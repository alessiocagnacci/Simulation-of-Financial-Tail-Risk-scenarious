import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

CSV_PATH = "/content/^fmib_d.csv"
lunghezza_finestra = 63
N_EPOCHE = 5000
BATCH_SIZE = 64
dim_latente = 16
N_PATH_GENERATI = 1000
N_CRITICO_PER_GENERATORE = 5
LAMBDA_GP = 10.0
lrC = 5e-5
lrG = 1e-4

# --- Architettura TCN ---
N_LIVELLI_TCN = 6
CANALI_TCN = 64
KERNEL_SIZE_TCN = 3
DROPOUT_TCN = 0.1

# --- Early stopping con Sliced-Wasserstein Distance (SWD) ---
USA_EARLY_STOPPING = True
EPOCHE_MINIME = 500
VERIFICA_OGNI = 25
PAZIENZA = 10
N_CAMPIONI_VERIFICA = 500
N_PROIEZIONI_SWD = 128

# --- Split cronologico: Train / Test ---
FRAZIONE_TRAIN = 0.85  # Il restante 15% è per il Test Set (Out-of-sample)

# --- Checkpoint su Google Drive ---
USA_CHECKPOINT = False
CHECKPOINT_PATH = "/content/drive/MyDrive/wgan_tcn2ndx_checkpoint.pt"
CHECKPOINT_OGNI = 25


# =============================================================================
# 1. Caricamento e Preprocessing Dati
# =============================================================================

def carica_da_csv(percorso_csv):
    df = pd.read_csv(percorso_csv)
    if "Close" not in df.columns:
        raise ValueError("Il CSV deve contenere una colonna 'Close' con i prezzi di chiusura")
    prezzi = df["Close"].values.astype(np.float64)
    rendimenti = np.diff(np.log(prezzi))
    return prezzi, rendimenti

def crea_finestre(rendimenti, lunghezza_finestra):
    finestre = []
    for i in range(len(rendimenti) - lunghezza_finestra + 1):
        finestre.append(rendimenti[i:i + lunghezza_finestra])
    return np.array(finestre, dtype=np.float32)

def dividi_train_test(rendimenti, frazione_train):
    """Ripartizione cronologica a 2 vie: Train / Test"""
    n_train = int(len(rendimenti) * frazione_train)
    return rendimenti[:n_train], rendimenti[n_train:], n_train

def standardizza(rendimenti, media, std):
    return (rendimenti - media) / std

def destandardizza(rendimenti_std, media, std):
    return rendimenti_std * std + media


# =============================================================================
# 2. Architetture TCN
# =============================================================================

class ChompCausale(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class BloccoResidualeTCN(nn.Module):
    def __init__(self, canali_in, canali_out, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(canali_in, canali_out, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = ChompCausale(padding)
        self.attivazione1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(canali_out, canali_out, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = ChompCausale(padding)
        self.attivazione2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(dropout)

        self.rete = nn.Sequential(
            self.conv1, self.chomp1, self.attivazione1, self.dropout1,
            self.conv2, self.chomp2, self.attivazione2, self.dropout2,
        )
        self.downsample = nn.Conv1d(canali_in, canali_out, 1) if canali_in != canali_out else None
        self.attivazione_finale = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.rete(x)
        residuo = x if self.downsample is None else self.downsample(x)
        return self.attivazione_finale(out + residuo)

class TCN(nn.Module):
    def __init__(self, canali_in, n_livelli, canali_nascosti, kernel_size, dropout):
        super().__init__()
        livelli = []
        for i in range(n_livelli):
            livelli.append(BloccoResidualeTCN(canali_in if i == 0 else canali_nascosti,
                                              canali_nascosti, kernel_size, 2**i, dropout))
        self.rete = nn.Sequential(*livelli)
    def forward(self, x): return self.rete(x)

class Generatore(nn.Module):
    def __init__(self, dim_latente, n_livelli, canali_nascosti, kernel_size, dropout):
        super().__init__()
        self.tcn = TCN(dim_latente, n_livelli, canali_nascosti, kernel_size, dropout)
        self.proiezione_finale = nn.Conv1d(canali_nascosti, 1, kernel_size=1)
    def forward(self, z): return self.proiezione_finale(self.tcn(z)).squeeze(1)

class Critico(nn.Module):
    def __init__(self, n_livelli, canali_nascosti, kernel_size, dropout):
        super().__init__()
        self.tcn = TCN(1, n_livelli, canali_nascosti, kernel_size, dropout)
        self.testa = nn.Linear(canali_nascosti, 1)
    def forward(self, x): return self.testa(self.tcn(x.unsqueeze(1)).mean(dim=2))


# =============================================================================
# 3. Sliced-Wasserstein Distance & Gradient Penalty
# =============================================================================

def calcola_gradient_penalty(critico, reali, generati, device):
    batch_size = reali.size(0)
    eps = torch.rand(batch_size, 1, device=device).expand_as(reali)
    interpolati = (eps * reali + (1 - eps) * generati).requires_grad_(True)
    score_interpolati = critico(interpolati)
    gradienti = torch.autograd.grad(outputs=score_interpolati, inputs=interpolati,
                                    grad_outputs=torch.ones_like(score_interpolati),
                                    create_graph=True, retain_graph=True)[0]
    return ((gradienti.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()

def calcola_sliced_wasserstein(X, Y, n_proiezioni=128, device="cpu"):
    X, Y = torch.as_tensor(X, dtype=torch.float32, device=device), torch.as_tensor(Y, dtype=torch.float32, device=device)
    theta = torch.randn(X.size(1), n_proiezioni, device=device)
    theta = theta / torch.norm(theta, dim=0, keepdim=True)

    X_sorted, _ = torch.sort(torch.matmul(X, theta), dim=0)
    Y_sorted, _ = torch.sort(torch.matmul(Y, theta), dim=0)

    n_x, n_y = X_sorted.size(0), Y_sorted.size(0)
    if n_x != n_y:
        q = torch.linspace(0.0, 1.0, steps=min(n_x, n_y), device=device)
        X_sorted = X_sorted[(q * (n_x - 1)).long()]
        Y_sorted = Y_sorted[(q * (n_y - 1)).long()]

    return torch.mean(torch.abs(X_sorted - Y_sorted)).item()

def calcola_distanza_reale_generato(finestre_target, generatore, dim_latente,
                                    lunghezza_finestra, device, n_campioni_verifica=500, n_proiezioni=128):
    generatore.eval()
    with torch.no_grad():
        z = torch.randn(n_campioni_verifica, dim_latente, lunghezza_finestra, device=device)
        batch_generato = generatore(z)

        idx = torch.randperm(len(finestre_target))[:min(len(finestre_target), n_campioni_verifica)]
        batch_reale = torch.tensor(finestre_target[idx], device=device)
        dist = calcola_sliced_wasserstein(batch_reale, batch_generato, n_proiezioni=n_proiezioni, device=device)
    generatore.train()
    return dist


# =============================================================================
# 4. Training Loop WGAN-GP
# =============================================================================

def allena_wgan_gp(finestre_train, dim_latente, lunghezza_finestra, n_epoche, batch_size, lrC, lrG,
                   n_critico_per_generatore, lambda_gp, n_livelli_tcn, canali_tcn, kernel_size_tcn, dropout_tcn,
                   device=None, verbose_ogni=200, usa_early_stopping=True, epoche_minime=500,
                   verifica_ogni=50, pazienza=10, n_campioni_verifica=500, n_proiezioni_swd=128):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training su device: {device}")

    tensore_dati = torch.tensor(finestre_train, dtype=torch.float32, device=device)
    n_campioni = tensore_dati.size(0)

    generatore = Generatore(dim_latente, n_livelli_tcn, canali_tcn, kernel_size_tcn, dropout_tcn).to(device)
    critico = Critico(n_livelli_tcn, canali_tcn, kernel_size_tcn, dropout_tcn).to(device)

    opt_g = optim.Adam(generatore.parameters(), lr=lrG, betas=(0.5, 0.9))
    opt_c = optim.Adam(critico.parameters(), lr=lrC, betas=(0.5, 0.9))

    storico_loss_c, storico_loss_g, storico_distanza = [], [], []
    miglior_distanza, miglior_stato_generatore, miglior_epoca, controlli_senza_miglioramento = float("inf"), None, 0, 0

    for epoca in range(1, n_epoche + 1):
        for _ in range(n_critico_per_generatore):
            batch_reale = tensore_dati[torch.randint(0, n_campioni, (batch_size,), device=device)]
            batch_generato = generatore(torch.randn(batch_size, dim_latente, lunghezza_finestra, device=device)).detach()

            gp = calcola_gradient_penalty(critico, batch_reale, batch_generato, device)
            loss_c = critico(batch_generato).mean() - critico(batch_reale).mean() + lambda_gp * gp

            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

        batch_generato = generatore(torch.randn(batch_size, dim_latente, lunghezza_finestra, device=device))
        loss_g = -critico(batch_generato).mean()

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        storico_loss_c.append(loss_c.item())
        storico_loss_g.append(loss_g.item())

        if epoca % verbose_ogni == 0 or epoca == 1:
            print(f"[Epoca {epoca:5d}] loss_critico={loss_c.item():.4f}  loss_generatore={loss_g.item():.4f}")

        # Valutazione SWD eseguita sul TRAIN SET
        if usa_early_stopping and epoca % verifica_ogni == 0:
            distanza = calcola_distanza_reale_generato(
                finestre_train, generatore, dim_latente, lunghezza_finestra,
                device, n_campioni_verifica=n_campioni_verifica, n_proiezioni=n_proiezioni_swd
            )
            storico_distanza.append((epoca, distanza))

            if distanza < miglior_distanza:
                miglior_distanza = distanza
                miglior_stato_generatore = copy.deepcopy(generatore.state_dict())
                miglior_epoca = epoca
                controlli_senza_miglioramento = 0
            else:
                controlli_senza_miglioramento += 1

            print(f"    -> [Verifica Train SWD Epoca {epoca}] SWD={distanza:.6f} (Migliore: {miglior_distanza:.6f} @ {miglior_epoca})")

            if epoca >= epoche_minime and controlli_senza_miglioramento >= pazienza:
                print(f"\nEarly stopping all'epoca {epoca}. Ripristino pesi dell'epoca {miglior_epoca}.")
                generatore.load_state_dict(miglior_stato_generatore)
                break

    if usa_early_stopping and miglior_stato_generatore is not None:
        generatore.load_state_dict(miglior_stato_generatore)

    return generatore, critico, storico_loss_c, storico_loss_g, storico_distanza


# =============================================================================
# 5. Generazione Grafici e Metriche
# =============================================================================

def genera_path_sintetici(generatore, n_path, dim_latente, lunghezza_finestra,
                          media_train, std_train, prezzo_iniziale, device):
    generatore.eval()
    with torch.no_grad():
        z = torch.randn(n_path, dim_latente, lunghezza_finestra, device=device)
        rend_std = generatore(z).cpu().numpy()
    rend_reali = destandardizza(rend_std, media_train, std_train)
    path_prezzi = prezzo_iniziale * np.exp(np.cumsum(rend_reali, axis=1))
    return rend_reali, path_prezzi

def calcola_var_es(rendimenti, livello=0.99):
    var = np.quantile(rendimenti, 1.0 - livello)
    es = rendimenti[rendimenti <= var].mean()
    return var, es

def plotta_risultati(prezzi_reali, path_sintetici, storico_loss_c, storico_loss_g,
                     rendimenti_reali, rendimenti_generati, storico_distanza, idx_test, out_png="risultati.png"):
    fig, assi = plt.subplots(3, 2, figsize=(14, 15))

    assi[0, 0].plot(prezzi_reali, color="black", label="Prezzo")
    assi[0, 0].axvline(idx_test, color="red", linestyle="--", label="Inizio Test")
    assi[0, 0].set_title("Serie Storica (Train / Test Split)")
    assi[0, 0].legend()

    for path in path_sintetici[:20]: assi[0, 1].plot(path, alpha=0.6)
    assi[0, 1].set_title("Path sintetici generati (TCN)")

    assi[1, 0].plot(storico_loss_c, label="Loss Critico")
    assi[1, 0].plot(storico_loss_g, label="Loss Generatore")
    assi[1, 0].set_title("Andamento Loss WGAN-GP")
    assi[1, 0].legend()

    assi[1, 1].hist(rendimenti_reali, bins=60, alpha=0.5, density=True, label="Reali (Train)")
    assi[1, 1].hist(rendimenti_generati.flatten(), bins=60, alpha=0.5, density=True, label="Generati")
    assi[1, 1].set_title("Distribuzione Rendimenti")
    assi[1, 1].legend()

    if storico_distanza:
        epoche, valori = zip(*storico_distanza)
        assi[2, 0].plot(epoche, valori, marker="o", color="teal")
        assi[2, 0].set_title("SWD su Train Set (In-Sample)")
    else:
        assi[2, 0].axis("off")

    assi[2, 1].axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()
def plotta_10_finestre(prezzi_reali, generatore, dim_latente, media_train, std_train,
                       lunghezza_finestra, idx_test_start, device, out_png="10_finestre_test.png"):
    fig, assi = plt.subplots(2, 5, figsize=(22, 9))
    assi = assi.flatten()

    min_idx = idx_test_start
    max_idx = max(0, len(prezzi_reali) - lunghezza_finestra - 1)
    indici_partenza = np.linspace(min_idx, max_idx, 10, dtype=int)
    asse_x = np.arange(lunghezza_finestra + 1)

    lista_cov_1std = []
    lista_cov_2std = []

    generatore.eval()
    for i, idx_start in enumerate(indici_partenza):
        p0 = prezzi_reali[idx_start]
        path_reale = prezzi_reali[idx_start : idx_start + lunghezza_finestra + 1]

        with torch.no_grad():
            z = torch.randn(N_PATH_GENERATI, dim_latente, lunghezza_finestra, device=device)
            rend_std = generatore(z).cpu().numpy()

        rend_gen = destandardizza(rend_std, media_train, std_train)
        path_sintetici = p0 * np.exp(np.cumsum(rend_gen, axis=1))

        media_path = np.insert(np.mean(path_sintetici, axis=0), 0, p0)
        std_path = np.insert(np.std(path_sintetici, axis=0), 0, 0.0)

        # --- CALCOLO EMPIRICO COVERAGE RATIO (SOLO DOWNSIDE RISK) ---
        reale_f = path_reale[1:]
        media_f = media_path[1:]
        std_f = std_path[1:]

        # Misuriamo quante volte la traiettoria reale non buca il limite inferiore
        copertura_1std = (reale_f >= media_f - std_f).mean()
        copertura_2std = (reale_f >= media_f - 2*std_f).mean()

        lista_cov_1std.append(copertura_1std)
        lista_cov_2std.append(copertura_2std)

        # Plot
        ax = assi[i]
        ax.plot(asse_x, path_reale, color="black", linewidth=2, label="Reale (Test)")
        ax.plot(asse_x, media_path, color="blue", linewidth=2, label="Media Gen.")
        ax.fill_between(asse_x, media_path - std_path, media_path + std_path, color="blue", alpha=0.3, label="±1 Std")
        ax.fill_between(asse_x, media_path - 2*std_path, media_path + 2*std_path, color="blue", alpha=0.1, label="±2 Std")

        # Titolo aggiornato con le metriche di copertura
        ax.set_title(f"Start: {idx_start}\nDownside Cov -1σ: {copertura_1std:.0%} | -2σ: {copertura_2std:.0%}")
        if i >= 5: ax.set_xlabel("Giorni passati")
        if i % 5 == 0: ax.set_ylabel("Prezzo")
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()

    # Stampa a console dei risultati aggregati
    print("\n" + "=" * 65)
    print("ANALISI DELLA COPERTURA DOWNSIDE (SULLE 10 FINESTRE DI TEST)")
    print("=" * 65)
    print(f"Copertura Media Rischio a -1 Std Dev : {np.mean(lista_cov_1std):.2%} (Target Teorico: ~84.1%)")
    print(f"Copertura Media Rischio a -2 Std Dev : {np.mean(lista_cov_2std):.2%} (Target Teorico: ~97.7%)")
    print("=" * 65)


# =============================================================================
# 6. ESECUZIONE
# =============================================================================

prezzi, rendimenti = carica_da_csv(CSV_PATH)
print(f"Caricati {len(prezzi)} prezzi da {CSV_PATH}")

# 1. Ripartizione cronologica a 2 vie: Train (85%) / Test (15%)
r_train, r_test, n_train = dividi_train_test(rendimenti, FRAZIONE_TRAIN)
print(f"Split cronologico -> Train: {len(r_train)}, Test: {len(r_test)}")

# 2. Standardizzazione stimata rigorosamente solo su Train
m0, s0 = r_train.mean(), r_train.std()
r_train_std = standardizza(r_train, m0, s0)
r_test_std = standardizza(r_test, m0, s0)

# Creazione finestre temporali 2D SOLO sul Train
finestre_train = crea_finestre(r_train_std, lunghezza_finestra)

# 3. Addestramento con Early Stopping in-sample
generatore, critico, loss_c, loss_g, storico_distanza = allena_wgan_gp(
    finestre_train, dim_latente, lunghezza_finestra, N_EPOCHE, BATCH_SIZE, lrC, lrG,
    N_CRITICO_PER_GENERATORE, LAMBDA_GP, N_LIVELLI_TCN, CANALI_TCN, KERNEL_SIZE_TCN, DROPOUT_TCN,
    device=None, verbose_ogni=200, usa_early_stopping=USA_EARLY_STOPPING,
    epoche_minime=EPOCHE_MINIME, verifica_ogni=VERIFICA_OGNI, pazienza=PAZIENZA,
    n_campioni_verifica=N_CAMPIONI_VERIFICA, n_proiezioni_swd=N_PROIEZIONI_SWD
)

device = next(generatore.parameters()).device

# 4. Generazione path sintetici
rendimenti_generati, path_sintetici = genera_path_sintetici(
    generatore, N_PATH_GENERATI, dim_latente, lunghezza_finestra, m0, s0, prezzi[0], device
)
rendimenti_gen_flat = rendimenti_generati.flatten()

# 5. Confronto statistico (Train vs Test vs Generati)
print("\n--- Confronto Statistico ---")
print(f"{'':15s} {'Train':>12s} {'Test':>12s} {'Generati':>12s}")
print(f"{'Media':15s} {r_train.mean():12.6f} {r_test.mean():12.6f} {rendimenti_gen_flat.mean():12.6f}")
print(f"{'Std':15s} {r_train.std():12.6f} {r_test.std():12.6f} {rendimenti_gen_flat.std():12.6f}")
print(f"{'Skew':15s} {pd.Series(r_train).skew():12.4f} {pd.Series(r_test).skew():12.4f} {pd.Series(rendimenti_gen_flat).skew():12.4f}")
print(f"{'Kurtosi':15s} {pd.Series(r_train).kurtosis():12.4f} {pd.Series(r_test).kurtosis():12.4f} {pd.Series(rendimenti_gen_flat).kurtosis():12.4f}")

var_r_95, es_r_95 = calcola_var_es(r_train, 0.95)
var_t_95, es_t_95 = calcola_var_es(r_test, 0.95)
var_g_95, es_g_95 = calcola_var_es(rendimenti_gen_flat, 0.95)

var_r_99, es_r_99 = calcola_var_es(r_train, 0.99)
var_t_99, es_t_99 = calcola_var_es(r_test, 0.99)
var_g_99, es_g_99 = calcola_var_es(rendimenti_gen_flat, 0.99)

print("\n--- Tail Risk ---")
print(f"{'':15s} {'Train':>12s} {'Test':>12s} {'Generati':>12s}")
print(f"{'VaR (95%)':15s} {var_r_95:12.4f} {var_t_95:12.4f} {var_g_95:12.4f}")
print(f"{'ES (95%)':15s} {es_r_95:12.4f} {es_t_95:12.4f} {es_g_95:12.4f}")
print(f"{'VaR (99%)':15s} {var_r_99:12.4f} {var_t_99:12.4f} {var_g_99:12.4f}")
print(f"{'ES (99%)':15s} {es_r_99:12.4f} {es_t_99:12.4f} {es_g_99:12.4f}")

# 6. Grafici Finali (Senza Val)
plotta_risultati(
    prezzi, path_sintetici, loss_c, loss_g, r_train, rendimenti_generati,
    storico_distanza, idx_test=n_train
)

plotta_10_finestre(
    prezzi, generatore, dim_latente, m0, s0,
    lunghezza_finestra, idx_test_start=n_train, device=device
)