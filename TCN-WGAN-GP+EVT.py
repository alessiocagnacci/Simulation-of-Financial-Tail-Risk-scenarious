import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, genpareto
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

# =============================================================================
# 1. CONFIGURAZIONE GENERALE
# =============================================================================
CSV_PATH = "/content/^fmib_d.csv"  # Percorso del singolo asset

LUNGHEZZA_FINESTRA = 63
EPOCHE_QUANTGAN = 5000  # Early stopping lo fermerà prima
BATCH_SIZE_QG = 64
DIM_LATENTE_QG = 16
N_PATH_GENERATI = 10000

# Regola TTUR per WGAN-GP e prevenzione mode-collapse
LR_CRITICO = 2e-4
LR_GENERATORE = 5e-5
N_LIVELLI_TCN_CRITICO = 6
N_LIVELLI_TCN_GENERATORE = 3

# Split Cronologico
FRAZIONE_TRAIN = 0.70
FRAZIONE_VAL = 0.15  # Il restante 0.15 è per il Test Set
SOGLIA_CODA_EVT = 0.05  # Splicing sul 5% peggiore dei dati

# Early Stopping QuantGAN (SWD)
USA_EARLY_STOPPING = True
EPOCHE_MINIME = 500
VERIFICA_OGNI = 25
PAZIENZA = 10
N_CAMPIONI_VERIFICA = 500
N_PROIEZIONI_SWD = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Hardware in uso: {DEVICE}")

# =============================================================================
# 2. MODULO QUANTGAN UNIVARIATO (TCN)
# =============================================================================

class ChompCausale(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class BloccoResidualeTCN(nn.Module):
    def __init__(self, canali_in, canali_out, kernel_size, dilation, dropout, usa_norm=False):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(canali_in, canali_out, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = ChompCausale(padding)
        self.norm1 = nn.InstanceNorm1d(canali_out) if usa_norm else nn.Identity()
        self.attivazione1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(canali_out, canali_out, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = ChompCausale(padding)
        self.norm2 = nn.InstanceNorm1d(canali_out) if usa_norm else nn.Identity()
        self.attivazione2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(dropout)

        self.rete = nn.Sequential(
            self.conv1, self.chomp1, self.norm1, self.attivazione1, self.dropout1,
            self.conv2, self.chomp2, self.norm2, self.attivazione2, self.dropout2
        )
        self.downsample = nn.Conv1d(canali_in, canali_out, 1) if canali_in != canali_out else None
        self.attivazione_finale = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.attivazione_finale(self.rete(x) + (x if self.downsample is None else self.downsample(x)))

class TCN(nn.Module):
    def __init__(self, canali_in, n_livelli, canali_nascosti, kernel_size, dropout, usa_norm=False):
        super().__init__()
        livelli = []
        for i in range(n_livelli):
            livelli.append(BloccoResidualeTCN(canali_in if i == 0 else canali_nascosti,
                                              canali_nascosti, kernel_size, 2**i, dropout, usa_norm))
        self.rete = nn.Sequential(*livelli)
    def forward(self, x): return self.rete(x)

class GeneratoreTCN(nn.Module):
    def __init__(self, dim_latente):
        super().__init__()
        self.tcn = TCN(dim_latente, N_LIVELLI_TCN_GENERATORE, 64, 3, 0.1, usa_norm=True)
        self.proj = nn.Conv1d(64, 1, 1)
    def forward(self, z): return self.proj(self.tcn(z)).squeeze(1)

class CriticoTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = TCN(1, N_LIVELLI_TCN_CRITICO, 64, 3, 0.1, usa_norm=False)
        self.testa = nn.Linear(64, 1)
    def forward(self, x): return self.testa(self.tcn(x.unsqueeze(1)).mean(dim=2))

def calcola_sliced_wasserstein(X, Y, n_proiezioni=128, device="cpu"):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    Y = torch.as_tensor(Y, dtype=torch.float32, device=device)
    dim = X.size(1)
    theta = torch.randn(dim, n_proiezioni, device=device)
    theta = theta / torch.norm(theta, dim=0, keepdim=True)
    X_proj = torch.matmul(X, theta)
    Y_proj = torch.matmul(Y, theta)
    X_sorted, _ = torch.sort(X_proj, dim=0)
    Y_sorted, _ = torch.sort(Y_proj, dim=0)
    n_x, n_y = X_sorted.size(0), Y_sorted.size(0)
    if n_x != n_y:
        q = torch.linspace(0.0, 1.0, steps=min(n_x, n_y), device=device)
        X_sorted = X_sorted[(q * (n_x - 1)).long()]
        Y_sorted = Y_sorted[(q * (n_y - 1)).long()]
    return torch.mean(torch.abs(X_sorted - Y_sorted)).item()

def calcola_distanza_reale_generato(finestre_val, generatore, dim_latente, lunghezza_finestra):
    generatore.eval()
    with torch.no_grad():
        z = torch.randn(N_CAMPIONI_VERIFICA, dim_latente, lunghezza_finestra, device=DEVICE)
        batch_generato = generatore(z)

        n_reali = len(finestre_val)
        n_selezionati = min(n_reali, N_CAMPIONI_VERIFICA)
        idx = torch.randperm(n_reali)[:n_selezionati]
        batch_reale = torch.tensor(finestre_val[idx], device=DEVICE)

        dist = calcola_sliced_wasserstein(batch_reale, batch_generato, n_proiezioni=N_PROIEZIONI_SWD, device=DEVICE)
    generatore.train()
    return dist

def allena_quantgan(finestre_train, finestre_val, dim_latente, lunghezza_finestra, epoche, name="Asset"):
    print(f"\n--- Training QuantGAN (TCN) - {name} ---")
    tensore_dati = torch.tensor(finestre_train, dtype=torch.float32, device=DEVICE)

    G = GeneratoreTCN(dim_latente).to(DEVICE)
    D = CriticoTCN().to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=LR_GENERATORE, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=LR_CRITICO, betas=(0.5, 0.9))

    storico_loss_c, storico_loss_g, storico_distanza = [], [], []
    miglior_distanza = float("inf")
    miglior_stato_G = None
    miglior_epoca = 0
    controlli_senza_miglioramento = 0

    for ep in range(epoche):
        for _ in range(5):
            idx = torch.randint(0, len(tensore_dati), (BATCH_SIZE_QG,), device=DEVICE)
            real = tensore_dati[idx]
            z = torch.randn(BATCH_SIZE_QG, dim_latente, lunghezza_finestra, device=DEVICE)
            fake = G(z).detach()

            alpha = torch.rand(BATCH_SIZE_QG, 1, device=DEVICE).expand_as(real)
            interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
            grad = torch.autograd.grad(outputs=D(interp), inputs=interp,
                                       grad_outputs=torch.ones_like(D(interp)),
                                       create_graph=True, retain_graph=True)[0]
            gp = ((grad.view(BATCH_SIZE_QG, -1).norm(2, dim=1) - 1) ** 2).mean()

            loss_D = D(fake).mean() - D(real).mean() + 10.0 * gp
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        z = torch.randn(BATCH_SIZE_QG, dim_latente, lunghezza_finestra, device=DEVICE)
        loss_G = -D(G(z)).mean()
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        storico_loss_c.append(loss_D.item())
        storico_loss_g.append(loss_G.item())

        if (ep + 1) % VERIFICA_OGNI == 0 or ep == 0:
            print(f"[{name}] Epoca {ep+1:4d} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

        if USA_EARLY_STOPPING and (ep + 1) % VERIFICA_OGNI == 0:
            distanza = calcola_distanza_reale_generato(finestre_val, G, dim_latente, lunghezza_finestra)
            storico_distanza.append((ep + 1, distanza))

            if distanza < miglior_distanza:
                miglior_distanza = distanza
                miglior_stato_G = copy.deepcopy(G.state_dict())
                miglior_epoca = ep + 1
                controlli_senza_miglioramento = 0
            else:
                controlli_senza_miglioramento += 1

            print(f"    -> [Val SWD] W-Dist: {distanza:.6f} (Best: {miglior_distanza:.6f} @ Epoca {miglior_epoca})")

            if (ep + 1) >= EPOCHE_MINIME and controlli_senza_miglioramento >= PAZIENZA:
                print(f"[{name}] Early stopping attivato. Ripristino pesi dell'epoca {miglior_epoca}.")
                G.load_state_dict(miglior_stato_G)
                break

    if USA_EARLY_STOPPING and miglior_stato_G is not None:
        G.load_state_dict(miglior_stato_G)

    return G, storico_loss_c, storico_loss_g, storico_distanza


# =============================================================================
# 3. MODULO SPLICING GPD UNIVARIATO (EVT PURA)
# =============================================================================

def splicing_univariato_gpd(rendimenti_quant, dati_reali, soglia_coda=0.05, name="Asset", verbose=True):
    if verbose: print(f"\n--- Avvio Splicing Univariato GPD [{name}] (Coda: {soglia_coda*100}%) ---")
    n_path, seq_len = rendimenti_quant.shape
    rend_flat = rendimenti_quant.flatten().copy()

    # FIT GPD SUI DATI REALI
    loss_reali = -dati_reali
    threshold_empirico = np.quantile(loss_reali, 1 - soglia_coda)
    eccessi = loss_reali[loss_reali > threshold_empirico] - threshold_empirico
    xi, loc, beta = genpareto.fit(eccessi, floc=0)

    # INDIVIDUAZIONE CLUSTER NEL QUANTGAN
    loss_quant = -rend_flat
    soglia_quant = np.quantile(loss_quant, 1 - soglia_coda)
    indici_estremi = np.where(loss_quant > soglia_quant)[0]
    n_estremi = len(indici_estremi)

    # GENERAZIONE E SPLICING
    u = np.random.uniform(0, 1, size=n_estremi)
    nuovi_eccessi = genpareto.ppf(u, xi, loc=0, scale=beta)
    nuove_loss_estreme = threshold_empirico + nuovi_eccessi

    indici_ordinati = indici_estremi[np.argsort(loss_quant[indici_estremi])]
    nuove_loss_estreme_ordinate = np.sort(nuove_loss_estreme)

    rend_flat[indici_ordinati] = -nuove_loss_estreme_ordinate

    return rend_flat.reshape(n_path, seq_len)

def calcola_var_es(rendimenti, livello):
    alpha = 1.0 - livello
    var = np.quantile(rendimenti, alpha)
    es = rendimenti[rendimenti <= var].mean()
    return var, es


# =============================================================================
# 4. MODULO GRAFICI E METRICHE DI COPERTURA
# =============================================================================

def plotta_risultati(prezzi_reali, path_sintetici_ibridi, storico_loss_c, storico_loss_g,
                     rendimenti_reali, rendimenti_generati_ibridi, storico_distanza,
                     idx_val, idx_test, out_png="qg_evt_risultati.png"):

    fig, assi = plt.subplots(3, 2, figsize=(14, 15))

    assi[0, 0].plot(prezzi_reali, color="black", label="Prezzo Reale")
    assi[0, 0].axvline(idx_val, color="orange", linestyle="--", label="Inizio Val")
    assi[0, 0].axvline(idx_test, color="red", linestyle="--", label="Inizio Test")
    assi[0, 0].set_title("Serie Storica Asset e Data Split")
    assi[0, 0].legend()

    for path in path_sintetici_ibridi[:20]:
        assi[0, 1].plot(path, alpha=0.6)
    assi[0, 1].set_title("Path Sintetici Ibridi Generati (QG + GPD)")

    assi[1, 0].plot(storico_loss_c, label="Loss Critico")
    assi[1, 0].plot(storico_loss_g, label="Loss Generatore")
    assi[1, 0].set_title("Andamento Loss Training")
    assi[1, 0].legend()

    assi[1, 1].hist(rendimenti_reali, bins=60, alpha=0.5, density=True, label="Reali (Train)")
    assi[1, 1].hist(rendimenti_generati_ibridi.flatten(), bins=60, alpha=0.5, density=True, label="Generati Ibridi")
    assi[1, 1].set_title("Distribuzione Rendimenti Complessivi")
    assi[1, 1].legend()

    if storico_distanza:
        epoche_verifica = [e for e, _ in storico_distanza]
        valori_distanza = [d for _, d in storico_distanza]
        assi[2, 0].plot(epoche_verifica, valori_distanza, marker="o", color="teal")
        assi[2, 0].set_title("Sliced-Wasserstein Distance (SWD) su Validation")
    else:
        assi[2, 0].axis("off")

    assi[2, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()
def plotta_10_finestre_ibride_con_copertura(prezzi_reali, generatore, r_train, lunghezza_finestra,
                                            idx_test_start, dim_latente, media_train, std_train,
                                            soglia_coda, device, out_png="qg_evt_10_finestre_coverage.png"):
    fig, assi = plt.subplots(2, 5, figsize=(22, 9))
    assi = assi.flatten()

    min_idx = idx_test_start
    max_idx = len(prezzi_reali) - lunghezza_finestra - 1

    if min_idx > max_idx:
        min_idx = max(0, max_idx)

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

        rend_q = (rend_std * std_train) + media_train
        rend_ibrido = splicing_univariato_gpd(rend_q, r_train, soglia_coda, verbose=False)

        path_sintetici = p0 * np.exp(np.cumsum(rend_ibrido, axis=1))
        media_path = np.insert(np.mean(path_sintetici, axis=0), 0, p0)
        std_path = np.insert(np.std(path_sintetici, axis=0), 0, 0.0)

        # --- CALCOLO EMPIRICO COVERAGE RATIO (SOLO DOWNSIDE RISK) ---
        reale_f = path_reale[1:]
        media_f = media_path[1:]
        std_f = std_path[1:]

        # Copertura = Quante volte il prezzo REALE si è mantenuto SOPRA la soglia di rischio
        copertura_1std = (reale_f >= media_f - std_f).mean()
        copertura_2std = (reale_f >= media_f - 2*std_f).mean()

        lista_cov_1std.append(copertura_1std)
        lista_cov_2std.append(copertura_2std)

        # PLOT
        ax = assi[i]
        ax.plot(asse_x, path_reale, color="black", linewidth=2, label="Reale (Test)")
        ax.plot(asse_x, media_path, color="blue", linewidth=2, label="Media Gen.")

        ax.fill_between(asse_x, media_path - std_path, media_path + std_path, color="blue", alpha=0.3, label="±1 Std")
        ax.fill_between(asse_x, media_path - 2*std_path, media_path + 2*std_path, color="blue", alpha=0.1, label="±2 Std")

        # Titoli aggiornati
        ax.set_title(f"Start: {idx_start}\nDownside Cov -1σ: {copertura_1std:.0%} | -2σ: {copertura_2std:.0%}")
        if i >= 5: ax.set_xlabel("Giorni passati")
        if i % 5 == 0: ax.set_ylabel("Prezzo")
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()

    # Stampa a console dei risultati aggregati (Downside Risk)
    print("\n" + "=" * 65)
    print("ANALISI DELLA COPERTURA DOWNSIDE (BACKTESTING SUL TEST SET)")
    print("=" * 65)
    print(f"Copertura Media Rischio a -1 Std Dev : {np.mean(lista_cov_1std):.2%} (Target Teorico Gaussiano: ~84.1%)")
    print(f"Copertura Media Rischio a -2 Std Dev : {np.mean(lista_cov_2std):.2%} (Target Teorico Gaussiano: ~97.7%)")
    print("Significato: Percentuale di giorni in cui il mercato NON ha subito perdite")
    print("maggiori rispetto al limite inferiore stimato dal modello generativo.")
    print("=" * 65)


# =============================================================================
# 5. ORCHESTRAZIONE MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 65)
    print("HYBRID TAIL RISK MODEL: QuantGAN (Bulk) + Univariate GPD (Tails)")
    print("=" * 65)

    df = pd.read_csv(CSV_PATH)
    prices = df['Close'].values
    log_ret = np.diff(np.log(prices + 1e-10))

    mask = ~np.isnan(log_ret) & ~np.isinf(log_ret)
    log_ret = log_ret[mask]

    n_tot = len(log_ret)
    n_train = int(n_tot * FRAZIONE_TRAIN)
    n_val = int(n_tot * (FRAZIONE_TRAIN + FRAZIONE_VAL))

    r_train = log_ret[:n_train]
    r_val = log_ret[n_train:n_val]
    r_test = log_ret[n_val:]

    print(f"Campioni - Train: {len(r_train)} | Val: {len(r_val)} | Test: {len(r_test)}")

    m0, s0 = r_train.mean(), r_train.std()

    r_train_std = (r_train - m0) / s0
    r_val_std = (r_val - m0) / s0

    finestre_train = np.array([r_train_std[i:i + LUNGHEZZA_FINESTRA]
                               for i in range(len(r_train_std) - LUNGHEZZA_FINESTRA + 1)])
    finestre_val = np.array([r_val_std[i:i + LUNGHEZZA_FINESTRA]
                             for i in range(len(r_val_std) - LUNGHEZZA_FINESTRA + 1)])

    qg, loss_c, loss_g, storico_distanza = allena_quantgan(
        finestre_train, finestre_val, DIM_LATENTE_QG, LUNGHEZZA_FINESTRA, EPOCHE_QUANTGAN, "Asset Principale"
    )

    print(f"\n--- Generazione path base QuantGAN ---")
    qg.eval()
    with torch.no_grad():
        z = torch.randn(N_PATH_GENERATI, DIM_LATENTE_QG, LUNGHEZZA_FINESTRA, device=DEVICE)
        rend_std = qg(z).cpu().numpy()

    rend_q = (rend_std * s0) + m0
    rend_ibrido = splicing_univariato_gpd(rend_q, r_train, SOGLIA_CODA_EVT, "Asset Principale")

    print("\n" + "=" * 65)
    print("RISULTATI FINALI (ASSET PRINCIPALE)")
    print("=" * 65)

    flat_reale = r_train
    flat_qg = rend_q.flatten()
    flat_ibrido = rend_ibrido.flatten()

    print(f"{'Metrica':<15} | {'Dati Reali':<12} | {'Solo QuantGAN':<15} | {'Spliced (QG+GPD)':<15}")
    print("-" * 65)
    print(f"{'Kurtosi':<15} | {pd.Series(flat_reale).kurtosis():<12.4f} | {pd.Series(flat_qg).kurtosis():<15.4f} | {pd.Series(flat_ibrido).kurtosis():<15.4f}")

    var_r_95, es_r_95 = calcola_var_es(flat_reale, 0.95)
    var_q_95, es_q_95 = calcola_var_es(flat_qg, 0.95)
    var_i_95, es_i_95 = calcola_var_es(flat_ibrido, 0.95)
    print(f"{'VaR (95%)':<15} | {var_r_95:<12.4f} | {var_q_95:<15.4f} | {var_i_95:<15.4f}")
    print(f"{'ES (95%)':<15} | {es_r_95:<12.4f} | {es_q_95:<15.4f} | {es_i_95:<15.4f}")

    var_r_99, es_r_99 = calcola_var_es(flat_reale, 0.99)
    var_q_99, es_q_99 = calcola_var_es(flat_qg, 0.99)
    var_i_99, es_i_99 = calcola_var_es(flat_ibrido, 0.99)
    print(f"{'VaR (99%)':<15} | {var_r_99:<12.4f} | {var_q_99:<15.4f} | {var_i_99:<15.4f}")
    print(f"{'ES (99%)':<15} | {es_r_99:<12.4f} | {es_q_99:<15.4f} | {es_i_99:<15.4f}")

    plt.figure(figsize=(12, 5))
    sns.kdeplot(flat_reale, label='Dati Reali (Train)', log_scale=(False, True), color='black', lw=2)
    sns.kdeplot(flat_qg, label='Solo QuantGAN (Code Deboli)', log_scale=(False, True), color='blue', linestyle='--')
    sns.kdeplot(flat_ibrido, label='Spliced Ibrido (Code perfette GPD)', log_scale=(False, True), color='red')
    plt.title("Distribuzione delle Code (Log-Scale) - Asset Principale")
    plt.xlim(-0.15, 0.15)
    plt.legend()
    plt.show()

    path_sintetici_ibridi = prices[0] * np.exp(np.cumsum(rend_ibrido, axis=1))

    plotta_risultati(
        prices, path_sintetici_ibridi, loss_c, loss_g, flat_reale, flat_ibrido,
        storico_distanza, idx_val=n_train, idx_test=n_val
    )

    plotta_10_finestre_ibride_con_copertura(
        prezzi_reali=prices, generatore=qg, r_train=r_train,
        lunghezza_finestra=LUNGHEZZA_FINESTRA, idx_test_start=n_val,
        dim_latente=DIM_LATENTE_QG, media_train=m0, std_train=s0,
        soglia_coda=SOGLIA_CODA_EVT, device=DEVICE
    )

if __name__ == "__main__":
    main()