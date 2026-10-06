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

# Sostituisci con i tuoi path reali.
ASSET_FILES = {
    "NKX":     "/content/^nkx_d-2.csv",
    "NDX":     "/content/^ndx_d copia.csv",
    "HSI":     "/content/^hsi_d.csv",
    "FTSEmib": "/content/^fmib_d.csv",
}
PRICE_COL = "Close"
DATE_COL = "Date"

PORTFOLIO_WEIGHTS = {
    "NKX": 0.25,
    "NDX": 0.25,
    "HSI": 0.25,
    "FTSEmib": 0.25,
}

LUNGHEZZA_FINESTRA = 63
EPOCHE_QUANTGAN = 5000
BATCH_SIZE_QG = 64
DIM_LATENTE_QG = 16
N_PATH_GENERATI = 500
SOGLIA_CODA_EVT = 0.05

# Regola TTUR per WGAN-GP e prevenzione mode-collapse
LR_CRITICO = 2e-4
LR_GENERATORE = 2e-5
N_LIVELLI_TCN_CRITICO = 6
N_LIVELLI_TCN_GENERATORE = 3

# Split Cronologico Marginale
FRAZIONE_TRAIN = 0.70
FRAZIONE_VAL = 0.15

# Early Stopping (SWD)
USA_EARLY_STOPPING = True
EPOCHE_MINIME = 500
VERIFICA_OGNI = 25
PAZIENZA = 10
N_CAMPIONI_VERIFICA = 500
N_PROIEZIONI_SWD = 128

# Copula e Portafoglio
N_SCENARI_PORTAFOGLIO = 20000  # Aumentato per avere un pool denso da cui campionare i path
FRAZIONE_TRAIN_COPULA = 0.75

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

        if (ep + 1) % 500 == 0 or ep == 0:
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

            if (ep + 1) >= EPOCHE_MINIME and controlli_senza_miglioramento >= PAZIENZA:
                print(f"[{name}] Early stopping all'epoca {ep+1}. Ripristino pesi dell'epoca {miglior_epoca}.")
                G.load_state_dict(miglior_stato_G)
                break

    if USA_EARLY_STOPPING and miglior_stato_G is not None:
        G.load_state_dict(miglior_stato_G)

    return G, storico_loss_c, storico_loss_g, storico_distanza


# =============================================================================
# 3. MODULO SPLICING GPD UNIVARIATO (EVT PURA)
# =============================================================================

def splicing_univariato_gpd(rendimenti_quant, dati_reali, soglia_coda=0.05, name="Asset", verbose=True):
    n_path, seq_len = rendimenti_quant.shape
    rend_flat = rendimenti_quant.flatten().copy()

    loss_reali = -dati_reali
    threshold_empirico = np.quantile(loss_reali, 1 - soglia_coda)
    eccessi = loss_reali[loss_reali > threshold_empirico] - threshold_empirico
    xi, loc, beta = genpareto.fit(eccessi, floc=0)

    loss_quant = -rend_flat
    soglia_quant = np.quantile(loss_quant, 1 - soglia_coda)
    indici_estremi = np.where(loss_quant > soglia_quant)[0]
    n_estremi = len(indici_estremi)

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
# 4. MODULO GRAFICI E METRICHE DI COPERTURA (MARGINALI E PORTAFOGLIO)
# =============================================================================

def plotta_risultati(prezzi_reali, path_sintetici_ibridi, storico_loss_c, storico_loss_g,
                     rendimenti_reali, rendimenti_generati_ibridi, storico_distanza,
                     idx_val, idx_test, out_png="risultati.png"):
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
    plt.close()

def plotta_10_finestre_ibride_con_copertura(prezzi_reali, generatore, r_train, lunghezza_finestra,
                                            idx_test_start, dim_latente, media_train, std_train,
                                            soglia_coda, device, asset_name, out_png="10_finestre.png"):
    fig, assi = plt.subplots(2, 5, figsize=(22, 9))
    assi = assi.flatten()

    min_idx = idx_test_start
    max_idx = len(prezzi_reali) - lunghezza_finestra - 1
    if min_idx > max_idx: min_idx = max(0, max_idx)

    indici_partenza = np.linspace(min_idx, max_idx, 10, dtype=int)
    asse_x = np.arange(lunghezza_finestra + 1)
    lista_cov_1std, lista_cov_2std = [], []

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

        reale_f, media_f, std_f = path_reale[1:], media_path[1:], std_path[1:]
        copertura_1std = (reale_f >= media_f - std_f).mean()
        copertura_2std = (reale_f >= media_f - 2*std_f).mean()

        lista_cov_1std.append(copertura_1std)
        lista_cov_2std.append(copertura_2std)

        ax = assi[i]
        ax.plot(asse_x, path_reale, color="black", linewidth=2, label="Reale (Test)")
        ax.plot(asse_x, media_path, color="blue", linewidth=2, label="Media Gen.")
        ax.fill_between(asse_x, media_path - std_path, media_path + std_path, color="blue", alpha=0.3, label="±1 Std")
        ax.fill_between(asse_x, media_path - 2*std_path, media_path + 2*std_path, color="blue", alpha=0.1, label="±2 Std")

        ax.set_title(f"Start: {idx_start}\nDownside Cov -1σ: {copertura_1std:.0%} | -2σ: {copertura_2std:.0%}")
        if i >= 5: ax.set_xlabel("Giorni passati")
        if i % 5 == 0: ax.set_ylabel("Prezzo")
        if i == 0: ax.legend()

    plt.suptitle(f"Analisi Copertura Downside Test Set - {asset_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# --- NUOVE FUNZIONI PER IL PORTAFOGLIO ---

def plotta_dashboard_portafoglio(storico_allineato, scenari_df, asset_order, pesi_dict, idx_test, out_png="PORTFOLIO_dashboard.png"):
    pesi = np.array([pesi_dict[a] for a in asset_order])

    # Ricostruzione Indice Storico di Portafoglio (Base 10k)
    pnl_reale_totale = storico_allineato @ pesi
    prezzo_portafoglio = 10000 * np.exp(np.insert(np.cumsum(pnl_reale_totale), 0, 0.0))
    pnl_sintetico_totale = scenari_df[asset_order].values @ pesi

    fig, assi = plt.subplots(2, 2, figsize=(14, 10))

    # 0,0: Valore Storico
    assi[0, 0].plot(prezzo_portafoglio, color="black", label="Valore Reale (Base 10k)")
    assi[0, 0].axvline(idx_test, color="red", linestyle="--", label="Inizio Test (Copula)")
    assi[0, 0].set_title("Valore Storico del Portafoglio e Data Split")
    assi[0, 0].legend()

    # 0,1: Path Sintetici (Campionati casualmente dagli scenari di copula)
    draws = np.random.choice(pnl_sintetico_totale, size=(20, LUNGHEZZA_FINESTRA), replace=True)
    path_sintetici = 10000 * np.exp(np.cumsum(draws, axis=1))
    for path in path_sintetici:
        assi[0, 1].plot(path, alpha=0.6)
    assi[0, 1].set_title("20 Path Sintetici Accoppiati (Copula OOS)")

    # 1,0: Distribuzione Bulk
    pnl_reale_train = pnl_reale_totale[:idx_test]
    assi[1, 0].hist(pnl_reale_train, bins=60, alpha=0.5, density=True, label="Reale (Train)")
    assi[1, 0].hist(pnl_sintetico_totale, bins=60, alpha=0.5, density=True, label="Sintetico")
    assi[1, 0].set_title("Distribuzione Rendimenti di Portafoglio (Bulk)")
    assi[1, 0].legend()

    # 1,1: Code Log-Scale
    sns.kdeplot(pnl_reale_train, label='Reale (Train)', log_scale=(False, True), color='black', lw=2, ax=assi[1, 1])
    sns.kdeplot(pnl_sintetico_totale, label='Sintetico (Accoppiato QG+GPD)', log_scale=(False, True), color='red', ax=assi[1, 1])
    assi[1, 1].set_title("Distribuzione delle Code Portafoglio (Log-Scale)")
    assi[1, 1].set_xlim(-0.10, 0.10)
    assi[1, 1].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plotta_10_finestre_portafoglio(storico_allineato, scenari_df, asset_order, pesi_dict, lunghezza_finestra, idx_test_start, out_png="PORTFOLIO_10_finestre.png"):
    pesi = np.array([pesi_dict[a] for a in asset_order])
    pnl_reale_totale = storico_allineato @ pesi
    prezzo_portafoglio = 10000 * np.exp(np.insert(np.cumsum(pnl_reale_totale), 0, 0.0))
    pnl_sintetico_totale = scenari_df[asset_order].values @ pesi

    fig, assi = plt.subplots(2, 5, figsize=(22, 9))
    assi = assi.flatten()

    min_idx = idx_test_start
    max_idx = len(prezzo_portafoglio) - lunghezza_finestra - 1
    if min_idx > max_idx: min_idx = max(0, max_idx)

    indici_partenza = np.linspace(min_idx, max_idx, 10, dtype=int)
    asse_x = np.arange(lunghezza_finestra + 1)
    lista_cov_1std, lista_cov_2std = [], []

    for i, idx_start in enumerate(indici_partenza):
        p0 = prezzo_portafoglio[idx_start]
        path_reale = prezzo_portafoglio[idx_start : idx_start + lunghezza_finestra + 1]

        # Campionamento degli scenari aggregati (Draw independenti dal pool accoppiato)
        draws = np.random.choice(pnl_sintetico_totale, size=(N_PATH_GENERATI, lunghezza_finestra), replace=True)
        path_sintetici = p0 * np.exp(np.cumsum(draws, axis=1))

        media_path = np.insert(np.mean(path_sintetici, axis=0), 0, p0)
        std_path = np.insert(np.std(path_sintetici, axis=0), 0, 0.0)

        # Copertura Downside
        reale_f, media_f, std_f = path_reale[1:], media_path[1:], std_path[1:]
        copertura_1std = (reale_f >= media_f - std_f).mean()
        copertura_2std = (reale_f >= media_f - 2*std_f).mean()

        lista_cov_1std.append(copertura_1std)
        lista_cov_2std.append(copertura_2std)

        ax = assi[i]
        ax.plot(asse_x, path_reale, color="black", linewidth=2, label="Reale (Test)")
        ax.plot(asse_x, media_path, color="blue", linewidth=2, label="Media Gen.")
        ax.fill_between(asse_x, media_path - std_path, media_path + std_path, color="blue", alpha=0.3, label="±1 Std")
        ax.fill_between(asse_x, media_path - 2*std_path, media_path + 2*std_path, color="blue", alpha=0.1, label="±2 Std")

        ax.set_title(f"Start: {idx_start}\nDownside Cov -1σ: {copertura_1std:.0%} | -2σ: {copertura_2std:.0%}")
        if i >= 5: ax.set_xlabel("Giorni passati")
        if i % 5 == 0: ax.set_ylabel("Valore PTF")
        if i == 0: ax.legend()

    plt.suptitle("Analisi Copertura Downside Test Set - PORTAFOGLIO AGGREGATO (Copula OOS)", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"\n[PORTAFOGLIO] Copertura Downside Media: -1σ = {np.mean(lista_cov_1std):.2%} | -2σ = {np.mean(lista_cov_2std):.2%}")


# =============================================================================
# 5. TRAINING E GENERAZIONE PER SINGOLO ASSET
# =============================================================================

def allena_e_genera_asset(nome_asset, prices):
    print(f"\n" + "="*50)
    print(f"ELABORAZIONE ASSET: {nome_asset}")
    print("="*50)

    log_ret = np.diff(np.log(prices + 1e-10))
    mask = ~np.isnan(log_ret) & ~np.isinf(log_ret)
    log_ret = log_ret[mask]

    n_tot = len(log_ret)
    n_train = int(n_tot * FRAZIONE_TRAIN)
    n_val = int(n_tot * (FRAZIONE_TRAIN + FRAZIONE_VAL))

    r_train = log_ret[:n_train]
    r_val = log_ret[n_train:n_val]
    r_test = log_ret[n_val:]

    m0, s0 = r_train.mean(), r_train.std()

    finestre_train = np.array([((r_train - m0)/s0)[i:i + LUNGHEZZA_FINESTRA]
                               for i in range(len(r_train) - LUNGHEZZA_FINESTRA + 1)])
    finestre_val = np.array([((r_val - m0)/s0)[i:i + LUNGHEZZA_FINESTRA]
                             for i in range(len(r_val) - LUNGHEZZA_FINESTRA + 1)])

    qg, loss_c, loss_g, storico_distanza = allena_quantgan(
        finestre_train, finestre_val, DIM_LATENTE_QG, LUNGHEZZA_FINESTRA, EPOCHE_QUANTGAN, nome_asset
    )

    qg.eval()
    with torch.no_grad():
        z = torch.randn(N_PATH_GENERATI, DIM_LATENTE_QG, LUNGHEZZA_FINESTRA, device=DEVICE)
        rend_std = qg(z).cpu().numpy()

    rend_q = (rend_std * s0) + m0
    rend_ibrido = splicing_univariato_gpd(rend_q, r_train, SOGLIA_CODA_EVT, nome_asset, verbose=False)
    path_sintetici_ibridi = prices[0] * np.exp(np.cumsum(rend_ibrido, axis=1))

    plotta_risultati(
        prices, path_sintetici_ibridi, loss_c, loss_g, r_train, rend_ibrido,
        storico_distanza, idx_val=n_train, idx_test=n_val, out_png=f"{nome_asset}_dashboard.png"
    )

    plotta_10_finestre_ibride_con_copertura(
        prezzi_reali=prices, generatore=qg, r_train=r_train, lunghezza_finestra=LUNGHEZZA_FINESTRA,
        idx_test_start=n_val, dim_latente=DIM_LATENTE_QG, media_train=m0, std_train=s0,
        soglia_coda=SOGLIA_CODA_EVT, device=DEVICE, asset_name=nome_asset, out_png=f"{nome_asset}_10_finestre.png"
    )

    return {
        'r_train': r_train,
        'r_test': r_test,
        'rend_quantgan': rend_q,
        'rend_ibrido': rend_ibrido,
        'pool_marginale': rend_ibrido.flatten(),
    }


# =============================================================================
# 6. ACCOPPIAMENTO VIA COPULA EMPIRICA (SCHAAKE SHUFFLE)
# =============================================================================

def costruisci_ranghi_storici(prezzi_dict, date_col=DATE_COL, price_col=PRICE_COL):
    asset_names = list(prezzi_dict.keys())
    dfs = []
    for name, path in prezzi_dict.items():
        df = pd.read_csv(path)
        sub = df[[date_col, price_col]].rename(columns={price_col: name})
        sub[date_col] = pd.to_datetime(sub[date_col])
        dfs.append(sub)

    merged = dfs[0]
    for sub in dfs[1:]:
        merged = pd.merge(merged, sub, on=date_col, how='inner')
    merged = merged.sort_values(date_col).reset_index(drop=True)

    print(f"\nRighe allineate per data (per la copula): {len(merged)}")
    prices = merged[asset_names].values
    log_returns = np.diff(np.log(prices + 1e-10), axis=0)
    mask = ~np.any(np.isnan(log_returns) | np.isinf(log_returns), axis=1)

    return log_returns[mask], asset_names

def split_storico_copula(storico_allineato, frazione_train=FRAZIONE_TRAIN_COPULA):
    n = len(storico_allineato)
    n_train = int(n * frazione_train)
    storico_train_copula = storico_allineato[:n_train]
    storico_test_copula = storico_allineato[n_train:]
    print(f"Split storico copula: train={len(storico_train_copula)} gg (per shuffle), "
          f"test={len(storico_test_copula)} gg (per valutazione OOS)")
    return storico_train_copula, storico_test_copula

def schaake_shuffle_coupling(pools_marginali, storico_allineato, asset_order, n_scenari, seed=42):
    rng = np.random.default_rng(seed)
    n_assets = len(asset_order)
    T_hist = storico_allineato.shape[0]

    ranghi_storici = np.zeros_like(storico_allineato)
    for j in range(n_assets):
        ranghi_storici[:, j] = pd.Series(storico_allineato[:, j]).rank(method='first').values - 1

    giorni_idx = rng.choice(T_hist, size=n_scenari, replace=(n_scenari > T_hist))
    scenari = np.zeros((n_scenari, n_assets))

    for j, asset in enumerate(asset_order):
        pool_ordinato = np.sort(pools_marginali[asset])
        n_pool = len(pool_ordinato)

        rank_giorni = ranghi_storici[giorni_idx, j]
        q_pos = np.round(rank_giorni / (T_hist - 1) * (n_pool - 1)).astype(int)
        q_pos = np.clip(q_pos, 0, n_pool - 1)
        scenari[:, j] = pool_ordinato[q_pos]

    return pd.DataFrame(scenari, columns=asset_order)


# =============================================================================
# 7. VALUTAZIONE DI PORTAFOGLIO E MAIN
# =============================================================================

def valuta_portafoglio(scenari_df, storico_allineato, asset_order, pesi_dict, livelli=(0.95, 0.99)):
    pesi = np.array([pesi_dict[a] for a in asset_order])
    pnl_reale = storico_allineato @ pesi
    pnl_sintetico = scenari_df[asset_order].values @ pesi

    print("\n" + "=" * 65)
    print("VALUTAZIONE DI PORTAFOGLIO (accoppiamento Schaake shuffle)")
    print("=" * 65)
    print(f"{'Metrica':<15} | {'Storico (reale OOS)':<20} | {'Sintetico (accoppiato)':<22}")
    print("-" * 65)
    print(f"{'Kurtosi':<15} | {pd.Series(pnl_reale).kurtosis():<20.4f} | {pd.Series(pnl_sintetico).kurtosis():<22.4f}")

    for liv in livelli:
        var_r, es_r = calcola_var_es(pnl_reale, liv)
        var_s, es_s = calcola_var_es(pnl_sintetico, liv)
        var_err = abs(var_s - var_r) / (abs(var_r) + 1e-10)
        es_err = abs(es_s - es_r) / (abs(es_r) + 1e-10)
        print(f"{'VaR ' + str(int(liv*100)) + '%':<15} | {var_r:<20.4f} | {var_s:<22.4f} (err: {var_err:.2%})")
        print(f"{'ES ' + str(int(liv*100)) + '%':<15} | {es_r:<20.4f} | {es_s:<22.4f} (err: {es_err:.2%})")

    return pnl_reale, pnl_sintetico

def valuta_dipendenza(scenari_df, storico_allineato, asset_order):
    corr_reale = pd.DataFrame(storico_allineato, columns=asset_order).corr()
    corr_sintetica = scenari_df[asset_order].corr()

    print("\n" + "-" * 65)
    print("MATRICE DI CORRELAZIONE: Storica (OOS) vs Sintetica (Accoppiata)")
    print("-" * 65)
    print("Storica:")
    print(corr_reale.round(3))
    print("\nSintetica:")
    print(corr_sintetica.round(3))


def main():
    print("=" * 65)
    print("PORTFOLIO TAIL RISK MODEL: N x (QuantGAN + EVT univariati)")
    print("                           + Copula Empirica (Schaake Shuffle)")
    print("=" * 65)

    asset_order = list(ASSET_FILES.keys())
    risultati_per_asset = {}

    # 1. Training Univariato per ogni Asset
    for nome_asset, path in ASSET_FILES.items():
        df = pd.read_csv(path)
        risultati_per_asset[nome_asset] = allena_e_genera_asset(nome_asset, df[PRICE_COL].values)

    # 2. Struttura di dipendenza storica
    storico_allineato, asset_order_check = costruisci_ranghi_storici(ASSET_FILES)
    storico_train_copula, storico_test_copula = split_storico_copula(storico_allineato)
    idx_test_copula = int(len(storico_allineato) * FRAZIONE_TRAIN_COPULA)

    # 3. Accoppiamento Copula (Schaake Shuffle)
    pools = {nome: risultati_per_asset[nome]['pool_marginale'] for nome in asset_order}
    scenari_df = schaake_shuffle_coupling(pools, storico_train_copula, asset_order, n_scenari=N_SCENARI_PORTAFOGLIO)

    # 4. Valutazione di Portafoglio (Test Out-of-Sample)
    pnl_reale, pnl_sintetico = valuta_portafoglio(scenari_df, storico_test_copula, asset_order, PORTFOLIO_WEIGHTS)
    valuta_dipendenza(scenari_df, storico_test_copula, asset_order)

    # 5. Generazione Dashboard di Portafoglio
    print("\nGenerazione grafici di Portafoglio in corso...")
    plotta_dashboard_portafoglio(
        storico_allineato, scenari_df, asset_order, PORTFOLIO_WEIGHTS, idx_test=idx_test_copula
    )

    plotta_10_finestre_portafoglio(
        storico_allineato, scenari_df, asset_order, PORTFOLIO_WEIGHTS,
        lunghezza_finestra=LUNGHEZZA_FINESTRA, idx_test_start=idx_test_copula
    )

if __name__ == "__main__":
    main()