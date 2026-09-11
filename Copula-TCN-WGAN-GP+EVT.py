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

# Un file CSV per asset. Sostituisci con i tuoi path reali.
ASSET_FILES = {
    "NKX":     "/content/^nkx_d-2.csv",
    "NDX":     "/content/^ndx_d copia.csv",
    "HSI":     "/content/^hsi_d.csv",
    "FTSEmib": "/content/^fmib_d.csv",
}
PRICE_COL = "Close"
DATE_COL = "Date"

# Pesi di portafoglio (devono sommare a 1). Modifica a piacere.
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
N_PATH_GENERATI = 300
FRAZIONE_TRAIN = 0.85
SOGLIA_CODA_EVT = 0.05

USA_EARLY_STOPPING = True
EPOCHE_MINIME = 500
VERIFICA_OGNI = 25
PAZIENZA = 40
N_CAMPIONI_VERIFICA = 500

# Numero di scenari di portafoglio da generare per la valutazione finale
N_SCENARI_PORTAFOGLIO = 5000


FRAZIONE_TRAIN_COPULA = 0.75

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Hardware in uso: {DEVICE}")


# =============================================================================
# 2. MODULO QUANTGAN UNIVARIATO (TCN) - invariato rispetto alla versione
#    gia' validata, solo parametrizzato per essere richiamabile per asset
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
        self.rete = nn.Sequential(
            nn.Conv1d(canali_in, canali_out, kernel_size, padding=padding, dilation=dilation),
            ChompCausale(padding), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Conv1d(canali_out, canali_out, kernel_size, padding=padding, dilation=dilation),
            ChompCausale(padding), nn.LeakyReLU(0.2), nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(canali_in, canali_out, 1) if canali_in != canali_out else None
        self.attivazione_finale = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.attivazione_finale(self.rete(x) + (x if self.downsample is None else self.downsample(x)))


class TCN(nn.Module):
    def __init__(self, canali_in, n_livelli, canali_nascosti, kernel_size, dropout):
        super().__init__()
        livelli = []
        for i in range(n_livelli):
            livelli.append(BloccoResidualeTCN(canali_in if i == 0 else canali_nascosti,
                                               canali_nascosti, kernel_size, 2**i, dropout))
        self.rete = nn.Sequential(*livelli)

    def forward(self, x):
        return self.rete(x)


class GeneratoreTCN(nn.Module):
    def __init__(self, dim_latente):
        super().__init__()
        self.tcn = TCN(dim_latente, 6, 64, 3, 0.1)
        self.proj = nn.Conv1d(64, 1, 1)

    def forward(self, z):
        return self.proj(self.tcn(z)).squeeze(1)


class CriticoTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = TCN(1, 6, 64, 3, 0.1)
        self.testa = nn.Linear(64, 1)

    def forward(self, x):
        return self.testa(self.tcn(x.unsqueeze(1)).mean(dim=2))


def calcola_distanza_reale_generato(rendimenti_riferimento_flat, generatore, dim_latente, lunghezza_finestra):
    generatore.eval()
    with torch.no_grad():
        z = torch.randn(N_CAMPIONI_VERIFICA, dim_latente, lunghezza_finestra, device=DEVICE)
        rendimenti_generati = generatore(z).cpu().numpy().flatten()
    generatore.train()
    return wasserstein_distance(rendimenti_riferimento_flat, rendimenti_generati)


def allena_quantgan(rendimenti_train_std, rendimenti_test_std, dim_latente, lunghezza_finestra, epoche, name="Asset"):
    print(f"\n--- Training QuantGAN (TCN) - {name} ---")
    finestre = np.array([rendimenti_train_std[i:i + lunghezza_finestra]
                          for i in range(len(rendimenti_train_std) - lunghezza_finestra + 1)])
    tensore_dati = torch.tensor(finestre, dtype=torch.float32, device=DEVICE)

    G = GeneratoreTCN(dim_latente).to(DEVICE)
    D = CriticoTCN().to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=5e-5, betas=(0.5, 0.9))

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

        if USA_EARLY_STOPPING and (ep + 1) % VERIFICA_OGNI == 0:
            distanza = calcola_distanza_reale_generato(rendimenti_test_std, G, dim_latente, lunghezza_finestra)
            if distanza < miglior_distanza:
                miglior_distanza = distanza
                miglior_stato_G = copy.deepcopy(G.state_dict())
                miglior_epoca = ep + 1
                controlli_senza_miglioramento = 0
            else:
                controlli_senza_miglioramento += 1

            print(f"[{name}] Epoca {ep+1}/{epoche} | W-Dist Test: {distanza:.6f} (Best: {miglior_distanza:.6f} @ {miglior_epoca})")

            if (ep + 1) >= EPOCHE_MINIME and controlli_senza_miglioramento >= PAZIENZA:
                print(f"[{name}] Early stopping attivato. Ripristino pesi dell'epoca {miglior_epoca}.")
                G.load_state_dict(miglior_stato_G)
                break

    if USA_EARLY_STOPPING and miglior_stato_G is not None:
        G.load_state_dict(miglior_stato_G)

    return G


# =============================================================================
# 3. MODULO SPLICING GPD UNIVARIATO (EVT PURA) - invariato
# =============================================================================

def splicing_univariato_gpd(rendimenti_quant, dati_reali, soglia_coda=0.05, name="Asset"):
    print(f"\n--- Avvio Splicing Univariato GPD [{name}] (Coda: {soglia_coda*100}%) ---")
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

    print(f"Sostituzione di {n_estremi} giorni di alta volatilità TCN con crolli puri GPD...")

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
# 4. TRAINING PER-ASSET: incapsula i passi B/C/D del main originale in una
#    funzione richiamabile per ciascun asset del portafoglio
# =============================================================================

def allena_e_genera_asset(nome_asset, log_ret_asset):
    """
    Allena il modello ibrido univariato per UN asset e ritorna
    un pool ampio di rendimenti generati (marginale, non ancora accoppiato
    con gli altri asset).
    """
    n_train = int(len(log_ret_asset) * FRAZIONE_TRAIN)
    r_train, r_test = log_ret_asset[:n_train], log_ret_asset[n_train:]

    m0, s0 = r_train.mean(), r_train.std()

    qg = allena_quantgan((r_train - m0) / s0, (r_test - m0) / s0,
                          DIM_LATENTE_QG, LUNGHEZZA_FINESTRA, EPOCHE_QUANTGAN,
                          nome_asset)

    qg.eval()
    with torch.no_grad():
        z = torch.randn(N_PATH_GENERATI, DIM_LATENTE_QG, LUNGHEZZA_FINESTRA, device=DEVICE)
        rend_std = qg(z).cpu().numpy()
    rend_q = (rend_std * s0) + m0

    rend_ibrido = splicing_univariato_gpd(rend_q, r_train, SOGLIA_CODA_EVT, nome_asset)

    return {
        'r_train': r_train,
        'r_test': r_test,
        'rend_quantgan': rend_q,
        'rend_ibrido': rend_ibrido,          # (N_PATH_GENERATI, LUNGHEZZA_FINESTRA)
        'pool_marginale': rend_ibrido.flatten(),  # pool piatto di draw i.i.d.
    }


# =============================================================================
# 5. ACCOPPIAMENTO VIA COPULA EMPIRICA (SCHAAKE SHUFFLE)
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

    print(f"Righe allineate per data (per la copula): {len(merged)}")

    prices = merged[asset_names].values
    log_returns = np.diff(np.log(prices + 1e-10), axis=0)
    mask = ~np.any(np.isnan(log_returns) | np.isinf(log_returns), axis=1)
    log_returns = log_returns[mask]

    return log_returns, asset_names  # (T_hist, n_assets)


def split_storico_copula(storico_allineato, frazione_train=FRAZIONE_TRAIN_COPULA):
    
    n = len(storico_allineato)
    n_train = int(n * frazione_train)
    storico_train_copula = storico_allineato[:n_train]
    storico_test_copula = storico_allineato[n_train:]
    print(f"Split storico copula: train={len(storico_train_copula)} giorni "
          f"(usati per i ranghi dello shuffle), "
          f"test={len(storico_test_copula)} giorni (mai visti, solo per la valutazione)")
    return storico_train_copula, storico_test_copula


def schaake_shuffle_coupling(pools_marginali, storico_allineato, asset_order, n_scenari, seed=42):
    """
    Accoppia i pool marginali generati indipendentemente usando la struttura
    di dipendenza di rango osservata storicamente (Schaake shuffle).

    pools_marginali: dict {asset_name: array 1D di draw generati (i.i.d.)}
    storico_allineato: (T_hist, n_assets) rendimenti storici reali allineati
                        per data (stesso ordine colonne di asset_order)
    asset_order: lista dei nomi asset, nello stesso ordine delle colonne di
                 storico_allineato
    n_scenari: quanti scenari di portafoglio generare

    Ritorna: DataFrame (n_scenari, n_assets) di rendimenti sintetici accoppiati
    """
    rng = np.random.default_rng(seed)
    n_assets = len(asset_order)
    T_hist = storico_allineato.shape[0]

    # Rango storico di ciascun asset per ogni giorno (0 = valore piu' basso)
    ranghi_storici = np.zeros_like(storico_allineato)
    for j in range(n_assets):
        ranghi_storici[:, j] = pd.Series(storico_allineato[:, j]).rank(method='first').values - 1

    # Campiona n_scenari "giorni template" dalla storia (con reimmissione se
    # n_scenari > T_hist)
    giorni_idx = rng.choice(T_hist, size=n_scenari, replace=(n_scenari > T_hist))

    scenari = np.zeros((n_scenari, n_assets))
    for j, asset in enumerate(asset_order):
        pool_ordinato = np.sort(pools_marginali[asset])
        n_pool = len(pool_ordinato)

        # posizione del rango storico rimappata nella scala del pool generato
        rank_giorni = ranghi_storici[giorni_idx, j]
        q_pos = np.round(rank_giorni / (T_hist - 1) * (n_pool - 1)).astype(int)
        q_pos = np.clip(q_pos, 0, n_pool - 1)

        scenari[:, j] = pool_ordinato[q_pos]

    return pd.DataFrame(scenari, columns=asset_order)


# =============================================================================
# 6. VALUTAZIONE DI PORTAFOGLIO
# =============================================================================

def valuta_portafoglio(scenari_df, storico_allineato, asset_order, pesi_dict, livelli=(0.95, 0.99)):
    pesi = np.array([pesi_dict[a] for a in asset_order])
    assert abs(pesi.sum() - 1.0) < 1e-6, "I pesi di portafoglio devono sommare a 1."

    pnl_reale = storico_allineato @ pesi
    pnl_sintetico = scenari_df[asset_order].values @ pesi

    print("\n" + "=" * 65)
    print("VALUTAZIONE DI PORTAFOGLIO (accoppiamento Schaake shuffle)")
    print("=" * 65)
    print(f"Pesi: {pesi_dict}")
    print(f"{'Metrica':<15} | {'Storico (reale)':<18} | {'Sintetico (accoppiato)':<22}")
    print("-" * 65)
    print(f"{'Kurtosi':<15} | {pd.Series(pnl_reale).kurtosis():<18.4f} | "
          f"{pd.Series(pnl_sintetico).kurtosis():<22.4f}")

    for liv in livelli:
        var_r, es_r = calcola_var_es(pnl_reale, liv)
        var_s, es_s = calcola_var_es(pnl_sintetico, liv)
        var_err = abs(var_s - var_r) / (abs(var_r) + 1e-10)
        es_err = abs(es_s - es_r) / (abs(es_r) + 1e-10)
        print(f"{'VaR ' + str(int(liv*100)) + '%':<15} | {var_r:<18.4f} | {var_s:<22.4f}  "
              f"(errore: {var_err:.2%})")
        print(f"{'ES ' + str(int(liv*100)) + '%':<15} | {es_r:<18.4f} | {es_s:<22.4f}  "
              f"(errore: {es_err:.2%})")

    return pnl_reale, pnl_sintetico


def valuta_dipendenza(scenari_df, storico_allineato, asset_order):
    """
    Confronto rapido: la matrice di correlazione storica vs quella dei
    rendimenti sintetici accoppiati. Non prova esplicitamente la dipendenza
    di CODA (per quella servirebbero le funzioni extremal_coefficient /
    dependence_score gia' presenti negli script WA-GAN), ma da' un primo
    controllo di sanita' sulla dipendenza media.
    """
    corr_reale = pd.DataFrame(storico_allineato, columns=asset_order).corr()
    corr_sintetica = scenari_df[asset_order].corr()

    print("\n" + "-" * 65)
    print("CORRELAZIONE: storica vs sintetica (accoppiata)")
    print("-" * 65)
    print("Storica:")
    print(corr_reale.round(3))
    print("\nSintetica:")
    print(corr_sintetica.round(3))


# =============================================================================
# 7. ORCHESTRAZIONE MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 65)
    print("PORTFOLIO TAIL RISK MODEL: N x QuantGAN+EVT univariati")
    print("                            + copula empirica (Schaake shuffle)")
    print("=" * 65)

    asset_order = list(ASSET_FILES.keys())

    # --- FASE 1: training univariato per ciascun asset (usa tutto lo
    #     storico disponibile di ciascuno, non allineato) ---
    risultati_per_asset = {}
    for nome_asset, path in ASSET_FILES.items():
        df = pd.read_csv(path)
        prices = df[PRICE_COL].values
        log_ret = np.diff(np.log(prices + 1e-10))
        mask = ~np.isnan(log_ret) & ~np.isinf(log_ret)
        log_ret = log_ret[mask]

        risultati_per_asset[nome_asset] = allena_e_genera_asset(nome_asset, log_ret)

    # --- FASE 2: struttura di dipendenza storica (allineata per data) ---
    storico_allineato, asset_order_check = costruisci_ranghi_storici(ASSET_FILES)
    assert asset_order_check == asset_order, "Ordine asset incoerente tra training e allineamento."

    # FIX: split temporale. Solo la parte "train" alimenta lo shuffle; la
    # parte "test" e' l'unico storico usato per il confronto finale, e non
    # viene mai vista dallo shuffle - cosi' il test sulla dipendenza e'
    # genuinamente out-of-sample, non circolare.
    storico_train_copula, storico_test_copula = split_storico_copula(storico_allineato)

    # --- FASE 3: accoppiamento Schaake shuffle (ranghi presi SOLO dal
    #     blocco train della copula) ---
    pools = {nome: risultati_per_asset[nome]['pool_marginale'] for nome in asset_order}
    scenari_df = schaake_shuffle_coupling(pools, storico_train_copula, asset_order,
                                           n_scenari=N_SCENARI_PORTAFOGLIO)

    # --- FASE 4: valutazione marginale (per asset, come nello script originale) ---
    print("\n" + "=" * 65)
    print("RISULTATI MARGINALI PER ASSET (invariati rispetto al modello univariato)")
    print("=" * 65)
    for nome_asset in asset_order:
        r = risultati_per_asset[nome_asset]
        flat_reale = r['r_train']
        flat_qg = r['rend_quantgan'].flatten()
        flat_ibrido = r['rend_ibrido'].flatten()

        print(f"\n--- {nome_asset} ---")
        print(f"{'Metrica':<15} | {'Reale':<12} | {'QuantGAN':<12} | {'Ibrido (QG+GPD)':<15}")
        print(f"{'Kurtosi':<15} | {pd.Series(flat_reale).kurtosis():<12.4f} | "
              f"{pd.Series(flat_qg).kurtosis():<12.4f} | {pd.Series(flat_ibrido).kurtosis():<15.4f}")
        var_r, es_r = calcola_var_es(flat_reale, 0.99)
        var_i, es_i = calcola_var_es(flat_ibrido, 0.99)
        print(f"VaR 99%: reale={var_r:.4f}, ibrido={var_i:.4f} "
              f"(errore {abs(var_i-var_r)/abs(var_r):.2%})")
        print(f"ES 99%:  reale={es_r:.4f}, ibrido={es_i:.4f} "
              f"(errore {abs(es_i-es_r)/abs(es_r):.2%})")

    # --- FASE 5: valutazione di portafoglio (SOLO contro lo storico test,
    #     mai visto dallo shuffle - confronto genuinamente out-of-sample) ---
    pnl_reale, pnl_sintetico = valuta_portafoglio(
        scenari_df, storico_test_copula, asset_order, PORTFOLIO_WEIGHTS
    )

    # --- FASE 6: controllo di sanita' sulla dipendenza (idem, out-of-sample) ---
    valuta_dipendenza(scenari_df, storico_test_copula, asset_order)

    # --- FASE 7: grafico ---
    plt.figure(figsize=(12, 5))
    sns.kdeplot(pnl_reale, label='Portafoglio storico (reale)', color='black', lw=2)
    sns.kdeplot(pnl_sintetico, label='Portafoglio sintetico (accoppiato)', color='red')
    plt.title("Distribuzione P&L di Portafoglio: Storico vs Sintetico")
    plt.legend()
    plt.show()

    return risultati_per_asset, scenari_df, pnl_reale, pnl_sintetico


if __name__ == "__main__":
    risultati_per_asset, scenari_df, pnl_reale, pnl_sintetico = main()