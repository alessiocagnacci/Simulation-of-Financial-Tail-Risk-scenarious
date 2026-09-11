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
CSV_PATH = "/content/^nkx_d-2.csv"  # Percorso del singolo asset

LUNGHEZZA_FINESTRA = 63
EPOCHE_QUANTGAN = 5000  # Early stopping lo fermerà prima
BATCH_SIZE_QG = 64
DIM_LATENTE_QG = 16
N_PATH_GENERATI = 300
FRAZIONE_TRAIN = 0.85
SOGLIA_CODA_EVT = 0.05  # Splicing sul 5% peggiore dei dati

# Early Stopping QuantGAN
USA_EARLY_STOPPING = True
EPOCHE_MINIME = 500
VERIFICA_OGNI = 25
PAZIENZA = 40
N_CAMPIONI_VERIFICA = 500

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
            livelli.append(BloccoResidualeTCN(canali_in if i == 0 else canali_nascosti, canali_nascosti, kernel_size, 2**i, dropout))
        self.rete = nn.Sequential(*livelli)
    def forward(self, x): return self.rete(x)

class GeneratoreTCN(nn.Module):
    def __init__(self, dim_latente):
        super().__init__()
        self.tcn = TCN(dim_latente, 6, 64, 3, 0.1)
        self.proj = nn.Conv1d(64, 1, 1)
    def forward(self, z): return self.proj(self.tcn(z)).squeeze(1)

class CriticoTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = TCN(1, 6, 64, 3, 0.1)
        self.testa = nn.Linear(64, 1)
    def forward(self, x): return self.testa(self.tcn(x.unsqueeze(1)).mean(dim=2))

def calcola_distanza_reale_generato(rendimenti_riferimento_flat, generatore, dim_latente, lunghezza_finestra):
    generatore.eval()
    with torch.no_grad():
        z = torch.randn(N_CAMPIONI_VERIFICA, dim_latente, lunghezza_finestra, device=DEVICE)
        rendimenti_generati = generatore(z).cpu().numpy().flatten()
    generatore.train()
    return wasserstein_distance(rendimenti_riferimento_flat, rendimenti_generati)

def allena_quantgan(rendimenti_train_std, rendimenti_test_std, dim_latente, lunghezza_finestra, epoche, name="Asset"):
    print(f"\n--- Training QuantGAN (TCN) - {name} ---")
    finestre = np.array([rendimenti_train_std[i:i + lunghezza_finestra] for i in range(len(rendimenti_train_std) - lunghezza_finestra + 1)])
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
            grad = torch.autograd.grad(outputs=D(interp), inputs=interp, grad_outputs=torch.ones_like(D(interp)), create_graph=True, retain_graph=True)[0]
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
# 3. MODULO SPLICING GPD UNIVARIATO (EVT PURA)
# =============================================================================

def splicing_univariato_gpd(rendimenti_quant, dati_reali, soglia_coda=0.05, name="Asset"):
    
    print(f"\n--- Avvio Splicing Univariato GPD [{name}] (Coda: {soglia_coda*100}%) ---")
    n_path, seq_len = rendimenti_quant.shape
    rend_flat = rendimenti_quant.flatten().copy()
    
    # 1. FIT GPD SUI DATI REALI
    loss_reali = -dati_reali
    threshold_empirico = np.quantile(loss_reali, 1 - soglia_coda)
    eccessi = loss_reali[loss_reali > threshold_empirico] - threshold_empirico
    xi, loc, beta = genpareto.fit(eccessi, floc=0)
    
    # 2. INDIVIDUAZIONE CLUSTER NEL QUANTGAN
    loss_quant = -rend_flat
    soglia_quant = np.quantile(loss_quant, 1 - soglia_coda)
    indici_estremi = np.where(loss_quant > soglia_quant)[0]
    n_estremi = len(indici_estremi)
    
    print(f"Sostituzione di {n_estremi} giorni di alta volatilità TCN con crolli puri GPD...")
    
    # 3. GENERAZIONE CROLLI PERFETTI
    u = np.random.uniform(0, 1, size=n_estremi)
    nuovi_eccessi = genpareto.ppf(u, xi, loc=0, scale=beta)
    nuove_loss_estreme = threshold_empirico + nuovi_eccessi
    
    # 4. SPLICING CON RANK-ORDERING
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
# 4. ORCHESTRAZIONE MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 65)
    print("HYBRID TAIL RISK MODEL: QuantGAN (Bulk) + Univariate GPD (Tails)")
    print("=" * 65)

    # A. Caricamento dati (Singolo Asset)
    df = pd.read_csv(CSV_PATH)
    prices = df['Close'].values
    log_ret = np.diff(np.log(prices + 1e-10))
    
    mask = ~np.isnan(log_ret) & ~np.isinf(log_ret)
    log_ret = log_ret[mask]
    
    n_train = int(len(log_ret) * FRAZIONE_TRAIN)
    r_train, r_test = log_ret[:n_train], log_ret[n_train:]
    
    m0, s0 = r_train.mean(), r_train.std()
    
    # B. Addestramento QuantGAN
    qg = allena_quantgan((r_train - m0)/s0, (r_test - m0)/s0, DIM_LATENTE_QG, LUNGHEZZA_FINESTRA, EPOCHE_QUANTGAN, "Asset Principale")

    # C. Inferenza QuantGAN (Generazione path base)
    print(f"\n--- Generazione path base QuantGAN ---")
    qg.eval()
    with torch.no_grad():
        z = torch.randn(N_PATH_GENERATI, DIM_LATENTE_QG, LUNGHEZZA_FINESTRA, device=DEVICE)
        rend_std = qg(z).cpu().numpy()
        
    rend_q = (rend_std * s0) + m0

    # D. Splicing GPD Univariato
    rend_ibrido = splicing_univariato_gpd(rend_q, r_train, SOGLIA_CODA_EVT, "Asset Principale")

    # E. Metriche e Confronti
    print("\n" + "=" * 65)
    print("RISULTATI FINALI (ASSET PRINCIPALE)")
    print("=" * 65)
    
    flat_reale = r_train
    flat_qg = rend_q.flatten()
    flat_ibrido = rend_ibrido.flatten()
    
    print(f"{'Metrica':<15} | {'Dati Reali':<12} | {'Solo QuantGAN':<15} | {'Spliced (QG+GPD)':<15}")
    print("-" * 65)
    
    print(f"{'Kurtosi':<15} | {pd.Series(flat_reale).kurtosis():<12.4f} | {pd.Series(flat_qg).kurtosis():<15.4f} | {pd.Series(flat_ibrido).kurtosis():<15.4f}")
    
    # Calcolo al 95% (soglia di giunzione)
    var_r_95, es_r_95 = calcola_var_es(flat_reale, 0.95)
    var_q_95, es_q_95 = calcola_var_es(flat_qg, 0.95)
    var_i_95, es_i_95 = calcola_var_es(flat_ibrido, 0.95)
    print(f"{'VaR (95%)':<15} | {var_r_95:<12.4f} | {var_q_95:<15.4f} | {var_i_95:<15.4f}")
    print(f"{'ES (95%)':<15} | {es_r_95:<12.4f} | {es_q_95:<15.4f} | {es_i_95:<15.4f}")

    # Calcolo al 99% (cuore della coda EVT)
    var_r_99, es_r_99 = calcola_var_es(flat_reale, 0.99)
    var_q_99, es_q_99 = calcola_var_es(flat_qg, 0.99)
    var_i_99, es_i_99 = calcola_var_es(flat_ibrido, 0.99)
    print(f"{'VaR (99%)':<15} | {var_r_99:<12.4f} | {var_q_99:<15.4f} | {var_i_99:<15.4f}")
    print(f"{'ES (99%)':<15} | {es_r_99:<12.4f} | {es_q_99:<15.4f} | {es_i_99:<15.4f}")
    
    # Grafico distribuzioni code
    plt.figure(figsize=(12, 5))
    sns.kdeplot(flat_reale, label='Dati Reali (Train)', log_scale=(False, True), color='black', lw=2)
    sns.kdeplot(flat_qg, label='Solo QuantGAN (Code Deboli)', log_scale=(False, True), color='blue', linestyle='--')
    sns.kdeplot(flat_ibrido, label='Spliced Ibrido (Code perfette GPD)', log_scale=(False, True), color='red')
    plt.title("Distribuzione delle Code (Log-Scale) - Asset Principale")
    plt.xlim(-0.15, 0.15)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()