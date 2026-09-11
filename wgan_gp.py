import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# CONFIGURAZIONE 
# =============================================================================

CSV_PATH = "/content/^nkx_d-2.csv"   # verifica che il nome/percorso coincida col file caricato
lunghezza_finestra = 63            # lunghezza delle sequenze di rendimenti
N_EPOCHE = 5000                    # tetto massimo di epoche; con early stopping si ferma prima se serve
BATCH_SIZE = 128
dim_latente = 64                   # dimensione del vettore di rumore in input al generatore
N_PATH_GENERATI = 300              # quanti path sintetici generare alla fine
N_CRITICO_PER_GENERATORE = 5       # aggiornamenti del critico per ogni aggiornamento del generatore
LAMBDA_GP = 15.0                   # peso della gradient penalty
lrC = 5e-5                         # learning rate del critico
lrG = 1e-4                         # learning rate del generatore
dimC_hidden = 256                  # capacita' (neuroni nascosti) del critico
dimG_hidden = 256                  # capacita' (neuroni nascosti) del generatore

# --- Early stopping ---
USA_EARLY_STOPPING = True
EPOCHE_MINIME = 500       # non fermarsi mai prima di queste epoche, anche se sembra convergere
VERIFICA_OGNI = 25        # ogni quante epoche controllare la distanza reale-vs-generato
PAZIENZA = 10             # quanti controlli consecutivi senza miglioramento prima di fermarsi
N_CAMPIONI_VERIFICA = 500  # quanti path sintetici generare ad ogni controllo per stimare la distanza

frazione_train = 0.85


# =============================================================================
# 1. Caricamento dati
# =============================================================================

def carica_da_csv(percorso_csv):
    """Carica un CSV con colonna 'Close' e calcola i rendimenti logaritmici."""
    df = pd.read_csv(percorso_csv)
    if "Close" not in df.columns:
        raise ValueError("Il CSV deve contenere una colonna 'Close' con i prezzi di chiusura")
    prezzi = df["Close"].values.astype(np.float64)
    rendimenti = np.diff(np.log(prezzi))
    return prezzi, rendimenti


def crea_finestre(rendimenti, lunghezza_finestra):
    # FIX: il parametro ora si chiama esattamente "lunghezza_finestra" (prima c'era un
    # typo, "lunghezza_finestr", che faceva usare per sbaglio la variabile globale
    # invece del valore passato come argomento).
    """Trasforma la serie di rendimenti in finestre sovrapposte di lunghezza fissa."""
    finestre = []
    for i in range(len(rendimenti) - lunghezza_finestra + 1):
        finestre.append(rendimenti[i:i + lunghezza_finestra])
    return np.array(finestre, dtype=np.float32)


def dividi_train_test(rendimenti, frazione_train):
    """
    Divide la serie di rendimenti in train/test in modo CRONOLOGICO:
    i dati piu' vecchi vanno in train, i piu' recenti in test.
    Non si fa uno split casuale perche' e' una serie storica: mescolare le date
    farebbe "vedere" al modello informazioni future durante il training (data leakage).
    """
    n_train = int(len(rendimenti) * frazione_train)
    rendimenti_train = rendimenti[:n_train]
    rendimenti_test = rendimenti[n_train:]
    return rendimenti_train, rendimenti_test


# =============================================================================
# 2. Architetture Generatore e Critico
# =============================================================================

class Generatore(nn.Module):
    """Prende un vettore di rumore latente e produce una sequenza di rendimenti."""

    def __init__(self, dim_latente, lunghezza_finestra, dimG_hidden):
        super().__init__()
        self.rete = nn.Sequential(
            nn.Linear(dim_latente, dimG_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(dimG_hidden, dimG_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(dimG_hidden, dimG_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(dimG_hidden, lunghezza_finestra),
            nn.Tanh(),
        )
        self.fattore_scala = 0.10  # rendimenti giornalieri plausibili entro +-10%

    def forward(self, z):
        return self.rete(z) * self.fattore_scala


class Critico(nn.Module):
    """Stima la 'qualita' Wasserstein' di una sequenza di rendimenti."""

    def __init__(self, lunghezza_finestra, dimC_hidden):
        super().__init__()
        self.rete = nn.Sequential(
            nn.Linear(lunghezza_finestra, dimC_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(dimC_hidden, dimC_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(dimC_hidden, dimC_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(dimC_hidden, 1),
        )

    def forward(self, x):
        return self.rete(x)


# =============================================================================
# 3. Gradient penalty
# =============================================================================

def calcola_gradient_penalty(critico, reali, generati, device):
    batch_size = reali.size(0)
    eps = torch.rand(batch_size, 1, device=device)
    eps = eps.expand_as(reali)

    interpolati = eps * reali + (1 - eps) * generati
    interpolati.requires_grad_(True)

    score_interpolati = critico(interpolati)

    gradienti = torch.autograd.grad(
        outputs=score_interpolati,
        inputs=interpolati,
        grad_outputs=torch.ones_like(score_interpolati),
        create_graph=True,
        retain_graph=True,
    )[0]

    norma_gradienti = gradienti.view(batch_size, -1).norm(2, dim=1)
    penalty = ((norma_gradienti - 1) ** 2).mean()
    return penalty


# =============================================================================
# 4. Training loop WGAN-GP
# =============================================================================

def calcola_distanza_reale_generato(rendimenti_riferimento_flat, generatore, dim_latente, device,
                                     n_campioni_verifica=500):
    
    generatore.eval()
    with torch.no_grad():
        z = torch.randn(n_campioni_verifica, dim_latente, device=device)
        rendimenti_generati = generatore(z).cpu().numpy().flatten()
    generatore.train()
    return wasserstein_distance(rendimenti_riferimento_flat, rendimenti_generati)


def allena_wgan_gp(dati_reali, dim_latente, lunghezza_finestra,
                    n_epoche, batch_size, lrC, lrG,
                    n_critico_per_generatore, lambda_gp, dimG_hidden, dimC_hidden,
                    rendimenti_verifica_flat=None,
                    device=None, verbose_ogni=200,
                    usa_early_stopping=True, epoche_minime=500,
                    verifica_ogni=50, pazienza=10, n_campioni_verifica=500):
    

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training su device: {device}")

    tensore_dati = torch.tensor(dati_reali, dtype=torch.float32, device=device)
    n_campioni = tensore_dati.size(0)

    if rendimenti_verifica_flat is None:
        rendimenti_verifica_flat = dati_reali.flatten()

    generatore = Generatore(dim_latente, lunghezza_finestra, dimG_hidden).to(device)
    critico = Critico(lunghezza_finestra, dimC_hidden).to(device)

    opt_g = optim.Adam(generatore.parameters(), lr=lrG, betas=(0.5, 0.9))
    opt_c = optim.Adam(critico.parameters(), lr=lrC, betas=(0.5, 0.9))

    storico_loss_c = []
    storico_loss_g = []
    storico_distanza = []  # (epoca, distanza) ad ogni controllo

    miglior_distanza = float("inf")
    miglior_stato_generatore = None
    miglior_epoca = 0
    controlli_senza_miglioramento = 0

    for epoca in range(1, n_epoche + 1):

        for _ in range(n_critico_per_generatore):
            idx = torch.randint(0, n_campioni, (batch_size,), device=device)
            batch_reale = tensore_dati[idx]

            z = torch.randn(batch_size, dim_latente, device=device)
            batch_generato = generatore(z).detach()

            score_reale = critico(batch_reale)
            score_generato = critico(batch_generato)
            gp = calcola_gradient_penalty(critico, batch_reale, batch_generato, device)

            loss_c = score_generato.mean() - score_reale.mean() + lambda_gp * gp

            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

        z = torch.randn(batch_size, dim_latente, device=device)
        batch_generato = generatore(z)
        loss_g = -critico(batch_generato).mean()

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        storico_loss_c.append(loss_c.item())
        storico_loss_g.append(loss_g.item())

        if epoca % verbose_ogni == 0 or epoca == 1:
            print(f"[Epoca {epoca:5d}] loss_critico={loss_c.item():.4f}  loss_generatore={loss_g.item():.4f}")

        # --- controllo periodico per l'early stopping ---
        if usa_early_stopping and epoca % verifica_ogni == 0:
            distanza = calcola_distanza_reale_generato(
                rendimenti_verifica_flat, generatore, dim_latente, device, n_campioni_verifica
            )
            storico_distanza.append((epoca, distanza))

            if distanza < miglior_distanza:
                miglior_distanza = distanza
                miglior_stato_generatore = copy.deepcopy(generatore.state_dict())
                miglior_epoca = epoca
                controlli_senza_miglioramento = 0
            else:
                controlli_senza_miglioramento += 1

            print(f"    -> [verifica epoca {epoca}] distanza reale-generato = {distanza:.6f} "
                  f"(migliore finora: {miglior_distanza:.6f} @ epoca {miglior_epoca})")

            if epoca >= epoche_minime and controlli_senza_miglioramento >= pazienza:
                print(f"\nEarly stopping attivato all'epoca {epoca}: "
                      f"nessun miglioramento per {pazienza} controlli consecutivi "
                      f"({pazienza * verifica_ogni} epoche). "
                      f"Ripristino i pesi del generatore migliore (epoca {miglior_epoca}).")
                generatore.load_state_dict(miglior_stato_generatore)
                break

    if usa_early_stopping and miglior_stato_generatore is not None:
        generatore.load_state_dict(miglior_stato_generatore)
        print(f"\nRipristinati i pesi del generatore migliore trovato all'epoca {miglior_epoca} "
              f"(distanza reale-generato = {miglior_distanza:.6f}).")

    return generatore, critico, storico_loss_c, storico_loss_g, storico_distanza


# =============================================================================
# 5. Generazione di path sintetici
# =============================================================================

def genera_path_sintetici(generatore, n_path, dim_latente, prezzo_iniziale, device):
    generatore.eval()
    with torch.no_grad():
        z = torch.randn(n_path, dim_latente, device=device)
        rendimenti_generati = generatore(z).cpu().numpy()

    path_prezzi = prezzo_iniziale * np.exp(np.cumsum(rendimenti_generati, axis=1))
    return rendimenti_generati, path_prezzi


# =============================================================================
# 6. Visualizzazione
# =============================================================================

def plotta_risultati(prezzi_reali, path_sintetici, storico_loss_c, storico_loss_g,
                      rendimenti_reali, rendimenti_generati, storico_distanza=None,
                      indice_split_train_test=None,
                      out_png="wgan_risultati.png"):

    n_righe, n_colonne = (3, 2) if storico_distanza else (2, 2)
    fig, assi = plt.subplots(n_righe, n_colonne, figsize=(14, 15 if storico_distanza else 10))

    assi[0, 0].plot(prezzi_reali, color="black")
    if indice_split_train_test is not None:
        assi[0, 0].axvline(indice_split_train_test, color="red", linestyle="--",
                            label="inizio test set")
        assi[0, 0].legend()
    assi[0, 0].set_title("Indice reale - tratteggio rosso = inizio test set")
    assi[0, 0].set_xlabel("Giorni")
    assi[0, 0].set_ylabel("Prezzo")

    for path in path_sintetici[:20]:
        assi[0, 1].plot(path, alpha=0.6)
    assi[0, 1].set_title("Path sintetici generati dalla WGAN-GP")
    assi[0, 1].set_xlabel("Giorni")
    assi[0, 1].set_ylabel("Prezzo simulato")

    assi[1, 0].plot(storico_loss_c, label="loss critico")
    assi[1, 0].plot(storico_loss_g, label="loss generatore")
    assi[1, 0].set_title("Andamento delle loss durante il training")
    assi[1, 0].set_xlabel("Epoca")
    assi[1, 0].legend()

    assi[1, 1].hist(rendimenti_reali, bins=60, alpha=0.5, density=True, label="reali")
    assi[1, 1].hist(rendimenti_generati.flatten(), bins=60, alpha=0.5, density=True,
                     label="generati")
    assi[1, 1].set_title("Distribuzione dei rendimenti: reali vs generati")
    assi[1, 1].legend()

    if storico_distanza:
        epoche_verifica = [e for e, _ in storico_distanza]
        valori_distanza = [d for _, d in storico_distanza]
        assi[2, 0].plot(epoche_verifica, valori_distanza, marker="o", markersize=3)
        assi[2, 0].set_title("Distanza reale-generato nel tempo (usata per l'early stopping)")
        assi[2, 0].set_xlabel("Epoca")
        assi[2, 0].set_ylabel("Distanza di Wasserstein")
        assi[2, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Grafico salvato in: {out_png}")

def calcola_var_es(rendimenti, livello_confidenza=0.99):
    """Calcola VaR ed ES per una serie di rendimenti."""
    # Assicurati che i rendimenti siano ordinati
    rendimenti_ordinati = np.sort(rendimenti.flatten())

    # Indice corrispondente al livello di confidenza per VaR
    idx_var = int(len(rendimenti_ordinati) * (1 - livello_confidenza))
    var = rendimenti_ordinati[idx_var]

    # Expected Shortfall: media dei rendimenti che superano il VaR (più estremi)
    es = rendimenti_ordinati[:idx_var].mean()

    return var, es


# =============================================================================
# 7. ESECUZIONE (parte principale, gira subito appena esegui la cella)
# =============================================================================

# 1. Dati
prezzi, rendimenti = carica_da_csv(CSV_PATH)
print(f"Caricati {len(prezzi)} prezzi da {CSV_PATH}")

# --- split cronologico train/test ---
rendimenti_train, rendimenti_test = dividi_train_test(rendimenti, frazione_train)
n_train = len(rendimenti_train)
print(f"Split cronologico: {n_train} rendimenti in train, {len(rendimenti_test)} in test "
      f"(frazione train = {frazione_train})")

finestre_train = crea_finestre(rendimenti_train, lunghezza_finestra)
print(f"Create {len(finestre_train)} finestre di training di lunghezza {lunghezza_finestra}")

if len(rendimenti_test) <= lunghezza_finestra:
    print("ATTENZIONE: il test set ha meno osservazioni della lunghezza finestra; "
          "la metrica di early stopping su singoli rendimenti resta comunque valida.")

# 2. Training WGAN-GP (allena SOLO su train; la distanza di early stopping usa il test set,
#    mai visto durante il training, per verificare che il generatore generalizzi davvero)
generatore, critico, loss_c, loss_g, storico_distanza = allena_wgan_gp(
    finestre_train,
    dim_latente,
    lunghezza_finestra,
    n_epoche=N_EPOCHE,
    batch_size=BATCH_SIZE,
    lrG=lrG, lrC=lrC,
    n_critico_per_generatore=N_CRITICO_PER_GENERATORE,
    lambda_gp=LAMBDA_GP,
    dimG_hidden=dimG_hidden, dimC_hidden=dimC_hidden,
    rendimenti_verifica_flat=rendimenti_test,
    usa_early_stopping=USA_EARLY_STOPPING,
    epoche_minime=EPOCHE_MINIME,
    verifica_ogni=VERIFICA_OGNI,
    pazienza=PAZIENZA,
    n_campioni_verifica=N_CAMPIONI_VERIFICA,
)

device = next(generatore.parameters()).device

# 3. Generazione di nuovi path sintetici dell'indice
rendimenti_generati, path_sintetici = genera_path_sintetici(
    generatore, n_path=N_PATH_GENERATI, dim_latente=dim_latente,
    prezzo_iniziale=prezzi[0], device=device,
)

# 4. Confronto statistico: generati vs TRAIN e vs TEST (fuori campione)
print("\n--- Confronto statistico: train (visto in training) vs test (fuori campione) vs generati ---")
print(f"{'':10s} {'train':>12s} {'test':>12s} {'generati':>12s}")
print(f"{'Media':10s} {rendimenti_train.mean():12.6f} {rendimenti_test.mean():12.6f} {rendimenti_generati.mean():12.6f}")
print(f"{'Std':10s} {rendimenti_train.std():12.6f} {rendimenti_test.std():12.6f} {rendimenti_generati.std():12.6f}")
print(f"{'Skew':10s} {pd.Series(rendimenti_train).skew():12.4f} {pd.Series(rendimenti_test).skew():12.4f} {pd.Series(rendimenti_generati.flatten()).skew():12.4f}")
print(f"{'Kurtosi':10s} {pd.Series(rendimenti_train).kurtosis():12.4f} {pd.Series(rendimenti_test).kurtosis():12.4f} {pd.Series(rendimenti_generati.flatten()).kurtosis():12.4f}")

livello_confidenza = 0.95  # 99% VaR e ES

# Calcola per il test set
var_test, es_test = calcola_var_es(rendimenti_test, livello_confidenza)

# Calcola per i dati generati
var_generati, es_generati = calcola_var_es(rendimenti_generati, livello_confidenza)

# Crea un DataFrame per visualizzare i risultati
dati_confronto_rischio = {
    'Metrica': ['VaR (95%)', 'ES (95%)'],
    'Test Reale': [var_test, es_test],
    'Generato WGAN-GP': [var_generati, es_generati]
}

df_confronto_rischio = pd.DataFrame(dati_confronto_rischio)
df_confronto_rischio['Differenza Assoluta'] = np.abs(df_confronto_rischio['Test Reale'] - df_confronto_rischio['Generato WGAN-GP'])

display(df_confronto_rischio)

# 5. Grafici
plotta_risultati(prezzi, path_sintetici, loss_c, loss_g, rendimenti, rendimenti_generati,
                  storico_distanza=storico_distanza, indice_split_train_test=n_train)