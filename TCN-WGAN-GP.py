import copy
import os
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
lunghezza_finestra = 63
N_EPOCHE = 5000                    # tetto massimo; con early stopping si ferma prima se serve
BATCH_SIZE = 64                    # abbassato rispetto alla versione MLP: le conv 1D con piu'
                                    # livelli costano di piu' per batch, soprattutto su CPU
dim_latente = 16                   # canali di rumore PER OGNI GIORNO della finestra (non piu' un
                                    # singolo vettore per l'intera sequenza). 8-16 e' un buon punto
                                    # di partenza per una TCN; valori piu' alti rallentano il training
                                    # senza benefici garantiti.
N_PATH_GENERATI = 300
N_CRITICO_PER_GENERATORE = 5
LAMBDA_GP = 10.0
lrC = 5e-5
lrG = 1e-4

# --- Architettura TCN ---
N_LIVELLI_TCN = 6          # numero di blocchi residui; ognuno raddoppia la dilatazione (1,2,4,8,16,32)
                            # con kernel_size=3 questo da' un campo recettivo di ~250 giorni,
                            # ampiamente sufficiente per lunghezza_finestra=63
CANALI_TCN = 64             # numero di canali (larghezza) di ciascun livello TCN
KERNEL_SIZE_TCN = 3
DROPOUT_TCN = 0.1

# --- Early stopping ---
USA_EARLY_STOPPING = True
EPOCHE_MINIME = 500
VERIFICA_OGNI = 25
PAZIENZA = 10
N_CAMPIONI_VERIFICA = 500

frazione_train = 0.85

# --- Checkpoint su Google Drive ---
USA_CHECKPOINT = False
CHECKPOINT_PATH = "/content/drive/MyDrive/wgan_tcn2_checkpoint.pt"   # nome diverso dalla versione
                                                                     # MLP: le architetture non sono
                                                                     # compatibili, non mischiare i checkpoint
CHECKPOINT_OGNI = 25


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
    """Trasforma la serie di rendimenti in finestre sovrapposte di lunghezza fissa."""
    finestre = []
    for i in range(len(rendimenti) - lunghezza_finestra + 1):
        finestre.append(rendimenti[i:i + lunghezza_finestra])
    return np.array(finestre, dtype=np.float32)


def dividi_train_test(rendimenti, frazione_train):
    """
    Divide la serie di rendimenti in train/test in modo CRONOLOGICO:
    i dati piu' vecchi vanno in train, i piu' recenti in test.
    """
    n_train = int(len(rendimenti) * frazione_train)
    return rendimenti[:n_train], rendimenti[n_train:]


def standardizza(rendimenti, media, std):
    """Applica (x - media) / std. Media e std vanno SEMPRE stimate solo sul training set."""
    return (rendimenti - media) / std


def destandardizza(rendimenti_std, media, std):
    """Inverte la standardizzazione per tornare alla scala reale dei rendimenti."""
    return rendimenti_std * std + media


# =============================================================================
# 2. Architetture Generatore e Critico (TCN)
# =============================================================================

class ChompCausale(nn.Module):
    """
    Rimuove gli ultimi 'chomp_size' passi temporali introdotti dal padding simmetrico
    di nn.Conv1d, in modo che la convoluzione sia CAUSALE: l'output al tempo t dipende
    solo da input a tempi <= t, mai dal futuro. Blocco standard delle TCN.
    """

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class BloccoResidualeTCN(nn.Module):
    
    def __init__(self, canali_in, canali_out, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(canali_in, canali_out, kernel_size,
                                padding=padding, dilation=dilation)
        self.chomp1 = ChompCausale(padding)
        self.attivazione1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(canali_out, canali_out, kernel_size,
                                padding=padding, dilation=dilation)
        self.chomp2 = ChompCausale(padding)
        self.attivazione2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(dropout)

        self.rete = nn.Sequential(
            self.conv1, self.chomp1, self.attivazione1, self.dropout1,
            self.conv2, self.chomp2, self.attivazione2, self.dropout2,
        )

        # se il numero di canali cambia, serve una proiezione 1x1 per la connessione residua
        self.downsample = (nn.Conv1d(canali_in, canali_out, 1)
                            if canali_in != canali_out else None)
        self.attivazione_finale = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.rete(x)
        residuo = x if self.downsample is None else self.downsample(x)
        return self.attivazione_finale(out + residuo)


class TCN(nn.Module):
    """Sequenza di blocchi residui con dilatazione che raddoppia ad ogni livello (1,2,4,8,...)."""

    def __init__(self, canali_in, n_livelli, canali_nascosti, kernel_size, dropout):
        super().__init__()
        livelli = []
        for i in range(n_livelli):
            dilation = 2 ** i
            in_ch = canali_in if i == 0 else canali_nascosti
            livelli.append(BloccoResidualeTCN(in_ch, canali_nascosti, kernel_size, dilation, dropout))
        self.rete = nn.Sequential(*livelli)

    def forward(self, x):
        return self.rete(x)


class Generatore(nn.Module):
    """
    Prende rumore latente PER-TIMESTEP (batch, dim_latente, lunghezza_finestra) e produce
    una sequenza di rendimenti standardizzati (batch, lunghezza_finestra), SENZA output
    vincolato (niente Tanh): lascia alla rete la liberta' di generare code pesanti.
    """

    def __init__(self, dim_latente, n_livelli, canali_nascosti, kernel_size, dropout):
        super().__init__()
        self.tcn = TCN(dim_latente, n_livelli, canali_nascosti, kernel_size, dropout)
        self.proiezione_finale = nn.Conv1d(canali_nascosti, 1, kernel_size=1)

    def forward(self, z):
        # z: (batch, dim_latente, lunghezza_finestra)
        out = self.tcn(z)                    # (batch, canali_nascosti, lunghezza_finestra)
        out = self.proiezione_finale(out)     # (batch, 1, lunghezza_finestra)
        return out.squeeze(1)                 # (batch, lunghezza_finestra)


class Critico(nn.Module):
    """
    Stima la 'qualita' Wasserstein' di una sequenza di rendimenti (standardizzati).
    Riceve (batch, lunghezza_finestra), applica una TCN, poi fa un average pooling
    lungo il tempo e produce uno score scalare. Niente BatchNorm: interferisce con
    la gradient penalty di WGAN-GP.
    """

    def __init__(self, n_livelli, canali_nascosti, kernel_size, dropout):
        super().__init__()
        self.tcn = TCN(1, n_livelli, canali_nascosti, kernel_size, dropout)
        self.testa = nn.Linear(canali_nascosti, 1)

    def forward(self, x):
        # x: (batch, lunghezza_finestra) -> aggiunge la dimensione dei canali (=1)
        x = x.unsqueeze(1)                    # (batch, 1, lunghezza_finestra)
        out = self.tcn(x)                     # (batch, canali_nascosti, lunghezza_finestra)
        out = out.mean(dim=2)                 # average pooling temporale -> (batch, canali_nascosti)
        return self.testa(out)                # (batch, 1)


# =============================================================================
# 3. Gradient penalty (identica alla versione MLP: agnostica rispetto all'architettura)
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

def calcola_distanza_reale_generato(rendimenti_riferimento_flat, generatore, dim_latente,
                                     lunghezza_finestra, device, n_campioni_verifica=500):
    
    generatore.eval()
    with torch.no_grad():
        z = torch.randn(n_campioni_verifica, dim_latente, lunghezza_finestra, device=device)
        rendimenti_generati = generatore(z).cpu().numpy().flatten()
    generatore.train()
    return wasserstein_distance(rendimenti_riferimento_flat, rendimenti_generati)


def _salva_checkpoint(checkpoint_path, epoca, generatore, critico, opt_g, opt_c,
                       storico_loss_c, storico_loss_g, storico_distanza,
                       miglior_distanza, miglior_stato_generatore, miglior_epoca,
                       controlli_senza_miglioramento):
    """Salva tutto cio' che serve per riprendere il training esattamente da qui."""
    cartella = os.path.dirname(checkpoint_path)
    if cartella:
        os.makedirs(cartella, exist_ok=True)
    torch.save({
        "epoca": epoca,
        "generatore": generatore.state_dict(),
        "critico": critico.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_c": opt_c.state_dict(),
        "storico_loss_c": storico_loss_c,
        "storico_loss_g": storico_loss_g,
        "storico_distanza": storico_distanza,
        "miglior_distanza": miglior_distanza,
        "miglior_stato_generatore": miglior_stato_generatore,
        "miglior_epoca": miglior_epoca,
        "controlli_senza_miglioramento": controlli_senza_miglioramento,
    }, checkpoint_path)


def allena_wgan_gp(dati_reali, dim_latente, lunghezza_finestra,
                    n_epoche, batch_size, lrC, lrG,
                    n_critico_per_generatore, lambda_gp,
                    n_livelli_tcn, canali_tcn, kernel_size_tcn, dropout_tcn,
                    rendimenti_verifica_flat=None,
                    device=None, verbose_ogni=200,
                    usa_early_stopping=True, epoche_minime=500,
                    verifica_ogni=50, pazienza=10, n_campioni_verifica=500,
                    usa_checkpoint=False, checkpoint_path=None, checkpoint_ogni=50):
    """
    dati_reali: finestre di training STANDARDIZZATE (usate per allenare critico e generatore).
    rendimenti_verifica_flat: rendimenti (1D, appiattiti, STANDARDIZZATI) MAI usati in
        training, usati solo per calcolare la distanza di Wasserstein per l'early stopping
        (tipicamente il test set).
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training su device: {device}")

    tensore_dati = torch.tensor(dati_reali, dtype=torch.float32, device=device)
    n_campioni = tensore_dati.size(0)

    if rendimenti_verifica_flat is None:
        rendimenti_verifica_flat = dati_reali.flatten()

    generatore = Generatore(dim_latente, n_livelli_tcn, canali_tcn, kernel_size_tcn, dropout_tcn).to(device)
    critico = Critico(n_livelli_tcn, canali_tcn, kernel_size_tcn, dropout_tcn).to(device)

    opt_g = optim.Adam(generatore.parameters(), lr=lrG, betas=(0.5, 0.9))
    opt_c = optim.Adam(critico.parameters(), lr=lrC, betas=(0.5, 0.9))

    storico_loss_c = []
    storico_loss_g = []
    storico_distanza = []

    miglior_distanza = float("inf")
    miglior_stato_generatore = None
    miglior_epoca = 0
    controlli_senza_miglioramento = 0
    epoca_iniziale = 1

    if usa_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Trovato checkpoint in {checkpoint_path}: riprendo il training da li'...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        generatore.load_state_dict(ckpt["generatore"])
        critico.load_state_dict(ckpt["critico"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_c.load_state_dict(ckpt["opt_c"])
        storico_loss_c = ckpt["storico_loss_c"]
        storico_loss_g = ckpt["storico_loss_g"]
        storico_distanza = ckpt["storico_distanza"]
        miglior_distanza = ckpt["miglior_distanza"]
        miglior_stato_generatore = ckpt["miglior_stato_generatore"]
        miglior_epoca = ckpt["miglior_epoca"]
        controlli_senza_miglioramento = ckpt["controlli_senza_miglioramento"]
        epoca_iniziale = ckpt["epoca"] + 1
        print(f"Ripreso dall'epoca {epoca_iniziale} "
              f"(miglior distanza finora: {miglior_distanza:.6f} @ epoca {miglior_epoca})")

    for epoca in range(epoca_iniziale, n_epoche + 1):

        for _ in range(n_critico_per_generatore):
            idx = torch.randint(0, n_campioni, (batch_size,), device=device)
            batch_reale = tensore_dati[idx]

            z = torch.randn(batch_size, dim_latente, lunghezza_finestra, device=device)
            batch_generato = generatore(z).detach()

            score_reale = critico(batch_reale)
            score_generato = critico(batch_generato)
            gp = calcola_gradient_penalty(critico, batch_reale, batch_generato, device)

            loss_c = score_generato.mean() - score_reale.mean() + lambda_gp * gp

            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

        z = torch.randn(batch_size, dim_latente, lunghezza_finestra, device=device)
        batch_generato = generatore(z)
        loss_g = -critico(batch_generato).mean()

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        storico_loss_c.append(loss_c.item())
        storico_loss_g.append(loss_g.item())

        if epoca % verbose_ogni == 0 or epoca == 1:
            print(f"[Epoca {epoca:5d}] loss_critico={loss_c.item():.4f}  loss_generatore={loss_g.item():.4f}")

        if usa_early_stopping and epoca % verifica_ogni == 0:
            distanza = calcola_distanza_reale_generato(
                rendimenti_verifica_flat, generatore, dim_latente, lunghezza_finestra,
                device, n_campioni_verifica
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
                if usa_checkpoint and checkpoint_path:
                    _salva_checkpoint(checkpoint_path, epoca, generatore, critico, opt_g, opt_c,
                                       storico_loss_c, storico_loss_g, storico_distanza,
                                       miglior_distanza, miglior_stato_generatore, miglior_epoca,
                                       controlli_senza_miglioramento)
                break

        if usa_checkpoint and checkpoint_path and epoca % checkpoint_ogni == 0:
            _salva_checkpoint(checkpoint_path, epoca, generatore, critico, opt_g, opt_c,
                               storico_loss_c, storico_loss_g, storico_distanza,
                               miglior_distanza, miglior_stato_generatore, miglior_epoca,
                               controlli_senza_miglioramento)

    if usa_early_stopping and miglior_stato_generatore is not None:
        generatore.load_state_dict(miglior_stato_generatore)
        print(f"\nRipristinati i pesi del generatore migliore trovato all'epoca {miglior_epoca} "
              f"(distanza reale-generato = {miglior_distanza:.6f}).")

    if usa_checkpoint and checkpoint_path:
        _salva_checkpoint(checkpoint_path, epoca, generatore, critico, opt_g, opt_c,
                           storico_loss_c, storico_loss_g, storico_distanza,
                           miglior_distanza, miglior_stato_generatore, miglior_epoca,
                           controlli_senza_miglioramento)
        print(f"Checkpoint finale salvato in {checkpoint_path}")

    return generatore, critico, storico_loss_c, storico_loss_g, storico_distanza


# =============================================================================
# 5. Generazione di path sintetici
# =============================================================================

def genera_path_sintetici(generatore, n_path, dim_latente, lunghezza_finestra,
                           media_train, std_train, prezzo_iniziale, device):
    """Genera path sintetici e li riporta in scala reale (de-standardizzati)."""
    generatore.eval()
    with torch.no_grad():
        z = torch.randn(n_path, dim_latente, lunghezza_finestra, device=device)
        rendimenti_generati_std = generatore(z).cpu().numpy()

    rendimenti_generati = destandardizza(rendimenti_generati_std, media_train, std_train)
    path_prezzi = prezzo_iniziale * np.exp(np.cumsum(rendimenti_generati, axis=1))
    return rendimenti_generati, path_prezzi


# =============================================================================
# 6. Visualizzazione (identica alla versione MLP)
# =============================================================================

def plotta_risultati(prezzi_reali, path_sintetici, storico_loss_c, storico_loss_g,
                      rendimenti_reali, rendimenti_generati, storico_distanza=None,
                      indice_split_train_test=None,
                      out_png="wgan_tcn_risultati.png"):

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
    assi[0, 1].set_title("Path sintetici generati dalla WGAN-GP (TCN)")
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
        assi[2, 0].set_title("Distanza reale-generato (spazio standardizzato) nel tempo")
        assi[2, 0].set_xlabel("Epoca")
        assi[2, 0].set_ylabel("Distanza di Wasserstein")
        assi[2, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Grafico salvato in: {out_png}")
def calcola_var_es(rendimenti, livello_confidenza=0.99):
    """
    Calcola Value at Risk (VaR) e Expected Shortfall (ES) empirici.
    Il livello di confidenza è tipicamente 0.95 o 0.99.
    Restituisce i valori (negativi) che rappresentano la perdita.
    """
    # L'alpha è la percentuale di coda (es. 1% per confidenza 99%)
    alpha = 1.0 - livello_confidenza
    
    # Il VaR è il quantile empirico di livello alpha
    var = np.quantile(rendimenti, alpha)
    
    # L'ES è la media dei rendimenti che sono inferiori o uguali al VaR
    es = rendimenti[rendimenti <= var].mean()
    
    return var, es

# =============================================================================
# 7. ESECUZIONE
# =============================================================================

# 0. Mount di Google Drive per il checkpoint
if USA_CHECKPOINT:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        print("Non sembra un ambiente Colab: disattivo il checkpoint su Drive.")
        USA_CHECKPOINT = False

# 1. Dati
prezzi, rendimenti = carica_da_csv(CSV_PATH)
print(f"Caricati {len(prezzi)} prezzi da {CSV_PATH}")

# --- split cronologico train/test ---
rendimenti_train, rendimenti_test = dividi_train_test(rendimenti, frazione_train)
n_train = len(rendimenti_train)
print(f"Split cronologico: {n_train} rendimenti in train, {len(rendimenti_test)} in test "
      f"(frazione train = {frazione_train})")

# --- standardizzazione (media/std stimate SOLO sul training set, niente leakage) ---
media_train = rendimenti_train.mean()
std_train = rendimenti_train.std()
print(f"Standardizzazione: media_train={media_train:.6f}, std_train={std_train:.6f}")

rendimenti_train_std = standardizza(rendimenti_train, media_train, std_train)
rendimenti_test_std = standardizza(rendimenti_test, media_train, std_train)

finestre_train = crea_finestre(rendimenti_train_std, lunghezza_finestra)
print(f"Create {len(finestre_train)} finestre di training di lunghezza {lunghezza_finestra}")

if len(rendimenti_test) <= lunghezza_finestra:
    print("ATTENZIONE: il test set ha meno osservazioni della lunghezza finestra; "
          "la metrica di early stopping su singoli rendimenti resta comunque valida.")

# 2. Training WGAN-GP (TCN)
generatore, critico, loss_c, loss_g, storico_distanza = allena_wgan_gp(
    finestre_train,
    dim_latente,
    lunghezza_finestra,
    n_epoche=N_EPOCHE,
    batch_size=BATCH_SIZE,
    lrG=lrG, lrC=lrC,
    n_critico_per_generatore=N_CRITICO_PER_GENERATORE,
    lambda_gp=LAMBDA_GP,
    n_livelli_tcn=N_LIVELLI_TCN, canali_tcn=CANALI_TCN,
    kernel_size_tcn=KERNEL_SIZE_TCN, dropout_tcn=DROPOUT_TCN,
    rendimenti_verifica_flat=rendimenti_test_std,
    usa_early_stopping=USA_EARLY_STOPPING,
    epoche_minime=EPOCHE_MINIME,
    verifica_ogni=VERIFICA_OGNI,
    pazienza=PAZIENZA,
    n_campioni_verifica=N_CAMPIONI_VERIFICA,
    usa_checkpoint=USA_CHECKPOINT,
    checkpoint_path=CHECKPOINT_PATH,
    checkpoint_ogni=CHECKPOINT_OGNI,
)

device = next(generatore.parameters()).device

# 3. Generazione di nuovi path sintetici dell'indice (gia' in scala reale, de-standardizzati)
rendimenti_generati, path_sintetici = genera_path_sintetici(
    generatore, n_path=N_PATH_GENERATI, dim_latente=dim_latente,
    lunghezza_finestra=lunghezza_finestra,
    media_train=media_train, std_train=std_train,
    prezzo_iniziale=prezzi[0], device=device,
)

# 4. Confronto statistico: generati vs TRAIN e vs TEST (fuori campione, tutto in scala reale)
print("\n--- Confronto statistico: train (visto in training) vs test (fuori campione) vs generati ---")
print(f"{'':10s} {'train':>12s} {'test':>12s} {'generati':>12s}")
print(f"{'Media':10s} {rendimenti_train.mean():12.6f} {rendimenti_test.mean():12.6f} {rendimenti_generati.mean():12.6f}")
print(f"{'Std':10s} {rendimenti_train.std():12.6f} {rendimenti_test.std():12.6f} {rendimenti_generati.std():12.6f}")
print(f"{'Skew':10s} {pd.Series(rendimenti_train).skew():12.4f} {pd.Series(rendimenti_test).skew():12.4f} {pd.Series(rendimenti_generati.flatten()).skew():12.4f}")
print(f"{'Kurtosi':10s} {pd.Series(rendimenti_train).kurtosis():12.4f} {pd.Series(rendimenti_test).kurtosis():12.4f} {pd.Series(rendimenti_generati.flatten()).kurtosis():12.4f}")
# 4. Confronto statistico: generati vs TRAIN e vs TEST (fuori campione, tutto in scala reale)
rendimenti_gen_flat = rendimenti_generati.flatten()

print("\n--- Confronto statistico: train vs test vs generati ---")
print(f"{'':15s} {'train':>12s} {'test':>12s} {'generati':>12s}")
print(f"{'Media':15s} {rendimenti_train.mean():12.6f} {rendimenti_test.mean():12.6f} {rendimenti_gen_flat.mean():12.6f}")
print(f"{'Std':15s} {rendimenti_train.std():12.6f} {rendimenti_test.std():12.6f} {rendimenti_gen_flat.std():12.6f}")
print(f"{'Skew':15s} {pd.Series(rendimenti_train).skew():12.4f} {pd.Series(rendimenti_test).skew():12.4f} {pd.Series(rendimenti_gen_flat).skew():12.4f}")
print(f"{'Kurtosi':15s} {pd.Series(rendimenti_train).kurtosis():12.4f} {pd.Series(rendimenti_test).kurtosis():12.4f} {pd.Series(rendimenti_gen_flat).kurtosis():12.4f}")

# --- Calcolo VaR e ES al 95% ---
var_train_95, es_train_95 = calcola_var_es(rendimenti_train, 0.95)
var_test_95, es_test_95 = calcola_var_es(rendimenti_test, 0.95)
var_gen_95, es_gen_95 = calcola_var_es(rendimenti_gen_flat, 0.95)

# --- Calcolo VaR e ES al 99% ---
var_train_99, es_train_99 = calcola_var_es(rendimenti_train, 0.99)
var_test_99, es_test_99 = calcola_var_es(rendimenti_test, 0.99)
var_gen_99, es_gen_99 = calcola_var_es(rendimenti_gen_flat, 0.99)

print("\n--- Analisi del Rischio (Tail Risk) ---")
print(f"{'VaR (95%)':15s} {var_train_95:12.4f} {var_test_95:12.4f} {var_gen_95:12.4f}")
print(f"{'ES (95%)':15s} {es_train_95:12.4f} {es_test_95:12.4f} {es_gen_95:12.4f}")
print(f"{'VaR (99%)':15s} {var_train_99:12.4f} {var_test_99:12.4f} {var_gen_99:12.4f}")
print(f"{'ES (99%)':15s} {es_train_99:12.4f} {es_test_99:12.4f} {es_gen_99:12.4f}")
# 5. Grafici
plotta_risultati(prezzi, path_sintetici, loss_c, loss_g, rendimenti, rendimenti_generati,
                  storico_distanza=storico_distanza, indice_split_train_test=n_train)

