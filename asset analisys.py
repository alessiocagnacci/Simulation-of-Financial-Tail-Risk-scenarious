import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Impostazione dello stile grafico
plt.style.use('seaborn-v0_8-darkgrid')

def plotta_prezzi_e_rendimenti(percorso_csv, colonna_data='Date', colonna_prezzo='Close', titolo="Asset"):
    try:
        df = pd.read_csv(percorso_csv)
    except FileNotFoundError:
        print(f"Errore: Il file '{percorso_csv}' non è stato trovato.")
        return

    # Gestione delle date per l'asse X
    if colonna_data in df.columns:
        df[colonna_data] = pd.to_datetime(df[colonna_data])
        df.set_index(colonna_data, inplace=True)
        asse_x = df.index
    else:
        print(f"Attenzione: Colonna '{colonna_data}' non trovata. Verrà usato l'indice numerico.")
        asse_x = range(len(df))

    if colonna_prezzo not in df.columns:
        print(f"Errore: La colonna '{colonna_prezzo}' non è presente nel CSV.")
        return

    # Calcolo dei log-rendimenti (mantiene l'allineamento con l'indice temporale)
    df['Log_Ret'] = np.log(df[colonna_prezzo] / df[colonna_prezzo].shift(1))

    # Creazione della figura con 2 subplot sovrapposti
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Subplot 1: Prezzo
    ax1.plot(asse_x, df[colonna_prezzo], color='black', linewidth=1.5, label=f"{titolo}")
    ax1.set_title(f"Time series and Log-Return: {titolo}", fontsize=14)
    ax1.set_ylabel("Price", fontsize=12)
    ax1.legend(loc="upper left")

    # Subplot 2: Log-Rendimenti
    # Partiamo da 1 per saltare il primo valore NaN generato dal .shift(1)
    ax2.plot(asse_x[1:], df['Log_Ret'].iloc[1:], color='teal', linewidth=1.0, alpha=0.8, label="Daily Log-Return")
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--') # Linea dello zero
    ax2.set_xlabel("Date" if colonna_data in df.columns else "Days", fontsize=12)
    ax2.set_ylabel("Log-Return", fontsize=12)
    ax2.legend(loc="upper left")

    plt.tight_layout()

    # Salva e mostra
    nome_out = f"storico_{titolo.replace(' ', '_')}.png"
    plt.savefig(nome_out, dpi=150)
    plt.show()

# Esecuzione
if __name__ == "__main__":
    CSV_PATH = "/content/^fmib_d.csv"  # Modifica con il percorso del tuo asset
    plotta_prezzi_e_rendimenti(CSV_PATH, titolo="FTSE MIB")
