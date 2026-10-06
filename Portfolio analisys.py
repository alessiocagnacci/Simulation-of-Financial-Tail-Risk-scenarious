import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Impostazione dello stile grafico
plt.style.use('seaborn-v0_8-darkgrid')

def analizza_portafoglio_ew(dizionario_csv, colonna_data='Date', colonna_prezzo='Close'):
    lista_df = []

    for nome, percorso in dizionario_csv.items():
        try:
            df = pd.read_csv(percorso)
            if colonna_data in df.columns:
                df[colonna_data] = pd.to_datetime(df[colonna_data])
                df.set_index(colonna_data, inplace=True)
            else:
                continue

            if colonna_prezzo not in df.columns:
                continue

            serie_prezzo = df[[colonna_prezzo]].rename(columns={colonna_prezzo: nome})
            lista_df.append(serie_prezzo)

        except FileNotFoundError:
            print(f"Errore: Il file '{percorso}' non è stato trovato.")
            return

    # Unione dei dataframe (mantiene solo i giorni di apertura comuni)
    df_portafoglio = pd.concat(lista_df, axis=1, join='inner').dropna()
    print(f"Dati allineati: {len(df_portafoglio)} giorni di contrattazione comuni trovati.")

    # 1. Prezzi Normalizzati Singoli Asset (Base 100)
    df_normalizzato = (df_portafoglio / df_portafoglio.iloc[0]) * 100

    # 2. Log-Rendimenti Singoli Asset
    df_log_ret = np.log(df_portafoglio / df_portafoglio.shift(1)).dropna()

    # 3. Log-Rendimenti Portafoglio (Equally Weighted)
    pesi = np.ones(len(dizionario_csv)) / len(dizionario_csv)
    serie_ret_portafoglio = df_log_ret.dot(pesi)

    # 4. Calcolo Equity Line Portafoglio EW (Base 100)
    # L'esponenziale della somma cumulativa dei log-rendimenti restituisce il rendimento composto
    prezzo_portafoglio = pd.Series(index=df_normalizzato.index, dtype=float)
    prezzo_portafoglio.iloc[0] = 100.0  # Valore di partenza
    prezzo_portafoglio.iloc[1:] = 100.0 * np.exp(np.cumsum(serie_ret_portafoglio))

    # Creazione della dashboard
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Pannello 1: Equity Line Portafoglio vs Asset Singoli
    for colonna in df_normalizzato.columns:
        ax1.plot(df_normalizzato.index, df_normalizzato[colonna], linewidth=1.0, alpha=0.4, label=colonna)

    # Linea del Portafoglio marcata e in evidenza
    ax1.plot(prezzo_portafoglio.index, prezzo_portafoglio, color='black', linewidth=2.5, label='EW Portfolio')

    ax1.set_title("Equity Line: Portfolio EW vs Asset (100 base)", fontsize=14)
    ax1.set_ylabel("Value (100 base)", fontsize=12)
    ax1.legend(loc="upper left")

    # Pannello 2: Log-Rendimenti del Portafoglio
    ax2.plot(serie_ret_portafoglio.index, serie_ret_portafoglio, color='purple', linewidth=1.0, alpha=0.8, label="Log-Return")
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')

    ax2.set_title("Daily Log-Return EW portfolio", fontsize=14)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Log-Return", fontsize=12)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("analisi_equity_portafoglio.png", dpi=150)
    plt.show()

# Esecuzione
if __name__ == "__main__":
    ASSET_FILES = {
        "FTSE MIB": "/content/^fmib_d.csv",
        "Nikkei 225": "/content/^nkx_d-2.csv",
        "Nasdaq 100": "/content/^ndx_d copia.csv",
        "Hang Seng": "/content/^hsi_d.csv"
    }

    analizza_portafoglio_ew(ASSET_FILES)