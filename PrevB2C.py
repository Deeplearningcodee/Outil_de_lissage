import os
import glob
import pandas as pd
import MacroParam
# Dossiers contenant les CSV
CSV_FOLDER      = 'CSV'
PREV_FOLDER     = 'Previsions'

# Recherche du fichier *_Detail_RAO_Commande*.csv dans CSV/
detail_pattern  = os.path.join(CSV_FOLDER, '*_Detail_RAO_Commande*.csv')
detail_files    = glob.glob(detail_pattern)
if not detail_files:
    raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {detail_pattern}")
DETAIL_CSV      = detail_files[0]

# Recherche du fichier *Previsions-B2C*.csv dans Previsions/
prev_pattern    = os.path.join(PREV_FOLDER, '*Previsions-B2C*.csv')
prev_files      = glob.glob(prev_pattern)
if not prev_files:
    raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {prev_pattern}")
LONGTERME_CSV   = prev_files[0]


def get_processed_data():
    # 1. Charger et préparer df_detail
    df_detail = pd.read_csv(
        DETAIL_CSV, sep=';', encoding='latin1',
        parse_dates=['DATE_COMMANDE'], dayfirst=True, engine='python'
    )
    # Renommer la clé produit
    for col in ('CDBASE','CDBase'):
        if col in df_detail.columns:
            df_detail.rename(columns={col:'CODE_METI'}, inplace=True)
            break
    # Calcul des bornes
    df_detail['Date_L1'] = MacroParam.DATE_COMMANDE + pd.Timedelta(days=2)
    df_detail['Date_L2'] = MacroParam.DATE_COMMANDE + pd.Timedelta(days=3)

    # 2. Charger et préparer df_prev
    df_prev = pd.read_csv(
        LONGTERME_CSV, sep=';', encoding='latin1',
        parse_dates=['Date'], dayfirst=True, engine='python'
    )
    # Renommer la clé produit
    for col in ('CDBASE','CDBase'):
        if col in df_prev.columns:
            df_prev.rename(columns={col:'CODE_METI'}, inplace=True)
            break
    # Conversion
    df_prev['Prevision_B2C'] = (
        df_prev['Prevision_B2C'].astype(str)
                    .str.replace(',','.')
                    .astype(float)
    )

    # Filter for dates within 3 weeks of 14/05/2025
    reference_date = MacroParam.DATE_COMMANDE
    three_weeks_ago = reference_date - pd.Timedelta(days=21)
    df_prev = df_prev[df_prev['Date'] >= three_weeks_ago]
    
    # 3. Fonctions de calcul
    def compute_prev_c1_l2_avg(row):
        mask = (
            (df_prev['CODE_METI']==row['CODE_METI']) &
            (df_prev['Date']>=row['DATE_COMMANDE']) &
            (df_prev['Date']<=row['Date_L2'])
        )
        vals = df_prev.loc[mask, 'Prevision_B2C']
        vals = vals.round(0)
        return vals.mean() if not vals.empty else 0

    def compute_prev_l1_l2_avg(row):
        mask = (
            (df_prev['CODE_METI']==row['CODE_METI']) &
            (df_prev['Date']>=row['Date_L1']) &
            (df_prev['Date']< row['Date_L2'])
        )
        vals = df_prev.loc[mask, 'Prevision_B2C']
        vals = vals.round(0)
        return vals.mean() if not vals.empty else 0

    # 4. Appliquer
    df_detail['Prev C1-L2 Avg'] = df_detail.apply(compute_prev_c1_l2_avg, axis=1)
    df_detail['Prev L1-L2 Avg'] = df_detail.apply(compute_prev_l1_l2_avg, axis=1)

    #rename Prev C1-L2 Avg to Prev C1-L2 
    df_detail.rename(columns={'Prev C1-L2 Avg': 'Prev C1-L2'}, inplace=True)
    df_detail.rename(columns={'Prev L1-L2 Avg': 'Prev L1-L2'}, inplace=True)
    
    # Return only the needed columns for merging
    result_df = df_detail[['CODE_METI', 'Prev C1-L2', 'Prev L1-L2']].drop_duplicates()
    return result_df

if __name__ == "__main__":
    # Original code execution when run as script
    # 5. Aperçu
    df_detail = get_processed_data()
    print(df_detail.head(10))
