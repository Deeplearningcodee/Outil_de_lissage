import os
import glob
import pandas as pd
import MacroParam
# Dossiers contenant les CSV
CSV_FOLDER      = 'CSV'

# Recherche du fichier *_Detail_RAO_Commande*.csv dans CSV/
detail_pattern  = os.path.join(CSV_FOLDER, '*_Detail_RAO_Commande*.csv')
detail_files    = glob.glob(detail_pattern)
if not detail_files:
    raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {detail_pattern}")
DETAIL_CSV      = detail_files[0]
# Recherche du fichier *Previsions_Flux_Tire_PPC_SQF*.csv dans CSV/
prev_pattern    = os.path.join(CSV_FOLDER, '*Previsions_Flux_Tire_PPC_SQF*.csv')
prev_files      = glob.glob(prev_pattern)
if not prev_files:
    raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {prev_pattern}")
PREV_CSV        = prev_files[0]


def get_processed_data():
    # 2. Charger le détail et parser la date de commande
    df_detail = pd.read_csv(
        DETAIL_CSV,
        sep=';',
        encoding='latin1',
        parse_dates=['DATE_COMMANDE', 'DATE_LIVRAISON', 'DATE_L2'], # MODIFIED
        dayfirst=True,
        engine='python'
    )
    
    df_detail['DATE_L1'] = df_detail['DATE_LIVRAISON']  # Utiliser la date de livraison V2
    df_detail['Date_L2_V2'] = df_detail['DATE_L2']  # Utiliser la date L2 V2

    # 4. Charger les prévisions promo
    df_prev = pd.read_csv(
        PREV_CSV,
        sep=';',
        encoding='latin1',
        parse_dates=['JourPromo'],
        dayfirst=True,
        engine='python'
    )

    # 5. Normaliser la clé produit
    for df in (df_detail, df_prev):
        if 'CDBASE' in df.columns:
            df.rename(columns={'CDBASE':'CODE_METI'}, inplace=True)
        elif 'CDBase' in df.columns:
            df.rename(columns={'CDBase':'CODE_METI'}, inplace=True)

    # 6. Nettoyer PrevisionJour en float
    df_prev['PrevisionJour'] = pd.to_numeric(
        df_prev['PrevisionJour'].astype(str).str.replace(',', '.', regex=False),
        errors='coerce'
    )
    # .sum() will treat NaNs as 0 unless all are NaN for a group.
    # If explicit 0s are needed for NaNs before sum, use .fillna(0)
    # df_prev['PrevisionJour'] = df_prev['PrevisionJour'].fillna(0)

    reference_date = MacroParam.DATE_COMMANDE
    three_weeks_ago = reference_date - pd.Timedelta(days=21)
    df_prev = df_prev[df_prev['JourPromo'] >= three_weeks_ago]

    start_date = MacroParam.DATE_COMMANDE

    # 8. Calcul de Prev Promo C1-L2
    def calc_prev_promo_c1_l2(row):
        mask = (
            (df_prev['CODE_METI'] == row['CODE_METI']) &
            (df_prev['JourPromo'] >= start_date) &
            (df_prev['JourPromo'] < row['Date_L2_V2']) 
        )
        return df_prev.loc[mask, 'PrevisionJour'].sum().round(0)

    # 9 calculer Prev Promo L1-L2
    def calc_prev_promo_l1_l2(row):
        mask = (
            (df_prev['CODE_METI'] == row['CODE_METI']) &
            (df_prev['JourPromo'] >= row ['DATE_L1']) &
            (df_prev['JourPromo'] < row['Date_L2_V2'])
        )
        return df_prev.loc[mask, 'PrevisionJour'].sum().round(0)

    df_detail['Prev Promo C1-L2'] = df_detail.apply(calc_prev_promo_c1_l2, axis=1)
    df_detail['Prev Promo L1-L2'] = df_detail.apply(calc_prev_promo_l1_l2, axis=1)

    # Return only the needed columns for merging
    result_df = df_detail[['CODE_METI', 'Prev Promo C1-L2', 'Prev Promo L1-L2']].drop_duplicates()
    return result_df

if __name__ == "__main__":
    # Original code execution when run as script
    # 9. Afficher un extrait
    df_detail = get_processed_data()
    print(df_detail[['CODE_METI', 'Prev Promo C1-L2', 'Prev Promo L1-L2']].head(10))
    #save to CSV for debugging
    output_csv = os.path.join(CSV_FOLDER, 'PrevPromo_Output.csv')
    df_detail.to_csv(output_csv, sep=';', index=False, encoding='latin1')


