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
        parse_dates=['DATE_COMMANDE'],
        dayfirst=True,
        engine='python'
    )

    # 3. Calculer Date_L2_V2 = DATE_COMMANDE + 3 jours
    df_detail['Date_L2_V2'] = MacroParam.DATE_COMMANDE + pd.Timedelta(days=3)

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
    df_prev['PrevisionJour'] = (
        df_prev['PrevisionJour']
            .astype(str)
            .str.replace(',','.',)
            .astype(float)
    )

    # Filter for dates within 3 weeks of 14/05/2025
    reference_date = MacroParam.DATE_COMMANDE
    three_weeks_ago = reference_date - pd.Timedelta(days=21)
    df_prev = df_prev[df_prev['JourPromo'] >= three_weeks_ago]

    # 7. Définir la date de début fixe (14/05/2025)
    start_date = MacroParam.DATE_COMMANDE

    # 8. Calcul de Prev Promo C1-L2
    def calc_prev_promo_c1_l2(row):
        mask = (
            (df_prev['CODE_METI'] == row['CODE_METI']) &
            (df_prev['JourPromo'] >= start_date) &
            (df_prev['JourPromo'] <= row['Date_L2_V2'])
        )
        return df_prev.loc[mask, 'PrevisionJour'].sum()

    # 9 calculer Prev Promo L1-L2
    def calc_prev_promo_l1_l2(row):
        mask = (
            (df_prev['CODE_METI'] == row['CODE_METI']) &
            (df_prev['JourPromo'] >= row['DATE_COMMANDE']+pd.Timedelta(days=2)) &
            (df_prev['JourPromo'] < row['Date_L2_V2'])
        )
        return df_prev.loc[mask, 'PrevisionJour'].sum()

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


