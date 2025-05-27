import pandas as pd
import numpy as np
import glob
import os
import MacroParam  
def get_processed_data():
    # =========================
    # 1. Charger les sources      
    # =========================

    # Dossier contenant les CSV
    CSV_FOLDER = 'CSV'
    # Recherche du fichier *_Detail_RAO_Commande*.csv
    detail_pattern = os.path.join(CSV_FOLDER, '*_Detail_RAO_Commande*.csv')
    detail_files = glob.glob(detail_pattern)
    if not detail_files:
        raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {detail_pattern}")
    DETAIL_CSV = detail_files[0]

    df_detail = pd.read_csv(
        DETAIL_CSV,
        sep=';', encoding='latin1', engine='python',
        parse_dates=['DATE_L2', 'DATE_COMMANDE'], dayfirst=True
    )

    
    Casse_FOLDER = 'Casse'
    # Recherche du fichier *_Casse*.csv
    casse_pattern = os.path.join(Casse_FOLDER, '*_Casse_Prev*.csv')
    casse_files = glob.glob(casse_pattern)
    if not casse_files:
        raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {casse_pattern}")
    CASSE_CSV = casse_files[0]
    
    df_casse_raw = pd.read_csv(
        CASSE_CSV,
        sep=';', encoding='latin1', engine='python',
        parse_dates=['TODAY'], dayfirst=True
    )

    # Filter for dates within 3 weeks of 14/05/2025
    reference_date = MacroParam.DATE_COMMANDE
    three_weeks_ago = reference_date - pd.Timedelta(days=21)
    df_casse_raw = df_casse_raw[df_casse_raw['TODAY'] >= three_weeks_ago]

    # =========================
    # 2. Normaliser les clés et colonnes
    # =========================
    df_detail = df_detail.rename(columns={
        'CDBASE': 'CODE_METI',
        'CLASSE_STOCKAGE': 'Classe_Stockage'
    })
    df_casse_raw = df_casse_raw.rename(columns={
        'CDBase': 'CODE_METI',
        'TODAY': 'Date',
        'Stock_Date_J_Fin_Journee': 'Casse_PrevQty'
    })

    # Conversion en numérique
    df_casse_raw['Casse_PrevQty'] = (
        df_casse_raw['Casse_PrevQty'].astype(str)
                    .str.replace(',', '.')
                    .astype(float)
    )

    # =========================
    # 3. Pivot 'Casse Prev TCD'
    # =========================
    pivot = df_casse_raw.pivot_table(
        index='CODE_METI',
        columns='Date',
        values='Casse_PrevQty',
        aggfunc='sum',
        fill_value=0
    ).sort_index(axis=1)
    dates = list(pivot.columns)

    # =========================
    # 4. Fonctions de calcul
    # =========================
    def position_casse(code):
        # si CODE_METI entre 2100 et 2580, renvoyer l'index dans le pivot
        if 2100 <= int(code) < 2580 and code in pivot.index:
            return pivot.index.get_loc(code) + 1
        return np.nan

    def position_col(date):
        # chercher la date dans les colonnes du pivot
        if pd.notna(date) and date in dates:
            return dates.index(date) + 1
        return np.nan

    def somme_c1_l2(code, end_date):
        if code in pivot.index and pd.notna(end_date):
            # somme depuis début jusqu'à end_date
            mask = pivot.columns <= end_date
            return round(pivot.loc[code, mask].sum(), 0)
        return 0

    def somme_l1_l2(code, start_date, end_date):
        if code in pivot.index and pd.notna(start_date) and pd.notna(end_date):
            mask = (pivot.columns >= start_date) & (pivot.columns <= end_date)
            return round(pivot.loc[code, mask].sum(), 0)
        return 0

    # =========================
    # 5. Appliquer les calculs
    # =========================
    df_detail['Position Casse Prev'] = df_detail['CODE_METI'].apply(position_casse)
    df_detail['Position L1']         = df_detail['DATE_L2'].apply(position_col)
    df_detail['Position L2']         = df_detail['DATE_COMMANDE'].apply(position_col)

    df_detail['Casse Prev C1-L2'] = df_detail.apply(
        lambda r: somme_c1_l2(r['CODE_METI'], r['DATE_COMMANDE']), axis=1
    )
    df_detail['Casse Prev L1-L2'] = df_detail.apply(
        lambda r: somme_l1_l2(r['CODE_METI'], r['DATE_L2'], r['DATE_COMMANDE']), axis=1
    )

    # Return only the needed columns for merging
    result_df = df_detail[['CODE_METI', 'Position Casse Prev', 'Casse Prev C1-L2', 'Casse Prev L1-L2']].drop_duplicates()
    return result_df

if __name__ == "__main__":
    # Original code execution when run as script
    # =========================
    # 6. Afficher 10 premières lignes
    # =========================
    df_detail = get_processed_data()
    print(df_detail.head(10))
    print(df_detail[df_detail['CODE_METI'] == 3699774].head(10))