import os
import glob
import pandas as pd
import numpy as np
import MacroParam
# Dossiers contenant vos CSV
CSV_FOLDER   = 'CSV'
PREV_FOLDER  = 'Previsions'

# Pattern pour le fichier détail
detail_pattern = os.path.join(CSV_FOLDER, '*_Detail_RAO_Commande*.csv')
detail_files   = glob.glob(detail_pattern)
if not detail_files:
    raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {detail_pattern}")
# On prend le premier match
DETAIL_CSV = detail_files[0]




def get_processed_data():
    # =========================
    # 1. Charger le détail
    # =========================
    df_detail = pd.read_csv(
        DETAIL_CSV,
        sep=';', encoding='latin1', engine='python'
    )
    # Normaliser la clé produit
    if 'CDBASE' in df_detail.columns:
        df_detail.rename(columns={'CDBASE':'CODE_METI'}, inplace=True)
    elif 'CDBase' in df_detail.columns:
        df_detail.rename(columns={'CDBase':'CODE_METI'}, inplace=True)

    # =========================
    # 2. Charger les livraisons
    #    pour window 14-17 mai 2025
    # =========================

    # in folder Livraisons_Client_A_Venir
    Encours_FOLDER="Livraisons_Client_A_Venir"
    #use glob to find the file
    liv_pattern = os.path.join(Encours_FOLDER, '*Livraisons_AllMags_A_Venir.csv')
    liv_files   = glob.glob(liv_pattern)
    if not liv_files:
        raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {liv_pattern}")
    # On prend le premier match
    LIV_CSV = liv_files[0]
    # Charger le fichier de livraisons
    df_liv = pd.read_csv(
        LIV_CSV,
        sep=';', encoding='latin1', engine='python',
        parse_dates=['DATE_Livraison'], dayfirst=True
    )

    # Normaliser clé
    if 'CDBASE' in df_liv.columns:
        df_liv.rename(columns={'CDBASE':'CODE_METI'}, inplace=True)
    elif 'CDBase' in df_liv.columns:
        df_liv.rename(columns={'CDBase':'CODE_METI'}, inplace=True)

    # Filter for dates within 3 weeks of 14/05/2025
    reference_date = MacroParam.DATE_COMMANDE
    three_weeks_ago = reference_date - pd.Timedelta(days=21)
    df_liv = df_liv[df_liv['DATE_Livraison'] >= three_weeks_ago]

    # Nettoyage quantité
    df_liv['QTE_totale'] = (
        df_liv['QTE_totale'].astype(str)
                      .str.replace(',', '.')
                      .astype(float)
    )

    # Filtrer par date
    start = MacroParam.DATE_COMMANDE
    end   = MacroParam.DATE_REF_JOUR_AC_STR
    df_win = df_liv[df_liv['DATE_Livraison'].between(start, end)]

    # =========================
    # 3. Pivot pour obtenir
    #    somme par code et par date
    # =========================
    pivot = df_win.pivot_table(
        index='CODE_METI',
        columns='DATE_Livraison',
        values='QTE_totale',
        aggfunc='sum',
        fill_value=0
    )
    # Garder uniquement ces 4 dates  DATE_COMMANDE   DATE_REF_JOUR_AB  DATE_REF_JOUR_AC  DATE_REF_JOUR_Ac+1
    dates = pd.date_range(
        start=MacroParam.DATE_COMMANDE,
        end=MacroParam.DATE_REF_JOUR_AC_STR,
        freq='D'
    )
    pivot = pivot.reindex(columns=dates, fill_value=0)

    # =========================
    # 4. Calcul des positions et cumuls
    # =========================
    # Positions dans le pivot
    # 1-based to match Excel
    pos_en = {code: idx+1 for idx, code in enumerate(pivot.index)}
    pos_l1 = 1  # premier jour de la fenêtre
    pos_l2 = len(dates)  # dernier jour de la fenêtre

    # Cumuls
    sum_window = pivot.sum(axis=1)            # somme sur toute la fenêtre
    sum_last = pivot[dates[-1]]                # somme du dernier jour uniquement

    # Créer DataFrame résultat
    df_res = df_detail[['CODE_METI']].drop_duplicates().copy()
    df_res['Position En-cours client'] = df_res['CODE_METI'].map(pos_en)
    df_res['Position L1']               = pos_l1
    df_res['Position L2']               = pos_l2
    df_res['En-cours client C1-L2']     = df_res['CODE_METI'].map(sum_window)
    df_res['En-cours client L1-L2']     = df_res['CODE_METI'].map(sum_last)
    
    return df_res

if __name__ == "__main__":
    # Original code execution when run as script
    # =========================
    # 5. Afficher les 10 premières lignes
    # =========================
    df_res = get_processed_data()
    print(df_res.head(10))