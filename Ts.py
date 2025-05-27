import os
import glob
import pandas as pd

# =========================
# 1. Détection automatique des chemins
# =========================
# (on suppose que tous les fichiers sont dans le répertoire courant ;
# si vous préférez un sous-dossier, il suffit de préfixer le pattern par "mon_dossier/")
CSV_FOLDER      = 'CSV'

# Recherche du fichier *_Detail_RAO_Commande*.csv dans CSV/
detail_pattern  = os.path.join(CSV_FOLDER, '*_Detail_RAO_Commande*.csv')
detail_files    = glob.glob(detail_pattern)
if not detail_files:
    raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {detail_pattern}")
DETAIL_CSV      = detail_files[0]

# Recherche du fichier *_Boite3*.csv dans CSV/
boite3_pattern  = os.path.join(CSV_FOLDER, '*SQF_Boite_3*.csv')
boite3_files    = glob.glob(boite3_pattern)
if not boite3_files:
    raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {boite3_pattern}")
BOITE3_CSV      = boite3_files[0]


TS_pattern       = os.path.join(CSV_FOLDER, '*Taux_Service.csv')
TS_files         = glob.glob(TS_pattern)
if not TS_files:
    raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {TS_pattern}")
# On prend le premier match
TS_CSV           = TS_files[0]


# Recherche du fichier *_MacroParam*.csv dans CSV/
macro_pattern  = os.path.join(CSV_FOLDER, 'MacroParam*.csv')
macro_files    = glob.glob(macro_pattern)
if not macro_files:
    raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {macro_pattern}")
MACROPARAM_CSV = macro_files[0]


def get_processed_data():
    """    Fonction qui calcule la colonne TS pour chaque CODE_METI et retourne un dataframe
    avec CODE_METI et TS pour merge avec les autres données.
    Toutes les valeurs TS sont sur une échelle de 0-1.
    """
    # =========================
    # 2. Charger le détail
    # =========================
    df_detail = pd.read_csv(
        DETAIL_CSV, sep=';', encoding='latin1', engine='python'
    )
    # Normaliser la clé produit
    for col in ('CDBASE','CDBase'):
        if col in df_detail.columns:
            df_detail.rename(columns={col:'CODE_METI'}, inplace=True)
            break

    # Assurer la présence de la colonne EAN
    if 'Ean_13' in df_detail.columns:
        df_detail['EAN'] = df_detail['Ean_13'].astype(str)  # Convert to string
    else:
        # Fallback via Boite3
        df_boite3 = pd.read_csv(
            BOITE3_CSV, sep=';', encoding='latin1', engine='python', dtype={'EAN': str}  # Set dtype for EAN
        )
        # Renommer CDBASE en CODE_METI
        for col in ('CDBASE','CDBase'):
            if col in df_boite3.columns:
                df_boite3.rename(columns={col:'CODE_METI'}, inplace=True)
                break
        # Ensure CODE_METI is string type to match df_detail
        df_boite3['CODE_METI'] = df_boite3['CODE_METI'].astype(str)
        df_detail['CODE_METI'] = df_detail['CODE_METI'].astype(str)
        
        # Merge with explicit string types
        df_detail = df_detail.merge(
            df_boite3[['CODE_METI','EAN']], on='CODE_METI', how='left'
        )

    # Make sure EAN column is string
    if 'EAN' in df_detail.columns:
        df_detail['EAN'] = df_detail['EAN'].fillna('').astype(str)

    # =========================
    # 3. Charger MacroParam
    # =========================
    df_macro = pd.read_csv(
        MACROPARAM_CSV, sep=';', encoding='latin1', engine='python'
    )
    # On conserve Zone de stockage et Correction TS
    df_macro = df_macro[['Zone de stockage','Correction TS']].rename(
        columns={'Zone de stockage':'CLASSE_STOCKAGE'}
    )
    # Convert Correction TS to float (0-1 scale)
    df_macro['Correction TS'] = df_macro['Correction TS'].astype(str).str.rstrip('%').replace('', '0').astype(float).div(100)

    # =========================
    # 4. Charger TS.xlsx
    # =========================
   
    df_ts =pd.read_csv(
        TS_CSV, sep=';', encoding='latin1', engine='python',dtype={'EAN': str}  # Explicitly set EAN to string
    )
    # Normaliser EAN format (sans séparateurs)
    df_ts['EAN'] = df_ts['EAN'].astype(str).str.replace(r'[^\d]', '', regex=True)
    # Conserver uniquement EAN et TS_Final_V2
    df_ts = df_ts[['EAN','TS_Final_V2']].copy()    # Convertir TS_Final_V2 en fraction numérique (0-1 scale)
    # Avant la conversion, vérifier si les valeurs sont déjà entre 0-1 ou entre 0-100
    # On présume que si la valeur max est <= 1, les données sont déjà à l'échelle 0-1
    ts_values = pd.to_numeric(df_ts['TS_Final_V2'].astype(str).str.rstrip('%').replace('', '0'), errors='coerce')
    
    # Si les valeurs sont déjà sur une échelle 0-1, pas besoin de diviser par 100
    if ts_values.max() <= 1:
        df_ts['TS_Base'] = ts_values
    else:
        # Sinon, convertir depuis une échelle 0-100 vers 0-1
        df_ts['TS_Base'] = ts_values.div(100)

    # =========================
    # 5. Fusionner tout
    # =========================
    # Make sure both EAN columns are string type before merging
    df_ts['EAN'] = df_ts['EAN'].astype(str)
    df_detail['EAN'] = df_detail['EAN'].astype(str)    # Now merge with string types
    df = df_detail.merge(df_ts[['EAN','TS_Base']], on='EAN', how='left')
    df['TS_Base'] = df['TS_Base'].fillna(1)  # défaut = 100% (échelle 0-1 donc 1 = 100%)

    # Convert CLASSE_STOCKAGE to same type before merging
    if 'CLASSE_STOCKAGE' in df.columns:
        df['CLASSE_STOCKAGE'] = pd.to_numeric(df['CLASSE_STOCKAGE'], errors='coerce')
        df_macro['CLASSE_STOCKAGE'] = pd.to_numeric(df_macro['CLASSE_STOCKAGE'], errors='coerce')

    df = df.merge(df_macro, on='CLASSE_STOCKAGE', how='left')
    df['Correction TS'] = df['Correction TS'].fillna(0.0)    # =========================
    # 6. Calcul final de TS (borné [0,1])
    # =========================
    # Calcul sur échelle 0-1
    df['TS'] = (df['TS_Base'] + df['Correction TS']).clip(lower=0, upper=1)
      # =========================
    # 7. Correction des valeurs incohérentes
    # =========================
    # Détecter les anomalies de TS - nous nous attendons à 0, 0.01, ou des valeurs proches de 1 (0.99-1.0)
    # Pour les valeurs comme 0.01 qui devraient être 1 (100%), on les corrige
    # Seuil: si une valeur est <= 0.02 et n'est pas 0, on présume que c'est une erreur et on la fixe à 1.0
    df['TS'] = df['TS'].apply(lambda x: 1.0 if (x > 0 and x <= 0.02) else x)
    
    # Si certains TS sont sur l'échelle 0-100 au lieu de 0-1, les diviser par 100
    df['TS'] = df['TS'].apply(lambda x: x / 100 if x > 1 and x <= 100 else x)
    
    # S'assurer que toutes les valeurs sont bien entre 0 et 1
    df['TS'] = df['TS'].clip(lower=0, upper=1)
    
    # Return only the needed columns for merging
    result_df = df[['CODE_METI', 'TS']].drop_duplicates()
    # Ensure CODE_METI is string type for consistency with other modules
    result_df['CODE_METI'] = result_df['CODE_METI'].astype(str)
      # Vérification finale des valeurs TS (doit être entre 0 et 1)
    print(f"Min TS value: {result_df['TS'].min()}, Max TS value: {result_df['TS'].max()}")
    
    return result_df

if __name__ == "__main__":
    # Test the function when module is run directly
    df = get_processed_data()
    print(f"Retrieved {len(df)} rows with TS values")
    print("\nSample data:")
    print(df.head(10))

