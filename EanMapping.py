import pandas as pd
import os
import glob


# Chemin vers le fichier contenant les EANs
# Dossier contenant les CSV
CSV_FOLDER = 'CSV'
boite3_pattern = os.path.join(CSV_FOLDER, '*Boite_3*.csv')
boite3_files = glob.glob(boite3_pattern)
if not boite3_files:
    raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {boite3_pattern}")
# On prend le premier match
BOITE3_CSV = boite3_files[0]


def get_processed_data():
    """
    Récupère le mapping CODE_METI -> EAN depuis le fichier Boite3
    
    Returns:
        DataFrame avec les colonnes CODE_METI et Ean_13
    """
    try:
        # Charger le fichier Boite3
        df_boite3 = pd.read_csv(
            BOITE3_CSV, 
            sep=';', 
            encoding='latin1', 
            engine='python',
            dtype={'EAN': str}  # Assurer que EAN est considéré comme une chaîne
        )
        
        # Renommer CDBASE en CODE_METI si nécessaire
        for col in ('CDBASE', 'CDBase'):
            if col in df_boite3.columns:
                df_boite3.rename(columns={col: 'CODE_METI'}, inplace=True)
                break
        
        # Assurer que CODE_METI est en format string
        df_boite3['CODE_METI'] = df_boite3['CODE_METI'].astype(str)
          # S'assurer que EAN est présent
        if 'EAN' not in df_boite3.columns:
            print(f"Attention: Colonne EAN non trouvée dans {BOITE3_CSV}")
            return pd.DataFrame(columns=['CODE_METI', 'Ean_13'])
          # Convertir EAN en string et nettoyer
        df_boite3['EAN'] = df_boite3['EAN'].astype(str).str.strip()
        
        # Renommer EAN en Ean_13 pour compatibilité avec FacteurAppro.py
        df_boite3.rename(columns={'EAN': 'Ean_13'}, inplace=True)
        
        # Ne conserver que les colonnes CODE_METI et Ean_13, et supprimer les doublons
        result_df = df_boite3[['CODE_METI', 'Ean_13']].drop_duplicates()
        
        print(f"Mapping EAN chargé: {len(result_df)} articles avec EAN")
        return result_df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {BOITE3_CSV}: {e}")
        return pd.DataFrame(columns=['CODE_METI', 'Ean_13'])

if __name__ == "__main__":
    # Test du module
    df = get_processed_data()
    print(df.head(10))
    print(f"Total des mappings CODE_METI -> EAN: {len(df)}")
