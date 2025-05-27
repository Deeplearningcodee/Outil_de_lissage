import pandas as pd
import os
import numpy as np

def get_processed_data():
    """
    Charge les données de Stock Mini Publication FL depuis le fichier Excel
    et les prépare pour les fusions dans le processus principal.
    
    Formule Excel à reproduire:
    =SIERREUR(INDEX(MiniPubliFL!F:F;EQUIV(G2;MiniPubliFL!C:C;0));-1)
    
    Returns:
        DataFrame contenant le mapping CODE_METI -> Mini Publication FL
    """
    file_path = os.path.join(os.path.dirname(__file__), 'Stock mini publication FL vrac.xlsx')
    
    if not os.path.exists(file_path):
        print(f"Erreur: Fichier {file_path} non trouvé")
        return pd.DataFrame(columns=['CODE_METI', 'Mini Publication FL'])
    
    try:
        # Charger le fichier Excel
        df_mini_publi = pd.read_excel(file_path, index_col=None)
        
        print(f"Fichier Mini Publication FL chargé avec {len(df_mini_publi)} lignes")
        print(f"Colonnes disponibles dans le fichier: {df_mini_publi.columns.tolist()}")
        
        # Vérifier que les colonnes nécessaires existent
        if 'METI' not in df_mini_publi.columns or 'Stock Mini Publication' not in df_mini_publi.columns:
            print("Erreur: Colonnes METI ou Stock Mini Publication manquantes")
            return pd.DataFrame(columns=['CODE_METI', 'Mini Publication FL'])
        
        # Créer un nouveau DataFrame avec uniquement les colonnes nécessaires
        # et renommer METI en CODE_METI pour la fusion avec les autres DataFrames
        result_df = pd.DataFrame({
            'CODE_METI': df_mini_publi['METI'].astype(str),
            'Mini Publication FL': df_mini_publi['Stock Mini Publication']
        })
        
        # Nettoyer les données et s'assurer que CODE_METI est en string
        result_df['CODE_METI'] = result_df['CODE_METI'].astype(str).str.strip()
        
        # Convertir Mini Publication FL en numérique, avec gestion des erreurs
        result_df['Mini Publication FL'] = pd.to_numeric(result_df['Mini Publication FL'], errors='coerce')
        
        # Remplacer les NaN par -1 selon la formule SIERREUR(...;-1)
        result_df['Mini Publication FL'] = result_df['Mini Publication FL'].fillna(-1)
        
        # Supprimer les doublons (garder la première occurrence)
        result_df = result_df.drop_duplicates(subset=['CODE_METI'], keep='first')
        
        print(f"Données Mini Publication FL traitées: {len(result_df)} lignes uniques")
        return result_df
        
    except Exception as e:
        print(f"Erreur lors du chargement du fichier Mini Publication FL: {e}")
        return pd.DataFrame(columns=['CODE_METI', 'Mini Publication FL'])

if __name__ == "__main__":
    # Test du module
    result = get_processed_data()
    print(result.head(10))
    print(f"Total des lignes: {len(result)}")
