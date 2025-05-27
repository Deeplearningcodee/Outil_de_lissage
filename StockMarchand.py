import pandas as pd
import os
import glob

def get_stock_marchand_data():
    """
    Charge les données du stock marchand depuis le fichier Excel qui commence par "Calcul des stocks marchands"
    dans le dossier Stock_Marchand.
    Filtre les données pour ne garder que les lignes où la colonne "Calcul" est égale à "Oui".
    
    Returns:
        DataFrame contenant les colonnes spécifiées du stock marchand
    """
    # Chemin du dossier Stock_Marchand
    stock_marchand_dir = os.path.join(os.path.dirname(__file__), 'Stock_Marchand')
    
    # Rechercher les fichiers commençant par "Calcul des stocks marchands"
    pattern = os.path.join(stock_marchand_dir, "Calcul des stocks marchands*")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Erreur: Aucun fichier commençant par 'Calcul des stocks marchands' trouvé dans {stock_marchand_dir}")
        return pd.DataFrame(columns=['FAMILLE_HYPER', 'Libellé_Famille', 'SS_FAMILLE_HYPER', 'Libellé_Sous_Famille', 
                                     'Ean_13', 'LIBELLE_REFERENCE', 'CODE_METI', 'Nb de commande / Semaine Final',
                                     'Macro-catégorie', 'Classe de stockage', 'Entrepôt', 'Période de vente finale (Jours)',
                                     'VMJ Utilisée Stock Sécurité', 'Stock Max', 'SM Final'])
    
    # Utiliser le fichier le plus récent si plusieurs sont trouvés
    file_path = max(files, key=os.path.getmtime)
    print(f"Chargement du fichier stock marchand: {os.path.basename(file_path)}")
    
    try:
        # Charger la feuille "Calcul" du fichier Excel
        df = pd.read_excel(file_path, sheet_name="Calcul", engine='openpyxl')
        
        # Vérifier que les colonnes nécessaires existent
        required_columns = ['FAMILLE_HYPER', 'Libellé_Famille', 'SS_FAMILLE_HYPER', 'Libellé_Sous_Famille', 
                           'Ean_13', 'LIBELLE_REFERENCE', 'CODE_METI', 'Nb de commande / Semaine Final',
                           'Macro-catégorie', 'Classe de stockage', 'Entrepôt', 'Période de vente finale (Jours)',
                           'VMJ Utilisée Stock Sécurité', 'Stock Max', 'SM Final']
        
        # Vérifier si la colonne "Calcul" existe
        if 'Calcul' not in df.columns:
            print(f"Attention: La colonne 'Calcul' est absente du fichier {os.path.basename(file_path)}")
            print("Le filtre sur 'Calcul' = 'Oui' ne sera pas appliqué.")
        else:
            # Filtrer pour ne garder que les lignes où Calcul = "Oui"
            df = df[df['Calcul'] == "Oui"]
            print(f"Filtrage appliqué: {len(df)} lignes où Calcul = 'Oui'")
        
        # Vérifier quelles colonnes sont présentes dans le DataFrame
        available_columns = df.columns.tolist()
        missing_columns = [col for col in required_columns if col not in available_columns]
        
        if missing_columns:
            print(f"Attention: Colonnes manquantes dans le fichier {os.path.basename(file_path)}: {missing_columns}")
            print(f"Colonnes disponibles: {available_columns}")
            
            # Essayer de récupérer les colonnes disponibles et créer des colonnes vides pour les manquantes
            result_df = pd.DataFrame()
            for col in required_columns:
                if col in df.columns:
                    result_df[col] = df[col]
                else:
                    result_df[col] = None
        else:
            # Filtrer pour ne garder que les colonnes requises
            result_df = df[required_columns].copy()
        
        # Convertir CODE_METI en string
        result_df['CODE_METI'] = result_df['CODE_METI'].astype(str)
        
        # Convertir les colonnes numériques
        numeric_cols = ['Nb de commande / Semaine Final', 'Période de vente finale (Jours)',
                        'VMJ Utilisée Stock Sécurité', 'Stock Max', 'SM Final']
        
        for col in numeric_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)
        
        # Supprimer les doublons
        result_df = result_df.drop_duplicates(subset=['CODE_METI'], keep='first')
        
        return result_df
        
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {os.path.basename(file_path)}: {e}")
        return pd.DataFrame(columns=required_columns)

if __name__ == "__main__":
    # Test du module
    result = get_stock_marchand_data()
    print(f"Nombre de lignes chargées: {len(result)}")
    print("\nAperçu des données:")
    print(result.head())
    
    # Vérification des données
    print("\nTypes de données:")
    print(result.dtypes)
    
    if 'CODE_METI' in result.columns:
        print(f"\nNombre de CODE_METI uniques: {result['CODE_METI'].nunique()}")