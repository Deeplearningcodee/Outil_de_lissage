import pandas as pd
import os
import glob

def get_stock_marchand_data():
    """
    Charge les données du stock marchand depuis le fichier Excel qui commence par "Outil Lissage Approvisionnements"
    Charge toutes les données de la feuille "Détail".
    
    Returns:
        DataFrame contenant les colonnes spécifiées du stock marchand
    """
    
    
    # Rechercher les fichiers commençant par "Outil Lissage Approvisionnements" in this directory
    pattern = os.path.join(  os.path.dirname(__file__), '*Outil Lissage Approvisionnements *.xlsm')
    files = glob.glob(pattern)
    
    if not files:
        print(f"Erreur: Aucun fichier commençant par 'Outil Lissage Approvisionnements' trouvé ")
        return pd.DataFrame(columns=['FAMILLE_HYPER', 'Libellé_Famille', 'SS_FAMILLE_HYPER', 'Libellé_Sous_Famille', 
                                     'Ean_13', 'LIBELLE_REFERENCE', 'CODE_METI', 'Nb de commande / Semaine Final',
                                     'Macro-catégorie', 'Classe de stockage', 'Entrepôt', 'Période de vente finale (Jours)',
                                     'VMJ Utilisée Stock Sécurité', 'Stock Max', 'SM Final'])
    
    # Utiliser le fichier le plus récent si plusieurs sont trouvés
    file_path = max(files, key=os.path.getmtime)
    print(f"Chargement du fichier stock marchand: {os.path.basename(file_path)}")
    try:
        # Charger la feuille "Détail" du fichier Excel
        df = pd.read_excel(file_path, sheet_name="Détail", engine='openpyxl')
        
        # Vérifier que les colonnes nécessaires existent
        required_columns = ['FAMILLE_HYPER', 'Libellé_Famille', 'SS_FAMILLE_HYPER', 'Libellé_Sous_Famille', 
                           'Ean_13', 'LIBELLE_REFERENCE', 'CODE_METI', 'Nb de commande / Semaine Final',
                           'Macro-catégorie', 'Classe de stockage', 'Entrepôt', 'Période de vente finale (Jours)',
                           'VMJ Utilisée Stock Sécurité', 'Stock Max', 'SM Final']
        
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
          # Convertir CODE_METI en string et nettoyer les décimales
        result_df['CODE_METI'] = pd.to_numeric(result_df['CODE_METI'], errors='coerce').fillna(0).astype(int).astype(str)
        
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

def generate_top_csv_files(df):
    """
    Génère les fichiers Top_500.csv et Top_3000.csv basés sur le classement VMJ.
    Structure: EAN Final;Libellé;Code Méti;VMJ;Classement;Ligne;Contrôle
    
    Args:
        df: DataFrame contenant les données du stock marchand
    """
    # Créer le dossier CSV s'il n'existe pas
    csv_dir = os.path.join(os.path.dirname(__file__), 'CSV')
    os.makedirs(csv_dir, exist_ok=True)
    
    # Vérifier que les colonnes nécessaires existent
    if 'VMJ Utilisée Stock Sécurité' not in df.columns:
        print("Erreur: La colonne 'VMJ Utilisée Stock Sécurité' est nécessaire pour générer les fichiers Top")
        return
    
    # Filtrer les lignes avec VMJ valide (non-null et > 0)
    df_valid = df[df['VMJ Utilisée Stock Sécurité'].notna() & (df['VMJ Utilisée Stock Sécurité'] > 0)].copy()
    
    if len(df_valid) == 0:
        print("Aucune donnée valide trouvée pour générer les fichiers Top")
        return
    
    # Trier par VMJ décroissant pour le classement
    df_sorted = df_valid.sort_values('VMJ Utilisée Stock Sécurité', ascending=False).reset_index(drop=True)
    
    # Ajouter le classement
    df_sorted['Classement'] = range(1, len(df_sorted) + 1)
    df_sorted['Ligne'] = range(1, len(df_sorted) + 1)
    df_sorted['Contrôle'] = ''  # Colonne vide pour contrôle
    
    # Préparer les colonnes pour le CSV
    columns_mapping = {
        'Ean_13': 'EAN Final',
        'LIBELLE_REFERENCE': 'Libellé',
        'CODE_METI': 'Code Méti',
        'VMJ Utilisée Stock Sécurité': 'VMJ',
        'Classement': 'Classement',
        'Ligne': 'Ligne',
        'Contrôle': 'Contrôle'
    }
    
    # Créer le DataFrame final avec les bonnes colonnes
    result_df = pd.DataFrame()
    for old_col, new_col in columns_mapping.items():
        if old_col in df_sorted.columns:
            result_df[new_col] = df_sorted[old_col]
        else:
            result_df[new_col] = ''
    
    # Générer Top_500.csv
    if len(result_df) >= 500:
        top_500 = result_df.head(500)
        top_500_path = os.path.join(csv_dir, 'Top_500.csv')
        top_500.to_csv(top_500_path, sep=';', index=False, encoding='utf-8-sig')
        print(f"Fichier Top_500.csv généré: {top_500_path}")
    else:
        print(f"Attention: Seulement {len(result_df)} lignes disponibles, impossible de générer Top_500.csv")
    
    # Générer Top_3000.csv
    if len(result_df) >= 3000:
        top_3000 = result_df.head(3000)
        top_3000_path = os.path.join(csv_dir, 'Top_3000.csv')
        top_3000.to_csv(top_3000_path, sep=';', index=False, encoding='utf-8-sig')
        print(f"Fichier Top_3000.csv généré: {top_3000_path}")
    else:
        print(f"Attention: Seulement {len(result_df)} lignes disponibles, impossible de générer Top_3000.csv")
        # Générer quand même le fichier avec toutes les données disponibles
        all_data_path = os.path.join(csv_dir, 'Top_3000.csv')
        result_df.to_csv(all_data_path, sep=';', index=False, encoding='utf-8-sig')
        print(f"Fichier Top_3000.csv généré avec {len(result_df)} lignes: {all_data_path}")

def process_and_generate_files():
    """
    Fonction principale qui charge les données et génère les fichiers CSV.
    """
    print("Chargement des données du stock marchand...")
    df = get_stock_marchand_data()
    
    if len(df) == 0:
        print("Aucune donnée chargée, arrêt du traitement.")
        return
    
    print(f"Données chargées: {len(df)} lignes")
    print("\nGénération des fichiers CSV...")
    generate_top_csv_files(df)
    print("Traitement terminé.")

if __name__ == "__main__":
    # Test du module et génération des fichiers CSV
    process_and_generate_files()
    
    # Test détaillé pour debug
    result = get_stock_marchand_data()
    print(f"\nNombre de lignes chargées: {len(result)}")
    print("\nAperçu des données:")
    print(result.head())
    
    # Vérification des données
    print("\nTypes de données:")
    print(result.dtypes)
    print(result.head())
    
    if 'CODE_METI' in result.columns:
        print(f"\nNombre de CODE_METI uniques: {result['CODE_METI'].nunique()}")
    
    if 'VMJ Utilisée Stock Sécurité' in result.columns:
        print(f"VMJ min: {result['VMJ Utilisée Stock Sécurité'].min()}")
        print(f"VMJ max: {result['VMJ Utilisée Stock Sécurité'].max()}")
        print(f"VMJ moyenne: {result['VMJ Utilisée Stock Sécurité'].mean():.2f}")