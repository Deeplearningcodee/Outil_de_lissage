import pandas as pd
import os
import glob
import MacroParam # Assuming MacroParam.py is in the same directory 'Outil_de_lissage'
from datetime import datetime # timedelta might not be needed if direct comparison works
import numpy as np # Added for np.number

def find_sqf_file():
    """
    Use glob to find SQF Outil Lissage Approvisionnements files in the current directory and parent directories.
    Returns the path to the most recent file found.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define search patterns
    patterns = [
        "SQF Outil Lissage Approvisionnements*.xlsm",
        "SQF Outil Lissage Approvisionnements*.xlsx"
    ]
    
    # Search in multiple directories
    search_dirs = [
        script_dir,  # Current script directory
        os.path.dirname(script_dir),  # Parent directory
        os.path.join(script_dir, ".."),  # Parent directory (alternative)
        os.path.join(script_dir, "data"),  # Data subdirectory
        os.path.join(script_dir, "files"),  # Files subdirectory
    ]
    
    found_files = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for pattern in patterns:
                search_pattern = os.path.join(search_dir, pattern)
                files = glob.glob(search_pattern)
                found_files.extend(files)
    
    if not found_files:
        print("Aucun fichier SQF Outil Lissage Approvisionnements trouvé.")
        print(f"Répertoires recherchés: {search_dirs}")
        print(f"Motifs recherchés: {patterns}")
        return None
    
    # Sort by modification time and return the most recent
    found_files.sort(key=os.path.getmtime, reverse=True)
    selected_file = found_files[0]
    
    print(f"Fichier SQF trouvé: {selected_file}")
    if len(found_files) > 1:
        print(f"Autres fichiers trouvés: {found_files[1:]}")
    
    return selected_file

def detect_sheet_with_data(file_path, required_columns):
    """
    Detect which sheet contains the required columns.
    Returns the sheet name that contains all required columns.
    """
    try:
        # Get all sheet names
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        print(f"Feuilles disponibles dans le fichier: {sheet_names}")
        
        for sheet_name in sheet_names:
            try:
                # Read just the header to check columns
                df_header = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)
                available_columns = df_header.columns.tolist()
                
                # Check if all required columns are present
                missing_columns = [col for col in required_columns if col not in available_columns]
                
                if not missing_columns:
                    print(f"Feuille '{sheet_name}' contient toutes les colonnes requises.")
                    return sheet_name
                else:
                    print(f"Feuille '{sheet_name}' - Colonnes manquantes: {missing_columns}")
                    
            except Exception as e:
                print(f"Erreur lors de la lecture de la feuille '{sheet_name}': {e}")
                continue
        
        # If no sheet has all columns, try to find the best match
        best_match = None
        best_score = 0
        
        for sheet_name in sheet_names:
            try:
                df_header = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)
                available_columns = df_header.columns.tolist()
                
                # Count how many required columns are present
                score = sum(1 for col in required_columns if col in available_columns)
                
                if score > best_score:
                    best_score = score
                    best_match = sheet_name
                    
            except Exception:
                continue
        
        if best_match and best_score > 0:
            print(f"Meilleure correspondance: feuille '{best_match}' avec {best_score}/{len(required_columns)} colonnes.")
            return best_match
        
        return None
        
    except Exception as e:
        print(f"Erreur lors de la détection de la feuille: {e}")
        return None

def load_pdc_perm_data():
    """
    Charge les données depuis le fichier SQF Outil Lissage Approvisionnements détecté automatiquement.
    Extrait les colonnes: Jour, Sec Hétérogène, Sec Homogène, Sec Méca, Surgelés, Frais Méca, Frais Manuel, Total
    """
    # Required columns to extract
    required_columns = ['Jour', 'Sec Hétérogène', 'Sec Homogène', 'Sec Méca', 'Surgelés', 'Frais Méca', 'Frais Manuel', 'Total']
    
    # Find the SQF file
    file_path = find_sqf_file()
    if not file_path:
        print("Aucun fichier SQF trouvé. Impossible de charger les données.")
        return pd.DataFrame()
    
    try:
        # Detect which sheet contains the required data
        sheet_name = detect_sheet_with_data(file_path, required_columns)
        
        if not sheet_name:
            print("Aucune feuille contenant les colonnes requises n'a été trouvée.")
            return pd.DataFrame()
        
        # Load the data from the detected sheet
        print(f"Chargement des données depuis la feuille '{sheet_name}'...")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        print(f"Colonnes disponibles: {df.columns.tolist()}")
        
        # Check which required columns are actually present
        available_required_cols = [col for col in required_columns if col in df.columns]
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            print(f"Colonnes manquantes: {missing_cols}")
        
        if 'Jour' not in available_required_cols:
            print("Erreur: La colonne 'Jour' est introuvable. Le chargement a échoué.")
            return pd.DataFrame()
        
        # Select only the available required columns
        df_selected = df[available_required_cols].copy()
        
        # Convert 'Jour' column to datetime
        df_selected['Jour'] = pd.to_datetime(df_selected['Jour'], errors='coerce')
        
        # Remove rows where 'Jour' couldn't be converted to a valid date
        initial_rows = len(df_selected)
        df_selected.dropna(subset=['Jour'], inplace=True)
        final_rows = len(df_selected)
        
        if initial_rows != final_rows:
            print(f"Suppression de {initial_rows - final_rows} lignes avec des dates invalides.")
        
        if df_selected.empty:
            print("Avertissement: Aucune donnée valide après conversion de la colonne 'Jour' et suppression des NaNs.")
            return pd.DataFrame()
        
        # Fill missing values in numeric columns with 0
        numeric_columns = [col for col in available_required_cols if col != 'Jour']
        for col in numeric_columns:
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce').fillna(0.0)
        
        # If 'Total' column is missing, calculate it
        if 'Total' not in df_selected.columns:
            product_columns = [col for col in ['Sec Hétérogène', 'Sec Homogène', 'Sec Méca', 'Surgelés', 'Frais Méca', 'Frais Manuel'] if col in df_selected.columns]
            if product_columns:
                df_selected['Total'] = df_selected[product_columns].sum(axis=1)
                print("Colonne 'Total' calculée automatiquement.")
        
        print(f"Données chargées avec succès depuis {os.path.basename(file_path)}, feuille '{sheet_name}'")
        print(f"Nombre de lignes: {len(df_selected)}")
        print(f"Colonnes chargées: {df_selected.columns.tolist()}")
        print(f"Période des données: du {df_selected['Jour'].min()} au {df_selected['Jour'].max()}")
        
        return df_selected
        
    except FileNotFoundError:
        print(f"Erreur: Le fichier {file_path} n'a pas été trouvé.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Erreur lors du chargement des données depuis {file_path}: {e}")
        return pd.DataFrame()

# Keep the old PDC_FILE_PATH for backward compatibility, but make it dynamic
def get_pdc_file_path():
    """Get the PDC file path dynamically."""
    sqf_file = find_sqf_file()
    if sqf_file:
        return sqf_file
    else:
        # Fallback to old path
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PDC', 'PDC.xlsx')

# Update the global variable to be dynamic
PDC_FILE_PATH = get_pdc_file_path()

def create_pdc_perm_summary(df_pdc_perm_input):
    """
    Crée un résumé basé sur les données de la feuille 'PDC Perm' (chargées comme df_pdc_perm_input)
    et la formule Excel fournie. Le résumé commence à partir de DATE_COMMANDE.
    """
    if df_pdc_perm_input.empty:
        print("Les données PDC Perm en entrée sont vides. Impossible de créer le résumé.")
        return pd.DataFrame()

    # Utiliser une copie pour éviter de modifier le DataFrame original
    df_pdc_perm = df_pdc_perm_input.copy()

    if 'Jour' not in df_pdc_perm.columns or df_pdc_perm['Jour'].isnull().all():
        print("La colonne 'Jour' est manquante, invalide, ou entièrement nulle dans les données PDC Perm. Impossible de créer le résumé.")
        return pd.DataFrame()
    
    # Définir 'Jour' comme index pour faciliter les recherches
    df_pdc_perm = df_pdc_perm.set_index('Jour')
    # S'assurer que l'index (dates) est unique, en gardant la première occurrence en cas de doublons
    df_pdc_perm = df_pdc_perm[~df_pdc_perm.index.duplicated(keep='first')]

    # Charger DATE_COMMANDE et la convertir en datetime
    date_commande_str = MacroParam.DATE_COMMANDE # Accès direct à la variable globale
    try:
        date_commande_dt = pd.to_datetime(date_commande_str, format='%d/%m/%Y')
    except ValueError:
        print(f"Erreur: DATE_COMMANDE '{date_commande_str}' dans MacroParam.py n'est pas dans le format attendu 'dd/mm/yyyy'.")
        return pd.DataFrame()

    # Filtrer les données pour commencer à partir de DATE_COMMANDE (inclus)
    df_pdc_perm_filtered = df_pdc_perm[df_pdc_perm.index >= date_commande_dt].copy()

    if df_pdc_perm_filtered.empty:
        print(f"Aucune donnée PDC Perm trouvée à partir de DATE_COMMANDE ({date_commande_str}). Le résumé sera vide.")
        return pd.DataFrame()

    # Charger les paramètres depuis le module MacroParam
    taux_service_amont_estime = MacroParam.get_param_value("taux_service_amont_estime", 0.92) 
    multiplier = 1.0

    # Définir les colonnes produit pour le tableau résumé
    summary_product_columns = ['Sec Hétérogène', 'Sec Homogène', 'Sec Méca', 'Surgelés', 'Frais Méca', 'Frais Manuel']
    
    # Filtrer pour utiliser uniquement les colonnes produit qui existent réellement dans df_pdc_perm_filtered
    valid_product_columns = [col for col in summary_product_columns if col in df_pdc_perm_filtered.columns]
    
    if not valid_product_columns:
        expected_cols_str = ", ".join(summary_product_columns)
        available_cols_str = ", ".join(df_pdc_perm_filtered.columns)
        print(f"Aucune des colonnes produit attendues ({expected_cols_str}) n'a été trouvée parmi les colonnes disponibles ({available_cols_str}) dans les données PDC Perm filtrées. Impossible de créer le résumé.")
        return pd.DataFrame()
    
    if len(valid_product_columns) < len(summary_product_columns):
        missing_cols = [col for col in summary_product_columns if col not in valid_product_columns]
        print(f"Avertissement: Certaines colonnes produit définies ({missing_cols}) sont absentes des données PDC Perm filtrées. Le résumé utilisera: {valid_product_columns}")

    output_columns = valid_product_columns + ['Total']
    summary_table = pd.DataFrame(index=df_pdc_perm_filtered.index, columns=output_columns, dtype=float)

    for date_k in summary_table.index:
        for col_l in output_columns:
            if col_l == "Total":
                summary_table.loc[date_k, "Total"] = summary_table.loc[date_k, valid_product_columns].sum()
            else:
                raw_pdc_value = df_pdc_perm_filtered.loc[date_k, col_l]
                raw_pdc_value = 0.0 if pd.isna(raw_pdc_value) else float(raw_pdc_value)

                if taux_service_amont_estime is not None and taux_service_amont_estime != 0:
                    calculated_value = raw_pdc_value / (1000 * taux_service_amont_estime)
                else:
                    calculated_value = 0.0 
                    if taux_service_amont_estime == 0:
                         print(f"Avertissement: taux_service_amont_estime est zéro pour la date {date_k}, colonne {col_l}. Résultat mis à 0.")

                summary_table.loc[date_k, col_l] = calculated_value * multiplier
    
    return summary_table

def format_pdc_perm_summary(df_summary):
    """
    Formate le résumé PDC Perm pour l'affichage ou la sortie.
    Arrondit toutes les valeurs numériques à 2 décimales.
    """
    print("Fonction format_pdc_perm_summary appelée.")
    if df_summary.empty:
        print("Le DataFrame de résumé est vide, aucun formatage appliqué.")
        return df_summary
    
    # Arrondir toutes les colonnes numériques à 2 décimales
    numeric_cols = df_summary.select_dtypes(include=np.number).columns
    df_summary[numeric_cols] = df_summary[numeric_cols].round(2)
    
    print("Les valeurs numériques du résumé PDC Perm ont été arrondies à 2 décimales.")
    return df_summary

def get_processed_pdc_perm_data():
    """
    Fonction principale pour obtenir les données PDC Perm traitées et formatées.
    """
    df_pdc = load_pdc_perm_data()
    if not df_pdc.empty:
        df_summary = create_pdc_perm_summary(df_pdc)
        if not df_summary.empty:
            df_formatted = format_pdc_perm_summary(df_summary)
            return df_formatted
        else:
            print("La création du résumé PDC Perm a résulté en un DataFrame vide.")
            return pd.DataFrame()
    else:
        print("Le chargement des données PDC Perm a échoué ou résulté en un DataFrame vide.")
        return pd.DataFrame()
    
def create_pdc_perm_summary_BRUT(df_pdc_perm_input):
    """
    Crée un résumé des données PDC Perm BRUTES (sans division par 1000*TSA).
    L'index est la date, les colonnes sont les types de produits.
    """
    if df_pdc_perm_input.empty:
        print("PDC.py - create_pdc_perm_summary_BRUT: Données PDC Perm en entrée sont vides.")
        return pd.DataFrame()

    df_pdc_perm = df_pdc_perm_input.copy()
    if 'Jour' not in df_pdc_perm.columns or df_pdc_perm['Jour'].isnull().all():
        print("PDC.py - create_pdc_perm_summary_BRUT: Colonne 'Jour' manquante ou invalide.")
        return pd.DataFrame()
    
    df_pdc_perm = df_pdc_perm.set_index('Jour')
    df_pdc_perm = df_pdc_perm[~df_pdc_perm.index.duplicated(keep='first')]

    date_commande_dt = pd.to_datetime(MacroParam.DATE_COMMANDE, format='%d/%m/%Y')
    df_pdc_perm_filtered = df_pdc_perm[df_pdc_perm.index >= date_commande_dt].copy()

    if df_pdc_perm_filtered.empty:
        print(f"PDC.py - create_pdc_perm_summary_BRUT: Aucune donnée PDC Perm trouvée à partir de DATE_COMMANDE.")
        return pd.DataFrame()

    summary_product_columns = ['Sec Hétérogène', 'Sec Homogène', 'Sec Méca', 'Surgelés', 'Frais Méca', 'Frais Manuel']
    valid_product_columns = [col for col in summary_product_columns if col in df_pdc_perm_filtered.columns]
    
    if not valid_product_columns:
        print("PDC.py - create_pdc_perm_summary_BRUT: Aucune colonne produit valide trouvée.")
        return pd.DataFrame()

    # Sélectionner uniquement les colonnes produits valides et l'index
    df_summary_brut = df_pdc_perm_filtered[valid_product_columns].copy()
    
    # Calculer le total si nécessaire
    df_summary_brut['Total'] = df_summary_brut[valid_product_columns].sum(axis=1)
    
    # S'assurer que les types sont numériques, remplacer NaN par 0 pour la somme
    for col in valid_product_columns:
        df_summary_brut[col] = pd.to_numeric(df_summary_brut[col], errors='coerce').fillna(0.0)
    df_summary_brut['Total'] = pd.to_numeric(df_summary_brut['Total'], errors='coerce').fillna(0.0)

    return df_summary_brut.round(2)

def get_RAW_pdc_perm_data_for_optim():
    """
    Fonction principale pour obtenir les données PDC Perm BRUTES, formatées pour l'optimisation.
    """
    print("PDC.py - get_RAW_pdc_perm_data_for_optim: Chargement des données brutes...")
    df_pdc = load_pdc_perm_data()
    if not df_pdc.empty:
        df_summary_brut = create_pdc_perm_summary_BRUT(df_pdc) 
        if not df_summary_brut.empty:
            return df_summary_brut
        else:
            print("PDC.py - get_RAW_pdc_perm_data_for_optim: La création du résumé BRUT a résulté en un DataFrame vide.")
            return pd.DataFrame()
    else:
        print("PDC.py - get_RAW_pdc_perm_data_for_optim: Le chargement des données PDC Perm a échoué.")
        return pd.DataFrame()

if __name__ == '__main__':
    # Test the updated functionality
    print("=== Test de détection automatique de fichier SQF ===")
    print(f"Fichier PDC détecté: {PDC_FILE_PATH}")
    
    print("\n=== Test de chargement des données ===")
    processed_data = get_processed_pdc_perm_data()
    
    if not processed_data.empty:
        print("\nDonnées PDC Perm traitées et formatées:")
        print(processed_data.head().to_string())
        print(f"\nShape: {processed_data.shape}")
    else:
        print("\nAucune donnée PDC Perm n'a été traitée ou le résultat est vide.")

    print("\n=== Test de chargement des données brutes ===")
    df_raw = load_pdc_perm_data()
    if not df_raw.empty:
        print("Données brutes chargées:")
        print(df_raw.head().to_string())
        print(f"Shape: {df_raw.shape}")
    else:
        print("Aucune donnée brute chargée.")