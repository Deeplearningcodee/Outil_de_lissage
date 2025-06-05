# Fichier: TypeProduits.py
import pandas as pd
import numpy as np
import os
import glob
import MacroParam

# Importer directement les dictionnaires et valeurs de MacroParam.py
try:
    from MacroParam import (
        TYPES_PRODUITS, 
        GESTION_AB_AC_MAPPING, 
        DATE_REF_JOUR_AB,
        DATE_REF_JOUR_AC
    )
    MACROPARAM_LOADED = True
except ImportError:
    print("AVERTISSEMENT (TypeProduits.py): MacroParam.py non trouvé. Utilisation de valeurs par défaut.")
    MACROPARAM_LOADED = False
    # Définir des valeurs par défaut si MacroParam.py n'est pas disponible
    TYPES_PRODUITS = {
        "1060": "Sec Méca", "1061": "Sec Homogène", "1062": "Sec Hétérogène",
        "1063": "Sec Hétérogène", "1064": "Sec Hétérogène", "1070": "Frais Méca",
        "1071": "Frais Manuel", "1080": "Surgelés"
    }
    GESTION_AB_AC_MAPPING = {
        "Sec Méca": "Oui", "Sec Homogène": "Oui", "Sec Hétérogène": "Oui",
        "Frais Méca": "Non", "Frais Manuel": "Non", "Surgelés": "Non"
    }
    DATE_REF_JOUR_AB = MacroParam.DATE_REF_JOUR_AB
    DATE_REF_JOUR_AC = MacroParam.DATE_REF_JOUR_AC


def calc_type_produits(df):
    """Calcule 'Type de produits' basé sur CLASSE_STOCKAGE."""
    print("  TypeProduits - Calcul de 'Type de produits'...")
    stockage_col_name = None
    if 'CLASSE_STOCKAGE' in df.columns:
        stockage_col_name = 'CLASSE_STOCKAGE'
    elif 'Classe_Stockage' in df.columns: # Ancienne nomenclature possible
        stockage_col_name = 'Classe_Stockage'
    
    if stockage_col_name is None:
        print("    ATTENTION (calc_type_produits): Colonne 'CLASSE_STOCKAGE' non trouvée. 'Type de produits' sera 'Autre'.")
        df['Type de produits'] = "Autre"
        return df

    # S'assurer que la colonne est en str pour le mapping, prendre la partie avant un éventuel '.'
    df['__temp_stockage_key__'] = df[stockage_col_name].astype(str).str.split('.').str[0].str.strip()
    df['Type de produits'] = df['__temp_stockage_key__'].map(TYPES_PRODUITS).fillna("Autre")
    df.drop(columns=['__temp_stockage_key__'], inplace=True, errors='ignore')
    print("  TypeProduits - 'Type de produits' calculé.")
    return df

def calc_type_produits_v2(df):
    """ Calcule "Type de produits V2" en utilisant DATE_LIVRAISON_V2 et les dates de référence."""
    print("  TypeProduits - Calcul de 'Type de produits V2'...")
    if 'Type de produits' not in df.columns:
        df = calc_type_produits(df) # S'assurer que 'Type de produits' existe
    
    if 'DATE_LIVRAISON_V2' not in df.columns:
        print("    ERREUR (calc_type_produits_v2): Colonne 'DATE_LIVRAISON_V2' manquante.")
        df['Type de produits V2'] = df['Type de produits'].astype(str).str.strip().str.lower()
        return df
    
    df['DATE_LIVRAISON_V2'] = pd.to_datetime(df['DATE_LIVRAISON_V2'], errors='coerce')

    # Récupérer les dates de référence globales
    # (déjà importées ou définies en haut du fichier)
    date_ref_ab = DATE_REF_JOUR_AB
    date_ref_ac = DATE_REF_JOUR_AC

    def determine_type_v2(row):
        type_produit_base = str(row.get('Type de produits', 'Autre')).strip()
        date_livraison_article = row.get('DATE_LIVRAISON_V2')

        if not type_produit_base.startswith('Sec'):
            return type_produit_base.lower()

        if pd.isna(date_livraison_article):
            # print(f"    DEBUG determine_type_v2: Date livraison NaN pour {type_produit_base}. Fallback à 'sec - autre'.")
            return "sec - autre" 

        gestion_ab_ac = GESTION_AB_AC_MAPPING.get(type_produit_base, "Non") # Valeur par défaut 'Non'

        if gestion_ab_ac == "Non" and date_livraison_article <= date_ref_ac:
            return type_produit_base.lower()
        else:
            if date_livraison_article == date_ref_ab:
                return f"{type_produit_base} - A/B".lower()
            elif date_livraison_article == date_ref_ac:
                return f"{type_produit_base} - A/C".lower()
            else:
                return "sec - autre" 

    df['Type de produits V2'] = df.apply(determine_type_v2, axis=1)
    print("  TypeProduits - 'Type de produits V2' calculé.")
    return df

def calc_top(df):
    """Calcule 'Top' (Top 500, Top 3000, Autre) en chargeant les données depuis le fichier Excel."""
    print("  TypeProduits - Calcul de 'Top'...")
    
    script_dir = os.path.dirname(__file__) 
    
    # Search for the Excel file using glob
    excel_pattern = os.path.join(script_dir, "*Lissage Approvisionnements*.xlsm")
    excel_files = glob.glob(excel_pattern)
    
    if not excel_files:
        excel_pattern = os.path.join(script_dir, "SQF Outil Lissage Approvisionnements*.xlsm")
        excel_files = glob.glob(excel_pattern)
    
    df_top500, df_top3000 = pd.DataFrame(), pd.DataFrame()
    ean_col_in_top_files = 'EAN Final' 
    codemeti_col_in_top_files = 'Code Méti'

    if excel_files:
        excel_file = excel_files[0]  # Use the first matching file
        print(f"    Lecture du fichier Excel: {os.path.basename(excel_file)}")
        
        try:
            # Read Top 500 sheet
            df_top500 = pd.read_excel(excel_file, sheet_name='Top 500', dtype=str)
            df_top500.columns = df_top500.columns.str.strip()
            print(f"    Feuille 'Top 500' chargée ({len(df_top500)} lignes). Colonnes: {df_top500.columns.tolist()}")
            
            if ean_col_in_top_files not in df_top500.columns:
                print(f"      ATTENTION: Colonne '{ean_col_in_top_files}' non trouvée dans 'Top 500'")
            if codemeti_col_in_top_files not in df_top500.columns:
                print(f"      ATTENTION: Colonne '{codemeti_col_in_top_files}' non trouvée dans 'Top 500'")
            
        except Exception as e:
            print(f"    ERREUR: Impossible de lire la feuille 'Top 500': {e}")
            df_top500 = pd.DataFrame()
        
        try:
            # Read Top 3000 sheet
            df_top3000 = pd.read_excel(excel_file, sheet_name='Top 3000', dtype=str)
            df_top3000.columns = df_top3000.columns.str.strip()
            print(f"    Feuille 'Top 3000' chargée ({len(df_top3000)} lignes). Colonnes: {df_top3000.columns.tolist()}")
            
            if ean_col_in_top_files not in df_top3000.columns:
                print(f"      ATTENTION: Colonne '{ean_col_in_top_files}' non trouvée dans 'Top 3000'")
            if codemeti_col_in_top_files not in df_top3000.columns:
                print(f"      ATTENTION: Colonne '{codemeti_col_in_top_files}' non trouvée dans 'Top 3000'")
            
        except Exception as e:
            print(f"    ERREUR: Impossible de lire la feuille 'Top 3000': {e}")
            df_top3000 = pd.DataFrame()
            
    else:
        print(f"    ERREUR (calc_top): Aucun fichier Excel correspondant trouvé avec le motif: {excel_pattern}")
        print(f"    Recherche dans le répertoire: {script_dir}")
        # Fallback: try direct filename
        direct_excel_file = os.path.join(script_dir, "SQF Outil Lissage Approvisionnements V15-6-11.xlsm")
        if os.path.exists(direct_excel_file):
            print(f"    Fichier Excel trouvé directement: {direct_excel_file}")
            try:
                df_top500 = pd.read_excel(direct_excel_file, sheet_name='Top 500', dtype=str)
                df_top500.columns = df_top500.columns.str.strip()
                df_top3000 = pd.read_excel(direct_excel_file, sheet_name='Top 3000', dtype=str)
                df_top3000.columns = df_top3000.columns.str.strip()
                print(f"    Données Excel chargées - Top 500: {len(df_top500)} lignes, Top 3000: {len(df_top3000)} lignes")
            except Exception as e:
                print(f"    ERREUR: Impossible de lire le fichier Excel: {e}")
        else:
            print(f"    ERREUR: Fichier Excel non trouvé à {direct_excel_file}")    # Build sets from Excel data
    top500_ean_set, top500_code_set = set(), set()
    if not df_top500.empty:
        if ean_col_in_top_files in df_top500.columns: 
            top500_ean_set = set(df_top500[ean_col_in_top_files].astype(str).str.strip().str.replace(",", "."))
        if codemeti_col_in_top_files in df_top500.columns: 
            top500_code_set = set(df_top500[codemeti_col_in_top_files].astype(str).str.strip())

    top3000_ean_set, top3000_code_set = set(), set()
    if not df_top3000.empty:
        if ean_col_in_top_files in df_top3000.columns: 
            top3000_ean_set = set(df_top3000[ean_col_in_top_files].astype(str).str.strip().str.replace(",", "."))
        if codemeti_col_in_top_files in df_top3000.columns: 
            top3000_code_set = set(df_top3000[codemeti_col_in_top_files].astype(str).str.strip())
    
    print(f"    Sets créés - Top 500: {len(top500_ean_set)} EAN, {len(top500_code_set)} Code Méti")
    print(f"    Sets créés - Top 3000: {len(top3000_ean_set)} EAN, {len(top3000_code_set)} Code Méti")
    
    ean_col_to_use = None
    if 'Ean_13' in df.columns: ean_col_to_use = 'Ean_13' 
    elif 'EAN Final' in df.columns: ean_col_to_use = 'EAN Final' 
    elif 'EAN' in df.columns: ean_col_to_use = 'EAN'
    
    codemeti_col_to_use = None
    if 'CODE_METI' in df.columns: codemeti_col_to_use = 'CODE_METI'
    # Si vous avez un 'Code Méti' dans df (le merged_df), ajoutez une condition ici aussi
    elif 'Code Méti' in df.columns: codemeti_col_to_use = 'Code Méti'


    def get_top_category(row):
        ean_val_str, code_meti_val_str = None, None
        
        if ean_col_to_use and pd.notna(row.get(ean_col_to_use)): 
            ean_val_str = str(row.get(ean_col_to_use)).strip().replace(",", ".") # CORRIGÉ: regex=False enlevé
            if ean_val_str.endswith('.0'):
                ean_val_str = ean_val_str[:-2]

        if codemeti_col_to_use and pd.notna(row.get(codemeti_col_to_use)): 
            code_meti_val_str = str(row.get(codemeti_col_to_use)).strip()
            if code_meti_val_str.endswith('.0'): 
                code_meti_val_str = code_meti_val_str[:-2]

        if (ean_val_str and ean_val_str in top500_ean_set) or \
           (code_meti_val_str and code_meti_val_str in top500_code_set):
            return "top 500"
        if (ean_val_str and ean_val_str in top3000_ean_set) or \
           (code_meti_val_str and code_meti_val_str in top3000_code_set):
            return "top 3000"
        return "autre"

    if not ean_col_to_use and not codemeti_col_to_use:
        print("    ATTENTION (calc_top): Ni colonne EAN ni CODE_METI utilisable trouvée dans le DataFrame principal. 'Top' sera 'autre'.")
        df['Top'] = "autre"
    else:
        print(f"    Utilisation de '{ean_col_to_use}' pour EAN et '{codemeti_col_to_use}' pour CODE_METI pour la recherche Top.")
        df['Top'] = df.apply(get_top_category, axis=1)
    
    print("  TypeProduits - 'Top' calculé.")
    return df

# La fonction get_processed_data doit aussi être corrigée pour le parsing des dates
def get_processed_data(input_df):
    print("TypeProduits - Début du traitement...")
    if not isinstance(input_df, pd.DataFrame):
        if isinstance(input_df, str) and os.path.exists(input_df):
            try:
                print(f"  TypeProduits - Lecture du fichier d'entrée: {input_df}")
                # Si init_merged_predictions.csv utilise des dates AAAA-MM-JJ, dayfirst=False est mieux
                df_copy = pd.read_csv(input_df, sep=';', encoding='latin1', low_memory=False)
            except Exception as e:
                raise ValueError(f"L'entrée de TypeProduits.get_processed_data n'est pas un DataFrame valide et n'a pas pu être lue comme CSV: {e}")
        else:
            raise ValueError("L'entrée de TypeProduits.get_processed_data doit être un DataFrame Pandas ou un chemin vers un fichier CSV valide.")
    else:
        df_copy = input_df.copy() 
    
    if 'DATE_LIVRAISON_V2' in df_copy.columns:
        # Pandas est bon pour deviner AAAA-MM-JJ, dayfirst=True n'est pas nécessaire et cause un warning si le format est YYYY-MM-DD
        df_copy['DATE_LIVRAISON_V2'] = pd.to_datetime(df_copy['DATE_LIVRAISON_V2'], errors='coerce') 

    df_copy = calc_type_produits(df_copy)
    df_copy = calc_type_produits_v2(df_copy) 
    df_copy = calc_top(df_copy) 
    
    print("TypeProduits - Traitement terminé.")
    return df_copy


if __name__ == "__main__":
    print("--- Test du module TypeProduits.py avec init_merged_predictions.csv ---")
    
    if not MACROPARAM_LOADED:
        print("\nATTENTION MAJEURE: MacroParam.py non chargé. Valeurs par défaut utilisées.\n")

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Note: Les fichiers Top_500.csv et Top_3000.csv seront créés automatiquement 
    # par calc_top() s'ils n'existent pas
    
    test_input_file = os.path.join(current_script_dir, "initial_merged_predictions.csv")

    if not os.path.exists(test_input_file):
        print(f"ATTENTION: Fichier de test d'entrée '{test_input_file}' non trouvé.")
        demo_init_data = {
            'CLASSE_STOCKAGE': ["1060", "1061", "1070", "1060", "1062", "1080"], 
            'DATE_LIVRAISON_V2': ["2025-05-22", "2025-05-22", "2025-05-22", "2025-05-21", "2025-05-21", "2025-05-22"], 
            'Ean_13': ['EAN_TOP500_DEMO1', 'EAN_TOP3000_DEMO1', 'EAN_AUTRE_TEST', '3270190000767', 'EAN_SHOULD_BE_OTHER', 'EAN_XYZ_IN_TOP3000'],
            'CODE_METI': ['CM_TOP500_DEMO1', 'CM_TOP3000_DEMO1', 'CM_AUTRE_TEST', '18087_TOP500_DEMO', 'CM_SHOULD_BE_OTHER', 'CM_ABC_IN_TOP3000']
        }
        pd.DataFrame(demo_init_data).to_csv(test_input_file, sep=';', index=False, encoding='latin1')
        print(f"Fichier de démo '{test_input_file}' créé. Veuillez le vérifier et relancer.")
        exit()
    else:
        print(f"Utilisation du fichier d'entrée: {test_input_file}")
        
    df_processed_from_file = get_processed_data(test_input_file) 
    
    print("\nRésultats du test de TypeProduits.py (depuis fichier):")
    cols_to_display = ['CLASSE_STOCKAGE', 'DATE_LIVRAISON_V2', 
                       'Ean_13', 'CODE_METI', 
                       'Type de produits', 'Type de produits V2', 'Top']
    # S'assurer que les colonnes existent avant de les afficher
    cols_to_print = [col for col in cols_to_display if col in df_processed_from_file.columns]
    if cols_to_print:
        print(df_processed_from_file[cols_to_print].to_string(index=False))
    else:
        print("Aucune des colonnes de débogage spécifiées n'a été trouvée dans le DataFrame traité.")

