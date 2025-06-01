import pandas as pd
import numpy as np
import os
import glob
from MacroParam import get_param_value # MODIFIED: Import get_param_value instead of facteur_appro_max_frais

# Chemins vers les fichiers source
FACTEUR_FAM_CSV = os.path.join('Facteur_Appro', 'FacteurAppro.csv')
FACTEUR_SF_CSV = os.path.join('Facteur_Appro', 'FacteurApproSF.csv')
FACTEUR_ART_CSV = os.path.join('Facteur_Appro', 'FacteurApproArt.csv')

def load_facteur_famille(file_path): # MODIFIED: Added file_path argument
    """
    Charge les facteurs d'approvisionnement par Famille
    """
    try:
        df = pd.read_csv(
            file_path, # MODIFIED: Use file_path argument
            sep=';',
            encoding='latin1',
            engine='python'
        )
        
        # Renommer les colonnes pour correspondre aux formules Excel
        df = df.rename(columns={
            'N° Famille': 'FAMILLE_HYPER',
            'Facteur': 'Facteur Multiplicatif'
        })
        
        # Convert FAMILLE_HYPER to string and normalize (e.g., "3070.0" to "3070")
        s = df['FAMILLE_HYPER'].astype(str)
        s = s.str.replace(r'\\.0$','', regex=True) # Corrected regex
        df['FAMILLE_HYPER'] = s
        
        # Revised logic for 'Facteur Multiplicatif'
        original_factor_str = df['Facteur Multiplicatif'].astype(str)
        new_factors = pd.Series(np.nan, index=df.index, dtype=float)

        cond_empty = (original_factor_str == '')
        new_factors.loc[cond_empty] = 1.0

        cond_percent = original_factor_str.str.contains('%', na=False) & ~cond_empty
        percent_strings_to_convert = original_factor_str.loc[cond_percent].str.rstrip('%')
        new_factors.loc[cond_percent] = pd.to_numeric(percent_strings_to_convert, errors='coerce') / 100.0

        cond_direct = ~cond_empty & ~cond_percent
        direct_strings_to_convert = original_factor_str.loc[cond_direct]
        new_factors.loc[cond_direct] = pd.to_numeric(direct_strings_to_convert, errors='coerce')
        
        new_factors.fillna(1.0, inplace=True) # Default for errors or unhandled cases
        df['Facteur Multiplicatif'] = new_factors
        
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {file_path}: {e}") # MODIFIED: Use file_path in error message
        return pd.DataFrame(columns=['FAMILLE_HYPER', 'Facteur Multiplicatif'])

def load_facteur_sous_famille(file_path): # MODIFIED: Added file_path argument
    """
    Charge les facteurs d'approvisionnement par Sous-Famille
    """
    try:
        df = pd.read_csv(
            file_path, # MODIFIED: Use file_path argument
            sep=';',
            encoding='latin1',
            engine='python'
        )
        
        # Renommer les colonnes pour correspondre aux formules Excel
        df = df.rename(columns={
            'N° Sous-Famille': 'SS_FAMILLE_HYPER',
            'Facteur': 'Facteur Multiplicatif'
        })
        
        # Convertir SS_FAMILLE_HYPER en string pour le merge et normalize
        s = df['SS_FAMILLE_HYPER'].astype(str)
        s = s.str.replace(r'\\.0$','', regex=True) # Corrected regex
        df['SS_FAMILLE_HYPER'] = s
        
        # Revised logic for 'Facteur Multiplicatif'
        original_factor_str = df['Facteur Multiplicatif'].astype(str)
        new_factors = pd.Series(np.nan, index=df.index, dtype=float)

        cond_empty = (original_factor_str == '')
        new_factors.loc[cond_empty] = 1.0

        cond_percent = original_factor_str.str.contains('%', na=False) & ~cond_empty
        percent_strings_to_convert = original_factor_str.loc[cond_percent].str.rstrip('%')
        new_factors.loc[cond_percent] = pd.to_numeric(percent_strings_to_convert, errors='coerce') / 100.0

        cond_direct = ~cond_empty & ~cond_percent
        direct_strings_to_convert = original_factor_str.loc[cond_direct]
        new_factors.loc[cond_direct] = pd.to_numeric(direct_strings_to_convert, errors='coerce')

        new_factors.fillna(1.0, inplace=True) # Default for errors or unhandled cases
        df['Facteur Multiplicatif'] = new_factors
        
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {file_path}: {e}") # MODIFIED: Use file_path in error message
        return pd.DataFrame(columns=['SS_FAMILLE_HYPER', 'Facteur Multiplicatif'])

def load_facteur_article(file_path): # MODIFIED: Added file_path argument
    """
    Charge les facteurs d'approvisionnement par Article (EAN)
    """
    try:
        df = pd.read_csv(
            file_path, # MODIFIED: Use file_path argument
            sep=';',
            encoding='latin1',
            engine='python'
        )
        
        # Renommer les colonnes pour correspondre aux formules Excel
        df = df.rename(columns={
            'EAN': 'Ean_13',
            'Coefficient': 'Facteur Multiplicatif'
        })
        
        # Convertir Ean_13 en string pour le merge et normalize
        s = df['Ean_13'].astype(str)
        s = s.str.replace(r'\\.0$','', regex=True) # Corrected regex
        df['Ean_13'] = s
        
        # Revised logic for 'Facteur Multiplicatif'
        original_factor_str = df['Facteur Multiplicatif'].astype(str)
        new_factors = pd.Series(np.nan, index=df.index, dtype=float)

        cond_empty = (original_factor_str == '')
        new_factors.loc[cond_empty] = 1.0

        cond_percent = original_factor_str.str.contains('%', na=False) & ~cond_empty
        percent_strings_to_convert = original_factor_str.loc[cond_percent].str.rstrip('%')
        new_factors.loc[cond_percent] = pd.to_numeric(percent_strings_to_convert, errors='coerce') / 100.0

        cond_direct = ~cond_empty & ~cond_percent
        direct_strings_to_convert = original_factor_str.loc[cond_direct]
        new_factors.loc[cond_direct] = pd.to_numeric(direct_strings_to_convert, errors='coerce')

        new_factors.fillna(1.0, inplace=True) # Default for errors or unhandled cases
        df['Facteur Multiplicatif'] = new_factors
        
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {file_path}: {e}") # MODIFIED: Use file_path in error message
        return pd.DataFrame(columns=['Ean_13', 'Facteur Multiplicatif'])

def update_facteur_csvs_from_excel():
    """
    Finds '*Outil Lissage Approvisionnements*.xlsm' in the script's directory,
    reads specified sheets, and updates the corresponding CSV files.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ might not be defined in some execution contexts (e.g. interactive)
        # Fallback to current working directory, though this might be less reliable
        script_dir = os.getcwd()
        print(f"Avertissement: __file__ non défini, utilisation de CWD: {script_dir} comme base pour la recherche de l'Excel.")

    excel_file_pattern = os.path.join(script_dir, '*Outil Lissage Approvisionnements*.xlsm')
    excel_files = glob.glob(excel_file_pattern)

    if not excel_files:
        print(f"Avertissement: Aucun fichier Excel correspondant à '{excel_file_pattern}' n'a été trouvé. Les CSV ne seront pas mis à jour.")
        return
    
    if len(excel_files) > 1:
        print(f"Avertissement: Plusieurs fichiers Excel trouvés: {excel_files}. Utilisation du premier: {excel_files[0]}.")
    
    excel_filepath = excel_files[0]
    print(f"Mise à jour des CSV à partir de: {excel_filepath}")

    config = [
        {
            'sheet_name': 'Facteur Appro',
            'csv_path': FACTEUR_FAM_CSV,
            'expected_csv_columns': ['N° Famille', 'Facteur']
        },
        {
            'sheet_name': 'Facteur Appro SF',
            'csv_path': FACTEUR_SF_CSV,
            'expected_csv_columns': ['N° Sous-Famille', 'Facteur']
        },
        {
            'sheet_name': 'Facteur Appro Art',
            'csv_path': FACTEUR_ART_CSV,
            'expected_csv_columns': ['EAN', 'Coefficient']
        }
    ]

    for item in config:
        try:
            df_sheet = pd.read_excel(excel_filepath, sheet_name=item['sheet_name'])
            
            missing_cols = [col for col in item['expected_csv_columns'] if col not in df_sheet.columns]
            if missing_cols:
                print(f"Erreur: Colonnes manquantes {missing_cols} dans la feuille '{item['sheet_name']}' du fichier Excel '{excel_filepath}'. Le CSV '{item['csv_path']}' ne sera pas mis à jour.")
                continue
                
            df_to_save = df_sheet # Changed to save all columns from the sheet
            
            # Construct full path for CSV relative to script_dir to ensure correct location
            # item['csv_path'] is like 'Facteur_Appro/file.csv'
            # The original FACTEUR_*_CSV constants are relative paths.
            # Writing to these relative paths assumes CWD is the script's directory.
            # This is consistent with how they are read by load_facteur_* functions.
            
            # Ensure the directory for the CSV exists
            # os.path.dirname(item['csv_path']) will give 'Facteur_Appro'
            # This path is relative to the CWD.
            csv_dir = os.path.dirname(item['csv_path'])
            if csv_dir and not os.path.exists(csv_dir):
                 os.makedirs(csv_dir, exist_ok=True)


            df_to_save.to_csv(item['csv_path'], sep=';', encoding='latin1', index=False)
            print(f"CSV '{item['csv_path']}' mis à jour avec les données de la feuille '{item['sheet_name']}'.")
        except Exception as e:
            print(f"Erreur lors de la mise à jour de {item['csv_path']} à partir de la feuille '{item['sheet_name']}' du fichier '{excel_filepath}': {e}")

def apply_facteur_appro(df, base_path="Facteur_Appro"):
    """
    Applique les facteurs d'approvisionnement au DataFrame principal.
    """
    # Récupérer la valeur de facteur_appro_max_frais en utilisant get_param_value
    facteur_appro_max_frais = get_param_value("facteur_appro_max_frais", 1.2) # MODIFIED: Call get_param_value

    # Charger les facteurs
    df_famille = load_facteur_famille(os.path.join(base_path, "FacteurAppro.csv"))
    df_sous_famille = load_facteur_sous_famille(os.path.join(base_path, "FacteurApproSF.csv"))
    df_article = load_facteur_article(os.path.join(base_path, "FacteurApproArt.csv"))

    # Vérifier que la colonne 'Casse Prev C1-L2' existe
    if 'Casse Prev C1-L2' not in df.columns:
        print("Colonne 'Casse Prev C1-L2' manquante, initialisation à 0")
        df['Casse Prev C1-L2'] = 0
    
    # 1. Facteur Multiplicatif Appro Famille
    # =SI(AT2>0;1;SIERREUR(INDEX('Facteur Appro'!G:G;EQUIV(A2;'Facteur Appro'!E:E;0));1))
    def get_facteur_famille(row):
        # Si Casse Prev C1-L2 > 0, retourner 1
        if pd.notna(row['Casse Prev C1-L2']) and row['Casse Prev C1-L2'] > 0:
            return 1.0
        
        # Sinon, chercher le facteur dans le DataFrame df_famille
        if 'FAMILLE_HYPER' in row and pd.notna(row['FAMILLE_HYPER']):
            # Normalize the ID from the row
            id_val_from_row = str(row['FAMILLE_HYPER'])
            if id_val_from_row.endswith('.0'):
                famille = id_val_from_row[:-2]
            else:
                famille = id_val_from_row
            
            match = df_famille[df_famille['FAMILLE_HYPER'] == famille] # MODIFIED: Changed df_fam to df_famille
            if not match.empty:
                return match['Facteur Multiplicatif'].values[0]
        
        # Par défaut, retourner 1
        return 1.0
    
    # 2. Facteur Multiplicatif Appro Sous-Famille
    # =SI(AT2>0;1;SIERREUR(INDEX('Facteur Appro SF'!G:G;EQUIV(C2;'Facteur Appro SF'!E:E;0));1))
    def get_facteur_sous_famille(row):
        # Si Casse Prev C1-L2 > 0, retourner 1
        if pd.notna(row['Casse Prev C1-L2']) and row['Casse Prev C1-L2'] > 0:
            return 1.0
        
        # Sinon, chercher le facteur dans le DataFrame df_sous_famille
        if 'SS_FAMILLE_HYPER' in row and pd.notna(row['SS_FAMILLE_HYPER']):
            # Normalize the ID from the row
            id_val_from_row = str(row['SS_FAMILLE_HYPER'])
            if id_val_from_row.endswith('.0'):
                sous_famille = id_val_from_row[:-2]
            else:
                sous_famille = id_val_from_row
            
            match = df_sous_famille[df_sous_famille['SS_FAMILLE_HYPER'] == sous_famille] # MODIFIED: Changed df_sf to df_sous_famille
            if not match.empty:
                return match['Facteur Multiplicatif'].values[0]
        
        # Par défaut, retourner 1
        return 1.0
    
    # 3. Facteur Multiplicatif Appro Article
    # =SI(AT2>0;1;SIERREUR(INDEX('Facteur Appro Art'!C:C;EQUIV(E2;'Facteur Appro Art'!A:A;0));""))
    def get_facteur_article(row):
        # Si Casse Prev C1-L2 > 0, retourner 1
        if pd.notna(row['Casse Prev C1-L2']) and row['Casse Prev C1-L2'] > 0:
            return 1.0
        
        # Sinon, chercher le facteur dans le DataFrame df_article
        if 'Ean_13' in row and pd.notna(row['Ean_13']):
            # Normalize the ID from the row
            id_val_from_row = str(row['Ean_13'])
            if id_val_from_row.endswith('.0'):
                ean = id_val_from_row[:-2]
            else:
                ean = id_val_from_row
            
            match = df_article[df_article['Ean_13'] == ean] # MODIFIED: Changed df_art to df_article
            if not match.empty:
                return match['Facteur Multiplicatif'].values[0]
        
        # If lookup fails or Ean_13 is not present, return np.nan
        return np.nan
    
    # Appliquer les fonctions pour calculer les facteurs
    df['Facteur Multiplicatif Appro Famille'] = df.apply(get_facteur_famille, axis=1)
    df['Facteur Multiplicatif Appro Sous-Famille'] = df.apply(get_facteur_sous_famille, axis=1)
    df['Facteur Multiplicatif Appro Article'] = df.apply(get_facteur_article, axis=1)
    
    # Calcul du Facteur Multiplicatif Appro final basé sur la logique Excel
    # Si Facteur Multiplicatif Appro Article existe (non NaN), on le prend.
    # Sinon, si Facteur Multiplicatif Appro Sous-Famille est différent de 1.0, on le prend.
    # Sinon (si Article est NaN et Sous-Famille est 1.0), on prend Facteur Multiplicatif Appro Famille.
    # Le tout est capé par facteur_appro_max_frais pour "2 - Frais" et 5.0 pour les autres.

    max_limit = np.where((df['Classe_Stockage'] == 1070.0) | (df['Classe_Stockage'] == 1071.0), facteur_appro_max_frais, 5.0) # MODIFIED: Replaced 'or' with '|' for element-wise operation

    
    val_br = df['Facteur Multiplicatif Appro Famille']
    val_bs = df['Facteur Multiplicatif Appro Sous-Famille']
    val_bt = df['Facteur Multiplicatif Appro Article'] 

    # Logique Excel:
    # =SI(NON(ESTNA(BT2));BT2;SI(BS2<>1;BS2;BR2))
    # Ensuite, appliquer le MIN avec la limite max_limit
    # =MIN(MAX_LIMIT; VALEUR_CALCULEE_CI_DESSUS)

    calculated_factor = np.where(pd.isna(val_bt),
                                 np.where(val_bs == 1.0, val_br, val_bs),
                                 val_bt)
    
    df['Facteur Multiplicatif Appro'] = np.minimum(max_limit, calculated_factor)
    
    return df

def get_processed_data(merged_df=None):
    """
    Fonction principale appelée par main.py
    Calcule les colonnes de facteur d'approvisionnement
    """
    update_facteur_csvs_from_excel() # Update CSV files from Excel first
    
    if merged_df is None:
        print("Erreur: DataFrame merged_df non fourni")
        return pd.DataFrame()
    
    # Appliquer les facteurs d'approvisionnement
    result_df = apply_facteur_appro(merged_df)
    
    return result_df

if __name__ == "__main__":
    # Test du module
    print("Test du module FacteurAppro")
    
    # Créer un DataFrame de test
    test_df = pd.read_csv("initial_merged_predictions.csv", sep=';', encoding='latin1', engine='python')

    
    # Appliquer les facteurs
    result = get_processed_data(test_df)
    
    # Afficher les résultats
    print(result.head())
    #save file
    result.to_csv("result_facteur_appro.csv", sep=';', encoding='latin1', index=False)
