import os
import glob
import pandas as pd
import StockMarchand

# =========================
# 1. Détection automatique des chemins
# =========================
CSV_FOLDER = 'CSV'
script_dir = os.path.dirname(os.path.abspath(__file__)) # Define script_dir

# Recherche du fichier *_Detail_RAO_Commande*.csv dans CSV/ ou script directory
detail_pattern_csv = os.path.join(CSV_FOLDER, '*_Detail_RAO_Commande*.csv')
detail_pattern_scriptdir = os.path.join(script_dir, '*_Detail_RAO_Commande*.csv') # Check in script dir too
detail_files = glob.glob(detail_pattern_csv) + glob.glob(detail_pattern_scriptdir)

if not detail_files:
    # Try searching in a 'CSV' subdirectory relative to the script's location
    detail_pattern_relative_csv = os.path.join(script_dir, CSV_FOLDER, '*_Detail_RAO_Commande*.csv')
    detail_files = glob.glob(detail_pattern_relative_csv)
    if not detail_files:
        # Fallback: Try to find it in the current working directory if all else fails
        detail_pattern_cwd = '*_Detail_RAO_Commande*.csv'
        detail_files = glob.glob(detail_pattern_cwd)
        if not detail_files:
            raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern Detail_RAO_Commande dans {CSV_FOLDER}, {script_dir}, {os.path.join(script_dir, CSV_FOLDER)} ou CWD.")

DETAIL_CSV = detail_files[0]
print(f"Ts.py: Using DETAIL_CSV: {DETAIL_CSV}")


# Recherche du fichier *Taux_Service*.csv
ts_pattern_csv = os.path.join(CSV_FOLDER, '*Taux_Service*.csv')
ts_pattern_scriptdir = os.path.join(script_dir, '*Taux_Service*.csv')
ts_files = glob.glob(ts_pattern_csv) + glob.glob(ts_pattern_scriptdir)

if not ts_files:
    ts_pattern_relative_csv = os.path.join(script_dir, CSV_FOLDER, '*Taux_Service*.csv')
    ts_files = glob.glob(ts_pattern_relative_csv)
    if not ts_files:
        ts_pattern_cwd = '*Taux_Service*.csv'
        ts_files = glob.glob(ts_pattern_cwd)
        if not ts_files:
            raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern Taux_Service dans {CSV_FOLDER}, {script_dir}, {os.path.join(script_dir, CSV_FOLDER)} ou CWD.")
TS_CSV = ts_files[0]
print(f"Ts.py: Using TS_CSV: {TS_CSV}")

# Recherche du fichier *_MacroParam*.csv
macro_pattern_csv = os.path.join(CSV_FOLDER, 'MacroParam*.csv')
macro_pattern_scriptdir = os.path.join(script_dir, 'MacroParam*.csv')
macro_files = glob.glob(macro_pattern_csv) + glob.glob(macro_pattern_scriptdir)

if not macro_files:
    macro_pattern_relative_csv = os.path.join(script_dir, CSV_FOLDER, 'MacroParam*.csv')
    macro_files = glob.glob(macro_pattern_relative_csv)
    if not macro_files:
        macro_pattern_cwd = 'MacroParam*.csv'
        macro_files = glob.glob(macro_pattern_cwd)
        if not macro_files:
            raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern MacroParam dans {CSV_FOLDER}, {script_dir}, {os.path.join(script_dir, CSV_FOLDER)} ou CWD.")
MACROPARAM_CSV = macro_files[0]
print(f"Ts.py: Using MACROPARAM_CSV: {MACROPARAM_CSV}")


def get_processed_data():
    """
    Fonction qui calcule la colonne TS pour chaque CODE_METI et retourne un dataframe
    avec CODE_METI et TS pour merge avec les autres données.
    Toutes les valeurs TS sont sur une échelle de 0-1.
    """
    print("Ts.py: Loading data from StockMarchand...")
    stock_marchand_df = StockMarchand.get_stock_marchand_data()
    
    df_detail = pd.DataFrame() # Initialize df_detail

    if stock_marchand_df is not None and not stock_marchand_df.empty:
        required_cols_sm = ['CODE_METI', 'Ean_13', 'Classe de stockage']
        missing_cols_sm = [col for col in required_cols_sm if col not in stock_marchand_df.columns]
        
        if not missing_cols_sm: # All required columns are in StockMarchand data
            print("Ts.py: Using data from StockMarchand.")
            df_detail = stock_marchand_df[required_cols_sm].copy()
            # Ensure 'Ean_13' is processed correctly to 'EAN'
            df_detail['EAN'] = pd.to_numeric(df_detail['Ean_13'], errors='coerce').fillna(0).astype(int).astype(str)
            df_detail['CODE_METI'] = df_detail['CODE_METI'].astype(str)
            df_detail['Classe_Stockage'] = df_detail['Classe de stockage'] # Use the original name for mapping
            print(f"Ts.py: Successfully loaded {len(df_detail)} records from StockMarchand")
            if 'EAN' in df_detail.columns:
                 print("DEBUG: Sample EAN values from StockMarchand after conversion:")
                 print(df_detail['EAN'].head(10).tolist())
        else:
            print(f"Ts.py WARNING: Missing columns in StockMarchand data: {missing_cols_sm}. Falling back to DETAIL_CSV.")
            df_detail = pd.DataFrame() # Reset df_detail before CSV fallback attempt
    else:
        print("Ts.py: StockMarchand data not available or empty, falling back to DETAIL_CSV...")
        df_detail = pd.DataFrame() # Ensure df_detail is reset

    # Fallback to DETAIL_CSV if df_detail is still empty (either SM failed, was empty, or had missing cols)
    if df_detail.empty:
        print(f"Ts.py: Attempting to load data from DETAIL_CSV: {DETAIL_CSV}")
        try:
            df_detail_csv = pd.read_csv(DETAIL_CSV, sep=';', encoding='latin1', engine='python', low_memory=False)
            # Rename CDBASE to CODE_METI
            renamed_cdbase = False
            for col_cdbase in ('CDBASE', 'CDBase'):
                if col_cdbase in df_detail_csv.columns:
                    df_detail_csv.rename(columns={col_cdbase: 'CODE_METI'}, inplace=True)
                    renamed_cdbase = True
                    break
            if not renamed_cdbase and 'CODE_METI' not in df_detail_csv.columns:
                 print(f"Ts.py WARNING: Neither CDBASE nor CODE_METI found in {DETAIL_CSV}.")
            
            # Create EAN from Ean_13
            if 'Ean_13' in df_detail_csv.columns:
                df_detail_csv['EAN'] = df_detail_csv['Ean_13'].astype(str)
            elif 'EAN' in df_detail_csv.columns: # If EAN column already exists
                 df_detail_csv['EAN'] = df_detail_csv['EAN'].astype(str)
            else:
                print(f"Ts.py WARNING: 'Ean_13' or 'EAN' column not found in {DETAIL_CSV}.")

            # Handle Classe_Stockage
            if 'Classe de stockage' in df_detail_csv.columns:
                 df_detail_csv['Classe_Stockage'] = df_detail_csv['Classe de stockage']
            elif 'CLASSE_STOCKAGE' in df_detail_csv.columns:
                 df_detail_csv['Classe_Stockage'] = df_detail_csv['CLASSE_STOCKAGE']
            else:
                print(f"Ts.py WARNING: 'Classe de stockage' or 'CLASSE_STOCKAGE' not found in {DETAIL_CSV}.")
            
            # Select only necessary columns if they exist
            final_cols_from_csv = []
            if 'CODE_METI' in df_detail_csv.columns: final_cols_from_csv.append('CODE_METI')
            if 'EAN' in df_detail_csv.columns: final_cols_from_csv.append('EAN')
            if 'Classe_Stockage' in df_detail_csv.columns: final_cols_from_csv.append('Classe_Stockage')
            
            if 'EAN' in final_cols_from_csv and 'CODE_METI' in final_cols_from_csv : # Check for essential columns
                 df_detail = df_detail_csv[final_cols_from_csv].copy()
                 print(f"Ts.py: Successfully loaded {len(df_detail)} records from {DETAIL_CSV}")
            else:
                 print(f"Ts.py ERROR: Essential columns (EAN, CODE_METI) not available in {DETAIL_CSV} for df_detail initialization.")
                 df_detail = pd.DataFrame()


        except FileNotFoundError:
            print(f"Ts.py ERROR: DETAIL_CSV '{DETAIL_CSV}' not found.")
            df_detail = pd.DataFrame()
        except Exception as e:
            print(f"Ts.py ERROR: Reading DETAIL_CSV '{DETAIL_CSV}' failed: {e}")
            df_detail = pd.DataFrame()

    # --- Central checks for essential columns in df_detail ---
    if df_detail.empty:
        print("Ts.py CRITICAL ERROR: df_detail is empty after all loading attempts. Cannot proceed.")
        return pd.DataFrame(columns=['CODE_METI', 'TS'])

    if 'CODE_METI' not in df_detail.columns:
        print("Ts.py CRITICAL ERROR: 'CODE_METI' column is missing in df_detail. Cannot proceed.")
        return pd.DataFrame(columns=['CODE_METI', 'TS'])
    df_detail['CODE_METI'] = df_detail['CODE_METI'].astype(str)

    if 'EAN' not in df_detail.columns:
        print("Ts.py CRITICAL ERROR: 'EAN' column is missing in df_detail. Cannot proceed with TS calculation.")
        return pd.DataFrame(columns=['CODE_METI', 'TS'])
    df_detail['EAN'] = df_detail['EAN'].fillna('').astype(str)
    
    if 'Classe_Stockage' not in df_detail.columns:
        print("Ts.py WARNING: 'Classe_Stockage' column missing in df_detail. TS correction by zone will use default 0.")
        df_detail['Classe_Stockage'] = pd.Series(dtype='float64') # Add empty column to prevent KeyErrors, will fillna(0) later for correction

    # =========================
    # 3. Charger MacroParam
    # =========================
    df_macro = pd.read_csv(MACROPARAM_CSV, sep=';', encoding='latin1', engine='python')
    df_macro = df_macro[['Zone de stockage', 'Correction TS']].rename(
        columns={'Zone de stockage': 'Classe_Stockage'}
    )
    df_macro['Correction TS'] = df_macro['Correction TS'].astype(str).str.rstrip('%').replace('', '0').astype(float).div(100)
    df_macro['Classe_Stockage'] = pd.to_numeric(df_macro['Classe_Stockage'], errors='coerce')


    # =========================
    # 4. Charger TS data (TS_CSV)
    # =========================
    df_ts = pd.read_csv(TS_CSV, sep=';', encoding='latin1', engine='python', dtype={'EAN': str})
    df_ts['EAN'] = df_ts['EAN'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df_ts = df_ts[['EAN', 'TS_Final_V2']].copy()
    
    print("DEBUG: Sample EAN values from TS CSV:")
    print(df_ts['EAN'].head(10).tolist())
    print(f"DEBUG: Total TS records: {len(df_ts)}")
    
    ts_values = pd.to_numeric(df_ts['TS_Final_V2'].astype(str).str.rstrip('%').replace('', '0'), errors='coerce')
    if ts_values.max() <= 1 and ts_values.max() > 0: # check if max is > 0 to avoid issues with all-zero columns
        df_ts['TS_Base'] = ts_values
    else:
        df_ts['TS_Base'] = ts_values.div(100)
    df_ts['TS_Base'].fillna(1.0, inplace=True) # Default to 100% if TS_Final_V2 was empty or non-numeric

    # =========================
    # 5. Fusionner tout
    # =========================
    df_ts['EAN'] = df_ts['EAN'].astype(str)
    # df_detail['EAN'] is already string and NaNs filled from checks above.
    # The line df_detail['EAN'] = df_detail['EAN'].astype(str) was here and caused the error. It's now removed.
    
    stock_eans = set(df_detail['EAN'].tolist())
    ts_eans_from_file = set(df_ts['EAN'].tolist()) # Renamed to avoid clash
    matches = stock_eans.intersection(ts_eans_from_file)
    print(f"DEBUG: df_detail EAN count: {len(stock_eans)}")
    print(f"DEBUG: TS CSV EAN count: {len(ts_eans_from_file)}")
    print(f"DEBUG: Matching EANs: {len(matches)}")
    
    if len(matches) == 0 and len(stock_eans) > 0 and len(ts_eans_from_file) > 0 : # only warn if both have EANs
        print("WARNING: No EAN matches found between df_detail and TS_CSV! Checking EAN format differences...")
        print("Sample df_detail EANs:", list(stock_eans)[:5])
        print("Sample TS CSV EANs:", list(ts_eans_from_file)[:5])
    
    df = df_detail.merge(df_ts[['EAN', 'TS_Base']], on='EAN', how='left')
    
    ts_from_file_count = df['TS_Base'].notna() & (df['TS_Base'] != 1) # Count actual values from file
    print(f"DEBUG: Records with TS_Base from TS_CSV (potentially before default fill): {df['TS_Base'].notna().sum()}")
    df['TS_Base'] = df['TS_Base'].fillna(1.0)  # Default = 100% (échelle 0-1 donc 1 = 100%)
    print(f"DEBUG: Records with TS from file (non-default): {ts_from_file_count.sum()}")
    print(f"DEBUG: Records with default TS (1.0 after fillna): {(df['TS_Base'] == 1.0).sum()}")

    df['Classe_Stockage'] = pd.to_numeric(df['Classe_Stockage'], errors='coerce')
    df = df.merge(df_macro, on='Classe_Stockage', how='left')
    df['Correction TS'] = df['Correction TS'].fillna(0.0)

    # =========================
    # 6. Calcul final de TS (borné [0,1])
    # =========================
    df['TS'] = (df['TS_Base'] + df['Correction TS']).clip(lower=0, upper=1)

    # =========================
    # 7. Correction des valeurs incohérentes
    # =========================
    df['TS'] = df['TS'].apply(lambda x: 1.0 if (x > 0 and x <= 0.02) else x)
    df['TS'] = df['TS'].apply(lambda x: x / 100 if x > 1 and x <= 100 else x) # If some TS were 0-100 scale
    df['TS'] = df['TS'].clip(lower=0, upper=1)
    
    result_df = df[['CODE_METI', 'TS']].drop_duplicates(subset=['CODE_METI'], keep='first')
    result_df['CODE_METI'] = result_df['CODE_METI'].astype(str)
    
    print(f"Ts.py: Final TS values - Min: {result_df['TS'].min()}, Max: {result_df['TS'].max()}")
    
    return result_df

if __name__ == "__main__":
    print("--- Test du module Ts.py ---")
    # Test the function when module is run directly
    # merged_input_for_test = pd.read_csv("initial_merged_predictions.csv", sep=';', encoding='latin1', engine='python', low_memory=False)
    # For a standalone test, get_processed_data now doesn't need any input df.
    
    # To ensure StockMarchand can be called, create a dummy file if it would normally be used by it.
    # This depends on StockMarchand.py's logic for finding its source file.
    # For now, let's assume StockMarchand.py can run or we fall back to DETAIL_CSV for the test.
    
    # Create dummy CSV files for testing if they don't exist
    dummy_files_created = False
    if not os.path.exists(DETAIL_CSV):
        print(f"Création du fichier dummy {DETAIL_CSV} pour le test...")
        pd.DataFrame({
            'CDBASE': ['1001', '1002', '1003'], 
            'Ean_13': ['1111111111111', '2222222222222', '3333333333333'],
            'CLASSE_STOCKAGE': [1060, 1070, 1060]
        }).to_csv(DETAIL_CSV, sep=';', index=False)
        dummy_files_created = True

    if not os.path.exists(TS_CSV):
        print(f"Création du fichier dummy {TS_CSV} pour le test...")
        pd.DataFrame({
            'EAN': ['1111111111111', '2222222222222', '4444444444444'], 
            'TS_Final_V2': ['98%', '0.95', '90%']
        }).to_csv(TS_CSV, sep=';', index=False)
        dummy_files_created = True

    if not os.path.exists(MACROPARAM_CSV):
        print(f"Création du fichier dummy {MACROPARAM_CSV} pour le test...")
        pd.DataFrame({
            'Zone de stockage': [1060, 1070, 1080], 
            'Correction TS': ['1%', '-2%', '0%']
        }).to_csv(MACROPARAM_CSV, sep=';', index=False)
        dummy_files_created = True

    df_output = get_processed_data()
    
    output_file = 'processed_ts_data_test.csv'
    df_output.to_csv(output_file, sep=';', index=False, encoding='latin1')
    print(f"Retrieved {len(df_output)} rows with TS values. Saved to {output_file}")
    print("\nSample data from test run:")
    print(df_output.head(10))

    # Clean up dummy files if they were created by this test
    if dummy_files_created:
        print("\nNettoyage des fichiers dummy créés...")
        for f_path in [DETAIL_CSV, TS_CSV, MACROPARAM_CSV]:
            # Only remove if it's a dummy file we know (e.g., by specific name or path)
            # For safety, this example won't auto-delete if paths are complex.
            # Add specific checks if needed, e.g. if "dummy" is in filename.
            # For now, we assume if they didn't exist before, they are safe to remove.
            # Be cautious with globbed file names.
            # This part is simplified for the example.
            pass # os.remove(f_path) 
        print("Nettoyage (simulé) terminé.")