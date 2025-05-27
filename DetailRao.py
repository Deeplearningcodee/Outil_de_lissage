# --- START OF FILE DetailRao.py ---
import os
import glob
import pandas as pd
import numpy as np # Nécessaire pour np.nan

# Dossier contenant les CSV
CSV_FOLDER = 'CSV' 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_folder_path = os.path.join(SCRIPT_DIR, CSV_FOLDER)

pattern = os.path.join(csv_folder_path, '*_Detail_RAO_Commande*.csv')
csv_files = glob.glob(pattern)

if not csv_files:
    pattern_current_dir = os.path.join(SCRIPT_DIR, '*_Detail_RAO_Commande*.csv')
    csv_files = glob.glob(pattern_current_dir)
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern Detail_RAO_Commande dans {csv_folder_path} ou {SCRIPT_DIR}. Assurez-vous que le fichier est présent et que le pattern est correct.")

DETAIL_CSV = csv_files[0] 

def get_processed_data():
    print(f"DetailRao: Chargement de {os.path.basename(DETAIL_CSV)}...")
    try:
        df_detail = pd.read_csv(
            DETAIL_CSV,
            sep=';',
            encoding='latin1',
            # Garder les colonnes de date comme objet initialement pour un contrôle plus fin si parse_dates échoue
            # Ou les parser directement si le format est consistent.
            # parse_dates=['DATE_LIVRAISON', 'DATE_L2', 'DATE_COMMANDE'], 
            # dayfirst=True,
            low_memory=False 
        )
    except Exception as e:
        print(f"ERREUR DetailRao: Impossible de charger {DETAIL_CSV}. Détails: {e}")
        return pd.DataFrame() 
        
    print(f"DetailRao: {len(df_detail)} lignes chargées initialement.")
    print(f"DetailRao DEBUG: Colonnes initiales lues depuis CSV: {df_detail.columns.tolist()}")

    # --- SECTION DE RENOMMAGE DES COLONNES ---
    # Dictionnaire: clé = nom original sensible à la casse (ou pattern), valeur = nouveau nom standardisé
    # Ce map doit être exhaustif pour toutes les colonnes que vous voulez renommer ou standardiser.
    rename_map = {
        # Cas spécifiques comme CDBASE
        'CDBASE': 'CODE_METI', 'CDBase': 'CODE_METI',
        'CLASSE_STOCKAGE': 'Classe_Stockage', 'Classe_stockage': 'Classe_Stockage',
        # Suppression des accents et standardisation
        'Libellé_Famille': 'Libelle_Famille',
        'Libellé_Sous_Famille': 'Libelle_Sous_Famille',
        'Macro-catégorie': 'Macro_categorie', # Ce nom sera utilisé pour la colonne créée
        'Entrepôt': 'Entrepot',
        'Période de vente finale (Jours)': 'Periode_vente_finale_Jours',
        'VMJ Utilisée Stock Sécurité': 'VMJ_Utilisee_Stock_Securite',
        'QTÉPRÉVISIONINIT1': 'QTEPREVISIONINIT1',
        'QTÉPRÉVISIONINIT2': 'QTEPREVISIONINIT2',
        'QTÉPROPOSÉE1': 'QTEPROPOSEE1',
        'QTÉPROPOSÉE2': 'QTEPROPOSEE2',
        'QTÉPROPOSÉEFINALE': 'QTEPROPOSEEFINALE',
        # Ajoutez d'autres colonnes ici si elles ont des accents ou des variations de casse
        # que vous voulez normaliser.
        # Par exemple:
        # 'Mini Publication FL': 'Mini_Publication_FL', (si vous voulez standardiser les espaces en underscores)
        # 'Nb de commande / Semaine Final': 'Nb_commande_Semaine_Final',
    }
    
    # Appliquer les renommages basés sur les clés exactes trouvées dans le rename_map
    # Pour gérer les variations de casse dans le fichier CSV source qui ne sont pas dans le map:
    actual_renames_to_apply = {}
    df_cols_upper = {col.upper(): col for col in df_detail.columns} # Map de noms de colonnes CSV en majuscules vers leur nom original

    for map_key_upper, new_name in {k.upper(): v for k, v in rename_map.items()}.items():
        if map_key_upper in df_cols_upper:
            actual_renames_to_apply[df_cols_upper[map_key_upper]] = new_name
            
    if actual_renames_to_apply:
        df_detail.rename(columns=actual_renames_to_apply, inplace=True)
        print(f"DetailRao DEBUG: Colonnes après renommages: {df_detail.columns.tolist()}")
    else:
        print("DetailRao DEBUG: Aucun renommage appliqué (vérifiez le rename_map et les colonnes CSV).")
    # --- FIN RENOMMAGE ---

    # --- CONVERSION DES DATES ---
    date_columns_to_parse = ['DATE_LIVRAISON', 'DATE_L2', 'DATE_COMMANDE']
    for col_date in date_columns_to_parse:
        if col_date in df_detail.columns:
            df_detail[col_date] = pd.to_datetime(df_detail[col_date], dayfirst=True, errors='coerce')
        else:
            print(f"DetailRao AVERTISSEMENT: Colonne de date '{col_date}' non trouvée pour parsing.")
    # --- FIN CONVERSION DES DATES ---


    # --- CONVERSION DES COLONNES NUMÉRIQUES AVEC VIRGULE DÉCIMALE ---
    # Utilisez les noms de colonnes APRÈS renommage
    cols_avec_virgule_decimal = [
        'QTEPROPOSEE1', 'QTEPROPOSEE2', 'QTEPREVISIONINIT1', 'QTEPREVISIONINIT2',
        'RAL', 'PCB', 'MINIMUM_COMMANDE', 'STOCK_REEL', 'SM Final',
        'Mini Publication FL', # Si elle peut avoir des virgules
        # 'Stock Max', 'VMJ_Utilisee_Stock_Securite', 'Periode_vente_finale_Jours',
        # Ajoutez toutes les colonnes numériques qui pourraient être lues comme des chaînes avec ','
    ] 
    for col_name in cols_avec_virgule_decimal:
        if col_name in df_detail.columns:
            if df_detail[col_name].dtype == 'object':
                # print(f"DetailRao DEBUG: Nettoyage décimal de la colonne '{col_name}'...")
                df_detail[col_name] = df_detail[col_name].astype(str).str.replace(',', '.', regex=False)
            # Convertir en numérique après le remplacement ou si déjà pas objet
            df_detail[col_name] = pd.to_numeric(df_detail[col_name], errors='coerce')
        # else:
            # print(f"DetailRao AVERTISSEMENT: Colonne '{col_name}' pour nettoyage décimal non trouvée.")
    print(f"DetailRao DEBUG: Types après conversion décimale pour QTEPROPOSEE1: {df_detail['QTEPROPOSEE1'].dtype if 'QTEPROPOSEE1' in df_detail.columns else 'N/A'}")
    # --- FIN CONVERSION NUMÉRIQUE ---


    if 'CODE_METI' not in df_detail.columns:
        print("AVERTISSEMENT (DetailRao): Colonne 'CODE_METI' critique non trouvée. Arrêt possible ou erreurs en aval.")
        # return pd.DataFrame() # Option: retourner vide si CODE_METI manque
    
    # Map Classe_Stockage to Macro_categorie (nom sans accent)
    if 'Classe_Stockage' in df_detail.columns:
        stockage_to_macro = {
            1060: 'Sec Méca', 1061: 'Sec Homogène', 1062: 'Sec Hétérogène',
            1063: 'Sec Hétérogène', 1064: 'Sec Hétérogène', 1070: 'Frais Méca',
            1071: 'Frais Manuel', 1080: 'Surgelés'
        }
        df_detail['Classe_Stockage'] = pd.to_numeric(df_detail['Classe_Stockage'], errors='coerce')
        df_detail['Macro_categorie'] = df_detail['Classe_Stockage'].map(stockage_to_macro).fillna('Autre')
    else:
        print("AVERTISSEMENT (DetailRao): Colonne 'Classe_Stockage' non trouvée pour mapper Macro_categorie.")
        df_detail['Macro_categorie'] = 'Autre'


    try:
        from MacroParam import DATE_COMMANDE as REF_DATE_COMMANDE
    except ImportError:
        print("AVERTISSEMENT (DetailRao): MacroParam.py non trouvé pour DATE_COMMANDE. Utilisation d'une date par défaut.")
        REF_DATE_COMMANDE = pd.to_datetime('2025-05-14', dayfirst=True) 
    
    date_commande_plus_1 = REF_DATE_COMMANDE + pd.Timedelta(days=1)
    date_commande_plus_2 = REF_DATE_COMMANDE + pd.Timedelta(days=2)

    # Position JUD - principalement pour la logique ESTERREUR($P2) dans les formules Excel.
    # Si P2 ne réfère pas à une colonne spécifique mais à la validité de la ligne,
    # on peut la créer ou ignorer cette partie ESTERREUR. Ici, on la crée.
    if 'Position JUD' not in df_detail.columns:
        df_detail['Position JUD'] = range(1, len(df_detail) + 1) 
    
    # Calcul de DATE_LIVRAISON_V2
    if 'DATE_LIVRAISON' not in df_detail.columns: 
        print("AVERTISSEMENT (DetailRao): Colonne 'DATE_LIVRAISON' non trouvée pour calculer DATE_LIVRAISON_V2.")
        df_detail['DATE_LIVRAISON'] = pd.NaT # Créer pour éviter erreur, sera remplie par défaut
        
    df_detail['DATE_LIVRAISON_V2'] = df_detail['DATE_LIVRAISON'].copy() 
    # Si DATE_LIVRAISON originale est NaT (vide ou erreur de parsing), mettre date_commande_plus_1
    df_detail.loc[df_detail['DATE_LIVRAISON_V2'].isna(), 'DATE_LIVRAISON_V2'] = date_commande_plus_1
    # S'assurer que c'est bien datetime à la fin
    df_detail['DATE_LIVRAISON_V2'] = pd.to_datetime(df_detail['DATE_LIVRAISON_V2'], errors='coerce')

    # Calcul de Date_L2_V2
    date_l2_col_name_final = 'DATE_L2' # Nom après renommage éventuel
    df_detail['Date_L2_V2'] = pd.NaT 
    if date_l2_col_name_final in df_detail.columns:
        # S'assurer que DATE_L2 est datetime (peut avoir été lue comme objet)
        df_detail[date_l2_col_name_final] = pd.to_datetime(df_detail[date_l2_col_name_final], errors='coerce')
        
        for idx, row_date_l2 in df_detail[date_l2_col_name_final].items():
            if pd.isna(row_date_l2):
                df_detail.loc[idx, 'Date_L2_V2'] = date_commande_plus_2
            else:
                df_detail.loc[idx, 'Date_L2_V2'] = max(row_date_l2, date_commande_plus_2)
    else:
        print(f"AVERTISSEMENT (DetailRao): Colonne '{date_l2_col_name_final}' non trouvée. Date_L2_V2 sera {date_commande_plus_2.strftime('%Y-%m-%d')}.")
        df_detail['Date_L2_V2'] = date_commande_plus_2
    df_detail['Date_L2_V2'] = pd.to_datetime(df_detail['Date_L2_V2'], errors='coerce')


    # --- Sélection et ordonnancement final des colonnes ---
    # Liste exhaustive des colonnes que vous voulez garder, avec leurs noms standardisés
    final_columns_order = [
        'FAMILLE_HYPER', 'Libelle_Famille', 'SS_FAMILLE_HYPER', 'Libelle_Sous_Famille',
        'Ean_13', 'LIBELLE_REFERENCE', 'CODE_METI', 
        'Nb de commande / Semaine Final', # Assurez-vous qu'elle existe ou supprimez-la de la liste
        'Macro_categorie', 
        'Classe_Stockage', 
        'Entrepot', # Si renommé depuis 'Entrepôt'
        'Periode_vente_finale_Jours', # Si renommé
        'VMJ_Utilisee_Stock_Securite', # Si renommé
        'Stock Max', 'SM Final',
        'FOURNISSEUR', 
        'DATE_LIVRAISON',    
        'DATE_LIVRAISON_V2', 
        'DATE_L2',           
        'Date_L2_V2',        
        'MINIMUM_COMMANDE', 'PCB', 'RAL', 
        'QTEPREVISIONINIT1', 'QTEPREVISIONINIT2', 
        'COCHE_RAO', 'Position JUD', 
        'QTEPROPOSEE1', 'QTEPROPOSEE2', 'QTEPROPOSEEFINALE', 
        'NUMERO_COMMANDE', 'IFLS', 
        'STOCK', # Si vous en avez besoin
        'STOCK_REEL',
        'Mini Publication FL',
        # ... ajoutez toutes les autres colonnes nécessaires pour les modules suivants ...
        # Ex: 'Prev C1-L2', 'Prev L1-L2', 'Prev Promo C1-L2', etc. si elles viennent de ce fichier.
        # Si elles sont calculées par d'autres modules, elles seront mergées plus tard dans main.py.
    ]
    
    # print(f"DetailRao DEBUG: Colonnes dans df_detail AVANT sélection finale par final_columns_order: {df_detail.columns.tolist()}")
    
    existing_final_columns = [col for col in final_columns_order if col in df_detail.columns]
    
    missing_from_final_list = [col for col in final_columns_order if col not in df_detail.columns]
    if missing_from_final_list:
        print(f"DetailRao AVERTISSEMENT: Colonnes listées dans final_columns_order MAIS ABSENTES de df_detail et ne seront pas retournées: {missing_from_final_list}")

    # print(f"DetailRao DEBUG: Colonnes finales qui seront effectivement conservées et retournées: {existing_final_columns}")
        
    if not existing_final_columns:
        print("DetailRao ERREUR: Aucune colonne à retourner. Vérifiez final_columns_order et les étapes précédentes.")
        return pd.DataFrame() 
        
    result_df = df_detail[existing_final_columns].copy()
    
    # Filtrage optionnel par date (souvent mieux de le faire dans main.py sur merged_df si nécessaire)
    # date_cols_to_filter = ['DATE_LIVRAISON_V2', 'Date_L2_V2']
    # # ... (logique de filtrage si besoin) ...
    
    print(f"DetailRao: Traitement terminé. {len(result_df)} lignes retournées. Colonnes finales: {result_df.columns.tolist()}")
    return result_df

if __name__ == "__main__":
    print("--- Test du module DetailRao.py ---")
    # Pour un test autonome plus robuste, il faudrait un fichier CSV de test.
    # Ce test simple exécute la fonction.
    df_output = get_processed_data()
    
    if df_output is not None and not df_output.empty:
        print(f"\nNombre de lignes après traitement: {len(df_output)}")
        print("\nColonnes dans le DataFrame résultant (test):")
        print(df_output.columns.tolist())
        print("\nExtrait des 5 premières lignes du résultat (test):")
        print(df_output.head().to_string())
        
        cols_to_check_type = ['QTEPROPOSEE1', 'QTEPROPOSEE2', 'RAL', 'DATE_LIVRAISON_V2', 'Date_L2_V2']
        for r_col in cols_to_check_type:
            if r_col in df_output.columns:
                print(f"Colonne '{r_col}': Type={df_output[r_col].dtype}. Aperçu: {df_output[r_col].head().tolist()}")
            else:
                print(f"AVERTISSEMENT TEST: Colonne '{r_col}' non trouvée dans la sortie.")
    else:
        print("Aucune donnée retournée par get_processed_data lors du test, ou une erreur s'est produite.")

# --- END OF FILE DetailRao.py ---