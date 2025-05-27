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
    """Calcule 'Top' (Top 500, Top 3000, Autre) en chargeant les fichiers CSV."""
    print("  TypeProduits - Calcul de 'Top'...")
    
    # Chemins vers les fichiers Top
    script_dir = os.path.dirname(__file__) # ou répertoire courant si CSV est là
    csv_folder = os.path.join(script_dir, 'CSV')
    if not os.path.exists(csv_folder):
        print(f"    ATTENTION (calc_top): Dossier CSV non trouvé à {csv_folder}. Recherche dans le répertoire du script.")
        csv_folder = script_dir # Fallback au répertoire du script

    top500_file = os.path.join(csv_folder, 'Top500.csv')
    top3000_file = os.path.join(csv_folder, 'Top3000.csv')

    df_top500, df_top3000 = pd.DataFrame(), pd.DataFrame() # Initialiser comme vides

    if os.path.exists(top500_file):
        try:
            df_top500 = pd.read_csv(top500_file, sep=';', encoding='latin1', low_memory=False)
            df_top500.columns = df_top500.columns.str.strip()
            print(f"    Fichier Top500.csv chargé ({len(df_top500)} lignes).")
        except Exception as e:
            print(f"    ERREUR (calc_top): Impossible de lire Top500.csv: {e}")
    else:
        print(f"    ATTENTION (calc_top): Fichier Top500.csv non trouvé à {top500_file}.")

    if os.path.exists(top3000_file):
        try:
            df_top3000 = pd.read_csv(top3000_file, sep=';', encoding='latin1', low_memory=False)
            df_top3000.columns = df_top3000.columns.str.strip()
            print(f"    Fichier Top3000.csv chargé ({len(df_top3000)} lignes).")
        except Exception as e:
            print(f"    ERREUR (calc_top): Impossible de lire Top3000.csv: {e}")
    else:
        print(f"    ATTENTION (calc_top): Fichier Top3000.csv non trouvé à {top3000_file}.")

    # Créer des sets pour une recherche rapide, en gérant les colonnes manquantes
    top500_ean_set, top500_code_set = set(), set()
    if not df_top500.empty:
        if 'EAN' in df_top500.columns: top500_ean_set = set(df_top500['EAN'].astype(str).str.strip())
        if 'CODE_METI' in df_top500.columns: top500_code_set = set(df_top500['CODE_METI'].astype(str).str.strip())

    top3000_ean_set, top3000_code_set = set(), set()
    if not df_top3000.empty:
        if 'EAN' in df_top3000.columns: top3000_ean_set = set(df_top3000['EAN'].astype(str).str.strip())
        if 'CODE_METI' in df_top3000.columns: top3000_code_set = set(df_top3000['CODE_METI'].astype(str).str.strip())
    
    # S'assurer que les colonnes Ean_13 et CODE_METI existent dans le df principal
    # Ean_13 est le nom dans Détail.xlsx, EAN peut être créé par Ts.py
    ean_col_to_use = None
    if 'Ean_13' in df.columns: ean_col_to_use = 'Ean_13'
    elif 'EAN' in df.columns: ean_col_to_use = 'EAN'
    
    codemeti_col_to_use = None
    if 'CODE_METI' in df.columns: codemeti_col_to_use = 'CODE_METI'


    def get_top_category(row):
        ean_val, code_meti_val = None, None
        if ean_col_to_use: ean_val = str(row.get(ean_col_to_use, '')).strip()
        if codemeti_col_to_use: code_meti_val = str(row.get(codemeti_col_to_use, '')).strip()

        if (ean_val and ean_val in top500_ean_set) or \
           (code_meti_val and code_meti_val in top500_code_set):
            return "top 500"
        if (ean_val and ean_val in top3000_ean_set) or \
           (code_meti_val and code_meti_val in top3000_code_set):
            return "top 3000"
        return "autre" # Standardisé en minuscules

    if not ean_col_to_use and not codemeti_col_to_use:
        print("    ATTENTION (calc_top): Ni 'Ean_13'/'EAN' ni 'CODE_METI' trouvées dans le DataFrame principal. 'Top' sera 'autre'.")
        df['Top'] = "autre"
    else:
        df['Top'] = df.apply(get_top_category, axis=1)
    
    print("  TypeProduits - 'Top' calculé.")
    return df

def get_processed_data(input_df):
    """
    Fonction principale pour calculer 'Type de produits', 'Type de produits V2', et 'Top'.
    Prend un DataFrame en entrée et retourne le DataFrame modifié.
    """
    print("TypeProduits - Début du traitement...")
    if not isinstance(input_df, pd.DataFrame):
        raise ValueError("L'entrée de TypeProduits.get_processed_data doit être un DataFrame Pandas.")
    
    df_copy = input_df.copy() # Travailler sur une copie pour éviter les SettingWithCopyWarning
    
    df_copy = calc_type_produits(df_copy)
    df_copy = calc_type_produits_v2(df_copy) 
    df_copy = calc_top(df_copy)
    
    print("TypeProduits - Traitement terminé.")
    return df_copy

if __name__ == "__main__":
    print("--- Test du module TypeProduits.py ---")
    
    # Vérifier si MacroParam a été chargé, sinon imprimer un avertissement plus visible
    if not MACROPARAM_LOADED:
        print("\nATTENTION MAJEURE: MacroParam.py n'a pas été chargé. Les valeurs par défaut sont utilisées pour TYPES_PRODUITS, GESTION_AB_AC_MAPPING, et les dates de référence. Les résultats peuvent ne pas correspondre à votre configuration Excel.\n")

    # Créer un DataFrame de test plus réaliste
    test_data = {
        'CLASSE_STOCKAGE': ["1060", "1060", "1061", "1070", "1080", "1062", "1060", "1060", "1063"], 
        'DATE_LIVRAISON_V2': [ # Assurez-vous que ces dates ont du sens par rapport à DATE_REF_JOUR_AB/AC
            DATE_REF_JOUR_AB - pd.Timedelta(days=1), # Devrait devenir "sec - autre" ou "sec méca"
            DATE_REF_JOUR_AB,                       # Devrait devenir "Sec Méca - A/B"
            DATE_REF_JOUR_AC,                       # Devrait devenir "Sec Homogène - A/C"
            DATE_REF_JOUR_AB,                       # "Frais Méca" (non Sec)
            DATE_REF_JOUR_AB,                       # "Surgelés" (non Sec)
            DATE_REF_JOUR_AB + pd.Timedelta(days=5), # "Sec Hétérogène" (devrait devenir "sec - autre")
            pd.NaT,                                 # Sec Méca avec date NaT
            DATE_REF_JOUR_AC,                       # Sec Méca, Gestion Non (si configuré dans GESTION_AB_AC_MAPPING)
            DATE_REF_JOUR_AB                        # Sec Hétérogène (pour tester GESTION_AB_AC_MAPPING)
        ],
        'Ean_13': ['EAN_SM_AUTRE', 'EAN_SM_AB', 'EAN_SH_AC', 'EAN_FM', 'EAN_SU', 'EAN_SHET_AUTRE', 'EAN_NAT', 'EAN_SECMECA_GESTNON', 'EAN_SHETERO_AB'],
        'CODE_METI': ['CM_SM_AUTRE', 'TOP500_CODE', 'TOP3000_CODE', 'CM_FM', 'CM_SU', 'CM_SHET_AUTRE', 'CM_NAT', 'CM_SECMECA_GESTNON', 'CM_SHETERO_AB']
    }
    test_df = pd.DataFrame(test_data)
    
    # Simuler des fichiers Top500 et Top3000 pour le test
    current_script_dir = os.path.dirname(__file__)
    csv_test_dir = os.path.join(current_script_dir, 'CSV')
    if not os.path.exists(csv_test_dir):
        os.makedirs(csv_test_dir)

    pd.DataFrame({'EAN': ['EAN_SM_AB'], 'CODE_METI': ['TOP500_CODE']}).to_csv(os.path.join(csv_test_dir, 'Top500.csv'), sep=';', index=False)
    pd.DataFrame({'EAN': ['EAN_SH_AC'], 'CODE_METI': ['TOP3000_CODE_WRONG_EAN']}).to_csv(os.path.join(csv_test_dir, 'Top3000.csv'), sep=';', index=False)
    print(f"Fichiers Top500.csv et Top3000.csv de test créés dans {csv_test_dir}")
    
    print(f"\nUtilisation DATE_REF_JOUR_AB: {DATE_REF_JOUR_AB.strftime('%Y-%m-%d') if pd.notna(DATE_REF_JOUR_AB) else 'NaT'}, DATE_REF_JOUR_AC: {DATE_REF_JOUR_AC.strftime('%Y-%m-%d') if pd.notna(DATE_REF_JOUR_AC) else 'NaT'}")
    print("GESTION_AB_AC_MAPPING utilisé:")
    for k, v in GESTION_AB_AC_MAPPING.items(): print(f"  {k}: {v}")

    test_df_processed = get_processed_data(test_df.copy()) 
    print("\nRésultats du test de TypeProduits.py:")
    cols_to_display = ['CLASSE_STOCKAGE', 'DATE_LIVRAISON_V2', 'Type de produits', 'Type de produits V2', 'Top']
    # Afficher seulement les colonnes qui existent réellement dans test_df_processed
    print(test_df_processed[[col for col in cols_to_display if col in test_df_processed.columns]].to_string())

    # Vérifications attendues
    print("\nVérifications attendues (exemples):")
    # Sec Méca, date_livraison == DATE_REF_JOUR_AB -> 'sec méca - a/b'
    row_sm_ab = test_df_processed[test_df_processed['Ean_13'] == 'EAN_SM_AB']
    if not row_sm_ab.empty:
        print(f"  EAN_SM_AB (Sec Méca, date=AB): Type V2 = '{row_sm_ab['Type de produits V2'].iloc[0]}', Top = '{row_sm_ab['Top'].iloc[0]}' (Attendu: 'sec méca - a/b', 'top 500')")

    # Sec Homogène, date_livraison == DATE_REF_JOUR_AC -> 'sec homogène - a/c'
    row_sh_ac = test_df_processed[test_df_processed['Ean_13'] == 'EAN_SH_AC']
    if not row_sh_ac.empty:
         print(f"  EAN_SH_AC (Sec Homogène, date=AC): Type V2 = '{row_sh_ac['Type de produits V2'].iloc[0]}', Top = '{row_sh_ac['Top'].iloc[0]}' (Attendu: 'sec homogène - a/c', 'top 3000')")

    # Frais Méca -> 'frais méca'
    row_fm = test_df_processed[test_df_processed['Ean_13'] == 'EAN_FM']
    if not row_fm.empty:
        print(f"  EAN_FM (Frais Méca): Type V2 = '{row_fm['Type de produits V2'].iloc[0]}', Top = '{row_fm['Top'].iloc[0]}' (Attendu: 'frais méca', 'autre')")
    
    # Sec Méca avec NaT date_livraison -> 'sec - autre'
    row_nat = test_df_processed[test_df_processed['Ean_13'] == 'EAN_NAT']
    if not row_nat.empty:
        print(f"  EAN_NAT (Sec Méca, date NaT): Type V2 = '{row_nat['Type de produits V2'].iloc[0]}' (Attendu: 'sec - autre')")