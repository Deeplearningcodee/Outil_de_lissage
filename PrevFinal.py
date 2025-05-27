import pandas as pd
import numpy as np
import os

# Import du module MacroParam pour récupérer les paramètres
from MacroParam import get_param_value as get_param_value_from_macroparam

# La fonction get_macro_param_value et get_exclusion_data restent inchangées
# ... (copiez-les ici depuis votre version précédente de PrevFinal.py) ...
def get_macro_param_value(param_name='coefficient_exception_prevision', default_value_local=0.0):
    # Appeler la fonction de MacroParam.py en passant la valeur par défaut locale
    return get_param_value_from_macroparam(param_name, default_value_local)

def get_exclusion_data():
    file_path = os.path.join(os.path.dirname(__file__), 'Exclusion.xlsx')
    if not os.path.exists(file_path):
        print(f"  PrevFinal: Fichier {os.path.basename(file_path)} non trouvé")
        return pd.DataFrame(columns=['CDBase', 'Cause'])
    try:
        df_exclusion = pd.read_excel(file_path, engine='openpyxl')
        if 'CDBase' in df_exclusion.columns:
            df_exclusion['CDBase'] = df_exclusion['CDBase'].astype(str)
        return df_exclusion
    except Exception as e:
        print(f"  PrevFinal: Erreur chargement Exclusion.xlsx: {e}")
        return pd.DataFrame(columns=['CDBase', 'Cause'])


def calculate_prev_finales(df): # Renommée pour calculer les deux
    """    
    Calcule les colonnes 'Prév C1-L2 Finale' (BF) et 'Prév L1-L2 Finale' (BG).
    Utilise 'Prev C1-L2' et 'Prev L1-L2' si disponibles.
    """
    print("  PrevFinal: Calcul des prévisions finales (BF, BG)...")
    df_exclusion = get_exclusion_data()
    coefficient_exception = get_macro_param_value('coefficient_exception_prevision')

    # --- Calcul de 'Prév C1-L2 Finale' (BF) ---
    # Formule: MAX(BD2; MAX(AO2; AT2)) * FacteurExclusion
    # AO2 est maintenant 'Prev C1-L2'
    
    required_cols_bf = [
        'En-cours client C1-L2', # BD2
        # 'Prev C1-L2',             # AO2 - ANCIEN NOM
        'Prev Promo C1-L2',       # AT2
        'Casse Prev C1-L2',       # AY2 (utilisé dans la formule Excel pour AT2, ici AT2 est direct)
                                  # La formule Excel pour BF est MAX(BD, MAX(AO, AT))
                                  # AO = Prev C1-L2 (maintenant Prev C1-L2)
                                  # AT = Prev Promo C1-L2
                                  # Le Casse Prev C1-L2 (AY) n'est pas directement dans MAX(AO,AT) mais peut influencer AT
                                  # D'après votre formule: =MAX(BD2;MAX(AO2;AT2)) où AT2 est 'Prev Promo C1-L2'
                                  # La formule originale pour 'Prév C1-L2 Finale' était :
                                  # MAX(BD2; MAX(AO2; AT2))
                                  # Si AO2 est 'Prev C1-L2' et AT2 est 'Prev Promo C1-L2', alors c'est correct.
                                  # Le Casse Prev C1-L2 (AY) est une source de données, pas une prévision aggrégée.
                                  # Si 'Prev C1-L2' (AO) est la prévision de base, et 'Prev Promo C1-L2' (AT) est la prévision promo,
                                  # alors la formule semble être MAX(EnCours, MAX(PrevBase, PrevPromo)).
                                  # Le 'Casse Prev C1-L2' (AY) n'est pas dans ce MAX direct,
                                  # sauf si AT2 ('Prev Promo C1-L2') est en fait MAX('Prev Promo C1-L2', 'Casse Prev C1-L2').
                                  # Votre formule Excel fournie est MAX(BD2;MAX(AO2;AT2))
                                  # AO2: 'Prev C1-L2' (devient 'Prev C1-L2')
                                  # AT2: 'Prev Promo C1-L2'
        'CODE_METI'
    ]
    # AJOUT: Vérifier si 'Prev C1-L2' existe, sinon utiliser 'Prev C1-L2'
    prev_c1_l2_col_name = 'Prev C1-L2'
    if prev_c1_l2_col_name not in df.columns:
        print(f"    PrevFinal: Colonne '{prev_c1_l2_col_name}' non trouvée, tentative avec 'Prev C1-L2'.")
        prev_c1_l2_col_name = 'Prev C1-L2'
        if prev_c1_l2_col_name not in df.columns:
             print(f"    PrevFinal: ERREUR - Ni '{prev_c1_l2_col_name}' ni 'Prev C1-L2' trouvées pour BF.")
             df['Prév C1-L2 Finale'] = 0 # ou np.nan
    if prev_c1_l2_col_name not in required_cols_bf : required_cols_bf.append(prev_c1_l2_col_name)


    for col in required_cols_bf:
        if col not in df.columns:
            print(f"    PrevFinal: AVERTISSEMENT - Colonne '{col}' manquante pour BF. Remplie avec 0.")
            df[col] = 0
    
    def calculate_bf_row(row):
        encours_bd = pd.to_numeric(row.get('En-cours client C1-L2',0), errors='coerce') or 0
        prev_ao = pd.to_numeric(row.get(prev_c1_l2_col_name,0), errors='coerce') or 0
        prev_promo_at = pd.to_numeric(row.get('Prev Promo C1-L2',0), errors='coerce') or 0
        
        base_value = max(encours_bd, max(prev_ao, prev_promo_at))
        
        code_meti = str(row['CODE_METI'])
        facteur_exclusion = 1.0 # Default

        # DEBUG pour METI_EXCLU_COEFF
        if code_meti == 'METI_EXCLU_COEFF':
            print(f"\nDEBUG BF pour {code_meti}:")
            print(f"  df_exclusion est vide: {df_exclusion.empty}")
            if not df_exclusion.empty:
                print(f"  'CDBase' in df_exclusion: {'CDBase' in df_exclusion.columns}")
                print(f"  'Cause' in df_exclusion: {'Cause' in df_exclusion.columns}")
                

        if not df_exclusion.empty and 'CDBase' in df_exclusion.columns and 'Cause' in df_exclusion.columns:
            matching_exclusion_rows = df_exclusion[df_exclusion['CDBase'] == code_meti]
            if not matching_exclusion_rows.empty:
                cause_series = matching_exclusion_rows['Cause']
                if not cause_series.empty:
                    cause_value_raw = cause_series.iloc[0] 
                    
                    # --- AJOUTER LE NETTOYAGE ICI ---
                    if isinstance(cause_value_raw, str):
                        cause_value_cleaned = cause_value_raw.strip() # Enlève les espaces au début/fin
                    else:
                        cause_value_cleaned = str(cause_value_raw).strip() # Convertir en str puis strip au cas où
                    # --- FIN DE L'AJOUT DU NETTOYAGE ---
                        
                    expected_cause_string = "Application d'un coefficient sur la prévision des flop casse" # Pas de .strip() ici car c'est notre référence propre
                    
                    if code_meti == 'METI_EXCLU_COEFF': # Garder ce bloc de debug
                        print(f"\nDEBUG BF pour {code_meti} (DANS LECTURE CAUSE):")
                        print(f"  Cause lue (brute): '{cause_value_raw}' (type: {type(cause_value_raw)})")
                        print(f"  Cause nettoyée: '{cause_value_cleaned}' (type: {type(cause_value_cleaned)})")
                        print(f"  Cause attendue: '{expected_cause_string}'")
                        print(f"  Correspondance (nettoyée vs attendue): {cause_value_cleaned == expected_cause_string}")
                        print(f"  coefficient_exception disponible: {coefficient_exception}")

                    if cause_value_cleaned == expected_cause_string: # Comparer avec la valeur nettoyée
                        facteur_exclusion = coefficient_exception
                        if code_meti == 'METI_EXCLU_COEFF':
                            print(f"  APPLIQUÉ facteur_exclusion = {facteur_exclusion} pour {code_meti} car les causes nettoyées correspondent.")
        
        if code_meti == 'METI_EXCLU_COEFF': # Garder ce bloc de debug final
             print(f"  Valeur finale facteur_exclusion pour {code_meti}: {facteur_exclusion}")
             print(f"  Calcul final BF: {base_value} * {facteur_exclusion} = {base_value * facteur_exclusion}")

        return base_value * facteur_exclusion
    
    df['Prév C1-L2 Finale'] = df.apply(calculate_bf_row, axis=1)

    # --- Calcul de 'Prév L1-L2 Finale' (BG) ---
    # Formule: MAX(BE2; MAX(AP2; AU2)) * FacteurExclusion
    # AP2 est maintenant 'Prev L1-L2'
    
    required_cols_bg = [
        'En-cours client L1-L2', # BE2
        # 'Prev L1-L2',             # AP2 - ANCIEN NOM
        'Prev Promo L1-L2',       # AU2
        'Casse Prev L1-L2',       # AZ2 (Similaire à AY2 pour BF, non directement dans le MAX(AP,AU) de la formule)
        'CODE_METI'
    ]
    # AJOUT: Vérifier si 'Prev L1-L2' existe, sinon utiliser 'Prev L1-L2'
    prev_l1_l2_col_name = 'Prev L1-L2'
    if prev_l1_l2_col_name not in df.columns:
        print(f"    PrevFinal: Colonne '{prev_l1_l2_col_name}' non trouvée, tentative avec 'Prev L1-L2'.")
        prev_l1_l2_col_name = 'Prev L1-L2'
        if prev_l1_l2_col_name not in df.columns:
            print(f"    PrevFinal: ERREUR - Ni '{prev_l1_l2_col_name}' ni 'Prev L1-L2' trouvées pour BG.")
            df['Prév L1-L2 Finale'] = 0 # ou np.nan
    if prev_l1_l2_col_name not in required_cols_bg : required_cols_bg.append(prev_l1_l2_col_name)

    for col in required_cols_bg:
        if col not in df.columns:
            print(f"    PrevFinal: AVERTISSEMENT - Colonne '{col}' manquante pour BG. Remplie avec 0.")
            df[col] = 0

    def calculate_bg_row(row):
        encours_be = pd.to_numeric(row.get('En-cours client L1-L2',0), errors='coerce') or 0
        prev_ap = pd.to_numeric(row.get(prev_l1_l2_col_name,0), errors='coerce') or 0 # Utilise le nom de colonne déterminé
        prev_promo_au = pd.to_numeric(row.get('Prev Promo L1-L2',0), errors='coerce') or 0
        
        base_value = max(encours_be, max(prev_ap, prev_promo_au))
        
        code_meti = str(row['CODE_METI'])
        facteur_exclusion = 1.0
        if not df_exclusion.empty and 'CDBase' in df_exclusion.columns and 'Cause' in df_exclusion.columns:
            if code_meti in df_exclusion['CDBase'].values:
                cause = df_exclusion.loc[df_exclusion['CDBase'] == code_meti, 'Cause'].values
                if len(cause) > 0 and cause[0] == "Application d'un coefficient sur la prévision des flop casse":
                    facteur_exclusion = coefficient_exception
        return base_value * facteur_exclusion

    df['Prév L1-L2 Finale'] = df.apply(calculate_bg_row, axis=1)
    
    print("  PrevFinal: Calculs BF et BG terminés.")
    return df


def get_processed_data(merged_df): # Signature modifiée pour accepter merged_df
    """
    Fonction principale pour être appelée depuis main.py.
    Calcule les colonnes 'Prév C1-L2 Finale' et 'Prév L1-L2 Finale' sur merged_df.
    
    Args:
        merged_df: DataFrame déjà fusionné contenant toutes les colonnes sources.
    
    Returns:
        DataFrame merged_df modifié avec les nouvelles colonnes.
    """
    if merged_df is None or merged_df.empty:
        print("  PrevFinal: ERREUR - DataFrame d'entrée (merged_df) est vide ou None.")
        # Retourner un DataFrame avec les colonnes attendues vides pour éviter des erreurs en aval
        return pd.DataFrame(columns=['CODE_METI', 'Prév C1-L2 Finale', 'Prév L1-L2 Finale'])
    
    # Appeler la fonction de calcul qui modifie merged_df en place
    merged_df_processed = calculate_prev_finales(merged_df)
    
    return merged_df_processed


if __name__ == "__main__":
    print("Test autonome de PrevFinal.py")

    # Créer un fichier Exclusion.xlsx de test s'il n'existe pas
    current_dir = os.path.dirname(__file__)
    exclusion_test_file = os.path.join(current_dir, 'Exclusion.xlsx')
    if not os.path.exists(exclusion_test_file):
        pd.DataFrame({
            'CDBase': ['METI_EXCLU_COEFF', 'METI_AUTRE_EXCLU'], 
            'Cause': ["Application d'un coefficient sur la prévision des flop casse", "Autre Cause"]
        }).to_excel(exclusion_test_file, index=False)
        print(f"Fichier de test '{os.path.basename(exclusion_test_file)}' créé.")

    # Simuler merged_df avec les colonnes nécessaires (y compris les 'Avg')
    test_data = {
        'CODE_METI': ['METI001', 'METI_EXCLU_COEFF', 'METI003'],
        'En-cours client C1-L2': [10, 5, 0],
        'Prev C1-L2': [100, 200, 50],  # NOUVEAU NOM
        'Prev Promo C1-L2': [20, 10, 0],
        'En-cours client L1-L2': [5, 2, 0],
        'Prev L1-L2': [80, 150, 40],   # NOUVEAU NOM
        'Prev Promo L1-L2': [15, 5, 0]
    }
    df_test_input = pd.DataFrame(test_data)
    df_test_input['CODE_METI'] = df_test_input['CODE_METI'].astype(str)


    print("\nDataFrame d'entrée pour PrevFinal:")
    print(df_test_input)
    
    # Simuler que MacroParam.get_param_value('coefficient_exception_prevision') retourne 0.5
    # Pour un test robuste, il faudrait mocker MacroParam ou le configurer.
    # Ici, on s'assure que la valeur par défaut est utilisée ou que le fichier est lisible.
    df_result = get_processed_data(df_test_input.copy()) # Passer une copie
    
    print("\nRésultats du test de PrevFinal:")
    print(df_result[['CODE_METI', 'Prév C1-L2 Finale', 'Prév L1-L2 Finale']].to_string())

    # Vérifications attendues:
    # METI001: BF = MAX(10, MAX(100, 20)) * 1 = 100. BG = MAX(5, MAX(80, 15)) * 1 = 80.
    # METI_EXCLU_COEFF: BF = MAX(5, MAX(200, 10)) * coeff. BG = MAX(2, MAX(150, 5)) * coeff.
    #   Si coeff = 0.1 (par défaut dans test si MacroParam.csv n'est pas configuré avec 0.0):
    #   BF = 200 * 0.1 = 20. BG = 150 * 0.1 = 15.
    # METI003: BF = MAX(0, MAX(50,0)) * 1 = 50. BG = MAX(0, MAX(40,0)) * 1 = 40.

    row_meti001 = df_result[df_result['CODE_METI'] == 'METI001'].iloc[0]
    assert abs(row_meti001['Prév C1-L2 Finale'] - 100) < 0.01, "Test BF METI001 échoué"
    assert abs(row_meti001['Prév L1-L2 Finale'] - 80) < 0.01, "Test BG METI001 échoué"
    print("Tests pour METI001 OK.")

    coeff_test = get_macro_param_value('coefficient_exception_prevision', 0.0) 
    row_meti_exclu = df_result[df_result['CODE_METI'] == 'METI_EXCLU_COEFF'].iloc[0]
    assert abs(row_meti_exclu['Prév C1-L2 Finale'] - (200 * coeff_test)) < 0.01, f"Test BF METI_EXCLU_COEFF échoué. Obtenu: {row_meti_exclu['Prév C1-L2 Finale']}, Attendu: {200*coeff_test}"
    assert abs(row_meti_exclu['Prév L1-L2 Finale'] - (150 * coeff_test)) < 0.01, f"Test BG METI_EXCLU_COEFF échoué. Obtenu: {row_meti_exclu['Prév L1-L2 Finale']}, Attendu: {150*coeff_test}"
    print(f"Tests pour METI_EXCLU_COEFF OK (avec coeff={coeff_test}).")

# --- END OF FILE PrevFinal.py ---