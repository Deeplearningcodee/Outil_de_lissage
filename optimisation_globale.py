# Fichier: optimisation_globale.py
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution, NonlinearConstraint
import os
from datetime import datetime 
import MacroParam

# --- Fichier et Module de Paramètres ---
try:
    # On utilise CommandeFinale.py pour la simulation de F_sim
    # car c'est ce que main.py utilise.
    import CommandeFinale as cf_main_module 
except ImportError as e:
    print(f"ERREUR: Module requis CommandeFinale.py introuvable. Erreur: {e}")
    exit()

try:
    from MacroParam import get_param_value, get_arrondi_pcb_seuils
except ImportError as e:
    print(f"ERREUR: Module MacroParam.py introuvable ou fonctions manquantes. Erreur: {e}")
    exit()

# --- Constantes Globales de Marge ---
MARGE_POUR_BOOST_ET_L_VAR_SOLVER = MacroParam.MARGE_POUR_BOOST_ET_L_VAR_SOLVER
MARGE_I_POUR_SOLVER_CONDITION = MacroParam.MARGE_I_POUR_SOLVER_CONDITION   
ALERTE_SURCHARGE_NIVEAU_1_DEFAULT = MacroParam.ALERTE_SURCHARGE_NIVEAU_1

current_row_state_for_solver = {} 

# --- Fonctions de Chargement et Pré-traitement des Données ---
def load_data(pdc_sim_filepath, detail_filepath, pdc_perm_filepath):
    print(f"DEBUG LoadData: Chargement de {pdc_sim_filepath} (PDC_Sim)...")
    try:
        df_pdc_sim = pd.read_excel(pdc_sim_filepath)
    except Exception:
        try:
            df_pdc_sim = pd.read_csv(pdc_sim_filepath) 
        except Exception as e_csv:
            raise ValueError(f"Erreur: Impossible de lire {pdc_sim_filepath}. Détails: {e_csv}")

    print(f"DEBUG LoadData: Chargement de {detail_filepath} (devrait être Détail.csv)...")
    try:
        df_detail = pd.read_csv(detail_filepath, sep=';', encoding='latin1', low_memory=False)
        print(f"  {detail_filepath} chargé avec succès comme CSV.")
    except Exception as e_csv:
        # Essayer avec Excel si CSV échoue, au cas où ce serait un .xlsx avec une extension .csv
        try:
            print(f"  Tentative de lecture de {detail_filepath} comme Excel...")
            df_detail = pd.read_excel(detail_filepath)
            print(f"  {detail_filepath} chargé avec succès comme Excel.")
        except Exception as e_excel:
             raise ValueError(f"Erreur: Impossible de lire {detail_filepath} comme CSV ou Excel. Détails CSV: {e_csv}, Détails Excel: {e_excel}")

    print(f"DEBUG LoadData: Chargement de {pdc_perm_filepath} (PDC Perm)...")
    try:
        # Lire la feuille "PDC Perm", en supposant que la première ligne est l'en-tête 
        # et la première colonne l'index des dates
        df_pdc_perm = pd.read_excel(pdc_perm_filepath, sheet_name="PDC", index_col=0)
        
        # Convertir l'index (qui devrait être les dates) en datetime
        df_pdc_perm.index = pd.to_datetime(df_pdc_perm.index, errors='coerce')
        df_pdc_perm.dropna(axis=0, how='all', inplace=True) # Enlever les lignes entièrement vides
        df_pdc_perm.dropna(axis=1, how='all', inplace=True) # Enlever les colonnes entièrement vides
        
        # Standardiser les noms de colonnes (Types de produits)
        df_pdc_perm.columns = [str(col).strip() for col in df_pdc_perm.columns]
        
        print(f"  {pdc_perm_filepath} chargé. Index (dates): {df_pdc_perm.index.name}. Colonnes (types produits): {df_pdc_perm.columns.tolist()[:5]}...")
    except Exception as e_perm:
        raise ValueError(f"Erreur: Impossible de lire ou de traiter {pdc_perm_filepath}. Vérifiez le nom de la feuille et la structure. Détails: {e_perm}")


    # --- Traitement de df_pdc_sim (PDC_Sim.xlsx) ---
    required_pdc_sim_cols = [
        'Type de produits V2', 'Type de produits', 'Jour livraison', 
        'PDC', 'En-cours', 'Commande SM à 100%', 'Tolérance',
        'Poids du A/C max', 'Top 500', 'Top 3000', 'Autre',
        'Limite Basse Top 500', 'Limite Basse Top 3000', 'Limite Basse Autre',
        'Limite Haute Top 500', 'Limite Haute Top 3000', 'Limite Haute Autre',
        'Min Facteur', 'Max Facteur', 'Boost PDC'
    ]
    for col in required_pdc_sim_cols:
        if col not in df_pdc_sim.columns:
             raise ValueError(f"Colonne requise '{col}' manquante dans {pdc_sim_filepath}")

    df_pdc_sim['Type de produits V2'] = df_pdc_sim['Type de produits V2'].astype(str).str.strip().str.lower()
    # Pour 'Type de produits', on le garde tel quel pour le lookup dans df_pdc_perm
    df_pdc_sim['Type de produits'] = df_pdc_sim['Type de produits'].astype(str).str.strip() 
    df_pdc_sim['Jour livraison'] = pd.to_datetime(df_pdc_sim['Jour livraison'], errors='coerce', dayfirst=False) # Excel utilise souvent format US par défaut si non spécifié
    
    percentage_like_cols_pdc_sim = ['Top 500', 'Top 3000', 'Autre', 'Boost PDC', 
                                    'Min Facteur', 'Max Facteur', 'Poids du A/C max',
                                    'Limite Basse Top 500', 'Limite Basse Top 3000', 'Limite Basse Autre',
                                    'Limite Haute Top 500', 'Limite Haute Top 3000', 'Limite Haute Autre']
    
    cols_to_convert_to_numeric_pdc_sim = [
        'PDC', 'En-cours', 'Commande SM à 100%', 'Tolérance', 'Poids du A/C max',
        'Top 500', 'Top 3000', 'Autre', 
        'Limite Basse Top 500', 'Limite Basse Top 3000', 'Limite Basse Autre',
        'Limite Haute Top 500', 'Limite Haute Top 3000', 'Limite Haute Autre',
        'Min Facteur', 'Max Facteur', 'Boost PDC'
    ]
    
    for col in cols_to_convert_to_numeric_pdc_sim:
        if col in df_pdc_sim.columns:
            if df_pdc_sim[col].dtype == 'object': 
                temp_col_str_series = df_pdc_sim[col].astype(str)
                df_pdc_sim[col] = temp_col_str_series.replace(r'^\s*$', np.nan, regex=True)
                had_percent_sign = df_pdc_sim[col].str.contains('%', na=False)
                cleaned_col = df_pdc_sim[col].str.rstrip('%').str.replace(',', '.', regex=False)
                df_pdc_sim[col] = pd.to_numeric(cleaned_col, errors='coerce')
                
                if col in percentage_like_cols_pdc_sim:
                    mask_had_percent_and_not_nan = had_percent_sign & (~df_pdc_sim[col].isnull())
                    df_pdc_sim.loc[mask_had_percent_and_not_nan, col] = \
                        df_pdc_sim.loc[mask_had_percent_and_not_nan, col] / 100.0
            
            if pd.api.types.is_numeric_dtype(df_pdc_sim[col]) and col in percentage_like_cols_pdc_sim:
                 if col not in ['PDC', 'En-cours', 'Commande SM à 100%', 'Tolérance']:
                    mask_large_percentage = (~df_pdc_sim[col].isnull()) & (df_pdc_sim[col].abs() > 5.0) # Ex: 100% est 100, pas 1.0
                    if mask_large_percentage.any():
                         df_pdc_sim.loc[mask_large_percentage, col] /= 100.0
    
    if 'Tolérance' in df_pdc_sim.columns and df_pdc_sim['Tolérance'].isnull().any():
        mask_nan_tolerance = df_pdc_sim['Tolérance'].isnull()
        if 'PDC' in df_pdc_sim.columns and pd.api.types.is_numeric_dtype(df_pdc_sim['PDC']):
            df_pdc_sim.loc[mask_nan_tolerance, 'Tolérance'] = \
                df_pdc_sim.loc[mask_nan_tolerance, 'PDC'] * ALERTE_SURCHARGE_NIVEAU_1_DEFAULT
        else: 
            print("AVERTISSEMENT LoadData: Col 'PDC' non trouvée/numérique pour recalculer 'Tolérance' NaN dans df_pdc_sim.")

    # --- Traitement de df_detail (Détail.csv / merged_predictions.csv) ---
    required_detail_cols_from_source = { # Noms attendus dans le fichier CSV/Excel source
        'Type de produits V2': 'BM', # Clé interne Python : Nom de colonne dans fichier Détail
        'DATE_LIVRAISON_V2': 'DATE_LIVRAISON_V2', 
        'Top': 'Top',
        'Borne Min Facteur multiplicatif lissage': 'Borne Min Facteur multiplicatif lissage',
        'Borne Max Facteur multiplicatif lissage': 'Borne Max Facteur multiplicatif lissage',
        'Commande Finale avec mini et arrondi SM à 100%': 'Commande Finale avec mini et arrondi SM à 100%', # BP
        'Commande Finale avec mini et arrondi SM à 100% avec TS': 'BQ', # BQ
        'Mini Publication FL': 'Mini Publication FL', 'COCHE_RAO': 'COCHE_RAO', 'STOCK_REEL': 'STOCK_REEL',
        'RAL': 'RAL', 'SM Final': 'SM Final', 'Prév C1-L2 Finale': 'Prév C1-L2 Finale',
        'Prév L1-L2 Finale': 'Prév L1-L2 Finale', 'Facteur Multiplicatif Appro': 'Facteur Multiplicatif Appro',
        'Casse Prev C1-L2': 'Casse Prev C1-L2', 'Casse Prev L1-L2': 'Casse Prev L1-L2',
        'Produit Bloqué': 'Produit Bloqué', 'Commande Max avec stock max': 'Commande Max avec stock max',
        'Position JUD': 'Position JUD', 'MINIMUM_COMMANDE': 'MINIMUM_COMMANDE', 'PCB': 'PCB', 'TS': 'TS',
        'CODE_METI': 'CODE_METI', 'Ean_13': 'Ean_13'
    }

    for original_name, internal_name_key in required_detail_cols_from_source.items():
        if original_name not in df_detail.columns:
            print(f"AVERTISSEMENT LoadData: Colonne '{original_name}' non trouvée dans {detail_filepath}. Elle sera NaN.")
            df_detail[original_name] = np.nan # Créer avec NaN si elle manque
        # Renommer si la clé interne est différente du nom original et que la clé interne n'existe pas déjà
        if internal_name_key != original_name and internal_name_key not in df_detail.columns:
            df_detail.rename(columns={original_name: internal_name_key}, inplace=True)
        # Si la clé interne est la même que l'original, rien à faire sur le nom.

    # Assurer que les colonnes clés pour le matching et calculs existent avec les noms internes attendus
    if 'DATE_LIVRAISON_V2' in df_detail.columns:
        df_detail['DATE_LIVRAISON_V2'] = pd.to_datetime(df_detail['DATE_LIVRAISON_V2'], errors='coerce', dayfirst=True)
    else: df_detail['DATE_LIVRAISON_V2'] = pd.NaT
    
    if 'BM' not in df_detail.columns: # Si après renommage 'BM' n'est pas là
        if 'Type de produits V2' in df_detail.columns: # Peut-être que le renommage n'a pas eu lieu
            df_detail.rename(columns={'Type de produits V2': 'BM'}, inplace=True)
        else:
            print("ERREUR CRITIQUE: 'BM' (ou 'Type de produits V2') non trouvé dans df_detail.")
            df_detail['BM'] = '' # Fallback pour éviter crash mais le matching échouera
    df_detail['BM'] = df_detail['BM'].fillna('').astype(str).str.strip().str.lower()
    
    if 'Top' not in df_detail.columns: df_detail['Top'] = 'autre'
    else: df_detail['Top'] = df_detail['Top'].fillna('autre')
    df_detail['Top'] = df_detail['Top'].astype(str).str.strip().str.lower()
    
    if 'BQ' not in df_detail.columns: # BQ = Commande Finale avec mini et arrondi SM à 100% avec TS
        if 'Commande Finale avec mini et arrondi SM à 100% avec TS' in df_detail.columns:
             df_detail.rename(columns={'Commande Finale avec mini et arrondi SM à 100% avec TS': 'BQ'}, inplace=True)
        else:
            print("ERREUR CRITIQUE: 'BQ' (ou nom long) non trouvé dans df_detail.")
            df_detail['BQ'] = 0
    df_detail['BQ'] = pd.to_numeric(df_detail['BQ'], errors='coerce').fillna(0)
        
    bp_col_name = 'Commande Finale avec mini et arrondi SM à 100%'
    if bp_col_name not in df_detail.columns: df_detail[bp_col_name] = 0
    else: df_detail[bp_col_name] = pd.to_numeric(df_detail[bp_col_name], errors='coerce').fillna(0)


    df_pdc_sim.dropna(subset=['Type de produits V2', 'Jour livraison', 'PDC', 'Tolérance'], inplace=True)
    df_detail.dropna(subset=['BM', 'DATE_LIVRAISON_V2', 'BQ', 'Top', bp_col_name], inplace=True)
    
    print("DEBUG LoadData: Aperçu df_pdc_sim après toutes conversions (quelques colonnes clés):")
    print(df_pdc_sim[[c for c in required_pdc_sim_cols if c in df_pdc_sim.columns]].head(3).to_string())

    print("DEBUG LoadData: Aperçu df_detail (Détail.csv) après toutes conversions (quelques colonnes clés):")
    cols_to_show_detail = ['BM', 'DATE_LIVRAISON_V2', 'Top', 'BQ', bp_col_name, 'SM Final', 'CODE_METI']
    print(df_detail[[c for c in cols_to_show_detail if c in df_detail.columns]].head(3).to_string())

    print("DEBUG LoadData: Aperçu df_pdc_perm (5 premières lignes):")
    print(df_pdc_perm.head().to_string())
    
    return df_pdc_sim, df_detail, df_pdc_perm

# --- Fonctions de Calcul et d'Optimisation ---
def recalculate_for_row(
    pdc_brut_perm_value,             # Argument 1
    commande_sm_100_ligne_sim,       # Argument 2
    en_cours_stock,                  # Argument 3
    df_detail_filtered,              # Argument 4
    j_factor_trial,                  # Argument 5
    k_factor_trial,                  # Argument 6
    l_factor_trial,                  # Argument 7
    h_boost_py_trial,                # Argument 8
    poids_ac_max_param_initial,      # Argument 9 (Poids A/C lu de la ligne de param de PDC_Sim)
    type_produit_v2_param_ligne,     # Argument 10 (Type V2 de la ligne de param de PDC_Sim)
    context=""                       # Argument 11 (optionnel)
):
    # --- Étape A: Calculer F_sim_total (en utilisant cf_main_module) ---
    simulated_total_cmd_opt = 0.0
    if not df_detail_filtered.empty:
        # On suppose que cf_main_module.get_total_cf_optimisee_vectorized existe et est fonctionnelle
        # ou que vous utilisez cf_main_module.get_cf_optimisee_for_detail_line dans une boucle ici.
        # Pour la cohérence avec les logs précédents, je remets la boucle avec get_cf_optimisee_for_detail_line
        # mais la version vectorisée est préférée pour la vitesse.
        log_ccdo_details_recalc = False
        MAX_DETAIL_LOG_LINES_RECALC = 0 # Mettre à >0 pour logger les CF de détail ici
        if context and ( "LHB_CALC1" in context.upper() or "TESTING_CCDO" in context.upper() or "SOLVER_FINAL" in context.upper()): # Log pour certains contextes
            log_ccdo_details_recalc = True

        for i_detail_recalc, (_, detail_row_as_series_recalc) in enumerate(df_detail_filtered.iterrows()):
            detail_row_dict_recalc = detail_row_as_series_recalc.to_dict()
            code_meti_detail_recalc = detail_row_dict_recalc.get('CODE_METI', detail_row_dict_recalc.get('Ean_13', f'idx_detail_{i_detail_recalc}'))
            
            cf_sim_detail_line = cf_main_module.get_cf_optimisee_for_detail_line( # Assurez-vous que c'est bien le nom
                detail_row_dict_recalc, 
                j_factor_trial, 
                k_factor_trial, 
                l_factor_trial
            )
            if log_ccdo_details_recalc and i_detail_recalc < MAX_DETAIL_LOG_LINES_RECALC:
                 print(f"        RECALC_ROW/CCDO - Ligne détail {code_meti_detail_recalc}: CF calculé = {cf_sim_detail_line:.2f} (Contexte: {context})")
            if i_detail_recalc == MAX_DETAIL_LOG_LINES_RECALC and log_ccdo_details_recalc and MAX_DETAIL_LOG_LINES_RECALC > 0 :
                 print(f"        (Limitation du log de détail à {MAX_DETAIL_LOG_LINES_RECALC} lignes pour {context})...")
            simulated_total_cmd_opt += cf_sim_detail_line
    f_sim_total = simulated_total_cmd_opt

    # --- Étape B: Calculer Poids_A/C_calculé_simulé (simplifié pour la dynamique) ---
    denom_poids_ac = commande_sm_100_ligne_sim + en_cours_stock
    poids_ac_calcule_simule = 1.0 
    if denom_poids_ac != 0:
        poids_ac_calcule_simule = (f_sim_total + en_cours_stock) / denom_poids_ac
    
    poids_ac_final_pour_pdc = 1.0
    # La variable type_produit_v2_param_ligne vient des paramètres de la ligne de PDC_Sim.xlsx
    if isinstance(type_produit_v2_param_ligne, str) and type_produit_v2_param_ligne.lower().endswith("a/c"):
        # Pour un produit A/C, le Poids A/C Max (lu des params, ex: 0.8) est une borne.
        # Le "vrai" poids est celui calculé, mais il est capé.
        # La logique exacte de la formule Excel pour I4 (Poids du A/C max) est :
        # =SI(DROITE(B4;3)="A/C"; MIN(0,8; INDEX('Macro-Param'!$J:$J;EQUIV(C4;'Macro-Param'!F:F;0))) ;1)
        # Si I4 est DYNAMIQUEMENT mis à jour avec le "Poids A/C Calculé", alors la valeur de I4
        # serait le résultat de la formule du "Poids A/C Calculé" mais capé par MIN(0.8, ...).
        # On utilise poids_ac_max_param_initial comme le résultat de MIN(0.8, INDEX(...))
        poids_ac_final_pour_pdc = min(poids_ac_max_param_initial, poids_ac_calcule_simule)
        # Assurer les bornes absolues
        if poids_ac_final_pour_pdc > 1.0: poids_ac_final_pour_pdc = 1.0
        if poids_ac_final_pour_pdc < 0.0: poids_ac_final_pour_pdc = 0.0
    else: # Pour A/B ou autre
        poids_ac_final_pour_pdc = 1.0

    # --- Étape C: Calculer PDC dynamique et appliquer boost Python ---
    if pd.isna(pdc_brut_perm_value):
        # print(f"AVERTISSEMENT (recalculate_for_row): pdc_brut_perm_value est NaN pour contexte {context}. Utilisation de 0.")
        pdc_brut_perm_value = 0.0 # Le PDC de base ne peut pas être NaN pour la suite
        
    pdc_dynamique = pdc_brut_perm_value * poids_ac_final_pour_pdc
    pdc_target_adjusted_with_python_boost = pdc_dynamique * (1 + h_boost_py_trial)
    
    # --- Étape D: Calculer H, I, K, L ---
    h_sim_diff = pdc_target_adjusted_with_python_boost - en_cours_stock - f_sim_total
    i_sim_abs_diff = abs(h_sim_diff)
    
    # K_var utilise maintenant pdc_dynamique (qui a le Poids A/C Calculé) comme dénominateur
    k_sim_var_pdc = h_sim_diff / pdc_dynamique if pdc_dynamique != 0 else 0.0
    l_sim_var_abs_pdc = abs(k_sim_var_pdc)
    
    # Log résumé pour certains contextes clés
    if context and "RECALC_SUMMARY" in context.upper() or \
       (context and ("LHB_FinalState" in context.upper() or "Opt_DecisionWithLHB_JKL" in context.upper() or "Solver_Final" in context.upper() or "Opt_TestUserMaxFact" in context.upper())):
        print(f"      RECALC_SUMMARY ({context}): F_sim_Total={f_sim_total:.2f}, I_sim={i_sim_abs_diff:.2f}, K_var={k_sim_var_pdc:.2%}")
        # print(f"        (PDC_Brut={pdc_brut_perm_value:.2f}, PoidsAC_Calc={poids_ac_calcule_simule:.3f} PoidsAC_App={poids_ac_final_pour_pdc:.3f} -> PDC_Dyn={pdc_dynamique:.2f}, HBoost={h_boost_py_trial:.2%} -> PDC_Adj={pdc_target_adjusted_with_python_boost:.2f})")


    return {
        'H_sim': h_sim_diff, 
        'I_sim': i_sim_abs_diff, 
        'K_sim_var_pdc': k_sim_var_pdc,
        'L_sim_var_abs_pdc': l_sim_var_abs_pdc, 
        'F_sim_total': f_sim_total
    }

def objective_minimize_scalar_j(j_val_trial): 
    state = current_row_state_for_solver # state contient maintenant pdc_brut_perm etc.
    sim_res = recalculate_for_row(
        state['pdc_brut_perm_value'], state['commande_sm_100_ligne_sim'],
        state['en_cours_stock'], state['df_detail_filtered'], 
        j_val_trial, j_val_trial, j_val_trial, 
        state['h_boost_py_current'], # h_boost est dans l'état
        state['poids_ac_max_param_initial'], state['type_produit_v2_param_ligne'],
        "solver_scalar_obj" )
    return sim_res['I_sim']

def objective_differential_evolution_jkl(jkl_factors_trial):
    state = current_row_state_for_solver
    j_trial, k_trial, l_trial = jkl_factors_trial[0], jkl_factors_trial[1], jkl_factors_trial[2]
    sim_res = recalculate_for_row(
        state['pdc_brut_perm_value'], state['commande_sm_100_ligne_sim'],
        state['en_cours_stock'], state['df_detail_filtered'], 
        j_trial, k_trial, l_trial, 
        state['h_boost_py_current'], 
        state['poids_ac_max_param_initial'], state['type_produit_v2_param_ligne'],
        "solver_de_obj" )
    return sim_res['I_sim']


def limite_haute_basse_python(
    current_state, # Dictionnaire contenant pdc_brut, cmd_sm100, en_cours, df_detail, etc. ET h_boost_py_current
    user_max_facteur_from_pdc_sim_row, 
    row_pdc_sim_params_for_limits, # pd.Series de la ligne PDC_Sim pour lire les limites USER initiales
    type_produits_v2_debug_logging 
):
    user_max_fact_str = f"{user_max_facteur_from_pdc_sim_row:.2%}" if pd.notna(user_max_facteur_from_pdc_sim_row) else 'NaN'
    print(f"    LHB_Python ({type_produits_v2_debug_logging}): Entrée user_max_fact={user_max_fact_str}")
    # Les valeurs comme pdc_brut_perm_value sont maintenant dans current_state

    work_j_initial, work_k_initial, work_l_initial = 1.0, 1.0, 1.0 
    
    # H_boost_py est géré via current_state
    current_state['h_boost_py_current'] = 0.0 # Initialiser pour cette passe LHB

    o_lim = float(row_pdc_sim_params_for_limits.get('Limite Basse Top 500', 1.0))
    p_lim = float(row_pdc_sim_params_for_limits.get('Limite Basse Top 3000', 1.0))
    q_lim = float(row_pdc_sim_params_for_limits.get('Limite Basse Autre', 1.0))
    s_lim = float(row_pdc_sim_params_for_limits.get('Limite Haute Top 500', 1.0))
    t_lim = float(row_pdc_sim_params_for_limits.get('Limite Haute Top 3000', 1.0))
    u_lim = float(row_pdc_sim_params_for_limits.get('Limite Haute Autre', 1.0))
    print(f"      LHB_Py - Limites OPQSTU initiales (lues de PDC_Sim): O={o_lim:.2f},P={p_lim:.2f},Q={q_lim:.2f}, S={s_lim:.2f},T={t_lim:.2f},U={u_lim:.2f}")
    # print(f"      LHB_Py - JKL initiaux pour calculs LHB: [{work_j_initial:.2f},{work_k_initial:.2f},{work_l_initial:.2f}], H_py_init={current_state['h_boost_py_current']:.2f}")

    sim_res_calc1 = recalculate_for_row(
        current_state['pdc_brut_perm_value'], 
        current_state['commande_sm_100_ligne_sim'],
        current_state['en_cours_stock'], 
        current_state['df_detail_filtered'],
        work_j_initial, work_k_initial, work_l_initial, 
        current_state['h_boost_py_current'],
        current_state['poids_ac_max_param_initial'], 
        current_state['type_produit_v2_param_ligne'],
        f"LHB_Calc1_JKL1_H{current_state['h_boost_py_current']:.2f}"
    )
    
    if sim_res_calc1['K_sim_var_pdc'] > MARGE_POUR_BOOST_ET_L_VAR_SOLVER: 
        current_state['h_boost_py_current'] = -MARGE_POUR_BOOST_ET_L_VAR_SOLVER
    elif sim_res_calc1['K_sim_var_pdc'] < -MARGE_POUR_BOOST_ET_L_VAR_SOLVER: 
        current_state['h_boost_py_current'] = MARGE_POUR_BOOST_ET_L_VAR_SOLVER
    print(f"      LHB_Py - h_boost_py_current (dans current_state) ajusté à: {current_state['h_boost_py_current']:.2%}")
    
    sim_res_calc2_after_boost = recalculate_for_row(
        current_state['pdc_brut_perm_value'], 
        current_state['commande_sm_100_ligne_sim'],
        current_state['en_cours_stock'], 
        current_state['df_detail_filtered'],
        work_j_initial, work_k_initial, work_l_initial, 
        current_state['h_boost_py_current'],
        current_state['poids_ac_max_param_initial'],
        current_state['type_produit_v2_param_ligne'],
        f"LHB_Calc2_JKL1_H{current_state['h_boost_py_current']:.2f}"
    )
    k_var_pdc_apres_boost = sim_res_calc2_after_boost['K_sim_var_pdc']
    print(f"      LHB_Py - K_var_pdc après H_boost_py (JKL=1): {k_var_pdc_apres_boost:.2%}")

    safe_user_max_facteur = float(user_max_facteur_from_pdc_sim_row) if pd.notna(user_max_facteur_from_pdc_sim_row) else 1.0
    final_j_start_lhb, final_k_start_lhb, final_l_start_lhb = work_j_initial, work_k_initial, work_l_initial

    if k_var_pdc_apres_boost > 0: 
        # print(f"      LHB_Py - K_var > 0. MODIFICATION Lims LHB: Basses O,P,Q=1. Hautes S,T,U={safe_user_max_facteur:.2%}.")
        o_lim, p_lim, q_lim = 1.0, 1.0, 1.0
        s_lim, t_lim, u_lim = safe_user_max_facteur, safe_user_max_facteur, safe_user_max_facteur
    else: 
        # print(f"      LHB_Py - K_var <= 0. MODIFICATION Lims LHB & JKL_start_LHB via cascade.")
        s_lim, t_lim, u_lim = 1.0, 1.0, 1.0 
        current_j_cascade, current_k_cascade, current_l_cascade = final_j_start_lhb, final_k_start_lhb, final_l_start_lhb

        current_l_cascade = 0.0; q_lim = 0.0             
        res_cascade_L0 = recalculate_for_row(
            current_state['pdc_brut_perm_value'], current_state['commande_sm_100_ligne_sim'],
            current_state['en_cours_stock'], current_state['df_detail_filtered'],
            current_j_cascade, current_k_cascade, current_l_cascade, 
            current_state['h_boost_py_current'],
            current_state['poids_ac_max_param_initial'], current_state['type_produit_v2_param_ligne'],
            "LHB_CascadeL0"
        )
        # print(f"        LHB_Cascade - K_var après L_start=0, Q_lim=0 (J_start={current_j_cascade:.1f},K_start={current_k_cascade:.1f}): {res_cascade_L0['K_sim_var_pdc']:.2%}")

        if res_cascade_L0['K_sim_var_pdc'] <= 0:
            current_k_cascade = 0.0; p_lim = 0.0         
            res_cascade_K0 = recalculate_for_row(
                current_state['pdc_brut_perm_value'], current_state['commande_sm_100_ligne_sim'],
                current_state['en_cours_stock'], current_state['df_detail_filtered'],
                current_j_cascade, current_k_cascade, current_l_cascade, 
                current_state['h_boost_py_current'],
                current_state['poids_ac_max_param_initial'], current_state['type_produit_v2_param_ligne'],
                "LHB_CascadeK0"
            )
            # print(f"        LHB_Cascade - K_var après K_start=0, P_lim=0 (J_start={current_j_cascade:.1f},L_start={current_l_cascade:.1f}): {res_cascade_K0['K_sim_var_pdc']:.2%}")

            if res_cascade_K0['K_sim_var_pdc'] <= 0:
                o_lim = 0.0     
                # print(f"        LHB_Cascade - K_var encore <=0. Set O_lim=0. (J_start={current_j_cascade:.1f},K_start={current_k_cascade:.1f},L_start={current_l_cascade:.1f})")
        
        final_j_start_lhb, final_k_start_lhb, final_l_start_lhb = current_j_cascade, current_k_cascade, current_l_cascade
        
    final_sim_results_lhb = recalculate_for_row(
        current_state['pdc_brut_perm_value'], current_state['commande_sm_100_ligne_sim'],
        current_state['en_cours_stock'], current_state['df_detail_filtered'], 
        final_j_start_lhb, final_k_start_lhb, final_l_start_lhb, 
        current_state['h_boost_py_current'], 
        current_state['poids_ac_max_param_initial'], current_state['type_produit_v2_param_ligne'],
        "LHB_FinalState"
    )
    
    # print(f"    LHB_Python ({type_produits_v2_debug_logging}): Sortie. JKL_start_LHB=[{final_j_start_lhb:.2f},{final_k_start_lhb:.2f},{final_l_start_lhb:.2f}], H_py={current_state['h_boost_py_current']:.2%}, Lims_Modifiées O={o_lim:.2f},P={p_lim:.2f},Q={q_lim:.2f}, S={s_lim:.2f},T={t_lim:.2f},U={u_lim:.2f}")
    return (final_j_start_lhb, final_k_start_lhb, final_l_start_lhb, current_state['h_boost_py_current'], 
            o_lim, p_lim, q_lim, s_lim, t_lim, u_lim, 
            final_sim_results_lhb)

# Dans optimisation_globale.py

def optimisation_macro_python(
    # JKL de départ et H_boost venant de la sortie de limite_haute_basse_python
    j_start_optim_lhb, 
    k_start_optim_lhb, 
    l_start_optim_lhb, 
    h_boost_py_from_lhb, # C'est le current_row_state_for_solver['h_boost_py_current'] final de LHB
    
    # row_params_and_lhb_limits est une pd.Series ou dict qui contient :
    #  - Les 'Min Facteur' et 'Max Facteur' originaux lus de PDC_Sim.xlsx
    #  - Les limites 'LHB_O_lim', 'LHB_P_lim', etc., qui ont été MODIFIÉES par limite_haute_basse_python
    row_params_and_lhb_limits 
):
    # Les données de base pour recalculate_for_row (PDC_brut, CmdSM100, EnCours, df_detail)
    # sont lues depuis la variable globale current_row_state_for_solver.
    # h_boost_py_from_lhb est aussi le h_boost_py_current qui sera utilisé par recalculate_for_row
    # via current_row_state_for_solver.
    
    print(f"    Optimisation_Macro: Entrée JKL_start_LHB=[{j_start_optim_lhb:.2f},{k_start_optim_lhb:.2f},{l_start_optim_lhb:.2f}], H_py_de_LHB={h_boost_py_from_lhb:.2%}")
    
    # Récupérer les limites LHB modifiées et les bornes utilisateur
    lim_bas_j_lhb = float(row_params_and_lhb_limits['LHB_O_lim'])
    lim_bas_k_lhb = float(row_params_and_lhb_limits['LHB_P_lim'])
    lim_bas_l_lhb = float(row_params_and_lhb_limits['LHB_Q_lim'])
    lim_haut_j_lhb = float(row_params_and_lhb_limits['LHB_S_lim'])
    lim_haut_k_lhb = float(row_params_and_lhb_limits['LHB_T_lim'])
    lim_haut_l_lhb = float(row_params_and_lhb_limits['LHB_U_lim'])
    
    user_min_facteur = float(row_params_and_lhb_limits['Min Facteur']) if pd.notna(row_params_and_lhb_limits['Min Facteur']) else 0.0
    user_max_facteur = float(row_params_and_lhb_limits['Max Facteur']) if pd.notna(row_params_and_lhb_limits['Max Facteur']) else 1.0

    print(f"      Opt - Limites LHB_Modifiées utilisées pour tests et capage: J:[{lim_bas_j_lhb:.2f}-{lim_haut_j_lhb:.2f}], K:[{lim_bas_k_lhb:.2f}-{lim_haut_k_lhb:.2f}], L:[{lim_bas_l_lhb:.2f}-{lim_haut_l_lhb:.2f}]")
    print(f"      Opt - Bornes User: MinFact={user_min_facteur:.2%}, MaxFact={user_max_facteur:.2%}")

    # Déterminer TypeLissage basé sur les limites O,P,Q MODIFIÉES par LHB
    # VBA: "If (Cells(i, PremiereColonneLimiteBasse) * Cells(i, PremiereColonneLimiteBasse + 1) * Cells(i, PremiereColonneLimiteBasse + 2)) >= 1 Then"
    if (lim_bas_j_lhb * lim_bas_k_lhb * lim_bas_l_lhb >= 1.0): 
        type_lissage = 1 # Hausse
    else: 
        type_lissage = 0 # Baisse
    print(f"      Opt - TypeLissage déterminé: {'Hausse' if type_lissage == 1 else 'Baisse'} (Prod Lims Basses LHB_Modifiées = {lim_bas_j_lhb * lim_bas_k_lhb * lim_bas_l_lhb:.2f})")

    # JKL actuels à retourner si pas de solveur ou si le solveur échoue
    # Initialisés avec les JKL de sortie de LHB
    current_j, current_k, current_l = j_start_optim_lhb, k_start_optim_lhb, l_start_optim_lhb
    # Le H_boost est celui de LHB, il est déjà dans current_row_state_for_solver['h_boost_py_current']
    # et est égal à h_boost_py_from_lhb.

    # Recalculer avec les JKL de LHB pour avoir un point de référence si le test MaxFacteur n'est pas concluant
    sim_results_with_lhb_jkl = recalculate_for_row(
        current_row_state_for_solver['pdc_brut_perm_value'],
        current_row_state_for_solver['commande_sm_100_ligne_sim'],
        current_row_state_for_solver['en_cours_stock'],
        current_row_state_for_solver['df_detail_filtered'],
        j_start_optim_lhb, k_start_optim_lhb, l_start_optim_lhb, 
        h_boost_py_from_lhb, # = current_row_state_for_solver['h_boost_py_current']
        current_row_state_for_solver['poids_ac_max_param_initial'],
        current_row_state_for_solver['type_produit_v2_param_ligne'],
        "Opt_InitialWithLHB_JKL"
    )

    # VBA: "If Cells(i, ColonneBorneMin) = Cells(i, ColonneBorneMax) Then"
    if user_min_facteur == user_max_facteur:
        val_fixee_user = user_max_facteur 
        # JKL finaux sont cette val_fixee_user, capée par les limites LHB
        current_j = min(max(val_fixee_user, lim_bas_j_lhb), lim_haut_j_lhb)
        current_k = min(max(val_fixee_user, lim_bas_k_lhb), lim_haut_k_lhb)
        current_l = min(max(val_fixee_user, lim_bas_l_lhb), lim_haut_l_lhb)
        print(f"      Opt - User Min/Max Facteur identical ({val_fixee_user:.2%}). JKL finaux (capés): J={current_j:.2%}, K={current_k:.2%}, L={current_l:.2%}")
        
        final_sim_results = recalculate_for_row(
            current_row_state_for_solver['pdc_brut_perm_value'], current_row_state_for_solver['commande_sm_100_ligne_sim'],
            current_row_state_for_solver['en_cours_stock'], current_row_state_for_solver['df_detail_filtered'],
            current_j, current_k, current_l, h_boost_py_from_lhb,
            current_row_state_for_solver['poids_ac_max_param_initial'], current_row_state_for_solver['type_produit_v2_param_ligne'],
            "Opt_UserMinMaxIdentical"
        )
        return False, type_lissage, current_j, current_k, current_l, h_boost_py_from_lhb, final_sim_results

    # VBA: "Else ... .Range(.Cells(i, PremiereColonneParametrage), ...).Value = Cells(i, ColonneBorneMax).Value"
    # Test avec user_max_facteur, capé par les limites LHB (O,P,Q,S,T,U)
    j_test_user_max = min(max(user_max_facteur, lim_bas_j_lhb), lim_haut_j_lhb)
    k_test_user_max = min(max(user_max_facteur, lim_bas_k_lhb), lim_haut_k_lhb)
    l_test_user_max = min(max(user_max_facteur, lim_bas_l_lhb), lim_haut_l_lhb)
    print(f"      Opt - Test avec JKL = User Max Facteur ({user_max_facteur:.2%}), capés par Lims LHB_Modifiées à JKL=[{j_test_user_max:.2f},{k_test_user_max:.2f},{l_test_user_max:.2f}]")
    
    sim_results_test_user_max = recalculate_for_row(
        current_row_state_for_solver['pdc_brut_perm_value'], current_row_state_for_solver['commande_sm_100_ligne_sim'],
        current_row_state_for_solver['en_cours_stock'], current_row_state_for_solver['df_detail_filtered'],
        j_test_user_max, k_test_user_max, l_test_user_max, h_boost_py_from_lhb,
        current_row_state_for_solver['poids_ac_max_param_initial'], current_row_state_for_solver['type_produit_v2_param_ligne'],
        "Opt_TestUserMaxFact"
    )
    print(f"      Opt - Résultats avec Test User Max Fact: I_sim={sim_results_test_user_max['I_sim']:.2f}, K_var={sim_results_test_user_max['K_sim_var_pdc']:.2%}")

    # Logique VBA pour décider quels JKL utiliser pour la condition du Solveur et comme JKL finaux si pas de Solveur.
    # VBA: "If Cells(i + DecalageParametreSimulation, ColonneResultatVariationRelative).Value <= 0 Then .Range(...JKL...).Value = 1"
    #       (Le .Value=1 signifie JKL de départ de LHB qui sont 1 ou 0 après cascade)
    
    results_for_solver_decision = None
    jkl_final_si_pas_solveur = None

    if sim_results_test_user_max['K_sim_var_pdc'] <= 0: 
        print(f"      Opt - K_var_pdc ({sim_results_test_user_max['K_sim_var_pdc']:.2%}) <= 0 après Test User Max. "
              f"Utilisation des JKL_start_LHB pour décision Solveur et comme JKL finaux (si pas de Solveur): [{j_start_optim_lhb:.2f},{k_start_optim_lhb:.2f},{l_start_optim_lhb:.2f}]")
        
        current_j, current_k, current_l = j_start_optim_lhb, k_start_optim_lhb, l_start_optim_lhb
        # results_for_solver_decision sont ceux calculés avec JKL de LHB au début de cette fonction
        results_for_solver_decision = sim_results_with_lhb_jkl
    else: # K_var_pdc > 0 après test UserMaxFacteur
        # Dans ce cas, VBA n'appelle PAS le solveur. Les JKL restent ceux du test MaxFacteur.
        print(f"      Opt - K_var_pdc ({sim_results_test_user_max['K_sim_var_pdc']:.2%}) > 0 après Test User Max. VBA n'appellerait pas le solveur. JKL finaux = JKL du Test User Max.")
        current_j, current_k, current_l = j_test_user_max, k_test_user_max, l_test_user_max
        # Les résultats pour la décision sont ceux du test MaxFacteur.
        results_for_solver_decision = sim_results_test_user_max
        # Forcer "pas de solveur" pour correspondre au VBA
        needs_solver_flag = False
        print(f"      Opt - Solver NON requis (car K_var > 0 après Test Max, comme VBA).")
        return needs_solver_flag, type_lissage, current_j, current_k, current_l, h_boost_py_from_lhb, results_for_solver_decision

    # Si on est ici, c'est que K_var_pdc du TestMax était <= 0.
    # On utilise results_for_solver_decision (calculé avec JKL de LHB) pour la condition d'appel.
    condition_L = results_for_solver_decision['L_sim_var_abs_pdc'] > MARGE_POUR_BOOST_ET_L_VAR_SOLVER
    condition_I = results_for_solver_decision['I_sim'] > MARGE_I_POUR_SOLVER_CONDITION
    
    needs_solver_flag = condition_L and condition_I

    if needs_solver_flag:
        print(f"      Opt - Solver REQUIS. (L_var_abs={results_for_solver_decision['L_sim_var_abs_pdc']:.2%} > {MARGE_POUR_BOOST_ET_L_VAR_SOLVER:.2%}, "
              f"ET I_sim={results_for_solver_decision['I_sim']:.2f} > {MARGE_I_POUR_SOLVER_CONDITION:.2f})")
        # Le solveur partira des JKL de LHB (j_start_optim_lhb, k_start_optim_lhb, l_start_optim_lhb)
        return True, type_lissage, j_start_optim_lhb, k_start_optim_lhb, l_start_optim_lhb, h_boost_py_from_lhb, results_for_solver_decision
    else:
        print(f"      Opt - Solver NON requis (conditions L et/ou I non remplies avec JKL de LHB).")
        # Les JKL finaux sont ceux de LHB (current_j,k,l ont été mis à j_start_optim_lhb etc.)
        return False, type_lissage, current_j, current_k, current_l, h_boost_py_from_lhb, results_for_solver_decision

# Dans optimisation_globale.py

def solver_macro_python(
    # JKL de départ venant de LHB (passés par optimisation_macro si solveur requis)
    j_start_solver, 
    k_start_solver, 
    l_start_solver, 
    # H_boost venant de LHB (est aussi dans current_row_state_for_solver)
    h_boost_py_for_solver, 
    type_lissage, 
    # row_params_with_lhb_limits contient les Lims LHB MODIFIÉES (LHB_O_lim etc.)
    # et potentiellement les Min/Max Facteur User si besoin (mais pas pour les bornes du solveur ici)
    row_params_with_lhb_limits 
):
    global current_row_state_for_solver # Nécessaire pour que les fonctions objectif y accèdent

    # Assurer que h_boost_py_current dans l'état global est bien celui fixé par LHB
    # car les fonctions objectif le lisent depuis là.
    current_row_state_for_solver['h_boost_py_current'] = h_boost_py_for_solver
    
    # Initialiser les JKL finaux avec les valeurs de départ (au cas où le solveur échoue)
    final_j, final_k, final_l = float(j_start_solver), float(k_start_solver), float(l_start_solver) 
    
    print(f"    Solver_Macro: Entrée JKL_start_LHB=[{final_j:.2f},{final_k:.2f},{final_l:.2f}], H_py={h_boost_py_for_solver:.2%}, TypeLissage={type_lissage}")

    # Récupérer les limites LHB modifiées pour les bornes du solveur
    lim_bas_j_lhb = float(row_params_with_lhb_limits['LHB_O_lim'])
    lim_haut_j_lhb = float(row_params_with_lhb_limits['LHB_S_lim'])
    lim_bas_k_lhb = float(row_params_with_lhb_limits['LHB_P_lim'])
    lim_haut_k_lhb = float(row_params_with_lhb_limits['LHB_T_lim'])
    lim_bas_l_lhb = float(row_params_with_lhb_limits['LHB_Q_lim'])
    lim_haut_l_lhb = float(row_params_with_lhb_limits['LHB_U_lim'])
    print(f"      Solver - Limites LHB_Modifiées utilisées pour bornes solveur: J:[{lim_bas_j_lhb:.2f}-{lim_haut_j_lhb:.2f}], K:[{lim_bas_k_lhb:.2f}-{lim_haut_k_lhb:.2f}], L:[{lim_bas_l_lhb:.2f}-{lim_haut_l_lhb:.2f}]")

    bounds_solver = [] # Les bornes effectives pour le solveur
    
    if type_lissage == 1: # Cas Hausse
        print(f"      Solver - Mode Hausse (minimize_scalar)")
        # Vérifier si les bornes pour J sont valides
        if lim_bas_j_lhb > lim_haut_j_lhb : 
            print(f"        Lims J invalides ([{lim_bas_j_lhb:.3f},{lim_haut_j_lhb:.3f}]). JKL finaux = JKL_start_LHB.")
            # final_j, final_k, final_l sont déjà les valeurs de départ
        elif lim_bas_j_lhb == lim_haut_j_lhb:
            print(f"        Lims J identiques [{lim_bas_j_lhb:.3f}]. JKL fixés à cette valeur.")
            final_j = final_k = final_l = lim_bas_j_lhb 
        else:
            bounds_solver_j = (lim_bas_j_lhb, lim_haut_j_lhb)
            res_scalar = minimize_scalar(objective_minimize_scalar_j, bounds=bounds_solver_j, method='bounded')
            if res_scalar.success: 
                j_optimal_scalar = res_scalar.x
                print(f"        minimize_scalar succès: J_optimal_brut={j_optimal_scalar:.4f}.")
                # Appliquer la logique de capage Excel pour K et L si TypeLissage=1
                # K et L suivent J, mais doivent aussi respecter leurs propres bornes LHB (P,T et Q,U)
                # même si elles ne sont pas des variables directes du solveur.
                final_j = j_optimal_scalar
                final_k_brut = final_j # K = J
                final_k = min(max(final_k_brut, lim_bas_k_lhb), lim_haut_k_lhb) # Capé par bornes de K (LHB)
                
                final_l_brut = final_k # L suit K (déjà capé par ses bornes P,T)
                final_l = min(max(final_l_brut, lim_bas_l_lhb), lim_haut_l_lhb) # Capé par bornes de L (LHB)
                print(f"        JKL après capage Excel style (K,L suivent J mais respectent leurs Lims LHB): [{final_j:.4f},{final_k:.4f},{final_l:.4f}]")
            else: 
                print(f"        minimize_scalar échec. JKL restent à JKL_start_LHB [{j_start_solver:.3f},{k_start_solver:.3f},{l_start_solver:.3f}]")
    
    else: # Cas Baisse (type_lissage == 0)
        print(f"      Solver - Mode Baisse (differential_evolution)")
        
        def sanitize_bound(low, high, default_low=0.0, default_high=1.0):
            low_f, high_f = float(low), float(high)
            if high_f < low_f: return (low_f, low_f) # Si inversé, fixer à la borne basse (ou haute)
            return (low_f, high_f) 

        bounds_solver = [ # Bornes pour DE
            sanitize_bound(lim_bas_j_lhb, lim_haut_j_lhb), 
            sanitize_bound(lim_bas_k_lhb, lim_haut_k_lhb), 
            sanitize_bound(lim_bas_l_lhb, lim_haut_l_lhb) 
        ]
        
        # Si toutes les bornes sont "plates" (min=max) ou invalides après sanitation
        if all(b[0] >= b[1] for b in bounds_solver): 
            print(f"        Toutes lims DE plates/invalides après sanitation. JKL fixés à leurs bornes basses LHB respectives.")
            final_j, final_k, final_l = bounds_solver[0][0], bounds_solver[1][0], bounds_solver[2][0]
        else:
            def constraint_j_ge_k_ge_l(jkl_arr): return np.array([jkl_arr[0] - jkl_arr[1], jkl_arr[1] - jkl_arr[2]])
            nlc = NonlinearConstraint(constraint_j_ge_k_ge_l, 0, np.inf)
            
            initial_guess_de = [final_j, final_k, final_l] # Partir des JKL de LHB
            for i_g in range(3): # Clipper x0 aux bornes sanitizées
                initial_guess_de[i_g] = min(max(initial_guess_de[i_g], bounds_solver[i_g][0]), bounds_solver[i_g][1])
            
            print(f"        DE 1er run. Bornes sanitizées: J:{bounds_solver[0]}, K:{bounds_solver[1]}, L:{bounds_solver[2]}. x0 clippé: {initial_guess_de}")
            solver_res_1 = differential_evolution(
                objective_differential_evolution_jkl, bounds_solver, constraints=[nlc], x0=initial_guess_de, 
                maxiter=50, popsize=30, tol=0.001, mutation=(0.5,1), recombination=0.7, disp=False, seed=42 # Params réduits pour vitesse
            )
            if solver_res_1.success: 
                final_j, final_k, final_l = solver_res_1.x
                print(f"        DE 1er run succès: JKL optimisé à [{final_j:.3f},{final_k:.3f},{final_l:.3f}]")
            else: 
                # Si échec, JKL restent à initial_guess_de (qui sont les JKL de LHB clippés)
                print(f"        DE 1er run échec. JKL restent à x0_clippé_LHB ({initial_guess_de}).")
            
            # Évaluation après le 1er run
            temp_res_after_1st_run = recalculate_for_row(
                current_row_state_for_solver['pdc_brut_perm_value'], current_row_state_for_solver['commande_sm_100_ligne_sim'],
                current_row_state_for_solver['en_cours_stock'], current_row_state_for_solver['df_detail_filtered'],
                final_j, final_k, final_l, h_boost_py_for_solver,
                current_row_state_for_solver['poids_ac_max_param_initial'], current_row_state_for_solver['type_produit_v2_param_ligne'],
                "Solver_Post1stDE"
            )
            
            # Condition VBA pour 2ème run: Si L_sim_var_abs_pdc > MARGE_POUR_BOOST_ET_L_VAR_SOLVER
            # (MARGE_POUR_BOOST_ET_L_VAR_SOLVER est 0.00, donc si L_var_abs > 0)
            if temp_res_after_1st_run['L_sim_var_abs_pdc'] > MARGE_POUR_BOOST_ET_L_VAR_SOLVER:
                print(f"        DE 2nd run requis (L_var_abs={temp_res_after_1st_run['L_sim_var_abs_pdc']:.2%} > {MARGE_POUR_BOOST_ET_L_VAR_SOLVER:.2%}).")
                x0_2nd_run = [final_j, final_k, final_l] 
                for i_g in range(3): x0_2nd_run[i_g] = min(max(x0_2nd_run[i_g], bounds_solver[i_g][0]), bounds_solver[i_g][1])

                solver_res_2 = differential_evolution(
                    objective_differential_evolution_jkl, bounds_solver, constraints=[nlc], x0=x0_2nd_run, 
                    maxiter=100, popsize=50, tol=0.001, mutation=(0.5,1), recombination=0.7, disp=False, seed=43 # Params un peu plus élevés                 
                )
                if solver_res_2.success: 
                    final_j, final_k, final_l = solver_res_2.x
                    print(f"        DE 2nd run succès: JKL optimisé à [{final_j:.3f},{final_k:.3f},{final_l:.3f}]")
                else: 
                    print(f"        DE 2nd run échec pour améliorer.")
            else: 
                print(f"        DE 1er run suffisant (L_var_abs={temp_res_after_1st_run['L_sim_var_abs_pdc']:.2%}).")

    # Assurer J >= K >= L si Baisse, après toutes les optimisations
    if type_lissage == 0:
        # Assurer que final_j,k,l sont des floats avant min/max
        fj, fk, fl = float(final_j), float(final_k), float(final_l)
        final_j = fj
        final_k = min(fk, fj)
        final_l = min(fl, final_k)
        print(f"      Solver - Après contrainte J>=K>=L (si Baisse): JKL=[{final_j:.3f},{final_k:.3f},{final_l:.3f}]")

    # Recalculer une dernière fois avec les JKL finaux pour obtenir la structure de résultats complète
    final_sim_results = recalculate_for_row(
        current_row_state_for_solver['pdc_brut_perm_value'], 
        current_row_state_for_solver['commande_sm_100_ligne_sim'],
        current_row_state_for_solver['en_cours_stock'], 
        current_row_state_for_solver['df_detail_filtered'], 
        final_j, final_k, final_l, 
        h_boost_py_for_solver, # Utiliser le h_boost fixé par LHB
        current_row_state_for_solver['poids_ac_max_param_initial'], 
        current_row_state_for_solver['type_produit_v2_param_ligne'],
        "Solver_Final"
    )
    print(f"    Solver_Macro - Sortie: JKL=[{final_j:.3f},{final_k:.3f},{final_l:.3f}], I_sim={final_sim_results['I_sim']:.2f}")
    return final_j, final_k, final_l, h_boost_py_for_solver, final_sim_results


# --- Boucle Principale ---
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PDC_SIM_FILE_INPUT = os.path.join(SCRIPT_DIR, "PDC_Sim_Input_For_Optim.xlsx") 
    DETAIL_FILE_INPUT = os.path.join(SCRIPT_DIR, "initial_merged_predictions.csv")
    # Chemin pour le fichier PDC Perm corrigé is in PDC/PDC.xlsx
    PDC_DIR = os.path.join(SCRIPT_DIR, "PDC")
    PDC_PERM_FILE_INPUT = os.path.join(PDC_DIR, "PDC.xlsx") 
    OUTPUT_FILE = os.path.join(SCRIPT_DIR, "PDC_Sim_Optimized_Python.xlsx")

    if not (os.path.exists(PDC_SIM_FILE_INPUT) and \
            os.path.exists(DETAIL_FILE_INPUT) and \
            os.path.exists(PDC_PERM_FILE_INPUT)):
        print(f"Erreur: Un des fichiers d'entrée est non trouvé dans {SCRIPT_DIR}")
        print(f"  PDC_Sim.xlsx: {os.path.exists(PDC_SIM_FILE_INPUT)}")
        print(f"  Détail.csv: {os.path.exists(DETAIL_FILE_INPUT)}")
        print(f"  PDC Perm.xlsx: {os.path.exists(PDC_PERM_FILE_INPUT)}")
        exit()

    print("--- DÉBUT DU CHARGEMENT DES DONNÉES ---")
    df_pdc_sim_input, df_detail, df_pdc_perm = load_data(
        PDC_SIM_FILE_INPUT, 
        DETAIL_FILE_INPUT, 
        PDC_PERM_FILE_INPUT
    )
    print("--- FIN DU CHARGEMENT DES DONNÉES ---")
    
    # Bloc de débogage pour les valeurs uniques (peut être commenté une fois le matching validé)
    print("\nDEBUG MATCHING - Valeurs uniques pour le matching AVANT LA BOUCLE :")
    print("  PDC_Sim 'Type de produits V2' (unique, lower):")
    print(sorted(df_pdc_sim_input['Type de produits V2'].unique()))
    print("  PDC_Sim 'Jour livraison' (unique, date):")
    print(sorted(df_pdc_sim_input['Jour livraison'].dt.date.unique()))
    print("  Détail.csv 'BM' (unique, lower):")
    if 'BM' in df_detail.columns: print(sorted(df_detail['BM'].unique()))
    else: print("    Colonne 'BM' non trouvée dans df_detail.")
    print("  Détail.csv 'DATE_LIVRAISON_V2' (unique, date):")
    if 'DATE_LIVRAISON_V2' in df_detail.columns:
        valid_dates_detail = df_detail['DATE_LIVRAISON_V2'].dropna()
        if not valid_dates_detail.empty: print(sorted(valid_dates_detail.dt.date.unique()))
        else: print("    Toutes les dates dans 'DATE_LIVRAISON_V2' de df_detail sont NaT ou la colonne est vide.")
    else: print("    Colonne 'DATE_LIVRAISON_V2' non trouvée dans df_detail.")
    print("-" * 30)
    
    df_pdc_sim_results = df_pdc_sim_input.copy()
    cols_python_results = [
        'PY_Opt_J', 'PY_Opt_K', 'PY_Opt_L', 'PY_Opt_H_Boost', 
        'PY_F_Sim', 'PY_I_Sim', 'PY_TypeLissage', 'PY_Comment_Optim',
        'LHB_O_lim', 'LHB_P_lim', 'LHB_Q_lim', 
        'LHB_S_lim', 'LHB_T_lim', 'LHB_U_lim',
        'LHB_J_start', 'LHB_K_start', 'LHB_L_start' 
    ]
    for col in cols_python_results: df_pdc_sim_results[col] = np.nan
    df_pdc_sim_results['PY_Comment_Optim'] = ""


    print(f"\n--- DÉBUT DE L'OPTIMISATION GLOBALE ({len(df_pdc_sim_input)} lignes à traiter) ---")
    print(f"Marges utilisées: Boost/L_var Solver = {MARGE_POUR_BOOST_ET_L_VAR_SOLVER:.2%}, I_abs Solver Cond = {MARGE_I_POUR_SOLVER_CONDITION:.2f}")

    for index, row_data_orig_pdc_sim in df_pdc_sim_input.iterrows():
        print(f"\nTraitement Ligne {index}: {row_data_orig_pdc_sim['Type de produits V2']} @ {row_data_orig_pdc_sim['Jour livraison'].strftime('%Y-%m-%d') if pd.notnull(row_data_orig_pdc_sim['Jour livraison']) else 'Date Invalide'}")
        
        current_row_params_for_macros = row_data_orig_pdc_sim.copy() 
        type_prod_v2_current = current_row_params_for_macros['Type de produits V2'] # Déjà en minuscules par load_data
        jour_liv_current = current_row_params_for_macros['Jour livraison']
        type_produit_current = current_row_params_for_macros['Type de produits'] # Pour lookup PDC Perm, gardé avec sa casse originale

        if pd.isna(jour_liv_current) or pd.isna(type_prod_v2_current) or pd.isna(current_row_params_for_macros['PDC']):
            df_pdc_sim_results.loc[index, 'PY_Comment_Optim'] = "Données clé manquantes (V2, Jour, PDC)"; continue
        
        # Récupérer le PDC_Brut_Perm
        pdc_brut_val = np.nan
        try:
            if pd.notna(jour_liv_current) and pd.notna(type_produit_current) and \
               jour_liv_current in df_pdc_perm.index and type_produit_current in df_pdc_perm.columns:
                 pdc_brut_val = df_pdc_perm.loc[jour_liv_current, type_produit_current]
            if pd.isna(pdc_brut_val): 
                print(f"  AVERTISSEMENT Ligne {index}: PDC Brut non trouvé dans 'PDC Perm' pour '{type_produit_current}' à {jour_liv_current.strftime('%Y-%m-%d')}. PDC Brut sera NaN.")
        except KeyError as e_key:
            print(f"  AVERTISSEMENT Ligne {index}: Clé non trouvée pour PDC Brut dans 'PDC Perm' ('{str(e_key)}'). PDC Brut sera NaN.")
        except Exception as e_lookup:
            print(f"  AVERTISSEMENT Ligne {index}: Erreur lookup PDC Brut: {e_lookup}. PDC Brut sera NaN.")

        # Valeurs fixes pour la ligne de simulation en cours
        commande_sm_100_val = float(current_row_params_for_macros['Commande SM à 100%']) if pd.notna(current_row_params_for_macros['Commande SM à 100%']) else 0.0
        en_cours_val = float(current_row_params_for_macros['En-cours']) if pd.notna(current_row_params_for_macros['En-cours']) else 0.0
        poids_ac_max_initial_val = float(current_row_params_for_macros['Poids du A/C max']) if pd.notna(current_row_params_for_macros['Poids du A/C max']) else 1.0
        
        df_detail_filt_current = df_detail[(df_detail['BM'] == type_prod_v2_current) & \
                                     (df_detail['DATE_LIVRAISON_V2'] == jour_liv_current)].copy()
        
        # Initialiser l'état pour cette ligne, qui sera utilisé par les fonctions objectif et potentiellement LHB
        current_row_state_for_solver = {
            'pdc_brut_perm_value': pdc_brut_val, # Peut être NaN si non trouvé
            'commande_sm_100_ligne_sim': commande_sm_100_val,
            'en_cours_stock': en_cours_val, 
            'df_detail_filtered': df_detail_filt_current, 
            'poids_ac_max_param_initial': poids_ac_max_initial_val,
            'type_produit_v2_param_ligne': type_prod_v2_current, # type_prod_v2 de la ligne de param (identique à la sim)
            'h_boost_py_current': 0.0 # Sera mis à jour par LHB
        }
        user_max_facteur_val = float(row_data_orig_pdc_sim['Max Facteur']) if pd.notna(row_data_orig_pdc_sim['Max Facteur']) else 1.0

        # --- Bloc de test pour cf_main_module.get_cf_optimisee_for_detail_line ---
        DEBUG_TARGET_INDEX_FOR_CCDO_TEST = 0 # Mettre à -1 pour désactiver ce test spécifique
        if index == DEBUG_TARGET_INDEX_FOR_CCDO_TEST: 
            print(f"  DEBUG Ligne {index} - Type: {type_prod_v2_current}, Date: {jour_liv_current}")
            print(f"    Taille de df_detail_filt_current: {len(df_detail_filt_current)}")
            detail_row_for_debug_dict = None 
            if not df_detail_filt_current.empty:
                bp_col_name_in_detail = 'Commande Finale avec mini et arrondi SM à 100%'
                lignes_detail_test_bp_positif = df_detail_filt_current[pd.to_numeric(df_detail_filt_current.get(bp_col_name_in_detail, 0), errors='coerce').fillna(0) > 0]
                
                if not lignes_detail_test_bp_positif.empty:
                    detail_row_for_debug_dict = lignes_detail_test_bp_positif.iloc[0].to_dict()
                    print("    Première ligne de détail AVEC BP > 0 trouvée pour le test cf_main_module:")
                elif not df_detail_filt_current.empty:
                    print("    AUCUNE ligne de détail trouvée avec BP > 0. Utilisation de la première ligne disponible.")
                    detail_row_for_debug_dict = df_detail_filt_current.iloc[0].to_dict()
                
                if detail_row_for_debug_dict:
                    cols_to_show_detail = ['BM', 'Top', 'BQ', bp_col_name_in_detail] # Ajouter d'autres si besoin
                    cols_present = [c for c in cols_to_show_detail if c in detail_row_for_debug_dict]
                    print(f"      Données détail pour test: {{ {', '.join([f'{k}: {detail_row_for_debug_dict.get(k)}' for k in cols_present])} }}")
                    
                    print(f"    TESTING cf_main_module.get_cf_optimisee_for_detail_line:")
                    cf_test_jkl1 = cf_main_module.get_cf_optimisee_for_detail_line(detail_row_for_debug_dict, 1.0, 1.0, 1.0)
                    print(f"      Résultat direct cf_main_module avec JKL=1.0: CF = {cf_test_jkl1}")
                    cf_test_jkl4 = cf_main_module.get_cf_optimisee_for_detail_line(detail_row_for_debug_dict, 4.0, 4.0, 4.0)
                    print(f"      Résultat direct cf_main_module avec JKL=4.0: CF = {cf_test_jkl4}")
            else: print("    df_detail_filt_current est VIDE. F_sim sera 0.")
        # --- Fin du Bloc de test ---
        
        if df_detail_filt_current.empty and pd.notna(current_row_params_for_macros['Commande SM à 100%']) and current_row_params_for_macros['Commande SM à 100%'] > 0 :
            print(f"  AVERTISSEMENT: Pas de lignes dans Détail pour {type_prod_v2_current} à {jour_liv_current.strftime('%Y-%m-%d') if pd.notna(jour_liv_current) else 'Date Invalide'}. F_sim sera 0.")
        
        # --- Étape 1: Limite_Haute_Basse ---
        j_lhb, k_lhb, l_lhb, h_boost_py_lhb_final, \
        o_lim_final_lhb, p_lim_final_lhb, q_lim_final_lhb, \
        s_lim_final_lhb, t_lim_final_lhb, u_lim_final_lhb, \
        results_after_lhb = limite_haute_basse_python( # Appel à la version à 4 arguments principaux
            current_row_state_for_solver,        
            user_max_facteur_val,               
            row_data_orig_pdc_sim, # Pour lire les Limites Basse/Haute USER initiales de PDC_Sim
            type_prod_v2_current                
        )
        # h_boost_py_lhb_final est le h_boost_py qui est DANS current_row_state_for_solver['h_boost_py_current']
        # après l'exécution de LHB. Les JKL de départ et limites modifiées sont retournés.
        
        # Stocker les limites LHB modifiées pour les passer aux étapes suivantes
        # et pour l'output Excel.
        params_for_opt_et_solve = row_data_orig_pdc_sim.copy()
        params_for_opt_et_solve['LHB_O_lim'] = o_lim_final_lhb
        params_for_opt_et_solve['LHB_P_lim'] = p_lim_final_lhb
        params_for_opt_et_solve['LHB_Q_lim'] = q_lim_final_lhb
        params_for_opt_et_solve['LHB_S_lim'] = s_lim_final_lhb
        params_for_opt_et_solve['LHB_T_lim'] = t_lim_final_lhb
        params_for_opt_et_solve['LHB_U_lim'] = u_lim_final_lhb
        
        df_pdc_sim_results.loc[index, ['LHB_O_lim', 'LHB_P_lim', 'LHB_Q_lim', 
                                     'LHB_S_lim', 'LHB_T_lim', 'LHB_U_lim']] = \
            [o_lim_final_lhb, p_lim_final_lhb, q_lim_final_lhb, 
             s_lim_final_lhb, t_lim_final_lhb, u_lim_final_lhb]
        df_pdc_sim_results.loc[index, ['LHB_J_start', 'LHB_K_start', 'LHB_L_start']] = [j_lhb, k_lhb, l_lhb]

        # --- Étape 2: Optimisation (décision Solveur) ---
        needs_solver, type_lissage_py, \
        j_post_om, k_post_om, l_post_om, \
        h_boost_py_post_om, \
        results_om_decision = optimisation_macro_python( 
            # Les 5 arguments positionnels attendus par la définition que nous visons :
            j_lhb,                        # 1. j_start_optim_lhb
            k_lhb,                        # 2. k_start_optim_lhb
            l_lhb,                        # 3. l_start_optim_lhb
            h_boost_py_lhb_final,         # 4. h_boost_py_from_lhb (celui sorti de LHB)
            params_for_opt_et_solve       # 5. row_params_and_lhb_limits
                                          #    (contient Lims LHB modifiées et Min/Max Facteur User)
        )
        df_pdc_sim_results.loc[index, 'PY_TypeLissage'] = type_lissage_py
        
        final_j_py, final_k_py, final_l_py = j_post_om, k_post_om, l_post_om
        final_h_boost_py = h_boost_py_post_om # Le H_boost ne change plus après LHB dans ce flux
        final_results_struct = results_om_decision
        
        # --- Étape 3: Solveur (si nécessaire) ---
        if needs_solver:
            # Le solveur utilise current_row_state_for_solver (pour les données de base et h_boost_py_current)
            # Il a besoin des JKL de départ de LHB (j_lhb, k_lhb, l_lhb)
            # et des limites LHB modifiées (dans params_for_opt_et_solve)
            final_j_py, final_k_py, final_l_py, \
            final_h_boost_py, \
            final_results_struct = solver_macro_python(
                j_lhb, k_lhb, l_lhb, # JKL de départ pour le solveur (ceux de LHB)
                h_boost_py_lhb_final, # H_boost fixé par LHB 
                type_lissage_py, 
                params_for_opt_et_solve # Contient Lims LHB modifiées pour les bornes du solveur
            )

        df_pdc_sim_results.loc[index, 'PY_Opt_J'] = final_j_py
        df_pdc_sim_results.loc[index, 'PY_Opt_K'] = final_k_py
        df_pdc_sim_results.loc[index, 'PY_Opt_L'] = final_l_py
        df_pdc_sim_results.loc[index, 'PY_Opt_H_Boost'] = final_h_boost_py
        df_pdc_sim_results.loc[index, 'PY_F_Sim'] = final_results_struct.get('F_sim_total', np.nan)
        df_pdc_sim_results.loc[index, 'PY_I_Sim'] = final_results_struct.get('I_sim', np.nan)
        
        current_comment = str(df_pdc_sim_results.loc[index, 'PY_Comment_Optim']) 
        df_pdc_sim_results.loc[index, 'PY_Comment_Optim'] = "Optimisé" if pd.notna(final_results_struct.get('I_sim')) \
                                                                else (current_comment if current_comment != "nan" and current_comment != "" else "Erreur/Non traité")
        
        print(f"  Résultats Ligne {index}: PY_JKL=[{final_j_py:.3%},{final_k_py:.3%},{final_l_py:.3%}], PY_H_Boost={final_h_boost_py:.2%}, PY_TypeLissage={type_lissage_py}, PY_F_sim={final_results_struct.get('F_sim_total',0):.2f}, PY_I_sim={final_results_struct.get('I_sim',0):.2f}")

    try:
        df_pdc_sim_results.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
        print(f"\nOptimisation terminée. Résultats sauvegardés dans {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nErreur lors de la sauvegarde dans {OUTPUT_FILE}: {e}")