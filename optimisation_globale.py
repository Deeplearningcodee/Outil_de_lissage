import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution, NonlinearConstraint
import os
from datetime import datetime 
# Assuming MacroParam.py exists and is importable
import MacroParam 

# --- Fichier et Module de Paramètres ---
try:
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

# --- VBA-Style Top Product Prioritization ---
def check_top_product_priority(product_type):
    if not product_type or pd.isna(product_type):
        return False
    product_type = str(product_type).strip().upper()
    top_categories = [
        'TOP_500', 'TOP500', 'TOP 500',
        'TOP_3000', 'TOP3000', 'TOP 3000',
        'PRIORITY', 'PRIORITAIRE', 'HIGH_PRIORITY'
    ]
    return any(cat in product_type for cat in top_categories)

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
        try:
            print(f"  Tentative de lecture de {detail_filepath} comme Excel...")
            df_detail = pd.read_excel(detail_filepath)
            print(f"  {detail_filepath} chargé avec succès comme Excel.")
        except Exception as e_excel:
             raise ValueError(f"Erreur: Impossible de lire {detail_filepath} comme CSV ou Excel. Détails CSV: {e_csv}, Détails Excel: {e_excel}")

    print(f"DEBUG LoadData: Chargement de {pdc_perm_filepath} (PDC Perm)...")
    try:
        df_pdc_perm = pd.read_excel(pdc_perm_filepath, sheet_name="PDC", index_col=0)
        df_pdc_perm.index = pd.to_datetime(df_pdc_perm.index, errors='coerce')
        df_pdc_perm.dropna(axis=0, how='all', inplace=True) 
        df_pdc_perm.dropna(axis=1, how='all', inplace=True) 
        df_pdc_perm.columns = [str(col).strip() for col in df_pdc_perm.columns]
        print(f"  {pdc_perm_filepath} chargé. Index (dates): {df_pdc_perm.index.name}. Colonnes (types produits): {df_pdc_perm.columns.tolist()[:5]}...")
    except Exception as e_perm:
        raise ValueError(f"Erreur: Impossible de lire ou de traiter {pdc_perm_filepath}. Vérifiez le nom de la feuille et la structure. Détails: {e_perm}")

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
    df_pdc_sim['Type de produits'] = df_pdc_sim['Type de produits'].astype(str).str.strip() 
    df_pdc_sim['Jour livraison'] = pd.to_datetime(df_pdc_sim['Jour livraison'], errors='coerce', dayfirst=False)
    
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
    
    # Force float dtypes for columns that will receive optimized values
    optimization_output_cols = [
        'Top 500', 'Top 3000', 'Autre', 'Boost PDC',
        'Limite Basse Top 500', 'Limite Basse Top 3000', 'Limite Basse Autre',
        'Limite Haute Top 500', 'Limite Haute Top 3000', 'Limite Haute Autre',
        'Poids du A/C max'
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
            
            # Ensure float dtype for optimization output columns to avoid dtype warnings
            if col in optimization_output_cols:
                df_pdc_sim[col] = df_pdc_sim[col].astype('float64')
            
            if pd.api.types.is_numeric_dtype(df_pdc_sim[col]) and col in percentage_like_cols_pdc_sim:
                 if col not in ['PDC', 'En-cours', 'Commande SM à 100%', 'Tolérance']:
                    mask_large_percentage = (~df_pdc_sim[col].isnull()) & (df_pdc_sim[col].abs() > 5.0) 
                    if mask_large_percentage.any():
                         df_pdc_sim.loc[mask_large_percentage, col] /= 100.0
    
    if 'Tolérance' in df_pdc_sim.columns and df_pdc_sim['Tolérance'].isnull().any():
        mask_nan_tolerance = df_pdc_sim['Tolérance'].isnull()
        if 'PDC' in df_pdc_sim.columns and pd.api.types.is_numeric_dtype(df_pdc_sim['PDC']):
            df_pdc_sim.loc[mask_nan_tolerance, 'Tolérance'] = \
                df_pdc_sim.loc[mask_nan_tolerance, 'PDC'] * ALERTE_SURCHARGE_NIVEAU_1_DEFAULT
        else: 
            print("AVERTISSEMENT LoadData: Col 'PDC' non trouvée/numérique pour recalculer 'Tolérance' NaN dans df_pdc_sim.")

    required_detail_cols_from_source = { 
        'Type de produits V2': 'BM', 
        'DATE_LIVRAISON_V2': 'DATE_LIVRAISON_V2', 
        'Top': 'Top',
        'Borne Min Facteur multiplicatif lissage': 'Borne Min Facteur multiplicatif lissage',
        'Borne Max Facteur multiplicatif lissage': 'Borne Max Facteur multiplicatif lissage',
        'Commande Finale avec mini et arrondi SM à 100%': 'BP', 
        'Commande Finale avec mini et arrondi SM à 100% avec TS': 'BQ', 
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
            df_detail[original_name] = np.nan 
        if internal_name_key != original_name and internal_name_key not in df_detail.columns:
             if original_name in df_detail.columns : # only rename if original exists
                df_detail.rename(columns={original_name: internal_name_key}, inplace=True)


    if 'DATE_LIVRAISON_V2' in df_detail.columns:
        df_detail['DATE_LIVRAISON_V2'] = pd.to_datetime(df_detail['DATE_LIVRAISON_V2'], errors='coerce', dayfirst=True)
    else: df_detail['DATE_LIVRAISON_V2'] = pd.NaT
    
    if 'BM' not in df_detail.columns: 
        if 'Type de produits V2' in df_detail.columns: 
            df_detail.rename(columns={'Type de produits V2': 'BM'}, inplace=True)
        else:
            print("ERREUR CRITIQUE: 'BM' (ou 'Type de produits V2') non trouvé dans df_detail.")
            df_detail['BM'] = '' 
    df_detail['BM'] = df_detail['BM'].fillna('').astype(str).str.strip().str.lower()
    
    if 'Top' not in df_detail.columns: df_detail['Top'] = 'autre'
    else: df_detail['Top'] = df_detail['Top'].fillna('autre')
    df_detail['Top'] = df_detail['Top'].astype(str).str.strip().str.lower()
    
    if 'BQ' not in df_detail.columns: 
        if 'Commande Finale avec mini et arrondi SM à 100% avec TS' in df_detail.columns:
             df_detail.rename(columns={'Commande Finale avec mini et arrondi SM à 100% avec TS': 'BQ'}, inplace=True)
        else: # Check if it was already named 'BQ' by a previous rename attempt or if the original 'BQ' mapping was to itself
            if 'BQ' not in df_detail.columns: # if still not found
                print("ERREUR CRITIQUE: 'BQ' (ou nom long) non trouvé dans df_detail.")
                df_detail['BQ'] = 0
    df_detail['BQ'] = pd.to_numeric(df_detail['BQ'], errors='coerce').fillna(0)
        
    bp_col_name_internal = 'BP' # Already defined as the internal key
    if bp_col_name_internal not in df_detail.columns: 
        if 'Commande Finale avec mini et arrondi SM à 100%' in df_detail.columns:
             df_detail.rename(columns={'Commande Finale avec mini et arrondi SM à 100%': bp_col_name_internal}, inplace=True)
        else: # if still not found
             print(f"AVERTISSEMENT: Colonne '{bp_col_name_internal}' (ou 'Commande Finale avec mini et arrondi SM à 100%') non trouvée. Mise à 0.")
             df_detail[bp_col_name_internal] = 0
    df_detail[bp_col_name_internal] = pd.to_numeric(df_detail[bp_col_name_internal], errors='coerce').fillna(0)


    df_pdc_sim.dropna(subset=['Type de produits V2', 'Jour livraison', 'PDC', 'Tolérance'], inplace=True)
    df_detail.dropna(subset=['BM', 'DATE_LIVRAISON_V2', 'BQ', 'Top', bp_col_name_internal], inplace=True)
    
    print("DEBUG LoadData: Aperçu df_pdc_sim après toutes conversions (quelques colonnes clés):")
    print(df_pdc_sim[[c for c in required_pdc_sim_cols if c in df_pdc_sim.columns]].head(3).to_string())

    print("DEBUG LoadData: Aperçu df_detail (Détail.csv) après toutes conversions (quelques colonnes clés):")
    cols_to_show_detail = ['BM', 'DATE_LIVRAISON_V2', 'Top', 'BQ', bp_col_name_internal, 'SM Final', 'CODE_METI']
    print(df_detail[[c for c in cols_to_show_detail if c in df_detail.columns]].head(3).to_string())

    print("DEBUG LoadData: Aperçu df_pdc_perm (5 premières lignes):")
    print(df_pdc_perm.head().to_string())
    
    return df_pdc_sim, df_detail, df_pdc_perm

# --- Fonctions de Calcul et d'Optimisation ---
def recalculate_for_row(
    pdc_brut_perm_value,            
    commande_sm_100_ligne_sim,      
    en_cours_stock,                 
    df_detail_filtered,             
    j_factor_trial,                 
    k_factor_trial,                 
    l_factor_trial,                 
    h_boost_py_trial,               
    poids_ac_max_param_initial,     
    type_produit_v2_param_ligne,    
    context=""                      
):
    simulated_total_cmd_opt = 0.0
    if not df_detail_filtered.empty:
        log_ccdo_details_recalc = False
        MAX_DETAIL_LOG_LINES_RECALC = 0 
        if context and ( "LHB_CALC1" in context.upper() or "TESTING_CCDO" in context.upper() or "SOLVER_FINAL" in context.upper()): 
            log_ccdo_details_recalc = True

        for i_detail_recalc, (_, detail_row_as_series_recalc) in enumerate(df_detail_filtered.iterrows()):
            detail_row_dict_recalc = detail_row_as_series_recalc.to_dict()
            code_meti_detail_recalc = detail_row_dict_recalc.get('CODE_METI', detail_row_dict_recalc.get('Ean_13', f'idx_detail_{i_detail_recalc}'))
            
            cf_sim_detail_line = cf_main_module.get_cf_optimisee_for_detail_line( 
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

    denom_poids_ac = commande_sm_100_ligne_sim + en_cours_stock
    poids_ac_calcule_simule = 1.0 
    if denom_poids_ac != 0:
        poids_ac_calcule_simule = (f_sim_total + en_cours_stock) / denom_poids_ac
    
    poids_ac_final_pour_pdc = 1.0
    if isinstance(type_produit_v2_param_ligne, str) and type_produit_v2_param_ligne.lower().endswith("a/c"):
        poids_ac_final_pour_pdc = min(poids_ac_max_param_initial, poids_ac_calcule_simule)
        if poids_ac_final_pour_pdc > 1.0: poids_ac_final_pour_pdc = 1.0
        if poids_ac_final_pour_pdc < 0.0: poids_ac_final_pour_pdc = 0.0
    else: 
        poids_ac_final_pour_pdc = 1.0

    if pd.isna(pdc_brut_perm_value):
        pdc_brut_perm_value = 0.0 
        
    pdc_dynamique = pdc_brut_perm_value * poids_ac_final_pour_pdc
    pdc_target_adjusted_with_python_boost = pdc_dynamique * (1 + h_boost_py_trial)
    
    h_sim_diff = pdc_target_adjusted_with_python_boost - en_cours_stock - f_sim_total
    i_sim_abs_diff = abs(h_sim_diff)
    
    k_sim_var_pdc = h_sim_diff / pdc_dynamique if pdc_dynamique != 0 else 0.0
    l_sim_var_abs_pdc = abs(k_sim_var_pdc)
    
    if context and "RECALC_SUMMARY" in context.upper() or \
       (context and ("LHB_FinalState" in context.upper() or "Opt_DecisionWithLHB_JKL" in context.upper() or "Solver_Final" in context.upper() or "Opt_TestUserMaxFact" in context.upper() or "Opt_InitialWithLHB_JKL" in context.upper() )):
        print(f"      RECALC_SUMMARY ({context}): F_sim_Total={f_sim_total:.2f}, I_sim={i_sim_abs_diff:.2f}, K_var={k_sim_var_pdc:.2%}")

    return {
        'H_sim': h_sim_diff, 
        'I_sim': i_sim_abs_diff, 
        'K_sim_var_pdc': k_sim_var_pdc,
        'L_sim_var_abs_pdc': l_sim_var_abs_pdc, 
        'F_sim_total': f_sim_total
    }

def objective_minimize_scalar_j(j_val_trial): 
    state = current_row_state_for_solver 
    # For TypeLissage=1 (Hausse), K and L follow J
    k_val_trial = j_val_trial
    l_val_trial = j_val_trial 
    sim_res = recalculate_for_row(
        state['pdc_brut_perm_value'], state['commande_sm_100_ligne_sim'],
        state['en_cours_stock'], state['df_detail_filtered'], 
        j_val_trial, k_val_trial, l_val_trial, 
        state['h_boost_py_current'], 
        state['poids_ac_max_param_initial'], state['type_produit_v2_param_ligne'],
        "solver_scalar_obj_hausse" )
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
    current_state, 
    user_max_facteur_from_pdc_sim_row, 
    row_pdc_sim_params_for_limits, # Original PDC_Sim row for user limits
    type_produits_v2_debug_logging 
):
    """
    Python equivalent of VBA Limite_Haute_Basse() function
    Exactly replicates the VBA logic for setting limits and boost PDC
    """
    user_max_fact_str = f"{user_max_facteur_from_pdc_sim_row:.2%}" if pd.notna(user_max_facteur_from_pdc_sim_row) else 'NaN'
    print(f"    LHB_Python ({type_produits_v2_debug_logging}): Entrée user_max_fact={user_max_fact_str}")

    # VBA equivalent: Initialize all limits to 1, boost to 0, parameters to 1 (100%)
    o_lim, p_lim, q_lim = 1.0, 1.0, 1.0  # Limite Basse Top 500, Top 3000, Autre
    s_lim, t_lim, u_lim = 1.0, 1.0, 1.0  # Limite Haute Top 500, Top 3000, Autre
    
    work_j_param, work_k_param, work_l_param = 1.0, 1.0, 1.0 # Parameters Top 500, Top 3000, Autre
    current_state['h_boost_py_current'] = 0.0 # Boost PDC
    
    print(f"      LHB_Py - Initialisation VBA: Lims O,P,Q=1 S,T,U=1, Params J,K,L=1, H_boost=0")

    # VBA: Calculate with parameters at 100% (1.0)
    sim_res_calc1 = recalculate_for_row(
        current_state['pdc_brut_perm_value'], 
        current_state['commande_sm_100_ligne_sim'],
        current_state['en_cours_stock'], 
        current_state['df_detail_filtered'],
        work_j_param, work_k_param, work_l_param, # JKL = 1,1,1
        current_state['h_boost_py_current'], # H_boost = 0
        current_state['poids_ac_max_param_initial'], 
        current_state['type_produit_v2_param_ligne'],
        f"LHB_Calc1_JKL1_H0"
    )
    
    # VBA: Write margin adjustment by product type
    marge_manoeuvre = MARGE_POUR_BOOST_ET_L_VAR_SOLVER
    if sim_res_calc1['K_sim_var_pdc'] > marge_manoeuvre: 
        current_state['h_boost_py_current'] = -marge_manoeuvre
    elif sim_res_calc1['K_sim_var_pdc'] < -marge_manoeuvre: 
        current_state['h_boost_py_current'] = marge_manoeuvre
    
    print(f"      LHB_Py - Après K_var_check: K_var={sim_res_calc1['K_sim_var_pdc']:.2%}, H_boost={current_state['h_boost_py_current']:.2%}")
    
    # VBA: Calculate again with adjusted boost
    sim_res_calc2 = recalculate_for_row(
        current_state['pdc_brut_perm_value'], 
        current_state['commande_sm_100_ligne_sim'],
        current_state['en_cours_stock'], 
        current_state['df_detail_filtered'],
        work_j_param, work_k_param, work_l_param, # JKL still = 1,1,1
        current_state['h_boost_py_current'], # H_boost = new value
        current_state['poids_ac_max_param_initial'],
        current_state['type_produit_v2_param_ligne'],
        f"LHB_Calc2_WithBoost"
    )
    
    variation_relative = sim_res_calc2['K_sim_var_pdc']
    print(f"      LHB_Py - Après Boost: K_var={variation_relative:.2%}")
    
    # VBA: If not enough to reach PDC, set limits
    safe_user_max_facteur = float(user_max_facteur_from_pdc_sim_row) if pd.notna(user_max_facteur_from_pdc_sim_row) else 4.0
    
    # VBA: If Cells(i + DecalageParametreSimulation, ColonneResultatVariationRelative).Value > 0 Then
    if variation_relative > 0:
        print(f"      LHB_Py - K_var > 0: Setting upper limits to user max bounds")
        # VBA: .Range(.Cells(i, PremiereColonneLimiteBasse), .Cells(i, PremiereColonneLimiteBasse + 2)).Value = 1
        # VBA: .Range(.Cells(i, PremiereColonneLimiteHaute), .Cells(i, PremiereColonneLimiteHaute + 2)).Value = Cells(i, ColonneBorneMax).Value
        o_lim, p_lim, q_lim = 1.0, 1.0, 1.0 # Keep lower limits at 1
        s_lim, t_lim, u_lim = safe_user_max_facteur, safe_user_max_facteur, safe_user_max_facteur # Set upper limits to max
        
    else:
        print(f"      LHB_Py - K_var <= 0: Testing with different bounds and adjusting")
        # VBA: Else - test with different bounds and adjust
        # VBA: .Range(.Cells(i, PremiereColonneLimiteHaute), .Cells(i, PremiereColonneLimiteHaute + 2)).Value = 1
        s_lim, t_lim, u_lim = 1.0, 1.0, 1.0 # Set upper limits to 1
        
        # VBA: Test Q_lim = 0, L_param = 0 (Autre segment)
        # VBA: Cells(i, PremiereColonneLimiteBasse + 2).Value = 0
        # VBA: Cells(i, PremiereColonneParametrage + 2).Value = 0
        q_lim = 0.0  # Limite Basse Autre = 0
        work_l_param = 0.0  # Parameter Autre = 0
        
        sim_res_test1 = recalculate_for_row(
            current_state['pdc_brut_perm_value'], current_state['commande_sm_100_ligne_sim'],
            current_state['en_cours_stock'], current_state['df_detail_filtered'],
            work_j_param, work_k_param, work_l_param, # J=1, K=1, L=0
            current_state['h_boost_py_current'],
            current_state['poids_ac_max_param_initial'], current_state['type_produit_v2_param_ligne'],
            "LHB_TestQ0L0"
        )
        print(f"        LHB_Test1 (Q=0,L=0): K_var={sim_res_test1['K_sim_var_pdc']:.2%}")
        
        # VBA: If Cells(i + DecalageParametreSimulation, ColonneResultatVariationRelative).Value <= 0 Then
        if sim_res_test1['K_sim_var_pdc'] <= 0:
            # VBA: Test P_lim = 0, K_param = 0 (Top 3000 segment)
            # VBA: Cells(i, PremiereColonneLimiteBasse + 1).Value = 0
            # VBA: Cells(i, PremiereColonneParametrage + 1).Value = 0
            # VBA: Cells(i, PremiereColonneLimiteHaute + 2).Value = 0
            p_lim = 0.0  # Limite Basse Top 3000 = 0
            work_k_param = 0.0  # Parameter Top 3000 = 0
            u_lim = 0.0  # Limite Haute Autre = 0 (VBA line)
            
            sim_res_test2 = recalculate_for_row(
                current_state['pdc_brut_perm_value'], current_state['commande_sm_100_ligne_sim'],
                current_state['en_cours_stock'], current_state['df_detail_filtered'],
                work_j_param, work_k_param, work_l_param, # J=1, K=0, L=0
                current_state['h_boost_py_current'],
                current_state['poids_ac_max_param_initial'], current_state['type_produit_v2_param_ligne'],
                "LHB_TestP0K0"
            )
            print(f"        LHB_Test2 (P=0,K=0): K_var={sim_res_test2['K_sim_var_pdc']:.2%}")
            
            # VBA: If Cells(i + DecalageParametreSimulation, ColonneResultatVariationRelative).Value <= 0 Then
            if sim_res_test2['K_sim_var_pdc'] <= 0:
                # VBA: Test O_lim = 0 (Top 500 segment)
                # VBA: Cells(i, PremiereColonneLimiteBasse).Value = 0
                # VBA: Cells(i, PremiereColonneLimiteHaute + 1).Value = 0
                o_lim = 0.0  # Limite Basse Top 500 = 0
                t_lim = 0.0  # Limite Haute Top 3000 = 0 (VBA line)
                print(f"        LHB_Test3: Setting O=0, T=0")
    
    # Final calculation with determined parameters and limits
    final_sim_results_lhb = recalculate_for_row(
        current_state['pdc_brut_perm_value'], current_state['commande_sm_100_ligne_sim'],
        current_state['en_cours_stock'], current_state['df_detail_filtered'], 
        work_j_param, work_k_param, work_l_param, 
        current_state['h_boost_py_current'], 
        current_state['poids_ac_max_param_initial'], current_state['type_produit_v2_param_ligne'],
        "LHB_Final"
    )
    
    print(f"    LHB_Python ({type_produits_v2_debug_logging}): Sortie. JKL_params=[{work_j_param:.2f},{work_k_param:.2f},{work_l_param:.2f}], H_boost={current_state['h_boost_py_current']:.2%}")
    print(f"      Limites finales: O={o_lim:.2f},P={p_lim:.2f},Q={q_lim:.2f}, S={s_lim:.2f},T={t_lim:.2f},U={u_lim:.2f}")
    return (work_j_param, work_k_param, work_l_param, current_state['h_boost_py_current'], 
            o_lim, p_lim, q_lim, s_lim, t_lim, u_lim, 
            final_sim_results_lhb)


def optimisation_macro_python(
    j_start_optim_lhb, 
    k_start_optim_lhb, 
    l_start_optim_lhb, 
    h_boost_py_from_lhb, 
    row_params_and_lhb_limits # Contains original Min/Max Factor AND LHB modified O,P,Q,S,T,U
):
    """
    Python equivalent of VBA Optimisation() function
    Exactly replicates the VBA logic for determining TypeLissage and solver conditions
    """
    print(f"    Optimisation_Macro: Entrée JKL_start_LHB=[{j_start_optim_lhb:.2f},{k_start_optim_lhb:.2f},{l_start_optim_lhb:.2f}], H_py_de_LHB={h_boost_py_from_lhb:.2%}")
    
    # Get LHB modified limits
    lim_bas_j_lhb = float(row_params_and_lhb_limits['LHB_O_lim'])
    lim_bas_k_lhb = float(row_params_and_lhb_limits['LHB_P_lim'])
    lim_bas_l_lhb = float(row_params_and_lhb_limits['LHB_Q_lim'])
    lim_haut_j_lhb = float(row_params_and_lhb_limits['LHB_S_lim'])
    lim_haut_k_lhb = float(row_params_and_lhb_limits['LHB_T_lim'])
    lim_haut_l_lhb = float(row_params_and_lhb_limits['LHB_U_lim'])
    
    # Get user bounds
    user_min_facteur = float(row_params_and_lhb_limits['Min Facteur']) if pd.notna(row_params_and_lhb_limits['Min Facteur']) else 0.0
    user_max_facteur = float(row_params_and_lhb_limits['Max Facteur']) if pd.notna(row_params_and_lhb_limits['Max Facteur']) else 1.0

    print(f"      Opt - Limites LHB: J:[{lim_bas_j_lhb:.2f}-{lim_haut_j_lhb:.2f}], K:[{lim_bas_k_lhb:.2f}-{lim_haut_k_lhb:.2f}], L:[{lim_bas_l_lhb:.2f}-{lim_haut_l_lhb:.2f}]")
    print(f"      Opt - Bornes User: MinFact={user_min_facteur:.2%}, MaxFact={user_max_facteur:.2%}")

    # VBA: Determine TypeLissage based on whether (LimBasse_J * LimBasse_K * LimBasse_L) >= 1
    # VBA: If (Cells(i, PremiereColonneLimiteBasse) * Cells(i, PremiereColonneLimiteBasse + 1) * Cells(i, PremiereColonneLimiteBasse + 2)) >= 1 Then
    if (lim_bas_j_lhb * lim_bas_k_lhb * lim_bas_l_lhb >= 1.0): 
        type_lissage = 1  # Hausse
    else: 
        type_lissage = 0  # Baisse
    print(f"      Opt - TypeLissage: {'Hausse' if type_lissage == 1 else 'Baisse'} (Prod Lims Basses = {lim_bas_j_lhb * lim_bas_k_lhb * lim_bas_l_lhb:.2f})")

    current_j, current_k, current_l = j_start_optim_lhb, k_start_optim_lhb, l_start_optim_lhb

    # VBA: If BorneMin égale à BorneMax
    # VBA: If Cells(i, ColonneBorneMin) = Cells(i, ColonneBorneMax) Then
    if user_min_facteur == user_max_facteur:
        print(f"      Opt - BorneMin = BorneMax ({user_max_facteur:.2%}): Setting parameters to this value")
        # VBA: .Range(.Cells(i, PremiereColonneParametrage), .Cells(i, PremiereColonneParametrage + 2)).Value = Cells(i, ColonneBorneMax).Value
        current_j, current_k, current_l = user_max_facteur, user_max_facteur, user_max_facteur
        
        final_sim_results = recalculate_for_row(
            current_row_state_for_solver['pdc_brut_perm_value'], current_row_state_for_solver['commande_sm_100_ligne_sim'],
            current_row_state_for_solver['en_cours_stock'], current_row_state_for_solver['df_detail_filtered'],
            current_j, current_k, current_l, h_boost_py_from_lhb,
            current_row_state_for_solver['poids_ac_max_param_initial'], current_row_state_for_solver['type_produit_v2_param_ligne'],
            "Opt_BorneMinMaxEqual"
        )
        return False, type_lissage, current_j, current_k, current_l, h_boost_py_from_lhb, final_sim_results

    else: # user_min_facteur != user_max_facteur
        # VBA: Else ... Vérifier avec la borne max
        # VBA: .Range(.Cells(i, PremiereColonneParametrage), .Cells(i, PremiereColonneParametrage + 2)).Value = Cells(i, ColonneBorneMax).Value
        print(f"      Opt - Testing with BorneMax ({user_max_facteur:.2%})")
        j_test_max, k_test_max, l_test_max = user_max_facteur, user_max_facteur, user_max_facteur
        
        sim_results_test_max = recalculate_for_row(
            current_row_state_for_solver['pdc_brut_perm_value'], current_row_state_for_solver['commande_sm_100_ligne_sim'],
            current_row_state_for_solver['en_cours_stock'], current_row_state_for_solver['df_detail_filtered'],
            j_test_max, k_test_max, l_test_max, h_boost_py_from_lhb,
            current_row_state_for_solver['poids_ac_max_param_initial'], current_row_state_for_solver['type_produit_v2_param_ligne'],
            "Opt_TestBorneMax"
        )
        print(f"      Opt - Results with BorneMax: K_var={sim_results_test_max['K_sim_var_pdc']:.2%}, I_sim={sim_results_test_max['I_sim']:.2f}")

        # VBA: If Cells(i + DecalageParametreSimulation, ColonneResultatVariationRelative).Value <= 0 Then 
        if sim_results_test_max['K_sim_var_pdc'] <= 0: 
            print(f"      Opt - K_var_pdc <= 0 after BorneMax test. Reset to LHB values and check solver conditions")
            # VBA: .Range(...JKL...).Value = 1 (meaning, reset to start values, which are from LHB)
            current_j, current_k, current_l = j_start_optim_lhb, k_start_optim_lhb, l_start_optim_lhb
            
            # Recalculate with LHB values for solver decision
            sim_results_lhb = recalculate_for_row(
                current_row_state_for_solver['pdc_brut_perm_value'], current_row_state_for_solver['commande_sm_100_ligne_sim'],
                current_row_state_for_solver['en_cours_stock'], current_row_state_for_solver['df_detail_filtered'],
                current_j, current_k, current_l, h_boost_py_from_lhb,
                current_row_state_for_solver['poids_ac_max_param_initial'], current_row_state_for_solver['type_produit_v2_param_ligne'],
                "Opt_ResetToLHB"
            )
            
            # VBA solver conditions:
            # If Feuil8.Cells(i + DecalageParametreSimulation, ColonneResultatVariationAbsolue) > margeManoeuvre And Feuil8.Cells(i + DecalageParametreSimulation, ColonneResultatDifferenceAbsolue) > margeManoeuvreUVC Then
            marge_manoeuvre = MARGE_POUR_BOOST_ET_L_VAR_SOLVER
            marge_manoeuvre_uvc = MARGE_I_POUR_SOLVER_CONDITION
            
            condition_variation_abs = sim_results_lhb['L_sim_var_abs_pdc'] > marge_manoeuvre
            condition_difference_abs = sim_results_lhb['I_sim'] > marge_manoeuvre_uvc
            needs_solver = condition_variation_abs and condition_difference_abs

            if needs_solver:
                print(f"      Opt - Solver REQUIRED (L_var_abs={sim_results_lhb['L_sim_var_abs_pdc']:.2%} > {marge_manoeuvre:.2%} AND I_sim={sim_results_lhb['I_sim']:.2f} > {marge_manoeuvre_uvc:.2f})")
                return True, type_lissage, j_start_optim_lhb, k_start_optim_lhb, l_start_optim_lhb, h_boost_py_from_lhb, sim_results_lhb
            else:
                print(f"      Opt - Solver NOT required. Final JKL from LHB")
                return False, type_lissage, current_j, current_k, current_l, h_boost_py_from_lhb, sim_results_lhb

        else: # K_var_pdc > 0 after test with BorneMax
            print(f"      Opt - K_var_pdc > 0 after BorneMax test. Use BorneMax values as final")
            # In VBA, this path doesn't call solver, just uses the BorneMax values
            current_j, current_k, current_l = j_test_max, k_test_max, l_test_max
            return False, type_lissage, current_j, current_k, current_l, h_boost_py_from_lhb, sim_results_test_max


def vba_style_evolutionary_solver(
    j_start_solver, k_start_solver, l_start_solver,
    h_boost_py_for_solver, type_lissage, row_params_with_lhb_limits
):
    global current_row_state_for_solver
    current_row_state_for_solver['h_boost_py_current'] = h_boost_py_for_solver
    
    final_j, final_k, final_l = float(j_start_solver), float(k_start_solver), float(l_start_solver)
    print(f"    VBA_Solver: Entrée JKL_start=[{final_j:.3f},{final_k:.3f},{final_l:.3f}], H_py={h_boost_py_for_solver:.2%}, TypeLissage={type_lissage}")

    lim_bas_j = float(row_params_with_lhb_limits['LHB_O_lim'])
    lim_haut_j = float(row_params_with_lhb_limits['LHB_S_lim']) 
    lim_bas_k = float(row_params_with_lhb_limits['LHB_P_lim'])
    lim_haut_k = float(row_params_with_lhb_limits['LHB_T_lim'])
    lim_bas_l = float(row_params_with_lhb_limits['LHB_Q_lim'])
    lim_haut_l = float(row_params_with_lhb_limits['LHB_U_lim'])
    print(f"      VBA_Solver - Bornes LHB pour Solveur: J:[{lim_bas_j:.3f}-{lim_haut_j:.3f}], K:[{lim_bas_k:.3f}-{lim_haut_k:.3f}], L:[{lim_bas_l:.3f}-{lim_haut_l:.3f}]")

    if type_lissage == 1:  # Hausse
        print(f"      VBA_Solver - Mode Hausse: Optimize J only (K=J, L=J).")
        # VBA: SolverOk ByChange:="$" & LettrePremiereColonneParametrage & "$" & i (only J)
        # VBA: SolverAdd CellRef:="$" & LettrePremiereColonneParametrage & "$" & i, Relation:=1, FormulaText:="$" & LettrePremiereColonneLimiteHaute & "$" & i (J <= S_lim)
        # VBA: SolverAdd CellRef:="$" & LettrePremiereColonneParametrage & "$" & i, Relation:=3, FormulaText:="$" & LettrePremiereColonneLimiteBasse & "$" & i (J >= O_lim)
        # VBA: Cells(i, PremiereColonneParametrage + 1).FormulaR1C1 = "=RC[-1]" (K_param = J_param)
        # VBA: Cells(i, PremiereColonneParametrage + 2).FormulaR1C1 = "=RC[-1]" (L_param = K_param -> L_param = J_param)
        
        bounds_j_only = (lim_bas_j, lim_haut_j)
        if bounds_j_only[0] >= bounds_j_only[1]: # If J is fixed or invalid bounds
             print(f"        J bounds invalid/equal for Hausse. J fixed to {bounds_j_only[0]:.3f}")
             final_j = bounds_j_only[0]
        else:
            # objective_minimize_scalar_j already sets K=J, L=J internally for its calculation
            res = minimize_scalar(objective_minimize_scalar_j, bounds=bounds_j_only, method='bounded',
                                options={'xatol': 0.01}) 
            if res.success:
                final_j = res.x
                print(f"        Hausse optimization success: J_optimal={final_j:.4f}")
            else:
                print(f"        Hausse optimization failed, J remains start value {final_j:.4f}")
        
        # K and L follow J
        final_k = final_j
        final_l = final_j
        print(f"        JKL after Hausse opt: J={final_j:.4f}, K={final_k:.4f}, L={final_l:.4f}")

    else:  # Baisse (type_lissage == 0)
        print(f"      VBA_Solver - Mode Baisse: Full J,K,L optimization with J>=K>=L.")
        bounds_de = [(lim_bas_j, lim_haut_j), (lim_bas_k, lim_haut_k), (lim_bas_l, lim_haut_l)]
        
        if any(b[0] >= b[1] for b in bounds_de if not (b[0]==b[1] and b[0] == 0 and b[1] == 0 and final_j == 0 and final_k == 0 and final_l == 0)): # Allow JKL=0 if bounds are 0-0
            # Check if all are fixed at 0, which can be valid if LHB cascade set them so
            all_zero_fixed = all(b[0]==0 and b[1]==0 for b in bounds_de) and \
                             j_start_solver==0 and k_start_solver==0 and l_start_solver==0
            if not all_zero_fixed:
                print(f"        Invalid/fixed bounds detected for Baisse, using LHB start values or bound limits directly.")
                # Use LHB start values if bounds are problematic, ensuring they are within the (potentially collapsed) bounds
                final_j = min(max(j_start_solver, lim_bas_j), lim_haut_j)
                final_k = min(max(k_start_solver, lim_bas_k), lim_haut_k)
                final_l = min(max(l_start_solver, lim_bas_l), lim_haut_l)
            # If all_zero_fixed, final_j,k,l are already 0,0,0 from start
        else:
            def prioritization_constraint(jkl_arr): # J>=K, K>=L
                return np.array([jkl_arr[0] - jkl_arr[1], jkl_arr[1] - jkl_arr[2]])
            constraint_jkl_order = NonlinearConstraint(prioritization_constraint, 0, np.inf)
            
            vba_params = {
                'popsize': 100, 'mutation': (0.07, 0.08), 'recombination': 0.9,
                'tol': 0.05, 'atol': 0.01, 'maxiter': 50, 
                'disp': False, 'seed': 42, 'polish': True, 
                'updating': 'immediate', 'workers': 1 
            }
            constraints_list = [constraint_jkl_order] # VBA adds J>=K, K>=L for Baisse
            # The check_top_product_priority is a Python addition, not in base VBA Solver sub directly
            # For closer VBA emulation, we'd stick to the simple J>=K>=L from SolverAdd lines.

            x0 = [final_j, final_k, final_l]
            for i_clip in range(3): x0[i_clip] = min(max(x0[i_clip], bounds_de[i_clip][0]), bounds_de[i_clip][1])
            print(f"        VBA Baisse DE Stage 1: x0_clipped={x0}, popsize={vba_params['popsize']}, maxiter={vba_params['maxiter']}")
            
            res1 = differential_evolution(
                objective_differential_evolution_jkl, bounds_de,
                constraints=constraints_list, x0=x0, **vba_params
            )
            
            if res1.success:
                final_j, final_k, final_l = res1.x
                print(f"        VBA Baisse Stage 1 success: JKL=[{final_j:.4f},{final_k:.4f},{final_l:.4f}]")
                
                temp_res_s1 = recalculate_for_row(
                    current_row_state_for_solver['pdc_brut_perm_value'], current_row_state_for_solver['commande_sm_100_ligne_sim'],
                    current_row_state_for_solver['en_cours_stock'], current_row_state_for_solver['df_detail_filtered'],
                    final_j, final_k, final_l, h_boost_py_for_solver,
                    current_row_state_for_solver['poids_ac_max_param_initial'], current_row_state_for_solver['type_produit_v2_param_ligne'],
                    "VBA_Baisse_S1_Check"
                )
                # VBA: If Feuil8.Cells(i + DecalageParametreSimulation, ColonneResultatVariationAbsolue) > margeManoeuvre Then Call Solveur(60, i, TypeLissage)
                # Python: margeManoeuvre is MARGE_POUR_BOOST_ET_L_VAR_SOLVER
                if temp_res_s1['L_sim_var_abs_pdc'] > MARGE_POUR_BOOST_ET_L_VAR_SOLVER: # Check L_var_abs
                    print(f"        VBA Baisse Stage 2 needed (L_var_abs_S1={temp_res_s1['L_sim_var_abs_pdc']:.3%})")
                    vba_params_s2 = vba_params.copy(); vba_params_s2['maxiter'] = 100; vba_params_s2['seed'] = 43 
                    x0_s2 = [final_j, final_k, final_l]
                    res2 = differential_evolution(
                        objective_differential_evolution_jkl, bounds_de,
                        constraints=constraints_list, x0=x0_s2, **vba_params_s2
                    )
                    if res2.success and res2.fun < res1.fun:
                        final_j, final_k, final_l = res2.x
                        print(f"        VBA Baisse Stage 2 improved: JKL=[{final_j:.4f},{final_k:.4f},{final_l:.4f}]")
                    else: print(f"        VBA Baisse Stage 2 no improvement or failed, keeping Stage 1 result.")
                else: print(f"        VBA Baisse Stage 1 sufficient (L_var_abs_S1={temp_res_s1['L_sim_var_abs_pdc']:.3%})")
            else: print(f"        VBA Baisse Stage 1 failed, keeping input JKL values.")
        
        # Ensure J >= K >= L for Baisse mode, even if DE slightly violates it due to float precision or if it failed.
        # This is implicitly handled by DE constraints but good to enforce.
        current_j_val = float(final_j)
        current_k_val = min(float(final_k), current_j_val)
        current_l_val = min(float(final_l), current_k_val)
        if not (np.isclose(final_j, current_j_val) and np.isclose(final_k, current_k_val) and np.isclose(final_l, current_l_val)):
            print(f"      VBA_Solver Baisse - Enforcing J>=K>=L: Was [{final_j:.4f},{final_k:.4f},{final_l:.4f}], Now [{current_j_val:.4f},{current_k_val:.4f},{current_l_val:.4f}]")
            final_j, final_k, final_l = current_j_val, current_k_val, current_l_val


    final_sim_results = recalculate_for_row(
        current_row_state_for_solver['pdc_brut_perm_value'],
        current_row_state_for_solver['commande_sm_100_ligne_sim'], 
        current_row_state_for_solver['en_cours_stock'],
        current_row_state_for_solver['df_detail_filtered'],
        final_j, final_k, final_l, h_boost_py_for_solver,
        current_row_state_for_solver['poids_ac_max_param_initial'],
        current_row_state_for_solver['type_produit_v2_param_ligne'],
        "VBA_Final_Solver"
    )
    
    print(f"    VBA_Solver - Sortie: JKL=[{final_j:.4f},{final_k:.4f},{final_l:.4f}], I_sim={final_sim_results['I_sim']:.2f}")
    return final_j, final_k, final_l, h_boost_py_for_solver, final_sim_results


def solver_macro_python(
    j_start_solver, k_start_solver, l_start_solver, 
    h_boost_py_for_solver, type_lissage, 
    row_params_with_lhb_limits 
):
    return vba_style_evolutionary_solver(
        j_start_solver, k_start_solver, l_start_solver,
        h_boost_py_for_solver, type_lissage, row_params_with_lhb_limits
    )

# --- Boucle Principale ---
if __name__ == "__main__":
    # SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # If running in notebook, __file__ might not be defined. Use cwd or specify path.
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.getcwd() 
        print(f"__file__ not defined, using SCRIPT_DIR = {SCRIPT_DIR}")

    PDC_SIM_FILE_INPUT = os.path.join(SCRIPT_DIR, "PDC_Sim_Input_For_Optim.xlsx") 
    DETAIL_FILE_INPUT = os.path.join(SCRIPT_DIR, "initial_merged_predictions.csv")
    PDC_DIR = os.path.join(SCRIPT_DIR, "PDC") # Ensure this "PDC" subfolder exists if PDC.xlsx is inside it
    if not os.path.exists(PDC_DIR): PDC_DIR = SCRIPT_DIR # Fallback to SCRIPT_DIR if PDC subfolder not found
    PDC_PERM_FILE_INPUT = os.path.join(PDC_DIR, "PDC.xlsx") 
    OUTPUT_FILE = os.path.join(SCRIPT_DIR, "PDC_Sim_Optimized_Python_VBA_Aligned.xlsx")

    # Create dummy files if they don't exist for testing
    if not os.path.exists(PDC_SIM_FILE_INPUT):
        print(f"Création d'un fichier dummy: {PDC_SIM_FILE_INPUT}")
        # Create a more representative dummy PDC_Sim_Input_For_Optim.xlsx
        dummy_pdc_sim_data = {
            'Type de produits V2': ['sec méca - a/b', 'sec méca - a/c', 'sec homogène - a/b', 'frais méca'],
            'Type de produits': ['Sec Méca', 'Sec Méca', 'Sec Homogène', 'Frais Méca'],
            'Jour livraison': [datetime(2025,5,30), datetime(2025,5,31), datetime(2025,5,30), datetime(2025,5,31)],
            'PDC': [22673, 54018.40, 283, 53981],
            'En-cours': [6812, 1609, 96, 14119],
            'Commande SM à 100%': [33787.73, 32532.80, 276.25, 21068.45],
            'Tolérance': [1133.65, 2700.92, 14.15, 2699.05],
            'Poids du A/C max': [1.0, 0.80, 1.0, 1.0], # 80% for A/C type
            'Top 500': [1.0,1.0,1.0,1.0], 'Top 3000': [1.0,1.0,1.0,1.0], 'Autre': [1.0,1.0,1.0,1.0],
            'Limite Basse Top 500': [1.0,1.0,1.0,1.0],'Limite Basse Top 3000': [1.0,1.0,1.0,1.0],'Limite Basse Autre': [1.0,1.0,1.0,1.0],
            'Limite Haute Top 500': [1.0,1.0,1.0,1.0],'Limite Haute Top 3000': [1.0,1.0,1.0,1.0],'Limite Haute Autre': [1.0,1.0,1.0,1.0],
            'Min Facteur': [0.0, 0.0, 0.0, 0.0],
            'Max Facteur': [4.0, 4.0, 4.0, 1.0], # Example: Frais Meca has MaxFacteur = 1
            'Boost PDC': [0.0,0.0,0.0,0.0]
        }
        pd.DataFrame(dummy_pdc_sim_data).to_excel(PDC_SIM_FILE_INPUT, index=False)

    if not os.path.exists(DETAIL_FILE_INPUT):
        print(f"Création d'un fichier dummy: {DETAIL_FILE_INPUT}")
        # Create a more representative dummy initial_merged_predictions.csv
        dummy_detail_data = {
            'Type de produits V2': ['sec méca - a/b', 'sec méca - a/b', 'sec méca - a/c', 'sec méca - a/c', 'sec homogène - a/b', 'frais méca', 'frais méca'],
            'DATE_LIVRAISON_V2': ['30/05/2025', '30/05/2025', '31/05/2025', '31/05/2025', '30/05/2025', '31/05/2025', '31/05/2025'],
            'Top': ['top 500', 'autre', 'top 3000', 'autre', 'autre', 'top 500', 'autre'],
            'Borne Min Facteur multiplicatif lissage': [0.1,0.1,0.1,0.1,0.1,0.1,0.1],
            'Borne Max Facteur multiplicatif lissage': [4.0,4.0,4.0,4.0,4.0,1.5,1.5],
            'Commande Finale avec mini et arrondi SM à 100%': [100, 50, 200, 30, 10, 1000, 500], # BP
            'Commande Finale avec mini et arrondi SM à 100% avec TS': [110, 55, 220, 33, 11, 1100, 550], # BQ
            'Mini Publication FL': [10,5,10,5,2,20,10], 'COCHE_RAO': [1,0,1,0,0,1,1], 'STOCK_REEL': [20,10,30,5,3,100,50],
            'RAL': [5,2,5,1,1,10,5], 'SM Final': [15,8,20,12,5,50,20], 
            'Prév C1-L2 Finale': [120,60,250,40,15,1200,600], 'Prév L1-L2 Finale': [115,58,240,38,14,1150,580],
            'Facteur Multiplicatif Appro': [1.0,1.0,1.0,1.0,1.0,1.0,1.0],
            'Casse Prev C1-L2': [0,0,0,0,0,0,0], 'Casse Prev L1-L2': [0,0,0,0,0,0,0],
            'Produit Bloqué': [0,0,0,0,0,0,0], 'Commande Max avec stock max': [1000,500,2000,300,100,5000,2000],
            'Position JUD': ['A','B','A','C','B','A','A'], 'MINIMUM_COMMANDE': [1,1,1,1,1,1,1], 'PCB': [1,1,1,1,1,1,1], 'TS': [10,5,20,3,1,100,50],
            'CODE_METI': [f'C{i}' for i in range(1,8)], 'Ean_13': [f'E{i}' for i in range(1,8)]
        }
        pd.DataFrame(dummy_detail_data).to_csv(DETAIL_FILE_INPUT, sep=';', index=False, encoding='latin1')

    if not os.path.exists(PDC_PERM_FILE_INPUT):
        print(f"Création d'un fichier dummy: {PDC_PERM_FILE_INPUT}")
        # Create a dummy PDC.xlsx
        pdc_perm_dates = pd.to_datetime([datetime(2025,5,30), datetime(2025,5,31)])
        dummy_pdc_perm_data = {
            'Sec Méca': [22673, 54018.40],
            'Sec Homogène': [283, 3742.40],
            'Sec Hétérogène': [19735, 26088.80],
            'Frais Méca': [53981, 53981], # Example value
            'Frais Manuel': [14577, 14577],
            'Surgelés': [9095, 9095]
        }
        df_dummy_pdc_perm = pd.DataFrame(dummy_pdc_perm_data, index=pdc_perm_dates)
        # Ensure the subfolder "PDC" exists if PDC_DIR points to it
        if PDC_DIR != SCRIPT_DIR and not os.path.exists(PDC_DIR):
            os.makedirs(PDC_DIR)
        df_dummy_pdc_perm.to_excel(PDC_PERM_FILE_INPUT, sheet_name="PDC")


    if not (os.path.exists(PDC_SIM_FILE_INPUT) and \
            os.path.exists(DETAIL_FILE_INPUT) and \
            os.path.exists(PDC_PERM_FILE_INPUT)):
        print(f"Erreur: Un des fichiers d'entrée est non trouvé.")
        print(f"  PDC_Sim_Input_For_Optim.xlsx: {os.path.exists(PDC_SIM_FILE_INPUT)} (Chemin: {PDC_SIM_FILE_INPUT})")
        print(f"  initial_merged_predictions.csv: {os.path.exists(DETAIL_FILE_INPUT)} (Chemin: {DETAIL_FILE_INPUT})")
        print(f"  PDC.xlsx: {os.path.exists(PDC_PERM_FILE_INPUT)} (Chemin: {PDC_PERM_FILE_INPUT})")
        exit()

    print("--- DÉBUT DU CHARGEMENT DES DONNÉES ---")
    df_pdc_sim_input, df_detail, df_pdc_perm = load_data(
        PDC_SIM_FILE_INPUT, DETAIL_FILE_INPUT, PDC_PERM_FILE_INPUT
    )
    print("--- FIN DU CHARGEMENT DES DONNÉES ---")
    
    df_pdc_sim_results = df_pdc_sim_input.copy()
    cols_python_results = [
        'PY_Opt_J', 'PY_Opt_K', 'PY_Opt_L', 'PY_Opt_H_Boost', 
        'PY_F_Sim', 'PY_I_Sim', 'PY_K_VarPDC', 'PY_TypeLissage', 'PY_Comment_Optim',
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
        type_prod_v2_current = current_row_params_for_macros['Type de produits V2'] 
        jour_liv_current = current_row_params_for_macros['Jour livraison']
        type_produit_current = current_row_params_for_macros['Type de produits'] 

        if pd.isna(jour_liv_current) or pd.isna(type_prod_v2_current) or pd.isna(current_row_params_for_macros['PDC']):
            df_pdc_sim_results.loc[index, 'PY_Comment_Optim'] = "Données clé manquantes (V2, Jour, PDC)"; continue
        
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

        commande_sm_100_val = float(current_row_params_for_macros['Commande SM à 100%']) if pd.notna(current_row_params_for_macros['Commande SM à 100%']) else 0.0
        en_cours_val = float(current_row_params_for_macros['En-cours']) if pd.notna(current_row_params_for_macros['En-cours']) else 0.0
        poids_ac_max_initial_val = float(current_row_params_for_macros['Poids du A/C max']) if pd.notna(current_row_params_for_macros['Poids du A/C max']) else 1.0
        
        df_detail_filt_current = df_detail[(df_detail['BM'] == type_prod_v2_current) & \
                                     (df_detail['DATE_LIVRAISON_V2'] == jour_liv_current)].copy()
        
        current_row_state_for_solver = {
            'pdc_brut_perm_value': pdc_brut_val, 
            'commande_sm_100_ligne_sim': commande_sm_100_val,
            'en_cours_stock': en_cours_val, 
            'df_detail_filtered': df_detail_filt_current, 
            'poids_ac_max_param_initial': poids_ac_max_initial_val,
            'type_produit_v2_param_ligne': type_prod_v2_current, 
            'h_boost_py_current': 0.0 
        }
        user_max_facteur_val = float(row_data_orig_pdc_sim['Max Facteur']) if pd.notna(row_data_orig_pdc_sim['Max Facteur']) else 1.0
        
        if df_detail_filt_current.empty and pd.notna(current_row_params_for_macros['Commande SM à 100%']) and current_row_params_for_macros['Commande SM à 100%'] > 0 :
            print(f"  AVERTISSEMENT: Pas de lignes dans Détail pour {type_prod_v2_current} à {jour_liv_current.strftime('%Y-%m-%d') if pd.notna(jour_liv_current) else 'Date Invalide'}. F_sim sera 0.")
        
        j_lhb, k_lhb, l_lhb, h_boost_py_lhb_final, \
        o_lim_final_lhb, p_lim_final_lhb, q_lim_final_lhb, \
        s_lim_final_lhb, t_lim_final_lhb, u_lim_final_lhb, \
        results_after_lhb = limite_haute_basse_python( 
            current_row_state_for_solver,        
            user_max_facteur_val,               
            row_data_orig_pdc_sim, 
            type_prod_v2_current                
        )
        
        params_for_opt_et_solve = row_data_orig_pdc_sim.copy() # Start with original params
        # Update with LHB modified limits
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
        
        # CRITICAL FIX: Update the original DataFrame with calculated limits so they appear in Excel output
        df_pdc_sim_input.loc[index, 'Limite Basse Top 500'] = float(o_lim_final_lhb)
        df_pdc_sim_input.loc[index, 'Limite Basse Top 3000'] = float(p_lim_final_lhb)
        df_pdc_sim_input.loc[index, 'Limite Basse Autre'] = float(q_lim_final_lhb)
        df_pdc_sim_input.loc[index, 'Limite Haute Top 500'] = float(s_lim_final_lhb)
        df_pdc_sim_input.loc[index, 'Limite Haute Top 3000'] = float(t_lim_final_lhb)
        df_pdc_sim_input.loc[index, 'Limite Haute Autre'] = float(u_lim_final_lhb)

        needs_solver, type_lissage_py, \
        j_post_om, k_post_om, l_post_om, \
        h_boost_py_post_om, \
        results_om_decision = optimisation_macro_python( 
            j_lhb, k_lhb, l_lhb,
            h_boost_py_lhb_final,  # This is current_row_state_for_solver['h_boost_py_current'] from LHB
            params_for_opt_et_solve # Contains original Min/Max Factor AND LHB modified limits      
        )
        df_pdc_sim_results.loc[index, 'PY_TypeLissage'] = type_lissage_py
        
        final_j_py, final_k_py, final_l_py = j_post_om, k_post_om, l_post_om
        final_h_boost_py = h_boost_py_post_om 
        final_results_struct = results_om_decision
        
        if needs_solver:
            final_j_py, final_k_py, final_l_py, \
            final_h_boost_py, \
            final_results_struct = solver_macro_python(
                j_post_om, k_post_om, l_post_om, # JKL to start solver (from OptimMacro decision logic)
                final_h_boost_py, # H_boost (already in current_row_state and passed along)
                type_lissage_py, 
                params_for_opt_et_solve # Contains LHB modified limits for solver bounds
            )

        df_pdc_sim_results.loc[index, 'PY_Opt_J'] = final_j_py
        df_pdc_sim_results.loc[index, 'PY_Opt_K'] = final_k_py
        df_pdc_sim_results.loc[index, 'PY_Opt_L'] = final_l_py
        df_pdc_sim_results.loc[index, 'PY_Opt_H_Boost'] = final_h_boost_py
        df_pdc_sim_results.loc[index, 'PY_F_Sim'] = final_results_struct.get('F_sim_total', np.nan)
        df_pdc_sim_results.loc[index, 'PY_I_Sim'] = final_results_struct.get('I_sim', np.nan)
        df_pdc_sim_results.loc[index, 'PY_K_VarPDC'] = final_results_struct.get('K_sim_var_pdc', np.nan)

        # *** CRITICAL VBA MATCH: Update Top 500, Top 3000, Autre with final optimized J,K,L values ***
        # In VBA, the optimized J,K,L parameters become the final Top 500, Top 3000, Autre values
        df_pdc_sim_input.loc[index, 'Top 500'] = float(final_j_py)
        df_pdc_sim_input.loc[index, 'Top 3000'] = float(final_k_py)  
        df_pdc_sim_input.loc[index, 'Autre'] = float(final_l_py)
        df_pdc_sim_input.loc[index, 'Boost PDC'] = float(final_h_boost_py)
        
        # *** CRITICAL VBA MATCH: Recalculate "Poids du A/C max" based on optimized commands ***
        # In VBA, this is recalculated after optimization, not used as a fixed input
        commande_optimisee_finale = final_results_struct.get('F_sim_total', 0)
        en_cours_actuel = df_pdc_sim_input.loc[index, 'En-cours']
        commande_sm_100_actuel = df_pdc_sim_input.loc[index, 'Commande SM à 100%']
        
        # Calculate new "Poids du A/C max" based on optimized results
        denominateur_poids_ac = commande_sm_100_actuel + en_cours_actuel
        if denominateur_poids_ac != 0:
            poids_ac_recalcule = commande_optimisee_finale / denominateur_poids_ac
        else:
            poids_ac_recalcule = 1.0
            
        # For A/C products, use the recalculated value; for A/B products, keep at 1.0
        if isinstance(type_prod_v2_current, str) and type_prod_v2_current.lower().endswith("a/c"):
            df_pdc_sim_input.loc[index, 'Poids du A/C max'] = float(poids_ac_recalcule)
        else:
            df_pdc_sim_input.loc[index, 'Poids du A/C max'] = 1.0
            
        # Update "Commande optimisée" with the final result
        df_pdc_sim_input.loc[index, 'Commande optimisée'] = float(commande_optimisee_finale)
        
        # *** CRITICAL VBA MATCH: Calculate additional output columns based on optimization results ***
        pdc_actuel = df_pdc_sim_input.loc[index, 'PDC']
        
        # Calculate "Différence PDC / Commande" = PDC - Commande optimisée  
        difference_pdc_commande = pdc_actuel - commande_optimisee_finale
        df_pdc_sim_input.loc[index, 'Différence PDC / Commande'] = float(difference_pdc_commande)
        
        # Calculate "Différence absolue" = abs(Différence PDC / Commande)
        df_pdc_sim_input.loc[index, 'Différence absolue'] = float(abs(difference_pdc_commande))
        
        # Calculate "Variation PDC" = (PDC - En-cours - Commande optimisée) / PDC
        if pdc_actuel != 0:
            variation_pdc = (pdc_actuel - en_cours_actuel - commande_optimisee_finale) / pdc_actuel
        else:
            variation_pdc = 0.0
        df_pdc_sim_input.loc[index, 'Variation PDC'] = float(variation_pdc)
        
        # Calculate "Variation absolue PDC" = abs(Variation PDC)
        df_pdc_sim_input.loc[index, 'Variation absolue PDC'] = float(abs(variation_pdc))
        
        # Set "Capage Borne Max ?" based on whether the result hit the maximum bounds
        # This would be "Oui" if the final J,K,L values are at their upper limits
        user_max_fact = df_pdc_sim_input.loc[index, 'Max Facteur']
        hit_upper_bound = (abs(final_j_py - user_max_fact) < 0.001 or 
                          abs(final_k_py - user_max_fact) < 0.001 or 
                          abs(final_l_py - user_max_fact) < 0.001)
        df_pdc_sim_input.loc[index, 'Capage Borne Max ?'] = "Oui" if hit_upper_bound else "Non"
        
        # Calculate "Moyenne" = (Top 500 + Top 3000 + Autre) / 3
        moyenne_jkl = (final_j_py + final_k_py + final_l_py) / 3.0
        df_pdc_sim_input.loc[index, 'Moyenne'] = float(moyenne_jkl)

        
        current_comment = str(df_pdc_sim_results.loc[index, 'PY_Comment_Optim']) 
        df_pdc_sim_results.loc[index, 'PY_Comment_Optim'] = "Optimisé" if pd.notna(final_results_struct.get('I_sim')) \
                                                                else (current_comment if current_comment != "nan" and current_comment != "" else "Erreur/Non traité")
        
        print(f"  Résultats Ligne {index}: PY_JKL=[{final_j_py:.3f},{final_k_py:.3f},{final_l_py:.3f}], PY_H_Boost={final_h_boost_py:.2%}, PY_TypeLissage={type_lissage_py}, PY_F_sim={final_results_struct.get('F_sim_total',0):.2f}, PY_I_sim={final_results_struct.get('I_sim',0):.2f}, PY_K_var={final_results_struct.get('K_sim_var_pdc',0):.2%}")

    try:
        # Write the original DataFrame with updated limits to Excel
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
            df_pdc_sim_input.to_excel(writer, index=False)
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            
            # Define percentage columns that should be formatted as percentages in Excel
            percentage_columns = ['Top 500', 'Top 3000', 'Autre', 'Boost PDC', 
                                'Min Facteur', 'Max Facteur', 'Poids du A/C max',
                                'Limite Basse Top 500', 'Limite Basse Top 3000', 'Limite Basse Autre',
                                'Limite Haute Top 500', 'Limite Haute Top 3000', 'Limite Haute Autre',
                                'Variation PDC', 'Variation absolue PDC', 'Moyenne']
            
            # Apply percentage formatting to the specified columns
            for col_name in percentage_columns:
                if col_name in df_pdc_sim_input.columns:
                    col_idx = df_pdc_sim_input.columns.get_loc(col_name) + 1  # Excel is 1-indexed
                    # Format the entire column as percentage with 2 decimal places
                    for row in range(2, len(df_pdc_sim_input) + 2):  # Skip header row
                        cell = worksheet.cell(row=row, column=col_idx)
                        cell.number_format = '0.00%'
        
        print(f"\nOptimisation terminée. Résultats sauvegardés dans {OUTPUT_FILE}")
        
        # Also save the detailed results for debugging
        RESULTS_DEBUG_FILE = OUTPUT_FILE.replace('.xlsx', '_Debug_Results.xlsx')
        df_pdc_sim_results.to_excel(RESULTS_DEBUG_FILE, index=False, engine='openpyxl')
        print(f"Résultats détaillés de debug sauvegardés dans {RESULTS_DEBUG_FILE}")
    except Exception as e:
        print(f"\nErreur lors de la sauvegarde dans {OUTPUT_FILE}: {e}")