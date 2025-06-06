#!/usr/bin/env python3
"""
Correct VBA logic implementation based on actual Excel formulas
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
import MacroParam

def calculate_by_facteur_lissage(detail_row, optimization_params):
    """
    Calculate BY (Facteur multiplicatif Lissage besoin brut) based on Excel formula:
    =MAX(BW2;MIN(BX2;SIERREUR(INDEX(Optimisation!$J$4:$L$15;EQUIV(BM2;Optimisation!$B$4:$B$15;0);EQUIV(BN2;Optimisation!$J$3:$L$3;0));1)))
    """
    # Get bounds
    bw_min = pd.to_numeric(detail_row.get('Borne Min Facteur multiplicatif lissage', 0), errors='coerce') or 0.0
    bx_max = pd.to_numeric(detail_row.get('Borne Max Facteur multiplicatif lissage', 10), errors='coerce') or 10.0
    
    # Get lookup values
    type_produit_v2 = str(detail_row.get('Type de produits V2', '')).strip()
    top_category = str(detail_row.get('Top', '')).strip().lower()
    
    # Map top category to column
    top_mapping = {
        'top 500': 'Top 500',
        'top 3000': 'Top 3000', 
        'autre': 'Autre'
    }
    
    factor_column = top_mapping.get(top_category, 'Autre')
    
    # Look up factor from optimization parameters
    try:
        matching_rows = optimization_params[optimization_params['Type de produits V2'] == type_produit_v2]
        if not matching_rows.empty:
            factor = pd.to_numeric(matching_rows.iloc[0].get(factor_column, 1.0), errors='coerce') or 1.0
        else:
            factor = 1.0
    except:
        factor = 1.0
    
    # Apply bounds: MAX(min, MIN(max, factor))
    result = max(bw_min, min(bx_max, factor))
    return result

def calculate_cd_commande_sans_arrondi(detail_row, by_factor):
    """
    Calculate CD (Commande optimisée sans arrondi) based on Excel formula:
    =SI(ET(BP2=0;BY2<=1);0;MAX(SI(OU(AK2<=0;AD2=46);0;AK2-SI(AJ2="";0;AJ2)-SI(Z2="";0;Z2));MIN(CC2;SI(BI2;0;MIN(SI(ESTERREUR($P2);0;MAX(0;(O2+BF2)*BU2*BY2+SI('Macro-Param'!$C$16="Oui";-(AJ2-AY2)-Z2;-AJ2-Z2)));SI(ESTERREUR($P2);0;MAX(0;(O2+BG2)*BU2*BY2+SI('Macro-Param'!$C$16="Oui";AZ2;0))))))))
    """
    # Get values with proper handling
    bp_cmd_sm100 = pd.to_numeric(detail_row.get('Commande Finale avec mini et arrondi SM à 100%', 0), errors='coerce') or 0.0
    
    # Initial condition: SI(ET(BP2=0;BY2<=1);0;...)
    if bp_cmd_sm100 == 0 and by_factor <= 1:
        return 0.0
    
    # Get other values
    ak_mini_pub = pd.to_numeric(detail_row.get('Mini Publication FL', 0), errors='coerce') or 0.0
    ad_coche_rao = pd.to_numeric(detail_row.get('COCHE_RAO', 0), errors='coerce') or 0.0
    aj_stock_reel = pd.to_numeric(detail_row.get('STOCK_REEL', 0), errors='coerce') or 0.0
    z_ral = pd.to_numeric(detail_row.get('RAL', 0), errors='coerce') or 0.0
    
    o_sm_final = pd.to_numeric(detail_row.get('SM Final', 0), errors='coerce') or 0.0
    bf_prev_c1_l2 = pd.to_numeric(detail_row.get('Prév C1-L2 Finale', 0), errors='coerce') or 0.0
    bg_prev_l1_l2 = pd.to_numeric(detail_row.get('Prév L1-L2 Finale', 0), errors='coerce') or 0.0
    bu_facteur_appro = pd.to_numeric(detail_row.get('Facteur Multiplicatif Appro', 1), errors='coerce') or 1.0
    
    ay_casse_c1_l2 = pd.to_numeric(detail_row.get('Casse Prev C1-L2', 0), errors='coerce') or 0.0
    az_casse_l1_l2 = pd.to_numeric(detail_row.get('Casse Prev L1-L2', 0), errors='coerce') or 0.0
    cc_cmd_max = pd.to_numeric(detail_row.get('Commande Max avec stock max', 99999999), errors='coerce') or 99999999.0
    
    bi_produit_bloque = detail_row.get('Produit Bloqué', False)
    if isinstance(bi_produit_bloque, str):
        bi_produit_bloque = bi_produit_bloque.lower() in ['true', 'oui', 'yes', '1']
    
    # Position JUD error check
    position_jud_val = detail_row.get('Position JUD', None)
    p_position_jud_is_error = pd.isna(pd.to_numeric(position_jud_val, errors='coerce'))
    
    # Get casse parameter
    casse_prev_active = MacroParam.get_param_value('Casse Prev Activé', 'Oui') == "Oui"
    
    # Calculate besoin_net: SI(OU(AK2<=0;AD2=46);0;AK2-SI(AJ2="";0;AJ2)-SI(Z2="";0;Z2))
    if ak_mini_pub <= 0 or ad_coche_rao == 46:
        besoin_net = 0.0
    else:
        besoin_net = ak_mini_pub - aj_stock_reel - z_ral
    
    # Calculate stock scenarios
    if bi_produit_bloque:
        quantite_stock_max_scenarios = 0.0
    else:
        # Scenario 1: SI(ESTERREUR($P2);0;MAX(0;(O2+BF2)*BU2*BY2+SI('Macro-Param'!$C$16="Oui";-(AJ2-AY2)-Z2;-AJ2-Z2)))
        if p_position_jud_is_error:
            scenario1 = 0.0
        else:
            if casse_prev_active:
                ajustement1 = -(aj_stock_reel - ay_casse_c1_l2) - z_ral
            else:
                ajustement1 = -aj_stock_reel - z_ral
            scenario1 = max(0.0, (o_sm_final + bf_prev_c1_l2) * bu_facteur_appro * by_factor + ajustement1)
        
        # Scenario 2: SI(ESTERREUR($P2);0;MAX(0;(O2+BG2)*BU2*BY2+SI('Macro-Param'!$C$16="Oui";AZ2;0)))
        if p_position_jud_is_error:
            scenario2 = 0.0
        else:
            if casse_prev_active:
                ajustement2 = az_casse_l1_l2
            else:
                ajustement2 = 0.0
            scenario2 = max(0.0, (o_sm_final + bg_prev_l1_l2) * bu_facteur_appro * by_factor + ajustement2)
        
        quantite_stock_max_scenarios = min(scenario1, scenario2)
    
    # MIN(CC2, quantite_stock_max_scenarios)
    valeur_avant_max = min(cc_cmd_max, quantite_stock_max_scenarios)
    
    # Final MAX(besoin_net, valeur_avant_max)
    result = max(besoin_net, valeur_avant_max)
    return max(0.0, result)

def calculate_ce_commande_avec_arrondi(detail_row, cd_value):
    """
    Calculate CE (Commande optimisée avec arrondi et mini) based on complex rounding formula
    """
    if pd.isna(cd_value) or cd_value == 0:
        return 0.0
    
    # Get PCB and minimum command values
    v_min_cmd = pd.to_numeric(detail_row.get('MINIMUM_COMMANDE', 0), errors='coerce') or 0.0
    w_pcb = pd.to_numeric(detail_row.get('PCB', 1), errors='coerce') or 1.0
    
    # Get product type for threshold lookup
    type_produit_v2 = str(detail_row.get('Type de produits V2', '')).strip()
    
    # Get thresholds (simplified - using defaults from MacroParam)
    try:
        from MacroParam import get_arrondi_pcb_seuils
        seuil1, seuil2 = get_arrondi_pcb_seuils(type_produit_v2)
    except:
        seuil1, seuil2 = 0.01, 0.5  # Default values
    
    # Apply rounding logic as per Excel formula
    diviseur_seuil1 = max(v_min_cmd, w_pcb)
    if diviseur_seuil1 == 0:
        diviseur_seuil1 = 1.0
    
    if cd_value / diviseur_seuil1 < seuil1:
        return 0.0
    
    # Complex rounding logic
    cd_div_pcb = cd_value / w_pcb
    
    if cd_div_pcb >= 1:
        partie_decimale = cd_div_pcb - np.floor(cd_div_pcb)
        if partie_decimale < seuil2:
            val_arrondie = np.floor(cd_div_pcb) * w_pcb
        else:
            val_arrondie = np.ceil(cd_div_pcb) * w_pcb
    else:
        if v_min_cmd > w_pcb:
            cd_div_min = cd_value / v_min_cmd
            partie_decimale_v = cd_div_min - np.floor(cd_div_min)
            if partie_decimale_v < seuil1:
                val_arrondie = np.floor(cd_div_min) * v_min_cmd
            else:
                val_arrondie = np.ceil(cd_div_min) * v_min_cmd
        else:
            val_arrondie = np.ceil(cd_div_pcb) * w_pcb
    
    return max(v_min_cmd, val_arrondie)

def calculate_cf_commande_avec_ts(detail_row, ce_value):
    """
    Calculate CF (Commande optimisée avec arrondi et mini et TS) = CE * TS
    """
    if pd.isna(ce_value):
        return 0.0
    
    ts_value = pd.to_numeric(detail_row.get('TS', 1), errors='coerce') or 1.0
    return ce_value * ts_value

def calculate_total_commande_optimisee(df_detail, optimization_params, type_produit_v2, date_livraison):
    """
    Calculate total 'Commande optimisée' for a given product type and delivery date
    This simulates the Excel SUMIFS formula but uses the existing CommandeFinale module
    """
    # Import the existing module that correctly implements Excel formulas
    import CommandeFinale as cf_main_module
    
    # Filter detail data (case insensitive for product type)
    mask = (df_detail['Type de produits V2'].str.lower() == type_produit_v2.lower()) & \
           (pd.to_datetime(df_detail['DATE_LIVRAISON_V2']).dt.normalize() == pd.to_datetime(date_livraison).normalize())
    
    filtered_detail = df_detail[mask].copy()
    
    if filtered_detail.empty:
        return 0.0
    
    # Prepare data for CommandeFinale module (add required column mappings)
    if 'BM' not in filtered_detail.columns and 'Type de produits V2' in filtered_detail.columns:
        filtered_detail['BM'] = filtered_detail['Type de produits V2']
    
    # Update BY factors based on current optimization parameters
    for idx, row in filtered_detail.iterrows():
        by_factor = calculate_by_facteur_lissage(row, optimization_params)
        filtered_detail.loc[idx, 'Facteur multiplicatif Lissage besoin brut'] = by_factor
    
    # Get factors for this product type  
    current_row = optimization_params[optimization_params['Type de produits V2'].str.lower() == type_produit_v2.lower()]
    if not current_row.empty:
        j_factor = current_row.iloc[0]['Top 500']
        k_factor = current_row.iloc[0]['Top 3000'] 
        l_factor = current_row.iloc[0]['Autre']
    else:
        j_factor = k_factor = l_factor = 1.0
    
    # Use the existing vectorized function
    total_cf = cf_main_module.get_total_cf_optimisee_vectorized(
        filtered_detail, j_factor, k_factor, l_factor)
    
    return total_cf

def calculate_excel_variation_relative(pdc_row, df_detail, optimization_params):
    """
    Calculate variation relative using Excel formulas:
    ColonneResultatVariationRelative = H18/E18 (Différence PDC / Commande) / PDC
    where H18 = PDC - En-cours - Commande optimisée
    """
    # Get base values
    pdc_base = pd.to_numeric(pdc_row.get('PDC', 0), errors='coerce') or 0.0
    en_cours = pd.to_numeric(pdc_row.get('En-cours', 0), errors='coerce') or 0.0
    poids_ac = pd.to_numeric(pdc_row.get('Poids du A/C max', 100), errors='coerce') or 100.0
    boost_pdc = pd.to_numeric(pdc_row.get('Boost PDC', 0), errors='coerce') or 0.0
    
    # Convert percentage to decimal if needed
    if poids_ac > 10:
        poids_ac = poids_ac / 100.0
    
    # Apply boost to PDC
    pdc_adjusted = pdc_base * poids_ac * (1.0 + boost_pdc)
    
    # Calculate total commande optimisée
    type_produit_v2 = pdc_row['Type de produits V2']
    date_livraison = pdc_row['Jour livraison']
    
    commande_optimisee = calculate_total_commande_optimisee(
        df_detail, optimization_params, type_produit_v2, date_livraison)
    
    # Calculate difference: H18 = PDC - En-cours - Commande optimisée
    difference = pdc_adjusted - en_cours - commande_optimisee
    
    # Calculate variation relative: H18/E18 (difference / PDC)
    if pdc_adjusted != 0:
        variation_relative = difference / pdc_adjusted
    else:
        variation_relative = 0.0
    
    return variation_relative, abs(difference)

def vba_limite_haute_basse_fixed(df_pdc_sim, df_detail):
    """
    Correct implementation of VBA Limite_Haute_Basse logic
    """
    print("VBA Limite_Haute_Basse: Starting...")
    
    # Get margin from MacroParam
    marge_manoeuvre = MacroParam.MARGE_POUR_BOOST_ET_L_VAR_SOLVER
    
    df_result = df_pdc_sim.copy()
    
    # Initialize all limits and parameters (VBA initial setup)
    df_result['Limite Basse Top 500'] = 1.0
    df_result['Limite Basse Top 3000'] = 1.0
    df_result['Limite Basse Autre'] = 1.0
    df_result['Limite Haute Top 500'] = 1.0
    df_result['Limite Haute Top 3000'] = 1.0
    df_result['Limite Haute Autre'] = 1.0
    df_result['Boost PDC'] = 0.0
    
    # Process each row
    for idx, row in df_result.iterrows():
        print(f"  Processing row {idx+1}/{len(df_result)}: {row['Type de produits V2']}")
        
        # Set parameters to 100% initially (VBA: .Range(...).Value = 1)
        df_result.loc[idx, 'Top 500'] = 1.0
        df_result.loc[idx, 'Top 3000'] = 1.0
        df_result.loc[idx, 'Autre'] = 1.0
        
        # Calculate variation using Excel formulas
        variation_relative, difference_absolue = calculate_excel_variation_relative(
            df_result.loc[idx], df_detail, df_result)
        
        # Adjust boost based on variation (VBA boost logic)
        if variation_relative > marge_manoeuvre:
            df_result.loc[idx, 'Boost PDC'] = -marge_manoeuvre
        elif variation_relative < -marge_manoeuvre:
            df_result.loc[idx, 'Boost PDC'] = marge_manoeuvre
        else:
            df_result.loc[idx, 'Boost PDC'] = 0.0
        
        # Recalculate with boost
        variation_relative, _ = calculate_excel_variation_relative(
            df_result.loc[idx], df_detail, df_result)
        
        # Set limits based on variation (VBA limit setting logic)
        if variation_relative > 0:
            # Still positive - set upper limits to max bounds
            max_facteur = pd.to_numeric(row.get('Max Facteur', 4.0), errors='coerce') or 4.0
            df_result.loc[idx, 'Limite Haute Top 500'] = max_facteur
            df_result.loc[idx, 'Limite Haute Top 3000'] = max_facteur
            df_result.loc[idx, 'Limite Haute Autre'] = max_facteur
            print(f"    → Upper limits set to {max_facteur} (positive variation)")
        else:
            # Negative - cascading reduction test
            # Test 1: Autre = 0
            df_result.loc[idx, 'Limite Basse Autre'] = 0.0
            df_result.loc[idx, 'Autre'] = 0.0
            
            variation_test, _ = calculate_excel_variation_relative(
                df_result.loc[idx], df_detail, df_result)
            
            if variation_test <= 0:
                # Test 2: Top 3000 = 0 also
                df_result.loc[idx, 'Limite Basse Top 3000'] = 0.0
                df_result.loc[idx, 'Top 3000'] = 0.0
                df_result.loc[idx, 'Limite Haute Autre'] = 0.0
                
                variation_test2, _ = calculate_excel_variation_relative(
                    df_result.loc[idx], df_detail, df_result)
                
                if variation_test2 <= 0:
                    # Test 3: Top 500 lower bound = 0
                    df_result.loc[idx, 'Limite Basse Top 500'] = 0.0
                    df_result.loc[idx, 'Limite Haute Top 3000'] = 0.0
            
            print(f"    → Bounds: LB=({df_result.loc[idx, 'Limite Basse Top 500']:.1f},{df_result.loc[idx, 'Limite Basse Top 3000']:.1f},{df_result.loc[idx, 'Limite Basse Autre']:.1f}) UB=({df_result.loc[idx, 'Limite Haute Top 500']:.1f},{df_result.loc[idx, 'Limite Haute Top 3000']:.1f},{df_result.loc[idx, 'Limite Haute Autre']:.1f})")
    
    return df_result

def vba_optimisation_fixed(df_pdc_sim, df_detail):
    """
    Correct implementation of VBA Optimisation logic
    """
    print("VBA Optimisation: Starting...")
    
    marge_manoeuvre = MacroParam.MARGE_POUR_BOOST_ET_L_VAR_SOLVER
    marge_manoeuvre_uvc = MacroParam.MARGE_I_POUR_SOLVER_CONDITION
    
    df_result = df_pdc_sim.copy()
    
    # Process each row
    for idx, row in df_result.iterrows():
        print(f"  Optimizing row {idx}: {row['Type de produits V2']}")
        
        # Determine TypeLissage
        limite_basse_product = (row['Limite Basse Top 500'] * 
                               row['Limite Basse Top 3000'] * 
                               row['Limite Basse Autre'])
        
        if limite_basse_product >= 1.0:
            type_lissage = 1  # Hausse
        else:
            type_lissage = 0  # Baisse
        
        print(f"    TypeLissage: {type_lissage} ({'Hausse' if type_lissage == 1 else 'Baisse'})")
        
        # Check if min == max bounds
        min_facteur = pd.to_numeric(row.get('Min Facteur', 0.0), errors='coerce') or 0.0
        max_facteur = pd.to_numeric(row.get('Max Facteur', 4.0), errors='coerce') or 4.0
        
        if min_facteur == max_facteur:
            # Set all to bound value
            df_result.loc[idx, 'Top 500'] = max_facteur
            df_result.loc[idx, 'Top 3000'] = max_facteur
            df_result.loc[idx, 'Autre'] = max_facteur
        else:
            # Test with max bounds first
            df_result.loc[idx, 'Top 500'] = max_facteur
            df_result.loc[idx, 'Top 3000'] = max_facteur
            df_result.loc[idx, 'Autre'] = max_facteur
            
            test_variation, _ = calculate_excel_variation_relative(
                df_result.loc[idx], df_detail, df_result)
            
            if test_variation <= 0:
                # Reset to 1.0 and check if solver is needed
                df_result.loc[idx, 'Top 500'] = 1.0
                df_result.loc[idx, 'Top 3000'] = 1.0
                df_result.loc[idx, 'Autre'] = 1.0
                
                # Check solver conditions
                variation_relative, difference_absolue = calculate_excel_variation_relative(
                    df_result.loc[idx], df_detail, df_result)
                
                variation_absolue = abs(variation_relative)
                solver_needed = (variation_absolue > marge_manoeuvre and 
                               difference_absolue > marge_manoeuvre_uvc)
                
                if solver_needed:
                    print(f"    Running Excel Solver simulation")
                    optimized_params = simulate_excel_solver(
                        df_result.loc[idx], df_detail, df_result, type_lissage)
                    df_result.loc[idx, 'Top 500'] = optimized_params[0]
                    df_result.loc[idx, 'Top 3000'] = optimized_params[1]
                    df_result.loc[idx, 'Autre'] = optimized_params[2]
                else:
                    print(f"    Solver not needed, keeping 1.0")
        
        # Update moyenne
        df_result.loc[idx, 'Moyenne'] = (df_result.loc[idx, 'Top 500'] + 
                                        df_result.loc[idx, 'Top 3000'] + 
                                        df_result.loc[idx, 'Autre']) / 3.0
        
        print(f"    Final: Top500={df_result.loc[idx, 'Top 500']:.3f}, Top3000={df_result.loc[idx, 'Top 3000']:.3f}, Autre={df_result.loc[idx, 'Autre']:.3f}")
    
    return df_result

def simulate_excel_solver(pdc_row, df_detail, optimization_params, type_lissage):
    """
    Simulate Excel Solver to minimize ColonneResultatDifferenceAbsolue
    """
    # Get bounds
    lim_bas_top500 = pdc_row['Limite Basse Top 500']
    lim_bas_top3000 = pdc_row['Limite Basse Top 3000']
    lim_bas_autre = pdc_row['Limite Basse Autre']
    lim_haut_top500 = pdc_row['Limite Haute Top 500']
    lim_haut_top3000 = pdc_row['Limite Haute Top 3000']
    lim_haut_autre = pdc_row['Limite Haute Autre']
    
    def objective_function(params):
        """Minimize difference absolue (Excel solver target)"""
        if type_lissage == 1:
            # Hausse: J=K=L (VBA sets formulas K=J, L=J)
            j, k, l = params[0], params[0], params[0]
        else:
            # Baisse: optimize all three separately
            j, k, l = params[0], params[1], params[2]
        
        # Create temporary optimization params
        temp_params = optimization_params.copy()
        temp_params.loc[pdc_row.name, 'Top 500'] = j
        temp_params.loc[pdc_row.name, 'Top 3000'] = k  
        temp_params.loc[pdc_row.name, 'Autre'] = l
        
        # Calculate variation using Excel formulas
        _, difference_absolue = calculate_excel_variation_relative(pdc_row, df_detail, temp_params)
        return difference_absolue
    
    # Set up optimization
    if type_lissage == 1:
        # Hausse: only optimize first parameter
        bounds = [(lim_bas_top500, lim_haut_top500)]
        constraints = []
    else:
        # Baisse: optimize all with J>=K>=L constraint
        bounds = [(lim_bas_top500, lim_haut_top500),
                 (lim_bas_top3000, lim_haut_top3000),
                 (lim_bas_autre, lim_haut_autre)]
        
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},  # J >= K
            {'type': 'ineq', 'fun': lambda x: x[1] - x[2]}   # K >= L
        ]
    
    # Try differential evolution first (mimics Excel Evolutionary Solver)
    try:
        if type_lissage == 1:
            def obj_with_constraints(params):
                return objective_function(params)
        else:
            def obj_with_constraints(params):
                if not (params[0] >= params[1] >= params[2]):
                    return 1e6
                return objective_function(params)
        
        result = differential_evolution(
            obj_with_constraints,
            bounds,
            seed=1,
            maxiter=20,  # Reduced for speed
            popsize=10   # Reduced for speed
        )
        
        if result.success:
            if type_lissage == 1:
                return result.x[0], result.x[0], result.x[0]
            else:
                return result.x[0], result.x[1], result.x[2]
    except:
        pass
    
    # Fallback to SLSQP
    try:
        initial_guess = [1.0] if type_lissage == 1 else [1.0, 1.0, 1.0]
        result = minimize(
            objective_function,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 0.01}
        )
        
        if result.success:
            if type_lissage == 1:
                return result.x[0], result.x[0], result.x[0]
            else:
                return result.x[0], result.x[1], result.x[2]
    except:
        pass
    
    # Ultimate fallback
    return 1.0, 1.0, 1.0

def run_vba_optimization_fixed(df_pdc_sim, df_detail):
    """
    Main function to run corrected VBA optimization
    """
    print("Starting corrected VBA optimization...")
    
    # Step 1: Limite_Haute_Basse
    df_after_lhb = vba_limite_haute_basse_fixed(df_pdc_sim, df_detail)
    
    # Step 2: Optimisation
    df_final = vba_optimisation_fixed(df_after_lhb, df_detail)
    
    print("VBA optimization completed.")
    return df_final