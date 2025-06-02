import pandas as pd
import numpy as np
from scipy.optimize import minimize
import MacroParam
import CommandeFinale as cf_main_module

def vba_limite_haute_basse(df_pdc_sim, df_detail, df_pdc_perm):
    """
    Implement exact VBA Limite_Haute_Basse logic
    """
    print("VBA Limite_Haute_Basse: Starting...")
    
    # Get margin from MacroParam (equivalent to Feuil13.Cells(8, 3))
    marge_manoeuvre = MacroParam.MARGE_POUR_BOOST_ET_L_VAR_SOLVER
    
    # Initialize all limits and parameters
    df_result = df_pdc_sim.copy()
    
    # Ensure numeric columns are float type to handle decimal optimization results
    numeric_cols = ['Limite Basse Top 500', 'Limite Basse Top 3000', 'Limite Basse Autre',
                   'Limite Haute Top 500', 'Limite Haute Top 3000', 'Limite Haute Autre',
                   'Top 500', 'Top 3000', 'Autre', 'Boost PDC', 'Moyenne']
    
    for col in numeric_cols:
        if col in df_result.columns:
            df_result[col] = df_result[col].astype(float)
    
    # Set initial values (equivalent to VBA initialization)
    df_result['Limite Basse Top 500'] = 1.0
    df_result['Limite Basse Top 3000'] = 1.0 
    df_result['Limite Basse Autre'] = 1.0
    df_result['Limite Haute Top 500'] = 1.0
    df_result['Limite Haute Top 3000'] = 1.0
    df_result['Limite Haute Autre'] = 1.0
    df_result['Boost PDC'] = 0.0
    
    # Process each row (equivalent to VBA For i = PremiereLigneParam To DerniereLigneParam)
    for idx, row in df_result.iterrows():
        print(f"  Processing row {idx}: {row['Type de produits V2']}")
        
        # Set parameters to 100% (equivalent to VBA .Range(...).Value = 1)
        df_result.loc[idx, 'Top 500'] = 1.0
        df_result.loc[idx, 'Top 3000'] = 1.0
        df_result.loc[idx, 'Autre'] = 1.0
        
        # Calculate variation (simulate Excel Calculate)
        variation_relative = calculate_variation_relative(row, df_detail, df_pdc_perm, 1.0, 1.0, 1.0, 0.0)
        
        # Adjust boost based on variation (equivalent to VBA boost logic)
        if variation_relative > marge_manoeuvre:
            df_result.loc[idx, 'Boost PDC'] = -marge_manoeuvre
        elif variation_relative < -marge_manoeuvre:
            df_result.loc[idx, 'Boost PDC'] = marge_manoeuvre
        else:
            df_result.loc[idx, 'Boost PDC'] = 0.0
            
        # Recalculate with boost
        boost_value = df_result.loc[idx, 'Boost PDC']
        variation_relative = calculate_variation_relative(row, df_detail, df_pdc_perm, 1.0, 1.0, 1.0, boost_value)
        
        # Set limits based on variation (equivalent to VBA limit setting logic)
        if variation_relative > 0:
            # If still positive, set upper limits to max bounds (VBA: Cells(i, ColonneBorneMax).Value)
            max_facteur = pd.to_numeric(row.get('Max Facteur', 4.0), errors='coerce')
            if pd.isna(max_facteur):
                max_facteur = 4.0
            
            df_result.loc[idx, 'Limite Haute Top 500'] = max_facteur
            df_result.loc[idx, 'Limite Haute Top 3000'] = max_facteur  
            df_result.loc[idx, 'Limite Haute Autre'] = max_facteur
            print(f"    Setting upper limits to {max_facteur} (still positive variation)")
        else:
            # VBA else logic - cascading reduction test
            print(f"    Variation negative, testing cascading reduction...")
            
            # Reset upper limits to 1.0 first
            df_result.loc[idx, 'Limite Haute Top 500'] = 1.0
            df_result.loc[idx, 'Limite Haute Top 3000'] = 1.0
            df_result.loc[idx, 'Limite Haute Autre'] = 1.0
            
            # Test 1: Set Autre = 0 (VBA: Cells(i, PremiereColonneLimiteBasse + 2).Value = 0)
            df_result.loc[idx, 'Limite Basse Autre'] = 0.0
            df_result.loc[idx, 'Autre'] = 0.0
            
            # Recalculate with updated row
            updated_row = df_result.loc[idx].copy()
            variation_test = calculate_variation_relative(updated_row, df_detail, df_pdc_perm, 1.0, 1.0, 0.0, boost_value)
            print(f"    Test 1 (Autre=0): variation = {variation_test:.3f}")
            
            if variation_test <= 0:
                # Test 2: Set Top 3000 = 0 also (VBA continues testing)
                df_result.loc[idx, 'Limite Basse Top 3000'] = 0.0
                df_result.loc[idx, 'Top 3000'] = 0.0  
                df_result.loc[idx, 'Limite Haute Autre'] = 0.0
                
                updated_row = df_result.loc[idx].copy()
                variation_test2 = calculate_variation_relative(updated_row, df_detail, df_pdc_perm, 1.0, 0.0, 0.0, boost_value)
                print(f"    Test 2 (Top3000=0): variation = {variation_test2:.3f}")
                
                if variation_test2 <= 0:
                    # Test 3: Set Top 500 = 0 (VBA final test)
                    df_result.loc[idx, 'Limite Basse Top 500'] = 0.0
                    df_result.loc[idx, 'Limite Haute Top 3000'] = 0.0
                    print(f"    Test 3: Setting Top 500 lower bound to 0")
        
        # Update moyenne
        df_result.loc[idx, 'Moyenne'] = (df_result.loc[idx, 'Top 500'] + 
                                        df_result.loc[idx, 'Top 3000'] + 
                                        df_result.loc[idx, 'Autre']) / 3.0
        
        print(f"    After LHB: Top500={df_result.loc[idx, 'Top 500']:.3f}, Top3000={df_result.loc[idx, 'Top 3000']:.3f}, Autre={df_result.loc[idx, 'Autre']:.3f}")
    
    return df_result

def vba_optimisation(df_pdc_sim, df_detail, df_pdc_perm):
    """
    Implement exact VBA Optimisation logic
    """
    print("VBA Optimisation: Starting...")
    
    marge_manoeuvre = MacroParam.MARGE_POUR_BOOST_ET_L_VAR_SOLVER
    marge_manoeuvre_uvc = MacroParam.MARGE_I_POUR_SOLVER_CONDITION
    
    df_result = df_pdc_sim.copy()
    
    # Ensure numeric columns are float type
    numeric_cols = ['Top 500', 'Top 3000', 'Autre', 'Moyenne']
    for col in numeric_cols:
        if col in df_result.columns:
            df_result[col] = df_result[col].astype(float)
    
    # Process each row
    for idx, row in df_result.iterrows():
        print(f"  Optimizing row {idx}: {row['Type de produits V2']}")
        
        # Determine TypeLissage (equivalent to VBA TypeLissage calculation)
        # VBA: If (Cells(i, PremiereColonneLimiteBasse) * Cells(i, PremiereColonneLimiteBasse + 1) * Cells(i, PremiereColonneLimiteBasse + 2)) >= 1
        limite_basse_top500 = pd.to_numeric(row.get('Limite Basse Top 500', 1), errors='coerce')
        limite_basse_top3000 = pd.to_numeric(row.get('Limite Basse Top 3000', 1), errors='coerce')
        limite_basse_autre = pd.to_numeric(row.get('Limite Basse Autre', 1), errors='coerce')
        
        if pd.isna(limite_basse_top500): limite_basse_top500 = 1.0
        if pd.isna(limite_basse_top3000): limite_basse_top3000 = 1.0  
        if pd.isna(limite_basse_autre): limite_basse_autre = 1.0
        
        limite_basse_product = limite_basse_top500 * limite_basse_top3000 * limite_basse_autre
        
        if limite_basse_product >= 1.0:
            type_lissage = 1  # Hausse (augmentation)
        else:
            type_lissage = 0  # Baisse (diminution)
            
        print(f"    TypeLissage: {type_lissage} ({'Hausse' if type_lissage == 1 else 'Baisse'})")
        
        # Check if BorneMin == BorneMax (VBA: If Cells(i, ColonneBorneMin) = Cells(i, ColonneBorneMax))
        min_facteur = row['Min Facteur'] if pd.notna(row['Min Facteur']) else 0.0
        max_facteur = row['Max Facteur'] if pd.notna(row['Max Facteur']) else 4.0
        
        if min_facteur == max_facteur:
            # Set all parameters to the bound value
            df_result.loc[idx, 'Top 500'] = max_facteur
            df_result.loc[idx, 'Top 3000'] = max_facteur
            df_result.loc[idx, 'Autre'] = max_facteur
        else:
            # Test with BorneMax (VBA: test with max bound)
            test_variation = calculate_variation_relative(row, df_detail, df_pdc_perm, 
                                                        max_facteur, max_facteur, max_facteur, 
                                                        row['Boost PDC'])
            
            if test_variation <= 0:
                # Reset to 100% and check if solver is needed
                df_result.loc[idx, 'Top 500'] = 1.0
                df_result.loc[idx, 'Top 3000'] = 1.0
                df_result.loc[idx, 'Autre'] = 1.0
                
                # Calculate metrics to determine if solver is needed
                variation_absolue, difference_absolue = calculate_solver_metrics(row, df_detail, df_pdc_perm, 
                                                                               1.0, 1.0, 1.0, row['Boost PDC'])
                
                # Check if solver is needed (VBA condition)
                solver_needed = (variation_absolue > marge_manoeuvre and 
                               difference_absolue > marge_manoeuvre_uvc)
                
                if solver_needed:
                    print(f"    Calling VBA Solver (TypeLissage={type_lissage})")
                    optimized_params = vba_solveur(row, df_detail, df_pdc_perm, type_lissage)
                    df_result.loc[idx, 'Top 500'] = optimized_params[0]
                    df_result.loc[idx, 'Top 3000'] = optimized_params[1] 
                    df_result.loc[idx, 'Autre'] = optimized_params[2]
                else:
                    print(f"    Solver not needed, keeping values at 1.0")
            else:
                # Use BorneMax values
                df_result.loc[idx, 'Top 500'] = max_facteur
                df_result.loc[idx, 'Top 3000'] = max_facteur
                df_result.loc[idx, 'Autre'] = max_facteur
        
        # Update moyenne
        df_result.loc[idx, 'Moyenne'] = (df_result.loc[idx, 'Top 500'] + 
                                        df_result.loc[idx, 'Top 3000'] + 
                                        df_result.loc[idx, 'Autre']) / 3.0
        
        print(f"    Final values: Top500={df_result.loc[idx, 'Top 500']:.3f}, Top3000={df_result.loc[idx, 'Top 3000']:.3f}, Autre={df_result.loc[idx, 'Autre']:.3f}")
    
    return df_result

def vba_solveur(row, df_detail, df_pdc_perm, type_lissage):
    """
    Implement VBA Solveur logic using scipy optimization to mimic Excel Evolutionary Solver
    """
    print(f"    VBA Solveur: TypeLissage={type_lissage}")
    
    # Get bounds from the row
    lim_bas_top500 = row['Limite Basse Top 500']
    lim_bas_top3000 = row['Limite Basse Top 3000'] 
    lim_bas_autre = row['Limite Basse Autre']
    lim_haut_top500 = row['Limite Haute Top 500']
    lim_haut_top3000 = row['Limite Haute Top 3000']
    lim_haut_autre = row['Limite Haute Autre']
    
    boost_pdc = row['Boost PDC']
    
    def objective_function(params):
        """Minimize absolute difference (equivalent to Excel Solver target)"""
        if type_lissage == 1:  # Hausse - only optimize first parameter
            j, k, l = params[0], params[0], params[0]  # K=J, L=J as per VBA
        else:  # Baisse - optimize all three with J>=K>=L constraint
            j, k, l = params[0], params[1], params[2]
        
        _, difference_absolue = calculate_solver_metrics(row, df_detail, df_pdc_perm, j, k, l, boost_pdc)
        return difference_absolue
    
    if type_lissage == 1:  # Hausse
        # Only optimize J (VBA: ByChange:="$" & LettrePremiereColonneParametrage & "$" & i)
        bounds = [(lim_bas_top500, lim_haut_top500)]
        x0 = [1.0]
        
        def constraint_func(params):
            return []  # No additional constraints for Hausse beyond bounds
            
    else:  # Baisse  
        # Optimize J, K, L with J>=K>=L constraint
        bounds = [(lim_bas_top500, lim_haut_top500),
                 (lim_bas_top3000, lim_haut_top3000), 
                 (lim_bas_autre, lim_haut_autre)]
        x0 = [1.0, 1.0, 1.0]
        
        def constraint_func(params):
            # VBA: J >= K >= L constraint
            return [params[0] - params[1],  # J >= K
                   params[1] - params[2]]   # K >= L
    
    # Setup constraints
    constraints = []
    if type_lissage == 0:  # Only for Baisse
        constraints.append({'type': 'ineq', 'fun': lambda x: constraint_func(x)[0]})
        constraints.append({'type': 'ineq', 'fun': lambda x: constraint_func(x)[1]})
    
    # Mimic Excel Evolutionary Solver parameters more closely
    # Excel VBA: PopulationSize:=100, MutationRate:=0.075, MaxTime:=TempsSolveur
    try:
        from scipy.optimize import differential_evolution
        use_differential_evolution = True
    except ImportError:
        use_differential_evolution = False
    
    best_result = None
    best_objective = float('inf')
    
    if use_differential_evolution:
        # Use differential evolution to mimic Excel's Evolutionary Solver
        try:
            def objective_with_constraints(params):
                # Apply constraints manually for differential evolution
                if type_lissage == 0:  # Baisse - enforce J>=K>=L
                    if len(params) == 3:
                        if not (params[0] >= params[1] >= params[2]):
                            return 1e6  # Large penalty for constraint violation
                return objective_function(params)
            
            result = differential_evolution(
                objective_with_constraints,
                bounds,
                seed=1,
                popsize=15,  # Reasonable population size
                mutation=(0.05, 0.2),  # Match Excel mutation rate range
                recombination=0.7,
                maxiter=50,  # Reasonable iteration limit
                atol=0.01,
                tol=0.01
            )
            
            if result.success:
                best_result = result
                best_objective = result.fun
        except Exception as e:
            print(f"      Differential evolution failed: {e}")
            use_differential_evolution = False
    
    # Fallback to SLSQP with multiple starts if differential evolution not available or fails
    if not use_differential_evolution or best_result is None:
        for seed in range(10):  # Multiple random starts
            np.random.seed(seed)
            
            # Random initial guess within bounds
            if type_lissage == 1:
                x0_random = [np.random.uniform(bounds[0][0], bounds[0][1])]
            else:
                x0_random = [np.random.uniform(b[0], b[1]) for b in bounds]
                # Ensure J>=K>=L for initial guess
                x0_random = sorted(x0_random, reverse=True)
            
            try:
                result = minimize(objective_function, x0_random, method='SLSQP', 
                                bounds=bounds, constraints=constraints,
                                options={'ftol': 0.01, 'maxiter': 100})
                
                if result.success and result.fun < best_objective:
                    best_result = result
                    best_objective = result.fun
                    
            except Exception as e:
                continue
    
    if best_result is not None and best_result.success:
        if type_lissage == 1:
            # For Hausse: J=K=L
            j_opt = best_result.x[0]
            return j_opt, j_opt, j_opt
        else:
            # For Baisse: return optimized J, K, L
            return best_result.x[0], best_result.x[1], best_result.x[2]
    else:
        # Fallback to initial values if optimization fails
        print(f"      Solver failed, using fallback values")
        return 1.0, 1.0, 1.0

def calculate_variation_relative(row, df_detail, df_pdc_perm, j_factor, k_factor, l_factor, boost_pdc):
    """Calculate variation relative as per Excel VBA logic
    
    This function simulates Excel's Calculate command that VBA relies on.
    It calculates: (PDC_adjusted - En_cours - Commande_optimisee) / PDC_adjusted
    """
    # Filter detail data for this row - match by Type de produits V2 and delivery date
    df_detail_filtered = filter_detail_for_row(row, df_detail)
    
    if df_detail_filtered.empty:
        print(f"    WARNING: No detail data found for {row['Type de produits V2']} on {row['Jour livraison']}")
        return 0.0
    
    # Calculate total optimized command using vectorized approach for accuracy
    total_cmd_opt = cf_main_module.get_total_cf_optimisee_vectorized(
        df_detail_filtered, j_factor, k_factor, l_factor)
    
    # Get base values - handle missing data properly
    pdc_base = pd.to_numeric(row.get('PDC', 0), errors='coerce')
    if pd.isna(pdc_base):
        pdc_base = 0.0
        
    poids_ac = pd.to_numeric(row.get('Poids du A/C max', 100), errors='coerce')
    if pd.isna(poids_ac):
        poids_ac = 100.0
    
    # Convert percentage to decimal if needed (Excel shows as %)
    if poids_ac > 10:  # Assume it's a percentage
        poids_ac = poids_ac / 100.0
    
    en_cours = pd.to_numeric(row.get('En-cours', 0), errors='coerce')
    if pd.isna(en_cours):
        en_cours = 0.0
        
    # Apply boost to PDC (VBA: PDC * Poids_AC * (1 + boost))
    pdc_adjusted = pdc_base * poids_ac * (1.0 + boost_pdc)
    
    # Calculate difference: PDC - En_cours - Commande_optimisee
    difference = pdc_adjusted - en_cours - total_cmd_opt
    
    # Calculate relative variation: difference / PDC_adjusted
    if pdc_adjusted != 0.0:
        variation_relative = difference / pdc_adjusted
    else:
        variation_relative = 0.0
    
    return variation_relative

def calculate_solver_metrics(row, df_detail, df_pdc_perm, j_factor, k_factor, l_factor, boost_pdc):
    """Calculate metrics needed for solver decision"""
    variation_relative = calculate_variation_relative(row, df_detail, df_pdc_perm, j_factor, k_factor, l_factor, boost_pdc)
    variation_absolue = abs(variation_relative)
    
    # Calculate difference absolue (used in solver target)
    df_detail_filtered = filter_detail_for_row(row, df_detail)
    total_cmd_opt = 0.0
    for _, detail_row in df_detail_filtered.iterrows():
        cf_value = cf_main_module.get_cf_optimisee_for_detail_line(
            detail_row.to_dict(), j_factor, k_factor, l_factor)
        total_cmd_opt += cf_value
    
    pdc_base = row['PDC'] if pd.notna(row['PDC']) else 0.0
    poids_ac = row['Poids du A/C max'] if pd.notna(row['Poids du A/C max']) else 1.0
    pdc_adjusted = pdc_base * poids_ac * (1 + boost_pdc)
    en_cours = row['En-cours'] if pd.notna(row['En-cours']) else 0.0
    
    difference_absolue = abs(pdc_adjusted - en_cours - total_cmd_opt)
    
    return variation_absolue, difference_absolue

def filter_detail_for_row(row, df_detail):
    """Filter detail data to match the PDC_Sim row
    
    Matches by Type de produits V2 and delivery date to get the corresponding detail rows
    """
    # Get the matching criteria
    type_v2_normalized = str(row['Type de produits V2']).strip().lower()
    
    # Handle different possible date column names
    date_cols = ['DATE_LIVRAISON_V2', 'Jour livraison', 'Date livraison', 'DATE_LIVRAISON']
    date_col = None
    for col in date_cols:
        if col in df_detail.columns:
            date_col = col
            break
    
    if date_col is None:
        print(f"    ERROR: No date column found in detail data. Available columns: {df_detail.columns.tolist()}")
        return df_detail.iloc[0:0]  # Return empty DataFrame with same structure
    
    try:
        jour_livraison = pd.to_datetime(row['Jour livraison']).normalize()
        
        # Create mask for filtering
        type_mask = df_detail['Type de produits V2'].astype(str).str.strip().str.lower() == type_v2_normalized
        date_mask = pd.to_datetime(df_detail[date_col]).dt.normalize() == jour_livraison
        
        mask = type_mask & date_mask
        filtered_df = df_detail[mask]
        
        return filtered_df
        
    except Exception as e:
        print(f"    ERROR in filter_detail_for_row: {e}")
        return df_detail.iloc[0:0]  # Return empty DataFrame

def run_vba_optimization(df_pdc_sim, df_detail, df_pdc_perm):
    """
    Main function to run the complete VBA optimization logic
    """
    print("Starting VBA-style optimization...")
    
    # Step 1: Limite_Haute_Basse
    df_after_lhb = vba_limite_haute_basse(df_pdc_sim, df_detail, df_pdc_perm)
    
    # Step 2: Optimisation 
    df_final = vba_optimisation(df_after_lhb, df_detail, df_pdc_perm)
    
    print("VBA-style optimization completed.")
    return df_final