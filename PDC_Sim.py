'''
Module for creating the "PDC - Simulation Commande" table.
'''
import pandas as pd
import os
import PDC, Optimisation, Encours, MacroParam, TypeProduits
from MacroParam import ALERTE_SURCHARGE_NIVEAU_1

# Define constants needed for calculation
TAUX_SERVICE_AMONT_ESTIME = 0.05  # 5% default value if not defined in MacroParam

# Define column names from the "Détail" sheet mapping for "Commande optimisée"
CF_COL_DETAIL = "Commande optimisée avec arrondi et mini et TS"
BM_COL_DETAIL = "Type de produits V2" # This is 'Type de produits V2' in parametres_commandes_appros
S_COL_DETAIL = "DATE_LIVRAISON_V2"    # This is 'DATE_LIVRAISON_V2' in parametres_commandes_appros
SM_COL_DETAIL = "Commande Finale avec mini et arrondi SM à 100% avec TS" # Column for "Commande SM a 100%"

# Define TOP categories
TOP_500 = "Top 500"
TOP_3000 = "Top 3000"
AUTRE = "Autre"

def create_pdc_simulation_table(
    raw_pdc_excel_data: pd.DataFrame, # Changed from pdc_perm_data, expects raw data
    parametres_commandes_appros: pd.DataFrame, # This is the "Détail" sheet data
    synth_cde_data: pd.DataFrame,
    macro_params: dict # Kept, though its direct use in PDC calculation is removed
) -> pd.DataFrame:
    '''
    Creates the "PDC - Simulation Commande" table.

    Args:
        raw_pdc_excel_data: DataFrame from PDC.load_pdc_perm_data() (raw Excel data).
        parametres_commandes_appros: DataFrame from Optimisation.create_parametres_commandes_appros() ("Détail" sheet).
        synth_cde_data: DataFrame from Encours.load_encours_data() (or similar).
        macro_params: Dictionary of macro parameters from MacroParam.load_macro_params().

    Returns:
        DataFrame representing the "PDC - Simulation Commande" table.
    '''
    # Initialize an empty list to store table rows
    table_data = []

    # Key columns needed from parametres_commandes_appros to define output rows
    key_cols_for_lookup = ['Type de produits V2', 'Type de produits', 'Jour de livraison']
    if not all(col in parametres_commandes_appros.columns for col in key_cols_for_lookup):
        missing_key_cols = [col for col in key_cols_for_lookup if col not in parametres_commandes_appros.columns]
        print(f"Error: Missing one or more key columns ({', '.join(missing_key_cols)}) in 'parametres_commandes_appros' needed to build the simulation table. Aborting.")
        return pd.DataFrame()    # params_lookup defines the unique rows ('Type de produits V2', 'Jour de livraison') for the output table
    # Note: BM_COL_DETAIL is 'Type de produits V2'
    params_lookup = parametres_commandes_appros.drop_duplicates(subset=[BM_COL_DETAIL], keep='first').set_index(BM_COL_DETAIL)

    # Pre-process parametres_commandes_appros (detail_df) for 'Commande optimisée' calculation
    detail_df_processed = parametres_commandes_appros.copy()
    
    # Ensure the required columns exist in the dataframe
    # Map the common column 'Jour de livraison' to S_COL_DETAIL if missing
    if S_COL_DETAIL not in detail_df_processed.columns and 'Jour de livraison' in detail_df_processed.columns:
        detail_df_processed[S_COL_DETAIL] = detail_df_processed['Jour de livraison']
        print(f"Added '{S_COL_DETAIL}' column based on 'Jour de livraison'")
    
    # Map 'Commande optimisée' or 'Commande SL a 100%' to CF_COL_DETAIL if missing
    if CF_COL_DETAIL not in detail_df_processed.columns:
        if 'Commande optimisée' in detail_df_processed.columns:
            detail_df_processed[CF_COL_DETAIL] = detail_df_processed['Commande optimisée']
            print(f"Added '{CF_COL_DETAIL}' column based on 'Commande optimisée'")
        elif 'Commande SL a 100%' in detail_df_processed.columns:
            detail_df_processed[CF_COL_DETAIL] = detail_df_processed['Commande SL a 100%']
            print(f"Added '{CF_COL_DETAIL}' column based on 'Commande SL a 100%'")
        elif 'Commande Finale avec mini et arrondi SM à 100% avec TS' in detail_df_processed.columns:
            detail_df_processed[CF_COL_DETAIL] = detail_df_processed['Commande Finale avec mini et arrondi SM à 100% avec TS']
            print(f"Added '{CF_COL_DETAIL}' column based on 'Commande Finale avec mini et arrondi SM à 100% avec TS'")
    
    # Make sure the SM column exists for "Commande SM a 100%" calculation
    if SM_COL_DETAIL not in detail_df_processed.columns and 'Commande Finale avec mini et arrondi SM à 100% avec TS' in detail_df_processed.columns:
        detail_df_processed[SM_COL_DETAIL] = detail_df_processed['Commande Finale avec mini et arrondi SM à 100% avec TS']
    
    required_cols_for_co = [BM_COL_DETAIL, S_COL_DETAIL, CF_COL_DETAIL]
    missing_cols_for_co = [col for col in required_cols_for_co if col not in detail_df_processed.columns]
    can_calculate_co = False

    # Check if we can calculate "Commande SM a 100%"
    required_cols_for_sm = [BM_COL_DETAIL, S_COL_DETAIL, SM_COL_DETAIL]
    missing_cols_for_sm = [col for col in required_cols_for_sm if col not in detail_df_processed.columns]
    can_calculate_sm = len(missing_cols_for_sm) == 0

    if not missing_cols_for_co:
        can_calculate_co = True
        try:
            detail_df_processed[S_COL_DETAIL + '_dt'] = pd.to_datetime(detail_df_processed[S_COL_DETAIL], format='%d/%m/%Y', errors='coerce')
            detail_df_processed[CF_COL_DETAIL] = pd.to_numeric(detail_df_processed[CF_COL_DETAIL], errors='coerce').fillna(0)
        except Exception as e:
            print(f"Error during pre-processing for 'Commande optimisée': {e}. 'Commande optimisée' will be 0.")
            can_calculate_co = False
    else:
        print(f"Warning: Missing columns required for 'Commande optimisée' calculation in 'parametres_commandes_appros': {', '.join(missing_cols_for_co)}. 'Commande optimisée' will be 0.")
    
    # Convert data types for "Commande SM a 100%" calculation
    if not missing_cols_for_sm:
        try:
            if S_COL_DETAIL + '_dt' not in detail_df_processed.columns:
                detail_df_processed[S_COL_DETAIL + '_dt'] = pd.to_datetime(detail_df_processed[S_COL_DETAIL], format='%d/%m/%Y', errors='coerce')
            detail_df_processed[SM_COL_DETAIL] = pd.to_numeric(detail_df_processed[SM_COL_DETAIL], errors='coerce').fillna(0)
            can_calculate_sm = True
        except Exception as e:
            print(f"Error during pre-processing for 'Commande SM a 100%': {e}. 'Commande SM a 100%' will be 0.")
            can_calculate_sm = False
    else:
        print(f"Warning: Missing columns required for 'Commande SM a 100%' calculation in 'parametres_commandes_appros': {', '.join(missing_cols_for_sm)}. 'Commande SM a 100%' will be 0.")

    # Prepare raw_pdc_excel_data for PDC lookup (remains the same)
    pdc_data_for_melt = raw_pdc_excel_data.reset_index() # 'Jour' becomes a column
    pdc_long_raw_values = pdc_data_for_melt.melt(
        id_vars=['Jour'],
        var_name='Type de produits',
        value_name='PDC_Raw_Value' # Using raw value from Excel
    )
    pdc_long_raw_values['Jour'] = pd.to_datetime(pdc_long_raw_values['Jour'])

    # Prepare synth_cde_data for lookup (remains the same)
    # Assuming synth_cde_data has 'Date Commande' (or similar for date) and columns for each 'Type de produits'
    # We need to melt it similar to pdc_perm_data if it's in wide format
    # For now, let's assume synth_cde_data is already suitable or will be processed later.
    # Placeholder: Adjust based on actual structure of synth_cde_data
    # Example: if synth_cde_data columns are product types and index is date:
    # synth_cde_long = synth_cde_data.reset_index().melt(
    # id_vars=['Date Commande'], # or actual date column name
    # var_name='Type de produits',
    # value_name='Encours_Value'
    # )
    # synth_cde_long['Date Commande'] = pd.to_datetime(synth_cde_long['Date Commande'])

    for type_prod_v2_key in params_lookup.index:
        try:
            type_prod = params_lookup.loc[type_prod_v2_key, 'Type de produits']
            jour_livraison_str = params_lookup.loc[type_prod_v2_key, 'Jour de livraison']
        except KeyError:
            print(f"Warning: Could not find 'Type de produits' or 'Jour de livraison' for '{type_prod_v2_key}' in params_lookup. Skipping row.")
            continue
        
        jour_livraison_dt = pd.to_datetime(jour_livraison_str, format='%d/%m/%Y', errors='coerce')
        if pd.NaT == jour_livraison_dt:
            print(f"Warning: Could not parse 'Jour de livraison' ('{jour_livraison_str}') for '{type_prod_v2_key}'. Skipping row.")
            continue        

        # --- PDC Calculation --- (implementing Excel formula with Top factors)
        pdc_value = 0

        matching_params = parametres_commandes_appros[
            parametres_commandes_appros['Type de produits V2'] == type_prod_v2_key
        ]
        
        # Step 1: Get the raw PDC value from PDC Perm matching date and product type
        # SI(B18="";"";INDEX('PDC Perm'!$A:$L;EQUIV(D18;'PDC Perm'!$A:$A;0);EQUIV(C18;'PDC Perm'!$1:$1;0))
        pdc_match = pdc_long_raw_values[
            (pdc_long_raw_values['Jour'] == jour_livraison_dt) &
            (pdc_long_raw_values['Type de produits'] == type_prod)
        ]
        
        if not pdc_match.empty:
            pdc_raw_value = pdc_match['PDC_Raw_Value'].iloc[0]
            
            # Step 2: Get Poids A/C final for this product type
            # INDEX($I$4:$I$14;EQUIV(B18;$B$4:$B$14;0))
            poids_ac_final = 1.0  # Default value if not found
            boost_pdc = 0.0  # Default value if not found
            top_500_factor = 1.0  # Default value
            top_3000_factor = 1.0  # Default value
            autre_factor = 1.0  # Default value
            
            # Find the row in parametres_commandes_appros for this type_prod_v2_key
            matching_params = parametres_commandes_appros[
                parametres_commandes_appros['Type de produits V2'] == type_prod_v2_key
            ]
            
            if not matching_params.empty:
                # Get Poids A/C final (column I in Excel), corresponds to Poids A/C final in our data
                if 'Poids A/C final' in matching_params.columns:
                    poids_ac_final_str = matching_params['Poids A/C final'].iloc[0]
                    if pd.notna(poids_ac_final_str) and poids_ac_final_str != "":
                        if isinstance(poids_ac_final_str, str) and '%' in poids_ac_final_str:
                            poids_ac_final = float(poids_ac_final_str.strip('%')) / 100.0
                        else:
                            poids_ac_final = float(poids_ac_final_str)
                
                # Step 3: Get Boost PDC (column H in Excel), corresponds to Boost PDC in our data
                # (1+INDEX($H$4:$H$14;EQUIV(B18;$B$4:$B$14;0)))
                if 'Boost PDC' in matching_params.columns:
                    boost_pdc_str = matching_params['Boost PDC'].iloc[0]
                    if pd.notna(boost_pdc_str) and boost_pdc_str != "":
                        if isinstance(boost_pdc_str, str) and '%' in boost_pdc_str:
                            # Handle both positive and negative percentage formats
                            if boost_pdc_str.startswith('-'):
                                boost_pdc = -float(boost_pdc_str.strip('-%')) / 100.0
                            else:
                                boost_pdc = float(boost_pdc_str.strip('%')) / 100.0
                        else:
                            boost_pdc = float(boost_pdc_str)
                
                # Get Top 500, Top 3000, and Autre factors
                if 'Top 500' in matching_params.columns:
                    top_500_val = matching_params['Top 500'].iloc[0]
                    if pd.notna(top_500_val):
                        top_500_factor = float(top_500_val)
                
                if 'Top 3000' in matching_params.columns:
                    top_3000_val = matching_params['Top 3000'].iloc[0]
                    if pd.notna(top_3000_val):
                        top_3000_factor = float(top_3000_val)
                
                if 'Autre' in matching_params.columns:
                    autre_val = matching_params['Autre'].iloc[0]
                    if pd.notna(autre_val):
                        autre_factor = float(autre_val)
            
            # For PDC calculation, we need to apply the Top factors according to article category
            # Implement this with a helper function called for each article in the raw PDC
            def apply_top_factors(raw_pdc, product_type, type_prod_v2_key):
                # Get the Top classification (Top 500, Top 3000, or Autre)
                # We'd need to access a list of articles with their Top classification
                # For simple demonstration, determine based on some logic
                
                # Here we should load Top article lists and check membership
                # For now, simulate with TypeProduits.get_top_type approximation
                top_type = TypeProduits.get_top_type(product_type, None)
                
                # Apply the appropriate factor based on Top classification
                if top_type == TOP_500:
                    factor = top_500_factor
                elif top_type == TOP_3000:
                    factor = top_3000_factor
                else:  # Autre
                    factor = autre_factor
                
                # Apply the factor to the raw PDC value
                adjusted_pdc = raw_pdc * factor
                return adjusted_pdc
            
            # For simplicity in this implementation, we'll apply a global top factor
            # In a more detailed implementation, this would be applied per article
            # Use an approximate top category weighting based on usual distributions
            weighted_factor = 0.1 * top_500_factor + 0.3 * top_3000_factor + 0.6 * autre_factor
            
            # Apply the complete formula: PDC_Raw_Value * Poids_A/C_final * (1 + Boost_PDC) * weighted_factor
            pdc_value = pdc_raw_value * poids_ac_final * (1 + boost_pdc) * weighted_factor
            
            # Log the calculation details
            print(f"PDC calculation for {type_prod_v2_key}:")
            print(f"  Raw PDC: {pdc_raw_value}")
            print(f"  Poids A/C final: {poids_ac_final}")
            print(f"  Boost PDC: {boost_pdc}")
            print(f"  Top factors - 500: {top_500_factor}, 3000: {top_3000_factor}, Autre: {autre_factor}")
            print(f"  Weighted factor: {weighted_factor}")
            print(f"  Final PDC value: {pdc_value}")
        
        # --- En-cours Calculation ---
        encours_value = 0
        if jour_livraison_dt in synth_cde_data.index and type_prod in synth_cde_data.columns:
            raw_encours = synth_cde_data.loc[jour_livraison_dt, type_prod]
            if pd.notna(raw_encours):
                 encours_value = raw_encours * 1000
        
        # --- Commande optimisée Calculation ---
        commande_optimisee_value = 0
        if can_calculate_co:
            # BM_COL_DETAIL is 'Type de produits V2'
            mask = (
                (detail_df_processed[BM_COL_DETAIL] == type_prod_v2_key) &
                (detail_df_processed[S_COL_DETAIL + '_dt'] == jour_livraison_dt) # Compare with processed datetime column
            )
            # Sum the numeric CF_COL_DETAIL for matched rows
            commande_optimisee_value = detail_df_processed.loc[mask, CF_COL_DETAIL].sum()
        
        # --- Commande SM a 100% Calculation ---
        commande_sm_100_value = 0
        if can_calculate_sm:
            # Find rows matching the current Type de produits V2 and Jour de livraison
            sm_mask = (
                (detail_df_processed[BM_COL_DETAIL] == type_prod_v2_key) &
                (detail_df_processed[S_COL_DETAIL + '_dt'] == jour_livraison_dt)
            )
            # Sum the values for the SM column for matched rows
            commande_sm_100_value = detail_df_processed.loc[sm_mask, SM_COL_DETAIL].sum()
        
        # Calcul de la différence PDC / Commande
        difference_pdc_commande = pdc_value - commande_optimisee_value - encours_value if type_prod_v2_key != "" else ""
          # Calcul de la différence absolue (valeur absolue de la différence PDC / Commande)
        # Formule: =SI(B18="";"";ABS(H18))
        if type_prod_v2_key != "" and difference_pdc_commande != "":
            difference_absolue = abs(difference_pdc_commande)
        else:
            difference_absolue = ""
        
        # Calcul de la tolérance (formule =SI(B18="";"";E18*'Macro-Param'!$C$6))
        # où 'Macro-Param'!$C$6 est la valeur Alerte surcharge niveau 1 (orange) (5%)
        if type_prod_v2_key != "" and pdc_value != "":
            tolerance = pdc_value * ALERTE_SURCHARGE_NIVEAU_1
        else:
            tolerance = ""

        # Calcul de la variation PDC (formule =SI(B18="";"";SIERREUR(H18/E18;0)))
        variation_pdc_calc = ""  # Default to empty string
        if type_prod_v2_key != "":  # Corresponds to B18 != "" (Type de produits V2)
            # H18 is difference_pdc_commande
            # E18 is pdc_value
            # Ensure pdc_value is treated as numeric; it's initialized to 0 or gets a value.
            # difference_pdc_commande is numeric if type_prod_v2_key != ""
            if pd.notna(pdc_value) and pdc_value != 0:
                try:
                    variation_pdc_calc = difference_pdc_commande / pdc_value
                except TypeError: 
                    # This might happen if difference_pdc_commande became non-numeric unexpectedly
                    variation_pdc_calc = 0 
            else: # pdc_value is 0 or NaN, so H18/E18 is an error. SIERREUR results in 0.
                variation_pdc_calc = 0
        # else: variation_pdc_calc remains "" if type_prod_v2_key is ""
            
        variation_absolue_pdc_calc = ""
        if type_prod_v2_key != "":
            if isinstance(variation_pdc_calc, (int, float)): # Check if it's a number
                variation_absolue_pdc_calc = abs(variation_pdc_calc)
            else: # Should be numeric 0 if pdc_value was 0 or NaN, or "" if type_prod_v2_key was ""
                  # If variation_pdc_calc is "", abs("") would error.
                  # Given the logic for variation_pdc_calc, if type_prod_v2_key != "",
                  # variation_pdc_calc is always a number (0 or a float).
                  # So this 'else' branch for non-numeric might not be strictly needed here
                  # if variation_pdc_calc is guaranteed to be numeric when type_prod_v2_key is not empty.
                  # However, to be safe, if it somehow ended up as non-numeric (e.g. empty string by mistake):
                variation_absolue_pdc_calc = 0 # Default to 0, which will become "0.00%"

        # --- Capage Borne Max ? Calculation ---
        # Formula: =SI(B18="";"";SI(ET(M4=G4;L18>0);"Oui";"Non"))
        # B18: type_prod_v2_key
        # M4: Moyenne from parametres_commandes_appros (for type_prod_v2_key)
        # G4: Max Facteur from parametres_commandes_appros (for type_prod_v2_key)
        # L18: variation_absolue_pdc_calc (numeric value)
        capage_borne_max_val = ""
        if type_prod_v2_key != "":
            moyenne_val = 0.0  # Default for M4 (Moyenne)
            max_facteur_val = 0.0  # Default for G4 (Max Facteur)

            # matching_params is the row from parametres_commandes_appros for the current type_prod_v2_key
            if not matching_params.empty:
                if 'Moyenne' in matching_params.columns:
                    moyenne_series = pd.to_numeric(matching_params['Moyenne'], errors='coerce')
                    if not moyenne_series.empty:
                        moyenne_val = moyenne_series.iloc[0]
                        if pd.isna(moyenne_val):
                            moyenne_val = 0.0
                    else:
                        moyenne_val = 0.0 # Should not happen if matching_params is not empty and has Moyenne
                else:
                    print(f"Warning: 'Moyenne' column not found in matching_params for {type_prod_v2_key}. Defaulting to 0.0 for Capage Borne Max calculation.")
                
                if 'Max Facteur' in matching_params.columns:
                    max_facteur_series = pd.to_numeric(matching_params['Max Facteur'], errors='coerce')
                    if not max_facteur_series.empty:
                        max_facteur_val = max_facteur_series.iloc[0]
                        if pd.isna(max_facteur_val):
                            max_facteur_val = 0.0
                    else:
                        max_facteur_val = 0.0 # Should not happen
                else:
                    print(f"Warning: 'Max Facteur' column not found in matching_params for {type_prod_v2_key}. Defaulting to 0.0 for Capage Borne Max calculation.")
            
            # Ensure variation_absolue_pdc_calc is numeric before comparison
            numeric_variation_absolue_pdc = 0.0
            if isinstance(variation_absolue_pdc_calc, (int, float)):
                numeric_variation_absolue_pdc = variation_absolue_pdc_calc
            elif isinstance(variation_absolue_pdc_calc, str) and variation_absolue_pdc_calc.endswith('%'):
                try:
                    numeric_variation_absolue_pdc = float(variation_absolue_pdc_calc.strip('%')) / 100.0
                except ValueError:
                    numeric_variation_absolue_pdc = 0.0 # Or handle error as appropriate
            
            condition_L18_gt_0 = numeric_variation_absolue_pdc > 0
            condition_M4_eq_G4 = (moyenne_val == max_facteur_val)
            
            if condition_M4_eq_G4 and condition_L18_gt_0:
                capage_borne_max_val = "Oui"
            else:
                capage_borne_max_val = "Non"
        jour_livraison_formatted_str = "" # Valeur par défaut si NaT
        if pd.notna(jour_livraison_dt):
                    jour_livraison_formatted_str = jour_livraison_dt.strftime('%d/%m/%Y')
        table_data.append({
            "Type de produits V2": type_prod_v2_key,
            "Type de produits": type_prod,
            "Jour livraison": jour_livraison_formatted_str, # Formatting date as string
            "PDC": pdc_value,
            "En-cours": encours_value,
            "Commande optimisée": commande_optimisee_value,
            "Commande SM a 100%": commande_sm_100_value,
            # Formule: =SI(B18="";"";E18-G18-F18)
            # SI(Type de produits V2="";"";PDC-Commande optimisée-En-cours)            "Différence PDC / Commande": difference_pdc_commande,
            # Formule: =SI(B18="";"";ABS(H18))
            # SI(Type de produits V2="";"";ABS(Différence PDC / Commande))
            "Différence absolue": difference_absolue,
            # Formule: =SI(B18="";"";E18*'Macro-Param'!$C$6)
            # SI(Type de produits V2="";"";PDC*TAUX_SERVICE_AMONT_ESTIME)
            "Tolérance": tolerance,
            "Variation PDC": variation_pdc_calc,
            "Variation absolue PDC": variation_absolue_pdc_calc,
            "Capage Borne Max ?": capage_borne_max_val
        })

    # Create DataFrame
    df_pdc_simulation = pd.DataFrame(table_data)

    # Round and format numeric columns
    if 'PDC' in df_pdc_simulation.columns:
        # Ensure PDC is numeric before rounding and converting
        df_pdc_simulation['PDC'] = pd.to_numeric(df_pdc_simulation['PDC'], errors='coerce')
        # Handle potential NaNs from coerce before astype(int)
        df_pdc_simulation['PDC'] = df_pdc_simulation['PDC'].round(0).astype('Int64') # Use Int64 for nullable integers
        # Format PDC as a string with a space for the thousands separator
        df_pdc_simulation['PDC'] = df_pdc_simulation['PDC'].apply(lambda x: f"{x:,}".replace(",", " ") if pd.notna(x) else None)
        
    if 'En-cours' in df_pdc_simulation.columns:
        # Round En-cours to 2 decimal places
        df_pdc_simulation['En-cours'] = df_pdc_simulation['En-cours'].round(2)
    
    if 'Commande optimisée' in df_pdc_simulation.columns:
        df_pdc_simulation['Commande optimisée'] = pd.to_numeric(df_pdc_simulation['Commande optimisée'], errors='coerce').fillna(0).round(0).astype('Int64')
        # Optional: String formatting like PDC if desired by user later
        # df_pdc_simulation['Commande optimisée'] = df_pdc_simulation['Commande optimisée'].apply(lambda x: f"{x:,}".replace(",", " ") if pd.notna(x) else None)
    
    # Format the "Commande SM a 100%" column
    if 'Commande SM a 100%' in df_pdc_simulation.columns:
        df_pdc_simulation['Commande SM a 100%'] = pd.to_numeric(df_pdc_simulation['Commande SM a 100%'], errors='coerce').fillna(0).round(0).astype('Int64')
        # Format with spaces for thousands separator
        df_pdc_simulation['Commande SM a 100%'] = df_pdc_simulation['Commande SM a 100%'].apply(lambda x: f"{x:,}".replace(",", " ") if pd.notna(x) else None)
      
    # Calculer la colonne "Différence PDC / Commande" avec la formule =SI(B18="";"";E18-G18-F18)
    # Type de produits V2 est vide => résultat vide, sinon PDC - Commande optimisée - En-cours
    if 'Différence PDC / Commande' in df_pdc_simulation.columns:
        # Créer une nouvelle colonne temporaire pour stocker les valeurs calculées
        df_temp = pd.DataFrame()
        
        # Convertir temporairement les colonnes en numérique pour le calcul        df_temp['PDC_numeric'] = pd.to_numeric(df_pdc_simulation['PDC'].str.replace(' ', ''), errors='coerce').fillna(0)
        df_temp['Commande_numeric'] = pd.to_numeric(df_pdc_simulation['Commande optimisée'], errors='coerce').fillna(0)
        df_temp['Commande_SM_numeric'] = pd.to_numeric(df_pdc_simulation['Commande SM a 100%'].astype(str).str.replace(' ', ''), errors='coerce').fillna(0)
        df_temp['Encours_numeric'] = pd.to_numeric(df_pdc_simulation['En-cours'], errors='coerce').fillna(0)
        
        # Calculer la différence (on continue d'utiliser Commande optimisée pour la différence)
        df_temp['Diff'] = df_temp['PDC_numeric'] - df_temp['Commande_numeric'] - df_temp['Encours_numeric']# Formater la colonne finale avec des valeurs vides si Type de produits V2 est vide
        formatted_diff = []
        formatted_abs_diff = []
        formatted_tolerance = []
        for idx, row in df_pdc_simulation.iterrows():
            if row['Type de produits V2'] == "":
                formatted_diff.append("")
                formatted_abs_diff.append("")
                formatted_tolerance.append("")
            else:
                # Différence PDC / Commande
                diff_value = df_temp.loc[idx, 'Diff']
                formatted_diff.append(f"{int(diff_value):,}".replace(",", " "))
                
                # Valeur absolue pour la colonne "Différence absolue"
                abs_diff_value = abs(diff_value)
                formatted_abs_diff.append(f"{int(abs_diff_value):,}".replace(",", " "))
                  # Tolérance (PDC * TAUX_SERVICE_AMONT_ESTIME)
                pdc_value = df_temp.loc[idx, 'PDC_numeric']
                tolerance_value = pdc_value * TAUX_SERVICE_AMONT_ESTIME
                # Arrondir à l'entier et formater avec séparateur d'espaces pour les milliers
                formatted_tolerance.append(f"{int(round(tolerance_value)):,}".replace(",", " "))
        
        # Assigner les valeurs formatées aux colonnes
        df_pdc_simulation['Différence PDC / Commande'] = formatted_diff
        df_pdc_simulation['Différence absolue'] = formatted_abs_diff
        df_pdc_simulation['Tolérance'] = formatted_tolerance

    # Format Variation PDC as percentage
    if 'Variation PDC' in df_pdc_simulation.columns:
        df_pdc_simulation['Variation PDC'] = pd.to_numeric(df_pdc_simulation['Variation PDC'], errors='coerce')
        df_pdc_simulation['Variation PDC'] = df_pdc_simulation['Variation PDC'].round(2) # Round to two decimal places
        df_pdc_simulation['Variation PDC'] = df_pdc_simulation['Variation PDC'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
            
    # Format Variation absolue PDC as percentage
    if 'Variation absolue PDC' in df_pdc_simulation.columns:
        df_pdc_simulation['Variation absolue PDC'] = pd.to_numeric(df_pdc_simulation['Variation absolue PDC'], errors='coerce')
        # Ensure it's rounded to two decimal places for percentage formatting (consistent with Variation PDC)
        df_pdc_simulation['Variation absolue PDC'] = df_pdc_simulation['Variation absolue PDC'].round(2) # Changed from round(4)
        df_pdc_simulation['Variation absolue PDC'] = df_pdc_simulation['Variation absolue PDC'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")

    return df_pdc_simulation

def run_simulation(use_temp_params=False, temp_file_path=None, save_final=True):
    """
    Run the PDC simulation and save results to CSV.
    This function can be imported and called from other modules.
    
    Args:
        use_temp_params: If True, use parameters from temp_file_path instead of creating new ones
        temp_file_path: Path to temporary parameters file (used for optimization)
        save_final: Whether to save results to CSV file
        
    Returns:
        DataFrame with PDC simulation results
    """
    # Load required data and run simulation
    print("Loading Macro Parameters...")
    macro_params_data = MacroParam.load_macro_params()
    
    print("Loading PDC Perm data...")
    pdc_perm_raw = PDC.load_pdc_perm_data()
    
    print("Creating PDC Perm summary table...")
    pdc_perm_summary_df = PDC.create_pdc_perm_summary(pdc_perm_raw)
    
    # Load the parameters
    if use_temp_params and temp_file_path and os.path.exists(temp_file_path):
        print(f"Loading temporary parameters from {temp_file_path}")
        params_commandes_appros_df = pd.read_csv(temp_file_path, sep=';', encoding='latin1')
    else:
        # Load the parameters from optimisation
        params_commandes_appros_df = Optimisation.create_parametres_commandes_appros()
    
    # Add the SM column if it doesn't exist
    if SM_COL_DETAIL not in params_commandes_appros_df.columns:
        params_commandes_appros_df[SM_COL_DETAIL] = 0
        print(f"Added '{SM_COL_DETAIL}' column with zero values")
    
    print("Loading En-cours data (Synthèse Cde)...")
    synthese_cde_df = Encours.get_processed_data(formatted=False)
    
    print("Creating PDC - Simulation Commande table...")
    pdc_simulation_df = create_pdc_simulation_table(
        pdc_perm_raw,
        params_commandes_appros_df,
        synthese_cde_df,
        macro_params_data
    )
    
    # Save results to CSV if requested
    if save_final:
        # Use absolute paths to ensure consistent file locations
        output_path = os.path.join(os.path.dirname(__file__), 'PDC_Simulation_Commande.csv')
        if use_temp_params:
            # When using temporary parameters for optimization, use a temporary output file
            output_path = os.path.join(os.path.dirname(__file__), 'temp_PDC_Simulation_Commande.csv')
        
        pdc_simulation_df.to_csv(output_path, sep=';', index=False, encoding='latin1')
        print(f"Saved PDC simulation results to {output_path}")
    
    print("\n--- PDC - Simulation Commande ---")
    print(pdc_simulation_df[['Type de produits V2', 'Type de produits', 'Variation PDC', 'Variation absolue PDC', 'Capage Borne Max ?']])
    
    return pdc_simulation_df

if __name__ == '__main__':
    # This is for testing the module directly.
    # You'll need to load the data from other modules first.

    print("Loading Macro Parameters...")
    macro_params_data = MacroParam.load_macro_params()
    # print(f"DATE_COMMANDE from MacroParam: {MacroParam.get_param_value('DATE_COMMANDE', macro_params_data)}")

    print("Loading PDC Perm data...")
    pdc_perm_raw = PDC.load_pdc_perm_data() # Corrected: No arguments
    # print("PDC Perm Raw Head:\\n", pdc_perm_raw.head())
    
    # The create_pdc_perm_summary function expects the raw data from load_pdc_perm_data
    # and then it filters and calculates based on DATE_COMMANDE and taux_service_amont_estime.
    # For PDC_Sim, we need the *output* of create_pdc_perm_summary, which is the actual 'PDC Perm' table.
    print("Creating PDC Perm summary table...")
    # Corrected: create_pdc_perm_summary in PDC.py takes only one argument (the pdc_perm_raw data)
    # and fetches MacroParam internally.
    pdc_perm_summary_df = PDC.create_pdc_perm_summary(pdc_perm_raw)    # print("PDC Perm Summary Head:\\n", pdc_perm_summary_df.head())
    
   
    params_commandes_appros_df = Optimisation.create_parametres_commandes_appros() # Corrected: No arguments
    
    # Add the SM column if it doesn't exist
    if SM_COL_DETAIL not in params_commandes_appros_df.columns:
        params_commandes_appros_df[SM_COL_DETAIL] = 0
        print(f"Added '{SM_COL_DETAIL}' column with zero values")
    
    # print("Paramètres commandes appros Head:\\n", params_commandes_appros_df.head())

    print("Loading En-cours data (Synthèse Cde)...")
    # Assuming Encours.py has a function load_encours_data() that returns the 'Synthèse Cde' table
    # For testing, we might need a mock or to ensure Encours.py is complete and provides this.
    # Let's use the provided get_processed_data from Encours.py, ensuring it's not formatted for display.
    
    synthese_cde_df = Encours.get_processed_data(formatted=False) # Corrected function call
       

    print("Creating PDC - Simulation Commande table...")
    pdc_simulation_df = create_pdc_simulation_table(
        pdc_perm_raw, # Pass the raw data from PDC.load_pdc_perm_data()
        params_commandes_appros_df,
        synthese_cde_df,
        macro_params_data
    )

    print("\\n--- PDC - Simulation Commande ---")
    #save to CSV for testing
    #save csv 
    
    
    pdc_simulation_df.to_csv('PDC_Simulation_Commande.csv', sep=';', index=False)
    print(pdc_simulation_df)

