import pandas as pd
import os
import numpy as np
# Assurez-vous que MacroParam.py est dans le même répertoire ou accessible via PYTHONPATH
try:
    from MacroParam import get_param_value, get_arrondi_pcb_seuils
except ImportError:
    print("ERREUR CRITIQUE (CommandeFinale.py): MacroParam.py non trouvé. Certaines valeurs par défaut seront utilisées.")
    # Fonctions factices si MacroParam n'est pas là
    def get_param_value(param_name, default_value):
        if param_name == 'Casse Prev Activé': return "Oui" 
        return default_value
    def get_arrondi_pcb_seuils(type_prod_v2):
        if "A/B" in type_prod_v2: return 0.01, 0.5 
        if "A/C" in type_prod_v2: return 0.01, 0.8 
        return 0.01, 0.5



def get_total_cf_optimisee_vectorized(df_detail_subset, j_factor, k_factor, l_factor):
    """
    Calcule la SOMME des Commande Finale (CF) optimisée pour un sous-ensemble de lignes de détail,
    en utilisant les facteurs J, K, L fournis, de manière vectorielle.
    """
    if df_detail_subset.empty:
        return 0.0

    df = df_detail_subset.copy() # Travailler sur une copie pour ajouter des colonnes temporaires

    # --- 1. Calcul Vectoriel de BY (Facteur multiplicatif Lissage besoin brut) ---
    bw_borne_min = pd.to_numeric(df.get('Borne Min Facteur multiplicatif lissage', 0), errors='coerce').fillna(0.0)
    bx_borne_max = pd.to_numeric(df.get('Borne Max Facteur multiplicatif lissage', 10), errors='coerce').fillna(10.0)
    # Assurer que Top est traité correctement (déjà en minuscules grâce à load_data)
    bn_top_category = df.get('Top', pd.Series(['autre'] * len(df), index=df.index)).fillna('autre')

    conditions = [
        bn_top_category == 'top 500',
        bn_top_category == 'top 3000',
        bn_top_category == 'autre'
    ]
    choices = [j_factor, k_factor, l_factor]
    facteur_optim_selon_top = np.select(conditions, choices, default=l_factor) # l_factor si 'autre' ou non trouvé
    
    df['BY_sim'] = np.maximum(bw_borne_min, np.minimum(bx_borne_max, facteur_optim_selon_top))

    # --- 2. Calcul Vectoriel de CD (Commande optimisée sans arrondi) ---
    bp_cmd_sm100_arrondi = pd.to_numeric(df.get('Commande Finale avec mini et arrondi SM à 100%', 0), errors='coerce').fillna(0.0)
    
    # Condition initiale pour CD
    cd_initial_zeros = (bp_cmd_sm100_arrondi == 0) & (df['BY_sim'] <= 1)
    
    ak_mini_pub = pd.to_numeric(df.get('Mini Publication FL', 0), errors='coerce').fillna(0.0)
    ad_coche_rao = pd.to_numeric(df.get('COCHE_RAO', 0), errors='coerce').fillna(0.0)
    aj_stock_reel = pd.to_numeric(df.get('STOCK_REEL', 0), errors='coerce').fillna(0.0)
    z_ral = pd.to_numeric(df.get('RAL', 0), errors='coerce').fillna(0.0)
    
    besoin_net_cd = ak_mini_pub - aj_stock_reel - z_ral
    besoin_net_cd[(ak_mini_pub <= 0) | (ad_coche_rao == 46)] = 0 # Appliquer conditions
    besoin_net_cd = np.maximum(0, besoin_net_cd) # Assurer non négatif ici aussi si la logique le veut

    o_sm_final = pd.to_numeric(df.get('SM Final', 0), errors='coerce').fillna(0.0)
    bf_prev_c1_l2 = pd.to_numeric(df.get('Prév C1-L2 Finale', 0), errors='coerce').fillna(0.0)
    bg_prev_l1_l2 = pd.to_numeric(df.get('Prév L1-L2 Finale', 0), errors='coerce').fillna(0.0)
    bu_facteur_appro = pd.to_numeric(df.get('Facteur Multiplicatif Appro', 1), errors='coerce').fillna(1.0)
    bu_facteur_appro[bu_facteur_appro == 0] = 1.0 # Éviter division par zéro ou facteur nul si 0 est invalide
    
    ay_casse_c1_l2 = pd.to_numeric(df.get('Casse Prev C1-L2', 0), errors='coerce').fillna(0.0)
    az_casse_l1_l2 = pd.to_numeric(df.get('Casse Prev L1-L2', 0), errors='coerce').fillna(0.0)
    cc_cmd_max_stock_max = pd.to_numeric(df.get('Commande Max avec stock max', 99999999), errors='coerce').fillna(99999999.0)
    
    bi_produit_bloque_str = df.get('Produit Bloqué', pd.Series(['Non'] * len(df), index=df.index)).fillna('Non').astype(str).str.lower()
    bi_produit_bloque = bi_produit_bloque_str.isin(['vrai', 'true', 'oui', 'yes', '1'])

    # Position JUD - plus complexe à vectoriser proprement si ERREUR() est clé.
    # Pour l'instant, on suppose que si P JUD est problématique, les scénarios sont 0.
    # Une approche simplifiée pour p_position_jud_is_error :
    position_jud_val = pd.to_numeric(df.get('Position JUD'), errors='coerce')
    p_position_jud_is_error = position_jud_val.isna() | (position_jud_val <= 0)

    casse_prev_active_str = get_param_value('Casse Prev Activé', 'Oui')
    casse_prev_active = (casse_prev_active_str == "Oui")

    # Scenarios
    ajustement1_cd = (-aj_stock_reel - z_ral)
    if casse_prev_active:
        ajustement1_cd = -(aj_stock_reel - ay_casse_c1_l2) - z_ral
    scenario1_cd = np.maximum(0, (o_sm_final + bf_prev_c1_l2) * bu_facteur_appro * df['BY_sim'] + ajustement1_cd)
    scenario1_cd[p_position_jud_is_error] = 0 # Si JUD est erreur, scenario = 0

    ajustement2_cd = pd.Series(np.zeros(len(df)), index=df.index)
    if casse_prev_active:
        ajustement2_cd = az_casse_l1_l2
    scenario2_cd = np.maximum(0, (o_sm_final + bg_prev_l1_l2) * bu_facteur_appro * df['BY_sim'] + ajustement2_cd)
    scenario2_cd[p_position_jud_is_error] = 0

    quantite_stock_max_scenarios = np.minimum(scenario1_cd, scenario2_cd)
    quantite_stock_max_scenarios[bi_produit_bloque] = 0

    valeur_avant_max_besoin_net = np.minimum(cc_cmd_max_stock_max, quantite_stock_max_scenarios)
    df['CD_sim'] = np.maximum(besoin_net_cd, valeur_avant_max_besoin_net)
    df.loc[cd_initial_zeros, 'CD_sim'] = 0.0 # Appliquer la condition initiale
    df['CD_sim'] = np.maximum(0, df['CD_sim']) # Assurer non négatif

    # --- 3. Calcul Vectoriel de CE ---
    v_min_commande = pd.to_numeric(df.get('MINIMUM_COMMANDE', 0), errors='coerce').fillna(0.0)
    w_pcb = pd.to_numeric(df.get('PCB', 1), errors='coerce').fillna(1.0)
    w_pcb[w_pcb == 0] = 1.0 # Éviter division par zéro

    # get_arrondi_pcb_seuils doit être appliquée ligne par ligne ou adaptée
    # Pour simplifier ici, on prend des valeurs moyennes ou les plus communes
    # Idéalement, get_arrondi_pcb_seuils retournerait deux Series ou on mapperait
    # Utiliser 'BM' qui est le nom interne pour 'Type de produits V2' dans df_detail
    col_type_prod_v2_dans_detail = 'BM' 
    if col_type_prod_v2_dans_detail not in df.columns:
        print(f"ERREUR CRITIQUE dans get_total_cf_optimisee_vectorized: Colonne '{col_type_prod_v2_dans_detail}' non trouvée dans le DataFrame de détail subset.")
        # Fallback ou lever une exception
        # Pour un fallback simple, on pourrait assigner des seuils par défaut, mais c'est risqué.
        # Pour l'instant, on va créer des Series vides pour éviter d'autres erreurs, mais F_sim sera probablement faux.
        seuil1 = pd.Series(np.nan, index=df.index)
        seuil2 = pd.Series(np.nan, index=df.index)
    else:
        # S'assurer que la colonne est de type str avant .apply
        type_prod_series = df[col_type_prod_v2_dans_detail].fillna('').astype(str)
        seuils_tuples = type_prod_series.apply(get_arrondi_pcb_seuils)
        
        # Convertir la Series de tuples en deux Series distinctes pour seuil1 et seuil2
        seuil1 = seuils_tuples.apply(lambda x: x[0] if isinstance(x, tuple) and len(x) == 2 else np.nan)
        seuil2 = seuils_tuples.apply(lambda x: x[1] if isinstance(x, tuple) and len(x) == 2 else np.nan)

    # S'assurer que seuil1 et seuil2 sont bien des Series numériques pour les opérations suivantes
    seuil1 = pd.to_numeric(seuil1, errors='coerce').fillna(0.01) # Remplacer NaN par un défaut
    seuil2 = pd.to_numeric(seuil2, errors='coerce').fillna(0.5)  # Remplacer NaN par un défaut

    df['CE_sim'] = 0.0 # Initialiser
    condition_ce_non_zero = ~( (pd.isna(df['CD_sim'])) | (df['CD_sim'] == 0) )
    
    if condition_ce_non_zero.any():
        cd_subset = df.loc[condition_ce_non_zero, 'CD_sim']
        v_min_subset = v_min_commande[condition_ce_non_zero]
        w_pcb_subset = w_pcb[condition_ce_non_zero]
        seuil1_subset = seuil1[condition_ce_non_zero]
        seuil2_subset = seuil2[condition_ce_non_zero]

        diviseur_seuil1_subset = np.maximum(v_min_subset, w_pcb_subset)
        diviseur_seuil1_subset[diviseur_seuil1_subset == 0] = 1.0

        condition_seuil1_ok = (cd_subset / diviseur_seuil1_subset) >= seuil1_subset
        
        val_arrondie_subset = pd.Series(np.zeros(len(cd_subset)), index=cd_subset.index)

        cd_div_pcb_subset = cd_subset / w_pcb_subset
        
        # Cas cd_div_pcb >= 1
        mask_ge1 = (cd_div_pcb_subset >= 1) & condition_seuil1_ok
        if mask_ge1.any():
            partie_decimale = cd_div_pcb_subset[mask_ge1] - np.floor(cd_div_pcb_subset[mask_ge1])
            arr_inf = np.floor(cd_div_pcb_subset[mask_ge1]) * w_pcb_subset[mask_ge1]
            arr_sup = np.ceil(cd_div_pcb_subset[mask_ge1]) * w_pcb_subset[mask_ge1]
            val_arrondie_subset[mask_ge1] = np.where(partie_decimale < seuil2_subset[mask_ge1], arr_inf, arr_sup)

        # Cas cd_div_pcb < 1
        mask_lt1 = (cd_div_pcb_subset < 1) & condition_seuil1_ok
        if mask_lt1.any():
            # Sub-cas v_min_commande > pcb
            mask_v_gt_w = mask_lt1 & (v_min_subset > w_pcb_subset)
            if mask_v_gt_w.any():
                cd_div_min_cmd = cd_subset[mask_v_gt_w] / v_min_subset[mask_v_gt_w]
                partie_decimale_v = cd_div_min_cmd - np.floor(cd_div_min_cmd)
                arr_inf_v = np.floor(cd_div_min_cmd) * v_min_subset[mask_v_gt_w]
                arr_sup_v = np.ceil(cd_div_min_cmd) * v_min_subset[mask_v_gt_w]
                val_arrondie_subset[mask_v_gt_w] = np.where(partie_decimale_v < seuil1_subset[mask_v_gt_w], arr_inf_v, arr_sup_v)
            
            # Sub-cas v_min_commande <= pcb (et cd_div_pcb < 1)
            mask_v_le_w = mask_lt1 & ~(v_min_subset > w_pcb_subset) # Le reste de mask_lt1
            if mask_v_le_w.any():
                 val_arrondie_subset[mask_v_le_w] = np.ceil(cd_div_pcb_subset[mask_v_le_w]) * w_pcb_subset[mask_v_le_w]
        
        df.loc[condition_ce_non_zero & condition_seuil1_ok, 'CE_sim'] = np.maximum(v_min_subset[condition_seuil1_ok], val_arrondie_subset[condition_seuil1_ok])
        df.loc[condition_ce_non_zero & ~condition_seuil1_ok, 'CE_sim'] = 0.0
    # --- 4. Calcul Vectoriel de CF ---
    bj_ts = pd.to_numeric(df.get('TS', 1), errors='coerce').fillna(1.0)
    cf_raw = df['CE_sim'].fillna(0.0) * bj_ts
    
    # Apply rounding logic: if < 1, make it 0; if >= 1, round down to integer
    df['CF_sim'] = np.where(cf_raw < 1, 0, np.floor(cf_raw))

    return df['CF_sim'].sum()

def calculate_cg_commande_optimisee_en_colis(df, df_qtemaxi_map=None): # Colonne CG
    print("Calcul de 'Commande optimisée avec arrondi et mini en colis' (CG)...")
    required_cols_cg = [
        'Quantité Maxi à charger ?',      # CI
        'Commande optimisée avec arrondi et mini', # CE
        'FOURNISSEUR',                    # Q 
        'IFLS',                           # AI
        'PCB'                             # W
    ]
    missing_cols = [col for col in required_cols_cg if col not in df.columns]
    if missing_cols:
        print(f"  ERREUR: Colonnes manquantes pour CG: {missing_cols}")
        df['Commande optimisée avec arrondi et mini en colis'] = 0
        return df

    if df_qtemaxi_map is None:
        script_dir = os.path.dirname(__file__)
        # Le dossier CSV est relatif au script actuel (CommandeFinale.py)
        # Si optimisation_globale.py est dans un autre dossier, ce chemin pourrait ne pas être bon
        # Il est plus sûr de passer le chemin du dossier CSV ou de le définir globalement
        csv_folder_path = os.path.join(script_dir, "CSV") 
        qtemaxi_file_path = os.path.join(csv_folder_path, "Qtite_Maxi.csv") 
        df_qtemaxi_map = {} 
        if os.path.exists(qtemaxi_file_path):
            try:
                # print(f"  Chargement de {qtemaxi_file_path}...")
                try: df_qtemaxi_temp = pd.read_csv(qtemaxi_file_path, sep=';', encoding='utf-8')
                except UnicodeDecodeError: df_qtemaxi_temp = pd.read_csv(qtemaxi_file_path, sep=';', encoding='latin1')
                
                key_col_qtemaxi = 'ID2'; value_col_qtemaxi = 'Nombre Maxi Unit�s De Commande' 
                if key_col_qtemaxi in df_qtemaxi_temp.columns and value_col_qtemaxi in df_qtemaxi_temp.columns:
                    df_qtemaxi_temp[key_col_qtemaxi] = df_qtemaxi_temp[key_col_qtemaxi].astype(str).str.strip()
                    df_qtemaxi_temp[value_col_qtemaxi] = pd.to_numeric(df_qtemaxi_temp[value_col_qtemaxi], errors='coerce')
                    df_qtemaxi_map_builder = df_qtemaxi_temp.dropna(subset=[value_col_qtemaxi])
                    df_qtemaxi_map = pd.Series(df_qtemaxi_map_builder[value_col_qtemaxi].values, index=df_qtemaxi_map_builder[key_col_qtemaxi]).to_dict()
                    # print(f"  Dictionnaire Qtite_Maxi chargé ({len(df_qtemaxi_map)} entrées).")
                # else: print(f"  ATTENTION: Colonnes clés manquantes dans {os.path.basename(qtemaxi_file_path)}.")
            except Exception as e: print(f"  ERREUR chargement {os.path.basename(qtemaxi_file_path)}: {e}.")
        # else: print(f"  ATTENTION: Fichier {os.path.basename(qtemaxi_file_path)} non trouvé.")
        # if not df_qtemaxi_map: print("    CG sera calculé sans Qtite_Maxi.")

    def calculate_cg_row(row):
        try:
            quantite_maxi_a_charger_ci = str(row.get('Quantité Maxi à charger ?', 'Non')).strip().lower()
            commande_optim_ce = pd.to_numeric(row.get('Commande optimisée avec arrondi et mini'), errors='coerce')
            pcb_w = pd.to_numeric(row.get('PCB'), errors='coerce')
            if pd.isna(commande_optim_ce): return 0
            if pd.isna(pcb_w) or pcb_w == 0: pcb_w = 1 
            if quantite_maxi_a_charger_ci == "oui":
                entrepot_q = str(row.get('FOURNISSEUR', '')).strip(); ifls_ai = str(row.get('IFLS', '')).strip()
                cle_qtemaxi = f"{entrepot_q}-{ifls_ai}" 
                valeur_qtemaxi_f = df_qtemaxi_map.get(cle_qtemaxi)
                if valeur_qtemaxi_f is not None and pd.notna(valeur_qtemaxi_f) and valeur_qtemaxi_f != 0:
                    return commande_optim_ce / valeur_qtemaxi_f
                return commande_optim_ce / pcb_w
            return commande_optim_ce / pcb_w
        except Exception: return 0 
    df['Commande optimisée avec arrondi et mini en colis'] = df.apply(calculate_cg_row, axis=1)
    # print("  Colonne 'Commande optimisée avec arrondi et mini en colis' (CG) calculée.")
    return df

# --- Fonctions de calcul ligne par ligne (AJOUTÉES/ADAPTÉES) ---
def _calculate_by_facteur_lissage_besoin_brut_row(detail_row_dict, j_factor_optim, k_factor_optim, l_factor_optim):
    bw_borne_min = pd.to_numeric(detail_row_dict.get('Borne Min Facteur multiplicatif lissage', 0), errors='coerce')
    bx_borne_max = pd.to_numeric(detail_row_dict.get('Borne Max Facteur multiplicatif lissage', 10), errors='coerce') 
    bn_top_category = str(detail_row_dict.get('Top','autre')).strip().lower()
    if pd.isna(bw_borne_min): bw_borne_min = 0.0
    if pd.isna(bx_borne_max): bx_borne_max = 10.0 
    facteur_optim_selon_top = 1.0
    if bn_top_category == 'top 500': facteur_optim_selon_top = j_factor_optim
    elif bn_top_category == 'top 3000': facteur_optim_selon_top = k_factor_optim
    elif bn_top_category == 'autre': facteur_optim_selon_top = l_factor_optim
    by_facteur = max(bw_borne_min, min(bx_borne_max, facteur_optim_selon_top))
    return by_facteur

def _calculate_cd_commande_optimisee_sans_arrondi_row(detail_row_dict, by_facteur_lissage):
    bp_cmd_sm100_arrondi = pd.to_numeric(detail_row_dict.get('Commande Finale avec mini et arrondi SM à 100%', 0), errors='coerce')
    if pd.isna(bp_cmd_sm100_arrondi): bp_cmd_sm100_arrondi = 0
    if bp_cmd_sm100_arrondi == 0 and by_facteur_lissage <= 1: return 0
    ak_mini_pub = pd.to_numeric(detail_row_dict.get('Mini Publication FL', 0), errors='coerce')
    ad_coche_rao = pd.to_numeric(detail_row_dict.get('COCHE_RAO', 0), errors='coerce')
    aj_stock_reel = pd.to_numeric(detail_row_dict.get('STOCK_REEL', 0), errors='coerce')
    z_ral = pd.to_numeric(detail_row_dict.get('RAL', 0), errors='coerce')
    if pd.isna(ak_mini_pub): ak_mini_pub = 0; 
    if pd.isna(ad_coche_rao): ad_coche_rao = 0; 
    if pd.isna(aj_stock_reel): aj_stock_reel = 0; 
    if pd.isna(z_ral): z_ral = 0;
    besoin_net_cd = 0
    if not (ak_mini_pub <= 0 or ad_coche_rao == 46): besoin_net_cd = ak_mini_pub - aj_stock_reel - z_ral
    o_sm_final = pd.to_numeric(detail_row_dict.get('SM Final', 0), errors='coerce')
    bf_prev_c1_l2 = pd.to_numeric(detail_row_dict.get('Prév C1-L2 Finale', 0), errors='coerce')
    bg_prev_l1_l2 = pd.to_numeric(detail_row_dict.get('Prév L1-L2 Finale', 0), errors='coerce')
    bu_facteur_appro = pd.to_numeric(detail_row_dict.get('Facteur Multiplicatif Appro', 1), errors='coerce')
    ay_casse_c1_l2 = pd.to_numeric(detail_row_dict.get('Casse Prev C1-L2', 0), errors='coerce')
    az_casse_l1_l2 = pd.to_numeric(detail_row_dict.get('Casse Prev L1-L2', 0), errors='coerce')
    cc_cmd_max_stock_max = pd.to_numeric(detail_row_dict.get('Commande Max avec stock max', 99999999), errors='coerce')
    bi_produit_bloque_str = str(detail_row_dict.get('Produit Bloqué', 'Non')).lower()
    bi_produit_bloque = bi_produit_bloque_str in ['vrai', 'true', 'oui', 'yes', '1', True]
    position_jud_val = detail_row_dict.get('Position JUD', None)
    p_position_jud_is_error = pd.isna(position_jud_val) or (pd.to_numeric(position_jud_val, errors='coerce') <= 0 if pd.notna(position_jud_val) and str(position_jud_val).strip() != "" else True)
    casse_prev_active = get_param_value('Casse Prev Activé', 'Oui') 
    if pd.isna(o_sm_final): o_sm_final = 0; 
    if pd.isna(bf_prev_c1_l2): bf_prev_c1_l2 = 0; 
    if pd.isna(bg_prev_l1_l2): bg_prev_l1_l2 = 0;
    if pd.isna(bu_facteur_appro) or bu_facteur_appro == 0 : bu_facteur_appro = 1;
    if pd.isna(ay_casse_c1_l2): ay_casse_c1_l2 = 0; 
    if pd.isna(az_casse_l1_l2): az_casse_l1_l2 = 0; 
    if pd.isna(cc_cmd_max_stock_max): cc_cmd_max_stock_max = 99999999;
    quantite_stock_max_scenarios = 0
    if bi_produit_bloque: quantite_stock_max_scenarios = 0
    else:
        scenario1_cd = 0
        if not p_position_jud_is_error:
            ajustement1_cd = (-aj_stock_reel - z_ral); 
            if casse_prev_active == "Oui": ajustement1_cd = -(aj_stock_reel - ay_casse_c1_l2) - z_ral
            scenario1_cd = max(0, (o_sm_final + bf_prev_c1_l2) * bu_facteur_appro * by_facteur_lissage + ajustement1_cd)
        scenario2_cd = 0
        if not p_position_jud_is_error:
            ajustement2_cd = 0; 
            if casse_prev_active == "Oui": ajustement2_cd = az_casse_l1_l2 
            scenario2_cd = max(0, (o_sm_final + bg_prev_l1_l2) * bu_facteur_appro * by_facteur_lissage + ajustement2_cd)
        quantite_stock_max_scenarios = min(scenario1_cd, scenario2_cd)
    valeur_avant_max_besoin_net = min(cc_cmd_max_stock_max, quantite_stock_max_scenarios)
    resultat_cd = max(besoin_net_cd, valeur_avant_max_besoin_net)
    return max(0, resultat_cd)

def _calculate_ce_commande_optimisee_avec_arrondi_et_mini_row(detail_row_dict, cd_commande_optim_sans_arrondi):
    if pd.isna(cd_commande_optim_sans_arrondi) or cd_commande_optim_sans_arrondi == 0: return 0
    v_min_commande = pd.to_numeric(detail_row_dict.get('MINIMUM_COMMANDE', 0), errors='coerce')
    w_pcb = pd.to_numeric(detail_row_dict.get('PCB', 1), errors='coerce') 
    if pd.isna(v_min_commande): v_min_commande = 0
    if pd.isna(w_pcb) or w_pcb == 0: w_pcb = 1
    bm_type_prod_v2 = str(detail_row_dict.get('Type de produits V2', '')).strip()
    seuil1, seuil2 = get_arrondi_pcb_seuils(bm_type_prod_v2) 
    diviseur_seuil1 = max(v_min_commande, w_pcb); 
    if diviseur_seuil1 == 0 : diviseur_seuil1 = 1 
    if cd_commande_optim_sans_arrondi / diviseur_seuil1 < seuil1: return 0
    else:
        val_arrondie = 0; cd_div_pcb = cd_commande_optim_sans_arrondi / w_pcb
        if cd_div_pcb >= 1:
            partie_decimale = cd_div_pcb - np.floor(cd_div_pcb)
            if partie_decimale < seuil2: val_arrondie = np.floor(cd_div_pcb) * w_pcb
            else: val_arrondie = np.ceil(cd_div_pcb) * w_pcb
        else: 
            if v_min_commande > w_pcb:
                cd_div_min_cmd = cd_commande_optim_sans_arrondi / v_min_commande
                partie_decimale_v = cd_div_min_cmd - np.floor(cd_div_min_cmd)
                if partie_decimale_v < seuil1: val_arrondie = np.floor(cd_div_min_cmd) * v_min_commande
                else: val_arrondie = np.ceil(cd_div_min_cmd) * v_min_commande
            else: val_arrondie = np.ceil(cd_div_pcb) * w_pcb
        return max(v_min_commande, val_arrondie)

def _calculate_cf_commande_optimisee_avec_arrondi_mini_ts_row(detail_row_dict, ce_commande_optim_arrondi_mini):
    if pd.isna(ce_commande_optim_arrondi_mini): return 0
    bj_ts = pd.to_numeric(detail_row_dict.get('TS', 1), errors='coerce')
    if pd.isna(bj_ts): bj_ts = 1 
    
    result = ce_commande_optim_arrondi_mini * bj_ts
    
    # Apply rounding logic: if < 1, make it 0; if >= 1, round down to integer
    if result < 1:
        return 0
    else:
        return int(np.floor(result))

# --- NOUVELLE FONCTION D'ORCHESTRATION LIGNE PAR LIGNE ---
def get_cf_optimisee_for_detail_line(detail_row_as_dict, j_factor, k_factor, l_factor):
    """
    Calcule la Commande Finale (CF) optimisée pour une seule ligne de détail,
    en utilisant les facteurs J, K, L fournis.
    Cette fonction sera appelée par optimisation_globale.py.
    """
    by = _calculate_by_facteur_lissage_besoin_brut_row(detail_row_as_dict, j_factor, k_factor, l_factor)
    cd = _calculate_cd_commande_optimisee_sans_arrondi_row(detail_row_as_dict, by)
    ce = _calculate_ce_commande_optimisee_avec_arrondi_et_mini_row(detail_row_as_dict, cd)
    cf = _calculate_cf_commande_optimisee_avec_arrondi_mini_ts_row(detail_row_as_dict, ce)
    return cf

# --- Fonctions existantes opérant sur DataFrame (BO, BP, BQ, puis CD, CE, CF via _row) ---

def calculate_commande_optimisee_sans_arrondi(df): # CD via _row
    print("Calcul de 'Commande optimisée sans arrondi' (CD)...")
    # ... (vérification colonnes comme avant) ...
    # Assurez-vous que 'Facteur multiplicatif Lissage besoin brut' (BY) est dans df
    if 'Facteur multiplicatif Lissage besoin brut' not in df.columns:
        print("  ERREUR (CD): Colonne 'Facteur multiplicatif Lissage besoin brut' (BY) manquante.")
        df['Commande optimisée sans arrondi'] = 0
        return df
        
    df['Commande optimisée sans arrondi'] = df.apply(
        lambda row: _calculate_cd_commande_optimisee_sans_arrondi_row(
            row.to_dict(), 
            row['Facteur multiplicatif Lissage besoin brut']
        ), axis=1
    )
    # print("  Colonne 'Commande optimisée sans arrondi' (CD) calculée.")
    return df

def calculate_ce_commande_optimisee_avec_arrondi_et_mini(df): # CE via _row
    print("Calcul de 'Commande optimisée avec arrondi et mini' (CE)...")
    # ... (vérification colonnes comme avant) ...
    if 'Commande optimisée sans arrondi' not in df.columns:
        print("  ERREUR (CE): Colonne 'Commande optimisée sans arrondi' (CD) manquante.")
        df['Commande optimisée avec arrondi et mini'] = 0
        return df

    df['Commande optimisée avec arrondi et mini'] = df.apply(
        lambda row: _calculate_ce_commande_optimisee_avec_arrondi_et_mini_row(
            row.to_dict(), 
            row['Commande optimisée sans arrondi']
        ), axis=1
    )
    # print("  Colonne 'Commande optimisée avec arrondi et mini' (CE) calculée.")
    return df

def calculate_cf_commande_optimisee_avec_arrondi_mini_ts(df): # CF via _row
    print("Calcul de 'Commande optimisée avec arrondi et mini et TS' (CF)...")
    # ... (vérification colonnes comme avant) ...
    if 'Commande optimisée avec arrondi et mini' not in df.columns:
        print("  ERREUR (CF): Colonne 'Commande optimisée avec arrondi et mini' (CE) manquante.")
        df['Commande optimisée avec arrondi et mini et TS'] = 0
        return df

    df['Commande optimisée avec arrondi et mini et TS'] = df.apply(
        lambda row: _calculate_cf_commande_optimisee_avec_arrondi_mini_ts_row(
            row.to_dict(), 
            row['Commande optimisée avec arrondi et mini']
        ), axis=1
    )
    # print("  Colonne 'Commande optimisée avec arrondi et mini et TS' (CF) calculée.")
    return df

# Les fonctions pour BO, BP, BQ peuvent rester telles quelles car elles n'utilisent pas BY
def calculate_commande_finale_sans_mini_ni_arrondi(df): # BO
    print("Calcul de 'Commande Finale sans mini ni arrondi SM à 100%' (BO)...")
    
    # Determine if 'Casse Prev Activé' from MacroParam once
    casse_prev_active_str = get_param_value('Casse Prev Activé', 'Oui')
    casse_prev_active = (casse_prev_active_str.lower() == "oui") if isinstance(casse_prev_active_str, str) else False

    # Check for required columns and add them with default 0 if missing (helps prevent KeyErrors in .get)
    required_columns_bo = [
        'Mini Publication FL', 'COCHE_RAO', 'RAL', 'STOCK_REEL', 'SM Final', 
        'Prév C1-L2 Finale', 'Prév L1-L2 Finale', 'Facteur Multiplicatif Appro',
        'Casse Prev C1-L2', 'Casse Prev L1-L2', 'Produit Bloqué', 
        'Commande Max avec stock max', 'Position JUD', 'CODE_METI' # CODE_METI for debugging
    ]
    for col_name in required_columns_bo:
        if col_name not in df.columns:
            print(f"  ATTENTION (BO): Colonne '{col_name}' manquante. Ajout avec valeur par défaut (0 ou False).")
            if col_name == 'Produit Bloqué':
                df[col_name] = False
            else:
                df[col_name] = 0
    
    has_position_jud_col = 'Position JUD' in df.columns

    def calculate_bo_row(row):
        try:
            # --- Extract and clean input values ---
            mini_publication = pd.to_numeric(row.get('Mini Publication FL'), errors='coerce')
            if pd.isna(mini_publication): mini_publication = 0.0
            
            coche_rao = pd.to_numeric(row.get('COCHE_RAO'), errors='coerce')
            if pd.isna(coche_rao): coche_rao = 0.0
            
            ral = pd.to_numeric(row.get('RAL'), errors='coerce')
            if pd.isna(ral): ral = 0.0
            
            stock_reel = pd.to_numeric(row.get('STOCK_REEL'), errors='coerce')
            if pd.isna(stock_reel): stock_reel = 0.0
            
            sm_final = pd.to_numeric(row.get('SM Final'), errors='coerce')
            if pd.isna(sm_final): sm_final = 0.0
            
            prev_c1_l2 = pd.to_numeric(row.get('Prév C1-L2 Finale'), errors='coerce')
            if pd.isna(prev_c1_l2): prev_c1_l2 = 0.0
            
            prev_l1_l2 = pd.to_numeric(row.get('Prév L1-L2 Finale'), errors='coerce')
            if pd.isna(prev_l1_l2): prev_l1_l2 = 0.0
            
            facteur_appro_bu = pd.to_numeric(row.get('Facteur Multiplicatif Appro'), errors='coerce')
            if pd.isna(facteur_appro_bu) or facteur_appro_bu == 0: facteur_appro_bu = 1.0 # Avoid 0 factor
            
            casse_c1_l2_ay = pd.to_numeric(row.get('Casse Prev C1-L2'), errors='coerce')
            if pd.isna(casse_c1_l2_ay): casse_c1_l2_ay = 0.0
            
            casse_l1_l2_az = pd.to_numeric(row.get('Casse Prev L1-L2'), errors='coerce')
            if pd.isna(casse_l1_l2_az): casse_l1_l2_az = 0.0
            
            produit_bloque_bi_val = row.get('Produit Bloqué', False) # Default to False if column missing
            if isinstance(produit_bloque_bi_val, str): 
                produit_bloque_bi = produit_bloque_bi_val.lower() in ['true', 'oui', 'yes', '1', 'vrai']
            elif isinstance(produit_bloque_bi_val, bool):
                produit_bloque_bi = produit_bloque_bi_val
            else: # try to convert to numeric then bool
                produit_bloque_bi_num = pd.to_numeric(produit_bloque_bi_val, errors='coerce')
                produit_bloque_bi = bool(produit_bloque_bi_num) if pd.notna(produit_bloque_bi_num) else False

            cmd_max_cc = pd.to_numeric(row.get('Commande Max avec stock max'), errors='coerce')
            if pd.isna(cmd_max_cc): cmd_max_cc = 99999999.0 # Default to a large number
            
            position_jud_p_val = pd.to_numeric(row.get('Position JUD'), errors='coerce') # Defaulted to 0 if missing by pre-check
            position_jud_is_error = pd.isna(position_jud_p_val)


            # --- Start Calculation Logic ---
            # If product is blocked, command is 0
            if produit_bloque_bi:
                return 0.0
            
            # Calculate besoin_net
            # SI(OU(AK2<=0;AD2=46);0;AK2-SI(AJ2="";0;AJ2)-SI(Z2="";0;Z2))
            if mini_publication <= 0 or coche_rao == 46:
                besoin_net = 0.0
            else:
                besoin_net = mini_publication - stock_reel - ral
            
            # Calculate stock_max_scenarios
            quantite_stock_max_scenarios = 0.0 # Default if JUD is error
            
            if not position_jud_is_error:
                # Scenario 1: MAX(0,(O2+BF2)*BU2+SI('Macro-Param'!$C$16="Oui";-(AJ2-AY2)-Z2;-AJ2-Z2)))
                if casse_prev_active:
                    ajustement1 = -(stock_reel - casse_c1_l2_ay) - ral
                else:
                    ajustement1 = -stock_reel - ral
                scenario1 = max(0.0, (sm_final + prev_c1_l2) * facteur_appro_bu + ajustement1)
                
                # Scenario 2: MAX(0,(O2+BG2)*BU2+SI('Macro-Param'!$C$16="Oui";AZ2;0)))
                if casse_prev_active:
                    ajustement2 = casse_l1_l2_az
                else:
                    ajustement2 = 0.0
                scenario2 = max(0.0, (sm_final + prev_l1_l2) * facteur_appro_bu + ajustement2)
                
                quantite_stock_max_scenarios = min(scenario1, scenario2)
        
            # MIN(CC2, quantite_stock_max_scenarios)
            valeur_avant_max_besoin_net = min(cmd_max_cc, quantite_stock_max_scenarios)
            
            # Final MAX(besoin_net, valeur_avant_max_besoin_net)
            resultat_bo = max(besoin_net, valeur_avant_max_besoin_net)
            
            return max(0.0, resultat_bo) # Ensure final result is not negative

        except Exception as e:
            code_meti_ex = row.get('CODE_METI', 'UNKNOWN_CODE_METI')
            print(f"  ERREUR (BO) in calculate_bo_row for CODE_METI {code_meti_ex}: {e}")
            import traceback
            # traceback.print_exc() # Uncomment for full stack trace during debugging
            return 0.0 # Default to 0 on any error

    df['Commande Finale sans mini ni arrondi SM à 100%'] = df.apply(calculate_bo_row, axis=1)
    print("  Colonne 'Commande Finale sans mini ni arrondi SM à 100%' (BO) calculée.")
    return df


def calculate_commande_finale_avec_mini_et_arrondi(df): # BP
    # ... (votre code existant pour BP) ...
    print("Calcul de 'Commande Finale avec mini et arrondi SM à 100%' (BP)...")
    required_columns_bp = ['Type de produits V2', 'Commande Finale sans mini ni arrondi SM à 100%', 'PCB', 'MINIMUM_COMMANDE']
    missing_columns = [col for col in required_columns_bp if col not in df.columns]
    if missing_columns:
        print(f"  ERREUR: Colonnes manquantes pour BP: {missing_columns}")
        for col_miss in missing_columns: df[col_miss] = 0
    def calculate_bp_row(row):
        try:
            bo_cmd_sans_arrondi = pd.to_numeric(row['Commande Finale sans mini ni arrondi SM à 100%'], errors='coerce'); 
            if pd.isna(bo_cmd_sans_arrondi): bo_cmd_sans_arrondi = 0
            if bo_cmd_sans_arrondi == 0: return 0
            pcb_w = pd.to_numeric(row['PCB'], errors='coerce'); 
            if pd.isna(pcb_w) or pcb_w == 0: pcb_w = 1
            mini_cmd_v = pd.to_numeric(row['MINIMUM_COMMANDE'], errors='coerce'); 
            if pd.isna(mini_cmd_v): mini_cmd_v = 0
            type_prod_bm = str(row.get('Type de produits V2', '')); 
            seuil1, seuil2 = get_arrondi_pcb_seuils(type_prod_bm) 
            diviseur_seuil1 = max(mini_cmd_v, pcb_w); 
            if diviseur_seuil1 == 0 : diviseur_seuil1 = 1 
            if bo_cmd_sans_arrondi / diviseur_seuil1 < seuil1: return 0
            else:
                val_arrondie = 0; bo_div_pcb = bo_cmd_sans_arrondi / pcb_w
                if bo_div_pcb >= 1:
                    partie_decimale = bo_div_pcb - np.floor(bo_div_pcb)
                    if partie_decimale < seuil2: val_arrondie = np.floor(bo_div_pcb) * pcb_w
                    else: val_arrondie = np.ceil(bo_div_pcb) * pcb_w
                else: 
                    if mini_cmd_v > pcb_w:
                        bo_div_min_cmd = bo_cmd_sans_arrondi / mini_cmd_v; partie_decimale_v = bo_div_min_cmd - np.floor(bo_div_min_cmd)
                        if partie_decimale_v < seuil1: val_arrondie = np.floor(bo_div_min_cmd) * mini_cmd_v
                        else: val_arrondie = np.ceil(bo_div_min_cmd) * mini_cmd_v
                    else: val_arrondie = np.ceil(bo_div_pcb) * pcb_w
                return max(mini_cmd_v, val_arrondie)
        except Exception: return 0
    df['Commande Finale avec mini et arrondi SM à 100%'] = df.apply(calculate_bp_row, axis=1)
    # print("  Colonne 'Commande Finale avec mini et arrondi SM à 100%' (BP) calculée.")
    return df

def calculate_commande_finale_avec_mini_et_arrondi_avec_ts(df): # BQ
    # ... (votre code existant pour BQ) ...    print("Calcul de 'Commande Finale avec mini et arrondi SM à 100% avec TS' (BQ)...")
    required_columns_bq = ['Commande Finale avec mini et arrondi SM à 100%', 'TS']
    missing_columns = [col for col in required_columns_bq if col not in df.columns]
    if missing_columns:
        print(f"  ERREUR: Colonnes manquantes pour BQ: {missing_columns}")
        for col_miss in missing_columns: df[col_miss] = 0
    
    def calculate_bq_row(row):
        try:
            bp_cmd_arrondi = pd.to_numeric(row['Commande Finale avec mini et arrondi SM à 100%'], errors='coerce'); 
            if pd.isna(bp_cmd_arrondi): bp_cmd_arrondi = 0
            ts_bj = pd.to_numeric(row['TS'], errors='coerce'); 
            if pd.isna(ts_bj): ts_bj = 1 
            result = bp_cmd_arrondi * ts_bj
            
            # Apply rounding logic: if < 1, make it 0; if >= 1, round down to integer
            if result < 1:
                return 0
            else:
                return int(np.floor(result))
        except Exception: return 0
    df['Commande Finale avec mini et arrondi SM à 100% avec TS'] = df.apply(calculate_bq_row, axis=1)
    # print("  Colonne 'Commande Finale avec mini et arrondi SM à 100% avec TS' (BQ) calculée.")
    return df

def get_processed_data(df, step="initial"):
    if step == "initial":
        print("CommandeFinale: Calcul des commandes initiales (BO, BP, BQ)...")
        df = calculate_commande_finale_sans_mini_ni_arrondi(df)      
        df = calculate_commande_finale_avec_mini_et_arrondi(df)       
        df = calculate_commande_finale_avec_mini_et_arrondi_avec_ts(df) 
    elif step == "optimise":
        print("CommandeFinale: Calcul des commandes optimisées (CD, CE, CF) utilisant BY...")
        if 'Facteur multiplicatif Lissage besoin brut' not in df.columns:
            print("  ERREUR: Colonne 'Facteur multiplicatif Lissage besoin brut' (BY) manquante pour calculer CD, CE, CF.")
            df['Commande optimisée sans arrondi'] = 0
            df['Commande optimisée avec arrondi et mini'] = 0
            df['Commande optimisée avec arrondi et mini et TS'] = 0
            return df
        df = calculate_commande_optimisee_sans_arrondi(df) # Utilise la version qui appelle _row
        df = calculate_ce_commande_optimisee_avec_arrondi_et_mini(df) # Utilise la version qui appelle _row
        df = calculate_cf_commande_optimisee_avec_arrondi_mini_ts(df) # Utilise la version qui appelle _row
    else:
        print(f"  ERREUR CommandeFinale: Étape inconnue '{step}'.")
    return df

if __name__ == "__main__":
    print("Test du module CommandeFinale")
    current_dir = os.path.dirname(__file__)
    
    csv_dir_test = os.path.join(current_dir, "CSV")
    if not os.path.exists(csv_dir_test):
        os.makedirs(csv_dir_test)

    # CORRECTION: Nom du fichier de test
    qtemaxi_test_path = os.path.join(csv_dir_test, "Qtite_Maxi.csv")
    qtemaxi_test_content = (
        "entrepot;EAN13;ifls;libelle article;Nombre Maxi Unit�s De Commande;pcb;ID;ID2;QteMaxUS\n"
        "619;3016570002150;R27775;75CL TAITTINGER PRESTIG.BR ET;12;6;619-3016570002150;619-R27775;12\n"
        "ENT02;EAN002;IFLS002;PRODUIT B;24;12;ENT02-EAN002;ENT02-IFLS002;24"
    )
    with open(qtemaxi_test_path, 'w', encoding='utf-8') as f:
        f.write(qtemaxi_test_content)
    print(f"Fichier de test {os.path.basename(qtemaxi_test_path)} créé dans {csv_dir_test}.")

    # ... (le reste du bloc de test peut rester tel quel car il utilise la fonction corrigée) ...
    # Simuler merged_predictions.csv
    test_file_path = os.path.join(current_dir, 'merged_predictions_for_test_cf.csv') 
    test_data_main = {
        'CODE_METI': ['M001', 'M002', 'M003'],
        'Mini Publication FL': [100, 200, 50],
        'COCHE_RAO': [0,0,0], 'RAL': [0,0,0], 'STOCK_REEL': [0,0,0], 'SM Final': [0,0,0],
        'Prév C1-L2 Finale': [0,0,0], 'Prév L1-L2 Finale': [0,0,0],
        'Facteur Multiplicatif Appro': [1,1,1],
        'Casse Prev C1-L2': [0,0,0], 'Casse Prev L1-L2': [0,0,0],
        'Produit Bloqué': [False, False, False], 'Commande Max avec stock max': [999,999,999],
        'Facteur multiplicatif Lissage besoin brut': [1.0, 1.2, 0.9],
        'Commande Finale avec mini et arrondi SM à 100%': [0,0,0],
        'Type de produits V2': ['TPV01', 'TPV02', 'TPV03'],
        'PCB': [6, 12, 1], 'MINIMUM_COMMANDE': [0,0,0], 'TS': [1,1,1],
        'Quantité Maxi à charger ?': ['Oui', 'Oui', 'Non'],
        'FOURNISSEUR': ['619', 'ENT02', 'ENT03'], 
        'IFLS': ['R27775', 'IFLS002', 'IFLS003']
    }
    df_test_main = pd.DataFrame(test_data_main)
    # Ajouter les colonnes qui seraient créées par les étapes précédentes pour un test plus complet
    df_test_main['Commande optimisée avec arrondi et mini'] = [120, 240, 50] # Valeurs d'exemple pour CE
    df_test_main.to_csv(test_file_path, sep=';', index=False, encoding='latin1')
    
    if os.path.exists(test_file_path):
        test_df = pd.read_csv(test_file_path, sep=';', encoding='latin1')
        
        print("\n--- Calcul Étape Initiale (BO, BP, BQ) ---")
        result_df_initial = get_processed_data(test_df.copy(), step="initial")
        
        print("\n--- Calcul Étape Optimisée (CD, CE, CF, CG) ---")
        # S'assurer que la colonne 'Commande optimisée avec arrondi et mini' existe avant d'appeler step="optimise"
        # car CG en dépend. Si get_processed_data(step="initial") ne la crée pas,
        # il faut s'assurer qu'elle est là d'une autre manière pour le test.
        # Dans le flux réel, elle serait calculée par la même fonction.
        # Pour le test ici, result_df_initial aura déjà cette colonne grâce à la simulation.
        result_df_optimise = get_processed_data(result_df_initial.copy(), step="optimise")
        #save to csv file
        
        print("\nRésultats Étape Optimisée (extrait):")
        cols_to_show = ['CODE_METI', 'FOURNISSEUR', 'IFLS', 'Quantité Maxi à charger ?',
                        'Commande optimisée avec arrondi et mini',
                        'PCB',
                        'Commande optimisée avec arrondi et mini en colis']
        print(result_df_optimise[[col for col in cols_to_show if col in result_df_optimise.columns]].head().to_string())

        # Vérification
        row_m001 = result_df_optimise[result_df_optimise['CODE_METI'] == 'M001']
        if not row_m001.empty:
            row_m001 = row_m001.iloc[0]
            ce_m001 = row_m001.get('Commande optimisée avec arrondi et mini', 0)
            cg_m001 = row_m001.get('Commande optimisée avec arrondi et mini en colis', 0)
            print(f"\nTest M001: CE={ce_m001}, CG attendu={ce_m001/12 if ce_m001 is not None else 0}, CG obtenu={cg_m001}")
            if ce_m001 is not None and abs(cg_m001 - (ce_m001 / 12)) < 0.001 :
                 print("  Test M001 OK")
            else:
                 print("  Test M001 ECHEC")


        row_m003 = result_df_optimise[result_df_optimise['CODE_METI'] == 'M003']
        if not row_m003.empty:
            row_m003 = row_m003.iloc[0]
            ce_m003 = row_m003.get('Commande optimisée avec arrondi et mini', 0)
            pcb_m003 = row_m003.get('PCB', 1)
            cg_m003 = row_m003.get('Commande optimisée avec arrondi et mini en colis', 0)
            print(f"Test M003: CE={ce_m003}, PCB={pcb_m003}, CG attendu={ce_m003/pcb_m003 if ce_m003 is not None and pcb_m003 !=0 else 0}, CG obtenu={cg_m003}")
            if ce_m003 is not None and pcb_m003 !=0 and abs(cg_m003 - (ce_m003/pcb_m003)) < 0.001:
                 print("  Test M003 OK")
            else:
                 print("  Test M003 ECHEC")


    else:
        print(f"Fichier de test principal {test_file_path} non trouvé.")