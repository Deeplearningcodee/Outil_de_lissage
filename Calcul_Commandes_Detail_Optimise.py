# Fichier: Calcul_Commandes_Detail_Optimise.py
import pandas as pd
import numpy as np

# Essayer d'importer depuis MacroParam. Si non trouvé, utiliser des valeurs par défaut pour le débogage.
try:
    from MacroParam import get_param_value, get_arrondi_pcb_seuils
    MACROPARAM_AVAILABLE = True
except ImportError:
    print("AVERTISSEMENT (Calcul_Commandes_Detail_Optimise.py): MacroParam.py non trouvé. Utilisation de valeurs par défaut pour certains paramètres.")
    MACROPARAM_AVAILABLE = False
    # Fonctions factices si MacroParam n'est pas là
    def get_param_value(param_name, default_value):
        if param_name == 'Casse Prev Activé': return "Oui" # Exemple
        return default_value
    def get_arrondi_pcb_seuils(type_prod_v2):
        # Valeurs par défaut typiques, à ajuster si nécessaire pour le test
        if "A/B" in type_prod_v2: return 0.01, 0.5 
        if "A/C" in type_prod_v2: return 0.01, 0.8 
        return 0.01, 0.5


def calculate_by_facteur_lissage_besoin_brut(detail_row, j_factor_optim, k_factor_optim, l_factor_optim):
    """
    Calcule Détail!BY (Facteur multiplicatif Lissage besoin brut)
    """
    bw_borne_min = pd.to_numeric(detail_row.get('Borne Min Facteur multiplicatif lissage', 0), errors='coerce')
    bx_borne_max = pd.to_numeric(detail_row.get('Borne Max Facteur multiplicatif lissage', 10), errors='coerce') 
    bn_top_category_orig = detail_row.get('Top','autre') # Garder l'original pour le log
    bn_top_category = str(bn_top_category_orig).strip().lower() # Standardiser

    if pd.isna(bw_borne_min): bw_borne_min = 0.0
    if pd.isna(bx_borne_max): bx_borne_max = 10.0 

    # print(f"        DEBUG CCDO.BY - Entrée: Detail_Top='{bn_top_category_orig}' (utilisé comme '{bn_top_category}'), J={j_factor_optim:.3f}, K={k_factor_optim:.3f}, L={l_factor_optim:.3f}")
    # print(f"        DEBUG CCDO.BY - Bornes Lissage: Min={bw_borne_min}, Max={bx_borne_max}")

    facteur_optim_selon_top = 1.0 # Default
    if bn_top_category == 'top 500':
        facteur_optim_selon_top = j_factor_optim
    elif bn_top_category == 'top 3000':
        facteur_optim_selon_top = k_factor_optim
    elif bn_top_category == 'autre': # Assurez-vous que 'autre' est la chaîne attendue
        facteur_optim_selon_top = l_factor_optim
    
    # print(f"        DEBUG CCDO.BY - facteur_optim_selon_top = {facteur_optim_selon_top:.3f}")
    
    by_facteur = max(bw_borne_min, min(bx_borne_max, facteur_optim_selon_top))
    # print(f"        DEBUG CCDO.BY - Sortie by_facteur: {by_facteur:.3f}")
    return by_facteur

def calculate_cd_commande_optimisee_sans_arrondi(detail_row, by_facteur_lissage):
    """
    Calcul de Détail!CD (Commande optimisée sans arrondi)
    """
    bp_cmd_sm100_arrondi = pd.to_numeric(detail_row.get('Commande Finale avec mini et arrondi SM à 100%', 0), errors='coerce')
    if pd.isna(bp_cmd_sm100_arrondi): bp_cmd_sm100_arrondi = 0

    #print(f"        DEBUG CCDO.CD - Entrée: by_facteur={by_facteur_lissage:.3f}, BP (Cmd SM100% Arroni)={bp_cmd_sm100_arrondi}")

    if bp_cmd_sm100_arrondi == 0 and by_facteur_lissage <= 1:
        #print(f"        DEBUG CCDO.CD - Condition BP=0 ET BY<=1 REMPLIE. CD=0.")
        return 0

    ak_mini_pub = pd.to_numeric(detail_row.get('Mini Publication FL', 0), errors='coerce')
    ad_coche_rao = pd.to_numeric(detail_row.get('COCHE_RAO', 0), errors='coerce')
    aj_stock_reel = pd.to_numeric(detail_row.get('STOCK_REEL', 0), errors='coerce')
    z_ral = pd.to_numeric(detail_row.get('RAL', 0), errors='coerce')
    
    if pd.isna(ak_mini_pub): ak_mini_pub = 0
    if pd.isna(ad_coche_rao): ad_coche_rao = 0
    if pd.isna(aj_stock_reel): aj_stock_reel = 0
    if pd.isna(z_ral): z_ral = 0

    besoin_net_cd = 0
    if not (ak_mini_pub <= 0 or ad_coche_rao == 46): 
        besoin_net_cd = ak_mini_pub - aj_stock_reel - z_ral
    
    o_sm_final = pd.to_numeric(detail_row.get('SM Final', 0), errors='coerce')
    bf_prev_c1_l2 = pd.to_numeric(detail_row.get('Prév C1-L2 Finale', 0), errors='coerce')
    bg_prev_l1_l2 = pd.to_numeric(detail_row.get('Prév L1-L2 Finale', 0), errors='coerce')
    bu_facteur_appro = pd.to_numeric(detail_row.get('Facteur Multiplicatif Appro', 1), errors='coerce')
    ay_casse_c1_l2 = pd.to_numeric(detail_row.get('Casse Prev C1-L2', 0), errors='coerce')
    az_casse_l1_l2 = pd.to_numeric(detail_row.get('Casse Prev L1-L2', 0), errors='coerce')
    cc_cmd_max_stock_max = pd.to_numeric(detail_row.get('Commande Max avec stock max', 99999999), errors='coerce')
    
    bi_produit_bloque_str = str(detail_row.get('Produit Bloqué', 'Non')).lower()
    bi_produit_bloque = bi_produit_bloque_str in ['vrai', 'true', 'oui', 'yes', '1', True]

    position_jud_val = detail_row.get('Position JUD', None)
    p_position_jud_is_error = pd.isna(position_jud_val) or \
                              (pd.to_numeric(position_jud_val, errors='coerce') <= 0 
                               if pd.notna(position_jud_val) and str(position_jud_val).strip() != "" 
                               else True)

    casse_prev_active = get_param_value('Casse Prev Activé', 'Oui') 

    if pd.isna(o_sm_final): o_sm_final = 0
    if pd.isna(bf_prev_c1_l2): bf_prev_c1_l2 = 0
    if pd.isna(bg_prev_l1_l2): bg_prev_l1_l2 = 0
    if pd.isna(bu_facteur_appro) or bu_facteur_appro == 0 : bu_facteur_appro = 1
    if pd.isna(ay_casse_c1_l2): ay_casse_c1_l2 = 0
    if pd.isna(az_casse_l1_l2): az_casse_l1_l2 = 0
    if pd.isna(cc_cmd_max_stock_max): cc_cmd_max_stock_max = 99999999

    quantite_stock_max_scenarios = 0 # Renommé pour clarté par rapport à quantite_via_stock_max_cd dans le code original
    if bi_produit_bloque:
        quantite_stock_max_scenarios = 0
    else:
        scenario1_cd = 0
        if not p_position_jud_is_error:
            ajustement1_cd = (-aj_stock_reel - z_ral)
            if casse_prev_active == "Oui":
                ajustement1_cd = -(aj_stock_reel - ay_casse_c1_l2) - z_ral
            scenario1_cd = max(0, (o_sm_final + bf_prev_c1_l2) * bu_facteur_appro * by_facteur_lissage + ajustement1_cd)

        scenario2_cd = 0
        if not p_position_jud_is_error:
            ajustement2_cd = 0 # L'ajustement pour scenario2 est différent
            if casse_prev_active == "Oui":
                ajustement2_cd = az_casse_l1_l2 
            scenario2_cd = max(0, (o_sm_final + bg_prev_l1_l2) * bu_facteur_appro * by_facteur_lissage + ajustement2_cd)
        
        quantite_stock_max_scenarios = min(scenario1_cd, scenario2_cd)

    valeur_avant_max_besoin_net = min(cc_cmd_max_stock_max, quantite_stock_max_scenarios)
    resultat_cd = max(besoin_net_cd, valeur_avant_max_besoin_net)
    
    cd_final_val = max(0, resultat_cd)
    # print(f"        DEBUG CCDO.CD - BesoinNet={besoin_net_cd:.2f}, QteStockMaxScenarios={quantite_stock_max_scenarios:.2f} (S1={scenario1_cd:.2f}, S2={scenario2_cd:.2f}), CmdMaxCC={cc_cmd_max_stock_max:.2f}")
    # print(f"        DEBUG CCDO.CD - ValeurAvantMaxBesoinNet={valeur_avant_max_besoin_net:.2f}, ResultatCD_avant_max0={resultat_cd:.2f}")
    # print(f"        DEBUG CCDO.CD - Sortie CD: {cd_final_val:.2f}")
    return cd_final_val


def calculate_ce_commande_optimisee_avec_arrondi_et_mini(detail_row, cd_commande_optim_sans_arrondi):
    """
    Calcul de Détail!CE (Commande optimisée avec arrondi et mini)
    """
    # print(f"        DEBUG CCDO.CE - Entrée: CD={cd_commande_optim_sans_arrondi:.2f}")
    if pd.isna(cd_commande_optim_sans_arrondi) or cd_commande_optim_sans_arrondi == 0:
        # print(f"        DEBUG CCDO.CE - CD est 0 ou NaN. CE=0.")
        return 0

    v_min_commande = pd.to_numeric(detail_row.get('MINIMUM_COMMANDE', 0), errors='coerce')
    w_pcb = pd.to_numeric(detail_row.get('PCB', 1), errors='coerce') 
    
    if pd.isna(v_min_commande): v_min_commande = 0
    if pd.isna(w_pcb) or w_pcb == 0: w_pcb = 1

    bm_type_prod_v2 = str(detail_row.get('Type de produits V2', '')).strip()
    seuil1, seuil2 = get_arrondi_pcb_seuils(bm_type_prod_v2) 
    # print(f"        DEBUG CCDO.CE - MinCmd={v_min_commande}, PCB={w_pcb}, Seuils=({seuil1},{seuil2}) pour TPV2='{bm_type_prod_v2}'")

    diviseur_seuil1 = max(v_min_commande, w_pcb) 
    if diviseur_seuil1 == 0 : diviseur_seuil1 = 1 

    if cd_commande_optim_sans_arrondi / diviseur_seuil1 < seuil1:
        # print(f"        DEBUG CCDO.CE - Condition Seuil1 non remplie ({cd_commande_optim_sans_arrondi / diviseur_seuil1:.2f} < {seuil1}). CE=0.")
        return 0
    else:
        val_arrondie = 0
        cd_div_pcb = cd_commande_optim_sans_arrondi / w_pcb
        if cd_div_pcb >= 1:
            partie_decimale = cd_div_pcb - np.floor(cd_div_pcb)
            if partie_decimale < seuil2:
                val_arrondie = np.floor(cd_div_pcb) * w_pcb
            else:
                val_arrondie = np.ceil(cd_div_pcb) * w_pcb
        else: 
            if v_min_commande > w_pcb:
                cd_div_min_cmd = cd_commande_optim_sans_arrondi / v_min_commande
                partie_decimale_v = cd_div_min_cmd - np.floor(cd_div_min_cmd)
                if partie_decimale_v < seuil1: 
                    val_arrondie = np.floor(cd_div_min_cmd) * v_min_commande
                else:
                    val_arrondie = np.ceil(cd_div_min_cmd) * v_min_commande
            else: 
                val_arrondie = np.ceil(cd_div_pcb) * w_pcb
        
        ce_final_val = max(v_min_commande, val_arrondie)
        # print(f"        DEBUG CCDO.CE - ValArrondie={val_arrondie:.2f}, Sortie CE: {ce_final_val:.2f}")
        return ce_final_val

def calculate_cf_commande_optimisee_avec_arrondi_mini_ts(detail_row, ce_commande_optim_arrondi_mini):
    """
    Calcul de Détail!CF (Commande optimisée avec arrondi et mini et TS)
    """
    # print(f"        DEBUG CCDO.CF - Entrée: CE={ce_commande_optim_arrondi_mini:.2f}")
    if pd.isna(ce_commande_optim_arrondi_mini):
        # print(f"        DEBUG CCDO.CF - CE est NaN. CF=0.")
        return 0
    
    bj_ts = pd.to_numeric(detail_row.get('TS', 1), errors='coerce')
    if pd.isna(bj_ts): bj_ts = 1 
    # print(f"        DEBUG CCDO.CF - TS={bj_ts}")

    cf_final_val = ce_commande_optim_arrondi_mini * bj_ts
    # print(f"        DEBUG CCDO.CF - Sortie CF: {cf_final_val:.2f}")
    return cf_final_val

def get_simulated_cf_for_detail_line(detail_row, j_factor_optim, k_factor_optim, l_factor_optim):
    """
    Fonction principale pour obtenir la CF simulée pour une ligne de détail,
    en utilisant les facteurs JKL optimisés.
    `detail_row` peut être un dict ou une pd.Series.
    """
    # print(f"    DEBUG get_simulated_cf_for_detail_line - CODE_METI: {detail_row.get('CODE_METI', 'N/A')}, JKL_optim: [{j_factor_optim:.3f},{k_factor_optim:.3f},{l_factor_optim:.3f}]")
    by = calculate_by_facteur_lissage_besoin_brut(detail_row, j_factor_optim, k_factor_optim, l_factor_optim)
    cd = calculate_cd_commande_optimisee_sans_arrondi(detail_row, by)
    ce = calculate_ce_commande_optimisee_avec_arrondi_et_mini(detail_row, cd)
    cf = calculate_cf_commande_optimisee_avec_arrondi_mini_ts(detail_row, ce)
    # print(f"    DEBUG get_simulated_cf_for_detail_line - Résultat CF: {cf:.2f}")
    return cf

# --- Section de test du module ---
def main_test_ccdo():
    print("--- Test du module Calcul_Commandes_Detail_Optimise ---")
    # Créer un exemple de DataFrame de détail pour le test
    # Doit contenir toutes les colonnes lues par les fonctions ci-dessus
    test_data_detail = {
        'CODE_METI': ['ART001'],
        'Top': ['Top 500'],
        'Borne Min Facteur multiplicatif lissage': [0.5],
        'Borne Max Facteur multiplicatif lissage': [1.5],
        'Commande Finale avec mini et arrondi SM à 100%': [10], # BP (pour la condition CD)
        'Mini Publication FL': [20], 'COCHE_RAO': [0], 'STOCK_REEL': [5], 'RAL': [2],
        'SM Final': [50], 'Prév C1-L2 Finale': [10], 'Prév L1-L2 Finale': [8],
        'Facteur Multiplicatif Appro': [1.0],
        'Casse Prev C1-L2': [1], 'Casse Prev L1-L2': [0.5],
        'Produit Bloqué': ['Non'], 'Commande Max avec stock max': [100],
        'Position JUD': [10], # Exemple de valeur valide
        'MINIMUM_COMMANDE': [6], 'PCB': [6], 'TS': [1.0],
        'Type de produits V2': ['Sec Méca - A/B'] # Pour get_arrondi_pcb_seuils
        # Ajouter d'autres colonnes si nécessaire pour couvrir tous les cas
    }
    df_test_detail = pd.DataFrame(test_data_detail)
    detail_row_dict = df_test_detail.iloc[0].to_dict()

    print("\nTest avec JKL = 1.0, 1.0, 1.0:")
    cf_sim_1 = get_simulated_cf_for_detail_line(detail_row_dict, 1.0, 1.0, 1.0)
    print(f"  CF Simulée (JKL=1.0): {cf_sim_1}") # Attendu non nul si BP !=0 ou BY > 1

    print("\nTest avec JKL = 1.2, 1.1, 0.9:")
    cf_sim_2 = get_simulated_cf_for_detail_line(detail_row_dict, 1.2, 1.1, 0.9)
    print(f"  CF Simulée (JKL variés): {cf_sim_2}")

    print("\nTest avec BP = 0 et JKL=1.0 (devrait donner CD=0 -> CF=0):")
    detail_row_bp0 = detail_row_dict.copy()
    detail_row_bp0['Commande Finale avec mini et arrondi SM à 100%'] = 0
    cf_sim_bp0 = get_simulated_cf_for_detail_line(detail_row_bp0, 1.0, 1.0, 1.0) # BY sera 1.0
    print(f"  CF Simulée (BP=0, JKL=1.0): {cf_sim_bp0}")
    
    print("\nTest avec BP = 0 et JKL=1.2 (CD ne devrait pas être 0 par la 1ere condition):")
    cf_sim_bp0_by_gt1 = get_simulated_cf_for_detail_line(detail_row_bp0, 1.2, 1.2, 1.2) # BY sera 1.2
    print(f"  CF Simulée (BP=0, JKL=1.2): {cf_sim_bp0_by_gt1}")

if __name__ == "__main__":
    main_test_ccdo()