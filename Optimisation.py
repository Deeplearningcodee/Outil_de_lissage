import pandas as pd
import numpy as np
import os
import datetime 

# Importer les modules qui fournissent les données sources
import PDC as ModulePDC 
import Encours as ModuleEncours 
import MacroParam
import TypeProduits
try:
    import Calcul_Commandes_Detail_Optimise as ccdo
    CCDO_IMPORTED = True
except ImportError:
    print("AVERTISSEMENT (ParametresApprosGenerator.py): Calcul_Commandes_Detail_Optimise.py (ccdo) non trouvé. CF initial sera 0.")
    CCDO_IMPORTED = False

from MacroParam import (
    TYPES_PRODUITS, 
    TYPES_PRODUITS_V2, 
    DATE_COMMANDE, 
    load_macro_params, 
    FACTEURS_LISSAGE,
    ALERTE_SURCHARGE_NIVEAU_1
)
def create_parametres_commandes_appros_table_internal():
    print("ParametresApprosGenerator: Création de la table des paramètres (partie haute)...")
    date_commande_dt = DATE_COMMANDE 
    df_macro_params_onglet = load_macro_params() # Charge les paramètres de l'onglet MacroParam
    
    rows = []
    # Utiliser les TYPES_PRODUITS de MacroParam (qui devrait être la source unique)
    unique_base_product_types = sorted(list(set(MacroParam.TYPES_PRODUITS.values())))

    # D'abord, créer une table de base avec Type de produits et Jour livraison
    base_param_rows = []
    for type_produit_base in unique_base_product_types:
        # Déterminer le délai standard basé UNIQUEMENT sur type_produit_base pour l'instant
        # La distinction A/B, A/C se fera après l'appel à TypeProduits.py
        # Cette logique de délai est simplifiée ici, car le délai réel peut dépendre de A/B ou A/C
        # qui n'est pas encore déterminé. On prendra un délai "moyen" ou le plus courant.
        # Pour l'exemple, on va juste prendre 2 jours, et TypeProduits.py créera les lignes A/B, A/C.
        # OU, on crée des entrées pour chaque combinaison possible de date A/B, A/C
        
        # Créons des lignes candidates pour les dates de livraison A/B et A/C si c'est un type "Sec"
        if type_produit_base.startswith("Sec"):
            dates_livraison_candidates = []
            if MacroParam.DATE_REF_JOUR_AB : dates_livraison_candidates.append(MacroParam.DATE_REF_JOUR_AB)
            if MacroParam.DATE_REF_JOUR_AC and MacroParam.DATE_REF_JOUR_AC != MacroParam.DATE_REF_JOUR_AB : 
                dates_livraison_candidates.append(MacroParam.DATE_REF_JOUR_AC)
            
            # S'il n'y a pas de date AB/AC distincte, ou pour les autres jours, on prend un délai standard.
            # La logique exacte de génération des lignes de paramètres doit correspondre à ce que vous attendez dans PDC_Sim.
            # Ici, on va générer pour AB et AC si applicable pour les "Sec"
            if not dates_livraison_candidates: # Fallback si pas de dates AB/AC
                 delai_standard_temp = 2 # Mettre une valeur par défaut
                 jour_livraison_dt_object = date_commande_dt + pd.Timedelta(days=delai_standard_temp)
                 if jour_livraison_dt_object.weekday() == 6: jour_livraison_dt_object += pd.Timedelta(days=1)
                 dates_livraison_candidates.append(jour_livraison_dt_object)

            for jour_liv_cand_dt in set(dates_livraison_candidates): # Utiliser set pour éviter doublons de dates
                 base_param_rows.append({
                     "Type de produits": type_produit_base,
                     "DATE_LIVRAISON_V2": jour_liv_cand_dt # Nom de colonne attendu par TypeProduits.py
                 })
        else: # Non "Sec"
            delai_standard = 2 # Délai standard pour non-Sec
            jour_livraison_dt_object = date_commande_dt + pd.Timedelta(days=delai_standard)
            if jour_livraison_dt_object.weekday() == 6: jour_livraison_dt_object += pd.Timedelta(days=1)
            base_param_rows.append({
                "Type de produits": type_produit_base,
                "DATE_LIVRAISON_V2": jour_livraison_dt_object
            })

    df_base_params = pd.DataFrame(base_param_rows)
    df_base_params.drop_duplicates(inplace=True) # S'assurer qu'on a des combinaisons uniques

    # Appeler TypeProduits.py pour calculer "Type de produits V2" de manière cohérente
    # TypeProduits.py s'attend à 'CLASSE_STOCKAGE' pour 'Type de produits'.
    # Il faut donc ajouter temporairement CLASSE_STOCKAGE ou adapter TypeProduits.py
    # Pour l'instant, on va supposer que 'Type de produits' est suffisant pour que TypeProduits.py
    # puisse déduire ce dont il a besoin ou qu'on adapte TypeProduits.py.
    # La fonction calc_type_produits_v2 utilise 'Type de produits' et 'DATE_LIVRAISON_V2'.
    print("  ParametresApprosGenerator: Appel de TypeProduits.get_processed_data sur df_base_params...")
    if not df_base_params.empty:
        # calc_type_produits dans TypeProduits a besoin de CLASSE_STOCKAGE.
        # Nous devons soit la fournir, soit modifier TypeProduits pour qu'il puisse fonctionner
        # sans, ou utiliser une version de Type de Produits basée sur une autre logique si CLASSE_STOCKAGE
        # n'est pas pertinente pour les paramètres eux-mêmes.
        # Pour cet exemple, on va mapper 'Type de produits' (base) vers une CLASSE_STOCKAGE fictive
        # si TypeProduits.py en a absolument besoin.
        # Le plus simple est que TypeProduits.calc_type_produits_v2 utilise directement 'Type de produits'
        # qui est déjà dans df_base_params.
        
        # Assurons-nous que les colonnes attendues par TypeProduits.py sont là
        if 'CLASSE_STOCKAGE' not in df_base_params.columns and 'Type de produits' in df_base_params.columns:
             # Créer une CLASSE_STOCKAGE fictive si TypeProduits.py en a besoin pour son map TYPES_PRODUITS
             # Cela suppose que le map TYPES_PRODUITS dans MacroParam est {classe_stockage: type_produit}
             inv_types_produits_map = {v: k for k, v in MacroParam.TYPES_PRODUITS.items()}
             df_base_params['CLASSE_STOCKAGE'] = df_base_params['Type de produits'].map(inv_types_produits_map).fillna('INCONNUE')


        df_params_with_v2 = TypeProduits.get_processed_data(df_base_params) 
        # get_processed_data devrait ajouter 'Type de produits' (si pas déjà là via CLASSE_STOCKAGE),
        # 'Type de produits V2', et 'Top' (bien que 'Top' ne soit pas utilisé pour la table de paramètres)
    else:
        df_params_with_v2 = pd.DataFrame() # Vide si df_base_params est vide

    # Maintenant, continuer à construire df_pdc_sim_output basé sur df_params_with_v2
    # qui contient 'Type de produits' et 'Type de produits V2' calculés de manière cohérente.
    
    final_rows_params = []
    if not df_params_with_v2.empty:
        for _, row_param_v2 in df_params_with_v2.iterrows():
            type_produit_base = row_param_v2['Type de produits']
            type_produit_v2 = row_param_v2['Type de produits V2'] # Vient de TypeProduits.py
            jour_livraison_dt_object = pd.to_datetime(row_param_v2['DATE_LIVRAISON_V2']) # Vient de df_base_params

            # Calcul du délai standard (pour information, pourrait être redondant si non utilisé ailleurs)
            delai_standard = (jour_livraison_dt_object.date() - date_commande_dt.date()).days
            
            # Logique pour Poids du A/C max (inchangée)
            poids_ac_max_val_num = 1.0 
            if type_produit_v2.endswith(" - a/c"): # Comparer avec la version normalisée
                poids_ac_final_macro_val = 0.0 
                if df_macro_params_onglet is not None and not df_macro_params_onglet.empty:
                    matched_row = df_macro_params_onglet[df_macro_params_onglet['Type de produits'] == type_produit_base]
                    if not matched_row.empty and 'Poids du A/C final' in matched_row.columns:
                        val_read = matched_row['Poids du A/C final'].iloc[0]
                        if isinstance(val_read, str) and '%' in val_read:
                             poids_ac_final_macro_val = float(val_read.strip('%')) / 100.0
                        elif pd.notna(val_read):
                             poids_ac_final_macro_val = float(val_read)
                poids_ac_max_val_num = min(0.8, poids_ac_final_macro_val)
            
            boost_pdc_param_init = 0.0 
            jkl_param_init = 1.0 # Ces JKL sont ceux affichés dans la section paramètre de PDC_Sim
            
            # Les Limites O,P,Q,S,T,U sont des entrées utilisateur (lues plus tard dans optimisation_globale)
            # Pour la génération initiale de PDC_Sim, elles sont à 100% (ou lues si fichier existe)
            opqstu_lim_init_basse = 1.0 # Limite Basse initiale
            opqstu_lim_init_haute = 1.0 # Limite Haute initiale pour la génération,
                                        # sera écrasée par les valeurs de l'Excel lu par optimisation_globale.py
                                        # Si ce script *crée* PDC_Sim.xlsx pour la première fois, ces valeurs sont utilisées.
                                        # Si PDC_Sim.xlsx existe, optimisation_globale.py lira les valeurs existantes.

            # Min/Max Facteur (bornes utilisateur globales)
            facteurs_specifiques_minmax = FACTEURS_LISSAGE.get(type_produit_v2, # Essayer avec V2 d'abord
                                                     FACTEURS_LISSAGE.get(type_produit_base, # Puis avec base
                                                                          {"Min Facteur": 0.0, "Max Facteur": 4.0}))
            min_facteur_val_num = facteurs_specifiques_minmax.get("Min Facteur", 0.0)
            max_facteur_val_num = facteurs_specifiques_minmax.get("Max Facteur", 4.0)

            final_rows_params.append({
                "Type de produits V2": type_produit_v2, 
                "Type de produits": type_produit_base,
                "Délai standard livraison (jours)": delai_standard, 
                "Jour livraison": jour_livraison_dt_object,
                "Min Facteur": min_facteur_val_num, 
                "Max Facteur": max_facteur_val_num,     
                "Boost PDC": boost_pdc_param_init, 
                "Poids du A/C max": poids_ac_max_val_num, 
                "Top 500": jkl_param_init, "Top 3000": jkl_param_init, "Autre": jkl_param_init,                 
                "Limite Basse Top 500": opqstu_lim_init_basse, 
                "Limite Basse Top 3000": opqstu_lim_init_basse, 
                "Limite Basse Autre": opqstu_lim_init_basse,   
                "Limite Haute Top 500": opqstu_lim_init_haute, # Normalement 1 pour la génération, mais lu ensuite
                "Limite Haute Top 3000": opqstu_lim_init_haute, 
                "Limite Haute Autre": opqstu_lim_init_haute    
            })
    
    df_params = pd.DataFrame(final_rows_params)
    # S'assurer qu'il n'y a pas de doublons de (Type de produits V2, Jour livraison)
    # La génération de df_params_with_v2 devrait déjà gérer cela si les dates sont distinctes.
    if not df_params.empty:
        df_params.drop_duplicates(subset=['Type de produits V2', 'Jour livraison'], inplace=True)
    
    if not df_params.empty and all(col in df_params.columns for col in ["Top 500", "Top 3000", "Autre"]):
        df_params['Moyenne'] = (df_params['Top 500'] + df_params['Top 3000'] + df_params['Autre']) / 3.0
    elif not df_params.empty:
        df_params['Moyenne'] = np.nan
        
    if 'Jour livraison' in df_params.columns: # S'assurer que la colonne existe
        df_params['Jour livraison'] = pd.to_datetime(df_params['Jour livraison'])
    
    print(f"  ParametresApprosGenerator: Table des paramètres (haute) créée avec {len(df_params)} lignes uniques.")
    return df_params

def get_base_type_from_v2(type_v2_lower_str):
    if pd.isna(type_v2_lower_str): return "Autre"
    type_v2_lower_str = str(type_v2_lower_str).strip()
    if type_v2_lower_str.endswith(" - a/b"): 
        base_candidate = type_v2_lower_str[:-6].strip()
        for k_mp_tp, v_mp_tp in MacroParam.TYPES_PRODUITS.items():
            if v_mp_tp.lower() == base_candidate: return v_mp_tp
        return base_candidate.title()
    if type_v2_lower_str.endswith(" - a/c"): 
        base_candidate = type_v2_lower_str[:-6].strip()
        for k_mp_tp, v_mp_tp in MacroParam.TYPES_PRODUITS.items():
            if v_mp_tp.lower() == base_candidate: return v_mp_tp
        return base_candidate.title()
    normalized_base_types_map = {bt.lower(): bt for bt in MacroParam.TYPES_PRODUITS.values()}
    if type_v2_lower_str in normalized_base_types_map:
        return normalized_base_types_map[type_v2_lower_str] 
    if type_v2_lower_str == "sec - autre": 
        for k_mp_tp, v_mp_tp in MacroParam.TYPES_PRODUITS.items():
            if v_mp_tp.startswith("Sec"): return v_mp_tp
        return "Sec Méca"
    return type_v2_lower_str.title()

def create_full_pdc_sim_table(df_detail_source: pd.DataFrame, 
                               df_pdc_perm_summary: pd.DataFrame, 
                               df_encours_summary: pd.DataFrame):
    print("ParametresApprosGenerator: Création de la table PDC_Sim complète...")

    if df_detail_source is None or df_detail_source.empty:
        print("  ERREUR: df_detail_source est vide. Impossible de générer la table PDC_Sim.")
        return pd.DataFrame()

    # --- Étape 1: Générer les lignes de paramètres voulues ---
    types_v2_souhaites_exacts = [
        "Sec Méca - A/B", "Sec Méca - A/C", 
        "Sec Homogène - A/B", "Sec Homogène - A/C",
        "Sec Hétérogène - A/B", "Sec Hétérogène - A/C",
        "Frais Méca", "Frais Manuel", "Surgelés"
    ]
    
    rows_for_pdc_sim = []
    for type_v2_exact in types_v2_souhaites_exacts:
        type_v2_lower = type_v2_exact.lower().strip()
        type_base = get_base_type_from_v2(type_v2_lower)
        gestion = MacroParam.GESTION_AB_AC_MAPPING.get(type_base, "Non")
        jours_livraison_pour_ce_type = []

        if type_v2_exact.endswith(" - A/B"):
            if MacroParam.DATE_REF_JOUR_AB and pd.notna(MacroParam.DATE_REF_JOUR_AB):
                jours_livraison_pour_ce_type.append(MacroParam.DATE_REF_JOUR_AB.normalize())
        elif type_v2_exact.endswith(" - A/C"):
            if MacroParam.DATE_REF_JOUR_AC and pd.notna(MacroParam.DATE_REF_JOUR_AC):
                jours_livraison_pour_ce_type.append(MacroParam.DATE_REF_JOUR_AC.normalize())
        elif gestion == "Non":
            if MacroParam.DATE_REF_JOUR_AC and pd.notna(MacroParam.DATE_REF_JOUR_AC):
                jours_livraison_pour_ce_type.append(MacroParam.DATE_REF_JOUR_AC.normalize())
            else:
                delai_std = 2 
                jour_liv_obj = DATE_COMMANDE + pd.Timedelta(days=delai_std)
                if jour_liv_obj.weekday() == 6: jour_liv_obj += pd.Timedelta(days=1)
                jours_livraison_pour_ce_type.append(jour_liv_obj.normalize())
                print(f"    ATTENTION: DATE_REF_JOUR_AC non définie pour type '{type_v2_exact}' (Gestion Non). Utilisation date par défaut: {jours_livraison_pour_ce_type[0].strftime('%Y-%m-%d')}")
        else:
            print(f"    ATTENTION: Type V2 '{type_v2_exact}' (Base: '{type_base}', Gestion: {gestion}) ne correspond pas à un suffixe A/B, A/C et n'est pas en gestion 'Non'. Ligne non générée pour les paramètres.")
            continue 

        for jour_liv_param in set(jours_livraison_pour_ce_type):
             rows_for_pdc_sim.append({
                 'mergekey_TypeV2': type_v2_lower,
                 'Type de produits V2_orig': type_v2_exact, 
                 'Type de produits': type_base,
                 'mergekey_JourLiv': jour_liv_param 
             })

    df_pdc_sim_output = pd.DataFrame(rows_for_pdc_sim)
    if not df_pdc_sim_output.empty:
        df_pdc_sim_output.drop_duplicates(subset=['mergekey_TypeV2', 'mergekey_JourLiv'], inplace=True)
        df_pdc_sim_output.dropna(subset=['mergekey_TypeV2', 'mergekey_JourLiv'], inplace=True)
    
    if df_pdc_sim_output.empty:
        print("  ERREUR: Aucune ligne de paramètre générée après filtrage des types/dates souhaités.")
        return pd.DataFrame()
    print(f"  ParametresApprosGenerator: {len(df_pdc_sim_output)} lignes de paramètres uniques générées.")

    # --- Étape 1.5: Préparer df_detail_normalized pour les agrégations ---
    # (Cette section était manquante ou mal placée dans la version précédente)
    df_detail_normalized = df_detail_source.copy()
    
    key_type_prod_v2_detail = 'Type de produits V2' 
    key_date_livraison_detail = 'DATE_LIVRAISON_V2'
    col_bq_detail_name_orig = 'Commande Finale avec mini et arrondi SM à 100% avec TS'

    # S'assurer que 'Type de produits V2' existe et est normalisé dans df_detail_normalized
    if key_type_prod_v2_detail not in df_detail_normalized.columns:
        print(f"  ATTENTION: Colonne '{key_type_prod_v2_detail}' manquante dans df_detail_source pour agrégation. Tentative de calcul via TypeProduits.py.")
        if 'CLASSE_STOCKAGE' not in df_detail_normalized.columns and 'Classe_Stockage' in df_detail_normalized.columns:
            df_detail_normalized.rename(columns={'Classe_Stockage': 'CLASSE_STOCKAGE'}, inplace=True)
        if key_date_livraison_detail not in df_detail_normalized.columns:
             df_detail_normalized[key_type_prod_v2_detail] = "INCONNU_V2_CALC_ERREUR_DETAIL"
        elif 'CLASSE_STOCKAGE' not in df_detail_normalized.columns:
            df_detail_normalized[key_type_prod_v2_detail] = "INCONNU_V2_CALC_ERREUR_DETAIL"
        else:
            df_detail_normalized = TypeProduits.get_processed_data(df_detail_normalized)
            if key_type_prod_v2_detail not in df_detail_normalized.columns:
                df_detail_normalized[key_type_prod_v2_detail] = "INCONNU_V2_POST_TP_DETAIL"
    
    if key_type_prod_v2_detail in df_detail_normalized.columns:
        df_detail_normalized['mergekey_TypeV2'] = df_detail_normalized[key_type_prod_v2_detail].astype(str).str.strip().str.lower()
    else: 
        df_detail_normalized['mergekey_TypeV2'] = "fallback_v2_key_manquante_detail"

    if key_date_livraison_detail in df_detail_normalized.columns:
        df_detail_normalized['mergekey_JourLiv'] = pd.to_datetime(df_detail_normalized[key_date_livraison_detail], errors='coerce').dt.normalize()
    else: 
        df_detail_normalized['mergekey_JourLiv'] = pd.NaT
        
    if col_bq_detail_name_orig not in df_detail_normalized.columns:
        df_detail_normalized[col_bq_detail_name_orig] = 0.0
    else:
        df_detail_normalized[col_bq_detail_name_orig] = pd.to_numeric(df_detail_normalized[col_bq_detail_name_orig], errors='coerce').fillna(0.0)


    # --- Étape 2: Enrichir df_pdc_sim_output (suite) ---
    df_pdc_sim_output['Jour livraison'] = df_pdc_sim_output['mergekey_JourLiv']
    df_pdc_sim_output['Délai standard livraison (jours)'] = (df_pdc_sim_output['Jour livraison'] - DATE_COMMANDE).dt.days

    df_macro_params_onglet = MacroParam.load_macro_params()
    map_poids_ac_macro = {}
    if df_macro_params_onglet is not None and not df_macro_params_onglet.empty and 'Poids du A/C final' in df_macro_params_onglet.columns:
        df_temp_macro = df_macro_params_onglet.copy()
        col_poids_ac_final = 'Poids du A/C final'
        if df_temp_macro[col_poids_ac_final].dtype == 'object':
            df_temp_macro['Poids_num'] = pd.to_numeric(df_temp_macro[col_poids_ac_final].astype(str).str.rstrip('%').str.replace(',', '.', regex=False), errors='coerce')
            # Diviser par 100 si la valeur est > 1 (pourcentage)
            df_temp_macro.loc[df_temp_macro['Poids_num'] > 1, 'Poids_num'] /= 100.0
        else:
            df_temp_macro['Poids_num'] = pd.to_numeric(df_temp_macro[col_poids_ac_final], errors='coerce')
        df_temp_macro.dropna(subset=['Poids_num'], inplace=True)
        map_poids_ac_macro = pd.Series(df_temp_macro['Poids_num'].values,index=df_temp_macro['Type de produits']).to_dict()
        
    def calculate_poids_ac(row):
        if row['mergekey_TypeV2'].endswith(" - a/c"):
            base_type = row['Type de produits'] 
            poids_final_macro = map_poids_ac_macro.get(base_type, 0.0) 
            return min(0.8, poids_final_macro if pd.notna(poids_final_macro) else 0.0)
        return 1.0
    df_pdc_sim_output['Poids du A/C max'] = df_pdc_sim_output.apply(calculate_poids_ac, axis=1)

    df_pdc_sim_output['Boost PDC'] = 0.0
    df_pdc_sim_output['Top 500'] = 1.0; df_pdc_sim_output['Top 3000'] = 1.0; df_pdc_sim_output['Autre'] = 1.0
    df_pdc_sim_output['Limite Basse Top 500']=1.0; df_pdc_sim_output['Limite Basse Top 3000']=1.0; df_pdc_sim_output['Limite Basse Autre']=1.0
    df_pdc_sim_output['Limite Haute Top 500']=1.0; df_pdc_sim_output['Limite Haute Top 3000']=1.0; df_pdc_sim_output['Limite Haute Autre']=1.0

    min_f, max_f = [], []
    for _, row in df_pdc_sim_output.iterrows():
        tpv2_key = row['mergekey_TypeV2'] 
        tp_base_orig = row['Type de produits']
        tp_base_norm = tp_base_orig.lower()
        f_spec = FACTEURS_LISSAGE.get(tpv2_key,FACTEURS_LISSAGE.get(tp_base_norm,FACTEURS_LISSAGE.get(tp_base_orig,{"Min Facteur":0.0,"Max Facteur":4.0})))
        min_f.append(f_spec.get("Min Facteur",0.0)); max_f.append(f_spec.get("Max Facteur",4.0))
    df_pdc_sim_output['Min Facteur']=min_f; df_pdc_sim_output['Max Facteur']=max_f
    
    if not df_pdc_sim_output.empty : df_pdc_sim_output['Moyenne'] = (df_pdc_sim_output['Top 500'] + df_pdc_sim_output['Top 3000'] + df_pdc_sim_output['Autre']) / 3.0
    
    # --- Calcul PDC et En-cours ---
    map_poids_ac_final_for_pdc = pd.Series(df_pdc_sim_output['Poids du A/C max'].values, index=df_pdc_sim_output['mergekey_TypeV2']).to_dict()
    pdc_vals = []
    if ModulePDC and df_pdc_perm_summary is not None and not df_pdc_perm_summary.empty:
        if not isinstance(df_pdc_perm_summary.index, pd.DatetimeIndex): df_pdc_perm_summary.index = pd.to_datetime(df_pdc_perm_summary.index, errors='coerce')
        for _, row_s in df_pdc_sim_output.iterrows():
            dt_lk, tp_lk_base, pdc_b = row_s['mergekey_JourLiv'], row_s['Type de produits'], 0.0
            if pd.notna(dt_lk) and dt_lk in df_pdc_perm_summary.index and tp_lk_base in df_pdc_perm_summary.columns: pdc_b = df_pdc_perm_summary.loc[dt_lk, tp_lk_base]
            if pd.isna(pdc_b): pdc_b = 0.0
            p_ac = map_poids_ac_final_for_pdc.get(row_s['mergekey_TypeV2'], 1.0)
            pdc_vals.append(pdc_b * p_ac) 
    else: pdc_vals = [0.0] * len(df_pdc_sim_output)
    df_pdc_sim_output['PDC'] = pdc_vals

    enc_vals = []
    if ModuleEncours and df_encours_summary is not None and not df_encours_summary.empty:
        if not isinstance(df_encours_summary.index, pd.DatetimeIndex): df_encours_summary.index = pd.to_datetime(df_encours_summary.index, errors='coerce')
        for _, row_s in df_pdc_sim_output.iterrows():
            dt_lk, tpv2_lk, val_e = row_s['mergekey_JourLiv'], row_s['mergekey_TypeV2'], 0.0
            if pd.notna(dt_lk) and dt_lk in df_encours_summary.index and tpv2_lk in df_encours_summary.columns: val_e = df_encours_summary.loc[dt_lk, tpv2_lk]
            if pd.isna(val_e): val_e = 0.0
            enc_vals.append(val_e)
    else: enc_vals = [0.0] * len(df_pdc_sim_output)
    df_pdc_sim_output['En-cours'] = enc_vals

    # --- Agrégations BQ et calcul de CF_initial pour Commande optimisée initiale ---
    print("  Calcul de CF_initial pour chaque ligne de détail (avec JKL=1,1,1)...")
    cf_initial_values = []
    if not df_detail_normalized.empty and CCDO_IMPORTED and ccdo is not None:
        # Assurer que les colonnes nécessaires pour ccdo sont dans df_detail_normalized
        # 'Top' est calculé par TypeProduits.py sur df_detail_normalized (si appelé)
        if 'Top' not in df_detail_normalized.columns:
            print("    ATTENTION: Colonne 'Top' manquante dans les détails pour calculer CF_initial. Tentative de calcul...")
            # Il faudrait CLASSE_STOCKAGE et DATE_LIVRAISON_V2 pour appeler TypeProduits.get_processed_data
            # Pour l'instant, on met un Top par défaut si manquant
            df_detail_normalized['Top'] = 'autre' 
            
        for _, detail_row in df_detail_normalized.iterrows():
            try:
                cf_initial = ccdo.get_simulated_cf_for_detail_line(detail_row.to_dict(), 1.0, 1.0, 1.0)
                cf_initial_values.append(cf_initial)
            except Exception as e_cf_calc:
                cf_initial_values.append(0.0)
        df_detail_normalized['CF_initial_pour_agregation'] = cf_initial_values
    elif not df_detail_normalized.empty: 
        if not CCDO_IMPORTED: print("    ATTENTION: ccdo non importé. 'CF_initial_pour_agregation' sera 0.")
        df_detail_normalized['CF_initial_pour_agregation'] = 0.0
    
    print("  Agrégation BQ pour 'Commande SM à 100%'...")
    agg_sm100 = df_detail_normalized.groupby(['mergekey_TypeV2', 'mergekey_JourLiv'], as_index=False)[col_bq_detail_name_orig].sum()
    agg_sm100.rename(columns={col_bq_detail_name_orig: 'temp_CmdSM100'}, inplace=True)
    df_pdc_sim_output = pd.merge(df_pdc_sim_output, agg_sm100, on=['mergekey_TypeV2', 'mergekey_JourLiv'], how='left')
    df_pdc_sim_output['Commande SM à 100%'] = df_pdc_sim_output['temp_CmdSM100'].fillna(0.0)
    df_pdc_sim_output.drop(columns=['temp_CmdSM100'], inplace=True, errors='ignore')

    print("  Agrégation CF_initial pour 'Commande optimisée' (état initial)...")
    if 'CF_initial_pour_agregation' in df_detail_normalized.columns:
        agg_cmd_opt_initial = df_detail_normalized.groupby(['mergekey_TypeV2', 'mergekey_JourLiv'], as_index=False)['CF_initial_pour_agregation'].sum()
        agg_cmd_opt_initial.rename(columns={'CF_initial_pour_agregation': 'temp_CmdOpt'}, inplace=True)
        df_pdc_sim_output = pd.merge(df_pdc_sim_output, agg_cmd_opt_initial, on=['mergekey_TypeV2', 'mergekey_JourLiv'], how='left')
        df_pdc_sim_output['Commande optimisée'] = df_pdc_sim_output['temp_CmdOpt'].fillna(0.0)
        df_pdc_sim_output.drop(columns=['temp_CmdOpt'], inplace=True, errors='ignore')
    else: 
        df_pdc_sim_output['Commande optimisée'] = 0.0
        
    df_pdc_sim_output.drop(columns=['mergekey_TypeV2', 'mergekey_JourLiv'], inplace=True, errors='ignore')
    df_pdc_sim_output['Type de produits V2'] = df_pdc_sim_output['Type de produits V2_orig'] 
    df_pdc_sim_output.drop(columns=['Type de produits V2_orig'], inplace=True, errors='ignore')

    # --- Calculs finaux pour la table de simulation ---
    df_pdc_sim_output['Tolérance'] = df_pdc_sim_output['PDC'] * ALERTE_SURCHARGE_NIVEAU_1
    df_pdc_sim_output['Tolérance'].fillna(0.0, inplace=True)
    df_pdc_sim_output['Différence PDC / Commande'] = df_pdc_sim_output['PDC'] - df_pdc_sim_output['En-cours'] - df_pdc_sim_output['Commande optimisée']
    df_pdc_sim_output['Différence absolue'] = df_pdc_sim_output['Différence PDC / Commande'].abs()
    df_pdc_sim_output['Variation PDC'] = np.where(
        (df_pdc_sim_output['PDC'].notna()) & (df_pdc_sim_output['PDC'] != 0), 
        df_pdc_sim_output['Différence PDC / Commande'] / df_pdc_sim_output['PDC'], 0.0)
    df_pdc_sim_output['Variation absolue PDC'] = df_pdc_sim_output['Variation PDC'].abs()
    df_pdc_sim_output['Capage Borne Max ?'] = "Non" 

    final_ordered_columns_excel = [
        "Type de produits V2", "Type de produits", "Délai standard livraison (jours)", "Jour livraison",
        "PDC", "En-cours", "Commande optimisée", 
        "Différence PDC / Commande", "Différence absolue", "Tolérance",
        "Variation PDC", "Variation absolue PDC", "Capage Borne Max ?", "Commande SM à 100%",
        "Poids du A/C max", "Top 500", "Top 3000", "Autre", "Moyenne",
        "Limite Basse Top 500", "Limite Basse Top 3000", "Limite Basse Autre",
        "Limite Haute Top 500", "Limite Haute Top 3000", "Limite Haute Autre",
        "Min Facteur", "Max Facteur", "Boost PDC"
    ]
    for col in final_ordered_columns_excel: 
        if col not in df_pdc_sim_output.columns:
            df_pdc_sim_output[col] = np.nan if col not in ["Capage Borne Max ?"] else "Non"
            
    if df_pdc_sim_output.empty:
        print("  ParametresApprosGenerator: Table PDC_Sim vide. Retour d'un DataFrame vide structuré.")
        return pd.DataFrame(columns=final_ordered_columns_excel)

    df_final_excel_output = df_pdc_sim_output[final_ordered_columns_excel].copy()
    
    print(f"ParametresApprosGenerator: Table PDC_Sim (input pour optimisation) créée avec {len(df_final_excel_output)} lignes.")
    # print("Aperçu des colonnes de commande (devraient refléter les agrégats BQ et CF_initial):")
    # print(df_final_excel_output[['Type de produits V2', 'Jour livraison', 'PDC', 'En-cours', 'Commande SM à 100%', 'Commande optimisée']].head(len(df_final_excel_output)).to_string())
    return df_final_excel_output

def generate_complete_pdc_sim_excel(df_detail_source_arg, output_excel_path='PDC_Sim.xlsx'):
    print(f"\nParametresApprosGenerator: Génération de la feuille Excel '{output_excel_path}'...")
    
    # df_detail_source_arg est déjà un DataFrame passé par main.py
    df_detail_data = df_detail_source_arg
    if not isinstance(df_detail_data, pd.DataFrame) or df_detail_data.empty:
        print(f"  ERREUR: df_detail_source passé à generate_complete_pdc_sim_excel est invalide ou vide.")
        return

    print(f"  Fichier détail source (DataFrame) chargé ({len(df_detail_data)} lignes).")

    print("  Chargement et traitement des données PDC Perm (version BRUTE)...")
    df_pdc_perm_summary = ModulePDC.get_RAW_pdc_perm_data_for_optim() 

    print("  Chargement et traitement des données EnCours...")
    df_encours_summary = ModuleEncours.get_processed_data(formatted=False) 

    df_pdc_sim_complet = create_full_pdc_sim_table(df_detail_data, df_pdc_perm_summary, df_encours_summary)

    if df_pdc_sim_complet.empty:
        print(f"  ERREUR: La table PDC_Sim complète est vide. Abandon de l'écriture Excel.")
        return

    try:
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            df_export_final = df_pdc_sim_complet.copy()
            if 'Jour livraison' in df_export_final.columns:
                df_export_final['Jour livraison'] = pd.to_datetime(df_export_final['Jour livraison'], errors='coerce').dt.date
            
            df_export_final.to_excel(writer, sheet_name='PDC_Sim', index=False, header=True)
            worksheet = writer.sheets['PDC_Sim']
            if 'Jour livraison' in df_export_final.columns:
                date_col_letter_idx = df_export_final.columns.get_loc("Jour livraison") + 1
                for row_idx_xl in range(2, worksheet.max_row + 1): 
                    cell = worksheet.cell(row=row_idx_xl, column=date_col_letter_idx)
                    if isinstance(cell.value, (datetime.datetime, datetime.date)): 
                        cell.number_format = 'DD/MM/YYYY' 
        print(f"Fichier Excel '{output_excel_path}' généré avec succès.")
    except Exception as e:
        print(f"ERREUR lors de l'écriture du fichier Excel '{output_excel_path}': {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    print("--- Test autonome de ParametresApprosGenerator avec merged_predictions.csv réel ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detail_source_path = os.path.join(script_dir, 'merged_predictions.csv') 
    if not os.path.exists(detail_source_path):
        print(f"ERREUR TEST: Fichier {detail_source_path} non trouvé! Veuillez le générer avec main.py.")
        exit()
    df_detail_real = pd.read_csv(detail_source_path, sep=';', encoding='latin1', low_memory=False)
    print(f"  Fichier '{os.path.basename(detail_source_path)}' chargé ({len(df_detail_real)} lignes).")
    
    # Appel de la fonction principale de ce module pour générer PDC_Sim.xlsx
    generate_complete_pdc_sim_excel(df_detail_real, output_excel_path='PDC_Sim_Input_For_Optim.xlsx')

    print("\n--- Fin du test autonome ---")