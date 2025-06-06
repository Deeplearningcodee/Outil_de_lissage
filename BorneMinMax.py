# --- START OF FILE BorneMinMax.py ---

import pandas as pd
import os
import numpy as np
import MacroParam # Assurez-vous que ce module est accessible
try:
    from CommandeFinale import _calculate_by_facteur_lissage_besoin_brut_row
except ImportError:
    print("ERREUR CRITIQUE (BorneMinMax.py): Impossible d'importer _calculate_by_facteur_lissage_besoin_brut_row de CommandeFinale.py")
    # Définir une fonction factice pour éviter un crash, mais BY ne sera pas correct
    def _calculate_by_facteur_lissage_besoin_brut_row(detail_row_dict, j, k, l):
        return 1.0 

def _calculer_borne_min_interne(row, params):
    """Fonction interne pour calculer BW."""
    exclusion = row.get('Exclusion Lissage', "RAS")
    periode_vente = pd.to_numeric(row.get('Période de vente finale (Jours)', 0), errors='coerce')
    if pd.isna(periode_vente): periode_vente = 0
    macro_categorie = str(row.get('Macro-catégorie', ""))
    top = str(row.get('Top', ""))
    nb_commande_semaine_final = row.get('Nb de commande / Semaine Final', 6)
    if pd.isna(nb_commande_semaine_final) or nb_commande_semaine_final == "": nb_commande_semaine_final = 6
    else:
        try: nb_commande_semaine_final = float(nb_commande_semaine_final)
        except: nb_commande_semaine_final = 6
    prev_promo_c1_l2 = pd.to_numeric(row.get('Prev Promo C1-L2', 0), errors='coerce')
    if pd.isna(prev_promo_c1_l2): prev_promo_c1_l2 = 0
    stock = pd.to_numeric(row.get('STOCK', 0), errors='coerce')
    if pd.isna(stock): stock = 0
    ral = pd.to_numeric(row.get('RAL', 0), errors='coerce')
    if pd.isna(ral) and 'RAL' not in row: ral_check = 0 
    else: ral_check = ral if pd.notna(ral) else 0

    condition1 = (exclusion == params['EXCLUSION_TOTALE'] or 
                  exclusion == params['AUGMENTATION_OK_DIMINUTION_IMPOSSIBLE'] or
                  exclusion == params['DIMINUTION_OK_AUGMENTATION_IMPOSSIBLE'])
    condition2 = pd.isna(exclusion) or exclusion == "" or exclusion == 0
    condition3 = (periode_vente < params['PERIODE_VENTE_MIN_AUGMENTATION'] and 
                  macro_categorie != "5 - FL" and 
                  top != "Autre")
    condition4 = nb_commande_semaine_final < params['NB_JOURS_COMMANDE_PAR_SEMAINE_MIN']
    condition5 = prev_promo_c1_l2 > 0 and params['EMPECHER_BAISSE_COMMANDES_PROMO'] == "Oui"
    
    if condition1 or condition2 or condition3 or condition4 or condition5:
        return 1.0
    elif stock == 0 and ral_check == 0:
         return params['TAUX_MIN_BESOIN_BRUT_RUPTURE']
    else:
        return 0.0

def _calculer_borne_max_interne(row, params):
    """Fonction interne pour calculer BX."""
    exclusion = row.get('Exclusion Lissage', "RAS")
    periode_vente = pd.to_numeric(row.get('Période de vente finale (Jours)', 0), errors='coerce')
    if pd.isna(periode_vente): periode_vente = 0
    macro_categorie = str(row.get('Macro-catégorie', ""))
    
    condition1 = (exclusion == params['EXCLUSION_TOTALE'] or 
                  exclusion == params['DIMINUTION_OK_AUGMENTATION_IMPOSSIBLE'])
    condition2 = pd.isna(exclusion) or exclusion == "" or exclusion == 0
    condition3 = (periode_vente < params['PERIODE_VENTE_MIN_AUGMENTATION'] and 
                  macro_categorie != "5 - FL")
    
    if condition1 or condition2 or condition3:
        return 1.0
    else:
        return 10.0

def calculate_initial_exclusions_and_bornes(merged_df):
    """
    Calcule 'Exclusion Lissage', 'Borne Min Facteur multiplicatif lissage' (BW),
    'Borne Max Facteur multiplicatif lissage' (BX), et initialise 
    'Facteur multiplicatif Lissage besoin brut' (BY) à 1.0.
    """
    print("BorneMinMax: Calcul initial des exclusions et bornes BW, BX...")
    params = {
        "PERIODE_VENTE_MIN_AUGMENTATION": MacroParam.get_param_value("periode_vente_min_augmentation", 12),
        "NB_JOURS_COMMANDE_PAR_SEMAINE_MIN": MacroParam.get_param_value("nb_jours_commande_par_semaine_min", 3),
        "TAUX_MIN_BESOIN_BRUT_RUPTURE": MacroParam.get_param_value("taux_min_besoin_brut_rupture", 0.5),
        "EMPECHER_BAISSE_COMMANDES_PROMO": MacroParam.get_param_value("empecher_baisse_commandes_promo", "Oui"),
        "EXCLUSION_TOTALE": "Exclusion Totale",
        "AUGMENTATION_OK_DIMINUTION_IMPOSSIBLE": "Augmentation OK - Diminution Impossible",
        "DIMINUTION_OK_AUGMENTATION_IMPOSSIBLE": "Diminution OK - Augmentation Impossible"
    }
    
    exclusion_file = os.path.join(os.path.dirname(__file__), 'Exclusion.xlsx')
    if not os.path.exists(exclusion_file):
        print(f"  Attention: Fichier {exclusion_file} non trouvé. 'Exclusion Lissage' sera 'RAS'.")
        merged_df['Exclusion Lissage'] = "RAS"
    else:
        try:
            # print(f"  Chargement du fichier d'exclusions: {os.path.basename(exclusion_file)}")
            exclusion_df = pd.read_excel(exclusion_file, engine='openpyxl')
            required_columns = ['CDBase', 'Type']
            missing_columns = [col for col in required_columns if col not in exclusion_df.columns]
            
            if missing_columns:
                print(f"  Attention: Colonnes manquantes dans Exclusion.xlsx: {missing_columns}. 'Exclusion Lissage' sera 'RAS'.")
                merged_df['Exclusion Lissage'] = "RAS"
            else:
                exclusion_df['CDBase'] = exclusion_df['CDBase'].astype(str)
                merged_df['CODE_METI'] = merged_df['CODE_METI'].astype(str) # Assurez-vous que CODE_METI existe et est str
                exclusion_dict = dict(zip(exclusion_df['CDBase'], exclusion_df['Type']))
                merged_df['Exclusion Lissage'] = merged_df['CODE_METI'].apply(
                    lambda code: exclusion_dict.get(code, "RAS")
                )
                # exclusion_count = merged_df[merged_df['Exclusion Lissage'] != "RAS"].shape[0]
                # print(f"  Nombre d'exclusions trouvées: {exclusion_count}")
        except Exception as e:
            print(f"  Erreur lors du chargement de Exclusion.xlsx: {e}. 'Exclusion Lissage' sera 'RAS'.")
            merged_df['Exclusion Lissage'] = "RAS"

    merged_df['Borne Min Facteur multiplicatif lissage'] = merged_df.apply(
        lambda row: _calculer_borne_min_interne(row, params), axis=1
    )
    merged_df['Borne Max Facteur multiplicatif lissage'] = merged_df.apply(
        lambda row: _calculer_borne_max_interne(row, params), axis=1
    )
    
    # Initialiser BY à 1.0
    merged_df['Facteur multiplicatif Lissage besoin brut'] = 1.0
    
    print("  Colonnes Exclusion Lissage, BW, BX calculées. BY initialisé à 1.0.")
    return merged_df

def update_facteur_lissage_besoin_brut_from_optim(merged_df): # merged_df est le DataFrame Détail enrichi
    print("BorneMinMax: Mise à jour de 'Facteur multiplicatif Lissage besoin brut' (BY) avec les résultats d'optimisation...")
    
    script_dir = os.path.dirname(__file__) 
    optim_results_file = os.path.join(script_dir, "PDC_Sim_Optimized_Python.xlsx")
    
    if not os.path.exists(optim_results_file):
        print(f"  ATTENTION: Fichier {optim_results_file} non trouvé. BY ne sera pas mis à jour avec les facteurs optimisés.")
        if 'Facteur multiplicatif Lissage besoin brut' not in merged_df.columns:
            merged_df['Facteur multiplicatif Lissage besoin brut'] = 1.0 
        return merged_df
        
    try:
        df_optim = pd.read_excel(optim_results_file, engine='openpyxl')
        
        col_opt_j = 'Top 500' 
        col_opt_k = 'Top 3000' 
        col_opt_l = 'Autre' 
        col_tpv2_optim_PDC = 'Type de produits V2' # Nom dans PDC_Sim_Optimized_Python.xlsx
        col_jour_optim_PDC = 'Jour livraison'      # Nom dans PDC_Sim_Optimized_Python.xlsx
        
        optim_cols_needed = [col_tpv2_optim_PDC, col_jour_optim_PDC, col_opt_j, col_opt_k, col_opt_l]
        
        missing_optim_cols = [col for col in optim_cols_needed if col not in df_optim.columns]
        if missing_optim_cols:
            print(f"  ATTENTION: Colonnes manquantes dans {optim_results_file}: {missing_optim_cols}. BY non mis à jour.")
            if 'Facteur multiplicatif Lissage besoin brut' not in merged_df.columns:
                 merged_df['Facteur multiplicatif Lissage besoin brut'] = 1.0
            return merged_df

        # --- Standardisation de df_optim ---
        df_optim[col_tpv2_optim_PDC] = df_optim[col_tpv2_optim_PDC].astype(str).str.strip().str.lower()
        df_optim[col_jour_optim_PDC] = pd.to_datetime(df_optim[col_jour_optim_PDC], errors='coerce')
        
        # Convertir les facteurs optimisés en numérique, remplacer NaN par 1.0 (pas d'effet de lissage) and round to 2 decimals
        for col_facteur in [col_opt_j, col_opt_k, col_opt_l]:
            df_optim[col_facteur] = pd.to_numeric(df_optim[col_facteur], errors='coerce').fillna(1.0).round(2)

        df_optim_for_merge = df_optim[optim_cols_needed].copy()
        df_optim_for_merge.dropna(subset=[col_tpv2_optim_PDC, col_jour_optim_PDC], inplace=True)
        df_optim_for_merge.drop_duplicates(subset=[col_tpv2_optim_PDC, col_jour_optim_PDC], keep='first', inplace=True)

        # --- Préparation de merged_df (DataFrame Détail) ---
        # Noms des colonnes dans merged_df (Détail) avant la fusion
        col_tpv2_detail = 'Type de produits V2' 
        col_jour_detail = 'DATE_LIVRAISON_V2'

        if col_tpv2_detail not in merged_df.columns or col_jour_detail not in merged_df.columns:
            print(f"  ERREUR BorneMinMax UpdateBY: Colonnes de jointure '{col_tpv2_detail}' ou '{col_jour_detail}' non trouvées dans merged_df (Détail).")
            if 'Facteur multiplicatif Lissage besoin brut' not in merged_df.columns:
                 merged_df['Facteur multiplicatif Lissage besoin brut'] = 1.0
            return merged_df

        # Assurer les types pour la fusion dans merged_df 
        merged_df[col_tpv2_detail] = merged_df[col_tpv2_detail].astype(str).str.strip().str.lower()
        merged_df[col_jour_detail] = pd.to_datetime(merged_df[col_jour_detail], errors='coerce')
        
        # --- Fusion ---
        print(f"  DEBUG BorneMinMax - Fusion sur '{col_tpv2_detail}' (Détail) avec '{col_tpv2_optim_PDC}' (Optim)")
        print(f"  DEBUG BorneMinMax - Fusion sur '{col_jour_detail}' (Détail) avec '{col_jour_optim_PDC}' (Optim)")
        
        # La fusion ajoute les colonnes PY_Opt_J, PY_Opt_K, PY_Opt_L à merged_df
        merged_df_with_factors = pd.merge(
            merged_df,
            df_optim_for_merge, 
            left_on=[col_tpv2_detail, col_jour_detail],
            right_on=[col_tpv2_optim_PDC, col_jour_optim_PDC],
            how='left' 
        )
        
        # Remplir les facteurs non trouvés avec 1.0 (pas de lissage optimisé)
        merged_df_with_factors[col_opt_j] = merged_df_with_factors[col_opt_j].fillna(1.0)
        merged_df_with_factors[col_opt_k] = merged_df_with_factors[col_opt_k].fillna(1.0)
        merged_df_with_factors[col_opt_l] = merged_df_with_factors[col_opt_l].fillna(1.0)

        nb_initial = len(merged_df)
        nb_matched = merged_df_with_factors[col_opt_j].notna().sum() # Vérifier après fillna n'est plus utile, on vérifie avant
                                                                     # Ou plutôt : vérifier combien de lignes ont reçu une valeur NON-1.0
        
        print(f"  DEBUG BorneMinMax - Fusion: {nb_initial} lignes initiales dans détail.")
        print(f"  DEBUG BorneMinMax - Nombre de lignes de détail où JKL optimisés ont été appliqués (non-NaN avant fillna): {merged_df_with_factors[col_opt_j].count() - merged_df_with_factors[col_opt_j].isna().sum()}") # Un peu alambiqué

        # --- Calcul de la colonne BY en utilisant les facteurs optimisés ---
        # S'assurer que les colonnes nécessaires pour _calculate_by_facteur_lissage_besoin_brut_row existent dans merged_df_with_factors
        required_for_by = ['Top', 'Borne Min Facteur multiplicatif lissage', 'Borne Max Facteur multiplicatif lissage']
        for col_by_req in required_for_by:
            if col_by_req not in merged_df_with_factors.columns:
                print(f"    ATTENTION BorneMinMax: Colonne '{col_by_req}' manquante pour recalculer BY. BY pourrait être incorrect.")
                # Mettre une valeur par défaut pour éviter un crash, mais le calcul sera faux
                if 'lissage' in col_by_req: merged_df_with_factors[col_by_req] = 0.0 if "Min" in col_by_req else 10.0
                else: merged_df_with_factors[col_by_req] = 'autre'
        
        print(f"  Calcul de 'Facteur multiplicatif Lissage besoin brut' (BY) en utilisant PY_Opt_J,K,L...")
        merged_df_with_factors['Facteur multiplicatif Lissage besoin brut'] = merged_df_with_factors.apply(
            lambda row: _calculate_by_facteur_lissage_besoin_brut_row(
                row.to_dict(), # Passer la ligne entière comme dictionnaire
                row[col_opt_j],
                row[col_opt_k],
                row[col_opt_l]
            ), axis=1
        )
        
        
        # Pour s'assurer qu'on retourne le df avec les colonnes originales de merged_df + BY mis à jour :
        final_cols = list(merged_df.columns)
        if 'Facteur multiplicatif Lissage besoin brut' not in final_cols:
            # Si merged_df n'avait pas la colonne, on la garde de merged_df_with_factors
            pass # Elle sera déjà dans merged_df_with_factors
        else:
            # Si merged_df avait déjà la colonne, on s'assure de prendre la version mise à jour
            # Cela est fait car merged_df_with_factors est basé sur merged_df
            pass

        # On peut choisir de ne retourner que les colonnes initiales de merged_df, avec BY mis à jour
        # Ou retourner merged_df_with_factors qui contient aussi PY_Opt_J,K,L
        # Pour l'instant, retournons merged_df_with_factors pour inspection.
        print("  Colonne 'Facteur multiplicatif Lissage besoin brut' (BY) mise à jour.")
        return merged_df_with_factors

    except Exception as e:
        print(f"  ERREUR lors de la mise à jour de BY depuis {optim_results_file}: {e}")
        if 'Facteur multiplicatif Lissage besoin brut' not in merged_df.columns:
             merged_df['Facteur multiplicatif Lissage besoin brut'] = 1.0
        return merged_df
        
if __name__ == "__main__":
    print("--- Test du module BorneMinMax.py ---")
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Utiliser abspath pour plus de robustesse

    # --- 1. Créer un fichier merged_predictions.csv simulé ---
    # Ce DataFrame simule le merged_df que main.py passerait à update_facteur_lissage_besoin_brut_from_optim
    # Il doit contenir les colonnes nécessaires lues par les fonctions de BorneMinMax
    # et celles nécessaires pour la fusion et le calcul de BY.
    data_for_merged_predictions = {
        'CODE_METI': ['M001', 'M002', 'M003', 'M004_S_NO_MATCH', 'M005_S_MATCH'],
        'Type de produits V2': [ # Sera renommé en BM par load_data puis potentiellement retour à TPV2
            'Sec Méca - A/B', 
            'Sec Méca - A/C', 
            'Frais Méca', 
            'Surgelés', # Cas où la date ne correspondra pas dans optim_results
            'Surgelés'  # Cas où la date correspondra
        ], 
        'DATE_LIVRAISON_V2': [
            pd.to_datetime('2025-05-21'), 
            pd.to_datetime('2025-05-21'), 
            pd.to_datetime('2025-05-22'),
            pd.to_datetime('2025-05-22'), # Date différente de l'optim pour M004
            pd.to_datetime('2025-05-22')  # Date qui correspondra pour M005
        ],
        'Top': ['Top 500', 'Autre', 'Top 3000', 'Autre', 'Autre'],
        # Colonnes pour _calculer_borne_min/max_interne
        'Exclusion Lissage': ["RAS", "Exclusion Totale", "RAS", "RAS", "RAS"],
        'Période de vente finale (Jours)': [20, 5, 15, 0, 30], # Pour tester conditions
        'Macro-catégorie': ['1 - TR', '1 - TR', '2 - FR', '3 - SU', '3 - SU'],
        'Nb de commande / Semaine Final': [4, 2, 5, 6, 6],
        'Prev Promo C1-L2': [0, 0, 0, 0, 0],
        'STOCK': [10, 5, 0, 0, 2], # M003 et M004 en rupture
        'RAL': [0, 0, 0, 0, 1]
    }
    df_merged_predictions_sim = pd.DataFrame(data_for_merged_predictions)
    # Standardiser comme le ferait load_data (important pour la fusion dans update_facteur)
    # Dans le flux réel, merged_df aurait 'BM' et 'DATE_LIVRAISON_V2' standardisés
    df_merged_predictions_sim.rename(columns={'Type de produits V2': 'BM'}, inplace=True)
    df_merged_predictions_sim['BM'] = df_merged_predictions_sim['BM'].astype(str).str.strip().str.lower()
    df_merged_predictions_sim['DATE_LIVRAISON_V2'] = pd.to_datetime(df_merged_predictions_sim['DATE_LIVRAISON_V2'])
    df_merged_predictions_sim['Top'] = df_merged_predictions_sim['Top'].astype(str).str.strip().str.lower()

    merged_predictions_filepath_test = os.path.join(current_dir, "test_merged_predictions.csv")
    df_merged_predictions_sim.to_csv(merged_predictions_filepath_test, sep=';', index=False, encoding='latin1')
    print(f"Fichier simulé '{os.path.basename(merged_predictions_filepath_test)}' créé.")

    # --- 2. Créer un fichier PDC_Sim_Optimized_Python.xlsx simulé ---
    data_for_optim_results = {
        'Type de produits V2': ['Sec Méca - A/B', 'Sec Méca - A/C', 'Frais Méca', 'Surgelés', 'Autre Cat'],
        'Jour livraison': [pd.to_datetime('2025-05-21'), pd.to_datetime('2025-05-21'), pd.to_datetime('2025-05-22'), pd.to_datetime('2025-05-22'), pd.to_datetime('2025-05-22')], 
        'PY_Opt_J': [1.10, 1.00, 0.90, 0.95, 1.0], # Facteur pour Top 500
        'PY_Opt_K': [1.15, 1.00, 0.85, 0.92, 1.0], # Facteur pour Top 3000
        'PY_Opt_L': [1.20, 0.95, 0.80, 0.88, 1.0]  # Facteur pour Autre
    }
    df_optim_results_sim = pd.DataFrame(data_for_optim_results)
    optim_results_filepath_test = os.path.join(current_dir, "PDC_Sim_Optimized_Python.xlsx")
    df_optim_results_sim.to_excel(optim_results_filepath_test, index=False)
    print(f"Fichier simulé '{os.path.basename(optim_results_filepath_test)}' créé.")

    # --- 3. Créer un fichier Exclusion.xlsx simulé (si calculate_initial_exclusions_and_bornes en a besoin) ---
    if not os.path.exists(os.path.join(current_dir, "Exclusion.xlsx")):
        pd.DataFrame({'CDBase': ['METI_EXCLU'], 'Type': ["Exclusion Totale"]}).to_excel(os.path.join(current_dir, "Exclusion.xlsx"), index=False)
        print("Fichier simulé 'Exclusion.xlsx' créé.")
    
    # --- Début des tests ---
    try:
        # Lire le merged_predictions.csv simulé
        df_to_test = pd.read_csv(merged_predictions_filepath_test, sep=';', encoding='latin1')
        # Convertir les colonnes de date
        df_to_test['DATE_LIVRAISON_V2'] = pd.to_datetime(df_to_test['DATE_LIVRAISON_V2'])


        print("\n--- Test de calculate_initial_exclusions_and_bornes ---")
        df_with_initial_bornes = calculate_initial_exclusions_and_bornes(df_to_test.copy())
        print("\nRésultats après calcul initial (BY devrait être 1.0) :")
        cols_to_show_initial = ['CODE_METI', 'BM', 'DATE_LIVRAISON_V2', 'Top',
                                'Exclusion Lissage', 'Borne Min Facteur multiplicatif lissage', 
                                'Borne Max Facteur multiplicatif lissage', 'Facteur multiplicatif Lissage besoin brut']
        print(df_with_initial_bornes[[c for c in cols_to_show_initial if c in df_with_initial_bornes.columns]].to_string())
        
        # Vérification que BY est bien initialisé à 1.0
        assert df_with_initial_bornes['Facteur multiplicatif Lissage besoin brut'].eq(1.0).all(), \
            "Erreur: BY n'a pas été initialisé à 1.0 partout."
        print("Vérification initiale de BY=1.0 : OK")

        # Vérification de quelques bornes calculées
        # METI001: Sec Méca - A/B, Top 500, Période vente > Min, Pas promo, Stock OK -> BW=0, BX=10
        row_m001_initial = df_with_initial_bornes[df_with_initial_bornes['CODE_METI'] == 'M001'].iloc[0]
        assert abs(row_m001_initial['Borne Min Facteur multiplicatif lissage'] - 0.0) < 0.001, "Erreur BW M001"
        assert abs(row_m001_initial['Borne Max Facteur multiplicatif lissage'] - 10.0) < 0.001, "Erreur BX M001"
        print("Vérification bornes M001 : OK")

        # METI002: Sec Méca - A/C, Exclusion Totale -> BW=1, BX=1
        row_m002_initial = df_with_initial_bornes[df_with_initial_bornes['CODE_METI'] == 'M002'].iloc[0]
        assert abs(row_m002_initial['Borne Min Facteur multiplicatif lissage'] - 1.0) < 0.001, "Erreur BW M002"
        assert abs(row_m002_initial['Borne Max Facteur multiplicatif lissage'] - 1.0) < 0.001, "Erreur BX M002"
        print("Vérification bornes M002 : OK")

        # METI003: Frais Méca, Stock=0, RAL=0 -> BW=0.5 (taux_min_besoin_brut_rupture), BX=10
        row_m003_initial = df_with_initial_bornes[df_with_initial_bornes['CODE_METI'] == 'M003'].iloc[0]
        taux_rupture = MacroParam.get_param_value("taux_min_besoin_brut_rupture", 0.5)
        assert abs(row_m003_initial['Borne Min Facteur multiplicatif lissage'] - taux_rupture) < 0.001, f"Erreur BW M003, attendu {taux_rupture}"
        assert abs(row_m003_initial['Borne Max Facteur multiplicatif lissage'] - 10.0) < 0.001, "Erreur BX M003"
        print("Vérification bornes M003 : OK")


        print("\n--- Test de update_facteur_lissage_besoin_brut_from_optim ---")
        # Utiliser le df qui a déjà les colonnes BW et BX calculées
        df_updated_by = update_facteur_lissage_besoin_brut_from_optim(df_with_initial_bornes.copy())
        print("\nRésultats après mise à jour de BY avec facteurs optimisés :")
        print(df_updated_by[['CODE_METI', 'BM', 'DATE_LIVRAISON_V2', 'Top',
                             'Borne Min Facteur multiplicatif lissage', 
                             'Borne Max Facteur multiplicatif lissage', 
                             'Facteur multiplicatif Lissage besoin brut']].to_string())

        # Vérifications spécifiques de BY
        # M001: Sec Méca - A/B (Top 500), J_optim=1.10. BW=0, BX=10. BY attendu = MAX(0, MIN(10, 1.10)) = 1.10
        row_m001_updated = df_updated_by[df_updated_by['CODE_METI'] == 'M001'].iloc[0]
        assert abs(row_m001_updated['Facteur multiplicatif Lissage besoin brut'] - 1.10) < 0.001, \
            f"Erreur BY M001. Attendu 1.10, Obtenu {row_m001_updated['Facteur multiplicatif Lissage besoin brut']}"
        print("Vérification BY M001 : OK")

        # M002: Sec Méca - A/C (Autre), L_optim=0.95. BW=1, BX=1 (car Exclusion Totale). BY attendu = MAX(1, MIN(1, 0.95)) = 1.0
        row_m002_updated = df_updated_by[df_updated_by['CODE_METI'] == 'M002'].iloc[0]
        assert abs(row_m002_updated['Facteur multiplicatif Lissage besoin brut'] - 1.0) < 0.001, \
            f"Erreur BY M002. Attendu 1.0, Obtenu {row_m002_updated['Facteur multiplicatif Lissage besoin brut']}"
        print("Vérification BY M002 : OK")
        
        # M003: Frais Méca (Top 3000), K_optim=0.85. BW=0.5, BX=10. BY attendu = MAX(0.5, MIN(10, 0.85)) = 0.85
        row_m003_updated = df_updated_by[df_updated_by['CODE_METI'] == 'M003'].iloc[0]
        assert abs(row_m003_updated['Facteur multiplicatif Lissage besoin brut'] - 0.85) < 0.001, \
            f"Erreur BY M003. Attendu 0.85, Obtenu {row_m003_updated['Facteur multiplicatif Lissage besoin brut']}"
        print("Vérification BY M003 : OK")

        # M004_S_NO_MATCH: Surgelés (Autre), pas de match de date dans optim_results. L_optim sera NaN -> 1.0 par défaut. BW=0.5, BX=10. BY attendu = MAX(0.5, MIN(10, 1.0)) = 1.0
        row_m004_updated = df_updated_by[df_updated_by['CODE_METI'] == 'M004_S_NO_MATCH'].iloc[0]
        assert abs(row_m004_updated['Facteur multiplicatif Lissage besoin brut'] - 1.0) < 0.001, \
            f"Erreur BY M004. Attendu 1.0, Obtenu {row_m004_updated['Facteur multiplicatif Lissage besoin brut']}"
        print("Vérification BY M004 (pas de match optim) : OK")
        
        # M005_S_MATCH: Surgelés (Autre), L_optim=0.88. BW=0.0 (car STOCK > 0), BX=10. BY attendu = MAX(0.0, MIN(10, 0.88)) = 0.88
        row_m005_updated = df_updated_by[df_updated_by['CODE_METI'] == 'M005_S_MATCH'].iloc[0]
        assert abs(row_m005_updated['Facteur multiplicatif Lissage besoin brut'] - 0.88) < 0.001, \
            f"Erreur BY M005. Attendu 0.88, Obtenu {row_m005_updated['Facteur multiplicatif Lissage besoin brut']}"
        print("Vérification BY M005 (match optim) : OK")

        print("\n--- Tous les tests pour BorneMinMax.py ont réussi ! ---")

    except Exception as e:
        print(f"\nERREUR PENDANT LE TEST du module BorneMinMax.py: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Nettoyer les fichiers de test
        if os.path.exists(merged_predictions_filepath_test):
            os.remove(merged_predictions_filepath_test)
        if os.path.exists(optim_results_filepath_test):
            os.remove(optim_results_filepath_test)
        # if os.path.exists(os.path.join(current_dir, "Exclusion.xlsx")): # Le laisser s'il est utilisé par d'autres tests
            # os.remove(os.path.join(current_dir, "Exclusion.xlsx"))
        print("Fichiers de test nettoyés.")