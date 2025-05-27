# --- START OF FILE CalculAutresColonnes.py ---

import pandas as pd
import numpy as np
import os
import MacroParam # Pour accéder aux paramètres comme Macro!$B$84

def load_qtemaxi_data():
    # ... (début de la fonction inchangé) ...
    script_dir = os.path.dirname(__file__)
    csv_folder_path = os.path.join(script_dir, "CSV")
    qtemaxi_file_path = os.path.join(csv_folder_path, "Qtite_Maxi.csv")
    
    qtemaxi_map_col_e_val = {} 
    qtemaxi_map_col_i_val = {} 

    if not os.path.exists(qtemaxi_file_path):
        print(f"  ATTENTION CalculAutresColonnes: Fichier {os.path.basename(qtemaxi_file_path)} non trouvé dans {csv_folder_path}.")
        return qtemaxi_map_col_e_val, qtemaxi_map_col_i_val

    try:
        # print(f"  CalculAutresColonnes: Chargement de {os.path.basename(qtemaxi_file_path)}...")
        try:
            df_qtemaxi = pd.read_csv(qtemaxi_file_path, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            df_qtemaxi = pd.read_csv(qtemaxi_file_path, sep=';', encoding='latin1')
        
        print(f"  DEBUG QteMaxi Load: Colonnes trouvées dans Qtite_Maxi.csv: {df_qtemaxi.columns.tolist()}")

        key_col = 'ID2' 
        # ----- CORRECTION ICI -----
        val_col_e_and_f = 'Nombre Maxi Unités De Commande' # Doit correspondre exactement à la sortie du print ci-dessus
        # --------------------------
        val_col_i = 'QteMaxUS' 

        if key_col not in df_qtemaxi.columns:
            print(f"  ATTENTION CalculAutresColonnes: Colonne clé '{key_col}' manquante dans {os.path.basename(qtemaxi_file_path)}.")
            return {}, {} 

        df_qtemaxi[key_col] = df_qtemaxi[key_col].astype(str).str.strip()
        # print(f"  DEBUG QteMaxi Load: Premières clés '{key_col}' après strip: {df_qtemaxi[key_col].head().tolist()}")

        if val_col_e_and_f in df_qtemaxi.columns:
            # print(f"  DEBUG QteMaxi Load: Premières valeurs brutes '{val_col_e_and_f}': {df_qtemaxi[val_col_e_and_f].head().tolist()}")
            df_qtemaxi[val_col_e_and_f] = pd.to_numeric(df_qtemaxi[val_col_e_and_f], errors='coerce')
            # print(f"  DEBUG QteMaxi Load: Premières valeurs '{val_col_e_and_f}' après to_numeric: {df_qtemaxi[val_col_e_and_f].head().tolist()}")
            
            temp_map_e = df_qtemaxi.dropna(subset=[val_col_e_and_f]) 
            qtemaxi_map_col_e_val = pd.Series(
                temp_map_e[val_col_e_and_f].values, 
                index=temp_map_e[key_col]
            ).to_dict()
        else:
            print(f"  ATTENTION CalculAutresColonnes: Colonne valeur '{val_col_e_and_f}' manquante pour le mappage E/F.") # Ce message ne devrait plus apparaître

        if val_col_i in df_qtemaxi.columns:
            # print(f"  DEBUG QteMaxi Load: Premières valeurs brutes '{val_col_i}': {df_qtemaxi[val_col_i].head().tolist()}")
            df_qtemaxi[val_col_i] = pd.to_numeric(df_qtemaxi[val_col_i], errors='coerce')
            # print(f"  DEBUG QteMaxi Load: Premières valeurs '{val_col_i}' après to_numeric: {df_qtemaxi[val_col_i].head().tolist()}")

            temp_map_i = df_qtemaxi.dropna(subset=[val_col_i]) 
            qtemaxi_map_col_i_val = pd.Series(
                temp_map_i[val_col_i].values, 
                index=temp_map_i[key_col]
            ).to_dict()
        else:
            print(f"  ATTENTION CalculAutresColonnes: Colonne valeur '{val_col_i}' manquante pour le mappage I.")
            
        print(f"  CalculAutresColonnes: Dictionnaires QteMaxi chargés (E/F: {len(qtemaxi_map_col_e_val)} entrées, I: {len(qtemaxi_map_col_i_val)} entrées).")
        # if len(qtemaxi_map_col_e_val) < 5 and len(qtemaxi_map_col_e_val) > 0:
        #     print(f"    Exemples map_e (clé:valeur): {list(qtemaxi_map_col_e_val.items())[:5]}")
        # if len(qtemaxi_map_col_i_val) < 5 and len(qtemaxi_map_col_i_val) > 0:
        #     print(f"    Exemples map_i (clé:valeur): {list(qtemaxi_map_col_i_val.items())[:5]}")

    except Exception as e:
        print(f"  ERREUR CalculAutresColonnes: Chargement/traitement de {os.path.basename(qtemaxi_file_path)}: {e}.")
        return {}, {}
        
    return qtemaxi_map_col_e_val, qtemaxi_map_col_i_val

def calculate_quantite_maxi_ch(df, qtemaxi_map_e, qtemaxi_map_i):
    """Calcul de 'Quantité Maxi ?' (CH)"""
    print("  Calcul de 'Quantité Maxi ?' (CH)...")
    
    def calculate_ch_row(row):
        try:
            fournisseur_q_brut = str(row.get('FOURNISSEUR', '')).strip()
            # Normaliser le fournisseur pour enlever ".0" si c'est un entier
            if fournisseur_q_brut.endswith(".0"):
                try:
                    # Essayer de convertir en int puis en str pour enlever .0
                    fournisseur_q = str(int(float(fournisseur_q_brut)))
                except ValueError:
                    fournisseur_q = fournisseur_q_brut # Garder tel quel si la conversion échoue
            else:
                fournisseur_q = fournisseur_q_brut

            ifls_ai = str(row.get('IFLS', '')).strip()
            pcb_w = pd.to_numeric(row.get('PCB'), errors='coerce')
            code_meti_debug = str(row.get('CODE_METI', 'N/A'))

            if pd.isna(pcb_w) or pcb_w == 0 : pcb_w = 1

            cle_qtemaxi = f"{fournisseur_q}-{ifls_ai}" # Clé avec fournisseur normalisé

            val_e = qtemaxi_map_e.get(cle_qtemaxi)
            val_i = qtemaxi_map_i.get(cle_qtemaxi)

            # ... (reste du code de débogage et de calcul) ...
            if row.name < 20 or code_meti_debug == 'VOTRE_CODE_METI_SPECIFIQUE_POUR_TEST': # Augmenter le nombre de lignes pour le debug
                # print(f"    DEBUG CH - Ligne index {row.name}, CODE_METI: {code_meti_debug}")
                # print(f"      FOURNISSEUR_brut: '{fournisseur_q_brut}', FOURNISSEUR_norm: '{fournisseur_q}', IFLS: '{ifls_ai}' => Clé QtiteMaxi: '{cle_qtemaxi}'")
                # print(f"      Valeur E depuis map: {val_e} (type: {type(val_e)})")
                # print(f"      Valeur I depuis map: {val_i} (type: {type(val_i)})")
                if val_e is None: print("        val_e est None (clé non trouvée dans map_e)")
                # ... (autres prints de debug) ...

            if val_e is not None and pd.notna(val_e) and val_i is not None and pd.notna(val_i):
                terme1 = val_e * pcb_w 
                terme2 = val_i       
                # if row.name < 5 or code_meti_debug == 'VOTRE_CODE_METI_SPECIFIQUE_POUR_TEST':
                #     print(f"        Calcul CH: min({terme1}, {terme2}) = {min(terme1, terme2)}")
                return min(terme1, terme2)
            else:
                # ...
                return 9999999 
        except Exception as e:
            print(f"  ERREUR Calcul CH pour ligne index {row.name} (CODE_METI: {code_meti_debug}): {e}")
            return 9999999
                
    df['Quantité Maxi ?'] = df.apply(calculate_ch_row, axis=1)
    return df

def calculate_quantite_maxi_a_charger_ci(df):
    """Calcul de 'Quantité Maxi à charger ?' (CI)"""
    print("  Calcul de 'Quantité Maxi à charger ?' (CI)...")
    
    def calculate_ci_row(row):
        ce_val = pd.to_numeric(row.get('Commande optimisée avec arrondi et mini'), errors='coerce')
        ch_val = pd.to_numeric(row.get('Quantité Maxi ?'), errors='coerce')

        if pd.isna(ce_val) or pd.isna(ch_val):
            return "Non" # Ou une autre gestion d'erreur/valeur par défaut
        
        return "Oui" if ce_val > ch_val else "Non"

    df['Quantité Maxi à charger ?'] = df.apply(calculate_ci_row, axis=1)
    return df

def calculate_algo_meti_cj(df):
    """Calcul de 'Algo Meti' (CJ)"""
    print("  Calcul de 'Algo Meti' (CJ)...")

    # --- NOUVEAUX NOMS DE COLONNES ---
    col_qte_proposee1 = 'QTEPROPOSEE1'
    col_qte_proposee2 = 'QTEPROPOSEE2'
    # --- FIN NOUVEAUX NOMS ---

    # Vérifier si les colonnes existent (avec les nouveaux noms)
    if col_qte_proposee1 not in df.columns or col_qte_proposee2 not in df.columns:
        print(f"  ATTENTION CJ: Colonnes '{col_qte_proposee1}' ou '{col_qte_proposee2}' manquantes. Algo Meti sera 0.")
        df['Algo Meti'] = 0
        return df

    def calculate_cj_row(row):
        try:
            code_meti_debug = str(row.get('CODE_METI', 'N/A'))

            ae_qte_proposee1_raw = row.get(col_qte_proposee1) # Utiliser le nouveau nom
            af_qte_proposee2_raw = row.get(col_qte_proposee2) # Utiliser le nouveau nom

            ae_qte_proposee1 = pd.to_numeric(ae_qte_proposee1_raw, errors='coerce')
            af_qte_proposee2 = pd.to_numeric(af_qte_proposee2_raw, errors='coerce')
            
            # ... (le bloc de débogage peut rester, il utilisera les _raw et _numeric) ...
            if  code_meti_debug == '20300': 
                print(f"    DEBUG CJ - Ligne index {row.name}, CODE_METI: {code_meti_debug}")
                print(f"      {col_qte_proposee1} (raw): '{ae_qte_proposee1_raw}' -> (numeric): {ae_qte_proposee1}")
                print(f"      {col_qte_proposee2} (raw): '{af_qte_proposee2_raw}' -> (numeric): {af_qte_proposee2}")
                if pd.isna(ae_qte_proposee1): print("        ae_qte_proposee1 est NaN après conversion")
                if pd.isna(af_qte_proposee2): print("        af_qte_proposee2 est NaN après conversion")


            if pd.isna(ae_qte_proposee1) or pd.isna(af_qte_proposee2):
                return 0 
            
            resultat = 1 if ae_qte_proposee1 <= af_qte_proposee2 else 2
            return resultat
        except Exception as e_cj:
            return 0

    df['Algo Meti'] = df.apply(calculate_cj_row, axis=1)
    return df

def calculate_sm_a_charger_test_ck(df):
    """Calcul de 'SM A Charger Test' (CK) - Formule complexe"""
    print("  Calcul de 'SM A Charger Test' (CK)...")

    # Récupération des paramètres globaux depuis MacroParam
    macro_b84_valeur_attendue = MacroParam.get_param_value(
        "EXCLUSION_SM_A_CHARGER_TEST", 
        "Application d'un coefficient sur la prévision des flop casse" # Valeur par défaut si non trouvée
    ) 
    macro_c21_coeff_exception = MacroParam.get_param_value(
        "coefficient_exception_prevision", # C'est le nom que vous avez indiqué pour C21
        0.0 # Valeur par défaut si non trouvée
    ) 
    macro_c14_stock_source_pour_calcul = MacroParam.get_param_value(
        "SOURCE_STOCK_POUR_SM_CHARGE", 
        "RAO" # Valeur par défaut si non trouvée
    ) 
    
    # Imprimer les paramètres une seule fois pour éviter la redondance dans la boucle
    # (peut être activé si __name__ != "__main__" pour n'imprimer qu'en mode import)
    # if 'printed_ck_params' not in calculate_sm_a_charger_test_ck.__dict__: # Astuce pour n'imprimer qu'une fois
    print(f"    DEBUG CK Params (utilisés pour toutes les lignes):")
    print(f"      B84_Exclusion_Attendue ('Macro!$B$84'): '{macro_b84_valeur_attendue}'")
    print(f"      C21_CoeffException ('Macro-Param'!$C$21): {macro_c21_coeff_exception}")
    print(f"      C14_StockSrc ('Macro!$C$14'): '{macro_c14_stock_source_pour_calcul}'")
        # calculate_sm_a_charger_test_ck.printed_ck_params = True


    def calculate_ck_row(row):
        code_meti_debug = str(row.get('CODE_METI', 'N/A'))
        # Ajustez cette condition pour cibler les lignes que vous voulez déboguer
        print_debug_this_row = code_meti_debug in ['20300', '7757253', '20642'] # Ajoutez d'autres CODE_METI si besoin

        try:
            # --- Récupération et nettoyage des valeurs de la ligne ---
            ci_qte_maxi_charge = str(row.get('Quantité Maxi à charger ?', 'Non')).strip().lower()
            bv_exclusion_lue = str(row.get('Exclusion Lissage', '')).strip()
            
            y_stock = pd.to_numeric(row.get('STOCK'), errors='coerce')
            z_ral = pd.to_numeric(row.get('RAL'), errors='coerce')
            ad_coche_rao_raw = row.get('COCHE_RAO') # Garder raw pour vérifier si c'est NaN vs 54.0
            ad_coche_rao = pd.to_numeric(ad_coche_rao_raw, errors='coerce')
            
            by_facteur_lissage = pd.to_numeric(row.get('Facteur multiplicatif Lissage besoin brut'), errors='coerce')
            cf_cmd_opt_ts = pd.to_numeric(row.get('Commande optimisée avec arrondi et mini et TS'), errors='coerce')
            cd_cmd_opt_sans_arrondi = pd.to_numeric(row.get('Commande optimisée sans arrondi'), errors='coerce')
            cj_algo_meti = pd.to_numeric(row.get('Algo Meti'), errors='coerce') 
            ce_cmd_opt_arrondi_mini = pd.to_numeric(row.get('Commande optimisée avec arrondi et mini'), errors='coerce')
            # Utiliser les noms de colonnes sans accents que vous avez définis
            aa_qte_prev_init1 = pd.to_numeric(row.get('QTEPREVISIONINIT1'), errors='coerce')
            ab_qte_prev_init2 = pd.to_numeric(row.get('QTEPREVISIONINIT2'), errors='coerce')
            w_pcb = pd.to_numeric(row.get('PCB'), errors='coerce')
            aj_stock_reel = pd.to_numeric(row.get('STOCK_REEL'), errors='coerce')

            # Remplacer les NaN par 0 pour la plupart des calculs numériques
            y_stock = y_stock if pd.notna(y_stock) else 0.0
            z_ral = z_ral if pd.notna(z_ral) else 0.0
            # ad_coche_rao peut être NaN, la comparaison `ad_coche_rao == 54` gérera cela (NaN != 54)
            by_facteur_lissage = by_facteur_lissage if pd.notna(by_facteur_lissage) else 0.0
            cf_cmd_opt_ts = cf_cmd_opt_ts if pd.notna(cf_cmd_opt_ts) else 0.0
            cd_cmd_opt_sans_arrondi = cd_cmd_opt_sans_arrondi if pd.notna(cd_cmd_opt_sans_arrondi) else 0.0
            cj_algo_meti = cj_algo_meti if pd.notna(cj_algo_meti) else 0 # Si Algo Meti est 0, la logique interne le traitera
            ce_cmd_opt_arrondi_mini = ce_cmd_opt_arrondi_mini if pd.notna(ce_cmd_opt_arrondi_mini) else 0.0
            aa_qte_prev_init1 = aa_qte_prev_init1 if pd.notna(aa_qte_prev_init1) else 0.0
            ab_qte_prev_init2 = ab_qte_prev_init2 if pd.notna(ab_qte_prev_init2) else 0.0
            w_pcb = w_pcb if pd.notna(w_pcb) and w_pcb != 0 else 1.0 # Éviter division par zéro pour ARRONDI.SUP(W/4)
            aj_stock_reel = aj_stock_reel if pd.notna(aj_stock_reel) else 0.0

            if print_debug_this_row:
                print(f"    DEBUG CK Ligne {row.name} ({code_meti_debug}):")
                print(f"      Inputs: CI='{ci_qte_maxi_charge}', BV_Lue='{bv_exclusion_lue}', Y={y_stock:.2f}, Z={z_ral:.2f}, AD_raw='{ad_coche_rao_raw}' (num:{ad_coche_rao}), BY={by_facteur_lissage:.2f}")
                print(f"              CF={cf_cmd_opt_ts:.2f}, CD={cd_cmd_opt_sans_arrondi:.2f}, CJ={cj_algo_meti}, CE={ce_cmd_opt_arrondi_mini:.2f}")
                print(f"              AA={aa_qte_prev_init1:.2f}, AB={ab_qte_prev_init2:.2f}, W={w_pcb:.2f}, AJ={aj_stock_reel:.2f}")

            # --- Logique de la formule Excel ---
            if ci_qte_maxi_charge == "oui":
                if print_debug_this_row: print(f"      CK Cond 1 (CI='oui'): Résultat -> 0")
                return 0.0

            # Condition modifiée pour BV ("hack" pour RAS)
            condition_bv_ok_pour_calcul = (bv_exclusion_lue == macro_b84_valeur_attendue or 
                                           bv_exclusion_lue == "RAS")

            condition_speciale_1 = (
                condition_bv_ok_pour_calcul and
                macro_c21_coeff_exception == 0.0 and # Comparaison explicite avec float
                (y_stock + z_ral < 1.0) and
                ad_coche_rao == 54.0 # Comparaison explicite avec float
            )
            if print_debug_this_row:
                print(f"      CK Éval Cond Spéciale 1:")
                print(f"        cond_bv_ok (BV_Lue='{bv_exclusion_lue}' vs B84_Att='{macro_b84_valeur_attendue}' OR RAS)? {condition_bv_ok_pour_calcul}")
                print(f"        C21_Coeff ({macro_c21_coeff_exception}) == 0.0? {macro_c21_coeff_exception == 0.0}")
                print(f"        (Y ({y_stock:.2f}) + Z ({z_ral:.2f}) < 1.0)? {(y_stock + z_ral < 1.0)}")
                print(f"        AD ({ad_coche_rao}) == 54.0? {ad_coche_rao == 54.0}")
                print(f"        => Cond Spéciale 1 est {condition_speciale_1}")

            if condition_speciale_1:
                if print_debug_this_row: print(f"      CK Cond 2 (Spéciale 1): Résultat -> 1.0")
                return 1.0
            
            condition_ad_ok = (ad_coche_rao == 54.0)
            condition_globale_pour_max_calcul = (condition_bv_ok_pour_calcul and condition_ad_ok)

            if print_debug_this_row:
                print(f"      CK Éval Cond Globale pour MAX_Calcul:")
                print(f"        cond_bv_ok? {condition_bv_ok_pour_calcul}")
                print(f"        cond_ad_ok (AD=54.0)? {condition_ad_ok}")
                print(f"        => Cond Globale pour MAX_Calcul est {condition_globale_pour_max_calcul}")

            if condition_globale_pour_max_calcul:
                # Terme X = SI(ET(BY2>0;CF2>0);1;0)
                terme_x_base = 1.0 if (by_facteur_lissage > 0 and cf_cmd_opt_ts > 0) else 0.0
                
                # Calcul interne pour Terme Y (avant arrondi)
                valeur_interne_pour_arrondi = 0.0
                if cd_cmd_opt_sans_arrondi == 0: # SI(CD7=0;0;...)
                    valeur_interne_pour_arrondi = 0.0
                else: # ...SI(CJ7=1;CE7-AA7-...;CE7-AB7-...)
                    arrondi_sup_w_div_4 = np.ceil(w_pcb / 4.0)
                    stock_pour_calcul_interne = y_stock # Default pour "RAO"
                    if macro_c14_stock_source_pour_calcul.upper() == "REEL":
                        stock_pour_calcul_interne = aj_stock_reel
                    
                    if cj_algo_meti == 1:
                        valeur_interne_pour_arrondi = ce_cmd_opt_arrondi_mini - aa_qte_prev_init1 - arrondi_sup_w_div_4 + stock_pour_calcul_interne + z_ral
                    else: # cj_algo_meti == 2 ou autre (ex: 0 si erreur de calcul pour CJ)
                        valeur_interne_pour_arrondi = ce_cmd_opt_arrondi_mini - ab_qte_prev_init2 - arrondi_sup_w_div_4
                
                # Le SIERREUR(interne_calcul;1) dans Excel
                # Ici, on suppose que si le calcul interne n'a pas levé d'exception Python, il est valide.
                # Si cd_cmd_opt_sans_arrondi était 0, valeur_interne_pour_arrondi est 0.
                # Si une erreur était attendue pour devenir 1, il faudrait la simuler.
                # Pour l'instant, on se base sur le calcul.
                terme_y_arrondi = np.round(valeur_interne_pour_arrondi, 0)
                
                # Implémentation de MAX(X; Y*C21; MAX(X;Y))
                terme_y_fois_c21 = terme_y_arrondi * macro_c21_coeff_exception
                max_x_y = max(terme_x_base, terme_y_arrondi)
                resultat_ck = max(terme_x_base, terme_y_fois_c21, max_x_y)
                
                if print_debug_this_row: 
                    print(f"        Terme X: {terme_x_base:.2f}")
                    print(f"        Valeur Interne (avant arrondi): {valeur_interne_pour_arrondi:.2f}, Terme Y (Arrondi): {terme_y_arrondi:.2f}")
                    print(f"        Calcul MAX(X={terme_x_base:.2f}; Y*C21={terme_y_fois_c21:.2f}; MAX(X,Y)={max_x_y:.2f})")
                    print(f"        ==> Résultat CK: {resultat_ck:.2f}")
                return resultat_ck
            else: 
                if print_debug_this_row: print(f"      CK Cond 3 (Globale pour MAX_Calcul) est FAUSSE. Résultat -> 0")
                return 0.0
        
        except Exception as e_ck_row_exception:
            if print_debug_this_row: print(f"    ERREUR Exception dans calculate_ck_row pour {code_meti_debug}: {e_ck_row_exception}")
            return 0.0 # SIERREUR global

    df['SM A Charger Test'] = df.apply(calculate_ck_row, axis=1)
    return df




def get_processed_data(df):
    """
    Fonction principale pour calculer les quatre nouvelles colonnes.
    Doit être appelée après que toutes les colonnes dépendantes (CE, CD, BY, CF, etc.) sont calculées.
    """
    print("CalculAutresColonnes: Démarrage du calcul des colonnes supplémentaires...")
    
    # Charger les données QteMaxi une seule fois
    qtemaxi_map_e, qtemaxi_map_i = load_qtemaxi_data()

    df = calculate_quantite_maxi_ch(df, qtemaxi_map_e, qtemaxi_map_i)
    df = calculate_quantite_maxi_a_charger_ci(df) # Dépend de CH et CE
    df = calculate_algo_meti_cj(df)                # Dépend de AE, AF
    df = calculate_sm_a_charger_test_ck(df)        # Dépend de nombreuses colonnes, y compris CI, CJ

    print("CalculAutresColonnes: Calculs terminés.")
    return df

# --- DANS CalculAutresColonnes.py, à la fin ---
if __name__ == "__main__":
    print("Test du module CalculAutresColonnes")
    current_dir = os.path.dirname(__file__)
    
    csv_dir_test = os.path.join(current_dir, "CSV")
    if not os.path.exists(csv_dir_test): 
        os.makedirs(csv_dir_test)
        print(f"Dossier de test créé: {csv_dir_test}")

    # Nom du fichier de test, distinct de votre fichier réel si possible, ou vérifier avant d'écrire
    qtemaxi_test_path = os.path.join(csv_dir_test, "Qtite_Maxi_TEST_DATA.csv") # Nom différent pour le test
    # OU si vous voulez utiliser le même nom mais seulement s'il n'existe pas :
    # qtemaxi_real_path = os.path.join(csv_dir_test, "Qtite_Maxi.csv")

    # Seulement créer le fichier de test s'il n'existe pas déjà, ou utiliser le nom de test spécifique
    # if not os.path.exists(qtemaxi_real_path): # Si vous voulez créer le fichier réel s'il manque
    if True: # Pour toujours créer le fichier de test Qtite_Maxi_TEST_DATA.csv pour cet exemple
        qtemaxi_test_content_for_script = (
            "entrepot;EAN13;ifls;libelle article;Nombre Maxi Unit�s De Commande;pcb;ID;ID2;QteMaxUS\n"
            "ENT1;EAN1;IFLS1;PROD A;10;1;ENT1-EAN1;ENT1-IFLS1;100\n"
            "ENT2;EAN2;IFLS2;PROD B;20;1;ENT2-EAN2;ENT2-IFLS2;50\n"
        )
        with open(qtemaxi_test_path, 'w', encoding='utf-8') as f: # Écrit dans le fichier de TEST
            f.write(qtemaxi_test_content_for_script)
        print(f"Fichier de test '{os.path.basename(qtemaxi_test_path)}' créé/mis à jour dans {csv_dir_test}.")
        # Maintenant, la fonction load_qtemaxi_data devra être modifiée pour utiliser ce fichier de test
        # si __name__ == "__main__", ou pointer vers le vrai fichier autrement. C'est un peu compliqué.

    # --- Alternative plus simple pour le test ---
    # Au lieu de lire un fichier dans le test, on peut créer le dictionnaire de map directement.
    # Cela évite tout risque avec les fichiers de données réelles.

    # Simuler les dictionnaires qtemaxi qui seraient chargés par load_qtemaxi_data()
    # pour le test autonome, sans lire de fichier externe pour le test.
    test_qtemaxi_map_e = {
        "ENT1-IFLS1": 10.0,
        "ENT2-IFLS2": 20.0
    }
    test_qtemaxi_map_i = {
        "ENT1-IFLS1": 100.0,
        "ENT2-IFLS2": 50.0
    }
    print("  CalculAutresColonnes (Test): Utilisation de dictionnaires QteMaxi simulés pour le test.")
    # Créer un DataFrame de test
    test_data = {
        # ... (vos données de test pour df_test) ...
         'CODE_METI': ['M1', 'M2', 'M3', 'M4'],
        'FOURNISSEUR': ['ENT1', 'ENT2', 'ENT1', 'ENT_NON_TROUVE'],
        'IFLS': ['IFLS1', 'IFLS2', 'IFLS_NON_TROUVE', 'IFLSX'],
        'PCB': [1, 1, 1, 1],
        'Commande optimisée avec arrondi et mini': [150, 40, 200, 30], # CE
        'QTÉPRÉPOSÉE1': [10, 30, 0, 0], # AE
        'QTÉPRÉPOSÉE2': [20, 20, 0, 0], # AF fixé - était QTÉPROPOSÉE2
        'Exclusion Lissage': ["Non Concerné SM à charger", "Autre", "Non Concerné SM à charger", "Autre"], 
        'STOCK': [0, 10, 0, 0], 'RAL': [0, 5, 0, 0], 'COCHE_RAO': [54, 0, 0, 0], 
        'Facteur multiplicatif Lissage besoin brut': [1.1, 1, 0, 0], 
        'Commande optimisée avec arrondi et mini et TS': [160, 40, 0, 0], 
        'Commande optimisée sans arrondi': [140, 38, 0, 0], 
        'QTÉPRÉVISIONINIT1': [100, 20, 0, 0], 'QTÉPRÉVISIONINIT2': [50, 15, 0, 0], 
        'STOCK_REEL': [5, 8, 0, 0], 
    }
    df_test = pd.DataFrame(test_data)
    
    # ---- MODIFICATION POUR LE TEST: Passer les maps simulées ----
    # Au lieu d'appeler get_processed_data(df_test.copy()) qui appellerait load_qtemaxi_data()
    # on appelle les fonctions de calcul individuellement en passant les maps de test.
    
    df_result_test = df_test.copy()
    df_result_test = calculate_quantite_maxi_ch(df_result_test, test_qtemaxi_map_e, test_qtemaxi_map_i)
    df_result_test = calculate_quantite_maxi_a_charger_ci(df_result_test)
    df_result_test = calculate_algo_meti_cj(df_result_test)
    df_result_test = calculate_sm_a_charger_test_ck(df_result_test)


    print("\n--- Résultats du test CalculAutresColonnes ---")
    # ... (le reste de vos affichages et assertions) ...
    cols_to_show = ['CODE_METI', 'FOURNISSEUR', 'IFLS', 'Quantité Maxi ?', 
                    'Commande optimisée avec arrondi et mini', 'Quantité Maxi à charger ?',
                    'QTÉPRÉPOSÉE1', 'QTÉPROPOSÉE2', 'Algo Meti',
                    'SM A Charger Test']
    print(df_result_test[[col for col in cols_to_show if col in df_result_test.columns]].to_string())

    # Assertions
    m1_row = df_result_test[df_result_test['CODE_METI'] == 'M1'].iloc[0]
    assert abs(m1_row['Quantité Maxi ?'] - 10) < 0.01, f"CH M1: {m1_row['Quantité Maxi ?']}"
    # ... (autres assertions) ...
    print("Tests basiques terminés.")