import pandas as pd
import numpy as np
from MacroParam import get_param_value
import datetime

def get_processed_data(df):
    """
    Calcule la colonne "Nb Jours Commande Max" selon la formule:
    =SI(Q2=312;INDEX('Macro-Param'!$B$45:$G$45;1;EQUIV(G2;'Macro-Param'!$B$34:$G$34;0));
      SI(NON(ESTNA(EQUIV(I2;'Macro-Param'!$T$13:$T$17;0)));
        SIERREUR(INDEX('Macro-Param'!$U:$Z;EQUIV(L2;'Macro-Param'!$T:$T;1);EQUIV(H2;'Macro-Param'!$U$4:$Z$4;0));
        'Macro-Param'!$U$5);12))
    
    Interprétation:
    - Si la zone de stockage (Q2) est 312, utiliser la valeur correspondant au type de produit (G2) dans la plage B45:G45
    - Sinon, si la famille de produit (I2) est dans la plage T13:T17, chercher la valeur correspondante 
      en fonction de la famille bis (L2) et du jour de la semaine (H2)
    - Si erreur, utiliser la valeur par défaut (U5 = 2.2)
    - Sinon, utiliser 12
    
    Args:
        df: DataFrame contenant les données nécessaires
    
    Returns:
        DataFrame avec la colonne "Nb Jours Commande Max" ajoutée
    """
    print("Calcul de la colonne 'Nb Jours Commande Max'...")
    # Vérifier que les colonnes nécessaires existent
    required_columns = ['CODE_METI']  # Liste minimale des colonnes requises
    recommended_columns = ['Période de vente finale (Jours)', 'Nb de commande / Semaine Final']  # Colonnes recommandées
    
    # Nous avons besoin d'au moins CODE_METI, mais les autres colonnes sont optionnelles
    missing_columns = [col for col in required_columns if col not in df.columns]
    missing_recommended = [col for col in recommended_columns if col not in df.columns]
    
    if missing_recommended:
        print(f"Attention: Colonnes recommandées manquantes pour la nouvelle méthode de calcul: {missing_recommended}")
    
    if missing_columns:
        print(f"Erreur: Colonnes manquantes pour le calcul de 'Nb Jours Commande Max': {missing_columns}")
        df['Nb Jours Commande Max'] = get_param_value("nb_jours_commande_max_default", 12)
        return df
    # Fonction pour calculer "Nb Jours Commande Max" pour chaque ligne
    def calc_nb_jours_commande_max(row):
        # Récupérer les paramètres nécessaires
        default_value = get_param_value("nb_jours_commande_max_default", 12)
        zone_values = get_param_value("nb_jours_commande_max_par_zone", {})
        familles_speciales = get_param_value("nb_jours_commande_max_familles", {})
        valeurs_par_jour = get_param_value("nb_jours_commande_max_par_jour", {})
        valeur_defaut_famille_special = get_param_value("nb_jours_commande_max_valeur_defaut_famille_special", 2.2)
        table_referentiel = get_param_value("nb_jours_commande_max_table", {})
        # Les messages de debug pour le produit 29440 ont été supprimés
        
        # Récupérer les valeurs des colonnes (si elles existent)
        zone_stockage = row.get('ZONE_STOCKAGE', None)
        type_produit = row.get('TYPE_PRODUIT', None)
        famille_produit = row.get('FAMILLE_PRODUIT', None)
        famille_produit_bis = row.get('FAMILLE_PRODUIT_BIS', None)
        
        # Déterminer le jour de la semaine (1-7)
        jour_semaine = datetime.datetime.now().isoweekday()
        if 'JOUR_SEMAINE' in row:
            try:
                jour_semaine = int(row['JOUR_SEMAINE'])
            except (ValueError, TypeError):
                pass
        
        # Le code pour les produits spéciaux a été retiré - tous les produits utilisent maintenant la table de référence
        code_meti = str(row.get('CODE_METI')).strip()
            
        # Nouvelle méthode: utiliser la table de référence basée sur Période de vente et Nb commandes par semaine
        periode_vente = row.get('Période de vente finale (Jours)')
        nb_commandes_semaine = row.get('Nb de commande / Semaine Final')
        
        # Vérifier si on doit traiter ce produit avec la table de référence ou utiliser la valeur par défaut
        # on utilise la valeur par défaut (12 jours)
        if (periode_vente is None or nb_commandes_semaine is None):
            return default_value
        if periode_vente is not None and nb_commandes_semaine is not None:
            # Utilisation de la table de référence
            try:
                # Vérifier si la période de vente est supérieure à la valeur maximale dans la table
                # Si c'est le cas, on utilise la valeur par défaut (12) comme dans la formule Excel
                periodes_disponibles = sorted(list(table_referentiel.keys()))
                periode_vente_float = float(periode_vente)
                max_periode_table = max(periodes_disponibles) if periodes_disponibles else 0
                
                # Si la période de vente dépasse la valeur max de la table, utiliser la valeur par défaut
                if periode_vente_float > max_periode_table:
                    return default_value  # 12 jours par défaut
                
                # Sinon, trouver la période de vente la plus proche (inférieure ou égale)
                periode_reference = None
                for p in periodes_disponibles:
                    if p <= periode_vente_float:
                        periode_reference = p
                    if p > periode_vente_float:
                        break
                
                if periode_reference is not None:
                    # Arrondir le nombre de commandes par semaine au nombre entier le plus proche
                    nb_commandes = int(round(float(nb_commandes_semaine)))
                    # Limiter aux valeurs entre 1 et 6
                    nb_commandes = max(1, min(6, nb_commandes))
                    
                    # Récupérer la valeur dans la table
                    if nb_commandes in table_referentiel[periode_reference]:
                        valeur_table = table_referentiel[periode_reference][nb_commandes]
                        # Message de debug pour le produit 29440 supprimé
                        return valeur_table
            except Exception as e:
                print(f"Erreur lors de l'utilisation de la table de référence: {e}")
                # Continuer avec les autres méthodes
        
        # 1. Si la zone de stockage est 312, utiliser la valeur correspondant au type de produit
        if zone_stockage == 312:
            return zone_values.get(str(zone_stockage), default_value)
        
        # 2. Si la famille de produit est dans la liste des familles spéciales
        if famille_produit in familles_speciales:
            try:
                # Chercher la valeur correspondante en fonction de la famille bis et du jour de la semaine
                # Si une erreur se produit, utiliser la valeur par défaut U5
                if famille_produit_bis and jour_semaine in valeurs_par_jour:
                    return valeurs_par_jour.get(jour_semaine, valeur_defaut_famille_special)
                else:
                    return valeur_defaut_famille_special
            except Exception:
                return valeur_defaut_famille_special
        
        # 3. Valeur par défaut
        return default_value
    
    # Appliquer la fonction à chaque ligne du DataFrame
    df['Nb Jours Commande Max'] = df.apply(calc_nb_jours_commande_max, axis=1)
    
    # Calcul de la colonne "Commande Max V0" selon la formule:
    # =MAX(BZ250*M250;W250;3;SI(ET(Y250=0;Z250=0);O250;0))
    # Où:
    # - BZ250 = "Nb Jours Commande Max"
    # - M250 = "VMJ Utilisée Stock Sécurité"
    # - W250 = "PCB"
    # - Y250 = "STOCK_MARCHAND" (colonne X)
    # - Z250 = "STOCK" (colonne Y)
    # - O250 = "SM Final"
    def calc_commande_max_v0(row):
        nb_jours_max = row.get('Nb Jours Commande Max', 0)
        vmj_securite = row.get('VMJ Utilisée Stock Sécurité', 0)
        pcb = row.get('PCB', 0)
        stock_marchand = row.get('STOCK_MARCHAND', 0)
        stock = row.get('STOCK', 0)
        sm_final = row.get('SM Final', 0)
        
        # Traiter les valeurs NaN/None
        nb_jours_max = 0 if pd.isna(nb_jours_max) else float(nb_jours_max)
        vmj_securite = 0 if pd.isna(vmj_securite) else float(vmj_securite)
        pcb = 0 if pd.isna(pcb) else float(pcb)
        stock_marchand = 0 if pd.isna(stock_marchand) else float(stock_marchand)
        stock = 0 if pd.isna(stock) else float(stock)
        sm_final = 0 if pd.isna(sm_final) else float(sm_final)
        
        # SI(ET(Y250=0;Z250=0);O250;0)
        condition_value = sm_final if (stock_marchand == 0 and stock == 0) else 0
        
        # MAX(BZ250*M250;W250;3;SI(ET(Y250=0;Z250=0);O250;0))
        return max(nb_jours_max * vmj_securite, pcb, 3, condition_value)
    
    # Vérifier que les colonnes nécessaires existent pour éviter les erreurs
    required_cols_max_v0 = ['VMJ Utilisée Stock Sécurité', 'PCB', 'STOCK_MARCHAND', 'STOCK', 'SM Final']
    missing_cols = [col for col in required_cols_max_v0 if col not in df.columns]
    if missing_cols:
        print(f"Attention: Colonnes manquantes pour le calcul de 'Commande Max V0': {missing_cols}")
        print("La colonne 'Commande Max V0' sera calculée avec les données disponibles.")
    
    df['Commande Max V0'] = df.apply(calc_commande_max_v0, axis=1)
    
    # Calcul de la colonne "Commande Max via Stock Max" selon la formule:
    # =SIERREUR(SI(N2="";"";MAX(0;N2-Y2-Z2+BF2-BG2));"")
    # Où:
    # - N2 = "Stock Max" (N)
    # - Y2 = "STOCK" (Y)
    # - Z2 = "RAL" (Z)  <- Correction: Z2 is RAL, not STOCK_MARCHAND
    # - BF2 = "Prév C1-L2 Finale" (BF)
    # - BG2 = "Prév L1-L2 Finale" (BG)
    
    def calc_commande_max_via_stock_max(row):
        """
        Calcule la colonne "Commande Max via Stock Max" selon la formule:
        =SIERREUR(SI(N2="";"";MAX(0;N2-Y2-Z2+BF2-BG2));"")
        
        Où:
        - N2 = "Stock Max"
        - Y2 = "STOCK"
        - Z2 = "RAL"
        - BF2 = "Prév C1-L2 Finale"
        - BG2 = "Prév L1-L2 Finale"
        """        # Récupérer les valeurs
        # Vérifier d'abord si la colonne 'Stock Max' existe dans le DataFrame
        if 'Stock Max' not in row:
            return ""
        
        stock_max = row.get('Stock Max', None)
        stock = row.get('STOCK', None)
        ral = row.get('RAL', None)
        prev_c1_l2_finale = row.get('Prév C1-L2 Finale', None)
        prev_l1_l2_finale = row.get('Prév L1-L2 Finale', None)
        code_meti = str(row.get('CODE_METI', '')).strip()
        
        # Debug flag pour le CODE_METI 1880332
        debug_flag = code_meti == '1880332'
        
        # SI(N2="") - Si Stock Max est vide, None, ou NaN, renvoyer une chaîne vide
        if stock_max == "" or pd.isna(stock_max)or stock_max == 0.0 or stock_max is None:
            if debug_flag:
                print(f"DEBUG 1880332: Stock Max is empty or None ({stock_max})")
            return ""
        
        # SIERREUR - Gérer les erreurs potentielles dans le calcul
        try:
            # Convertir les valeurs en nombres (avec des valeurs par défaut si NaN)
            stock = 0 if pd.isna(stock) else float(stock)
            ral = 0 if pd.isna(ral) else float(ral)
            prev_c1_l2_finale = 0 if pd.isna(prev_c1_l2_finale) else float(prev_c1_l2_finale)
            prev_l1_l2_finale = 0 if pd.isna(prev_l1_l2_finale) else float(prev_l1_l2_finale)
            
            # MAX(0;N2-Y2-Z2+BF2-BG2)
            result = max(0, stock_max - stock - ral + prev_c1_l2_finale - prev_l1_l2_finale)
            
            # Debug pour CODE_METI 1880332
            if debug_flag:
                print(f"DEBUG for CODE_METI 1880332:")
                print(f"  Stock Max: {stock_max}")
                print(f"  STOCK: {stock}")
                print(f"  RAL: {ral}")
                print(f"  Prév C1-L2 Finale: {prev_c1_l2_finale}")
                print(f"  Prév L1-L2 Finale: {prev_l1_l2_finale}")
                print(f"  Calculation: max(0, {stock_max} - {stock} - {ral} + {prev_c1_l2_finale} - {prev_l1_l2_finale}) = {result}")
            
            return result
            
        except Exception as e:
            # Si une erreur se produit (SIERREUR), retourner une chaîne vide
            if debug_flag:
                print(f"DEBUG 1880332: Error in calculation: {e}")
            return ""
      # Vérifier que les colonnes nécessaires existent pour éviter les erreurs
    required_cols_via_stock_max = ['Stock Max', 'STOCK', 'RAL', 'Prév C1-L2 Finale', 'Prév L1-L2 Finale']
    missing_cols = [col for col in required_cols_via_stock_max if col not in df.columns]
    
    if missing_cols:
        print(f"Attention: Colonnes manquantes pour le calcul de 'Commande Max via Stock Max': {missing_cols}")
        print("La colonne 'Commande Max via Stock Max' sera calculée avec des valeurs vides si 'Stock Max' est manquant.")
        
        # Si la colonne 'Stock Max' est manquante, on remplit toute la colonne avec des chaînes vides
        if 'Stock Max' not in df.columns:
            df['Commande Max via Stock Max'] = ""
        else:
            df['Commande Max via Stock Max'] = df.apply(calc_commande_max_via_stock_max, axis=1)
    else:
        df['Commande Max via Stock Max'] = df.apply(calc_commande_max_via_stock_max, axis=1)
      # Calcul de la colonne "Commande Max avec stock max" selon la formule:
    # =SI(OU(CB2="";'Macro-Param'!$C$14="Non");CA2;MIN(CA2;CB2))
    # Où:
    # - CB2 = "Commande Max via Stock Max" (CB)
    # - CA2 = "Commande Max V0" (CA)
    # - 'Macro-Param'!$C$14 = "Activation stock max"
    def calc_commande_max_avec_stock_max(row):
        # Vérifier si les colonnes nécessaires existent
        if 'Commande Max V0' not in row:
            return 0
            
        commande_max_v0 = row.get('Commande Max V0', 0)
        commande_max_via_stock_max = row.get('Commande Max via Stock Max', "")
        activation_stock_max = get_param_value("Activation stock max", "Oui")
        
        # Traiter les valeurs NaN/None
        commande_max_v0 = 0 if pd.isna(commande_max_v0) else float(commande_max_v0)
        
        # SI(OU(CB2="";'Macro-Param'!$C$14="Non");CA2;MIN(CA2;CB2))
        if commande_max_via_stock_max == "" or pd.isna(commande_max_via_stock_max) or activation_stock_max == "Non":
            return commande_max_v0
        else:
            try:
                commande_max_via_stock_max = float(commande_max_via_stock_max)
                return min(commande_max_v0, commande_max_via_stock_max)
            except (ValueError, TypeError):
                # Si la conversion échoue, retourner Commande Max V0
                return commande_max_v0
    
    # Vérifier que les colonnes nécessaires existent
    required_cols_avec_stock_max = ['Commande Max V0', 'Commande Max via Stock Max']
    missing_cols = [col for col in required_cols_avec_stock_max if col not in df.columns]
    
    if missing_cols:
        print(f"Attention: Colonnes manquantes pour le calcul de 'Commande Max avec stock max': {missing_cols}")
        if 'Commande Max V0' in df.columns:
            print("La colonne 'Commande Max avec stock max' sera égale à 'Commande Max V0'.")
            df['Commande Max avec stock max'] = df['Commande Max V0']
        else:
            print("La colonne 'Commande Max avec stock max' sera remplie avec des zéros.")
            df['Commande Max avec stock max'] = 0
    else:
        df['Commande Max avec stock max'] = df.apply(calc_commande_max_avec_stock_max, axis=1)
    
    return df


if __name__ == "__main__":
    # Test du module    
    test_df = pd.DataFrame({
        'CODE_METI': ['123', '456', '789', '101', '202', '303', '404'],
        'ZONE_STOCKAGE': [312, 100, None, None, None, None, None],
        'TYPE_PRODUIT': ['Sec', 'Frais', 'Surgelé', 'Sec', 'Frais', 'Surgelé', 'Sec'],
        'FAMILLE_PRODUIT': ['Standard', '5 - FL', 'Standard', 'Standard', 'Standard', 'Standard', 'Standard'],
        'FAMILLE_PRODUIT_BIS': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'JOUR_SEMAINE': [1, 2, 3, 4, 5, 6, 7],
        'Période de vente finale (Jours)': [None, None, None, 3, 5, 12, 20],
        'Nb de commande / Semaine Final': [None, None, None, 6, 3, 1, 2],
        # Colonnes additionnelles pour le calcul de "Commande Max V0"
        'VMJ Utilisée Stock Sécurité': [10, 8, 6, 5, 4, 3, 2],
        'PCB': [5, 4, 3, 6, 12, 24, 6],
        'STOCK_MARCHAND': [100, 50, 0, 20, 10, 0, 5],
        'STOCK': [34, 60, 0, 25, 15, 0, 8],
        'SM Final': [150, 80, 50, 30, 20, 40, 10],
        # Colonnes additionnelles pour le calcul de "Commande Max via Stock Max"
        'Stock Max': [115.28, 100, "", 50, 30, 60, 20],
        'RAL': [90, 5, 0, 8, 3, 2, 1],  # Ajout de la colonne RAL pour le calcul correct
        'Prév C1-L2 Finale': [14, 5, 0, 2, 3, 5, 1],
        'Prév L1-L2 Finale': [0, 3, 0, 1, 2, 3, 1]
    
    })
    
    result_df = get_processed_data(test_df)
    
    print("\nRésultats du calcul Nb Jours Commande Max:")
    print(result_df[['CODE_METI', 'ZONE_STOCKAGE', 'FAMILLE_PRODUIT', 
                     'Période de vente finale (Jours)', 'Nb de commande / Semaine Final', 
                     'Nb Jours Commande Max', 'Commande Max V0', 
                     'Commande Max via Stock Max', 'Commande Max avec stock max']])
    
    print("\nTest de la table de référence:")
    print("Période de vente = 3, Nb commandes = 6 → Valeur attendue = 2.2")
    print("Période de vente = 5, Nb commandes = 3 → Valeur attendue = 2.6")
    print("Période de vente = 12, Nb commandes = 1 → Valeur attendue = 3.5")
    print("Période de vente = 20, Nb commandes = 2 → Valeur attendue = 6.0")
