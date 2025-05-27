import pandas as pd
import os
import MacroParam # Assuming MacroParam.py is in the same directory 'Outil_de_lissage'
from datetime import datetime # timedelta might not be needed if direct comparison works
import numpy as np # Added for np.number

# Définition du chemin vers le fichier PDC.xlsx
# Assurez-vous que ce chemin est correct et que le fichier existe à cet emplacement.
# __file__ est le chemin du script PDC.py
# os.path.dirname(__file__) est le répertoire contenant PDC.py (Outil_de_lissage)
# os.path.join(os.path.dirname(__file__), 'PDC', 'PDC.xlsx') construit le chemin complet
PDC_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PDC', 'PDC.xlsx')

def load_pdc_perm_data():
    """
    Charge les données depuis la feuille 'PDC' du fichier PDC.xlsx.
    Assure que la colonne 'Jour' est correctement formatée en datetime.
    """
    try:
        # User confirmed sheet_name='PDC' for the Excel file.
        df = pd.read_excel(PDC_FILE_PATH, sheet_name='PDC')
        
        if 'Jour' not in df.columns:
            print("Erreur: La colonne 'Jour' est introuvable dans la feuille 'PDC'. Le chargement a échoué.")
            return pd.DataFrame()
        
        # Convertir la colonne 'Jour' en datetime
        df['Jour'] = pd.to_datetime(df['Jour'], errors='coerce')
        # Supprimer les lignes où 'Jour' n'a pas pu être converti en date valide
        df.dropna(subset=['Jour'], inplace=True)

        if df.empty:
            print("Avertissement: Aucune donnée valide après conversion de la colonne 'Jour' et suppression des NaNs.")
            return pd.DataFrame()
            
        print(f"Données chargées avec succès depuis la feuille 'PDC' de {PDC_FILE_PATH}")
        return df
    except FileNotFoundError:
        print(f"Erreur: Le fichier {PDC_FILE_PATH} n'a pas été trouvé.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Erreur lors du chargement des données depuis {PDC_FILE_PATH}, feuille 'PDC': {e}")
        return pd.DataFrame()

def create_pdc_perm_summary(df_pdc_perm_input):
    """
    Crée un résumé basé sur les données de la feuille 'PDC Perm' (chargées comme df_pdc_perm_input)
    et la formule Excel fournie. Le résumé commence à partir de DATE_COMMANDE.
    Formule Excel (conceptuellement, AUJOURDHUI() est remplacé par DATE_COMMANDE):
    =SI(L12="";"";
        SI(L$12="Total";SOMME(DECALER($K13;;1;;COLONNE(L12)-COLONNE($K13)-1));
            SIERREUR(INDEX('PDC Perm'!$A:$S;EQUIV($K13;'PDC Perm'!$A:$A;0);EQUIV(L$12;'PDC Perm'!$1:$1;0));0) /
            (1000*'Macro-Param'!$C$5)
        ) *
        SIERREUR(SI($K13>=DATE_COMMANDE+2+SI('Macro-Param'!$C$10="J-1";1;0);
                    INDEX('Macro-Param'!$L:$L;EQUIV(L$12;'Macro-Param'!$F:$F;0));1);1)
    )
    $K13: Date (index of the summary table)
    L$12: Product Type or "Total" (column header of the summary table)
    'PDC Perm'!$A:$A : Date column in 'PDC Perm' sheet (here, 'Jour' column)
    'PDC Perm'!$1:$1 : Header row in 'PDC Perm' sheet (product type column names)
    'Macro-Param'!$C$5: taux_service_amont_estime
    'Macro-Param'!$C$10: jour_passage_commande
    'Macro-Param'!$L:$L: An empty column, leading to the multiplier part evaluating to 1.
    DATE_COMMANDE: Date de début pour le traitement, tirée de MacroParam.DATE_COMMANDE.
    """
    if df_pdc_perm_input.empty:
        print("Les données PDC Perm en entrée sont vides. Impossible de créer le résumé.")
        return pd.DataFrame()

    # Utiliser une copie pour éviter de modifier le DataFrame original
    df_pdc_perm = df_pdc_perm_input.copy()

    if 'Jour' not in df_pdc_perm.columns or df_pdc_perm['Jour'].isnull().all():
        print("La colonne 'Jour' est manquante, invalide, ou entièrement nulle dans les données PDC Perm. Impossible de créer le résumé.")
        return pd.DataFrame()
    
    # Définir 'Jour' comme index pour faciliter les recherches
    # Assurez-vous que la colonne 'Jour' est déjà au format datetime grâce à load_pdc_perm_data
    df_pdc_perm = df_pdc_perm.set_index('Jour')
    # S'assurer que l'index (dates) est unique, en gardant la première occurrence en cas de doublons
    df_pdc_perm = df_pdc_perm[~df_pdc_perm.index.duplicated(keep='first')]

    # Charger DATE_COMMANDE et la convertir en datetime
    date_commande_str = MacroParam.DATE_COMMANDE # Accès direct à la variable globale
    try:
        date_commande_dt = pd.to_datetime(date_commande_str, format='%d/%m/%Y')
    except ValueError:
        print(f"Erreur: DATE_COMMANDE '{date_commande_str}' dans MacroParam.py n'est pas dans le format attendu 'dd/mm/yyyy'.")
        return pd.DataFrame()

    # Filtrer les données pour commencer à partir de DATE_COMMANDE (inclus)
    df_pdc_perm_filtered = df_pdc_perm[df_pdc_perm.index >= date_commande_dt].copy() # Utiliser .copy() pour éviter SettingWithCopyWarning

    if df_pdc_perm_filtered.empty:
        print(f"Aucune donnée PDC Perm trouvée à partir de DATE_COMMANDE ({date_commande_str}). Le résumé sera vide.")
        return pd.DataFrame()

    # Charger les paramètres depuis le module MacroParam
    taux_service_amont_estime = MacroParam.get_param_value("taux_service_amont_estime", 0.92) 
    # jour_passage_commande = MacroParam.get_param_value("jour_passage_commande", "J") # Non utilisé car le multiplicateur est 1
    
    # Le multiplicateur complexe de la formule Excel se simplifie à 1.0 car 'Macro-Param'!$L:$L est vide.
    # La condition $K13>=DATE_COMMANDE+2+SI(...) faisait partie de ce multiplicateur.
    # Puisque le multiplicateur est 1, cette condition n'affecte plus directement le calcul principal ici.
    multiplier = 1.0

    # Définir les colonnes produit pour le tableau résumé, basées sur la structure Excel observée.
    summary_product_columns = ['Sec Hétérogène', 'Sec Homogène', 'Sec Méca', 'Surgelés', 'Frais Méca', 'Frais Manuel']
    
    # Filtrer pour utiliser uniquement les colonnes produit qui existent réellement dans df_pdc_perm_filtered chargé
    valid_product_columns = [col for col in summary_product_columns if col in df_pdc_perm_filtered.columns]
    
    if not valid_product_columns:
        expected_cols_str = ", ".join(summary_product_columns)
        available_cols_str = ", ".join(df_pdc_perm_filtered.columns)
        print(f"Aucune des colonnes produit attendues ({expected_cols_str}) n'a été trouvée parmi les colonnes disponibles ({available_cols_str}) dans les données PDC Perm filtrées. Impossible de créer le résumé.")
        return pd.DataFrame()
    
    if len(valid_product_columns) < len(summary_product_columns):
        missing_cols = [col for col in summary_product_columns if col not in valid_product_columns]
        print(f"Avertissement: Certaines colonnes produit définies ({missing_cols}) sont absentes des données PDC Perm filtrées. Le résumé utilisera: {valid_product_columns}")

    output_columns = valid_product_columns + ['Total']
    # L'index du summary_table est maintenant basé sur les dates filtrées
    summary_table = pd.DataFrame(index=df_pdc_perm_filtered.index, columns=output_columns, dtype=float)

    for date_k in summary_table.index: # date_k correspond à $K13 (maintenant filtré par DATE_COMMANDE)
        for col_l in output_columns: # col_l correspond à L$12
            if col_l == "Total":
                summary_table.loc[date_k, "Total"] = summary_table.loc[date_k, valid_product_columns].sum()
            else:
                raw_pdc_value = df_pdc_perm_filtered.loc[date_k, col_l]
                raw_pdc_value = 0.0 if pd.isna(raw_pdc_value) else float(raw_pdc_value)

                if taux_service_amont_estime is not None and taux_service_amont_estime != 0:
                    calculated_value = raw_pdc_value / (1000 * taux_service_amont_estime)
                else:
                    calculated_value = 0.0 
                    if taux_service_amont_estime == 0:
                         print(f"Avertissement: taux_service_amont_estime est zéro pour la date {date_k}, colonne {col_l}. Résultat mis à 0.")

                summary_table.loc[date_k, col_l] = calculated_value * multiplier # multiplier est 1.0
    
    return summary_table

def format_pdc_perm_summary(df_summary):
    """
    Formate le résumé PDC Perm pour l'affichage ou la sortie.
    Arrondit toutes les valeurs numériques à 2 décimales.
    """
    print("Fonction format_pdc_perm_summary appelée.")
    if df_summary.empty:
        print("Le DataFrame de résumé est vide, aucun formatage appliqué.")
        return df_summary
    
    # Arrondir toutes les colonnes numériques à 2 décimales
    # S'assure que seules les colonnes numériques sont arrondies pour éviter les erreurs sur les colonnes non numériques (si elles existaient)
    numeric_cols = df_summary.select_dtypes(include=np.number).columns
    df_summary[numeric_cols] = df_summary[numeric_cols].round(2)
    
    print("Les valeurs numériques du résumé PDC Perm ont été arrondies à 2 décimales.")
    return df_summary

def get_processed_pdc_perm_data():
    """
    Fonction principale pour obtenir les données PDC Perm traitées et formatées.
    """
    df_pdc = load_pdc_perm_data()
    if not df_pdc.empty:
        df_summary = create_pdc_perm_summary(df_pdc)
        if not df_summary.empty:
            df_formatted = format_pdc_perm_summary(df_summary)
            return df_formatted
        else:
            print("La création du résumé PDC Perm a résulté en un DataFrame vide.")
            return pd.DataFrame()
    else:
        print("Le chargement des données PDC Perm a échoué ou résulté en un DataFrame vide.")
        return pd.DataFrame()
    
def create_pdc_perm_summary_BRUT(df_pdc_perm_input): # Nouvelle fonction ou version modifiée
    """
    Crée un résumé des données PDC Perm BRUTES (sans division par 1000*TSA).
    L'index est la date, les colonnes sont les types de produits.
    """
    if df_pdc_perm_input.empty:
        print("PDC.py - create_pdc_perm_summary_BRUT: Données PDC Perm en entrée sont vides.")
        return pd.DataFrame()

    df_pdc_perm = df_pdc_perm_input.copy()
    if 'Jour' not in df_pdc_perm.columns or df_pdc_perm['Jour'].isnull().all():
        print("PDC.py - create_pdc_perm_summary_BRUT: Colonne 'Jour' manquante ou invalide.")
        return pd.DataFrame()
    
    df_pdc_perm = df_pdc_perm.set_index('Jour')
    df_pdc_perm = df_pdc_perm[~df_pdc_perm.index.duplicated(keep='first')]

    date_commande_dt = pd.to_datetime(MacroParam.DATE_COMMANDE, format='%d/%m/%Y')
    df_pdc_perm_filtered = df_pdc_perm[df_pdc_perm.index >= date_commande_dt].copy()

    if df_pdc_perm_filtered.empty:
        print(f"PDC.py - create_pdc_perm_summary_BRUT: Aucune donnée PDC Perm trouvée à partir de DATE_COMMANDE.")
        return pd.DataFrame()

    summary_product_columns = ['Sec Hétérogène', 'Sec Homogène', 'Sec Méca', 'Surgelés', 'Frais Méca', 'Frais Manuel']
    valid_product_columns = [col for col in summary_product_columns if col in df_pdc_perm_filtered.columns]
    
    if not valid_product_columns:
        print("PDC.py - create_pdc_perm_summary_BRUT: Aucune colonne produit valide trouvée.")
        return pd.DataFrame()

    # Sélectionner uniquement les colonnes produits valides et l'index
    df_summary_brut = df_pdc_perm_filtered[valid_product_columns].copy()
    
    # Calculer le total si nécessaire (optionnel pour cette version brute)
    df_summary_brut['Total'] = df_summary_brut[valid_product_columns].sum(axis=1)
    
    # S'assurer que les types sont numériques, remplacer NaN par 0 pour la somme
    for col in valid_product_columns:
        df_summary_brut[col] = pd.to_numeric(df_summary_brut[col], errors='coerce').fillna(0.0)
    df_summary_brut['Total'] = pd.to_numeric(df_summary_brut['Total'], errors='coerce').fillna(0.0)

    return df_summary_brut.round(2) # Arrondir à la fin

def get_RAW_pdc_perm_data_for_optim(): # Nouvelle fonction à appeler depuis ParametresApprosGenerator
    """
    Fonction principale pour obtenir les données PDC Perm BRUTES, formatées pour l'optimisation.
    """
    print("PDC.py - get_RAW_pdc_perm_data_for_optim: Chargement des données brutes...")
    df_pdc = load_pdc_perm_data() # Charge les données avec les grandes valeurs
    if not df_pdc.empty:
        # On utilise create_pdc_perm_summary_BRUT qui ne fait PAS la division
        df_summary_brut = create_pdc_perm_summary_BRUT(df_pdc) 
        if not df_summary_brut.empty:
            # Pas besoin de format_pdc_perm_summary si create_pdc_perm_summary_BRUT fait déjà l'arrondi
            return df_summary_brut
        else:
            print("PDC.py - get_RAW_pdc_perm_data_for_optim: La création du résumé BRUT a résulté en un DataFrame vide.")
            return pd.DataFrame()
    else:
        print("PDC.py - get_RAW_pdc_perm_data_for_optim: Le chargement des données PDC Perm a échoué.")
        return pd.DataFrame()

if __name__ == '__main__':
    # Ceci est un exemple de la façon dont vous pourriez utiliser ce module
    print(f"Chargement du fichier PDC Perm depuis: {PDC_FILE_PATH}")
    processed_data = get_processed_pdc_perm_data()
    
    
    if not processed_data.empty:
        print("\nDonnées PDC Perm traitées et formatées:")
        # Afficher l'index (dates) et les colonnes
        print(processed_data.head().to_string())
    else:
        print("\nAucune donnée PDC Perm n'a été traitée ou le résultat est vide.")

    
    df=load_pdc_perm_data()
    print(df.head())