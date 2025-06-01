import os
import pandas as pd
import numpy as np
import glob
import datetime
from dateutil.relativedelta import relativedelta
from MacroParam import TYPES_PRODUITS, DATE_COMMANDE
import MacroParam

def find_latest_encours_file(directory_path=None):
    """
    Trouve le fichier le plus récent contenant "Synthese_En-Cours" dans son nom
    dans le répertoire spécifié.
    
    Args:
        directory_path: Chemin du répertoire où chercher (par défaut, dossier CSV)
        
    Returns:
        Chemin du fichier le plus récent contenant "Synthese_En-Cours" dans son nom,
        ou None si aucun fichier correspondant n'est trouvé.
    """
    if directory_path is None:
        directory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CSV')
    
    # Motif pour rechercher tous les fichiers CSV pertinents
    pattern = os.path.join(directory_path, "*Synthese_En-Cours.csv")
    
    # Liste tous les fichiers correspondants
    files = glob.glob(pattern)
    
    if not files:
        print(f"Aucun fichier contenant 'Synthese_En-Cours' trouvé dans {directory_path}")
        return None
    
    # Trier les fichiers par date de modification (le plus récent en premier)
    latest_file = max(files, key=os.path.getmtime)
    
    print(f"Fichier En-Cours le plus récent trouvé : {latest_file}")
    return latest_file

def load_encours_data(file_path=None):
    """
    Charge les données d'en-cours à partir du fichier CSV spécifié ou du dernier fichier disponible.
    
    Args:
        file_path: Chemin du fichier CSV contenant les données d'en-cours (optionnel)
        
    Returns:
        DataFrame contenant les données d'en-cours
    """
    # Si aucun fichier n'est spécifié, trouver le plus récent
    if file_path is None:
        file_path = find_latest_encours_file()
        if file_path is None:
            print("Aucun fichier d'en-cours trouvé.")
            return pd.DataFrame()  # Retourner un DataFrame vide
    
    try:
        # Charger le fichier CSV
        df = pd.read_csv(file_path, sep=';', encoding='latin1')
        
        # Vérifier qu'on a les colonnes obligatoires
        required_columns = ['CLASSE_STOCKAGE', 'DATE_LIVRAISON', 'QUANTITE_CONFIRMEE', 'Commande_Origine']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Erreur: Colonnes manquantes dans {file_path}: {missing_columns}")
            return pd.DataFrame()
        
        # Renommer la colonne QUANTITE_CONFIRMEE en QUANTITE pour simplifier
        if 'QUANTITE_CONFIRMEE' in df.columns and 'QUANTITE' not in df.columns:
            df.rename(columns={'QUANTITE_CONFIRMEE': 'QUANTITE'}, inplace=True)
        
        # Convertir CLASSE_STOCKAGE en chaîne de caractères
        df['CLASSE_STOCKAGE'] = df['CLASSE_STOCKAGE'].astype(str)
        
        # Convertir DATE_LIVRAISON en datetime
        df['DATE_LIVRAISON'] = pd.to_datetime(df['DATE_LIVRAISON'], format='%d/%m/%Y', errors='coerce')
        
        # Ajouter une colonne Type de produits
        df['Type de produits'] = 'Autre'  # Valeur par défaut
        
        # Utiliser le dictionnaire TYPES_PRODUITS pour déterminer le type de produit
        for zone_stockage, type_produit in TYPES_PRODUITS.items():
            # Nous comparons CLASSE_STOCKAGE avec les clés du dictionnaire
            df.loc[df['CLASSE_STOCKAGE'] == str(zone_stockage), 'Type de produits'] = type_produit
        
        # Ajouter la colonne "Commande à exclure"
        df['Commande à exclure'] = np.where(df['Commande_Origine'] == 'PROMO', 'Oui', 'Non')
        
        print(f"Données En-cours chargées avec succès: {len(df)} lignes")
        return df
        
    except Exception as e:
        print(f"Erreur lors du chargement du fichier d'en-cours {file_path}: {e}")
        return pd.DataFrame()

def create_encours_summary(df=None, date_start=None, days_to_include=7):
    """
    Crée un tableau récapitulatif des en-cours par type de produit et par date.
    """
    if df is None:
        df = load_encours_data() 
        
    if df.empty:
        print("  Encours: Aucune donnée d'en-cours disponible pour créer le tableau récapitulatif.")
        return pd.DataFrame()
    
    if date_start is None:
        processed_date_start = DATE_COMMANDE 
    elif isinstance(date_start, str):
        try:
            processed_date_start = pd.to_datetime(date_start, format='%d/%m/%Y', errors='raise')
        except ValueError:
            print(f"  Encours ERREUR: date_start string '{date_start}' n'est pas au format attendu '%d/%m/%Y'. Utilisation de DATE_COMMANDE.")
            processed_date_start = DATE_COMMANDE
    elif isinstance(date_start, (datetime.datetime, pd.Timestamp)):
        processed_date_start = date_start
    else:
        print(f"  Encours AVERTISSEMENT: Type de date_start inconnu ({type(date_start)}). Utilisation de DATE_COMMANDE.")
        processed_date_start = DATE_COMMANDE
    
    # Utiliser les Type de produits V2 pour les colonnes, comme dans PDC_Sim
    # Cela suppose que votre fichier d'en-cours ou le mapping dans load_encours_data
    # peut produire une colonne 'Type de produits V2'.
    # Si 'Type de produits' est utilisé, ajustez en conséquence.
    # Pour être cohérent avec la feuille PDC_Sim, il faudrait agréger par 'Type de produits V2'.
    # La fonction create_parametres_commandes_appros_table_internal() crée les lignes par 'Type de produits V2'
    # Donc, le résumé EnCours doit aussi être par 'Type de produits V2'.

    # Récupérer dynamiquement les Type de produits V2 depuis MacroParam
    all_type_produits_v2 = []
    for base_type, versions in MacroParam.TYPES_PRODUITS_V2.items():
        for version in versions:
            if version:
                all_type_produits_v2.append(f"{base_type} - {version}")
            else:
                all_type_produits_v2.append(base_type)
    # Garder seulement les types uniques et les trier pour un ordre cohérent
    product_types_v2_ordered = sorted(list(set(all_type_produits_v2)))

    # S'assurer que le df d'en-cours a une colonne 'Type de produits V2'
    # Si ce n'est pas le cas, il faut la créer.
    # Supposons que load_encours_data ou une étape intermédiaire le fait.
    # Pour ce test, on va supposer que 'Type de produits' est suffisant, mais idéalement, ce serait 'Type de produits V2'.
    # Si votre df d'encours a 'Type de produits' et que la clé de recherche dans PDC_Sim est 'Type de produits V2',
    # il faudra un mapping ou une adaptation.
    # Pour l'instant, on va utiliser la colonne 'Type de produits' existante de df, et les colonnes du résumé seront
    # basées sur les valeurs uniques de cette colonne, ou sur product_types_ordered s'ils correspondent.

    if 'Type de produits' not in df.columns: # ou 'Type de produits V2' si c'est ce que vous utilisez
        print("  Encours ERREUR: Colonne 'Type de produits' (ou 'Type de produits V2') manquante pour l'agrégation.")
        return pd.DataFrame()

    dates_for_summary = [processed_date_start + pd.Timedelta(days=i) for i in range(days_to_include)]
    data_rows = []
    
    if 'DATE_LIVRAISON' not in df.columns or df['DATE_LIVRAISON'].isnull().all():
        print("  Encours ERREUR: Colonne 'DATE_LIVRAISON' manquante ou vide.")
        # Retourner un DataFrame avec les bonnes colonnes mais rempli de 0
        return pd.DataFrame(0, index=dates_for_summary, columns=product_types_v2_ordered + ['Total']).rename_axis('Date')


    df['DATE_LIVRAISON'] = pd.to_datetime(df['DATE_LIVRAISON'], errors='coerce')
    df_filtered_for_dates = df.dropna(subset=['DATE_LIVRAISON'])

    # Le résumé doit avoir les 'Type de produits V2' en colonnes
    # Si df_filtered_for_dates n'a que 'Type de produits', il faut mapper vers 'Type de produits V2'
    # Pour l'instant, je vais pivoter sur 'Type de produits' et ParametresApprosGenerator devra gérer la correspondance.
    # IDÉALEMENT: df (en entrée de create_encours_summary) devrait déjà avoir 'Type de produits V2'
    # OU, load_encours_data devrait le créer.

    # Pivotons les données d'en-cours    # Assurons-nous que QUANTITE est numérique
    if 'QUANTITE' not in df_filtered_for_dates.columns:
        print("  Encours ERREUR: Colonne 'QUANTITE' manquante.")
        return pd.DataFrame(0, index=dates_for_summary, columns=product_types_v2_ordered + ['Total']).rename_axis('Date')
    df_filtered_for_dates['QUANTITE'] = pd.to_numeric(df_filtered_for_dates['QUANTITE'], errors='coerce').fillna(0)
    
    # Exclure les commandes PROMO (où 'Commande à exclure' == 'Oui')
    df_to_pivot = df_filtered_for_dates[df_filtered_for_dates['Commande à exclure'] != 'Oui'].copy()
    print(f"  Encours: {len(df_filtered_for_dates) - len(df_to_pivot)} commandes PROMO exclues.")
    
    # Créer la colonne 'Type de produits V2' en distribuant les types de base selon A/B et A/C
    # basé sur la proximité de la date de livraison avec les dates de référence
    def determine_ab_ac_variant(row):
        base_type = row['Type de produits']
        delivery_date = row['DATE_LIVRAISON']
        
        # Si le type de produit n'a pas de variants A/B et A/C, retourner le type de base
        if base_type not in MacroParam.TYPES_PRODUITS_V2:
            return base_type
        
        variants = MacroParam.TYPES_PRODUITS_V2[base_type]
        
        # Si pas de variants définis (liste vide ou contient seulement ""), retourner le type de base
        if not variants or (len(variants) == 1 and variants[0] == ""):
            return base_type
        
        # Pour les types avec variants A/B et A/C, déterminer selon la date
        if "A/B" in variants and "A/C" in variants:
            # Calculer la distance aux dates de référence
            distance_ab = abs((delivery_date - MacroParam.DATE_REF_JOUR_AB).days)
            distance_ac = abs((delivery_date - MacroParam.DATE_REF_JOUR_AC).days)
            
            # Choisir le variant le plus proche
            if distance_ab <= distance_ac:
                return f"{base_type} - A/B"
            else:
                return f"{base_type} - A/C"
        
        # Cas par défaut: retourner le type de base
        return base_type
    
    # Appliquer la logique de détermination A/B vs A/C
    df_to_pivot = df_to_pivot.copy()
    df_to_pivot['Type de produits V2'] = df_to_pivot.apply(determine_ab_ac_variant, axis=1)
    
    print("  Encours: Colonne 'Type de produits V2' créée avec distribution A/B/A/C basée sur les dates de livraison.")


    pivot_df = pd.pivot_table(
        df_to_pivot,
        values='QUANTITE',
        index='DATE_LIVRAISON', # Sera l'index du résultat
        columns='Type de produits V2', # Colonnes du résultat
        aggfunc='sum',
        fill_value=0 # Remplir les NaN avec 0 après le pivot
    )
    
    # --- MODIFICATION ICI: Supprimer la division par 1000.0 ---
    # pivot_df = pivot_df / 1000.0 # ANCIENNE LIGNE AVEC DIVISION
    # Les valeurs sont maintenant à leur échelle originale.

    # Réindexer pour inclure toutes les dates et tous les types de produits V2 ordonnés
    summary_df = pivot_df.reindex(index=dates_for_summary, columns=product_types_v2_ordered, fill_value=0)
    
    # Calculer le total
    summary_df['Total'] = summary_df.sum(axis=1)
    summary_df.index.name = 'Date'
    
    print(f"  Encours: Tableau récapitulatif BRUT (sans /1000) créé avec {len(summary_df)} lignes.")
    return summary_df



def format_encours_summary(df):
    """
    Formate le tableau récapitulatif des en-cours pour l'affichage.
    Attend maintenant des valeurs à l'échelle originale.
    """
    if df.empty:
        return df
    formatted_df = df.copy()
    for col in formatted_df.columns:
        # Afficher avec 0 décimale si c'est un entier, sinon 1 décimale.
        # Ou toujours 0 décimale pour les quantités.
        formatted_df[col] = formatted_df[col].apply(
            lambda x: '-' if pd.isna(x) or x == 0 else f"{x:,.0f}" # Format avec séparateur de milliers, 0 décimales
        )
    return formatted_df

def get_processed_data(file_path=None, date_start=None, days_to_include=7, formatted=True):
    """
    Fonction principale pour obtenir le tableau récapitulatif des en-cours.
    Si formatted=False, retourne les données numériques brutes (non divisées par 1000).
    """
    df_loaded = load_encours_data(file_path)
    summary_df_raw = create_encours_summary(df_loaded, date_start, days_to_include)
    
    if not summary_df_raw.empty:
        # Normaliser les noms de colonnes pour correspondre à mergekey_TypeV2
        summary_df_raw.columns = [str(col).lower().strip() if col != 'Total' else 'Total' for col in summary_df_raw.columns]
        print("  Encours: Noms de colonnes de summary_df_raw normalisés en minuscules.")
        # S'assurer que l'index est bien DatetimeIndex
        if not isinstance(summary_df_raw.index, pd.DatetimeIndex):
            summary_df_raw.index = pd.to_datetime(summary_df_raw.index, errors='coerce')
            summary_df_raw.dropna(axis=0, subset=[summary_df_raw.index.name], inplace=True) # Enlever les dates invalides

    return summary_df_raw # Retourner les données numériques brutes normalisées



if __name__ == "__main__":
    print("Test du module Encours...")
    
    # Obtenir le tableau récapitulatif
    summary_df = get_processed_data()

    
    if not summary_df.empty:
        print("\nTableau récapitulatif des en-cours:")
        print(summary_df)
        #save to CSV
        output_file = "encours_summary.csv"
        summary_df.to_csv(output_file, sep=';', index=True, encoding='latin1')
    
        
        
     
   