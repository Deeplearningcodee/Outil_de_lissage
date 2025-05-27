import os
import pandas as pd
import numpy as np
import glob
import datetime
from dateutil.relativedelta import relativedelta
from MacroParam import TYPES_PRODUITS, DATE_COMMANDE

def find_latest_boite3_file(directory_path=None):
    """
    Trouve le fichier le plus récent contenant "Synthese_Boite_3" dans son nom
    dans le répertoire spécifié.
    
    Args:
        directory_path: Chemin du répertoire où chercher (par défaut, dossier CSV)
        
    Returns:
        Chemin du fichier le plus récent contenant "Synthese_Boite_3" dans son nom,
        ou None si aucun fichier correspondant n'est trouvé.
    """
    if directory_path is None:
        directory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CSV')
    
    # Motif pour rechercher tous les fichiers CSV pertinents
    pattern = os.path.join(directory_path, "*Synthese_Boite_3.csv")
    
    # Liste tous les fichiers correspondants
    files = glob.glob(pattern)
    
    if not files:
        print(f"Aucun fichier contenant 'Synthese_Boite_3' trouvé dans {directory_path}")
        return None
    
    # Trier les fichiers par date de modification (le plus récent en premier)
    latest_file = max(files, key=os.path.getmtime)
    
    print(f"Fichier Boite3 le plus récent trouvé : {latest_file}")
    return latest_file

def load_boite3_data(file_path=None):
    """
    Charge les données de Boite 3 à partir du fichier CSV spécifié ou du dernier fichier disponible.
    
    Args:
        file_path: Chemin du fichier CSV contenant les données de Boite 3 (optionnel)
        
    Returns:
        DataFrame contenant les données de Boite 3
    """
    # Si aucun fichier n'est spécifié, trouver le plus récent
    if file_path is None:
        file_path = find_latest_boite3_file()
        if file_path is None:
            print("Aucun fichier de Boite 3 trouvé.")
            return pd.DataFrame()  # Retourner un DataFrame vide
    
    try:
        # Charger le fichier CSV
        df = pd.read_csv(file_path, sep=';', encoding='latin1')
        
        # Vérifier qu'on a les colonnes obligatoires
        required_columns = ['CLASSE_STOCKAGE', 'DATE_LIVRAISON', 'QUANTITE_CONFIRMEE']
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
        
        print(f"Données Boite3 chargées avec succès: {len(df)} lignes")
        return df
        
    except Exception as e:
        print(f"Erreur lors du chargement du fichier Boite3 {file_path}: {e}")
        return pd.DataFrame()

def create_boite3_summary(df=None, date_start=None, days_to_include=7):
    """
    Crée un tableau récapitulatif des Boite 3 par type de produit et par date.
    
    Args:
        df: DataFrame contenant les données de Boite 3 (optionnel)
        date_start: Date de début (par défaut, date de commande)
        days_to_include: Nombre de jours à inclure (par défaut, 7)
        
    Returns:
        DataFrame contenant le résumé des Boite 3
    """
    if df is None:
        df = load_boite3_data()
        
    if df.empty:
        print("Aucune donnée de Boite 3 disponible pour créer le tableau récapitulatif.")
        return pd.DataFrame()
    
    # Si date_start n'est pas spécifié, utiliser la date de commande
    if date_start is None:
        date_start = datetime.datetime.strptime(DATE_COMMANDE, '%d/%m/%Y')
    elif isinstance(date_start, str):
        date_start = datetime.datetime.strptime(date_start, '%d/%m/%Y')
    
    # Liste des types de produits à inclure dans l'ordre
    product_types = ['Sec Méca', 'Sec Homogène', 'Sec Hétérogène', 'Frais Méca', 'Frais Manuel', 'Surgelés']
    
    # Créer un dictionnaire de dates pour l'axe vertical
    dates = [date_start + datetime.timedelta(days=i) for i in range(days_to_include)]
    date_str = [d.strftime('%d/%m/%Y') for d in dates]
    
    # Créer une liste pour stocker les données
    data_rows = []
    
    # Pour chaque date
    for date_obj in dates:
        row_data = {'Date': date_obj.strftime('%d/%m/%Y')}
        total_row = 0
        
        # Pour chaque type de produit
        for prod_type in product_types:
            # Filtrer les données:
            # 1. Date de livraison = date actuelle
            # 2. Type de produit = prod_type
            filtered_df = df[(df['DATE_LIVRAISON'].dt.date == date_obj.date()) & 
                            (df['Type de produits'] == prod_type)]
            
            # Somme des quantités et conversion en k (diviser par 1000)
            value = filtered_df['QUANTITE'].sum() / 1000 if not filtered_df.empty else 0
            
            # Ajouter la valeur à la ligne
            row_data[prod_type] = value
            total_row += value
        
        # Ajouter le total de la ligne
        row_data['Total'] = total_row
        
        # Ajouter la ligne aux données
        data_rows.append(row_data)
    
    # Créer le DataFrame
    summary_df = pd.DataFrame(data_rows)
    
    # Définir 'Date' comme index
    if 'Date' in summary_df.columns:
        summary_df.set_index('Date', inplace=True)
    
    # Réordonner les colonnes pour avoir Total à la fin
    if 'Total' in summary_df.columns:
        cols = [col for col in summary_df.columns if col != 'Total'] + ['Total']
        summary_df = summary_df[cols]
    
    return summary_df

def format_boite3_summary(df):
    """
    Formate le tableau récapitulatif de Boite 3 pour l'affichage.
    
    Args:
        df: DataFrame contenant le résumé de Boite 3
        
    Returns:
        DataFrame formaté pour l'affichage
    """
    if df.empty:
        return df
    
    # Copier le DataFrame pour éviter de modifier l'original
    formatted_df = df.copy()
    
    # Formater les nombres pour l'affichage
    for col in formatted_df.columns:
        # Remplacer les valeurs nulles par '-' et arrondir à 1 décimale
        formatted_df[col] = formatted_df[col].apply(
            lambda x: '-' if x == 0 or pd.isna(x) else f"{x:.1f}")
    
    return formatted_df

def get_processed_data(file_path=None, date_start=None, days_to_include=7, formatted=True):
    """
    Fonction principale pour obtenir le tableau récapitulatif de Boite 3.
    
    Args:
        file_path: Chemin du fichier CSV contenant les données de Boite 3 (optionnel)
        date_start: Date de début (par défaut, date de commande)
        days_to_include: Nombre de jours à inclure (par défaut, 7)
        formatted: Si True, formate le tableau pour l'affichage
        
    Returns:
        DataFrame contenant le résumé de Boite 3
    """
    # Charger les données
    df = load_boite3_data(file_path)
    
    # Créer le tableau récapitulatif
    summary_df = create_boite3_summary(df, date_start, days_to_include)
    
    # Formater le tableau si demandé
    if formatted and not summary_df.empty:
        summary_df = format_boite3_summary(summary_df)
    
    return summary_df

if __name__ == "__main__":
    print("Test du module Boite3...")
    
    # Obtenir le tableau récapitulatif
    summary_df = get_processed_data()
    
    if not summary_df.empty:
        print("\nTableau récapitulatif de Boite 3:")
        print(summary_df)
    else:
        print("Impossible de créer le tableau récapitulatif de Boite 3.")