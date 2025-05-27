"""
Module pour calculer la colonne 'Commande optimisée' pour le tableau PDC - Simulation Commande.
Ce module émule la formule Excel SUMIFS pour la commande optimisée:
=SI(B18="";"";SOMME.SI.ENS(Détail!CF:CF;Détail!BM:BM;B18;Détail!S:S;D18))
"""
import pandas as pd
import numpy as np

def get_processed_data(merged_df):
    """
    Calcule les valeurs de la colonne "Commande optimisée" basées sur les données merged_predictions.csv.
    
    Cette fonction implémente la formule Excel:
    =SI(B18="";"";SOMME.SI.ENS(Détail!CF:CF;Détail!BM:BM;B18;Détail!S:S;D18))
    
    Signification: Si "Type de produits V2" est vide, retourner vide;
                  sinon, sommer les valeurs de "Commande optimisée avec arrondi et mini et TS" (CF)
                  où "Type de produits V2" (BM) égale B18 et "DATE_LIVRAISON_V2" (S) égale D18
    
    Args:
        merged_df (pd.DataFrame): Le dataframe fusionné contenant toutes les colonnes requises
    
    Returns:
        pd.DataFrame: Un dataframe avec les colonnes "Type de produits V2", "Jour de livraison" 
                      et "Commande optimisée"
    """
    print("Calcul de la colonne 'Commande optimisée' pour PDC - Simulation Commande...")
    
    # Vérifier si les colonnes requises sont dans le dataframe
    required_cols = ['Type de produits V2', 'DATE_LIVRAISON_V2', 'Commande optimisée avec arrondi et mini et TS']
    if not all(col in merged_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in merged_df.columns]
        print(f"Attention: Colonnes requises manquantes pour le calcul de 'Commande optimisée': {missing_cols}")
        print("Création d'un DataFrame vide avec la structure requise.")
        return pd.DataFrame(columns=['Type de produits V2', 'Jour de livraison', 'Commande optimisée'])
    
    # Créer une copie du dataframe d'entrée avec uniquement les colonnes nécessaires
    df_copy = merged_df.copy()
    
    # S'assurer que les types de données sont corrects
    df_copy['Type de produits V2'] = df_copy['Type de produits V2'].astype(str)
    
    # Convertir DATE_LIVRAISON_V2 en datetime si ce n'est pas déjà fait
    if df_copy['DATE_LIVRAISON_V2'].dtype != 'datetime64[ns]':
        try:
            df_copy['DATE_LIVRAISON_V2'] = pd.to_datetime(df_copy['DATE_LIVRAISON_V2'], format='%d/%m/%Y', errors='coerce')
        except Exception as e:
            print(f"Erreur lors de la conversion de DATE_LIVRAISON_V2 en datetime: {e}")
            df_copy['DATE_LIVRAISON_V2'] = pd.to_datetime(df_copy['DATE_LIVRAISON_V2'], errors='coerce')
    
    # S'assurer que 'Commande optimisée avec arrondi et mini et TS' est numérique
    df_copy['Commande optimisée avec arrondi et mini et TS'] = pd.to_numeric(
        df_copy['Commande optimisée avec arrondi et mini et TS'], errors='coerce'
    ).fillna(0)
    
    # Obtenir les combinaisons uniques de Type de produits V2 et DATE_LIVRAISON_V2
    unique_product_types = df_copy['Type de produits V2'].dropna().unique()
    
    # Créer un nouveau dataframe pour stocker les résultats
    result_data = []
    
    for prod_type in unique_product_types:
        # Ignorer les types de produits vides
        if pd.isna(prod_type) or prod_type == "":
            continue
            
        # Obtenir les dates uniques pour ce type de produit
        type_data = df_copy[df_copy['Type de produits V2'] == prod_type]
        if type_data.empty:
            continue
            
        dates = type_data['DATE_LIVRAISON_V2'].dropna().unique()
        
        for date in dates:
            # Ignorer les dates NaT
            if pd.isna(date):
                continue
                
            # Calculer la somme pour ce type de produit et cette date
            mask = (df_copy['Type de produits V2'] == prod_type) & (df_copy['DATE_LIVRAISON_V2'] == date)
            
            # Vérifier s'il y a des lignes correspondantes
            if not df_copy.loc[mask].empty:
                sum_value = df_copy.loc[mask, 'Commande optimisée avec arrondi et mini et TS'].sum()
                
                # Trouver le 'Type de produits' correspondant
                type_produits = type_data['Type de produits'].iloc[0] if 'Type de produits' in type_data.columns else prod_type
                
                # Ajouter le résultat à nos données de sortie
                result_data.append({
                    'Type de produits V2': prod_type,
                    'Type de produits': type_produits,
                    'Jour de livraison': date.strftime('%d/%m/%Y'),  # Formater la date en chaîne DD/MM/YYYY
                    'Commande optimisée': sum_value
                })
    
    # Créer le dataframe de résultat
    result_df = pd.DataFrame(result_data)
    
    print(f"Calcul de 'Commande optimisée' terminé avec succès. {len(result_df)} lignes générées.")
    return result_df

if __name__ == "__main__":
    # Tester le module avec le fichier merged_predictions
    try:
        print("Chargement des prédictions fusionnées...")
        merged_df = pd.read_csv('merged_predictions.csv', sep=';', encoding='latin1')
        print(f"Chargé {len(merged_df)} lignes depuis merged_predictions.csv")
        
        result = get_processed_data(merged_df)
        print(result.head())
        
        # Sauvegarder le résultat pour vérification
        result.to_csv('pdc_commande_optimisee.csv', sep=';', index=False)
        print("Résultats sauvegardés dans pdc_commande_optimisee.csv")
    except Exception as e:
        print(f"Erreur lors du test du module PDC_CommandeOptimisee: {e}")
