import pandas as pd
import numpy as np
import datetime

# Import functions from other modules
import Encours
import Boite3
from MacroParam import DATE_COMMANDE # Assuming DATE_COMMANDE is used as a default start date

# Define the product types in the desired order for columns
PRODUCT_TYPES = ['Sec Méca', 'Sec Homogène', 'Sec Hétérogène', 'Frais Méca', 'Frais Manuel', 'Surgelés']

def create_total_summary(encours_summary_df, boite3_summary_df):
    """
    Crée un tableau récapitulatif "Total" en additionnant les données de En-cours et Boite 3.
    
    Args:
        encours_summary_df: DataFrame numérique du résumé des En-cours.
        boite3_summary_df: DataFrame numérique du résumé de Boite 3.
        
    Returns:
        DataFrame contenant le résumé "Total".
    """
    if encours_summary_df.empty and boite3_summary_df.empty:
        print("Les données En-cours et Boite 3 sont vides. Impossible de créer le tableau Total.")
        return pd.DataFrame()

    # Ensure consistent indexing for addition, using the index from encours_summary_df or boite3_summary_df
    # This assumes both dataframes will have a similar date index if not empty
    common_index = encours_summary_df.index if not encours_summary_df.empty else boite3_summary_df.index

    # Reindex both dataframes to the common index and PRODUCT_TYPES columns, filling missing values with 0
    encours_reindexed = encours_summary_df.reindex(index=common_index, columns=PRODUCT_TYPES).fillna(0)
    boite3_reindexed = boite3_summary_df.reindex(index=common_index, columns=PRODUCT_TYPES).fillna(0)
    
    # Add the reindexed dataframes
    total_summary_df = encours_reindexed.add(boite3_reindexed)
    
    # Recalculate the 'Total' column for the new summary
    total_summary_df['Total'] = total_summary_df[PRODUCT_TYPES].sum(axis=1)
    
    return total_summary_df

def format_total_summary(df):
    """
    Formate le tableau récapitulatif "Total" pour l'affichage.
    
    Args:
        df: DataFrame contenant le résumé "Total".
        
    Returns:
        DataFrame formaté pour l'affichage.
    """
    if df.empty:
        return df
    
    formatted_df = df.copy()
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(
            lambda x: '-' if pd.isna(x) or x == 0 else f"{x:.1f}"
        )
    return formatted_df

def get_processed_total_data(date_start_str=None, days_to_include=7, formatted=True):
    """
    Fonction principale pour obtenir le tableau récapitulatif "Total".
    
    Args:
        date_start_str: Date de début (chaîne au format '%d/%m/%Y', par défaut DATE_COMMANDE).
        days_to_include: Nombre de jours à inclure (par défaut, 7).
        formatted: Si True, formate le tableau pour l'affichage.
        
    Returns:
        DataFrame contenant le résumé "Total".
    """
    
    # Determine the start date for fetching data
    # The sub-modules (Encours, Boite3) handle their own default start date (DATE_COMMANDE)
    # if date_start_str is None.

    encours_summary_df = Encours.get_processed_data(
        file_path=None, 
        date_start=date_start_str, 
        days_to_include=days_to_include, 
        formatted=False
    )
    
    boite3_summary_df = Boite3.get_processed_data(
        file_path=None, 
        date_start=date_start_str, 
        days_to_include=days_to_include, 
        formatted=False
    )
    
    # Create the total summary
    total_summary = create_total_summary(encours_summary_df, boite3_summary_df)
    
    # Format if requested
    if formatted and not total_summary.empty:
        total_summary = format_total_summary(total_summary)
        
    return total_summary

if __name__ == "__main__":
    print("Test du module Total...")
    
    # Example: Get data for 7 days starting from DATE_COMMANDE
    summary_total_df = get_processed_total_data() 
    
    if not summary_total_df.empty:
        print("\nTableau récapitulatif Total:")
        print(summary_total_df)
    else:
        print("Impossible de créer le tableau récapitulatif Total.")

    # Example: Get data for 5 days starting from a specific date
    # specific_date_dt = datetime.datetime.strptime(DATE_COMMANDE, '%d/%m/%Y') + datetime.timedelta(days=2)
    # specific_date_str = specific_date_dt.strftime('%d/%m/%Y')
    # summary_total_df_specific_date = get_processed_total_data(date_start_str=specific_date_str, days_to_include=5)
    # if not summary_total_df_specific_date.empty:
    #     print(f"\nTableau récapitulatif Total pour 5 jours à partir du {specific_date_str}:")
    #     print(summary_total_df_specific_date)
    # else:
    #     print(f"Impossible de créer le tableau récapitulatif Total pour 5 jours à partir du {specific_date_str}.")
