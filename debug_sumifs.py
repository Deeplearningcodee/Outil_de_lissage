#!/usr/bin/env python3
"""
Debug the SUMIFS calculation to match Excel exactly
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

def load_real_data():
    """Load the actual data"""
    df_pdc_sim = pd.read_excel('PDC_Sim_Input_For_Optim.xlsx', sheet_name='PDC_Sim')
    df_detail = pd.read_csv('merged_predictions.csv', sep=';', encoding='latin1')
    return df_pdc_sim, df_detail

def debug_excel_sumifs():
    """Debug the Excel SUMIFS formula step by step"""
    
    df_pdc_sim, df_detail = load_real_data()
    
    print("="*80)
    print("DEBUGGING EXCEL SUMIFS CALCULATION")
    print("="*80)
    
    # Take the first row as example: Sec Méca - A/B
    test_row = df_pdc_sim.iloc[0]
    type_produit_v2 = test_row['Type de produits V2']  # "Sec Méca - A/B"
    date_livraison = test_row['Jour livraison']        # 2025-05-30
    
    print(f"Test case: {type_produit_v2} on {date_livraison}")
    print(f"Excel shows: Commande optimisée = {test_row['Commande optimisée']:.0f}")
    
    # Excel SUMIFS formula: =SUMIFS(Détail!CF:CF;Détail!BM:BM;B18;Détail!S:S;D18)
    # CF = "Commande optimisée avec arrondi et mini et TS"
    # BM = "Type de produits V2" 
    # S = "DATE_LIVRAISON_V2"
    
    print(f"\n1. Filtering detail data...")
    print(f"   Looking for Type de produits V2 = '{type_produit_v2}'")
    print(f"   Looking for DATE_LIVRAISON_V2 = {date_livraison}")
    
    # Check what's in the detail data
    print(f"\n2. Detail data analysis:")
    print(f"   Total detail rows: {len(df_detail)}")
    print(f"   Unique Type de produits V2: {df_detail['Type de produits V2'].unique()}")
    print(f"   Unique DATE_LIVRAISON_V2: {df_detail['DATE_LIVRAISON_V2'].unique()[:5]}...")
    
    # Filter by Type de produits V2 (case insensitive)
    type_mask = df_detail['Type de produits V2'].str.lower() == type_produit_v2.lower()
    type_filtered = df_detail[type_mask]
    print(f"   Rows matching type '{type_produit_v2}' (case insensitive): {len(type_filtered)}")
    
    if len(type_filtered) == 0:
        print("   ERROR: No rows match the product type!")
        return
    
    # Check date format and filtering
    print(f"\n3. Date filtering analysis:")
    print(f"   DATE_LIVRAISON_V2 data type: {df_detail['DATE_LIVRAISON_V2'].dtype}")
    print(f"   Sample DATE_LIVRAISON_V2 values: {type_filtered['DATE_LIVRAISON_V2'].head().tolist()}")
    
    # Convert dates for comparison
    detail_dates = pd.to_datetime(type_filtered['DATE_LIVRAISON_V2']).dt.normalize()
    target_date = pd.to_datetime(date_livraison).normalize()
    
    print(f"   Target date (normalized): {target_date}")
    print(f"   Detail dates (normalized): {detail_dates.unique()}")
    
    # Filter by date
    date_mask = detail_dates == target_date
    final_filtered = type_filtered[date_mask]
    print(f"   Rows matching both type and date: {len(final_filtered)}")
    
    if len(final_filtered) == 0:
        print("   ERROR: No rows match both product type and date!")
        return
    
    # Check the CF column (Commande optimisée avec arrondi et mini et TS)
    print(f"\n4. CF Column analysis:")
    cf_column = 'Commande optimisée avec arrondi et mini et TS'
    
    if cf_column not in df_detail.columns:
        print(f"   ERROR: Column '{cf_column}' not found!")
        print(f"   Available columns containing 'Commande': {[col for col in df_detail.columns if 'Commande' in col]}")
        return
    
    cf_values = final_filtered[cf_column]
    print(f"   CF values to sum: {len(cf_values)} values")
    print(f"   CF values sample: {cf_values.head().tolist()}")
    print(f"   CF values stats:")
    print(f"     Sum: {cf_values.sum():.2f}")
    print(f"     Mean: {cf_values.mean():.2f}")
    print(f"     Min: {cf_values.min():.2f}")
    print(f"     Max: {cf_values.max():.2f}")
    print(f"     Non-zero values: {(cf_values != 0).sum()}")
    
    # Compare with Excel result
    excel_result = test_row['Commande optimisée']
    python_result = cf_values.sum()
    
    print(f"\n5. COMPARISON:")
    print(f"   Excel SUMIFS result:  {excel_result:.2f}")
    print(f"   Python sum result:    {python_result:.2f}")
    print(f"   Difference:           {abs(excel_result - python_result):.2f}")
    print(f"   Match?                {'✓ YES' if abs(excel_result - python_result) < 1 else '✗ NO'}")
    
    if abs(excel_result - python_result) > 1:
        print(f"\n   DEBUGGING THE DIFFERENCE...")
        
        # Check if CF values are calculated correctly
        print(f"   Checking if CF values are properly calculated...")
        
        # Show a few sample rows with all relevant columns
        sample_cols = ['Type de produits V2', 'DATE_LIVRAISON_V2', 'Top', 'SM Final', 
                      'Commande optimisée sans arrondi', 'Commande optimisée avec arrondi et mini',
                      'Commande optimisée avec arrondi et mini et TS']
        
        available_cols = [col for col in sample_cols if col in final_filtered.columns]
        print(f"   Sample data:")
        print(final_filtered[available_cols].head())

def main():
    debug_excel_sumifs()

if __name__ == "__main__":
    main()