#!/usr/bin/env python3
"""
Test if our calculation matches Excel when all factors = 1.0
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

import vba_logic_fixed

def test_baseline_calculation():
    """Test calculation with baseline parameters (all 1.0)"""
    
    print("="*80)
    print("TESTING BASELINE CALCULATION (all factors = 1.0)")
    print("="*80)
    
    # Load data
    df_pdc_sim = pd.read_excel('PDC_Sim_Input_For_Optim.xlsx', sheet_name='PDC_Sim')
    df_detail = pd.read_csv('merged_predictions.csv', sep=';', encoding='latin1')
    
    # Test first row
    test_row = df_pdc_sim.iloc[0].copy()
    
    # Set all factors to 1.0 (baseline)
    test_row['Top 500'] = 1.0
    test_row['Top 3000'] = 1.0
    test_row['Autre'] = 1.0
    test_row['Boost PDC'] = 0.0
    
    # Create optimization params DataFrame (all 1.0)
    optimization_params = df_pdc_sim.copy()
    optimization_params['Top 500'] = 1.0
    optimization_params['Top 3000'] = 1.0
    optimization_params['Autre'] = 1.0
    optimization_params['Boost PDC'] = 0.0
    
    type_produit_v2 = test_row['Type de produits V2']
    date_livraison = test_row['Jour livraison']
    
    print(f"Test case: {type_produit_v2} on {date_livraison}")
    print(f"Excel baseline: Commande optimisée = {test_row['Commande optimisée']:.2f}")
    
    # Calculate using our function
    python_result = vba_logic_fixed.calculate_total_commande_optimisee(
        df_detail, optimization_params, type_produit_v2, date_livraison)
    
    print(f"Python result:  Commande optimisée = {python_result:.2f}")
    print(f"Difference:     {abs(python_result - test_row['Commande optimisée']):.2f}")
    
    # Calculate variation
    variation_relative, difference_absolue = vba_logic_fixed.calculate_excel_variation_relative(
        test_row, df_detail, optimization_params)
    
    # Expected variation from Excel
    expected_variation = test_row['Variation PDC']
    
    print(f"\nVariation comparison:")
    print(f"Excel variation:  {expected_variation:.1%}")
    print(f"Python variation: {variation_relative:.1%}")
    print(f"Difference:       {abs(variation_relative - expected_variation):.3f}")
    
    # Check if we match
    command_match = abs(python_result - test_row['Commande optimisée']) < 1.0
    variation_match = abs(variation_relative - expected_variation) < 0.01
    
    print(f"\nResults:")
    print(f"Command calculation matches: {'✓ YES' if command_match else '✗ NO'}")
    print(f"Variation calculation matches: {'✓ YES' if variation_match else '✗ NO'}")
    
    if not command_match or not variation_match:
        print(f"\nDEBUGGING...")
        
        # Check what's different
        if not command_match:
            print("Command calculation issues:")
            # Let's check if we're using the right CF column
            mask = (df_detail['Type de produits V2'].str.lower() == type_produit_v2.lower()) & \
                   (pd.to_datetime(df_detail['DATE_LIVRAISON_V2']).dt.normalize() == pd.to_datetime(date_livraison).normalize())
            filtered = df_detail[mask]
            
            # Compare stored CF vs calculated CF
            stored_cf_sum = filtered['Commande optimisée avec arrondi et mini et TS'].sum()
            print(f"  Stored CF sum (from CSV): {stored_cf_sum:.2f}")
            print(f"  Calculated CF sum: {python_result:.2f}")
            print(f"  Should match stored CF for baseline case")

def main():
    test_baseline_calculation()

if __name__ == "__main__":
    main()