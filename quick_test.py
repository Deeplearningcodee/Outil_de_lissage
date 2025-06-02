#!/usr/bin/env python3
"""
Quick test of corrected VBA optimization (first row only)
"""

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))

import vba_logic_fixed

def quick_test():
    """Quick test with just first row"""
    
    # Load data
    df_pdc_sim = pd.read_excel('PDC_Sim_Input_For_Optim.xlsx', sheet_name='PDC_Sim')
    df_detail = pd.read_csv('merged_predictions.csv', sep=';', encoding='latin1')
    
    print("QUICK TEST - First row only")
    print("="*50)
    
    # Test just first row (Sec Méca - A/B)
    test_row = df_pdc_sim.iloc[0:1].copy()
    
    print(f"Testing: {test_row.iloc[0]['Type de produits V2']}")
    print(f"Current variation: {test_row.iloc[0]['Variation PDC']:.1%}")
    print(f"Expected result: Top500=100%, Top3000=81%, Autre=0%")
    
    # Run limite_haute_basse only 
    print(f"\nRunning Limite_Haute_Basse...")
    df_after_lhb = vba_logic_fixed.vba_limite_haute_basse_fixed(test_row, df_detail)
    
    print(f"After LHB:")
    row = df_after_lhb.iloc[0]
    print(f"  Boost PDC: {row['Boost PDC']:.3f}")
    print(f"  Bounds: LB=({row['Limite Basse Top 500']:.1f}, {row['Limite Basse Top 3000']:.1f}, {row['Limite Basse Autre']:.1f})")
    print(f"  Bounds: UB=({row['Limite Haute Top 500']:.1f}, {row['Limite Haute Top 3000']:.1f}, {row['Limite Haute Autre']:.1f})")
    
    # Check if the limits make sense
    # With -79% variation (too low), we should get:
    # - Positive variation after boost? -> Upper limits to max
    # - Still negative? -> Cascading reduction (lower bounds set to 0)
    
    print(f"\nThis tells us what type of optimization is needed:")
    if row['Limite Haute Top 500'] > 1:
        print(f"  ✓ Upper limits > 1 -> Algorithm will try to INCREASE factors")
    else:
        print(f"  ✓ Upper limits = 1, some lower bounds = 0 -> Algorithm will try to DECREASE some factors")

def main():
    quick_test()

if __name__ == "__main__":
    main()