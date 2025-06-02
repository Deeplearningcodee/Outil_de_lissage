#!/usr/bin/env python3
"""
Test VBA optimization with real Excel data
"""

import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

import vba_logic_fixed
import MacroParam

def load_real_data():
    """Load the actual Excel data"""
    print("Loading real Excel data...")
    
    # Load PDC_Sim data
    df_pdc_sim = pd.read_excel('PDC_Sim_Input_For_Optim.xlsx', sheet_name='PDC_Sim')
    print(f"PDC_Sim loaded: {df_pdc_sim.shape}")
    
    # Load processed detail data (not raw data)
    detail_file = 'merged_predictions.csv'
    try:
        df_detail = pd.read_csv(detail_file, sep=';', encoding='latin1')
    except UnicodeDecodeError:
        df_detail = pd.read_csv(detail_file, sep=';', encoding='utf-8')
    
    print(f"Detail data loaded: {df_detail.shape}")
    print(f"Detail columns: {df_detail.columns.tolist()[:10]}...")  # First 10 columns
    
    return df_pdc_sim, df_detail

def print_initial_vs_expected():
    """Print comparison of initial data vs expected Excel results"""
    
    print("\n" + "="*80)
    print("REAL DATA ANALYSIS")
    print("="*80)
    
    df_pdc_sim, df_detail = load_real_data()
    
    print("\nCurrent PDC_Sim parameters (BEFORE optimization):")
    print("-" * 50)
    
    for idx, row in df_pdc_sim.iterrows():
        type_prod = row['Type de produits V2']
        print(f"\n{type_prod}:")
        print(f"  Current factors: Top500={row['Top 500']}, Top3000={row['Top 3000']}, Autre={row['Autre']}")
        print(f"  PDC={row['PDC']:.0f}, En-cours={row['En-cours']:.0f}")
        print(f"  Current Commande optimisée={row['Commande optimisée']:.0f}")
        print(f"  Current Variation={row['Variation PDC']:.1%}")
        
        # Show expected results
        if type_prod == 'Sec Méca - A/B':
            print(f"  Expected (Excel): Top500=100%, Top3000=81%, Autre=0%")
        elif type_prod == 'Sec Méca - A/C':
            print(f"  Expected (Excel): Top500=100%, Top3000=100%, Autre=92%")
        elif type_prod == 'Sec Homogène - A/B':
            print(f"  Expected (Excel): Top500=100%, Top3000=100%, Autre=100%")

def test_with_real_data():
    """Test VBA optimization with real Excel data"""
    
    print("\n" + "="*80)
    print("TESTING VBA OPTIMIZATION WITH REAL DATA")
    print("="*80)
    
    # Load real data
    df_pdc_sim, df_detail = load_real_data()
    
    # Show data structure matching
    print(f"\nData structure check:")
    print(f"PDC_Sim has columns: {df_pdc_sim.columns.tolist()}")
    
    # Check if detail data has required columns
    required_detail_cols = ['Type de produits V2', 'DATE_LIVRAISON_V2', 'Top', 'SM Final', 
                           'Prév C1-L2 Finale', 'Prév L1-L2 Finale', 'Mini Publication FL',
                           'STOCK_REEL', 'RAL', 'Facteur Multiplicatif Appro', 'PCB', 'TS']
    
    missing_cols = [col for col in required_detail_cols if col not in df_detail.columns]
    if missing_cols:
        print(f"WARNING: Missing detail columns: {missing_cols}")
        
        # Check for similar column names
        print("Available detail columns:")
        for col in df_detail.columns:
            print(f"  {col}")
    
    # Convert percentage columns
    percentage_cols = ['Min Facteur', 'Max Facteur', 'Boost PDC', 'Poids du A/C max']
    for col in percentage_cols:
        if col in df_pdc_sim.columns:
            # Values are already in decimal form (0-4 for factors, 0-1 for percentages)
            df_pdc_sim[col] = pd.to_numeric(df_pdc_sim[col], errors='coerce')
    
    # Ensure floating point columns
    float_cols = ['Top 500', 'Top 3000', 'Autre', 'Moyenne', 
                 'Limite Basse Top 500', 'Limite Basse Top 3000', 'Limite Basse Autre',
                 'Limite Haute Top 500', 'Limite Haute Top 3000', 'Limite Haute Autre']
    
    for col in float_cols:
        if col in df_pdc_sim.columns:
            df_pdc_sim[col] = pd.to_numeric(df_pdc_sim[col], errors='coerce').astype(float)
    
    print(f"\nRunning VBA optimization on real data...")
    
    try:
        # Run VBA optimization
        df_optimized = vba_logic_fixed.run_vba_optimization_fixed(df_pdc_sim, df_detail)
        
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS vs EXCEL EXPECTED")
        print("="*80)
        
        for idx, row in df_optimized.iterrows():
            type_prod = row['Type de produits V2']
            print(f"\n{type_prod}:")
            print("-" * 40)
            
            print(f"  PYTHON RESULT:")
            print(f"    Top 500:  {row['Top 500']:.1%}")
            print(f"    Top 3000: {row['Top 3000']:.1%}")
            print(f"    Autre:    {row['Autre']:.1%}")
            print(f"    Moyenne:  {row['Moyenne']:.1%}")
            
            print(f"  EXCEL EXPECTED:")
            if type_prod == 'Sec Méca - A/B':
                print(f"    Top 500:  100.0% ✓")
                print(f"    Top 3000: 81.0%")
                print(f"    Autre:    0.0%")
            elif type_prod == 'Sec Méca - A/C':
                print(f"    Top 500:  100.0%")
                print(f"    Top 3000: 100.0%") 
                print(f"    Autre:    92.0%")
            elif type_prod == 'Sec Homogène - A/B':
                print(f"    Top 500:  100.0%")
                print(f"    Top 3000: 100.0%")
                print(f"    Autre:    100.0%")
            
            # Calculate match score
            if type_prod == 'Sec Méca - A/B':
                expected = [1.0, 0.81, 0.0]
                actual = [row['Top 500'], row['Top 3000'], row['Autre']]
                diff = sum(abs(a - e) for a, e in zip(actual, expected))
                print(f"  DIFFERENCE: {diff:.3f} (closer to 0 is better)")
        
        # Save results
        output_file = "real_data_optimization_results.csv"
        df_optimized.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Testing VBA Optimization with Real Excel Data")
    print("=" * 50)
    
    # First show the initial data vs expected
    print_initial_vs_expected()
    
    # Then run the test
    success = test_with_real_data()
    
    if success:
        print("\n✅ Real data test completed!")
    else:
        print("\n❌ Real data test failed!")

if __name__ == "__main__":
    main()