#!/usr/bin/env python3
"""
Test script for the fixed VBA optimization implementation
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

import vba_logic_fixed
import MacroParam

def create_test_data():
    """Create test data that matches the Excel structure"""
    
    # Create PDC_Sim data matching Excel results
    pdc_sim_data = [
        {
            'Type de produits V2': 'Sec Méca - A/B',
            'Type de produits': 'Sec Méca',
            'Délai standard livraison (jours)': 2,
            'Jour livraison': '30/05/2025',
            'Min Facteur': '0%',
            'Max Facteur': '400%',
            'Boost PDC': '0%',
            'Poids du A/C max': '100%',
            'Top 500': 100.0,
            'Top 3000': 100.0,
            'Autre': 100.0,
            'Moyenne': 100.0,
            'Limite Basse Top 500': 1.0,
            'Limite Basse Top 3000': 1.0,
            'Limite Basse Autre': 1.0,
            'Limite Haute Top 500': 1.0,
            'Limite Haute Top 3000': 1.0,
            'Limite Haute Autre': 1.0,
            'PDC': 22673,
            'En-cours': 6812
        },
        {
            'Type de produits V2': 'Sec Méca - A/C',
            'Type de produits': 'Sec Méca',
            'Délai standard livraison (jours)': 3,
            'Jour livraison': '31/05/2025',
            'Min Facteur': '0%',
            'Max Facteur': '400%',
            'Boost PDC': '0%',
            'Poids du A/C max': '45%',
            'Top 500': 100.0,
            'Top 3000': 100.0,
            'Autre': 100.0,
            'Moyenne': 100.0,
            'Limite Basse Top 500': 1.0,
            'Limite Basse Top 3000': 1.0,
            'Limite Basse Autre': 1.0,
            'Limite Haute Top 500': 1.0,
            'Limite Haute Top 3000': 1.0,
            'Limite Haute Autre': 1.0,
            'PDC': 30619,
            'En-cours': 1609
        },
        {
            'Type de produits V2': 'Sec Homogène - A/B',
            'Type de produits': 'Sec Homogène',
            'Délai standard livraison (jours)': 2,
            'Jour livraison': '30/05/2025',
            'Min Facteur': '0%',
            'Max Facteur': '400%',
            'Boost PDC': '0%',
            'Poids du A/C max': '100%',
            'Top 500': 100.0,
            'Top 3000': 100.0,
            'Autre': 100.0,
            'Moyenne': 100.0,
            'Limite Basse Top 500': 1.0,
            'Limite Basse Top 3000': 1.0,
            'Limite Basse Autre': 1.0,
            'Limite Haute Top 500': 1.0,
            'Limite Haute Top 3000': 1.0,
            'Limite Haute Autre': 1.0,
            'PDC': 283,
            'En-cours': 96
        }
    ]
    
    df_pdc_sim = pd.DataFrame(pdc_sim_data)
    
    # Convert percentage strings to floats
    percentage_cols = ['Min Facteur', 'Max Facteur', 'Boost PDC', 'Poids du A/C max']
    for col in percentage_cols:
        df_pdc_sim[col] = df_pdc_sim[col].str.replace('%', '').astype(float) / 100
    
    # Convert date
    df_pdc_sim['Jour livraison'] = pd.to_datetime(df_pdc_sim['Jour livraison'], dayfirst=True)
    
    # Create sample detail data that would produce realistic commands
    detail_data = []
    
    # Define realistic command values that should sum close to the expected results
    expected_commands = {
        'Sec Méca - A/B': 15282,    # Expected from Excel: PDC=22673, En-cours=6812, so Command≈15861
        'Sec Méca - A/C': 29704,    # Expected from Excel: PDC=30619, En-cours=1609, so Command≈29010  
        'Sec Homogène - A/B': 272   # Expected from Excel: PDC=283, En-cours=96, so Command≈187
    }
    
    for _, row in df_pdc_sim.iterrows():
        type_prod = row['Type de produits V2']
        expected_cmd = expected_commands[type_prod]
        
        # Create detail rows that sum to approximately the expected command
        num_products = 10
        avg_cmd_per_product = expected_cmd / num_products
        
        for i in range(num_products):
            # Distribute products across Top categories
            if i < 3:
                top_category = 'top 500'
            elif i < 6:
                top_category = 'top 3000'
            else:
                top_category = 'autre'
            
            # Calculate SM and Prev values that would result in the target command
            base_cmd = avg_cmd_per_product * np.random.uniform(0.5, 1.5)
            sm_final = base_cmd * 0.7  # SM contributes ~70%
            prev_c1_l2 = base_cmd * 0.2  # Prev C1-L2 contributes ~20%
            prev_l1_l2 = base_cmd * 0.1  # Prev L1-L2 contributes ~10%
            
            detail_data.append({
                'Type de produits V2': type_prod,
                'DATE_LIVRAISON_V2': row['Jour livraison'],
                'Top': top_category,
                'Mini Publication FL': base_cmd * 0.8,  # Mini pub slightly less than command
                'COCHE_RAO': 0,
                'RAL': 0,
                'STOCK_REEL': base_cmd * 0.1,  # Small stock
                'SM Final': sm_final,
                'Prév C1-L2 Finale': prev_c1_l2,
                'Prév L1-L2 Finale': prev_l1_l2,
                'Facteur Multiplicatif Appro': 1.0,
                'Casse Prev C1-L2': 0,
                'Casse Prev L1-L2': 0,
                'Produit Bloqué': False,
                'Commande Max avec stock max': 999999,
                'PCB': 1.0,
                'MINIMUM_COMMANDE': 0,
                'TS': 1.0,
                'Position JUD': 1,
                'Borne Min Facteur multiplicatif lissage': 0.0,
                'Borne Max Facteur multiplicatif lissage': 10.0,
                'BM': type_prod,
                'Commande Finale avec mini et arrondi SM à 100%': base_cmd,
                'Type de produits V2': type_prod
            })
    
    df_detail = pd.DataFrame(detail_data)
    
    # Create empty df_pdc_perm (not used in this test)
    df_pdc_perm = pd.DataFrame()
    
    return df_pdc_sim, df_detail, df_pdc_perm

def print_comparison_results(df_original, df_optimized):
    """Print comparison between original and optimized results"""
    
    print("\n" + "="*80)
    print("VBA OPTIMIZATION RESULTS COMPARISON")
    print("="*80)
    
    comparison_cols = ['Type de produits V2', 'Top 500', 'Top 3000', 'Autre', 'Moyenne', 
                      'Limite Basse Top 500', 'Limite Basse Top 3000', 'Limite Basse Autre',
                      'Limite Haute Top 500', 'Limite Haute Top 3000', 'Limite Haute Autre']
    
    for idx, row in df_optimized.iterrows():
        type_prod = row['Type de produits V2']
        print(f"\n{type_prod}:")
        print("-" * 40)
        
        # Show optimization parameters
        print(f"  Optimization Factors:")
        print(f"    Top 500:  {row['Top 500']:.1%}")
        print(f"    Top 3000: {row['Top 3000']:.1%}")  
        print(f"    Autre:    {row['Autre']:.1%}")
        print(f"    Moyenne:  {row['Moyenne']:.1%}")
        
        # Show limits
        print(f"  Limite Basse: Top500={row['Limite Basse Top 500']:.1f}, Top3000={row['Limite Basse Top 3000']:.1f}, Autre={row['Limite Basse Autre']:.1f}")
        print(f"  Limite Haute: Top500={row['Limite Haute Top 500']:.1f}, Top3000={row['Limite Haute Top 3000']:.1f}, Autre={row['Limite Haute Autre']:.1f}")
        
        # Show expected vs actual
        if type_prod == 'Sec Méca - A/B':
            print(f"  Expected (Excel): Top500=100%, Top3000=81%, Autre=0%")
        elif type_prod == 'Sec Méca - A/C':
            print(f"  Expected (Excel): Top500=100%, Top3000=100%, Autre=92%")
        elif type_prod == 'Sec Homogène - A/B':
            print(f"  Expected (Excel): Top500=100%, Top3000=100%, Autre=100%")

def main():
    """Main test function"""
    print("Testing Fixed VBA Optimization Implementation")
    print("=" * 50)
    
    # Create test data
    print("Creating test data...")
    df_pdc_sim, df_detail, df_pdc_perm = create_test_data()
    
    print(f"PDC_Sim rows: {len(df_pdc_sim)}")
    print(f"Detail rows: {len(df_detail)}")
    
    # Run the VBA optimization
    print("\nRunning VBA optimization...")
    try:
        df_optimized = vba_logic_fixed.run_vba_optimization_fixed(df_pdc_sim, df_detail)
        
        # Print results
        print_comparison_results(df_pdc_sim, df_optimized)
        
        # Save results for inspection
        output_file = "vba_optimization_test_results.csv"
        df_optimized.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")