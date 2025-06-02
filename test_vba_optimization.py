#!/usr/bin/env python3
"""
Test script to use VBA-style optimization logic instead of optimisation_globale.py
"""
import pandas as pd
import numpy as np
import os

# Import existing modules
import Optimisation
import vba_logic_optimization

# Import other required modules
import DetailRao
import PrevB2C
import PrevPromo  
import PrevCasse
import PrevEncours
import Ts
import PrevFinal
import FacteurAppro
import TypeProduits
import MiniPubliFL
import EntrepotProduitBloque
import NbJoursCommandeMax
import CommandeFinale
import BorneMinMax
import MacroParam
import CalculAutresColonnes

def test_vba_optimization():
    """Test the VBA optimization logic"""
    print("=== Testing VBA Optimization Logic ===")
    
    # Load the data that would normally be processed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detail_source_path = os.path.join(script_dir, 'merged_predictions.csv')
    
    if not os.path.exists(detail_source_path):
        print(f"ERROR: {detail_source_path} not found! Please run main.py first.")
        return
        
    print("Loading detail data...")
    df_detail = pd.read_csv(detail_source_path, sep=';', encoding='latin1', low_memory=False)
    print(f"Loaded {len(df_detail)} detail rows")
    
    # Load PDC data 
    print("Loading PDC data...")
    import PDC as ModulePDC
    df_pdc_perm = ModulePDC.get_RAW_pdc_perm_data_for_optim()
    
    # Load En-cours data
    print("Loading En-cours data...")
    import Encours as ModuleEncours  
    df_encours = ModuleEncours.get_processed_data(formatted=False)
    
    # Generate initial PDC_Sim table using existing logic
    print("Generating initial PDC_Sim table...")
    df_pdc_sim_initial = Optimisation.create_full_pdc_sim_table(df_detail, df_pdc_perm, df_encours)
    
    if df_pdc_sim_initial.empty:
        print("ERROR: Initial PDC_Sim table is empty!")
        return
        
    print(f"Initial PDC_Sim table created with {len(df_pdc_sim_initial)} rows")
    print("\nInitial values (first 3 rows):")
    cols_to_show = ['Type de produits V2', 'Top 500', 'Top 3000', 'Autre', 'Min Facteur', 'Max Facteur']
    print(df_pdc_sim_initial[cols_to_show].head(3).to_string())
    
    # Apply VBA optimization logic
    print("\n=== Applying VBA Optimization Logic ===")
    df_pdc_sim_optimized = vba_logic_optimization.run_vba_optimization(
        df_pdc_sim_initial, df_detail, df_pdc_perm
    )
    
    print("\n=== VBA Optimization Results ===")
    print("Optimized values:")
    print(df_pdc_sim_optimized[cols_to_show].to_string())
    
    # Compare with Excel expected results
    print("\n=== Comparison with Excel Results ===")
    expected_results = {
        'sec méca - a/b': (1.00, 0.81, 0.00),
        'sec méca - a/c': (1.00, 1.00, 0.92), 
        'sec homogène - a/b': (1.00, 1.00, 1.00),
        'sec homogène - a/c': (1.95, 1.95, 1.95),
        'sec hétérogène - a/b': (1.06, 1.06, 1.06), 
        'sec hétérogène - a/c': (1.00, 1.00, 1.00),
        'frais méca': (1.00, 1.00, 1.00),
        'frais manuel': (1.00, 1.00, 0.54),
        'surgelés': (1.04, 1.04, 1.04)
    }
    
    for idx, row in df_pdc_sim_optimized.iterrows():
        type_v2_key = row['Type de produits V2'].lower().strip()
        if type_v2_key in expected_results:
            expected = expected_results[type_v2_key]
            actual = (row['Top 500'], row['Top 3000'], row['Autre'])
            print(f"{type_v2_key:20} | Expected: {expected} | Actual: {actual[0]:.2f}, {actual[1]:.2f}, {actual[2]:.2f}")
    
    # Save results
    output_path = 'PDC_Sim_VBA_Optimized.xlsx'
    print(f"\nSaving results to {output_path}...")
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_export = df_pdc_sim_optimized.copy()
            if 'Jour livraison' in df_export.columns:
                df_export['Jour livraison'] = pd.to_datetime(df_export['Jour livraison'], errors='coerce').dt.date
            df_export.to_excel(writer, sheet_name='PDC_Sim_VBA', index=False, header=True)
        print(f"Results saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    test_vba_optimization()