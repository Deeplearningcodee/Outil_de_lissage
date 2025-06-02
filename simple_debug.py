#!/usr/bin/env python3
"""Simple debug script to understand the BP column value"""

import pandas as pd
import numpy as np

def simple_debug():
    print("=== Simple Debug: Check BP Column Value ===")
    
    # Use the same data loading as the test scripts
    try:
        # Try different ways to load the data
        try:
            df_detail = pd.read_csv('merged_predictions.csv', sep=';', encoding='latin1', low_memory=False)
            print("Loaded with latin1 encoding and semicolon separator")
        except:
            df_detail = pd.read_csv('merged_predictions.csv', encoding='latin1', low_memory=False)
            print("Loaded with latin1 encoding")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Total rows in dataset: {len(df_detail)}")
    
    # Filter for CODE_METI 20684
    test_rows = df_detail[df_detail['CODE_METI'] == 20684].copy()
    print(f"Found {len(test_rows)} rows for CODE_METI 20684")
    
    if len(test_rows) == 0:
        print("No test rows found!")
        return
    
    # Check for the BP column
    bp_columns = [col for col in df_detail.columns if 'Commande Finale' in col and 'arrondi' in col]
    print(f"\nFound BP-related columns: {bp_columns}")
    
    # Also check for other key columns
    key_columns = ['CODE_METI', 'BM', 'Top', 'Borne Min Facteur', 'Borne Max Facteur']
    for col in key_columns:
        matching_cols = [c for c in df_detail.columns if col.lower() in c.lower()]
        print(f"Columns matching '{col}': {matching_cols}")
    
    # Show first row values
    print(f"\nFirst row data for CODE_METI 20684:")
    for col in test_rows.columns:
        val = test_rows[col].iloc[0]
        print(f"  {col}: {val}")
        if len(str(col)) > 100:  # Avoid very long output
            break

if __name__ == "__main__":
    simple_debug()
