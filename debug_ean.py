#!/usr/bin/env python3
import StockMarchand
import pandas as pd

# Load StockMarchand data
print("Loading StockMarchand data...")
df = StockMarchand.get_stock_marchand_data()

print("Sample EAN values from StockMarchand:")
print(df['Ean_13'].head(10).tolist())
print(f"EAN data type: {df['Ean_13'].dtype}")

# Check conversion
print("\nConverting EAN to match TS format:")
converted = pd.to_numeric(df['Ean_13'], errors='coerce').fillna(0).astype(int).astype(str)
print("Converted EAN values:")
print(converted.head(10).tolist())

# Load TS data for comparison
print("\nLoading TS data...")
df_ts = pd.read_csv('CSV/Taux_Service.csv', sep=';', encoding='latin1', dtype={'EAN': str})
print("Sample EAN values from TS CSV:")
print(df_ts['EAN'].head(10).tolist())
print(f"TS EAN data type: {df_ts['EAN'].dtype}")

# Check for matches
print("\nChecking for matches...")
stock_eans = set(converted.tolist())
ts_eans = set(df_ts['EAN'].tolist())
matches = stock_eans.intersection(ts_eans)
print(f"Number of matching EANs: {len(matches)}")
print(f"Total StockMarchand EANs: {len(stock_eans)}")
print(f"Total TS EANs: {len(ts_eans)}")

# Show some matches
if matches:
    print("\nSample matching EANs:")
    for ean in list(matches)[:10]:
        print(f"  {ean}")
