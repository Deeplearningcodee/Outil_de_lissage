import pandas as pd
import traceback

print("=== DÉBUT DE LA VÉRIFICATION ===")

# Check what sheets are available in the Excel file
try:
    print("Vérification des feuilles disponibles...")
    xl_file = pd.ExcelFile('PDC_Sim_Optimized_Python_VBA_Aligned.xlsx')
    print(f"Feuilles disponibles: {xl_file.sheet_names}")
    
    # Try to read the first sheet
    sheet_name = xl_file.sheet_names[0]
    print(f"Lecture de la feuille: {sheet_name}")
    
    df = pd.read_excel('PDC_Sim_Optimized_Python_VBA_Aligned.xlsx', sheet_name=sheet_name)
    print("✓ Fichier lu avec succès!")
    print(f"Nombre de lignes: {len(df)}")
    print(f"Nombre de colonnes: {len(df.columns)}")
    print()
    
    print("Colonnes disponibles:")
    for i, col in enumerate(df.columns):
        print(f"  {i}: {col}")
    print()
    
    print("=== VERIFICATION: Top 500, Top 3000, Autre (doivent maintenant montrer les valeurs optimisées) ===")
    for i, row in df.iterrows():
        produit = row['Type de produits V2']
        top500 = row['Top 500']
        top3000 = row['Top 3000'] 
        autre = row['Autre']
        print(f"Ligne {i}: {produit} -> Top500={top500:.3f}, Top3000={top3000:.3f}, Autre={autre:.3f}")
    
    print()
    print("=== VERIFICATION: Poids du A/C max (doit être recalculé dynamiquement pour A/C) ===")
    for i, row in df.iterrows():
        produit = row['Type de produits V2']
        poids_ac = row['Poids du A/C max']
        if 'a/c' in produit.lower():
            print(f"Ligne {i}: {produit} -> Poids du A/C max = {poids_ac:.3f} (doit être recalculé)")
        else:
            print(f"Ligne {i}: {produit} -> Poids du A/C max = {poids_ac:.3f} (reste fixe)")
    
    print()
    print("=== VERIFICATION: Colonnes de calcul additionnelles ===")
    for col in ['Différence PDC / Commande', 'Variation PDC', 'Capage Borne Max ?']:
        if col in df.columns:
            print(f"\n{col}:")
            for i, row in df.iterrows():
                produit = row['Type de produits V2']
                val = row[col] if not pd.isna(row[col]) else "NaN"
                print(f"  Ligne {i}: {produit} -> {val}")
        else:
            print(f"\nCOLONNE MANQUANTE: {col}")
    
except Exception as e:
    print(f"Erreur lors de la lecture du fichier: {e}")
    print("Traceback complet:")
    traceback.print_exc()
