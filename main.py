import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import the other modules
import DetailRao
import PrevB2C
import PrevPromo
import PrevCasse
import PrevEncours
import Ts
import PrevFinal
import FacteurAppro
import TypeProduits
import MiniPubliFL  # Ajout du module MiniPubliFL
import EntrepotProduitBloque  # Ajout du module pour gérer les colonnes Entrepôt/Produit Bloqué
import NbJoursCommandeMax  # Ajout du module pour gérer la colonne Nb Jours Commande Max

import CommandeFinale
import BorneMinMax
import MacroParam
import CalculAutresColonnes
import Exclusion

# Define reference date
reference_date = MacroParam.DATE_COMMANDE
# Calculate 3 weeks earlier
three_weeks_ago = reference_date - pd.Timedelta(days=21)

# 1. Load stock merchant data
print("Loading stock merchant data...")
# Méthode précédente
# stock_df = pd.read_csv(
#     'stkmarchand_CES880.csv',
#     sep=';',
#     header=None,
#     names=['CODE_METI', 'SM_final']
# )

# Nouvelle méthode: importer depuis StockMarchand.py

import StockMarchand
stock_marchand_df = StockMarchand.get_stock_marchand_data()

# importer les Exclusions 

exclusion_df = Exclusion.copy_exclusion_sheet()

# Si le chargement du stock marchand a échoué, revenir à l'ancienne méthode
# if stock_marchand_df is None or stock_marchand_df.empty:
#     print("Fallback: Chargement des données de stock marchand depuis stkmarchand_CES880.csv")
#     stock_df = pd.read_csv(
#         'stkmarchand_CES880.csv',
#         sep=';',
#         header=None,
#         names=['CODE_METI', 'SM_final']
#     )
# else:
#     print(f"Stock marchand chargé avec succès: {len(stock_marchand_df)} lignes")
#     # Créer un DataFrame compatible avec le reste du code
#     stock_df = pd.DataFrame({
#         'CODE_METI': stock_marchand_df['CODE_METI'],
#         'SM_final': stock_marchand_df['SM Final']
#     })

# 2. Get data from each module in the desired order
print("Processing Detail RAO data...")
rao_df = DetailRao.get_processed_data()

print("Loading Mini Publication FL data...")
mini_publi_df = MiniPubliFL.get_processed_data()

print("Processing B2C predictions...")
b2c_df = PrevB2C.get_processed_data()

print("Processing Promo predictions...")
promo_df = PrevPromo.get_processed_data()

print("Processing Casse predictions...")
casse_df = PrevCasse.get_processed_data()

print("Processing Encours predictions...")
encours_df = PrevEncours.get_processed_data()


print("Processing TS values...")
ts_df = Ts.get_processed_data()

# print("Loading EAN mapping data...")
# ean_df = EanMapping.get_processed_data()

# 3. Merge all dataframes by CODE_METI
print("Merging all data...")
# Start with the RAO detail data as the base
# if rao_df is not None and not rao_df.empty:
#     merged_df = rao_df.copy()
# else:
#     merged_df = stock_df.copy()

merged_df = rao_df.copy() 

# Ensure CODE_METI is the same type in all dataframes (convert to string)
merged_df['CODE_METI'] = merged_df['CODE_METI'].astype(str)

# Ajouter les données Mini Publication FL juste après le détail RAO
if mini_publi_df is not None and not mini_publi_df.empty:
    mini_publi_df['CODE_METI'] = mini_publi_df['CODE_METI'].astype(str)
    merged_df = pd.merge(merged_df, mini_publi_df, on='CODE_METI', how='left')
    # Remplir les valeurs manquantes avec -1 (équivalent à SIERREUR(...;-1))
    merged_df['Mini Publication FL'] = merged_df['Mini Publication FL'].fillna(-1)

# Merge with each prediction dataframe in the specified order
dfs_to_merge = [b2c_df, promo_df, casse_df, encours_df, ts_df]
for df in dfs_to_merge:
    if df is not None and not df.empty:
        # Convert CODE_METI to string before merging
        df['CODE_METI'] = df['CODE_METI'].astype(str)
        merged_df = pd.merge(merged_df, df, on='CODE_METI', how='left')

# 4. Filter for dates within 3 weeks of the reference date
# Note: We've already filtered the data in each module but we can double check
date_cols = [col for col in merged_df.columns if 'date' in col.lower() or 'DATE' in col]
for col in date_cols:
    if merged_df[col].dtype == 'datetime64[ns]':
        merged_df = merged_df[merged_df[col] >= three_weeks_ago]

# 4.5 Calculate final predictions ("Prév C1-L2 Finale" and "Prév L1-L2 Finale")
print("Calculating final predictions...")
merged_df = PrevFinal.get_processed_data(merged_df)

# NOTE: Call to FacteurAppro moved to after stock_marchand_df merge (see step 4.76.1)
# # 4.6 Calculate supply multiplier factors
# print("Calculating supply multiplier factors...")
# merged_df = FacteurAppro.get_processed_data(merged_df) 

# 4.7 Calculate product types and top categories
print("Calculating product types and top categories...")
merged_df = TypeProduits.get_processed_data(merged_df) 



# 4.75 Add Entrepôt Bloqué and Produit Bloqué columns
print("Adding Entrepôt Bloqué and Produit Bloqué columns...")
merged_df = EntrepotProduitBloque.get_processed_data(merged_df)

# 4.76 Add VMJ Utilisée Stock Sécurité and Période de vente finale (Jours) columns and other stock data
print("Adding VMJ Utilisée Stock Sécurité, Période de vente finale (Jours) and other stock data...")    # Si on a réussi à charger le stock marchand complet, on l'utilise
if 'stock_marchand_df' in locals() and not stock_marchand_df.empty:
    print("Fusion avec les données complètes du stock marchand...")
    
    # Sauvegarde du comptage initial
    initial_record_count = len(merged_df)
    sm_record_count = len(stock_marchand_df)
    
    # Conversion des types pour la fusion
    stock_marchand_df['CODE_METI'] = stock_marchand_df['CODE_METI'].astype(str)
    merged_df['CODE_METI'] = merged_df['CODE_METI'].astype(str)
    
    # Au lieu d'un INNER JOIN, on va faire une approche à deux étapes:
    # 1. Conserver l'ancien dataframe pour les données RAO/prévisions
    old_merged_df = merged_df.copy()
    
    # 2. Partir du StockMarchand comme base et enrichir avec les données de old_merged_df
    merged_df = stock_marchand_df.copy()
    merged_df = pd.merge(merged_df, old_merged_df, on='CODE_METI', how='left')
      # Afficher les comptages
    print(f"Utilisation des {sm_record_count} enregistrements du stock marchand comme base de données")
    print(f"Enrichissement avec les données des {initial_record_count} enregistrements RAO/prévisions")
    
    # Liste des colonnes à vérifier et remplir si manquantes
    stock_columns = ['FAMILLE_HYPER', 'Libellé_Famille', 'SS_FAMILLE_HYPER', 'Libellé_Sous_Famille',
                    'Macro-catégorie', 'Classe de stockage', 'Entrepôt', 'Période de vente finale (Jours)',
                    'VMJ Utilisée Stock Sécurité', 'Stock Max', 'SM Final', 'Nb de commande / Semaine Final']
    
    for col in stock_columns:
        if col in merged_df.columns:
            if col in ['Période de vente finale (Jours)', 'VMJ Utilisée Stock Sécurité', 'Stock Max', 'SM Final', 'Nb de commande / Semaine Final']:
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)
            else:
                merged_df[col] = merged_df[col].fillna('')

# 4.76.1 Calculate supply multiplier factors (MOVED HERE)
print("Calculating supply multiplier factors...")
merged_df = FacteurAppro.get_processed_data(merged_df) 

# 4.77 Add Nb Jours Commande Max column
print("Adding Nb Jours Commande Max column...")
merged_df = NbJoursCommandeMax.get_processed_data(merged_df)


# 4.8 Calculate Commande Finale sans mini ni arrondi SM à 100%
print("Calcul des commandes finales SM à 100% (BO, BP, BQ)...")
merged_df = CommandeFinale.get_processed_data(merged_df, step="initial")

# 4.9 Ajouter la colonne Exclusion Lissage
print("Calcul initial des exclusions et bornes min/max pour le lissage (BW, BX, BY initialisé à 1)...")
merged_df = BorneMinMax.calculate_initial_exclusions_and_bornes(merged_df)


#save the initial merged dataframe to a CSV file
print("Saving initial merged data to CSV file...")
output_initial_file = 'initial_merged_predictions.csv'
merged_df.to_csv(output_initial_file, sep=';', index=False, encoding='latin1')


import Optimisation
Optimisation.generate_complete_pdc_sim_excel(
    output_excel_path='PDC_Sim_Input_For_Optim.xlsx', 
    df_detail_source_arg=merged_df # merged_df contient maintenant BQ et CF

)
# --- VBA-STYLE OPTIMIZATION LOGIC ---
# Replace optimisation_globale.py with VBA logic implementation
print("Running VBA-style optimization logic...")

# Import corrected VBA logic
import vba_logic_fixed

# Load the PDC_Sim input data that was just generated
df_pdc_sim_input = pd.read_excel('PDC_Sim_Input_For_Optim.xlsx')
print(f"Loaded PDC_Sim input with {len(df_pdc_sim_input)} rows for optimization")

# Run corrected VBA optimization using the processed merged_df as detail data
df_pdc_sim_optimized = vba_logic_fixed.run_vba_optimization_fixed(
    df_pdc_sim_input, merged_df
)

# Save optimized results to the expected filename
print("Saving VBA optimization results to PDC_Sim_Optimized_Python.xlsx...")
try:
    with pd.ExcelWriter('PDC_Sim_Optimized_Python.xlsx', engine='openpyxl') as writer:
        df_export_optim = df_pdc_sim_optimized.copy()
        if 'Jour livraison' in df_export_optim.columns:
            df_export_optim['Jour livraison'] = pd.to_datetime(df_export_optim['Jour livraison'], errors='coerce').dt.date
        df_export_optim.to_excel(writer, sheet_name='PDC_Sim', index=False, header=True)
        
        # Format date column
        worksheet = writer.sheets['PDC_Sim']
        if 'Jour livraison' in df_export_optim.columns:
            date_col_idx = df_export_optim.columns.get_loc("Jour livraison") + 1
            for row_idx in range(2, worksheet.max_row + 1):
                cell = worksheet.cell(row=row_idx, column=date_col_idx)
                if hasattr(cell.value, 'strftime'):
                    cell.number_format = 'DD/MM/YYYY'
    print("VBA optimization results saved successfully!")
except Exception as e:
    print(f"Error saving optimization results: {e}")
    import traceback
    traceback.print_exc() 



# --- APRÈS que PDC_Sim_Optimized_Python.xlsx a été généré ---
# Étape 5: Mise à jour de BY avec les facteurs optimisés
print("Mise à jour du Facteur multiplicatif Lissage besoin brut (BY) avec les résultats optimisés...")
merged_df = BorneMinMax.update_facteur_lissage_besoin_brut_from_optim(merged_df)

# Étape 6: Calcul des commandes optimisées (CD, CE, CF) qui utilisent le BY mis à jour
print("Calcul des commandes optimisées (CD, CE, CF) utilisant le BY mis à jour...")
merged_df = CommandeFinale.get_processed_data(merged_df, step="optimise")

# Étape 7: Calcul des autres colonnes

merged_df = CalculAutresColonnes.get_processed_data(merged_df)

# 5. Save the merged file
output_file = 'merged_predictions.csv'
print(f"Saving merged data to {output_file}...")
merged_df.to_csv(output_file, sep=';', index=False, encoding='latin1')

print("Processing completed!")
print(f"Total records: {len(merged_df)}")
print(f"Columns in merged file: {merged_df.columns.tolist()}")

# Display a sample of the merged data
print("\nSample of merged data:")
print(merged_df.head(5))



