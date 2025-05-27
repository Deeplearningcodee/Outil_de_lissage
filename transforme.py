import os
#transforme xlsx file to csv
import pandas as pd
import numpy as np
import glob

#file name is 'Ts.xlsx' make it to 'Taux_Service.csv'
def transform_xlsx_to_csv():
    # Chemin du fichier
    xlsx_file_path = os.path.join(os.path.dirname(__file__), 'Ts.xlsx')
    
    # Vérifier si le fichier existe
    if not os.path.exists(xlsx_file_path):
        print(f"Erreur: Fichier {xlsx_file_path} non trouvé")
        return
    
    # Charger le fichier Excel
    df = pd.read_excel(xlsx_file_path, index_col=None)
    
    # Normaliser les colonnes
    df.rename(columns={'CDBASE': 'CODE_METI'}, inplace=True)
    
    # Convertir en CSV
    csv_file_path = os.path.join(os.path.dirname(__file__), 'Taux_Service.csv')
    df.to_csv(csv_file_path, sep=';', index=False, encoding='latin1')
    
    print(f"Fichier converti et enregistré sous {csv_file_path}")


#main function
if __name__ == "__main__":
    transform_xlsx_to_csv()
    