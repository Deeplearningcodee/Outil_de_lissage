import pandas as pd
import os
import numpy as np

# Chemins vers les fichiers source
FACTEUR_FAM_CSV = os.path.join('Facteur_Appro', 'FacteurAppro.csv')
FACTEUR_SF_CSV = os.path.join('Facteur_Appro', 'FacteurApproSF.csv')
FACTEUR_ART_CSV = os.path.join('Facteur_Appro', 'FacteurApproArt.csv')

def load_facteur_famille():
    """
    Charge les facteurs d'approvisionnement par Famille
    """
    try:
        df = pd.read_csv(
            FACTEUR_FAM_CSV,
            sep=';',
            encoding='latin1',
            engine='python'
        )
        
        # Renommer les colonnes pour correspondre aux formules Excel
        df = df.rename(columns={
            'N° Famille': 'FAMILLE_HYPER',
            'Facteur': 'Facteur Multiplicatif'
        })
        
        # Convertir FAMILLE_HYPER en string pour le merge
        df['FAMILLE_HYPER'] = df['FAMILLE_HYPER'].astype(str)
        
        # Convertir le facteur en valeur numérique (sans le %)
        df['Facteur Multiplicatif'] = df['Facteur Multiplicatif'].astype(str).str.rstrip('%').replace('', '100').astype(float) / 100
        
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {FACTEUR_FAM_CSV}: {e}")
        return pd.DataFrame(columns=['FAMILLE_HYPER', 'Facteur Multiplicatif'])

def load_facteur_sous_famille():
    """
    Charge les facteurs d'approvisionnement par Sous-Famille
    """
    try:
        df = pd.read_csv(
            FACTEUR_SF_CSV,
            sep=';',
            encoding='latin1',
            engine='python'
        )
        
        # Renommer les colonnes pour correspondre aux formules Excel
        df = df.rename(columns={
            'N° Sous-Famille': 'SS_FAMILLE_HYPER',
            'Facteur': 'Facteur Multiplicatif'
        })
        
        # Convertir SS_FAMILLE_HYPER en string pour le merge
        df['SS_FAMILLE_HYPER'] = df['SS_FAMILLE_HYPER'].astype(str)
        
        # Convertir le facteur en valeur numérique (sans le %)
        df['Facteur Multiplicatif'] = df['Facteur Multiplicatif'].astype(str).str.rstrip('%').replace('', '100').astype(float) / 100
        
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {FACTEUR_SF_CSV}: {e}")
        return pd.DataFrame(columns=['SS_FAMILLE_HYPER', 'Facteur Multiplicatif'])

def load_facteur_article():
    """
    Charge les facteurs d'approvisionnement par Article (EAN)
    """
    try:
        df = pd.read_csv(
            FACTEUR_ART_CSV,
            sep=';',
            encoding='latin1',
            engine='python'
        )
        
        # Renommer les colonnes pour correspondre aux formules Excel
        df = df.rename(columns={
            'EAN': 'Ean_13',
            'Coefficient': 'Facteur Multiplicatif'
        })
        
        # Convertir Ean_13 en string pour le merge
        df['Ean_13'] = df['Ean_13'].astype(str)
        
        # Convertir le facteur en valeur numérique (sans le %)
        df['Facteur Multiplicatif'] = df['Facteur Multiplicatif'].astype(str).str.rstrip('%').replace('', '100').astype(float) / 100
        
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {FACTEUR_ART_CSV}: {e}")
        return pd.DataFrame(columns=['Ean_13', 'Facteur Multiplicatif'])

def apply_facteur_appro(df):
    """
    Applique les facteurs d'approvisionnement pour chaque niveau:
    - Famille (FAMILLE_HYPER)
    - Sous-Famille (SS_FAMILLE_HYPER)
    - Article (Ean_13)
    
    Formule Facteur Multiplicatif Appro Famille:
    =SI(AT2>0;1;SIERREUR(INDEX('Facteur Appro'!G:G;EQUIV(A2;'Facteur Appro'!E:E;0));1))
    
    Formule Facteur Multiplicatif Appro Sous-Famille:
    =SI(AT2>0;1;SIERREUR(INDEX('Facteur Appro SF'!G:G;EQUIV(C2;'Facteur Appro SF'!E:E;0));1))
    
    Formule Facteur Multiplicatif Appro Article:
    =SI(AT2>0;1;SIERREUR(INDEX('Facteur Appro Art'!C:C;EQUIV(E2;'Facteur Appro Art'!A:A;0));""))
    
    AT2 représente la colonne 'Casse Prev C1-L2'
    """
    # Vérifier que la colonne 'Casse Prev C1-L2' existe
    if 'Casse Prev C1-L2' not in df.columns:
        print("Colonne 'Casse Prev C1-L2' manquante, initialisation à 0")
        df['Casse Prev C1-L2'] = 0
    
    # Charger les données des facteurs
    df_fam = load_facteur_famille()
    df_sf = load_facteur_sous_famille()
    df_art = load_facteur_article()
    
    # 1. Facteur Multiplicatif Appro Famille
    # =SI(AT2>0;1;SIERREUR(INDEX('Facteur Appro'!G:G;EQUIV(A2;'Facteur Appro'!E:E;0));1))
    def get_facteur_famille(row):
        # Si Casse Prev C1-L2 > 0, retourner 1
        if pd.notna(row['Casse Prev C1-L2']) and row['Casse Prev C1-L2'] > 0:
            return 1.0
        
        # Sinon, chercher le facteur dans le DataFrame df_fam
        if 'FAMILLE_HYPER' in row and pd.notna(row['FAMILLE_HYPER']):
            famille = str(row['FAMILLE_HYPER'])
            match = df_fam[df_fam['FAMILLE_HYPER'] == famille]
            if not match.empty:
                return match['Facteur Multiplicatif'].values[0]
        
        # Par défaut, retourner 1
        return 1.0
    
    # 2. Facteur Multiplicatif Appro Sous-Famille
    # =SI(AT2>0;1;SIERREUR(INDEX('Facteur Appro SF'!G:G;EQUIV(C2;'Facteur Appro SF'!E:E;0));1))
    def get_facteur_sous_famille(row):
        # Si Casse Prev C1-L2 > 0, retourner 1
        if pd.notna(row['Casse Prev C1-L2']) and row['Casse Prev C1-L2'] > 0:
            return 1.0
        
        # Sinon, chercher le facteur dans le DataFrame df_sf
        if 'SS_FAMILLE_HYPER' in row and pd.notna(row['SS_FAMILLE_HYPER']):
            sous_famille = str(row['SS_FAMILLE_HYPER'])
            match = df_sf[df_sf['SS_FAMILLE_HYPER'] == sous_famille]
            if not match.empty:
                return match['Facteur Multiplicatif'].values[0]
        
        # Par défaut, retourner 1
        return 1.0
    
    # 3. Facteur Multiplicatif Appro Article
    # =SI(AT2>0;1;SIERREUR(INDEX('Facteur Appro Art'!C:C;EQUIV(E2;'Facteur Appro Art'!A:A;0));""))
    def get_facteur_article(row):
        # Si Casse Prev C1-L2 > 0, retourner 1
        if pd.notna(row['Casse Prev C1-L2']) and row['Casse Prev C1-L2'] > 0:
            return 1.0
        
        # Sinon, chercher le facteur dans le DataFrame df_art
        if 'Ean_13' in row and pd.notna(row['Ean_13']):
            ean = str(row['Ean_13'])
            match = df_art[df_art['Ean_13'] == ean]
            if not match.empty:
                return match['Facteur Multiplicatif'].values[0]
        
        # Par défaut, retourner chaîne vide (comme dans la formule Excel)
        return 1.0
    
    # Appliquer les fonctions pour calculer les facteurs
    df['Facteur Multiplicatif Appro Famille'] = df.apply(get_facteur_famille, axis=1)
    df['Facteur Multiplicatif Appro Sous-Famille'] = df.apply(get_facteur_sous_famille, axis=1)
    df['Facteur Multiplicatif Appro Article'] = df.apply(get_facteur_article, axis=1)
    
    # Calculer le facteur multiplicatif global
    # On prend le produit des trois facteurs, en remplaçant les chaînes vides par 1
    df['Facteur Multiplicatif Appro'] = df['Facteur Multiplicatif Appro Famille'] * \
                                       df['Facteur Multiplicatif Appro Sous-Famille'] * \
                                       df['Facteur Multiplicatif Appro Article']
    
    return df

def get_processed_data(merged_df=None):
    """
    Fonction principale appelée par main.py
    Calcule les colonnes de facteur d'approvisionnement
    """
    if merged_df is None:
        print("Erreur: DataFrame merged_df non fourni")
        return pd.DataFrame()
    
    # Appliquer les facteurs d'approvisionnement
    result_df = apply_facteur_appro(merged_df)
    
    return result_df

if __name__ == "__main__":
    # Test du module
    print("Test du module FacteurAppro")
    
    # Créer un DataFrame de test
    test_df = pd.DataFrame({
        'FAMILLE_HYPER': ['1000', '1010'],
        'SS_FAMILLE_HYPER': ['10000', '10100'],
        'Ean_13': ['8410076470812', '1234567890123'],
        'Casse Prev C1-L2': [0, 10]
    })
    
    # Appliquer les facteurs
    result = get_processed_data(test_df)
    
    # Afficher les résultats
    print(result[['FAMILLE_HYPER', 'SS_FAMILLE_HYPER', 'Ean_13', 'Casse Prev C1-L2',
                 'Facteur Multiplicatif Appro Famille',
                 'Facteur Multiplicatif Appro Sous-Famille',
                 'Facteur Multiplicatif Appro Article',
                 'Facteur Multiplicatif Appro']])
