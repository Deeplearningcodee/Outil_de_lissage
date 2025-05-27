import pandas as pd
import numpy as np
from MacroParam import get_param_value

def add_bloque_columns(df):
    """
    Ajoute les colonnes 'Entrepôt Bloqué' et 'Produit Bloqué'.
    
    - 'Entrepôt Bloqué' est défini par le paramètre dans MacroParam (False par défaut)
    - 'Produit Bloqué' est calculé selon la formule =OU(BH2;AD2=46)
      où BH2 est 'Entrepôt Bloqué' et AD2 est 'COCHE_RAO'
    
    Args:
        df: DataFrame contenant la colonne 'COCHE_RAO'
    
    Returns:
        DataFrame avec les colonnes 'Entrepôt Bloqué' et 'Produit Bloqué' ajoutées
    """
    # Récupérer la valeur du paramètre 'Entrepôt Bloqué' depuis MacroParam
    entrepot_bloque = get_param_value('Entrepôt Bloqué', False)
    
    # Vérifier que la colonne nécessaire existe
    if 'COCHE_RAO' not in df.columns:
        print("Avertissement: La colonne 'COCHE_RAO' est manquante. Impossible de calculer 'Produit Bloqué'")
        df['Entrepôt Bloqué'] = entrepot_bloque
        df['Produit Bloqué'] = entrepot_bloque  # Par défaut, produit bloqué = entrepôt bloqué si pas de 'COCHE_RAO'
        return df
    
    # Ajouter la colonne 'Entrepôt Bloqué' avec la valeur du paramètre
    df['Entrepôt Bloqué'] = entrepot_bloque
    
    # Calculer 'Produit Bloqué' selon la formule =OU(BH2;AD2=46)
    def calc_produit_bloque(row):
        # Convertir COCHE_RAO en numérique pour la comparaison
        coche_rao = pd.to_numeric(row.get('COCHE_RAO', 0), errors='coerce')
        if pd.isna(coche_rao):
            coche_rao = 0
            
        # Appliquer la formule =OU('Entrepôt Bloqué';'COCHE_RAO'=46)
        return bool(row['Entrepôt Bloqué'] or coche_rao == 46)
    
    # Ajouter la colonne 'Produit Bloqué'
    df['Produit Bloqué'] = df.apply(calc_produit_bloque, axis=1)
    
    return df

def get_processed_data(df=None):
    """
    Fonction principale pour ajouter les colonnes 'Entrepôt Bloqué' et 'Produit Bloqué'
    
    Args:
        df: DataFrame à traiter. Si None, un DataFrame vide est créé.
    
    Returns:
        DataFrame avec les colonnes 'Entrepôt Bloqué' et 'Produit Bloqué' ajoutées
    """
    if df is None:
        # Si aucun DataFrame n'est fourni, on retourne un DataFrame avec juste les colonnes ajoutées
        return pd.DataFrame({
            'Entrepôt Bloqué': [get_param_value('Entrepôt Bloqué', False)],
            'Produit Bloqué': [get_param_value('Entrepôt Bloqué', False)]
        })
    
    # Ajouter les colonnes au DataFrame fourni
    return add_bloque_columns(df)

if __name__ == "__main__":
    # Test du module
    test_df = pd.DataFrame({
        'CODE_METI': ['123456', '789012', '345678'],
        'COCHE_RAO': [0, 46, 12]
    })
    
    result_df = get_processed_data(test_df)
    print(result_df[['CODE_METI', 'COCHE_RAO', 'Entrepôt Bloqué', 'Produit Bloqué']])
