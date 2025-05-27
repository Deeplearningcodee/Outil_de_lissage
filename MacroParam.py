import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

DATE_COMMANDE_STR = "27/05/2025" 
DATE_COMMANDE = pd.to_datetime(DATE_COMMANDE_STR, dayfirst=True)

DATE_REF_JOUR_AB_STR = "30/05/2025" 
DATE_REF_JOUR_AC_STR = "31/05/2025" 

DATE_REF_JOUR_AB = pd.to_datetime(DATE_REF_JOUR_AB_STR, dayfirst=True)
DATE_REF_JOUR_AC = pd.to_datetime(DATE_REF_JOUR_AC_STR, dayfirst=True)

ALERTE_SURCHARGE_NIVEAU_1 = 0.05  # 5%
MARGE_POUR_BOOST_ET_L_VAR_SOLVER = 0.00  
MARGE_I_POUR_SOLVER_CONDITION = 300.0  

TAUX_SERVICE_AMONT_ESTIME = 0.92  # 92%

GESTION_AB_AC_MAPPING = {
    "Sec Méca": "Oui",
    "Sec Homogène": "Oui",
    "Sec Hétérogène": "Oui", # Les 3 Sec Hétérogène ont "Oui"
    "Frais Méca": "Non",
    "Frais Manuel": "Non",
    "Surgelés": "Non",
    "Autre": "Non" # Cas par défaut si type non trouvé
}


# Définition des règles d'arrondi PCB selon l'image partagée
ARRONDI_PCB = {
    'Sec Méca - A/B': {'1er COLIS': 0.01, '2 EME COLIS OU PLUS': 0.01},
    'Sec Méca - A/C': {'1er COLIS': 0.01, '2 EME COLIS OU PLUS': 0.01},
    'Sec Homogène - A/B': {'1er COLIS': 0.01, '2 EME COLIS OU PLUS': 0.01},
    'Sec Homogène - A/C': {'1er COLIS': 0.01, '2 EME COLIS OU PLUS': 0.01},
    'Sec Hétérogène - A/B': {'1er COLIS': 0.01, '2 EME COLIS OU PLUS': 0.01},
    'Sec Hétérogène - A/C': {'1er COLIS': 0.01, '2 EME COLIS OU PLUS': 0.01},
    'Frais Méca': {'1er COLIS': 0.01, '2 EME COLIS OU PLUS': 0.01},
    'Frais Manuel': {'1er COLIS': 0.01, '2 EME COLIS OU PLUS': 0.01},
    'Surgelés': {'1er COLIS': 0.01, '2 EME COLIS OU PLUS': 0.01},
    'Sec - Autre': {'1er COLIS': 0.01, '2 EME COLIS OU PLUS': 0.01},
    '': {'1er COLIS': 0.01, '2 EME COLIS OU PLUS': 0.01}  # Valeur par défaut
}

# Définition des types de produits et types de produits V2
TYPES_PRODUITS = {
    "1060": "Sec Méca",
    "1061": "Sec Homogène",
    "1062": "Sec Hétérogène",
    "1063": "Sec Hétérogène",
    "1064": "Sec Hétérogène",
    "1070": "Frais Méca",
    "1071": "Frais Manuel",
    "1080": "Surgelés"
}

TYPES_PRODUITS_V2 = {
    "Sec Méca": ["A/B", "A/C"],
    "Sec Homogène": ["A/B", "A/C"],
    "Sec Hétérogène": ["A/B", "A/C"],
    "Frais Méca": [""],
    "Frais Manuel": [""],
    "Surgelés": [""]
}

# Définition des facteurs de lissage Min et Max par Type de produits V2
FACTEURS_LISSAGE = {
    "Sec Méca - A/B": {"Min Facteur": 0.0, "Max Facteur": 1.0},
    "Sec Méca - A/C": {"Min Facteur": 0.0, "Max Facteur": 4.0},
    "Sec Homogène - A/B": {"Min Facteur": 0.0, "Max Facteur": 4.0},
    "Sec Homogène - A/C": {"Min Facteur": 0.0, "Max Facteur": 4.0},
    "Sec Hétérogène - A/B": {"Min Facteur": 0.0, "Max Facteur": 4.0},
    "Sec Hétérogène - A/C": {"Min Facteur": 0.0, "Max Facteur": 4.0},
    "Frais Méca": {"Min Facteur": 0.0, "Max Facteur": 1.0},
    "Frais Manuel": {"Min Facteur": 0.0, "Max Facteur": 1.0},
    "Surgelés": {"Min Facteur": 0.0, "Max Facteur": 2.5},
    # Add other types as needed, or a default mechanism
}

def get_delivery_dates(start_date_str=DATE_COMMANDE_STR, num_days=14, excluded_weekday=6):
    """
    Génère une liste de dates de livraison à partir d'une date de début,
    en excluant un jour spécifique de la semaine (par défaut dimanche=6).
    """
    start_date = pd.to_datetime(start_date_str, dayfirst=True)
    delivery_dates = []
    current_date = start_date
    while len(delivery_dates) < num_days:
        if current_date.weekday() != excluded_weekday: # Lundi=0, Dimanche=6
            delivery_dates.append(current_date)
        current_date += timedelta(days=1)
    return delivery_dates

def get_arrondi_pcb_seuils(type_produit):
    """
    Récupère les seuils d'arrondi PCB pour un type de produit donné
    
    Args:
        type_produit: Type de produit (correspond au type dans l'onglet Arrondi PCB)
        
    Returns:
        Tuple (seuil_1er_colis, seuil_2eme_colis_ou_plus)
    """
    # Si le type de produit est présent dans le dictionnaire, renvoyer les seuils
    if type_produit in ARRONDI_PCB:
        return (ARRONDI_PCB[type_produit]['1er COLIS'], ARRONDI_PCB[type_produit]['2 EME COLIS OU PLUS'])
    # Sinon, renvoyer les valeurs par défaut
    return (0.01, 0.01)

def load_macro_params(file_path=None):
    """
    Load Macro-Param data from CSV file
    
    Args:
        file_path: Path to the MacroParam.csv file (optional)
        
    Returns:
        DataFrame with Macro-Param data
    """
    # Use default path if none provided
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), 'CSV/MacroParam.csv')
    
    try:
        # Try to load from CSV file
        df = pd.read_csv(
            file_path,
            sep=';',
            encoding='latin1'
        )
        
        # Make sure required columns exist
        required_columns = ['Zone de stockage', 'Type de produits', 'Correction TS']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns in {file_path}: {missing_columns}")
            print("Falling back to default values")
            return get_default_params()
            
        # Convert percentage strings to floats
        if 'Correction TS' in df.columns:
            df['Correction TS'] = df['Correction TS'].apply(
                lambda x: float(str(x).replace('%', '').replace(',', '.')) / 100 
                if isinstance(x, str) and '%' in str(x) else x
            )
            
        return df
        
    except Exception as e:
        print(f"Warning: Could not load MacroParam data from {file_path}: {e}")
        print("Falling back to default values")
        # If we can't load from file, use default values
        return get_default_params()

def get_param_value(name, default_value=None):
    """
    Get a specific parameter value from the MacroParam.csv file
    
    Args:
        name: Parameter name
        default_value: Default value if parameter not found
    
    Returns:
        Parameter value
    """    # Add additional parameters that aren't in the main table    
    additional_params = {
        # Paramètres pour Optimisation.py
        "periode_vente_min_augmentation": 12,  # 'Macro-Param'!$C$3 (Période de vente mini pour augmentation sur le frais (jours))
        "nb_jours_commande_par_semaine_min": 3,  # 'Macro-Param'!$C$4 (Nombre de jours de commande par semaine mini pour diminution PDC)
        "taux_service_amont_estime": TAUX_SERVICE_AMONT_ESTIME ,
        "jour_passage_commande": "J", # 'Macro-Param'!$C$10 (Jour Passage Commande)
        "taux_min_besoin_brut_rupture": 0.5,  # 'Macro-Param'!$C$13 (Taux min besoin brut en cas rupture)
        "empecher_baisse_commandes_promo": "Oui",  # 'Macro-Param'!$C$18 (Empêcher la baisse des commandes des articles en promos flux tirés?)
        "EXCLUSION_SM_A_CHARGER_TEST": "Application d'un coefficient sur la prévision des flop casse", # CORRIGÉ - Doit correspondre à Macro!$B$84
        "SOURCE_STOCK_POUR_SM_CHARGE": "RAO", # Correspond à Macro!$C$14 (déjà "RAO")
        # Autres paramètres existants
        "coefficient_exception_prevision": 0.0,
        "facteur_appro_max_frais": 1.2,  # 120% for "Facteur appro max Frais"
        "Casse Prev Activé": "Oui",  # Added parameter for Casse Prev Activé
        "Entrepôt Bloqué": False,    # Paramètre pour indiquer si l'entrepôt est bloqué
        "Activation stock max": "Oui",  # Paramètre pour activer l'utilisation du stock max (C14)
        # Paramètres pour le calcul de "Nb Jours Commande Max"
        "nb_jours_commande_max_default": 12,  # Valeur par défaut (12 jours) à utiliser si le produit n'existe pas dans la table
        "nb_jours_commande_max_par_zone": {   # Valeurs spécifiques par zone de stockage
            "312": 12                         # Zone 312 = 12 jours
        },
        "nb_jours_commande_max_familles": {   # Liste des familles spéciales
            "5 - FL": True                    # Famille produits frais
        },
        "nb_jours_commande_max_par_jour": {   # Valeurs par jour de la semaine (1=lundi, 7=dimanche)
            1: 1.6,                           # Lundi
            2: 1.6,                           # Mardi
            3: 1.6,                           # Mercredi
            4: 1.6,                           # Jeudi
            5: 1.6,                           # Vendredi
            6: 1.6,                           # Samedi
            7: 1.6                            # Dimanche
        },
        "nb_jours_commande_max_valeur_defaut_famille_special": 2.2,  # Valeur par défaut pour les familles spéciales (U5)
        
        # Nouvelle table de référence pour Nombre Jours Commande Max en fonction de Période de vente et Nombre de commandes par semaine
        "nb_jours_commande_max_table": {
            # Structure: {période_vente: {commandes_semaine: valeur}}
            0: {1: 1.6, 2: 1.6, 3: 1.6, 4: 1.6, 5: 1.6, 6: 1.6},
            3: {1: 2.2, 2: 2.2, 3: 2.2, 4: 2.2, 5: 2.2, 6: 2.2},
            5: {1: 2.6, 2: 2.6, 3: 2.6, 4: 2.6, 5: 2.6, 6: 2.6},
            8: {1: 3.0, 2: 3.0, 3: 3.0, 4: 3.0, 5: 3.0, 6: 3.0},
            12: {1: 3.5, 2: 3.5, 3: 3.5, 4: 3.5, 5: 3.0, 6: 3.0},
            15: {1: 4.0, 2: 4.0, 3: 3.5, 4: 3.5, 5: 3.0, 6: 3.0},
            20: {1: 12.0, 2: 6.0, 3: 3.5, 4: 3.5, 5: 3.0, 6: 3.0}
        }
    }
    
    if name in additional_params:
        return additional_params[name]
    
    return default_value

def get_default_params():
    """
    Return default Macro-Param data as a DataFrame
    Used as fallback when CSV file can't be loaded
    """
    # Define the data as in the Excel screenshot
    data = [
        {"Zone de stockage": 1060, "Type de produits": "Sec Méca", "Gestion A/B - A/C": "Oui", 
         "Poids du A/C calculé": 1.0, "Poids du A/C manuel": None, 
         "Poids du A/C final": 1.0, "Correction TS": 0.0},
        {"Zone de stockage": 1061, "Type de produits": "Sec Homogène", "Gestion A/B - A/C": "Oui", 
         "Poids du A/C calculé": 1.0, "Poids du A/C manuel": None, 
         "Poids du A/C final": 1.0, "Correction TS": 0.0},
        {"Zone de stockage": 1062, "Type de produits": "Sec Hétérogène", "Gestion A/B - A/C": "Oui", 
         "Poids du A/C calculé": 1.0, "Poids du A/C manuel": None, 
         "Poids du A/C final": 1.0, "Correction TS": 0.0},
        {"Zone de stockage": 1063, "Type de produits": "Sec Hétérogène", "Gestion A/B - A/C": "Oui", 
         "Poids du A/C calculé": 1.0, "Poids du A/C manuel": None, 
         "Poids du A/C final": 1.0, "Correction TS": 0.0},
        {"Zone de stockage": 1064, "Type de produits": "Sec Hétérogène", "Gestion A/B - A/C": "Oui", 
         "Poids du A/C calculé": 1.0, "Poids du A/C manuel": None, 
         "Poids du A/C final": 1.0, "Correction TS": 0.0},
        {"Zone de stockage": 1070, "Type de produits": "Frais Méca", "Gestion A/B - A/C": "Non", 
         "Poids du A/C calculé": 1.0, "Poids du A/C manuel": None, 
         "Poids du A/C final": 1.0, "Correction TS": 0.0},
        {"Zone de stockage": 1071, "Type de produits": "Frais Manuel", "Gestion A/B - A/C": "Non", 
         "Poids du A/C calculé": 1.0, "Poids du A/C manuel": None, 
         "Poids du A/C final": 1.0, "Correction TS": 0.0},
        {"Zone de stockage": 1080, "Type de produits": "Surgelés", "Gestion A/B - A/C": "Non", 
         "Poids du A/C calculé": 1.0, "Poids du A/C manuel": None, 
         "Poids du A/C final": 1.0, "Correction TS": 0.0},
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

def create_csv_if_needed():
    """
    Create MacroParam.csv file if it doesn't already exist
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'MacroParam.csv')
    
    if not os.path.exists(csv_path):
        print(f"Creating MacroParam.csv at {csv_path}")
        params = get_default_params()
        
        # Format percentages for better readability
        for col in ['Poids du A/C calculé', 'Poids du A/C final', 'Correction TS']:
            if col in params.columns:
                params[col] = params[col].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "")
        
        params.to_csv(csv_path, sep=';', index=False, encoding='latin1')
        print("MacroParam.csv created successfully")
    else:
        print(f"MacroParam.csv already exists at {csv_path}")

if __name__ == "__main__":
    # When run directly, create the CSV file if needed
    create_csv_if_needed()
    
    # Test loading the parameters
    params = load_macro_params()
    print("Loaded parameters:")
    print(params)
