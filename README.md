# Outil de Lissage

Ce dépôt contient l'ensemble des scripts Python utilisés pour  lisser les besoins d'approvisionnement et générer le plan de commande final. Les différents modules chargent des fichiers CSV/Excel, appliquent des calculs complexes puis produisent un fichier Excel optimisé.

## Installation

1. Installez **Python 3.10** ou plus récent.
2. Installez les dépendances listées dans `requirements.txt` :
   ```bash
   python -m pip install -r requirements.txt
   ```

## Exécution

Depuis la racine du projet, lancez :
```bash
python main.py
```
Le script consolide les données puis crée notamment `merged_predictions.csv` et `PDC_Sim_Optimized_Python.xlsx`.

## Description des fichiers principaux

- **main.py** – point d'entrée orchestrant l'ensemble des traitements.
- **MacroParam.py** – paramètres globaux (dates de référence, marges, etc.).
- **DetailRao.py** – charge et prépare le détail RAO depuis le dossier `CSV`.
- **PrevB2C.py** – génère les prévisions du canal B2C.
- **PrevPromo.py** – traite les prévisions liées aux promotions.
- **PrevCasse.py** – ajoute les prévisions de casse produits.
- **PrevEncours.py** – intègre les prévisions d'encours.
- **PrevFinal.py** – calcule les prévisions finales combinées.
- **Boite3.py** – produit la synthèse « Boîte 3 ».
- **MiniPubliFL.py** – récupère les informations de mini publication FL.
- **StockMarchand.py** – lit les données de stock marchand.
- **FacteurAppro.py** – calcule les coefficients d'approvisionnement.
- **TypeProduits.py** – catégorise les produits (familles, classes, etc.).
- **EntrepotProduitBloque.py** – repère entrepôts ou produits bloqués.
- **NbJoursCommandeMax.py** – détermine le nombre de jours maximum de commande.
- **CommandeFinale.py** – construit les commandes finales hors lissage.
- **BorneMinMax.py** – fixe les bornes mini/maxi pour le lissage.
- **Optimisation.py** – prépare le fichier `PDC_Sim_Input_For_Optim.xlsx`.
- **vba_logic_fixed.py** – réimplémente la logique d'optimisation VBA.
- **CalculAutresColonnes.py** et **Calcul_Commandes_Detail_Optimise.py** – fonctions utilitaires pour enrichir le détail commande.
- **Ts.py**, **Total.py** – scripts annexes pour le traitement des valeurs de TS et calculs totaux.
- **optimisation_globale.py** – ancienne approche d'optimisation (optionnelle).
- **test.py** – petit script de test/démonstration.

Plusieurs sous-dossiers contiennent les fichiers d'entrée :`CSV`, `Casse`, `Facteur_Appro`, `Livraisons_Client_A_Venir`, `PDC`, `Previsions`. Les fichiers Excel du projet (`.xlsx` ou `.xlsm`) servent d'exemple de données ou de résultat.

## Détail des modules appelés par `main.py`

Ce qui suit décrit plus précisément le rôle de chaque module importé par `main.py` :

### MacroParam.py
Centralise les paramètres globaux du projet (dates de référence, coefficients,
options diverses). La fonction `get_param_value()` permet de récupérer un
paramètre avec une valeur par défaut. Le module peut également générer un
`MacroParam.csv` de base si nécessaire.

### DetailRao.py
Charge le fichier `*_Detail_RAO_Commande*.csv`, normalise les colonnes
(`CODE_METI`, dates, valeurs numériques) et renvoie un `DataFrame` propre prêt à
être fusionné.

### PrevB2C.py
À partir des données RAO et du fichier de prévisions B2C, calcule les colonnes
`Prev C1-L2` et `Prev L1-L2` pour chaque produit.

### PrevPromo.py
Extrait les prévisions liées aux promotions (`Prev Promo C1-L2` et
`Prev Promo L1-L2`) en se basant sur les dates fournies dans `MacroParam`.

### PrevCasse.py
Combine le détail RAO avec les historiques de casse afin de produire les
colonnes `Casse Prev C1-L2` et `Casse Prev L1-L2` ainsi que des positions pour
l'analyse.

### PrevEncours.py
Lit les livraisons clients à venir et construit des positions ainsi que des
totaux `En-cours client C1-L2` et `En-cours client L1-L2` sur la fenêtre de
temps définie.

### Ts.py
Calcule la valeur de taux de service (`TS`) pour chaque produit en croisant le
stock marchand et un fichier `Taux_Service*.csv`.

### PrevFinal.py
À partir du `merged_df` intermédiaire, produit les colonnes finales de
prévision :`Prév C1-L2 Finale` et `Prév L1-L2 Finale`. Le module tient compte
d'une éventuelle feuille `Exclusion.xlsx` et des coefficients définis dans
`MacroParam`.

### FacteurAppro.py
Applique des coefficients multiplicatifs par famille, sous-famille ou EAN afin
d'obtenir la colonne `Facteur Multiplicatif Appro`. Les valeurs proviennent des
fichiers de la sous-arborescence `Facteur_Appro`.

### TypeProduits.py
Détermine les types de produits (et leur version « V2 ») à partir de la classe
de stockage et calcule aussi la catégorie Top 500/Top 3000 grâce aux fichiers
`Top_500.csv` et `Top_3000.csv`.

### MiniPubliFL.py
Lit `Stock mini publication FL vrac.xlsx` pour obtenir une valeur de « Mini
Publication FL » par `CODE_METI`.

### StockMarchand.py
Ouvre l'Excel « Outil Lissage Approvisionnements…xlsm » afin de récupérer les
données complètes de stock marchand utilisées dans plusieurs calculs. Le module
aussi génére les fichiers `Top_500.csv` et `Top_3000.csv`.

### EntrepotProduitBloque.py
Ajoute les colonnes `Entrepôt Bloqué` et `Produit Bloqué` en se basant sur un
paramètre global et sur la colonne `COCHE_RAO` du détail RAO.

### NbJoursCommandeMax.py
Calcule la colonne `Nb Jours Commande Max` à l'aide de tables de référence de
`MacroParam` puis déduit diverses quantités maximales de commande.

### CommandeFinale.py
Implémente la logique Excel de calcul des commandes finales et Autres colonnes Commande optimisée sans arrondi, Commande optimisée avec arrondi et mini,
Commande optimisée avec arrondi et mini et TS en tenant compte des facteurs de lissage, du stock et des quantités
minimum.

### BorneMinMax.py
Définit les bornes `Borne Min Facteur multiplicatif lissage` et `Borne Max Facteur multiplicatif lissage` du facteur de lissage et contient également la
mise à jour du facteur `Facteur multiplicatif Lissage besoin brut` à partir des résultats d'optimisation.

### CalculAutresColonnes.py
Regroupe plusieurs calculs additionnels : quantité maxi à charger, indicateur
`Algo Meti`, tests sur la mini publication, etc.

### Optimisation.py
Génère le fichier `PDC_Sim_Input_For_Optim.xlsx` et prépare les tables de
paramètres destinées au processus d'optimisation.

### vba_logic_fixed.py
Réimplémente en Python la logique d'optimisation auparavant écrite en VBA ; il
est appelé après la génération du fichier d'entrée pour produire
`PDC_Sim_Optimized_Python.xlsx`.
