"""
PROJET :
---------
Vitrine_pipeline

TITRE :
---------
1_subset_test_creation.py

OBJECTIF PRINCIPAL :
---------------------
Ce script sélectionne aléatoirement 100 phrases à partir d'un fichier CSV
d'entrée et génère un nouveau fichier CSV de test. Le fichier CSV d'entrée
se trouve dans data/raw/subset/radar_subset_en.csv et le CSV de sortie sera 
stocké dans data/processed/subset/radar_subset_test_en.csv.

DÉPENDANCES :
-------------
- os
- pandas
- random

FONCTIONNALITÉS PRINCIPALES :
-----------------------------
1) Charger le fichier CSV source.
2) Sélectionner aléatoirement 100 phrases 
3) Écrire le nouveau CSV de test dans le répertoire de sortie.

Auteur :
---------
Antoine Lemor
"""

import os
import pandas as pd
import random

##############################################################################
#                          Fonction principale
##############################################################################
def main():
    """
    Charge le fichier CSV, sélectionne 100 phrases aléatoires et écrit le résultat.
    """
    try:
        # Définition des chemins relatifs
        chemin_source = os.path.join('data', 'raw', 'subset', 'radar_subset.csv')
        chemin_sortie_dir = os.path.join('data', 'processed', 'subset')
        chemin_sortie = os.path.join(chemin_sortie_dir, 'radar_subset_test.csv')
        
        # Création du répertoire de sortie s'il n'existe pas
        if not os.path.isdir(chemin_sortie_dir):
            os.makedirs(chemin_sortie_dir)
            # Répertoire créé pour stocker le fichier de sortie
        
        # Lecture du fichier CSV source
        df = pd.read_csv(chemin_source)
        # Vérification du nombre de lignes pour éviter une erreur si le fichier contient moins de 100 lignes
        if len(df) < 100:
            print("[ERREUR] Le fichier source contient moins de 100 phrases.")
            return
        
        # Sélection aléatoire de 100 phrases via la colonne "body"
        df_echantillon = df.sample(n=100, random_state=random.randint(0, 10000))
        
        # Écriture du fichier CSV de test
        df_echantillon.to_csv(chemin_sortie, index=False)
        print(f"[INFO] Fichier CSV de test généré avec succès : {chemin_sortie}")
    
    except Exception as e:
        print(f"[ERREUR] Une erreur est survenue : {e}")

if __name__ == '__main__':
    main()
