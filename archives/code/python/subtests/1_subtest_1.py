"""
PROJET :
--------
vitrine_pipeline

TITRE :
--------
1_subtest_1.py

OBJECTIF PRINCIPAL :
---------------------
Ce script lit un fichier JSONL annoté, extrait le contenu du champ "text" de chaque ligne,
et écrit ces données dans un fichier CSV à une seule colonne ("body") pour faciliter les analyses ultérieures.

DÉPENDANCES :
-------------
- os
- json
- csv

FONCTIONNALITÉS PRINCIPALES :
-----------------------------
1) Lecture du fichier JSONL annoté.
2) Extraction de la valeur associée à "text" pour chaque entrée.
3) Écriture d'un fichier CSV comportant une colonne "body" avec les valeurs extraites.

OBJECTIFS :
----------
1) Tester les performances de modèles avec des prompts par catégories spécifiques, plutôt qu'un prompt général.

Auteur :
--------
Antoine Lemor
"""

import os
import json
import csv

# ##############################################################################
# A. DÉFINITION DES CHEMINS D'ACCÈS
# ##############################################################################
# Chemin relatif vers le fichier JSONL d'entrée
input_path = os.path.join('data', 'processed', 'validation', 'annotated_jsonl', 'JerGil.jsonl')

# Chemin relatif vers le répertoire de sortie et le fichier CSV
output_dir = os.path.join('data', 'processed', 'subset', 'subtest')
output_csv_path = os.path.join(output_dir, 'subtest_1.csv')

# ##############################################################################
# B. CRÉATION DU RÉPERTOIRE DE SORTIE
# ##############################################################################
# Crée le répertoire de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# ##############################################################################
# C. LECTURE DU FICHIER JSONL ET EXTRACTION DES DONNÉES
# ##############################################################################
rows = []
with open(input_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        line = line.strip()
        if line:
            data = json.loads(line)
            # Récupère la valeur associée à "text"
            rows.append(data.get("text", ""))

# ##############################################################################
# D. ÉCRITURE DES DONNÉES DANS UN FICHIER CSV
# ##############################################################################
with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Écrit l'en-tête de colonne "body"
    writer.writerow(["body"])
    # Parcourt chaque texte extrait et l'écrit dans le CSV
    for text in rows:
        writer.writerow([text])
