"""
PROJET :
--------
vitrine_pipeline

TITRE :
--------
1_subtest_2.py (Nouvelle version)

OBJECTIF PRINCIPAL :
---------------------
Ce script lit plusieurs fichiers JSONL annotés (annotator_1 à annotator_5), extrait le
texte ainsi que les métadonnées de chaque ligne, et écrit ces données dans un fichier CSV unique.
Le CSV s'appelle 'LLM_SUBSET.csv' et contient uniquement les phrases uniques, dans leur ordre
d'apparition. Chaque colonne du CSV correspond à 'text' ou à un champ des métadonnées.

STRUCTURE D'ENTRÉE :
--------------------
Plusieurs fichiers JSONL :
    data/processed/validation/annotator_1.jsonl
    data/processed/validation/annotator_2.jsonl
    data/processed/validation/annotator_3.jsonl
    data/processed/validation/annotator_4.jsonl
    data/processed/validation/annotator_5.jsonl

Chaque ligne du JSONL est de la forme :
{
  "text": "Le contenu textuel...",
  "metadata": {
      "id": "...",
      "extraction_date": "...",
      "media_id": "...",
      "lang": "...",
      ...
  }
}

STRUCTURE DE SORTIE :
---------------------
- Fichier CSV : data/processed/validation/LLM_SUBSET.csv
- Colonnes :
  1) text
  2) Une colonne par champ des métadonnées (id, extraction_date, media_id, lang, etc.)
- Chaque ligne du CSV représente une phrase unique, dans l'ordre d'apparition au fil
  des différents JSONL (annotator_1, puis annotator_2, etc.).

DÉPENDANCES :
-------------
- os
- json
- csv

UTILISATION :
-------------
Exécutez ce script pour générer data/processed/validation/LLM_SUBSET.csv avec toutes les
phrases uniques et leurs métadonnées depuis les 5 fichiers JSONL.
"""

import os
import json
import csv

def main():
    # 1) Définition des chemins d'entrée et de sortie
    base_dir = os.path.join("data", "processed", "validation")
    input_files = [
        os.path.join(base_dir, "annotator_1.jsonl"),
        os.path.join(base_dir, "annotator_2.jsonl"),
        os.path.join(base_dir, "annotator_3.jsonl"),
        os.path.join(base_dir, "annotator_4.jsonl"),
        os.path.join(base_dir, "annotator_5.jsonl")
    ]
    output_csv = os.path.join(base_dir, "LLM_SUBSET.csv")

    # 2) Structures de données pour stocker les lignes uniques
    #    et l'ordre d'apparition
    seen_texts = set()       # Pour filtrer les textes dupliqués
    rows = []                # Pour conserver l'ordre d'apparition
    metadata_keys = set()    # Pour collecter tous les noms de champs de métadonnées

    # 3) Lecture de chaque fichier JSONL en séquence
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Fichier introuvable : {file_path} — Ignoré.")
            continue

        with open(file_path, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                # Récupérer le texte et la metadata
                text_value = data.get("text", "")
                metadata_dict = data.get("metadata", {})

                # Vérifier si ce texte est déjà enregistré
                if text_value in seen_texts:
                    # Si déjà vu, on ignore cette ligne
                    continue
                else:
                    # Nouveau texte : on l'ajoute à la sortie
                    seen_texts.add(text_value)
                    row = {"text": text_value}

                    # Ajouter les métadonnées à la ligne
                    for k, v in metadata_dict.items():
                        row[k] = v
                        metadata_keys.add(k)

                    rows.append(row)

    # 4) Création de la liste finale des colonnes
    #    On veut "text" en première colonne, puis toutes les métadonnées
    #    dans l'ordre alphabétique (ou tout autre ordre désiré).
    columns = ["text"] + sorted(metadata_keys)

    # 5) Écriture du CSV final
    with open(output_csv, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            # S'assurer que toutes les colonnes sont présentes
            # (certaines métadonnées pourraient être manquantes dans certaines lignes)
            for col in columns:
                if col not in row:
                    row[col] = ""
            writer.writerow(row)

    print(f"Fichier CSV généré : {output_csv}")

if __name__ == "__main__":
    main()
