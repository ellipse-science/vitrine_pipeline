# ---------------------------------------------------------------------------------
# PROJET : vitrine_pipeline
#
# TITRE : 2_JSONL.py
#
# OBJECTIF PRINCIPAL :
# ---------------------
# Ce script génère des fichiers JSONL pour l’annotation avec Doccano en utilisant
# un fichier CSV contenant le texte et les annotations des LLMs. Il gère deux cas :
#   - Un seul annotateur : toutes les lignes annotées disponibles sont utilisées,
#     sans découpage 20 % / 80 %.
#   - Plusieurs annotateurs : 20 % des lignes sont communes à tous et 80 % sont
#     distribuées équitablement entre eux.
#
# DÉPENDANCES :
# -------------
# - os
# - json
# - random
# - math
# - csv
#
# FONCTIONNALITÉS PRINCIPALES :
# -----------------------------
# 1) Demande à l'utilisateur le chemin du fichier CSV et le nombre d’annotateurs.
# 2) Lit la/les colonne(s) de type JSON contenant les annotations (ex. "body_annotated",
#    "comment_annotated", etc.), et ajoute un suffixe de colonne pour différencier les labels.
# 3) Si plusieurs annotateurs, divise les données en 20 % communes et 80 % uniques
#    (réparties équitablement entre eux). Si un seul annotateur, aucune division.
# 4) Génère un fichier de configuration pour Doccano contenant tous les labels détectés.
# 5) Crée pour chaque annotateur un fichier JSONL dans le format attendu par Doccano.
# 6) Affiche un résumé statistique de la répartition (nombre de phrases, distribution
#    des labels, etc.).
#
# Auteur :
# ---------
# Antoine Lemor
# ---------------------------------------------------------------------------------

import os
import json
import random
import csv

def main():
    """
    Fonction principale pour orchestrer la création des fichiers JSONL et du fichier
    de configuration Doccano.
    
    Étapes :
        1) Demander à l'utilisateur le chemin du fichier CSV et le nombre d'annotateurs.
        2) Lire le CSV et stocker toutes les lignes dans une structure interne.
        3) Demander quelles colonnes contiennent les annotations JSON ("_annotated" par ex.).
           Créer un fichier de configuration Doccano avec tous les labels détectés,
           suffixés par le nom de la colonne.
        4) Selon le nombre d’annotateurs :
           - Si 1 annotateur : toutes les données sont utilisées (aucun découpage).
           - Si > 1 annotateurs : effectuer un découpage 20 % commun / 80 % unique.
        5) Répartir les données uniques équitablement entre les annotateurs s’ils sont
           plusieurs.
        6) Générer les fichiers JSONL (un par annotateur) avec la répartition adéquate.
        7) Afficher un résumé statistique des ensembles d'annotation.
    """
    # --------------------------------------------------------------------------
    # Définition de la seed en dur pour la reproductibilité
    # --------------------------------------------------------------------------
    seed_value = 42
    random.seed(seed_value)

    # --------------------------------------------------------------------------
    # 1. Demander les entrées utilisateur : chemin du CSV et nombre d'annotateurs
    # --------------------------------------------------------------------------
    csv_file_path = input("Entrez le chemin du fichier CSV : ").strip()
    while not os.path.isfile(csv_file_path):
        print(f"Erreur : Le fichier '{csv_file_path}' n'existe pas.")
        csv_file_path = input("Entrez un chemin valide pour le fichier CSV : ").strip()

    try:
        num_annotators = int(input("Entrez le nombre d'annotateurs (entier) : ").strip())
        if num_annotators <= 0:
            raise ValueError
    except ValueError:
        print("Erreur : Nombre d'annotateurs invalide. Doit être un entier positif.")
        return

    # --------------------------------------------------------------------------
    # 2. Lecture du CSV et préparation des données
    # --------------------------------------------------------------------------
    data_rows = []
    with open(csv_file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames  # Pour lister les colonnes disponibles
        for row in reader:
            data_rows.append(row)

    if not data_rows:
        print("Erreur : Le fichier CSV est vide ou mal formaté.")
        return

    # --------------------------------------------------------------------------
    # Affichage des colonnes disponibles et demande de sélection
    # --------------------------------------------------------------------------
    print("Colonnes disponibles dans le CSV :")
    for h in headers:
        print(f"  - {h}")

    annotation_cols_input = input(
        "Entrez les colonnes contenant les annotations JSON (séparées par des virgules) : "
    ).strip()

    # Filtrer pour ne garder que les colonnes valides
    if annotation_cols_input:
        selected_annotation_cols = [
            col.strip() for col in annotation_cols_input.split(",")
            if col.strip() in headers
        ]
    else:
        selected_annotation_cols = []

    if not selected_annotation_cols:
        print("Aucune colonne d'annotation sélectionnée. Fin du script.")
        return

    # --------------------------------------------------------------------------
    # 3. Répartition : Si plusieurs annotateurs, 20 % / 80 %. Si un seul, tout est pris.
    # --------------------------------------------------------------------------
    random.shuffle(data_rows)  # Mélange global

    if num_annotators == 1:
        # Un seul annotateur : aucune division
        common_data = data_rows  # Tout va à l'annotateur
        unique_data = []
        print(f"\nUn seul annotateur : {len(common_data)} lignes seront utilisées sans découpage.")
    else:
        # Plusieurs annotateurs
        total_count = len(data_rows)
        common_count = int(0.2 * total_count)
        unique_count = total_count - common_count

        common_data = data_rows[:common_count]   # 20 %
        unique_data = data_rows[common_count:]   # 80 %
        print(f"\nPlusieurs annotateurs : {total_count} lignes au total.")
        print(f"Répartition : {common_count} lignes communes / {unique_count} lignes uniques.")

    # --------------------------------------------------------------------------
    # 4. Répartition des données uniques entre les annotateurs (si plusieurs)
    # --------------------------------------------------------------------------
    annotators_unique_data = []
    if num_annotators > 1:
        unique_count_per_annot = len(unique_data) // num_annotators

        start_idx = 0
        for _ in range(num_annotators):
            end_idx = start_idx + unique_count_per_annot
            annotators_unique_data.append(unique_data[start_idx:end_idx])
            start_idx = end_idx
    else:
        annotators_unique_data = [[]]  # Cohérence pour un seul annotateur

    # --------------------------------------------------------------------------
    # 5. Extraction de tous les labels à partir des colonnes JSON sélectionnées
    # --------------------------------------------------------------------------
    all_labels = set()
    for row in data_rows:
        for col in selected_annotation_cols:
            try:
                ann_json = json.loads(row.get(col, "{}"))
                themes = ann_json.get("themes", [])
                for theme in themes:
                    # Ex. "politics_body_annotated"
                    label_with_suffix = f"{theme}_{col}"
                    all_labels.add(label_with_suffix)
            except json.JSONDecodeError:
                # Ignorer les lignes mal formées
                continue

    all_labels = sorted(all_labels)

    # --------------------------------------------------------------------------
    # 6. Création du fichier de configuration des labels Doccano
    # --------------------------------------------------------------------------
    output_dir = "data/processed/validation"
    os.makedirs(output_dir, exist_ok=True)

    color_palette = [
        "#F44336", "#E91E63", "#9C27B0", "#673AB7", "#3F51B5",
        "#2196F3", "#03A9F4", "#00BCD4", "#009688", "#4CAF50",
        "#8BC34A", "#CDDC39", "#FFC107", "#FF9800", "#FF5722",
        "#795548", "#9E9E9E", "#607D8B"
    ]

    doccano_labels_config = []
    for i, label in enumerate(all_labels):
        color = color_palette[i % len(color_palette)]
        doccano_labels_config.append({"label": label, "color": color})

    config_json = {"label": doccano_labels_config}
    config_path = os.path.join(output_dir, "doccano_config.json")

    with open(config_path, mode="w", encoding="utf-8") as cf:
        json.dump(config_json, cf, ensure_ascii=False, indent=2)

    print(f"Fichier de configuration Doccano créé à : {config_path}")

    # --------------------------------------------------------------------------
    # 7. Génération des fichiers JSONL (un par annotateur)
    # --------------------------------------------------------------------------
    annotators_file_paths = []

    if num_annotators == 1:
        # Un fichier unique
        file_path = os.path.join(output_dir, "annotator_1.jsonl")
        random.shuffle(common_data)  # Re-mélange final

        with open(file_path, mode="w", encoding="utf-8") as out_f:
            for row in common_data:
                text = row.get("body", "")
                row_labels = []
                for col in selected_annotation_cols:
                    try:
                        ann_json = json.loads(row.get(col, "{}"))
                        themes = ann_json.get("themes", [])
                        for theme in themes:
                            row_labels.append(f"{theme}_{col}")
                    except json.JSONDecodeError:
                        continue

                doccano_line = {"text": text, "label": row_labels}
                out_f.write(json.dumps(doccano_line, ensure_ascii=False) + "\n")

        annotators_file_paths.append(file_path)

    else:
        # Plusieurs annotateurs
        for i in range(num_annotators):
            annotator_id = i + 1
            file_name = f"annotator_{annotator_id}.jsonl"
            file_path = os.path.join(output_dir, file_name)

            # Combine 20 % communes + part unique
            combined_data = common_data + annotators_unique_data[i]
            random.shuffle(combined_data)

            with open(file_path, mode="w", encoding="utf-8") as out_f:
                for row in combined_data:
                    text = row.get("body", "")
                    row_labels = []
                    for col in selected_annotation_cols:
                        try:
                            ann_json = json.loads(row.get(col, "{}"))
                            themes = ann_json.get("themes", [])
                            for theme in themes:
                                row_labels.append(f"{theme}_{col}")
                        except json.JSONDecodeError:
                            continue

                    doccano_line = {"text": text, "label": row_labels}
                    out_f.write(json.dumps(doccano_line, ensure_ascii=False) + "\n")

            annotators_file_paths.append(file_path)

    # --------------------------------------------------------------------------
    # 8. Résumé et statistiques dans le terminal
    # --------------------------------------------------------------------------
    print("\n===== Résumé de la répartition =====")
    print(f"Nombre d'annotateurs : {num_annotators}")

    if num_annotators == 1:
        print(f"Nombre total de lignes utilisées : {len(common_data)}")
    else:
        print(f"Nombre total de lignes : {len(data_rows)}")
        print(f"Lignes communes (20 %) : {len(common_data)}")
        print(f"Lignes uniques (80 %) : {len(unique_data)}")
        print(f"Lignes uniques par annotateur (approx.) : {len(unique_data) // num_annotators}")

    # Parcourir chaque fichier JSONL pour afficher le nombre total de phrases
    # et la répartition des labels
    for i, file_path in enumerate(annotators_file_paths, start=1):
        with open(file_path, mode="r", encoding="utf-8") as f_in:
            lines = f_in.readlines()

        total_count = len(lines)
        label_counts = {lbl: 0 for lbl in all_labels}

        for line in lines:
            record = json.loads(line)
            labels_list = record.get("label", [])

            for lbl in labels_list:
                if lbl in label_counts:
                    label_counts[lbl] += 1

        print(f"\nAnnotateur {i} :")
        print(f"  Fichier JSONL : {file_path}")
        print(f"  Nombre total de phrases : {total_count}")
        print("  Répartition des labels :")
        for lbl, cnt in label_counts.items():
            print(f"    {lbl} : {cnt}")

    print("\nLes fichiers JSONL et doccano_config.json se trouvent dans :")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
