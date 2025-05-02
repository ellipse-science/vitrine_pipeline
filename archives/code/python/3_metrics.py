"""
 PROJET : vitrine_pipeline

 TITRE : 2_JSONL.py

 OBJECTIF PRINCIPAL :
 ---------------------
 Ce script génère des fichiers JSONL pour l’annotation avec Doccano en utilisant
 un fichier CSV contenant le texte et les annotations des LLMs. Il gère deux cas :
   - Un seul annotateur : toutes les lignes annotées disponibles sont utilisées,
     sans découpage 20 % / 80 %.
   - Plusieurs annotateurs : 20 % des lignes sont communes à tous et 80 % sont
     distribuées équitablement entre eux.

 DÉPENDANCES :
 -------------
 - os
 - json
 - random
 - csv

 FONCTIONNALITÉS PRINCIPALES :
 -----------------------------
 1) Lister les CSV disponibles dans "data/processed/subset" et demander à
    l'utilisateur de sélectionner celui souhaité.
 2) Demander le nombre d’annotateurs (entier positif).
 3) Lire la/les colonne(s) de type JSON contenant les annotations (ex. "body_annotated",
    "comment_annotated", etc.), et ajouter un suffixe de colonne pour différencier les labels.
 4) Demander quelles métadonnées issues du CSV original doivent figurer dans le JSONL.
    Elles seront ajoutées sous la forme doccano_line["metadata"] = {...}.
 5) Si plusieurs annotateurs, diviser les données en 20 % communes et 80 % uniques,
    réparties équitablement. Si un seul annotateur, aucune division.
 6) Générer un fichier de configuration Doccano (doccano_config.json) contenant
    tous les labels détectés (après filtrage).
 7) Créer pour chaque annotateur un fichier JSONL dans le format attendu par Doccano
    (ou un seul fichier si un seul annotateur).
 8) Afficher un résumé statistique de la répartition (nombre de phrases, distribution
    des labels, etc.) et enregistrer ce même résumé dans un fichier CSV de log.

 Auteur :
 ---------
 Antoine Lemor
"""

import os
import json
import random
import csv

def parse_annotation_values(json_str: str, col_name: str):
    """
    Convertit le contenu JSON d'une cellule en un ensemble de labels.
    Pour chaque clé dans le JSON, si la valeur associée est :
      - Une liste : on récupère chaque élément valide pour en faire un label,
        suffixé par '_<clé>_<colonne>'.
      - Une chaîne : on en fait directement un label, suffixé par '_<clé>_<colonne>'.
    Ne sont conservés que les labels non nuls, non 'null', et de longueur > 1.
    Exemples :
      {
        "themes": ["immigration", "education"],
        "sentiment": "neutral",
        "political_parties": null,
        "specific_themes": null
      }
      -> ["immigration_themes_<col_name>", "education_themes_<col_name>", "neutral_sentiment_<col_name>"]
    """
    try:
        ann_dict = json.loads(json_str)
        if not isinstance(ann_dict, dict):
            return []
    except json.JSONDecodeError:
        # En cas d'erreur de parsing ou si ce n'est pas un dict, on renvoie une liste vide
        return []

    all_labels = []

    for key, val in ann_dict.items():
        if val is None:
            # Exclut explicitement None
            continue

        if isinstance(val, list):
            # Si c'est une liste, on parcourt chaque élément
            for item in val:
                if item and isinstance(item, str):
                    candidate = item.strip()
                    if (
                        candidate.lower() != "null"
                        and len(candidate) > 1
                    ):
                        # On suffixe par '_<clé>_<colonne>'
                        label = f"{candidate}_{key}_{col_name}"
                        all_labels.append(label)

        elif isinstance(val, str):
            candidate = val.strip()
            if (
                candidate.lower() != "null"
                and len(candidate) > 1
            ):
                label = f"{candidate}_{key}_{col_name}"
                all_labels.append(label)
        # On ignore d'autres types (int, dict, etc.)

    return all_labels


def main():
    """
    Fonction principale pour orchestrer la création des fichiers JSONL et du fichier
    de configuration Doccano, en incluant les métadonnées choisies et le reporting
    des distributions d'annotation dans un CSV.

    Étapes :
        1) Lister les fichiers CSV dans 'data/processed/subset' et demander à l’utilisateur
           de sélectionner celui souhaité.
        2) Demander le nombre d’annotateurs.
        3) Lire le CSV et stocker toutes les lignes dans une structure interne.
        4) Demander quelles colonnes contiennent les annotations JSON ("_annotated" par ex.).
           Puis demander quelles colonnes de métadonnées doivent être intégrées.
        5) Créer un fichier de configuration Doccano avec tous les labels détectés,
           suffixés par le nom de la colonne et de la clé.
        6) Filtrer les lignes pour ne conserver que celles déjà annotées (au moins un label).
        7) Selon le nombre d’annotateurs :
           - Si 1 annotateur : toutes les données annotées sont utilisées (aucun découpage).
           - Si > 1 annotateurs : effectuer un découpage 20 % commun / 80 % unique.
        8) Répartir les données uniques équitablement entre les annotateurs s’ils sont
           plusieurs.
        9) Générer les fichiers JSONL (un par annotateur) avec la répartition adéquate,
           en incluant les métadonnées choisies.
        10) Afficher un résumé statistique des ensembles d'annotation dans le terminal
            et enregistrer ces statistiques dans un fichier CSV de log.
    """
    # --------------------------------------------------------------------------
    # Définition de la seed en dur pour la reproductibilité
    # --------------------------------------------------------------------------
    seed_value = 42
    random.seed(seed_value)

    # --------------------------------------------------------------------------
    # 1. Lister les fichiers CSV dans 'data/processed/subset' et demander à
    #    l'utilisateur de sélectionner celui souhaité
    # --------------------------------------------------------------------------
    subset_dir = "data/processed/subset/subtest/full_prompt"
    if not os.path.isdir(subset_dir):
        print(f"Erreur : Le répertoire '{subset_dir}' est introuvable.")
        return

    available_files = [f for f in os.listdir(subset_dir) if f.lower().endswith(".csv")]
    if not available_files:
        print(f"Aucun fichier CSV trouvé dans '{subset_dir}'.")
        return

    print("Fichiers CSV disponibles dans 'data/processed/subset' :")
    for idx, fname in enumerate(available_files, start=1):
        print(f"  {idx}. {fname}")

    file_choice = input("Sélectionnez un fichier CSV en entrant son numéro : ").strip()
    try:
        file_index = int(file_choice) - 1
        if file_index < 0 or file_index >= len(available_files):
            raise IndexError
    except (ValueError, IndexError):
        print("Erreur : choix de fichier invalide.")
        return

    csv_file_path = os.path.join(subset_dir, available_files[file_index])

    # --------------------------------------------------------------------------
    # 2. Demander le nombre d'annotateurs
    # --------------------------------------------------------------------------
    try:
        num_annotators = int(input("Entrez le nombre d'annotateurs (entier positif) : ").strip())
        if num_annotators <= 0:
            raise ValueError
    except ValueError:
        print("Erreur : Nombre d'annotateurs invalide. Doit être un entier positif.")
        return

    # --------------------------------------------------------------------------
    # 3. Lecture du CSV et préparation des données
    # --------------------------------------------------------------------------
    data_rows = []
    with open(csv_file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames  # Pour lister les colonnes disponibles
        if not headers:
            print("Erreur : Impossible de lire les en-têtes du CSV.")
            return

        for row in reader:
            data_rows.append(row)

    if not data_rows:
        print("Erreur : Le fichier CSV est vide ou mal formaté.")
        return

    # --------------------------------------------------------------------------
    # Affichage des colonnes disponibles et demande de sélection (annotations)
    # --------------------------------------------------------------------------
    print("\nColonnes disponibles dans le CSV :")
    for h in headers:
        print(f"  - {h}")

    annotation_cols_input = input(
        "\nEntrez les colonnes contenant les annotations JSON (séparées par des virgules) : "
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
    # Demande de sélection des colonnes de métadonnées à inclure
    # --------------------------------------------------------------------------
    print("\nColonnes disponibles (rappel) :")
    for h in headers:
        print(f"  - {h}")

    metadata_cols_input = input(
        "\nEntrez les colonnes que vous souhaitez inclure comme métadonnées "
        "(séparées par des virgules), ou laissez vide pour n'en inclure aucune : "
    ).strip()

    if metadata_cols_input:
        selected_metadata_cols = [
            col.strip() for col in metadata_cols_input.split(",")
            if col.strip() in headers
        ]
    else:
        selected_metadata_cols = []

    # --------------------------------------------------------------------------
    # 4. Filtrer les lignes pour ne conserver que celles déjà annotées
    #    (avec au moins un label)
    # --------------------------------------------------------------------------
    total_initial = len(data_rows)
    filtered_rows = []

    for row in data_rows:
        ligne_annotee = False
        # On boucle sur chaque colonne d'annotation
        for col in selected_annotation_cols:
            value = row.get(col, "").strip()
            if value:
                # Récupération et filtrage générique
                labels_found = parse_annotation_values(value, col)
                if labels_found:
                    # S'il y a au moins un label, on marque la ligne comme annotée
                    ligne_annotee = True
                    break
        if ligne_annotee:
            filtered_rows.append(row)

    data_rows = filtered_rows

    if not data_rows:
        print("Aucune ligne annotée (avec labels valides) détectée dans le CSV.")
        return

    print(f"{len(data_rows)} lignes annotées détectées sur {total_initial} lignes totales.\n")

    # --------------------------------------------------------------------------
    # 5. Répartition : Si plusieurs annotateurs, 20 % communes / 80 % uniques.
    #    Si un seul annotateur, toutes les données annotées sont utilisées.
    # --------------------------------------------------------------------------
    random.shuffle(data_rows)  # Mélange global

    if num_annotators == 1:
        # Un seul annotateur : aucune division
        common_data = data_rows  # Tout va à l'annotateur
        unique_data = []
        print(f"Un seul annotateur : {len(common_data)} lignes seront utilisées sans découpage.")
    else:
        total_count = len(data_rows)
        common_count = int(0.2 * total_count)
        unique_count = total_count - common_count

        common_data = data_rows[:common_count]   # 20 %
        unique_data = data_rows[common_count:]   # 80 %
        print(f"Plusieurs annotateurs : {total_count} lignes annotées au total.")
        print(f"Répartition : {common_count} lignes communes / {unique_count} lignes uniques.")

    # --------------------------------------------------------------------------
    # 6. Répartition des données uniques entre les annotateurs (si plusieurs)
    # --------------------------------------------------------------------------
    annotators_unique_data = []
    if num_annotators > 1:
        unique_count_per_annot = len(unique_data) // num_annotators

        start_idx = 0
        for _ in range(num_annotators):
            end_idx = start_idx + unique_count_per_annot
            annotators_unique_data.append(unique_data[start_idx:end_idx])
            start_idx = end_idx
        # Nota: il peut rester un reliquat si la division n'est pas exacte.
        # On pourrait l'attribuer aléatoirement à certains annotateurs, si besoin.
    else:
        annotators_unique_data = [[]]  # Cohérence pour un seul annotateur

    # --------------------------------------------------------------------------
    # 7. Extraction de tous les labels à partir des colonnes JSON sélectionnées
    # --------------------------------------------------------------------------
    all_labels = set()
    for row in data_rows:
        for col in selected_annotation_cols:
            value = row.get(col, "").strip()
            if value:
                labels_found = parse_annotation_values(value, col)
                for lbl in labels_found:
                    all_labels.add(lbl)

    all_labels = sorted(all_labels)

    # --------------------------------------------------------------------------
    # 8. Création du fichier de configuration des labels Doccano
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

    print(f"\nFichier de configuration Doccano créé à : {config_path}")

    # --------------------------------------------------------------------------
    # 9. Génération des fichiers JSONL (un par annotateur), en incluant
    #    les métadonnées sélectionnées
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

                # Récupération des labels (toutes clés confondues) pour chaque colonne
                for col in selected_annotation_cols:
                    value = row.get(col, "{}").strip()
                    labels_found = parse_annotation_values(value, col)
                    row_labels.extend(labels_found)

                # Ajout des métadonnées
                metadata_dict = {mcol: row.get(mcol, "") for mcol in selected_metadata_cols}

                doccano_line = {
                    "text": text,
                    "label": row_labels,
                    "metadata": metadata_dict
                }
                out_f.write(json.dumps(doccano_line, ensure_ascii=False) + "\n")

        annotators_file_paths.append(file_path)

    else:
        # Plusieurs annotateurs
        for i in range(num_annotators):
            annotator_id = i + 1
            file_name = f"annotator_{annotator_id}.jsonl"
            file_path = os.path.join(output_dir, file_name)

            # Combiner les données communes et les données uniques pour cet annotateur
            combined_data = common_data + annotators_unique_data[i]
            random.shuffle(combined_data)

            with open(file_path, mode="w", encoding="utf-8") as out_f:
                for row in combined_data:
                    text = row.get("body", "")
                    row_labels = []

                    for col in selected_annotation_cols:
                        value = row.get(col, "{}").strip()
                        labels_found = parse_annotation_values(value, col)
                        row_labels.extend(labels_found)

                    # Ajout des métadonnées
                    metadata_dict = {mcol: row.get(mcol, "") for mcol in selected_metadata_cols}

                    doccano_line = {
                        "text": text,
                        "label": row_labels,
                        "metadata": metadata_dict
                    }
                    out_f.write(json.dumps(doccano_line, ensure_ascii=False) + "\n")

            annotators_file_paths.append(file_path)

    # --------------------------------------------------------------------------
    # 10. Résumé et statistiques : affichage dans le terminal + export CSV
    # --------------------------------------------------------------------------
    print("\n===== Résumé de la répartition =====")
    print(f"Nombre d'annotateurs : {num_annotators}")

    if num_annotators == 1:
        print(f"Nombre total de lignes utilisées : {len(common_data)}")
    else:
        total_annotated = len(data_rows)
        print(f"Nombre total de lignes annotées : {total_annotated}")
        print(f"Lignes communes (20 %) : {len(common_data)}")
        print(f"Lignes uniques (80 %) : {len(unique_data)}")
        print(f"Lignes uniques par annotateur (approx.) : {len(unique_data) // num_annotators}")

    # Préparation du contenu pour le CSV de distribution
    distribution_rows = []
    header_distribution = ["Annotateur", "Fichier JSONL", "Nombre total de phrases"] + all_labels

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

        # Pour le CSV
        row_for_csv = [f"Annotateur_{i}", file_path, total_count] + [label_counts[lbl] for lbl in all_labels]
        distribution_rows.append(row_for_csv)

    distribution_csv_path = os.path.join(output_dir, "annotation_distribution_summary.csv")
    with open(distribution_csv_path, mode="w", encoding="utf-8", newline="") as dist_csv:
        writer = csv.writer(dist_csv, delimiter=",")
        writer.writerow(header_distribution)
        for row in distribution_rows:
            writer.writerow(row)

    print("\nUn rapport CSV de la distribution des annotations a été enregistré ici :")
    print(f"  {distribution_csv_path}")
    print("\nLes fichiers JSONL et doccano_config.json se trouvent dans :")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
