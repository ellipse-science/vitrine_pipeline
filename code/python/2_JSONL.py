"""
# ---------------------------------------------------------------------------------
# PROJET : vitrine_pipeline
#
# TITRE : 2_JSONL.py
#
# OBJECTIF PRINCIPAL :
# ---------------------
# Ce script génère des fichiers JSONL pour l’annotation avec Doccano en utilisant
# un fichier CSV contenant le texte et des annotations préexistantes. Il assure une
# répartition de 50 % d’entrées en anglais (EN) et 50 % en français (FR), en
# sélectionnant 20 % de phrases communes à tous les annotateurs et 80 % de phrases
# uniques par annotateur.
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
# 2) Lit et équilibre les données du CSV (50 % EN, 50 % FR) et les divise en
#    parties communes (20 %) et uniques (80 %) [si plusieurs annotateurs].
# 3) Génère un fichier de configuration pour Doccano contenant les labels.
# 4) Crée pour chaque annotateur un fichier JSONL contenant les phrases et leurs
#    labels (au format Doccano).
# 5) Affiche un résumé statistique de la répartition (nombre de phrases, langues,
#    distribution des labels, etc.).
#
# Auteur :
# ---------
# Antoine Lemor
# ---------------------------------------------------------------------------------
"""

import os
import json
import random
import math
import csv

def main():
    """
    Fonction principale pour orchestrer la création des fichiers JSONL et du fichier
    de configuration Doccano.
    
    Étapes :
        1) Demander à l'utilisateur le chemin du fichier CSV et le nombre d'annotateurs.
        2) Lire le CSV et filtrer les données pour assurer une répartition 50 % EN et 50 % FR.
        3) Si plusieurs annotateurs, diviser les données en 20 % communes et 80 % uniques.
           Si un seul annotateur, prendre toutes les données équilibrées.
        4) Demander quelles colonnes contiennent les annotations JSON ("_annotated" par ex.).
           Créer un fichier de configuration Doccano avec tous les labels, suffixés par le
           nom de la colonne.
        5) Générer les fichiers JSONL (un par annotateur) avec la répartition correcte.
        6) Afficher un résumé statistique des ensembles d'annotation.
    """
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
    # 2. Lecture des données depuis le CSV et stockage des lignes
    #    On suppose qu'au moins une colonne 'lang' (pour EN/FR) et une
    #    colonne 'body' (texte principal) existent.
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
    # Demander à l'utilisateur quelles colonnes contiennent les annotations JSON
    # (typiquement "body_annotated", "comment_annotated", etc.)
    # Les labels extraits de la colonne X seront suffixés par "_X".
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
    # 3. Séparer les lignes par langue (EN, FR) pour assurer une répartition 50-50
    # --------------------------------------------------------------------------
    en_rows = [r for r in data_rows if r.get("lang", "").upper() == "EN"]
    fr_rows = [r for r in data_rows if r.get("lang", "").upper() == "FR"]

    if not en_rows or not fr_rows:
        print("Attention : Au moins un ensemble (EN ou FR) est vide. "
              "La répartition 50-50 ne peut être assurée.")
        print("Arrêt du script pour éviter la création de répartitions incomplètes.")
        return

    # Randomise et garde la proportion 50-50
    random.shuffle(en_rows)
    random.shuffle(fr_rows)
    min_count = min(len(en_rows), len(fr_rows))
    en_rows = en_rows[:min_count]
    fr_rows = fr_rows[:min_count]

    balanced_data = en_rows + fr_rows
    random.shuffle(balanced_data)

    total_balanced = len(balanced_data)
    print(f"{total_balanced} lignes sélectionnées au total (50 % EN, 50 % FR).")

    # --------------------------------------------------------------------------
    # 4. Si un seul annotateur, pas de répartition 20 % / 80 %, on prend tout.
    #    Sinon, on répartit 20 % communes + 80 % uniques comme initialement.
    # --------------------------------------------------------------------------
    if num_annotators == 1:
        # Pas de découpage. Toutes les données vont à l'annotateur unique
        common_data = balanced_data  # On considère tout comme "common"
        unique_data = []
        print("Un seul annotateur : toutes les données équilibrées seront utilisées sans pondération.")
    else:
        # Division en 20 % communes / 80 % uniques
        common_count = int(0.2 * total_balanced)
        unique_count = total_balanced - common_count

        # Randomise encore pour éviter les biais
        random.shuffle(balanced_data)
        common_data = balanced_data[:common_count]
        unique_data = balanced_data[common_count:]

        # S'assurer que les données communes sont équilibrées EN-FR
        common_en = [r for r in common_data if r.get("lang", "").upper() == "EN"]
        common_fr = [r for r in common_data if r.get("lang", "").upper() == "FR"]
        random.shuffle(common_en)
        random.shuffle(common_fr)

        half_common = common_count // 2
        picked_en_for_common = min(half_common, len(common_en))
        picked_fr_for_common = min(half_common, len(common_fr))

        final_common_en = common_en[:picked_en_for_common]
        final_common_fr = common_fr[:picked_fr_for_common]
        final_common_data = final_common_en + final_common_fr

        needed_for_common = common_count - len(final_common_data)
        if needed_for_common > 0:
            leftover_en = common_en[picked_en_for_common:]
            leftover_fr = common_fr[picked_fr_for_common:]
            combined_leftover = leftover_en + leftover_fr
            random.shuffle(combined_leftover)
            final_common_data += combined_leftover[:needed_for_common]

        common_data = final_common_data
        actual_common_count = len(common_data)

    # --------------------------------------------------------------------------
    # 5. (si plusieurs annotateurs) Répartir les données uniques entre les annotateurs
    #    en garantissant 50-50 EN-FR
    # --------------------------------------------------------------------------
    annotators_unique_data = []
    if num_annotators > 1:
        unique_en = [r for r in unique_data if r.get("lang", "").upper() == "EN"]
        unique_fr = [r for r in unique_data if r.get("lang", "").upper() == "FR"]
        random.shuffle(unique_en)
        random.shuffle(unique_fr)

        unique_count = len(unique_data)
        unique_count_per_annot = unique_count // num_annotators
        half_unique_per_annot = unique_count_per_annot // 2

        for _ in range(num_annotators):
            en_slice = unique_en[:half_unique_per_annot]
            fr_slice = unique_fr[:half_unique_per_annot]
            combined_slice = en_slice + fr_slice
            annotators_unique_data.append(combined_slice)

            # Enlève les lignes déjà attribuées
            unique_en = unique_en[half_unique_per_annot:]
            unique_fr = unique_fr[half_unique_per_annot:]
    else:
        # Un seul annotateur => pas de notion d'unique par annotateur
        annotators_unique_data = [[]]  # Liste vide pour rester cohérent

    # --------------------------------------------------------------------------
    # 6. Rassembler tous les labels en analysant les colonnes sélectionnées
    #    pour les annotations JSON. Les labels sont suffixés par le nom de la colonne.
    # --------------------------------------------------------------------------
    all_labels = set()
    for row in data_rows:
        for col in selected_annotation_cols:
            try:
                annotated_json = json.loads(row.get(col, "{}"))
                themes = annotated_json.get("themes", [])
                for theme in themes:
                    # Suffixe le label par le nom de la colonne (ex: politics_body_annotated)
                    label_with_suffix = f"{theme}_{col}"
                    all_labels.add(label_with_suffix)
            except json.JSONDecodeError:
                # Passer les lignes mal formatées
                continue

    all_labels = sorted(all_labels)

    # --------------------------------------------------------------------------
    # 7. Création du fichier de configuration des labels Doccano
    # --------------------------------------------------------------------------
    # Enregistrement dans un dossier "data/processed/validation"
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
    # 8. Générer les fichiers JSONL pour chaque annotateur
    #    (données communes + uniques). Si un seul annotateur, on a tout dans un seul fichier.
    # --------------------------------------------------------------------------
    annotators_file_paths = []
    if num_annotators == 1:
        # Un annotateur : on combine tout dans un unique fichier
        single_annotator_data = common_data  # qui contient déjà toutes les données équilibrées
        file_name = f"annotator_1.jsonl"
        file_path = os.path.join(output_dir, file_name)
        random.shuffle(single_annotator_data)

        with open(file_path, mode="w", encoding="utf-8") as out_f:
            for row in single_annotator_data:
                text = row.get("body", "")
                row_labels = []
                # Récupérer les labels sur toutes les colonnes sélectionnées
                for col in selected_annotation_cols:
                    try:
                        ann_json = json.loads(row.get(col, "{}"))
                        themes = ann_json.get("themes", [])
                        for theme in themes:
                            label_with_suffix = f"{theme}_{col}"
                            row_labels.append(label_with_suffix)
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

            # Combine les données communes et uniques pour chaque annotateur
            combined_annotator_data = common_data + annotators_unique_data[i]
            random.shuffle(combined_annotator_data)

            with open(file_path, mode="w", encoding="utf-8") as out_f:
                for row in combined_annotator_data:
                    text = row.get("body", "")
                    row_labels = []
                    # Récupérer les labels sur toutes les colonnes sélectionnées
                    for col in selected_annotation_cols:
                        try:
                            ann_json = json.loads(row.get(col, "{}"))
                            themes = ann_json.get("themes", [])
                            for theme in themes:
                                label_with_suffix = f"{theme}_{col}"
                                row_labels.append(label_with_suffix)
                        except json.JSONDecodeError:
                            continue

                    doccano_line = {"text": text, "label": row_labels}
                    out_f.write(json.dumps(doccano_line, ensure_ascii=False) + "\n")

            annotators_file_paths.append(file_path)

    # --------------------------------------------------------------------------
    # 9. Afficher un résumé de la répartition dans le terminal
    # --------------------------------------------------------------------------
    print("\n===== Résumé de la répartition =====")
    print(f"Nombre d'annotateurs : {num_annotators}")
    print(f"Lignes totales utilisées (équilibrées) : {total_balanced}")

    if num_annotators > 1:
        print(f"Lignes communes (partagées par tous) : {len(common_data)}")
        print(f"Lignes uniques totales : {len(unique_data)}")
        print(f"Lignes uniques par annotateur (approx.) : {len(unique_data) // num_annotators}")
    else:
        print("Aucune répartition 20 % / 80 % car un seul annotateur.")

    # Lis chaque fichier JSONL pour afficher les statistiques
    for i, file_path in enumerate(annotators_file_paths, start=1):
        with open(file_path, mode="r", encoding="utf-8") as f_in:
            lines = f_in.readlines()

        total_count = len(lines)
        en_count, fr_count = 0, 0
        # Initialiser le compteur de labels
        label_counts = {lbl: 0 for lbl in all_labels}

        for line in lines:
            record = json.loads(line)
            text = record.get("text", "")
            labels_list = record.get("label", [])
            # Comptage des labels
            for lbl in labels_list:
                if lbl in label_counts:
                    label_counts[lbl] += 1

            # Très approximatif pour distinguer EN/FR (selon mots clés)
            if any(word in text.lower() for word in [" le ", " la ", " les ", " être ", " c'est ", " suis "]):
                fr_count += 1
            else:
                en_count += 1

        print(f"\nAnnotateur {i} :")
        print(f"  Fichier JSONL : {file_path}")
        print(f"  Nombre total de phrases : {total_count}")
        print(f"  (Approx.) Phrases en FR : {fr_count}, en EN : {en_count}")
        print("  Répartition des labels :")
        for lbl, cnt in label_counts.items():
            print(f"    {lbl} : {cnt}")

    print("\nLes fichiers JSONL et doccano_config.json se trouvent dans :")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
