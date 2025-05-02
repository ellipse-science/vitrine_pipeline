"""
 PROJET : vitrine_pipeline

 TITRE : 2_JSONL_for_blind_annotation.py

 OBJECTIF PRINCIPAL :
 ---------------------
 Ce script génère des fichiers JSONL pour l’annotation manuelle (à l’aveugle) avec Doccano.

 CARACTÉRISTIQUES  :
 -------------------
 1) Le script parcourt un répertoire spécifique ("data/raw/subset") à la recherche de fichiers
    CSV. L’utilisateur choisit un des fichiers à traiter.

 2) L’utilisateur indique le nombre d’annotateurs. Deux cas se présentent :
    - Un annotateur unique : toutes les lignes vont dans un seul fichier JSONL (pas de découpage).
    - Plusieurs annotateurs : le script crée un partage 20 % / 80 %.
      * 20 % des lignes ("communes") sont dupliquées pour tous les annotateurs.
      * 80 % restants ("uniques") sont répartis équitablement et de manière aléatoire
        entre les annotateurs.

 3) Le script demande à l’utilisateur :
    - Quelle(s) colonne(s) du CSV contient le texte à annoter (en général, une seule).
      Seule la première colonne listée sera effectivement prise pour le champ "text" du JSONL.
      Les suivantes pourraient éventuellement être ignorées (ou servir d’autres champs).
    - Quelles colonnes seront incluses dans la clé "metadata" du JSONL.

 4) Le script produit pour chaque annotateur un fichier JSONL dans le format attendu
    par Doccano (où "text" correspond à la colonne texte choisie, et "metadata" est
    un dictionnaire comprenant les colonnes additionnelles sélectionnées).

 5) Les 20 % de lignes communes apparaissent toujours en premier dans les fichiers JSONL,
    suivies des lignes uniques de l’annotateur.

 6) Le script génère un fichier CSV de répartition ("annotation_distribution_summary.csv")
    dans le même répertoire que le fichier CSV d’origine. Ce fichier indique :
      - Le nombre total de lignes.
      - Le nombre de lignes communes.
      - Le nombre de lignes uniques.
      - Pour chaque annotateur : combien de lignes au total, dont combien de lignes communes
        et combien de lignes uniques.

 DÉPENDANCES :
 -------------
 - os
 - csv
 - json
 - random

 Auteur :
 ---------
 Antoine Lemor
"""

import os
import csv
import json
import random

def main():
    """
    Script principal pour générer des JSONL d’annotation à l’aveugle.
    Le flux d’exécution est le suivant :

        1) Lister et sélectionner un fichier CSV dans 'data/raw/subset'.
        2) Demander le nombre d’annotateurs.
        3) Demander quelles colonnes serviront de texte à annoter (on utilisera la première).
        4) Demander quelles colonnes (facultatif) seront placées dans 'metadata'.
        5) Découper les données :
           - Si un annotateur, toutes les lignes vont directement dans le JSONL unique.
           - Si plusieurs annotateurs, 20 % des lignes sont communes à tous les annotateurs,
             et 80 % sont réparties équitablement entre eux. Les lignes communes sont
             toujours en premier dans le JSONL, suivies des lignes uniques.
        6) Générer les fichiers JSONL pour chaque annotateur, un fichier de configuration
           Doccano (vide de labels), et un CSV de répartition dans le même répertoire que
           le fichier CSV sélectionné.
    """

    # --------------------------------------------------------------------------
    # 1. Recherche et sélection du fichier CSV dans "data/raw/subset"
    # --------------------------------------------------------------------------
    data_dir = "data/raw/subset"
    if not os.path.isdir(data_dir):
        print(f"Erreur : Le répertoire '{data_dir}' est introuvable.")
        return

    # Liste des fichiers CSV disponibles
    available_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    if not available_files:
        print(f"Aucun fichier CSV trouvé dans '{data_dir}'.")
        return

    print("Fichiers CSV disponibles dans 'data/raw/subset' :")
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

    csv_file_name = available_files[file_index]
    csv_file_path = os.path.join(data_dir, csv_file_name)

    # --------------------------------------------------------------------------
    # 2. Demander le nombre d’annotateurs
    # --------------------------------------------------------------------------
    try:
        num_annotators = int(input("Entrez le nombre d'annotateurs (entier positif) : ").strip())
        if num_annotators <= 0:
            raise ValueError
    except ValueError:
        print("Erreur : Nombre d'annotateurs invalide. Doit être un entier positif.")
        return

    # --------------------------------------------------------------------------
    # Lecture du CSV et stockage des lignes
    # --------------------------------------------------------------------------
    with open(csv_file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if not headers:
            print("Erreur : Impossible de lire les en-têtes du CSV.")
            return

        data_rows = list(reader)

    if not data_rows:
        print("Erreur : Le fichier CSV est vide ou mal formaté.")
        return

    total_rows = len(data_rows)
    print(f"\nLe fichier '{csv_file_name}' contient {total_rows} lignes (hors en-tête).\n")

    # --------------------------------------------------------------------------
    # 3. Demander quelles colonnes utiliser pour le texte
    #    (on n’en utilisera qu’une seule pour doccano_line["text"], la première listée)
    # --------------------------------------------------------------------------
    print("Colonnes disponibles :")
    for col in headers:
        print(f"  - {col}")

    text_cols_input = input(
        "\nEntrez la/les colonne(s) qui contiennent le texte à annoter (séparées par des virgules) : "
    ).strip()

    if not text_cols_input:
        print("Erreur : aucune colonne de texte spécifiée. Script interrompu.")
        return

    selected_text_cols = [c.strip() for c in text_cols_input.split(",") if c.strip() in headers]
    if not selected_text_cols:
        print("Erreur : aucune colonne de texte valide n'a été trouvée. Script interrompu.")
        return

    main_text_col = selected_text_cols[0]
    print(f"\nLa première colonne de texte utilisée sera : '{main_text_col}'.\n")

    # --------------------------------------------------------------------------
    # 4. Demander quelles colonnes inclure dans les métadonnées
    # --------------------------------------------------------------------------
    print("Rappel des colonnes disponibles :")
    for col in headers:
        print(f"  - {col}")

    metadata_cols_input = input(
        "\nEntrez les colonnes à inclure dans les métadonnées (séparées par des virgules), "
        "ou laissez vide pour n'en inclure aucune : "
    ).strip()

    if metadata_cols_input:
        selected_metadata_cols = [
            c.strip() for c in metadata_cols_input.split(",") if c.strip() in headers
        ]
    else:
        selected_metadata_cols = []

    # --------------------------------------------------------------------------
    # 5. Réaliser le découpage 20 % / 80 % si plusieurs annotateurs
    # --------------------------------------------------------------------------
    # Mélange aléatoire pour garantir le caractère aléatoire du choix commun/unique
    random.seed(42)  # seed pour reproductibilité
    random.shuffle(data_rows)

    if num_annotators == 1:
        # Pas de découpage
        common_data = data_rows
        unique_data = []
        print(f"\nUn seul annotateur : {len(common_data)} lignes seront annotées sans découpage.\n")
    else:
        total_count = len(data_rows)
        common_count = int(0.2 * total_count)
        unique_count = total_count - common_count

        common_data = data_rows[:common_count]
        unique_data = data_rows[common_count:]

        print(f"\nPlusieurs annotateurs : {total_count} lignes totales.")
        print(f"   → {common_count} lignes communes (20 %).")
        print(f"   → {unique_count} lignes uniques (80 %).\n")

    # --------------------------------------------------------------------------
    # 6. Répartir les lignes uniques entre les annotateurs si besoin
    # --------------------------------------------------------------------------
    if num_annotators > 1:
        unique_count_per_annot = unique_count // num_annotators
        annotators_unique_data = []
        start_idx = 0
        for i in range(num_annotators):
            end_idx = start_idx + unique_count_per_annot
            # Les annotateurs reçoivent chacun un segment de unique_data
            annotators_unique_data.append(unique_data[start_idx:end_idx])
            start_idx = end_idx

        # Si la division n'est pas exacte, il peut rester un reliquat
        # qu'on peut répartir un par un aux annotateurs suivants
        remainder = unique_count - (unique_count_per_annot * num_annotators)
        for i in range(remainder):
            annotators_unique_data[i].append(unique_data[start_idx])
            start_idx += 1

    else:
        # Pour un seul annotateur, pas de distribution
        annotators_unique_data = [list()]  # liste vide pour cohérence

    # --------------------------------------------------------------------------
    # 7. Générer les fichiers JSONL pour chaque annotateur
    #    Le dossier de sortie peut être, par exemple, "data/processed/validation"
    # --------------------------------------------------------------------------
    output_dir = "data/processed/validation"
    os.makedirs(output_dir, exist_ok=True)

    annotators_files = []
    for i in range(num_annotators):
        annotator_id = i + 1
        file_name = f"annotator_{annotator_id}.jsonl"
        out_path = os.path.join(output_dir, file_name)

        # Combiner : lignes communes (en premier) + lignes uniques
        if num_annotators == 1:
            combined_data = common_data
        else:
            combined_data = common_data + annotators_unique_data[i]

        # NE PAS random.shuffle ici, car on veut que les communes soient en premier

        with open(out_path, mode="w", encoding="utf-8") as out_f:
            for row in combined_data:
                text_value = row.get(main_text_col, "")

                # Construction d'un dictionnaire de métadonnées
                metadata_dict = {}
                for meta_col in selected_metadata_cols:
                    metadata_dict[meta_col] = row.get(meta_col, "")

                # Ligne au format Doccano
                # Annotation à l’aveugle => pas de "label"
                doccano_line = {
                    "text": text_value,
                    "metadata": metadata_dict
                }

                out_f.write(json.dumps(doccano_line, ensure_ascii=False) + "\n")

        annotators_files.append(out_path)

    print("Fichiers JSONL créés :")
    for path in annotators_files:
        print(f"  - {path}")

    # --------------------------------------------------------------------------
    # 8. Générer un fichier de configuration Doccano (vide, sans labels)
    # --------------------------------------------------------------------------
    config = {"label": []}  # Aucune liste de labels, annotation à l’aveugle
    config_path = os.path.join(output_dir, "doccano_config.json")
    with open(config_path, mode="w", encoding="utf-8") as cf:
        json.dump(config, cf, ensure_ascii=False, indent=2)

    print(f"\nFichier de configuration Doccano (vide) créé : {config_path}")

    # --------------------------------------------------------------------------
    # 9. Générer le CSV de distribution dans le même répertoire que le CSV source
    # --------------------------------------------------------------------------
    summary_csv_name = "annotation_distribution_summary.csv"
    summary_csv_path = os.path.join(data_dir, summary_csv_name)

    # Calculs de répartition globaux
    total_lines = len(data_rows)
    num_common = len(common_data)
    num_unique = total_lines - num_common

    # En-tête : Annotateur, Fichier, Nb total de lignes, Nb lignes communes, Nb lignes uniques
    with open(summary_csv_path, mode="w", encoding="utf-8", newline="") as dist_csv:
        columns = [
            "Annotateur",
            "Fichier JSONL",
            "Nombre total de lignes",
            "Dont lignes communes",
            "Dont lignes uniques"
        ]
        writer = csv.writer(dist_csv, delimiter=",")
        writer.writerow(["Nombre total de lignes CSV", total_lines])
        writer.writerow(["Nombre de lignes communes (20%)", num_common])
        writer.writerow(["Nombre de lignes uniques (80%)", num_unique])
        writer.writerow([])  # ligne vide
        writer.writerow(columns)

        for i, jsonl_path in enumerate(annotators_files, start=1):
            if num_annotators == 1:
                # Tout est commun
                lines_for_annot = len(common_data)
                common_for_annot = lines_for_annot
                unique_for_annot = 0
            else:
                lines_for_annot = len(common_data) + len(annotators_unique_data[i-1])
                common_for_annot = len(common_data)
                unique_for_annot = len(annotators_unique_data[i-1])

            row_data = [
                f"Annotateur_{i}",
                jsonl_path,
                lines_for_annot,
                common_for_annot,
                unique_for_annot
            ]
            writer.writerow(row_data)

    print(f"\nUn fichier de synthèse de la répartition a été enregistré ici : {summary_csv_path}")
    print("\nTerminé !")


if __name__ == "__main__":
    main()
