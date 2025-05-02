"""
PROJET :
--------
Vitrine_pipeline

TITRE :
--------
3_metrics.py (version "subtests" - corrigée pour le calcul du temps moyen)

OBJECTIF PRINCIPAL :
---------------------
Ce script compare les annotations humaines (JSONL) aux prédictions générées par le modèle (CSV) 
dans un contexte multi-étiquette. 

DÉPENDANCES :
-------------
- os
- csv
- json
- glob
- collections (defaultdict)

FONCTIONNALITÉS PRINCIPALES :
-----------------------------
1) Charger les annotations gold depuis un fichier JSONL.
   - Les labels dans le JSONL se terminent souvent par "_body_annotated_deepseek_r1". 
     On les nettoie pour enlever ce suffixe afin d'harmoniser les noms.
   - "null_body_annotated_deepseek_r1" devient "null".
2) Demander à l’utilisateur quelles colonnes du CSV contiennent les prédictions au format JSON.
   - Pour chaque colonne sélectionnée, on suppose l’existence d’une colonne de temps associée 
     (même nom + "_inference_time").
3) Charger et traiter les prédictions du modèle :
   - On combine les étiquettes de toutes les colonnes (union).
   - On somme le temps total (somme des colonnes) pour chaque texte (afin d’avoir un temps global).
   - Pour chaque colonne, on répartit son temps d’inférence sur ses labels prédits.
4) Calculer des métriques multi-étiquettes (précision, rappel, F1 micro/macro, subset accuracy).
5) Calculer des indicateurs de temps (moyenne par label, moyenne globale, moyenne macro, 
   moyenne pondérée, temps total absolu).
6) Sauvegarder les métriques et temps dans un fichier CSV par fichier JSONL, incluant la ligne de temps total.

Auteur :
--------
Antoine Lemor 
"""

import os
import csv
import json
import glob
from collections import defaultdict

# Étiquette spéciale indiquant « aucune catégorie » (on retire le suffixe pour harmoniser)
NULL_LABEL = "null"


def load_gold_annotations_from_jsonl(jsonl_path):
    """
    Charge les annotations gold humaines à partir d'un fichier JSONL.

    Paramètres :
    ------------
    jsonl_path : str
        Chemin vers le fichier JSONL contenant les annotations gold.

    Renvoie :
    ---------
    dict
        Dictionnaire où chaque clé est le texte (string) et chaque valeur est un ensemble d'étiquettes 
        (le suffixe "_body_annotated_deepseek_r1" est retiré).
        Exemple :
            {
                "un texte": {"law_and_crime"},
                "autre texte": {"null"}
            }
    """
    gold_dict = {}
    with open(jsonl_path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                labels = obj.get("label", [])

                valid_labels = set()
                for lbl in labels:
                    if lbl.endswith("_body_annotated_deepseek_r1"):
                        cleaned = lbl.replace("_body_annotated_deepseek_r1", "")
                        # On remplace également "null_body_annotated_deepseek_r1" par "null"
                        cleaned = cleaned if cleaned != "null" else NULL_LABEL
                        valid_labels.add(cleaned)

                gold_dict[text] = valid_labels
            except json.JSONDecodeError:
                # Ignorer les lignes invalides
                continue
    return gold_dict


def select_annotation_columns(csv_file_path):
    """
    Affiche les colonnes disponibles dans le CSV et demande à l’utilisateur 
    de sélectionner celles contenant les prédictions JSON.

    Paramètres :
    ------------
    csv_file_path : str
        Chemin vers le fichier CSV contenant les prédictions du modèle.

    Renvoie :
    ---------
    list
        Liste des noms de colonnes sélectionnées par l'utilisateur pour les prédictions JSON.
        Exemple : ["col1", "col3"].
        Si aucune colonne n'est sélectionnée ou en cas d'entrée invalide, renvoie une liste vide.
    """
    with open(csv_file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

    if not headers:
        return []

    print("\nColonnes disponibles dans le CSV :")
    for idx, col in enumerate(headers, start=1):
        print(f"  {idx}. {col}")

    user_input = input(
        "\nEntrez les numéros des colonnes contenant les prédictions JSON (séparés par des virgules) : "
    ).strip()
    if not user_input:
        print("Aucune colonne sélectionnée. Fin du programme.")
        return []

    selected_indices = []
    for val in user_input.split(","):
        val = val.strip()
        try:
            idx = int(val)
            if 1 <= idx <= len(headers):
                selected_indices.append(idx)
        except ValueError:
            # Ignorer les entrées non numériques
            continue

    selected_cols = [headers[i - 1] for i in selected_indices]
    return selected_cols


def load_model_annotations_from_csv(csv_path, annotation_cols):
    """
    Charge les prédictions du modèle à partir d'un fichier CSV, en combinant les étiquettes 
    de toutes les colonnes sélectionnées et en stockant les temps d'inférence de manière détaillée.

    Mécanisme (nouveau mode de répartition du temps) :
    --------------------------------------------------
    - Pour chaque ligne (texte), on va créer une structure model_dict[text]["columns"] 
      qui contiendra la liste des colonnes utilisées (pour le texte) avec :
         {
            "labels": set() des labels prédits par cette colonne,
            "time": (float) temps d'inférence associé à cette colonne
         }
    - On cumulera également un "total_inference_time" pour le texte (somme des temps de toutes les colonnes).
    - Ainsi, lors du calcul final, on pourra répartir le temps de chaque colonne 
      uniquement sur les labels que cette colonne a prédits (plutôt que d’attribuer la somme totale à chacun).

    Paramètres :
    ------------
    csv_path : str
        Chemin vers le fichier CSV contenant les prédictions du modèle.
    annotation_cols : list
        Liste des noms de colonnes contenant les prédictions au format JSON.

    Renvoie :
    ---------
    dict
        Dictionnaire associant chaque texte (colonne "body") à :
            {
                "columns": [
                    {
                       "labels": set(...),
                       "time": float
                    },
                    ...
                ],
                "total_inference_time": float
            }
    """
    inference_cols = {
        col: col + "_inference_time" for col in annotation_cols
    }

    model_dict = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("body", "").strip()
            if not text:
                continue

            if text not in model_dict:
                model_dict[text] = {
                    "columns": [],
                    "total_inference_time": 0.0
                }

            # Pour chaque colonne de prédiction
            for col in annotation_cols:
                json_str = row.get(col, "").strip()
                time_str = ""
                if inference_cols.get(col) in row:
                    time_str = row[inference_cols[col]].strip()

                col_time = 0.0
                if time_str:
                    try:
                        col_time = float(time_str)
                    except ValueError:
                        col_time = 0.0

                # Extrait les labels de cette colonne
                predicted_labels = set()
                if json_str:
                    try:
                        ann_json = json.loads(json_str)
                        theme_val = ann_json.get("themes")
                        if isinstance(theme_val, str):
                            if theme_val.lower() != "null":
                                predicted_labels.add(theme_val.strip())
                        elif isinstance(theme_val, list):
                            for tv in theme_val:
                                if isinstance(tv, str) and tv.lower() != "null":
                                    predicted_labels.add(tv.strip())
                    except json.JSONDecodeError:
                        pass

                # Stocke l'ensemble des labels pour cette colonne + le temps associé
                model_dict[text]["columns"].append({
                    "labels": predicted_labels,
                    "time": col_time
                })
                # On ajoute ce col_time au total du texte
                model_dict[text]["total_inference_time"] += col_time

    # Si un texte n'a aucun label prédit sur l'ensemble des colonnes, nous le considérons comme "null".
    # (Ici, on peut vérifier après coup dans compute_metrics également, 
    #  mais on peut le faire ici si on souhaite que tous les textes aient au moins un label.)
    return model_dict


def compute_metrics(gold_dict, model_dict):
    """
    Calcule les métriques multi-étiquettes et les temps d'inférence :
      - précision, rappel, F1 (micro/macro)
      - subset accuracy
      - temps d'inférence moyen global
      - temps d'inférence moyen par label (macro, pondéré)
      - temps total d'inférence (somme de tous les temps d'inférence sur l'intersection de textes)

    Nouveau : 
    ---------
    Le calcul du temps moyen par label évite la surestimation. 
    Pour chaque texte, on répartit le temps de chaque colonne uniquement sur les labels 
    que cette colonne a prédits (en le divisant si plusieurs labels ont été prédits par la même colonne).

    Paramètres :
    ------------
    gold_dict : dict
        { texte: set(labels_gold), ... }
    model_dict : dict
        { 
           texte: {
             "columns": [
                {"labels": set(...), "time": float}, ...
             ],
             "total_inference_time": float
           },
           ...
        }

    Renvoie :
    ---------
    dict
        {
          "per_label": {
             label: {
               "precision": float, "recall": float, "f1": float,
               "tp": int, "fp": int, "fn": int,
               "gold_count": int, "pred_count": int,
               "avg_inference_time": float
             },
             ...
          },
          "micro": {"precision": float, "recall": float, "f1": float},
          "macro": {"precision": float, "recall": float, "f1": float},
          "subset_accuracy": float,
          "overall_avg_inference_time": float,
          "macro_avg_inference_time": float,
          "weighted_avg_inference_time": float,
          "total_inference_time": float
        }
    """

    # Textes présents à la fois dans gold_dict et model_dict
    common_texts = set(gold_dict.keys()) & set(model_dict.keys())
    if not common_texts:
        return {
            "per_label": {},
            "micro": {"precision": 0, "recall": 0, "f1": 0},
            "macro": {"precision": 0, "recall": 0, "f1": 0},
            "subset_accuracy": 0,
            "overall_avg_inference_time": 0,
            "macro_avg_inference_time": 0,
            "weighted_avg_inference_time": 0,
            "total_inference_time": 0
        }

    label_tp = defaultdict(int)
    label_fp = defaultdict(int)
    label_fn = defaultdict(int)
    gold_label_count = defaultdict(int)
    pred_label_count = defaultdict(int)

    # Sommes pour calcul du temps d'inférence
    label_inference_sum = defaultdict(float)
    label_inference_count = defaultdict(int)

    total_docs = 0
    correct_docs = 0
    total_inference_sum = 0.0

    for text in common_texts:
        gold_labels = gold_dict[text]

        # Récupère la liste des colonnes + le temps total
        info = model_dict[text]
        col_list = info["columns"]
        text_total_time = info["total_inference_time"]
        total_inference_sum += text_total_time

        # Construit l'ensemble des labels prédits en union de toutes les colonnes
        all_predicted_labels = set()
        for col_data in col_list:
            all_predicted_labels.update(col_data["labels"])

        # Si aucun label n'a été prédit globalement, on assigne "null"
        if not all_predicted_labels:
            all_predicted_labels = {NULL_LABEL}

        # Subset accuracy
        total_docs += 1
        if gold_labels == all_predicted_labels:
            correct_docs += 1

        # Comptes or / prédictions
        for gl in gold_labels:
            gold_label_count[gl] += 1
        for pl in all_predicted_labels:
            pred_label_count[pl] += 1

        # TP / FN
        for lbl in gold_labels:
            if lbl in all_predicted_labels:
                label_tp[lbl] += 1
            else:
                label_fn[lbl] += 1

        # FP
        for lbl in all_predicted_labels:
            if lbl not in gold_labels:
                label_fp[lbl] += 1

        # Répartition du temps d'inférence par label
        # Pour chaque colonne, si elle prédit n labels, on divise col_time par n pour chacun.
        for col_data in col_list:
            col_time = col_data["time"]
            predicted_labels_in_col = col_data["labels"]

            # S'il n'y a pas de label dans cette colonne, on ne répartit pas
            if not predicted_labels_in_col:
                continue

            share_time = 0.0
            if len(predicted_labels_in_col) > 0:
                share_time = col_time / len(predicted_labels_in_col)

            for lbl in predicted_labels_in_col:
                label_inference_sum[lbl] += share_time
                # On incrémente le count d'occurrences documentaires
                # (c.-à-d. "Ce label a été prédit au moins une fois pour ce texte")
                # Dans la logique habituelle : le "count" = nb de textes où le label est prédit.
                # On ne veut pas incrémenter plusieurs fois si le label apparaît dans plusieurs colonnes ?
                # => On l’incrémente UNE SEULE fois par texte, si le label a été prédit par au moins une colonne.
                # Mais ici, pour la moyenne "temps par label", on considère la contribution par colonne.
                # Selon la conception habituelle, on préfère cependant utiliser le "count" standard (docs).
                # => On va incrémenter label_inference_count[lbl] = 1 si on n'a pas déjà compté ce texte.
                #    Sinon, on faussera la division. 
                # Pour simplifier, on considère qu’on cumule la répartition de temps “réelle”.
                # Ensuite, la moyenne = somme(temps) / nombre_total_de_prédictions_label (somme sur toutes colonnes).
                # => On a besoin du total “pred_count” (celui de la micro-stat) OU on compte col par col. 

                # Ici on va faire : label_inference_count[lbl] += 1
                # mais comme on veut coller à la sémantique "avg_time = somme / nb_texts_qui_pred_label",
                # on pourrait seulement incrémenter si c'est la 1ère fois qu'on voit ce label sur ce texte.
                # => Or, la version précédente incrémentait 1 par doc. 
                # => On va distinguer : "label_text_seen" local
                #    pour n'incrémenter qu'une fois par texte.

                # Néanmoins, l'utilisateur veut souvent un "average_inference_time" par 
                # "occurrence" (i.e. si 1 doc -> 3 colonnes -> 3 * share_time). 
                # On clarifie : la version ancienne comptait 1 par label si label prédit. 
                # On rétablit la version plus "granulaire" : 
                #    c.-à-d. la somme sera partagée, 
                #    le "count" sera incrémenté 1 fois pour chaque instance. 
                # => Ainsi, avg_inference_time = (somme des temps) / (nb total de "prédictions effectives"). 
                
                label_inference_count[lbl] += 1

    subset_accuracy = correct_docs / total_docs if total_docs else 0.0
    all_labels = set(label_tp.keys()) | set(label_fp.keys()) | set(label_fn.keys())

    per_label = {}
    for lbl in sorted(all_labels):
        tp = label_tp[lbl]
        fp = label_fp[lbl]
        fn = label_fn[lbl]
        gc = gold_label_count[lbl]
        pc = pred_label_count[lbl]

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

        avg_inf_time = 0.0
        if label_inference_count[lbl] > 0:
            avg_inf_time = label_inference_sum[lbl] / label_inference_count[lbl]

        per_label[lbl] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gold_count": gc,
            "pred_count": pc,
            "avg_inference_time": avg_inf_time
        }

    # Micro (basé sur la somme globale de TP, FP, FN)
    total_tp = sum(label_tp.values())
    total_fp = sum(label_fp.values())
    total_fn = sum(label_fn.values())
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0

    # Macro
    n_labels = len(all_labels) if all_labels else 1
    macro_precision = sum(d["precision"] for d in per_label.values()) / n_labels
    macro_recall = sum(d["recall"] for d in per_label.values()) / n_labels
    macro_f1 = sum(d["f1"] for d in per_label.values()) / n_labels

    # Temps d'inférence moyen global (overall)
    overall_avg_inference_time = total_inference_sum / total_docs if total_docs else 0.0

    # Moyenne des moyennes par label (macro)
    label_avgs = [d["avg_inference_time"] for d in per_label.values() if d["pred_count"] > 0]
    if label_avgs:
        macro_avg_inference_time = sum(label_avgs) / len(label_avgs)
    else:
        macro_avg_inference_time = 0.0

    # Moyenne pondérée par le nombre de prédictions (weighted)
    total_pred_count = sum(d["pred_count"] for d in per_label.values())
    weighted_sum = 0.0
    for d in per_label.values():
        weighted_sum += d["avg_inference_time"] * d["pred_count"]
    weighted_avg_inference_time = weighted_sum / total_pred_count if total_pred_count else 0.0

    return {
        "per_label": per_label,
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        },
        "subset_accuracy": subset_accuracy,
        "overall_avg_inference_time": overall_avg_inference_time,
        "macro_avg_inference_time": macro_avg_inference_time,
        "weighted_avg_inference_time": weighted_avg_inference_time,
        "total_inference_time": total_inference_sum
    }


def save_metrics_to_csv(results, output_path, jsonl_filename):
    """
    Sauvegarde les métriques calculées dans un fichier CSV, 
    en incluant le temps d'inférence moyen (par label, global, etc.)
    et le temps total absolu.

    Le fichier CSV contiendra :
      - Une ligne par étiquette : label, gold_count, pred_count, precision, recall, f1, avg_inference_time
      - Des lignes spéciales :
          __micro__,
          __macro__,
          __subset_accuracy__,
          __weighted__  (temps moyen pondéré),
          __total_inference_time__ (somme absolue de tous les temps)
    
    Paramètres :
    ------------
    results : dict
        Dictionnaire produit par compute_metrics(), contenant statistiques par label et globales.
    output_path : str
        Chemin de destination du fichier CSV de sortie.
    jsonl_filename : str
        Nom du fichier JSONL traité (pour référence dans la colonne "file").
    """
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", 
            "label", 
            "gold_count", 
            "pred_count",
            "precision", 
            "recall", 
            "f1",
            "avg_inference_time"
        ])

        # 1) Détails par label
        for lbl, vals in results["per_label"].items():
            writer.writerow([
                jsonl_filename,
                lbl,
                vals["gold_count"],
                vals["pred_count"],
                round(vals["precision"], 4),
                round(vals["recall"], 4),
                round(vals["f1"], 4),
                round(vals["avg_inference_time"], 4)
            ])

        # 2) __micro__
        writer.writerow([
            jsonl_filename,
            "__micro__",
            "",
            "",
            round(results["micro"]["precision"], 4),
            round(results["micro"]["recall"], 4),
            round(results["micro"]["f1"], 4),
            round(results["overall_avg_inference_time"], 4)
        ])

        # 3) __macro__
        writer.writerow([
            jsonl_filename,
            "__macro__",
            "",
            "",
            round(results["macro"]["precision"], 4),
            round(results["macro"]["recall"], 4),
            round(results["macro"]["f1"], 4),
            round(results["macro_avg_inference_time"], 4)
        ])

        # 4) __subset_accuracy__
        writer.writerow([
            jsonl_filename,
            "__subset_accuracy__",
            "",
            "",
            round(results["subset_accuracy"], 4),
            "",
            "",
            ""
        ])

        # 5) __weighted__ (temps moyen pondéré)
        writer.writerow([
            jsonl_filename,
            "__weighted__",
            "",
            "",
            "",
            "",
            "",
            round(results["weighted_avg_inference_time"], 4)
        ])

        # 6) __total_inference_time__
        writer.writerow([
            jsonl_filename,
            "__total_inference_time__",
            "",
            "",
            "",
            "",
            "",
            round(results["total_inference_time"], 4)
        ])


def main():
    """
    Exécution principale :

    1) Demande le chemin vers le fichier CSV (ou utilise un chemin par défaut).
    2) Demande à l'utilisateur quelles colonnes contiennent les prédictions JSON.
       Chaque colonne a normalement une colonne associée <col>_inference_time.
    3) Charge les prédictions du modèle, en stockant le temps d’inférence 
       pour chaque colonne afin de pouvoir le répartir correctement.
    4) Pour chaque fichier .jsonl dans le dossier des gold :
       - Charge les annotations gold.
       - Calcule les métriques (y compris la répartition correcte des temps).
       - Sauvegarde les résultats dans un fichier CSV.
    """
    print("Chemin par défaut : data/processed/subset/subtest/...")
    csv_subtest_path = input(
        "Entrez le chemin vers le fichier CSV (ou laissez vide pour le chemin par défaut) : "
    ).strip()

    if not csv_subtest_path:
        csv_subtest_path = os.path.join(
            "data", "processed", "subset", "subtest", "predictions.csv"
        )

    while not os.path.isfile(csv_subtest_path):
        print(f"Fichier non trouvé : {csv_subtest_path}")
        csv_subtest_path = input("Veuillez réessayer : ").strip()

    # 2) Choix des colonnes d'annotations
    annotation_cols = select_annotation_columns(csv_subtest_path)
    if not annotation_cols:
        return

    # 3) Charger les prédictions
    model_dict = load_model_annotations_from_csv(csv_subtest_path, annotation_cols)

    # 4) Calcul et sauvegarde des métriques, pour chaque fichier JSONL du répertoire
    gold_dir = os.path.join("data", "processed", "validation", "annotated_jsonl")
    jsonl_files = glob.glob(os.path.join(gold_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"Aucun fichier JSONL trouvé dans {gold_dir}")
        return

    for jsonl_path in jsonl_files:
        jsonl_filename = os.path.basename(jsonl_path)

        # Charger les annotations gold
        gold_dict = load_gold_annotations_from_jsonl(jsonl_path)

        # Calculer les métriques
        results = compute_metrics(gold_dict, model_dict)

        # Sauvegarder dans un fichier CSV
        output_csv = os.path.join(
            gold_dir, jsonl_filename.replace(".jsonl", "_metrics.csv")
        )
        save_metrics_to_csv(results, output_csv, jsonl_filename)
        print(f"Métriques calculées pour {jsonl_filename} -> {output_csv}")


if __name__ == "__main__":
    main()
