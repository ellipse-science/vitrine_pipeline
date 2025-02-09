#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJET :
--------
Vitrine_pipeline

TITRE :
--------
3_metrics.py (version "subtests")

OBJECTIF PRINCIPAL :
---------------------
Ce script compare les annotations humaines (JSONL) aux prédictions générées par le modèle (CSV)
pour un cas où chaque requête d'inférence du LLM correspond à une seule catégorie (un seul prompt)

Le CSV se trouve cette fois dans data/processed/subset/subtest.
Les annotations gold se trouvent dans data/processed/validation/annotated_jsonl.

Il calcule les métriques multi-label (précision, rappel, F1 en micro et macro, ainsi que la subset accuracy)
et exporte les résultats dans un fichier CSV pour chaque fichier JSONL détecté.

DÉPENDANCES :
-------------
- os
- csv
- json
- glob
- collections (defaultdict)

FONCTIONNEMENT SPÉCIFIQUE :
-----------------------------
1) Le script ouvre le CSV pour lister les colonnes disponibles et demande à l'utilisateur
   de sélectionner celles contenant le JSON d'annotations (chaque JSON correspondant à une catégorie).
2) Le script lit ensuite toutes les lignes du CSV (situé dans data/processed/subset/subtest)
   et combine les prédictions extraites des colonnes choisies.
3) Les données gold sont chargées depuis chaque fichier .jsonl (annotations humaines) situé dans
   data/processed/validation/annotated_jsonl.
4) Pour chaque fichier JSONL, on ne compare que l'intersection des textes (champ "body" dans le CSV
   vs. champ "text" dans le JSONL).
5) Les métriques sont calculées (TP, FP, FN, précision, rappel, F1, micro-average, macro-average et subset accuracy)
   en excluant les 'null'
6) Un fichier CSV de métriques est exporté pour chaque fichier JSONL.

Auteur :
---------
Antoine Lemor
"""

import os
import csv
import json
import glob
from collections import defaultdict

# Nom du label à ignorer
IGNORED_LABEL = "null_body_annotated_deepseek_r1"

def load_gold_annotations_from_jsonl(jsonl_path):
    """
    Charge les annotations humaines (gold) depuis un fichier JSONL.

    Paramètres :
    ------------
    jsonl_path : str
        Chemin vers le fichier JSONL contenant les annotations gold.

    Retourne :
    -----------
    dict
        Dictionnaire où chaque clé est le texte (obj["text"]) et la valeur est un
        ensemble de labels se terminant par "_body_annotated_deepseek_r1" (à l'exception du label "null").
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
                # Conserver uniquement les labels se terminant par le suffixe ET ne pas conserver "null"
                valid_labels = {lbl for lbl in labels if lbl.endswith("_body_annotated_deepseek_r1") and lbl != IGNORED_LABEL}
                gold_dict[text] = valid_labels
            except json.JSONDecodeError:
                continue
    return gold_dict

def select_annotation_columns(csv_file_path):
    """
    Affiche la liste des colonnes disponibles dans le CSV et demande à l'utilisateur
    de sélectionner lesquelles contiennent le JSON d'annotations à analyser.

    Retourne :
    -----------
    list
        Liste des colonnes sélectionnées.
    """
    with open(csv_file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
    
    print("\nColonnes disponibles dans le CSV :")
    for idx, col in enumerate(headers, start=1):
        print(f"  {idx}. {col}")
    
    selected_cols_input = input(
        "\nEntrez les numéros de colonnes contenant les annotations JSON (séparés par des virgules) : "
    ).strip()
    
    if not selected_cols_input:
        print("Aucune colonne sélectionnée. Fin du script.")
        return []
    
    selected_indices = []
    for val in selected_cols_input.split(","):
        val = val.strip()
        try:
            idx = int(val)
            if 1 <= idx <= len(headers):
                selected_indices.append(idx)
        except ValueError:
            continue

    selected_cols = [headers[i - 1] for i in selected_indices]
    return selected_cols

def load_model_annotations_from_csv(csv_path, annotation_cols):
    """
    Charge les annotations du modèle à partir d'un fichier CSV,
    où chaque colonne de la liste 'annotation_cols' contient un JSON décrivant la prédiction d'une catégorie.

    Paramètres :
    ------------
    csv_path : str
        Chemin vers le CSV contenant les prédictions du modèle.
    annotation_cols : list
        Liste des noms de colonnes contenant le JSON des prédictions.

    Retourne :
    -----------
    dict
        model_dict[text] = set(labels_predits)
        Pour chaque ligne, on extrait le champ "body" et on combine les labels
        provenant de toutes les colonnes sélectionnées, en ajoutant le suffixe
        "_body_annotated_deepseek_r1" et en ignorant le label "null".
    """
    model_dict = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("body", "").strip()
            if not text:
                continue

            combined_labels = set()
            for col in annotation_cols:
                json_str = row.get(col, "").strip()
                if not json_str:
                    continue
                try:
                    ann_json = json.loads(json_str)
                    # On tente de récupérer la valeur du champ "themes"
                    theme_val = ann_json.get("themes")
                    if isinstance(theme_val, str):
                        lbl = f"{theme_val}_body_annotated_deepseek_r1"
                        if lbl != IGNORED_LABEL:
                            combined_labels.add(lbl)
                    elif isinstance(theme_val, list):
                        for tv in theme_val:
                            lbl = f"{tv}_body_annotated_deepseek_r1"
                            if lbl != IGNORED_LABEL:
                                combined_labels.add(lbl)
                    # Possibilité de tester d'autres clés si besoin
                except json.JSONDecodeError:
                    continue
            model_dict[text] = combined_labels
    return model_dict

def compute_metrics(gold_dict, model_dict):
    """
    Calcule les métriques multi-label sur l'intersection des textes présents dans les deux sources.

    Retourne :
    -----------
    dict
        Dictionnaire regroupant les métriques par label, ainsi que les scores micro-average,
        macro-average et la subset accuracy.
    """
    # Comparer uniquement les textes communs
    common_texts = set(gold_dict.keys()) & set(model_dict.keys())

    label_tp = defaultdict(int)
    label_fp = defaultdict(int)
    label_fn = defaultdict(int)
    gold_label_count = defaultdict(int)
    model_label_count = defaultdict(int)
    total_docs = 0
    correct_docs = 0

    for text in common_texts:
        gold_labels = gold_dict[text]
        pred_labels = model_dict[text]

        for gl in gold_labels:
            gold_label_count[gl] += 1
        for pl in pred_labels:
            model_label_count[pl] += 1

        total_docs += 1
        if gold_labels == pred_labels:
            correct_docs += 1

        for lbl in gold_labels:
            if lbl in pred_labels:
                label_tp[lbl] += 1
            else:
                label_fn[lbl] += 1
        for lbl in pred_labels:
            if lbl not in gold_labels:
                label_fp[lbl] += 1

    subset_accuracy = correct_docs / total_docs if total_docs else 0.0
    all_labels = set(label_tp.keys()) | set(label_fp.keys()) | set(label_fn.keys())

    per_label = {}
    for lbl in sorted(all_labels):
        tp = label_tp[lbl]
        fp = label_fp[lbl]
        fn = label_fn[lbl]
        gc = gold_label_count[lbl]
        pc = model_label_count[lbl]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        per_label[lbl] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gold_count": gc,
            "pred_count": pc
        }

    total_tp = sum(label_tp.values())
    total_fp = sum(label_fp.values())
    total_fn = sum(label_fn.values())
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    n_labels = len(all_labels) if all_labels else 1
    macro_precision = sum(d["precision"] for d in per_label.values()) / n_labels
    macro_recall = sum(d["recall"] for d in per_label.values()) / n_labels
    macro_f1 = sum(d["f1"] for d in per_label.values()) / n_labels

    return {
        "per_label": per_label,
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "subset_accuracy": subset_accuracy
    }

def save_metrics_to_csv(results, output_path, jsonl_filename):
    """
    Enregistre les métriques calculées dans un fichier CSV.
    Le fichier contiendra une ligne par label, puis des lignes spéciales pour __micro__,
    __macro__ et __subset_accuracy__.
    """
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "gold_count", "pred_count", "precision", "recall", "f1"])
        for lbl, vals in results["per_label"].items():
            writer.writerow([
                jsonl_filename,
                lbl,
                vals["gold_count"],
                vals["pred_count"],
                round(vals["precision"], 4),
                round(vals["recall"], 4),
                round(vals["f1"], 4)
            ])
        writer.writerow([
            jsonl_filename,
            "__micro__",
            "",
            "",
            round(results["micro"]["precision"], 4),
            round(results["micro"]["recall"], 4),
            round(results["micro"]["f1"], 4)
        ])
        writer.writerow([
            jsonl_filename,
            "__macro__",
            "",
            "",
            round(results["macro"]["precision"], 4),
            round(results["macro"]["recall"], 4),
            round(results["macro"]["f1"], 4)
        ])
        writer.writerow([
            jsonl_filename,
            "__subset_accuracy__",
            "",
            "",
            round(results["subset_accuracy"], 4),
            "",
            ""
        ])

def main():
    """
    Étapes principales :
    1) Demander le chemin du CSV dans data/processed/subset/subtest (ou utiliser le chemin par défaut).
    2) Lister les colonnes du CSV et demander à l'utilisateur lesquelles contiennent le JSON d'annotations.
    3) Charger et combiner les prédictions du modèle depuis ces colonnes.
    4) Pour chaque fichier .jsonl dans data/processed/validation/annotated_jsonl :
         - Charger les annotations gold (en ignorant "null")
         - Calculer les métriques sur l'intersection des textes
         - Exporter un CSV de métriques correspondant.
    """
    print("Chemin par défaut : data/processed/subset/subtest/...")
    csv_subtest_path = input("Entrez le chemin du fichier CSV (ou laissez vide pour le chemin par défaut) : ").strip()
    if not csv_subtest_path:
        csv_subtest_path = os.path.join("data", "processed", "subset", "subtest", "predictions.csv")

    while not os.path.isfile(csv_subtest_path):
        print(f"Fichier introuvable : {csv_subtest_path}")
        csv_subtest_path = input("Réessayez : ").strip()

    annotation_cols = select_annotation_columns(csv_subtest_path)
    if not annotation_cols:
        return

    model_dict = load_model_annotations_from_csv(csv_subtest_path, annotation_cols)

    gold_dir = os.path.join("data", "processed", "validation", "annotated_jsonl")
    jsonl_files = glob.glob(os.path.join(gold_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"Aucun fichier JSONL trouvé dans {gold_dir}")
        return

    for jsonl_path in jsonl_files:
        jsonl_filename = os.path.basename(jsonl_path)
        gold_dict = load_gold_annotations_from_jsonl(jsonl_path)
        results = compute_metrics(gold_dict, model_dict)
        output_csv = os.path.join(gold_dir, jsonl_filename.replace(".jsonl", "_metrics.csv"))
        save_metrics_to_csv(results, output_csv, jsonl_filename)
        print(f"Métriques calculées pour {jsonl_filename} -> {output_csv}")

if __name__ == "__main__":
    main()
