"""
PROJET :
--------
Vitrine_pipeline

TITRE :
--------
3_metrics.py

OBJECTIF PRINCIPAL :
---------------------
Ce script compare les annotations humaines (gold) issues de fichiers JSONL aux prédictions générées par le modèle (CSV).
Il calcule les métriques multi-label (précision, rappel, F1 en micro et macro, ainsi que la subset accuracy)
et exporte les résultats dans un fichier CSV pour chaque fichier JSONL.

DÉPENDANCES :
-------------
- os
- csv
- json
- glob
- collections (defaultdict)

FONCTIONNALITÉS PRINCIPALES :
-----------------------------
1) Chargement des annotations humaines (JSONL) en filtrant les labels pertinents.
2) Chargement des annotations du modèle (CSV) et reconstitution des labels avec suffixe.
3) Calcul des métriques comparatives sur l'intersection des textes.
4) Exportation des métriques dans des fichiers CSV.

Auteur :
---------
Antoine Lemor
"""

import os
import csv
import json
import glob
from collections import defaultdict

def load_gold_annotations_from_jsonl(jsonl_path):
    """
    Charge les annotations humaines depuis un fichier JSONL.
    
    Paramètres :
    ------------
    jsonl_path : str
        Chemin vers le fichier JSONL contenant les annotations gold.
    
    Retourne :
    -----------
    dict
        Dictionnaire où chaque clé est un texte et la valeur est un ensemble de labels 
        (seulement ceux se terminant par "_body_annotated_deepseek_r1").
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
                # Récupération des labels et filtrage par suffixe
                labels = obj.get("label", [])
                valid_labels = {lbl for lbl in labels if lbl.endswith("_body_annotated_deepseek_r1")}
                gold_dict[text] = valid_labels
            except json.JSONDecodeError:
                # Ignorer la ligne en cas d'erreur de décodage
                continue
    return gold_dict

def load_model_annotations_from_csv(csv_path):
    """
    Charge les annotations du modèle à partir d'un fichier CSV.
    
    Paramètres :
    ------------
    csv_path : str
        Chemin vers le fichier CSV contenant les prédictions du modèle.
    
    Retourne :
    -----------
    dict
        Dictionnaire où chaque clé est un texte et la valeur est un ensemble de labels,
        reconstruit avec le suffixe "_body_annotated_deepseek_r1".
    """
    model_dict = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("body", "").strip()
            if not text:
                continue
            ann_str = row.get("body_annotated_deepseek_r1", "").strip()
            if not ann_str:
                model_dict[text] = set()
                continue
            try:
                ann_json = json.loads(ann_str)
                themes = ann_json.get("themes") or []
                # Reconstruction des labels avec ajout du suffixe
                labels = {f"{theme}_body_annotated_deepseek_r1" for theme in themes}
                model_dict[text] = labels
            except json.JSONDecodeError:
                model_dict[text] = set()
    return model_dict

def compute_metrics(gold_dict, model_dict):
    """
    Calcule les métriques multi-label sur l'intersection des textes présents dans les deux sources.
    
    Paramètres :
    ------------
    gold_dict : dict
        Dictionnaire des annotations humaines (gold).
    model_dict : dict
        Dictionnaire des annotations du modèle.
    
    Retourne :
    -----------
    dict
        Dictionnaire regroupant les métriques par label, ainsi que les scores micro-average,
        macro-average et la subset accuracy.
    """
    # On compare uniquement les textes communs aux deux sources
    common_texts = set(gold_dict.keys()) & set(model_dict.keys())

    label_tp = defaultdict(int)
    label_fp = defaultdict(int)
    label_fn = defaultdict(int)

    gold_label_count = defaultdict(int)
    model_label_count = defaultdict(int)

    total_docs = 0
    correct_docs = 0  # Nombre de documents où les labels sont identiques exactement

    for text in common_texts:
        gold_labels = gold_dict.get(text, set())
        pred_labels = model_dict.get(text, set())

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

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0

    n_labels = len(all_labels) if all_labels else 1
    macro_precision = sum(d["precision"] for d in per_label.values()) / n_labels
    macro_recall = sum(d["recall"] for d in per_label.values()) / n_labels
    macro_f1 = sum(d["f1"] for d in per_label.values()) / n_labels

    results = {
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
        "subset_accuracy": subset_accuracy
    }
    return results

def save_metrics_to_csv(results, output_path, jsonl_filename):
    """
    Exporte les métriques calculées dans un fichier CSV.
    
    Le CSV contiendra les colonnes :
        file, label, gold_count, pred_count, precision, recall, f1.
    Des lignes supplémentaires sont ajoutées pour __micro__, __macro__ et __subset_accuracy__.
    
    Paramètres :
    ------------
    results : dict
        Dictionnaire contenant les métriques calculées.
    output_path : str
        Chemin vers le fichier de sortie CSV.
    jsonl_filename : str
        Nom du fichier JSONL d'origine, utilisé pour identifier les données exportées.
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
    Fonction principale du script.
    
    Étapes :
    ------------
    1) Charger les annotations du modèle à partir d'un fichier CSV.
    2) Pour chaque fichier JSONL dans le répertoire des annotations humaines, 
       charger les annotations gold et calculer les métriques sur l'intersection des textes.
    3) Exporter les métriques calculées dans un fichier CSV associé.
    """
    # Chemin vers le CSV contenant les annotations du modèle
    csv_model_path = os.path.join("data", "processed", "subset", "radar_subset_deepseek.csv")
    model_dict = load_model_annotations_from_csv(csv_model_path)

    # Répertoire contenant les fichiers JSONL (annotations humaines)
    gold_dir = os.path.join("data", "processed", "validation", "annotated_jsonl")
    jsonl_files = glob.glob(os.path.join(gold_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"Aucun fichier JSONL trouvé dans {gold_dir}")
        return

    for jsonl_path in jsonl_files:
        jsonl_filename = os.path.basename(jsonl_path)
        # Chargement des annotations humaines à partir du fichier JSONL
        gold_dict = load_gold_annotations_from_jsonl(jsonl_path)

        # Calcul des métriques sur l'intersection des textes communs
        results = compute_metrics(gold_dict, model_dict)

        # Génération du nom du fichier CSV de sortie
        base_name = jsonl_filename.replace(".jsonl", "_metrics.csv")
        output_csv_path = os.path.join(gold_dir, base_name)

        save_metrics_to_csv(results, output_csv_path, jsonl_filename)
        print(f"Métriques calculées pour {jsonl_filename} -> {output_csv_path}")

if __name__ == "__main__":
    main()
