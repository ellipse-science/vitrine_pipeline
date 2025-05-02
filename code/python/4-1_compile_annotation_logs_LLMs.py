"""
PROJECT:
-------
vitrine_pipeline

TITLE:
------
4-1_compile_annotation_logs_LLMs.py

MAIN OBJECTIVE:
---------------
Ce script compile automatiquement les métriques issues des logs d’entraînement 
des modèles spécialisés LLM. Il permet de :
1) Extraire, pour chaque log, les métriques par epoch (macro-F1, weighted-F1, recall 
   de la classe "1", pertes, etc.) ainsi que la distribution des labels en entraînement 
   et validation.
2) Agréger ces métriques en cas de multi-fold (cross-validation).
3) Calculer un score global par epoch via : score = alpha * macro_F1 + beta * weighted_F1 + gamma * recall_class_1.
4) Sélectionner la meilleure epoch (avec tie-break sur test_loss puis train_loss).
5) Générer un fichier CSV récapitulant, pour chaque modèle, la meilleure epoch ainsi que 
   toutes les métriques et distributions associées.

Dépendances:
-------------
- Python >= 3.7
- pandas, numpy, re, os

Auteur:
-------
Antoine Lemor
"""

import os
import re
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paramètres de sélection et de scoring
# --------------------------------------------------------------------------
overfit_threshold = 0.65  # Si (test_loss - train_loss) > ce seuil, on considère l'epoch surapprise
alpha = 0.6               # Poids de macro-F1
beta = 0.2                # Poids de weighted-F1
gamma = 3.2               # Poids du recall de la classe "1"

min_positive_recall = 0.0  # Recall minimum pour la classe positive 

# --------------------------------------------------------------------------
# Définition des chemins (adaptés à la structure du projet)
# --------------------------------------------------------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

# Répertoire contenant les logs d'entraînement LLM
log_output_dir = os.path.join(base_path, "..", "..", "data", "processed", "validation", "LLMs_training")

# Répertoire de sortie du CSV récapitulatif (ici dans subvalidation)
csv_output_dir = os.path.join(base_path, "..", "..", "data", "processed", "validation", "LLMs_training", "metrics_summary")
csv_output_path = os.path.join(csv_output_dir, "models_metrics_summary_advanced.csv")
os.makedirs(log_output_dir, exist_ok=True)
os.makedirs(csv_output_dir, exist_ok=True)

# --------------------------------------------------------------------------
# Reconnaissance automatique des logs et regroupement par modèle
# --------------------------------------------------------------------------
log_files = [f for f in os.listdir(log_output_dir) if f.endswith("_training_log.txt")]

fold_pattern = re.compile(r"^(.*)_fold(\d+)_training_log\.txt$")
model_to_folds = {}
for lf in log_files:
    match = fold_pattern.match(lf)
    if match:
        model_name, fold_num = match.groups()
        if model_name not in model_to_folds:
            model_to_folds[model_name] = []
        model_to_folds[model_name].append(lf)
    else:
        # Cas sans information de fold : on le traite comme un seul fold
        model_name_alt = lf.replace("_training_log.txt", "")
        if model_name_alt not in model_to_folds:
            model_to_folds[model_name_alt] = []
        model_to_folds[model_name_alt].append(lf)

# --------------------------------------------------------------------------
# Fonction utilitaire : analyse du rapport de classification
# --------------------------------------------------------------------------
def parse_classification_report(report_str):
    """
    Analyse un rapport de classification brut et retourne un dictionnaire 
    contenant les métriques globales et par classe (precision, recall, f1-score, support).
    """
    lines = [line.strip() for line in report_str.strip().split("\n")]
    report_data = {}
    if len(lines) < 2:
        return report_data

    for line in lines:
        if not line:
            continue
        tokens = line.split()
        # Exemple : "accuracy                           0.95       286"
        if tokens[0] == "accuracy":
            try:
                report_data["accuracy"] = float(tokens[-2])
            except:
                pass
            continue

        # Cas "macro avg" ou "weighted avg"
        joined_first_two = " ".join(tokens[:2])
        if joined_first_two in ["macro avg", "weighted avg"]:
            label = joined_first_two
            try:
                precision, recall, f1_score, support = map(float, tokens[2:])
                report_data[label] = {
                    "precision": precision,
                    "recall": recall,
                    "f1-score": f1_score,
                    "support": support,
                }
            except:
                pass
            continue

        # Pour les classes (par exemple "0", "1")
        if len(tokens) >= 5:
            label_candidate = tokens[0]
            if label_candidate.isdigit() or label_candidate in ["0", "1"]:
                try:
                    precision, recall, f1_score, support = map(float, tokens[1:5])
                    report_data[label_candidate] = {
                        "precision": precision,
                        "recall": recall,
                        "f1-score": f1_score,
                        "support": support,
                    }
                except:
                    pass

    return report_data

# --------------------------------------------------------------------------
# Extraction des métriques et distributions depuis un log unique
# --------------------------------------------------------------------------
def parse_single_log(full_log_path):
    """
    Extrait, depuis un log d'entraînement, les métriques par epoch ainsi que 
    la distribution des labels en entraînement et validation.
    """
    if not os.path.exists(full_log_path) or os.path.getsize(full_log_path) == 0:
        return {}

    with open(full_log_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return {}

    # Extraction de la distribution des labels en entraînement et validation
    train_class_0_count = 0
    train_class_1_count = 0
    test_class_0_count = 0
    test_class_1_count = 0

    lines = content.split("\n")
    nb_lines = len(lines)
    i = 0
    while i < nb_lines:
        line = lines[i]
        if "Training label distribution" in line:
            dist_train = {}
            j = i + 1
            while j < nb_lines:
                subline = lines[j].strip()
                if subline.startswith("Name: count"):
                    break
                m = re.match(r"(\d+)\s+(\d+)", subline)
                if m:
                    label_found = m.group(1)
                    count_found = int(m.group(2))
                    dist_train[label_found] = count_found
                j += 1
            train_class_0_count = dist_train.get('0', 0)
            train_class_1_count = dist_train.get('1', 0)
            i = j
            continue

        if "Validation label distribution" in line:
            dist_test = {}
            j = i + 1
            while j < nb_lines:
                subline = lines[j].strip()
                if subline.startswith("Name: count"):
                    break
                m = re.match(r"(\d+)\s+(\d+)", subline)
                if m:
                    label_found = m.group(1)
                    count_found = int(m.group(2))
                    dist_test[label_found] = count_found
                j += 1
            test_class_0_count = dist_test.get('0', 0)
            test_class_1_count = dist_test.get('1', 0)
            i = j
            continue
        i += 1

    # Extraction des métriques par epoch
    epoch_pattern = re.compile(r"^[=]{4,}\s*Epoch\s+(\d+)\s*/\s*(\d+)\s*[=]{4,}", re.MULTILINE)
    matches = list(epoch_pattern.finditer(content))
    if not matches:
        return {
            "epochs_data": {},
            "train_class_0_count": train_class_0_count,
            "train_class_1_count": train_class_1_count,
            "test_class_0_count": test_class_0_count,
            "test_class_1_count": test_class_1_count,
        }

    epoch_data = {}
    for i, match in enumerate(matches):
        epoch_num = int(match.group(1))
        start_pos = match.end()
        end_pos = matches[i+1].start() if (i + 1) < len(matches) else len(content)
        block = content[start_pos:end_pos]

        m_train = re.search(r"Average training loss:\s*([\d.]+)", block)
        train_loss = float(m_train.group(1)) if m_train else float("inf")
        m_test = re.search(r"Average test loss:\s*([\d.]+)", block)
        test_loss = float(m_test.group(1)) if m_test else float("inf")

        # Recherche du rapport de classification
        classif_start = re.search(r"(?:^|\n)\s*precision\s+recall\s+f1-score\s+support\s*\n", block)
        if not classif_start:
            continue
        start_idx = classif_start.end()
        classif_content = block[start_idx:].strip()
        next_epoch_marker = re.search(r"^[=]{4,}\s*Epoch\s", classif_content, re.MULTILINE)
        if next_epoch_marker:
            classif_content = classif_content[:next_epoch_marker.start()].strip()

        report_data = parse_classification_report(classif_content)
        if "macro avg" not in report_data:
            continue

        macro_f1 = report_data["macro avg"]["f1-score"]
        weighted_f1 = report_data.get("weighted avg", {}).get("f1-score", 0.0)
        class_1_precision = report_data.get("1", {}).get("precision", 0.0)
        class_1_recall = report_data.get("1", {}).get("recall", 0.0)
        class_1_f1_score = report_data.get("1", {}).get("f1-score", 0.0)
        class_1_support = report_data.get("1", {}).get("support", 0.0)
        class_0_support = report_data.get("0", {}).get("support", 0.0)

        epoch_data[epoch_num] = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "recall_1": class_1_recall,
            "class_0_support": class_0_support,
            "class_1_support": class_1_support,
            "class_1_precision": class_1_precision,
            "class_1_f1_score": class_1_f1_score,
        }

    return {
        "epochs_data": epoch_data,
        "train_class_0_count": train_class_0_count,
        "train_class_1_count": train_class_1_count,
        "test_class_0_count": test_class_0_count,
        "test_class_1_count": test_class_1_count,
    }

# --------------------------------------------------------------------------
# Agrégation des métriques sur plusieurs folds (cross-validation)
# --------------------------------------------------------------------------
def aggregate_folds_metrics(list_of_epoch_dicts):
    if not list_of_epoch_dicts:
        return {}

    common_epochs = set(list_of_epoch_dicts[0].keys())
    for d in list_of_epoch_dicts[1:]:
        common_epochs = common_epochs.intersection(d.keys())

    aggregated = {}
    for ep in sorted(common_epochs):
        macro_f1_vals = []
        weighted_f1_vals = []
        test_loss_vals = []
        train_loss_vals = []
        recall_1_vals = []
        class_1_precision_vals = []
        class_1_f1_vals = []
        class_0_support_total = 0.0
        class_1_support_total = 0.0
        overfit_detected = False

        for fold_dict in list_of_epoch_dicts:
            ep_metrics = fold_dict[ep]
            if (ep_metrics["test_loss"] - ep_metrics["train_loss"]) > overfit_threshold:
                overfit_detected = True
                break
            macro_f1_vals.append(ep_metrics["macro_f1"])
            weighted_f1_vals.append(ep_metrics["weighted_f1"])
            test_loss_vals.append(ep_metrics["test_loss"])
            train_loss_vals.append(ep_metrics["train_loss"])
            recall_1_vals.append(ep_metrics["recall_1"])
            class_1_precision_vals.append(ep_metrics["class_1_precision"])
            class_1_f1_vals.append(ep_metrics["class_1_f1_score"])
            class_0_support_total += ep_metrics["class_0_support"]
            class_1_support_total += ep_metrics["class_1_support"]

        if overfit_detected:
            continue

        aggregated[ep] = {
            "macro_f1": np.mean(macro_f1_vals),
            "weighted_f1": np.mean(weighted_f1_vals),
            "test_loss": np.mean(test_loss_vals),
            "train_loss": np.mean(train_loss_vals),
            "recall_1": np.mean(recall_1_vals),
            "class_1_precision": np.mean(class_1_precision_vals),
            "class_1_f1_score": np.mean(class_1_f1_vals),
            "class_0_support": class_0_support_total,
            "class_1_support": class_1_support_total,
        }

    return aggregated

# --------------------------------------------------------------------------
# Calcul du score global pour une epoch
# --------------------------------------------------------------------------
def compute_epoch_score(macro_f1, weighted_f1, recall_1):
    return alpha * macro_f1 + beta * weighted_f1 + gamma * recall_1

# --------------------------------------------------------------------------
# Sélection de la meilleure epoch en fonction du score global et des pertes (tie-break)
# --------------------------------------------------------------------------
def select_best_epoch(epochs_metrics):
    best_epoch = None
    best_data = None
    best_score = -1.0

    for ep, vals in epochs_metrics.items():
        macro_f1 = vals["macro_f1"]
        weighted_f1 = vals["weighted_f1"]
        recall_1 = vals["recall_1"]
        test_loss = vals["test_loss"]
        train_loss = vals["train_loss"]

        if recall_1 < min_positive_recall:
            continue

        current_score = compute_epoch_score(macro_f1, weighted_f1, recall_1)
        if current_score > best_score:
            best_epoch = ep
            best_data = vals
            best_score = current_score
        elif abs(current_score - best_score) < 1e-9:
            # En cas d'égalité, on compare sur test_loss puis train_loss
            if test_loss < best_data["test_loss"]:
                best_epoch = ep
                best_data = vals
                best_score = current_score
            elif abs(test_loss - best_data["test_loss"]) < 1e-9:
                if train_loss < best_data["train_loss"]:
                    best_epoch = ep
                    best_data = vals
                    best_score = current_score

    if best_epoch is None:
        return None, None
    return best_epoch, best_data

# --------------------------------------------------------------------------
# Boucle principale : parsing des logs, agrégation, sélection et compilation CSV
# --------------------------------------------------------------------------
final_rows = []

for model_name, fold_logs in model_to_folds.items():
    fold_epoch_dicts = []
    train_class_0_counts = []
    train_class_1_counts = []
    test_class_0_counts = []
    test_class_1_counts = []

    for lf in fold_logs:
        path_log = os.path.join(log_output_dir, lf)
        parsed = parse_single_log(path_log)
        if "epochs_data" in parsed and parsed["epochs_data"]:
            fold_epoch_dicts.append(parsed["epochs_data"])
            train_class_0_counts.append(parsed.get("train_class_0_count", 0))
            train_class_1_counts.append(parsed.get("train_class_1_count", 0))
            test_class_0_counts.append(parsed.get("test_class_0_count", 0))
            test_class_1_counts.append(parsed.get("test_class_1_count", 0))

    if not fold_epoch_dicts:
        continue

    if len(fold_epoch_dicts) == 1:
        single_dict = fold_epoch_dicts[0]
        aggregated_metrics = {}
        for ep, metrics in single_dict.items():
            if (metrics["test_loss"] - metrics["train_loss"]) > overfit_threshold:
                continue
            aggregated_metrics[ep] = {
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "test_loss": metrics["test_loss"],
                "train_loss": metrics["train_loss"],
                "recall_1": metrics["recall_1"],
                "class_1_precision": metrics["class_1_precision"],
                "class_1_f1_score": metrics["class_1_f1_score"],
                "class_0_support": metrics["class_0_support"],
                "class_1_support": metrics["class_1_support"],
            }
        total_train_class_0_count = train_class_0_counts[0]
        total_train_class_1_count = train_class_1_counts[0]
        total_test_class_0_count = test_class_0_counts[0]
        total_test_class_1_count = test_class_1_counts[0]
    else:
        aggregated_metrics = aggregate_folds_metrics(fold_epoch_dicts)
        total_train_class_0_count = sum(train_class_0_counts)
        total_train_class_1_count = sum(train_class_1_counts)
        total_test_class_0_count = sum(test_class_0_counts)
        total_test_class_1_count = sum(test_class_1_counts)

    if not aggregated_metrics:
        continue

    best_ep, best_vals = select_best_epoch(aggregated_metrics)
    if best_ep is None:
        continue

    row = {
        "model_name": model_name,
        "best_epoch": best_ep,
        "score": compute_epoch_score(best_vals["macro_f1"], best_vals["weighted_f1"], best_vals["recall_1"]),
        "macro_f1": best_vals["macro_f1"],
        "weighted_f1": best_vals["weighted_f1"],
        "test_loss": best_vals["test_loss"],
        "train_loss": best_vals["train_loss"],
        "class_1_precision": best_vals["class_1_precision"],
        "class_1_recall": best_vals["recall_1"],
        "class_1_f1_score": best_vals["class_1_f1_score"],
        "class_1_support": best_vals["class_1_support"],
        "class_0_support": best_vals["class_0_support"],
        "train_class_0_count": total_train_class_0_count,
        "train_class_1_count": total_train_class_1_count,
        "test_class_0_count": total_test_class_0_count,
        "test_class_1_count": total_test_class_1_count,
    }
    final_rows.append(row)

df = pd.DataFrame(final_rows)
df.to_csv(csv_output_path, index=False)
print(f"[END] Advanced metrics summary saved in: {csv_output_path}")
