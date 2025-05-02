"""
PROJET :
--------
Vitrine_pipeline

TITRE :
-------
3bis_metrics_all.py

OBJECTIF PRINCIPAL :
-------------------
Ce script calcule des métriques d'évaluation (précision, rappel, F1-score) pour comparer
les prédictions de différentes méthodes par rapport à un standard de référence (gold standard).
Il traite des données multi-étiquettes dans plusieurs catégories et génère des métriques 
à la fois au niveau des étiquettes individuelles et des catégories entières.

DÉPENDANCES :
------------
- os
- json
- csv
- collections (defaultdict)
- glob

FONCTIONNALITÉS PRINCIPALES :
----------------------------
1) Lecture des annotations de référence (gold standard) depuis un fichier JSONL
2) Lecture des prédictions de différentes méthodes depuis des fichiers CSV
3) Calcul des métriques de performance (précision, rappel, F1) pour chaque étiquette 
   et chaque catégorie (thèmes, partis politiques, thèmes spécifiques, sentiment)
4) Génération de métriques micro et macro moyennées pour chaque catégorie
5) Production d'un fichier CSV contenant toutes les métriques calculées

Auteur :
-------
Antoine Lemor
"""

import os
import json
import csv
from collections import defaultdict
import glob


###############################################################################
#                  1. FONCTIONS D'ANALYSE DU STANDARD DE RÉFÉRENCE
###############################################################################

def parse_gold_labels(gold_label_list):
 """
 Analyse la liste des étiquettes de référence (gold standard) sous les formes :
   - "theme_energy_jeremy"
   - "political_parties_CPC_shdin"
   - "sentiment_positive_antoine"
   - "specific_themes_welfare_state_jdrouin"
 en ignorant les doublons provenant de plusieurs annotateurs.

 Paramètres :
 -----------
 gold_label_list : list
     Liste des étiquettes de référence brutes.

 Renvoie :
 --------
 dict
     Un dictionnaire avec les clés :
     {
       "themes": set(...),
       "political_parties": set(...),
       "specific_themes": set(...),
       "sentiment": set(...)
     }

 Détails d'implémentation :
 -------------------------
 - Découpe sur les caractères underscore; la première partie est la "catégorie" 
   (theme, sentiment, etc.), la dernière partie est le nom de l'annotateur, 
   et les parties intermédiaires forment l'étiquette réelle.
 - Si l'étiquette est 'null', elle est ignorée.
 - Conversion : "theme" -> "themes", 
                "political" (avec "parties_") -> "political_parties",
                "sentiment" -> "sentiment",
                "specific" ou "specific_themes" -> "specific_themes".
 """
 label_dict = {
     "themes": set(),
     "political_parties": set(),
     "specific_themes": set(),
     "sentiment": set()
 }

 for lbl in gold_label_list:
     parts = lbl.split('_')
     if len(parts) < 2:
         # Not a valid label with category + something + annotator
         continue

     category = parts[0].lower().strip()
     label_value_list = parts[1:-1]
     if not label_value_list:
         # Means there's no actual label text
         continue

     label_text = "_".join(label_value_list).lower().strip()
     # Ignore if label_text is 'null'
     if label_text == "null":
         continue

     # Category mapping
     if category == "theme":
         label_dict["themes"].add(label_text)
     elif category == "political":
         # Could be "political_parties_CPC_jeremy" => parts = 
         # ["political", "parties", "CPC", "jeremy"]
         # label_text = "parties_CPC"
         if label_text.startswith("parties_"):
             label_text = label_text.replace("parties_", "", 1)
         if label_text != "null":
             label_dict["political_parties"].add(label_text)
     elif category == "political_parties":
         label_dict["political_parties"].add(label_text)
     elif category == "sentiment":
         label_dict["sentiment"].add(label_text)
     elif category == "specific":
         # e.g. "specific_themes_welfare_state_jeremy"
         # label_text might be "themes_welfare_state"
         # remove "themes_"
         if label_text.startswith("themes_"):
             label_text = label_text.replace("themes_", "", 1)
         if label_text != "null":
             label_dict["specific_themes"].add(label_text)
     elif category == "specific_themes":
         label_dict["specific_themes"].add(label_text)
     else:
         pass  # ignore anything else not recognized

 return label_dict


def build_gold_dict(gold_path):
 """
 Lit le fichier gold_count.jsonl ligne par ligne, en analysant chaque objet JSON :
   {
     "text": "...",
     "label": ["sentiment_positive_jeremy", "theme_environment_shdin", ...],
     ...
   }

 Paramètres :
 -----------
 gold_path : str
     Chemin d'accès au fichier JSONL contenant les annotations de référence.

 Renvoie :
 --------
 dict
     Un dictionnaire où :
     gold_dict[texte] = {
       "themes": set(...),
       "political_parties": set(...),
       "specific_themes": set(...),
       "sentiment": set(...)
     }
     Si un texte apparaît plusieurs fois, les ensembles d'étiquettes sont fusionnés.
 """
 gold_dict = {}
 with open(gold_path, "r", encoding="utf-8") as f:
     for line in f:
         line = line.strip()
         if not line:
             continue
         data = json.loads(line)
         txt = data.get("text", "").strip()
         if not txt:
             continue
         label_list = data.get("label", [])
         parsed = parse_gold_labels(label_list)

         if txt not in gold_dict:
             gold_dict[txt] = {
                 "themes": set(),
                 "political_parties": set(),
                 "specific_themes": set(),
                 "sentiment": set()
             }

         gold_dict[txt]["themes"].update(parsed["themes"])
         gold_dict[txt]["political_parties"].update(parsed["political_parties"])
         gold_dict[txt]["specific_themes"].update(parsed["specific_themes"])
         gold_dict[txt]["sentiment"].update(parsed["sentiment"])

 return gold_dict


###############################################################################
#                  2. FONCTIONS D'ANALYSE DES PRÉDICTIONS
###############################################################################

def parse_prediction_json(pred_str):
 """
 Analyse la chaîne JSON de prédictions de chaque ligne CSV. Exemple :

 {
   "themes": ["environment", "energy"] ou "null",
   "political_parties": "CPC" ou ["CPC", "NDP"] ou "null",
   "specific_themes": ["public_finance"] ou "null",
   "sentiment": "negative" ou ["negative"] ou "null"
 }

 Paramètres :
 -----------
 pred_str : str
     Chaîne JSON contenant les prédictions.

 Renvoie :
 --------
 dict
     Un dictionnaire d'ensembles :
     {
       "themes": set(...),
       "political_parties": set(...),
       "specific_themes": set(...),
       "sentiment": set(...)
     }

 Notes :
 ------
 - Si le champ est "null" ou manquant, c'est un ensemble vide.
 - Si le champ est une chaîne unique, elle est convertie en un ensemble à 1 élément (sauf si c'est "null").
 - Si le champ est une liste, tous les éléments non-"null" sont ajoutés à l'ensemble.
 """
 cat_pred = {
     "themes": set(),
     "political_parties": set(),
     "specific_themes": set(),
     "sentiment": set()
 }
 if not pred_str:
     return cat_pred

 try:
     data = json.loads(pred_str)
 except (json.JSONDecodeError, TypeError):
     # If not valid JSON, return empty sets
     return cat_pred

 # Helper to convert a raw JSON value (string or list) to a set
 def val_to_set(val):
     if val is None or val == "null":
         return set()
     if isinstance(val, str):
         # If it's a single string
         val_lower = val.strip().lower()
         return set([]) if val_lower == "null" else set([val_lower])
     elif isinstance(val, list):
         s = set()
         for x in val:
             if x and x.lower() != "null":
                 s.add(x.strip().lower())
         return s
     elif isinstance(val, dict):
         # In case there's a dictionary by mistake. We ignore it.
         return set()
     else:
         return set()

 cat_pred["themes"] = val_to_set(data.get("themes", None))
 cat_pred["political_parties"] = val_to_set(data.get("political_parties", None))
 cat_pred["specific_themes"] = val_to_set(data.get("specific_themes", None))
 cat_pred["sentiment"] = val_to_set(data.get("sentiment", None))

 return cat_pred


def build_predictions_dict(csv_path):
 """
 Lit un CSV avec les colonnes :
   - text
   - predictions (chaîne JSON)
   - inference_time (flottant, facultatif)

 Paramètres :
 -----------
 csv_path : str
     Chemin d'accès au fichier CSV contenant les prédictions.

 Renvoie :
 --------
 tuple
     (pred_dict, total_inference_time, num_rows) où :
     - pred_dict[texte] = {
         "themes": set(...),
         "political_parties": set(...),
         "specific_themes": set(...),
         "sentiment": set(...)
       }
     - total_inference_time est la somme de tous les 'inference_time' numériques
       si présent
     - num_rows est le nombre total de lignes dans le CSV (en excluant l'en-tête)
 """
 pred_dict = {}
 total_inference_time = 0.0
 num_rows = 0

 with open(csv_path, "r", encoding="utf-8") as f:
     reader = csv.DictReader(f)
     for row in reader:
         num_rows += 1
         txt = row.get("text", "").strip()
         if not txt:
             continue

         pred_json = row.get("predictions", "")
         parsed = parse_prediction_json(pred_json)
         pred_dict[txt] = parsed

         # Sum inference_time if present
         if "inference_time" in row and row["inference_time"]:
             try:
                 val = float(row["inference_time"])
                 total_inference_time += val
             except ValueError:
                 pass

 return pred_dict, total_inference_time, num_rows


###############################################################################
#                3. CALCUL DES MÉTRIQUES (MULTI-ÉTIQUETTES)
###############################################################################

def compute_labelwise_confusion(gold_list, pred_list, label_set):
 """
 Calcule les matrices de confusion par étiquette.

 Paramètres :
 -----------
 gold_list : list
     gold_list[i] = ensemble des étiquettes de référence pour l'élément i
 pred_list : list
     pred_list[i] = ensemble des étiquettes prédites pour l'élément i
 label_set : set
     Union des étiquettes de référence et prédites (pour le comptage de confusion)

 Renvoie :
 --------
 tuple
     tp[étiquette], fp[étiquette], fn[étiquette] pour chaque étiquette dans label_set.
     (Nous ne calculons PAS explicitement TN car nous en avons uniquement besoin pour les micro/macro.)
 """
 tp = defaultdict(int)
 fp = defaultdict(int)
 fn = defaultdict(int)

 for i in range(len(gold_list)):
     g = gold_list[i]
     p = pred_list[i]
     for lbl in label_set:
         in_gold = (lbl in g)
         in_pred = (lbl in p)
         if in_gold and in_pred:
             tp[lbl] += 1
         elif in_gold and not in_pred:
             fn[lbl] += 1
         elif not in_gold and in_pred:
             fp[lbl] += 1
         else:
             # TN - not needed for multi-label micro/macro
             pass

 return tp, fp, fn


def precision_recall_f1_from_confusion(tp, fp, fn, label):
 """
 Calcule la précision, le rappel et le F1-score pour une seule étiquette à partir des dictionnaires tp, fp, fn.

 Paramètres :
 -----------
 tp : dict
     Dictionnaire des vrais positifs par étiquette
 fp : dict
     Dictionnaire des faux positifs par étiquette
 fn : dict
     Dictionnaire des faux négatifs par étiquette
 label : str
     L'étiquette pour laquelle calculer les métriques

 Renvoie :
 --------
 tuple
     (précision, rappel, f1) pour l'étiquette spécifiée
 """
 tp_val = tp[label]
 fp_val = fp[label]
 fn_val = fn[label]

 if (tp_val + fp_val) > 0:
     precision = tp_val / (tp_val + fp_val)
 else:
     precision = 0.0

 if (tp_val + fn_val) > 0:
     recall = tp_val / (tp_val + fn_val)
 else:
     recall = 0.0

 if (precision + recall) > 0:
     f1 = 2 * precision * recall / (precision + recall)
 else:
     f1 = 0.0

 return precision, recall, f1


def compute_multi_label_metrics(gold_list, pred_list, full_label_set):
 """
 Calcule les métriques multi-étiquettes avec moyenne micro et macro.

 Étapes :
 -------
 1. Calcul des TP, FP, FN par étiquette.
 2. Macro-moyenne = moyenne de toutes les métriques (P,R,F1) au niveau des étiquettes.
 3. Micro-moyenne = somme des TP, somme des FP, somme des FN pour toutes les étiquettes -> puis calcul de (P,R,F1).

 Paramètres :
 -----------
 gold_list : list
     Liste de sets contenant les étiquettes de référence
 pred_list : list
     Liste de sets contenant les étiquettes prédites
 full_label_set : set
     Ensemble complet de toutes les étiquettes (référence + prédictions)

 Renvoie :
 --------
 dict
   {
      "label_metrics": {
          étiquette_x: {"precision":..., "recall":..., "f1":..., "tp":..., "fp":..., "fn":...},
          ...
      },
      "macro_precision": ...,
      "macro_recall": ...,
      "macro_f1": ...,
      "micro_precision": ...,
      "micro_recall": ...,
      "micro_f1": ...
   }

 Note: full_label_set doit inclure toute étiquette qui apparaît dans les données de référence ou qui a été prédite
 pour des calculs corrects de la matrice de confusion.
 """
 tp, fp, fn = compute_labelwise_confusion(gold_list, pred_list, full_label_set)

 label_metrics = {}
 for lbl in full_label_set:
     p, r, f1 = precision_recall_f1_from_confusion(tp, fp, fn, lbl)
     label_metrics[lbl] = {
         "precision": p,
         "recall": r,
         "f1": f1,
         "tp": tp[lbl],
         "fp": fp[lbl],
         "fn": fn[lbl]
     }

 n_labels = len(full_label_set)
 if n_labels == 0:
     macro_p = macro_r = macro_f1 = 0.0
 else:
     macro_p = sum(m["precision"] for m in label_metrics.values()) / n_labels
     macro_r = sum(m["recall"] for m in label_metrics.values()) / n_labels
     macro_f1 = sum(m["f1"] for m in label_metrics.values()) / n_labels

 sum_tp = sum(tp.values())
 sum_fp = sum(fp.values())
 sum_fn = sum(fn.values())

 if (sum_tp + sum_fp) > 0:
     micro_p = sum_tp / (sum_tp + sum_fp)
 else:
     micro_p = 0.0

 if (sum_tp + sum_fn) > 0:
     micro_r = sum_tp / (sum_tp + sum_fn)
 else:
     micro_r = 0.0

 if (micro_p + micro_r) > 0:
     micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
 else:
     micro_f1 = 0.0

 return {
     "label_metrics": label_metrics,
     "macro_precision": macro_p,
     "macro_recall": macro_r,
     "macro_f1": macro_f1,
     "micro_precision": micro_p,
     "micro_recall": micro_r,
     "micro_f1": micro_f1
 }


def compute_metrics_for_category(gold_dict, pred_dict, category, gold_label_set):
 """
 Calcule les métriques pour une catégorie spécifique.

 Paramètres :
 -----------
 gold_dict : dict
     gold_dict[texte][catégorie] -> ensemble d'étiquettes de référence
 pred_dict : dict
     pred_dict[texte][catégorie] -> ensemble d'étiquettes prédites
 category : str
     La catégorie à évaluer
 gold_label_set : set
     Ensemble de toutes les étiquettes qui apparaissent dans les données de référence pour cette catégorie
     (utilisé pour filtrer les lignes de sortie afin de sauter les étiquettes jamais présentes dans les références)

 Renvoie :
 --------
 tuple
     (metrics_result, nb_matched_texts) :
     - metrics_result est le dictionnaire renvoyé par compute_multi_label_metrics
     - nb_matched_texts est le nombre de textes qui apparaissent à la fois dans gold_dict et pred_dict
 """
 common_texts = sorted(set(gold_dict.keys()).intersection(pred_dict.keys()))
 gold_list = []
 pred_list = []

 # Collect union of labels for confusion matrix
 # (this includes labels predicted that might not be in gold so we can count FP)
 union_label_set = set()

 for txt in common_texts:
     gset = gold_dict[txt][category]
     pset = pred_dict[txt][category]
     gold_list.append(gset)
     pred_list.append(pset)
     union_label_set.update(gset)
     union_label_set.update(pset)

 # Now compute multi-label metrics
 metrics_result = compute_multi_label_metrics(gold_list, pred_list, union_label_set)
 nb_matched_texts = len(common_texts)

 return metrics_result, nb_matched_texts


###############################################################################
#                            4. FONCTION PRINCIPALE
###############################################################################

def main():
 """
 Fonction principale pour :
 1. Lire le standard de référence depuis JSONL.
 2. Lire chaque CSV dans data/processed/subset.
 3. Pour chaque méthode, calculer les métriques étiquette par étiquette pour chaque catégorie
    plus les micro/macro dans cette catégorie.
 4. Calculer également une catégorie "ALL" combinant les catégories (pour les LLMs, les 4;
    pour le dictionnaire, ignorer "specific_themes" si non utilisé).
 5. Écrire un seul CSV au format vertical :
      method_name, nb_predictions, nb_gold_texts, nb_gold_matched, total_inference_time,
      category, label, precision, recall, f1
    avec des lignes supplémentaires pour "(micro)" et "(macro)" à la fin de chaque bloc de catégorie
    et un bloc supplémentaire category="ALL" résumant l'ensemble.
 """
 gold_path = "data/processed/validation/subvalidation/gold_count.jsonl"
 subset_dir = "data/processed/subset"
 output_csv = "data/processed/validation/all_metrics_gold.csv"

 # 1. Build gold dictionary
 gold_dict = build_gold_dict(gold_path)
 nb_gold_texts = len(gold_dict)

 # Gather sets of gold labels for each category (only those that appear in gold)
 gold_labels_by_cat = {
     "themes": set(),
     "political_parties": set(),
     "specific_themes": set(),
     "sentiment": set()
 }
 for txt, cat_dict in gold_dict.items():
     for cat, labels in cat_dict.items():
         gold_labels_by_cat[cat].update(labels)

 # 2. Gather CSV files in subset_dir
 csv_files = glob.glob(os.path.join(subset_dir, "*.csv"))

 # 3. Prepare data structure for final results
 #    We will output in "vertical" format:
 #    method, nb_predictions, nb_gold_texts, nb_gold_matched, total_inference_time,
 #    category, label, precision, recall, f1
 #    (We can also store micro/macro separately with label="(micro)" or "(macro)")
 rows_for_csv = []

 # 4. For each CSV => parse predictions => compute metrics
 for csv_path in csv_files:
     method_name = os.path.splitext(os.path.basename(csv_path))[0]
     pred_dict, total_inf_time, nb_predictions = build_predictions_dict(csv_path)

     # Determine if we skip "specific_themes" for this method 
     # (e.g. if method_name or the CSV is recognized as 'dictionary', 
     #  or if user wants to skip it because dictionary never predicted it).
     # For safety, let's see if the user specifically wants it. 
     # We'll use a simple check:
     skip_specific_themes = False
     if "dictionary" in method_name.lower():
         skip_specific_themes = True

     # We track how many gold texts matched for *all categories combined*
     # We'll accumulate sets of matched texts across the categories we evaluate
     matched_texts_overall = set()

     # 4a. Compute category-level metrics
     categories_to_process = ["themes", "political_parties", "specific_themes", "sentiment"]
     if skip_specific_themes:
         categories_to_process = ["themes", "political_parties", "sentiment"]

     for cat in categories_to_process:
         # Compute metrics
         cat_result, nb_matched_texts = compute_metrics_for_category(gold_dict, pred_dict, cat, gold_labels_by_cat[cat])
         # We'll gather these matched texts to eventually compute total matched across categories
         # but since we only want texts that appear in the gold for that category, we keep 
         # a set intersection. Actually we want the texts in intersection for all categories?
         # Let's just gather them in a union sense: these are texts for which we have gold/pred 
         # for cat. It's a slight overcount if some text has no label in cat. 
         # The user specifically wanted a single "nb_gold_matched"? 
         # We'll keep the maximum. 
         matched_texts_overall.update(set(gold_dict.keys()).intersection(pred_dict.keys()))

         # label-level (only produce rows for labels that appear in gold to avoid 
         # listing spurious predicted labels)
         # We still compute confusion for the union of gold+pred, 
         # but display rows only for gold labels.
         label_metrics = cat_result["label_metrics"]
         # All labels used in confusion are label_metrics.keys()
         # But we only display those that appear in gold_labels_by_cat[cat].
         gold_labels_in_cat = gold_labels_by_cat[cat]
         for lbl in sorted(gold_labels_in_cat):
             if lbl not in label_metrics:
                 # That means it never appeared in predictions nor is in gold for the intersection
                 # (But if it's truly in gold, it should be in label_metrics unless no texts matched)
                 continue
             p = label_metrics[lbl]["precision"]
             r = label_metrics[lbl]["recall"]
             f1 = label_metrics[lbl]["f1"]
             rows_for_csv.append({
                 "method_name": method_name,
                 "nb_predictions": nb_predictions,
                 "nb_gold_texts": nb_gold_texts,
                 "nb_gold_matched": nb_matched_texts,  # how many texts matched for this category
                 "total_inference_time": total_inf_time,
                 "category": cat,
                 "label": lbl,
                 "precision": p,
                 "recall": r,
                 "f1": f1
             })

         # Then the micro, macro lines
         rows_for_csv.append({
             "method_name": method_name,
             "nb_predictions": nb_predictions,
             "nb_gold_texts": nb_gold_texts,
             "nb_gold_matched": nb_matched_texts,
             "total_inference_time": total_inf_time,
             "category": cat,
             "label": "(micro)",
             "precision": cat_result["micro_precision"],
             "recall": cat_result["micro_recall"],
             "f1": cat_result["micro_f1"]
         })
         rows_for_csv.append({
             "method_name": method_name,
             "nb_predictions": nb_predictions,
             "nb_gold_texts": nb_gold_texts,
             "nb_gold_matched": nb_matched_texts,
             "total_inference_time": total_inf_time,
             "category": cat,
             "label": "(macro)",
             "precision": cat_result["macro_precision"],
             "recall": cat_result["macro_recall"],
             "f1": cat_result["macro_f1"]
         })

     # 4b. "ALL" category metrics (combine categories that this method supports)
     # Collect sets from each text across the relevant categories
     all_cat_common_texts = set(gold_dict.keys()).intersection(pred_dict.keys())
     # For multi-label confusion, we build gold_list & pred_list of sets 
     # that combine these categories.
     gold_list_all = []
     pred_list_all = []
     for txt in sorted(all_cat_common_texts):
         gold_combined = set()
         pred_combined = set()

         for cat in categories_to_process:
             gold_combined.update(gold_dict[txt][cat])
             pred_combined.update(pred_dict[txt][cat])

         gold_list_all.append(gold_combined)
         pred_list_all.append(pred_combined)

     # Then we unify all possible labels from these categories in gold+pred 
     # for confusion counting. We'll unify all gold_labels from categories_to_process 
     # plus all predicted labels as well (extracted from pred_dict).
     # But we'll build that from the actual items to handle any spurious predicted labels.
     union_all_labels = set()
     for gset in gold_list_all:
         union_all_labels.update(gset)
     for pset in pred_list_all:
         union_all_labels.update(pset)

     all_result = compute_multi_label_metrics(gold_list_all, pred_list_all, union_all_labels)
     # We do NOT produce label-by-label for "ALL" again, as that would duplicate 
     # what we've done category-by-category. The user only wants micro/macro for "ALL".
     # We'll define nb_matched_all as the intersection for all categories or just any text matched
     # We'll do the union approach so that nb_matched_all is the number of texts that appear in 
     # both gold and pred for any category in categories_to_process.
     nb_matched_all = len(all_cat_common_texts)

     rows_for_csv.append({
         "method_name": method_name,
         "nb_predictions": nb_predictions,
         "nb_gold_texts": nb_gold_texts,
         "nb_gold_matched": nb_matched_all,
         "total_inference_time": total_inf_time,
         "category": "ALL",
         "label": "(micro)",
         "precision": all_result["micro_precision"],
         "recall": all_result["micro_recall"],
         "f1": all_result["micro_f1"]
     })
     rows_for_csv.append({
         "method_name": method_name,
         "nb_predictions": nb_predictions,
         "nb_gold_texts": nb_gold_texts,
         "nb_gold_matched": nb_matched_all,
         "total_inference_time": total_inf_time,
         "category": "ALL",
         "label": "(macro)",
         "precision": all_result["macro_precision"],
         "recall": all_result["macro_recall"],
         "f1": all_result["macro_f1"]
     })

 # 5. Write out final CSV in vertical format
 # Columns: method_name, nb_predictions, nb_gold_texts, nb_gold_matched, 
 #          total_inference_time, category, label, precision, recall, f1
 os.makedirs(os.path.dirname(output_csv), exist_ok=True)
 fieldnames = [
     "method_name",
     "nb_predictions",
     "nb_gold_texts",
     "nb_gold_matched",
     "total_inference_time",
     "category",
     "label",
     "precision",
     "recall",
     "f1"
 ]

 with open(output_csv, "w", encoding="utf-8", newline="") as f:
     writer = csv.DictWriter(f, fieldnames=fieldnames)
     writer.writeheader()
     for row in rows_for_csv:
         writer.writerow(row)


if __name__ == "__main__":
 main()
