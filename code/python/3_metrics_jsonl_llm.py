"""
 PROJET : vitrine_pipeline

 TITRE : 3_metrics_jsonl_llm.py

 OBJECTIF :
 -----------
 Ce script calcule à la fois :
   1) Des métriques d'accord interannotateurs (Krippendorff’s Alpha) entre les fichiers
      .jsonl annotés manuellement
   2) Des fichiers récapitulant les cas de plein accord (full_agreement) et de désaccord
      (disagreement) pour chaque label
   3) Des métriques d'efficacité (précision, rappel, F1) pour un LLM donné par rapport
      à un consensus de ces annotateurs humains (appelé consensus majoritaire), incluant :
      - Subset accuracy (la proportion de textes pour lesquels la prédiction LLM correspond
        exactement au consensus humain)
      - Le temps d'inférence total pour toutes les lignes réellement prises en compte
        (celles qui sont dans le consensus)
   4) Un Alpha (Krippendorff) séparé entre le LLM et chacun des annotateurs humains
      (ligne __alpha_global__(annotateur)), ainsi qu’un Alpha global combinant
      le LLM et tous les annotateurs dans une seule ligne __alpha_global__.

 Auteur :
 --------
 Antoine Lemor
"""

import os
import json
import csv
import glob
import statistics
from collections import defaultdict, Counter
import itertools


###############################################################################
# 0) Fonction utilitaire supplémentaire pour lire les textes de base.jsonl
###############################################################################
def lire_textes_base(filepath):
    """
    Lit le fichier base.jsonl et retourne un set contenant tous les textes présents.

    Paramètres
    ----------
    filepath : str
        Chemin vers le fichier base.jsonl

    Retour
    ------
    set
        Ensemble des textes extraits du fichier.
    """
    base_set = set()
    if not os.path.exists(filepath):
        return base_set
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                txt = data.get("text", "")
                if txt:
                    base_set.add(txt)
            except Exception:
                continue
    return base_set


###############################################################################
# 1) Fonctions utilitaires de lecture de fichiers et de sélection
###############################################################################
def lire_fichiers_jsonl(dossier):
    """
    Lit tous les fichiers .jsonl dans le dossier spécifié et retourne
    la liste des chemins complets de ces fichiers.

    Paramètres
    ----------
    dossier : str
        Chemin vers le dossier où se trouvent les .jsonl

    Retour
    ------
    list
        Liste des chemins de fichiers .jsonl
    """
    pattern = os.path.join(dossier, "*.jsonl")
    return glob.glob(pattern)


def lire_fichiers_csv(dossier):
    """
    Lit tous les fichiers .csv dans le dossier spécifié et retourne
    la liste des chemins complets de ces fichiers.

    Paramètres
    ----------
    dossier : str
        Chemin vers le dossier où se trouvent les .csv

    Retour
    ------
    list
        Liste des chemins complets de fichiers .csv
    """
    pattern = os.path.join(dossier, "*.csv")
    return glob.glob(pattern)


def demander_selection_fichiers(liste_fichiers, message_invitation):
    """
    Affiche la liste des fichiers disponibles et demande à l'utilisateur
    de sélectionner un ou plusieurs fichiers (indices séparés par des virgules).

    Paramètres
    ----------
    liste_fichiers : list
        Liste des chemins complets de fichiers
    message_invitation : str
        Message affiché pour inviter l'utilisateur à choisir

    Retour
    ------
    list
        Liste des chemins sélectionnés par l'utilisateur
    """
    if not liste_fichiers:
        print("Aucun fichier trouvé.")
        return []

    print(message_invitation)
    for i, f in enumerate(liste_fichiers):
        print(f"{i} -> {f}")
    choix = input("Veuillez entrer les indices des fichiers à sélectionner, séparés par des virgules : ")
    indices = [x.strip() for x in choix.split(",") if x.strip().isdigit()]

    fichiers_selectionnes = []
    for idx in indices:
        idx_int = int(idx)
        if 0 <= idx_int < len(liste_fichiers):
            fichiers_selectionnes.append(liste_fichiers[idx_int])
    return fichiers_selectionnes


###############################################################################
# 2) Lecture et fusion des annotations JSONL
###############################################################################
def nom_court_jsonl(path):
    """
    Extrait un nom de fichier (sans extension) à partir du chemin path.
    Exemple: "/home/test/annotations.jsonl" -> "annotations"
    """
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def lire_annotations_jsonl(filepath, annotateur_id):
    """
    Lit un fichier .jsonl annoté, et retourne un dictionnaire
    { text: { annotateur_id: set(labels) } }

    Paramètres
    ----------
    filepath : str
        Chemin du fichier .jsonl
    annotateur_id : str
        Identifiant de l'annotateur (nom court du fichier .jsonl)

    Retour
    ------
    dict
        Clé = text (str),
        Valeur = { annotateur_id: set(labels) }
    """
    dico = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            txt = data.get("text", "")
            raw_labels = data.get("label", [])
            if isinstance(raw_labels, list):
                assigned_labels = set(raw_labels)
            else:
                assigned_labels = set()

            if txt not in dico:
                dico[txt] = {}
            dico[txt][annotateur_id] = assigned_labels
    return dico


def fusionner_annotations(dicts_list):
    """
    Fusionne plusieurs dictionnaires d'annotations de la forme 
    { text: {annotateur: set(labels)} }
    en un seul dictionnaire global.

    Paramètres
    ----------
    dicts_list : list
        Liste de dictionnaires, chacun obtenu via lire_annotations_jsonl()

    Retour
    ------
    dict
        Clé = text,
        Valeur = { annotateur_id: set(labels), ...}
    """
    fusion = {}
    for d in dicts_list:
        for txt, anots in d.items():
            if txt not in fusion:
                fusion[txt] = {}
            for ann_id, labels in anots.items():
                fusion[txt][ann_id] = labels
    return fusion


def extraire_tous_les_labels(global_dict):
    """
    Extrait l'ensemble de tous les labels possibles dans un dictionnaire global.

    Paramètres
    ----------
    global_dict : dict
        { text : { annotateur : set(labels), ... }, ... }

    Retour
    ------
    set
        Ensemble de tous les labels rencontrés
    """
    all_labels = set()
    for ann_dict in global_dict.values():
        for labels in ann_dict.values():
            all_labels.update(labels)
    return all_labels


###############################################################################
# 3) Création des matrices d'annotation et calcul de Krippendorff’s Alpha
###############################################################################
def construire_matrice_interannotateurs(global_dict, label):
    """
    Construit une matrice de jugements (list of list) pour le calcul
    de Krippendorff’s Alpha sur un label donné.  
    Chaque colonne correspond à un annotateur, chaque ligne correspond à un texte.

    On encode 1 si l'annotateur a attribué le label, 0 sinon.

    Paramètres
    ----------
    global_dict : dict
        { text : { annotateur_id : set(labels), ...}, ...}
    label : str
        Le label pour lequel on veut construire la matrice

    Retour
    ------
    data_matrix : list of list
        data_matrix[i][j] = 1 ou 0
    annotateurs : list
        Liste ordonnée des annotateurs (colonnes)
    textes : list
        Liste ordonnée des textes (lignes)
    """
    annotateurs = sorted(
        set(itertools.chain.from_iterable(
            [ann_dict.keys() for ann_dict in global_dict.values()]
        ))
    )
    textes = sorted(global_dict.keys())

    data_matrix = []
    for txt in textes:
        row = []
        for ann in annotateurs:
            labels_annot = global_dict[txt].get(ann, set())
            val = 1 if label in labels_annot else 0
            row.append(val)
        data_matrix.append(row)

    return data_matrix, annotateurs, textes


def krippendorff_alpha(data_matrix):
    """
    Calcule Krippendorff's Alpha (pour des données nominales binaires)
    à partir d'une matrice data_matrix[i][j].
    Valeurs 0 ou 1 (absence/présence du label).

    Si la matrice n'est pas exploitable (trop peu d'annotateurs ou items),
    renvoie NaN.
    """
    if not data_matrix:
        return float('nan')
    if len(data_matrix[0]) <= 1:
        return float('nan')

    N = len(data_matrix)
    M = len(data_matrix[0])
    if N == 0 or M == 0:
        return float('nan')

    # Discordance observée (Do)
    Do_num = 0.0
    Do_den = 0.0
    for i in range(N):
        row = data_matrix[i]
        for r in range(M):
            for s in range(r+1, M):
                disc = 1 if row[r] != row[s] else 0
                Do_num += disc
                Do_den += 1
    if Do_den == 0:
        return float('nan')
    Do = Do_num / Do_den

    # Discordance attendue (De)
    flat_values = []
    for i in range(N):
        for j in range(M):
            flat_values.append(data_matrix[i][j])

    c = Counter(flat_values)
    total_obs = len(flat_values)
    if total_obs == 0:
        return float('nan')

    p0 = c[0] / total_obs
    p1 = c[1] / total_obs
    De = 2.0 * p0 * p1

    if De == 0:
        return float('nan')

    alpha = 1 - (Do / De)
    return alpha


def calculer_alpha_global(global_dict):
    """
    Calcule l'alpha global en considérant tous les labels 0/1 empilés ensemble.
    On concatène les matrices (label par label) dans une grande matrice.

    Retourne un float (Krippendorff's Alpha global).
    """
    all_labels = sorted(extraire_tous_les_labels(global_dict))
    big_data_matrix = []

    for lbl in all_labels:
        mat_lbl, _, _ = construire_matrice_interannotateurs(global_dict, lbl)
        big_data_matrix.extend(mat_lbl)

    if not big_data_matrix:
        return float('nan')
    return krippendorff_alpha(big_data_matrix)


# Nouvelle fonction pour filtrer les labels commençant par "sentiment_"
def filtrer_dict_sans_sentiment(global_dict):
    new_dict = {}
    for txt, ann_dict in global_dict.items():
        new_ann = {}
        for annot, labels in ann_dict.items():
            new_ann[annot] = {l for l in labels if not l.startswith("sentiment_")}
        new_dict[txt] = new_ann
    return new_dict

def calculer_alpha_global_sans_sentiment(global_dict):
    filtered = filtrer_dict_sans_sentiment(global_dict)
    return calculer_alpha_global(filtered)

# Nouvelle fonction pour calculer Fleiss' Kappa sur une matrice binaire
def fleiss_kappa(data_matrix):
    if not data_matrix:
        return float('nan')
    N = len(data_matrix)
    n = len(data_matrix[0])
    P_i = []
    for row in data_matrix:
        counts = {}
        for val in row:
            counts[val] = counts.get(val, 0) + 1
        if n <= 1:
            P_i.append(0)
        else:
            P_row = sum(c*(c-1) for c in counts.values()) / (n*(n-1))
            P_i.append(P_row)
    P_bar = sum(P_i) / N
    total = N * n
    tot_counts = {}
    for row in data_matrix:
        for val in row:
            tot_counts[val] = tot_counts.get(val, 0) + 1
    p_j = [cnt / total for cnt in sorted(tot_counts)]
    P_e = sum(p**2 for p in p_j)
    if (1 - P_e) == 0:
        return float('nan')
    return (P_bar - P_e) / (1 - P_e)

def fleiss_kappa_global(global_dict):
    all_labels = sorted(extraire_tous_les_labels(global_dict))
    big_data_matrix = []
    for lbl in all_labels:
        mat, _, _ = construire_matrice_interannotateurs(global_dict, lbl)
        big_data_matrix.extend(mat)
    return fleiss_kappa(big_data_matrix)

def fleiss_kappa_global_sans_sentiment(global_dict):
    filtered = filtrer_dict_sans_sentiment(global_dict)
    return fleiss_kappa_global(filtered)


###############################################################################
# 4) Export alpha.csv + colonnes par fichier
###############################################################################
def afficher_repartition_annotateurs_par_label(global_dict):
    """
    Retourne un dictionnaire:
    { annotateur : { label: (count_1, count_0) } }

    count_1 = nombre de textes où l'annotateur a mis le label
    count_0 = nombre de textes où l'annotateur n'a pas mis le label
    """
    all_labels = sorted(extraire_tous_les_labels(global_dict))
    all_annotateurs = sorted(
        set(itertools.chain.from_iterable(
            [ann_dict.keys() for ann_dict in global_dict.values()])
        )
    )

    repartition = {
        ann: {lbl: [0, 0] for lbl in all_labels} for ann in all_annotateurs
    }

    for txt, ann_dict in global_dict.items():
        for ann in all_annotateurs:
            labels = ann_dict.get(ann, set())
            for lbl in all_labels:
                if lbl in labels:
                    repartition[ann][lbl][0] += 1
                else:
                    repartition[ann][lbl][1] += 1

    return repartition


def ecrire_alpha_csv(global_dict, jsonl_names, output_path="data/processed/validation/alpha.csv"):
    """
    Écrit un CSV contenant les informations d'accord interannotateurs (Krippendorff’s Alpha)
    pour chaque label et l'alpha global, entre les différents fichiers annotés.

    Les colonnes de répartition sont <nom_fichier>_positif et <nom_fichier>_negatif.

    Paramètres
    ----------
    global_dict : dict
        Dictionnaire global d'annotations
    jsonl_names : list
        Liste des chemins .jsonl sélectionnés
    output_path : str
        Chemin de sortie, ex: "data/processed/validation/alpha.csv"
    """
    # 1) Calcul alpha par label
    all_labels = sorted(extraire_tous_les_labels(global_dict))
    alpha_par_label = {}
    for lbl in all_labels:
        mat, _, _ = construire_matrice_interannotateurs(global_dict, lbl)
        alpha_par_label[lbl] = krippendorff_alpha(mat)

    alpha_global_ = calculer_alpha_global(global_dict)
    alpha_global_sans = calculer_alpha_global_sans_sentiment(global_dict)
    fleiss_global = fleiss_kappa_global(global_dict)
    fleiss_global_sans = fleiss_kappa_global_sans_sentiment(global_dict)

    # 2) Répartition (positif/négatif)
    short_names = [nom_court_jsonl(n) for n in jsonl_names]
    rep = afficher_repartition_annotateurs_par_label(global_dict)

    # 3) Écriture
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["Krippendorff’s Alpha - InterAnnotateurs"])
        writer.writerow(["JSONL sélectionnés :"] + jsonl_names)
        writer.writerow([])

        header = ["Label", "Alpha_label"]
        for sn in short_names:
            header.append(f"{sn}_positif")
            header.append(f"{sn}_negatif")
        writer.writerow(header)

        for lbl in all_labels:
            row = [lbl, alpha_par_label[lbl]]
            for sn in short_names:
                if sn in rep:
                    pos, neg = rep[sn][lbl]
                else:
                    pos, neg = 0, 0
                row.append(pos)
                row.append(neg)
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(["__alpha_global__ (Krippendorff)", alpha_global_])
        writer.writerow(["__alpha_global__ sans sentiment (Krippendorff)", alpha_global_sans])
        writer.writerow(["__fleiss_kappa_global__", fleiss_global])
        writer.writerow(["__fleiss_kappa_global__ sans sentiment", fleiss_global_sans])

    print(f"[OK] Fichier alpha.csv écrit : {output_path}")


###############################################################################
# 5) Export des cas full agreement et disagreement par label
###############################################################################
def export_full_and_disagreement(global_dict_with_llm, annotator_names):
    """
    Crée pour chaque label deux CSV :
      - full_agreement/<label>.csv : textes où il y a accord parfait
      - disagreement/<label>.csv   : textes où il y a désaccord

    L'accord parfait (full agreement) pour un label signifie :
      - Soit tous les annotateurs l'ont attribué (1),
      - Soit aucun annotateur ne l'a attribué (0).

    Le désaccord est le cas où au moins un annotateur diffère (certains 1, d'autres 0).

    Dans chaque CSV, on crée une colonne "text", puis une colonne par annotateur
    (y compris le LLM, dont le nom est par exemple col_pred) indiquant 1 ou 0
    selon que l'annotateur a mis le label.
    
    Paramètres
    ----------
    global_dict_with_llm : dict
        Dictionnaire combinant annotateurs humains + LLM,
        sous la forme : { text : { "annotateur": set(labels), ..., "col_pred": set(labels) } }
    annotator_names : list
        Liste contenant les noms courts des annotateurs humains + le nom de la colonne du LLM
        (ex. ["jeremy", "shdin", "antoine", "deepseek-r1"])
    """
    base_dir = "data/processed/validation/subvalidation"
    out_full = os.path.join(base_dir, "full_agreement")
    out_dis = os.path.join(base_dir, "disagreement")
    os.makedirs(out_full, exist_ok=True)
    os.makedirs(out_dis, exist_ok=True)

    all_labels = sorted(extraire_tous_les_labels(global_dict_with_llm))
    all_texts = sorted(global_dict_with_llm.keys())

    for lbl in all_labels:
        full_path = os.path.join(out_full, f"{lbl}.csv")
        dis_path = os.path.join(out_dis, f"{lbl}.csv")

        with open(full_path, "w", encoding="utf-8", newline="") as f_full, \
             open(dis_path, "w", encoding="utf-8", newline="") as f_dis:
            w_full = csv.writer(f_full, delimiter=",")
            w_dis = csv.writer(f_dis, delimiter=",")

            header = ["text"] + annotator_names
            w_full.writerow(header)
            w_dis.writerow(header)

            for txt in all_texts:
                row_vals = []
                for ann in annotator_names:
                    labels_for_ann = global_dict_with_llm[txt].get(ann, set())
                    val = 1 if lbl in labels_for_ann else 0
                    row_vals.append(val)
                all_identical = (len(set(row_vals)) == 1)
                row_to_write = [txt] + row_vals

                if all_identical:
                    w_full.writerow(row_to_write)
                else:
                    w_dis.writerow(row_to_write)


###############################################################################
# 6) Calcul du consensus majoritaire
###############################################################################
def calculer_consensus_majoritaire(global_dict):
    """
    Pour chaque text, calcule l'ensemble des labels validés par la majorité
    (strictement plus de la moitié des annotateurs).

    Retour
    ------
    consensus_dict : dict
        text -> set(labels majoritaires)
    """
    all_texts = sorted(global_dict.keys())
    all_annotateurs = sorted(
        set(itertools.chain.from_iterable(
            [ann_dict.keys() for ann_dict in global_dict.values()]
        ))
    )
    nb_annot = len(all_annotateurs)
    threshold = nb_annot / 2.0
    all_labels = sorted(extraire_tous_les_labels(global_dict))

    consensus_dict = {}
    for txt in all_texts:
        labels_consensus = set()
        for lbl in all_labels:
            count_lbl = sum(1 for ann in all_annotateurs if lbl in global_dict[txt].get(ann, set()))
            if count_lbl > threshold:
                labels_consensus.add(lbl)
        consensus_dict[txt] = labels_consensus
    return consensus_dict


###############################################################################
# 7) Lecture CSV prédiction LLM + calcul performance
###############################################################################
def lire_csv_predictions(filepath):
    """
    Lit un fichier CSV et retourne (rows, fieldnames).
    rows: list de dict
    fieldnames: liste des noms de colonnes
    """
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames


def extraire_labels_pred_llm(raw_pred, valid_labels=None):
    """
    Extrait un set de labels à partir d'une prédiction LLM sous forme JSON ou string.

    Format JSON attendu (exemple):
      {
        "themes": ["environment", ...] ou "null",
        "political_parties": "CPC" ou ["CPC"] ou "null",
        "specific_themes": ...,
        "sentiment": ...
      }

    En cas d'échec de la charge JSON, on effectue un split par virgule.

    Si valid_labels est fourni (set), seuls les labels présents dans valid_labels
    seront conservés (les autres seront ignorés, considérés comme réponse nulle).
    """
    pred_labels = set()
    if not raw_pred.strip():
        return pred_labels

    try:
        parsed_pred = json.loads(raw_pred)
        them = parsed_pred.get("themes", None)
        if isinstance(them, list):
            for val in them:
                if val and val != "null":
                    pred_labels.add(f"theme_{val}")
        elif isinstance(them, str) and them.lower() != "null":
            pred_labels.add(f"theme_{them}")

        pol = parsed_pred.get("political_parties", None)
        if isinstance(pol, list):
            for val in pol:
                if val and val.lower() != "null":
                    pred_labels.add(f"political_parties_{val}")
        elif isinstance(pol, str) and pol.lower() != "null":
            pred_labels.add(f"political_parties_{pol}")

        spec = parsed_pred.get("specific_themes", None)
        if isinstance(spec, list):
            for val in spec:
                if val and val.lower() != "null":
                    pred_labels.add(f"specific_themes_{val}")
        elif isinstance(spec, str) and spec.lower() != "null":
            pred_labels.add(f"specific_themes_{spec}")

        sent = parsed_pred.get("sentiment", None)
        if sent and sent.lower() != "null":
            pred_labels.add(f"sentiment_{sent}")

    except:
        parts = [x.strip() for x in raw_pred.split(",")]
        for p in parts:
            if p:
                pred_labels.add(p)

    if valid_labels is not None:
        pred_labels = pred_labels.intersection(valid_labels)
    return pred_labels


def calculer_scores_llm(rows_csv, global_dict, col_text, col_pred, col_time):
    """
    Calcule :
      - Un dictionnaire metrics_dict : {label: {tp, fp, fn, gold_count, pred_count, inference_times}}
      - Un consensus_dict (text -> set de labels majoritaires)
      - subset_accuracy : proportion de textes où LLM == consensus
      - total_inference_time : somme des temps d'inférence pour tous les textes considérés

    Paramètres
    ----------
    rows_csv : list de dict
        Contenu du CSV
    global_dict : dict
        Annotations humaines {text: {annotateur: set(labels)}}
    col_text : str
        Nom de la colonne contenant le texte
    col_pred : str
        Nom de la colonne contenant la prédiction du LLM
    col_time : str
        Nom de la colonne contenant le temps d'inférence

    Retour
    ------
    metrics_dict, consensus_dict, subset_accuracy, total_inference_time
    """
    consensus_dict = calculer_consensus_majoritaire(global_dict)
    metrics_dict = defaultdict(lambda: {
        "tp": 0, "fp": 0, "fn": 0,
        "gold_count": 0, "pred_count": 0,
        "inference_times": []
    })
    nb_exact_match = 0
    nb_total_consensus = 0
    total_inference_time = 0.0

    # Calculer l'ensemble des labels manuels valides
    valid_labels = extraire_tous_les_labels(global_dict)
    consensus_texts = set(consensus_dict.keys())

    for row in rows_csv:
        txt = row.get(col_text, "").strip()
        if txt not in consensus_texts:
            continue

        gold_labels = consensus_dict[txt]
        raw_pred = row.get(col_pred, "")
        pred_labels = extraire_labels_pred_llm(raw_pred, valid_labels)

        nb_total_consensus += 1
        if pred_labels == gold_labels:
            nb_exact_match += 1

        t_inf_str = row.get(col_time, "")
        try:
            t_inf = float(t_inf_str)
        except:
            t_inf = 0.0
        total_inference_time += t_inf

        all_labels_texte = gold_labels.union(pred_labels)
        for lbl in all_labels_texte:
            if lbl in gold_labels:
                metrics_dict[lbl]["gold_count"] += 1
            if lbl in pred_labels:
                metrics_dict[lbl]["pred_count"] += 1

            if (lbl in gold_labels) and (lbl in pred_labels):
                metrics_dict[lbl]["tp"] += 1
                metrics_dict[lbl]["inference_times"].append(t_inf)
            elif (lbl not in gold_labels) and (lbl in pred_labels):
                metrics_dict[lbl]["fp"] += 1
                metrics_dict[lbl]["inference_times"].append(t_inf)
            elif (lbl in gold_labels) and (lbl not in pred_labels):
                metrics_dict[lbl]["fn"] += 1

    subset_accuracy = nb_exact_match / nb_total_consensus if nb_total_consensus > 0 else 0.0
    return metrics_dict, consensus_dict, subset_accuracy, total_inference_time


def calculer_scores_aggreges(metrics_dict, subset_accuracy=0.0, total_inference_time=0.0):
    """
    Calcule micro-F1, macro-F1, weighted-F1 en plus de subset_accuracy et total_inference_time.

    Retour
    ------
    dict
      {
        "micro": (p, r, f1),
        "macro": (p, r, f1),
        "weighted": (p, r, f1),
        "subset_accuracy": float,
        "total_inference_time": float
      }
    """
    sum_tp = sum(m["tp"] for m in metrics_dict.values())
    sum_fp = sum(m["fp"] for m in metrics_dict.values())
    sum_fn = sum(m["fn"] for m in metrics_dict.values())

    micro_precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    label_precisions = []
    label_recalls = []
    label_f1s = []
    total_support = 0
    for lbl, d in metrics_dict.items():
        tp = d["tp"]
        fp = d["fp"]
        fn = d["fn"]
        denom_p = tp + fp
        denom_r = tp + fn
        p_ = tp / denom_p if denom_p else 0.0
        r_ = tp / denom_r if denom_r else 0.0
        f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
        label_precisions.append(p_)
        label_recalls.append(r_)
        label_f1s.append(f1_)
        total_support += (tp + fn)

    macro_precision = statistics.mean(label_precisions) if label_precisions else 0.0
    macro_recall = statistics.mean(label_recalls) if label_recalls else 0.0
    macro_f1 = statistics.mean(label_f1s) if label_f1s else 0.0

    if total_support == 0:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    else:
        sum_wp, sum_wr, sum_wf = 0.0, 0.0, 0.0
        for lbl, d in metrics_dict.items():
            tp = d["tp"]
            fp = d["fp"]
            fn = d["fn"]
            denom_p = tp + fp
            denom_r = tp + fn
            p_ = tp / denom_p if denom_p else 0.0
            r_ = tp / denom_r if denom_r else 0.0
            f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
            support = tp + fn
            sum_wp += p_ * support
            sum_wr += r_ * support
            sum_wf += f1_ * support

        weighted_precision = sum_wp / total_support
        weighted_recall = sum_wr / total_support
        weighted_f1 = sum_wf / total_support

    return {
        "micro": (micro_precision, micro_recall, micro_f1),
        "macro": (macro_precision, macro_recall, macro_f1),
        "weighted": (weighted_precision, weighted_recall, weighted_f1),
        "subset_accuracy": subset_accuracy,
        "total_inference_time": total_inference_time
    }


###############################################################################
# 8) Calculer un alpha séparé (LLM vs chaque fichier .jsonl) et un alpha global combiné
###############################################################################
def calculer_alpha_llm_vs_annotateurs(rows_csv, global_dict, col_text, col_pred):
    """
    Calcule, pour chaque annotateur (chaque fichier .jsonl), Krippendorff’s Alpha
    entre le LLM et cet annotateur, par label + global.

    Retourne un dict :
      {
        "<nom_fichier>": {
            "<label>": alpha_val,
            ...
            "__alpha_global__": alpha_glob
        },
        ...
      }
    """
    # Calculer l'ensemble des labels manuels valides
    valid_labels = extraire_tous_les_labels(global_dict)
    llm_dict = {}
    for row in rows_csv:
        txt = row.get(col_text, "").strip()
        raw_pred = row.get(col_pred, "")
        pred_labels = extraire_labels_pred_llm(raw_pred, valid_labels)
        if txt not in llm_dict:
            llm_dict[txt] = {}
        llm_dict[txt]["LLM"] = pred_labels

    all_annotateurs = sorted(
        set(itertools.chain.from_iterable(
            [ann_dict.keys() for ann_dict in global_dict.values()]
        ))
    )

    results = {}
    for ann in all_annotateurs:
        mini_dict = {}
        for txt, ann_dict in global_dict.items():
            if ann in ann_dict and txt in llm_dict:
                sets_ = {ann: ann_dict[ann], "LLM": llm_dict[txt]["LLM"]}
                mini_dict[txt] = sets_

        labels_here = extraire_tous_les_labels(mini_dict)
        alpha_lbl_dict = {}
        for lbl in sorted(labels_here):
            mat, _, _ = construire_matrice_interannotateurs(mini_dict, lbl)
            alpha_lbl_dict[lbl] = krippendorff_alpha(mat)
        alpha_lbl_dict["__alpha_global__"] = calculer_alpha_global(mini_dict)
        results[ann] = alpha_lbl_dict

    return results


def construire_dict_annot_llm_complet(global_dict, rows_csv, col_text, col_pred):
    """
    Construit un dictionnaire global contenant tous les annotateurs + une clé "LLM".
    { text: { ann1: set(...), ann2: set(...), ..., "LLM": set(...) } }

    On ajoute la clé "LLM" seulement pour les textes présents dans le CSV.
    """
    # Calculer l'ensemble des labels manuels valides
    valid_labels = extraire_tous_les_labels(global_dict)
    big_dict = {}
    for txt, ann_dict in global_dict.items():
        big_dict[txt] = {}
        for ann, labset in ann_dict.items():
            big_dict[txt][ann] = labset

    for row in rows_csv:
        txt = row.get(col_text, "").strip()
        if txt not in big_dict:
            continue
        raw_pred = row.get(col_pred, "")
        pred_labels = extraire_labels_pred_llm(raw_pred, valid_labels)
        big_dict[txt]["LLM"] = pred_labels

    return big_dict


###############################################################################
# 9) Écriture de metrics.csv
###############################################################################
def ecrire_metrics_csv(metrics_dict, aggregats, consensus_dict,
                       alpha_llm_vs_ann, annotateurs_order, alpha_global_combine,
                       output_path="data/processed/validation/metrics.csv"):
    """
    Écrit metrics.csv, contenant : 
    - une ligne par label : label, gold_count, pred_count, precision, recall, F1, avg_inference_time
      + alpha_(annotateur1), alpha_(annotateur2), ...
    - puis en bas : __micro__, __macro__, __subset_accuracy__, __weighted__, 
      __alpha_global__(annotateur), et __alpha_global__
      enfin, __total_inference_time__ en dernière ligne.

    Paramètres
    ----------
    metrics_dict : dict[label] = { "tp", "fp", "fn", "gold_count", "pred_count", "inference_times": [...]}
    aggregats : dict
        {
          "micro": (p, r, f1),
          "macro": (p, r, f1),
          "weighted": (p, r, f1),
          "subset_accuracy": float,
          "total_inference_time": float
        }
    consensus_dict : dict
        (fourni pour compatibilité ; pas utilisé directement)
    alpha_llm_vs_ann : dict 
        { "annot_name": { lbl: alpha_val, ..., "__alpha_global__": alphaG } }
    annotateurs_order : list (noms courts, ex: ["jeremy", "shdin", "antoine"])
    alpha_global_combine : float
        Alpha combiné (LLM + tous annotateurs)
    output_path : str
        Chemin de sortie du fichier metrics.csv
    """
    header = [
        "label", "gold_count", "pred_count",
        "precision", "recall", "F1",
        "avg_inference_time"
    ]
    for ann in annotateurs_order:
        header.append(f"alpha_{ann}")

    rows = []
    sorted_labels = sorted(metrics_dict.keys())

    for lbl in sorted_labels:
        tp = metrics_dict[lbl]["tp"]
        fp = metrics_dict[lbl]["fp"]
        fn = metrics_dict[lbl]["fn"]
        gold_count = metrics_dict[lbl]["gold_count"]
        pred_count = metrics_dict[lbl]["pred_count"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1_ = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        times = metrics_dict[lbl]["inference_times"]
        avg_time = statistics.mean(times) if times else 0.0

        row_data = [
            lbl,
            gold_count,
            pred_count,
            round(prec, 4),
            round(rec, 4),
            round(f1_, 4),
            round(avg_time, 4)
        ]
        for ann in annotateurs_order:
            alpha_val = alpha_llm_vs_ann.get(ann, {}).get(lbl, float('nan'))
            row_data.append(alpha_val)

        rows.append(row_data)

    micro_p, micro_r, micro_f1 = aggregats["micro"]
    macro_p, macro_r, macro_f1 = aggregats["macro"]
    w_p, w_r, w_f1 = aggregats["weighted"]
    subset_acc = aggregats["subset_accuracy"]
    total_inf_time = aggregats["total_inference_time"]

    def agg_row(label_agg, p, r, f1):
        return [
            label_agg, "", "", round(p, 4), round(r, 4), round(f1, 4), ""
        ] + ["" for _ in annotateurs_order]

    rows.append(agg_row("__micro__", micro_p, micro_r, micro_f1))
    rows.append(agg_row("__macro__", macro_p, macro_r, macro_f1))

    row_subset = [
        "__subset_accuracy__", "", "", "", "", round(subset_acc, 4), ""
    ] + ["" for _ in annotateurs_order]
    rows.append(row_subset)

    rows.append(agg_row("__weighted__", w_p, w_r, w_f1))

    for ann in annotateurs_order:
        alpha_g = alpha_llm_vs_ann.get(ann, {}).get("__alpha_global__", float('nan'))
        row_alpha_g = [
            f"__alpha_global__({ann})", "", "", "", "", "", ""
        ]
        for a2 in annotateurs_order:
            if a2 == ann:
                row_alpha_g.append(alpha_g)
            else:
                row_alpha_g.append("")
        rows.append(row_alpha_g)

    alpha_global_row = [
        "__alpha_global__", "", "", "", "", "", ""
    ]
    if annotateurs_order:
        alpha_global_row.append(alpha_global_combine)
        for _ in annotateurs_order[1:]:
            alpha_global_row.append("")
    rows.append(alpha_global_row)

    row_total_time = [
        "__total_inference_time__", "", "", "", "", "", round(total_inf_time, 4)
    ]
    for _ in annotateurs_order:
        row_total_time.append("")
    rows.append(row_total_time)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["Métriques de performance du modèle (Consensus majoritaire vs LLM)"])
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    print(f"[OK] Fichier metrics.csv écrit : {output_path}")

###############################################################################
# NEW : Lecture du gold_count.jsonl et normalisation des labels
###############################################################################
def lire_gold_count_jsonl(filepath):
    """
    Lit le fichier gold_count.jsonl et retourne un dict { text: set(normalized_labels) }.
    Pour chaque label, seul le préfixe (avant le dernier '_') est conservé.
    """
    gold = {}
    if not os.path.exists(filepath):
        return gold
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                txt = data.get("text", "").strip()
                raw_labels = data.get("label", [])
                norm_labels = set()
                for lab in raw_labels:
                    if "_" in lab:
                        norm_labels.add(lab.rsplit("_", 1)[0])
                    else:
                        norm_labels.add(lab)
                if txt:
                    gold[txt] = norm_labels
            except Exception:
                continue
    return gold

###############################################################################
# NEW : Calcul des métriques à partir du gold_count.jsonl
###############################################################################
def calculer_scores_llm_bis(rows_csv, gold_standard, col_text, col_pred, col_time):
    """
    Calcule les métriques d'efficacité en comparant les prédictions LLM (CSV)
    au gold standard provenant de gold_count.jsonl.
    Retourne (metrics_dict, subset_accuracy, total_inference_time).
    """
    metrics_dict = defaultdict(lambda: {
        "tp": 0, "fp": 0, "fn": 0,
        "gold_count": 0, "pred_count": 0,
        "inference_times": []
    })
    nb_exact_match = 0
    nb_total = 0
    total_inference_time = 0.0

    # Déduire l'ensemble des labels du gold standard
    valid_labels = set()
    for labs in gold_standard.values():
        valid_labels.update(labs)

    for row in rows_csv:
        txt = row.get(col_text, "").strip()
        if txt not in gold_standard:
            continue

        gold_labels = gold_standard[txt]
        raw_pred = row.get(col_pred, "")
        pred_labels = extraire_labels_pred_llm(raw_pred, valid_labels)

        nb_total += 1
        if pred_labels == gold_labels:
            nb_exact_match += 1

        try:
            t_inf = float(row.get(col_time, "0"))
        except:
            t_inf = 0.0
        total_inference_time += t_inf

        all_labels = gold_labels.union(pred_labels)
        for lbl in all_labels:
            if lbl in gold_labels:
                metrics_dict[lbl]["gold_count"] += 1
            if lbl in pred_labels:
                metrics_dict[lbl]["pred_count"] += 1
            if (lbl in gold_labels) and (lbl in pred_labels):
                metrics_dict[lbl]["tp"] += 1
                metrics_dict[lbl]["inference_times"].append(t_inf)
            elif (lbl not in gold_labels) and (lbl in pred_labels):
                metrics_dict[lbl]["fp"] += 1
                metrics_dict[lbl]["inference_times"].append(t_inf)
            elif (lbl in gold_labels) and (lbl not in pred_labels):
                metrics_dict[lbl]["fn"] += 1

    subset_accuracy = nb_exact_match / nb_total if nb_total > 0 else 0.0
    return metrics_dict, subset_accuracy, total_inference_time

###############################################################################
# NEW : Écriture de metrics_bis.csv
###############################################################################
def ecrire_metrics_bis_csv(metrics_dict, aggregats, output_path="data/processed/validation/metrics_bis.csv"):
    """
    Écrit metrics_bis.csv contenant :
      - une ligne par label : label, gold_count, pred_count, precision, recall, F1, avg_inference_time
      - puis en bas, les agrégats : __micro__, __macro__, __subset_accuracy__, __weighted__, __total_inference_time__
    """
    header = ["label", "gold_count", "pred_count", "precision", "recall", "F1", "avg_inference_time"]
    rows = []
    sorted_labels = sorted(metrics_dict.keys())
    for lbl in sorted_labels:
        tp = metrics_dict[lbl]["tp"]
        fp = metrics_dict[lbl]["fp"]
        fn = metrics_dict[lbl]["fn"]
        gold_count = metrics_dict[lbl]["gold_count"]
        pred_count = metrics_dict[lbl]["pred_count"]
        prec = tp / (tp+fp) if (tp+fp) > 0 else 0.0
        rec = tp / (tp+fn) if (tp+fn) > 0 else 0.0
        f1_ = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        times = metrics_dict[lbl]["inference_times"]
        avg_time = statistics.mean(times) if times else 0.0
        rows.append([lbl, gold_count, pred_count, round(prec,4), round(rec,4), round(f1_,4), round(avg_time,4)])

    micro_p, micro_r, micro_f1 = aggregats["micro"]
    macro_p, macro_r, macro_f1 = aggregats["macro"]
    subset_acc = aggregats["subset_accuracy"]
    total_inf_time = aggregats["total_inference_time"]

    rows.append(["__micro__", "", "", round(micro_p,4), round(micro_r,4), round(micro_f1,4), ""])
    rows.append(["__macro__", "", "", round(macro_p,4), round(macro_r,4), round(macro_f1,4), ""])
    rows.append(["__subset_accuracy__", "", "", "", "", round(subset_acc,4), ""])
    rows.append(["__total_inference_time__", "", "", "", "", "", round(total_inf_time,4)])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["Métriques d'efficacité (Gold_count vs LLM)"])
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f"[OK] Fichier metrics_bis.csv écrit : {output_path}")



###############################################################################
# 10) MAIN
###############################################################################
def main():
    """
    Point d'entrée principal du script.

    1) Sélection des .jsonl annotés -> fusion et calcul de Krippendorff’s Alpha (alpha.csv).
       (Ici, on compare uniquement les annotateurs humains)
    2) Sélection d'un fichier CSV de prédictions LLM -> calcul des métriques
       (précision, rappel, F1, subset_accuracy, total_inference_time),
       alpha séparé LLM vs annotateurs et alpha global combiné.
    3) À la fin, on produit également les fichiers de full_agreement et disagreement,
       cette fois-ci en incluant le LLM (dont le "nom d'annotateur" est col_pred).
    4) Écriture de metrics.csv final.
    """
    ###########################################################################
    # Étape 1) Sélection des .jsonl annotés et calcul interannotateur (alpha.csv)
    ###########################################################################
    dossier_jsonl = os.path.join("data", "processed", "validation", "annotated_jsonl")
    fichiers_jsonl = lire_fichiers_jsonl(dossier_jsonl)
    jsonl_choisis = demander_selection_fichiers(
        fichiers_jsonl,
        "Sélectionnez les fichiers .jsonl à fusionner pour calculer Krippendorff’s Alpha :"
    )
    if not jsonl_choisis:
        print("Aucun fichier .jsonl sélectionné, le script s'arrête.")
        return

    list_of_dicts = []
    for fpath in jsonl_choisis:
        annot_id = nom_court_jsonl(fpath)
        dico_ = lire_annotations_jsonl(fpath, annot_id)
        list_of_dicts.append(dico_)
    full_global_dict = fusionner_annotations(list_of_dicts)

    # Demander si on doit exclure les textes de base.jsonl pour le comparatif interannotateurs
    rep_inter = input("Voulez-vous exclure les annotations contenues dans 'data/processed/validation/annotated_jsonl/base.jsonl' pour le calcul comparatif interannotateurs ? (oui/non) : ").strip().lower()
    exclude_base_inter = (rep_inter == "oui")
    base_file = os.path.join("data", "processed", "validation", "annotated_jsonl", "base.jsonl")
    base_texts = lire_textes_base(base_file)
    if base_texts:
        print(f"{len(base_texts)} textes trouvés dans base.jsonl.")
    else:
        print("Aucun texte trouvé dans base.jsonl ou fichier non existant.")

    if exclude_base_inter and base_texts:
        inter_global_dict = {txt: anns for txt, anns in full_global_dict.items() if txt not in base_texts}
        print(f"Après exclusion, {len(inter_global_dict)} textes restent pour le calcul interannotateurs.")
    else:
        inter_global_dict = full_global_dict

    alpha_csv_path = "data/processed/validation/alpha.csv"
    ecrire_alpha_csv(inter_global_dict, jsonl_choisis, output_path=alpha_csv_path)

    ###########################################################################
    # Étape 2) Sélection du CSV de prédictions LLM et calcul des métriques
    ###########################################################################
    dossier_csv = os.path.join("data", "processed", "subset")
    fichiers_csv = lire_fichiers_csv(dossier_csv)
    csv_choisis = demander_selection_fichiers(
        fichiers_csv,
        "Sélectionnez le fichier .csv pour calculer les métriques d'efficacité :"
    )
    if not csv_choisis:
        print("Aucun fichier CSV sélectionné, le script s'arrête.")
        return

    csv_path = csv_choisis[0]
    rows_csv, fieldnames = lire_csv_predictions(csv_path)

    print("\nColonnes disponibles :", fieldnames)
    col_text = input("Entrez le nom de la colonne contenant le texte original (col_text) : ").strip()
    while col_text not in fieldnames:
        col_text = input("Colonne invalide pour le texte. Recommencez : ").strip()

    col_pred = input("Entrez le nom de la colonne contenant la prédiction du LLM (col_pred) : ").strip()
    while col_pred not in fieldnames:
        col_pred = input("Colonne invalide pour la prédiction. Recommencez : ").strip()

    while col_pred == col_text:
        print("La colonne de prédiction (col_pred) ne peut pas être la même que la colonne du texte (col_text).")
        col_pred = input("Veuillez entrer une colonne valide pour la prédiction : ").strip()
        while col_pred not in fieldnames:
            col_pred = input("Colonne invalide pour la prédiction. Recommencez : ").strip()

    col_time = input("Entrez le nom de la colonne contenant le temps d'inférence (col_time) : ").strip()
    while col_time not in fieldnames:
        col_time = input("Colonne invalide pour le temps d'inférence. Recommencez : ").strip()

    while col_time == col_pred:
        print("La colonne du temps d'inférence (col_time) ne peut pas être la même que la colonne de prédiction (col_pred).")
        col_time = input("Veuillez entrer une colonne valide pour le temps d'inférence : ").strip()
        while col_time not in fieldnames:
            col_time = input("Colonne invalide pour le temps d'inférence. Recommencez : ").strip()

    # Demander si on doit exclure les textes de base.jsonl pour le comparatif avec le LLM
    rep_llm = input("Voulez-vous exclure les annotations contenues dans 'data/processed/validation/annotated_jsonl/base.jsonl' pour le calcul comparatif avec le LLM ? (oui/non) : ").strip().lower()
    exclude_base_llm = (rep_llm == "oui")
    if exclude_base_llm and base_texts:
        llm_global_dict = {txt: anns for txt, anns in full_global_dict.items() if txt not in base_texts}
        print(f"Après exclusion, {len(llm_global_dict)} textes restent pour le calcul comparatif avec le LLM.")
    else:
        llm_global_dict = full_global_dict

    metrics_dict, consensus_dict, subset_acc, total_time = calculer_scores_llm(
        rows_csv, llm_global_dict, col_text, col_pred, col_time
    )
    if len(metrics_dict) == 0:
        print(
            "\nATTENTION : Aucune ligne du CSV ne correspond aux textes annotés.\n"
            "Le fichier metrics.csv risque de contenir uniquement des valeurs nulles ou NaN."
        )

    aggregats = calculer_scores_aggreges(metrics_dict, subset_acc, total_time)

    ###########################################################################
    # Étape 3) Construction d'un dictionnaire combinant HUMAIN + LLM et export full_agreement/disagreement
    ###########################################################################
    big_dict = construire_dict_annot_llm_complet(llm_global_dict, rows_csv, col_text, col_pred)
    big_dict_with_llm = {}
    for txt, ann_dict in big_dict.items():
        big_dict_with_llm[txt] = {}
        for ann_name, ann_labels in ann_dict.items():
            if ann_name == "LLM":
                big_dict_with_llm[txt][col_pred] = ann_labels
            else:
                big_dict_with_llm[txt][ann_name] = ann_labels

    short_names = [nom_court_jsonl(fp) for fp in jsonl_choisis]
    annotateurs_plus_llm = short_names + [col_pred]
    export_full_and_disagreement(big_dict_with_llm, annotateurs_plus_llm)

    ###########################################################################
    # Étape 4) Calcul alpha LLM vs annotateurs + écriture de metrics.csv
    ###########################################################################
    llm_vs_ann = calculer_alpha_llm_vs_annotateurs(rows_csv, llm_global_dict, col_text, col_pred)
    alpha_global_combine = calculer_alpha_global(big_dict)  # big_dict inclut 'LLM'

    metrics_out = "data/processed/validation/metrics.csv"
    ecrire_metrics_csv(
        metrics_dict,
        aggregats,
        consensus_dict,
        llm_vs_ann,
        short_names,  # Conserver l'ordre des annotateurs humains
        alpha_global_combine,
        output_path=metrics_out
    )

    ###########################################################################
    # Calcul des métriques à partir du gold_count.jsonl et écriture de metrics_bis.csv
    ###########################################################################
    gold_count_file = os.path.join("data", "processed", "validation", "subvalidation", "gold_count.jsonl")
    gold_standard = lire_gold_count_jsonl(gold_count_file)
    if not gold_standard:
        print("Aucun gold standard trouvé dans gold_count.jsonl.")
    else:
        metrics_dict_bis, subset_acc_bis, total_time_bis = calculer_scores_llm_bis(
            rows_csv, gold_standard, col_text, col_pred, col_time
        )
        aggregats_bis = calculer_scores_aggreges(metrics_dict_bis, subset_acc_bis, total_time_bis)
        metrics_bis_out = os.path.join("data", "processed", "validation", "metrics_bis.csv")
        ecrire_metrics_bis_csv(metrics_dict_bis, aggregats_bis, output_path=metrics_bis_out)

if __name__ == "__main__":
    main()