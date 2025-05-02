"""
PROJET:
--------
vitrine_pipeline

FICHIER:
--------
4_training_from_LLMs.py

DESCRIPTION:
------------
Ce script entraîne des modèles BERT/Camembert spécialisés basés sur les annotations produites par des LLM locaux,
en s’appuyant sur la bibliothèque 'AugmentedSocialScientistFork' (fourche maison de AugmentedSocialScientist) 
pour gérer la logique des transformers. Il :

    1) Sélectionne de manière interactive un fichier CSV et une colonne d’annotation.
    2) Crée des ensembles de données JSONL binaires (train/validation) par étiquette d’annotation (excluant 'null').
    3) Entraîne Camembert pour le français et BERT pour l’anglais avec 20 époques normales.
    4) Sauvegarde les logs pour chaque époque dans data/processed/validation/LLMs_training.
    5) Déclenche automatiquement un apprentissage renforcé (époques supplémentaires avec suréchantillonnage de la classe) si
       le score F1 de la classe 1 est inférieur à 0,7 après l’entraînement normal.
    6) Implémente une logique de secours qui sélectionne la 5ᵉ époque renforcée si aucune époque renforcée
       ne dépasse 0 en F1 pour la classe 1 (càd reste à 0). Si une époque dépasse 0, la
       première de celles-ci est sélectionnée si elle apporte une amélioration du F1 de la classe 1.
    7) Agrège les métriques finales dans un seul CSV (data/processed/validation/all_best_models.csv)
       contenant uniquement les métriques du modèle final pour chaque paire d’étiquettes entraînée.

AUTEUR:
-------
Antoine Lemor
"""

import os
import sys
import json
import glob
import shutil
import random
import datetime
import time
import csv
import numpy as np
import pandas as pd
import torch

from torch.types import Device
from tqdm.auto import tqdm

from AugmentedSocialScientistFork.models import Bert, Camembert

# -----------------------------
# REMPLACER LES CHEMINS ABSOLUS PAR DES CHEMINS RELATIFS
# -----------------------------
# Nous utilisons base_path = "." pour garder tous les chemins relatifs et portables.
base_path = "."

# --------------------------------------------------------------------
# SÉLECTION DU DEVICE
# --------------------------------------------------------------------
def get_device() -> torch.device:
    """
    Détecte le GPU disponible (CUDA ou MPS) ou utilise le CPU par défaut.

    Returns
    -------
    torch.device
        Le device sélectionné pour l'entraînement.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Utilisation du GPU (CUDA).")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Utilisation du GPU (MPS).")
    else:
        device = torch.device("cpu")
        print("Utilisation du CPU.")
    return device


# --------------------------------------------------------------------
# JOURNALISATION SIMPLE
# --------------------------------------------------------------------
class Logger:
    """
    Logger qui duplique stdout à la fois dans la console et dans un fichier de log.

    Attributes
    ----------
    terminal : file-like
        Référence originale de stdout.
    log : file-like
        Fichier où les logs sont écrits.
    """

    def __init__(self, filename: str):
        """
        Initialise le logger.

        Parameters
        ----------
        filename : str
            Chemin vers le fichier où les logs seront écrits.
        """
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        """
        Écrit un message à la fois sur stdout et dans le fichier de log.

        Parameters
        ----------
        message : str
            Message à logger.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        """
        Vide le buffer du log. Requis pour certaines bibliothèques I/O.
        """
        pass

    def close(self) -> None:
        """
        Ferme le fichier de log.
        """
        self.log.close()


# --------------------------------------------------------------------
# CHARGER JSONL DANS UN DATAFRAME
# --------------------------------------------------------------------
def load_jsonl_to_dataframe(filepath: str) -> pd.DataFrame:
    """
    Charge un fichier JSONL dans un DataFrame pandas.

    Parameters
    ----------
    filepath : str
        Chemin vers le fichier JSONL.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les données JSONL.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# --------------------------------------------------------------------
# CRÉER LES ENSEMBLES DE DONNÉES D'ENTRAÎNEMENT/VALIDATION
# --------------------------------------------------------------------
def create_training_datasets(csv_path: str, annotation_column: str) -> None:
    """
    Construit des datasets JSONL binaires.
    label = 1  → ligne contient la paire (clé, label)
    label = 0  → ligne a une annotation quelconque mais pas cette paire
    Aucune ligne n'est ajoutée si la cellule d'annotation est vide/null.
    """
    output_base = os.path.join(base_path, "data", "processed", "training_LLMs")

    # ── 1. Lecture du CSV ────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"Chargé {len(df):,} lignes depuis {csv_path}")

    # ── 2. Recensement des labels possibles par clé ──────────────────
    candidate_labels: dict[str, set[str]] = {}
    for annot in df[annotation_column].dropna():
        try:
            d = json.loads(annot)
        except Exception:
            continue
        for k, v in d.items():
            if v in (None, [], "", "null"):
                continue
            if isinstance(v, list):
                candidate_labels.setdefault(k, set()).update(
                    {lab for lab in v if str(lab).lower() != "null"}
                )
            else:
                if str(v).lower() != "null":
                    candidate_labels.setdefault(k, set()).add(v)

    # ── 3. Construction des exemples ────────────────────────────────
    data_by_pair: dict[str, dict[str, list[dict]]] = {}
    for _, row in df.iterrows():
        try:
            annotations = json.loads(row.get(annotation_column, "{}"))
        except Exception:
            continue  # annotation malformée ⇒ ignorer
        if not annotations:  # pas d'annotation ⇒ aucun exemple
            continue

        lang = str(row.get("lang", "")).strip().upper()
        text = str(row.get("text", "")).strip()

        for key, labels in candidate_labels.items():
            row_val = annotations.get(key)
            for lab in labels:
                if isinstance(row_val, list):
                    is_positive = lab in row_val
                else:
                    is_positive = row_val == lab

                pair = f"{key}_{lab}".lower().replace(" ", "_")
                data_by_pair \
                    .setdefault(pair, {}) \
                    .setdefault(lang, []) \
                    .append({"text": text, "label": 1 if is_positive else 0})

    # ── 4. Split 80/20 et sauvegarde ────────────────────────────────
    split_ratio = 0.8
    random.seed(42)
    for pair, langs in data_by_pair.items():
        for lang, examples in langs.items():
            if not examples:
                continue
            random.shuffle(examples)
            idx = int(len(examples) * split_ratio)
            train_ex, val_ex = examples[:idx], examples[idx:]

            train_dir = os.path.join(output_base, pair, "train", lang)
            val_dir = os.path.join(output_base, pair, "validation", lang)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            with open(os.path.join(train_dir, f"{pair}_train_{lang}.jsonl"), "w", encoding="utf-8") as f:
                for ex in train_ex:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            with open(os.path.join(val_dir, f"{pair}_validation_{lang}.jsonl"), "w", encoding="utf-8") as f:
                for ex in val_ex:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            print(f"[{pair} | {lang}] → {len(train_ex):,} train / {len(val_ex):,} val")

# --------------------------------------------------------------------
# FICHIERS DE MODÈLE REQUIS PAR LANGUE
# --------------------------------------------------------------------
def get_required_files_for_language(language: str) -> list[str]:
    """
    Retourne la liste des fichiers de modèle requis selon la langue.
    Pour le Camembert français, 'sentencepiece.bpe.model' est nécessaire,
    pour le BERT anglais, 'vocab.txt' est nécessaire.

    Parameters
    ----------
    language : str
        Code langue ('FR' ou 'EN'/autres).

    Returns
    -------
    list[str]
        Noms de fichiers requis pour le modèle (config.json, pytorch_model.bin, plus un vocabulaire).
    """
    if language == "FR":
        return ["config.json", "pytorch_model.bin", "sentencepiece.bpe.model"]
    return ["config.json", "pytorch_model.bin", "vocab.txt"]


# --------------------------------------------------------------------
# COMPTER LES FICHIERS DE MODÈLE DANS LE RÉPERTOIRE
# --------------------------------------------------------------------
def get_model_file_count(model_dir: str, language: str) -> int:
    """
    Compte combien des fichiers requis sont présents dans un répertoire de modèle.

    Parameters
    ----------
    model_dir : str
        Chemin vers le répertoire du modèle.
    language : str
        Code langue.

    Returns
    -------
    int
        Nombre de fichiers requis trouvés.
    """
    required = get_required_files_for_language(language)
    if not os.path.exists(model_dir):
        return 0
    return sum(1 for fname in required if os.path.exists(os.path.join(model_dir, fname)))


# --------------------------------------------------------------------
# EXCEPTION PERSONNALISÉE POUR IGNORER L'ENTRAÎNEMENT
# --------------------------------------------------------------------
class SkipTrainingException(Exception):
    """
    Exception indiquant que l'entraînement doit être ignoré en raison
    de l'absence d'étiquettes positives.
    """
    pass


# --------------------------------------------------------------------
# ENTRAÎNEMENT DES MODÈLES
# --------------------------------------------------------------------
def train_models() -> None:
    """
    Parcourt les datasets générés (train/validation), instancie le modèle
    approprié (Camembert pour FR, BERT pour EN), et lance l'entraînement
    avec 20 époques normales via la bibliothèque AugmentedSocialScientistFork.

    Si le meilleur F1 de la classe 1 après l'entraînement normal est
    inférieur à 0.7, l'apprentissage renforcé est déclenché automatiquement.
    La logique de secours sélectionne la 5ᵉ époque renforcée si aucune
    n'améliore le F1 de la classe 1. Sinon, la première à apporter une amélioration est retenue.

    Enfin, un unique CSV data/processed/validation/all_best_models.csv est
    créé/mis à jour avec les métriques du modèle final pour chaque paire d'étiquettes.
    """
    annotation_base = os.path.join(base_path, 'data', 'processed', 'training_LLMs')
    model_output = os.path.join(base_path, 'models')
    log_output = os.path.join(base_path, 'data', 'processed', 'validation', 'LLMs_training')
    os.makedirs(log_output, exist_ok=True)
    os.makedirs(model_output, exist_ok=True)

    all_best_csv = os.path.join(log_output, 'all_best_models.csv')
    # Créer les en-têtes CSV si non existant
    if not os.path.exists(all_best_csv):
        with open(all_best_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_loss",
                "precision_0", "recall_0", "f1_0", "support_0",
                "precision_1", "recall_1", "f1_1", "support_1",
                "macro_f1", "saved_model_path", "training_phase"
            ])

    non_trained = []
    counts = {'fully': 0, 'partial': 0, 'not_started': 0, 'skipped': 0}

    # Pour chaque paire d'étiquettes et sous-dossier de langue
    for pair in os.listdir(annotation_base):
        pair_path = os.path.join(annotation_base, pair)
        train_root = os.path.join(pair_path, 'train')
        if not os.path.isdir(train_root):
            continue

        for lang in os.listdir(train_root):
            # Rechercher les fichiers JSONL d'entraînement/validation
            train_files = glob.glob(os.path.join(pair_path, 'train', lang, f"*train*_{lang}.jsonl"))
            val_files = glob.glob(os.path.join(pair_path, 'validation', lang, f"*validation*_{lang}.jsonl"))
            if not train_files or not val_files:
                continue

            for train_file in train_files:
                model_name = os.path.basename(train_file).replace('_train_', '_').replace('.jsonl', '')
                model_dir = os.path.join(model_output, f"{model_name}.model")
                file_count = get_model_file_count(model_dir, lang)
                needed = len(get_required_files_for_language(lang))

                if file_count == needed:
                    counts['fully'] += 1
                elif file_count == 0:
                    counts['not_started'] += 1
                else:
                    counts['partial'] += 1

                # Identifier le fichier de validation correspondant
                base_val = os.path.basename(train_file).replace('_train_', '_validation_')
                val_file = next((v for v in val_files if base_val in v), None)
                if not val_file:
                    continue

                print(f"[TRAIN] Démarrage de l'entraînement pour {pair} | Lang: {lang} -> {model_name}")
                device = get_device()

                # Sélectionner Camembert (FR) ou BERT (autres/EN)
                if lang == 'FR':
                    print("Instantiation de Camembert pour le français.")
                    model_wrapper = Camembert(model_name="camembert-base", device=device)
                else:
                    print("Instantiation de BERT pour l'anglais.")
                    model_wrapper = Bert(model_name="bert-base-cased", device=device)

                try:
                    # Charger les données d'entraînement/validation
                    train_df = load_jsonl_to_dataframe(train_file)
                    val_df = load_jsonl_to_dataframe(val_file)
                    if train_df.empty or val_df.empty:
                        raise ValueError("Données d'entraînement ou de validation vides")

                    # Vérifier la présence d'étiquettes positives
                    train_counts = train_df['label'].value_counts()
                    val_counts = val_df['label'].value_counts()
                    print("Distribution des labels en entraînement :", train_counts.to_dict())
                    print("Distribution des labels en validation :", val_counts.to_dict())

                    if not (train_df['label'] > 0).any() or not (val_df['label'] > 0).any():
                        counts['skipped'] += 1
                        raise SkipTrainingException("Aucune étiquette positive présente.")

                    # Encoder les données avec la méthode de la bibliothèque
                    train_loader = model_wrapper.encode(
                        train_df.text.values,
                        train_df.label.values.astype(int)
                    )
                    val_loader = model_wrapper.encode(
                        val_df.text.values,
                        val_df.label.values.astype(int)
                    )

                    # Nous stockons les logs dans un sous-dossier spécifique à ce modèle
                    sub_metrics_dir = os.path.join(log_output, model_name)
                    os.makedirs(sub_metrics_dir, exist_ok=True)

                    # Exécuter l'entraînement sur 20 époques normales, puis gérer automatiquement l'apprentissage renforcé si nécessaire
                    result_scores = model_wrapper.run_training(
                        train_dataloader=train_loader,
                        test_dataloader=val_loader,
                        n_epochs=20,
                        lr=5e-5,
                        random_state=42,
                        save_model_as=model_name,
                        pos_weight=None,
                        metrics_output_dir=sub_metrics_dir,
                        best_model_criteria="combined",
                        f1_class_1_weight=0.9,
                        reinforced_learning=True,
                        n_epochs_reinforced=5,
                        rescue_low_class1_f1=True,
                        f1_1_rescue_threshold=0.0
                    )
                    # result_scores = (précision, rappel, f1, support) du meilleur modèle final

                    print(f"Entraînement terminé pour {model_name}. Scores finaux : {result_scores}")

                    # Analyser les métriques du meilleur modèle final depuis les logs CSV de la bibliothèque
                    best_model_csv = os.path.join(sub_metrics_dir, "best_models.csv")
                    final_epoch = final_train_loss = final_val_loss = None
                    final_prec0 = final_rec0 = final_f10 = final_sup0 = None
                    final_prec1 = final_rec1 = final_f11 = final_sup1 = None
                    final_macro_f1 = final_phase = final_path = None

                    if os.path.exists(best_model_csv):
                        with open(best_model_csv, 'r', encoding='utf-8') as f:
                            csv_reader = csv.DictReader(f)
                            rows = list(csv_reader)
                            if rows:
                                last_row = rows[-1]
                                final_epoch = last_row["epoch"]
                                final_train_loss = last_row["train_loss"]
                                final_val_loss = last_row["val_loss"]
                                final_prec0 = last_row["precision_0"]
                                final_rec0 = last_row["recall_0"]
                                final_f10 = last_row["f1_0"]
                                final_sup0 = last_row["support_0"]
                                final_prec1 = last_row["precision_1"]
                                final_rec1 = last_row["recall_1"]
                                final_f11 = last_row["f1_1"]
                                final_sup1 = last_row["support_1"]
                                final_macro_f1 = last_row["macro_f1"]
                                final_phase = last_row["training_phase"]
                                final_path = last_row["saved_model_path"]

                    # Gestion de la logique de secours si aucune époque RL n'a amélioré le F1 classe 1
                    forced_pick = False
                    if final_phase == "reinforced" and final_f11 and float(final_f11) == 0.0:
                        reinforced_csv = os.path.join(sub_metrics_dir, "reinforced_training_metrics.csv")
                        if os.path.exists(reinforced_csv):
                            with open(reinforced_csv, 'r', encoding='utf-8') as f:
                                r_reader = csv.DictReader(f)
                                epochs_rl = list(r_reader)
                                overcame_zero = any(float(r["f1_1"]) > 0.0 for r in epochs_rl)
                                if not overcame_zero:
                                    forced_epoch = min(5, len(epochs_rl))
                                    forced_path = f"./models/{model_name}_reinforced_epoch_{forced_epoch}"
                                    if os.path.exists(forced_path):
                                        final_path_renamed = f"./models/{model_name}"
                                        if os.path.exists(final_path_renamed):
                                            shutil.rmtree(final_path_renamed)
                                        os.rename(forced_path, final_path_renamed)
                                        final_path = final_path_renamed
                                        forced_pick = True
                                        print(f"[LOG FORCÉ] Forçage de l'époque RL {forced_epoch} comme modèle final.")

                    # Si on a forcé l'époque 5, relire ses métriques
                    if forced_pick:
                        reinforced_csv = os.path.join(sub_metrics_dir, "reinforced_training_metrics.csv")
                        if os.path.exists(reinforced_csv):
                            with open(reinforced_csv, 'r', encoding='utf-8') as f:
                                r_reader = csv.DictReader(f)
                                forced_rows = list(r_reader)
                                forced_epoch_value = min(5, len(forced_rows))
                                for row in forced_rows:
                                    if row["epoch"] == str(forced_epoch_value):
                                        forced_epoch_data = row
                                        break
                                if forced_epoch_data:
                                    final_epoch = forced_epoch_data["epoch"]
                                    final_train_loss = forced_epoch_data["train_loss"]
                                    final_val_loss = forced_epoch_data["val_loss"]
                                    final_prec0 = forced_epoch_data["precision_0"]
                                    final_rec0 = forced_epoch_data["recall_0"]
                                    final_f10 = forced_epoch_data["f1_0"]
                                    final_sup0 = forced_epoch_data["support_0"]
                                    final_prec1 = forced_epoch_data["precision_1"]
                                    final_rec1 = forced_epoch_data["recall_1"]
                                    final_f11 = forced_epoch_data["f1_1"]
                                    final_sup1 = forced_epoch_data["support_1"]
                                    final_macro_f1 = forced_epoch_data["macro_f1"]
                                    final_phase = "reinforced_forced_epoch5"

                    # Écrire la ligne finale dans all_best_models.csv
                    with open(all_best_csv, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            final_epoch,
                            final_train_loss,
                            final_val_loss,
                            final_prec0, final_rec0, final_f10, final_sup0,
                            final_prec1, final_rec1, final_f11, final_sup1,
                            final_macro_f1,
                            final_path or "Aucun modèle final",
                            final_phase or "inconnu"
                        ])

                except SkipTrainingException as ste:
                    print(f"[IGNORÉ] {ste}")
                    non_trained.append({'model': model_name, 'reason': str(ste)})
                    if os.path.exists(model_dir):
                        shutil.rmtree(model_dir, ignore_errors=True)
                except Exception as e:
                    print(f"[ERREUR] Échec de l'entraînement pour {model_name} : {e}")
                    non_trained.append({'model': model_name, 'reason': str(e)})
                    if os.path.exists(model_dir):
                        shutil.rmtree(model_dir, ignore_errors=True)
                finally:
                    pass
    # Sauvegarde du résumé des modèles non entraînés
    if non_trained:
        summary_path = os.path.join(annotation_base, 'non_trained_models.csv')
        pd.DataFrame(non_trained).to_csv(summary_path, index=False, encoding='utf-8')
        print(f"Résumé des modèles non entraînés enregistré dans {summary_path}")

    # Affichage du résumé final
    print("===== RÉSUMÉ FINAL =====")
    print(f"Modèles entièrement entraînés : {counts['fully']}")
    print(f"Non démarrés : {counts['not_started']}")
    print(f"Partiellement entraînés : {counts['partial']}")
    print(f"Ignorés (aucune positive) : {counts['skipped']}")
    print("=========================")


# --------------------------------------------------------------------
# POINT D'ENTRÉE PRINCIPAL
# --------------------------------------------------------------------
if __name__ == "__main__":
    """
    Point d'entrée interactif qui :
      1) Liste les fichiers CSV disponibles dans data/processed/subset.
      2) Invite l'utilisateur à en sélectionner un et à préciser la colonne d'annotation.
      3) Génère les datasets train/validation (create_training_datasets).
      4) Lance la logique d'entraînement (train_models).
    """
    subset_dir = os.path.join(base_path, 'data', 'processed', 'subset')
    csv_files = [f for f in os.listdir(subset_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"Aucun fichier CSV trouvé dans {subset_dir}")
        sys.exit(1)

    print(f"Fichiers CSV disponibles dans {subset_dir} :")
    for idx, fname in enumerate(csv_files, 1):
        print(f"{idx} - {fname}")
    choice = input("Sélectionnez le numéro du fichier CSV à utiliser : ")
    try:
        idx = int(choice) - 1
        selected = csv_files[idx]
    except Exception:
        print("Choix invalide.")
        sys.exit(1)
    csv_path = os.path.join(subset_dir, selected)
    print(f"Fichier sélectionné : {csv_path}")

    # Sélection de la colonne d'annotation
    try:
        cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception as e:
        print(f"Erreur lecture colonnes CSV : {e}")
        sys.exit(1)
    print("Colonnes disponibles :")
    for idx, col in enumerate(cols, 1):
        print(f"{idx} - {col}")
    col_choice = input("Sélectionnez le numéro de la colonne d'annotation : ")
    try:
        annotation_col = cols[int(col_choice) - 1]
    except Exception:
        print("Sélection de colonne invalide.")
        sys.exit(1)
    print(f"Colonne d'annotation utilisée : {annotation_col}")

    print("=== Génération des datasets d'entraînement ===")
    create_training_datasets(csv_path, annotation_col)
    print("=== Lancement de l'entraînement des modèles ===")
    train_models()
    print("=== Script terminé ===")
