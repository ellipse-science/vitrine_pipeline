"""
PROJECT:
--------
4_training_from_LLMs.py

TITLE:
------
Entraînement de modèles spécialisés BERT à partir d’annotations de LLMs locaux

OBJECTIF PRINCIPAL:
-------------------
1. Demander à l'utilisateur de choisir, parmi les fichiers CSV disponibles dans 
   "data/processed/subset", celui à utiliser pour l'entraînement.
2. Une fois le fichier sélectionné, afficher les colonnes du CSV et demander à 
   l'utilisateur de choisir celle contenant les annotations.
3. Lire les annotations depuis le fichier sélectionné et, pour chaque clé présente 
   dans la colonne d'annotations, détecter tous les labels (pour les valeurs de type 
   liste ou valeur unique) et créer de façon générique des bases dichotomiques.
   - Pour chaque couple (clé, label), chaque ligne produira une entrée JSONL de la forme :
     {"text": "texte de l'article", "label": 1}   (label = 1 si le label est présent, 0 sinon)
   - Les fichiers générés seront rangés dans une arborescence par langue et par couple, par exemple :
     data/processed/training_LLMs/themes_international_affairs/train/EN/themes_international_affairs_train_EN.jsonl  
     data/processed/training_LLMs/themes_international_affairs/validation/FR/themes_international_affairs_validation_FR.jsonl
4. Procéder ensuite à l’entraînement en utilisant Camembert pour le français et Bert pour l’anglais,
   avec des logs enregistrés dans "data/processed/validation/LLMs_training" et les modèles sauvegardés dans "models".

Dépendances:
-------------
- os, sys, json, glob, shutil, random  
- pandas, torch  
- AugmentedSocialScientist.models (pour Camembert et Bert)

Auteur:
-------
Antoine Lemor
"""

import os
import sys
import json
import glob
import shutil
import random
import pandas as pd
import torch

from AugmentedSocialScientistFork.models import Camembert, Bert

# --------------------------------------------------------------------
# FONCTION POUR DÉTECTER LE DEVICE (CUDA, MPS OU CPU)
# --------------------------------------------------------------------
def get_device():
    """
    Détecte une GPU (CUDA ou MPS) ou retourne le CPU.

    Returns:
        torch.device: le device à utiliser.
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
# LOGGER SIMPLE POUR ENREGISTRER LES SORTIES
# --------------------------------------------------------------------
class Logger(object):
    """
    Logger redirigeant la sortie standard (stdout) vers un fichier et la console.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def close(self):
        self.log.close()

# --------------------------------------------------------------------
# FONCTION POUR CHARGER UN FICHIER JSONL EN DATAFRAME
# --------------------------------------------------------------------
def load_jsonl_to_dataframe(filepath):
    """
    Charge un fichier JSONL dans une DataFrame pandas.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# --------------------------------------------------------------------
# FONCTION DE CRÉATION DES BASES D'ENTRAÎNEMENT ET VALIDATION
# --------------------------------------------------------------------
def create_training_datasets(csv_path, annotation_column):
    """
    1. Lit le CSV d'annotations sélectionné par l'utilisateur.
    2. Analyse de façon générique la colonne choisie (contenant les annotations) pour détecter
       l'ensemble des clés et de leurs labels présents.
    3. Pour chaque couple (clé, label) et par langue (selon la colonne 'lang'),
       crée une base dichotomique où la présence du label vaut 1, l'absence 0.
    4. Réalise un split train/validation (80/20 par défaut) et écrit les fichiers JSONL dans l'arborescence :
       data/processed/training_LLMs/{clé}_{label}/train/{LANG}/{clé}_{label}_train_{LANG}.jsonl
       data/processed/training_LLMs/{clé}_{label}/validation/{LANG}/{clé}_{label}_validation_{LANG}.jsonl
    """
    # Définition des chemins (en tenant compte que ce script est dans code/python)
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_base = os.path.join(base_path, "..", "..", "data", "processed", "training_LLMs")

    # Lecture du CSV sélectionné
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV chargé depuis {csv_path}")
    except Exception as e:
        print(f"Erreur lors de la lecture du CSV: {e}")
        sys.exit(1)

    # Première passe : collecte des labels candidats pour chaque clé
    candidate_labels = {}  # dict: clé -> set(labels)
    for idx, row in df.iterrows():
        try:
            annotations = json.loads(row.get(annotation_column, "{}"))
        except Exception as e:
            annotations = {}
        for key, value in annotations.items():
            if value is None:
                continue
            # Si la valeur est une liste, ajouter chaque élément
            if isinstance(value, list):
                for lab in value:
                    if lab is not None:
                        candidate_labels.setdefault(key, set()).add(lab)
            else:
                candidate_labels.setdefault(key, set()).add(value)

    # Préparation des données par couple (clé, label) et par langue
    data_by_pair = {}  # dict: pair_name -> dict(lang -> list of examples)
    for idx, row in df.iterrows():
        lang = str(row.get("lang", "")).strip().upper()
        text = row.get("text", "").strip()
        try:
            annotations = json.loads(row.get(annotation_column, "{}"))
        except Exception:
            annotations = {}
        # Pour chaque clé détectée globalement
        for key, labs in candidate_labels.items():
            row_value = annotations.get(key, None)
            for lab in labs:
                pair_name = f"{key}_{lab}"
                pair_name = pair_name.lower().replace(" ", "_")
                if row_value is None:
                    binary_label = 0
                else:
                    if isinstance(row_value, list):
                        binary_label = 1 if lab in row_value else 0
                    else:
                        binary_label = 1 if row_value == lab else 0
                if pair_name not in data_by_pair:
                    data_by_pair[pair_name] = {}
                if lang not in data_by_pair[pair_name]:
                    data_by_pair[pair_name][lang] = []
                data_by_pair[pair_name][lang].append({"text": text, "label": binary_label})

    # Split train/validation et écriture des fichiers JSONL
    split_ratio = 0.8  # 80% training, 20% validation
    random.seed(42)  # pour reproductibilité

    for pair_name, lang_dict in data_by_pair.items():
        for lang, examples in lang_dict.items():
            if not examples:
                continue
            random.shuffle(examples)
            split_index = int(len(examples) * split_ratio) if len(examples) > 1 else len(examples)
            train_examples = examples[:split_index]
            val_examples = examples[split_index:]
            train_dir = os.path.join(output_base, pair_name, "train", lang)
            val_dir = os.path.join(output_base, pair_name, "validation", lang)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            train_file = os.path.join(train_dir, f"{pair_name}_train_{lang}.jsonl")
            val_file = os.path.join(val_dir, f"{pair_name}_validation_{lang}.jsonl")
            with open(train_file, "w", encoding="utf-8") as f_train:
                for ex in train_examples:
                    f_train.write(json.dumps(ex, ensure_ascii=False) + "\n")
            with open(val_file, "w", encoding="utf-8") as f_val:
                for ex in val_examples:
                    f_val.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"Création de {train_file} ({len(train_examples)} exemples) et {val_file} ({len(val_examples)} exemples).")

# --------------------------------------------------------------------
# FONCTION POUR DÉTERMINER LES FICHIERS REQUIS EN FONCTION DE LA LANGUE
# --------------------------------------------------------------------
def get_required_files_for_language(language: str):
    """
    Retourne la liste des fichiers nécessaires pour un modèle selon la langue.
    """
    if language == "FR":
        return ["config.json", "pytorch_model.bin", "sentencepiece.bpe.model"]
    else:
        return ["config.json", "pytorch_model.bin", "vocab.txt"]

# --------------------------------------------------------------------
# FONCTION POUR COMPTER LES FICHIERS MODÈLES DANS UN RÉPERTOIRE
# --------------------------------------------------------------------
def get_model_file_count(model_dir, language):
    """
    Compte le nombre de fichiers requis présents dans 'model_dir'.
    """
    required_files = get_required_files_for_language(language)
    present_count = 0
    if not os.path.exists(model_dir):
        return 0
    for file_name in required_files:
        if os.path.exists(os.path.join(model_dir, file_name)):
            present_count += 1
    return present_count

# --------------------------------------------------------------------
# EXCEPTION PERSONNALISÉE POUR PASSER L'ENTRAÎNEMENT
# --------------------------------------------------------------------
class SkipTrainingException(Exception):
    """
    Exception personnalisée indiquant que l’entraînement doit être sauté.
    """
    pass

# --------------------------------------------------------------------
# FONCTION PRINCIPALE D'ENTRAÎNEMENT DES MODÈLES
# --------------------------------------------------------------------
def train_models():
    """
    Parcourt les bases d'entraînement générées, détecte pour chaque couple (clé, label) et langue 
    les fichiers train et validation, puis lance la procédure d'entraînement (en instanciant Camembert ou Bert).
    
    Les logs sont enregistrés dans "data/processed/validation/LLMs_training/".
    Les modèles sont sauvegardés dans le dossier "models" à la racine.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    annotation_base_dir = os.path.join(base_path, "..", "..", "data", "processed", "training_LLMs")
    model_output_dir = os.path.join(base_path, "..", "..", "models")
    log_output_dir = os.path.join(base_path, "..", "..", "data", "processed", "validation", "LLMs_training")
    training_data_dir = os.path.join(base_path, "..", "..", "data", "processed", "training_LLMs")
    
    os.makedirs(log_output_dir, exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)
    
    fully_trained_count = 0
    partial_count = 0
    not_started_count = 0
    skipped_count = 0
    non_trained_models = []

    pair_dirs = [d for d in os.listdir(annotation_base_dir) if os.path.isdir(os.path.join(annotation_base_dir, d))]
    
    for pair in pair_dirs:
        pair_path = os.path.join(annotation_base_dir, pair)
        lang_dirs = []
        train_path = os.path.join(pair_path, "train")
        if os.path.exists(train_path):
            lang_dirs = os.listdir(train_path)
        for lang in lang_dirs:
            train_filepath_pattern = os.path.join(pair_path, "train", lang, f"*train*_{lang}.jsonl")
            val_filepath_pattern = os.path.join(pair_path, "validation", lang, f"*validation*_{lang}.jsonl")
            train_files = glob.glob(train_filepath_pattern)
            val_files = glob.glob(val_filepath_pattern)
            if not train_files or not val_files:
                continue

            for train_file in train_files:
                model_name = os.path.basename(train_file).replace('_train_', '_').replace(".jsonl", "")
                model_dir = os.path.join(model_output_dir, f"{model_name}.model")
                file_count = get_model_file_count(model_dir, lang)
                required_count = len(get_required_files_for_language(lang))
                if file_count == required_count:
                    status = "fully_trained"
                    fully_trained_count += 1
                elif file_count == 0:
                    status = "not_started"
                    not_started_count += 1
                else:
                    status = "partial"
                    partial_count += 1

                base_name = os.path.basename(train_file).replace('_train_', '_validation_')
                matching_val_file = None
                for val_file in val_files:
                    if base_name in val_file:
                        matching_val_file = val_file
                        break
                if not matching_val_file:
                    continue

                print(f"[TRAIN] Début de l'entraînement pour {pair} | Langue: {lang} -> {model_name}")

                device = get_device()
                if lang == "FR":
                    print("Instanciation du modèle Camembert pour le français.")
                    model = Camembert(device=device)
                else:
                    print("Instanciation du modèle Bert pour l'anglais.")
                    model = Bert(device=device)
                try:
                    model.to(device)
                except AttributeError:
                    print("Attention : La méthode .to(device) n'est pas supportée par ce modèle.")

                log_filepath = os.path.join(log_output_dir, f"{model_name}_training_log.txt")
                print(f"Configuration des logs dans : {log_filepath}")
                logger = Logger(log_filepath)
                sys.stdout = logger
                print(f"[LOG] Début de l'enregistrement des logs pour {model_name} en {lang}")

                scores = None
                train_label_counts = None
                val_label_counts = None

                try:
                    train_data = load_jsonl_to_dataframe(train_file)
                    val_data = load_jsonl_to_dataframe(matching_val_file)
                    print(f"Données chargées pour {pair} en {lang}")
                    if train_data.empty or val_data.empty:
                        print(f"Données vides pour {pair} en {lang}")
                        raise ValueError("Données d'entraînement ou de validation vides")
                    
                    train_label_counts = train_data['label'].value_counts()
                    val_label_counts = val_data['label'].value_counts()
                    print(f"Distribution des labels en entraînement pour {pair} en {lang} :")
                    print(train_label_counts)
                    print(f"Distribution des labels en validation pour {pair} en {lang} :")
                    print(val_label_counts)

                    if not (train_data['label'] > 0).any() or not (val_data['label'] > 0).any():
                        print("[SKIP] Entraînement sauté car aucun label positif n'est présent.")
                        skipped_count += 1
                        raise SkipTrainingException("Aucun label positif dans train ou validation.")
                    
                    min_annotations = 1
                    if len(train_data) < min_annotations or len(val_data) < min_annotations:
                        print(f"Annotations insuffisantes pour {pair} en {lang}.")
                        raise ValueError("Annotations insuffisantes")
                    
                    train_loader = model.encode(
                        train_data.text.values,
                        train_data.label.values.astype(int)
                    )
                    val_loader = model.encode(
                        val_data.text.values,
                        val_data.label.values.astype(int)
                    )
                    print("Encodage des données terminé.")

                    relative_model_output_path = f"{model_name}.model"
                    print(f"Sauvegarde du modèle dans : {relative_model_output_path}")

                    scores = model.run_training(
                        train_loader,
                        val_loader,
                        lr=5e-5,
                        n_epochs=20,
                        random_state=42,
                        save_model_as=relative_model_output_path
                    )
                    print(f"Entraînement terminé pour {model_name}")
                except SkipTrainingException as ste:
                    print(f"[INFO] {ste}")
                    non_trained_models.append({
                        "model_name": model_name,
                        "train_distribution": train_label_counts.to_dict() if train_label_counts is not None else {},
                        "val_distribution": val_label_counts.to_dict() if val_label_counts is not None else {}
                    })
                    if os.path.exists(model_dir):
                        shutil.rmtree(model_dir, ignore_errors=True)
                        print(f"Répertoire supprimé pour le modèle non entraîné: {model_dir}")
                except Exception as e:
                    print(f"Erreur lors de l'entraînement pour {model_name}: {e}")
                    non_trained_models.append({
                        "model_name": model_name,
                        "train_distribution": train_label_counts.to_dict() if train_label_counts is not None else {},
                        "val_distribution": val_label_counts.to_dict() if val_label_counts is not None else {}
                    })
                    if os.path.exists(model_dir):
                        shutil.rmtree(model_dir, ignore_errors=True)
                        print(f"Répertoire supprimé pour le modèle non entraîné: {model_dir}")
                finally:
                    sys.stdout = sys.__stdout__
                    logger.close()
                    print(f"[LOG] Fin des logs pour {model_name} en {lang}")
                
                if scores is not None:
                    print(f"[TRAIN] Entraînement complété pour {model_name}, scores: {scores}")
                else:
                    if train_label_counts is not None and val_label_counts is not None:
                        print(f"[TRAIN] Entraînement sauté ou échoué pour {model_name} (voir logs).")
                    else:
                        print(f"[TRAIN] Entraînement non démarré pour {model_name} (aucune donnée ou exception).")
    
    if non_trained_models:
        non_trained_csv_path = os.path.join(training_data_dir, "non_trained_models.csv")
        df_non_trained = pd.DataFrame(non_trained_models)
        df_non_trained.to_csv(non_trained_csv_path, index=False, encoding='utf-8')
        print(f"[INFO] non_trained_models.csv créé à : {non_trained_csv_path}")

    print("===== RÉSUMÉ FINAL =====")
    print(f"Modèles entièrement entraînés : {fully_trained_count}")
    print(f"Modèles non démarrés : {not_started_count}")
    print(f"Modèles partiellement entraînés : {partial_count}")
    print(f"Modèles sautés (aucun label positif) : {skipped_count}")
    print("================================")

# --------------------------------------------------------------------
# POINT D'ENTRÉE PRINCIPALE
# --------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Sélection interactive du fichier CSV dans data/processed/subset
    base_path = os.path.dirname(os.path.abspath(__file__))
    subset_dir = os.path.join(base_path, "..", "..", "data", "processed", "subset")
    csv_files = [f for f in os.listdir(subset_dir) if f.endswith(".csv")]
    if not csv_files:
        print("Aucun fichier CSV trouvé dans", subset_dir)
        sys.exit(1)
    print("Fichiers CSV disponibles dans", subset_dir, ":")
    for idx, csvf in enumerate(csv_files):
        print(f"{idx + 1} - {csvf}")
    choice = input("Veuillez choisir le numéro du fichier CSV à utiliser: ")
    try:
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(csv_files):
            print("Choix invalide.")
            sys.exit(1)
    except Exception as e:
        print("Erreur lors de la sélection.")
        sys.exit(1)
    selected_csv_file = os.path.join(subset_dir, csv_files[choice_idx])
    print("Fichier sélectionné :", selected_csv_file)

    # 2. Sélection interactive de la colonne contenant les annotations
    try:
        df_temp = pd.read_csv(selected_csv_file, nrows=0)
        columns = list(df_temp.columns)
    except Exception as e:
        print("Erreur lors de la lecture du fichier CSV:", e)
        sys.exit(1)
    print("Colonnes disponibles dans le fichier CSV:")
    for idx, col in enumerate(columns):
        print(f"{idx + 1} - {col}")
    col_choice = input("Veuillez choisir le numéro de la colonne à utiliser pour les annotations: ")
    try:
        col_choice_idx = int(col_choice) - 1
        if col_choice_idx < 0 or col_choice_idx >= len(columns):
            print("Choix invalide.")
            sys.exit(1)
    except Exception as e:
        print("Erreur lors de la sélection de colonne.")
        sys.exit(1)
    annotation_column = columns[col_choice_idx]
    print("Colonne sélectionnée pour les annotations :", annotation_column)

    print("=== Création des bases d'entraînement ===")
    create_training_datasets(selected_csv_file, annotation_column)
    print("=== Lancement de l'entraînement des modèles ===")
    train_models()
    print("=== Script terminé ===")
