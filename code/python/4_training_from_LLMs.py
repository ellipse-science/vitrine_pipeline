"""
PROJECT:
--------
POLISCIENCE

FILE:
-----
2_training_from_LLMs.py

DESCRIPTION:
------------
This script trains specialized BERT/Camembert models based on annotations produced by local LLMs,
relying on the 'AugmentedSocialScientistFork' library to handle the core Transformer logic. It:
    1) Interactively selects a CSV file and annotation column.
    2) Creates binary train/validation JSONL datasets per annotation label (excluding 'null').
    3) Trains Camembert for French and BERT for English with 20 normal epochs.
    4) Saves logs for each epoch to data/processed/validation/LLMs_training.
    5) Automatically triggers reinforced learning (extra epochs with class oversampling) if
       the F1 score for class 1 is below 0.6 after normal training.
    6) Implements rescue logic that selects the 5th reinforced epoch if no reinforced epoch
       surpasses 0 in F1 for class 1 (i.e., remains at 0). If an epoch does surpass 0, the
       first such epoch is selected if it yields any improvement in class 1 F1.
    7) Aggregates final metrics in a single CSV (data/processed/validation/all_best_models.csv)
       containing only the final model’s metrics for each trained label.

AUTHOR:
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

# Import the specialized library containing BERT/Camembert wrappers
# that handle training, prediction, and reinforced learning logic.
from AugmentedSocialScientistFork.models import Bert, Camembert

# -----------------------------
# REPLACE ABSOLUTE PATHS WITH RELATIVE PATHS
# -----------------------------
# We use base_path = "." to keep all paths relative and portable.
base_path = "."

# --------------------------------------------------------------------
# DEVICE SELECTION
# --------------------------------------------------------------------
def get_device() -> torch.device:
    """
    Detect available GPU (CUDA or MPS) or default to CPU.
    
    Returns
    -------
    torch.device
        The selected device for training.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU (CUDA).")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using GPU (MPS).")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device


# --------------------------------------------------------------------
# SIMPLE LOGGER
# --------------------------------------------------------------------
class Logger:
    """
    Logger that duplicates stdout to both console and a log file.

    Attributes
    ----------
    terminal : file-like
        Original stdout reference.
    log : file-like
        The file where logs are written.
    """

    def __init__(self, filename: str):
        """
        Initialize the logger.

        Parameters
        ----------
        filename : str
            The path to the file where logs will be written.
        """
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        """
        Write a message to both stdout and the log file.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        """
        Flush the log file. Required for some I/O libraries.
        """
        pass

    def close(self) -> None:
        """
        Close the log file.
        """
        self.log.close()


# --------------------------------------------------------------------
# LOAD JSONL TO DATAFRAME
# --------------------------------------------------------------------
def load_jsonl_to_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load a JSONL file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        The path to the JSONL file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the JSONL data.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# --------------------------------------------------------------------
# CREATE TRAIN/VALIDATION DATASETS
# --------------------------------------------------------------------
def create_training_datasets(csv_path: str, annotation_column: str) -> None:
    """
    Build binary JSONL datasets.
    label = 1  → row contains the (key, label) pair
    label = 0  → row has *any* annotation but not this pair
    No row is added if the annotation cell is empty/null.
    """
    output_base = os.path.join(base_path, "data", "processed", "training_LLMs")

    # ── 1. Lecture du CSV ────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path}")

    # ── 2. Recensement des labels possibles par key ──────────────────
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
            continue  # annotation malformed ⇒ skip
        if not annotations:                # pas d'annotation ⇒ aucun exemple
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
            val_dir   = os.path.join(output_base, pair, "validation", lang)
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
# REQUIRED MODEL FILES PER LANGUAGE
# --------------------------------------------------------------------
def get_required_files_for_language(language: str) -> list[str]:
    """
    Return a list of required model files based on language.
    For French Camembert, 'sentencepiece.bpe.model' is needed,
    for English BERT, 'vocab.txt' is needed.

    Parameters
    ----------
    language : str
        Language code ('FR' or 'EN'/others).

    Returns
    -------
    list[str]
        Filenames required for the model (config.json, pytorch_model.bin, plus a vocab file).
    """
    if language == "FR":
        return ["config.json", "pytorch_model.bin", "sentencepiece.bpe.model"]
    return ["config.json", "pytorch_model.bin", "vocab.txt"]


# --------------------------------------------------------------------
# COUNT MODEL FILES IN DIRECTORY
# --------------------------------------------------------------------
def get_model_file_count(model_dir: str, language: str) -> int:
    """
    Count how many of the required files are present in a model directory.

    Parameters
    ----------
    model_dir : str
        Path to the model directory.
    language : str
        Language code.

    Returns
    -------
    int
        Number of required files found.
    """
    required = get_required_files_for_language(language)
    if not os.path.exists(model_dir):
        return 0
    return sum(1 for fname in required if os.path.exists(os.path.join(model_dir, fname)))


# --------------------------------------------------------------------
# CUSTOM EXCEPTION TO SKIP TRAINING
# --------------------------------------------------------------------
class SkipTrainingException(Exception):
    """
    Exception indicating that training should be skipped due to no positive labels.
    """
    pass


# --------------------------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------------------------
def train_models() -> None:
    """
    Iterate over generated train/validation datasets, instantiate the appropriate model
    (Camembert for FR, BERT for EN), and run training with 20 normal epochs via the
    AugmentedSocialScientistFork library. Logs and model outputs are saved in
    data/processed/validation/LLMs_training.

    If the best F1 for class 1 after normal training is below 0.6, the library's reinforced
    learning is automatically triggered. The rescue logic is also leveraged: if no epoch
    of reinforced training surpasses 0 in class 1 F1 (when the normal training had class 1 F1=0),
    the 5th epoch is forcibly selected. Otherwise, the first epoch that surpasses 0 is selected
    as the new best if it yields an improvement.

    Finally, a single CSV file data/processed/validation/all_best_models.csv is created/updated
    containing only the metrics of each final selected model for each label pair.
    """
    annotation_base = os.path.join(base_path, 'data', 'processed', 'training_LLMs')
    model_output = os.path.join(base_path, 'models')
    log_output = os.path.join(base_path, 'data', 'processed', 'validation', 'LLMs_training')
    os.makedirs(log_output, exist_ok=True)
    os.makedirs(model_output, exist_ok=True)

    all_best_csv = os.path.join(base_path, 'data', 'processed', 'validation', 'LLMs_training', 'all_best_models.csv')
    # Create CSV headers if not existing
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

    # For each label pair and language subfolder
    for pair in os.listdir(annotation_base):
        pair_path = os.path.join(annotation_base, pair)
        train_root = os.path.join(pair_path, 'train')
        if not os.path.isdir(train_root):
            continue

        for lang in os.listdir(train_root):
            # Find train/validation JSONL files
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

                # Find matching validation file
                base_val = os.path.basename(train_file).replace('_train_', '_validation_')
                val_file = next((v for v in val_files if base_val in v), None)
                if not val_file:
                    continue

                print(f"[TRAIN] Starting training for {pair} | Lang: {lang} -> {model_name}")
                device = get_device()

                # Select Camembert (FR) or BERT (others/EN)
                if lang == 'FR':
                    print("Instantiating Camembert for French.")
                    model_wrapper = Camembert(model_name="camembert-base", device=device)
                else:
                    print("Instantiating BERT for English.")
                    model_wrapper = Bert(model_name="bert-base-cased", device=device)

                # Logging to a dedicated file
                log_file = os.path.join(log_output, f"{model_name}_training_log.txt")
                logger = Logger(log_file)
                sys.stdout = logger
                print(f"[LOG] Logging to {log_file}")

                try:
                    # Load train/validation data
                    train_df = load_jsonl_to_dataframe(train_file)
                    val_df = load_jsonl_to_dataframe(val_file)
                    if train_df.empty or val_df.empty:
                        raise ValueError("Empty train or validation data")

                    # Check for presence of positive labels
                    train_counts = train_df['label'].value_counts()
                    val_counts = val_df['label'].value_counts()
                    print("Label distribution in training:", train_counts.to_dict())
                    print("Label distribution in validation:", val_counts.to_dict())

                    if not (train_df['label'] > 0).any() or not (val_df['label'] > 0).any():
                        counts['skipped'] += 1
                        raise SkipTrainingException("No positive labels present.")

                    # Encode data using the library's method
                    train_loader = model_wrapper.encode(
                        train_df.text.values, 
                        train_df.label.values.astype(int)
                    )
                    val_loader = model_wrapper.encode(
                        val_df.text.values, 
                        val_df.label.values.astype(int)
                    )

                    # We store logs in a subfolder specific to this model
                    sub_metrics_dir = os.path.join(log_output, model_name)
                    os.makedirs(sub_metrics_dir, exist_ok=True)

                    # Run training with 20 normal epochs, then automatically handle RL if needed
                    # Use 'run_training' from the library with the relevant parameters
                    result_scores = model_wrapper.run_training(
                        train_dataloader=train_loader,
                        test_dataloader=val_loader,
                        n_epochs=20,               # 20 normal epochs
                        lr=5e-5,
                        random_state=42,
                        save_model_as=model_name,  # final model folder name
                        pos_weight=None,           # if we want weighting from the start, set here
                        metrics_output_dir=sub_metrics_dir,
                        best_model_criteria="combined",
                        f1_class_1_weight=0.9,     # place heavier emphasis on class 1 F1
                        reinforced_learning=True,  # let the library run extra epochs if needed
                        n_epochs_reinforced=5,     # do 5 RL epochs if triggered
                        rescue_low_class1_f1=True, # if best normal F1(class1)=0, any improvement is captured
                        f1_1_rescue_threshold=0.0
                    )
                    # result_scores is (precision, recall, f1, support) from final best model

                    print(f"Training completed for {model_name}. Final scores: {result_scores}")

                    # Parse the final best model's metrics from the library's CSV logs.
                    # The library writes each best model (normal or RL) to best_models.csv
                    # in sub_metrics_dir. The last entry in that file is the final best.
                    best_model_csv = os.path.join(sub_metrics_dir, "best_models.csv")
                    final_epoch = None
                    final_train_loss = None
                    final_val_loss = None
                    final_prec0 = None
                    final_rec0 = None
                    final_f10 = None
                    final_sup0 = None
                    final_prec1 = None
                    final_rec1 = None
                    final_f11 = None
                    final_sup1 = None
                    final_macro_f1 = None
                    final_phase = None
                    final_path = None

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

                    # In some edge cases, the library might not pick an improved RL epoch if
                    # no improvement is found. However, the original requirement:
                    # "If no epoch in RL surpasses 0 in class 1 F1, forcibly pick epoch 5"
                    # We'll handle that post-hoc by checking reinforced logs if needed.
                    # -------------
                    # The library already attempts rescue, but does NOT forcibly pick the last epoch if
                    # there's no improvement above 0. So let's do that check if needed:
                    # -------------
                    forced_pick = False
                    if final_phase == "reinforced" and final_f11 and float(final_f11) == 0.0:
                        # Means we ended up with RL, but class1 F1 is still 0. 
                        # The original script forcibly picks the 5th RL epoch if it never improved.
                        # We'll see if the library actually saved an epoch 5 folder. If not, we do so manually.
                        # Summaries of RL are in reinforced_training_metrics.csv. We can parse it:
                        reinforced_csv = os.path.join(sub_metrics_dir, "reinforced_training_metrics.csv")
                        if os.path.exists(reinforced_csv):
                            with open(reinforced_csv, 'r', encoding='utf-8') as f:
                                r_reader = csv.DictReader(f)
                                epochs_rl = list(r_reader)
                                # Check if any overcame 0
                                overcame_zero = any(float(r["f1_1"]) > 0.0 for r in epochs_rl)
                                if not overcame_zero:
                                    # forcibly pick 5th epoch
                                    # If there are fewer than 5 RL epochs, pick the last
                                    total_rl = len(epochs_rl)
                                    forced_epoch = min(5, total_rl)
                                    # We'll unify this with the library's naming scheme 
                                    forced_path = f"./models/{model_name}_reinforced_epoch_{forced_epoch}"
                                    if not os.path.exists(forced_path):
                                        # Possibly library never saved it (since it wasn't 'best'). 
                                        # We'll note it, but can't do much else unless we re-run training for epoch 5 
                                        print("[FORCE PICK LOGIC] No forced epoch path found on disk. Using normal best.")
                                    else:
                                        # We rename forced_path to final 
                                        final_path_renamed = f"./models/{model_name}"
                                        if os.path.exists(final_path_renamed):
                                            shutil.rmtree(final_path_renamed)
                                        os.rename(forced_path, final_path_renamed)
                                        final_path = final_path_renamed
                                        forced_pick = True
                                        print(f"[FORCE PICK LOGIC] Forcing RL epoch {forced_epoch} as final model.")
                    
                    # If we forcibly picked epoch 5, let's parse its metrics from reinforced_training_metrics.csv
                    if forced_pick:
                        # Re-read that forced epoch's row
                        reinforced_csv = os.path.join(sub_metrics_dir, "reinforced_training_metrics.csv")
                        if os.path.exists(reinforced_csv):
                            with open(reinforced_csv, 'r', encoding='utf-8') as f:
                                r_reader = csv.DictReader(f)
                                forced_rows = [r for r in r_reader]
                                # last line might be 5 if 5 RL epochs exist
                                # but we need the one that specifically is the forced_epoch
                                # for simplicity, we can pick forced_epoch-1 index if the CSV is in order:
                                # but let's do a direct filter
                                forced_epoch_data = None
                                total_rl = len(forced_rows)
                                forced_epoch_value = min(5, total_rl)
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

                    # Write final row to all_best_models.csv
                    with open(all_best_csv, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            final_epoch,
                            final_train_loss,
                            final_val_loss,
                            final_prec0, final_rec0, final_f10, final_sup0,
                            final_prec1, final_rec1, final_f11, final_sup1,
                            final_macro_f1,
                            final_path if final_path else "No final model",
                            final_phase if final_phase else "unknown"
                        ])

                except SkipTrainingException as ste:
                    print(f"[SKIP] {ste}")
                    non_trained.append({'model': model_name, 'reason': str(ste)})
                    # If partial artifacts exist, remove them
                    if os.path.exists(model_dir):
                        shutil.rmtree(model_dir, ignore_errors=True)
                except Exception as e:
                    print(f"[ERROR] Training failed for {model_name}: {e}")
                    non_trained.append({'model': model_name, 'reason': str(e)})
                    if os.path.exists(model_dir):
                        shutil.rmtree(model_dir, ignore_errors=True)
                finally:
                    # Restore stdout and close logger
                    sys.stdout = sys.__stdout__
                    logger.close()

    # Save summary of non-trained models
    if non_trained:
        summary_path = os.path.join(annotation_base, 'non_trained_models.csv')
        pd.DataFrame(non_trained).to_csv(summary_path, index=False, encoding='utf-8')
        print(f"Non-trained models summary saved to {summary_path}")

    # Print final summary
    print("===== FINAL SUMMARY =====")
    print(f"Fully trained models: {counts['fully']}")
    print(f"Not started: {counts['not_started']}")
    print(f"Partially trained: {counts['partial']}")
    print(f"Skipped (no positives): {counts['skipped']}")
    print("=========================")


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    """
    Interactive entry point that:
      1) Lists available CSV files in data/processed/subset.
      2) Prompts user to select one and specify the annotation column.
      3) Generates train/validation datasets (create_training_datasets).
      4) Launches the training logic (train_models).
    """
    subset_dir = os.path.join(base_path, 'data', 'processed', 'subset')
    csv_files = [f for f in os.listdir(subset_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {subset_dir}")
        sys.exit(1)

    print(f"Available CSV files in {subset_dir}:")
    for idx, fname in enumerate(csv_files, 1):
        print(f"{idx} - {fname}")
    choice = input("Select the number of the CSV file to use: ")
    try:
        idx = int(choice) - 1
        selected = csv_files[idx]
    except Exception:
        print("Invalid choice.")
        sys.exit(1)
    csv_path = os.path.join(subset_dir, selected)
    print(f"Selected file: {csv_path}")

    # Select annotation column
    try:
        cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception as e:
        print(f"Error reading CSV columns: {e}")
        sys.exit(1)
    print("Available columns:")
    for idx, col in enumerate(cols, 1):
        print(f"{idx} - {col}")
    col_choice = input("Select the annotation column number: ")
    try:
        annotation_col = cols[int(col_choice) - 1]
    except Exception:
        print("Invalid column selection.")
        sys.exit(1)
    print(f"Using annotation column: {annotation_col}")

    print("=== Generating training datasets ===")
    create_training_datasets(csv_path, annotation_col)
    print("=== Starting model training ===")
    train_models()
    print("=== Script completed ===")
