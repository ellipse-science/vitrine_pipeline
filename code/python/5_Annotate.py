"""
PROJET :
-------
vitrine_pipeline

FICHIER
----
5_Annotate.py

DESCRIPTION :
------------

Le script permet d'annoter un fichier CSV ou une base de données SQL en utilisant des modèles de classification
pré-entraînés. Il prend en charge l'annotation par lots et la parallélisation sur CPU et GPU. 

1) Le script peut gérer soit un fichier CSV, soit une base de données SQL pour l'annotation.
2) L'utilisateur est invité à choisir le format de base (.csv ou sql).
3) Pour CSV :
    - L'utilisateur fournit le chemin absolu vers le fichier CSV.
    - Le code affiche toutes les colonnes et demande quelle colonne contient le texte à annoter
      et quelle colonne contient le code de langue.
    - Tous les modèles disponibles (se terminant par _EN ou _FR) dans `models/` sont listés, et
      l'utilisateur sélectionne ceux à appliquer. Chaque modèle sélectionné est utilisé pour annoter
      uniquement les lignes qui correspondent à sa langue (EN ou FR).
    - Pour chaque modèle choisi, une nouvelle colonne (nommée d'après la "catégorie" du modèle) est
      créée (si manquante). Les lignes qui manquent de valeurs dans cette colonne sont traitées
      par lots (CPU/GPU parallèles), et les prédictions sont stockées dans le DataFrame.
    - Une fois la colonne du modèle entièrement traitée, le CSV est sauvegardé pour éviter de perdre
      la progression.

4) Pour SQL :
    - L'utilisateur spécifie les paramètres de connexion à la base de données de manière interactive.
    - Le script se connecte à la BDD, liste les tables disponibles (l'utilisateur en choisit une),
      puis liste les colonnes pour que l'utilisateur puisse choisir la colonne de texte et lacsv colonne de langue.
    - Tous les modèles disponibles sont listés, et l'utilisateur sélectionne ceux à appliquer.
    - Le script traite chaque modèle choisi en parallèle, en découpant les lignes en lots. Il annote uniquement
      les lignes dont la langue correspond au modèle (EN ou FR).
    - Après avoir terminé tous les lots pour la colonne d'un modèle, une seule mise à jour en masse est
      effectuée sur cette table pour éviter les commits lents ligne par ligne.
    - Le script préserve la logique de concurrence pour l'utilisation GPU+CPU.

5) Parallélisation et concurrence :
    - Le code conserve la logique multi-périphériques de la version précédente : l'utilisateur peut
      choisir d'exécuter sur CPU, GPU, ou les deux.
    - Une file d'attente de périphériques garantit qu'au plus un worker GPU est assigné (si le GPU est sélectionné),
      avec des workers supplémentaires sur CPU si nécessaire.
    - Nous utilisons la prédiction par lots et les barres de progression TQDM pour chaque lot.

6) Gestion des erreurs et repli :
    - Si un texte déclenche une exception, sa prédiction devient None sans bloquer
      les autres lignes.
    - Si une exception au niveau du lot se produit, le code se replie sur l'annotation par texte.

7) Entrées utilisateur supplémentaires :
    - Combien de lignes annoter (un entier ou "all").
    - S'il faut échantillonner aléatoirement les lignes ou les prendre depuis le début.

Le script final utilise les mêmes méthodes que le code précédent, refactorisé pour la
flexibilité CSV/SQL et préservant les fonctionnalités établies et les barres de progression.

AUTEUR
------
Antoine Lemor
"""

from __future__ import annotations

import os
import sys
import random
import queue
import time
import psycopg2
import psycopg2.extras

import numpy as np
import torch
import pandas as pd

from collections import defaultdict
from multiprocessing import (
     Pool,
     cpu_count,
     Manager,
     RLock,
     current_process,
)
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm
from psycopg2 import sql
from psycopg2.extras import execute_values


# --------------------------------------------------------------------------------------
#                                 Constantes globales (tailles des chunks)
# --------------------------------------------------------------------------------------

CHUNK_SIZE: int = 1000  # nombre de lignes par lot lors de la parallélisation

# --------------------------------------------------------------------------------------
#                              UTILITAIRES POUR LA DÉCOUVERTE DES MODÈLES
# --------------------------------------------------------------------------------------

def parse_model_filename(dirname: str) -> Tuple[str, str]:
     """
     Extrait (catégorie, langue) du nom d'un répertoire de modèle.
     Le nom du répertoire doit se terminer par "_EN" ou "_FR".

     Parameters
     ----------
     dirname : str
          Le nom du répertoire qui se termine par "_EN" ou "_FR".

     Returns
     -------
     Tuple[str, str]
          (catégorie, langue) où langue est "EN" ou "FR".

     Raises
     ------
     ValueError
          Si `dirname` ne se termine pas par "_EN" ou "_FR".
     """
     base = dirname.rstrip("/")
     if base.endswith("_EN"):
          return base[:-3], "EN"
     elif base.endswith("_FR"):
          return base[:-3], "FR"
     else:
          raise ValueError(
                f"[DÉCOUVERTE-MODÈLE] «{dirname}» ne se termine pas par _EN ou _FR ; langue inconnue."
          )


def load_models(models_dir: str) -> Dict[str, Dict[str, str]]:
     """
     Parcourt `models_dir` et rassemble les sous-dossiers se terminant par "_EN"/"_FR".
     Construit un dictionnaire : {catégorie: {langue: chemin_vers_dossier_modèle}}.

     Parameters
     ----------
     models_dir : str
          Chemin vers le répertoire de niveau supérieur contenant les sous-dossiers de modèles.

     Returns
     -------
     Dict[str, Dict[str, str]]
          Une correspondance imbriquée de catégorie → langue → chemin du dossier.
     """
     if not os.path.isdir(models_dir):
          sys.exit(f"[FATAL] Répertoire des modèles non trouvé : «{models_dir}»")

     model_map: Dict[str, Dict[str, str]] = {}
     for entry in os.listdir(models_dir):
          path = os.path.join(models_dir, entry)
          if not os.path.isdir(path):
                continue
          try:
                cat, lang = parse_model_filename(entry)
          except ValueError:
                continue
          model_map.setdefault(cat, {})[lang] = path

     if not model_map:
          sys.exit(f"[FATAL] Aucun dossier *_EN ou *_FR valide dans «{models_dir}».")

     return model_map


# --------------------------------------------------------------------------------------
#                                 DEVICE SELECTION
# --------------------------------------------------------------------------------------

def pick_devices_interactively() -> List[str]:
     """
     Demande à l'utilisateur s'il souhaite utiliser uniquement le CPU, uniquement le GPU, ou les deux si le GPU est disponible.
     Distribue ces périphériques entre les processus workers.

     Returns
     -------
     List[str]
          Une liste de chaînes de périphériques, par ex. ["cpu", "cpu", ...] ou ["cuda", "cpu", ...]
     """
     # Nombre de processus à générer
     max_workers = cpu_count()

     # Vérifier si le GPU est disponible
     gpu_available = torch.cuda.is_available()
     # Vérifier également MPS pour Apple Silicon si nécessaire
     mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

     # Si un utilisateur a MPS mais pas CUDA, nous traitons le choix "GPU" comme MPS.
     if not gpu_available and mps_available:
          print("[INFO] CUDA non trouvé mais MPS est disponible (Apple Silicon).")
          gpu_available = True  # Nous traiterons MPS comme 'gpu' pour l'invite utilisateur

     print("\nChoisissez le(s) périphérique(s) à utiliser pour le traitement parallèle :")
     print("  1) CPU uniquement")
     print("  2) GPU (CUDA/MPS) uniquement (si disponible)")
     print("  3) CPU et GPU (si GPU disponible)")
     choice = input("Choix du périphérique [1/2/3] ? ").strip()

     if choice not in {"1", "2", "3"}:
          print("[WARN] Choix invalide, utilisation du CPU uniquement par défaut.")
          choice = "1"

     #choix 1 : tous les CPU
     if choice == "1" or not gpu_available:
          return ["cpu"] * max_workers 

     elif choice == "2":
          # Tous les GPU ou 1 worker GPU (optionnel). Habituellement, 1 worker GPU suffit.
          # Le reste peut être CPU si nous voulions la concurrence, mais interprétons "GPU uniquement" strictement :
          return ["cuda"] * min(max_workers, 1) if torch.cuda.is_available() else ["mps"] * min(max_workers, 1)

     else:  # choix == "3"
          # CPU et GPU
          devices = []
          gpu_name = "cuda" if torch.cuda.is_available() else "mps"
          devices.append(gpu_name)              # 1 worker GPU
          devices.extend(["cpu"] * (max_workers - 1))   # TOUS les cœurs CPU
          return devices


# --------------------------------------------------------------------------------------
#                     CLASSES DE MODÈLES DE REMPLACEMENT (si non installées)
# --------------------------------------------------------------------------------------
try:
     from AugmentedSocialScientistFork.models import Camembert, Bert
except ImportError:
     class Camembert:
          """
          Modèle Camembert de remplacement pour démonstration uniquement.
          """
          def __init__(self, device: torch.device):
                pass

          def encode(self, texts, *args, **kwargs):
                return None

          def predict_with_model(self, loader, model_path, proba=False, progress_bar=False):
                n_samples = len(loader) if loader else 0
                # Retourne des probabilités aléatoires pour la démonstration
                return np.random.rand(n_samples, 2)

     class Bert:
          """
          Modèle Bert de remplacement pour démonstration uniquement.
          """
          def __init__(self, device: torch.device):
                pass

          def encode(self, texts, *args, **kwargs):
                return None

          def predict_with_model(self, loader, model_path, proba=False, progress_bar=False):
                n_samples = len(loader) if loader else 0
                return np.random.rand(n_samples, 2)


# --------------------------------------------------------------------------------------
#                          PRÉDICTION PAR LOT 
# --------------------------------------------------------------------------------------

def predict_batch_safely(
     model: Bert | Camembert,
     texts: List[str],
     model_path: str,
) -> List[Optional[int]]:
     """
     Prédit une liste de textes, retournant les indices de classe numériques ou None en cas d'erreur.
     Si le lot entier échoue, une solution de repli texte par texte est utilisée.

     Parameters
     ----------
     model : Bert or Camembert
          Une instance du modèle de classification.
     texts : List[str]
          Échantillons de texte à prédire.
     model_path : str
          Chemin vers les poids sauvegardés ou le répertoire.

     Returns
     -------
     List[Optional[int]]
          Prédictions pour chaque texte, ou None si ce texte a généré une erreur.
     """
     try:
          loader = model.encode(texts, None, len(texts), progress_bar=False)
          probs = model.predict_with_model(loader, model_path, proba=True, progress_bar=False)
          return np.argmax(probs, axis=1).tolist()
     except Exception:
          # Repli : prédire individuellement en cas d'échec du lot
          results: List[Optional[int]] = []
          for t in texts:
                try:
                     loader_1 = model.encode([t], None, 1, progress_bar=False)
                     probs_1 = model.predict_with_model(loader_1, model_path, proba=True, progress_bar=False)
                     pred = int(np.argmax(probs_1, axis=1)[0])
                     results.append(pred)
                except Exception:
                     results.append(None)
          return results


# --------------------------------------------------------------------------------------
#                         VARIABLES GLOBALES DES WORKERS & INITIALISATION
# --------------------------------------------------------------------------------------
WORKER_DEVICE: str = "cpu"
WORKER_BS: int = 25
WORKER_POS: int | None = None

def _worker_init(
     devices_q: "queue.Queue[str]",
     cpu_bs: int,
     gpu_bs: int,
     pos_q: "queue.Queue[int]",
) -> None:
     """
     Initialise chaque worker avec :
     1) Attribution de périphérique (CPU/GPU) depuis devices_q.
     2) Taille de lot basée sur CPU ou GPU.
     3) Position de la ligne de la barre de progression TQDM depuis pos_q.

     Parameters
     ----------
     devices_q : queue.Queue
          File d'attente des noms de périphériques ('cpu', 'cuda', ou 'mps').
     cpu_bs : int
          Taille de lot pour les workers CPU.
     gpu_bs : int
          Taille de lot pour les workers GPU.
     pos_q : queue.Queue
          File d'attente des positions entières pour la sortie multiligne TQDM.
     """
     global WORKER_DEVICE, WORKER_BS, WORKER_POS

     try:
          WORKER_DEVICE = devices_q.get_nowait()
     except queue.Empty:
          WORKER_DEVICE = "cpu"

     WORKER_BS = gpu_bs if WORKER_DEVICE in ("cuda", "mps") else cpu_bs
     WORKER_POS = pos_q.get_nowait()

     # Réduire le bruit de journalisation si transformers est installé :
     try:
          import transformers
          transformers.logging.set_verbosity_error()
     except ImportError:
          pass


# --------------------------------------------------------------------------------------
#                       FONCTION WORKER POUR LE TRAITEMENT DES LOTS
# --------------------------------------------------------------------------------------

def _worker_chunk(task: Dict[str, Any]) -> Dict[str, Any]:
     """
     Une fonction worker qui traite un lot de lignes (SQL ou CSV).
     Elle charge le modèle approprié (Bert ou Camembert) et prédit par lots.

     Pour le mode SQL :
        - 'rows' contient des tuples de (doc_id, sentence_id, text).
        - Les mises à jour retournées sont [(pred, doc_id, sentence_id), ...].

     Pour le mode CSV :
        - 'rows' contient des tuples de (row_index, text).
        - Les mises à jour retournées sont [(pred, row_index), ...].

     Parameters
     ----------
     task : Dict[str, Any]
          Un dictionnaire contenant :
          - "mode": "sql" ou "csv"
          - "table": ou "csv_path" (inutilisé dans le lot lui-même, mais retourné)
          - "column": le nom de la colonne pour stocker les prédictions
          - "model_path": chemin vers les poids du modèle
          - "lang": langue ("EN" ou "FR") pour le modèle
          - "rows": liste de tuples de lignes (selon le mode)

     Returns
     -------
     Dict[str, Any]
          - "mode": identique à l'entrée
          - "column": identique à l'entrée
          - "updates": les résultats prédits sous la forme décrite ci-dessus
          - "table" ou "csv_path": celui qui a été donné
     """
     mode = task["mode"]
     lang = task["lang"]
     model_path = task["model_path"]
     column = task["column"]

     device = torch.device(WORKER_DEVICE)
     ModelClass = Camembert if lang == "FR" else Bert
     model = ModelClass(device=device)

     updates: Union[List[Tuple[Optional[int], int, int]],
                         List[Tuple[Optional[int], int]]] = []

     if mode == "sql":
          # les lignes sont (doc_id, sentence_id, text)
          doc_sent_ids = [(r[0], r[1]) for r in task["rows"]]
          texts = [r[2] for r in task["rows"]]

          bar = tqdm(
                total=len(texts),
                desc=f"{column} ({current_process().name})", # Traduit la description
                position=WORKER_POS,
                unit="sent",
                leave=False,
                dynamic_ncols=True,
          )

          for i in range(0, len(texts), WORKER_BS):
                sub_txt = texts[i : i + WORKER_BS]
                sub_ids = doc_sent_ids[i : i + WORKER_BS]

                preds = predict_batch_safely(model, sub_txt, model_path)
                for (doc_id, sent_id), pred in zip(sub_ids, preds):
                     updates.append((pred, doc_id, sent_id))

                bar.update(len(sub_txt))

          bar.close()

          return {
                "mode": mode,
                "table": task["table"],
                "column": column,
                "updates": updates,
          }

     else:  # mode == "csv"
          # les lignes sont (row_index, text)
          row_ids = [r[0] for r in task["rows"]]
          texts = [r[1] for r in task["rows"]]

          bar = tqdm(
                total=len(texts),
                desc=f"{column} ({current_process().name})", # Traduit la description
                position=WORKER_POS,
                unit="lignes", # Traduit l'unité
                leave=False,
                dynamic_ncols=True,
          )

          for i in range(0, len(texts), WORKER_BS):
                sub_txt = texts[i : i + WORKER_BS]
                sub_ids = row_ids[i : i + WORKER_BS]

                preds = predict_batch_safely(model, sub_txt, model_path)
                for row_idx, pred in zip(sub_ids, preds):
                     updates.append((pred, row_idx))

                bar.update(len(sub_txt))

          bar.close()

          return {
                "mode": mode,
                "csv_path": task["csv_path"],
                "column": column,
                "updates": updates,
          }


# --------------------------------------------------------------------------------------
#                             MISE À JOUR EN MASSE POUR SQL
# --------------------------------------------------------------------------------------

def bulk_update_column(
     cur: psycopg2.extensions.cursor,
     tbl: str,
     col: str,
     updates: List[Tuple[Optional[int], str, int]],
) -> None:
     """
     Copie 'updates' dans une table temporaire UNLOGGED, puis met à jour la table cible en une seule passe.
     Cela préserve les performances en évitant les commits ligne par ligne.

     Parameters
     ----------
     cur : psycopg2.extensions.cursor
          Curseur ouvert vers la base de données.
     tbl : str
          Nom de la table à mettre à jour.
     col : str
          Nom de la colonne pour stocker les prédictions.
     updates : List[Tuple[Optional[int], str, int]]
          Les prédictions, chacune un tuple (pred, doc_id, sentence_id).
     """
     # Un nom court pour la table temporaire
     tmp_name = f"_tmp_{abs(hash(tbl + col)) % 10_000_000}"
     tmp_ident = sql.Identifier(tmp_name)
     tbl_ident = sql.Identifier(tbl)
     col_ident = sql.Identifier(col)

     # 1) Supprimer + créer temp unlogged
     cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(tmp_ident))
     cur.execute(
          sql.SQL("CREATE UNLOGGED TABLE {} (pred INTEGER, doc_id TEXT, sentence_id INTEGER)")
          .format(tmp_ident)
     )

     # 2) Insérer via execute_values
     execute_values(
          cur,
          sql.SQL("INSERT INTO {} (pred, doc_id, sentence_id) VALUES %s").format(tmp_ident),
          updates,
          page_size=1000,
     )

     # 3) UPDATE ... FROM temp
     cur.execute(
          sql.SQL("""
                UPDATE public.{} AS t
                    SET {} = s.pred
                  FROM {} AS s
                 WHERE t.doc_id = s.doc_id
                    AND t.sentence_id = s.sentence_id
          """).format(tbl_ident, col_ident, tmp_ident)
     )

     # 4) Supprimer temp
     cur.execute(sql.SQL("DROP TABLE {}").format(tmp_ident))


# --------------------------------------------------------------------------------------
#                               FONCTIONS LIÉES AU SQL
# --------------------------------------------------------------------------------------

def ensure_column(
     cur: psycopg2.extensions.cursor,
     table: str,
     column: str
) -> None:
     """
     S'assure qu'une colonne INTEGER nommée `column` existe dans `table`.
     Si elle n'existe pas, la crée.

     Parameters
     ----------
     cur : psycopg2.extensions.cursor
          Un curseur ouvert vers la BDD.
     table : str
          Nom de la table.
     column : str
          Nom de la colonne à assurer.
     """
     cur.execute(
          f"""
          ALTER TABLE public."{table}"
          ADD COLUMN IF NOT EXISTS "{column}" INTEGER;
          """
     )


def ensure_index(cur: psycopg2.extensions.cursor, table: str) -> None:
     """
     Crée l'index composite (doc_id, sentence_id) sur <table> s'il manque.
     Nom de l'index = <table>__pk_idx  (double “__”, < 63 caractères)

     Parameters
     ----------
     cur : psycopg2.extensions.cursor
          Curseur ouvert.
     table : str
          Nom de la table sur laquelle créer l'index.
     """
     idx_name = f"{table}__pk_idx"
     cur.execute(
          sql.SQL(
                "CREATE INDEX IF NOT EXISTS {idx} "
                "ON public.{tbl} (doc_id, sentence_id);"
          ).format(
                idx=sql.Identifier(idx_name),
                tbl=sql.Identifier(table),
          )
     )


def fetch_to_annotate(
     cur: psycopg2.extensions.cursor,
     table: str,
     column: str,
     lang: str,
     limit_n: Optional[int],
     use_random: bool,
     text_col: str,
     lang_col: str,
) -> List[psycopg2.extras.DictRow]:
     """
     Récupère les lignes nécessitant une annotation pour une colonne spécifique, avec une limite optionnelle
     et un échantillonnage aléatoire. Nous ne récupérons que les lignes dont <lang_col> correspond à <lang>.

     Parameters
     ----------
     cur : psycopg2.extensions.cursor
          Curseur BDD actif.
     table : str
          Nom de la table.
     column : str
          Nom de la colonne à filtrer (IS NULL).
     lang : str
          Code langue ("EN"/"FR").
     limit_n : Optional[int]
          Nombre max de lignes à récupérer. Si None, récupérer tout.
     use_random : bool
          Indique s'il faut utiliser ORDER BY RANDOM() ou simplement un tri standard.
     text_col : str
          Le nom de la colonne contenant le texte.
     lang_col : str
          Le nom de la colonne contenant le code de langue.

     Returns
     -------
     List[psycopg2.extras.DictRow]
          Les lignes, chacune contenant (doc_id, sentence_id, <text_col>).
     """
     if limit_n is not None:
          if use_random:
                query = f"""
                     SELECT doc_id, sentence_id, {text_col}
                     FROM public."{table}"
                     WHERE "{column}" IS NULL
                        AND {lang_col} = %s
                     ORDER BY RANDOM()
                     LIMIT {limit_n};
                """
          else:
                query = f"""
                     SELECT doc_id, sentence_id, {text_col}
                     FROM public."{table}"
                     WHERE "{column}" IS NULL
                        AND {lang_col} = %s
                     LIMIT {limit_n};
                """
          cur.execute(query, (lang,))
     else:
          query = f"""
                SELECT doc_id, sentence_id, {text_col}
                FROM public."{table}"
                WHERE "{column}" IS NULL
                  AND {lang_col} = %s;
          """
          cur.execute(query, (lang,))

     return cur.fetchall()


# --------------------------------------------------------------------------------------
#                               ORCHESTRATION PRINCIPALE
# --------------------------------------------------------------------------------------

def main() -> None:
     """
     Orchestre le flux d'annotation pour un fichier CSV ou une base de données SQL, en utilisant
     des workers CPU/GPU parallèles et des prédictions par lots.

     Étapes :
     ------
     1) Demander à l'utilisateur le format de base : .csv ou sql.
     2) Si CSV :
         - Demander le chemin absolu du fichier CSV.
         - Le charger (pandas).
         - Afficher les colonnes ; demander les noms des colonnes de texte et de langue.
         - Lire tous les modèles disponibles dans `models/` ; demander à l'utilisateur de choisir ceux à utiliser.
         - Demander à l'utilisateur l'utilisation des périphériques (CPU, GPU, ou les deux), les tailles de lot, la limite, l'échantillonnage aléatoire.
         - Pour chaque modèle choisi :
            * Déterminer sa langue à partir du nom du dossier (_EN ou _FR).
            * Créer une nouvelle colonne dans le DataFrame si manquante (nommée d'après la catégorie du modèle).
            * Identifier les lignes manquant une valeur dans cette colonne. Parmi ces lignes, filtrer sur la langue correspondante.
            * Les découper en lots et exécuter les prédictions parallèles.
            * Après avoir terminé les prédictions pour cette colonne de modèle, écrire le CSV sur le disque.
     3) Si SQL :
         - Demander interactivement les paramètres de connexion à la BDD.
         - Se connecter et lister les tables disponibles (l'utilisateur en choisit une).
         - Afficher les colonnes ; demander les noms des colonnes de texte et de langue.
         - Lire tous les modèles disponibles dans `models/` ; demander à l'utilisateur de choisir ceux à utiliser.
         - Demander à l'utilisateur l'utilisation des périphériques (CPU, GPU, ou les deux), les tailles de lot, la limite, l'échantillonnage aléatoire.
         - Pour chaque modèle choisi :
            * Déterminer sa langue à partir du nom du dossier (_EN ou _FR).
            * S'assurer que la table a une colonne avec le nom du modèle.
            * Récupérer les lignes qui manquent cette colonne (IS NULL) et correspondent à la langue.
            * Les découper en lots et exécuter les prédictions parallèles.
            * Mettre à jour en masse la table pour cette colonne en une seule passe après avoir terminé tous les lots.
     """
     print("\n[DÉBUT] Script d'annotation - Mode CSV ou SQL\n")
     base_format = input("Quel format de base souhaitez-vous utiliser ? (indiquez csv ou sql) : ").strip().lower()

     if base_format not in {"csv", "sql"}:
          sys.exit("[ERREUR] Choix de format invalide. Veuillez relancer et choisir '.csv' ou 'sql'.")

     # --------------------------------------------------------------------------
     # Recueillir l'entrée utilisateur pour l'utilisation des périphériques et la taille de lot
     # --------------------------------------------------------------------------
     devices = pick_devices_interactively()
     cpu_bs_default: int = 25
     gpu_bs_default: int = 50

     # --- Taille batch CPU -------------------------------------------------------
     if any(d == "cpu" for d in devices):
         inp_cpu = input(f"CPU batch size (default {cpu_bs_default}): ").strip()
         try:
             cpu_bs: int = int(inp_cpu) if inp_cpu else cpu_bs_default
         except ValueError:
             print("[WARN] Invalid CPU batch size; using default.")
             cpu_bs = cpu_bs_default
     else:
         cpu_bs = cpu_bs_default

     # --- Taille batch GPU -------------------------------------------------------
     if any(d in {"cuda", "mps"} for d in devices):
         inp_gpu = input(f"GPU batch size (default {gpu_bs_default}): ").strip()
         try:
             gpu_bs: int = int(inp_gpu) if inp_gpu else gpu_bs_default
         except ValueError:
             print("[WARN] Invalid GPU batch size; using default.")
             gpu_bs = gpu_bs_default
     else:
         gpu_bs = gpu_bs_default

     # --------------------------------------------------------------------------
     # Demander la limite de lignes et l'échantillonnage aléatoire
     # --------------------------------------------------------------------------
     limit_str = input("Combien de lignes souhaitez-vous annoter ? (entier ou 'all') : ").strip().lower()
     if limit_str == "" or limit_str == "all":
          limit_n = None
     else:
          try:
                limit_n = int(limit_str)
          except ValueError:
                print("[WARN] Limite invalide. Utilisation de 'all'.")
                limit_n = None

     rnd = input("Utiliser l'échantillonnage aléatoire ? [o/N] : ").strip().lower()
     use_random = rnd.startswith("o") # 'o' pour 'oui'

     # --------------------------------------------------------------------------
     # Charger les modèles depuis `models/` et laisser l'utilisateur sélectionner dans la liste complète
     # --------------------------------------------------------------------------
     script_dir = os.path.dirname(os.path.abspath(__file__))
     models_dir = os.path.join(script_dir, "..", "..", "models")
     all_models = load_models(models_dir)

     # Les aplatir en une liste de (cat, lang, path), en ignorant l'ancien filtrage par langue
     model_entries = []
     for cat, lang_map in sorted(all_models.items()):
          for lang, path in sorted(lang_map.items()):
                model_entries.append((cat, lang, path))

     print("\n[INFO] Les modèles suivants ont été trouvés :")
     for idx, (cat, lang, path) in enumerate(model_entries, start=1):
          print(f"  {idx}) {cat}_{lang} -> {path}")

     model_choice = input("Sélectionnez les numéros de modèles (séparés par virgule) ou appuyez sur Entrée pour tous : ").strip()
     if model_choice:
          chosen_idxs = []
          try:
                chosen_idxs = [int(x.strip()) for x in model_choice.split(",")]
          except ValueError:
                chosen_idxs = []
          chosen_models = [model_entries[i - 1] for i in chosen_idxs if 1 <= i <= len(model_entries)]
          if not chosen_models:
                print("[WARN] Sélection invalide. Utilisation de tous les modèles.")
                chosen_models = model_entries
     else:
          chosen_models = model_entries

     # --------------------------------------------------------------------------
     # MODE CSV
     # --------------------------------------------------------------------------
     if base_format == "csv":
          # 1) Demander le chemin absolu du CSV et charger
          csv_path = input("Entrez le chemin absolu vers le fichier CSV : ").strip()
          if not os.path.isfile(csv_path):
                sys.exit(f"[ERREUR] Fichier CSV non trouvé : {csv_path}")

          df = pd.read_csv(csv_path, dtype=str).fillna("")
          # 2) Afficher les colonnes, demander à l'utilisateur les colonnes de texte et de langue
          print("\nColonnes disponibles dans le CSV :")
          for col in df.columns:
                print(f"  - {col}")
          text_col = input("\nQuelle colonne contient le texte à annoter ? (entrez le nom entier)").strip()
          lang_col = input("Quelle colonne contient le code de langue ? (entrez le nom entier)").strip()
          if text_col not in df.columns or lang_col not in df.columns:
                sys.exit("[ERREUR] Choix de colonnes invalides.")

          # Nous utiliserons un seul pool multiprocessing
          manager = Manager()
          devices_q = manager.Queue()
          for d in devices:
                devices_q.put(d)

          pos_q = manager.Queue()
          # Nous limiterons le nombre de lignes utilisées par TQDM dans la console
          num_workers = len(devices)
          first_worker_pos = 1
          for p in range(first_worker_pos, first_worker_pos + num_workers):
                pos_q.put(p)

          tqdm.set_lock(RLock())
          pool = Pool(
                processes=num_workers,
                initializer=_worker_init,
                initargs=(devices_q, cpu_bs, gpu_bs, pos_q),
          )

          # Pour chaque modèle choisi, nous effectuons l'annotation au niveau de la colonne
          try:
                for (cat, mlang, mpath) in chosen_models:
                     # Créer la colonne si manquante
                     if cat not in df.columns:
                          df[cat] = ""

                     # Nous avons besoin du sous-ensemble de lignes où la colonne 'cat' est vide (c.-à-d. non annotée)
                     # et la langue de la ligne correspond à la langue du modèle
                     # Nous traiterons la chaîne vide ou None comme "manquant"
                     mask_missing = (df[cat] == "") | (df[cat].isna())
                     mask_lang = (df[lang_col] == mlang)

                     # Combiner
                     target_mask = mask_missing & mask_lang
                     idx_to_annotate = df.index[target_mask].tolist()

                     n_rows = len(idx_to_annotate)
                     if n_rows == 0:
                          print(f"\n[IGNORÉ] Aucune ligne à annoter pour le modèle «{cat}_{mlang}».")
                          continue

                     # Limiter ou randomiser éventuellement
                     if limit_n is not None and limit_n < n_rows:
                          if use_random:
                                idx_to_annotate = random.sample(idx_to_annotate, limit_n)
                          else:
                                idx_to_annotate = idx_to_annotate[:limit_n]
                          n_rows = len(idx_to_annotate)

                     # Diviser en lots
                     tasks = []
                     for i in range(0, n_rows, CHUNK_SIZE):
                          chunk_indices = idx_to_annotate[i : i + CHUNK_SIZE]
                          rows_data = [(idx, df.loc[idx, text_col]) for idx in chunk_indices]

                          tasks.append({
                                "mode": "csv",
                                "csv_path": csv_path,
                                "column": cat,
                                "model_path": mpath,
                                "lang": mlang,
                                "rows": rows_data,
                          })

                     print(f"\n[INFO] Annotation de {n_rows} lignes pour le modèle «{cat}_{mlang}» par lots de {CHUNK_SIZE}...")
                     model_bar = tqdm(
                          total=len(tasks),
                          desc=f"[{cat}_{mlang}]",
                          position=0,
                          unit="lot", # Traduit l'unité
                          leave=True,
                          dynamic_ncols=True,
                     )

                     updates_buffer: List[Tuple[Optional[int], int]] = []

                     for result in pool.imap_unordered(_worker_chunk, tasks):
                          # result["updates"] -> liste de (pred, row_idx)
                          updates_buffer.extend(result["updates"])
                          model_bar.update(1)

                     model_bar.close()

                     # Appliquer maintenant ces mises à jour au DataFrame
                     # updates_buffer = [(pred, row_idx), ...]
                     for pred, row_idx in updates_buffer:
                          # pred peut être None ou un entier
                          if pred is None:
                                df.at[row_idx, cat] = ""
                          else:
                                df.at[row_idx, cat] = str(pred)

                     # Une fois la colonne entièrement traitée, sauvegarder en CSV
                     df.to_csv(csv_path, index=False)
                     print(f"[SAUVEGARDÉ] CSV mis à jour avec les prédictions pour la colonne «{cat}».")

          finally:
                pool.close()
                pool.join()

          print("\n[TERMINÉ] Annotation CSV terminée.")

     # --------------------------------------------------------------------------
     # MODE SQL
     # --------------------------------------------------------------------------
     else:
          # 1) Demander les paramètres BDD
          print("\nEntrez les paramètres de connexion PostgreSQL :")
          host = input("Hôte (défaut 'localhost') : ").strip() or "localhost"
          port_s = input("Port (défaut 5432) : ").strip()
          dbname = input("Nom de la base de données : ").strip()
          user = input("Nom d'utilisateur : ").strip()
          pwd = input("Mot de passe (laisser vide si aucun) : ").strip()

          try:
                port = int(port_s) if port_s else 5432
          except ValueError:
                port = 5432

          db_params: Dict[str, Any] = {
                "host": host,
                "port": port,
                "dbname": dbname,
                "user": user,
                "password": pwd,
          }

          # Connexion
          try:
                conn = psycopg2.connect(**db_params)
          except psycopg2.Error as e:
                sys.exit(f"[ERREUR] Impossible de se connecter à la BDD : {e}")
          conn.autocommit = True
          cur = conn.cursor()

          # 2) Lister les tables pour que l'utilisateur choisisse
          cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema='public';
          """)
          available_tables = [r[0] for r in cur.fetchall()]
          if not available_tables:
                sys.exit("[ERREUR] Aucune table trouvée dans le schéma 'public'.")

          print("\n[INFO] Tables disponibles :")
          for i, t in enumerate(available_tables, start=1):
                print(f"   {i}) {t}")
          tbl_choice = input("Sélectionnez le numéro de la table : ").strip()
          try:
                tbl_idx = int(tbl_choice)
                table_name = available_tables[tbl_idx - 1]
          except Exception:
                sys.exit("[ERREUR] Choix de table invalide.")

          # 3) Afficher les colonnes, demander les colonnes de texte et de langue
          cur.execute(f'SELECT * FROM public."{table_name}" LIMIT 0;')
          col_names = [desc[0] for desc in cur.description]

          print(f"\nColonnes dans la table «{table_name}» :")
          for coln in col_names:
                print(f"  - {coln}")

          text_col = input("\nQuelle colonne contient le texte à annoter ? ").strip()
          lang_col = input("Quelle colonne contient le code de langue ? ").strip()
          if text_col not in col_names or lang_col not in col_names:
                sys.exit("[ERREUR] Choix de colonnes invalides.")

          # 4) Configurer le pool parallèle
          manager = Manager()
          devices_q = manager.Queue()
          for d in devices:
                devices_q.put(d)

          pos_q = manager.Queue()
          num_workers = len(devices)
          first_worker_pos = 1
          for p in range(first_worker_pos, first_worker_pos + num_workers):
                pos_q.put(p)

          tqdm.set_lock(RLock())
          pool = Pool(
                processes=num_workers,
                initializer=_worker_init,
                initargs=(devices_q, cpu_bs, gpu_bs, pos_q),
          )

          # 5) Pour chaque modèle choisi, annoter
          try:
                for (cat, mlang, mpath) in chosen_models:
                     # Assurer la colonne
                     ensure_column(cur, table_name, cat)
                     ensure_index(cur, table_name)

                     # Récupérer les lignes nécessaires
                     rows = fetch_to_annotate(
                          cur,
                          table_name,
                          cat,
                          mlang,
                          limit_n,
                          use_random,
                          text_col,
                          lang_col
                     )

                     n_rows = len(rows)
                     if n_rows == 0:
                          print(f"\n[IGNORÉ] Aucune ligne à annoter pour le modèle «{cat}_{mlang}».")
                          continue

                     # Découper les lignes en lots
                     tasks = []
                     for i in range(0, n_rows, CHUNK_SIZE):
                          chunk_rows = rows[i : i + CHUNK_SIZE]
                          tasks.append({
                                "mode": "sql",
                                "table": table_name,
                                "column": cat,
                                "model_path": mpath,
                                "lang": mlang,
                                "rows": chunk_rows,
                          })

                     print(f"\n[INFO] Annotation de {n_rows} lignes dans la table «{table_name}» pour le modèle «{cat}_{mlang}».")
                     table_bar = tqdm(
                          total=len(tasks),
                          desc=f"{table_name}.{cat}",
                          position=0,
                          unit="lot", # Traduit l'unité
                          leave=True,
                          dynamic_ncols=True,
                     )

                     buffer_updates: List[Tuple[Optional[int], str, int]] = []

                     for result in pool.imap_unordered(_worker_chunk, tasks):
                          buffer_updates.extend(result["updates"])
                          table_bar.update(1)

                     table_bar.close()

                     # Mise à jour en masse
                     if buffer_updates:
                          t0 = time.perf_counter()
                          bulk_update_column(cur, table_name, cat, buffer_updates)
                          delta = time.perf_counter() - t0
                          print(f"[COMMIT] {len(buffer_updates):,} lignes mises à jour pour {table_name}.{cat} en {delta:.2f}s")

          finally:
                pool.close()
                pool.join()
                cur.close()
                conn.close()

          print("\n[TERMINÉ] Annotation SQL terminée.")

     print("\n[SORTIE] Toutes les annotations demandées ont été traitées.")
     # Fin de main()


if __name__ == "__main__":
     main()