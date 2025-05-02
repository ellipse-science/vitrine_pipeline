"""
PROJET :
--------
Vitrine_pipeline

TITRE :
--------
LLM_annotation_tool.py

OBJECTIF PRINCIPAL :
---------------------
Ce script permet d’annoter des textes en interrogeant un modèle de langage.
Il traite des données provenant d’un fichier CSV, Excel, Parquet, RData/rds
ou d’une base PostgreSQL, gère la création d’identifiants uniques, et met
à jour ou sauvegarde les annotations sous forme de JSON, après vérification
et nettoyage de la réponse du modèle.

Il est maintenant possible de traiter un même texte avec plusieurs prompts
successivement. Les résultats JSON obtenus pour un même texte sont fusionnés
en un seul JSON final.

DÉPENDANCES :
-------------
- tqdm
- ollama
- sqlalchemy
- pandas
- pyreadr (pour RData/RDS)
- openpyxl ou xlrd (pour Excel) selon la version de votre fichier
- pyarrow (pour Parquet)
- os
- sys
- json
- re
- math
- concurrent.futures
- logging
- subprocess
- time

FONCTIONNALITÉS PRINCIPALES :
-----------------------------
1) Connexion à une base PostgreSQL ou lecture d’un fichier CSV, Excel,
   Parquet, RData/rds.
2) Ajout et vérification des colonnes d’annotation et d’identifiants uniques.
3) Création automatique d’une colonne temps d’inférence, nommée
   <colonne_annotation>_inference_time (type numérique).
4) Chargement, vérification et analyse détaillée d’un ou de plusieurs prompts.
5) Sélection et utilisation d’un modèle Ollama présent sur la machine.
6) Nettoyage et validation de la sortie JSON (jusqu'à 5 tentatives).
7) Fusion des JSON obtenus (en cas de multiple prompts) en un seul JSON final.
8) Mise à jour des annotations (et du temps d'inférence global) dans la base
   de données ou dans le fichier, avec sauvegarde immédiate ligne par ligne.
9) Exécution parallèle des tâches d’annotation (optionnel).

Auteur :
--------
Antoine Lemor
"""

import pandas as pd
import os
import math
from tqdm import tqdm
from ollama import generate
import concurrent.futures
import random
from sqlalchemy import create_engine, text, JSON, bindparam
from sqlalchemy.exc import SQLAlchemyError
import logging
import sys
import re
import json
import subprocess
import time  # pour mesurer le temps d'inférence

# Pour lire/écrire RData et RDS
try:
    import pyreadr
    HAS_PYREADR = True
except ImportError:
    HAS_PYREADR = False
    logging.warning("Le package 'pyreadr' n'est pas installé. Lecture/écriture RData/RDS indisponible.")

# Configuration des logs (sans timestamp) et suppression des logs indésirables
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Réduire la verbosité des bibliothèques tierces
for logger_name in ['urllib3', 'requests', 'ollama']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


###############################################################################
#                           Fonctions utilitaires                              #
###############################################################################

def connect_to_postgresql(dbname, host, port, user, password):
    """
    Se connecte à la base de données PostgreSQL et retourne un moteur SQLAlchemy.
    """
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_string)
    return engine

def add_annotation_column_db(engine, table_name, annotation_column):
    """
    Ajoute la colonne d'annotation (JSONB) à la table PostgreSQL si elle n'existe pas déjà.
    Ajoute également la colonne de temps d'inférence associée.
    """
    with engine.begin() as connection:
        # Création de la colonne d'annotation
        alter_table_query = f"""
        ALTER TABLE {table_name}
        ADD COLUMN IF NOT EXISTS {annotation_column} JSONB;
        """
        connection.execute(text(alter_table_query))
        logging.info(f"Colonne '{annotation_column}' ajoutée à la table '{table_name}' ou déjà existante.")

        # Création de la colonne temps d'inférence
        time_column = f"{annotation_column}_inference_time"
        alter_table_time_query = f"""
        ALTER TABLE {table_name}
        ADD COLUMN IF NOT EXISTS {time_column} DOUBLE PRECISION;
        """
        connection.execute(text(alter_table_time_query))
        logging.info(f"Colonne '{time_column}' (temps d'inférence) ajoutée ou déjà existante.")

def add_annotation_column_df(df, annotation_column):
    """
    Ajoute la colonne d'annotation au DataFrame si elle n'existe pas déjà.
    Ajoute également la colonne de temps d'inférence associée.
    """
    if annotation_column not in df.columns:
        df[annotation_column] = pd.NA
        logging.info(f"Colonne '{annotation_column}' ajoutée au DataFrame.")
    else:
        logging.info(f"Colonne '{annotation_column}' existe déjà dans le DataFrame.")

    time_column = f"{annotation_column}_inference_time"
    if time_column not in df.columns:
        df[time_column] = pd.NA
        logging.info(f"Colonne '{time_column}' (temps d'inférence) ajoutée au DataFrame.")
    else:
        logging.info(f"Colonne '{time_column}' existe déjà dans le DataFrame.")

    return df

def create_unique_id_df(df, text_column):
    """
    Crée une nouvelle colonne <text_column>_id_for_llm contenant un identifiant unique
    de 1 à len(df) dans un DataFrame (CSV, Excel, Parquet, RData).
    """
    new_col = f"{text_column}_id_for_llm"
    if new_col in df.columns:
        logging.warning(f"La colonne '{new_col}' existe déjà. On ne la recrée pas.")
        return df, new_col
    
    df[new_col] = range(1, len(df) + 1)
    logging.info(f"Nouvelle colonne '{new_col}' créée pour l'identifiant unique.")
    return df, new_col

def create_unique_id_db(engine, table_name, text_column):
    """
    Crée une nouvelle colonne <text_column>_id_for_llm dans la table PostgreSQL si elle n'existe pas,
    puis la remplit avec un identifiant unique (auto-incrémenté).
    """
    new_col = f"{text_column}_id_for_llm"
    with engine.begin() as connection:
        # Créer la colonne si elle n'existe pas (type bigserial)
        alter_table_query = f"""
        ALTER TABLE {table_name}
        ADD COLUMN IF NOT EXISTS {new_col} BIGSERIAL;
        """
        connection.execute(text(alter_table_query))
        
        logging.info(f"Nouvelle colonne '{new_col}' (BIGSERIAL) vérifiée/créée dans la table '{table_name}'.")
    return new_col


###############################################################################
#                     Gestion et chargement des prompts                       #
###############################################################################

def load_prompt(prompt_path):
    """
    Lit le fichier de prompt et sépare le texte principal de la section
    '**Clés JSON Attendues**'. On récupère ensuite la portion attendue pour
    extraire les clés. Si la section n'existe pas, on renvoie un prompt brut
    sans clés.
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read()

    if '**Clés JSON Attendues**' in prompt_text:
        # On sépare le prompt principal de la section attendue
        llm_prompt, expected_keys_section = prompt_text.split('**Clés JSON Attendues**', 1)
        llm_prompt = llm_prompt.strip()
        expected_keys_section = expected_keys_section.strip()
        # Extraction des clés via la fonction ci-dessous
        expected_keys = extract_expected_keys(expected_keys_section)
    else:
        llm_prompt = prompt_text.strip()
        expected_keys = []

    return llm_prompt, expected_keys

def extract_expected_keys(expected_keys_section):
    """
    Tente de détecter de manière plus robuste des clés JSON dans la section.
    1) On recherche d'abord le bloc JSON entre { ... }.
    2) On corrige rapidement les cas de ';' au lieu de ':'.
    3) On parse le bloc corrigé pour récupérer les clés.
    4) Si échec, on retombe sur l'ancienne méthode (regex globale).
    """
    # Recherche du premier bloc d'accolades
    block_match = re.search(r'\{(.*?)\}', expected_keys_section, flags=re.DOTALL)
    if block_match:
        block_str = block_match.group(0)  # Inclut les accolades
        # Correction simple (cas fréquent: ';' à la place de ':')
        block_str_fixed = re.sub(r'"\s*;\s*"', '": "', block_str)

        try:
            loaded = json.loads(block_str_fixed)
            # On récupère simplement les clés
            return list(loaded.keys())
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback : ancienne méthode => on cherche n'importe quelle suite de caractères entre guillemets
    keys = re.findall(r'"([^"]+)"', expected_keys_section)
    # On retire les doublons éventuels (en conservant l'ordre d'apparition)
    final_keys = []
    for k in keys:
        if k not in final_keys:
            final_keys.append(k)

    return final_keys

def verify_prompt_structure(base_prompt, expected_keys):
    """
    Vérifie si la structure du prompt est correcte, et affiche les informations
    sur les clés JSON détectées.
    """
    print("\n=== Vérification de la structure du prompt ===")
    if not base_prompt:
        print("Avertissement : le prompt principal est vide ou non détecté.")
    else:
        print(f"Longueur du prompt principal : {len(base_prompt)} caractères.")

    if expected_keys:
        print(f"Clés JSON détectées : {expected_keys}")
        print("Structure du prompt : OK")
    else:
        print("Aucune clé JSON n'a été détectée.")
        print("Soit le segment '**Clés JSON Attendues**' est absent, soit le bloc n'est pas reconnu.")
    print("================================================\n")


###############################################################################
#                    Fonctions de dialogue / interactions                     #
###############################################################################

def prompt_user_yes_no(question):
    """
    Affiche une question à l'utilisateur, qui doit répondre oui ou non.
    """
    while True:
        choice = input(f"{question} (oui/non) : ").strip().lower()
        if choice in ['oui', 'o']:
            return True
        elif choice in ['non', 'n']:
            return False
        else:
            print("Veuillez répondre par 'oui' ou 'non'.")

def prompt_user_choice(question, choices):
    """
    Demande à l'utilisateur de choisir parmi une liste d'options (choices).
    """
    choices_lower = [choice.lower() for choice in choices]
    while True:
        choice = input(f"{question} ({'/'.join(choices)}): ").strip().lower()
        if choice in choices_lower:
            return choice
        else:
            print(f"Veuillez choisir parmi : {', '.join(choices)}.")

def prompt_user_int(question, min_value=1, max_value=None, default=None):
    """
    Demande à l'utilisateur de saisir un nombre entier dans une plage spécifiée,
    avec la possibilité d'une valeur par défaut.
    """
    while True:
        try:
            inp = input(f"{question} : ").strip()
            if inp == '' and default is not None:
                value = default
            else:
                value = int(inp)
            if value < min_value:
                print(f"Veuillez entrer un nombre entier ≥ {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Veuillez entrer un nombre entier ≤ {max_value}.")
                continue
            return value
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre entier.")

def prompt_user_model_parameters(num_processes):
    """
    Demande à l'utilisateur de définir les paramètres du modèle.
    Explique l'impact de chaque paramètre, et fixe num_thread = num_processes.
    """
    print("\nVeuillez fournir les paramètres du modèle. (Entrée = valeur par défaut)\n")

    def get_param(name, default, description):
        while True:
            inp = input(f"{name} (par défaut {default}) - {description} : ").strip()
            if inp == '':
                return default
            try:
                if isinstance(default, float):
                    return float(inp)
                elif isinstance(default, int):
                    return int(inp)
                else:
                    return inp
            except ValueError:
                print(f"Entrée invalide. Veuillez entrer un(e) {type(default).__name__} valide.")

    temperature = get_param(
        "temperature",
        0.8,
        "Contrôle la créativité du modèle. Plus élevé → plus varié, plus faible → plus déterministe."
    )
    seed = get_param(
        "seed",
        42,
        "Graine aléatoire pour reproductibilité. Même seed => mêmes résultats pour un prompt identique."
    )
    top_p = get_param(
        "top_p",
        0.97,
        "Limite la probabilité cumulative des tokens. Plus élevé → plus grande diversité."
    )

    num_thread = num_processes
    print(f"num_thread (automatiquement défini à {num_thread}) : nombre de threads par processus.")
    num_predict = get_param(
        "num_predict",
        200,
        "Nombre maximal de tokens à générer. Plus élevé => réponse plus longue."
    )

    options = {
        "temperature": temperature,
        "seed": seed,
        "top_p": top_p,
        "num_thread": num_thread,
        "num_predict": num_predict
    }

    print("\nParamètres du modèle définis :")
    for key, value in options.items():
        print(f"  {key} : {value}")

    return options

def calculate_sample_size(total_comments):
    """
    Calcule la taille d'échantillon pour 95% d'IC, marge d'erreur à 5%, p=0.5.
    """
    confidence_level = 0.95
    z = 1.96
    p = 0.5
    e = 0.05

    numerator = (z**2) * p * (1 - p)
    denominator = e**2

    sample_size = (numerator / denominator) / (
        1 + ((numerator / denominator - 1) / total_comments)
    )
    sample_size = min(math.ceil(sample_size), total_comments)
    return sample_size


###############################################################################
#                Nettoyage/correction JSON pour UN prompt                     #
###############################################################################

def clean_json_output(output: str, expected_keys: list) -> str:
    """
    Tente de convertir la sortie du modèle en JSON valide :
    1) Parsing direct de la réponse brute
    2) Si échec, on recherche un bloc JSON dans la réponse, on corrige, on parse
    3) On ne garde que les clés qui figurent dans 'expected_keys' (si elles existent)
       en remplissant avec None le cas échéant (ou si 'expected_keys' est vide, on garde tout).
    4) On renvoie une string JSON finale (ou None si échec).
    """
    # 1) Tentative de parsing direct
    try:
        raw_data = json.loads(output)
        if expected_keys:
            # Conserver uniquement les clés attendues
            final_data = {k: raw_data.get(k, None) for k in expected_keys}
        else:
            final_data = raw_data
        return json.dumps(final_data, ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    # 2) Recherche/Nettoyage d'un éventuel bloc JSON dans le texte
    text_cleaned = output.strip('`').strip()
    text_cleaned = re.sub(r'^```.*?\n', '', text_cleaned, flags=re.MULTILINE)
    text_cleaned = re.sub(r'\n```$', '', text_cleaned, flags=re.MULTILINE)
    # Supprime les commentaires style // ou /* */
    text_cleaned = re.sub(r'//.*?$|/\*.*?\*/', '', text_cleaned, flags=re.DOTALL | re.MULTILINE)

    # On récupère tous les blocs { ... }
    candidates = re.findall(r'\{.*?\}', text_cleaned, flags=re.DOTALL)
    if not candidates:
        return None

    # On essaie de parser en partant du dernier bloc (cas fréquent)
    for candidate in reversed(candidates):
        c_str = candidate.replace("'", '"')
        c_str = re.sub(r',\s*([}\]])', r'\1', c_str)  # retire virgules avant } ou ]
        c_str = re.sub(r'"\s*;\s*"', '": "', c_str)

        try:
            json_data = json.loads(c_str)
            if expected_keys:
                final_data = {k: json_data.get(k, None) for k in expected_keys}
            else:
                final_data = json_data
            return json.dumps(final_data, ensure_ascii=False)
        except json.JSONDecodeError:
            continue

    return None


###############################################################################
#        Appels au modèle et gestion d'annotation (un ou plusieurs prompts)   #
###############################################################################

def analyze_text_with_model(model_name, prompt, options):
    """
    Envoie le prompt au modèle via Ollama et retourne la réponse brute.
    Affiche le prompt pour information.
    """
    print("\n=== Requête au modèle ===")
    print("Prompt envoyé :")
    print(prompt)
    print("=========================\n")

    try:
        response = generate(model_name, prompt, options=options)
        return response['response'].strip()
    except Exception as e:
        logging.error(f"Erreur lors de l'appel au modèle : {e}")
        return None

def merge_json_objects(json_list):
    """
    Reçoit une liste de JSON (dict) et fusionne leurs clés.  
    Si la même clé apparaît dans plusieurs JSON, la dernière occurrence l'emporte.  
    (On peut ajuster ce comportement en fonction du besoin, ici on suppose la dernière prime).
    """
    merged = {}
    for d in json_list:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            merged[k] = v
    return merged

def process_comment_multiple_prompts(args):
    """
    Fonction utilisée en multiprocessing pour le cas "plusieurs prompts".

    Pour chaque phrase :
      - on applique successivement chaque prompt,
      - on nettoie/valide la réponse (jusqu'à 5 tentatives par prompt),
      - on stocke chaque JSON partiel,
      - on fusionne les JSON en un seul,
      - on renvoie l'identifiant, le JSON final et le temps total d'inférence.
    """
    (
        index,
        row,
        list_of_prompts,      # [(prompt_text, expected_keys), (prompt_text, expected_keys), ...]
        model_name,
        options,
        text_column,
        identifier_column
    ) = args

    start_time = time.perf_counter()
    identifier = row[identifier_column]
    text_to_annotate = row[text_column]

    # Contient les JSON "nets" au fur et à mesure
    collected_jsons = []

    for i, (base_prompt, expected_keys) in enumerate(list_of_prompts, start=1):
        # Instructions supplémentaires
        instructions_sup = (
            "\nIMPORTANT:\n"
            "- Répondez exclusivement avec un objet JSON strictement valide.\n"
            "- Aucun texte/commentaire hors de l'objet.\n"
            "- Les clés attendues sont exactement celles spécifiées (si spécifiées).\n"
        )

        prompt_final = f"{base_prompt}\n\n{instructions_sup}\n\nTexte à analyser :\n{text_to_annotate}"

        max_attempts = 5
        cleaned_json_str = None

        # Tentatives multiples pour ce prompt
        for attempt in range(1, max_attempts + 1):
            output = analyze_text_with_model(model_name, prompt_final, options)
            if output:
                print("\n=== Réponse brute du modèle ===")
                print(output)
                print("================================\n")

            if not output:
                logging.error(f"Pas de réponse du modèle pour l'ID {identifier} (Prompt #{i}, tentative {attempt}).")
            else:
                cleaned_json_str = clean_json_output(output, expected_keys)
                if cleaned_json_str is not None:
                    print(f"=== Réponse nettoyée (Prompt #{i}) ===")
                    print(cleaned_json_str)
                    print("==================================\n")
                    break

            logging.info(f"ID {identifier}, Prompt #{i} : JSON invalide ou vide, tentative {attempt}/{max_attempts}.")
            if attempt == 3:
                prompt_final += (
                    "\n\nRAPPEL: La sortie doit être un unique objet JSON, "
                    "sans texte hors de cet objet.\nRespectez la structure JSON."
                )

        if cleaned_json_str is None:
            logging.error(
                f"Échec: pas de JSON valide pour l'ID {identifier}, Prompt #{i} après {max_attempts} tentatives."
            )
            # Si un prompt a échoué, on arrête (on peut choisir de continuer, mais ici on renvoie None)
            end_time = time.perf_counter()
            return identifier, None, (end_time - start_time)

        # Conversion en dict
        try:
            cleaned_json_dict = json.loads(cleaned_json_str)
        except json.JSONDecodeError:
            logging.error(f"Impossible de parser l'ID {identifier}, Prompt #{i} malgré le nettoyage.")
            end_time = time.perf_counter()
            return identifier, None, (end_time - start_time)

        collected_jsons.append(cleaned_json_dict)

    # Fin de la boucle sur les prompts
    # Fusion finale
    final_json_dict = merge_json_objects(collected_jsons)
    final_json_str = json.dumps(final_json_dict, ensure_ascii=False)

    end_time = time.perf_counter()
    inference_time = end_time - start_time
    return identifier, final_json_str, inference_time


def process_comment_single_prompt(args):
    """
    Fonction en multiprocessing pour le cas "un seul prompt" (version initiale).
    On la conserve pour clarté et pour ne pas changer la logique existante.
    """
    (
        index,
        row,
        base_prompt,
        expected_keys,
        model_name,
        options,
        text_column,
        identifier_column
    ) = args

    start_time = time.perf_counter()
    identifier = row[identifier_column]
    text_to_annotate = row[text_column]

    instructions_sup = (
        "\nIMPORTANT:\n"
        "- Répondez exclusivement avec un objet JSON strictement valide.\n"
        "- Aucun texte/commentaire hors de l'objet.\n"
        "- Les clés attendues sont exactement celles spécifiées.\n"
    )
    prompt_final = f"{base_prompt}\n\n{instructions_sup}\n\nTexte à analyser :\n{text_to_annotate}"

    max_attempts = 5
    cleaned_json_str = None

    for attempt in range(1, max_attempts + 1):
        output = analyze_text_with_model(model_name, prompt_final, options)
        if output:
            print("\n=== Réponse brute du modèle ===")
            print(output)
            print("================================\n")

        if not output:
            logging.error(f"Pas de réponse du modèle pour l'ID {identifier} (tentative {attempt}).")
        else:
            cleaned_json_str = clean_json_output(output, expected_keys)
            if cleaned_json_str is not None:
                print("=== Réponse nettoyée (utilisée comme annotation) ===")
                print(cleaned_json_str)
                print("===============================================\n")
                break

        logging.info(f"ID {identifier} : JSON invalide ou vide, tentative {attempt}/{max_attempts}.")

        if attempt == 3:
            prompt_final += (
                "\n\nRAPPEL: La sortie doit être un unique objet JSON, "
                "sans texte hors de cet objet.\nRespectez la structure JSON."
            )

    end_time = time.perf_counter()
    inference_time = end_time - start_time

    if cleaned_json_str is None:
        logging.error(f"Échec: pas de JSON valide pour l'ID {identifier} après {max_attempts} tentatives.")
        return identifier, None, inference_time

    try:
        # On reconvertit en JSON dict (pour finalité), puis re-dumps
        cleaned_json_dict = json.loads(cleaned_json_str)
        final_json_str = json.dumps(cleaned_json_dict, ensure_ascii=False)
        return identifier, final_json_str, inference_time
    except json.JSONDecodeError:
        logging.error(f"Impossible de parser l'ID {identifier} malgré le nettoyage.")
        return identifier, None, inference_time


###############################################################################
#       Mise à jour des annotations (et du temps d'inférence) en base/DF      #
###############################################################################

def update_annotation_db(engine, table_name, identifier_column, identifier,
                         annotation_column, annotation_json):
    """
    Met à jour la colonne JSON d'annotation dans la table PostgreSQL, ligne par ligne.
    """
    update_query = text(f"""
        UPDATE {table_name}
        SET {annotation_column} = :annotation_json
        WHERE {identifier_column} = :identifier
    """).bindparams(bindparam('annotation_json', type_=JSON))

    with engine.begin() as connection:
        try:
            connection.execute(update_query, {
                'identifier': identifier,
                'annotation_json': json.loads(annotation_json) if annotation_json else None
            })
        except SQLAlchemyError as e:
            logging.error(f"Erreur lors de la mise à jour pour l'ID {identifier}: {e}")

def update_inference_time_db(engine, table_name, identifier_column, identifier,
                             annotation_time_column, inference_time):
    """
    Met à jour la colonne du temps d'inférence (float/double) dans la table PostgreSQL, ligne par ligne.
    """
    update_time_query = text(f"""
        UPDATE {table_name}
        SET {annotation_time_column} = :inference_time
        WHERE {identifier_column} = :identifier
    """)

    with engine.begin() as connection:
        try:
            connection.execute(update_time_query, {
                'identifier': identifier,
                'inference_time': inference_time
            })
        except SQLAlchemyError as e:
            logging.error(f"Erreur lors de la mise à jour du temps d'inférence pour l'ID {identifier}: {e}")

def update_annotation_df(df, identifier, identifier_column, annotation_column, annotation_json):
    """
    Met à jour la colonne d'annotation dans le DataFrame pour l'ID donné.
    Utilisée pour CSV, Excel, Parquet, RData, RDS. Retourne le DataFrame modifié.
    """
    if identifier_column not in df.columns:
        logging.error(f"Colonne ID '{identifier_column}' introuvable dans le DataFrame.")
        return df

    mask = (df[identifier_column] == identifier)
    if mask.any():
        df.loc[mask, annotation_column] = annotation_json
    else:
        logging.error(f"Identifiant '{identifier}' non trouvé dans '{identifier_column}'.")
    return df

def update_inference_time_df(df, identifier, identifier_column,
                             annotation_time_column, inference_time):
    """
    Met à jour la colonne du temps d'inférence dans le DataFrame pour l'ID donné.
    """
    if identifier_column not in df.columns:
        logging.error(f"Colonne ID '{identifier_column}' introuvable dans le DataFrame.")
        return df

    mask = (df[identifier_column] == identifier)
    if mask.any():
        df.loc[mask, annotation_time_column] = inference_time
    else:
        logging.error(f"Identifiant '{identifier}' non trouvé dans '{identifier_column}'.")
    return df


###############################################################################
#                         Sauvegarde (ligne par ligne)                         #
###############################################################################

def read_data_file(file_path, file_format):
    """
    Lit le fichier selon le format spécifié et retourne un DataFrame.
    file_format doit être l'un de : 'csv', 'excel', 'parquet', 'rdata', 'rds'.
    """
    file_format = file_format.lower()

    if file_format == 'csv':
        return pd.read_csv(file_path)
    elif file_format == 'excel':
        return pd.read_excel(file_path)
    elif file_format == 'parquet':
        return pd.read_parquet(file_path)
    elif file_format in ['rdata', 'rds']:
        if not HAS_PYREADR:
            raise ImportError("Le package pyreadr est requis pour lire RData/rds.")
        result = pyreadr.read_r(file_path)
        df = list(result.values())[0]
        return df
    else:
        raise ValueError(f"Format inconnu ou non pris en charge : {file_format}")

def write_data_file(df, file_path, file_format):
    """
    Écrit le DataFrame dans le fichier selon le format spécifié.
    Opération faite après chaque annotation pour ne rien perdre.
    """
    file_format = file_format.lower()

    if file_format == 'csv':
        df.to_csv(file_path, index=False)
    elif file_format == 'excel':
        df.to_excel(file_path, index=False)
    elif file_format == 'parquet':
        df.to_parquet(file_path, index=False)
    elif file_format == 'rdata':
        if not HAS_PYREADR:
            raise ImportError("Le package pyreadr est requis pour écrire RData.")
        pyreadr.write_rdata({'data': df}, file_path)
    elif file_format == 'rds':
        if not HAS_PYREADR:
            raise ImportError("Le package pyreadr est requis pour écrire RDS.")
        pyreadr.write_rds(df, file_path)
    else:
        raise ValueError(f"Format inconnu ou non pris en charge : {file_format}")


###############################################################################
#                         Nettoyage global (optionnel)                         #
###############################################################################

def clear_existing_annotations_db(engine, table_name, annotation_column):
    """
    Réinitialise la colonne d'annotation JSON dans la table PostgreSQL.
    """
    clear_query = text(f"UPDATE {table_name} SET {annotation_column} = NULL;")
    with engine.begin() as connection:
        try:
            connection.execute(clear_query)
            logging.info(f"Les annotations dans '{annotation_column}' ont été effacées.")
        except SQLAlchemyError as e:
            logging.error(f"Erreur lors de la suppression: {e}")

def clear_existing_annotations_df(df, annotation_column):
    """
    Réinitialise la colonne d'annotation dans un DataFrame (CSV, Excel...).
    """
    df[annotation_column] = pd.NA
    logging.info(f"Les annotations dans '{annotation_column}' ont été effacées.")
    return df


###############################################################################
#                  Liste des modèles Ollama disponibles                       #
###############################################################################

def list_ollama_models():
    """
    Utilise la commande 'ollama list' pour récupérer la liste des modèles disponibles.
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error("Impossible d'obtenir la liste des modèles Ollama. Erreur : " + result.stderr)
            return []
        
        lines = result.stdout.strip().splitlines()
        models_cleaned = []

        for line in lines:
            # Exclure la ligne d'en-tête ou les lignes vides
            if "NAME" in line and "MODIFIED" in line:
                continue
            if not line.strip():
                continue

            parts = line.split()
            # Gestion d'une éventuelle numérotation en début de ligne
            if parts[0].endswith('.'):
                model_name = parts[1]
            else:
                model_name = parts[0]
            models_cleaned.append(model_name)

        return models_cleaned

    except FileNotFoundError:
        logging.error("Ollama n'est pas installé ou introuvable dans le PATH.")
        return []


###############################################################################
#                            Script principal                                  #
###############################################################################

def main():
    print("=== Application d'Annotation ===\n")

    # (NOUVEAU) On propose plusieurs sources de données
    data_source = prompt_user_choice(
        "Dans quel format sont les données à annoter ?",
        ['csv', 'excel', 'parquet', 'rdata', 'rds', 'postgresql']
    )

    ###########################################################################
    #           Sélection du modèle Ollama (commun aux deux cas)              #
    ###########################################################################
    models_available = list_ollama_models()
    if not models_available:
        logging.error("Aucun modèle Ollama n'a été trouvé ou 'ollama list' a échoué.")
        sys.exit(1)

    print("\n=== Modèles Ollama disponibles ===")
    for i, m in enumerate(models_available, start=1):
        print(f"{i}. {m}")
    print("==================================\n")

    model_choice = prompt_user_int("Entrez le numéro du modèle à utiliser", 1, len(models_available))
    model_name = models_available[model_choice - 1]
    logging.info(f"Modèle sélectionné : {model_name}")

    #
    # (NOUVEAU) Vérification : veut-on utiliser un seul prompt ou plusieurs ?
    #
    multiple_prompts = prompt_user_yes_no("Voulez-vous utiliser plusieurs prompts successifs pour chaque texte ?")

    if multiple_prompts:
        n_prompts = prompt_user_int("Combien de prompts souhaitez-vous utiliser ?", 2)
        list_of_prompts = []
        for i in range(1, n_prompts + 1):
            prompt_path = input(f"Chemin complet vers le prompt #{i} (.txt) : ").strip()
            while not os.path.isfile(prompt_path):
                print("Chemin invalide ou fichier inexistant. Réessayez.")
                prompt_path = input(f"Chemin complet vers le prompt #{i} (.txt) : ").strip()

            base_prompt_i, expected_keys_i = load_prompt(prompt_path)
            verify_prompt_structure(base_prompt_i, expected_keys_i)
            list_of_prompts.append((base_prompt_i, expected_keys_i))
    else:
        # Un seul prompt
        prompt_path = input("Chemin complet vers le fichier prompt (.txt) : ").strip()
        while not os.path.isfile(prompt_path):
            print("Chemin invalide ou fichier inexistant. Réessayez.")
            prompt_path = input("Chemin complet vers le fichier prompt (.txt) : ").strip()

        base_prompt, expected_keys = load_prompt(prompt_path)
        verify_prompt_structure(base_prompt, expected_keys)
        # On stocke tout de même dans un conteneur similaire (plus simple pour la suite).
        list_of_prompts = [(base_prompt, expected_keys)]

    ###########################################################################
    #                  CAS 1 : Fichiers (CSV, Excel, Parquet, RData/rds)      #
    ###########################################################################
    if data_source in ['csv', 'excel', 'parquet', 'rdata', 'rds']:
        file_path = input(f"Chemin complet vers le fichier .{data_source} : ").strip()
        while not os.path.isfile(file_path):
            print("Chemin invalide ou fichier inexistant. Réessayez.")
            file_path = input(f"Chemin vers le fichier .{data_source} : ").strip()

        # Lecture du DataFrame
        try:
            df_loaded = read_data_file(file_path, data_source)
        except Exception as e:
            logging.error(f"Erreur lors de la lecture du fichier {file_path} : {e}")
            sys.exit(1)

        print(f"Colonnes disponibles : {', '.join(df_loaded.columns)}")

        # 1) Colonne texte
        text_column = input("Nom de la colonne contenant le texte à annoter : ").strip()
        while text_column not in df_loaded.columns:
            print("Colonne inexistante. Réessayez.")
            text_column = input("Nom de la colonne texte : ").strip()

        # (NOUVEAU) Vérification si une colonne d'annotations existe déjà et si l'on souhaite la poursuivre
        resume_annotation = prompt_user_yes_no(
            "Existe-t-il déjà une colonne contenant des annotations d'un modèle et souhaitez-vous poursuivre l'annotation dans cette colonne ?"
        )
        if resume_annotation:
            annotation_column = input("Nom de la colonne existante contenant les annotations : ").strip()
            while annotation_column not in df_loaded.columns:
                print("Colonne inexistante. Réessayez.")
                annotation_column = input("Nom de la colonne existante contenant les annotations : ").strip()
            identifier_column = input("Nom de la colonne d'identifiant unique utilisée précédemment : ").strip()
            while identifier_column not in df_loaded.columns:
                print("Colonne inexistante. Réessayez.")
                identifier_column = input("Nom de la colonne d'identifiant unique utilisée précédemment : ").strip()
            inference_time_column = input("Nom de la colonne de temps d'inférence utilisée précédemment : ").strip()
            while inference_time_column not in df_loaded.columns:
                print("Colonne inexistante. Réessayez.")
                inference_time_column = input("Nom de la colonne de temps d'inférence utilisée précédemment : ").strip()
            annotated_rows = df_loaded[annotation_column].notna().sum()
            logging.info(f"{annotated_rows} lignes ont déjà été annotées dans la colonne '{annotation_column}'.")
            # Conserver le DataFrame complet et filtrer uniquement les lignes non annotées pour le traitement
            full_df = df_loaded.copy()
            df_to_annotate = df_loaded[df_loaded[annotation_column].isna()].copy()
            if df_to_annotate.empty:
                print("Toutes les lignes ont déjà été annotées. Aucune nouvelle annotation à réaliser.")
                sys.exit(0)
            total_comments = len(df_to_annotate)
            sample_size = total_comments
            print(f"\nAprès filtrage, {total_comments} lignes sont disponibles pour annotation.\n")
        else:
            # 2) Option de créer un identifiant unique
            create_new_id = prompt_user_yes_no(
                "Voulez-vous créer une variable d'identification unique basée sur la colonne de texte ?"
            )
            if create_new_id:
                full_df, new_id_col = create_unique_id_df(df_loaded, text_column)
                identifier_column = new_id_col
                print(f"La nouvelle colonne '{new_id_col}' servira d'ID unique.")
            else:
                identifier_column = input("Nom de la colonne d'identifiant unique : ").strip()
                while identifier_column not in df_loaded.columns:
                    print("Colonne inexistante. Réessayez.")
                    identifier_column = input("Nom de la colonne d'identifiant unique : ").strip()

            # 3) Colonne d'annotation + colonne temps d'inférence
            annotation_column = input("Nom de la nouvelle colonne pour les annotations (JSON) : ").strip()
            full_df = add_annotation_column_df(df_loaded, annotation_column)
            # Pour une nouvelle annotation, on traite l'ensemble du DataFrame
            df_to_annotate = full_df.copy()
            total_comments = len(full_df)

            # 4) Calcul échantillon
            calculate_ic = prompt_user_yes_no("Calculer taille d'échantillon (IC 95%) ?")
            if calculate_ic:
                choice_ic = prompt_user_choice(
                    "Calcul basé sur le nombre de lignes ou une variable unique ?",
                    ['lignes', 'variable']
                )
                if choice_ic == 'variable':
                    unique_var = input("Variable avec un compte unique : ").strip()
                    while unique_var not in full_df.columns:
                        print("Colonne inexistante. Réessayez.")
                        unique_var = input("Variable unique : ").strip()
                sample_size = calculate_sample_size(total_comments)
                print(f"\nTaille d'échantillon recommandée : {sample_size} lignes.\n")
            else:
                sample_size = total_comments
                print(f"\nNombre total de lignes disponibles : {sample_size}\n")

        # 5) Combien annoter
        num_to_annotate = prompt_user_int(
            f"Combien de données annoter ? (max {total_comments})", 1, total_comments
        )
        print(f"{num_to_annotate} lignes choisies.\n")

        # 6) Sélection aléatoire ?
        random_selection = prompt_user_yes_no(
            "Voulez-vous sélectionner ces lignes de manière aléatoire ?"
        )

        # (NOUVEAU) Chemin de sauvegarde
        output_file_path = input(
            f"Chemin de sauvegarde du fichier annoté (.{data_source}) : "
        ).strip()
        output_file_path = os.path.abspath(output_file_path)
        parent_dir = os.path.dirname(output_file_path)

        if not os.path.exists(parent_dir):
            print(f"Le répertoire '{parent_dir}' n'existe pas. Le créer ? (oui/non) : ", end='')
            create_dir = prompt_user_yes_no("")
            if create_dir:
                try:
                    os.makedirs(parent_dir)
                    logging.info(f"Répertoire '{parent_dir}' créé.")
                except Exception as e:
                    logging.error(f"Impossible de créer '{parent_dir}': {e}")
                    sys.exit(1)
            else:
                logging.error("Impossible de sauvegarder sans répertoire valide.")
                sys.exit(1)

        # 7) Multiprocessing
        num_processes = prompt_user_int("Combien de processus parallèles ?", 1)
        options = prompt_user_model_parameters(num_processes)

        # 8) Sélection des lignes à annoter
        if num_to_annotate < len(df_to_annotate):
            if random_selection:
                df_subset = df_to_annotate.sample(n=num_to_annotate, random_state=42)
            else:
                df_subset = df_to_annotate.iloc[:num_to_annotate]
        else:
            df_subset = df_to_annotate

        logging.info(f"{len(df_subset)} données sélectionnées pour annotation.")

        # 9) Préparation des tâches pour le multiprocessing
        args_list = []
        if multiple_prompts:
            # On enverra la liste de prompts entière
            for index, row in df_subset.iterrows():
                args_list.append((
                    index, row,
                    list_of_prompts,  # [(prompt_text, expected_keys), ...]
                    model_name,
                    options,
                    text_column,
                    identifier_column
                ))
        else:
            # Single prompt
            (single_base_prompt, single_expected_keys) = list_of_prompts[0]
            for index, row in df_subset.iterrows():
                args_list.append((
                    index,
                    row,
                    single_base_prompt,
                    single_expected_keys,
                    model_name,
                    options,
                    text_column,
                    identifier_column
                ))

        total_to_annotate = len(args_list)
        logging.info(f"Annotation de {total_to_annotate} textes.")

        # 10) Exécution parallèle
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            try:
                if multiple_prompts:
                    futures = {executor.submit(process_comment_multiple_prompts, a): a[0] for a in args_list}
                else:
                    futures = {executor.submit(process_comment_single_prompt, a): a[0] for a in args_list}

                with tqdm(total=total_to_annotate, desc='Analyse des données', unit='annotation') as pbar:
                    batch_results = []
                    idx = 0

                    for future in concurrent.futures.as_completed(futures):
                        identifier, output_json, inference_time = future.result()

                        # Mise à jour et sauvegarde immédiate dans le DataFrame complet
                        if output_json is not None:
                            full_df = update_annotation_df(
                                full_df,
                                identifier,
                                identifier_column,
                                annotation_column,
                                output_json
                            )
                            full_df = update_inference_time_df(
                                full_df,
                                identifier,
                                identifier_column,
                                f"{annotation_column}_inference_time",
                                inference_time
                            )
                            try:
                                write_data_file(full_df, output_file_path, data_source)
                            except Exception as e:
                                logging.error(f"Erreur lors de la sauvegarde ligne par ligne : {e}")
                            batch_results.append((identifier, output_json, inference_time))

                        idx += 1
                        if idx % 10 == 0 and batch_results:
                            selected = random.choice(batch_results)
                            tqdm.write(f"\nExemple d'annotation pour l'ID {selected[0]} :")
                            tqdm.write(json.dumps(json.loads(selected[1]), ensure_ascii=False, indent=2))
                            tqdm.write(f"Temps d'inférence : {selected[2]:.4f} s")
                            tqdm.write("-" * 80)
                            batch_results = []

                        if idx % 500 == 0:
                            percentage = (idx / total_to_annotate) * 100
                            tqdm.write(f"Progression : {idx}/{total_to_annotate} ({percentage:.2f}%).")

                        pbar.update(1)

            except KeyboardInterrupt:
                logging.warning("Interrompu par l'utilisateur.")
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)
            except Exception as e:
                logging.error(f"Erreur inattendue : {e}")
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)

        logging.info("Analyse terminée (Fichier).")

    ###########################################################################
    #                  CAS 2 : Lecture via PostgreSQL                          #
    ###########################################################################
    elif data_source == 'postgresql':
        #
        # 1) Connexion PostgreSQL
        #
        dbname = input("Nom de la base de données : ").strip()
        host = input("Hôte de la base (défaut 'localhost') : ").strip() or 'localhost'
        port_input = input("Port de la base (défaut 5432) : ").strip()
        port = int(port_input) if port_input else 5432
        user = input("Nom d'utilisateur : ").strip()
        password = input("Mot de passe : ").strip()

        try:
            engine = connect_to_postgresql(dbname, host, port, user, password)
            logging.info("Connexion PostgreSQL réussie.")
        except Exception as e:
            logging.error(f"Connexion PostgreSQL impossible : {e}")
            sys.exit(1)

        table_name = input("Nom de la table à annoter : ").strip()

        # 2) Vérifier l'accessibilité de la table
        try:
            with engine.connect() as connection:
                _ = connection.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1;"))
            logging.info(f"La table '{table_name}' est accessible.")
        except SQLAlchemyError as e:
            logging.error(f"Erreur lors de la vérification : {e}")
            sys.exit(1)

        # 3) Liste des colonnes de la table
        with engine.connect() as connection:
            result = connection.execute(text(
                f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}';"
            ))
            columns = [row[0] for row in result]

        print(f"Colonnes disponibles dans '{table_name}' : {', '.join(columns)}")

        # 4) Colonne texte
        text_column = input("Nom de la colonne contenant le texte à annoter : ").strip()
        while text_column not in columns:
            print("Colonne inexistante. Réessayez.")
            text_column = input("Nom de la colonne texte : ").strip()

        # (NOUVEAU) Vérification si une colonne d'annotations existe déjà et si l'on souhaite la poursuivre
        resume_annotation = prompt_user_yes_no(
            "Existe-t-il déjà une colonne contenant des annotations d'un modèle et souhaitez-vous poursuivre l'annotation dans cette colonne ?"
        )
        if resume_annotation:
            annotation_column = input("Nom de la colonne existante contenant les annotations (JSONB) : ").strip()
            while annotation_column not in columns:
                print("Colonne inexistante. Réessayez.")
                annotation_column = input("Nom de la colonne existante contenant les annotations (JSONB) : ").strip()
            identifier_column = input("Nom de la colonne d'identifiant unique utilisée précédemment : ").strip()
            while identifier_column not in columns:
                print("Colonne inexistante. Réessayez.")
                identifier_column = input("Nom de la colonne d'identifiant unique utilisée précédemment : ").strip()
            annotation_time_column = input("Nom de la colonne de temps d'inférence utilisée précédemment : ").strip()
            while annotation_time_column not in columns:
                print("Colonne inexistante. Réessayez.")
                annotation_time_column = input("Nom de la colonne de temps d'inférence utilisée précédemment : ").strip()
            with engine.connect() as connection:
                result_count = connection.execute(text(
                    f"SELECT COUNT(*) FROM {table_name} WHERE {annotation_column} IS NOT NULL;"
                ))
                annotated_rows = result_count.fetchone()[0]
            logging.info(f"{annotated_rows} lignes ont déjà été annotées dans la colonne '{annotation_column}'.")
            # Comptage des lignes non annotées pour reprendre l'annotation
            with engine.connect() as connection:
                result_count = connection.execute(text(
                    f"SELECT COUNT(*) FROM {table_name} WHERE {annotation_column} IS NULL;"
                ))
                total_comments = result_count.fetchone()[0]
            if total_comments == 0:
                print("Toutes les lignes ont déjà été annotées. Aucune nouvelle annotation à réaliser.")
                sys.exit(0)
        else:
            # 5) Option: créer un identifiant unique ou en choisir un existant
            create_new_id = prompt_user_yes_no(
                "Voulez-vous créer une variable d'identification unique basée sur la colonne de texte ?"
            )
            if create_new_id:
                from sqlalchemy import exc
                try:
                    new_id_col = create_unique_id_db(engine, table_name, text_column)
                    identifier_column = new_id_col
                    print(f"La nouvelle colonne '{new_id_col}' servira d'ID unique.")
                except exc.SQLAlchemyError as e:
                    logging.error(f"Erreur lors de la création de la colonne unique: {e}")
                    sys.exit(1)
            else:
                identifier_column = input(
                    "Veuillez indiquer la colonne d'identifiant unique (ex: 'comment_id') : "
                ).strip()
                while identifier_column not in columns:
                    print("Colonne inexistante. Réessayez.")
                    identifier_column = input("Nom de la colonne d'ID : ").strip()

            # 6) Colonne annotation + colonne temps
            annotation_column = input("Nom de la colonne pour les annotations (JSONB) : ").strip()
            add_annotation_column_db(engine, table_name, annotation_column)
            annotation_time_column = f"{annotation_column}_inference_time"
            # Pour le calcul du total, lire le nombre total de lignes en base
            with engine.connect() as connection:
                result_count = connection.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
                total_comments = result_count.fetchone()[0]

        # 7) Calcul échantillon
        if not resume_annotation:
            calculate_ic = prompt_user_yes_no("Calculer taille d'échantillon (IC 95%) ?")
            if calculate_ic:
                choice_ic = prompt_user_choice(
                    "Calcul basé sur lignes ou variable unique ?",
                    ['lignes', 'variable']
                )
                if choice_ic == 'variable':
                    unique_var = input("Nom de la variable unique : ").strip()
                    while unique_var not in columns:
                        print("Colonne inexistante. Réessayez.")
                        unique_var = input("Variable unique : ").strip()
                sample_size = calculate_sample_size(total_comments)
                print(f"\nTaille d'échantillon recommandée : {sample_size} lignes.\n")
            else:
                sample_size = total_comments
                print(f"\nNombre total de lignes disponibles : {sample_size}\n")
        else:
            print(f"\nIl y a {total_comments} lignes disponibles pour annotation (lignes non annotées).\n")

        # 8) Combien annoter
        num_to_annotate = prompt_user_int(
            f"Combien de données annoter ? (max {total_comments})", 1, total_comments
        )
        print(f"{num_to_annotate} lignes sélectionnées.\n")

        # 8bis) Sélection aléatoire ?
        random_selection = prompt_user_yes_no(
            "Voulez-vous sélectionner ces lignes de manière aléatoire ?"
        )

        # 8ter) Chemin de sauvegarde CSV pour les lignes annotées seulement
        output_csv_path = input("Chemin de sauvegarde (.csv) pour les lignes annotées seulement : ").strip()
        output_csv_path = os.path.abspath(output_csv_path)
        parent_dir = os.path.dirname(output_csv_path)

        if not os.path.exists(parent_dir):
            print(f"Le répertoire '{parent_dir}' n'existe pas. Le créer ? (oui/non) : ", end='')
            create_dir = prompt_user_yes_no("")
            if create_dir:
                try:
                    os.makedirs(parent_dir)
                    logging.info(f"Répertoire '{parent_dir}' créé.")
                except Exception as e:
                    logging.error(f"Impossible de créer '{parent_dir}': {e}")
                    sys.exit(1)
            else:
                logging.error("Impossible de sauvegarder sans répertoire valide.")
                sys.exit(1)

        # 9) Multiprocessing
        num_processes = prompt_user_int("Combien de processus parallèles ?", 1)
        options = prompt_user_model_parameters(num_processes)

        # 10) Sélection des données à annoter en base
        with engine.connect() as connection:
            if resume_annotation:
                if random_selection:
                    query = text(
                        f"SELECT {identifier_column}, {text_column} FROM {table_name} "
                        f"WHERE {annotation_column} IS NULL "
                        f"ORDER BY RANDOM() LIMIT {num_to_annotate};"
                    )
                else:
                    query = text(
                        f"SELECT {identifier_column}, {text_column} FROM {table_name} "
                        f"WHERE {annotation_column} IS NULL "
                        f"ORDER BY {identifier_column} ASC LIMIT {num_to_annotate};"
                    )
            else:
                if num_to_annotate < total_comments:
                    if random_selection:
                        query = text(
                            f"SELECT {identifier_column}, {text_column} FROM {table_name} "
                            f"ORDER BY RANDOM() LIMIT {num_to_annotate};"
                        )
                    else:
                        query = text(
                            f"SELECT {identifier_column}, {text_column} FROM {table_name} "
                            f"ORDER BY {identifier_column} ASC LIMIT {num_to_annotate};"
                        )
                else:
                    query = text(f"SELECT {identifier_column}, {text_column} FROM {table_name};")

            df_to_annotate = pd.read_sql_query(query, connection)
            logging.info(f"Nombre de lignes réellement sélectionnées : {len(df_to_annotate)}")

        # 11) Préparation des arguments pour le multiprocessing
        args_list = []
        if multiple_prompts:
            for index, row in df_to_annotate.iterrows():
                args_list.append((
                    index,
                    row,
                    list_of_prompts,  # liste de tuples (prompt, expected_keys)
                    model_name,
                    options,
                    text_column,
                    identifier_column
                ))
        else:
            (single_base_prompt, single_expected_keys) = list_of_prompts[0]
            for index, row in df_to_annotate.iterrows():
                args_list.append((
                    index,
                    row,
                    single_base_prompt,
                    single_expected_keys,
                    model_name,
                    options,
                    text_column,
                    identifier_column
                ))

        total_to_annotate = len(args_list)
        logging.info(f"Début annotation de {total_to_annotate} textes.")

        # 12) Exécution parallèle
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            try:
                if multiple_prompts:
                    futures = {executor.submit(process_comment_multiple_prompts, a): a[0] for a in args_list}
                else:
                    futures = {executor.submit(process_comment_single_prompt, a): a[0] for a in args_list}

                with tqdm(total=total_to_annotate, desc='Analyse des données', unit='annotation') as pbar:
                    batch_results = []
                    idx = 0

                    for future in concurrent.futures.as_completed(futures):
                        identifier, output_json, inference_time = future.result()
                        if output_json is not None:
                            # Mise à jour en base de données (ligne par ligne)
                            update_annotation_db(
                                engine,
                                table_name,
                                identifier_column,
                                identifier,
                                annotation_column,
                                output_json
                            )
                            # Mise à jour du temps d'inférence
                            update_inference_time_db(
                                engine,
                                table_name,
                                identifier_column,
                                identifier,
                                annotation_time_column,
                                inference_time
                            )

                            batch_results.append((identifier, output_json, inference_time))

                        idx += 1
                        if idx % 10 == 0 and batch_results:
                            selected_result = random.choice(batch_results)
                            pbar.write(f"\nExemple d'annotation pour l'ID {selected_result[0]} :")
                            pbar.write(json.dumps(json.loads(selected_result[1]), ensure_ascii=False, indent=2))
                            pbar.write(f"Temps d'inférence : {selected_result[2]:.4f} s")
                            pbar.write("-" * 80)
                            batch_results = []

                        if idx % 500 == 0:
                            percentage = (idx / total_to_annotate) * 100
                            pbar.write(f"Progression : {idx}/{total_to_annotate} ({percentage:.2f}%).")

                        pbar.update(1)

            except KeyboardInterrupt:
                logging.warning("Interrompu par l'utilisateur.")
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)
            except Exception as e:
                logging.error(f"Erreur inattendue : {e}")
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)

        # 13) Export CSV des lignes effectivement annotées
        with engine.connect() as connection:
            annotated_query = text(f"""
                SELECT {identifier_column}, {text_column}, {annotation_column}, {annotation_time_column}
                FROM {table_name}
                WHERE {annotation_column} IS NOT NULL
            """)
            df_annotated = pd.read_sql_query(annotated_query, connection)

        if df_annotated.empty:
            print("\nAucune ligne n'a été annotée en base. Le fichier de sortie sera vide.")
        else:
            try:
                df_annotated.to_csv(output_csv_path, index=False)
                logging.info(f"Annotations sauvegardées (lignes annotées seulement) dans '{output_csv_path}'.")
            except Exception as e:
                logging.error(f"Erreur lors de la sauvegarde CSV : {e}")
                sys.exit(1)

        logging.info("Analyse terminée (PostgreSQL).")


if __name__ == '__main__':
    main()
