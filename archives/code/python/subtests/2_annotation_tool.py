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

NOUVELLE FONCTIONNALITÉ POUR TESTER DES PROMPTS SÉPARÉS ET L'EFFICACITÉ DES
MODÈLES AVEC DES PROMPTS MOINS COMPLEXES:
----------------------------------------------
1) Le script demande maintenant où se trouve le dossier des prompts (.txt).
2) Il va y chercher tous les fichiers .txt.
3) Pour chaque prompt .txt :
   - Il lit le contenu et extrait le thème spécifique (d'après l'exemple de JSON).
   - Il crée (ou vérifie) la colonne correspondante dans les données, 
     portant le même nom que le thème (ex: "culture_and_nationalism").
   - Il exécute le processus d’annotation décrit plus bas, 
     puis met à jour le jeu de données avec les réponses de ce prompt.
4) Chaque prompt est donc appliqué à la même sélection de données, 
   et les annotations sont stockées dans autant de colonnes qu'il y a de prompts.
5) Pour chaque colonne d'annotation créée, on génère automatiquement une seconde
   colonne intitulée "<nom_de_la_colonne_annotation>_inference_time". Elle contient,
   pour chaque ligne annotée, le temps d’inférence du modèle pour cette annotation
   (en secondes).

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
3) Chargement, vérification et analyse détaillée d’un ou plusieurs prompts annotés.
4) Sélection et utilisation d’un modèle Ollama présent sur la machine.
5) Nettoyage et validation de la sortie JSON.
6) Mise à jour des annotations dans la base de données ou dans le fichier,
   avec sauvegarde immédiate ligne par ligne pour ne rien perdre.
7) Exécution parallèle des tâches d’annotation (optionnel).
8) Itération sur tous les prompts .txt d'un dossier, pour créer
   autant de colonnes d'annotation que de prompts.
9) Création automatique des colonnes de temps d'inférence,
   et enregistrement de la durée d'exécution pour chaque ligne annotée.

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
import time        

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
    Pour les temps d'inférence, on ajoute une colonne float (double precision).
    """
    annotation_time_column = f"{annotation_column}_inference_time"
    with engine.begin() as connection:
        # Colonne JSONB
        alter_table_query_annotation = f"""
        ALTER TABLE {table_name}
        ADD COLUMN IF NOT EXISTS {annotation_column} JSONB;
        """
        connection.execute(text(alter_table_query_annotation))
        logging.info(f"Colonne '{annotation_column}' ajoutée à la table '{table_name}' ou déjà existante.")

        # Colonne temps d'inférence
        alter_table_query_time = f"""
        ALTER TABLE {table_name}
        ADD COLUMN IF NOT EXISTS {annotation_time_column} DOUBLE PRECISION;
        """
        connection.execute(text(alter_table_query_time))
        logging.info(f"Colonne '{annotation_time_column}' ajoutée à la table '{table_name}' ou déjà existante.")

def add_annotation_column_df(df, annotation_column):
    """
    Ajoute la colonne d'annotation et la colonne de temps d'inférence
    au DataFrame (CSV/Excel/Parquet/RData/rds) si elles n'existent pas déjà.
    """
    if annotation_column not in df.columns:
        df[annotation_column] = pd.NA
        logging.info(f"Colonne '{annotation_column}' ajoutée au DataFrame.")
    else:
        logging.info(f"Colonne '{annotation_column}' existe déjà dans le DataFrame.")

    annotation_time_column = f"{annotation_column}_inference_time"
    if annotation_time_column not in df.columns:
        df[annotation_time_column] = pd.NA
        logging.info(f"Colonne '{annotation_time_column}' ajoutée au DataFrame.")
    else:
        logging.info(f"Colonne '{annotation_time_column}' existe déjà dans le DataFrame.")

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
        alter_table_query = f"""
        ALTER TABLE {table_name}
        ADD COLUMN IF NOT EXISTS {new_col} BIGSERIAL;
        """
        connection.execute(text(alter_table_query))
        logging.info(f"Nouvelle colonne '{new_col}' (BIGSERIAL) vérifiée/créée dans la table '{table_name}'.")
    return new_col

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
        # Généralement, read_r renvoie un dict {objet1: df, objet2: df...}
        # On suppose qu'il y a un seul objet ou qu'on prend le premier
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
        "Limite la probabilité cumulative des tokens. Plus élevé => plus grande diversité."
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
#                      Nettoyage et correction du JSON                        #
###############################################################################

def clean_json_output(output: str, expected_keys: list) -> str:
    """
    Tente de convertir la sortie du modèle en JSON valide :
    1) Parsing direct
    2) Nettoyage + extraction
    3) Forçage des clés attendues
    """
    # 1) Tentative directe
    try:
        json_data = json.loads(output)
        final_data = {k: json_data.get(k, None) for k in expected_keys}
        return json.dumps(final_data, ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    # 2) Nettoyage + extraction
    text_cleaned = output.strip('`').strip()
    text_cleaned = re.sub(r'^```.*?\n', '', text_cleaned, flags=re.MULTILINE)
    text_cleaned = re.sub(r'\n```$', '', text_cleaned, flags=re.MULTILINE)
    text_cleaned = re.sub(r'//.*?$|/\*.*?\*/', '', text_cleaned, flags=re.DOTALL | re.MULTILINE)

    candidates = re.findall(r'\{.*?\}', text_cleaned, flags=re.DOTALL)
    if not candidates:
        return None

    for candidate in reversed(candidates):
        c_str = candidate.replace("'", '"')
        c_str = re.sub(r',\s*([}\]])', r'\1', c_str)

        try:
            json_data = json.loads(c_str)
            final_data = {k: json_data.get(k, None) for k in expected_keys}
            return json.dumps(final_data, ensure_ascii=False)
        except json.JSONDecodeError:
            continue

    return None

###############################################################################
#                      Appel du modèle et traitement                           #
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

def process_comment(args):
    """
    Appelé en multiprocessing:
      - Construit prompt final
      - Mesure le temps d'inférence
      - Appelle le modèle
      - Nettoie la sortie
      - Retourne (identifiant, dict JSON ou None, inference_time).
    """
    (
        index,
        row,
        base_prompt,
        model_name,
        options,
        expected_keys,
        text_column,
        identifier_column
    ) = args

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
    inference_time = None

    for attempt in range(1, max_attempts + 1):
        start_time = time.time()
        output = analyze_text_with_model(model_name, prompt_final, options)
        end_time = time.time()
        current_inference_time = end_time - start_time

        if output:
            print("\n=== Réponse brute du modèle ===")
            print(output)
            print("================================\n")
        else:
            logging.error(f"Pas de réponse du modèle pour l'ID {identifier} (tentative {attempt}).")

        cleaned_json_str = None
        if output:
            cleaned_json_str = clean_json_output(output, expected_keys)

        if cleaned_json_str is not None:
            inference_time = current_inference_time
            print("=== Réponse nettoyée (utilisée comme annotation) ===")
            print(cleaned_json_str)
            print("===============================================\n")
            break
        else:
            logging.info(f"ID {identifier} : JSON invalide ou vide, tentative {attempt}/{max_attempts}.")

            if attempt == 3:
                prompt_final += (
                    "\n\nRAPPEL: La sortie doit être un unique objet JSON, "
                    "sans texte hors de cet objet.\nRespectez la structure JSON."
                )

    if cleaned_json_str is None:
        logging.error(f"Échec: pas de JSON valide pour l'ID {identifier} après {max_attempts} tentatives.")
        return identifier, None, None

    try:
        cleaned_json = json.loads(cleaned_json_str)
        return identifier, cleaned_json, inference_time
    except json.JSONDecodeError:
        logging.error(f"Impossible de parser l'ID {identifier} malgré le nettoyage.")
        return identifier, None, None

###############################################################################
#          Mise à jour des annotations (ligne par ligne)                      #
###############################################################################

def update_annotation_db(engine, table_name, identifier_column, identifier, 
                         annotation_column, annotation_json, inference_time):
    """
    Met à jour la colonne JSON d'annotation et la colonne de temps d'inférence 
    dans la table PostgreSQL, ligne par ligne.
    """
    annotation_time_col = f"{annotation_column}_inference_time"
    update_query = text(f"""
        UPDATE {table_name}
        SET {annotation_column} = :annotation_json,
            {annotation_time_col} = :inference_time
        WHERE {identifier_column} = :identifier
    """).bindparams(bindparam('annotation_json', type_=JSON))

    with engine.begin() as connection:
        try:
            connection.execute(update_query, {
                'identifier': identifier,
                'annotation_json': annotation_json,
                'inference_time': inference_time
            })
        except SQLAlchemyError as e:
            logging.error(f"Erreur lors de la mise à jour pour l'ID {identifier}: {e}")

def update_annotation_df(df, identifier, identifier_column, annotation_column, 
                         annotation_json, inference_time):
    """
    Met à jour la colonne d'annotation et la colonne de temps d'inférence 
    dans le DataFrame pour l'ID donné.
    Utilisée pour CSV, Excel, Parquet, RData, RDS. Retourne le DataFrame modifié.
    """
    if identifier_column not in df.columns:
        logging.error(f"Colonne ID '{identifier_column}' introuvable dans le DataFrame.")
        return df

    mask = (df[identifier_column] == identifier)
    if mask.any():
        df.loc[mask, annotation_column] = json.dumps(annotation_json, ensure_ascii=False)
        time_col = f"{annotation_column}_inference_time"
        df.loc[mask, time_col] = inference_time
    else:
        logging.error(f"Identifiant '{identifier}' non trouvé dans '{identifier_column}'.")
    return df

###############################################################################
#                         Sauvegarde (ligne par ligne)                         #
###############################################################################

def clear_existing_annotations_db(engine, table_name, annotation_column):
    """
    Réinitialise la colonne d'annotation JSON dans la table PostgreSQL.
    (Ici, on pourrait aussi réinitialiser la colonne de temps d'inférence si besoin.)
    """
    annotation_time_col = f"{annotation_column}_inference_time"
    clear_query = text(f"""
        UPDATE {table_name}
        SET {annotation_column} = NULL,
            {annotation_time_col} = NULL;
    """)
    with engine.begin() as connection:
        try:
            connection.execute(clear_query)
            logging.info(f"Les annotations dans '{annotation_column}' et '{annotation_time_col}' ont été effacées.")
        except SQLAlchemyError as e:
            logging.error(f"Erreur lors de la suppression: {e}")

def clear_existing_annotations_df(df, annotation_column):
    """
    Réinitialise la colonne d'annotation dans un DataFrame (CSV, Excel...).
    Ainsi que la colonne de temps d'inférence correspondante.
    """
    df[annotation_column] = pd.NA
    annotation_time_col = f"{annotation_column}_inference_time"
    if annotation_time_col in df.columns:
        df[annotation_time_col] = pd.NA
    logging.info(f"Les annotations dans '{annotation_column}' (et '{annotation_time_col}' s'il existe) ont été effacées.")
    return df

###############################################################################
#                  Liste des modèles Ollama disponibles                       #
###############################################################################

def list_ollama_models():
    """
    Utilise la commande 'ollama list' pour récupérer la liste des modèles disponibles.
    Extrait uniquement le vrai nom du modèle pour pouvoir l'utiliser dans l'appel 'ollama generate'.
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

            if parts[0].endswith('.'):
                # cas "2." => le nom du modèle est parts[1]
                model_name = parts[1]
            else:
                # cas "deepseek-r1:70b" directement => parts[0]
                model_name = parts[0]

            models_cleaned.append(model_name)

        return models_cleaned

    except FileNotFoundError:
        logging.error("Ollama n'est pas installé ou introuvable dans le PATH.")
        return []

###############################################################################
#                      Extraction du thème depuis le prompt                    #
###############################################################################

def load_prompt(prompt_path):
    """
    Charge le contenu intégral du prompt depuis un fichier texte.
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read()
    return prompt_text

def parse_theme_name(prompt_text):
    """
    Extrait le nom du thème (par exemple "culture_and_nationalism")
    depuis la section "Exemple de JSON :" du prompt.
    On cherche par exemple :
      Exemple de JSON :
      {
        "themes": ["culture_and_nationalism"]
      }

    On renvoie "culture_and_nationalism".

    S'il n'est pas trouvé, on renvoie None.
    """
    pattern = r'"themes"\s*:\s*\[\s*"([^"]+)"'
    match = re.search(pattern, prompt_text)
    if match:
        return match.group(1).strip()
    else:
        return None

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
    #                   Sélection du modèle Ollama                            #
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

    ###########################################################################
    #                   Lecture / Connexion aux données                       #
    ###########################################################################

    if data_source in ['csv', 'excel', 'parquet', 'rdata', 'rds']:
        file_path = input(f"Chemin complet vers le fichier .{data_source} : ").strip()
        while not os.path.isfile(file_path):
            print("Chemin invalide ou fichier inexistant. Réessayez.")
            file_path = input(f"Chemin vers le fichier .{data_source} : ").strip()

        # Lecture du DataFrame
        try:
            df = read_data_file(file_path, data_source)
        except Exception as e:
            logging.error(f"Erreur lors de la lecture du fichier {file_path} : {e}")
            sys.exit(1)

        print(f"Colonnes disponibles : {', '.join(df.columns)}")

        # 1) Colonne texte
        text_column = input("Nom de la colonne contenant le texte à annoter : ").strip()
        while text_column not in df.columns:
            print("Colonne inexistante. Réessayez.")
            text_column = input("Nom de la colonne texte : ").strip()

        # 2) Option de créer un identifiant unique
        create_new_id = prompt_user_yes_no(
            "Voulez-vous créer une variable d'identification unique basée sur la colonne de texte ?"
        )
        if create_new_id:
            df, new_id_col = create_unique_id_df(df, text_column)
            identifier_column = new_id_col
            print(f"La nouvelle colonne '{new_id_col}' servira d'ID unique.")
        else:
            # Sinon on demande l'existant
            identifier_column = input("Nom de la colonne d'identifiant unique : ").strip()
            while identifier_column not in df.columns:
                print("Colonne inexistante. Réessayez.")
                identifier_column = input("Nom de la colonne d'identifiant unique : ").strip()

        # 3) Calcul échantillon
        calculate_ic = prompt_user_yes_no("Calculer taille d'échantillon (IC 95%) ?")
        if calculate_ic:
            choice_ic = prompt_user_choice(
                "Calcul basé sur le nombre de lignes ou une variable unique ?",
                ['lignes', 'variable']
            )
            if choice_ic == 'variable':
                unique_var = input("Variable avec un compte unique : ").strip()
                while unique_var not in df.columns:
                    print("Colonne inexistante. Réessayez.")
                    unique_var = input("Variable unique : ").strip()
                # Ici on pourrait affiner, mais on laisse simple.
            total_comments = len(df)
            sample_size = calculate_sample_size(total_comments)
            print(f"\nTaille d'échantillon recommandée : {sample_size} lignes.\n")
        else:
            sample_size = len(df)
            print(f"\nNombre total de lignes disponibles : {sample_size}\n")

        # 4) Combien annoter
        num_to_annotate = prompt_user_int(
            f"Combien de données annoter ? (max {len(df)})", 1, len(df)
        )
        print(f"{num_to_annotate} lignes choisies.\n")

        # 5) Sélection aléatoire ?
        random_selection = prompt_user_yes_no(
            "Voulez-vous sélectionner ces lignes de manière aléatoire ?"
        )

        # 6) Chemin de sauvegarde pour la mise à jour au fil de l'eau
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

        # 8) Sélection : DataFrame à annoter
        if num_to_annotate < len(df):
            if random_selection:
                df_to_annotate = df.sample(n=num_to_annotate, random_state=42)
            else:
                df_to_annotate = df.iloc[:num_to_annotate]
        else:
            df_to_annotate = df

        logging.info(f"{len(df_to_annotate)} données sélectionnées pour annotation.")

        #######################################################################
        # (NOUVEAU) 9) Chemin du dossier des prompts, on applique chacun      #
        #######################################################################
        prompt_dir = input("Chemin vers le dossier contenant vos prompts (.txt) : ").strip()
        while not os.path.isdir(prompt_dir):
            print("Dossier invalide ou introuvable. Réessayez.")
            prompt_dir = input("Chemin vers le dossier de prompts (.txt) : ").strip()

        # Liste de tous les fichiers .txt
        prompt_files = [f for f in os.listdir(prompt_dir) if f.lower().endswith('.txt')]
        if not prompt_files:
            print("Aucun fichier .txt trouvé dans ce dossier. Abandon.")
            sys.exit(1)

        print("\nFichiers .txt détectés :")
        for pf in prompt_files:
            print(f" - {pf}")

        # Pour chaque prompt, on crée la colonne qui correspond au thème 
        # + la colonne de temps d'inférence, et on exécute l'annotation
        for pf in prompt_files:
            full_prompt_path = os.path.join(prompt_dir, pf)
            base_prompt = load_prompt(full_prompt_path)
            theme_name = parse_theme_name(base_prompt)

            if not theme_name:
                logging.warning(f"Impossible de déterminer un thème pour {pf}. Colonne = 'theme_inconnu'.")
                theme_name = "theme_inconnu"

            # On force la clé attendue à ["themes"], 
            # car dans les prompts c'est toujours "themes" : [...]
            expected_keys = ["themes"]

            # Création (ou vérification) de la colonne correspondante + temps d'inférence
            df = add_annotation_column_df(df, theme_name)

            # Préparation des tâches pour ce prompt
            args_list = []
            for index, row in df_to_annotate.iterrows():
                args_list.append((
                    index, row, base_prompt, model_name, options,
                    expected_keys, text_column, identifier_column
                ))

            total_to_annotate = len(args_list)
            logging.info(f"\n--- Début annotation de {total_to_annotate} textes avec le prompt '{pf}' ---")
            logging.info(f"Colonne de destination : {theme_name}")

            # Annotation en multiprocessing
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                try:
                    futures = {executor.submit(process_comment, a): a[0] for a in args_list}
                    with tqdm(total=total_to_annotate, desc=f'Analyse ({pf})', unit='annotation') as pbar:
                        batch_results = []
                        idx = 0

                        for future in concurrent.futures.as_completed(futures):
                            identifier, output, inference_time = future.result()

                            if output is not None:
                                # Mise à jour dans le DataFrame
                                df = update_annotation_df(
                                    df,
                                    identifier,
                                    identifier_column,
                                    theme_name,
                                    output,
                                    inference_time
                                )
                                # Sauvegarde après chaque annotation
                                try:
                                    write_data_file(df, output_file_path, data_source)
                                except Exception as e:
                                    logging.error(f"Erreur lors de la sauvegarde ligne par ligne : {e}")

                                batch_results.append((identifier, output, inference_time))

                            idx += 1
                            if idx % 10 == 0 and batch_results:
                                selected = random.choice(batch_results)
                                pbar.write(
                                    f"\nExemple d'annotation pour l'ID {selected[0]} (col: {theme_name}) :"
                                )
                                pbar.write(json.dumps(selected[1], ensure_ascii=False, indent=2))
                                pbar.write(f"Temps d'inférence : {selected[2]:.4f} sec")
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

            logging.info(f"Annotation terminée pour le prompt '{pf}' (colonne '{theme_name}').")

        print("\n=== Annotation terminée pour tous les prompts ===\n")

    ###########################################################################
    #                  CAS PostgreSQL                                         #
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

        # 2) Vérifier table
        try:
            with engine.connect() as connection:
                _ = connection.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1;"))
            logging.info(f"La table '{table_name}' est accessible.")
        except SQLAlchemyError as e:
            logging.error(f"Erreur lors de la vérification : {e}")
            sys.exit(1)

        # 3) Liste colonnes
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

        # 5) Option: créer identifiant unique ou en choisir un
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

        # 6) Calcul échantillon
        with engine.connect() as connection:
            result_count = connection.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
            total_comments = result_count.fetchone()[0]

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

        # 7) Combien annoter
        num_to_annotate = prompt_user_int(
            f"Combien de données annoter ? (max {total_comments})", 1, total_comments
        )
        print(f"{num_to_annotate} lignes sélectionnées.\n")

        # 8) Sélection aléatoire ?
        random_selection = prompt_user_yes_no(
            "Voulez-vous sélectionner ces lignes de manière aléatoire ?"
        )

        # 8bis) Chemin de sauvegarde CSV pour les lignes annotées (option)
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

        # 10) Sélection des données à annoter
        with engine.connect() as connection:
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

        # 11) Chemin du dossier des prompts
        prompt_dir = input("Chemin vers le dossier contenant vos prompts (.txt) : ").strip()
        while not os.path.isdir(prompt_dir):
            print("Dossier invalide ou introuvable. Réessayez.")
            prompt_dir = input("Chemin vers le dossier de prompts (.txt) : ").strip()

        prompt_files = [f for f in os.listdir(prompt_dir) if f.lower().endswith('.txt')]
        if not prompt_files:
            print("Aucun fichier .txt trouvé dans ce dossier. Abandon.")
            sys.exit(1)

        print("\nFichiers .txt détectés :")
        for pf in prompt_files:
            print(f" - {pf}")

        # 12) Pour chaque prompt, on crée la colonne + celle du temps d'inférence, on annote
        for pf in prompt_files:
            full_prompt_path = os.path.join(prompt_dir, pf)
            base_prompt = load_prompt(full_prompt_path)
            theme_name = parse_theme_name(base_prompt)
            if not theme_name:
                logging.warning(f"Impossible de déterminer un thème pour {pf}. Colonne = 'theme_inconnu'.")
                theme_name = "theme_inconnu"

            expected_keys = ["themes"]

            # On crée la colonne dans la table (JSONB) + le champ temps d'inférence
            add_annotation_column_db(engine, table_name, theme_name)

            args_list = []
            for index, row in df_to_annotate.iterrows():
                args_list.append((
                    index, row, base_prompt, model_name, options,
                    expected_keys, text_column, identifier_column
                ))

            total_to_annotate = len(args_list)
            logging.info(f"\n--- Début annotation de {total_to_annotate} textes avec le prompt '{pf}' ---")
            logging.info(f"Colonne de destination : {theme_name}")

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                try:
                    futures = {executor.submit(process_comment, a): a[0] for a in args_list}
                    with tqdm(total=total_to_annotate, desc=f'Analyse ({pf})', unit='annotation') as pbar:
                        batch_results = []
                        idx = 0

                        for future in concurrent.futures.as_completed(futures):
                            identifier, output, inference_time = future.result()
                            if output is not None:
                                # Mise à jour en base de données
                                update_annotation_db(
                                    engine,
                                    table_name,
                                    identifier_column,
                                    identifier,
                                    theme_name,
                                    output,
                                    inference_time
                                )
                                batch_results.append((identifier, output, inference_time))

                            idx += 1
                            if idx % 10 == 0 and batch_results:
                                selected_result = random.choice(batch_results)
                                pbar.write(
                                    f"\nExemple d'annotation pour l'ID {selected_result[0]} (col: {theme_name}) :"
                                )
                                pbar.write(json.dumps(selected_result[1], ensure_ascii=False, indent=2))
                                pbar.write(f"Temps d'inférence : {selected_result[2]:.4f} sec")
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

            logging.info(f"Annotation terminée pour le prompt '{pf}' (colonne '{theme_name}').")

        # On récupère les données annotées pour les sauvegarder dans un CSV
        with engine.connect() as connection:
            annotated_query = text(f"""
                SELECT * FROM {table_name}
            """)
            # On prend toutes les colonnes, y compris celles nouvellement créées.
            df_annotated = pd.read_sql_query(annotated_query, connection)

        if df_annotated.empty:
            print("\nAucune ligne n'a été annotée (ou table vide). Le fichier de sortie sera vide.")
        else:
            # Écrit le CSV avec toutes les colonnes, y compris celles nouvellement créées
            try:
                df_annotated.to_csv(output_csv_path, index=False)
                logging.info(f"Annotations sauvegardées (toutes lignes) dans '{output_csv_path}'.")
            except Exception as e:
                logging.error(f"Erreur lors de la sauvegarde CSV : {e}")
                sys.exit(1)

        logging.info("Analyse terminée (PostgreSQL).")


if __name__ == "__main__":
    main()
