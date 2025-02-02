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
    Ajoute la colonne d'annotation à la table PostgreSQL si elle n'existe pas déjà.
    """
    with engine.begin() as connection:
        alter_table_query = f"""
        ALTER TABLE {table_name}
        ADD COLUMN IF NOT EXISTS {annotation_column} JSONB;
        """
        connection.execute(text(alter_table_query))
        logging.info(f"Colonne '{annotation_column}' ajoutée à la table '{table_name}' ou déjà existante.")

def add_annotation_column_csv(df, annotation_column):
    """
    Ajoute la colonne d'annotation au DataFrame CSV si elle n'existe pas déjà.
    """
    if annotation_column not in df.columns:
        df[annotation_column] = pd.NA
        logging.info(f"Colonne '{annotation_column}' ajoutée au DataFrame.")
    else:
        logging.info(f"Colonne '{annotation_column}' existe déjà dans le DataFrame.")

def create_unique_id_csv(df, text_column):
    """
    Crée une nouvelle colonne <text_column>_id_for_llm contenant un identifiant unique
    de 1 à len(df).
    """
    new_col = f"{text_column}_id_for_llm"
    if new_col in df.columns:
        logging.warning(f"La colonne '{new_col}' existe déjà. On ne la recrée pas.")
        return new_col
    
    df[new_col] = range(1, len(df) + 1)
    logging.info(f"Nouvelle colonne '{new_col}' créée pour l'identifiant unique dans le CSV.")
    return new_col

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

def load_prompt(prompt_path):
    """
    Charge le prompt depuis un fichier texte et extrait, si présents, les noms de clés JSON attendues.
    Le séparateur doit être '**Clés JSON Attendues**'.
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read()

    # Séparation du prompt principal et des clés attendues
    if '**Clés JSON Attendues**' in prompt_text:
        llm_prompt, expected_keys_section = prompt_text.split('**Clés JSON Attendues**', 1)
        expected_keys = extract_expected_keys(expected_keys_section)
    else:
        llm_prompt = prompt_text
        expected_keys = []

    return llm_prompt.strip(), expected_keys

def extract_expected_keys(expected_keys_section):
    """
    Extrait la liste de clés attendues depuis la section dédiée du prompt,
    en cherchant des guillemets.
    """
    keys = re.findall(r'"([^"]+)"', expected_keys_section)
    return keys

def verify_prompt_structure(base_prompt, expected_keys):
    """
    Vérifie si la structure du prompt est correcte, affiche les informations
    sur la partie principale et sur les clés JSON détectées.
    """
    print("\n=== Vérification de la structure du prompt ===")
    print(f"Longueur du prompt principal : {len(base_prompt)} caractères.")
    if expected_keys:
        print(f"Clés JSON détectées : {expected_keys}")
        print("Structure du prompt : OK, des clés JSON ont été trouvées.")
    else:
        print("Aucune clé JSON n'a été détectée dans le prompt.")
        print("Assurez-vous que la section '**Clés JSON Attendues**' est présente ou que des clés y figurent.")
    print("================================================\n")

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
    # 1) Tentative direct
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
#                      Appel au modèle et traitement                           #
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
      - Appelle le modèle
      - Nettoie la sortie
      - Retourne (identifiant, dict JSON ou None).
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

    if cleaned_json_str is None:
        logging.error(f"Échec: pas de JSON valide pour l'ID {identifier} après {max_attempts} tentatives.")
        return identifier, None

    try:
        cleaned_json = json.loads(cleaned_json_str)
        return identifier, cleaned_json
    except json.JSONDecodeError:
        logging.error(f"Impossible de parser l'ID {identifier} malgré le nettoyage.")
        return identifier, None

###############################################################################
#          Mise à jour des annotations (CSV ou PostgreSQL)                    #
###############################################################################

def update_annotation_db(engine, table_name, identifier_column, identifier, annotation_column, annotation_json):
    """
    Met à jour la colonne JSON d'annotation dans la table PostgreSQL.
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
                'annotation_json': annotation_json
            })
        except SQLAlchemyError as e:
            logging.error(f"Erreur lors de la mise à jour pour l'ID {identifier}: {e}")

def update_annotation_csv(df, identifier, identifier_column, annotation_column, annotation_json):
    """
    Met à jour la colonne d'annotation dans le DataFrame CSV pour l'ID donné.
    """
    if identifier_column not in df.columns:
        logging.error(f"Colonne ID '{identifier_column}' introuvable dans le DataFrame.")
        return df

    if not df[df[identifier_column] == identifier].empty:
        df.loc[df[identifier_column] == identifier, annotation_column] = json.dumps(annotation_json, ensure_ascii=False)
    else:
        logging.error(f"Identifiant '{identifier}' non trouvé dans '{identifier_column}'.")
    return df

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

def clear_existing_annotations_csv(df, annotation_column):
    """
    Réinitialise la colonne d'annotation dans un DataFrame CSV.
    """
    df[annotation_column] = pd.NA
    logging.info(f"Les annotations dans '{annotation_column}' ont été effacées.")
    return df

###############################################################################
#                      Vérification du prompt (nouveau)                       #
###############################################################################

def verify_prompt_structure(base_prompt, expected_keys):
    """
    Vérifie si la structure du prompt est correcte, affiche les informations
    sur la partie principale et sur les clés JSON détectées.
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
        print("Aucune clé JSON n'a été détectée dans le prompt.")
        print("Assurez-vous d'avoir la section '**Clés JSON Attendues**' ou d'y déclarer des clés.")
    print("================================================\n")

###############################################################################
#                            Script principal                                  #
###############################################################################

def main():
    print("=== Application d'Annotation ===\n")

    # Choix CSV vs PostgreSQL
    data_source = prompt_user_choice(
        "Les variables à annoter se trouvent dans une base .csv ou PostgreSQL ?",
        ['csv', 'postgresql']
    )

    if data_source == 'csv':
        #
        # 1) Lecture CSV
        #
        csv_path = input("Chemin complet vers le fichier .csv : ").strip()
        while not os.path.isfile(csv_path):
            print("Chemin invalide ou fichier inexistant. Réessayez.")
            csv_path = input("Chemin complet vers le fichier .csv : ").strip()

        df = pd.read_csv(csv_path)
        print(f"Colonnes disponibles : {', '.join(df.columns)}")

        # 2) Colonne texte
        text_column = input("Nom de la colonne contenant le texte à annoter : ").strip()
        while text_column not in df.columns:
            print("Colonne inexistante. Réessayez.")
            text_column = input("Nom de la colonne texte : ").strip()

        # 3) Option de créer un identifiant unique
        create_new_id = prompt_user_yes_no(
            "Voulez-vous créer une variable d'identification unique basée sur la colonne de texte ?"
        )
        if create_new_id:
            new_id_col = create_unique_id_csv(df, text_column)
            identifier_column = new_id_col
            print(f"La nouvelle colonne '{new_id_col}' servira d'ID unique.")
        else:
            # Si on ne crée pas, on demande l'existant
            identifier_column = input(
                "Veuillez indiquer le nom de la colonne d'identifiant unique (ex: 'video_id') : "
            ).strip()
            while identifier_column not in df.columns:
                print("Colonne inexistante. Réessayez.")
                identifier_column = input("Nom de la colonne d'identifiant unique : ").strip()

        # 4) Colonne d'annotation
        annotation_column = input("Nom de la nouvelle colonne pour les annotations : ").strip()
        add_annotation_column_csv(df, annotation_column)

        # 5) Calcul échantillon
        calculate_ic = prompt_user_yes_no("Calculer taille d'échantillon représentative (IC 95%) ?")
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
            
            sample_size = calculate_sample_size(len(df))
            print(f"\nTaille d'échantillon recommandée : {sample_size} lignes.\n")
        else:
            sample_size = len(df)
            print(f"\nNombre total de lignes disponibles : {sample_size}\n")

        # 6) Combien annoter
        num_to_annotate = prompt_user_int(
            f"Combien de données annoter ? (max {len(df)})", 1, len(df)
        )
        print(f"{num_to_annotate} lignes choisies.\n")

        # 6bis) Sélection aléatoire ?
        random_selection = prompt_user_yes_no(
            "Voulez-vous sélectionner ces lignes de manière aléatoire ?"
        )

        # 6ter) Chemin de sauvegarde (avant annotation)
        output_csv_path = input("Chemin de sauvegarde du fichier annoté (.csv) : ").strip()
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

        # 7) Prompt
        prompt_path = input("Chemin complet vers le fichier prompt (.txt) : ").strip()
        while not os.path.isfile(prompt_path):
            print("Chemin invalide ou fichier inexistant. Réessayez.")
            prompt_path = input("Chemin complet du fichier prompt (.txt) : ").strip()

        # 8) Multiprocessing
        num_processes = prompt_user_int("Combien de processus parallèles ?", 1)
        options = prompt_user_model_parameters(num_processes)

        # 9) Chargement du prompt
        base_prompt, expected_keys = load_prompt(prompt_path)

        # 9bis) Vérification de la structure du prompt
        verify_prompt_structure(base_prompt, expected_keys)

        # 10) Sélection
        if num_to_annotate < len(df):
            if random_selection:
                # Échantillonnage aléatoire
                df_to_annotate = df.sample(n=num_to_annotate, random_state=42)
            else:
                df_to_annotate = df.iloc[:num_to_annotate]
        else:
            df_to_annotate = df

        logging.info(f"{len(df_to_annotate)} données sélectionnées pour annotation.")

        # 11) Préparation des tâches
        args_list = []
        for index, row in df_to_annotate.iterrows():
            args_list.append((
                index, row, base_prompt, 'deepseek-r1:70b', options,
                expected_keys, text_column, identifier_column
            ))

        total_to_annotate = len(args_list)
        logging.info(f"Annotation de {total_to_annotate} textes.")

        batch_size = 10
        batch_results = []
        idx = 0

        # 12) Exécution parallèle
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            try:
                futures = {executor.submit(process_comment, a): a[0] for a in args_list}
                with tqdm(total=total_to_annotate, desc='Analyse des données', unit='annotation') as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        identifier, output = future.result()
                        # Si on a un output correct, on met à jour la DataFrame d'origine
                        if output is not None:
                            df = update_annotation_csv(df, identifier, identifier_column, annotation_column, output)
                            batch_results.append((identifier, output))

                        idx += 1
                        if idx % batch_size == 0 and batch_results:
                            selected = random.choice(batch_results)
                            tqdm.write(f"\nExemple d'annotation pour l'ID {selected[0]} :")
                            tqdm.write(json.dumps(selected[1], ensure_ascii=False, indent=2))
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

        # 13) Sauvegarde CSV : on n'enregistre QUE les lignes annotées
        df_annotated = df[df[annotation_column].notna()]
        if len(df_annotated) == 0:
            print("\nAucune ligne n'a été annotée. Le fichier de sortie sera vide.")
        else:
            try:
                df_annotated.to_csv(output_csv_path, index=False)
                logging.info(f"Annotations sauvegardées (lignes annotées seulement) dans '{output_csv_path}'.")
            except Exception as e:
                logging.error(f"Erreur lors de la sauvegarde CSV : {e}")
                sys.exit(1)

        logging.info("Analyse terminée (CSV).")

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

        # 6) Colonne annotation
        annotation_column = input("Nom de la colonne pour les annotations (JSONB) : ").strip()
        add_annotation_column_db(engine, table_name, annotation_column)

        # 7) Calcul échantillon
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

        # 8) Combien annoter
        num_to_annotate = prompt_user_int(
            f"Combien de données annoter ? (max {total_comments})", 1, total_comments
        )
        print(f"{num_to_annotate} lignes sélectionnées.\n")

        # 8bis) Sélection aléatoire ?
        random_selection = prompt_user_yes_no(
            "Voulez-vous sélectionner ces lignes de manière aléatoire ?"
        )

        # 8ter) Chemin de sauvegarde CSV pour les lignes annotées
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

        # 9) Prompt
        prompt_path = input("Chemin complet vers le fichier prompt (.txt) : ").strip()
        while not os.path.isfile(prompt_path):
            print("Chemin invalide ou fichier inexistant. Réessayez.")
            prompt_path = input("Chemin complet prompt (.txt) : ").strip()

        # 10) Multiprocessing
        num_processes = prompt_user_int("Combien de processus parallèles ?", 1)
        options = prompt_user_model_parameters(num_processes)

        # 11) Chargement prompt
        base_prompt, expected_keys = load_prompt(prompt_path)

        # 11bis) Vérification structure prompt
        verify_prompt_structure(base_prompt, expected_keys)

        # 12) Sélection 
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

        # 13) Préparation
        args_list = []
        for index, row in df_to_annotate.iterrows():
            args_list.append((
                index, row, base_prompt, 'deepseek-r1:70b', options,
                expected_keys, text_column, identifier_column
            ))

        total_to_annotate = len(args_list)
        logging.info(f"Début annotation de {total_to_annotate} textes.")

        batch_size = 10
        batch_results = []
        idx = 0

        # 14) Multiprocessing
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            try:
                futures = {executor.submit(process_comment, a): a[0] for a in args_list}
                with tqdm(total=total_to_annotate, desc='Analyse des données', unit='annotation') as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        identifier, output = future.result()
                        if output is not None:
                            # On n'a pas mis à jour la BDD df => On créera un df en mémoire 
                            # Pour la sortie, on stockera un petit DataFrame local.
                            # Mais on met déjà en DB la mise à jour : 
                            update_annotation_db(
                                engine, table_name, identifier_column, identifier,
                                annotation_column, output
                            )
                            batch_results.append((identifier, output))

                        idx += 1
                        if idx % batch_size == 0 and batch_results:
                            selected_result = random.choice(batch_results)
                            pbar.write(f"\nExemple d'annotation pour l'ID {selected_result[0]} :")
                            pbar.write(json.dumps(selected_result[1], ensure_ascii=False, indent=2))
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

        # 14bis) On récupère seulement les lignes annotées pour la sauvegarde CSV
        # Il faut recharger la table partiellement ou construire un DataFrame local 
        # Commençons par re-sélectionner (ID, annotation_column)
        with engine.connect() as connection:
            # On suppose qu'on a fait le 'WHERE annotation_column IS NOT NULL'
            annotated_query = text(f"""
                SELECT {identifier_column}, {text_column}, {annotation_column}
                FROM {table_name}
                WHERE {annotation_column} IS NOT NULL
            """)
            df_annotated = pd.read_sql_query(annotated_query, connection)

        if df_annotated.empty:
            print("\nAucune ligne n'a été annotée en base. Le fichier de sortie sera vide.")
        else:
            # Sauvegarde
            try:
                df_annotated.to_csv(output_csv_path, index=False)
                logging.info(f"Annotations sauvegardées (lignes annotées seulement) dans '{output_csv_path}'.")
            except Exception as e:
                logging.error(f"Erreur lors de la sauvegarde CSV : {e}")
                sys.exit(1)

        logging.info("Analyse terminée (PostgreSQL).")


if __name__ == "__main__":
    main()
