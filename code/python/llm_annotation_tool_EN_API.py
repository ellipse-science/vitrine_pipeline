#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PROJECT:
--------
Vitrine_pipeline

TITLE:
--------
LLM_annotation_tool.py

MAIN OBJECTIVE:
---------------------
This script annotates texts by querying a language model.
It processes data from a CSV, Excel, Parquet, RData/rds file or a PostgreSQL database,
manages the creation of unique identifiers, and updates or saves the annotations in JSON format,
after verifying and cleaning the model's response.

It is now possible to process the same text with multiple successive prompts.
The JSON results obtained for the same text are merged into a single final JSON.

(NEW) FUNCTIONALITY:
---------------------
1) At the beginning of the code, the user is asked whether they wish to use the OpenAI API (new 1.0+ version).
   - If yes, they must enter their OpenAI API key and then choose the OpenAI model (e.g., "gpt-3.5-turbo" or "gpt-4").
   - If no, the code proceeds with the local Ollama models.

NOTE ON OPENAI LIBRARY (>=1.0.0):
---------------------------------
OpenAI has changed its Python library interface in version 1.0.0 and above.
Calling `openai.ChatCompletion.create(...)` is no longer supported.
Instead, we must import and instantiate a client from `openai import OpenAI`,
and then call methods like:

    client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        ...
    )

Please check the official documentation or the user notices for more details on usage.

DEPENDENCIES:
-------------
- tqdm
- ollama
- sqlalchemy
- pandas
- pyreadr (for RData/RDS)
- openpyxl or xlrd (for Excel) depending on your file version
- pyarrow (for Parquet)
- os
- sys
- json
- re
- math
- concurrent.futures
- logging
- subprocess
- time
- (NEW) openai>=1.0.0 (if you want to use the OpenAI API)

MAIN FEATURES:
-----------------------------
1) Connect to a PostgreSQL database or read a CSV, Excel, Parquet, RData/rds file.
2) Add and verify annotation columns and unique identifiers.
3) Automatically create an inference time column, named <annotation_column>_inference_time (numeric type).
4) Load, verify, and perform detailed analysis of one or several prompts.
5) Select and use an Ollama model present on the machine OR use the new OpenAI Python API (>=1.0.0).
6) Clean and validate the JSON output (up to 5 attempts).
7) Merge JSON objects obtained (in the case of multiple prompts) into a single final JSON.
8) Update the annotations (and overall inference time) in the database or file, with immediate line-by-line saving.
9) Parallel execution of annotation tasks (optional).

Author:
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
import time  # to measure inference time

# (NEW) OpenAI import for ChatGPT API usage (>=1.0.0)
try:
    from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError, APIError
    HAS_NEW_OPENAI = True
except ImportError:
    # Either openai is not installed or is <1.0.0
    HAS_NEW_OPENAI = False
    OpenAI = None

# For reading/writing RData and RDS
try:
    import pyreadr
    HAS_PYREADR = True
except ImportError:
    HAS_PYREADR = False
    logging.warning("The 'pyreadr' package is not installed. Reading/writing RData/RDS unavailable.")

# Logging configuration (without timestamp) and suppression of unwanted logs
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Reduce verbosity of third-party libraries
for logger_name in ['urllib3', 'requests', 'ollama']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


###############################################################################
#                           Utility Functions                                 #
###############################################################################

def connect_to_postgresql(dbname, host, port, user, password):
    """
    Connects to the PostgreSQL database and returns a SQLAlchemy engine.
    """
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_string)
    return engine

def add_annotation_column_db(engine, table_name, annotation_column):
    """
    Adds the annotation column (JSONB) to the PostgreSQL table if it does not already exist.
    Also adds the associated inference time column.
    """
    with engine.begin() as connection:
        # Create the annotation column
        alter_table_query = f"""
        ALTER TABLE {table_name}
        ADD COLUMN IF NOT EXISTS {annotation_column} JSONB;
        """
        connection.execute(text(alter_table_query))
        logging.info(f"Column '{annotation_column}' added to table '{table_name}' or already exists.")

        # Create the inference time column
        time_column = f"{annotation_column}_inference_time"
        alter_table_time_query = f"""
        ALTER TABLE {table_name}
        ADD COLUMN IF NOT EXISTS {time_column} DOUBLE PRECISION;
        """
        connection.execute(text(alter_table_time_query))
        logging.info(f"Column '{time_column}' (inference time) added or already exists.")

def add_annotation_column_df(df, annotation_column):
    """
    Adds the annotation column to the DataFrame if it does not already exist.
    Also adds the associated inference time column.
    """
    if annotation_column not in df.columns:
        df[annotation_column] = pd.NA
        logging.info(f"Column '{annotation_column}' added to the DataFrame.")
    else:
        logging.info(f"Column '{annotation_column}' already exists in the DataFrame.")

    time_column = f"{annotation_column}_inference_time"
    if time_column not in df.columns:
        df[time_column] = pd.NA
        logging.info(f"Column '{time_column}' (inference time) added to the DataFrame.")
    else:
        logging.info(f"Column '{time_column}' already exists in the DataFrame.")

    return df

def create_unique_id_df(df, text_column):
    """
    Creates a new column <text_column>_id_for_llm containing a unique identifier
    from 1 to len(df) in a DataFrame (CSV, Excel, Parquet, RData).
    """
    new_col = f"{text_column}_id_for_llm"
    if new_col in df.columns:
        logging.warning(f"Column '{new_col}' already exists. It will not be recreated.")
        return df, new_col
    
    df[new_col] = range(1, len(df) + 1)
    logging.info(f"New column '{new_col}' created for unique identification.")
    return df, new_col

def create_unique_id_db(engine, table_name, text_column):
    """
    Creates a new column <text_column>_id_for_llm in the PostgreSQL table if it does not exist,
    then fills it with a unique (auto-incremented) identifier.
    """
    new_col = f"{text_column}_id_for_llm"
    with engine.begin() as connection:
        # Create the column if it does not exist (type bigserial)
        alter_table_query = f"""
        ALTER TABLE {table_name}
        ADD COLUMN IF NOT EXISTS {new_col} BIGSERIAL;
        """
        connection.execute(text(alter_table_query))
        
        logging.info(f"New column '{new_col}' (BIGSERIAL) verified/created in table '{table_name}'.")
    return new_col


###############################################################################
#                    Prompt Management and Loading                            #
###############################################################################

def load_prompt(prompt_path):
    """
    Reads the prompt file and separates the main text from the
    '**Expected JSON Keys**' section. Then retrieves the expected keys.
    If the section does not exist, returns a raw prompt without keys.
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read()

    if '**Expected JSON Keys**' in prompt_text:
        # Separate the main prompt from the expected section
        llm_prompt, expected_keys_section = prompt_text.split('**Expected JSON Keys**', 1)
        llm_prompt = llm_prompt.strip()
        expected_keys_section = expected_keys_section.strip()
        expected_keys = extract_expected_keys(expected_keys_section)
    else:
        llm_prompt = prompt_text.strip()
        expected_keys = []

    return llm_prompt, expected_keys

def extract_expected_keys(expected_keys_section):
    """
    Attempts to robustly detect JSON keys in the section.
    1) First, search for the JSON block between { ... }.
    2) Quickly correct cases of ';' instead of ':'.
    3) Parse the corrected block to retrieve the keys.
    4) If it fails, fall back on the old method (global regex).
    """
    block_match = re.search(r'\{(.*?)\}', expected_keys_section, flags=re.DOTALL)
    if block_match:
        block_str = block_match.group(0)  # Includes braces
        block_str_fixed = re.sub(r'"\s*;\s*"', '": "', block_str)
        try:
            loaded = json.loads(block_str_fixed)
            return list(loaded.keys())
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: old method => search for any sequence of characters between quotes
    keys = re.findall(r'"([^"]+)"', expected_keys_section)
    final_keys = []
    for k in keys:
        if k not in final_keys:
            final_keys.append(k)
    return final_keys

def verify_prompt_structure(base_prompt, expected_keys):
    """
    Verifies whether the prompt structure is correct and displays
    information about the detected JSON keys.
    """
    print("\n=== Verification of Prompt Structure ===")
    if not base_prompt:
        print("Warning: the main prompt is empty or not detected.")
    else:
        print(f"Length of main prompt: {len(base_prompt)} characters.")

    if expected_keys:
        print(f"Detected JSON keys: {expected_keys}")
        print("Prompt structure: OK")
    else:
        print("No JSON keys have been detected.")
        print("Either the '**Expected JSON Keys**' segment is missing or the block is not recognized.")
    print("==========================================\n")


###############################################################################
#                    Dialogue / Interaction Functions                         #
###############################################################################

def prompt_user_yes_no(question):
    """
    Displays a question to the user, who must answer yes or no.
    """
    while True:
        choice = input(f"{question} (yes/no): ").strip().lower()
        if choice in ['yes', 'y']:
            return True
        elif choice in ['no', 'n']:
            return False
        else:
            print("Please answer with 'yes' or 'no'.")

def prompt_user_choice(question, choices):
    """
    Asks the user to choose from a list of options.
    """
    choices_lower = [choice.lower() for choice in choices]
    while True:
        choice = input(f"{question} ({'/'.join(choices)}): ").strip().lower()
        if choice in choices_lower:
            return choice
        else:
            print(f"Please choose among: {', '.join(choices)}.")

def prompt_user_int(question, min_value=1, max_value=None, default=None):
    """
    Asks the user to enter an integer within a specified range,
    with the possibility of a default value.
    """
    while True:
        try:
            inp = input(f"{question}: ").strip()
            if inp == '' and default is not None:
                value = default
            else:
                value = int(inp)
            if value < min_value:
                print(f"Please enter an integer ≥ {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Please enter an integer ≤ {max_value}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def prompt_user_model_parameters(num_processes):
    """
    Asks the user to define the model parameters.
    Explains the impact of each parameter, and sets num_thread = num_processes.
    """
    print("\nPlease provide the model parameters. (Enter = default value)\n")

    def get_param(name, default, description):
        while True:
            inp = input(f"{name} (default {default}) - {description}: ").strip()
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
                print(f"Invalid input. Please enter a valid {type(default).__name__}.")

    temperature = get_param(
        "temperature",
        0.8,
        "Controls the model's creativity. Higher → more varied, lower → more deterministic."
    )
    seed = get_param(
        "seed",
        42,
        "Random seed for reproducibility. Same seed => same results for an identical prompt."
    )
    top_p = get_param(
        "top_p",
        0.97,
        "Limits the cumulative probability of tokens. Higher → greater diversity."
    )

    num_thread = num_processes
    print(f"num_thread (automatically set to {num_thread}): number of threads per process.")
    num_predict = get_param(
        "num_predict",
        200,
        "Maximum number of tokens to generate. Higher => longer response."
    )

    options = {
        "temperature": temperature,
        "seed": seed,
        "top_p": top_p,
        "num_thread": num_thread,
        "num_predict": num_predict
    }

    print("\nModel parameters set:")
    for key, value in options.items():
        print(f"  {key} : {value}")

    return options

def calculate_sample_size(total_comments):
    """
    Calculates the sample size for 95% CI, 5% margin of error, p=0.5.
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
#                JSON Cleaning/Correction for A Single Prompt                #
###############################################################################

def clean_json_output(output: str, expected_keys: list) -> str:
    """
    Attempts to convert the model's output into valid JSON:
    1) Directly parse the raw response.
    2) If that fails, search for a JSON block in the response, clean it, and parse.
    3) Retain only the keys that appear in 'expected_keys' (if provided)
       filling with None if necessary (or if 'expected_keys' is empty, keep everything).
    4) Returns a final JSON string (or None if unsuccessful).
    """
    # 1) Attempt direct parsing
    try:
        raw_data = json.loads(output)
        if expected_keys:
            final_data = {k: raw_data.get(k, None) for k in expected_keys}
        else:
            final_data = raw_data
        return json.dumps(final_data, ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    # 2) Try to find any JSON block in the text
    text_cleaned = output.strip('`').strip()
    text_cleaned = re.sub(r'^```.*?\n', '', text_cleaned, flags=re.MULTILINE)
    text_cleaned = re.sub(r'\n```$', '', text_cleaned, flags=re.MULTILINE)
    # Remove comments like // or /* */
    text_cleaned = re.sub(r'//.*?$|/\*.*?\*/', '', text_cleaned, flags=re.DOTALL | re.MULTILINE)

    candidates = re.findall(r'\{.*?\}', text_cleaned, flags=re.DOTALL)
    if not candidates:
        return None

    for candidate in reversed(candidates):
        c_str = candidate.replace("'", '"')
        c_str = re.sub(r',\s*([}\]])', r'\1', c_str)  # remove commas before } or ]
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
#       Model Calls and Annotation Management (Single or Multiple Prompts)    #
###############################################################################

def analyze_text_with_model(model_name, prompt, options):
    """
    Sends the prompt to either a local model (using Ollama) or the OpenAI API.
    
    If using the OpenAI API, this function lazily initializes the client in the worker
    process to avoid pickling issues with non-picklable objects such as thread locks.
    
    Parameters:
        model_name (str): Name of the model to use.
        prompt (str): The prompt to send to the model.
        options (dict): A dictionary containing model parameters and API usage flags.
        
    Returns:
        str: The stripped response from the model, or None if an error occurred.
    """
    use_openai_api = options.get("use_openai_api", False)
    if use_openai_api:
        # Check if the OpenAI client is already initialized in this process.
        openai_client = options.get("openai_client", None)
        if openai_client is None:
            # Retrieve the API key from options and initialize the client.
            openai_api_key = options.get("openai_api_key", None)
            if openai_api_key is None:
                logging.error("OpenAI API key is missing in options.")
                return None
            try:
                openai_client = OpenAI(
                    api_key=openai_api_key,
                    max_retries=2,      # Default number of retries
                    timeout=600.0       # Timeout in seconds
                )
                # Store the newly created client in options for reuse in the same process.
                options["openai_client"] = openai_client
            except Exception as e:
                logging.error(f"Failed to initialize the OpenAI client: {e}")
                return None
        try:
            # Set parameters for the API call.
            temperature = float(options.get("temperature", 0.8))
            top_p = float(options.get("top_p", 1.0))
            max_tokens = int(options.get("num_predict", 256))
            response = openai_client.chat.completions.create(
                model=options.get("openai_model"),
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                logging.error("OpenAI returned an empty list of choices.")
                return None
        except (APIConnectionError, APIStatusError, APITimeoutError, APIError, Exception) as e:
            logging.error(f"Error during OpenAI API call: {e}")
            return None
    else:
        # Use local Ollama model for inference.
        print("\n=== Model Request ===")
        print("Prompt sent:")
        print(prompt)
        print("=====================\n")
        try:
            response = generate(model_name, prompt, options=options)
            return response['response'].strip()
        except Exception as e:
            logging.error(f"Error during local model call: {e}")
            return None


def merge_json_objects(json_list):
    """
    Receives a list of JSON objects (dicts) and merges their keys.
    If the same key appears in multiple JSON objects, the last occurrence takes precedence.
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
    Function used in multiprocessing for the "multiple prompts" case.

    For each text:
      - apply each prompt successively,
      - clean/validate the response (up to 5 attempts per prompt),
      - store each partial JSON,
      - merge the JSON objects into one final JSON,
      - return the identifier, the final JSON, and the total inference time.
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

    collected_jsons = []

    for i, (base_prompt, expected_keys) in enumerate(list_of_prompts, start=1):
        instructions_sup = (
            "\nIMPORTANT:\n"
            "- Respond exclusively with a strictly valid JSON object.\n"
            "- No text/comments outside of the object.\n"
            "- The expected keys are exactly those specified (if provided).\n"
        )

        prompt_final = f"{base_prompt}\n\n{instructions_sup}\n\nText to analyze:\n{text_to_annotate}"

        max_attempts = 5
        cleaned_json_str = None

        for attempt in range(1, max_attempts + 1):
            output = analyze_text_with_model(model_name, prompt_final, options)
            if output:
                print("\n=== Raw Model Response ===")
                print(output)
                print("================================\n")

            if not output:
                logging.error(f"No model response for ID {identifier} (Prompt #{i}, attempt {attempt}).")
            else:
                cleaned_json_str = clean_json_output(output, expected_keys)
                if cleaned_json_str is not None:
                    print(f"=== Cleaned Response (Prompt #{i}) ===")
                    print(cleaned_json_str)
                    print("=====================================\n")
                    break

            logging.info(f"ID {identifier}, Prompt #{i}: Invalid or empty JSON, attempt {attempt}/{max_attempts}.")
            if attempt == 3:
                prompt_final += (
                    "\n\nREMINDER: The output must be a single JSON object, "
                    "with no text outside of this object.\nPlease adhere to the JSON structure."
                )

        if cleaned_json_str is None:
            logging.error(
                f"Failure: no valid JSON for ID {identifier}, Prompt #{i} after {max_attempts} attempts."
            )
            end_time = time.perf_counter()
            return identifier, None, (end_time - start_time)

        try:
            cleaned_json_dict = json.loads(cleaned_json_str)
        except json.JSONDecodeError:
            logging.error(f"Unable to parse ID {identifier}, Prompt #{i} despite cleaning.")
            end_time = time.perf_counter()
            return identifier, None, (end_time - start_time)

        collected_jsons.append(cleaned_json_dict)

    final_json_dict = merge_json_objects(collected_jsons)
    final_json_str = json.dumps(final_json_dict, ensure_ascii=False)

    end_time = time.perf_counter()
    inference_time = end_time - start_time
    return identifier, final_json_str, inference_time


def process_comment_single_prompt(args):
    """
    Multiprocessing function for the "single prompt" case.
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
        "- Respond exclusively with a strictly valid JSON object.\n"
        "- No text/comments outside of the object.\n"
        "- The expected keys are exactly those specified.\n"
    )
    prompt_final = f"{base_prompt}\n\n{instructions_sup}\n\nText to analyze:\n{text_to_annotate}"

    max_attempts = 5
    cleaned_json_str = None

    for attempt in range(1, max_attempts + 1):
        output = analyze_text_with_model(model_name, prompt_final, options)
        if output:
            print("\n=== Raw Model Response ===")
            print(output)
            print("================================\n")

        if not output:
            logging.error(f"No model response for ID {identifier} (attempt {attempt}).")
        else:
            cleaned_json_str = clean_json_output(output, expected_keys)
            if cleaned_json_str is not None:
                print("=== Cleaned Response (used as annotation) ===")
                print(cleaned_json_str)
                print("============================================\n")
                break

        logging.info(f"ID {identifier}: Invalid or empty JSON, attempt {attempt}/{max_attempts}.")

        if attempt == 3:
            prompt_final += (
                "\n\nREMINDER: The output must be a single JSON object, "
                "with no text outside of this object.\nPlease adhere to the JSON structure."
            )

    end_time = time.perf_counter()
    inference_time = end_time - start_time

    if cleaned_json_str is None:
        logging.error(f"Failure: no valid JSON for ID {identifier} after {max_attempts} attempts.")
        return identifier, None, inference_time

    try:
        cleaned_json_dict = json.loads(cleaned_json_str)
        final_json_str = json.dumps(cleaned_json_dict, ensure_ascii=False)
        return identifier, final_json_str, inference_time
    except json.JSONDecodeError:
        logging.error(f"Unable to parse ID {identifier} despite cleaning.")
        return identifier, None, inference_time


###############################################################################
#       Update Annotations (and Inference Time) in DB/DataFrame              #
###############################################################################

def update_annotation_db(engine, table_name, identifier_column, identifier,
                         annotation_column, annotation_json):
    """
    Updates the JSON annotation column in the PostgreSQL table, row by row.
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
            logging.error(f"Error updating ID {identifier}: {e}")

def update_inference_time_db(engine, table_name, identifier_column, identifier,
                             annotation_time_column, inference_time):
    """
    Updates the inference time column (float/double) in the PostgreSQL table, row by row.
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
            logging.error(f"Error updating inference time for ID {identifier}: {e}")

def update_annotation_df(df, identifier, identifier_column, annotation_column, annotation_json):
    """
    Updates the annotation column in the DataFrame for the given ID.
    Used for CSV, Excel, Parquet, RData, RDS.
    """
    if identifier_column not in df.columns:
        logging.error(f"ID column '{identifier_column}' not found in the DataFrame.")
        return df

    mask = (df[identifier_column] == identifier)
    if mask.any():
        df.loc[mask, annotation_column] = annotation_json
    else:
        logging.error(f"Identifier '{identifier}' not found in '{identifier_column}'.")
    return df

def update_inference_time_df(df, identifier, identifier_column,
                             annotation_time_column, inference_time):
    """
    Updates the inference time column in the DataFrame for the given ID.
    """
    if identifier_column not in df.columns:
        logging.error(f"ID column '{identifier_column}' not found in the DataFrame.")
        return df

    mask = (df[identifier_column] == identifier)
    if mask.any():
        df.loc[mask, annotation_time_column] = inference_time
    else:
        logging.error(f"Identifier '{identifier}' not found in '{identifier_column}'.")
    return df


###############################################################################
#                         Line-by-Line Saving                                 #
###############################################################################

def read_data_file(file_path, file_format):
    """
    Reads the file according to the specified format and returns a DataFrame.
    file_format must be one of: 'csv', 'excel', 'parquet', 'rdata', 'rds'.
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
            raise ImportError("The pyreadr package is required to read RData/rds.")
        result = pyreadr.read_r(file_path)
        df = list(result.values())[0]
        return df
    else:
        raise ValueError(f"Unknown or unsupported format: {file_format}")

def write_data_file(df, file_path, file_format):
    """
    Writes the DataFrame to a file according to the specified format.
    Operation performed after each annotation to avoid data loss.
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
            raise ImportError("The pyreadr package is required to write RData.")
        pyreadr.write_rdata({'data': df}, file_path)
    elif file_format == 'rds':
        if not HAS_PYREADR:
            raise ImportError("The pyreadr package is required to write RDS.")
        pyreadr.write_rds(df, file_path)
    else:
        raise ValueError(f"Unknown or unsupported format: {file_format}")


###############################################################################
#                         Global Cleaning (Optional)                          #
###############################################################################

def clear_existing_annotations_db(engine, table_name, annotation_column):
    """
    Resets the JSON annotation column in the PostgreSQL table.
    """
    clear_query = text(f"UPDATE {table_name} SET {annotation_column} = NULL;")
    with engine.begin() as connection:
        try:
            connection.execute(clear_query)
            logging.info(f"Annotations in '{annotation_column}' have been cleared.")
        except SQLAlchemyError as e:
            logging.error(f"Error during deletion: {e}")

def clear_existing_annotations_df(df, annotation_column):
    """
    Resets the annotation column in a DataFrame (CSV, Excel, etc.).
    """
    df[annotation_column] = pd.NA
    logging.info(f"Annotations in '{annotation_column}' have been cleared.")
    return df


###############################################################################
#                  List of Available Ollama Models                            #
###############################################################################

def list_ollama_models():
    """
    Uses the 'ollama list' command to retrieve the list of available models.
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error("Unable to retrieve the list of Ollama models. Error: " + result.stderr)
            return []
        
        lines = result.stdout.strip().splitlines()
        models_cleaned = []

        for line in lines:
            # Exclude header or empty lines
            if "NAME" in line and "MODIFIED" in line:
                continue
            if not line.strip():
                continue

            parts = line.split()
            if parts[0].endswith('.'):
                model_name = parts[1]
            else:
                model_name = parts[0]
            models_cleaned.append(model_name)

        return models_cleaned

    except FileNotFoundError:
        logging.error("Ollama is not installed or not found in the PATH.")
        return []


###############################################################################
#                              Main Script                                    #
###############################################################################

def main():
    print("=== Annotation Application ===\n")

    # (NEW) Ask if user wants to use the new OpenAI API client
    use_openai_api = prompt_user_yes_no("Do you want to use Open AI API ?")
    openai_api_key = None
    openai_model_name = None
    openai_client = None

    if use_openai_api:
        # We check if the new library is available
        if not HAS_NEW_OPENAI or OpenAI is None:
            print("The new OpenAI library (>=1.0.0) is not installed or not detected.")
            print("Please install with: pip install --upgrade openai")
            sys.exit(1)

        openai_api_key = input("Enter your API key: ").strip()
        if not openai_api_key:
            print("No API key provided, cannot proceed with OpenAI usage.")
            sys.exit(1)

        # Create the new OpenAI client instance
        try:
            # We can also set max_retries / timeouts if needed
            openai_client = OpenAI(
                api_key=openai_api_key,
                # You can customize these:
                max_retries=2,      # default is 2
                timeout=600.0,      # default is 600 seconds
            )
        except Exception as e:
            logging.error(f"Failed to initialize the OpenAI client: {e}")
            sys.exit(1)

        # Propose a few typical models but allow custom choice
        print("\nExemples de modèles OpenAI: gpt-3.5-turbo, gpt-4, etc.")
        model_choice = input("Which model do you want to use? (defaut: gpt-3.5-turbo) ").strip()
        if not model_choice:
            model_choice = "gpt-3.5-turbo"  # Default fallback
        openai_model_name = model_choice
        print(f"You chose the following model : {openai_model_name}\n")

    else:
        # If not using OpenAI, we will rely on local Ollama
        pass

    # If not using OpenAI, let's list local Ollama models
    if not use_openai_api:
        models_available = list_ollama_models()
        if not models_available:
            logging.error("No Ollama model was found or 'ollama list' failed. Unable to proceed.")
            sys.exit(1)

        print("\n=== Available Ollama Models ===")
        for i, m in enumerate(models_available, start=1):
            print(f"{i}. {m}")
        print("================================\n")

        model_choice_idx = prompt_user_int("Enter the number of the model to use", 1, len(models_available))
        model_name = models_available[model_choice_idx - 1]
        logging.info(f"Selected local model: {model_name}")
    else:
        # In the OpenAI case, keep model_name for consistency
        model_name = openai_model_name

    # (NEW) Check: do you want to use a single prompt or multiple prompts?
    multiple_prompts = prompt_user_yes_no("Do you want to use multiple successive prompts for each text?")

    if multiple_prompts:
        n_prompts = prompt_user_int("How many prompts do you want to use?", 2)
        list_of_prompts = []
        for i in range(1, n_prompts + 1):
            prompt_path = input(f"Full path to prompt #{i} (.txt): ").strip()
            while not os.path.isfile(prompt_path):
                print("Invalid path or file does not exist. Please try again.")
                prompt_path = input(f"Full path to prompt #{i} (.txt): ").strip()

            base_prompt_i, expected_keys_i = load_prompt(prompt_path)
            verify_prompt_structure(base_prompt_i, expected_keys_i)
            list_of_prompts.append((base_prompt_i, expected_keys_i))
    else:
        # Single prompt
        prompt_path = input("Full path to the prompt file (.txt): ").strip()
        while not os.path.isfile(prompt_path):
            print("Invalid path or file does not exist. Please try again.")
            prompt_path = input("Full path to the prompt file (.txt): ").strip()

        base_prompt, expected_keys = load_prompt(prompt_path)
        verify_prompt_structure(base_prompt, expected_keys)
        list_of_prompts = [(base_prompt, expected_keys)]

    # (NEW) Offer multiple data sources
    data_source = prompt_user_choice(
        "In what format are the data to annotate?",
        ['csv', 'excel', 'parquet', 'rdata', 'rds', 'postgresql']
    )

    # Next, we define how many parallel processes
    num_processes = prompt_user_int("How many parallel processes?", 1)
    # Then we ask user for model parameters (temp, top_p, etc.)
    options = prompt_user_model_parameters(num_processes)

    # (NEW) Store usage info for model calls.
    options["use_openai_api"] = use_openai_api
    options["openai_model"] = openai_model_name
    if use_openai_api:
        # Instead of passing the non-picklable OpenAI client, store the API key.
        options["openai_api_key"] = openai_api_key
        # Remove any existing openai_client to avoid pickling issues.
        options.pop("openai_client", None)


    ###########################################################################
    #                  CASE 1: Files (CSV, Excel, Parquet, RData/rds)         #
    ###########################################################################
    if data_source in ['csv', 'excel', 'parquet', 'rdata', 'rds']:
        file_path = input(f"Full path to the .{data_source} file: ").strip()
        while not os.path.isfile(file_path):
            print("Invalid path or file does not exist. Please try again.")
            file_path = input(f"Full path to the .{data_source} file: ").strip()

        # Read the DataFrame
        try:
            df_loaded = read_data_file(file_path, data_source)
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            sys.exit(1)

        print(f"Available columns: {', '.join(df_loaded.columns)}")

        # 1) Text column
        text_column = input("Name of the column containing the text to annotate: ").strip()
        while text_column not in df_loaded.columns:
            print("Column does not exist. Please try again.")
            text_column = input("Name of the text column: ").strip()

        # (NEW) Check if an annotation column already exists and if you want to continue using it
        resume_annotation = prompt_user_yes_no(
            "Does a column containing model annotations already exist and do you want to continue annotation in that column?"
        )
        if resume_annotation:
            annotation_column = input("Name of the existing column containing annotations: ").strip()
            while annotation_column not in df_loaded.columns:
                print("Column does not exist. Please try again.")
                annotation_column = input("Name of the existing column containing annotations: ").strip()
            identifier_column = input("Name of the previously used unique identifier column: ").strip()
            while identifier_column not in df_loaded.columns:
                print("Column does not exist. Please try again.")
                identifier_column = input("Name of the previously used unique identifier column: ").strip()
            inference_time_column = input("Name of the previously used inference time column: ").strip()
            while inference_time_column not in df_loaded.columns:
                print("Column does not exist. Please try again.")
                inference_time_column = input("Name of the previously used inference time column: ").strip()

            annotated_rows = df_loaded[annotation_column].notna().sum()
            logging.info(f"{annotated_rows} rows have already been annotated in column '{annotation_column}'.")

            full_df = df_loaded.copy()
            df_to_annotate = df_loaded[df_loaded[annotation_column].isna()].copy()
            if df_to_annotate.empty:
                print("All rows have already been annotated. No new annotation to perform.")
                sys.exit(0)
            total_comments = len(df_to_annotate)
            sample_size = total_comments
            print(f"\nAfter filtering, {total_comments} rows are available for annotation.\n")

        else:
            # 2) Option to create a unique identifier
            create_new_id = prompt_user_yes_no(
                "Do you want to create a unique identifier variable based on the text column?"
            )
            if create_new_id:
                full_df, new_id_col = create_unique_id_df(df_loaded, text_column)
                identifier_column = new_id_col
                print(f"The new column '{new_id_col}' will serve as the unique ID.")
            else:
                identifier_column = input("Name of the unique identifier column: ").strip()
                while identifier_column not in df_loaded.columns:
                    print("Column does not exist. Please try again.")
                    identifier_column = input("Name of the unique identifier column: ").strip()

            annotation_column = input("Name of the new column for annotations (JSON): ").strip()
            full_df = add_annotation_column_df(df_loaded, annotation_column)
            df_to_annotate = full_df.copy()
            total_comments = len(full_df)

            # 4) Sample size calculation
            calculate_ic = prompt_user_yes_no("Calculate sample size (95% CI)?")
            if calculate_ic:
                choice_ic = prompt_user_choice(
                    "Calculation based on row count or a unique variable?",
                    ['rows', 'variable']
                )
                if choice_ic == 'variable':
                    unique_var = input("Variable with a unique count: ").strip()
                    while unique_var not in full_df.columns:
                        print("Column does not exist. Please try again.")
                        unique_var = input("Unique variable: ").strip()
                sample_size = calculate_sample_size(total_comments)
                print(f"\nRecommended sample size: {sample_size} rows.\n")
            else:
                sample_size = total_comments
                print(f"\nTotal number of available rows: {sample_size}\n")

        # 5) How many to annotate
        num_to_annotate = prompt_user_int(
            f"How many data rows to annotate? (max {total_comments})", 1, total_comments
        )
        print(f"{num_to_annotate} rows selected.\n")

        # 6) Random selection?
        random_selection = prompt_user_yes_no(
            "Do you want to select these rows randomly?"
        )

        # (NEW) Save file path
        output_file_path = input(
            f"Save path for the annotated file (.{data_source}): "
        ).strip()
        output_file_path = os.path.abspath(output_file_path)
        parent_dir = os.path.dirname(output_file_path)
        if not os.path.exists(parent_dir):
            print(f"The directory '{parent_dir}' does not exist. Create it? (yes/no): ", end='')
            create_dir = prompt_user_yes_no("")
            if create_dir:
                try:
                    os.makedirs(parent_dir)
                    logging.info(f"Directory '{parent_dir}' created.")
                except Exception as e:
                    logging.error(f"Unable to create '{parent_dir}': {e}")
                    sys.exit(1)
            else:
                logging.error("Cannot save without a valid directory.")
                sys.exit(1)

        # 7) Select rows to annotate
        if num_to_annotate < len(df_to_annotate):
            if random_selection:
                df_subset = df_to_annotate.sample(n=num_to_annotate, random_state=42)
            else:
                df_subset = df_to_annotate.iloc[:num_to_annotate]
        else:
            df_subset = df_to_annotate

        logging.info(f"{len(df_subset)} data rows selected for annotation.")

        # 8) Prepare tasks for multiprocessing
        args_list = []
        if multiple_prompts:
            for index, row in df_subset.iterrows():
                args_list.append((
                    index, row,
                    list_of_prompts,
                    model_name,
                    options,
                    text_column,
                    identifier_column
                ))
        else:
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
        logging.info(f"Annotating {total_to_annotate} texts.")

        # 9) Parallel execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            try:
                if multiple_prompts:
                    futures = {executor.submit(process_comment_multiple_prompts, a): a[0] for a in args_list}
                else:
                    futures = {executor.submit(process_comment_single_prompt, a): a[0] for a in args_list}

                with tqdm(total=total_to_annotate, desc='Data Analysis', unit='annotation') as pbar:
                    batch_results = []
                    idx = 0

                    for future in concurrent.futures.as_completed(futures):
                        identifier, output_json, inference_time = future.result()

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
                                logging.error(f"Error during line-by-line saving: {e}")
                            batch_results.append((identifier, output_json, inference_time))

                        idx += 1
                        if idx % 10 == 0 and batch_results:
                            selected = random.choice(batch_results)
                            tqdm.write(f"\nExample annotation for ID {selected[0]}:")
                            tqdm.write(json.dumps(json.loads(selected[1]), ensure_ascii=False, indent=2))
                            tqdm.write(f"Inference time: {selected[2]:.4f} s")
                            tqdm.write("-" * 80)
                            batch_results = []

                        if idx % 500 == 0:
                            percentage = (idx / total_to_annotate) * 100
                            tqdm.write(f"Progress: {idx}/{total_to_annotate} ({percentage:.2f}%).")

                        pbar.update(1)

            except KeyboardInterrupt:
                logging.warning("Interrupted by the user.")
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)

        logging.info("Analysis completed (File).")

    ###########################################################################
    #                  CASE 2: Reading via PostgreSQL                         #
    ###########################################################################
    elif data_source == 'postgresql':
        dbname = input("Database name: ").strip()
        host = input("Database host (default 'localhost'): ").strip() or 'localhost'
        port_input = input("Database port (default 5432): ").strip()
        port = int(port_input) if port_input else 5432
        user = input("Username: ").strip()
        password = input("Password: ").strip()

        try:
            engine = connect_to_postgresql(dbname, host, port, user, password)
            logging.info("PostgreSQL connection successful.")
        except Exception as e:
            logging.error(f"PostgreSQL connection failed: {e}")
            sys.exit(1)

        table_name = input("Name of the table to annotate: ").strip()

        try:
            with engine.connect() as connection:
                _ = connection.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1;"))
            logging.info(f"Table '{table_name}' is accessible.")
        except SQLAlchemyError as e:
            logging.error(f"Error during verification: {e}")
            sys.exit(1)

        with engine.connect() as connection:
            result = connection.execute(text(
                f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}';"
            ))
            columns = [row[0] for row in result]

        print(f"Available columns in '{table_name}': {', '.join(columns)}")

        text_column = input("Name of the column containing the text to annotate: ").strip()
        while text_column not in columns:
            print("Column does not exist. Please try again.")
            text_column = input("Name of the text column: ").strip()

        resume_annotation = prompt_user_yes_no(
            "Does a column containing model annotations already exist and do you want to continue annotation in that column?"
        )
        if resume_annotation:
            annotation_column = input("Name of the existing column containing annotations (JSONB): ").strip()
            while annotation_column not in columns:
                print("Column does not exist. Please try again.")
                annotation_column = input("Name of the existing column containing annotations (JSONB): ").strip()
            identifier_column = input("Name of the previously used unique identifier column: ").strip()
            while identifier_column not in columns:
                print("Column does not exist. Please try again.")
                identifier_column = input("Name of the previously used unique identifier column: ").strip()
            annotation_time_column = input("Name of the previously used inference time column: ").strip()
            while annotation_time_column not in columns:
                print("Column does not exist. Please try again.")
                annotation_time_column = input("Name of the previously used inference time column: ").strip()

            with engine.connect() as connection:
                result_count = connection.execute(text(
                    f"SELECT COUNT(*) FROM {table_name} WHERE {annotation_column} IS NOT NULL;"
                ))
                annotated_rows = result_count.fetchone()[0]
            logging.info(f"{annotated_rows} rows have already been annotated in '{annotation_column}'.")

            with engine.connect() as connection:
                result_count = connection.execute(text(
                    f"SELECT COUNT(*) FROM {table_name} WHERE {annotation_column} IS NULL;"
                ))
                total_comments = result_count.fetchone()[0]
            if total_comments == 0:
                print("All rows have already been annotated. No new annotation to perform.")
                sys.exit(0)

        else:
            create_new_id = prompt_user_yes_no(
                "Do you want to create a unique identifier variable based on the text column?"
            )
            if create_new_id:
                from sqlalchemy import exc
                try:
                    new_id_col = create_unique_id_db(engine, table_name, text_column)
                    identifier_column = new_id_col
                    print(f"The new column '{new_id_col}' will serve as the unique ID.")
                except exc.SQLAlchemyError as e:
                    logging.error(f"Error creating the unique column: {e}")
                    sys.exit(1)
            else:
                identifier_column = input("Please specify the unique identifier column: ").strip()
                while identifier_column not in columns:
                    print("Column does not exist. Please try again.")
                    identifier_column = input("Name of the ID column: ").strip()

            annotation_column = input("Name of the column for annotations (JSONB): ").strip()
            add_annotation_column_db(engine, table_name, annotation_column)
            annotation_time_column = f"{annotation_column}_inference_time"

            with engine.connect() as connection:
                result_count = connection.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
                total_comments = result_count.fetchone()[0]

        if not resume_annotation:
            calculate_ic = prompt_user_yes_no("Calculate sample size (95% CI)?")
            if calculate_ic:
                choice_ic = prompt_user_choice(
                    "Calculation based on rows or a unique variable?",
                    ['rows', 'variable']
                )
                if choice_ic == 'variable':
                    unique_var = input("Name of the unique variable: ").strip()
                    while unique_var not in columns:
                        print("Column does not exist. Please try again.")
                        unique_var = input("Unique variable: ").strip()
                sample_size = calculate_sample_size(total_comments)
                print(f"\nRecommended sample size: {sample_size} rows.\n")
            else:
                sample_size = total_comments
                print(f"\nTotal number of available rows: {sample_size}\n")
        else:
            print(f"\nThere are {total_comments} rows available for annotation (non-annotated rows).\n")
            sample_size = total_comments

        num_to_annotate = prompt_user_int(
            f"How many data rows to annotate? (max {total_comments})", 1, total_comments
        )
        print(f"{num_to_annotate} rows selected.\n")

        random_selection = prompt_user_yes_no("Do you want to select these rows randomly?")

        output_csv_path = input("Save path (.csv) for annotated rows only: ").strip()
        output_csv_path = os.path.abspath(output_csv_path)
        parent_dir = os.path.dirname(output_csv_path)
        if not os.path.exists(parent_dir):
            print(f"The directory '{parent_dir}' does not exist. Create it? (yes/no): ", end='')
            create_dir = prompt_user_yes_no("")
            if create_dir:
                try:
                    os.makedirs(parent_dir)
                    logging.info(f"Directory '{parent_dir}' created.")
                except Exception as e:
                    logging.error(f"Unable to create '{parent_dir}': {e}")
                    sys.exit(1)
            else:
                logging.error("Cannot save without a valid directory.")
                sys.exit(1)

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
            logging.info(f"Number of rows actually selected: {len(df_to_annotate)}")

        args_list = []
        if multiple_prompts:
            for index, row in df_to_annotate.iterrows():
                args_list.append((
                    index,
                    row,
                    list_of_prompts,
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
        logging.info(f"Starting annotation of {total_to_annotate} texts.")

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            try:
                if multiple_prompts:
                    futures = {executor.submit(process_comment_multiple_prompts, a): a[0] for a in args_list}
                else:
                    futures = {executor.submit(process_comment_single_prompt, a): a[0] for a in args_list}

                with tqdm(total=total_to_annotate, desc='Data Analysis', unit='annotation') as pbar:
                    batch_results = []
                    idx = 0

                    for future in concurrent.futures.as_completed(futures):
                        identifier, output_json, inference_time = future.result()
                        if output_json is not None:
                            update_annotation_db(
                                engine,
                                table_name,
                                identifier_column,
                                identifier,
                                annotation_column,
                                output_json
                            )
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
                            pbar.write(f"\nExample annotation for ID {selected_result[0]}:")
                            pbar.write(json.dumps(json.loads(selected_result[1]), ensure_ascii=False, indent=2))
                            pbar.write(f"Inference time: {selected_result[2]:.4f} s")
                            pbar.write("-" * 80)
                            batch_results = []

                        if idx % 500 == 0:
                            percentage = (idx / total_to_annotate) * 100
                            pbar.write(f"Progress: {idx}/{total_to_annotate} ({percentage:.2f}%).")

                        pbar.update(1)

            except KeyboardInterrupt:
                logging.warning("Interrupted by the user.")
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)

        with engine.connect() as connection:
            annotated_query = text(f"""
                SELECT {identifier_column}, {text_column}, {annotation_column}, {annotation_time_column}
                FROM {table_name}
                WHERE {annotation_column} IS NOT NULL
            """)
            df_annotated = pd.read_sql_query(annotated_query, connection)

        if df_annotated.empty:
            print("\nNo rows were annotated in the database. The output file will be empty.")
        else:
            try:
                df_annotated.to_csv(output_csv_path, index=False)
                logging.info(f"Annotations saved (annotated rows only) in '{output_csv_path}'.")
            except Exception as e:
                logging.error(f"Error saving CSV: {e}")
                sys.exit(1)

        logging.info("Analysis completed (PostgreSQL).")


if __name__ == '__main__':
    main()