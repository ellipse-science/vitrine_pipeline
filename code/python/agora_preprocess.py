"""
PROJECT:
-------
CCF-Canadian-Climate-Framing moded for vitrine Agora

TITLE:
------
1_Preprocess.py

MAIN OBJECTIVE:
---------------
Preprocess the media database CSV by loading data, generating sentence contexts in parallel,
and saving the processed data.

Author:
-------
Antoine Lemor
"""

# ----------------------------------------------------
# Import necessary libraries
# ----------------------------------------------------
import os
import pandas as pd
import spacy
from tqdm import tqdm 
from joblib import Parallel, delayed
import spacy


# ----------------------------------------------------
# Instead of global spaCy model loading here,
# we define a helper function to load models on demand
# within each worker process.
# ----------------------------------------------------
def get_nlp(language):
    """
    Lazily load and return the spaCy model corresponding to the language.
    """
    global _nlp_models
    try:
        _nlp_models
    except NameError:
        _nlp_models = {}
    
    language = language.upper()
    if language == 'FR':
        if 'FR' not in _nlp_models:
            _nlp_models['FR'] = spacy.load('fr_dep_news_trf')
        return _nlp_models['FR']
    else:
        if 'EN' not in _nlp_models:
            _nlp_models['EN'] = spacy.load('en_core_web_lg')
        return _nlp_models['EN']

def tokenize_and_context(text, language):
    """
    Tokenize the text and build two-sentence contexts.
    """
    nlp = get_nlp(language)
    if not isinstance(text, str):
        return ['']  # Handle missing or non-string texts safely
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    contexts = []
    for i in range(len(sentences)):
        if i > 0:
            context = ' '.join(sentences[i-1:i+1])
            contexts.append(context)
        else:
            contexts.append(sentences[i])
    return contexts if contexts else ['']

def process_chunk(df_chunk, start_doc_id):
    """
    Process a chunk of the DataFrame: assign doc_id, tokenize, and split into contexts.
    """
    processed_texts = []
    current_doc_id = start_doc_id

    for _, row in df_chunk.iterrows():
        current_doc_id += 1
        contexts = tokenize_and_context(row['text'], row['language'])

        for sentence_id, context in enumerate(contexts):
            processed_texts.append({
                 'doc_id': current_doc_id,
                'author': row.get('speaker_full_name', ''),
                 'date': row.get('event_date', ''),
                'object_of_business_title': row.get('object_of_business_title',''),
                'subject_of_business_title': row.get('subject_of_business_title',''),
                'language': row.get('language', ''),
                'sentences': context
             })
    
    return processed_texts, current_doc_id

def split_dataframe(df, n_splits):
    """
    Split a DataFrame into n_splits roughly equal chunks.
    """
    chunk_size = len(df) // n_splits
    chunks = []
    for i in range(n_splits):
        start_i = i * chunk_size
        end_i = len(df) if i == (n_splits - 1) else (i + 1) * chunk_size
        chunk = df.iloc[start_i:end_i]
        chunks.append(chunk)
    return chunks

# ---------------------------------------------
# Main section of the script
# ---------------------------------------------
if __name__ == "__main__":

    # Manually specify the input CSV path
    csv_path = "/Users/shannondinan/Library/CloudStorage/Dropbox/RESEARCH/_COLLABORATIONS/_GitHub/depFiscales/_SharedFolder_depFiscales/data/donnee_1995-2025_for_spaCy.csv"

    # Automatically define the output CSV path
    output_path = os.path.splitext(csv_path)[0] + "_processed.csv"

    # ---------------------------
    # Step 1: Load the CSV file
    # ---------------------------
    print(f"Loading CSV file: {csv_path}")
    with tqdm(total=1, desc="Loading CSV file") as pbar:
        df = pd.read_csv(csv_path)
        pbar.update(1)

    # ----------------------------------------------------------------
    # Step 2: Remove any existing doc_id column and prepare DataFrame
    # ----------------------------------------------------------------
    print(f"Removing any existing doc_id column and preparing DataFrame")
    with tqdm(total=1, desc="Cleaning DataFrame") as pbar:
        if 'doc_id' in df.columns:
            print("Dropping existing 'doc_id' column...")
            df.drop(columns=['doc_id'], inplace=True)
        pbar.update(1)

# ----------------------------------------------------------------
# Step 2: Remove any existing doc_id column and prepare DataFrame
# ----------------------------------------------------------------
print(f"Removing any existing doc_id column and preparing DataFrame")
with tqdm(total=1, desc="Cleaning DataFrame") as pbar:
    if 'doc_id' in df.columns:
        print("Dropping existing 'doc_id' column...")
        df.drop(columns=['doc_id'], inplace=True)
    pbar.update(1)

# -----------------------------------------------
# Step 3: Parallel tokenization & new doc_id
# -----------------------------------------------
print("Tokenizing texts and building two-sentence contexts in parallel...")

n_cores = min(8, os.cpu_count())  # Use up to 8 cores to be safe
print(f"Using {n_cores} CPU cores for parallel processing.")

df_chunks = split_dataframe(df, n_cores)

start_doc_id = 0
tasks = []
for chunk in df_chunks:
    tasks.append((chunk, start_doc_id))
    start_doc_id += len(chunk)

with tqdm(total=len(tasks), desc="Processing chunks") as pbar:
    results = []
    for result in Parallel(n_jobs=n_cores)(
        delayed(process_chunk)(t[0], t[1]) for t in tasks
    ):
        results.append(result)
        pbar.update(1)

all_processed = []
with tqdm(total=len(results), desc="Compiling results") as pbar:
    for processed_texts, _ in results:
        all_processed.extend(processed_texts)
        pbar.update(1)

with tqdm(total=1, desc="Creating DataFrame") as pbar:
    processed_df = pd.DataFrame(all_processed)
    pbar.update(1)

    # ------------------------------------------------
    # Step 4: Save the new DataFrame to CSV
    # ------------------------------------------------
    print(f"Saving the processed DataFrame to CSV: {output_path}")
    with tqdm(total=1, desc="Saving to CSV") as pbar:
        processed_df.to_csv(output_path, index=False, header=True)
        pbar.update(1)

    print("✅ Processing and saving completed successfully.")
