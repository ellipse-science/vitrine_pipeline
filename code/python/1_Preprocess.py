"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
1_Preprocess.py

MAIN OBJECTIVE:
---------------
Preprocess the media database CSV by loading data, generating sentence contexts in parallel,
and saving the processed data. No date conversion or word recounting is required this time.

Dependencies:
-------------
- os
- pandas
- spacy
- tqdm
- joblib

MAIN FEATURES:
-------------
1) Load and preprocess CSV data.
2) Remove any existing 'doc_id' and create a new one.
3) Tokenize text into 2-sentence contexts in parallel.
4) Store processed data into a new CSV file.

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

# ----------------------------------------------------
# Instead of global spaCy model loading here,
# we define a helper function to load models on demand
# within each worker process.
# ----------------------------------------------------
def get_nlp(language):
    """
    Lazily load and return the spaCy model corresponding to the language.
    This ensures that each worker process has its own instance, avoiding
    shared read-only memory issues.
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

    Parameters:
    ----------
    text : str
        The text to tokenize.
    language : str
        Language code ('FR' or otherwise 'EN').

    Returns:
    -------
    list
        A list of sentence contexts with up to two sentences joined together.
    """
    nlp = get_nlp(language)  # Use the lazy-loaded model per process
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    contexts = []
    for i in range(len(sentences)):
        if i > 0:  # Ensure there is a previous sentence
            context = ' '.join(sentences[i-1:i+1])
            contexts.append(context)
        else:
            contexts.append(sentences[i])
    return contexts if contexts else ['']

def process_chunk(df_chunk, start_doc_id):
    """
    Process a chunk of the DataFrame by:
      - Assigning a new doc_id for each article.
      - Tokenizing text into up to two-sentence contexts.
      - Building a list of processed items.

    Parameters:
    ----------
    df_chunk : pd.DataFrame
        A slice of the main DataFrame.
    start_doc_id : int
        Starting document ID for this chunk to ensure unique IDs overall.

    Returns:
    -------
    list
        A list of dictionaries containing processed rows (split by sentence contexts).
    int
        The last doc_id used (for chaining to next chunk).
    """
    processed_texts = []
    current_doc_id = start_doc_id

    for _, row in df_chunk.iterrows():
        current_doc_id += 1  # Increment ID for each article
        contexts = tokenize_and_context(row['text'], row['language'])

        for sentence_id, context in enumerate(contexts):
            processed_texts.append({
                # New doc_id
                'doc_id': current_doc_id,
                'news_type': row['news_type'],
                'title': row['title'],
                'author': row['author'],
                'media': row['media'] if pd.notna(row['media']) else 'media : not provided',
                'words_count': row['words_count'],  # We keep the original words_count
                'date': row['date'],
                'language': row['language'],
                'page_number': row['page_number'],
                'sentences': context
            })
    
    return processed_texts, current_doc_id

def split_dataframe(df, n_splits):
    """
    Split a DataFrame into n_splits roughly equal chunks.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame to be split.
    n_splits : int
        Number of parts to split into.

    Returns:
    -------
    list of pd.DataFrame
        A list of DataFrame chunks.
    """
    chunk_size = len(df) // n_splits
    chunks = []
    for i in range(n_splits):
        start_i = i * chunk_size
        # For the last chunk, take everything remaining
        end_i = len(df) if i == (n_splits - 1) else (i + 1) * chunk_size
        chunk = df.iloc[start_i:end_i]
        chunks.append(chunk)
    return chunks

# ---------------------------------------------
# Main section of the script
# ---------------------------------------------
if __name__ == "__main__":

    # Relative path to the folder containing the script
    script_dir = os.path.dirname(__file__)

    # Relative path to the CSV file in the Database folder
    csv_path = os.path.join(script_dir, '..', '..', 'Database', 'Database', 'CCF.media_database.csv')

    # ---------------------------
    # Step 1: Load the CSV file
    # ---------------------------
    print("Loading CSV file...")
    with tqdm(total=1, desc="Loading CSV file") as pbar:
        df = pd.read_csv(csv_path)
        pbar.update(1)

    # ----------------------------------------------------------------
    # Step 2: Remove any existing doc_id column and prepare DataFrame
    # ----------------------------------------------------------------
    if 'doc_id' in df.columns:
        print("Dropping existing 'doc_id' column...")
        df.drop(columns=['doc_id'], inplace=True)

    # -----------------------------------------------
    # Step 3: Parallel tokenization & new doc_id
    # -----------------------------------------------
    print("Tokenizing texts and building two-sentence contexts in parallel...")

    # Determine number of cores (note: on Mac with a M2 Ultra, os.cpu_count() can be high)
    n_cores = os.cpu_count()  # Alternatively, force a specific number if besoin: n_cores = 8
    print(f"Using {n_cores} CPU cores for parallel processing.")

    # Split DataFrame into chunks
    df_chunks = split_dataframe(df, n_cores)

    # We'll keep track of the last doc_id used for each chunk
    # Start from 0 so the first chunk's doc_id will begin at 1
    start_doc_id = 0

    # Prepare arguments for each chunk to pass to joblib.Parallel
    tasks = []
    for chunk in df_chunks:
        tasks.append((chunk, start_doc_id))
        # Estimate next start_doc_id
        start_doc_id += len(chunk)

    # Run parallel processing using delayed
    # Each item in tasks is (df_chunk, chunk_start_doc_id)
    results = Parallel(n_jobs=n_cores)(
        delayed(process_chunk)(t[0], t[1]) for t in tqdm(tasks, desc="Processing chunks")
    )

    # Combine all processed results into a single list
    # Each element of 'results' is (processed_texts, last_doc_id_for_chunk)
    all_processed = []
    for processed_texts, _ in results:
        all_processed.extend(processed_texts)

    # Create a new DataFrame from the processed data
    processed_df = pd.DataFrame(all_processed)

    # ------------------------------------------------
    # Step 4: Save the new DataFrame to CSV
    # ------------------------------------------------
    output_path = os.path.join(script_dir, '..', '..', 'Database', 'Database', 'CCF.media_processed_texts.csv')
    print("Saving the processed DataFrame to CSV...")
    with tqdm(total=1, desc="Saving to CSV") as pbar:
        processed_df.to_csv(output_path, index=False, header=True)
        pbar.update(1)

    print("Processing and saving completed.")