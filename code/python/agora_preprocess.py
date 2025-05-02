"""
PROJECT:
-------
Vitrine Pipeline

TITLE:
------
agora_preprocess.py

MAIN OBJECTIVE:
---------------
Preprocess legislative and media database CSV by loading data, generating sentence contexts in parallel,
and saving the processed data. No date conversion or word recounting is required this time.

Dependencies:
-------------
- os
- pandas
- spacy
- tqdm
- joblib
- matplotlib

MAIN FEATURES:
-------------
1) Load and preprocess CSV data.
2) Remove any existing 'doc_id' and create a new one.
3) Tokenize text into one- or two-sentence contexts in parallel.
4) Store processed data into a new CSV file.

Author:
-------
Antoine Lemor & Shannon Dinan
"""

# ----------------------------------------------------
# Import necessary libraries
# ----------------------------------------------------
import os
import pandas as pd
import spacy
from tqdm import tqdm 
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Lazy-loading of spaCy models: load each model on demand in a worker process
# ----------------------------------------------------
def get_nlp(language):
    """
    Lazily load and return the spaCy model corresponding to the language.
    This ensures that each worker process has its own instance.
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

# ----------------------------------------------------
# Tokenize and build one- or two-sentence contexts
# ----------------------------------------------------
def tokenize_and_context(text, language):
    """
    Tokenize the text and build up to two-sentence contexts.

    Parameters:
    -----------
    text : str
        The text to tokenize.
    language : str
        Language code (e.g. 'FR' or 'EN').

    Returns:
    --------
    list
        A list of contexts composed of one or two joined sentences.
    """
    nlp = get_nlp(language)  # Ensure lazy loading
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

# ----------------------------------------------------
# Process a chunk of the DataFrame: tokenization and new doc_id assignment
# ----------------------------------------------------
def process_chunk(df_chunk, start_doc_id, language):
    """
    Process a chunk of the DataFrame:
      - Assign a new doc_id for each article.
      - Tokenize text into up to two-sentence contexts.
      - Build a list of processed items.

    Parameters:
    -----------
    df_chunk : pd.DataFrame
        A slice of the DataFrame.
    start_doc_id : int
        The starting document ID for this chunk.
    language : str
        The language code.

    Returns:
    --------
    list
        A list of dictionaries containing processed rows (one per context).
    int
        The last doc_id used.
    """
    processed_texts = []
    current_doc_id = start_doc_id

    for _, row in df_chunk.iterrows():
        current_doc_id += 1  # New doc_id for each article
        contexts = tokenize_and_context(row['text'], language)
        # For each context, we assign the same doc_id (could be extended to include a context index)
        for context in contexts:
            processed_texts.append({
                'doc_id': current_doc_id,
                'news_type': row.get('news_type', None),
                'title': row.get('title', None),
                'author': row.get('author', None) if pd.notna(row.get('author', None)) else 'not provided',
                'media': row.get('media', None) if pd.notna(row.get('media', None)) else 'not provided',
                'words_count': row.get('words_count', None),
                'date': row.get('date', None),
                'language': row.get('language', language),
                'page_number': row.get('page_number', None),
                'sentences': context
            })
    return processed_texts, current_doc_id

# ----------------------------------------------------
# Split DataFrame into roughly equal chunks for parallel processing.
# ----------------------------------------------------
def split_dataframe(df, n_splits):
    """
    Split a DataFrame into n_splits roughly equal chunks.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to split.
    n_splits : int
        Number of desired chunks.

    Returns:
    --------
    list of pd.DataFrame
        List of DataFrame chunks.
    """
    chunk_size = len(df) // n_splits
    chunks = []
    for i in range(n_splits):
        start_i = i * chunk_size
        end_i = len(df) if i == (n_splits - 1) else (i + 1) * chunk_size
        chunks.append(df.iloc[start_i:end_i])
    return chunks

# ----------------------------------------------------
# Create contexts DataFrame from lists or a full DataFrame.
# ----------------------------------------------------
def create_contexts_df(
    texts=None,
    ids=None,
    language="FR",
    author_ids=None,
    media=None,
    spacy_model=None,
    n_jobs=-2,
    input_df=None,              # New: Optionally pass a DataFrame directly
    text_column="intervention_text",  # Column name for the text
    id_column="intervention_id",      # Column name for unique IDs
    author_column="author",           # Column name for authors
    media_column="media"              # Column name for media
):
    """
    Create a DataFrame of contexts (1-2 sentence blocks) from input texts.
    This function can accept either:
      - Lists (texts, ids, etc.), or
      - A full DataFrame via `input_df`, along with column names.
      
    Parameters:
    -----------
    texts : list of str
        List of texts to process.
    ids : list
        List of document IDs corresponding to texts (optional).
    language : str
        Language code ("FR" or "EN").
    author_ids : list
        List of authors (optional).
    media : list
        List of media names (optional).
    spacy_model : spacy language model
        A preloaded spaCy language model.
    n_jobs : int
        Number of CPU cores to use (-2 means all except one).
    input_df : pd.DataFrame, optional
        If provided, the function will extract texts and other fields from this DataFrame.
    text_column : str
        Column name for text in input_df.
    id_column : str
        Column name for unique IDs in input_df.
    author_column : str
        Column name for author data in input_df.
    media_column : str
        Column name for media data in input_df.

    Returns:
    --------
    pd.DataFrame
        A DataFrame of tokenized contexts with new document IDs.
    """

    # If a DataFrame is provided, extract needed columns as lists:
    if input_df is not None:
        texts = input_df[text_column].tolist()
        ids = input_df[id_column].tolist() if id_column in input_df.columns else None
        author_ids = input_df[author_column].tolist() if author_column in input_df.columns else None
        media = input_df[media_column].tolist() if media_column in input_df.columns else None
        # Create a unified DataFrame with minimal required fields.
        # We expect at least a text column; add language info as well.
        input_df = input_df.copy()
        if 'text' not in input_df.columns:
            input_df.rename(columns={text_column: 'text'}, inplace=True)
        if 'language' not in input_df.columns:
            input_df['language'] = language
    else:
        # If lists are provided, create a DataFrame.
        input_df = pd.DataFrame({
            'text': texts,
            'language': language
        })
        if ids is not None:
            input_df[id_column] = ids
        if author_ids is not None:
            input_df[author_column] = author_ids
        if media is not None:
            input_df[media_column] = media

    # Determine number of parallel jobs
    n_cores = n_jobs if n_jobs != -2 else (os.cpu_count() - 1)
    if n_cores < 1:
        n_cores = 1

    # Split DataFrame into chunks for parallel processing
    df_chunks = split_dataframe(input_df, n_cores)

    start_doc_id = 0  # Starting doc_id
    tasks = []
    for chunk in df_chunks:
        tasks.append((chunk, start_doc_id))
        start_doc_id += len(chunk)

    # Process chunks in parallel
    results = Parallel(n_jobs=n_cores)(
        delayed(process_chunk)(t[0], t[1], language) for t in tasks
    )

    # Combine results from each chunk
    all_processed = []
    for processed_texts, _ in results:
        all_processed.extend(processed_texts)

    # Return the processed data as a DataFrame
    contexts_df = pd.DataFrame(all_processed)
    return contexts_df

# ----------------------------------------------------
# Plotting function: Distribution of contexts per document
# ----------------------------------------------------
def plot_sentence_distribution(contexts_df):
    """
    Plot the distribution of number of contexts per document.
    
    Parameters:
    -----------
    contexts_df : pd.DataFrame
        The DataFrame containing a 'doc_id' column.
    """
    counts = contexts_df.groupby('doc_id').size()

    plt.figure(figsize=(10, 6))
    counts.hist(bins=50)
    plt.xlabel('Number of contexts per document')
    plt.ylabel('Frequency')
    plt.title('Distribution of Contexts per Document')
    plt.grid(True)
    plt.show()

# ----------------------------------------------------
# (Optional) Main section for command-line testing
# ----------------------------------------------------
if __name__ == "__main__":
    # Example usage: adapt paths and parameters as needed
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess CSV file into sentence contexts.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the processed CSV.")
    parser.add_argument("--language", type=str, default="FR", help="Language code ('FR' or 'EN').")
    parser.add_argument("--n_jobs", type=int, default=-2, help="Number of CPU cores to use (-2 means all except one).")
    parser.add_argument("--text_column", type=str, default="intervention_text", help="Column name for text.")
    parser.add_argument("--id_column", type=str, default="intervention_id", help="Column name for unique IDs.")
    parser.add_argument("--author_column", type=str, default="author", help="Column name for author.")
    parser.add_argument("--media_column", type=str, default="media", help="Column name for media.")
    args = parser.parse_args()

    # Load the input CSV
    print("Loading CSV file...")
    df = pd.read_csv(args.input_csv)

    # Process the DataFrame into contexts
    print("Processing texts into contexts...")
    contexts_df = create_contexts_df(
        input_df=df,
        language=args.language,
        n_jobs=args.n_jobs,
        text_column=args.text_column,
        id_column=args.id_column,
        author_column=args.author_column,
        media_column=args.media_column
    )

    # Save the processed DataFrame
    print(f"Saving processed data to {args.output_csv}...")
    contexts_df.to_csv(args.output_csv, index=False)
    
    # Plot sentence distribution
    print("Plotting distribution of contexts per document...")
    plot_sentence_distribution(contexts_df)

    print("Processing and plotting completed.")