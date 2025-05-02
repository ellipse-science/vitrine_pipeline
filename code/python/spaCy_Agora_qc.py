import pandas as pd
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Load spaCy model
print("Loading spaCy model...")
# nlp = spacy.load("en_core_web_lg") # English sentences
nlp = spacy.load("fr_dep_news_trf") # French sentences
print("spaCy model loaded.")

# Load your CSV file
print("Loading CSV file...")
df = pd.read_csv("/Users/shannondinan/Library/CloudStorage/Dropbox/RESEARCH/_COLLABORATIONS/_GitHub/depFiscales/_SharedFolder_depFiscales/data/donnee_1995-2025.csv")
print(f"CSV loaded successfully. {len(df)} rows.")

# Function to tokenize sentences
def tokenize_sentences(text):
    if pd.isnull(text):  # Handle missing values safely
        return []
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# Apply to your column with a progress bar
print("Starting sentence tokenization...")
tqdm.pandas(desc="Tokenizing sentences")
df["sentences"] = df["intervention_text"].progress_apply(tokenize_sentences)

# Check the first few tokenized sentences to confirm
print("Tokenization complete. Here's a preview of the first few rows:")
print(df["sentences"].head())

# Save the tokenized data to a new CSV
output_file = "/Users/shannondinan/Library/CloudStorage/Dropbox/RESEARCH/_COLLABORATIONS/_GitHub/depFiscales/_SharedFolder_depFiscales/data/donnée_1995-2025_tokenized.csv"
print(f"Saving the tokenized data to {output_file}...")
df.to_csv(output_file, index=False)
print("File saved successfully.")

# Number of sentences per intervention
sentence_counts = df["sentences"].apply(len)

# Plot a histogram
plt.figure(figsize=(10, 6))
sns.histplot(sentence_counts, bins=30, kde=True, color="skyblue")
plt.title("Distribution of Sentences per Intervention")
plt.xlabel("Number of Sentences")
plt.ylabel("Number of Interventions")
plt.grid(True)
plt.show()