"""
This script performs:
1. Text cleaning and preprocessing (tokenization, stopword & special character removal)
2. Text vectorization using Bag of Words (BoW) and TF-IDF
3. Topic extraction using:
   - Latent Dirichlet Allocation (LDA)
   - Latent Semantic Analysis (LSA)
"""

# === Import Libraries ===
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

# Download necessary NLTK data
nltk.download('punkt') # Tokenizer
nltk.download('punkt_tab') # Had to add this line to fix a known issue with some NLTK installations
nltk.download('stopwords') # Downlaods a list of common stopwords

# === Step 1: Load Dataset ===
# Defined the dataset path
DATA_PATH = "dataset/complaints_processed.csv"

# The dataset contains columns: '#', 'product', 'narrative'
df = pd.read_csv(DATA_PATH)

# Keep only the 'narrative' column and drop missing values
texts = df['narrative'].dropna().tolist()

print(f"Loaded {len(texts)} complaint narratives.")
print("Dataset Loading ✅")

# === Step 2: Text Cleaning Function ===
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    # Join tokens back into a single string
    return " ".join(tokens)

# Clean all narratives
cleaned_texts = [clean_text(t) for t in texts]
print("Text Cleaning ✅")

# === Step 3: Vectorization ===

# Bag of Words
bow_vectorizer = CountVectorizer(max_features=5000)
bow = bow_vectorizer.fit_transform(cleaned_texts)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf = tfidf_vectorizer.fit_transform(cleaned_texts)

print("Vectorization ✅")
print(f" - BoW matrix shape: {bow.shape}")
print(f" - TF-IDF matrix shape: {tfidf.shape}")

# === Step 4: Topic Modeling ===
NUM_TOPICS = 5  # Number of topics to extract

# --- LDA using Bag of Words ---
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)
lda.fit(bow)

# --- LSA using TF-IDF ---
lsa = TruncatedSVD(n_components=NUM_TOPICS, random_state=42)
lsa.fit(tfidf)
print("Topic Modeling ✅")

# === Step 5: Display Topics ===
def display_topics(model, feature_names, n_top_words=10, model_name="Model"):
    print(f"\n--- Top {n_top_words} words per topic ({model_name}) ---")
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"Topic {topic_idx + 1}: {' | '.join(top_words)}")

# Display topics for LDA (BoW)
display_topics(lda, bow_vectorizer.get_feature_names_out(), model_name="LDA (BoW)")

# Display topics for LSA (TF-IDF)
display_topics(lsa, tfidf_vectorizer.get_feature_names_out(), model_name="LSA (TF-IDF)")
