"""
What does the code do?
1. Text cleaning and preprocessing (tokenization, stopword & POS-based filtering)
2. Text vectorization using Bag of Words (BoW) and TF-IDF
3. Topic extraction using:
   - Latent Dirichlet Allocation (LDA)
   - Latent Semantic Analysis (LSA)
4. Coherence evaluation using Gensim
"""

# === Imports ===
import re
import time
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# === NLTK Setup ===
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# === Constants ===
DATA_PATH = "dataset/complaints_processed.csv"
NUM_TOPICS = 5
MAX_FEATURES = 5000
TOP_N_WORDS = 10
POS_TAGS_TO_KEEP = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}

# === Utility Functions ===
def clean_text(text, stop_words):
    """Cleans text: lowercase, remove special chars, tokenize, remove stopwords, and keep nouns/adjectives."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [t for t in word_tokenize(text) if t not in stop_words and len(t) > 2]
    tagged = pos_tag(tokens)
    filtered = [w for w, tag in tagged if tag in POS_TAGS_TO_KEEP]
    return filtered, " ".join(filtered)


def load_and_preprocess_data(path):
    """Loads CSV, cleans text, and returns tokenized and cleaned versions."""
    df = pd.read_csv(path, usecols=["narrative"]).dropna()
    stop_words = set(stopwords.words("english"))

    print(f"[INFO] Loaded {len(df)} complaint narratives.")
    print("[INFO] Cleaning and filtering text...")

    results = [clean_text(t, stop_words) for t in df["narrative"]]
    tokens_list, cleaned_texts = zip(*results)

    print("[INFO] Text preprocessing complete.")
    return list(tokens_list), list(cleaned_texts)


def vectorize_texts(cleaned_texts):
    """Vectorizes texts using BoW and TF-IDF."""
    print("[INFO] Vectorizing texts...")
    bow_vectorizer = CountVectorizer(max_features=MAX_FEATURES)
    tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    bow = bow_vectorizer.fit_transform(cleaned_texts)
    tfidf = tfidf_vectorizer.fit_transform(cleaned_texts)
    print(f"[INFO] BoW shape: {bow.shape} | TF-IDF shape: {tfidf.shape}")
    return bow, tfidf, bow_vectorizer, tfidf_vectorizer


def display_topics(model, feature_names, model_name, n_top_words=10):
    """Displays top words per topic and returns them as lists."""
    print(f"\n[RESULT] Top {n_top_words} words per topic ({model_name}):")
    topics = []
    for i, topic in enumerate(model.components_):
        top_words = [feature_names[j] for j in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"  Topic {i+1}: {' | '.join(top_words)}")
        topics.append(top_words)
    return topics


def calculate_coherence(topics, tokens_list, model_name="Model"):
    """Calculates Gensim C_v coherence score."""
    dictionary = Dictionary(tokens_list)
    coherence_model = CoherenceModel(topics=topics, texts=tokens_list,
                                     dictionary=dictionary, coherence="c_v")
    score = coherence_model.get_coherence()
    print(f"[METRIC] Coherence Score (C_v) for {model_name}: {score:.4f}")
    return score


def run_topic_modeling():
    """Main function: load data, vectorize, model topics, and evaluate coherence."""
    start = time.time()
    print("[START] Topic Modeling Pipeline initiated.\n")

    tokens_list, cleaned_texts = load_and_preprocess_data(DATA_PATH)
    bow, tfidf, bow_vectorizer, tfidf_vectorizer = vectorize_texts(cleaned_texts)

    print("\n[INFO] Training LDA (BoW)...")
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)
    lda.fit(bow)
    lda_topics = display_topics(lda, bow_vectorizer.get_feature_names_out(), "LDA (BoW)", TOP_N_WORDS)
    calculate_coherence(lda_topics, tokens_list, "LDA (BoW)")

    print("\n[INFO] Training LSA (TF-IDF)...")
    lsa = TruncatedSVD(n_components=NUM_TOPICS, random_state=42)
    lsa.fit(tfidf)
    lsa_topics = display_topics(lsa, tfidf_vectorizer.get_feature_names_out(), "LSA (TF-IDF)", TOP_N_WORDS)
    calculate_coherence(lsa_topics, tokens_list, "LSA (TF-IDF)")

    print(f"\n[END] Pipeline completed in {time.time() - start:.2f} seconds ✅")


if __name__ == "__main__":
    try:
        run_topic_modeling()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
