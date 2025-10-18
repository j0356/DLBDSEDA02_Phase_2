"""
What does the code do?
1. Text cleaning and preprocessing (tokenization, stopword & POS-based filtering)
2. Text vectorization using Bag of Words (BoW) and TF-IDF
3. Topic extraction using:
   - Latent Dirichlet Allocation (LDA)
   - Latent Semantic Analysis (LSA)
4. Coherence evaluation using Gensim
5. Find optimal number of topics
6. Visualize topics and keywords
"""

# === Imports ===
import re
import time
import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt
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
OUTPUT_DIR = "output"
NUM_TOPICS_RANGE = range(2, 5)  # Test 2 to 4 topics
MAX_FEATURES = 5000
TOP_N_WORDS = 5
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


def find_optimal_topics(bow, tfidf, bow_vectorizer, tfidf_vectorizer, tokens_list):
    """Tests different numbers of topics and finds optimal based on coherence."""
    print("\n[INFO] Testing different numbers of topics...")
    
    lda_scores = []
    lsa_scores = []
    
    for n_topics in NUM_TOPICS_RANGE:
        # LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
        lda.fit(bow)
        lda_topics = [
            [bow_vectorizer.get_feature_names_out()[j] 
             for j in topic.argsort()[:-TOP_N_WORDS - 1:-1]]
            for topic in lda.components_
        ]
        lda_scores.append(calculate_coherence(lda_topics, tokens_list, f"LDA-{n_topics}"))
        
        # LSA
        lsa = TruncatedSVD(n_components=n_topics, random_state=42)
        lsa.fit(tfidf)
        lsa_topics = [
            [tfidf_vectorizer.get_feature_names_out()[j] 
             for j in topic.argsort()[:-TOP_N_WORDS - 1:-1]]
            for topic in lsa.components_
        ]
        lsa_scores.append(calculate_coherence(lsa_topics, tokens_list, f"LSA-{n_topics}"))
    
    optimal_lda = NUM_TOPICS_RANGE[lda_scores.index(max(lda_scores))]
    optimal_lsa = NUM_TOPICS_RANGE[lsa_scores.index(max(lsa_scores))]
    
    print(f"\n[OPTIMAL] Best number of topics for LDA: {optimal_lda}")
    print(f"[OPTIMAL] Best number of topics for LSA: {optimal_lsa}")
    
    return lda_scores, lsa_scores, optimal_lda, optimal_lsa


def plot_coherence(lda_scores, lsa_scores):
    """Plots coherence scores for different numbers of topics."""
    plt.figure(figsize=(10, 6))
    plt.plot(NUM_TOPICS_RANGE, lda_scores, marker='o', label='LDA (BoW)', linewidth=2)
    plt.plot(NUM_TOPICS_RANGE, lsa_scores, marker='s', label='LSA (TF-IDF)', linewidth=2)
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score (C_v)')
    plt.title('Coherence Score vs Number of Topics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'coherence_scores.png')
    plt.savefig(filepath, dpi=150)
    print(f"\n[SAVED] Coherence plot saved as '{filepath}'")


def visualize_topics(model, feature_names, model_name, n_topics):
    """Creates bar chart visualization of top words per topic."""
    fig, axes = plt.subplots(n_topics, 1, figsize=(10, 3 * n_topics))
    if n_topics == 1:
        axes = [axes]
    
    for i, (topic, ax) in enumerate(zip(model.components_, axes)):
        top_indices = topic.argsort()[:-TOP_N_WORDS - 1:-1]
        top_words = [feature_names[j] for j in top_indices]
        top_weights = topic[top_indices]
        
        ax.barh(range(len(top_words)), top_weights)
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words)
        ax.invert_yaxis()
        ax.set_xlabel('Weight')
        ax.set_title(f'Topic {i+1}')
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'Topic Keywords ({model_name})', fontsize=14, y=1.00)
    plt.tight_layout()
    filename = f'topics_{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150)
    print(f"[SAVED] Topic visualization saved as '{filepath}'")


def run_topic_modeling():
    """Main function: load data, vectorize, model topics, and evaluate coherence."""
    start = time.time()
    print("[START] Topic Modeling Pipeline initiated.\n")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokens_list, cleaned_texts = load_and_preprocess_data(DATA_PATH)
    bow, tfidf, bow_vectorizer, tfidf_vectorizer = vectorize_texts(cleaned_texts)

    # Find optimal number of topics
    lda_scores, lsa_scores, optimal_lda, optimal_lsa = find_optimal_topics(
        bow, tfidf, bow_vectorizer, tfidf_vectorizer, tokens_list
    )
    
    # Plot coherence scores
    plot_coherence(lda_scores, lsa_scores)

    # Train final models with optimal topics
    print("\n[INFO] Training LDA with optimal topics...")
    lda = LatentDirichletAllocation(n_components=optimal_lda, random_state=42)
    lda.fit(bow)
    lda_topics = display_topics(lda, bow_vectorizer.get_feature_names_out(), "LDA (BoW)", TOP_N_WORDS)
    calculate_coherence(lda_topics, tokens_list, "LDA (BoW)")
    visualize_topics(lda, bow_vectorizer.get_feature_names_out(), "LDA BoW", optimal_lda)

    print("\n[INFO] Training LSA with optimal topics...")
    lsa = TruncatedSVD(n_components=optimal_lsa, random_state=42)
    lsa.fit(tfidf)
    lsa_topics = display_topics(lsa, tfidf_vectorizer.get_feature_names_out(), "LSA (TF-IDF)", TOP_N_WORDS)
    calculate_coherence(lsa_topics, tokens_list, "LSA (TF-IDF)")
    visualize_topics(lsa, tfidf_vectorizer.get_feature_names_out(), "LSA TF-IDF", optimal_lsa)

    print(f"\n[END] Pipeline completed in {time.time() - start:.2f} seconds ✅")


if __name__ == "__main__":
    try:
        run_topic_modeling()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")