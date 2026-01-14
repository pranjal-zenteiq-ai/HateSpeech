import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
from wordcloud import WordCloud
import numpy as np


def load_and_clean(path="HateSpeechDataset.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Content", "Label"])
    # Convert to numeric and remove non-binary labels
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"])
    df["Label"] = df["Label"].astype(int)
    df = df[df["Label"].isin([0, 1])]
    if "Content_int" in df.columns:
        df = df.drop(columns=["Content_int"])
    return df


def plot_class_distribution(df):
    counts = df["Label"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color=["#2ecc71", "#e74c3c"])
    plt.xticks([0, 1], ["Clean", "Hate"], rotation=0)
    plt.title("Class Distribution", fontsize=14, fontweight="bold")
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: class_distribution.png")
    plt.show()


def plot_text_length_distribution(df):
    df = df.copy()
    df["len"] = df["Content"].astype(str).apply(len)
    plt.figure(figsize=(10, 5))
    plt.hist(df[df["Label"] == 0]["len"], bins=50, alpha=0.7, label="Clean", color="#2ecc71")
    plt.hist(df[df["Label"] == 1]["len"], bins=50, alpha=0.7, label="Hate", color="#e74c3c")
    plt.legend(fontsize=11)
    plt.title("Text Length Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Characters", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig("text_length_distribution.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: text_length_distribution.png")
    plt.show()


def plot_top_words(df, label, top_n=20):
    """Plot top words using frequency (more interpretable than raw TF-IDF)."""
    texts = df[df["Label"] == label]["Content"]
    
    # Use CountVectorizer for raw frequency (more interpretable)
    count_vec = CountVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 1),
        min_df=5,  # Ignore words appearing in fewer than 5 documents
        max_df=0.8  # Ignore very common words
    )
    
    X_count = count_vec.fit_transform(texts)
    words = count_vec.get_feature_names_out()
    
    # Raw frequency scores (more direct interpretation)
    freq_scores = X_count.sum(axis=0).A1
    top_idx = freq_scores.argsort()[-top_n:][::-1]
    
    top_words = [words[i] for i in top_idx]
    top_scores = freq_scores[top_idx]
    
    label_name = "Hate Speech" if label == 1 else "Clean Text"
    color = "#e74c3c" if label == 1 else "#2ecc71"
    
    plt.figure(figsize=(11, 6))
    plt.barh(top_words[::-1], top_scores[::-1], color=color, alpha=0.8)
    plt.title(f"Top {top_n} Most Frequent Words: {label_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Frequency", fontsize=12)
    plt.tight_layout()
    filename = f"top_words_{label_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename}")
    plt.show()


def plot_hate_ratio_by_length(df):
    df = df.copy()
    df["len"] = df["Content"].astype(str).apply(len)
    bins = pd.cut(df["len"], bins=[0, 50, 100, 200, 500, 2000])
    grouped = df.groupby(bins, observed=True)["Label"].mean()
    
    plt.figure(figsize=(10, 5))
    grouped.plot(kind="bar", color="#3498db", alpha=0.8)
    plt.title("Hate Speech Ratio by Text Length", fontsize=14, fontweight="bold")
    plt.ylabel("Hate Ratio", fontsize=12)
    plt.xlabel("Text Length (characters)", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("hate_ratio_by_length.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: hate_ratio_by_length.png")
    plt.show()


def plot_wordcloud(df, label):
    """Generate and display wordcloud for each class."""
    texts = df[df["Label"] == label]["Content"]
    combined_text = " ".join(texts.astype(str).values)
    
    label_name = "Hate Speech" if label == 1 else "Clean Text"
    bg_color = "#ffebee" if label == 1 else "#e8f5e9"
    
    wordcloud = WordCloud(
        width=1200, 
        height=600,
        background_color=bg_color,
        stopwords="english",
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(combined_text)
    
    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud: {label_name}", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    filename = f"wordcloud_{label_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename}")
    plt.show()


def run_all(path="HateSpeechDataset.csv"):
    print("\n" + "="*60)
    print("HATE SPEECH DATASET VISUALIZATION")
    print("="*60 + "\n")
    
    df = load_and_clean(path)
    print(f"Dataset loaded: {len(df)} samples")
    print(f"  - Clean (0): {(df['Label'] == 0).sum()}")
    print(f"  - Hate (1): {(df['Label'] == 1).sum()}\n")
    
    plot_class_distribution(df)
    plot_text_length_distribution(df)
    plot_top_words(df, 1)   # Hate
    plot_top_words(df, 0)   # Clean
    plot_wordcloud(df, 1)   # Hate wordcloud
    plot_wordcloud(df, 0)   # Clean wordcloud
    plot_hate_ratio_by_length(df)
    
    print("\n" + "="*60)
    print("All visualizations completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all()
