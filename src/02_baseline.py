"""
02_baseline.py
--------------
Establishes a TF-IDF + Logistic Regression baseline for the
support ticket classification task.

Why a baseline?
  Before fine-tuning BERT, we need a reference point. If BERT
  only marginally outperforms a simple model, it may not be worth
  the compute cost. A strong baseline also helps us understand
  which categories are inherently hard to classify.

Expected performance on this dataset: F1 ~ 0.78-0.83
BERT should significantly outperform this.

Install: pip install scikit-learn pandas
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_splits():
    """Load preprocessed train/test splits."""
    train_df = pd.read_csv("data/train.csv")
    test_df  = pd.read_csv("data/test.csv")
    label_map = pd.read_csv("data/label_map.csv")

    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    return train_df, test_df, label_map


def build_tfidf_features(train_df, test_df):
    """
    Convert raw text to TF-IDF feature vectors.

    TF-IDF (Term Frequency-Inverse Document Frequency):
      - TF: how often a word appears in a document
      - IDF: how rare the word is across all documents
      - High TF-IDF = word is frequent in this doc but rare overall = informative

    Settings:
      - max_features=20000: keep top 20K most informative n-grams
      - ngram_range=(1,2): use both single words and two-word phrases
        e.g. "cancel" alone vs "cancel subscription" — the bigram
        is far more informative for intent classification
      - sublinear_tf=True: apply log normalization to term frequency
        to reduce the dominance of very frequent terms
    """
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2  # ignore terms that appear in fewer than 2 documents
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_test  = vectorizer.transform(test_df["text"])

    print(f"TF-IDF feature matrix: {X_train.shape}")
    return X_train, X_test, vectorizer


def train_baseline(X_train, y_train):
    """
    Train Logistic Regression classifier on TF-IDF features.

    Settings:
      - class_weight='balanced': automatically adjusts weights
        inversely proportional to class frequency — handles imbalance
      - max_iter=1000: enough iterations for convergence on this feature size
      - C=1.0: default regularization strength
    """
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("Baseline model trained.")
    return clf


def evaluate_baseline(clf, X_test, test_df, label_map):
    """
    Evaluate the baseline model and print a full classification report.
    Saves per-class F1 scores and a confusion matrix.
    """
    os.makedirs("outputs", exist_ok=True)

    y_pred = clf.predict(X_test)
    y_true = test_df["label"]

    intent_names = label_map["intent"].tolist()

    # Overall metrics
    acc    = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro")
    f1_wt  = f1_score(y_true, y_pred, average="weighted")

    print("\n── Baseline Results ─────────────────────────────")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Macro F1:           {f1_mac:.4f}")
    print(f"  Weighted F1:        {f1_wt:.4f}")
    print("\n── Per-class Report ─────────────────────────────")
    print(classification_report(y_true, y_pred, target_names=intent_names))

    # Save metrics for comparison with BERT
    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": round(acc, 4),
        "macro_f1": round(f1_mac, 4),
        "weighted_f1": round(f1_wt, 4)
    }
    with open("outputs/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, intent_names, "outputs/baseline_confusion_matrix.png")

    return metrics


def plot_confusion_matrix(y_true, y_pred, intent_names, path):
    """Plot and save normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=intent_names,
        yticklabels=intent_names,
        linewidths=0.3, linecolor="white",
        ax=ax
    )
    ax.set_title("Baseline — Normalized Confusion Matrix\n(TF-IDF + Logistic Regression)", fontsize=14, pad=16)
    ax.set_xlabel("Predicted intent", fontsize=11)
    ax.set_ylabel("True intent", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {path}")


if __name__ == "__main__":
    train_df, test_df, label_map = load_splits()

    X_train, X_test, vectorizer = build_tfidf_features(train_df, test_df)

    clf = train_baseline(X_train, train_df["label"])

    metrics = evaluate_baseline(clf, X_test, test_df, label_map)

    print(f"\nBaseline metrics saved to outputs/baseline_metrics.json")
