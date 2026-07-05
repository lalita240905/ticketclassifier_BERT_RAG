"""
01_preprocessing.py
-------------------
Loads the Bitext Customer Support dataset from HuggingFace,
cleans the ticket text, encodes labels, and saves train/test
splits ready for both the baseline and BERT fine-tuning.

Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset
~27K labeled support tickets across 27 intent categories.

Install: pip install datasets pandas scikit-learn
"""

import pandas as pd
import numpy as np
import os
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data() -> pd.DataFrame:
    """
    Load Bitext customer support dataset from HuggingFace.
    Downloads automatically on first run — no manual download needed.

    Key columns:
      - instruction: the raw customer support ticket text
      - intent:      the category label (billing, technical, account, etc.)
    """
    print("Loading Bitext dataset from HuggingFace...")
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

    # Dataset only has a train split — we will create our own test split
    df = pd.DataFrame(dataset["train"])

    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nIntent distribution:\n{df['intent'].value_counts()}")

    return df


def clean_text(text: str) -> str:
    """
    Light text cleaning for support tickets:
    - Lowercase
    - Remove excess whitespace
    - Strip leading/trailing spaces
    - Remove special characters but keep punctuation that carries meaning
      (question marks, apostrophes)

    We keep cleaning minimal — BERT's tokenizer handles most normalization
    internally and aggressive cleaning can remove useful signal.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\?\'\.\,\!]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_data(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
    1. Clean ticket text
    2. Encode intent labels as integers
    3. Split into train (80%) and test (20%) with stratification
       to ensure each intent category is represented in both splits
    """
    # Clean text
    df["text"] = df["instruction"].apply(clean_text)

    # Encode labels
    # LabelEncoder maps each intent string to a unique integer
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["intent"])

    print(f"\nNumber of intent categories: {df['label'].nunique()}")
    print(f"Label mapping (first 10):")
    for i, cls in enumerate(le.classes_[:10]):
        print(f"  {i}: {cls}")

    # Stratified train/test split
    # Stratify ensures class proportions are preserved in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    train_df = pd.DataFrame({"text": X_train, "label": y_train}).reset_index(drop=True)
    test_df  = pd.DataFrame({"text": X_test,  "label": y_test}).reset_index(drop=True)

    print(f"\nTrain size: {len(train_df):,}")
    print(f"Test size:  {len(test_df):,}")

    return train_df, test_df, le


def compute_class_weights(train_df: pd.DataFrame) -> dict:
    """
    Compute class weights to handle class imbalance.

    Some intent categories have significantly more examples than others.
    Weighted cross-entropy loss penalizes misclassification of minority
    classes more heavily, preventing the model from ignoring them.

    Weight formula: total_samples / (n_classes * samples_in_class)
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(train_df["label"])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"]
    )

    weight_dict = dict(zip(classes, weights))
    print(f"\nClass weight range: [{min(weights):.3f}, {max(weights):.3f}]")

    return weight_dict


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = load_data()
    train_df, test_df, le = prepare_data(df)
    class_weights = compute_class_weights(train_df)

    # Save splits
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    # Save label mapping for evaluation
    label_map = pd.DataFrame({
        "label": range(len(le.classes_)),
        "intent": le.classes_
    })
    label_map.to_csv("data/label_map.csv", index=False)

    print("\nSaved: data/train.csv, data/test.csv, data/label_map.csv")
    print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
