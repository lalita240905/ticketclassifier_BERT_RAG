"""
04_evaluation.py
----------------
Full evaluation of the fine-tuned BERT model:
  1. Per-class precision, recall, F1
  2. Confusion matrix
  3. Baseline vs BERT comparison
  4. Attention weight analysis on misclassified examples

Attention analysis:
  BERT's attention mechanism assigns weights to each input token
  when making a prediction. By inspecting these weights on
  misclassified examples, we can understand WHY the model failed —
  e.g. it may have attended to generic words ("please", "help")
  instead of intent-specific terms ("cancel", "refund", "broken").
  These insights can inform improvements to ticket intake form design.

Run after: 01_preprocessing.py, 02_baseline.py, 03_bert_finetune.py
"""

import pandas as pd
import numpy as np
import os
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix
)


OUTPUT_DIR = "outputs/bert_model"
MAX_LENGTH = 128


def load_model_and_data():
    """Load fine-tuned model, tokenizer, and test data."""
    print(f"Loading model from {OUTPUT_DIR}...")
    tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR)
    model     = BertForSequenceClassification.from_pretrained(
        OUTPUT_DIR,
        output_attentions=True  # required for attention weight extraction
    )
    model.eval()

    test_df   = pd.read_csv("data/test.csv")
    label_map = pd.read_csv("data/label_map.csv")

    return model, tokenizer, test_df, label_map


def predict_batch(model, tokenizer, texts, batch_size=64):
    """
    Run inference on a list of texts in batches.
    Returns predicted labels and raw logits.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds  = []
    all_logits = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoding = tokenizer(
            batch,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoding)

        logits = outputs.logits.cpu().numpy()
        preds  = np.argmax(logits, axis=-1)

        all_preds.extend(preds)
        all_logits.extend(logits)

    return np.array(all_preds), np.array(all_logits)


def evaluate_bert(model, tokenizer, test_df, label_map):
    """Full evaluation with per-class metrics and confusion matrix."""
    os.makedirs("outputs", exist_ok=True)

    texts  = test_df["text"].tolist()
    y_true = test_df["label"].values
    intent_names = label_map["intent"].tolist()

    print("Running inference on test set...")
    y_pred, logits = predict_batch(model, tokenizer, texts)

    acc    = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro")
    f1_wt  = f1_score(y_true, y_pred, average="weighted")

    print("\n── BERT Evaluation Results ──────────────────────")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  Macro F1:     {f1_mac:.4f}")
    print(f"  Weighted F1:  {f1_wt:.4f}")
    print("\n── Per-class Report ─────────────────────────────")
    print(classification_report(y_true, y_pred, target_names=intent_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=intent_names,
        yticklabels=intent_names,
        linewidths=0.3, linecolor="white", ax=ax
    )
    ax.set_title("BERT — Normalized Confusion Matrix\n(bert-base-uncased fine-tuned)", fontsize=14, pad=16)
    ax.set_xlabel("Predicted intent", fontsize=11)
    ax.set_ylabel("True intent", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/bert_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Confusion matrix saved to outputs/bert_confusion_matrix.png")

    return y_pred, {"accuracy": acc, "macro_f1": f1_mac, "weighted_f1": f1_wt}


def compare_models():
    """
    Load baseline and BERT metrics and produce a side-by-side comparison chart.
    Shows the relative improvement from fine-tuning.
    """
    with open("outputs/baseline_metrics.json") as f:
        baseline = json.load(f)
    with open("outputs/bert_metrics.json") as f:
        bert = json.load(f)

    metrics  = ["accuracy", "macro_f1", "weighted_f1"]
    labels   = ["Accuracy", "Macro F1", "Weighted F1"]
    baseline_vals = [baseline[m] for m in metrics]
    bert_vals     = [bert[m]      for m in metrics]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label="TF-IDF + Logistic Regression", color="#888780")
    bars2 = ax.bar(x + width/2, bert_vals,     width, label="BERT fine-tuned",               color="#1D9E75")

    for bar, val in zip(list(bars1) + list(bars2), baseline_vals + bert_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # Annotate relative improvement
    for i, (b, m) in enumerate(zip(baseline_vals, bert_vals)):
        lift = (m - b) / b * 100
        ax.annotate(
            f"+{lift:.1f}%",
            xy=(x[i] + width/2, m + 0.015),
            ha="center", fontsize=8.5, color="#1D9E75", fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs BERT Fine-tuned — Model Comparison", fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Model comparison chart saved to outputs/model_comparison.png")


def attention_analysis(model, tokenizer, test_df, label_map, y_pred, n_examples=3):
    """
    Inspect attention weights on misclassified examples.

    For each misclassified ticket:
      1. Tokenize the text
      2. Extract attention weights from the last BERT layer
      3. Average across all attention heads
      4. Plot token-level attention as a heatmap

    High attention weight on generic words (e.g. "please", "help")
    vs. intent-specific words (e.g. "refund", "cancel") helps explain
    why the model confused two similar-sounding categories.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    intent_names = label_map["intent"].tolist()
    y_true       = test_df["label"].values

    # Find misclassified examples
    misclassified = test_df[y_pred != y_true].copy()
    misclassified["predicted"] = y_pred[y_pred != y_true]

    print(f"\n{len(misclassified):,} misclassified examples found.")
    print(f"Showing attention analysis for {n_examples} examples...\n")

    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 5 * n_examples))
    if n_examples == 1:
        axes = [axes]

    for idx, (ax, (_, row)) in enumerate(zip(axes, misclassified.head(n_examples).iterrows())):
        text      = row["text"]
        true_lbl  = intent_names[int(row["label"])]
        pred_lbl  = intent_names[int(row["predicted"])]

        # Tokenize
        inputs = tokenizer(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Forward pass with attention output
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Extract last layer attention: shape (1, n_heads, seq_len, seq_len)
        # Average over all attention heads → shape (seq_len, seq_len)
        last_layer_attn = outputs.attentions[-1][0]
        avg_attn = last_layer_attn.mean(dim=0).cpu().numpy()

        # Use CLS token's attention to all other tokens as the relevance signal
        # The CLS token aggregates information for the classification decision
        cls_attn = avg_attn[0, :]

        # Remove [CLS] and [SEP] tokens for cleaner visualization
        tokens_clean = tokens[1:-1]
        attn_clean   = cls_attn[1:-1]

        # Normalize to [0, 1]
        attn_norm = (attn_clean - attn_clean.min()) / (attn_clean.max() - attn_clean.min() + 1e-9)

        # Plot as horizontal heatmap
        attn_2d = attn_norm.reshape(1, -1)
        im = ax.imshow(attn_2d, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens_clean)))
        ax.set_xticklabels(tokens_clean, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks([])
        ax.set_title(
            f"Example {idx+1}  |  True: '{true_lbl}'  →  Predicted: '{pred_lbl}'\n"
            f"Text: \"{text[:120]}{'...' if len(text) > 120 else ''}\"",
            fontsize=9.5, pad=8
        )
        plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.015, label="Attention weight")

    plt.suptitle("BERT Attention Analysis — Misclassified Examples\n(CLS token attention, last layer, averaged across heads)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/attention_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Attention analysis saved to outputs/attention_analysis.png")


if __name__ == "__main__":
    model, tokenizer, test_df, label_map = load_model_and_data()

    y_pred, bert_metrics = evaluate_bert(model, tokenizer, test_df, label_map)

    compare_models()

    attention_analysis(model, tokenizer, test_df, label_map, y_pred, n_examples=3)

    print("\nAll evaluation outputs saved to outputs/")
