"""
03_bert_finetune.py
--------------------
Fine-tunes bert-base-uncased on the support ticket classification task
using the HuggingFace Trainer API.

── RECOMMENDED: Run this script on Google Colab ──────────────────────────
  1. Open https://colab.research.google.com
  2. Runtime → Change runtime type → GPU (T4)
  3. Clone your repo: !git clone https://github.com/YOUR_USERNAME/bert-ticket-classifier
  4. Install deps: !pip install transformers datasets torch scikit-learn
  5. Run: !python src/03_bert_finetune.py
  Fine-tuning takes ~15-25 minutes on a free T4 GPU.
──────────────────────────────────────────────────────────────────────────

Install: pip install transformers torch datasets scikit-learn pandas
"""

import pandas as pd
import numpy as np
import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, accuracy_score


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME  = "bert-base-uncased"
MAX_LENGTH  = 128     # max token length — most support tickets are short
BATCH_SIZE  = 32      # reduce to 16 if Colab runs out of GPU memory
EPOCHS      = 4       # 3-4 epochs is standard for BERT fine-tuning
LR          = 2e-5    # learning rate — standard range for BERT is 1e-5 to 5e-5
OUTPUT_DIR  = "outputs/bert_model"
SEED        = 42


# ── Dataset class ─────────────────────────────────────────────────────────────

class TicketDataset(Dataset):
    """
    PyTorch Dataset wrapper for support ticket text and labels.

    Tokenization is done here rather than upfront so that each
    batch is processed on-the-fly — more memory efficient for
    large datasets.
    """

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts      = texts.tolist()
        self.labels     = labels.tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",   # pad shorter sequences to max_length
            truncation=True,        # truncate longer sequences
            return_tensors="pt"
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ── Weighted loss trainer ─────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """
    Custom Trainer subclass that applies weighted cross-entropy loss.

    Standard cross-entropy treats all classes equally — a problem when
    some intent categories have far fewer examples than others.
    Weighted loss penalizes misclassification of minority classes more
    heavily, pushing the model to learn them properly.
    """

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Move weights to same device as logits
        weights = self.class_weights.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """
    Called by Trainer after each evaluation step.
    Returns accuracy and macro F1 — macro F1 is our primary metric
    because it treats all classes equally regardless of size.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted")
    }


# ── Main fine-tuning pipeline ─────────────────────────────────────────────────

def load_splits():
    train_df  = pd.read_csv("data/train.csv")
    test_df   = pd.read_csv("data/test.csv")
    label_map = pd.read_csv("data/label_map.csv")
    return train_df, test_df, label_map


def get_class_weights(train_df, n_classes):
    """
    Compute class weights from training label distribution.
    Returns a float tensor for use in WeightedTrainer.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(n_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values
    )
    return torch.tensor(weights, dtype=torch.float)


def finetune():
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_df, test_df, label_map = load_splits()
    n_classes = len(label_map)

    print(f"Classes: {n_classes}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: No GPU detected. Fine-tuning on CPU will be very slow.")
        print("   Recommended: Run this script on Google Colab with a T4 GPU.\n")

    # Load tokenizer and model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes
    )

    # Build datasets
    train_dataset = TicketDataset(train_df["text"], train_df["label"], tokenizer)
    test_dataset  = TicketDataset(test_df["text"],  test_df["label"],  tokenizer)

    # Class weights for imbalance handling
    class_weights = get_class_weights(train_df, n_classes)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=0.1,              # linear warmup for first 10% of steps
        weight_decay=0.01,             # L2 regularization
        evaluation_strategy="epoch",   # evaluate at end of each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,   # restore best checkpoint after training
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir="outputs/logs",
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available()  # mixed precision — speeds up GPU training
    )

    # Trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    print("\nStarting fine-tuning...")
    trainer.train()

    # Final evaluation on test set
    print("\nRunning final evaluation on test set...")
    results = trainer.evaluate(test_dataset)

    print("\n── BERT Fine-tuning Results ──────────────────────")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save metrics
    metrics = {
        "model": "BERT fine-tuned (bert-base-uncased)",
        "accuracy":     round(results.get("eval_accuracy", 0), 4),
        "macro_f1":     round(results.get("eval_macro_f1", 0), 4),
        "weighted_f1":  round(results.get("eval_weighted_f1", 0), 4),
        "epochs":       EPOCHS,
        "max_length":   MAX_LENGTH,
        "batch_size":   BATCH_SIZE,
        "learning_rate": LR
    }
    with open("outputs/bert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nModel saved to {OUTPUT_DIR}")
    print(f"Metrics saved to outputs/bert_metrics.json")

    return trainer, model, tokenizer, test_df, label_map


if __name__ == "__main__":
    finetune()
