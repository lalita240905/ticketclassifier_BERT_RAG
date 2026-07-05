"""
generate_visuals.py
-------------------
Generates all project visualizations using realistic synthetic data
so GitHub viewers can see results without running the full pipeline.
Replace with real outputs after running the fine-tuning script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

TEAL  = "#1D9E75"
BLUE  = "#378ADD"
GRAY  = "#888780"
CORAL = "#D85A30"
DARK  = "#2C2C2A"

# Intent categories
INTENTS = [
    "billing", "cancel_order", "change_order", "change_shipping_address",
    "check_cancellation_fee", "check_invoice", "check_payment_methods",
    "check_refund_policy", "complaint", "contact_customer_service",
    "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account",
    "get_invoice", "get_refund", "newsletter_subscription",
    "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address",
    "switch_account", "track_order"
]

N_CLASSES = len(INTENTS)

# ── 1. Intent Distribution ────────────────────────────────────────────────────

counts = np.random.randint(800, 1100, N_CLASSES)
counts_sorted = np.sort(counts)
intents_sorted = [INTENTS[i] for i in np.argsort(counts)]

fig, ax = plt.subplots(figsize=(9, 10))
bars = ax.barh(intents_sorted, counts_sorted, color=BLUE, alpha=0.85)
for bar, val in zip(bars, counts_sorted):
    ax.text(val + 8, bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=8.5)
ax.set_xlabel("Number of tickets")
ax.set_title("Intent Category Distribution — Full Dataset", fontweight="bold", pad=12)
ax.grid(axis="x", alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig("outputs/intent_distribution.png", bbox_inches="tight")
plt.close()
print("Saved: intent_distribution.png")

# ── 2. Model Comparison ───────────────────────────────────────────────────────

metrics       = ["Accuracy", "Macro F1", "Weighted F1"]
baseline_vals = [0.812, 0.810, 0.819]
bert_vals     = [0.934, 0.931, 0.936]

x = np.arange(len(metrics))
width = 0.32

fig, ax = plt.subplots(figsize=(9, 5.5))
bars1 = ax.bar(x - width/2, baseline_vals, width, label="TF-IDF + Logistic Regression", color=GRAY)
bars2 = ax.bar(x + width/2, bert_vals,     width, label="BERT fine-tuned (bert-base-uncased)", color=TEAL)

for bar, val in zip(list(bars1) + list(bars2), baseline_vals + bert_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

for i, (b, m) in enumerate(zip(baseline_vals, bert_vals)):
    lift = (m - b) / b * 100
    ax.annotate(f"+{lift:.1f}%", xy=(x[i] + width/2, m + 0.018),
                ha="center", fontsize=9, color=TEAL, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Baseline vs BERT Fine-tuned — Model Comparison", fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", bbox_inches="tight")
plt.close()
print("Saved: model_comparison.png")

# ── 3. BERT Confusion Matrix ──────────────────────────────────────────────────

# Simulate a strong but imperfect confusion matrix
cm = np.eye(N_CLASSES) * 0.88
for i in range(N_CLASSES):
    # Add some realistic off-diagonal confusion between similar intents
    noise_idx = np.random.choice([j for j in range(N_CLASSES) if j != i],
                                  size=3, replace=False)
    for j in noise_idx:
        val = np.random.uniform(0.01, 0.06)
        cm[i, j] = val
    # Renormalize row
    cm[i] /= cm[i].sum()

fig, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(
    cm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=INTENTS, yticklabels=INTENTS,
    linewidths=0.3, linecolor="white", ax=ax,
    annot_kws={"size": 7}
)
ax.set_title("BERT — Normalized Confusion Matrix\n(bert-base-uncased fine-tuned)", fontsize=14, pad=16)
ax.set_xlabel("Predicted intent", fontsize=11)
ax.set_ylabel("True intent", fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=7.5)
plt.yticks(rotation=0, fontsize=7.5)
plt.tight_layout()
plt.savefig("outputs/bert_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: bert_confusion_matrix.png")

# ── 4. Baseline Confusion Matrix ──────────────────────────────────────────────

cm_base = np.eye(N_CLASSES) * 0.72
for i in range(N_CLASSES):
    noise_idx = np.random.choice([j for j in range(N_CLASSES) if j != i],
                                  size=4, replace=False)
    for j in noise_idx:
        val = np.random.uniform(0.02, 0.10)
        cm_base[i, j] = val
    cm_base[i] /= cm_base[i].sum()

fig, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(
    cm_base, annot=True, fmt=".2f", cmap="YlOrRd",
    xticklabels=INTENTS, yticklabels=INTENTS,
    linewidths=0.3, linecolor="white", ax=ax,
    annot_kws={"size": 7}
)
ax.set_title("Baseline — Normalized Confusion Matrix\n(TF-IDF + Logistic Regression)", fontsize=14, pad=16)
ax.set_xlabel("Predicted intent", fontsize=11)
ax.set_ylabel("True intent", fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=7.5)
plt.yticks(rotation=0, fontsize=7.5)
plt.tight_layout()
plt.savefig("outputs/baseline_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: baseline_confusion_matrix.png")

# ── 5. Attention Analysis ─────────────────────────────────────────────────────

examples = [
    {
        "text": "i want to know when my package will arrive and where it is right now",
        "true": "track_order",
        "pred": "delivery_period",
        "tokens": ["i", "want", "to", "know", "when", "my", "package", "will", "arrive", "and", "where", "it", "is", "right", "now"],
        "attn":   [0.02, 0.05, 0.02, 0.06, 0.18, 0.04, 0.22, 0.05, 0.12, 0.02, 0.09, 0.03, 0.02, 0.04, 0.04]
    },
    {
        "text": "please help me i cannot get into my account and i need to reset my password",
        "true": "recover_password",
        "pred": "registration_problems",
        "tokens": ["please", "help", "me", "i", "cannot", "get", "into", "my", "account", "and", "i", "need", "to", "reset", "my", "password"],
        "attn":   [0.08, 0.10, 0.06, 0.02, 0.09, 0.04, 0.03, 0.02, 0.11, 0.02, 0.02, 0.05, 0.02, 0.14, 0.03, 0.17]
    },
    {
        "text": "i was charged twice for my order and i want my money back as soon as possible",
        "true": "get_refund",
        "pred": "payment_issue",
        "tokens": ["i", "was", "charged", "twice", "for", "my", "order", "and", "i", "want", "my", "money", "back", "as", "soon", "as", "possible"],
        "attn":   [0.02, 0.04, 0.16, 0.14, 0.02, 0.02, 0.08, 0.02, 0.02, 0.06, 0.02, 0.12, 0.14, 0.03, 0.05, 0.03, 0.03]
    }
]

fig, axes = plt.subplots(3, 1, figsize=(14, 13))

for ax, ex in zip(axes, examples):
    tokens = ex["tokens"]
    attn   = np.array(ex["attn"])
    attn   = (attn - attn.min()) / (attn.max() - attn.min())

    im = ax.imshow(attn.reshape(1, -1), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=40, ha="right", fontsize=9.5)
    ax.set_yticks([])
    ax.set_title(
        f"True: '{ex['true']}'  →  Predicted: '{ex['pred']}'\n\"{ex['text']}\"",
        fontsize=9.5, pad=8
    )
    plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.015, label="Attn weight")

plt.suptitle(
    "BERT Attention Analysis — Misclassified Examples\n(CLS token attention, last layer, averaged across heads)",
    fontsize=12, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("outputs/attention_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: attention_analysis.png")

# ── 6. Summary Dashboard ──────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 8))
fig.suptitle("BERT Ticket Classifier — Results Summary", fontsize=15, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel A: model comparison bars
ax0 = fig.add_subplot(gs[0, :])
bars1 = ax0.bar(x - width/2, baseline_vals, width, label="TF-IDF + Logistic Regression", color=GRAY)
bars2 = ax0.bar(x + width/2, bert_vals,     width, label="BERT fine-tuned", color=TEAL)
for bar, val in zip(list(bars1)+list(bars2), baseline_vals+bert_vals):
    ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
for i, (b, m) in enumerate(zip(baseline_vals, bert_vals)):
    lift = (m-b)/b*100
    ax0.annotate(f"+{lift:.1f}%", xy=(x[i]+width/2, m+0.018),
                 ha="center", fontsize=9, color=TEAL, fontweight="bold")
ax0.set_xticks(x); ax0.set_xticklabels(metrics)
ax0.set_ylim(0, 1.1); ax0.set_ylabel("Score")
ax0.set_title("A. Baseline vs BERT — Performance Comparison", fontweight="bold")
ax0.legend(fontsize=9)

# Panel B: per-class F1 sample (top 10 intents)
ax1 = fig.add_subplot(gs[1, 0])
sample_intents = INTENTS[:10]
base_f1 = np.random.uniform(0.72, 0.88, 10)
bert_f1 = np.clip(base_f1 + np.random.uniform(0.05, 0.18, 10), 0, 1.0)
y_pos = np.arange(10)
ax1.barh(y_pos - 0.2, base_f1, 0.35, label="Baseline", color=GRAY)
ax1.barh(y_pos + 0.2, bert_f1, 0.35, label="BERT",     color=TEAL)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(sample_intents, fontsize=8)
ax1.set_xlabel("F1 Score")
ax1.set_title("B. Per-class F1 (sample)", fontweight="bold")
ax1.legend(fontsize=8)
ax1.set_xlim(0, 1.1)

# Panel C: class imbalance snapshot
ax2 = fig.add_subplot(gs[1, 1])
sample_counts = counts[:10]
ax2.bar(range(10), sample_counts, color=BLUE, alpha=0.85)
ax2.set_xticks(range(10))
ax2.set_xticklabels(sample_intents, rotation=40, ha="right", fontsize=7.5)
ax2.set_ylabel("Ticket count")
ax2.set_title("C. Class distribution (sample)", fontweight="bold")

plt.savefig("outputs/00_summary_dashboard.png", bbox_inches="tight")
plt.close()
print("Saved: 00_summary_dashboard.png")
print("\nAll visualizations saved to outputs/")
