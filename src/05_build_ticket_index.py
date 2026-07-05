"""
05_build_ticket_index.py
-------------------------
Builds a semantic search index over historical support tickets so that,
at inference time, we can retrieve past tickets that are *similar in
meaning* to a new incoming ticket — not just similar in keywords.

This is the "R" (retrieval) half of the RAG engine:
  1. Load the full historical ticket corpus (instruction + response +
     intent + category). We need the *response* column here, which
     01_preprocessing.py drops — it only keeps text/label for the
     classifier. So this script reloads the raw dataset directly.
  2. Embed every historical ticket's text with a sentence-transformer
     (dense vector that captures semantic meaning, not just words).
  3. Build a FAISS index over those embeddings for fast nearest-neighbor
     lookup at query time.
  4. Persist the index + a metadata table (instruction, response, intent,
     category) so 06_rag_engine.py can load it without re-embedding.

Install: pip install sentence-transformers faiss-cpu pandas datasets
"""

import os
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss


# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast, 384-dim, strong for short text
INDEX_DIR        = "outputs/rag_index"
BATCH_SIZE       = 128


def load_raw_corpus() -> pd.DataFrame:
    """
    Load the full Bitext dataset with all columns intact.

    Unlike 01_preprocessing.py (which keeps only text + label for the
    classifier), the RAG engine needs the `response` column — that's
    the historical agent resolution we retrieve and ground drafts on.
    """
    print("Loading Bitext dataset from HuggingFace...")
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = pd.DataFrame(dataset["train"])

    required = {"instruction", "response", "intent", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    df = df.dropna(subset=["instruction", "response"]).reset_index(drop=True)
    df["ticket_id"] = df.index.astype(str)

    print(f"Loaded {len(df):,} historical tickets with resolutions.")
    return df[["ticket_id", "instruction", "response", "intent", "category"]]


def embed_corpus(texts: list[str], model_name: str = EMBED_MODEL_NAME) -> tuple[np.ndarray, SentenceTransformer]:
    """
    Encode ticket text into dense semantic vectors.

    We embed the raw `instruction` text (not the heavily-regex-cleaned
    version used for the classifier) because the embedding model already
    handles casing/punctuation, and preserving natural phrasing helps
    semantic similarity match how a real agent would read the ticket.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Embedding {len(texts):,} tickets...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize so inner product == cosine similarity
    )
    return embeddings.astype("float32"), model


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a flat (exact, brute-force) FAISS index using inner product.

    Since embeddings are L2-normalized, inner product is equivalent to
    cosine similarity. IndexFlatIP is exact — no approximation — which
    is fine at this corpus scale (tens of thousands of tickets). For
    millions of tickets, swap in IndexIVFFlat or HNSW for speed.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Built FAISS index: {index.ntotal:,} vectors, dim={dim}")
    return index


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    df = load_raw_corpus()
    embeddings, model = embed_corpus(df["instruction"].tolist())
    index = build_faiss_index(embeddings)

    # Persist index
    faiss.write_index(index, os.path.join(INDEX_DIR, "tickets.index"))

    # Persist metadata (everything needed to display/ground a retrieved ticket)
    df.to_parquet(os.path.join(INDEX_DIR, "metadata.parquet"), index=False)

    # Persist config so the retrieval engine knows which embedding model to reload
    config = {
        "embed_model_name": EMBED_MODEL_NAME,
        "embedding_dim": int(embeddings.shape[1]),
        "n_tickets": int(len(df)),
        "metric": "cosine (via normalized inner product)",
    }
    with open(os.path.join(INDEX_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to {INDEX_DIR}/:")
    print("  tickets.index     — FAISS vector index")
    print("  metadata.parquet  — ticket_id, instruction, response, intent, category")
    print("  config.json       — embedding model + index metadata")


if __name__ == "__main__":
    main()
