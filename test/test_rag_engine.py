"""
test_rag_engine_smoke.py
-------------------------
End-to-end smoke test for the RAG engine using a tiny synthetic ticket
corpus and a deterministic hash-based "embedder" in place of a real
sentence-transformer model. This lets us validate the retrieval +
generation plumbing (index build -> load -> retrieve -> draft) without
needing network access to download model weights.

Run: python -m pytest tests/test_rag_engine_smoke.py -v
"""

import json
import os
import sys
import hashlib
import numpy as np
import pandas as pd
import faiss
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


_STOPWORDS = {
    "i", "a", "an", "the", "is", "was", "for", "my", "your", "to", "you",
    "do", "it", "as", "in", "on", "from", "before", "one", "and", "or",
    "please", "can", "shows", "how", "never", "same",
}


class FakeEmbedder:
    """
    Deterministic bag-of-words-ish embedder: hashes each non-stopword into
    one of EMBED_DIM buckets and sums, so texts sharing meaningful words get
    non-trivial cosine similarity — enough to validate retrieval ranking
    logic without downloading a real model. Stopwords are excluded so
    incidental overlap on filler words doesn't swamp the topical signal.
    """

    EMBED_DIM = 64

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kwargs):
        vectors = []
        for text in texts:
            vec = np.zeros(self.EMBED_DIM, dtype="float32")
            for word in text.lower().split():
                if word in _STOPWORDS:
                    continue
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.EMBED_DIM
                vec[bucket] += 1.0
            norm = np.linalg.norm(vec)
            if normalize_embeddings and norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype="float32")


SYNTHETIC_TICKETS = [
    {"instruction": "I was charged twice for my order please refund one charge",
     "response": "I've refunded the duplicate charge to your original payment method; it should post within 5-7 business days.",
     "intent": "refund", "category": "billing"},
    {"instruction": "There is a duplicate charge on my credit card from your store",
     "response": "I can confirm the duplicate charge and have issued a refund; you'll see it in 5-7 business days.",
     "intent": "refund", "category": "billing"},
    {"instruction": "How do I cancel my subscription before the renewal date",
     "response": "I've canceled your subscription effective immediately; you won't be billed again.",
     "intent": "cancel", "category": "account"},
    {"instruction": "My package never arrived and tracking shows it as delivered",
     "response": "I've opened a lost-package investigation with the carrier and will send a replacement if it isn't located within 48 hours.",
     "intent": "shipping", "category": "delivery"},
]


@pytest.fixture
def rag_index_dir(tmp_path, monkeypatch):
    """Build a tiny FAISS index from the synthetic corpus, mimicking 05_build_ticket_index.py."""
    df = pd.DataFrame(SYNTHETIC_TICKETS)
    df["ticket_id"] = df.index.astype(str)

    embedder = FakeEmbedder()
    embeddings = embedder.encode(df["instruction"].tolist())

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_dir = tmp_path / "rag_index"
    index_dir.mkdir()
    faiss.write_index(index, str(index_dir / "tickets.index"))
    df.to_parquet(index_dir / "metadata.parquet", index=False)
    with open(index_dir / "config.json", "w") as f:
        json.dump({"embed_model_name": "fake", "embedding_dim": embeddings.shape[1], "n_tickets": len(df)}, f)

    # Patch SentenceTransformer so TicketRAGEngine loads our FakeEmbedder instead
    # of trying to download a real model from HuggingFace.
    import importlib
    rag_module = importlib.import_module("06_rag_engine")
    monkeypatch.setattr(rag_module, "SentenceTransformer", lambda name: FakeEmbedder())

    return str(index_dir), rag_module


def test_retrieval_ranks_semantically_similar_ticket_first(rag_index_dir):
    index_dir, rag_module = rag_index_dir
    engine = rag_module.TicketRAGEngine(
        index_dir=index_dir, use_classifier=False, anthropic_api_key=None
    )

    query = "I got billed twice for the same order, can you fix the duplicate charge?"
    results = engine.retrieve(query, top_k=2)

    assert len(results) == 2
    # Both top results should be the refund/billing tickets, not the unrelated
    # cancellation or shipping tickets.
    assert all(r.intent == "refund" for r in results)


def test_extractive_fallback_grounds_draft_in_retrieved_response(rag_index_dir):
    index_dir, rag_module = rag_index_dir
    engine = rag_module.TicketRAGEngine(
        index_dir=index_dir, use_classifier=False, anthropic_api_key=None
    )

    query = "I was double charged on my last purchase"
    result = engine.recommend_next_best_action(query, top_k=2)

    assert result.generation_mode == "extractive_fallback"
    # The draft must actually contain wording from a retrieved resolution —
    # i.e. it's grounded, not hallucinated.
    assert "refunded the duplicate charge" in result.draft_response
    assert len(result.retrieved) == 2


def test_no_matching_tickets_returns_safe_message(rag_index_dir):
    index_dir, rag_module = rag_index_dir
    engine = rag_module.TicketRAGEngine(
        index_dir=index_dir, use_classifier=False, anthropic_api_key=None
    )

    # Monkeypatch retrieve to simulate an empty result set (e.g. empty corpus).
    engine.retrieve = lambda *a, **k: []
    result = engine.recommend_next_best_action("some totally novel issue", top_k=5)

    assert "No sufficiently similar past tickets" in result.draft_response
    assert result.generation_mode == "extractive_fallback"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
