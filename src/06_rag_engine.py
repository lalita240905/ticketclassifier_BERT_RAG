"""
06_rag_engine.py
-----------------
The core RAG engine: given a new incoming ticket, this module

  1. RETRIEVES the most semantically similar historical tickets and
     their resolutions (using the FAISS index built by
     05_build_ticket_index.py), optionally re-ranked using the
     fine-tuned BERT classifier's predicted intent as a relevance
     signal.
  2. RECOMMENDS a next-best-action: a draft reply grounded in those
     retrieved resolutions, plus a confidence signal and citations
     back to the source tickets, so an agent can review and send
     rather than starting from a blank page.

Generation has two modes, chosen automatically:
  - LLM-grounded (if ANTHROPIC_API_KEY is set): Claude drafts the reply,
    instructed to use ONLY the retrieved resolutions as source material.
  - Extractive fallback (no API key): a template stitches together the
    retrieved resolutions directly — no generation model required, so
    the pipeline still works out of the box.

Install: pip install sentence-transformers faiss-cpu pandas torch transformers anthropic
"""

import os
import json
import numpy as np
import pandas as pd
import faiss
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer

try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    from google import genai as google_genai
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


# ── Config ────────────────────────────────────────────────────────────────────

INDEX_DIR      = "outputs/rag_index"
CLASSIFIER_DIR = "outputs/bert_model"
LABEL_MAP_PATH = "data/label_map.csv"
MAX_LENGTH     = 128
CLAUDE_MODEL   = "claude-sonnet-4-6"
GEMINI_MODEL   = "gemini-2.5-flash"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RetrievedTicket:
    ticket_id: str
    instruction: str
    response: str
    intent: str
    category: str
    similarity: float
    intent_match: bool


@dataclass
class NextBestAction:
    ticket_text: str
    predicted_intent: str | None
    predicted_confidence: float | None
    retrieved: list[RetrievedTicket]
    draft_response: str
    generation_mode: str  # "gemini_grounded" | "claude_grounded" | "extractive_fallback"


# ── Engine ────────────────────────────────────────────────────────────────────

class TicketRAGEngine:
    """
    Loads the semantic index (+ optionally the fine-tuned BERT classifier)
    once, then serves retrieval + draft-response generation for new tickets.
    """

    def __init__(
        self,
        index_dir: str = INDEX_DIR,
        classifier_dir: str = CLASSIFIER_DIR,
        label_map_path: str = LABEL_MAP_PATH,
        use_classifier: bool = True,
        gemini_api_key: str | None = None,
        anthropic_api_key: str | None = None,
    ):
        self.index_dir = index_dir

        # -- Load retrieval index + metadata --
        config_path = os.path.join(index_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No index found at {index_dir}. Run 05_build_ticket_index.py first."
            )
        with open(config_path) as f:
            self.index_config = json.load(f)

        self.index = faiss.read_index(os.path.join(index_dir, "tickets.index"))
        self.metadata = pd.read_parquet(os.path.join(index_dir, "metadata.parquet"))
        self.embed_model = SentenceTransformer(self.index_config["embed_model_name"])

        # -- Load fine-tuned BERT classifier (optional, for intent re-ranking) --
        self.classifier = None
        self.tokenizer = None
        self.label_map = None
        if use_classifier and _HAS_TRANSFORMERS and os.path.isdir(classifier_dir):
            try:
                self.tokenizer = BertTokenizer.from_pretrained(classifier_dir)
                self.classifier = BertForSequenceClassification.from_pretrained(classifier_dir)
                self.classifier.eval()
                self.label_map = pd.read_csv(label_map_path)
                print("Loaded fine-tuned BERT classifier for intent re-ranking.")
            except Exception as e:
                print(f"Could not load classifier ({e}); continuing retrieval-only.")

        # -- LLM generation client (optional) --
        # Gemini is tried first (free-tier friendly), then Anthropic, then
        # we fall back to the no-LLM extractive draft.
        self.llm_client = None
        self.llm_provider = None  # "gemini" | "claude" | None

        gemini_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        claude_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

        if gemini_key and _HAS_GEMINI:
            self.llm_client = google_genai.Client(api_key=gemini_key)
            self.llm_provider = "gemini"
            print("Gemini API key found — using Gemini-grounded draft generation.")
        elif claude_key and _HAS_ANTHROPIC:
            self.llm_client = anthropic.Anthropic(api_key=claude_key)
            self.llm_provider = "claude"
            print("Anthropic API key found — using Claude-grounded draft generation.")
        else:
            print("No LLM API key found — using extractive fallback draft generation.")

    # ── Intent prediction ─────────────────────────────────────────────────

    def predict_intent(self, text: str) -> tuple[str | None, float | None]:
        """Predict the ticket's intent using the fine-tuned BERT classifier, if loaded."""
        if self.classifier is None:
            return None, None

        encoding = self.tokenizer(
            text, max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.classifier(**encoding).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())
        intent = self.label_map.loc[self.label_map["label"] == pred_id, "intent"].iloc[0]
        return intent, confidence

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        text: str,
        top_k: int = 5,
        predicted_intent: str | None = None,
        intent_boost_weight: float = 0.15,
        candidate_multiplier: int = 4,
    ) -> list[RetrievedTicket]:
        """
        Semantic search over historical tickets, optionally re-ranked so
        that tickets sharing the classifier's predicted intent surface
        higher — without discarding a strong semantic match just because
        its label happens to differ.

        Final score = (1 - w) * cosine_similarity + w * intent_match
        where w = intent_boost_weight (0 disables re-ranking).
        """
        query_vec = self.embed_model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        n_candidates = min(top_k * candidate_multiplier, self.index.ntotal)
        similarities, indices = self.index.search(query_vec, n_candidates)

        candidates = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            row = self.metadata.iloc[idx]
            intent_match = bool(predicted_intent) and (row["intent"] == predicted_intent)
            final_score = (
                (1 - intent_boost_weight) * float(sim) + intent_boost_weight * float(intent_match)
                if predicted_intent
                else float(sim)
            )
            candidates.append(
                RetrievedTicket(
                    ticket_id=str(row["ticket_id"]),
                    instruction=row["instruction"],
                    response=row["response"],
                    intent=row["intent"],
                    category=row["category"],
                    similarity=float(sim),
                    intent_match=intent_match,
                )
            )
            candidates[-1].__dict__["_final_score"] = final_score

        candidates.sort(key=lambda c: c.__dict__["_final_score"], reverse=True)
        return candidates[:top_k]

    # ── Draft generation ──────────────────────────────────────────────────

    @staticmethod
    def _build_grounding_prompt(text: str, retrieved: list[RetrievedTicket]) -> tuple[str, str]:
        """Shared system/user prompt construction for any LLM provider."""
        context_blocks = "\n\n".join(
            f"[Past ticket {r.ticket_id} | intent: {r.intent} | similarity: {r.similarity:.2f}]\n"
            f"Customer said: {r.instruction}\n"
            f"Agent resolved with: {r.response}"
            for r in retrieved
        )

        system_prompt = (
            "You are drafting a customer support agent's reply. You must ground your draft "
            "STRICTLY in the resolutions from the retrieved past tickets provided below — do "
            "not invent policies, refund amounts, timelines, or steps that are not present in "
            "them. If the retrieved tickets conflict or don't clearly cover the new ticket, say "
            "so plainly and draft a best-effort reply flagged for agent review. Keep the tone "
            "professional and concise. Do not include a greeting/sign-off placeholder like "
            "'[Agent Name]' — just write the body."
        )

        user_prompt = (
            f"New ticket from customer:\n\"{text}\"\n\n"
            f"Retrieved past tickets and their resolutions:\n\n{context_blocks}\n\n"
            "Draft the agent's reply to the new ticket."
        )
        return system_prompt, user_prompt

    def _generate_claude_grounded(self, text: str, retrieved: list[RetrievedTicket]) -> str:
        """Ask Claude to draft a reply, grounded strictly in retrieved resolutions."""
        system_prompt, user_prompt = self._build_grounding_prompt(text, retrieved)

        response = self.llm_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def _generate_gemini_grounded(self, text: str, retrieved: list[RetrievedTicket]) -> str:
        """Ask Gemini to draft a reply, grounded strictly in retrieved resolutions."""
        system_prompt, user_prompt = self._build_grounding_prompt(text, retrieved)

        response = self.llm_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=google_genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=500,
            ),
        )
        return response.text.strip()

    def _generate_extractive_fallback(self, text: str, retrieved: list[RetrievedTicket]) -> str:
        """
        No-LLM fallback: build a draft directly from the retrieved resolutions,
        with no generation model required. This keeps the pipeline usable
        out of the box (no API key), at the cost of a more template-y draft.
        """
        if not retrieved:
            return (
                "No sufficiently similar past tickets were found. Please draft a reply "
                "manually and consider flagging this as a new issue type."
            )

        top = retrieved[0]
        draft_lines = [
            f"Thanks for reaching out. Based on similar past cases (closest match: "
            f"ticket {top.ticket_id}, {top.similarity:.0%} similar, intent: {top.intent}), "
            f"here's the suggested resolution:",
            "",
            top.response.strip(),
        ]

        # Surface any materially different alternative resolutions from other
        # retrieved tickets with a different intent, in case the top match
        # doesn't fully fit.
        alternatives = [
            r for r in retrieved[1:]
            if r.intent != top.intent and r.response.strip() != top.response.strip()
        ]
        if alternatives:
            draft_lines.append("")
            draft_lines.append("Alternative approaches from related past tickets:")
            for alt in alternatives[:2]:
                draft_lines.append(f"  • ({alt.intent}, {alt.similarity:.0%} similar) {alt.response.strip()}")

        draft_lines.append("")
        draft_lines.append(
            "[Auto-drafted from historical resolutions — review before sending. "
            "No LLM configured: set ANTHROPIC_API_KEY for a more natural, synthesized draft.]"
        )
        return "\n".join(draft_lines)

    def generate_draft(self, text: str, retrieved: list[RetrievedTicket]) -> tuple[str, str]:
        """Returns (draft_response, generation_mode)."""
        if self.llm_provider == "gemini":
            try:
                return self._generate_gemini_grounded(text, retrieved), "gemini_grounded"
            except Exception as e:
                print(f"Gemini generation failed ({e}); falling back to extractive draft.")
        elif self.llm_provider == "claude":
            try:
                return self._generate_claude_grounded(text, retrieved), "claude_grounded"
            except Exception as e:
                print(f"Claude generation failed ({e}); falling back to extractive draft.")
        return self._generate_extractive_fallback(text, retrieved), "extractive_fallback"

    # ── Full pipeline ─────────────────────────────────────────────────────

    def recommend_next_best_action(self, text: str, top_k: int = 5) -> NextBestAction:
        """
        Full pipeline for one incoming ticket:
          classify intent -> retrieve similar resolved tickets -> draft a
          grounded reply. This is the single entry point an agent-facing
          app (see 07_demo_app.py) should call.
        """
        predicted_intent, predicted_confidence = self.predict_intent(text)
        retrieved = self.retrieve(text, top_k=top_k, predicted_intent=predicted_intent)
        draft, mode = self.generate_draft(text, retrieved)

        return NextBestAction(
            ticket_text=text,
            predicted_intent=predicted_intent,
            predicted_confidence=predicted_confidence,
            retrieved=retrieved,
            draft_response=draft,
            generation_mode=mode,
        )


# ── CLI demo ──────────────────────────────────────────────────────────────────

def _print_result(result: NextBestAction):
    print("\n" + "=" * 70)
    print(f"TICKET: {result.ticket_text}")
    if result.predicted_intent:
        print(f"Predicted intent: {result.predicted_intent} ({result.predicted_confidence:.1%} confidence)")
    print(f"\nTop {len(result.retrieved)} similar past tickets:")
    for r in result.retrieved:
        flag = " <- intent match" if r.intent_match else ""
        print(f"  [{r.ticket_id}] sim={r.similarity:.2f} intent={r.intent}{flag}")
        print(f"      Q: {r.instruction[:90]}")
        print(f"      A: {r.response[:90]}")
    print(f"\nDraft response (mode: {result.generation_mode}):")
    print("-" * 70)
    print(result.draft_response)
    print("=" * 70)


if __name__ == "__main__":
    import sys

    engine = TicketRAGEngine()

    example_ticket = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Hi, I was charged twice for my last order and I need one of the charges refunded."
    )
    result = engine.recommend_next_best_action(example_ticket, top_k=5)
    _print_result(result)