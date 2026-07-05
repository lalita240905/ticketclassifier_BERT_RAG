"""
07_demo_app.py
----------------
Agent-facing demo UI for the RAG-powered next-best-action engine.

Paste in an incoming ticket and get back:
  - the predicted intent (from the fine-tuned BERT classifier)
  - the most similar past tickets + how they were resolved
  - a draft reply grounded in those resolutions, ready to edit and send

Run: streamlit run src/07_demo_app.py
"""

import sys
import os
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

rag_module = import_module("06_rag_engine")
TicketRAGEngine = rag_module.TicketRAGEngine


st.set_page_config(page_title="Ticket RAG Assistant", layout="wide")
st.title("🎫 Next-Best-Action Assistant")
st.caption(
    "Semantic search over historical tickets + a draft reply grounded in past resolutions."
)


@st.cache_resource(show_spinner="Loading ticket index and models...")
def get_engine():
    return TicketRAGEngine()


try:
    engine = get_engine()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `python src/05_build_ticket_index.py` first to build the semantic index.")
    st.stop()

with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Similar tickets to retrieve", min_value=1, max_value=10, value=5)
    st.markdown("---")
    if engine.llm_provider == "gemini":
        st.success("LLM-grounded generation active (Gemini)")
    elif engine.llm_provider == "claude":
        st.success("LLM-grounded generation active (Claude)")
    else:
        st.warning("No GEMINI_API_KEY / ANTHROPIC_API_KEY set — using extractive fallback drafts")
    if engine.classifier is not None:
        st.success("BERT intent classifier loaded")
    else:
        st.info("No classifier loaded — retrieval is semantic-only (no intent re-ranking)")

ticket_text = st.text_area(
    "Incoming ticket",
    placeholder="e.g. Hi, I was charged twice for my last order and need one charge refunded.",
    height=120,
)

if st.button("Get recommendation", type="primary") and ticket_text.strip():
    with st.spinner("Retrieving similar tickets and drafting a response..."):
        result = engine.recommend_next_best_action(ticket_text, top_k=top_k)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Analysis")
        if result.predicted_intent:
            st.metric("Predicted intent", result.predicted_intent,
                      delta=f"{result.predicted_confidence:.1%} confidence")
        st.markdown(f"**Similar past tickets** ({len(result.retrieved)})")
        for r in result.retrieved:
            match_flag = " 🎯 intent match" if r.intent_match else ""
            with st.expander(f"[{r.ticket_id}] {r.similarity:.0%} similar — {r.intent}{match_flag}"):
                st.markdown(f"**Customer said:** {r.instruction}")
                st.markdown(f"**Resolution:** {r.response}")

    with col2:
        st.subheader("Draft response")
        mode_labels = {
            "gemini_grounded": "LLM-grounded (Gemini)",
            "claude_grounded": "LLM-grounded (Claude)",
            "extractive_fallback": "Extractive fallback",
        }
        mode_label = mode_labels.get(result.generation_mode, result.generation_mode)
        st.caption(f"Generation mode: {mode_label}")
        st.text_area("Editable draft", value=result.draft_response, height=350, key="draft_output")
        st.caption("Review before sending — always verify against current policy.")

elif ticket_text.strip() == "":
    st.info("Paste in a ticket above and click **Get recommendation**.")