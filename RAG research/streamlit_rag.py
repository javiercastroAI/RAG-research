#!/usr/bin/env python3
"""
Simple Streamlit front-end for the Downloads RAG system.
 - One-click index rebuild.
 - Chat-style Q&A with LLM (if OPENAI_API_KEY is set) or extractive fallback.
"""

import os
from pathlib import Path
from typing import List, Tuple

import streamlit as st

import rag_downloads as rag
from rag_downloads import Chunk


st.set_page_config(page_title="dnAI GPT for RAG research", layout="wide")

# Persist chat state across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ground_truth_input" not in st.session_state:
    st.session_state.ground_truth_input = ""


def get_cached_index(doc_dir: Path, index_path: Path, model: str):
    # Streamlit caches by doc_dir/index_path/model so we reuse embeddings across requests.
    @st.cache_resource(show_spinner=False)
    def _load(doc_dir_str: str, index_path_str: str, model_name: str):
        return rag.maybe_rebuild(Path(doc_dir_str), Path(index_path_str), model_name)

    return _load(str(doc_dir), str(index_path), model)


def format_sources(hits: List[Tuple[Chunk, float]]) -> None:
    for idx, (chunk, score) in enumerate(hits, start=1):
        path = Path(chunk.source).resolve()
        snippet = chunk.text[:400] + ("..." if len(chunk.text) > 400 else "")
        with st.expander(f"{idx}. {path.name} — score {score:.3f}"):
            st.write(snippet)
            if path.exists():
                try:
                    data = path.read_bytes()
                    mime = "application/pdf" if path.suffix.lower() == ".pdf" else "text/plain"
                    st.download_button(
                        "Open / Download",
                        data=data,
                        file_name=path.name,
                        mime=mime,
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.warning(f"Could not load file: {exc}")
            else:
                st.error("File not found on disk.")


def render_similarity_meta(index: dict):
    meta = index.get("meta", {})
    exact = meta.get("exact_duplicates_removed", [])
    similar = meta.get("semantic_similar_pairs", [])
    auto = meta.get("semantic_auto_dropped", [])

    with st.expander("Duplicates & Similarity (from last index build)", expanded=False):
        st.markdown("**Exact duplicates removed (kept newest)**")
        if exact:
            st.table(exact)
        else:
            st.caption("None recorded.")

        st.markdown("**Semantic similarity pairs (score >= threshold)**")
        if similar:
            st.dataframe(similar, use_container_width=True, hide_index=True)
        else:
            st.caption("None recorded.")

        st.markdown("**Auto-dropped similar files (if enabled)**")
        if auto:
            st.table(auto)
        else:
            st.caption("Auto-drop was not applied or no files were dropped.")

# Persist chat state across reruns so the text boxes don't clear after clicking Ask.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ground_truth_input" not in st.session_state:
    st.session_state.ground_truth_input = ""

st.title("🤖 dnAI GPT for RAG research")
st.caption("Indexes ~/Downloads/rag and answers with citations. If no OpenAI key is set, you get extractive evidence.")

with st.sidebar:
    st.subheader("Settings")
    doc_dir_input = st.text_input("Document folder", value=str(rag.DEFAULT_DOC_DIR))
    doc_dir = rag.resolve_doc_dir(Path(doc_dir_input)) if doc_dir_input else rag.DEFAULT_DOC_DIR
    index_path = Path(
        st.text_input("Index cache path", value=str(rag.DEFAULT_INDEX_PATH))
    )
    model = st.text_input("Encoder model", value=rag.DEFAULT_MODEL_NAME)
    top_k = st.slider("Top K", min_value=1, max_value=15, value=5, step=1)
    max_context = st.slider("Max context chars", min_value=1000, max_value=20000, value=rag.MAX_CONTEXT_CHARS, step=500)
    llm_model = st.text_input("LLM model", value=os.getenv("RAG_COMPLETION_MODEL", "gpt-4o-mini"))
    st.caption(f"OPENAI_API_KEY detected: {'yes' if os.getenv('OPENAI_API_KEY') else 'no'}")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    max_tokens = st.slider("Max tokens", min_value=100, max_value=1200, value=400, step=50)
    similar_threshold = st.slider("Similar doc threshold (cosine)", min_value=0.8, max_value=0.99, value=rag.DEFAULT_SIMILAR_THRESHOLD, step=0.01)
    auto_drop_similar = st.checkbox("Auto-drop older in similar pairs", value=False)
    use_true_ragas = st.checkbox("Compute RAGAS metrics (needs OPENAI_API_KEY & ragas)", value=False)
    st.text_area(
        "Ground truth (optional, for RAGAS recall)",
        key="ground_truth_input",
        height=100,
        help="Provide a reference answer to improve Context Recall; otherwise the generated answer is used.",
    )
    if st.button("Clear chat history"):
        st.session_state.messages = []

    rebuild = st.button("Rebuild index", type="primary")

if rebuild:
    with st.spinner(f"Building index in {doc_dir} ..."):
        rag.build_index(
            doc_dir,
            index_path,
            model,
            similar_threshold=similar_threshold,
            auto_drop_similar=auto_drop_similar,
        )
    st.success(f"Index rebuilt at {index_path}")

# Chat input at bottom
prompt = st.chat_input("Ask about the corpus")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        index = get_cached_index(doc_dir, index_path, model)
    except Exception as exc:
        st.error(f"Index load failed: {exc}")
    else:
        with st.spinner("Retrieving..."):
            hits = rag.top_k(prompt, index, k=top_k, model_name=model)
            compact_chunks, compact_scores = rag._compact_hits(hits, max_chars=max_context)
            answer = rag.generate_answer(
                prompt,
                compact_chunks,
                compact_scores,
                llm_model=llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            ragas_scores = {}
            if use_true_ragas:
                try:
                    with st.spinner("Computing true RAGAS metrics..."):
                        ragas_scores = rag.evaluate_ragas_true(
                            prompt,
                            answer,
                            compact_chunks,
                            llm_model=llm_model,
                            embed_model=rag.DEFAULT_EMBED_MODEL,
                            ground_truth=st.session_state.ground_truth_input.strip() or None,
                        )
                except Exception as exc:
                    st.error(f"True RAGAS failed: {exc}")
            hit_views = [
                {
                    "source": str(chunk.source),
                    "score": score,
                    "preview": chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""),
                }
                for chunk, score in hits
            ]
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer if answer else "_No answer returned._",
                    "ragas": ragas_scores,
                    "hits": hit_views,
                }
            )
        # Optionally show index meta after each turn
        render_similarity_meta(index)

# Display chat history (render after any new messages are added so latest turn shows immediately)
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("ragas"):
                    r = msg["ragas"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(
                        "Faithfulness",
                        f"{r['faithfulness']:.3f}",
                        help="How well the answer is supported by the retrieved context.",
                    )
                    c2.metric(
                        "Answer Relevancy",
                        f"{r['answer_relevancy']:.3f}",
                        help="How directly the answer addresses the user question.",
                    )
                    c3.metric(
                        "Context Precision",
                        f"{r['context_precision']:.3f}",
                        help="Fraction of retrieved context that is actually useful for the answer.",
                    )
                    c4.metric(
                        "Context Recall",
                        f"{r['context_recall']:.3f}",
                        help="How completely the retrieved context covers what the question needs.",
                    )
                if msg.get("hits"):
                    with st.expander("Sources"):
                        for i, h in enumerate(msg["hits"], 1):
                            st.markdown(
                                f"**{i}. {Path(h['source']).name}** — score {h['score']:.3f}\n\n"
                                f"{h['preview']}"
                            )
