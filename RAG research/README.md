# dnAI GPT for RAG research

Author: Javier Castro — javiercastro@aiready.es

A lightweight Retrieval-Augmented Generation (RAG) toolkit for your `~/Downloads/rag` folder. It ships with:
- **ChatGPT-style UI** (`streamlit_rag.py`) with citations, RAGAS metrics, and optional OpenAI completions.
- **CLI utilities** (`rag_downloads.py`) to index, chat, or ask one-off questions.
- **Convenience launcher** (`./rag`) to rebuild the index and start chatting.

## Contents
- `rag_downloads.py` — indexing, retrieval, and answer generation.
- `streamlit_rag.py` — Streamlit front-end.
- `rag` — shell wrapper: `./rag --chat` (rebuilds index then starts chat).
- `.rag_downloads.index.pkl` — cached embeddings (auto-generated).

## Requirements
- Python 3.9+
- Python packages: `sentence-transformers`, `pypdf`, `openai`, `streamlit`, `ragas`, `datasets`, `langchain-openai`, `numpy`.
  ```bash
  python3 -m pip install --user sentence-transformers pypdf openai streamlit ragas datasets langchain-openai
  ```
- Documents located in `~/Downloads/rag` (default). Supported: PDF, TXT, MD, IPYNB.
- Optional: `OPENAI_API_KEY` for LLM answers and true RAGAS metrics.

## Quick start
### Streamlit UI (ChatGPT-like)
```bash
cd "RAG research"
streamlit run streamlit_rag.py
```
- Type in the bottom chat box; answers, RAGAS metrics (if enabled), and source previews appear immediately.
- Toggle “Compute RAGAS metrics” in the sidebar (requires `OPENAI_API_KEY`).
- Optional ground-truth box improves Context Recall scoring.

### CLI
```bash
cd "RAG research"
python3 rag_downloads.py --index              # build or refresh index
python3 rag_downloads.py --ask "Question?"    # single-shot answer
python3 rag_downloads.py --chat               # interactive loop
# Or with wrapper:
./rag --chat
```

## Configuration
- `OPENAI_API_KEY` — enables LLM answers and true RAGAS.
- `RAG_COMPLETION_MODEL` — OpenAI chat model (default: `gpt-4o-mini`).
- `RAG_EMBED_MODEL` — embedding model for RAGAS (default: `text-embedding-3-small`).
- `RAG_MODEL_NAME` — SentenceTransformer for indexing/search (default: `all-MiniLM-L6-v2`).
- `--doc-dir` and `--index-path` CLI flags override defaults.

## RAGAS metrics
When enabled, the app computes:
- **Faithfulness** — answer supported by retrieved context.
- **Answer Relevancy** — answer addresses the user question.
- **Context Precision** — fraction of retrieved context that was useful.
- **Context Recall** — how completely context covers what the question needs (better with ground truth provided).

## Data & caching
- Index cache: `.rag_downloads.index.pkl` (auto-managed).
- Duplicate handling: exact dupes dropped; semantic similarity optionally deduped at index build.

## Development
- Lint/format: not enforced; keep code ASCII and small helper comments only when needed.
- Tests: none included; you can run ad-hoc checks via `python3 -m py_compile *.py`.

## Security & privacy
- Secrets: keep `OPENAI_API_KEY` in environment or a local `.env` that is *not* committed.
- Documents stay local; retrieval uses your local embeddings and optional OpenAI calls for generation/scoring.
- See `SECURITY.md` for reporting and hardening notes.

## License
MIT License. See `LICENSE`.
