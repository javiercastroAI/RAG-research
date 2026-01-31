#!/usr/bin/env python3
"""
Lightweight RAG CLI for documents in ~/Downloads/rag.

Features
--------
- Indexes PDFs / text / Markdown / notebooks into cached embeddings.
- Uses SentenceTransformers (local) by default; OpenAI completions optional for answers.
- Modes: --index (build cache), --ask "question", --chat (interactive loop).

Examples
--------
$ python rag_downloads.py --index
$ python rag_downloads.py --ask "What does the XXXX report say about AI risks?"
$ python rag_downloads.py --chat
"""

import argparse
import json
import os
import pickle
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any
import hashlib
import itertools

import numpy as np


def _load_env():
    """Lightweight .env loader (no dependency on python-dotenv)."""
    env_paths = [
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
        Path.home() / ".env",
    ]
    for env_path in env_paths:
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


# Load .env early so defaults pick up values
_load_env()

# Third‑party, small and already available/installed in this environment
from sentence_transformers import SentenceTransformer

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - runtime guard
    PdfReader = None
    PDF_IMPORT_ERROR = exc
else:
    PDF_IMPORT_ERROR = None

# Optional OpenAI generation (only used if API key is present)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover - runtime guard
    OpenAI = None


DEFAULT_DOC_DIR = Path.home() / "Downloads" / "rag"
DEFAULT_INDEX_PATH = Path(__file__).parent / ".rag_downloads.index.pkl"
DEFAULT_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "all-MiniLM-L6-v2")
# OpenAI-backed defaults for generation and embeddings (used by optional RAGAS flow)
DEFAULT_COMPLETION_MODEL = os.getenv("RAG_COMPLETION_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
MAX_CONTEXT_CHARS = 6000  # cap total characters sent to LLM context
DEFAULT_SIMILAR_THRESHOLD = 0.92


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def iter_files(doc_dir: Path) -> Iterable[Path]:
    exts = {".pdf", ".txt", ".md", ".ipynb"}
    for path in sorted(doc_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def resolve_doc_dir(doc_dir: Path) -> Path:
    """
    Make doc_dir resilient to common typos like 'dowloads/rag'.
    Tries a few fallbacks before failing with a helpful message.
    """
    candidates = []
    # User provided
    candidates.append(doc_dir)
    # Normalize tilde / relative
    candidates.append(doc_dir.expanduser().resolve())
    # Common misspelling
    candidates.append(Path.home() / "Downloads" / "rag")
    candidates.append(Path.home() / "downloads" / "rag")
    candidates.append(Path.cwd() / "Downloads" / "rag")
    candidates.append(Path.cwd() / "downloads" / "rag")
    for candidate in candidates:
        try:
            if candidate.is_dir():
                return candidate
        except Exception:
            continue
    raise FileNotFoundError(
        f"Document directory not found: {doc_dir}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def file_md5(path: Path, block_size: int = 1 << 20) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def dedup_exact(files: List[Path]) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """Drop exact duplicates by hash, keep newest mtime."""
    kept: Dict[str, Path] = {}
    info: Dict[str, float] = {}
    dropped = []
    for f in files:
        h = file_md5(f)
        mtime = f.stat().st_mtime
        if h not in kept:
            kept[h] = f
            info[h] = mtime
        else:
            if mtime > info[h]:
                dropped.append({"duplicate_of": str(kept[h]), "dropped": str(kept[h]), "kept": str(f), "reason": "newer duplicate"})
                kept[h] = f
                info[h] = mtime
            else:
                dropped.append({"duplicate_of": str(kept[h]), "dropped": str(f), "kept": str(kept[h]), "reason": "older duplicate"})
    return list(kept.values()), dropped


def find_semantic_similars(
    paths: List[Path],
    texts: List[str],
    model: SentenceTransformer,
    threshold: float,
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    if not paths:
        return [], np.array([])
    doc_embs = model.encode(texts, batch_size=8, convert_to_numpy=True)
    sims = cosine_sim(doc_embs, doc_embs)
    pairs = []
    for i, j in itertools.combinations(range(len(paths)), 2):
        score = sims[i, j]
        if score >= threshold:
            pairs.append((i, j, float(score)))
    return pairs, doc_embs


def _normalize_index_path(index_path: Path) -> Path:
    """Allow users to pass a directory; turn it into a file path."""
    if index_path.is_dir():
        index_path = index_path / ".rag_downloads.index.pkl"
    # If a trailing slash was given and the path doesn't exist yet, assume directory intent
    if str(index_path).endswith(os.sep):
        index_path = Path(str(index_path).rstrip(os.sep)) / ".rag_downloads.index.pkl"
    # Ensure parent exists
    index_path.parent.mkdir(parents=True, exist_ok=True)
    return index_path


def read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError(
            f"pypdf is required for PDF support. Install via `python3 -m pip install --user pypdf` "
            f"(import error: {PDF_IMPORT_ERROR})"
        )
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def read_ipynb(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path.read_text(errors="ignore")
    cells = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "markdown":
            cells.append("".join(cell.get("source", [])))
        elif cell.get("cell_type") == "code":
            cells.append("".join(cell.get("source", [])))
    return "\n".join(cells)


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    if path.suffix.lower() == ".ipynb":
        return read_ipynb(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    clean = " ".join(text.split())
    if len(clean) <= chunk_chars:
        return [clean]
    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_chars, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(end - overlap, 0)
    return chunks


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, keepdims=True) + 1e-9)
    return np.dot(a_norm, b_norm.T)


def build_index(
    doc_dir: Path,
    index_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    chunk_chars: int = CHUNK_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP,
    similar_threshold: float = DEFAULT_SIMILAR_THRESHOLD,
    auto_drop_similar: bool = False,
) -> dict:
    index_path = _normalize_index_path(index_path)
    doc_dir = resolve_doc_dir(doc_dir)

    model = get_embedder(model_name)
    print(f"[index] scanning {doc_dir} ...")
    files = list(iter_files(doc_dir))
    files, exact_dropped = dedup_exact(files)
    if exact_dropped:
        print(f"[index] removed {len(exact_dropped)} exact duplicates; kept newest versions.")

    texts: List[str] = []
    kept_files: List[Path] = []
    for fpath in files:
        try:
            text = read_text(fpath)
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"[warn] skip {fpath.name}: {exc}")
            continue
        texts.append(text)
        kept_files.append(fpath)

    if not kept_files:
        raise RuntimeError("No indexable documents found.")

    similar_pairs, doc_embs = find_semantic_similars(kept_files, texts, model, similar_threshold)
    auto_drop_notes = []
    if similar_pairs:
        print(f"[index] flagged {len(similar_pairs)} semantically similar pairs (>= {similar_threshold}).")
        if auto_drop_similar:
            # drop older file in each pair
            to_drop = set()
            for i, j, _ in similar_pairs:
                fi, fj = kept_files[i], kept_files[j]
                newer, older = (fi, fj) if fi.stat().st_mtime >= fj.stat().st_mtime else (fj, fi)
                to_drop.add(older)
                auto_drop_notes.append({"kept": str(newer), "dropped": str(older), "reason": "semantic similar auto-drop"})
            kept_files = [f for f in kept_files if f not in to_drop]
            texts = [t for f, t in zip(kept_files, texts) if f in kept_files]
            print(f"[index] auto-dropped {len(to_drop)} similar files (kept newest).")

    chunks: List[Chunk] = []
    for fpath, text in zip(kept_files, texts):
        for idx, piece in enumerate(chunk_text(text, chunk_chars, chunk_overlap)):
            chunks.append(Chunk(text=piece, source=str(fpath), chunk_id=idx))
    if not chunks:
        raise RuntimeError("No indexable documents found after processing.")

    print(f"[index] embedding {len(chunks)} chunks with {model_name} ...")
    embeddings = model.encode([c.text for c in chunks], batch_size=16, show_progress_bar=True, convert_to_numpy=True)

    meta = {
        "model_name": model_name,
        "doc_dir": str(doc_dir),
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "files": {str(p): {"mtime": p.stat().st_mtime, "size": p.stat().st_size} for p in kept_files},
        "exact_duplicates_removed": exact_dropped,
        "semantic_similar_pairs": [
            {"a": str(kept_files[i]), "b": str(kept_files[j]), "score": score} for i, j, score in similar_pairs
        ],
        "semantic_auto_dropped": auto_drop_notes,
    }
    payload = {"meta": meta, "chunks": chunks, "embeddings": embeddings}
    index_path.write_bytes(pickle.dumps(payload))
    print(f"[index] saved -> {index_path}")
    return payload


def load_index(index_path: Path) -> dict:
    index_path = _normalize_index_path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    return pickle.loads(index_path.read_bytes())


def maybe_rebuild(index_path: Path, doc_dir: Path, model_name: str) -> dict:
    index_path = _normalize_index_path(index_path)
    doc_dir = resolve_doc_dir(doc_dir)
    try:
        data = load_index(index_path)
    except FileNotFoundError:
        print("[index] cache missing; building fresh...")
        return build_index(doc_dir, index_path, model_name)

    # Check for stale files (simple mtime/size check)
    stale = False
    current_files = {str(p): (p.stat().st_mtime, p.stat().st_size) for p in iter_files(doc_dir)}
    cached_files = {k: (v["mtime"], v["size"]) for k, v in data["meta"].get("files", {}).items()}
    if set(current_files.keys()) != set(cached_files.keys()):
        stale = True
    else:
        for path, info in current_files.items():
            if info != cached_files.get(path):
                stale = True
                break

    if stale or data["meta"].get("model_name") != model_name:
        print("[index] cache stale; rebuilding...")
        return build_index(doc_dir, index_path, model_name)
    return data


_EMBEDDERS = {}


def get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _EMBEDDERS:
        _EMBEDDERS[model_name] = SentenceTransformer(model_name)
    return _EMBEDDERS[model_name]


def top_k(query: str, index: dict, k: int = 5, model_name: str = DEFAULT_MODEL_NAME) -> List[Tuple[Chunk, float]]:
    model = get_embedder(model_name)
    q_emb = model.encode([query], convert_to_numpy=True)
    sims = cosine_sim(index["embeddings"], q_emb)[..., 0]
    top_ids = sims.argsort()[::-1][:k]
    return [(index["chunks"][i], float(sims[i])) for i in top_ids]


def _compact_hits(hits: List[Tuple[Chunk, float]], max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[List[Chunk], List[float]]:
    selected_chunks: List[Chunk] = []
    selected_scores: List[float] = []
    budget = max_chars
    for chunk, score in hits:
        needed = len(chunk.text)
        if needed > budget and selected_chunks:
            break
        selected_chunks.append(chunk)
        selected_scores.append(score)
        budget -= needed
        if budget <= 0:
            break
    return selected_chunks, selected_scores


def _extractive_summary(question: str, contexts: List[Chunk], scores: List[float]) -> str:
    if not contexts:
        return ""
    lines = [f"No LLM configured; showing top evidence for: {question}"]
    for idx, (chunk, score) in enumerate(zip(contexts, scores), start=1):
        snippet = textwrap.shorten(chunk.text, width=400, placeholder=" ...")
        lines.append(f"[{idx}] {Path(chunk.source).name} (score {score:.3f}): {snippet}")
    return "\n".join(lines)


def generate_answer(
    question: str,
    contexts: List[Chunk],
    scores: List[float],
    llm_model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    if OpenAI is None or os.getenv("OPENAI_API_KEY") is None:
        return _extractive_summary(question, contexts, scores)
    client = OpenAI()
    context_block = "\n\n".join(
        f"[{i+1}] {Path(c.source).name} (chunk {c.chunk_id}, score {scores[i]:.3f}):\n{c.text}"
        for i, c in enumerate(contexts)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise research assistant. Use only the provided context. "
                "Cite the bracket numbers when referencing evidence. If information is insufficient, say so."
            ),
        },
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context_block}"},
    ]
    try:
        resp = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"[warn] OpenAI completion failed: {exc}")
        return ""
    return resp.choices[0].message.content.strip()


def evaluate_ragas_true(
    question: str,
    answer: str,
    contexts: List[Chunk],
    llm_model: str = DEFAULT_COMPLETION_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    ground_truth: str = None,
) -> dict:
    """
    Compute official RAGAS metrics using OpenAI-backed LLM and embeddings.

    Requirements:
      - OPENAI_API_KEY set in the environment.
      - ragas, datasets, and langchain-openai installed.
    """
    if not contexts or not answer:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }
    if os.getenv("OPENAI_API_KEY") is None:
        raise RuntimeError("OPENAI_API_KEY is required for true RAGAS metrics.")

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "Install ragas, datasets, and langchain-openai to compute true RAGAS metrics."
        ) from exc

    # If caller did not supply reference text, use the produced answer as a best-effort proxy.
    gt_text = ground_truth if ground_truth is not None else answer

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [[c.text for c in contexts]],
            "ground_truth": [gt_text],
        }
    )

    llm = ChatOpenAI(model=llm_model, temperature=0)
    embeddings = OpenAIEmbeddings(model=embed_model)

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            is_async=False,
            raise_exceptions=False,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(f"RAGAS evaluation failed: {exc}")

    scores = {}
    wanted = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    for key in wanted:
        val = result.get(key)
        try:
            if isinstance(val, (list, tuple)):
                val = val[0] if val else None
            if val is None:
                continue
            scores[key] = float(val)
        except Exception:
            continue

    for key in wanted:
        scores.setdefault(key, 0.0)
    return scores


def print_hits(hits: List[Tuple[Chunk, float]]):
    for idx, (chunk, score) in enumerate(hits, start=1):
        print("\n" + "-" * 80)
        print(f"[{idx}] {Path(chunk.source).name}  (chunk {chunk.chunk_id})  score={score:.3f}")
        preview = textwrap.shorten(chunk.text, width=800, placeholder=" ...")
        print(preview)
    print("\n" + "-" * 80)


def interactive_loop(
    index: dict,
    model_name: str,
    max_context: int = MAX_CONTEXT_CHARS,
    llm_model: str = os.getenv("RAG_COMPLETION_MODEL", "gpt-4o-mini"),
    temperature: float = 0.2,
    max_tokens: int = 400,
):
    print("Chat mode. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        hits = top_k(question, index, k=5, model_name=model_name)
        compact_chunks, compact_scores = _compact_hits(hits, max_chars=max_context)
        answer = generate_answer(
            question,
            compact_chunks,
            compact_scores,
            llm_model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if answer:
            print("\nAnswer:\n" + answer)
        print_hits(hits)


def cli():
    parser = argparse.ArgumentParser(description="RAG over ~/Downloads/rag")
    parser.add_argument("--doc-dir", type=Path, default=DEFAULT_DOC_DIR, help="Folder with source docs")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH, help="Cache file path")
    parser.add_argument("--index", action="store_true", help="Rebuild the index cache")
    parser.add_argument("--ask", type=str, help="Single question to answer")
    parser.add_argument("--chat", action="store_true", help="Interactive Q&A loop")
    parser.add_argument("--top-k", type=int, default=5, help="Results to return")
    parser.add_argument("--max-context", type=int, default=MAX_CONTEXT_CHARS, help="Max characters of context sent to LLM")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name")
    parser.add_argument("--llm-model", type=str, default=os.getenv("RAG_COMPLETION_MODEL", "gpt-4o-mini"), help="OpenAI chat model")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=400, help="LLM max tokens")
    parser.add_argument("--similar-threshold", type=float, default=DEFAULT_SIMILAR_THRESHOLD, help="Cosine threshold to flag semantically similar documents")
    parser.add_argument("--auto-drop-similar", action="store_true", help="Automatically drop older doc in similar pairs")
    args = parser.parse_args()

    if args.index:
        build_index(
            args.doc_dir,
            args.index_path,
            args.model,
            similar_threshold=args.similar_threshold,
            auto_drop_similar=args.auto_drop_similar,
        )
        if not (args.ask or args.chat):
            return

    index = maybe_rebuild(args.index_path, args.doc_dir, args.model)

    if args.ask:
        hits = top_k(args.ask, index, k=args.top_k, model_name=args.model)
        compact_chunks, compact_scores = _compact_hits(hits, max_chars=args.max_context)
        answer = generate_answer(
            args.ask,
            compact_chunks,
            compact_scores,
            llm_model=args.llm_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if answer:
            print("\nAnswer:\n" + answer)
        print_hits(hits)
    elif args.chat:
        interactive_loop(
            index,
            args.model,
            max_context=args.max_context,
            llm_model=args.llm_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
