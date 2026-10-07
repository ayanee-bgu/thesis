"""
Lightweight RAG retrieval tool for the Retriever agent.
Searches paper summaries using TF-IDF cosine similarity (no embedding model needed).
"""

import math
import re
from pathlib import Path


# Directories to search (relative to thesis root, resolved at query time)
DEFAULT_SEARCH_DIRS = [
    "papers/summaries",
    "knowledge_base",
]

MAX_CHUNKS = 3
CHUNK_CHARS = 800


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    n = len(tokens) or 1
    return {t: c / n for t, c in counts.items()}


def _idf(docs: list[list[str]], vocab: set[str]) -> dict[str, float]:
    N = len(docs)
    result = {}
    for term in vocab:
        df = sum(1 for doc in docs if term in doc)
        result[term] = math.log((N + 1) / (df + 1)) + 1
    return result


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a.get(t, 0) * b.get(t, 0) for t in b)
    norm_a = math.sqrt(sum(v * v for v in a.values())) or 1
    norm_b = math.sqrt(sum(v * v for v in b.values())) or 1
    return dot / (norm_a * norm_b)


def _load_documents(thesis_root: Path) -> list[dict]:
    docs = []
    for rel_dir in DEFAULT_SEARCH_DIRS:
        search_dir = thesis_root / rel_dir
        if not search_dir.exists():
            continue
        for path in search_dir.rglob("*.md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append({"path": path, "text": text, "tokens": _tokenize(text)})
    return docs


def retrieve(query: str, thesis_root: Path, top_k: int = MAX_CHUNKS) -> str:
    """
    Search all paper summaries and knowledge base files for the query.
    Returns a formatted string of the top-k most relevant chunks.
    """
    docs = _load_documents(thesis_root)
    if not docs:
        return "[rag_retrieval] No documents found in knowledge base."

    query_tokens = _tokenize(query)
    if not query_tokens:
        return "[rag_retrieval] Empty query."

    all_doc_tokens = [d["tokens"] for d in docs]
    vocab = set(query_tokens)
    idf = _idf(all_doc_tokens, vocab)

    def tfidf(tokens: list[str]) -> dict[str, float]:
        tf = _tf(tokens)
        return {t: tf.get(t, 0) * idf.get(t, 1) for t in vocab}

    q_vec = tfidf(query_tokens)
    scored = [(d, _cosine(q_vec, tfidf(d["tokens"]))) for d in docs]
    scored.sort(key=lambda x: x[1], reverse=True)

    top = scored[:top_k]
    if not top or top[0][1] == 0:
        return "[rag_retrieval] No relevant documents found."

    parts = []
    for doc, score in top:
        snippet = doc["text"][:CHUNK_CHARS].strip()
        parts.append(f"**{doc['path'].name}** (score={score:.3f})\n{snippet}")

    return "[rag_retrieval] Relevant documents:\n\n" + "\n\n---\n\n".join(parts)
