"""
Shared schema and storage for cross-session memory.

Five formats, each representing a different level of abstraction:

  F1_raw        verbatim session log  (question, queries, answer, correct)
  F2_fact       extracted factual claim  (doc-specific)
  F3_procedure  generalizable search strategy  (cross-doc, by question type)
  F4_program    parameterized query program with slot variables  (cross-doc, executable)
  F5_episode    compressed doc summary + character index  (doc-specific)

All formats expose a `content` string used for TF-IDF/BM25/embed indexing and
a `format_verbatim()` method that renders a compact human-readable block for
inclusion in an LLM prompt.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

FORMATS = ["F1_raw", "F2_fact", "F3_procedure", "F4_program", "F5_episode"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class MemoryNode:
    id: str
    format: str                    # one of FORMATS
    doc_kind: str                  # gutenberg | movie
    content: str                   # primary text indexed for retrieval
    meta: dict = field(default_factory=dict)
    source_sessions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @staticmethod
    def new(fmt: str, doc_kind: str, content: str, meta: dict, source_session: str) -> "MemoryNode":
        return MemoryNode(
            id=str(uuid.uuid4()),
            format=fmt,
            doc_kind=doc_kind,
            content=content,
            meta=meta,
            source_sessions=[source_session],
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MemoryNode":
        return MemoryNode(**d)

    def format_verbatim(self) -> str:
        """Compact human-readable rendering for inclusion in an LLM prompt."""
        header = f"[MEMORY/{self.format} | {self.doc_kind}]"
        m = self.meta

        if self.format == "F1_raw":
            queries = " → ".join(m.get("queries_issued", []))
            correct = "✓" if m.get("correct") else "✗"
            return (
                f"{header}\n"
                f"Q: {m.get('question', '')}\n"
                f"Searched: {queries}\n"
                f"A: {m.get('answer', '')} [{correct}]"
            )

        if self.format == "F2_fact":
            conf = m.get("confidence", 0)
            return (
                f"{header} | {m.get('doc_title', '')} | conf={conf:.0%}\n"
                f"{self.content}"
            )

        if self.format == "F3_procedure":
            sr = m.get("success_rate")
            sr_str = f" | success={sr:.0%}" if sr is not None else ""
            fail = m.get("failure_modes", "")
            fail_str = f"\nAvoid: {fail}" if fail else ""
            return (
                f"{header} | {m.get('question_type', '')}{sr_str}\n"
                f"{self.content}{fail_str}"
            )

        if self.format == "F4_program":
            slots = ", ".join(m.get("slots", []))
            queries = " | ".join(m.get("query_sequence", []))
            sr = m.get("success_rate")
            sr_str = f" | success={sr:.0%}" if sr is not None else ""
            return (
                f"{header} | {m.get('name', '')}{sr_str}\n"
                f"Slots: {slots}\n"
                f"Queries: {queries}"
            )

        if self.format == "F5_episode":
            chars = "; ".join(f"{k}: {v}" for k, v in list(m.get("characters", {}).items())[:5])
            return (
                f"{header} | {m.get('doc_title', '')}\n"
                f"{self.content}\n"
                f"Characters: {chars}"
            )

        return f"{header}\n{self.content}"


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

class MemoryPool:
    """Append-only store of MemoryNodes, persisted as per-format JSONL files.

    build_index() pre-computes a retrieval index over a format's nodes so
    query() can be called cheaply at eval time.
    """

    def __init__(self, pool_dir: Path):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self._nodes: list[MemoryNode] = []
        self._indexes: dict[str, Any] = {}

    # ── persistence ─────────────────────────────────────────────────────────

    def add(self, node: MemoryNode) -> None:
        self._nodes.append(node)

    def flush(self) -> None:
        """Append all in-memory nodes to their format-specific JSONL files."""
        buckets: dict[str, list[MemoryNode]] = {}
        for n in self._nodes:
            buckets.setdefault(n.format, []).append(n)
        for fmt, nodes in buckets.items():
            path = self.pool_dir / f"{fmt}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                for n in nodes:
                    f.write(json.dumps(n.to_dict()) + "\n")
        self._nodes.clear()

    def load(self, formats: list[str] | None = None) -> None:
        """Load nodes from disk into memory.  Call once before querying."""
        self._nodes.clear()
        targets = formats or FORMATS
        for fmt in targets:
            path = self.pool_dir / f"{fmt}.jsonl"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._nodes.append(MemoryNode.from_dict(json.loads(line)))

    def nodes_for(self, fmt: str, doc_kind: str | None = None) -> list[MemoryNode]:
        nodes = [n for n in self._nodes if n.format == fmt]
        if doc_kind:
            nodes = [n for n in nodes if n.doc_kind == doc_kind]
        return nodes

    def nodes_for_formats(self, fmts: list[str], doc_kind: str | None = None) -> list[MemoryNode]:
        out = []
        for fmt in fmts:
            out.extend(self.nodes_for(fmt, doc_kind))
        return out

    def stats(self) -> dict:
        counts = {}
        for fmt in FORMATS:
            counts[fmt] = sum(1 for n in self._nodes if n.format == fmt)
        return {"total": len(self._nodes), "by_format": counts}

    # ── index building ───────────────────────────────────────────────────────

    def build_index(
        self,
        nodes: list[MemoryNode],
        index_type: str = "tfidf",
        embed_fn: Callable | None = None,
    ) -> "MemoryIndex":
        if index_type == "bm25":
            return BM25MemoryIndex(nodes)
        if index_type == "embed":
            return EmbedMemoryIndex(nodes, embed_fn)
        return TFIDFMemoryIndex(nodes)


# ---------------------------------------------------------------------------
# Retrieval indices over MemoryNode lists
# ---------------------------------------------------------------------------

def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TFIDFMemoryIndex:
    def __init__(self, nodes: list[MemoryNode]):
        self.nodes = nodes
        self._doc_tokens = [_tok(n.content) for n in nodes]
        self._N = len(nodes) + 1

    def _idf(self, term: str) -> float:
        df = sum(1 for toks in self._doc_tokens if term in toks)
        return math.log((self._N + 1) / (df + 1)) + 1

    def _vec(self, tokens: list[str], vocab: set[str]) -> dict[str, float]:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n = len(tokens) or 1
        return {t: (tf.get(t, 0) / n) * self._idf(t) for t in vocab}

    @staticmethod
    def _cos(a: dict, b: dict) -> float:
        dot = sum(a.get(t, 0) * b.get(t, 0) for t in b)
        na = math.sqrt(sum(v * v for v in a.values())) or 1
        nb = math.sqrt(sum(v * v for v in b.values())) or 1
        return dot / (na * nb)

    def query(self, text: str, top_k: int = 1) -> list[tuple[MemoryNode, float]]:
        q_toks = _tok(text)
        vocab = set(q_toks)
        if not vocab or not self.nodes:
            return [(self.nodes[0], 0.0)] if self.nodes else []
        q_vec = self._vec(q_toks, vocab)
        scored = [(self._cos(q_vec, self._vec(dt, vocab)), i) for i, dt in enumerate(self._doc_tokens)]
        scored.sort(reverse=True)
        return [(self.nodes[i], s) for s, i in scored[:top_k]]


class BM25MemoryIndex:
    def __init__(self, nodes: list[MemoryNode]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("pip install rank_bm25")
        self.nodes = nodes
        self._bm25 = BM25Okapi([_tok(n.content) for n in nodes])

    def query(self, text: str, top_k: int = 1) -> list[tuple[MemoryNode, float]]:
        import numpy as np
        q_toks = _tok(text)
        if not q_toks or not self.nodes:
            return [(self.nodes[0], 0.0)] if self.nodes else []
        scores = self._bm25.get_scores(q_toks)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(self.nodes[int(i)], float(scores[i])) for i in idxs]


class EmbedMemoryIndex:
    def __init__(self, nodes: list[MemoryNode], embed_fn: Callable):
        import numpy as np
        self.nodes = nodes
        self._embed_fn = embed_fn
        if nodes:
            vecs = np.array(embed_fn([n.content for n in nodes]))
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            self._vecs = vecs / np.maximum(norms, 1e-9)
        else:
            self._vecs = np.zeros((0, 384))

    def query(self, text: str, top_k: int = 1) -> list[tuple[MemoryNode, float]]:
        import numpy as np
        if not self.nodes:
            return []
        q = np.array(self._embed_fn([text])[0])
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self._vecs @ q
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(self.nodes[int(i)], float(scores[i])) for i in idxs]


# ---------------------------------------------------------------------------
# Verbatim context formatter
# ---------------------------------------------------------------------------

def format_verbatim_context(nodes: list[MemoryNode], max_words: int) -> str:
    """Format up to max_words of memory nodes as a readable context block."""
    if not nodes:
        return ""
    lines = []
    word_count = 0
    for node in nodes:
        block = node.format_verbatim()
        block_words = len(block.split())
        if word_count + block_words > max_words:
            break
        lines.append(block)
        word_count += block_words
    return "\n\n".join(lines)
