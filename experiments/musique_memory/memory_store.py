"""
Memory schema for the MuSiQue procedural memory experiment.

Seven formats representing increasing levels of abstraction over past sessions:

  F1_raw        verbatim oracle session log
  F2_chain      gold question-decomposition chain  (no LLM needed — from dataset)
  F3_procedure  generalizable natural-language strategy per question type
  F4_program    parameterized query program with slot variables
  F5_decomp     canonical decomposition pattern for a question-type cluster
  F6_preference approach preferences learned from successful sessions
  F7_correction common mistakes and corrections extracted from failed pivots

Key difference from E11 (NarrativeQA): here EVERY format can potentially
transfer across questions because the unit of knowledge is a *reasoning
procedure*, not a document fact.  F2_chain is derived from the gold
decomposition, making it a near-oracle ceiling on procedural memory.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

FORMATS = ["F1_raw", "F2_chain", "F3_procedure", "F4_program", "F5_decomp",
           "F6_preference", "F7_correction"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class MemoryNode:
    id: str
    format: str
    hop: int                       # 2 | 3 | 4  — used for hop-filtered retrieval
    content: str                   # primary text indexed for retrieval
    meta: dict = field(default_factory=dict)
    source_sessions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @staticmethod
    def new(fmt: str, hop: int, content: str, meta: dict, source_session: str) -> "MemoryNode":
        return MemoryNode(
            id=str(uuid.uuid4()),
            format=fmt, hop=hop, content=content, meta=meta,
            source_sessions=[source_session],
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MemoryNode":
        return MemoryNode(**d)

    def format_verbatim(self) -> str:
        m = self.meta
        header = f"[MEMORY/{self.format} | hop{self.hop}]"

        if self.format == "F1_raw":
            steps = " → ".join(m.get("queries", []))
            correct = "✓" if m.get("correct") else "✗"
            return (
                f"{header}\n"
                f"Q: {m.get('question', '')}\n"
                f"Queries: {steps}\n"
                f"A: {m.get('answer', '')} [{correct}]"
            )

        if self.format == "F2_chain":
            return (
                f"{header} | {m.get('question_type', '')}\n"
                f"Q: {m.get('question', '')}\n"
                + self.content
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

        if self.format == "F5_decomp":
            examples = "; ".join(m.get("example_questions", [])[:2])
            return (
                f"{header} | {m.get('question_type', '')}\n"
                f"Pattern: {self.content}\n"
                f"Examples: {examples}"
            )

        if self.format == "F6_preference":
            sr = m.get("success_rate")
            sr_str = f" | success={sr:.0%}" if sr is not None else ""
            avoid = m.get("avoid", "")
            avoid_str = f"\nAvoid: {avoid}" if avoid else ""
            return (
                f"{header} | {m.get('question_type', '')}{sr_str}\n"
                f"Preferred approach: {self.content}{avoid_str}"
            )

        if self.format == "F7_correction":
            wrong = m.get("wrong_pattern", "")
            return (
                f"{header} | {m.get('question_type', '')}\n"
                f"Common mistake: {wrong}\n"
                f"Correction: {self.content}"
            )

        return f"{header}\n{self.content}"


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

class MemoryPool:
    def __init__(self, pool_dir: Path):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self._nodes: list[MemoryNode] = []

    def add(self, node: MemoryNode) -> None:
        self._nodes.append(node)

    def flush(self) -> None:
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
        self._nodes.clear()
        for fmt in (formats or FORMATS):
            path = self.pool_dir / f"{fmt}.jsonl"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._nodes.append(MemoryNode.from_dict(json.loads(line)))

    def nodes_for(self, fmt: str, hop: int | None = None) -> list[MemoryNode]:
        nodes = [n for n in self._nodes if n.format == fmt]
        if hop is not None:
            nodes = [n for n in nodes if n.hop == hop]
        return nodes

    def nodes_for_formats(self, fmts: list[str], hop: int | None = None) -> list[MemoryNode]:
        out = []
        for fmt in fmts:
            out.extend(self.nodes_for(fmt, hop))
        return out

    def stats(self) -> dict:
        by_fmt = {fmt: sum(1 for n in self._nodes if n.format == fmt) for fmt in FORMATS}
        return {"total": len(self._nodes), "by_format": by_fmt}

    def build_index(self, nodes: list[MemoryNode], index_type: str = "tfidf",
                    embed_fn=None) -> "TFIDFMemoryIndex":
        return TFIDFMemoryIndex(nodes)


# ---------------------------------------------------------------------------
# Retrieval index
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

    def query(self, text: str, top_k: int = 1,
              success_weight: float = 0.0) -> list[tuple[MemoryNode, float]]:
        """
        Retrieve top_k nodes by composite score:
          score = (1 - success_weight) * tfidf_similarity + success_weight * success_rate
        success_weight=0 is pure TF-IDF; success_weight=0.3 mixes in historical performance.
        """
        q_toks = _tok(text)
        vocab = set(q_toks)
        if not vocab or not self.nodes:
            return [(self.nodes[0], 0.0)] if self.nodes else []
        q_vec = self._vec(q_toks, vocab)
        scored = []
        for i, dt in enumerate(self._doc_tokens):
            sim = self._cos(q_vec, self._vec(dt, vocab))
            sr = self.nodes[i].meta.get("success_rate") or 0.0
            composite = (1.0 - success_weight) * sim + success_weight * sr
            scored.append((composite, i))
        scored.sort(reverse=True)
        return [(self.nodes[i], s) for s, i in scored[:top_k]]


# ---------------------------------------------------------------------------
# Verbatim formatter
# ---------------------------------------------------------------------------

def format_verbatim_context(nodes: list[MemoryNode], max_words: int) -> str:
    if not nodes:
        return ""
    lines, wc = [], 0
    for node in nodes:
        block = node.format_verbatim()
        bw = len(block.split())
        if wc + bw > max_words:
            break
        lines.append(block)
        wc += bw
    return "\n\n".join(lines)
