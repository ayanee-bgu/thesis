"""
Session-end consolidator: distils raw agent trajectories into the procedure graph.

Workflow (called once per session, after all trajectories are collected):
  1. For each trajectory, ask an LLM to extract step-level and script-level procedures.
  2. Deduplicate against existing graph nodes using TF-IDF cosine similarity.
  3. Detect conflicts: if two procedures share a precondition but prescribe different actions,
     add a conflicts_with edge and log the pair.
  4. Write new/merged nodes and edges to the ProcedureGraph.

Deduplication uses the same TF-IDF logic as rag_retrieval.py (no embedding model required).
Pass a sentence-transformer encode function via `embed_fn` for higher precision if available.
"""

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import requests

from src.memory.graph_store import ProcedureGraph

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 120

DISTIL_PROMPT = """You are a memory distillation expert. Given a task trajectory (a sequence of agent actions and observations), extract two types of procedural memory:

1. SCRIPT: A single high-level abstract procedure template (2-4 sentences) that captures the general strategy for this task type. Should be reusable across similar tasks.

2. STEPS: A numbered list of fine-grained concrete steps (each 1-2 sentences). Should capture the specific actions taken in order.

3. DEPENDENCIES: Which steps depend on which (e.g., step 3 requires step 2 to be done first). List as "step_N depends_on step_M".

4. CONFLICTS (optional): If this trajectory reveals a strategy that contradicts a known procedure, note it briefly.

Respond in this exact JSON format:
{
  "script": "<abstract procedure template>",
  "steps": ["<step 1>", "<step 2>", ...],
  "dependencies": ["step_2 depends_on step_1", ...],
  "conflicts_hint": "<optional: brief description of any contradiction with prior approaches, or null>"
}"""

DEDUP_THRESHOLD = 0.82  # TF-IDF cosine similarity above this → treat as duplicate

CHUNK_CHARS = 4000      # distillation call budget per chunk
CHUNK_OVERLAP_CHARS = 400  # shared context between consecutive chunks

_STEP_BOUNDARY = re.compile(r"\n(?=Step\s*\d+\s*:)", re.IGNORECASE)


def chunk_trajectory(text: str, chunk_chars: int = CHUNK_CHARS, overlap_chars: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """
    Split a trajectory into <=chunk_chars windows instead of hard-truncating it.
    Splits on "Step N:" boundaries when present (keeps a step intact rather than
    cutting mid-sentence); falls back to character windows otherwise. Consecutive
    chunks overlap by overlap_chars so a procedure spanning a boundary isn't lost
    to either half — the same fix applied to the retrieval-oracle's chunking
    (avoids the truncation failure mode that made full_context lose information
    past a fixed budget, here applied to what the consolidator ever sees).
    """
    if len(text) <= chunk_chars:
        return [text]

    boundaries = [m.start() for m in _STEP_BOUNDARY.finditer(text)]
    if not boundaries:
        # No step markers found — fall back to plain overlapping character windows.
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_chars
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap_chars
        return chunks

    segments = []
    prev = 0
    for b in boundaries:
        segments.append(text[prev:b])
        prev = b
    segments.append(text[prev:])

    chunks = []
    current = ""
    for seg in segments:
        if current and len(current) + len(seg) > chunk_chars:
            chunks.append(current)
            overlap_tail = current[-overlap_chars:] if overlap_chars else ""
            current = overlap_tail + seg
        else:
            current += seg
    if current:
        chunks.append(current)
    return chunks


_KEYWORD_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "was", "were", "are", "that", "this", "with", "by", "as", "from", "step",
    "task", "action", "observation", "query", "reference", "plan", "eval",
}


def extract_keywords(text: str, top_n: int = 6) -> list[str]:
    """
    Cheap TF top-N content words (no LLM call) — a coarse tag stored alongside
    the distilled content so ContextInjector can prefilter on keyword overlap
    before running similarity scoring, instead of scoring every node's full
    content text against the query every time.
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    freq: dict[str, int] = {}
    for t in tokens:
        if t in _KEYWORD_STOPWORDS or len(t) <= 2:
            continue
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in ranked[:top_n]]


@dataclass
class ConsolidationLog:
    session_id: str
    source_agent: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trajectories_processed: int = 0
    nodes_added: int = 0
    nodes_merged: int = 0
    nodes_deprecated: int = 0
    conflicts_detected: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "source_agent": self.source_agent,
            "timestamp": self.timestamp,
            "trajectories_processed": self.trajectories_processed,
            "nodes_added": self.nodes_added,
            "nodes_merged": self.nodes_merged,
            "nodes_deprecated": self.nodes_deprecated,
            "conflicts_detected": self.conflicts_detected,
            "errors": self.errors,
        }


class SessionConsolidator:
    def __init__(
        self,
        graph: ProcedureGraph,
        model: str = "llama3:8b",
        dedup_threshold: float = DEDUP_THRESHOLD,
        embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
    ):
        self.graph = graph
        self.model = model
        self.dedup_threshold = dedup_threshold
        self.embed_fn = embed_fn  # optional: sentence-transformer encode function

    def consolidate(
        self,
        trajectories: list[str],
        session_id: str,
        source_agent: str,
        graph_path: Optional[Path] = None,
        task_types: Optional[list[str]] = None,
    ) -> ConsolidationLog:
        """
        Process a list of trajectory strings from one session.
        Updates self.graph in-place and saves it to graph_path if provided.
        Returns a ConsolidationLog describing what changed.

        task_types, if given, is a parallel list (one per trajectory) of coarse
        tags (e.g. the source task id) attached to every node distilled from
        that trajectory — used by ContextInjector as a prefilter.
        """
        log = ConsolidationLog(session_id=session_id, source_agent=source_agent)

        for i, traj in enumerate(trajectories):
            log.trajectories_processed += 1
            task_type = task_types[i] if task_types and i < len(task_types) else ""
            try:
                extracted = self._distil_chunked(traj)
            except Exception as e:
                log.errors.append(f"Distillation failed: {e}")
                continue

            script_id = self._add_or_merge(
                content=extracted["script"],
                level="script",
                source_agent=source_agent,
                session_id=session_id,
                task_type=task_type,
                keywords=extract_keywords(extracted["script"] + " " + traj[:2000]),
                log=log,
            )

            step_ids = []
            for step_text in extracted.get("steps", []):
                sid = self._add_or_merge(
                    content=step_text,
                    level="step",
                    source_agent=source_agent,
                    session_id=session_id,
                    task_type=task_type,
                    keywords=extract_keywords(step_text),
                    log=log,
                )
                step_ids.append(sid)
                # script generalizes each step
                if script_id and not self.graph._g.has_edge(script_id, sid):
                    self.graph.add_edge(script_id, sid, "generalizes")

            # Wire step dependencies
            dep_pattern = re.compile(r"step_?(\d+)\s+depends_on\s+step_?(\d+)", re.IGNORECASE)
            for dep_str in extracted.get("dependencies", []):
                m = dep_pattern.search(dep_str)
                if m:
                    i, j = int(m.group(1)) - 1, int(m.group(2)) - 1
                    if 0 <= i < len(step_ids) and 0 <= j < len(step_ids):
                        src, dst = step_ids[i], step_ids[j]
                        if src != dst and not self.graph._g.has_edge(src, dst):
                            self.graph.add_edge(src, dst, "depends_on")

            # Conflict detection (heuristic: check hint against existing conflicts)
            hint = extracted.get("conflicts_hint")
            if hint:
                conflict_pairs = self._detect_conflicts(script_id, hint)
                for a, b in conflict_pairs:
                    if not self.graph._g.has_edge(a, b):
                        self.graph.add_edge(a, b, "conflicts_with")
                    log.conflicts_detected.append((a, b))

        if graph_path:
            self.graph.save(graph_path)

        return log

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _distil(self, trajectory: str) -> dict:
        """Call LLM to extract script + steps + dependencies from one <=CHUNK_CHARS trajectory chunk."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DISTIL_PROMPT},
                {"role": "user", "content": f"TRAJECTORY:\n{trajectory}"},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_gpu": 0},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return self._parse_json(content)

    def _distil_chunked(self, trajectory: str) -> dict:
        """
        Split long trajectories via chunk_trajectory() instead of hard-truncating
        at CHUNK_CHARS, distil each chunk separately, and merge the results —
        so procedures described past the old 4000-char cutoff are no longer
        silently dropped before the LLM ever sees them.
        """
        chunks = chunk_trajectory(trajectory)
        if len(chunks) == 1:
            return self._distil(chunks[0])

        scripts, steps, deps, conflict_hints = [], [], [], []
        for chunk in chunks:
            extracted = self._distil(chunk)
            if extracted.get("script"):
                scripts.append(extracted["script"])
            offset = len(steps)
            steps.extend(extracted.get("steps", []))
            for dep_str in extracted.get("dependencies", []):
                m = re.search(r"step_?(\d+)\s+depends_on\s+step_?(\d+)", dep_str, re.IGNORECASE)
                if m:
                    deps.append(f"step_{int(m.group(1)) + offset} depends_on step_{int(m.group(2)) + offset}")
            if extracted.get("conflicts_hint"):
                conflict_hints.append(extracted["conflicts_hint"])

        return {
            "script": " ".join(scripts) if scripts else "",
            "steps": steps,
            "dependencies": deps,
            "conflicts_hint": "; ".join(conflict_hints) if conflict_hints else None,
        }

    def _parse_json(self, text: str) -> dict:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in LLM response")
        return json.loads(text[start:end])

    def _add_or_merge(
        self,
        content: str,
        level: str,
        source_agent: str,
        session_id: str,
        log: ConsolidationLog,
        task_type: str = "",
        keywords: Optional[list[str]] = None,
    ) -> str:
        """Add a procedure node, or merge with an existing near-duplicate. Returns node_id."""
        existing = [n for n in self.graph.active_nodes() if n["level"] == level]
        if existing:
            best_id, best_score = self._best_match(content, existing)
            if best_score >= self.dedup_threshold:
                merged_content = self._merge_content(
                    self.graph.node_content(best_id), content
                )
                node = self.graph._g.nodes[best_id]
                node["content"] = merged_content
                node["reinforce_count"] = node.get("reinforce_count", 1) + 1
                if keywords:
                    node["keywords"] = sorted(set(node.get("keywords", [])) | set(keywords))
                log.nodes_merged += 1
                return best_id

        nid = self.graph.add_procedure(
            content, level, source_agent, session_id,
            task_type=task_type, keywords=keywords,
        )
        log.nodes_added += 1
        return nid

    def recluster(self) -> ConsolidationLog:
        """
        Graph-wide re-consolidation pass, independent of any single incoming
        trajectory: re-checks every active node against every other node at the
        same level and merges any pair the incremental per-trajectory dedup
        missed (e.g. two nodes added far apart in time, before the vocabulary
        overlap needed for TF-IDF similarity had accumulated). Intended to be run
        periodically (e.g. every N sessions) rather than after every trajectory,
        since it's O(n^2) in active node count.
        """
        log = ConsolidationLog(session_id="recluster", source_agent="recluster")
        for level in ("script", "step"):
            nodes = [n for n in self.graph.active_nodes() if n["level"] == level]
            i = 0
            while i < len(nodes):
                keep = nodes[i]
                rest = nodes[i + 1:]
                if not rest:
                    break
                best_id, best_score = self._best_match(keep["content"], rest)
                if best_score >= self.dedup_threshold and best_id:
                    merged_content = self._merge_content(
                        self.graph.node_content(keep["id"]), self.graph.node_content(best_id)
                    )
                    self.graph.merge_nodes(keep["id"], best_id, merged_content=merged_content)
                    log.nodes_merged += 1
                    nodes[i] = dict(nodes[i])
                    nodes[i]["content"] = merged_content
                    nodes = [n for n in nodes if n["id"] != best_id]
                    # Don't increment: re-check the merged node against remaining candidates
                else:
                    i += 1
        return log

    def _best_match(self, query: str, candidates: list[dict]) -> tuple[str, float]:
        """Return (node_id, similarity) of the most similar candidate."""
        if self.embed_fn:
            return self._embed_match(query, candidates)
        return self._tfidf_match(query, candidates)

    def _tfidf_match(self, query: str, candidates: list[dict]) -> tuple[str, float]:
        def tokenize(text: str) -> list[str]:
            return re.findall(r"[a-z0-9]+", text.lower())

        q_tokens = tokenize(query)
        all_tokens = [tokenize(c["content"]) for c in candidates]
        vocab = set(q_tokens)
        N = len(all_tokens) + 1

        def idf(term: str) -> float:
            df = sum(1 for toks in all_tokens if term in toks)
            return math.log((N + 1) / (df + 1)) + 1

        def tfidf_vec(tokens: list[str]) -> dict[str, float]:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            n = len(tokens) or 1
            return {t: (tf.get(t, 0) / n) * idf(t) for t in vocab}

        def cosine(a: dict, b: dict) -> float:
            dot = sum(a.get(t, 0) * b.get(t, 0) for t in b)
            na = math.sqrt(sum(v * v for v in a.values())) or 1
            nb = math.sqrt(sum(v * v for v in b.values())) or 1
            return dot / (na * nb)

        q_vec = tfidf_vec(q_tokens)
        best_id, best_score = "", 0.0
        for cand, toks in zip(candidates, all_tokens):
            score = cosine(q_vec, tfidf_vec(toks))
            if score > best_score:
                best_score = score
                best_id = cand["id"]
        return best_id, best_score

    def _embed_match(self, query: str, candidates: list[dict]) -> tuple[str, float]:
        """Use provided embed_fn for higher-quality similarity."""
        import numpy as np

        texts = [query] + [c["content"] for c in candidates]
        vecs = self.embed_fn(texts)
        q_vec = np.array(vecs[0])
        best_id, best_score = "", 0.0
        for i, cand in enumerate(candidates):
            c_vec = np.array(vecs[i + 1])
            score = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-9))
            if score > best_score:
                best_score = score
                best_id = cand["id"]
        return best_id, best_score

    def _merge_content(self, existing: str, new: str) -> str:
        """Simple merge: keep existing if very similar, else append new insight."""
        if len(new) > len(existing) * 1.2:
            return new
        return existing

    def _detect_conflicts(self, script_id: str, hint: str) -> list[tuple[str, str]]:
        """
        Heuristic conflict detection: look for existing script nodes with high
        similarity to the hint (which describes a contradictory approach).
        Returns pairs (new_script_id, conflicting_existing_id).
        """
        if not hint or not script_id:
            return []
        scripts = [n for n in self.graph.active_nodes() if n["level"] == "script" and n["id"] != script_id]
        if not scripts:
            return []
        best_id, score = self._tfidf_match(hint, scripts)
        if score >= 0.5 and best_id:
            return [(script_id, best_id)]
        return []
