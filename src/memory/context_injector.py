"""
Context injector: retrieves a relevant subgraph from the ProcedureGraph and
formats it as a structured prompt prefix for agent context injection.

Retrieval strategy:
  1. Find seed nodes: score all active nodes against the task description
     using TF-IDF cosine similarity (or embed_fn if provided).
  2. Expand: traverse k-hop neighbourhood of top-N seeds.
  3. Rank: weight nodes by (similarity_score × edge_depth_discount).
  4. Budget: include nodes until token_budget characters are consumed.
  5. Format: render as a structured block distinguishing scripts from steps,
     with dependency chains shown inline.
"""

import math
import re
from pathlib import Path
from typing import Callable, Optional

import networkx as nx

from src.memory.graph_store import ProcedureGraph

DEFAULT_K_HOPS = 2
DEFAULT_TOP_SEEDS = 5
DEFAULT_TOKEN_BUDGET = 2000  # approximate character budget for injected context
DEPTH_DISCOUNT = 0.7  # score multiplier per hop away from seed
REINFORCE_WEIGHT = 0.15  # per-doubling score bonus for nodes reinforced across sessions


class ContextInjector:
    def __init__(
        self,
        graph: ProcedureGraph,
        k_hops: int = DEFAULT_K_HOPS,
        top_seeds: int = DEFAULT_TOP_SEEDS,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
        use_keyword_prefilter: bool = False,
        use_reinforcement_weighting: bool = False,
    ):
        self.graph = graph
        self.k_hops = k_hops
        self.top_seeds = top_seeds
        self.token_budget = token_budget
        self.embed_fn = embed_fn
        self.use_keyword_prefilter = use_keyword_prefilter
        self.use_reinforcement_weighting = use_reinforcement_weighting

    def inject(self, task_description: str, task_type: str = "") -> str:
        """
        Return a formatted context string to prepend to the agent's prompt.
        Returns an empty string if the graph is empty.

        If use_keyword_prefilter is set, nodes are first narrowed to those
        matching task_type or sharing a keyword with task_description before
        similarity scoring — a cheap grounding step (mirrors the retrieval-oracle
        query-grounding fix) meant to stop semantically-similar-but-wrong-topic
        nodes from ever reaching the scoring stage.
        """
        if not self.graph.active_nodes():
            return ""
        subgraph, ranked, _ = self._retrieve(task_description, task_type)
        if not subgraph.nodes:
            return ""
        return self._format(subgraph, ranked)

    def retrieval_stats(self, task_description: str) -> dict:
        """Return metadata about what would be retrieved (for evaluation/logging)."""
        active = self.graph.active_nodes()
        if not active:
            return {"seed_count": 0, "subgraph_nodes": 0, "chars": 0}
        subgraph, ranked, seed_scores = self._retrieve(task_description)
        top_seeds = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_seeds]
        context = self._format(subgraph, ranked)
        return {
            "seed_count": len(top_seeds),
            "top_seed_scores": [(nid, round(s, 3)) for nid, s in top_seeds],
            "subgraph_nodes": subgraph.number_of_nodes(),
            "subgraph_edges": subgraph.number_of_edges(),
            "chars": len(context),
        }

    def _retrieve(
        self, task_description: str, task_type: str = ""
    ) -> tuple["nx.DiGraph", list[tuple[str, float]], dict[str, float]]:
        """
        Shared retrieval core for inject() and retrieval_stats().
        Returns (subgraph, ranked_node_scores, seed_scores).
        ranked_node_scores covers both seeds and BFS-expanded neighbours.
        seed_scores covers only the directly-scored active nodes (pre-BFS).
        """
        active = self.graph.active_nodes()

        if self.use_keyword_prefilter:
            prefiltered = self._keyword_prefilter(task_description, task_type, active)
            if prefiltered:
                active = prefiltered

        seed_scores = self._score_nodes(task_description, active)
        if self.use_reinforcement_weighting:
            seed_scores = self._apply_reinforcement(seed_scores, active)
        top_seeds = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_seeds]

        if not top_seeds or top_seeds[0][1] == 0.0:
            return nx.DiGraph(), [], seed_scores

        seed_ids = [nid for nid, _ in top_seeds]
        subgraph = self.graph.get_subgraph(seed_ids, k_hops=self.k_hops)

        node_scores: dict[str, float] = {}
        for nid, score in seed_scores.items():
            if nid in subgraph.nodes:
                node_scores[nid] = score

        visited = set(seed_ids)
        frontier = list(seed_ids)
        for hop in range(1, self.k_hops + 1):
            next_frontier = []
            for n in frontier:
                for nb in list(subgraph.successors(n)) + list(subgraph.predecessors(n)):
                    if nb not in visited:
                        parent_score = node_scores.get(n, 0.0)
                        node_scores[nb] = max(
                            node_scores.get(nb, 0.0),
                            parent_score * (DEPTH_DISCOUNT ** hop),
                        )
                        next_frontier.append(nb)
                        visited.add(nb)
            frontier = next_frontier

        ranked = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        return subgraph, ranked, seed_scores

    # ------------------------------------------------------------------
    # Grounding / reinforcement
    # ------------------------------------------------------------------

    def _keyword_prefilter(self, query: str, task_type: str, nodes: list[dict]) -> list[dict]:
        """Narrow to nodes matching task_type or sharing >=1 keyword with the query."""
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        matched = []
        for n in nodes:
            if task_type and n.get("task_type") == task_type:
                matched.append(n)
                continue
            node_keywords = set(n.get("keywords", []))
            if node_keywords & query_tokens:
                matched.append(n)
        return matched

    def _apply_reinforcement(self, scores: dict[str, float], nodes: list[dict]) -> dict[str, float]:
        """Boost score by log2(reinforce_count) * REINFORCE_WEIGHT — a node five
        sessions independently converged on outranks one seen only once."""
        counts = {n["id"]: n.get("reinforce_count", 1) for n in nodes}
        return {
            nid: score * (1 + REINFORCE_WEIGHT * math.log2(max(counts.get(nid, 1), 1) + 1))
            for nid, score in scores.items()
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_nodes(self, query: str, nodes: list[dict]) -> dict[str, float]:
        """Return {node_id: similarity_score} for all active nodes."""
        if self.embed_fn:
            return self._embed_score(query, nodes)
        return self._tfidf_score(query, nodes)

    def _tfidf_score(self, query: str, nodes: list[dict]) -> dict[str, float]:
        def tokenize(text: str) -> list[str]:
            return re.findall(r"[a-z0-9]+", text.lower())

        q_tokens = tokenize(query)
        if not q_tokens:
            return {n["id"]: 0.0 for n in nodes}

        all_tokens = [tokenize(n["content"]) for n in nodes]
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
        return {
            node["id"]: cosine(q_vec, tfidf_vec(toks))
            for node, toks in zip(nodes, all_tokens)
        }

    def _embed_score(self, query: str, nodes: list[dict]) -> dict[str, float]:
        import numpy as np

        texts = [query] + [n["content"] for n in nodes]
        vecs = self.embed_fn(texts)
        q_vec = np.array(vecs[0])
        scores = {}
        for i, node in enumerate(nodes):
            c_vec = np.array(vecs[i + 1])
            score = float(
                np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-9)
            )
            scores[node["id"]] = score
        return scores

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format(self, subgraph: nx.DiGraph, ranked: list[tuple[str, float]]) -> str:
        lines = ["[PROCEDURAL MEMORY]"]
        chars_used = len(lines[0]) + 1
        scripts_seen: set[str] = set()
        steps_seen: set[str] = set()

        for nid, score in ranked:
            if nid not in subgraph.nodes:
                continue
            attrs = subgraph.nodes[nid]
            level = attrs.get("level", "step")
            content = attrs.get("content", "").strip()
            if not content:
                continue

            if level == "script" and nid not in scripts_seen:
                block = f"\nSTRATEGY: {content}"
                # Append step children inline
                step_children = [
                    s for s in subgraph.successors(nid)
                    if subgraph.edges[nid, s].get("relation") == "generalizes"
                    and s not in steps_seen
                ]
                if step_children:
                    step_lines = []
                    for i, sc in enumerate(step_children, 1):
                        sc_content = subgraph.nodes[sc].get("content", "").strip()
                        if sc_content:
                            step_lines.append(f"  {i}. {sc_content}")
                            steps_seen.add(sc)
                    if step_lines:
                        block += "\nSteps:\n" + "\n".join(step_lines)
                scripts_seen.add(nid)

            elif level == "step" and nid not in steps_seen:
                block = f"\nSTEP: {content}"
                steps_seen.add(nid)
            else:
                continue

            if chars_used + len(block) > self.token_budget:
                break
            lines.append(block)
            chars_used += len(block)

        if len(lines) == 1:
            return ""

        lines.append("\n[END PROCEDURAL MEMORY]\n")
        return "\n".join(lines)
