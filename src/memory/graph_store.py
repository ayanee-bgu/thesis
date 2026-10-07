"""
Procedure graph: NetworkX-backed directed graph for storing agent procedural memory.

Nodes represent procedures at two levels:
  - "step"   : fine-grained, single-action instruction
  - "script" : higher-level, task-template abstraction

Typed edges encode relationships:
  - depends_on    : src requires dst to be completed first
  - generalizes   : src is a more abstract form of dst (script -> step)
  - conflicts_with: src and dst are contradictory procedures for the same precondition
  - derived_from  : src was distilled from the same trajectory/session as dst

Serialises to JSON; no server required.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx

VALID_RELATIONS = frozenset({"depends_on", "generalizes", "conflicts_with", "derived_from"})
VALID_LEVELS = frozenset({"step", "script"})

DEFAULT_GRAPH_PATH = Path("knowledge_base/graph/procedure_graph.json")


class ProcedureGraph:
    def __init__(self):
        self._g: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_procedure(
        self,
        content: str,
        level: str,
        source_agent: str,
        session_id: str,
        node_id: Optional[str] = None,
        task_type: str = "",
        keywords: Optional[list[str]] = None,
    ) -> str:
        """Add a procedure node. Returns the node_id.

        task_type/keywords are optional coarse tags (task id, topic, entity names)
        used by ContextInjector as a cheap prefilter before similarity scoring —
        mirrors the query-grounding fix in the retrieval-oracle work: a keyword
        match is a much cheaper and more reliable signal than raw-content cosine
        similarity when the graph has many superficially-similar nodes.
        """
        if level not in VALID_LEVELS:
            raise ValueError(f"level must be one of {VALID_LEVELS}, got '{level}'")
        nid = node_id or str(uuid.uuid4())[:8]
        self._g.add_node(
            nid,
            content=content,
            level=level,
            source_agent=source_agent,
            session_id=session_id,
            task_type=task_type,
            keywords=keywords or [],
            reinforce_count=1,
            created_at=datetime.now(timezone.utc).isoformat(),
            deprecated=False,
            deprecation_reason=None,
        )
        return nid

    def deprecate(self, node_id: str, reason: str = "") -> None:
        """Soft-delete a node (keeps it in graph for provenance; excluded from retrieval)."""
        if node_id not in self._g:
            raise KeyError(f"Node '{node_id}' not found")
        self._g.nodes[node_id]["deprecated"] = True
        self._g.nodes[node_id]["deprecation_reason"] = reason

    def merge_nodes(self, keep: str, drop: str, merged_content: Optional[str] = None) -> str:
        """
        Merge `drop` into `keep`: redirect all edges, deprecate `drop`.
        Returns `keep`.

        Bumps `keep`'s reinforce_count by `drop`'s — a node independently
        rediscovered across many sessions accumulates a higher count, which
        ContextInjector can use to rank reinforced procedures above ones seen once.
        """
        if keep not in self._g or drop not in self._g:
            raise KeyError(f"Both nodes must exist: keep={keep}, drop={drop}")
        if merged_content:
            self._g.nodes[keep]["content"] = merged_content
        if keep != drop:
            self._g.nodes[keep]["reinforce_count"] = (
                self._g.nodes[keep].get("reinforce_count", 1) + self._g.nodes[drop].get("reinforce_count", 1)
            )
        drop_keywords = self._g.nodes[drop].get("keywords", [])
        if drop_keywords:
            keep_keywords = set(self._g.nodes[keep].get("keywords", []))
            keep_keywords.update(drop_keywords)
            self._g.nodes[keep]["keywords"] = sorted(keep_keywords)

        for pred in list(self._g.predecessors(drop)):
            rel = self._g.edges[pred, drop]["relation"]
            if pred != keep and not self._g.has_edge(pred, keep):
                self._g.add_edge(pred, keep, relation=rel)

        for succ in list(self._g.successors(drop)):
            rel = self._g.edges[drop, succ]["relation"]
            if succ != keep and not self._g.has_edge(keep, succ):
                self._g.add_edge(keep, succ, relation=rel)

        self.deprecate(drop, reason=f"merged_into:{keep}")
        return keep

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, src: str, dst: str, relation: str) -> None:
        """Add a typed directed edge between two procedure nodes."""
        if relation not in VALID_RELATIONS:
            raise ValueError(f"relation must be one of {VALID_RELATIONS}, got '{relation}'")
        if src not in self._g:
            raise KeyError(f"Source node '{src}' not found")
        if dst not in self._g:
            raise KeyError(f"Destination node '{dst}' not found")
        self._g.add_edge(src, dst, relation=relation)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_subgraph(self, seed_node_ids: list[str], k_hops: int = 2) -> nx.DiGraph:
        """
        Return a k-hop neighbourhood subgraph centred on seed_node_ids.
        Deprecated nodes are excluded from the result.
        """
        seed_node_ids = [n for n in seed_node_ids if n in self._g]
        visited: set[str] = set(seed_node_ids)
        frontier = list(seed_node_ids)

        for _ in range(k_hops):
            next_frontier = []
            for n in frontier:
                for succ in self._g.successors(n):
                    if self._g.edges[n, succ].get("relation") != "conflicts_with":
                        next_frontier.append(succ)
                for pred in self._g.predecessors(n):
                    if self._g.edges[pred, n].get("relation") != "conflicts_with":
                        next_frontier.append(pred)
            frontier = [n for n in next_frontier if n not in visited]
            visited.update(frontier)

        active = [n for n in visited if not self._g.nodes[n].get("deprecated", False)]
        return self._g.subgraph(active).copy()

    def active_nodes(self) -> list[dict]:
        """All non-deprecated nodes as a list of attribute dicts (including 'id')."""
        return [
            {"id": n, **dict(self._g.nodes[n])}
            for n in self._g.nodes
            if not self._g.nodes[n].get("deprecated", False)
        ]

    def node_content(self, node_id: str) -> str:
        return self._g.nodes[node_id]["content"]

    def conflicts(self) -> list[tuple[str, str]]:
        """Return all (src, dst) pairs connected by a conflicts_with edge."""
        return [
            (u, v)
            for u, v, data in self._g.edges(data=True)
            if data.get("relation") == "conflicts_with"
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = DEFAULT_GRAPH_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [{"id": n, **dict(attrs)} for n, attrs in self._g.nodes(data=True)],
            "edges": [
                {"src": u, "dst": v, **dict(attrs)}
                for u, v, attrs in self._g.edges(data=True)
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_GRAPH_PATH) -> "ProcedureGraph":
        """Load from JSON. Returns an empty graph if the file does not exist."""
        pg = cls()
        path = Path(path)
        if not path.exists():
            return pg
        data = json.loads(path.read_text(encoding="utf-8"))
        for node in data.get("nodes", []):
            node = dict(node)
            nid = node.pop("id")
            pg._g.add_node(nid, **node)
        for edge in data.get("edges", []):
            edge = dict(edge)
            pg._g.add_edge(edge.pop("src"), edge.pop("dst"), **edge)
        return pg

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of active (non-deprecated) nodes."""
        return sum(1 for n in self._g.nodes if not self._g.nodes[n].get("deprecated", False))

    def __repr__(self) -> str:
        return f"ProcedureGraph(nodes={len(self)}, edges={self._g.number_of_edges()})"
