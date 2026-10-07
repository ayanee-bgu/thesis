"""
Optional sentence-embedding backend for graph_store/consolidator/context_injector.

Every embed_fn parameter across src/memory expects: list[str] -> list[list[float]].
This module is the one place that actually loads a model, so consolidator.py and
context_injector.py stay dependency-free unless a caller opts in.

Usage:
    from src.memory.embeddings import get_embed_fn
    embed_fn = get_embed_fn()  # raises ImportError with install instructions if missing
    consolidator = SessionConsolidator(graph, model=..., embed_fn=embed_fn)
"""

from __future__ import annotations

from typing import Callable, Optional

_MODEL_CACHE: dict[str, object] = {}


def get_embed_fn(model_name: str = "all-MiniLM-L6-v2") -> Callable[[list[str]], list[list[float]]]:
    """
    Returns an embed_fn backed by sentence-transformers, loaded once and cached
    by model_name across calls (avoids reloading the model on every consolidate()).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers not installed. Run: pip install sentence-transformers\n"
            "This is the untested lever flagged across the retrieval-oracle and memory-controller "
            "reports — TF-IDF is the default everywhere until this is installed."
        ) from e

    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    model = _MODEL_CACHE[model_name]

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, convert_to_numpy=False, show_progress_bar=False)

    return embed_fn


def try_get_embed_fn(model_name: str = "all-MiniLM-L6-v2") -> Optional[Callable[[list[str]], list[list[float]]]]:
    """Same as get_embed_fn but returns None instead of raising if unavailable."""
    try:
        return get_embed_fn(model_name)
    except ImportError:
        return None
