"""
E10: NarrativeQA Oracle Ablation — which component of the interactive retrieval
oracle actually drives accuracy in the full-context-infeasible regime?

Holds dataset / model / harness fixed from E9 and ablates the oracle mechanism
across six variants:

  Variant           Index     Grounded  ±1 Neighbors
  oracle_baseline   TF-IDF    no        no     (= E9 interactive_retrieval post-fix)
  oracle_grounded   TF-IDF    yes       no
  oracle_bm25       BM25      no        no
  oracle_embed      Embed     no        no
  oracle_neighbor   TF-IDF    no        yes
  oracle_full       Embed     yes       yes

Three static baselines from E9 (no_context, full_context, summary_doc) are kept
as the outer comparison frame so "does the best oracle close the gap to the
human summary?" remains answerable.

New metric: retrieval_hit_rate — fraction of a question's retrieved primary
chunks that contain >=50% of any gold answer's content words. Separates
"oracle never retrieved the evidence" from "oracle retrieved it but LLM
reasoned wrong."

Dependencies beyond E9:
  pip install rank_bm25 sentence-transformers numpy

Usage:
  python experiments/narrativeqa_oracle_v2/main.py [options]

  --no-llm             Bypass Ollama (structural test only)
  --condition COND     Comma-separated conditions (default: all)
  --num-tasks N        Override num_per_kind
  --max-queries K      Override retrieval.max_queries
  --log-level LEVEL
  --fresh              Ignore cached results and recompute from scratch
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import string
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import requests
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = Path(__file__).parent / "config.yaml"

STATIC_CONDITIONS = ["no_context", "full_context", "summary_doc"]

# Each oracle variant is defined by three axes: retrieval index, query grounding,
# and whether to expand the best hit to include its ±1 neighboring chunks.
ORACLE_CONFIGS: dict[str, dict] = {
    "oracle_baseline": {"index": "tfidf", "grounded": False, "n_neighbors": 0},
    "oracle_grounded": {"index": "tfidf", "grounded": True,  "n_neighbors": 0},
    "oracle_bm25":     {"index": "bm25",  "grounded": False, "n_neighbors": 0},
    "oracle_embed":    {"index": "embed", "grounded": False, "n_neighbors": 0},
    "oracle_neighbor": {"index": "tfidf", "grounded": False, "n_neighbors": 1},
    "oracle_full":     {"index": "embed", "grounded": True,  "n_neighbors": 1},
}
ORACLE_CONDITIONS = list(ORACLE_CONFIGS.keys())
CONDITIONS = STATIC_CONDITIONS + ORACLE_CONDITIONS


# ---------------------------------------------------------------------------
# Config / env
# ---------------------------------------------------------------------------

def load_config(overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if overrides:
        for k, v in overrides.items():
            cfg[k] = v
    return cfg


def load_dotenv_manual(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Dataset loading  (identical to E9)
# ---------------------------------------------------------------------------

def load_narrativeqa(cfg: dict, logger: logging.Logger) -> dict[str, list[dict]]:
    load_dotenv_manual(ROOT / ".env")
    from datasets import load_dataset
    ds = load_dataset(cfg["experiment"]["dataset"], split=cfg["experiment"]["split"])
    logger.info(f"[dataset] Loaded {len(ds)} rows from {cfg['experiment']['dataset']}")
    num_per_kind = cfg["experiment"]["num_per_kind"]
    kinds = cfg["experiment"]["kinds"]
    out: dict[str, list[dict]] = {k: [] for k in kinds}
    for ex in ds:
        kind = ex["document"]["kind"]
        if kind in out and len(out[kind]) < num_per_kind:
            out[kind].append(ex)
        if all(len(v) >= num_per_kind for v in out.values()):
            break
    for label, examples in out.items():
        logger.info(f"[dataset] {label}: {len(examples)} examples selected")
    return out


def chunk_narrative_document(
    text: str, chunk_words: int, overlap_words: int = 0
) -> list[tuple[str, str]]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks, chunk_idx, i = [], 0, 0
    while i < len(words):
        chunks.append((f"[chunk {chunk_idx}]", " ".join(words[i:i + chunk_words])))
        i += step
        chunk_idx += 1
    return chunks


def truncate_to_budget(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " ...[truncated]"


# ---------------------------------------------------------------------------
# Scoring  (identical to E9)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize_answer(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.lower().split())


def exact_match_score(prediction: str, candidates: list[str]) -> float:
    norm = normalize_answer(prediction)
    return 1.0 if any(norm == normalize_answer(c) for c in candidates) else 0.0


def f1_score(prediction: str, candidates: list[str]) -> float:
    def _f1(pred_tokens, gold_tokens):
        if not pred_tokens or not gold_tokens:
            return 1.0 if pred_tokens == gold_tokens else 0.0
        common = {t: min(pred_tokens.count(t), gold_tokens.count(t)) for t in pred_tokens}
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        p = num_same / len(pred_tokens)
        r = num_same / len(gold_tokens)
        return 2 * p * r / (p + r)
    pred_tokens = normalize_answer(prediction).split()
    return max(_f1(pred_tokens, normalize_answer(c).split()) for c in candidates)


_CONTAINMENT_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "was", "were", "are", "that", "this", "with", "by", "as", "from", "which",
    "what", "who", "when", "where", "did", "does", "how", "many", "much",
    "same", "also", "has", "have", "had", "it", "its", "his", "her", "their",
}


def answer_containment(given_text: str, gold_answers: list[str]) -> float:
    if not given_text or not gold_answers:
        return 0.0
    given_tokens = set(_tokenize(given_text))
    for ans in gold_answers:
        if not ans:
            continue
        ans_tokens = {t for t in _tokenize(ans) if t not in _CONTAINMENT_STOPWORDS and len(t) > 2}
        if not ans_tokens:
            continue
        if len(ans_tokens & given_tokens) / len(ans_tokens) >= 0.5:
            return 1.0
    return 0.0


def retrieval_hit_rate(primary_chunks: list[str], gold_answers: list[str]) -> float:
    """Fraction of retrieved primary chunks that contain >=50% of any gold answer's
    content words. Separates retrieval failure from reasoning failure."""
    if not primary_chunks or not gold_answers:
        return 0.0
    hits = sum(1 for c in primary_chunks if answer_containment(c, gold_answers) > 0.0)
    return hits / len(primary_chunks)


# ---------------------------------------------------------------------------
# DFS chain-following metric  (identical to E9)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "was", "were", "are", "that", "this", "with", "by", "as", "from", "which",
    "what", "who", "when", "where", "did", "does", "how", "many", "much",
    "same", "also", "has", "have", "had", "it", "its", "his", "her", "their",
}


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 2}


def compute_dfs_score(queries: list[str], retrieved_chunks: list[str]) -> float | None:
    if len(queries) < 2:
        return None
    hits = sum(
        1 for i in range(1, len(queries))
        if _content_tokens(retrieved_chunks[i - 1]) & _content_tokens(queries[i])
    )
    return hits / (len(queries) - 1)


# ---------------------------------------------------------------------------
# Retrieval indices
# ---------------------------------------------------------------------------

class TFIDFIndex:
    """TF-IDF cosine retrieval.  Same logic as E9 RetrievalIndex."""

    def __init__(self, chunks: list[tuple[str, str]]):
        self.chunks = chunks
        self._doc_tokens = [_tokenize(text) for _, text in chunks]
        self._N = len(chunks) + 1

    def _idf(self, term: str) -> float:
        df = sum(1 for toks in self._doc_tokens if term in toks)
        return math.log((self._N + 1) / (df + 1)) + 1

    def _tfidf_vec(self, tokens: list[str], vocab: set[str]) -> dict[str, float]:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n = len(tokens) or 1
        return {t: (tf.get(t, 0) / n) * self._idf(t) for t in vocab}

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        dot = sum(a.get(t, 0) * b.get(t, 0) for t in b)
        na = math.sqrt(sum(v * v for v in a.values())) or 1
        nb = math.sqrt(sum(v * v for v in b.values())) or 1
        return dot / (na * nb)

    def query(self, query_text: str, n_neighbors: int = 0) -> list[tuple[str, str, float]]:
        q_tokens = _tokenize(query_text)
        vocab = set(q_tokens)
        if not vocab:
            best_idx, best_score = 0, 0.0
        else:
            q_vec = self._tfidf_vec(q_tokens, vocab)
            best_idx, best_score = 0, -1.0
            for i, toks in enumerate(self._doc_tokens):
                score = self._cosine(q_vec, self._tfidf_vec(toks, vocab))
                if score > best_score:
                    best_idx, best_score = i, score
        results = [(self.chunks[best_idx][0], self.chunks[best_idx][1], best_score)]
        if n_neighbors:
            for offset in [-1, 1]:
                nb = best_idx + offset
                if 0 <= nb < len(self.chunks):
                    results.append((self.chunks[nb][0], self.chunks[nb][1], 0.0))
        return results


class BM25Index:
    """BM25Okapi retrieval via rank_bm25."""

    def __init__(self, chunks: list[tuple[str, str]]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 required for oracle_bm25: pip install rank_bm25")
        self.chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(text) for _, text in chunks])

    def query(self, query_text: str, n_neighbors: int = 0) -> list[tuple[str, str, float]]:
        import numpy as np
        q_tokens = _tokenize(query_text)
        if not q_tokens:
            best_idx, best_score = 0, 0.0
        else:
            scores = self._bm25.get_scores(q_tokens)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
        results = [(self.chunks[best_idx][0], self.chunks[best_idx][1], best_score)]
        if n_neighbors:
            for offset in [-1, 1]:
                nb = best_idx + offset
                if 0 <= nb < len(self.chunks):
                    results.append((self.chunks[nb][0], self.chunks[nb][1], 0.0))
        return results


class EmbedIndex:
    """Cosine similarity retrieval via sentence-transformer embeddings.
    Chunk vectors are pre-computed at build time; only the query is encoded
    at query time, so per-question latency is low."""

    def __init__(self, chunks: list[tuple[str, str]], embed_fn: Callable):
        import numpy as np
        self.chunks = chunks
        self._embed_fn = embed_fn
        vecs = np.array(embed_fn([text for _, text in chunks]))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self._vecs = vecs / np.maximum(norms, 1e-9)

    def query(self, query_text: str, n_neighbors: int = 0) -> list[tuple[str, str, float]]:
        import numpy as np
        q_vec = np.array(self._embed_fn([query_text])[0])
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        scores = self._vecs @ q_vec
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        results = [(self.chunks[best_idx][0], self.chunks[best_idx][1], best_score)]
        if n_neighbors:
            for offset in [-1, 1]:
                nb = best_idx + offset
                if 0 <= nb < len(self.chunks):
                    results.append((self.chunks[nb][0], self.chunks[nb][1], float(scores[nb])))
        return results


def build_index(variant: str, chunks: list[tuple[str, str]], embed_fn=None):
    index_type = ORACLE_CONFIGS[variant]["index"]
    if index_type == "bm25":
        return BM25Index(chunks)
    if index_type == "embed":
        if embed_fn is None:
            raise ValueError(f"{variant} requires embed_fn — pass --embed or check oracle.embed_model in config")
        return EmbedIndex(chunks, embed_fn)
    return TFIDFIndex(chunks)


# ---------------------------------------------------------------------------
# LLM helper  (identical to E9)
# ---------------------------------------------------------------------------

_CONNECT_TIMEOUT = 15
_MAX_RETRIES = 3
_RETRY_BACKOFF = 15


def _llm_call(messages: list[dict], model: str, ollama_url: str, timeout: int) -> tuple[str, int, float]:
    payload = {
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512, "num_gpu": 0},
    }
    t0 = time.time()
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(ollama_url, json=payload, timeout=(_CONNECT_TIMEOUT, timeout))
            resp.raise_for_status()
            data = resp.json()
            content = data["message"]["content"].strip()
            tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            return content, tokens, round(time.time() - t0, 2)
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                logging.getLogger(__name__).warning(
                    f"LLM call failed (attempt {attempt}/{_MAX_RETRIES}): {exc}. Retrying in {_RETRY_BACKOFF}s..."
                )
                time.sleep(_RETRY_BACKOFF)
    raise last_exc


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANSWER_STYLE = "Answer in as few words as possible (a name, phrase, or short sentence) — no explanation."

NO_CONTEXT_SYSTEM = f"You are answering a question about a book or movie from your own knowledge. {ANSWER_STYLE} If you don't know, give your best guess."
FULL_CONTEXT_SYSTEM = f"Answer the question using ONLY the story text below (it may be an excerpt). {ANSWER_STYLE}\n\n{{passages}}"
SUMMARY_QA_SYSTEM = f"Answer the question using ONLY the summary below. {ANSWER_STYLE}\n\nSUMMARY:\n{{summary}}"
INTERACTIVE_ANSWER_SYSTEM = f"Answer the question using ONLY the retrieved information below. {ANSWER_STYLE}"

# Ungrounded query-gen system (oracle_baseline, oracle_bm25, oracle_embed, oracle_neighbor)
QUERY_GEN_SYSTEM = (
    "You are researching a question about a book or movie step by step, one search query at a time. "
    "Use what you learned from previous results to decide your next query if you still need more "
    "information. Do not repeat a query for information already retrieved. "
    "If you already have enough information to answer, reply with just DONE. "
    "Otherwise reply with ONLY a short search query (2-6 words), nothing else."
)


def make_grounded_query_gen_system(doc_title: str, doc_kind: str) -> str:
    """Grounded variant: tells the model the exact title so queries target the
    right characters/entities from the start rather than generic searches."""
    label = {"gutenberg": "novel", "movie": "movie script"}.get(doc_kind, doc_kind)
    return (
        f"You are researching a question about '{doc_title}' ({label}) step by step, "
        "one search query at a time. "
        "Use what you learned from previous results to decide your next query if you still need more "
        "information. Do not repeat a query for information already retrieved. "
        "If you already have enough information to answer, reply with just DONE. "
        "Otherwise reply with ONLY a short search query (2-6 words), nothing else."
    )


# ---------------------------------------------------------------------------
# Per-condition answer functions
# ---------------------------------------------------------------------------

def answer_no_context(question: str, model: str, url: str, timeout: int) -> tuple[str, int, float, dict]:
    ans, tok, el = _llm_call(
        [{"role": "system", "content": NO_CONTEXT_SYSTEM},
         {"role": "user", "content": f"Question: {question}\nAnswer:"}],
        model, url, timeout,
    )
    return ans, tok, el, {"given_text": "", "primary_chunks": []}


def answer_full_context(question: str, doc_text: str, max_words: int, model: str, url: str, timeout: int) -> tuple[str, int, float, dict]:
    truncated = truncate_to_budget(doc_text, max_words)
    ans, tok, el = _llm_call(
        [{"role": "system", "content": FULL_CONTEXT_SYSTEM.format(passages=truncated)},
         {"role": "user", "content": f"Question: {question}\nAnswer:"}],
        model, url, timeout,
    )
    return ans, tok, el, {"given_text": truncated, "primary_chunks": []}


def answer_summary_doc_provided(question: str, summary_text: str, model: str, url: str, timeout: int) -> tuple[str, int, float, dict]:
    ans, tok, el = _llm_call(
        [{"role": "system", "content": SUMMARY_QA_SYSTEM.format(summary=summary_text)},
         {"role": "user", "content": f"Question: {question}\nAnswer:"}],
        model, url, timeout,
    )
    return ans, tok, el, {"given_text": summary_text, "primary_chunks": []}


def answer_oracle_variant(
    question: str,
    chunks: list[tuple[str, str]],
    k: int,
    variant: str,
    model: str,
    url: str,
    timeout: int,
    doc_title: str = "",
    doc_kind: str = "",
    embed_fn=None,
) -> tuple[str, int, float, dict]:
    cfg = ORACLE_CONFIGS[variant]
    n_neighbors = cfg["n_neighbors"]

    index = build_index(variant, chunks, embed_fn)
    query_gen_sys = (
        make_grounded_query_gen_system(doc_title, doc_kind)
        if cfg["grounded"] else QUERY_GEN_SYSTEM
    )

    queries: list[str] = []
    primary_titles: list[str] = []
    primary_chunks: list[str] = []   # best-match chunk per round; used for DFS + query history
    expansion_chunks: list[str] = [] # neighbor chunks (±1); added to answer prompt only
    total_tokens = 0
    t0 = time.time()
    early_stop = False

    for _round in range(k):
        history_lines = [
            f"QUERY {i+1}: {q}\nRESULT {i+1}: [{t}] {c}"
            for i, (q, t, c) in enumerate(zip(queries, primary_titles, primary_chunks))
        ]
        user_msg = f"QUESTION: {question}"
        if history_lines:
            user_msg += "\n\n" + "\n\n".join(history_lines)
        user_msg += "\n\nWhat is your next search query?"

        query_text, q_tok, _ = _llm_call(
            [{"role": "system", "content": query_gen_sys}, {"role": "user", "content": user_msg}],
            model, url, timeout,
        )
        total_tokens += q_tok

        if query_text.strip().upper().startswith("DONE"):
            early_stop = True
            break

        retrieved = index.query(query_text, n_neighbors=n_neighbors)
        p_title, p_text, _score = retrieved[0]
        queries.append(query_text)
        primary_titles.append(p_title)
        primary_chunks.append(p_text)
        for nb_title, nb_text, _ in retrieved[1:]:
            expansion_chunks.append(nb_text)

    # Build deduplicated answer prompt: primary chunks first, then neighbor context
    seen: set[str] = set()
    unique_history_lines = []
    for i, (q, t, c) in enumerate(zip(queries, primary_titles, primary_chunks)):
        if c not in seen:
            unique_history_lines.append(f"QUERY {i+1}: {q}\nRESULT {i+1}: [{t}] {c}")
            seen.add(c)

    unique_expansions = [c for c in dict.fromkeys(expansion_chunks) if c not in seen]
    expansion_block = (
        "\n\nADDITIONAL CONTEXT (neighboring passages):\n" + "\n\n".join(unique_expansions)
        if unique_expansions else ""
    )

    final_user = (
        f"QUESTION: {question}\n\n"
        + "\n\n".join(unique_history_lines)
        + expansion_block
        + "\n\nAnswer:"
    )
    ans, a_tok, _ = _llm_call(
        [{"role": "system", "content": INTERACTIVE_ANSWER_SYSTEM},
         {"role": "user", "content": final_user}],
        model, url, timeout,
    )
    total_tokens += a_tok
    elapsed = round(time.time() - t0, 2)

    all_retrieved = list(dict.fromkeys(primary_chunks + unique_expansions))
    dfs_score = compute_dfs_score(queries, primary_chunks)
    extra = {
        "queries": queries,
        "retrieved_titles": primary_titles,
        "given_titles": list(dict.fromkeys(primary_titles)),
        "given_text": "\n\n".join(all_retrieved),
        "primary_chunks": primary_chunks,
        "num_queries": len(queries),
        "early_stop": early_stop,
        "n_expansion_chunks": len(unique_expansions),
        "dfs_score": dfs_score,
    }
    return ans, total_tokens, elapsed, extra


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_condition(
    condition: str,
    examples: list[dict],
    qtype: str,
    cfg: dict,
    logger: logging.Logger,
    no_llm: bool = False,
    label: str | None = None,
    embed_fn=None,
) -> list[dict]:
    label = label or condition
    model = cfg["model"]["name"]
    url = cfg["model"]["ollama_url"]
    timeout = cfg["model"]["timeout"]
    k = cfg["retrieval"]["max_queries"]
    n_neighbors = cfg["retrieval"]["n_neighbors"]
    max_words = cfg["experiment"]["full_context_max_words"]
    chunk_words = cfg["experiment"]["chunk_words"]
    overlap_words = cfg["experiment"].get("chunk_overlap_words", chunk_words // 5)

    # For oracle conditions, propagate the configured n_neighbors into the variant cfg.
    # This lets config.yaml override the default without changing the variant table.
    if condition in ORACLE_CONDITIONS:
        effective_cfg = dict(ORACLE_CONFIGS[condition])
        if effective_cfg["n_neighbors"] > 0:
            effective_cfg["n_neighbors"] = n_neighbors
        ORACLE_CONFIGS[condition] = effective_cfg

    results = []
    for ex in examples:
        doc = ex["document"]
        question = ex["question"]["text"]
        candidates = [a["text"] for a in ex["answers"] if a.get("text")]
        if not candidates:
            candidates = [""]
        doc_id = doc["id"]
        doc_text = doc["text"]
        doc_word_count = doc.get("word_count") or len(doc_text.split())
        summary_text = doc["summary"]["text"]
        doc_title = doc.get("summary", {}).get("title") or doc_id.replace("_", " ").title()
        doc_kind = doc.get("kind", "")

        t0 = time.time()
        if no_llm:
            ans, tok, el, extra = (
                f"[MOCK for {doc_id}]", 0, 0.0,
                {"given_text": "", "primary_chunks": [], "num_queries": 0,
                 "early_stop": False, "n_expansion_chunks": 0, "dfs_score": None}
            )
        else:
            try:
                if condition == "no_context":
                    ans, tok, el, extra = answer_no_context(question, model, url, timeout)
                elif condition == "full_context":
                    ans, tok, el, extra = answer_full_context(question, doc_text, max_words, model, url, timeout)
                elif condition == "summary_doc":
                    ans, tok, el, extra = answer_summary_doc_provided(question, summary_text, model, url, timeout)
                elif condition in ORACLE_CONDITIONS:
                    chunks = chunk_narrative_document(doc_text, chunk_words, overlap_words)
                    ans, tok, el, extra = answer_oracle_variant(
                        question, chunks, k, condition, model, url, timeout,
                        doc_title=doc_title, doc_kind=doc_kind, embed_fn=embed_fn,
                    )
                else:
                    raise ValueError(f"Unknown condition: {condition}")
            except Exception as e:
                logger.warning(f"[{label}] {doc_id} error: {e}", exc_info=True)
                ans, tok, el, extra = (
                    "", 0, 0.0,
                    {"given_text": "", "primary_chunks": [], "num_queries": None,
                     "early_stop": False, "n_expansion_chunks": 0, "dfs_score": None}
                )

        em = exact_match_score(ans, candidates)
        f1 = f1_score(ans, candidates)
        containment = answer_containment(extra.get("given_text", ""), candidates)
        primary_chunks = extra.get("primary_chunks", [])
        hit_rate = retrieval_hit_rate(primary_chunks, candidates) if primary_chunks else None
        total_elapsed = round(time.time() - t0, 2)

        result = {
            "id": doc_id,
            "type": qtype,
            "condition": label,
            "question": question,
            "gold_answer": candidates[0],
            "predicted_answer": ans,
            "em": em,
            "f1": f1,
            "tokens": tok,
            "elapsed_s": total_elapsed,
            "supporting_fact_recall": containment,
            "retrieval_hit_rate": hit_rate,
            "num_queries": extra.get("num_queries"),
            "early_stop": extra.get("early_stop", False),
            "n_expansion_chunks": extra.get("n_expansion_chunks", 0),
            "dfs_score": extra.get("dfs_score"),
            "queries": extra.get("queries"),
            "doc_word_count": doc_word_count,
        }
        results.append(result)
        hit_str = f" hit={hit_rate:.0%}" if hit_rate is not None else ""
        logger.info(
            f"[{label}/{qtype}] {doc_id}: em={em:.0f} f1={f1:.2f} "
            f"contain={containment:.0f}{hit_str} tok={tok} s={total_elapsed:.1f}"
        )
    return results


# ---------------------------------------------------------------------------
# Model warmup
# ---------------------------------------------------------------------------

def warmup_model(model: str, url: str, timeout: int, logger: logging.Logger) -> None:
    payload = {
        "model": model, "messages": [{"role": "user", "content": "Ready?"}],
        "stream": False, "options": {"temperature": 0.0, "num_predict": 1, "num_gpu": 0},
    }
    for attempt in range(1, 6):
        try:
            t0 = time.time()
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                logger.info(f"[warmup] Model ready ({round(time.time()-t0, 1)}s)")
                return
        except Exception as e:
            logger.warning(f"[warmup] Attempt {attempt}/5: {e}, retrying...")
        time.sleep(5)
    logger.warning("[warmup] Model did not respond cleanly — proceeding anyway.")


# ---------------------------------------------------------------------------
# Summary / saving
# ---------------------------------------------------------------------------

def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def compute_summary(task_results: list[dict]) -> dict:
    if not task_results:
        return {}
    hit_rates = [r["retrieval_hit_rate"] for r in task_results if r["retrieval_hit_rate"] is not None]
    dfs_scores = [r["dfs_score"] for r in task_results if r["dfs_score"] is not None]
    early_stops = [r for r in task_results if r.get("early_stop")]
    return {
        "num_tasks": len(task_results),
        "exact_match": _mean([r["em"] for r in task_results]),
        "f1": _mean([r["f1"] for r in task_results]),
        "mean_tokens": _mean([r["tokens"] for r in task_results]),
        "mean_elapsed_s": _mean([r["elapsed_s"] for r in task_results]),
        "mean_supporting_fact_recall": _mean([r["supporting_fact_recall"] for r in task_results]),
        "mean_retrieval_hit_rate": _mean(hit_rates) if hit_rates else None,
        "mean_dfs_score": _mean(dfs_scores) if dfs_scores else None,
        "mean_num_queries": _mean([r["num_queries"] for r in task_results if r["num_queries"] is not None]),
        "early_stop_rate": len(early_stops) / len(task_results),
    }


def load_existing_condition_results(condition: str, results_dir: Path) -> dict[str, list[dict]]:
    path = results_dir / f"{condition}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("results_by_type", {})
    except Exception:
        return {}


def save_condition_results(
    condition: str,
    all_results_by_type: dict[str, list[dict]],
    results_dir: Path,
    cfg: dict,
) -> dict:
    summary_by_type = {qtype: compute_summary(res) for qtype, res in all_results_by_type.items()}
    overall = compute_summary([r for res in all_results_by_type.values() for r in res])
    out = {
        "condition": condition,
        "config": {"num_per_kind": cfg["experiment"]["num_per_kind"], "max_queries": cfg["retrieval"]["max_queries"]},
        "results_by_type": all_results_by_type,
        "summary_by_type": summary_by_type,
        "summary_overall": overall,
    }
    (results_dir / f"{condition}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Embed function loader
# ---------------------------------------------------------------------------

def load_embed_fn(model_name: str, logger: logging.Logger) -> Callable:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers required: pip install sentence-transformers")
    logger.info(f"[embed] Loading {model_name} ...")
    t0 = time.time()
    model = SentenceTransformer(model_name)
    logger.info(f"[embed] Model loaded in {round(time.time()-t0, 1)}s")
    def embed_fn(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, show_progress_bar=False).tolist()
    return embed_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E10: NarrativeQA oracle ablation")
    parser.add_argument("--condition", default=None,
                        help=f"Comma-separated conditions (default: all). Valid: {', '.join(CONDITIONS)}")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true", help="Bypass Ollama for structural testing")
    parser.add_argument("--fresh", action="store_true", help="Ignore cached results and recompute")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = load_config()
    if args.num_tasks:
        cfg["experiment"]["num_per_kind"] = args.num_tasks
    if args.max_queries is not None:
        cfg["retrieval"]["max_queries"] = args.max_queries

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    results_dir = ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.condition:
        selected_conditions = [c.strip() for c in args.condition.split(",")]
        invalid = [c for c in selected_conditions if c not in CONDITIONS]
        if invalid:
            logger.error(f"Unknown conditions: {invalid}. Valid: {CONDITIONS}")
            sys.exit(1)
    else:
        selected_conditions = CONDITIONS

    # Load embed model only if an embed-based oracle is selected
    embed_fn = None
    needs_embed = any(ORACLE_CONFIGS.get(c, {}).get("index") == "embed" for c in selected_conditions)
    if needs_embed and not args.no_llm:
        embed_fn = load_embed_fn(cfg["oracle"]["embed_model"], logger)

    logger.info("Loading NarrativeQA dataset...")
    examples_by_type = load_narrativeqa(cfg, logger)

    if not args.no_llm:
        warmup_model(cfg["model"]["name"], cfg["model"]["ollama_url"], cfg["model"]["timeout"], logger)

    all_condition_results = []
    for condition in selected_conditions:
        logger.info(f"\n{'─'*60}\nCondition: {condition}\n{'─'*60}")
        results_by_type = (
            load_existing_condition_results(condition, results_dir)
            if not args.fresh else {}
        )
        for qtype, examples in examples_by_type.items():
            cached = results_by_type.get(qtype)
            if cached and len(cached) == len(examples):
                logger.info(f"[{condition}/{qtype}] {len(cached)} cached results, skipping")
                continue
            results_by_type[qtype] = run_condition(
                condition, examples, qtype, cfg, logger,
                no_llm=args.no_llm, embed_fn=embed_fn,
            )
            save_condition_results(condition, results_by_type, results_dir, cfg)
        out = save_condition_results(condition, results_by_type, results_dir, cfg)
        all_condition_results.append(out)

    # Write cross-condition summary
    summary_rows = [
        {"condition": r["condition"], "overall": r["summary_overall"],
         **{f"by_type_{qt}": s for qt, s in r["summary_by_type"].items()}}
        for r in all_condition_results
    ]
    (results_dir / "summary.json").write_text(
        json.dumps({"conditions": summary_rows}, indent=2), encoding="utf-8"
    )
    logger.info(f"\nResults written to {results_dir}")

    # ── Print results table ──────────────────────────────────────────────────
    print("\n=== E10 RESULTS — NarrativeQA Oracle Ablation (overall) ===")
    hdr = (
        f"{'Condition':<20} {'EM':>6} {'F1':>6} {'Contain':>8} "
        f"{'HitRate':>8} {'DFS':>6} {'Qs':>5} {'Stop%':>6} {'Tok':>7}"
    )
    print(hdr)
    print("─" * len(hdr))

    def _fmt_row(name: str, s: dict) -> str:
        dfs = s.get("mean_dfs_score")
        hit = s.get("mean_retrieval_hit_rate")
        stop = s.get("early_stop_rate")
        nq = s.get("mean_num_queries")
        return (
            f"{name:<20} "
            f"{s.get('exact_match', 0):>6.1%} "
            f"{s.get('f1', 0):>6.1%} "
            f"{s.get('mean_supporting_fact_recall', 0):>8.1%} "
            f"{(f'{hit:.1%}' if hit is not None else '—'):>8} "
            f"{(f'{dfs:.1%}' if dfs is not None else '—'):>6} "
            f"{(f'{nq:.1f}' if nq is not None else '—'):>5} "
            f"{(f'{stop:.0%}' if stop is not None else '—'):>6} "
            f"{s.get('mean_tokens', 0):>7.0f}"
        )

    # Print static baselines first, then oracle ladder
    print("── static baselines ──")
    for r in all_condition_results:
        if r["condition"] in STATIC_CONDITIONS:
            print(_fmt_row(r["condition"], r["summary_overall"]))
    print("── oracle variants ──")
    for r in all_condition_results:
        if r["condition"] in ORACLE_CONDITIONS:
            print(_fmt_row(r["condition"], r["summary_overall"]))

    for qtype in cfg["experiment"]["kinds"]:
        print(f"\n── {qtype} ──")
        print(hdr)
        print("─" * len(hdr))
        for r in all_condition_results:
            s = r["summary_by_type"].get(qtype, {})
            print(_fmt_row(r["condition"], s))


if __name__ == "__main__":
    main()
