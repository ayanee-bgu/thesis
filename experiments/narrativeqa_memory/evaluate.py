"""
Phase 2 — Memory evaluation.

Runs 15 conditions on the NarrativeQA validation split against a fixed memory
pool built by populate.py.

Conditions:
  Baselines (no memory):   no_context | full_context | oracle_full
  Per format × method:     {F1,F2,F3,F4,F5,Fall} × {verbatim,interactive}
  Fused oracle+doc:        Fall_interactive_plus_doc

Usage:
  python experiments/narrativeqa_memory/evaluate.py [options]

  --no-llm             Structural test only
  --condition COND     Comma-separated subset (default: all)
  --num-tasks N        Override num_per_kind
  --max-queries K      Override retrieval.max_queries
  --fresh              Ignore cached results
  --log-level LEVEL
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

from experiments.narrativeqa_memory.memory_store import (
    FORMATS, MemoryNode, MemoryPool, TFIDFMemoryIndex, BM25MemoryIndex, EmbedMemoryIndex,
    format_verbatim_context, _tok,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

BASELINE_CONDITIONS = ["no_context", "full_context", "oracle_full"]
FORMAT_NAMES = ["F1", "F2", "F3", "F4", "F5", "Fall"]
MEMORY_CONDITIONS = (
    [f"{f}_verbatim" for f in FORMAT_NAMES]
    + [f"{f}_interactive" for f in FORMAT_NAMES]
    + ["Fall_interactive_plus_doc"]
)
CONDITIONS = BASELINE_CONDITIONS + MEMORY_CONDITIONS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_val_examples(cfg: dict, logger: logging.Logger) -> dict[str, list[dict]]:
    load_dotenv(ROOT / ".env")
    from datasets import load_dataset

    ev = cfg["evaluation"]
    ds = load_dataset(ev["dataset"], split=ev["split"])
    logger.info(f"[dataset] Loaded {len(ds)} val rows")
    num = ev["num_per_kind"]
    kinds = ev["kinds"]
    out: dict[str, list[dict]] = {k: [] for k in kinds}
    for ex in ds:
        kind = ex["document"]["kind"]
        if kind in out and len(out[kind]) < num:
            out[kind].append(ex)
        if all(len(v) >= num for v in out.values()):
            break
    for k, v in out.items():
        logger.info(f"[dataset] {k}: {len(v)} val examples")
    return out


def chunk_document(text: str, chunk_words: int, overlap_words: int) -> list[tuple[str, str]]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks, idx, i = [], 0, 0
    while i < len(words):
        chunks.append((f"[chunk {idx}]", " ".join(words[i:i + chunk_words])))
        i += step
        idx += 1
    return chunks


def truncate_to_budget(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " ...[truncated]"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.lower().split())


def exact_match_score(pred: str, golds: list[str]) -> float:
    n = normalize_answer(pred)
    return 1.0 if any(n == normalize_answer(g) for g in golds) else 0.0


def f1_score(pred: str, golds: list[str]) -> float:
    def _f1(pt, gt):
        if not pt or not gt:
            return 1.0 if pt == gt else 0.0
        ns = sum({t: min(pt.count(t), gt.count(t)) for t in pt}.values())
        return 0.0 if ns == 0 else 2 * ns / (len(pt) + len(gt))
    pt = normalize_answer(pred).split()
    return max(_f1(pt, normalize_answer(g).split()) for g in golds)


_STOP = {
    "the","a","an","of","in","on","at","to","for","and","or","is","was","were",
    "are","that","this","with","by","as","from","which","what","who","when",
    "where","did","does","how","many","much","same","also","has","have","had",
    "it","its","his","her","their",
}


def answer_containment(text: str, golds: list[str]) -> float:
    if not text or not golds:
        return 0.0
    given = set(_tok(text))
    for g in golds:
        g_toks = {t for t in _tok(g) if t not in _STOP and len(t) > 2}
        if g_toks and len(g_toks & given) / len(g_toks) >= 0.5:
            return 1.0
    return 0.0


def retrieval_hit_rate(chunks: list[str], golds: list[str]) -> float | None:
    if not chunks:
        return None
    return sum(1 for c in chunks if answer_containment(c, golds) > 0) / len(chunks)


def compute_dfs_score(queries: list[str], chunks: list[str]) -> float | None:
    if len(queries) < 2:
        return None
    hits = sum(
        1 for i in range(1, len(queries))
        if {t for t in _tok(chunks[i-1]) if t not in _STOP and len(t) > 2}
           & {t for t in _tok(queries[i]) if t not in _STOP and len(t) > 2}
    )
    return hits / (len(queries) - 1)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

_CONNECT_TIMEOUT = 15
_MAX_RETRIES = 3
_RETRY_BACKOFF = 15


def _llm_call(messages: list[dict], model: str, url: str, timeout: int) -> tuple[str, int, float]:
    payload = {
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512, "num_gpu": 0},
    }
    t0 = time.time()
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            r = requests.post(url, json=payload, timeout=(_CONNECT_TIMEOUT, timeout))
            r.raise_for_status()
            d = r.json()
            return (d["message"]["content"].strip(),
                    d.get("prompt_eval_count", 0) + d.get("eval_count", 0),
                    round(time.time() - t0, 2))
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                logging.getLogger(__name__).warning(f"LLM attempt {attempt}: {exc}, retrying...")
                time.sleep(_RETRY_BACKOFF)
    raise last_exc


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANSWER_STYLE = "Answer in as few words as possible (a name, phrase, or short sentence) — no explanation."

NO_CONTEXT_SYSTEM = f"You are answering a question about a book or movie from your own knowledge. {ANSWER_STYLE}"
FULL_CONTEXT_SYSTEM = f"Answer using ONLY the story text below. {ANSWER_STYLE}\n\n{{passages}}"
ORACLE_ANSWER_SYSTEM = f"Answer using ONLY the retrieved information below. {ANSWER_STYLE}"

MEMORY_VERBATIM_SYSTEM = (
    "You have access to a knowledge base from previous Q&A sessions about similar stories. "
    "Use it to help answer the question. {ANSWER_STYLE}"
)

MEMORY_QUERY_GEN_SYSTEM = (
    "You are searching a knowledge base built from previous Q&A sessions about similar stories. "
    "Search for relevant facts, strategies, or procedures that could help answer the question. "
    "Do not repeat a query you already made. "
    "Reply DONE if you have enough information. Otherwise reply with only a short search query (2-6 words)."
)

FUSED_QUERY_GEN_SYSTEM = (
    "You can search two sources: a knowledge base from previous sessions (facts, strategies, procedures) "
    "and the current story text. "
    "Do not repeat a query you already made. "
    "Reply DONE if you have enough. Otherwise reply with only a short search query (2-6 words)."
)

MEMORY_INTERACTIVE_ANSWER_SYSTEM = (
    f"Answer using ONLY the retrieved knowledge and story passages below. {ANSWER_STYLE}"
)

# Oracle full query gen (grounded)
def make_doc_query_gen_system(doc_title: str, doc_kind: str) -> str:
    label = {"gutenberg": "novel", "movie": "movie script"}.get(doc_kind, doc_kind)
    return (
        f"You are researching a question about '{doc_title}' ({label}) step by step. "
        "Use previous results to guide next queries. Do not repeat queries. "
        "Reply DONE if you have enough. Otherwise reply with only a short search query (2-6 words)."
    )


# ---------------------------------------------------------------------------
# Document retrieval index  (TF-IDF; mirrors E10 oracle_full)
# ---------------------------------------------------------------------------

class TFIDFDocIndex:
    def __init__(self, chunks: list[tuple[str, str]]):
        self.chunks = chunks
        self._doc_tokens = [_tok(text) for _, text in chunks]
        self._N = len(chunks) + 1

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

    def query(self, query_text: str, n_neighbors: int = 1) -> list[tuple[str, str]]:
        q_toks = _tok(query_text)
        vocab = set(q_toks)
        if not vocab:
            best_idx = 0
        else:
            q_vec = self._vec(q_toks, vocab)
            best_idx = max(range(len(self._doc_tokens)),
                           key=lambda i: self._cos(q_vec, self._vec(self._doc_tokens[i], vocab)))
        results = [self.chunks[best_idx]]
        for offset in [-1, 1]:
            nb = best_idx + offset
            if 0 <= nb < len(self.chunks):
                results.append(self.chunks[nb])
        return results


# ---------------------------------------------------------------------------
# Answer functions — baselines
# ---------------------------------------------------------------------------

def answer_no_context(q, model, url, timeout):
    ans, tok, el = _llm_call(
        [{"role": "system", "content": NO_CONTEXT_SYSTEM},
         {"role": "user", "content": f"Question: {q}\nAnswer:"}],
        model, url, timeout,
    )
    return ans, tok, el, {"given_text": "", "primary_chunks": []}


def answer_full_context(q, doc_text, max_words, model, url, timeout):
    trunc = truncate_to_budget(doc_text, max_words)
    ans, tok, el = _llm_call(
        [{"role": "system", "content": FULL_CONTEXT_SYSTEM.format(passages=trunc)},
         {"role": "user", "content": f"Question: {q}\nAnswer:"}],
        model, url, timeout,
    )
    return ans, tok, el, {"given_text": trunc, "primary_chunks": []}


def answer_oracle_full(q, chunks, k, doc_title, doc_kind, n_neighbors, model, url, timeout):
    index = TFIDFDocIndex(chunks)
    qgen_sys = make_doc_query_gen_system(doc_title, doc_kind)
    queries, p_titles, p_chunks, exp_chunks = [], [], [], []
    for _ in range(k):
        hist = "\n\n".join(f"QUERY {i+1}: {q2}\nRESULT {i+1}: [{t}] {c}"
                           for i, (q2, t, c) in enumerate(zip(queries, p_titles, p_chunks)))
        user = f"QUESTION: {q}"
        if hist:
            user += f"\n\n{hist}"
        user += "\n\nNext query?"
        qt, tok, _ = _llm_call(
            [{"role": "system", "content": qgen_sys}, {"role": "user", "content": user}],
            model, url, timeout,
        )
        if qt.strip().upper().startswith("DONE"):
            break
        retrieved = index.query(qt, n_neighbors=n_neighbors)
        queries.append(qt)
        p_titles.append(retrieved[0][0])
        p_chunks.append(retrieved[0][1])
        for _, c in retrieved[1:]:
            exp_chunks.append(c)

    seen: set[str] = set()
    hist_lines = []
    for i, (q2, t, c) in enumerate(zip(queries, p_titles, p_chunks)):
        if c not in seen:
            hist_lines.append(f"QUERY {i+1}: {q2}\nRESULT {i+1}: [{t}] {c}")
            seen.add(c)

    final_user = f"QUESTION: {q}\n\n" + "\n\n".join(hist_lines) + "\n\nAnswer:"
    ans, a_tok, el = _llm_call(
        [{"role": "system", "content": ORACLE_ANSWER_SYSTEM},
         {"role": "user", "content": final_user}],
        model, url, timeout,
    )
    all_text = "\n\n".join(dict.fromkeys(p_chunks + exp_chunks))
    dfs = compute_dfs_score(queries, p_chunks)
    return ans, a_tok, el, {
        "given_text": all_text, "primary_chunks": p_chunks,
        "queries": queries, "num_queries": len(queries), "dfs_score": dfs,
    }


# ---------------------------------------------------------------------------
# Answer functions — memory verbatim
# ---------------------------------------------------------------------------

def answer_memory_verbatim(
    q: str, memory_nodes: list[MemoryNode], max_words: int,
    model: str, url: str, timeout: int,
) -> tuple[str, int, float, dict]:
    memory_text = format_verbatim_context(memory_nodes, max_words)
    system = (
        "You have access to a knowledge base from previous Q&A sessions about similar stories.\n\n"
        f"KNOWLEDGE BASE:\n{memory_text}\n\n"
        f"{ANSWER_STYLE}"
    )
    ans, tok, el = _llm_call(
        [{"role": "system", "content": system},
         {"role": "user", "content": f"Question: {q}\nAnswer:"}],
        model, url, timeout,
    )
    return ans, tok, el, {"given_text": memory_text, "primary_chunks": [], "num_queries": 0}


# ---------------------------------------------------------------------------
# Answer functions — memory interactive
# ---------------------------------------------------------------------------

def answer_memory_interactive(
    q: str,
    memory_index,   # any MemoryIndex with .query(text, top_k)
    k: int,
    model: str,
    url: str,
    timeout: int,
    doc_index=None,  # optional second index for Fall_interactive_plus_doc
) -> tuple[str, int, float, dict]:
    qgen_sys = FUSED_QUERY_GEN_SYSTEM if doc_index else MEMORY_QUERY_GEN_SYSTEM
    queries: list[str] = []
    retrieved_labels: list[str] = []
    retrieved_chunks: list[str] = []
    total_tokens = 0

    for _ in range(k):
        hist = "\n\n".join(
            f"QUERY {i+1}: {q2}\nRESULT {i+1}: {c}"
            for i, (q2, c) in enumerate(zip(queries, retrieved_chunks))
        )
        user = f"QUESTION: {q}"
        if hist:
            user += f"\n\n{hist}"
        user += "\n\nNext search query?"

        qt, tok, _ = _llm_call(
            [{"role": "system", "content": qgen_sys}, {"role": "user", "content": user}],
            model, url, timeout,
        )
        total_tokens += tok
        if qt.strip().upper().startswith("DONE"):
            break

        # Search memory first; if fused, also search doc and take best
        mem_results = memory_index.query(qt, top_k=1)
        if not mem_results:
            continue
        mem_node, mem_score = mem_results[0]
        best_text = mem_node.format_verbatim()
        best_label = f"[MEMORY/{mem_node.format}]"

        if doc_index:
            doc_results = doc_index.query(qt, n_neighbors=0)
            doc_title_chunk, doc_text = doc_results[0]
            # Use doc chunk if memory hit was weak (score < 0.1) or if doc has more overlap
            doc_overlap = len(set(_tok(qt)) & set(_tok(doc_text)))
            mem_overlap = len(set(_tok(qt)) & set(_tok(mem_node.content)))
            if doc_overlap > mem_overlap or mem_score < 0.05:
                best_text = doc_text
                best_label = doc_title_chunk

        queries.append(qt)
        retrieved_labels.append(best_label)
        retrieved_chunks.append(best_text)

    seen: set[str] = set()
    hist_lines = []
    for i, (q2, lbl, c) in enumerate(zip(queries, retrieved_labels, retrieved_chunks)):
        if c not in seen:
            hist_lines.append(f"QUERY {i+1}: {q2}\nRESULT {i+1}: {lbl}\n{c}")
            seen.add(c)

    final_user = f"QUESTION: {q}\n\n" + "\n\n".join(hist_lines) + "\n\nAnswer:"
    ans, a_tok, el = _llm_call(
        [{"role": "system", "content": MEMORY_INTERACTIVE_ANSWER_SYSTEM},
         {"role": "user", "content": final_user}],
        model, url, timeout,
    )
    total_tokens += a_tok
    dfs = compute_dfs_score(queries, retrieved_chunks)
    return ans, total_tokens, el, {
        "given_text": "\n\n".join(dict.fromkeys(retrieved_chunks)),
        "primary_chunks": [],  # memory nodes, not doc chunks
        "queries": queries,
        "num_queries": len(queries),
        "dfs_score": dfs,
    }


# ---------------------------------------------------------------------------
# Condition dispatcher
# ---------------------------------------------------------------------------

def run_example(
    condition: str,
    ex: dict,
    cfg: dict,
    pool: MemoryPool,
    embed_fn,
    logger: logging.Logger,
    no_llm: bool,
) -> dict:
    model = cfg["model"]["name"]
    url = cfg["model"]["ollama_url"]
    timeout = cfg["model"]["timeout"]
    ev = cfg["evaluation"]
    ret = cfg["retrieval"]
    k = ret["max_queries"]
    n_neighbors = ret["n_neighbors"]
    max_words = ev["full_context_max_words"]
    chunk_words = ev["chunk_words"]
    overlap_words = ev.get("chunk_overlap_words", chunk_words // 5)
    verbatim_max_words = ret["verbatim_max_words"]
    filter_by_kind = ret.get("filter_by_kind", True)

    doc = ex["document"]
    question = ex["question"]["text"]
    golds = [a["text"] for a in ex["answers"] if a.get("text")] or [""]
    doc_id = doc["id"]
    doc_text = doc["text"]
    doc_kind = doc.get("kind", "")
    doc_word_count = doc.get("word_count") or len(doc_text.split())
    doc_title = doc.get("summary", {}).get("title") or doc_id
    summary_text = doc["summary"]["text"]

    if no_llm:
        ans, tok, el = f"[MOCK]", 0, 0.0
        extra = {"given_text": "", "primary_chunks": [], "num_queries": 0, "dfs_score": None, "queries": []}
    else:
        try:
            # ── baselines ──────────────────────────────────────────────────
            if condition == "no_context":
                ans, tok, el, extra = answer_no_context(question, model, url, timeout)

            elif condition == "full_context":
                ans, tok, el, extra = answer_full_context(question, doc_text, max_words, model, url, timeout)

            elif condition == "oracle_full":
                chunks = chunk_document(doc_text, chunk_words, overlap_words)
                ans, tok, el, extra = answer_oracle_full(
                    question, chunks, k, doc_title, doc_kind, n_neighbors, model, url, timeout,
                )

            # ── memory verbatim ────────────────────────────────────────────
            elif condition.endswith("_verbatim"):
                fmt_name = condition[:-len("_verbatim")]
                fmts = FORMATS if fmt_name == "Fall" else [f"{fmt_name}_raw" if fmt_name == "F1"
                                                            else f"{fmt_name}_fact" if fmt_name == "F2"
                                                            else f"{fmt_name}_procedure" if fmt_name == "F3"
                                                            else f"{fmt_name}_program" if fmt_name == "F4"
                                                            else f"{fmt_name}_episode"]
                kind_filter = doc_kind if filter_by_kind else None
                nodes = pool.nodes_for_formats(fmts, kind_filter)
                ans, tok, el, extra = answer_memory_verbatim(
                    question, nodes, verbatim_max_words, model, url, timeout,
                )

            # ── memory interactive ─────────────────────────────────────────
            elif condition.endswith("_interactive") or condition == "Fall_interactive_plus_doc":
                plus_doc = condition == "Fall_interactive_plus_doc"
                if plus_doc:
                    fmts = FORMATS
                else:
                    fmt_name = condition[:-len("_interactive")]
                    fmts = FORMATS if fmt_name == "Fall" else [f"{fmt_name}_raw" if fmt_name == "F1"
                                                                else f"{fmt_name}_fact" if fmt_name == "F2"
                                                                else f"{fmt_name}_procedure" if fmt_name == "F3"
                                                                else f"{fmt_name}_program" if fmt_name == "F4"
                                                                else f"{fmt_name}_episode"]
                kind_filter = doc_kind if filter_by_kind else None
                nodes = pool.nodes_for_formats(fmts, kind_filter)
                if not nodes:
                    # Fall back to no_context if pool is empty for this kind
                    ans, tok, el, extra = answer_no_context(question, model, url, timeout)
                else:
                    mem_index = pool.build_index(nodes, index_type="tfidf", embed_fn=embed_fn)
                    doc_index = None
                    if plus_doc:
                        chunks = chunk_document(doc_text, chunk_words, overlap_words)
                        doc_index = TFIDFDocIndex(chunks)
                    ans, tok, el, extra = answer_memory_interactive(
                        question, mem_index, k, model, url, timeout, doc_index=doc_index,
                    )
            else:
                raise ValueError(f"Unknown condition: {condition}")

        except Exception as e:
            logger.warning(f"[{condition}] {doc_id} error: {e}", exc_info=True)
            ans, tok, el, extra = "", 0, 0.0, {
                "given_text": "", "primary_chunks": [], "num_queries": None, "dfs_score": None, "queries": [],
            }

    em = exact_match_score(ans, golds)
    f1 = f1_score(ans, golds)
    containment = answer_containment(extra.get("given_text", ""), golds)
    hit_rate = retrieval_hit_rate(extra.get("primary_chunks", []) or [], golds)

    result = {
        "id": doc_id,
        "doc_kind": doc_kind,
        "condition": condition,
        "question": question,
        "gold_answer": golds[0],
        "predicted_answer": ans,
        "em": em,
        "f1": f1,
        "tokens": tok,
        "elapsed_s": el,
        "supporting_fact_recall": containment,
        "retrieval_hit_rate": hit_rate,
        "num_queries": extra.get("num_queries"),
        "dfs_score": extra.get("dfs_score"),
        "queries": extra.get("queries"),
        "doc_word_count": doc_word_count,
    }
    hit_str = f" hit={hit_rate:.0%}" if hit_rate is not None else ""
    logger.info(
        f"[{condition}/{doc_kind}] {doc_id[:12]}: em={em:.0f} f1={f1:.2f} "
        f"contain={containment:.0f}{hit_str} tok={tok}"
    )
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def compute_summary(results: list[dict]) -> dict:
    if not results:
        return {}
    hit_rates = [r["retrieval_hit_rate"] for r in results if r["retrieval_hit_rate"] is not None]
    dfs_scores = [r["dfs_score"] for r in results if r["dfs_score"] is not None]
    return {
        "num": len(results),
        "exact_match": _mean([r["em"] for r in results]),
        "f1": _mean([r["f1"] for r in results]),
        "mean_tokens": _mean([r["tokens"] for r in results]),
        "mean_supporting_fact_recall": _mean([r["supporting_fact_recall"] for r in results]),
        "mean_retrieval_hit_rate": _mean(hit_rates) if hit_rates else None,
        "mean_dfs_score": _mean(dfs_scores) if dfs_scores else None,
        "mean_num_queries": _mean([r["num_queries"] for r in results if r["num_queries"] is not None]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_format_names(fmts_spec: str) -> list[str]:
    """Map 'F1_raw' style to FORMATS entries."""
    result = []
    for token in fmts_spec.split(","):
        token = token.strip()
        if token.startswith("F") and not token.startswith("Fall"):
            num = token[1]
            mapping = {"1": "F1_raw", "2": "F2_fact", "3": "F3_procedure",
                       "4": "F4_program", "5": "F5_episode"}
            result.append(mapping.get(num, token))
        else:
            result.append(token)
    return result


def main():
    parser = argparse.ArgumentParser(description="E11 Phase 2: memory evaluation")
    parser.add_argument("--condition", default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)
    cfg = load_config()

    if args.num_tasks:
        cfg["evaluation"]["num_per_kind"] = args.num_tasks
    if args.max_queries is not None:
        cfg["retrieval"]["max_queries"] = args.max_queries

    if args.condition:
        selected = [c.strip() for c in args.condition.split(",")]
        invalid = [c for c in selected if c not in CONDITIONS]
        if invalid:
            logger.error(f"Unknown: {invalid}. Valid: {CONDITIONS}")
            sys.exit(1)
    else:
        selected = CONDITIONS

    # Load memory pool
    pool = MemoryPool(ROOT / cfg["output"]["memory_dir"])
    pool.load()
    logger.info(f"[pool] {pool.stats()}")

    # Load embed model only if needed
    embed_fn = None
    needs_embed = any("embed" in c for c in selected)
    if needs_embed and not args.no_llm:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[embed] Loading {cfg['oracle']['embed_model']}...")
            st = SentenceTransformer(cfg["oracle"]["embed_model"])
            embed_fn = lambda texts: st.encode(texts, show_progress_bar=False).tolist()
        except ImportError:
            logger.warning("[embed] sentence-transformers not installed; embed conditions will use tfidf fallback")

    logger.info("Loading validation split...")
    val_by_kind = load_val_examples(cfg, logger)
    val_flat = [ex for exs in val_by_kind.values() for ex in exs]

    results_dir = ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for condition in selected:
        logger.info(f"\n{'─'*60}\nCondition: {condition}\n{'─'*60}")
        out_path = results_dir / f"{condition}.json"

        if not args.fresh and out_path.exists():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            cached = existing.get("results", [])
            if len(cached) == len(val_flat):
                logger.info(f"[{condition}] {len(cached)} cached results, skipping")
                all_summaries.append({"condition": condition, "summary": existing.get("summary", {})})
                continue

        results = []
        for ex in val_flat:
            r = run_example(condition, ex, cfg, pool, embed_fn, logger, no_llm=args.no_llm)
            results.append(r)

        summary = compute_summary(results)
        out = {"condition": condition, "results": results, "summary": summary}
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        all_summaries.append({"condition": condition, "summary": summary})

    # Cross-condition summary
    (results_dir / "summary.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8"
    )

    # Results table
    print("\n=== E11 RESULTS — NarrativeQA Memory Ablation ===")
    hdr = f"{'Condition':<30} {'EM':>6} {'F1':>6} {'Contain':>8} {'HitRate':>8} {'DFS':>6} {'Qs':>5} {'Tok':>7}"
    print(hdr)
    print("─" * len(hdr))

    def _row(name: str, s: dict) -> str:
        hit = s.get("mean_retrieval_hit_rate")
        dfs = s.get("mean_dfs_score")
        nq  = s.get("mean_num_queries")
        return (
            f"{name:<30} "
            f"{s.get('exact_match', 0):>6.1%} "
            f"{s.get('f1', 0):>6.1%} "
            f"{s.get('mean_supporting_fact_recall', 0):>8.1%} "
            f"{(f'{hit:.1%}' if hit is not None else '—'):>8} "
            f"{(f'{dfs:.1%}' if dfs is not None else '—'):>6} "
            f"{(f'{nq:.1f}' if nq is not None else '—'):>5} "
            f"{s.get('mean_tokens', 0):>7.0f}"
        )

    print("── baselines ──")
    for row in all_summaries:
        if row["condition"] in BASELINE_CONDITIONS:
            print(_row(row["condition"], row["summary"]))
    print("── verbatim ──")
    for row in all_summaries:
        if row["condition"].endswith("_verbatim"):
            print(_row(row["condition"], row["summary"]))
    print("── interactive ──")
    for row in all_summaries:
        if row["condition"].endswith("_interactive") or row["condition"] == "Fall_interactive_plus_doc":
            print(_row(row["condition"], row["summary"]))


if __name__ == "__main__":
    main()
