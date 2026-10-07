"""
Phase 2 — MuSiQue procedural memory evaluation.

13 conditions across three axes:

  Baselines (no memory):
    no_context          question only
    full_context        all 20 passages in one block
    oracle_tfidf        K-round TF-IDF oracle over passages  (E7 baseline)

  Verbatim memory (memory dumped as prompt context; oracle still runs over passages):
    F1_verbatim         raw session transcripts
    F2_verbatim         gold decomposition chains  (near-oracle ceiling)
    F3_verbatim         natural-language procedures
    F4_verbatim         parameterized query programs
    F5_verbatim         canonical decomposition patterns
    Fall_verbatim       all formats fused

  Guided oracle (memory injected into query-gen system prompt; oracle then runs
                 K rounds over the 20 passages — this is the KEY test):
    F3_guided           procedure hint in system prompt
    F4_guided           program hint in system prompt
    Fall_guided         all-format hint in system prompt

  Fused (memory pool + passages queried in same interactive loop):
    Fall_fused          interleaved memory and passage retrieval

Key metrics:
  EM / F1             answer correctness
  supporting_recall   fraction of gold supporting passages retrieved
  dfs_score           chain-following in oracle queries

Usage:
  python experiments/musique_memory/evaluate.py [options]

  --no-llm             Structural test
  --condition COND     Comma-separated subset
  --num-tasks N        Override num_per_hop
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
from typing import Optional

import requests
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.musique_memory.memory_store import (
    FORMATS, MemoryNode, MemoryPool, TFIDFMemoryIndex,
    format_verbatim_context, _tok,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
_HOP_RE = re.compile(r"^(\d+)hop")

BASELINE_CONDITIONS   = ["no_context", "full_context", "oracle_tfidf"]
VERBATIM_CONDITIONS   = ["F1_verbatim", "F2_verbatim", "F3_verbatim",
                          "F4_verbatim", "F5_verbatim",
                          "F6_verbatim", "F7_verbatim", "Fall_verbatim"]
GUIDED_CONDITIONS     = ["F3_guided", "F3_typed_guided", "F3_interactive",
                          "F4_guided", "F4_slot_fill",
                          "F6_guided", "F7_guided", "Fall_guided"]
FUSED_CONDITIONS      = ["Fall_fused"]
CONDITIONS = BASELINE_CONDITIONS + VERBATIM_CONDITIONS + GUIDED_CONDITIONS + FUSED_CONDITIONS

# Maps condition name → list of FORMATS to pull from pool
_VERBATIM_FMT_MAP = {
    "F1_verbatim": ["F1_raw"],
    "F2_verbatim": ["F2_chain"],
    "F3_verbatim": ["F3_procedure"],
    "F4_verbatim": ["F4_program"],
    "F5_verbatim": ["F5_decomp"],
    "F6_verbatim": ["F6_preference"],
    "F7_verbatim": ["F7_correction"],
    "Fall_verbatim": FORMATS,
}
_GUIDED_FMT_MAP = {
    "F3_guided":        ["F3_procedure"],
    "F3_typed_guided":  ["F3_procedure"],
    "F3_interactive":   ["F3_procedure", "F6_preference", "F7_correction"],
    "F4_guided":        ["F4_program"],
    "F4_slot_fill":     ["F4_program"],
    "F6_guided":        ["F6_preference"],
    "F7_guided":        ["F7_correction"],
    "Fall_guided":      FORMATS,
}

_QWORD_RE = re.compile(r"^(who|what|where|when|which|how)", re.IGNORECASE)

def classify_question_type(question: str, hop: int) -> str:
    """Extract hop{N}_{wh1}_{wh2}... tag from question wh-words, matching populate.py."""
    words = question.lower().split()
    wh_words = [m.group(1) for w in words if (m := _QWORD_RE.match(w))][:hop]
    if not wh_words:
        wh_words = ["what"]
    return "hop{}_{}" .format(hop, "_".join(wh_words))


# ---------------------------------------------------------------------------
# Config / env
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
    logger.info(f"[dataset] {len(ds)} val rows")
    buckets = {f"hop{h}": [] for h in ev["hop_buckets"]}
    for ex in ds:
        m = _HOP_RE.match(ex["id"])
        if not m:
            continue
        label = f"hop{m.group(1)}"
        if label in buckets and len(buckets[label]) < ev["num_per_hop"]:
            buckets[label].append(ex)
        if all(len(v) >= ev["num_per_hop"] for v in buckets.values()):
            break
    for label, exs in buckets.items():
        logger.info(f"[dataset] {label}: {len(exs)} val examples")
    return buckets


def format_passages(paragraphs: list[dict]) -> list[tuple[str, str]]:
    return [(p["title"], f"{p['title']}: {p['paragraph_text']}") for p in paragraphs]


def gold_titles(example: dict) -> set[str]:
    return {p["title"] for p in example["paragraphs"] if p["is_supporting"]}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.lower().split())


def exact_match_score(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return 1.0 if pt == gt else 0.0
    ns = sum({t: min(pt.count(t), gt.count(t)) for t in pt}.values())
    return 0.0 if ns == 0 else 2 * ns / (len(pt) + len(gt))


def supporting_recall(retrieved_titles: list[str], gold_sup: set[str]) -> float:
    if not gold_sup:
        return 0.0
    return len({t for t in retrieved_titles if t in gold_sup}) / len(gold_sup)


_STOP = {"the","a","an","of","in","on","at","to","for","and","or","is","was","were",
         "are","that","this","with","by","as","from","which","what","who","when","where"}


def compute_dfs_score(queries: list[str], retrieved_texts: list[str]) -> float | None:
    if len(queries) < 2:
        return None
    hits = sum(
        1 for i in range(1, len(queries))
        if {t for t in _tok(retrieved_texts[i-1]) if t not in _STOP and len(t) > 2}
           & {t for t in _tok(queries[i]) if t not in _STOP and len(t) > 2}
    )
    return hits / (len(queries) - 1)


# ---------------------------------------------------------------------------
# LLM
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
# Query cleaning  (fixes QUERY_CONTAM failure mode)
# ---------------------------------------------------------------------------

def clean_query(raw: str) -> str:
    """Strip procedure preamble from LLM output; keep only the search query.

    The model sometimes echoes the procedure description before the actual
    query.  We strip everything before the last occurrence of a "query:"
    marker, then take the last substantive line, then cap at 10 words.
    """
    q = raw.strip()
    # Extract text after the last "search query:" / "query:" marker
    for marker in ["search query:", "query:", "search:"]:
        idx = q.lower().rfind(marker)
        if idx >= 0:
            q = q[idx + len(marker):].strip()
            break
    # Take last non-empty, non-parenthetical line
    lines = [l.strip() for l in q.splitlines()
             if l.strip() and not l.strip().startswith("(")]
    if lines:
        q = lines[-1]
    # Strip leading hop-type prefix (e.g. "hop3_who_what: ...")
    q = re.sub(r"^hop\d+_\w+:\s*", "", q)
    # Strip surrounding quotes
    q = q.strip("\"'")
    # Cap at 10 words to prevent sentence-length queries polluting TF-IDF
    words = q.split()
    if len(words) > 10:
        q = " ".join(words[:10])
    return q.strip()


# ---------------------------------------------------------------------------
# Slot-filling for F4_slot_fill condition
# ---------------------------------------------------------------------------

def _json_from_response(text: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def fill_slots_from_question(
    question: str,
    program_node: "MemoryNode",
    model: str,
    url: str,
    timeout: int,
) -> list[str]:
    """
    One LLM call to fill slot values from the question, then render the
    query_sequence templates.  Returns a list of concrete queries to issue
    before falling back to free-form generation.
    """
    m = program_node.meta
    slots = m.get("slots", [])
    query_seq = m.get("query_sequence", [])
    if not slots or not query_seq:
        return []

    slot_lines = "\n".join(f'  "{s}": "..."' for s in slots)
    prompt = (
        f"Question: {question}\n\n"
        f"Query program '{m.get('name', '')}' has these slot variables: {', '.join(slots)}\n"
        f"Example queries: {' → '.join(query_seq[:3])}\n\n"
        f"Extract the slot values from the question. Write only JSON:\n"
        f"{{\n{slot_lines}\n}}"
    )
    resp, _, _ = _llm_call(
        [{"role": "system", "content": "You extract named entities from questions. Output only JSON."},
         {"role": "user", "content": prompt}],
        model, url, timeout,
    )
    filled = _json_from_response(resp) or {}
    # Render templates with filled values; keep unfilled slots as-is
    rendered = []
    for template in query_seq:
        q = template
        for slot, val in filled.items():
            q = q.replace(f"{{{slot}}}", str(val))
        rendered.append(q)
    return rendered


# ---------------------------------------------------------------------------
# Passage-level TF-IDF index
# ---------------------------------------------------------------------------

class PassageIndex:
    def __init__(self, passages: list[tuple[str, str]]):
        self.passages = passages
        self._doc_tokens = [_tok(text) for _, text in passages]
        self._N = len(passages) + 1

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

    def query(self, query_text: str) -> tuple[str, str, float]:
        q_toks = _tok(query_text)
        vocab = set(q_toks)
        if not vocab:
            return self.passages[0][0], self.passages[0][1], 0.0
        q_vec = self._vec(q_toks, vocab)
        best_idx, best_score = 0, -1.0
        for i, toks in enumerate(self._doc_tokens):
            s = self._cos(q_vec, self._vec(toks, vocab))
            if s > best_score:
                best_idx, best_score = i, s
        return self.passages[best_idx][0], self.passages[best_idx][1], best_score


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANSWER_STYLE = "Answer in as few words as possible — a name, date, or short phrase. No explanation."

NO_CONTEXT_SYSTEM = f"Answer the multi-hop question from your own knowledge. {ANSWER_STYLE}"
FULL_CONTEXT_SYSTEM = f"Answer using ONLY the passages below. {ANSWER_STYLE}\n\n{{passages}}"
ANSWER_FROM_RETRIEVED = f"Answer using ONLY the retrieved passages below. {ANSWER_STYLE}"

BASE_QUERY_GEN = (
    "You are answering a multi-hop question step by step. "
    "Each query retrieves one passage. Use results to guide next queries. "
    "Reply DONE if you have enough to answer. "
    "Otherwise reply with only a short search query (3-7 words)."
)


def make_guided_query_gen_system(memory_context: str) -> str:
    """Inject strategy advice from a past session. Framed as guidance, not as the solution."""
    return (
        "You are solving a multi-hop question by searching a knowledge base step by step. "
        "A past session that faced similar questions has shared this strategic advice:\n\n"
        f"{memory_context}\n\n"
        "Use this strategy to guide HOW you search — the approach, the ordering, what to anchor on. "
        "You must find the answer yourself through your own queries. "
        "Each query retrieves one passage. Reply with only a short search query (3-7 words), "
        "or DONE when ready to answer."
    )


INTERACTIVE_QUERY_GEN = (
    "You are solving a multi-hop question by searching a knowledge base. "
    "For each round, output exactly ONE of these actions:\n\n"
    "  SEARCH: <query>    — search for a passage (3-7 words)\n"
    "  QUESTION: <query>  — ask the strategy advisor how to approach this "
    "(use only when genuinely unsure how to proceed — not for factual lookups)\n"
    "  DONE               — when ready to synthesize your answer\n\n"
    "The strategy advisor remembers how similar questions were solved before. "
    "It will give you strategic guidance, not the answer. "
    "Output only one action per turn, nothing else."
)


# ---------------------------------------------------------------------------
# Oracle runner (shared by oracle_tfidf, guided, fused, interactive)
# ---------------------------------------------------------------------------

def run_oracle(
    question: str,
    passages: list[tuple[str, str]],
    k: int,
    model: str,
    url: str,
    timeout: int,
    query_gen_system: str,
    memory_index: Optional[TFIDFMemoryIndex] = None,
    prefilled_queries: list[str] | None = None,
    interactive_memory_index: Optional[TFIDFMemoryIndex] = None,
) -> tuple[str, int, float, dict]:
    """
    Runs a K-round retrieval oracle over `passages`.

    memory_index           — fused mode: also searches memory pool each round
    prefilled_queries      — F4_slot_fill: use pre-rendered queries for first N rounds
    interactive_memory_index — F3_interactive: agent may output QUESTION: to request
                              strategy advice from the memory pool mid-session

    QUESTION: rounds do not count as passage retrievals; they inject strategy
    advice into the conversation history and are excluded from DFS/recall metrics.
    """
    passage_index = PassageIndex(passages)

    # Separate tracking: search rounds (for metrics) vs all history (for LLM context)
    search_queries: list[str] = []
    search_titles: list[str] = []
    search_texts: list[str] = []
    all_history: list[tuple[str, str, str]] = []   # (label, title_or_tag, text)
    questions_asked: list[dict] = []
    total_tokens = 0
    t0 = time.time()

    for round_num in range(k):
        # Build context for LLM from all history (searches + strategy exchanges)
        hist_parts = []
        for label, tag, txt in all_history:
            if tag == "[STRATEGY]":
                hist_parts.append(f"STRATEGY REQUEST: {label}\nSTRATEGY ADVICE: {txt}")
            else:
                hist_parts.append(f"SEARCH: {label}\nRESULT: [{tag}] {txt}")
        hist = "\n\n".join(hist_parts)

        # Determine the action to take this round
        if prefilled_queries and round_num < len(prefilled_queries):
            raw_response = prefilled_queries[round_num]
            tok = 0
        else:
            user = f"QUESTION: {question}"
            if hist:
                user += f"\n\n{hist}"
            user += "\n\nNext action?"
            raw_response, tok, _ = _llm_call(
                [{"role": "system", "content": query_gen_system},
                 {"role": "user", "content": user}],
                model, url, timeout,
            )

        total_tokens += tok
        raw_stripped = raw_response.strip()

        if raw_stripped.upper().startswith("DONE"):
            break

        # Interactive: agent asks strategy advisor a question
        if raw_stripped.upper().startswith("QUESTION:") and interactive_memory_index is not None:
            q_text = raw_stripped[len("QUESTION:"):].strip()
            hits = interactive_memory_index.query(q_text, top_k=1, success_weight=0.3)
            if hits:
                strategy_text = hits[0][0].format_verbatim()
            else:
                strategy_text = "No prior strategy found for this question type."
            questions_asked.append({"question": q_text, "strategy": strategy_text})
            all_history.append((q_text, "[STRATEGY]", strategy_text))
            continue  # don't count as a search round

        # Regular search round
        # Strip "SEARCH:" prefix if agent used that syntax
        if raw_stripped.upper().startswith("SEARCH:"):
            raw_stripped = raw_stripped[len("SEARCH:"):].strip()

        qt = clean_query(raw_stripped)
        title, text, p_score = passage_index.query(qt)

        # Fused: also retrieve from memory pool; use if relevance comparable
        if memory_index:
            mem_hits = memory_index.query(qt, top_k=1, success_weight=0.3)
            if mem_hits:
                mem_node, mem_score = mem_hits[0]
                if mem_score > p_score * 0.8:
                    text = f"[MEMORY]: {mem_node.format_verbatim()}\n[PASSAGE]: {text}"

        search_queries.append(qt)
        search_titles.append(title)
        search_texts.append(text)
        all_history.append((qt, title, text))

    # Deduplicate search results for final answer synthesis
    seen: set[str] = set()
    unique = [(q, t, c) for q, t, c in zip(search_queries, search_titles, search_texts)
              if c not in seen and not seen.add(c)]

    final_user = (
        f"QUESTION: {question}\n\n"
        + "\n\n".join(
            f"SEARCH {i+1}: {q}\nRESULT {i+1}: [{t}] {c}"
            for i, (q, t, c) in enumerate(unique)
        )
        + "\n\nAnswer:"
    )
    ans, a_tok, _ = _llm_call(
        [{"role": "system", "content": ANSWER_FROM_RETRIEVED},
         {"role": "user", "content": final_user}],
        model, url, timeout,
    )
    total_tokens += a_tok
    elapsed = round(time.time() - t0, 2)
    dfs = compute_dfs_score(search_queries, search_texts)

    return ans, total_tokens, elapsed, {
        "queries": search_queries,
        "retrieved_titles": search_titles,
        "num_queries": len(search_queries),
        "dfs_score": dfs,
        "questions_asked": questions_asked,
    }


# ---------------------------------------------------------------------------
# Per-condition dispatch
# ---------------------------------------------------------------------------

def run_example(
    condition: str,
    ex: dict,
    cfg: dict,
    pool: MemoryPool,
    logger: logging.Logger,
    no_llm: bool,
) -> dict:
    model = cfg["model"]["name"]
    url = cfg["model"]["ollama_url"]
    timeout = cfg["model"]["timeout"]
    k = cfg["retrieval"]["max_queries"]
    verbatim_max_words = cfg["retrieval"]["verbatim_max_words"]
    filter_by_hop = cfg["retrieval"].get("filter_by_hop", True)

    question = ex["question"]
    gold_answer = ex["answer"]
    qid = ex["id"]
    m = _HOP_RE.match(qid)
    hop = int(m.group(1)) if m else 2
    passages = format_passages(ex["paragraphs"])
    gold_sup = gold_titles(ex)

    all_passages_text = "\n\n".join(f"[{p['title']}] {p['paragraph_text']}" for p in ex["paragraphs"])
    hop_filter = hop if filter_by_hop else None

    if no_llm:
        return {
            "id": qid, "hop": hop, "condition": condition,
            "question": question, "gold_answer": gold_answer,
            "predicted_answer": "[MOCK]",
            "em": 0.0, "f1": 0.0, "tokens": 0, "elapsed_s": 0.0,
            "supporting_recall": 0.0, "dfs_score": None,
            "num_queries": 0, "queries": [],
        }

    extra: dict = {}
    try:
        # ── baselines ──────────────────────────────────────────────────────
        if condition == "no_context":
            ans, tok, el = _llm_call(
                [{"role": "system", "content": NO_CONTEXT_SYSTEM},
                 {"role": "user", "content": f"Question: {question}\nAnswer:"}],
                model, url, timeout,
            )
            retrieved_titles, queries, dfs, num_q = [], [], None, 0

        elif condition == "full_context":
            ans, tok, el = _llm_call(
                [{"role": "system", "content": FULL_CONTEXT_SYSTEM.format(passages=all_passages_text)},
                 {"role": "user", "content": f"Question: {question}\nAnswer:"}],
                model, url, timeout,
            )
            retrieved_titles, queries, dfs, num_q = [], [], None, 0

        elif condition == "oracle_tfidf":
            ans, tok, el, extra = run_oracle(
                question, passages, k, model, url, timeout, BASE_QUERY_GEN,
            )
            retrieved_titles = extra["retrieved_titles"]
            queries = extra["queries"]
            dfs = extra["dfs_score"]
            num_q = extra["num_queries"]

        # ── verbatim memory ────────────────────────────────────────────────
        elif condition in _VERBATIM_FMT_MAP:
            fmts = _VERBATIM_FMT_MAP[condition]
            nodes = pool.nodes_for_formats(fmts, hop_filter)
            memory_text = format_verbatim_context(nodes, verbatim_max_words)
            # Memory as context hint; oracle still runs over passages
            qgen_sys = make_guided_query_gen_system(memory_text) if memory_text else BASE_QUERY_GEN
            ans, tok, el, extra = run_oracle(
                question, passages, k, model, url, timeout, qgen_sys,
            )
            retrieved_titles = extra["retrieved_titles"]
            queries = extra["queries"]
            dfs = extra["dfs_score"]
            num_q = extra["num_queries"]

        # ── guided oracle ──────────────────────────────────────────────────
        elif condition in _GUIDED_FMT_MAP:
            fmts = _GUIDED_FMT_MAP[condition]
            nodes = pool.nodes_for_formats(fmts, hop_filter)

            if condition == "F3_typed_guided":
                # Typed retrieval: filter to matching question_type, fall back to all
                q_type = classify_question_type(question, hop)
                typed_nodes = [n for n in nodes if n.meta.get("question_type") == q_type]
                retrieval_nodes = typed_nodes if typed_nodes else nodes
                if retrieval_nodes:
                    mem_index = pool.build_index(retrieval_nodes)
                    hits = mem_index.query(question, top_k=2, success_weight=0.3)
                    memory_context = "\n\n".join(node.format_verbatim() for node, _ in hits)
                else:
                    memory_context = ""
                qgen_sys = make_guided_query_gen_system(memory_context) if memory_context else BASE_QUERY_GEN
                ans, tok, el, extra = run_oracle(
                    question, passages, k, model, url, timeout, qgen_sys,
                )

            elif condition == "F3_interactive":
                # Interactive: agent can ask the memory pool strategy questions mid-session
                interactive_idx = pool.build_index(nodes) if nodes else None
                ans, tok, el, extra = run_oracle(
                    question, passages, k, model, url, timeout,
                    INTERACTIVE_QUERY_GEN,
                    interactive_memory_index=interactive_idx,
                )

            elif condition == "F4_slot_fill":
                if nodes:
                    mem_index = pool.build_index(nodes)
                    hits = mem_index.query(question, top_k=1, success_weight=0.3)
                else:
                    hits = []
                prefilled: list[str] = []
                if hits:
                    prefilled = fill_slots_from_question(
                        question, hits[0][0], model, url, timeout,
                    )
                ans, tok, el, extra = run_oracle(
                    question, passages, k, model, url, timeout,
                    BASE_QUERY_GEN, prefilled_queries=prefilled or None,
                )

            else:
                # Standard guided: retrieve top-2 by composite score, inject as strategy
                if nodes:
                    mem_index = pool.build_index(nodes)
                    hits = mem_index.query(question, top_k=2, success_weight=0.3)
                    memory_context = "\n\n".join(node.format_verbatim() for node, _ in hits)
                else:
                    memory_context = ""
                qgen_sys = make_guided_query_gen_system(memory_context) if memory_context else BASE_QUERY_GEN
                ans, tok, el, extra = run_oracle(
                    question, passages, k, model, url, timeout, qgen_sys,
                )

            retrieved_titles = extra["retrieved_titles"]
            queries = extra["queries"]
            dfs = extra["dfs_score"]
            num_q = extra["num_queries"]

        # ── fused (memory pool + passages in same loop) ────────────────────
        elif condition == "Fall_fused":
            nodes = pool.nodes_for_formats(FORMATS, hop_filter)
            mem_index = pool.build_index(nodes) if nodes else None
            ans, tok, el, extra = run_oracle(
                question, passages, k, model, url, timeout, BASE_QUERY_GEN,
                memory_index=mem_index,
            )
            retrieved_titles = extra["retrieved_titles"]
            queries = extra["queries"]
            dfs = extra["dfs_score"]
            num_q = extra["num_queries"]

        else:
            raise ValueError(f"Unknown condition: {condition}")

    except Exception as e:
        logger.warning(f"[{condition}] {qid} error: {e}", exc_info=True)
        ans, tok, el = "", 0, 0.0
        retrieved_titles, queries, dfs, num_q = [], [], None, 0
        extra = {}

    em = exact_match_score(ans, gold_answer)
    f1 = f1_score(ans, gold_answer)
    sup_rec = supporting_recall(retrieved_titles, gold_sup)

    logger.info(
        f"[{condition}/hop{hop}] {qid[:12]}: "
        f"em={em:.0f} f1={f1:.2f} sup={sup_rec:.0%} "
        f"dfs={f'{dfs:.0%}' if dfs is not None else '—'} tok={tok}"
    )
    return {
        "id": qid, "hop": hop, "condition": condition,
        "question": question, "gold_answer": gold_answer,
        "predicted_answer": ans,
        "em": em, "f1": f1, "tokens": tok, "elapsed_s": el,
        "supporting_recall": sup_rec,
        "dfs_score": dfs,
        "num_queries": num_q,
        "queries": queries,
        "questions_asked": extra.get("questions_asked", []),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def compute_summary(results: list[dict]) -> dict:
    if not results:
        return {}
    dfs = [r["dfs_score"] for r in results if r["dfs_score"] is not None]
    return {
        "num": len(results),
        "exact_match": _mean([r["em"] for r in results]),
        "f1": _mean([r["f1"] for r in results]),
        "mean_tokens": _mean([r["tokens"] for r in results]),
        "mean_supporting_recall": _mean([r["supporting_recall"] for r in results]),
        "mean_dfs_score": _mean(dfs) if dfs else None,
        "mean_num_queries": _mean([r["num_queries"] for r in results if r["num_queries"] is not None]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E12 Phase 2: MuSiQue memory evaluation")
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
        cfg["evaluation"]["num_per_hop"] = args.num_tasks
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

    pool = MemoryPool(ROOT / cfg["output"]["memory_dir"])
    pool.load()
    logger.info(f"[pool] {pool.stats()}")

    logger.info("Loading val split...")
    val_by_hop = load_val_examples(cfg, logger)
    val_flat = [ex for exs in val_by_hop.values() for ex in exs]

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
                logger.info(f"[{condition}] {len(cached)} cached, skipping")
                all_summaries.append({"condition": condition, "summary": existing.get("summary", {})})
                continue

        results = [run_example(condition, ex, cfg, pool, logger, args.no_llm)
                   for ex in val_flat]

        summary = compute_summary(results)
        out_path.write_text(json.dumps({"condition": condition, "results": results, "summary": summary}, indent=2), encoding="utf-8")
        all_summaries.append({"condition": condition, "summary": summary})

    (results_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    # ── Results table ────────────────────────────────────────────────────────
    print("\n=== E12 RESULTS — MuSiQue Procedural Memory ===")
    hdr = f"{'Condition':<24} {'EM':>6} {'F1':>6} {'Sup.Rec':>8} {'DFS':>6} {'Qs':>4} {'Tok':>7}"
    print(hdr)
    print("─" * len(hdr))

    def _row(name: str, s: dict) -> str:
        dfs = s.get("mean_dfs_score")
        nq  = s.get("mean_num_queries")
        return (
            f"{name:<24} "
            f"{s.get('exact_match', 0):>6.1%} "
            f"{s.get('f1', 0):>6.1%} "
            f"{s.get('mean_supporting_recall', 0):>8.1%} "
            f"{(f'{dfs:.1%}' if dfs is not None else '—'):>6} "
            f"{(f'{nq:.1f}' if nq is not None else '—'):>4} "
            f"{s.get('mean_tokens', 0):>7.0f}"
        )

    print("── baselines ──")
    for row in all_summaries:
        if row["condition"] in BASELINE_CONDITIONS:
            print(_row(row["condition"], row["summary"]))
    print("── verbatim (memory→oracle) ──")
    for row in all_summaries:
        if row["condition"] in VERBATIM_CONDITIONS:
            print(_row(row["condition"], row["summary"]))
    print("── guided oracle ──")
    for row in all_summaries:
        if row["condition"] in GUIDED_CONDITIONS:
            print(_row(row["condition"], row["summary"]))
    print("── fused ──")
    for row in all_summaries:
        if row["condition"] in FUSED_CONDITIONS:
            print(_row(row["condition"], row["summary"]))

    # By hop breakdown
    for hop_label in [f"hop{h}" for h in cfg["evaluation"]["hop_buckets"]]:
        hop = int(hop_label.replace("hop", ""))
        print(f"\n── {hop_label} ──")
        print(hdr)
        print("─" * len(hdr))
        hop_summaries = []
        for row in all_summaries:
            out_path = results_dir / f"{row['condition']}.json"
            if out_path.exists():
                data = json.loads(out_path.read_text(encoding="utf-8"))
                hop_results = [r for r in data.get("results", []) if r.get("hop") == hop]
                hop_summaries.append({"condition": row["condition"], "summary": compute_summary(hop_results)})
        for row in hop_summaries:
            print(_row(row["condition"], row["summary"]))


if __name__ == "__main__":
    main()
