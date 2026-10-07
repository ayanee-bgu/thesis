"""
E8: QASPER — Interactive Retrieval vs Static Context, by paper length.

Research question: E7 found full_context's advantage over interactive_retrieval
(seen in E5/E6 on 10-passage benchmarks) reverses at MuSiQue's 20-passage scale
— dumping more context starts actively hurting once there's enough of it to
confuse the model among distractors. QASPER pushes this into a new regime:
single research papers ranging ~1,400-6,400 words (~2,000-8,300 tokens),
right at the edge of llama3:8b's 8192-token context window. Does full_context
degrade PROGRESSIVELY as paper length approaches the window limit, before it's
literally blown outright (that harder case is E9, on much longer documents)?

Four conditions, same mechanisms as E5-E7:
  no_context          — question only, no paper (cold-start baseline)
  full_context        — the paper's full text (all sections) in the prompt
  summary_doc         — one static LLM-generated summary of the paper
  interactive_retrieval — up to K=3 queries against a TF-IDF retrieval controller
                         over the paper's SECTIONS (each section = one chunk)

Length buckets (run separately, same conditions): short / medium / long —
tertile-bucketed by paper word count across the QASPER validation set.

Evidence recall here is TEXT-CONTAINMENT, not title-overlap (QASPER's gold
`evidence` field is raw quoted text snippets, not passage titles like
HotpotQA/2Wiki): fraction of gold evidence snippets that appear verbatim in
whatever text the model was actually given.

Usage:
  python experiments/qasper_transfer/main.py [options]

  --no-llm             Bypass Ollama (structural test only)
  --condition COND     Comma-separated conditions to run (default: all)
  --num-tasks N        Override num_per_bucket
  --max-queries K      Override retrieval.max_queries
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

import requests
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = Path(__file__).parent / "config.yaml"

CONDITIONS = ["no_context", "full_context", "summary_doc", "interactive_retrieval"]


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
    """Minimal .env loader — sets os.environ without overriding existing vars."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _paper_word_count(full_text: dict) -> int:
    return sum(len(" ".join(paras).split()) for paras in full_text["paragraphs"])


def _build_gold_candidates(question_answers: dict) -> tuple[list[str], set[str]]:
    """
    QASPER's per-question `answers` entry is {'answer': [per-annotator dicts], ...}.
    Returns (list of candidate gold-answer strings, set of gold evidence text snippets).
    """
    candidates = []
    evidence: set[str] = set()
    for a in question_answers["answer"]:
        if a.get("unanswerable"):
            candidates.append("Unanswerable")
        elif a.get("yes_no") is not None:
            candidates.append("Yes" if a["yes_no"] else "No")
        elif a.get("free_form_answer"):
            candidates.append(a["free_form_answer"])
        elif a.get("extractive_spans"):
            candidates.append("; ".join(a["extractive_spans"]))
        for ev in a.get("evidence") or []:
            if ev and not ev.startswith("FLOAT SELECTED"):
                evidence.add(ev)
    if not candidates:
        candidates = ["Unanswerable"]
    return candidates, evidence


def load_qasper(cfg: dict, logger: logging.Logger) -> dict[str, list[dict]]:
    """
    Load QASPER validation set, flatten to (paper, question) pairs, tertile-bucket
    by paper word count into short/medium/long, take the first num_per_bucket
    flattened examples per bucket (deterministic — same slice every run).
    Returns {"short": [...], "medium": [...], "long": [...]}.
    """
    load_dotenv_manual(ROOT / ".env")
    from datasets import load_dataset

    ds = load_dataset(
        cfg["experiment"]["dataset"],
        split=cfg["experiment"]["split"],
        revision=cfg["experiment"]["revision"],
    )
    logger.info(f"[dataset] Loaded {len(ds)} papers from {cfg['experiment']['dataset']}")

    word_counts = [_paper_word_count(ds[i]["full_text"]) for i in range(len(ds))]
    sorted_wc = sorted(word_counts)
    n = len(sorted_wc)
    lo_cut = sorted_wc[n // 3]
    hi_cut = sorted_wc[(2 * n) // 3]
    logger.info(f"[dataset] length tertile cuts: short<{lo_cut} <= medium<{hi_cut} <= long (words)")

    def bucket_for(wc: int) -> str:
        if wc < lo_cut:
            return "short"
        if wc < hi_cut:
            return "medium"
        return "long"

    num_per_bucket = cfg["experiment"]["num_per_bucket"]
    out: dict[str, list[dict]] = {b: [] for b in cfg["experiment"]["length_buckets"]}

    for i in range(len(ds)):
        paper = ds[i]
        wc = word_counts[i]
        bucket = bucket_for(wc)
        if bucket not in out or len(out[bucket]) >= num_per_bucket:
            continue
        qas = paper["qas"]
        for qidx in range(len(qas["question"])):
            if len(out[bucket]) >= num_per_bucket:
                break
            candidates, evidence = _build_gold_candidates(qas["answers"][qidx])
            out[bucket].append({
                "id": qas["question_id"][qidx],
                "question": qas["question"][qidx],
                "answer_candidates": candidates,
                "gold_evidence": evidence,
                "full_text": paper["full_text"],
                "paper_word_count": wc,
            })
        if all(len(v) >= num_per_bucket for v in out.values()):
            break

    for label, examples in out.items():
        logger.info(f"[dataset] {label}: {len(examples)} examples selected")
    return out


def format_passages_qasper(full_text: dict) -> list[tuple[str, str]]:
    """One chunk per paper SECTION — already naturally segmented, no invented chunking."""
    chunks = []
    for name, paras in zip(full_text["section_name"], full_text["paragraphs"]):
        title = name if name else "(untitled section)"
        chunks.append((title, f"{title}: " + " ".join(paras)))
    return chunks


def truncate_to_budget(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ...[truncated]"


# ---------------------------------------------------------------------------
# Answer normalization / scoring (standard SQuAD/HotpotQA style, multi-candidate)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, candidates: list[str]) -> float:
    norm_pred = normalize_answer(prediction)
    return 1.0 if any(norm_pred == normalize_answer(c) for c in candidates) else 0.0


def f1_score(prediction: str, candidates: list[str]) -> float:
    def _f1(pred_tokens, gold_tokens):
        if not pred_tokens or not gold_tokens:
            return 1.0 if pred_tokens == gold_tokens else 0.0
        common = {}
        for t in pred_tokens:
            common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    pred_tokens = normalize_answer(prediction).split()
    return max(_f1(pred_tokens, normalize_answer(c).split()) for c in candidates)


def evidence_containment_recall(given_text: str, gold_evidence: set[str]) -> float:
    """Fraction of gold evidence snippets that appear verbatim in what the model was given."""
    if not gold_evidence:
        return 1.0
    if not given_text:
        return 0.0
    hits = sum(1 for ev in gold_evidence if ev in given_text)
    return hits / len(gold_evidence)


# ---------------------------------------------------------------------------
# TF-IDF retrieval controller (same tokenize/idf/cosine pattern as
# src/memory/context_injector.py, adapted to (title, text) passage chunks)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class RetrievalIndex:
    """TF-IDF cosine retrieval over a single paper's section chunks."""

    def __init__(self, chunks: list[tuple[str, str]]):
        self.chunks = chunks  # [(title, text), ...]
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

    def query(self, query_text: str) -> tuple[str, str, float]:
        q_tokens = _tokenize(query_text)
        vocab = set(q_tokens)
        if not vocab:
            return self.chunks[0][0], self.chunks[0][1], 0.0
        q_vec = self._tfidf_vec(q_tokens, vocab)
        best_idx, best_score = 0, -1.0
        for i, toks in enumerate(self._doc_tokens):
            d_vec = self._tfidf_vec(toks, vocab)
            score = self._cosine(q_vec, d_vec)
            if score > best_score:
                best_idx, best_score = i, score
        title, text = self.chunks[best_idx]
        return title, text, best_score


# ---------------------------------------------------------------------------
# DFS / chain-following metric
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
    hits = 0
    for i in range(1, len(queries)):
        prev_tokens = _content_tokens(retrieved_chunks[i - 1])
        cur_tokens = _content_tokens(queries[i])
        if prev_tokens & cur_tokens:
            hits += 1
    return hits / (len(queries) - 1)


# ---------------------------------------------------------------------------
# LLM helper
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
            elapsed = round(time.time() - t0, 2)
            return content, tokens, elapsed
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

ANSWER_STYLE = (
    "Answer in as few words as possible (a name, number, or short phrase) — no explanation. "
    "If the passages do not contain the answer, reply exactly: Unanswerable."
)

NO_CONTEXT_SYSTEM = f"You are answering a question about an NLP research paper from your own knowledge. {ANSWER_STYLE}"

FULL_CONTEXT_SYSTEM = f"Answer the question using ONLY the paper text below. {ANSWER_STYLE}\n\n{{passages}}"

SUMMARIZE_SYSTEM = (
    "Summarize the key facts, methods, and results in the paper below in 250-400 words. "
    "Preserve specific numbers, dataset names, and method names exactly."
)

SUMMARY_QA_SYSTEM = f"Answer the question using ONLY the summary below. {ANSWER_STYLE}\n\nSUMMARY:\n{{summary}}"

QUERY_GEN_SYSTEM = (
    "You are researching a question about an NLP paper step by step, one search query at a time. "
    "Use what you learned from previous results to decide your next query if you still need more "
    "information. Do not repeat a query for information already retrieved. "
    "If you already have enough information to answer, reply with just DONE. "
    "Otherwise reply with ONLY a short search query (2-6 words), nothing else."
)

INTERACTIVE_ANSWER_SYSTEM = f"Answer the question using ONLY the retrieved information below. {ANSWER_STYLE}"


# ---------------------------------------------------------------------------
# Per-condition answer functions
# ---------------------------------------------------------------------------

def answer_no_context(question: str, model: str, url: str, timeout: int) -> tuple[str, int, float, dict]:
    ans, tok, el = _llm_call(
        [{"role": "system", "content": NO_CONTEXT_SYSTEM},
         {"role": "user", "content": f"Question: {question}\nAnswer:"}],
        model, url, timeout,
    )
    return ans, tok, el, {"given_text": ""}


def answer_full_context(question: str, chunks: list[tuple[str, str]], max_words: int, model: str, url: str, timeout: int) -> tuple[str, int, float, dict]:
    passages_text = truncate_to_budget("\n\n".join(text for _, text in chunks), max_words)
    system = FULL_CONTEXT_SYSTEM.format(passages=passages_text)
    ans, tok, el = _llm_call(
        [{"role": "system", "content": system},
         {"role": "user", "content": f"Question: {question}\nAnswer:"}],
        model, url, timeout,
    )
    return ans, tok, el, {"given_text": passages_text}


def answer_summary_doc(question: str, chunks: list[tuple[str, str]], model: str, url: str, timeout: int) -> tuple[str, int, float, dict]:
    passages_text = "\n\n".join(text for _, text in chunks)
    summary, sum_tok, sum_el = _llm_call(
        [{"role": "system", "content": SUMMARIZE_SYSTEM},
         {"role": "user", "content": passages_text}],
        model, url, timeout,
    )
    qa_system = SUMMARY_QA_SYSTEM.format(summary=summary)
    ans, qa_tok, qa_el = _llm_call(
        [{"role": "system", "content": qa_system},
         {"role": "user", "content": f"Question: {question}\nAnswer:"}],
        model, url, timeout,
    )
    return ans, sum_tok + qa_tok, sum_el + qa_el, {"given_text": summary, "summary_chars": len(summary)}


def answer_interactive_retrieval(
    question: str, chunks: list[tuple[str, str]], k: int, model: str, url: str, timeout: int,
) -> tuple[str, int, float, dict]:
    index = RetrievalIndex(chunks)
    queries: list[str] = []
    retrieved_titles: list[str] = []
    retrieved_chunks: list[str] = []
    total_tokens = 0
    t0 = time.time()

    for round_i in range(k):
        history_lines = []
        for i, (q, title, chunk_text) in enumerate(zip(queries, retrieved_titles, retrieved_chunks)):
            history_lines.append(f"QUERY {i+1}: {q}\nRESULT {i+1}: [{title}] {chunk_text}")
        history_text = "\n\n".join(history_lines)
        user_msg = f"QUESTION: {question}"
        if history_text:
            user_msg += f"\n\n{history_text}"
        user_msg += "\n\nWhat is your next search query?"

        query_text, q_tok, _ = _llm_call(
            [{"role": "system", "content": QUERY_GEN_SYSTEM}, {"role": "user", "content": user_msg}],
            model, url, timeout,
        )
        total_tokens += q_tok

        if query_text.strip().upper().startswith("DONE"):
            break

        title, chunk_text, _score = index.query(query_text)
        queries.append(query_text)
        retrieved_titles.append(title)
        retrieved_chunks.append(chunk_text)

    seen: set[str] = set()
    unique_history_lines = []
    for i, (q, t, c) in enumerate(zip(queries, retrieved_titles, retrieved_chunks)):
        if c not in seen:
            unique_history_lines.append(f"QUERY {i+1}: {q}\nRESULT {i+1}: [{t}] {c}")
            seen.add(c)

    final_user = f"QUESTION: {question}\n\n" + "\n\n".join(unique_history_lines) + "\n\nAnswer:"
    ans, a_tok, _ = _llm_call(
        [{"role": "system", "content": INTERACTIVE_ANSWER_SYSTEM},
         {"role": "user", "content": final_user}],
        model, url, timeout,
    )
    total_tokens += a_tok
    elapsed = round(time.time() - t0, 2)

    dfs_score = compute_dfs_score(queries, retrieved_chunks)
    unique_titles = list(dict.fromkeys(retrieved_titles))
    extra = {
        "queries": queries,
        "retrieved_titles": retrieved_titles,
        "given_titles": unique_titles,
        "given_text": "\n\n".join(dict.fromkeys(retrieved_chunks)),
        "num_queries": len(queries),
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
) -> list[dict]:
    label = label or condition
    model = cfg["model"]["name"]
    url = cfg["model"]["ollama_url"]
    timeout = cfg["model"]["timeout"]
    k = cfg["retrieval"]["max_queries"]
    max_words = cfg["experiment"]["full_context_max_words"]

    results = []
    for ex in examples:
        question = ex["question"]
        candidates = ex["answer_candidates"]
        gold_evidence = ex["gold_evidence"]
        chunks = format_passages_qasper(ex["full_text"])

        t0 = time.time()
        if no_llm:
            ans, tok, el, extra = f"[MOCK for {ex['id']}]", 0, 0.0, {"given_text": ""}
        else:
            try:
                if condition == "no_context":
                    ans, tok, el, extra = answer_no_context(question, model, url, timeout)
                elif condition == "full_context":
                    ans, tok, el, extra = answer_full_context(question, chunks, max_words, model, url, timeout)
                elif condition == "summary_doc":
                    ans, tok, el, extra = answer_summary_doc(question, chunks, model, url, timeout)
                elif condition == "interactive_retrieval":
                    ans, tok, el, extra = answer_interactive_retrieval(question, chunks, k, model, url, timeout)
                else:
                    raise ValueError(condition)
            except Exception as e:
                logger.warning(f"[{label}] {ex['id']} error: {e}")
                ans, tok, el, extra = "", 0, 0.0, {"given_text": ""}

        em = exact_match_score(ans, candidates)
        f1 = f1_score(ans, candidates)
        recall = evidence_containment_recall(extra.get("given_text", ""), gold_evidence)
        total_elapsed = round(time.time() - t0, 2)

        result = {
            "id": ex["id"],
            "type": qtype,
            "condition": label,
            "question": question,
            "gold_answer": candidates[0],
            "predicted_answer": ans,
            "em": em,
            "f1": f1,
            "tokens": tok,
            "elapsed_s": total_elapsed,
            "supporting_fact_recall": recall,
            "num_queries": extra.get("num_queries"),
            "dfs_score": extra.get("dfs_score"),
            "queries": extra.get("queries"),
            "paper_word_count": ex["paper_word_count"],
        }
        results.append(result)
        logger.info(
            f"[{label}/{qtype}] {ex['id']}: em={em:.0f} f1={f1:.2f} tok={tok} "
            f"recall={recall:.2f} s={total_elapsed:.1f}"
        )
    return results


# ---------------------------------------------------------------------------
# Model warmup
# ---------------------------------------------------------------------------

def warmup_model(model: str, url: str, timeout: int, logger: logging.Logger, max_attempts: int = 5) -> None:
    payload = {
        "model": model, "messages": [{"role": "user", "content": "Ready?"}], "stream": False,
        "options": {"temperature": 0.0, "num_predict": 1, "num_gpu": 0},
    }
    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.time()
            resp = requests.post(url, json=payload, timeout=timeout)
            elapsed = round(time.time() - t0, 1)
            if resp.status_code == 200:
                logger.info(f"[warmup] Model ready ({elapsed}s)")
                return
            logger.warning(f"[warmup] Attempt {attempt}/{max_attempts}: HTTP {resp.status_code}, retrying...")
        except Exception as e:
            logger.warning(f"[warmup] Attempt {attempt}/{max_attempts}: {e}, retrying...")
        time.sleep(5)
    logger.warning("[warmup] Model did not respond cleanly after all attempts — proceeding anyway.")


# ---------------------------------------------------------------------------
# Summary / saving
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def compute_summary(task_results: list[dict]) -> dict:
    if not task_results:
        return {}
    n = len(task_results)
    dfs_scores = [r["dfs_score"] for r in task_results if r["dfs_score"] is not None]
    return {
        "num_tasks": n,
        "exact_match": _mean([r["em"] for r in task_results]),
        "f1": _mean([r["f1"] for r in task_results]),
        "mean_tokens": _mean([r["tokens"] for r in task_results]),
        "mean_elapsed_s": _mean([r["elapsed_s"] for r in task_results]),
        "mean_supporting_fact_recall": _mean([r["supporting_fact_recall"] for r in task_results]),
        "mean_dfs_score": _mean(dfs_scores) if dfs_scores else None,
        "mean_num_queries": _mean([r["num_queries"] for r in task_results if r["num_queries"] is not None]) or None,
    }


def load_existing_condition_results(condition: str, results_dir: Path) -> dict[str, list[dict]]:
    path = results_dir / f"{condition}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("results_by_type", {})
    except Exception:
        return {}


def save_condition_results(condition: str, all_results_by_type: dict[str, list[dict]], results_dir: Path, cfg: dict) -> dict:
    summary_by_type = {qtype: compute_summary(res) for qtype, res in all_results_by_type.items()}
    overall = compute_summary([r for res in all_results_by_type.values() for r in res])
    out = {
        "condition": condition,
        "config": {
            "num_per_bucket": cfg["experiment"]["num_per_bucket"],
            "max_queries": cfg["retrieval"]["max_queries"],
        },
        "results_by_type": all_results_by_type,
        "summary_by_type": summary_by_type,
        "summary_overall": overall,
    }
    out_path = results_dir / f"{condition}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E8: QASPER Interactive Retrieval vs Static Context")
    parser.add_argument("--condition", default=None,
                        help=f"Comma-separated conditions to run (default: all). Valid: {', '.join(CONDITIONS)}")
    parser.add_argument("--num-tasks", type=int, default=None, help="Override num_per_bucket")
    parser.add_argument("--max-queries", type=int, default=None, help="Override retrieval.max_queries (K)")
    parser.add_argument("--no-llm", action="store_true", help="Bypass Ollama for structural testing")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any cached per-condition results and recompute everything from scratch")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = load_config()
    if args.num_tasks:
        cfg["experiment"]["num_per_bucket"] = args.num_tasks
    if args.max_queries is not None:
        cfg["retrieval"]["max_queries"] = args.max_queries

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                         format="%(asctime)s %(levelname)s %(message)s")
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

    logger.info("Loading QASPER dataset...")
    examples_by_type = load_qasper(cfg, logger)

    if not args.no_llm:
        warmup_model(cfg["model"]["name"], cfg["model"]["ollama_url"], cfg["model"]["timeout"], logger)

    all_condition_results = []
    for condition in selected_conditions:
        logger.info(f"\n{'─'*50}\nCondition: {condition}\n{'─'*50}")
        results_by_type = load_existing_condition_results(condition, results_dir) if not args.fresh else {}
        for qtype, examples in examples_by_type.items():
            cached = results_by_type.get(qtype)
            if cached and len(cached) == len(examples):
                logger.info(f"[{condition}/{qtype}] resuming — {len(cached)} cached results found, skipping recompute")
                continue
            results_by_type[qtype] = run_condition(condition, examples, qtype, cfg, logger, no_llm=args.no_llm)
            save_condition_results(condition, results_by_type, results_dir, cfg)
        out = save_condition_results(condition, results_by_type, results_dir, cfg)
        all_condition_results.append(out)

    summary_path = results_dir / "summary.json"
    summary_rows = []
    for r in all_condition_results:
        row = {"condition": r["condition"], "overall": r["summary_overall"]}
        row.update({f"by_type_{qtype}": s for qtype, s in r["summary_by_type"].items()})
        summary_rows.append(row)
    summary_path.write_text(json.dumps({"conditions": summary_rows}, indent=2), encoding="utf-8")
    logger.info(f"\nResults written to {results_dir}")

    print("\n=== E8 RESULTS SUMMARY (overall, all length buckets combined) ===")
    hdr = f"{'Condition':<24} {'EM':>7} {'F1':>7} {'Tokens':>9} {'Sec':>8} {'EvRecall':>10} {'DFS':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_condition_results:
        s = r["summary_overall"]
        dfs = s.get("mean_dfs_score")
        print(
            f"{r['condition']:<24} {s.get('exact_match', 0):>7.1%} {s.get('f1', 0):>7.1%} "
            f"{s.get('mean_tokens', 0):>9.0f} {s.get('mean_elapsed_s', 0):>8.1f} "
            f"{s.get('mean_supporting_fact_recall', 0):>10.1%} "
            f"{(f'{dfs:.1%}' if dfs is not None else '—'):>7}"
        )

    print("\n=== By length bucket ===")
    for qtype in cfg["experiment"]["length_buckets"]:
        print(f"\n-- {qtype} --")
        print(hdr)
        print("-" * len(hdr))
        for r in all_condition_results:
            s = r["summary_by_type"].get(qtype, {})
            dfs = s.get("mean_dfs_score")
            print(
                f"{r['condition']:<24} {s.get('exact_match', 0):>7.1%} {s.get('f1', 0):>7.1%} "
                f"{s.get('mean_tokens', 0):>9.0f} {s.get('mean_elapsed_s', 0):>8.1f} "
                f"{s.get('mean_supporting_fact_recall', 0):>10.1%} "
                f"{(f'{dfs:.1%}' if dfs is not None else '—'):>7}"
            )


if __name__ == "__main__":
    main()
