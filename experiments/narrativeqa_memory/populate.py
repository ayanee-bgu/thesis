"""
Phase 1 — Memory population.

Loads NarrativeQA train split, processes each document with the oracle
(same config as oracle_full from E10), then calls LLM extractors to produce
memory nodes in all five formats and saves them to the pool.

Population is deterministic: same train docs every run.  Results are
append-only so you can resume after interruption with --resume.

Usage:
  python experiments/narrativeqa_memory/populate.py [options]

  --no-llm          Structural test only (no Ollama calls)
  --resume          Skip docs already in the pool
  --log-level LEVEL
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.narrativeqa_memory.memory_store import (
    FORMATS, MemoryNode, MemoryPool, TFIDFMemoryIndex, _tok,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"


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
# Dataset loading  (grouped by document)
# ---------------------------------------------------------------------------

def load_train_docs(cfg: dict, logger: logging.Logger) -> dict[str, list[dict]]:
    """Return {kind: [doc_record, ...]} where each doc_record bundles the
    document and all of its selected questions."""
    load_dotenv(ROOT / ".env")
    from datasets import load_dataset

    pop = cfg["population"]
    ds = load_dataset(pop["dataset"], split=pop["split"])
    logger.info(f"[dataset] Loaded {len(ds)} train rows")

    # Group by (kind, doc_id), preserving insertion order
    seen: dict[str, dict] = {}  # doc_id → {doc, questions}
    kind_counts: dict[str, int] = {k: 0 for k in pop["kinds"]}

    for ex in ds:
        kind = ex["document"]["kind"]
        if kind not in kind_counts:
            continue
        doc_id = ex["document"]["id"]
        if doc_id not in seen:
            if kind_counts[kind] >= pop["num_docs_per_kind"]:
                continue
            seen[doc_id] = {"doc": ex["document"], "kind": kind, "questions": []}
            kind_counts[kind] += 1
        if len(seen[doc_id]["questions"]) < pop["max_questions_per_doc"]:
            seen[doc_id]["questions"].append({
                "question": ex["question"]["text"],
                "gold_answers": [a["text"] for a in ex["answers"] if a.get("text")],
            })
        if all(v >= pop["num_docs_per_kind"] for v in kind_counts.values()):
            break

    by_kind: dict[str, list[dict]] = {k: [] for k in pop["kinds"]}
    for doc_id, rec in seen.items():
        by_kind[rec["kind"]].append({"doc_id": doc_id, **rec})

    for kind, docs in by_kind.items():
        logger.info(f"[dataset] {kind}: {len(docs)} docs selected")
    return by_kind


# ---------------------------------------------------------------------------
# Oracle runner  (TF-IDF + grounded + ±1 neighbor — mirrors oracle_full)
# ---------------------------------------------------------------------------

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
# LLM helpers
# ---------------------------------------------------------------------------

_CONNECT_TIMEOUT = 15
_MAX_RETRIES = 3
_RETRY_BACKOFF = 15


def _llm(messages: list[dict], model: str, url: str, timeout: int) -> str:
    payload = {
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512, "num_gpu": 0},
    }
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            r = requests.post(url, json=payload, timeout=(_CONNECT_TIMEOUT, timeout))
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                logging.getLogger(__name__).warning(f"LLM attempt {attempt}: {exc}, retrying...")
                time.sleep(_RETRY_BACKOFF)
    raise last_exc


def _json_from_response(text: str) -> dict | list | None:
    """Extract the first JSON object or array from a response string."""
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Oracle session runner
# ---------------------------------------------------------------------------

def make_grounded_query_gen_system(doc_title: str, doc_kind: str) -> str:
    label = {"gutenberg": "novel", "movie": "movie script"}.get(doc_kind, doc_kind)
    return (
        f"You are researching a question about '{doc_title}' ({label}) step by step, "
        "one search query at a time. Use previous results to guide next queries. "
        "Do not repeat a query you already made. "
        "Reply DONE if you have enough to answer, otherwise reply with only a short search query (2-6 words)."
    )

INTERACTIVE_ANSWER_SYSTEM = (
    "Answer the question using ONLY the retrieved information below. "
    "Answer in as few words as possible — a name, phrase, or short sentence. No explanation."
)


def run_oracle_session(
    question: str,
    chunks: list[tuple[str, str]],
    k: int,
    doc_title: str,
    doc_kind: str,
    model: str,
    url: str,
    timeout: int,
    n_neighbors: int = 1,
) -> dict:
    """Run one oracle_full session and return the trajectory dict."""
    index = TFIDFDocIndex(chunks)
    query_gen_sys = make_grounded_query_gen_system(doc_title, doc_kind)
    queries, primary_titles, primary_chunks, expansion_chunks = [], [], [], []
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

        query_text = _llm(
            [{"role": "system", "content": query_gen_sys}, {"role": "user", "content": user_msg}],
            model, url, timeout,
        )
        if query_text.strip().upper().startswith("DONE"):
            early_stop = True
            break

        retrieved = index.query(query_text, n_neighbors=n_neighbors)
        p_title, p_text = retrieved[0]
        queries.append(query_text)
        primary_titles.append(p_title)
        primary_chunks.append(p_text)
        for nb_title, nb_text in retrieved[1:]:
            expansion_chunks.append(nb_text)

    seen: set[str] = set()
    unique_pairs = []
    for q, t, c in zip(queries, primary_titles, primary_chunks):
        if c not in seen:
            unique_pairs.append((q, t, c))
            seen.add(c)
    for c in expansion_chunks:
        seen.add(c)

    final_user = (
        f"QUESTION: {question}\n\n"
        + "\n\n".join(f"QUERY {i+1}: {q}\nRESULT {i+1}: [{t}] {c}" for i, (q, t, c) in enumerate(unique_pairs))
        + "\n\nAnswer:"
    )
    answer = _llm(
        [{"role": "system", "content": INTERACTIVE_ANSWER_SYSTEM},
         {"role": "user", "content": final_user}],
        model, url, timeout,
    )

    return {
        "question": question,
        "predicted_answer": answer,
        "queries": queries,
        "primary_chunks": primary_chunks,
        "expansion_chunks": expansion_chunks,
        "early_stop": early_stop,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    return " ".join(s.lower().split())


def f1_score(pred: str, golds: list[str]) -> float:
    def _f1(pt, gt):
        if not pt or not gt:
            return 1.0 if pt == gt else 0.0
        common = {t: min(pt.count(t), gt.count(t)) for t in pt}
        ns = sum(common.values())
        if ns == 0:
            return 0.0
        return 2 * ns / (len(pt) + len(gt))
    pt = normalize_answer(pred).split()
    return max(_f1(pt, normalize_answer(g).split()) for g in golds)


# ---------------------------------------------------------------------------
# Extraction prompts  (F2 – F5)
# ---------------------------------------------------------------------------

def extract_F2_facts(
    doc_title: str, doc_kind: str, question: str, answer: str, chunk: str,
    model: str, url: str, timeout: int,
) -> list[str]:
    """Return a list of factual claim strings."""
    prompt = (
        f"Story: {doc_title} ({doc_kind})\n"
        f"Question: {question}\n"
        f"Correct answer: {answer}\n"
        f"Key text retrieved: {chunk[:400]}\n\n"
        "Write 1-3 factual claims from this story that would help answer similar questions. "
        "Each claim on its own line starting with 'FACT: '. "
        "Be specific: include character names, relationships, and events."
    )
    response = _llm(
        [{"role": "system", "content": "You are a precise fact extractor. Output only FACT: lines."},
         {"role": "user", "content": prompt}],
        model, url, timeout,
    )
    facts = []
    for line in response.splitlines():
        line = line.strip()
        if line.upper().startswith("FACT:"):
            claim = line[5:].strip()
            if len(claim) > 10:
                facts.append(claim)
    return facts


def extract_F3_procedure(
    doc_kind: str, question_type: str, examples: list[dict],
    model: str, url: str, timeout: int,
) -> dict | None:
    """Return a procedure dict or None if extraction fails."""
    ex_lines = "\n".join(
        f"- Q: {e['question']}\n  Searches: {' → '.join(e['queries'])}\n  Correct: {e['correct']}"
        for e in examples[:5]
    )
    prompt = (
        f"Document type: {doc_kind}\n"
        f"Question category: {question_type}\n"
        f"Examples:\n{ex_lines}\n\n"
        "What search strategy works for this type of question? "
        "Write a JSON object with fields:\n"
        '  "description": "2-sentence strategy",\n'
        '  "failure_modes": "1 sentence on what to avoid"\n'
        "JSON only, no explanation."
    )
    response = _llm(
        [{"role": "system", "content": "You are a search strategy analyst. Output only JSON."},
         {"role": "user", "content": prompt}],
        model, url, timeout,
    )
    parsed = _json_from_response(response)
    if not isinstance(parsed, dict):
        return None
    return {
        "description": str(parsed.get("description", "")),
        "failure_modes": str(parsed.get("failure_modes", "")),
    }


def extract_F4_program(
    doc_kind: str, procedure_description: str, question_examples: list[str],
    model: str, url: str, timeout: int,
) -> dict | None:
    """Return a query program dict or None if extraction fails."""
    examples = "\n".join(f"- {q}" for q in question_examples[:4])
    prompt = (
        f"Procedure: {procedure_description}\n"
        f"Example questions:\n{examples}\n\n"
        "Create a parameterized query program. "
        "Use {{slot_name}} for variable parts. "
        "Write a JSON object with fields:\n"
        '  "name": "short_snake_case_name",\n'
        '  "slots": ["slot1", "slot2"],\n'
        '  "query_sequence": ["{slot1} {slot2}", "alternative query"]\n'
        "JSON only."
    )
    response = _llm(
        [{"role": "system", "content": "You are a query program designer. Output only JSON."},
         {"role": "user", "content": prompt}],
        model, url, timeout,
    )
    parsed = _json_from_response(response)
    if not isinstance(parsed, dict):
        return None
    return {
        "name": str(parsed.get("name", "unnamed_program")),
        "slots": list(parsed.get("slots", [])),
        "query_sequence": list(parsed.get("query_sequence", [])),
    }


def extract_F5_episode(
    doc_title: str, doc_kind: str, doc_word_count: int, qa_pairs: list[dict],
    model: str, url: str, timeout: int,
) -> dict | None:
    """Return an episode dict or None if extraction fails."""
    qa_lines = "\n".join(
        f"- Q: {p['question']}  A: {p['answer']} ({'✓' if p['correct'] else '✗'})"
        for p in qa_pairs[:8]
    )
    prompt = (
        f"Story: {doc_title} ({doc_kind}, ~{doc_word_count} words)\n"
        f"Q&A sessions:\n{qa_lines}\n\n"
        "Extract:\n"
        "1. A 3-sentence plot summary (characters, key events, relationships)\n"
        "2. Key characters: name → role description\n\n"
        "Write a JSON object:\n"
        '  "summary": "...",\n'
        '  "characters": {"Name": "role"}\n'
        "JSON only."
    )
    response = _llm(
        [{"role": "system", "content": "You are a story summarizer. Output only JSON."},
         {"role": "user", "content": prompt}],
        model, url, timeout,
    )
    parsed = _json_from_response(response)
    if not isinstance(parsed, dict):
        return None
    return {
        "summary": str(parsed.get("summary", "")),
        "characters": dict(parsed.get("characters", {})),
    }


# ---------------------------------------------------------------------------
# Question-type classifier  (lightweight — no LLM call)
# ---------------------------------------------------------------------------

_QTYPE_PATTERNS = [
    (r"\bwho\b", "who_X"),
    (r"\bwhat.*(?:happen|occur|take place)\b", "what_happened"),
    (r"\bwhy\b", "why_X"),
    (r"\bwhere\b", "where_X"),
    (r"\bhow\b", "how_X"),
    (r"\bwhen\b", "when_X"),
    (r"\bwhich\b", "which_X"),
    (r"\bwhat\b", "what_X"),
]


def classify_question_type(question: str) -> str:
    q = question.lower()
    for pattern, label in _QTYPE_PATTERNS:
        if re.search(pattern, q):
            return label
    return "other"


# ---------------------------------------------------------------------------
# Procedure deduplication / aggregation
# ---------------------------------------------------------------------------

def _procedure_sim(a: str, b: str) -> float:
    """TF-IDF cosine between two procedure descriptions (no corpus IDF — use raw TF)."""
    a_toks = _tok(a)
    b_toks = _tok(b)
    vocab = set(a_toks) | set(b_toks)
    if not vocab:
        return 0.0
    def tf_vec(toks):
        n = len(toks) or 1
        return {t: toks.count(t) / n for t in vocab}
    av, bv = tf_vec(a_toks), tf_vec(b_toks)
    dot = sum(av[t] * bv[t] for t in vocab)
    na = math.sqrt(sum(v * v for v in av.values())) or 1
    nb = math.sqrt(sum(v * v for v in bv.values())) or 1
    return dot / (na * nb)


class ProcedureAggregator:
    """Accumulates procedure candidates across sessions; deduplicates by
    TF-IDF similarity and reinforces existing procedures when new evidence arrives."""

    def __init__(self, dedup_threshold: float, min_sessions: int):
        self.dedup_threshold = dedup_threshold
        self.min_sessions = min_sessions
        self._procedures: list[dict] = []  # {description, failure_modes, question_type, doc_kind, sessions, successes, questions}

    def add(self, description: str, failure_modes: str, question_type: str,
            doc_kind: str, session_id: str, success: bool, questions: list[str]) -> None:
        if not description:
            return
        for p in self._procedures:
            if (p["question_type"] == question_type and p["doc_kind"] == doc_kind
                    and _procedure_sim(description, p["description"]) >= self.dedup_threshold):
                p["sessions"].append(session_id)
                p["successes"] += int(success)
                p["questions"].extend(questions)
                return
        self._procedures.append({
            "description": description,
            "failure_modes": failure_modes,
            "question_type": question_type,
            "doc_kind": doc_kind,
            "sessions": [session_id],
            "successes": int(success),
            "questions": questions,
        })

    def mature_procedures(self) -> list[dict]:
        return [p for p in self._procedures if len(p["sessions"]) >= self.min_sessions]


# ---------------------------------------------------------------------------
# Main population loop
# ---------------------------------------------------------------------------

def populate(cfg: dict, logger: logging.Logger, no_llm: bool = False, resume: bool = False) -> None:
    pop = cfg["population"]
    model = cfg["model"]["name"]
    url = cfg["model"]["ollama_url"]
    timeout = cfg["model"]["timeout"]
    k = pop["oracle"]["max_queries"]
    n_neighbors = pop["oracle"]["n_neighbors"]
    chunk_words = pop["chunk_words"]
    overlap_words = pop["chunk_overlap_words"]
    min_fact_f1 = pop["extraction"]["min_fact_f1"]
    min_program_sr = pop["extraction"]["min_program_success_rate"]

    pool = MemoryPool(Path(ROOT / cfg["output"]["memory_dir"]))
    proc_agg = ProcedureAggregator(
        dedup_threshold=pop["extraction"]["procedure_dedup_threshold"],
        min_sessions=pop["extraction"]["min_procedure_sessions"],
    )

    # Resume: track already-processed doc_ids
    processed_doc_ids: set[str] = set()
    if resume:
        pool.load()
        for n in pool._nodes:
            for sid in n.source_sessions:
                processed_doc_ids.add(sid.split(":")[0])
        pool._nodes.clear()
        logger.info(f"[resume] Skipping {len(processed_doc_ids)} already-processed docs")

    docs_by_kind = load_train_docs(cfg, logger)

    for kind, docs in docs_by_kind.items():
        for doc_rec in docs:
            doc_id = doc_rec["doc_id"]
            if doc_id in processed_doc_ids:
                logger.info(f"[populate] skip {doc_id} (already done)")
                continue

            doc = doc_rec["doc"]
            doc_title = doc.get("summary", {}).get("title") or doc_id
            doc_text = doc["text"]
            doc_word_count = doc.get("word_count") or len(doc_text.split())
            questions = doc_rec["questions"]

            logger.info(f"\n[populate] {doc_title} ({kind}, {doc_word_count} words, {len(questions)} Qs)")

            if no_llm:
                # Structural test: emit one mock node per format and move on
                session_id = f"{doc_id}:mock"
                for fmt in FORMATS:
                    pool.add(MemoryNode.new(
                        fmt, kind, f"[mock {fmt}]",
                        {"question": "mock?", "answer": "mock", "doc_title": doc_title},
                        session_id,
                    ))
                pool.flush()
                continue

            chunks = chunk_document(doc_text, chunk_words, overlap_words)
            doc_trajectories = []

            for q_rec in questions:
                question = q_rec["question"]
                gold = q_rec["gold_answers"] or [""]
                session_id = f"{doc_id}:{hash(question) & 0xFFFF:04x}"

                try:
                    traj = run_oracle_session(
                        question, chunks, k, doc_title, kind,
                        model, url, timeout, n_neighbors,
                    )
                except Exception as e:
                    logger.warning(f"[oracle] {doc_id} '{question[:40]}' failed: {e}")
                    continue

                score = f1_score(traj["predicted_answer"], gold)
                traj["gold_answers"] = gold
                traj["f1"] = score
                traj["correct"] = score >= 0.5
                doc_trajectories.append({"session_id": session_id, **traj})

                # F1 — raw transcript (always)
                queries_text = " | ".join(traj["queries"])
                content_f1 = f"{question} {traj['predicted_answer']} {queries_text}"
                pool.add(MemoryNode.new(
                    "F1_raw", kind, content_f1,
                    {
                        "question": question,
                        "answer": traj["predicted_answer"],
                        "queries_issued": traj["queries"],
                        "correct": traj["correct"],
                        "doc_title": doc_title,
                    },
                    session_id,
                ))

                # F2 — facts (only from correct answers)
                if traj["correct"] and traj["primary_chunks"]:
                    best_chunk = traj["primary_chunks"][0]
                    try:
                        facts = extract_F2_facts(
                            doc_title, kind, question, traj["predicted_answer"],
                            best_chunk, model, url, timeout,
                        )
                    except Exception as e:
                        logger.warning(f"[F2] extraction failed: {e}")
                        facts = []
                    for claim in facts:
                        pool.add(MemoryNode.new(
                            "F2_fact", kind, claim,
                            {"claim": claim, "doc_title": doc_title,
                             "source_question": question, "confidence": float(score)},
                            session_id,
                        ))

                logger.info(
                    f"  Q: {question[:60]}\n"
                    f"  A: {traj['predicted_answer']} | F1={score:.2f} | queries={len(traj['queries'])}"
                )

            if not doc_trajectories:
                continue

            # F3 — procedures (per question type, accumulated across all questions)
            for traj in doc_trajectories:
                qtype = classify_question_type(traj["question"])
                proc_result = None
                if traj["queries"]:
                    try:
                        proc_result = extract_F3_procedure(
                            kind, qtype,
                            [{"question": traj["question"], "queries": traj["queries"], "correct": traj["correct"]}],
                            model, url, timeout,
                        )
                    except Exception as e:
                        logger.warning(f"[F3] extraction failed: {e}")
                if proc_result:
                    proc_agg.add(
                        description=proc_result["description"],
                        failure_modes=proc_result["failure_modes"],
                        question_type=qtype,
                        doc_kind=kind,
                        session_id=doc_id,
                        success=traj["correct"],
                        questions=[traj["question"]],
                    )

            # F5 — episode (one per doc, from all Q&A pairs)
            qa_pairs = [
                {"question": t["question"], "answer": t["predicted_answer"], "correct": t["correct"]}
                for t in doc_trajectories
            ]
            try:
                episode = extract_F5_episode(
                    doc_title, kind, doc_word_count, qa_pairs, model, url, timeout,
                )
            except Exception as e:
                logger.warning(f"[F5] extraction failed: {e}")
                episode = None
            if episode and episode.get("summary"):
                pool.add(MemoryNode.new(
                    "F5_episode", kind, episode["summary"],
                    {"doc_title": doc_title, "summary": episode["summary"],
                     "characters": episode.get("characters", {})},
                    doc_id,
                ))

            pool.flush()
            logger.info(f"[populate] flushed pool after {doc_title}")

    # After all docs: materialize F3 procedures and F4 programs
    logger.info("\n[populate] Materializing F3 procedures and F4 programs...")
    for proc in proc_agg.mature_procedures():
        success_rate = proc["successes"] / max(len(proc["sessions"]), 1)
        content_f3 = f"{proc['question_type']}: {proc['description']}"
        pool.add(MemoryNode.new(
            "F3_procedure", proc["doc_kind"], content_f3,
            {
                "question_type": proc["question_type"],
                "description": proc["description"],
                "failure_modes": proc["failure_modes"],
                "success_rate": success_rate,
                "num_sessions": len(proc["sessions"]),
            },
            proc["sessions"][0],
        ))

        # F4 — promote to program if success rate is high enough
        if success_rate >= pop["extraction"]["min_program_success_rate"] and not no_llm:
            try:
                program = extract_F4_program(
                    proc["doc_kind"], proc["description"],
                    proc["questions"][:4], model, url, timeout,
                )
            except Exception as e:
                logger.warning(f"[F4] extraction failed: {e}")
                program = None
            if program and program.get("query_sequence"):
                content_f4 = f"{program['name']}: {proc['description']} " + " ".join(program["query_sequence"])
                pool.add(MemoryNode.new(
                    "F4_program", proc["doc_kind"], content_f4,
                    {
                        "name": program["name"],
                        "slots": program["slots"],
                        "query_sequence": program["query_sequence"],
                        "description": proc["description"],
                        "success_rate": success_rate,
                    },
                    proc["sessions"][0],
                ))

    pool.flush()

    # Print summary
    pool.load()
    stats = pool.stats()
    logger.info(f"\n[populate] Done. Pool stats: {json.dumps(stats, indent=2)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E11 Phase 1: populate memory pool")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)
    cfg = load_config()
    populate(cfg, logger, no_llm=args.no_llm, resume=args.resume)


if __name__ == "__main__":
    main()
