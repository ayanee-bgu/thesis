"""
Phase 1 — Memory population from MuSiQue train split.

For each train question:
  1. Run K-round TF-IDF oracle over the 20 candidate passages
  2. Store F1 (raw session log) unconditionally
  3. Store F2 (gold decomposition chain) directly from dataset — no LLM call
  4. Classify question type from decomposition pattern
  5. Accumulate procedure evidence for F3/F4 extraction

After all questions:
  6. Extract F3 procedures from aggregated clusters (one LLM call per cluster)
  7. Promote to F4 programs where success rate is sufficient
  8. Extract F5 canonical decomposition patterns per question type

Usage:
  python experiments/musique_memory/populate.py [options]

  --no-llm    Structural test (skip Ollama, emit mock nodes)
  --resume    Skip question IDs already in the pool
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

from experiments.musique_memory.memory_store import (
    FORMATS, MemoryNode, MemoryPool, TFIDFMemoryIndex, _tok,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
_HOP_RE = re.compile(r"^(\d+)hop")


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

def load_train_examples(cfg: dict, logger: logging.Logger) -> dict[str, list[dict]]:
    load_dotenv(ROOT / ".env")
    from datasets import load_dataset

    pop = cfg["population"]
    ds = load_dataset(pop["dataset"], split=pop["split"])
    logger.info(f"[dataset] Loaded {len(ds)} train rows")

    buckets = {f"hop{h}": [] for h in pop["hop_buckets"]}
    for ex in ds:
        m = _HOP_RE.match(ex["id"])
        if not m:
            continue
        label = f"hop{m.group(1)}"
        if label in buckets and len(buckets[label]) < pop["num_per_hop"]:
            buckets[label].append(ex)
        if all(len(v) >= pop["num_per_hop"] for v in buckets.values()):
            break

    for label, exs in buckets.items():
        logger.info(f"[dataset] {label}: {len(exs)} train examples")
    return buckets


def format_passages(paragraphs: list[dict]) -> list[tuple[str, str]]:
    return [(p["title"], f"{p['title']}: {p['paragraph_text']}") for p in paragraphs]


def gold_titles(example: dict) -> set[str]:
    return {p["title"] for p in example["paragraphs"] if p["is_supporting"]}


# ---------------------------------------------------------------------------
# TF-IDF passage index
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
        t, text = self.passages[best_idx]
        return t, text, best_score


# ---------------------------------------------------------------------------
# LLM helper
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


def _json_from(text: str) -> dict | list | None:
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

QUERY_GEN_SYSTEM = (
    "You are answering a multi-hop question step by step using search queries. "
    "Each search retrieves one passage. Use previous results to guide next queries. "
    "Reply DONE if you have enough to answer. Otherwise reply with only a short search query (3-7 words)."
)

ANSWER_SYSTEM = (
    "Answer the question using ONLY the retrieved passages below. "
    "Answer in as few words as possible — a name, date, or short phrase. No explanation."
)


def run_oracle_session(
    question: str,
    passages: list[tuple[str, str]],
    k: int,
    model: str,
    url: str,
    timeout: int,
) -> dict:
    index = PassageIndex(passages)
    queries, retrieved_titles, retrieved_texts = [], [], []

    for _ in range(k):
        hist = "\n\n".join(
            f"QUERY {i+1}: {q}\nRESULT {i+1}: [{t}] {c}"
            for i, (q, t, c) in enumerate(zip(queries, retrieved_titles, retrieved_texts))
        )
        user = f"QUESTION: {question}"
        if hist:
            user += f"\n\n{hist}"
        user += "\n\nNext search query?"

        qt = _llm(
            [{"role": "system", "content": QUERY_GEN_SYSTEM},
             {"role": "user", "content": user}],
            model, url, timeout,
        )
        if qt.strip().upper().startswith("DONE"):
            break

        title, text, _ = index.query(qt)
        queries.append(qt)
        retrieved_titles.append(title)
        retrieved_texts.append(text)

    seen: set[str] = set()
    unique_pairs = [(q, t, c) for q, t, c in zip(queries, retrieved_titles, retrieved_texts) if c not in seen and not seen.add(c)]

    final_user = (
        f"QUESTION: {question}\n\n"
        + "\n\n".join(f"QUERY {i+1}: {q}\nRESULT {i+1}: [{t}] {c}" for i, (q, t, c) in enumerate(unique_pairs))
        + "\n\nAnswer:"
    )
    answer = _llm(
        [{"role": "system", "content": ANSWER_SYSTEM},
         {"role": "user", "content": final_user}],
        model, url, timeout,
    )
    return {
        "queries": queries,
        "retrieved_titles": retrieved_titles,
        "retrieved_texts": retrieved_texts,
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    return " ".join(s.lower().split())


def f1_score(pred: str, gold: str) -> float:
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return 1.0 if pt == gt else 0.0
    ns = sum({t: min(pt.count(t), gt.count(t)) for t in pt}.values())
    return 0.0 if ns == 0 else 2 * ns / (len(pt) + len(gt))


def supporting_recall(retrieved_titles: list[str], gold_supporting: set[str]) -> float:
    if not gold_supporting:
        return 0.0
    found = {t for t in retrieved_titles if t in gold_supporting}
    return len(found) / len(gold_supporting)


# ---------------------------------------------------------------------------
# Question type classifier
# ---------------------------------------------------------------------------

_QWORD_RE = re.compile(r"^(who|what|where|when|which|how)", re.IGNORECASE)


def classify_question_type(question: str, decomposition: list[dict], hop: int) -> str:
    """
    Classify by the sequence of wh-words in the sub-questions.
    e.g. hop2_what_who = "what film starred X? who directed [film]?"
    This creates transferable types: all "hop2_what_who" questions share the
    same bridge structure regardless of which entities are involved.
    """
    sub_words = []
    for sub in decomposition[:hop]:
        sq = sub.get("question", "")
        m = _QWORD_RE.match(sq.strip())
        sub_words.append(m.group(1).lower() if m else "what")
    return f"hop{hop}_{'_'.join(sub_words)}"


# ---------------------------------------------------------------------------
# F2: gold decomposition chain  (no LLM — derived directly from dataset)
# ---------------------------------------------------------------------------

def make_f2_chain_content(question: str, decomposition: list[dict]) -> str:
    """
    Convert gold sub-questions into a chain string used as the indexable content.
    The chain is abstract enough to describe the reasoning procedure without
    committing to the specific entities of this particular question.
    """
    steps = []
    for i, sub in enumerate(decomposition):
        sq = sub.get("question", "")
        sa = sub.get("answer", "")
        steps.append(f"Step {i+1}: {sq} → [{sa}]")
    return " | ".join(steps)


# ---------------------------------------------------------------------------
# F3 procedure extractor  (one LLM call per question-type cluster)
# ---------------------------------------------------------------------------

def extract_f3_procedure(
    hop: int, question_type: str, examples: list[dict],
    model: str, url: str, timeout: int,
) -> dict | None:
    ex_lines = "\n".join(
        f"- Q: {e['question']}\n"
        f"  Queries issued: {' → '.join(e['queries'])}\n"
        f"  Supporting passages found: {e['support_found']}/{e['support_total']}\n"
        f"  Correct: {e['correct']}"
        for e in examples[:6]
    )
    prompt = (
        f"Hop count: {hop}\n"
        f"Question pattern: {question_type}\n\n"
        f"Examples:\n{ex_lines}\n\n"
        "What search strategy works for this type of multi-hop question? "
        "Write a JSON object:\n"
        '{\n'
        '  "description": "2-sentence strategy for how to decompose and search",\n'
        '  "failure_modes": "1 sentence on what to avoid"\n'
        '}\n'
        "JSON only."
    )
    resp = _llm(
        [{"role": "system", "content": "You are a multi-hop reasoning strategist. Output only JSON."},
         {"role": "user", "content": prompt}],
        model, url, timeout,
    )
    parsed = _json_from(resp)
    if not isinstance(parsed, dict):
        return None
    return {
        "description": str(parsed.get("description", "")),
        "failure_modes": str(parsed.get("failure_modes", "")),
    }


# ---------------------------------------------------------------------------
# F4 program extractor
# ---------------------------------------------------------------------------

def extract_f4_program(
    hop: int, procedure_description: str, question_examples: list[str],
    model: str, url: str, timeout: int,
) -> dict | None:
    examples = "\n".join(f"- {q}" for q in question_examples[:4])
    prompt = (
        f"Hop count: {hop}\n"
        f"Procedure: {procedure_description}\n\n"
        f"Example questions:\n{examples}\n\n"
        "Create a parameterized search program using {slot_name} for variable parts. "
        "Write JSON:\n"
        '{\n'
        '  "name": "short_snake_case_name",\n'
        '  "slots": ["slot1", "slot2"],\n'
        '  "query_sequence": ["{slot1}", "{slot1} {slot2}", "..."],\n'
        '  "bridge_slot": "slot that is the intermediate entity (bridge)"\n'
        '}\n'
        "JSON only."
    )
    resp = _llm(
        [{"role": "system", "content": "You are a query program designer. Output only JSON."},
         {"role": "user", "content": prompt}],
        model, url, timeout,
    )
    parsed = _json_from(resp)
    if not isinstance(parsed, dict):
        return None
    return {
        "name": str(parsed.get("name", "unnamed")),
        "slots": list(parsed.get("slots", [])),
        "query_sequence": list(parsed.get("query_sequence", [])),
        "bridge_slot": str(parsed.get("bridge_slot", "")),
    }


# ---------------------------------------------------------------------------
# F6 preference extractor  (what approach style worked in successful sessions)
# ---------------------------------------------------------------------------

def extract_f6_preference(
    hop: int, question_type: str, examples: list[dict],
    model: str, url: str, timeout: int,
) -> dict | None:
    successes = [e for e in examples if e["correct"]] or examples[:5]
    ex_lines = "\n".join(
        f"- Q: {e['question']}\n"
        f"  Queries issued: {' → '.join(e['queries'])}\n"
        f"  Correct: {e['correct']}"
        for e in successes[:5]
    )
    prompt = (
        f"Hop count: {hop}, Question type: {question_type}\n\n"
        f"Successful or best sessions:\n{ex_lines}\n\n"
        "What specific query style or ordering preference made these sessions work? "
        "Focus on: which entity to anchor first, query specificity, relationship framing. "
        "Write JSON:\n"
        '{\n'
        '  "approach": "1-2 sentences on the preferred search style",\n'
        '  "avoid": "1 sentence on what to avoid"\n'
        '}\n'
        "JSON only."
    )
    resp = _llm(
        [{"role": "system", "content": "You are a multi-hop search strategist. Output only JSON."},
         {"role": "user", "content": prompt}],
        model, url, timeout,
    )
    parsed = _json_from(resp)
    if not isinstance(parsed, dict):
        return None
    return {
        "approach": str(parsed.get("approach", "")),
        "avoid": str(parsed.get("avoid", "")),
    }


# ---------------------------------------------------------------------------
# F7 correction extractor  (common first-query mistakes and how to fix them)
# ---------------------------------------------------------------------------

def extract_f7_correction(
    hop: int, question_type: str, examples: list[dict],
    model: str, url: str, timeout: int,
) -> dict | None:
    # Only use sessions that had to pivot (more than 1 query)
    pivots = [e for e in examples if len(e["queries"]) > 1]
    if not pivots:
        return None
    ex_lines = "\n".join(
        f"- Q: {e['question']}\n"
        f"  First query (likely wrong): {e['queries'][0]}\n"
        f"  Corrected to: {' → '.join(e['queries'][1:])}\n"
        f"  Correct answer found: {e['correct']}"
        for e in pivots[:5]
    )
    prompt = (
        f"Hop count: {hop}, Question type: {question_type}\n\n"
        f"Sessions that pivoted their search approach:\n{ex_lines}\n\n"
        "What is the common mistake in the first query, and what correction works? "
        "Write JSON:\n"
        '{\n'
        '  "wrong_pattern": "1 sentence describing the typical first-query mistake",\n'
        '  "correction": "1-2 sentences on how to fix it and what to do instead"\n'
        '}\n'
        "JSON only."
    )
    resp = _llm(
        [{"role": "system", "content": "You are a multi-hop search error analyst. Output only JSON."},
         {"role": "user", "content": prompt}],
        model, url, timeout,
    )
    parsed = _json_from(resp)
    if not isinstance(parsed, dict):
        return None
    return {
        "wrong_pattern": str(parsed.get("wrong_pattern", "")),
        "correction": str(parsed.get("correction", "")),
    }


# ---------------------------------------------------------------------------
# F5 decomposition pattern extractor
# ---------------------------------------------------------------------------

def extract_f5_decomp(
    hop: int, question_type: str, example_questions: list[str],
    example_chains: list[str], model: str, url: str, timeout: int,
) -> str | None:
    q_lines = "\n".join(f"  - {q}" for q in example_questions[:4])
    c_lines = "\n".join(f"  - {c}" for c in example_chains[:3])
    prompt = (
        f"Hop count: {hop}, Question type: {question_type}\n\n"
        f"Example questions:\n{q_lines}\n\n"
        f"Example decompositions:\n{c_lines}\n\n"
        "Write ONE sentence describing the canonical abstract decomposition pattern "
        "for this question type, using placeholders like [Entity], [Attribute], [Bridge]. "
        "Do NOT mention specific names or facts. Pattern only."
    )
    resp = _llm(
        [{"role": "system", "content": "You are a reasoning pattern analyst. Output one sentence only."},
         {"role": "user", "content": prompt}],
        model, url, timeout,
    )
    resp = resp.strip().strip('"').strip("'")
    return resp if len(resp) > 10 else None


# ---------------------------------------------------------------------------
# Procedure aggregator
# ---------------------------------------------------------------------------

def _sim(a: str, b: str) -> float:
    at, bt = _tok(a), _tok(b)
    vocab = set(at) | set(bt)
    if not vocab:
        return 0.0
    def tf(toks):
        n = len(toks) or 1
        return {t: toks.count(t) / n for t in vocab}
    av, bv = tf(at), tf(bt)
    dot = sum(av[t] * bv[t] for t in vocab)
    na = math.sqrt(sum(v * v for v in av.values())) or 1
    nb = math.sqrt(sum(v * v for v in bv.values())) or 1
    return dot / (na * nb)


class ProcedureAggregator:
    def __init__(self, dedup_threshold: float, min_sessions: int):
        self.threshold = dedup_threshold
        self.min_sessions = min_sessions
        self._clusters: list[dict] = []

    def add(self, question_type: str, hop: int, question: str,
            queries: list[str], correct: bool,
            support_found: int, support_total: int, session_id: str) -> None:
        key = (question_type, hop)
        for c in self._clusters:
            if (c["question_type"], c["hop"]) == key:
                c["sessions"].append(session_id)
                c["successes"] += int(correct)
                c["examples"].append({
                    "question": question, "queries": queries,
                    "correct": correct,
                    "support_found": support_found, "support_total": support_total,
                })
                return
        self._clusters.append({
            "question_type": question_type,
            "hop": hop,
            "sessions": [session_id],
            "successes": int(correct),
            "examples": [{
                "question": question, "queries": queries,
                "correct": correct,
                "support_found": support_found, "support_total": support_total,
            }],
        })

    def mature_clusters(self) -> list[dict]:
        return [c for c in self._clusters if len(c["sessions"]) >= self.min_sessions]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def populate(cfg: dict, logger: logging.Logger, no_llm: bool = False, resume: bool = False) -> None:
    pop = cfg["population"]
    model = cfg["model"]["name"]
    url = cfg["model"]["ollama_url"]
    timeout = cfg["model"]["timeout"]
    k = pop["oracle"]["max_queries"]
    ext = pop["extraction"]

    pool = MemoryPool(ROOT / cfg["output"]["memory_dir"])
    agg = ProcedureAggregator(
        dedup_threshold=ext["procedure_dedup_threshold"],
        min_sessions=ext["min_procedure_sessions"],
    )

    processed: set[str] = set()
    if resume:
        pool.load()
        for n in pool._nodes:
            for sid in n.source_sessions:
                processed.add(sid)
        pool._nodes.clear()
        logger.info(f"[resume] {len(processed)} sessions already processed")

    train_by_hop = load_train_examples(cfg, logger)

    for hop_label, examples in train_by_hop.items():
        hop = int(hop_label.replace("hop", ""))
        for ex in examples:
            qid = ex["id"]
            if qid in processed:
                logger.info(f"[skip] {qid}")
                continue

            question = ex["question"]
            gold_answer = ex["answer"]
            decomposition = ex.get("question_decomposition") or []
            passages = format_passages(ex["paragraphs"])
            gold_sup = gold_titles(ex)
            question_type = classify_question_type(question, decomposition, hop)

            logger.info(f"[populate] {qid} | {question_type} | {question[:60]}")

            if no_llm:
                pool.add(MemoryNode.new("F1_raw", hop, f"[mock] {question}", {
                    "question": question, "queries": ["mock"], "answer": "mock", "correct": False,
                }, qid))
                pool.add(MemoryNode.new("F2_chain", hop, f"[mock chain] {question}", {
                    "question": question, "question_type": question_type, "chain": "mock",
                }, qid))
                pool.flush()
                continue

            try:
                traj = run_oracle_session(question, passages, k, model, url, timeout)
            except Exception as e:
                logger.warning(f"[oracle] {qid} failed: {e}")
                continue

            score = f1_score(traj["answer"], gold_answer)
            correct = score >= 0.4
            sup_recall = supporting_recall(traj["retrieved_titles"], gold_sup)

            logger.info(
                f"  A: {traj['answer']} | gold: {gold_answer} | "
                f"F1={score:.2f} sup_recall={sup_recall:.0%} queries={len(traj['queries'])}"
            )

            # F1: raw session log
            queries_str = " → ".join(traj["queries"])
            pool.add(MemoryNode.new("F1_raw", hop, f"{question} {queries_str} {traj['answer']}", {
                "question": question,
                "queries": traj["queries"],
                "answer": traj["answer"],
                "correct": correct,
                "question_type": question_type,
            }, qid))

            # F2: gold decomposition chain (no LLM needed)
            if decomposition:
                chain_content = make_f2_chain_content(question, decomposition)
                pool.add(MemoryNode.new("F2_chain", hop, f"{question} {chain_content}", {
                    "question": question,
                    "question_type": question_type,
                    "chain": chain_content,
                    "num_steps": len(decomposition),
                }, qid))

            # Accumulate for F3/F4/F5
            agg.add(
                question_type=question_type,
                hop=hop,
                question=question,
                queries=traj["queries"],
                correct=correct,
                support_found=int(len({t for t in traj["retrieved_titles"] if t in gold_sup})),
                support_total=len(gold_sup),
                session_id=qid,
            )

            pool.flush()

    # Materialize F3, F4, F5 from clusters
    logger.info(f"\n[populate] Materializing F3/F4/F5 from {len(agg.mature_clusters())} clusters...")
    for cluster in agg.mature_clusters():
        qt = cluster["question_type"]
        hop = cluster["hop"]
        sessions = cluster["sessions"]
        success_rate = cluster["successes"] / max(len(sessions), 1)
        examples = cluster["examples"]

        if not no_llm:
            # F3: procedure
            proc = extract_f3_procedure(hop, qt, examples, model, url, timeout)
        else:
            proc = {"description": f"[mock procedure for {qt}]", "failure_modes": ""}

        if proc and proc.get("description"):
            f3_content = f"{qt}: {proc['description']}"
            pool.add(MemoryNode.new("F3_procedure", hop, f3_content, {
                "question_type": qt,
                "description": proc["description"],
                "failure_modes": proc.get("failure_modes", ""),
                "success_rate": success_rate,
                "num_sessions": len(sessions),
            }, sessions[0]))

            # F4: program (promote from F3 when success rate is sufficient)
            if success_rate >= ext["min_program_success_rate"] and not no_llm:
                program = extract_f4_program(
                    hop, proc["description"],
                    [e["question"] for e in examples[:4]],
                    model, url, timeout,
                )
            else:
                program = None

            if program and program.get("query_sequence"):
                f4_content = f"{program['name']}: " + " → ".join(program["query_sequence"])
                pool.add(MemoryNode.new("F4_program", hop, f4_content, {
                    "name": program["name"],
                    "slots": program["slots"],
                    "query_sequence": program["query_sequence"],
                    "bridge_slot": program.get("bridge_slot", ""),
                    "description": proc["description"],
                    "success_rate": success_rate,
                    "question_type": qt,
                }, sessions[0]))

        # F6: approach preference (what style worked)
        if not no_llm:
            pref = extract_f6_preference(hop, qt, examples, model, url, timeout)
        else:
            pref = {"approach": f"[mock preference for {qt}]", "avoid": ""}

        if pref and pref.get("approach"):
            f6_content = pref["approach"]
            pool.add(MemoryNode.new("F6_preference", hop, f"{qt}: {f6_content}", {
                "question_type": qt,
                "avoid": pref.get("avoid", ""),
                "success_rate": success_rate,
                "num_sessions": len(sessions),
            }, sessions[0]))

        # F7: error corrections (wrong→right pivots)
        if not no_llm:
            corr = extract_f7_correction(hop, qt, examples, model, url, timeout)
        else:
            corr = {"wrong_pattern": f"[mock mistake for {qt}]", "correction": "[mock fix]"}

        if corr and corr.get("correction"):
            f7_content = corr["correction"]
            pool.add(MemoryNode.new("F7_correction", hop, f"{qt}: {f7_content}", {
                "question_type": qt,
                "wrong_pattern": corr.get("wrong_pattern", ""),
                "num_sessions": len(sessions),
            }, sessions[0]))

        # F5: canonical decomposition pattern
        chains = [e2["question"] for e2 in cluster["examples"][:4]]
        chain_descs = [make_f2_chain_content(e2["question"], []) for e2 in cluster["examples"][:3]]

        if not no_llm:
            pattern = extract_f5_decomp(
                hop, qt, chains, chain_descs, model, url, timeout,
            )
        else:
            pattern = f"[mock pattern for {qt}]"

        if pattern:
            pool.add(MemoryNode.new("F5_decomp", hop, f"{qt}: {pattern}", {
                "question_type": qt,
                "pattern": pattern,
                "example_questions": [e2["question"] for e2 in examples[:3]],
                "num_sessions": len(sessions),
            }, sessions[0]))

    pool.flush()
    pool.load()
    logger.info(f"\n[populate] Done. {json.dumps(pool.stats(), indent=2)}")


def main():
    parser = argparse.ArgumentParser(description="E12 Phase 1: populate MuSiQue memory pool")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)
    populate(load_config(), logger, no_llm=args.no_llm, resume=args.resume)


if __name__ == "__main__":
    main()
