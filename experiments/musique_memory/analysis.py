"""
E12 Error analysis — F3_guided hop3 failures.

Three failure modes:
  SYNTH_FAIL   supporting_recall > 0  → evidence retrieved, synthesis wrong
  QUERY_CONTAM queries[0] contains procedure text → formatting bug
  NO_CHAIN     supporting_recall = 0, dfs < 0.5  → wrong retrieval, no chain

Also cross-references the memory pool to check procedure type match.

Usage:
  python experiments/musique_memory/analysis.py [--results-dir PATH] [--condition F3_guided]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.musique_memory.memory_store import MemoryPool, TFIDFMemoryIndex, _tok, FORMATS

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Question-type pattern words from evaluate.py
_QWORD_RE = re.compile(r"^(who|what|where|when|which|how)", re.IGNORECASE)

# Heuristic: a query has been contaminated by procedure text if it exceeds
# 12 words OR contains a "hop\d_" prefix OR contains a colon followed by more than 4 words
def is_contaminated(query: str) -> bool:
    q = query.strip()
    if re.search(r"hop\d+_", q):
        return True
    if re.search(r":\s+\w+.{20,}", q):   # colon followed by long text
        return True
    if len(q.split()) > 14:
        return True
    return False


def classify_failure(r: dict) -> str:
    sup = r.get("supporting_recall", 0)
    dfs = r.get("dfs_score") or 0
    queries = r.get("queries", [])

    if queries and is_contaminated(queries[0]):
        return "QUERY_CONTAM"
    if sup > 0:
        return "SYNTH_FAIL"
    if dfs >= 0.5:
        return "WRONG_CHAIN"
    return "NO_CHAIN"


def _infer_question_type(question: str, hop: int) -> str:
    """Rough type from question words — mirrors populate.py classifier."""
    words = question.lower().split()
    # grab first wh-word
    first = next((m.group(1) for w in words[:5] if (m := _QWORD_RE.match(w))), "what")
    return f"hop{hop}_{first}"


def classify_match(question: str, hop: int, pool_procs: list[dict]) -> tuple[str, str, bool]:
    """Return (question_type, retrieved_proc_type, type_match)."""
    q_type = _infer_question_type(question, hop)
    if not pool_procs:
        return q_type, "—", False
    # Simulate TF-IDF retrieval: find most similar procedure content to the question
    q_toks = _tok(question)
    best_proc = None
    best_score = -1.0
    for p in pool_procs:
        p_toks = _tok(p["content"])
        vocab = set(q_toks) | set(p_toks)
        if not vocab:
            continue
        def tf(toks):
            n = len(toks) or 1
            return {t: toks.count(t) / n for t in vocab}
        qv, pv = tf(q_toks), tf(p_toks)
        dot = sum(qv.get(t, 0) * pv.get(t, 0) for t in vocab)
        na = math.sqrt(sum(v*v for v in qv.values())) or 1
        nb = math.sqrt(sum(v*v for v in pv.values())) or 1
        score = dot / (na * nb)
        if score > best_score:
            best_score, best_proc = score, p
    if best_proc is None:
        return q_type, "—", False
    retrieved_type = best_proc["meta"].get("question_type", "?")
    return q_type, retrieved_type, q_type == retrieved_type


def clean_query_preview(q: str, max_chars: int = 80) -> str:
    q = q.replace("\n", " ").strip()
    return q[:max_chars] + ("…" if len(q) > max_chars else "")


def load_pool_procedures(pool_dir: Path, hop: int) -> list[dict]:
    path = pool_dir / "F3_procedure.jsonl"
    if not path.exists():
        return []
    procs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                p = json.loads(line)
                if p.get("hop") == hop:
                    procs.append(p)
    return procs


def analyse_condition(
    results: list[dict],
    pool_dir: Path,
    hop: int,
    verbose: bool = True,
) -> dict:
    hop_results = [r for r in results if r["hop"] == hop]
    failures = [r for r in hop_results if r["em"] == 0]
    successes = [r for r in hop_results if r["em"] > 0]
    pool_procs = load_pool_procedures(pool_dir, hop)

    mode_counts: Counter = Counter()
    type_mismatches = 0
    synth_fail_details = []
    contam_details = []
    wrong_chain_details = []

    for r in failures:
        mode = classify_failure(r)
        mode_counts[mode] += 1
        q_type, ret_type, matched = classify_match(r["question"], hop, pool_procs)

        if not matched:
            type_mismatches += 1

        detail = {
            "id": r["id"],
            "question": r["question"],
            "gold": r["gold_answer"],
            "predicted": r["predicted_answer"],
            "sup_recall": r["supporting_recall"],
            "dfs": r["dfs_score"],
            "queries": r["queries"],
            "q_type": q_type,
            "retrieved_proc_type": ret_type,
            "type_match": matched,
        }
        if mode == "SYNTH_FAIL":
            synth_fail_details.append(detail)
        elif mode == "QUERY_CONTAM":
            contam_details.append(detail)
        else:
            wrong_chain_details.append(detail)

    if not verbose:
        return {"failures": len(failures), "by_mode": dict(mode_counts), "type_mismatches": type_mismatches}

    n = len(hop_results)
    print(f"\n{'═'*70}")
    print(f"  hop{hop}  —  {len(successes)}/{n} correct  ({len(failures)} failures)")
    print(f"{'═'*70}")

    print(f"\n  Failure breakdown:")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"    {mode:<16} {count}/{len(failures)}  ({count/len(failures):.0%})")
    if type_mismatches:
        print(f"    Procedure type mismatch (among all failures): {type_mismatches}/{len(failures)}")

    print(f"\n  Pool procedures available for hop{hop}: {len(pool_procs)}")
    for p in pool_procs:
        qt = p["meta"]["question_type"]
        sr = p["meta"].get("success_rate", 0)
        ns = p["meta"].get("num_sessions", 0)
        desc = p["meta"]["description"][:80]
        print(f"    [{qt}] sr={sr:.0%} n={ns}: {desc}")

    if contam_details:
        print(f"\n  ── QUERY_CONTAM ({len(contam_details)}) ──")
        print("  First query includes procedure text — TF-IDF receives garbage.")
        for d in contam_details[:3]:
            print(f"\n    Q: {d['question'][:70]}")
            print(f"    gold='{d['gold']}' | predicted='{d['predicted']}'")
            print(f"    query[0]: {clean_query_preview(d['queries'][0])}")
            if len(d["queries"]) > 1:
                print(f"    query[1]: {clean_query_preview(d['queries'][1])}")
            print(f"    proc_type match: q={d['q_type']} retrieved={d['retrieved_proc_type']} {'✓' if d['type_match'] else '✗'}")

    if synth_fail_details:
        print(f"\n  ── SYNTH_FAIL ({len(synth_fail_details)}) ──")
        print("  Gold passages retrieved but answer synthesis wrong.")
        for d in synth_fail_details[:3]:
            print(f"\n    Q: {d['question'][:70]}")
            print(f"    gold='{d['gold']}' | predicted='{d['predicted']}'")
            print(f"    sup_recall={d['sup_recall']:.0%}  dfs={d['dfs'] or 0:.0%}")
            print(f"    queries: {' → '.join(clean_query_preview(q, 40) for q in d['queries'])}")
            print(f"    proc_type match: q={d['q_type']} retrieved={d['retrieved_proc_type']} {'✓' if d['type_match'] else '✗'}")

    if wrong_chain_details:
        print(f"\n  ── WRONG_CHAIN / NO_CHAIN ({len(wrong_chain_details)}) ──")
        for d in wrong_chain_details[:3]:
            print(f"\n    Q: {d['question'][:70]}")
            print(f"    gold='{d['gold']}' | predicted='{d['predicted']}'")
            print(f"    sup_recall={d['sup_recall']:.0%}  dfs={d['dfs'] or 0:.0%}")
            print(f"    queries: {' → '.join(clean_query_preview(q, 35) for q in d['queries'])}")

    return {"failures": len(failures), "by_mode": dict(mode_counts), "type_mismatches": type_mismatches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--condition", default="F3_guided")
    parser.add_argument("--hops", default="2,3")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    results_dir = Path(args.results_dir) if args.results_dir else ROOT / cfg["output"]["results_dir"]
    pool_dir = ROOT / cfg["output"]["memory_dir"]

    result_path = results_dir / f"{args.condition}.json"
    if not result_path.exists():
        print(f"No results found at {result_path}")
        sys.exit(1)

    data = json.loads(result_path.read_text(encoding="utf-8"))
    results = data["results"]
    hops = [int(h) for h in args.hops.split(",")]

    print(f"\nError analysis: {args.condition}  ({len(results)} total results)")

    all_stats = {}
    for hop in hops:
        stats = analyse_condition(results, pool_dir, hop, verbose=True)
        all_stats[hop] = stats

    print(f"\n{'═'*70}")
    print("  SUMMARY")
    print(f"{'═'*70}")
    for hop, stats in all_stats.items():
        n_fail = stats["failures"]
        by_mode = stats["by_mode"]
        print(f"\n  hop{hop}: {n_fail} failures")
        for mode, count in sorted(by_mode.items(), key=lambda x: -x[1]):
            print(f"    {mode:<16} {count}  ({count/n_fail:.0%})")

    print()
    print("  Root causes:")
    print("  1. QUERY_CONTAM — procedure text leaks into search query.")
    print("     Fix: strip query output to last short phrase / after 'query:' marker.")
    print("  2. SYNTH_FAIL — evidence retrieved, reasoning wrong.")
    print("     Fix: better final-answer prompt (chain-of-thought) or stronger model.")
    print("  3. WRONG_CHAIN — procedure guided wrong chain.")
    print("     Fix: larger pool (n≥10 per procedure type) + better extraction prompt.")


if __name__ == "__main__":
    main()
