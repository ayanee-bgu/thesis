"""
Analyze E8 QASPER results — does full_context degrade as paper length approaches
llama3:8b's context window?

Usage:
  python experiments/qasper_transfer/analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results" / "qasper_transfer"
CONDITIONS = ["no_context", "full_context", "summary_doc", "interactive_retrieval"]
LENGTH_BUCKETS = ["short", "medium", "long"]


def load_results() -> dict[str, dict]:
    out = {}
    for cond in CONDITIONS:
        path = RESULTS_DIR / f"{cond}.json"
        if path.exists():
            out[cond] = json.loads(path.read_text(encoding="utf-8"))
    return out


def load_summary() -> dict:
    path = RESULTS_DIR / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _row(cond: str, s: dict) -> str:
    dfs = s.get("mean_dfs_score")
    dfs_str = f"{dfs:.1%}" if dfs is not None else "-"
    return (
        f"{cond:<24} {s.get('exact_match', 0):>7.1%} {s.get('f1', 0):>7.1%} "
        f"{s.get('mean_tokens', 0):>9.0f} {s.get('mean_elapsed_s', 0):>8.1f} "
        f"{s.get('mean_supporting_fact_recall', 0):>10.1%} {dfs_str:>8}"
    )


HEADER = f"{'Condition':<24} {'EM':>7} {'F1':>7} {'Tokens':>9} {'Sec':>8} {'EvRecall':>10} {'DFS':>8}"


def print_overall_table(summary: dict) -> None:
    print("\n" + "=" * 80)
    print("E8: QASPER — Overall (short + medium + long combined)")
    print("=" * 80)
    conds = summary.get("conditions", [])
    if not conds:
        print("No summary data found.")
        return
    print(HEADER)
    print("-" * len(HEADER))
    for c in conds:
        print(_row(c["condition"], c.get("overall", {})))


def print_by_bucket_tables(summary: dict) -> None:
    conds = summary.get("conditions", [])
    if not conds:
        return
    for bucket in LENGTH_BUCKETS:
        print(f"\n-- {bucket} papers --")
        print(HEADER)
        print("-" * len(HEADER))
        for c in conds:
            s = c.get(f"by_type_{bucket}", {})
            print(_row(c["condition"], s))


def print_full_context_degradation(summary: dict, results: dict[str, dict]) -> None:
    conds = {c["condition"]: c for c in summary.get("conditions", [])}
    fc = conds.get("full_context")
    if not fc:
        return
    print("\nDoes full_context degrade as paper length approaches the context window?")
    print(f"{'Bucket':<10} {'MeanWords':>10} {'EM':>8} {'F1':>8} {'EvRecall':>10}")
    print("-" * 50)
    for bucket in LENGTH_BUCKETS:
        s = fc.get(f"by_type_{bucket}", {})
        rows = results.get("full_context", {}).get("results_by_type", {}).get(bucket, [])
        mean_words = sum(r["paper_word_count"] for r in rows) / len(rows) if rows else 0
        print(f"{bucket:<10} {mean_words:>10.0f} {s.get('exact_match', 0):>8.1%} "
              f"{s.get('f1', 0):>8.1%} {s.get('mean_supporting_fact_recall', 0):>10.1%}")


def print_recall_analysis(results: dict[str, dict]) -> None:
    print("\nEvidence-Containment Recall (did the model actually see the gold evidence text?):")
    hdr = f"{'Condition':<24}" + "".join(f" {b:>10}" for b in LENGTH_BUCKETS)
    print(hdr)
    print("-" * len(hdr))
    for cond in CONDITIONS:
        if cond not in results:
            continue
        row = f"{cond:<24}"
        for bucket in LENGTH_BUCKETS:
            items = results[cond]["results_by_type"].get(bucket, [])
            recalls = [r["supporting_fact_recall"] for r in items]
            mean_r = sum(recalls) / len(recalls) if recalls else 0
            row += f" {mean_r:>9.1%} "
        print(row)


def make_charts(summary: dict, results: dict[str, dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[analyze] matplotlib not available — skipping charts")
        return

    conds = summary.get("conditions", [])
    if not conds:
        return

    names = [c["condition"] for c in conds]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"][:len(conds)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. F1 vs length bucket, one line per condition
    for i, cond in enumerate(conds):
        f1s = [cond.get(f"by_type_{b}", {}).get("f1", 0) * 100 for b in LENGTH_BUCKETS]
        axes[0].plot(LENGTH_BUCKETS, f1s, "o-", label=names[i], color=colors[i])
    axes[0].set_xlabel("Paper length bucket")
    axes[0].set_ylabel("F1 (%)")
    axes[0].set_title("F1 vs Paper Length, by Condition")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 2. full_context EvRecall vs length bucket (does evidence visibility drop?)
    fc = next((c for c in conds if c["condition"] == "full_context"), None)
    if fc:
        rec = [fc.get(f"by_type_{b}", {}).get("mean_supporting_fact_recall", 0) * 100 for b in LENGTH_BUCKETS]
        bars = axes[1].bar(LENGTH_BUCKETS, rec, color="#2196F3")
        axes[1].set_ylabel("Evidence-containment recall (%)")
        axes[1].set_title("full_context: Evidence Recall vs Paper Length")
        axes[1].set_ylim(0, 110)
        for bar, val in zip(bars, rec):
            axes[1].text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
        axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_dir = RESULTS_DIR / "plots"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "e8_length_degradation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {out}")


def main() -> None:
    results = load_results()
    summary = load_summary()

    if not results and not summary:
        print("No results found. Run main.py first.")
        return

    print_overall_table(summary)
    print_by_bucket_tables(summary)
    print_full_context_degradation(summary, results)
    print_recall_analysis(results)
    make_charts(summary, results)


if __name__ == "__main__":
    main()
