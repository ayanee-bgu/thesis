"""
Analyze E6 2WikiMultihopQA Interactive Retrieval vs Static Context results.

Usage:
  python experiments/wikihop_transfer/analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results" / "wikihop_transfer"
CONDITIONS = ["no_context", "full_context", "summary_doc", "interactive_retrieval"]
QUESTION_TYPES = ["compositional", "comparison", "bridge_comparison", "inference"]
# Only `comparison` has no dependency chain to follow (two independent, already-named
# entities) — the other three all require some form of multi-hop chaining/inference.
BFS_SHAPED = {"comparison"}


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


HEADER = f"{'Condition':<24} {'EM':>7} {'F1':>7} {'Tokens':>9} {'Sec':>8} {'SF-Recall':>10} {'DFS':>8}"


def print_overall_table(summary: dict) -> None:
    print("\n" + "=" * 80)
    print("E6: 2WikiMultihopQA — Overall (all 4 types combined)")
    print("=" * 80)
    conds = summary.get("conditions", [])
    if not conds:
        print("No summary data found.")
        return
    print(HEADER)
    print("-" * len(HEADER))
    for c in conds:
        print(_row(c["condition"], c.get("overall", {})))


def print_by_type_tables(summary: dict) -> None:
    conds = summary.get("conditions", [])
    if not conds:
        return
    for qtype in QUESTION_TYPES:
        shape = "BFS-shaped: two independent entities" if qtype in BFS_SHAPED else "DFS-shaped: requires chaining/inference"
        print(f"\n-- {qtype} ({shape}) --")
        print(HEADER)
        print("-" * len(HEADER))
        for c in conds:
            s = c.get(f"by_type_{qtype}", {})
            print(_row(c["condition"], s))


def print_dfs_vs_bfs_delta(summary: dict) -> None:
    conds = {c["condition"]: c for c in summary.get("conditions", [])}
    if not conds:
        return
    dfs_types = [t for t in QUESTION_TYPES if t not in BFS_SHAPED]
    print(f"\nDoes interactive_retrieval's advantage concentrate on DFS-shaped types ({', '.join(dfs_types)}) vs BFS-shaped (comparison)?")
    print(f"{'Condition':<24} {'Mean DFS-type F1':>17} {'Comparison F1':>14} {'DFS - Comparison':>17}")
    print("-" * 76)
    for cond in CONDITIONS:
        if cond not in conds:
            continue
        dfs_f1s = [conds[cond].get(f"by_type_{t}", {}).get("f1", 0) for t in dfs_types]
        mean_dfs_f1 = sum(dfs_f1s) / len(dfs_f1s) if dfs_f1s else 0
        comp_f1 = conds[cond].get("by_type_comparison", {}).get("f1", 0)
        print(f"{cond:<24} {mean_dfs_f1:>17.1%} {comp_f1:>14.1%} {mean_dfs_f1 - comp_f1:>+16.1%}")


def print_interactive_query_analysis(results: dict[str, dict]) -> None:
    inter = results.get("interactive_retrieval")
    if not inter:
        return
    print("\nInteractive Retrieval — Query Chain Analysis:")
    for qtype in QUESTION_TYPES:
        rows = inter["results_by_type"].get(qtype, [])
        dfs_scores = [r["dfs_score"] for r in rows if r["dfs_score"] is not None]
        mean_dfs = sum(dfs_scores) / len(dfs_scores) if dfs_scores else None
        print(f"\n  {qtype}: mean chain-following (DFS) rate = "
              f"{f'{mean_dfs:.1%}' if mean_dfs is not None else 'n/a'} over {len(dfs_scores)} multi-query tasks")
        for r in rows[:2]:
            qs = r.get("queries") or []
            print(f"    [{r['id'][:12]}] dfs={r['dfs_score']}: {' -> '.join(qs)}")


def print_recall_analysis(results: dict[str, dict]) -> None:
    print("\nSupporting-Fact Recall (did the model actually see the gold passages?):")
    hdr = f"{'Condition':<24}" + "".join(f" {t[:14]:>15}" for t in QUESTION_TYPES)
    print(hdr)
    print("-" * len(hdr))
    for cond in CONDITIONS:
        if cond not in results:
            continue
        row = f"{cond:<24}"
        for qtype in QUESTION_TYPES:
            items = results[cond]["results_by_type"].get(qtype, [])
            recalls = [r["supporting_fact_recall"] for r in items]
            mean_r = sum(recalls) / len(recalls) if recalls else 0
            row += f" {mean_r:>14.1%} "
        print(row)


def make_charts(summary: dict) -> None:
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
    type_colors = {"compositional": "#2196F3", "comparison": "#FF9800", "bridge_comparison": "#4CAF50", "inference": "#9C27B0"}

    tokens = [c.get("overall", {}).get("mean_tokens", 0) for c in conds]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 1. Grouped bar: F1 by condition x question type (4 types)
    x = np.arange(len(names))
    w = 0.8 / len(QUESTION_TYPES)
    for i, qtype in enumerate(QUESTION_TYPES):
        f1s = [c.get(f"by_type_{qtype}", {}).get("f1", 0) * 100 for c in conds]
        offset = (i - (len(QUESTION_TYPES) - 1) / 2) * w
        axes[0].bar(x + offset, f1s, w, label=qtype, color=type_colors.get(qtype))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    axes[0].set_ylabel("F1 (%)")
    axes[0].set_title("F1 by Condition x Question Type")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3, axis="y")

    # 2. Token cost by condition
    bars = axes[1].bar([n.replace("_", "\n") for n in names], tokens, color=colors)
    axes[1].set_ylabel("Mean Tokens")
    axes[1].set_title("Token Cost by Condition")
    for bar, val in zip(bars, tokens):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    # 3. DFS chain-following score for interactive_retrieval, by type
    inter = next((c for c in conds if c["condition"] == "interactive_retrieval"), None)
    if inter:
        vals = [(inter.get(f"by_type_{t}", {}).get("mean_dfs_score") or 0) * 100 for t in QUESTION_TYPES]
        bars = axes[2].bar(QUESTION_TYPES, vals, color=[type_colors.get(t) for t in QUESTION_TYPES])
        axes[2].set_xticklabels(QUESTION_TYPES, fontsize=7)
        axes[2].set_ylabel("Chain-following rate (%)")
        axes[2].set_title("interactive_retrieval: DFS Score by Type")
        axes[2].set_ylim(0, 110)
        for bar, val in zip(bars, vals):
            axes[2].text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
        axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_dir = RESULTS_DIR / "plots"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "e6_dfs_vs_bfs.png"
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
    print_by_type_tables(summary)
    print_dfs_vs_bfs_delta(summary)
    print_recall_analysis(results)
    print_interactive_query_analysis(results)
    make_charts(summary)


if __name__ == "__main__":
    main()
