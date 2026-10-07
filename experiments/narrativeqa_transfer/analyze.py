"""
Analyze E9 NarrativeQA results — when full_context must be truncated to fit,
how much of its failure is because the answer was structurally unreachable
(truncated away) vs visible-but-still-wrong? Does interactive_retrieval, which
searches the FULL untruncated document, recover the structurally-unreachable
cases?

Usage:
  python experiments/narrativeqa_transfer/analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results" / "narrativeqa_transfer"
CONDITIONS = ["no_context", "full_context", "summary_doc", "interactive_retrieval"]
KINDS = ["gutenberg", "movie"]


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


HEADER = f"{'Condition':<24} {'EM':>7} {'F1':>7} {'Tokens':>9} {'Sec':>8} {'Contain':>10} {'DFS':>8}"


def print_overall_table(summary: dict) -> None:
    print("\n" + "=" * 80)
    print("E9: NarrativeQA — Overall (gutenberg + movie combined)")
    print("=" * 80)
    conds = summary.get("conditions", [])
    if not conds:
        print("No summary data found.")
        return
    print(HEADER)
    print("-" * len(HEADER))
    for c in conds:
        print(_row(c["condition"], c.get("overall", {})))


def print_by_kind_tables(summary: dict) -> None:
    conds = summary.get("conditions", [])
    if not conds:
        return
    for kind in KINDS:
        print(f"\n-- {kind} --")
        print(HEADER)
        print("-" * len(HEADER))
        for c in conds:
            s = c.get(f"by_type_{kind}", {})
            print(_row(c["condition"], s))


def print_structurally_unreachable_analysis(results: dict[str, dict]) -> None:
    fc = results.get("full_context")
    ir = results.get("interactive_retrieval")
    if not fc:
        return
    fc_rows = [r for res in fc["results_by_type"].values() for r in res]
    ir_by_id = {}
    if ir:
        for res in ir["results_by_type"].values():
            for r in res:
                ir_by_id[r["id"]] = r

    failures = [r for r in fc_rows if r["f1"] < 0.3]
    unreachable = [r for r in failures if r["supporting_fact_recall"] == 0.0]
    visible_wrong = [r for r in failures if r["supporting_fact_recall"] == 1.0]

    print(f"\nfull_context failures (F1 < 0.3): {len(failures)}/{len(fc_rows)}")
    print(f"  structurally unreachable (answer truncated away, containment=0): {len(unreachable)}")
    print(f"  visible but still wrong (containment=1, same failure mode as E7 distractor confusion): {len(visible_wrong)}")

    if ir_by_id and unreachable:
        recovered = [r for r in unreachable if ir_by_id.get(r["id"], {}).get("f1", 0) >= 0.3]
        print(f"\n  Of the {len(unreachable)} structurally-unreachable failures, "
              f"interactive_retrieval recovers (F1 >= 0.3) on {len(recovered)} "
              f"({len(recovered)/len(unreachable):.0%})")
        for r in unreachable[:3]:
            ir_r = ir_by_id.get(r["id"], {})
            print(f"    Q: {r['question'][:80]}")
            print(f"       gold={r['gold_answer']!r} | full_context={r['predicted_answer'][:50]!r} (f1={r['f1']:.2f})"
                  f" | interactive={ir_r.get('predicted_answer', 'n/a')[:50]!r} (f1={ir_r.get('f1', 0):.2f}, contain={ir_r.get('supporting_fact_recall', 0):.0f})")


def print_containment_by_condition(results: dict[str, dict]) -> None:
    print("\nAnswer-Containment Rate (did the model's given context contain the gold answer at all?):")
    hdr = f"{'Condition':<24}" + "".join(f" {k:>12}" for k in KINDS)
    print(hdr)
    print("-" * len(hdr))
    for cond in CONDITIONS:
        if cond not in results:
            continue
        row = f"{cond:<24}"
        for kind in KINDS:
            items = results[cond]["results_by_type"].get(kind, [])
            rate = sum(r["supporting_fact_recall"] for r in items) / len(items) if items else 0
            row += f" {rate:>11.1%} "
        print(row)


def make_charts(summary: dict, results: dict[str, dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not available — skipping charts")
        return

    conds = summary.get("conditions", [])
    if not conds:
        return

    names = [c["condition"] for c in conds]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"][:len(conds)]
    f1s = [c.get("overall", {}).get("f1", 0) * 100 for c in conds]
    containment = [c.get("overall", {}).get("mean_supporting_fact_recall", 0) * 100 for c in conds]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars = axes[0].bar([n.replace("_", "\n") for n in names], f1s, color=colors)
    axes[0].set_ylabel("F1 (%)")
    axes[0].set_title("F1 by Condition")
    for bar, val in zip(bars, f1s):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    bars = axes[1].bar([n.replace("_", "\n") for n in names], containment, color=colors)
    axes[1].set_ylabel("Answer-containment rate (%)")
    axes[1].set_title("Does the given context even contain the answer?")
    axes[1].set_ylim(0, 110)
    for bar, val in zip(bars, containment):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_dir = RESULTS_DIR / "plots"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "e9_infeasible_context.png"
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
    print_by_kind_tables(summary)
    print_containment_by_condition(results)
    print_structurally_unreachable_analysis(results)
    make_charts(summary, results)


if __name__ == "__main__":
    main()
