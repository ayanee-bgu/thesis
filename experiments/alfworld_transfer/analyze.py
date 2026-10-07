"""
Analyze E4 ALFWorld Knowledge Transfer results.

Usage:
  python experiments/alfworld_transfer/analyze.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results" / "alfworld_transfer"
CONDITIONS = ["no_memory", "graph_session", "full_handoff", "interactive_oracle"]


def load_results() -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    for cond in CONDITIONS:
        path = RESULTS_DIR / f"{cond}.json"
        if path.exists():
            results[cond] = json.loads(path.read_text(encoding="utf-8"))
    return results


def load_summary() -> dict:
    path = RESULTS_DIR / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def print_condition_table(summary: dict) -> None:
    print("\n" + "="*80)
    print("E4: ALFWorld Plan Generation — Condition Comparison")
    print("="*80)
    conds = summary.get("conditions", [])
    if not conds:
        print("No summary data found.")
        return

    header = f"{'Condition':<22} {'Success':>8} {'Seq':>7} {'Loose':>7} {'Tokens':>8} {'Tok/Suc':>10} {'OracleQ':>8}"
    print(header)
    print("-"*75)
    for c in conds:
        tok_suc = c.get("tokens_per_successful_task", float("inf"))
        tok_suc_str = f"{tok_suc:,.0f}" if tok_suc != float("inf") else "inf"
        print(
            f"{c['condition']:<22} "
            f"{c['success_rate']:>7.1%} "
            f"{c.get('sequence_correct_rate', 0):>7.1%} "
            f"{c.get('loose_correct_rate', 0):>7.1%} "
            f"{c['mean_total_tokens']:>8,.0f} "
            f"{tok_suc_str:>10} "
            f"{c.get('mean_oracle_questions_asked', 0):>8.1f}"
        )
    print("="*80)


def print_efficiency_table(summary: dict) -> None:
    conds = summary.get("conditions", [])
    if not conds:
        return
    baseline = next((c for c in conds if c["condition"] == "no_memory"), None)
    if not baseline:
        return
    base_tok = baseline["tokens_per_successful_task"]
    print("\nEfficiency (relative to no_memory):")
    print(f"{'Condition':<22} {'Success':>8} {'Tok/Suc':>12} {'Relative':>10} {'Total(s)':>10}")
    print("-"*68)
    for c in conds:
        ts = c["tokens_per_successful_task"]
        rel = ts / base_tok if base_tok > 0 and ts != float("inf") else float("inf")
        rel_str = f"{rel:.2f}x" if rel != float("inf") else "inf"
        ts_str = f"{ts:,.0f}" if ts != float("inf") else "inf"
        print(
            f"{c['condition']:<22} "
            f"{c['success_rate']:>7.1%} "
            f"{ts_str:>12} "
            f"{rel_str:>10} "
            f"{c['mean_total_elapsed_s']:>10.0f}"
        )


def print_per_task_breakdown(results: dict[str, list[dict]]) -> None:
    all_tasks: list[str] = []
    for res_list in results.values():
        for r in res_list:
            if r["task_id"] not in all_tasks:
                all_tasks.append(r["task_id"])
    all_tasks.sort()

    if not all_tasks:
        return

    print("\nPer-Task Breakdown (S=success, Q=seq_correct):")
    header = f"{'Task':<45}"
    for cond in CONDITIONS:
        if cond in results:
            short = cond.replace("no_memory", "no_mem").replace("graph_session", "graph").replace("full_handoff", "full").replace("interactive_oracle", "oracle")
            header += f" {short:>8}"
    print(header)
    print("-"*80)
    for task_id in all_tasks:
        short_id = task_id[:44]
        row = f"{short_id:<45}"
        for cond in CONDITIONS:
            if cond not in results:
                continue
            task_res = next((r for r in results[cond] if r["task_id"] == task_id), None)
            if task_res:
                s = "Y" if task_res["success"] else "N"
                q = "Y" if task_res["sequence_correct"] else "N"
                row += f"  {s}/{q}     "
            else:
                row += "  -/-     "
        print(row)
    print()
    print("(S=composite success, Q=exact sequence match)")


def print_oracle_question_analysis(results: dict[str, list[dict]]) -> None:
    oracle_res = results.get("interactive_oracle", [])
    if not oracle_res:
        return
    all_questions = []
    for r in oracle_res:
        all_questions.extend(r.get("oracle_questions", []))
    if not all_questions:
        return

    print(f"\nOracle Question Analysis ({len(all_questions)} total questions):")
    words = Counter()
    for q in all_questions:
        for w in q.lower().split():
            w = w.strip("?.,;:")
            if len(w) > 4:
                words[w] += 1
    top = words.most_common(15)
    print("Top keywords:", ", ".join(f"{w}({n})" for w, n in top))
    print("\nSample questions:")
    for q in all_questions[:5]:
        print(f"  > {q[:100]}")


def print_action_analysis(results: dict[str, list[dict]]) -> None:
    print("\nAction Sequence Analysis:")
    for cond in CONDITIONS:
        if cond not in results:
            continue
        correct = sum(1 for r in results[cond] if r["sequence_correct"])
        total = len(results[cond])
        print(f"\n  {cond}:")
        for r in results[cond]:
            gt = r.get("gt_actions", [])
            gen = r.get("generated_actions", [])
            match = "OK" if r["sequence_correct"] else "MISMATCH"
            print(f"    {r['task_id'][:35]:35} GT:{gt} -> Gen:{gen} [{match}]")


def compare_with_e3(summary: dict) -> None:
    e3_success = {"no_memory": 0.60, "graph_session": 0.60, "full_handoff": 0.40, "interactive_oracle": 0.20}
    conds = {c["condition"]: c for c in summary.get("conditions", [])}
    if not conds:
        return
    print("\nComparison with E3 (TravelPlanner) — context pollution test:")
    print(f"{'Condition':<22} {'E3 Success':>12} {'E4 Success':>12} {'Delta':>8}")
    print("-"*58)
    for cond in CONDITIONS:
        if cond not in conds:
            continue
        e3 = e3_success.get(cond, None)
        e4 = conds[cond]["success_rate"]
        if e3 is not None:
            delta = e4 - e3
            delta_str = f"{delta:+.1%}"
        else:
            delta_str = "N/A"
        e3_str = f"{e3:.1%}" if e3 is not None else "N/A"
        print(f"{cond:<22} {e3_str:>12} {e4:>12.1%} {delta_str:>8}")
    print()
    # Assess context pollution
    nm = conds.get("no_memory", {}).get("success_rate", None)
    fh = conds.get("full_handoff", {}).get("success_rate", None)
    if nm is not None and fh is not None:
        if fh < nm:
            print("Context pollution: YES — full_handoff degrades vs no_memory (same as E3)")
        elif fh > nm:
            print("Context pollution: NO — full_handoff IMPROVES vs no_memory (ALFWorld benefits from context)")
        else:
            print("Context pollution: NEUTRAL — no difference")


def make_efficiency_chart(summary: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not available — skipping chart")
        return

    conds = summary.get("conditions", [])
    if not conds:
        return

    names = [c["condition"].replace("_", "\n") for c in conds]
    success = [c["success_rate"] * 100 for c in conds]
    tokens = [c["mean_total_tokens"] for c in conds]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Efficiency frontier
    ax1.scatter(tokens, success, c=colors[:len(conds)], s=150, zorder=5)
    for i, (x, y, name) in enumerate(zip(tokens, success, names)):
        ax1.annotate(name, (x, y), textcoords="offset points", xytext=(8, 0), fontsize=9)
    ax1.set_xlabel("Mean Total Tokens")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("E4 ALFWorld: Efficiency Frontier\n(lower-right = better)")
    ax1.grid(True, alpha=0.3)

    # Success rate bar chart
    short = ["no_mem", "graph", "full", "oracle"][:len(conds)]
    bars = ax2.bar(short, success, color=colors[:len(conds)])
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Success Rate by Condition")
    ax2.set_ylim(0, 110)
    for bar, val in zip(bars, success):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.0f}%",
                 ha="center", va="bottom", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = RESULTS_DIR / "efficiency_frontier.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {out}")


def main() -> None:
    results = load_results()
    summary = load_summary()

    if not results and not summary:
        print("No results found. Run main.py first.")
        return

    print_condition_table(summary)
    print_efficiency_table(summary)
    print_per_task_breakdown(results)
    print_oracle_question_analysis(results)
    print_action_analysis(results)
    compare_with_e3(summary)
    make_efficiency_chart(summary)


if __name__ == "__main__":
    main()
