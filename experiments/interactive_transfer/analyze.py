"""
E3 Analysis: metrics tables + efficiency frontier chart.

Usage:
  python experiments/interactive_transfer/analyze.py
  python experiments/interactive_transfer/analyze.py --results-dir results/interactive_transfer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

CONDITIONS = ["no_memory", "graph_session", "full_handoff", "interactive_oracle"]
CONDITION_LABELS = {
    "no_memory": "No Memory",
    "graph_session": "Graph (E1)",
    "full_handoff": "Full Handoff",
    "interactive_oracle": "Interactive Oracle",
}


def load_results(results_dir: Path) -> list[dict]:
    results = []
    for cond in CONDITIONS:
        path = results_dir / f"{cond}.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            print(f"[warn] Missing: {path}")
    return results


def print_comparison_table(results: list[dict]):
    print("\n=== E3: CONDITION COMPARISON ===")
    cols = [
        ("Success Rate", "success_rate", ".1%"),
        ("Mean Total Tok", "mean_total_tokens", ".0f"),
        ("Mean Plan Tok", "mean_plan_tokens", ".0f"),
        ("Mean Oracle Tok", "mean_oracle_tokens", ".0f"),
        ("Tok/Success", "tokens_per_successful_task", ".0f"),
        ("Mem Chars", "mean_memory_chars", ".0f"),
        ("Oracle Qs", "mean_oracle_questions_asked", ".1f"),
    ]
    header = f"{'Condition':<22}" + "".join(f"{label:>16}" for label, _, _ in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        s = r["summary"]
        cond = r["condition"]
        row = f"{CONDITION_LABELS.get(cond, cond):<22}"
        for _, key, fmt in cols:
            val = s.get(key, 0) or 0
            row += f"{format(val, fmt):>16}"
        print(row)


def print_efficiency_table(results: list[dict]):
    """Tokens vs success — the cost-efficiency view."""
    print("\n=== EFFICIENCY: TOKENS PER SUCCESSFUL TASK ===")
    print(f"{'Condition':<22} {'Success':>10} {'Total Tok':>12} {'Tok/Success':>14} {'vs no_memory':>14}")
    print("-" * 74)
    baseline_tps = None
    rows = []
    for r in results:
        s = r["summary"]
        tps = s.get("tokens_per_successful_task", 0) or 0
        sr = s.get("success_rate", 0) or 0
        tt = s.get("mean_total_tokens", 0) or 0
        if r["condition"] == "no_memory":
            baseline_tps = tps
        rows.append((r["condition"], sr, tt, tps))

    for cond, sr, tt, tps in rows:
        if baseline_tps and baseline_tps > 0:
            ratio = tps / baseline_tps
            vs_str = f"{ratio:.2f}x"
        else:
            vs_str = "—"
        label = CONDITION_LABELS.get(cond, cond)
        print(f"{label:<22} {sr:>9.1%} {tt:>12.0f} {tps:>14.0f} {vs_str:>14}")


def print_latency_table(results: list[dict]):
    """Wall-clock latency breakdown — tests whether smaller context = faster inference."""
    print("\n=== INFERENCE LATENCY BREAKDOWN (seconds per task) ===")
    print(f"{'Condition':<22} {'Plan call':>11} {'Oracle calls':>14} {'Total':>8} {'Oracle/call':>13} {'vs no_memory':>14}")
    print("-" * 84)
    baseline_total = None
    for r in results:
        s = r["summary"]
        if r["condition"] == "no_memory":
            baseline_total = s.get("mean_total_elapsed_s", 0) or 0
        plan_s = s.get("mean_plan_elapsed_s", 0) or 0
        oracle_s = s.get("mean_oracle_elapsed_s", 0) or 0
        total_s = s.get("mean_total_elapsed_s", 0) or 0
        per_call_s = s.get("mean_oracle_call_elapsed_s", 0) or 0
        if baseline_total and baseline_total > 0:
            ratio = total_s / baseline_total
            vs_str = f"{ratio:.2f}x"
        else:
            vs_str = "—"
        label = CONDITION_LABELS.get(r["condition"], r["condition"])
        per_call_str = f"{per_call_s:.1f}s" if per_call_s > 0 else "—"
        print(f"{label:<22} {plan_s:>10.1f}s {oracle_s:>13.1f}s {total_s:>7.1f}s {per_call_str:>13} {vs_str:>14}")
    print()
    print("Note: 'Plan call' = single planning LLM call latency.")
    print("      'Oracle calls' = total time spent in oracle Q&A (interactive_oracle only).")
    print("      Smaller plan call latency for interactive_oracle vs full_handoff indicates")
    print("      that reduced context size leads to faster per-call inference.")


def print_oracle_question_analysis(results: list[dict]):
    """What questions did the student ask? (interactive_oracle only)"""
    oracle_result = next((r for r in results if r["condition"] == "interactive_oracle"), None)
    if not oracle_result:
        return

    all_questions = []
    for task in oracle_result.get("tasks", []):
        all_questions.extend(task.get("oracle_questions", []))

    if not all_questions:
        print("\n[oracle analysis] No questions recorded.")
        return

    print(f"\n=== ORACLE QUESTION ANALYSIS (n={len(all_questions)} questions) ===")
    # Simple keyword frequency — what topics come up most?
    keywords = Counter()
    for q in all_questions:
        words = q.lower().split()
        for w in words:
            if len(w) > 4 and w not in {"what", "which", "where", "when", "with", "from", "that", "this", "have", "does", "were", "they"}:
                keywords[w] += 1

    print("Top 15 keywords in oracle questions:")
    for word, count in keywords.most_common(15):
        print(f"  {word}: {count}")

    print(f"\nSample questions (first 5):")
    for q in all_questions[:5]:
        print(f"  - {q[:120]}")


def plot_efficiency_frontier(results: list[dict], out_path: Path):
    """Success rate vs total tokens scatter + bar chart."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[analyze] matplotlib not installed — skipping plot")
        return

    colors = {
        "no_memory": "#9E9E9E",
        "graph_session": "#2196F3",
        "full_handoff": "#F44336",
        "interactive_oracle": "#4CAF50",
    }
    markers = {"no_memory": "o", "graph_session": "s", "full_handoff": "^", "interactive_oracle": "*"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: efficiency frontier (success vs total tokens)
    ax = axes[0]
    for r in results:
        s = r["summary"]
        cond = r["condition"]
        x = s.get("mean_total_tokens", 0) or 0
        y = (s.get("success_rate", 0) or 0) * 100
        ax.scatter(x, y, c=colors.get(cond, "black"), marker=markers.get(cond, "o"),
                   s=150, zorder=5, label=CONDITION_LABELS.get(cond, cond))
        ax.annotate(CONDITION_LABELS.get(cond, cond), (x, y),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean Total Tokens (plan + oracle)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Efficiency Frontier\n(upper-left = better)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: stacked bar — plan tokens + oracle tokens per condition
    ax2 = axes[1]
    conds = [r["condition"] for r in results]
    labels = [CONDITION_LABELS.get(c, c) for c in conds]
    plan_toks = [r["summary"].get("mean_plan_tokens", 0) or 0 for r in results]
    oracle_toks = [r["summary"].get("mean_oracle_tokens", 0) or 0 for r in results]

    x_pos = range(len(conds))
    ax2.bar(x_pos, plan_toks, label="Plan tokens", color="#42A5F5")
    ax2.bar(x_pos, oracle_toks, bottom=plan_toks, label="Oracle tokens", color="#EF5350")
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Mean tokens per task")
    ax2.set_title("Token Breakdown by Condition")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Right-most: stacked bar — plan latency + oracle latency per condition
    ax3 = axes[2]
    plan_s = [r["summary"].get("mean_plan_elapsed_s", 0) or 0 for r in results]
    oracle_s = [r["summary"].get("mean_oracle_elapsed_s", 0) or 0 for r in results]
    ax3.bar(x_pos, plan_s, label="Plan call (s)", color="#42A5F5")
    ax3.bar(x_pos, oracle_s, bottom=plan_s, label="Oracle calls (s)", color="#FF7043")
    ax3.set_xticks(list(x_pos))
    ax3.set_xticklabels(labels, rotation=15, ha="right")
    ax3.set_ylabel("Mean seconds per task")
    ax3.set_title("Inference Latency Breakdown\n(smaller context → faster plan call?)")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[analyze] Plot saved to {out_path}")
    plt.close()


def print_per_task_breakdown(results: list[dict]):
    """Per-task success grid for easy comparison."""
    all_task_ids: list[str] = []
    for r in results:
        for task in r.get("tasks", []):
            tid = task["task_id"]
            if tid not in all_task_ids:
                all_task_ids.append(tid)

    if not all_task_ids:
        return

    cond_results: dict[str, dict[str, bool]] = {}
    for r in results:
        cond = r["condition"]
        cond_results[cond] = {t["task_id"]: t["success"] for t in r.get("tasks", [])}

    available_conds = [c for c in CONDITIONS if c in cond_results]
    print("\n=== PER-TASK BREAKDOWN (Y=success, N=fail) ===")
    header = f"{'Task':<14}" + "".join(f"{CONDITION_LABELS.get(c, c)[:14]:>16}" for c in available_conds)
    print(header)
    print("-" * len(header))
    for tid in all_task_ids:
        row = f"{tid:<14}"
        for cond in available_conds:
            val = cond_results[cond].get(tid)
            if val is None:
                row += f"{'?':>16}"
            else:
                row += f"{'Y' if val else 'N':>16}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="E3 analysis")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else ROOT / "results" / "interactive_transfer"
    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}. Run main.py first.")
        sys.exit(1)

    print_comparison_table(results)
    print_efficiency_table(results)
    print_latency_table(results)
    print_per_task_breakdown(results)
    print_oracle_question_analysis(results)
    plot_efficiency_frontier(results, results_dir / "efficiency_frontier.png")


if __name__ == "__main__":
    main()
