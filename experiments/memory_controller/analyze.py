"""
E1 analysis: load results JSON files, compute metrics, print tables, save plots.

Usage:
    python experiments/memory_controller/analyze.py [--results-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "memory_controller"

CONDITION_ORDER = ["no_memory", "memp_flat", "graph_pertraj", "graph_session"]
CONDITION_LABELS = {
    "no_memory": "No Memory",
    "memp_flat": "MemP Flat",
    "graph_pertraj": "Graph (per-traj)",
    "graph_session": "Graph (session-end)",
}


def load_results(results_dir: Path) -> dict[str, dict]:
    data = {}
    for cond in CONDITION_ORDER:
        p = results_dir / f"{cond}.json"
        if p.exists():
            data[cond] = json.loads(p.read_text(encoding="utf-8"))
    return data


def per_task_table(results: dict[str, dict]):
    print("\n=== PER-CONDITION SUMMARY ===")
    print(f"{'Condition':<22} {'N':>4} {'Success%':>10} {'Avg Steps':>10} {'Avg Tokens':>12} {'Retrievals':>12} {'Graph Nodes':>12}")
    print("-" * 86)
    for cond, data in results.items():
        s = data.get("summary", {})
        tasks = data.get("tasks", [])
        nodes = data.get("graph_nodes") or "-"
        print(
            f"{CONDITION_LABELS.get(cond, cond):<22} {len(tasks):>4} "
            f"{s.get('success_rate', 0):>9.1%} {s.get('mean_steps', 0):>10.1f} "
            f"{s.get('mean_tokens', 0):>12.0f} {s.get('total_retrieval_calls', 0):>12} "
            f"{str(nodes):>12}"
        )


def conflict_report(results: dict[str, dict]):
    graph_conds = {k: v for k, v in results.items() if v.get("memory_type") == "graph"}
    if not graph_conds:
        return
    print("\n=== CONFLICT DETECTION (graph conditions) ===")
    for cond, data in graph_conds.items():
        tasks = data.get("tasks", [])
        total_conflicts = sum(t.get("conflicts_detected", 0) for t in tasks)
        print(f"  {cond}: {total_conflicts} conflicts detected across {len(tasks)} tasks")


def token_efficiency(results: dict[str, dict]):
    baseline = results.get("no_memory", {}).get("summary", {}).get("mean_tokens")
    if not baseline or baseline == 0:
        return
    print("\n=== TOKEN EFFICIENCY (vs. no_memory baseline) ===")
    for cond, data in results.items():
        mean_tok = data.get("summary", {}).get("mean_tokens", 0)
        diff = mean_tok - baseline
        sign = "+" if diff >= 0 else ""
        success_rate = data.get("summary", {}).get("success_rate", 0)
        base_success = results.get("no_memory", {}).get("summary", {}).get("success_rate", 0)
        success_delta = success_rate - base_success
        print(
            f"  {CONDITION_LABELS.get(cond, cond):<22}: "
            f"tokens {sign}{diff:+.0f} ({mean_tok:.0f} total), "
            f"success Δ={success_delta:+.1%}"
        )


def plot_results(results: dict[str, dict], out_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n[analyze] matplotlib not installed — skipping plots")
        return

    conds = [c for c in CONDITION_ORDER if c in results]
    labels = [CONDITION_LABELS.get(c, c) for c in conds]
    success_rates = [results[c]["summary"].get("success_rate", 0) * 100 for c in conds]
    mean_tokens = [results[c]["summary"].get("mean_tokens", 0) for c in conds]
    mean_steps = [results[c]["summary"].get("mean_steps", 0) for c in conds]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("E1: Graph Memory Controller vs. MemP Baselines", fontsize=12)

    colors = ["#999999", "#4C72B0", "#55A868", "#C44E52"][:len(conds)]

    for ax, values, ylabel, title in zip(
        axes,
        [success_rates, mean_tokens, mean_steps],
        ["Success Rate (%)", "Mean Tokens / Task", "Mean Steps / Task"],
        ["Task Success Rate", "Token Cost", "Steps to Completion"],
    ):
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8
            )

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "e1_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n[analyze] Plot saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze E1 memory controller results")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}. Run main.py first.")
        sys.exit(1)

    per_task_table(results)
    token_efficiency(results)
    conflict_report(results)
    plot_results(results, results_dir / "plots")


if __name__ == "__main__":
    main()
