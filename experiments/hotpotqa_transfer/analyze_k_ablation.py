"""
Analyze the E5 K-ablation: does a bigger interactive_retrieval query budget
close the retrieval-recall gap identified in REPORT.md?

Loads K=2,5,7,10 from interactive_retrieval_k{K}.json (produced by
k_ablation.py) plus K=3 from interactive_retrieval.json (the main E5 run —
same mechanism, same questions, not rerun), and compares against the static
summary_doc/full_context ceilings from the main run.

Usage:
  python experiments/hotpotqa_transfer/analyze_k_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results" / "hotpotqa_transfer"
QUESTION_TYPES = ["bridge", "comparison"]
K_VALUES = [2, 3, 5, 7, 10]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def compute_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    dfs = [r["dfs_score"] for r in rows if r["dfs_score"] is not None]
    return {
        "num_tasks": len(rows),
        "exact_match": _mean([r["em"] for r in rows]),
        "f1": _mean([r["f1"] for r in rows]),
        "mean_tokens": _mean([r["tokens"] for r in rows]),
        "mean_elapsed_s": _mean([r["elapsed_s"] for r in rows]),
        "mean_supporting_fact_recall": _mean([r["supporting_fact_recall"] for r in rows]),
        "mean_dfs_score": _mean(dfs) if dfs else None,
        "full_recall_rate": sum(1 for r in rows if r["supporting_fact_recall"] == 1.0) / len(rows),
    }


def load_k_data() -> dict[int, dict]:
    out = {}
    for k in K_VALUES:
        fname = "interactive_retrieval.json" if k == 3 else f"interactive_retrieval_k{k}.json"
        path = RESULTS_DIR / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        by_type = data["results_by_type"]
        out[k] = {
            "overall": compute_summary([r for res in by_type.values() for r in res]),
            "bridge": compute_summary(by_type.get("bridge", [])),
            "comparison": compute_summary(by_type.get("comparison", [])),
        }
    return out


def load_static_baselines() -> dict[str, dict]:
    out = {}
    for cond in ["full_context", "summary_doc", "no_context"]:
        path = RESULTS_DIR / f"{cond}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        by_type = data["results_by_type"]
        out[cond] = {
            "overall": compute_summary([r for res in by_type.values() for r in res]),
            "bridge": compute_summary(by_type.get("bridge", [])),
            "comparison": compute_summary(by_type.get("comparison", [])),
        }
    return out


def print_table(k_data: dict[int, dict], scope: str) -> None:
    print(f"\n-- {scope} --")
    hdr = f"{'K':>4} {'EM':>7} {'F1':>7} {'Tokens':>8} {'Sec':>7} {'SF-Recall':>10} {'FullRecall%':>12} {'DFS':>7}"
    print(hdr)
    print("-" * len(hdr))
    for k in K_VALUES:
        if k not in k_data:
            continue
        s = k_data[k][scope]
        dfs = s.get("mean_dfs_score")
        dfs_str = f"{dfs:.1%}" if dfs is not None else "-"
        print(
            f"{k:>4} {s.get('exact_match', 0):>7.1%} {s.get('f1', 0):>7.1%} "
            f"{s.get('mean_tokens', 0):>8.0f} {s.get('mean_elapsed_s', 0):>7.1f} "
            f"{s.get('mean_supporting_fact_recall', 0):>10.1%} "
            f"{s.get('full_recall_rate', 0):>12.1%} {dfs_str:>7}"
        )


def print_static_reference(baselines: dict[str, dict]) -> None:
    print("\n-- Static baselines (for reference — not part of the K sweep) --")
    hdr = f"{'Condition':<14} {'EM':>7} {'F1':>7} {'Tokens':>8}"
    print(hdr)
    print("-" * len(hdr))
    for cond, scopes in baselines.items():
        s = scopes["overall"]
        print(f"{cond:<14} {s.get('exact_match', 0):>7.1%} {s.get('f1', 0):>7.1%} {s.get('mean_tokens', 0):>8.0f}")


def make_chart(k_data: dict[int, dict], baselines: dict[str, dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze_k_ablation] matplotlib not available — skipping chart")
        return

    ks = [k for k in K_VALUES if k in k_data]
    em_overall = [k_data[k]["overall"].get("exact_match", 0) * 100 for k in ks]
    f1_overall = [k_data[k]["overall"].get("f1", 0) * 100 for k in ks]
    recall_overall = [k_data[k]["overall"].get("mean_supporting_fact_recall", 0) * 100 for k in ks]
    full_recall = [k_data[k]["overall"].get("full_recall_rate", 0) * 100 for k in ks]
    tokens = [k_data[k]["overall"].get("mean_tokens", 0) for k in ks]

    em_bridge = [k_data[k]["bridge"].get("exact_match", 0) * 100 for k in ks]
    em_comp = [k_data[k]["comparison"].get("exact_match", 0) * 100 for k in ks]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. EM/F1/recall vs K, with static ceilings
    axes[0].plot(ks, em_overall, "o-", label="EM", color="#2196F3")
    axes[0].plot(ks, f1_overall, "s-", label="F1", color="#4CAF50")
    axes[0].plot(ks, recall_overall, "^-", label="SF-recall", color="#FF9800")
    if "summary_doc" in baselines:
        axes[0].axhline(baselines["summary_doc"]["overall"]["f1"] * 100, color="#4CAF50", linestyle="--", alpha=0.5, label="summary_doc F1 ceiling")
    if "full_context" in baselines:
        axes[0].axhline(baselines["full_context"]["overall"]["f1"] * 100, color="#9E9E9E", linestyle=":", alpha=0.7, label="full_context F1")
    axes[0].set_xlabel("K (query budget)")
    axes[0].set_ylabel("%")
    axes[0].set_title("interactive_retrieval vs K")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # 2. EM by question type vs K
    axes[1].plot(ks, em_bridge, "o-", label="bridge (DFS-shaped)", color="#2196F3")
    axes[1].plot(ks, em_comp, "s-", label="comparison (BFS-shaped)", color="#FF9800")
    axes[1].set_xlabel("K (query budget)")
    axes[1].set_ylabel("EM (%)")
    axes[1].set_title("EM by Question Type vs K")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 3. Token cost vs K, with static costs for reference
    axes[2].plot(ks, tokens, "o-", color="#F44336", label="interactive_retrieval")
    if "summary_doc" in baselines:
        axes[2].axhline(baselines["summary_doc"]["overall"]["mean_tokens"], color="#4CAF50", linestyle="--", alpha=0.6, label="summary_doc")
    if "full_context" in baselines:
        axes[2].axhline(baselines["full_context"]["overall"]["mean_tokens"], color="#9E9E9E", linestyle=":", alpha=0.7, label="full_context")
    axes[2].set_xlabel("K (query budget)")
    axes[2].set_ylabel("Mean Tokens")
    axes[2].set_title("Token Cost vs K")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = RESULTS_DIR / "plots"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "e5_k_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: {out}")


def main():
    k_data = load_k_data()
    if not k_data:
        print("No K-ablation results found. Run k_ablation.py first.")
        return
    baselines = load_static_baselines()

    print("=" * 80)
    print("E5 K-Ablation: interactive_retrieval accuracy/cost vs query budget K")
    print("=" * 80)
    print_table(k_data, "overall")
    print_table(k_data, "bridge")
    print_table(k_data, "comparison")
    print_static_reference(baselines)
    make_chart(k_data, baselines)


if __name__ == "__main__":
    main()
