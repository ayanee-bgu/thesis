"""
Analyze the E7 hop-count x K ablation: does the accuracy-optimal query budget
K for interactive_retrieval shift with how many hops a MuSiQue question needs?

Loads interactive_retrieval_k{K}.json for each K in config's retrieval.k_values
(produced by k_ablation.py), grouped by hop bucket (hop2/hop3/hop4), and
compares against the static no_context/full_context/summary_doc ceilings from
main.py, also grouped by hop bucket.

Usage:
  python experiments/musique_transfer/analyze_k_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results" / "musique_transfer"
HOP_BUCKETS = ["hop2", "hop3", "hop4"]


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


def _load_by_hop(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    by_type = data["results_by_type"]
    out = {"overall": compute_summary([r for res in by_type.values() for r in res])}
    for hop in HOP_BUCKETS:
        out[hop] = compute_summary(by_type.get(hop, []))
    return out


def load_k_data(k_values: list[int]) -> dict[int, dict]:
    out = {}
    for k in k_values:
        data = _load_by_hop(RESULTS_DIR / f"interactive_retrieval_k{k}.json")
        if data:
            out[k] = data
    return out


def load_static_baselines() -> dict[str, dict]:
    out = {}
    for cond in ["no_context", "full_context", "summary_doc"]:
        data = _load_by_hop(RESULTS_DIR / f"{cond}.json")
        if data:
            out[cond] = data
    return out


def infer_k_values() -> list[int]:
    ks = []
    for p in RESULTS_DIR.glob("interactive_retrieval_k*.json"):
        try:
            ks.append(int(p.stem.replace("interactive_retrieval_k", "")))
        except ValueError:
            continue
    return sorted(ks)


def print_grid(k_data: dict[int, dict], metric: str, label: str) -> None:
    ks = sorted(k_data.keys())
    print(f"\n-- {label}: hop bucket x K --")
    hdr = f"{'Hop bucket':<12}" + "".join(f"{'K='+str(k):>10}" for k in ks)
    print(hdr)
    print("-" * len(hdr))
    for hop in HOP_BUCKETS:
        row = f"{hop:<12}"
        for k in ks:
            val = k_data[k].get(hop, {}).get(metric)
            row += f"{(f'{val:.1%}' if val is not None else '-'):>10}"
        print(row)


def print_static_reference(baselines: dict[str, dict]) -> None:
    print("\n-- Static baselines by hop bucket (for reference — not part of the K sweep) --")
    hdr = f"{'Condition':<14}" + "".join(f"{h:>12}" for h in HOP_BUCKETS) + f"{'overall':>12}"
    print(hdr)
    print("-" * len(hdr))
    for cond, scopes in baselines.items():
        row = f"{cond:<14}"
        for hop in HOP_BUCKETS + ["overall"]:
            f1 = scopes.get(hop, {}).get("f1")
            row += f"{(f'{f1:.1%}' if f1 is not None else '-'):>12}"
        print(row)


def make_chart(k_data: dict[int, dict], baselines: dict[str, dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze_k_ablation] matplotlib not available — skipping chart")
        return

    ks = sorted(k_data.keys())
    colors = {"hop2": "#2196F3", "hop3": "#4CAF50", "hop4": "#F44336"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 1. F1 vs K, one line per hop bucket
    for hop in HOP_BUCKETS:
        f1s = [k_data[k].get(hop, {}).get("f1", 0) * 100 for k in ks]
        axes[0].plot(ks, f1s, "o-", label=hop, color=colors.get(hop))
    if "summary_doc" in baselines:
        axes[0].axhline(baselines["summary_doc"]["overall"]["f1"] * 100, color="gray", linestyle="--", alpha=0.6, label="summary_doc overall")
    axes[0].set_xlabel("K (query budget)")
    axes[0].set_ylabel("F1 (%)")
    axes[0].set_title("interactive_retrieval F1 vs K, by hop count")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 2. Full-recall rate vs K, one line per hop bucket
    for hop in HOP_BUCKETS:
        fr = [k_data[k].get(hop, {}).get("full_recall_rate", 0) * 100 for k in ks]
        axes[1].plot(ks, fr, "s-", label=hop, color=colors.get(hop))
    axes[1].set_xlabel("K (query budget)")
    axes[1].set_ylabel("Full-recall rate (%)")
    axes[1].set_title("Retrieval Success vs K, by hop count")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 3. Token cost vs K, one line per hop bucket
    for hop in HOP_BUCKETS:
        toks = [k_data[k].get(hop, {}).get("mean_tokens", 0) for k in ks]
        axes[2].plot(ks, toks, "^-", label=hop, color=colors.get(hop))
    axes[2].set_xlabel("K (query budget)")
    axes[2].set_ylabel("Mean Tokens")
    axes[2].set_title("Token Cost vs K, by hop count")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = RESULTS_DIR / "plots"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "e7_hopcount_k_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: {out}")


def main():
    k_values = infer_k_values()
    if not k_values:
        print("No K-ablation results found. Run k_ablation.py first.")
        return
    k_data = load_k_data(k_values)
    baselines = load_static_baselines()

    print("=" * 90)
    print("E7 Hop-Count x K Ablation: interactive_retrieval accuracy/cost vs (hop count, K)")
    print("=" * 90)
    print_grid(k_data, "exact_match", "EM")
    print_grid(k_data, "f1", "F1")
    print_grid(k_data, "full_recall_rate", "Full-Recall Rate")
    print_static_reference(baselines)
    make_chart(k_data, baselines)


if __name__ == "__main__":
    main()
