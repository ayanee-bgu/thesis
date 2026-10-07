"""
E7 K-ablation: does the accuracy-optimal query budget K for interactive_retrieval
scale with how many hops a MuSiQue question actually requires?

E5 found K=3 near-optimal on HotpotQA (all bridge/comparison questions are
effectively 2-hop) and that raising K further made things WORSE (context
dilution outweighs recall gains). This reruns interactive_retrieval at
K in {3, 5, 7} (from config.yaml's retrieval.k_values) across MuSiQue's
2/3/4-hop buckets, to see whether the "K=3 is near-optimal, more hurts"
finding is a HotpotQA-specific artifact of its fixed 2-hop structure, or a
more general context-dilution ceiling independent of hop count.

Each K value is saved to its own file (interactive_retrieval_k{K}.json,
containing all 3 hop buckets), using the same incremental per-hop-bucket
save + auto-resume machinery as main.py.

Usage:
  python experiments/musique_transfer/k_ablation.py [options]

  --k-values "3,5,7"   Comma-separated K values to run (default: from config.yaml)
  --num-tasks N        Override num_per_hop
  --no-llm             Structural smoke test, no real Ollama calls
  --fresh              Ignore cached per-K results, recompute everything
  --log-level LEVEL
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import (  # noqa: E402
    ROOT,
    load_config,
    load_musique,
    load_existing_condition_results,
    run_condition,
    save_condition_results,
    warmup_model,
)


def main():
    parser = argparse.ArgumentParser(description="E7 K-ablation for interactive_retrieval, by hop count")
    parser.add_argument("--k-values", default=None,
                        help="Comma-separated K values to run (default: retrieval.k_values from config.yaml)")
    parser.add_argument("--num-tasks", type=int, default=None, help="Override num_per_hop")
    parser.add_argument("--no-llm", action="store_true", help="Bypass Ollama for structural testing")
    parser.add_argument("--fresh", action="store_true", help="Ignore cached per-K results, recompute everything")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = load_config()
    if args.num_tasks:
        cfg["experiment"]["num_per_hop"] = args.num_tasks

    k_values = [int(k.strip()) for k in args.k_values.split(",")] if args.k_values else cfg["retrieval"]["k_values"]

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                         format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    results_dir = ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MuSiQue dataset...")
    examples_by_type = load_musique(cfg, logger)

    if not args.no_llm:
        warmup_model(cfg["model"]["name"], cfg["model"]["ollama_url"], cfg["model"]["timeout"], logger)

    for k in k_values:
        label = f"interactive_retrieval_k{k}"
        logger.info(f"\n{'='*60}\nK-ablation: K={k}  (label={label})\n{'='*60}")

        run_cfg = dict(cfg)
        run_cfg["retrieval"] = {**cfg["retrieval"], "max_queries": k}

        results_by_type = load_existing_condition_results(label, results_dir) if not args.fresh else {}
        for qtype, examples in examples_by_type.items():
            cached = results_by_type.get(qtype)
            if cached and len(cached) == len(examples):
                logger.info(f"[{label}/{qtype}] resuming — {len(cached)} cached results found, skipping recompute")
                continue
            results_by_type[qtype] = run_condition(
                "interactive_retrieval", examples, qtype, run_cfg, logger, no_llm=args.no_llm, label=label,
            )
            save_condition_results(label, results_by_type, results_dir, run_cfg)
        save_condition_results(label, results_by_type, results_dir, run_cfg)

    logger.info(f"\nK-ablation complete. Results in {results_dir} (interactive_retrieval_k{{K}}.json per K).")


if __name__ == "__main__":
    main()
