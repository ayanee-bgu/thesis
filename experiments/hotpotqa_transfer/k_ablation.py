"""
E5 K-ablation: does a bigger interactive_retrieval query budget close the
retrieval-recall gap identified in REPORT.md (accuracy craters when the K=3
budget misses a gold passage; is competitive when it doesn't)?

Reruns ONLY the interactive_retrieval mechanism at several K values, on the
same HotpotQA questions used in the main E5 run (first num_per_type of each
type, same seed/slice — deterministic). K=3 itself is not rerun; the existing
results/hotpotqa_transfer/interactive_retrieval.json from the main run IS the
K=3 data point and is picked up directly by analyze_k_ablation.py.

Each K value is saved to its own file (interactive_retrieval_k{K}.json) using
the same incremental per-type save + auto-resume machinery as main.py, so an
interruption partway through one K value (or partway through the whole sweep)
does not lose completed work.

Usage:
  python experiments/hotpotqa_transfer/k_ablation.py [options]

  --k-values "2,5,7,10"   Comma-separated K values to run (default: 2,5,7,10)
  --num-tasks N           Override num_per_type (default: from config.yaml)
  --no-llm                Structural smoke test, no real Ollama calls
  --fresh                 Ignore cached per-K results, recompute everything
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
    load_hotpotqa,
    load_existing_condition_results,
    run_condition,
    save_condition_results,
    warmup_model,
)


def main():
    parser = argparse.ArgumentParser(description="E5 K-ablation for interactive_retrieval")
    parser.add_argument("--k-values", default="2,5,7,10",
                        help="Comma-separated K values to run (default: 2,5,7,10). K=3 reuses the main E5 run.")
    parser.add_argument("--num-tasks", type=int, default=None, help="Override num_per_type")
    parser.add_argument("--no-llm", action="store_true", help="Bypass Ollama for structural testing")
    parser.add_argument("--fresh", action="store_true", help="Ignore cached per-K results, recompute everything")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    cfg = load_config()
    if args.num_tasks:
        cfg["experiment"]["num_per_type"] = args.num_tasks

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                         format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    results_dir = ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading HotpotQA dataset...")
    examples_by_type = load_hotpotqa(cfg, logger)

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
