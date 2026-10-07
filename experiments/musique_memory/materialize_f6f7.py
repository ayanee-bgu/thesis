"""
Materialize F6_preference and F7_correction nodes from existing F1_raw pool data.

Reconstructs ProcedureAggregator clusters from stored F1 sessions, then calls
the same extractors used in populate.py — no oracle re-runs needed.

Usage:
  python experiments/musique_memory/materialize_f6f7.py [--no-llm]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.musique_memory.memory_store import MemoryNode, MemoryPool
from experiments.musique_memory.populate import (
    extract_f6_preference,
    extract_f7_correction,
    _llm,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model = cfg["model"]["name"]
    url = cfg["model"]["ollama_url"]
    timeout = cfg["model"]["timeout"]
    pool_dir = ROOT / cfg["output"]["memory_dir"]

    pool = MemoryPool(pool_dir)
    pool.load(formats=["F1_raw"])
    f1_nodes = pool.nodes_for("F1_raw")
    logger.info(f"Loaded {len(f1_nodes)} F1 nodes")

    # Reconstruct clusters from F1 metadata
    clusters: dict[tuple[str, int], dict] = {}
    for node in f1_nodes:
        m = node.meta
        qt = m.get("question_type", "unknown")
        hop = node.hop
        key = (qt, hop)
        if key not in clusters:
            clusters[key] = {
                "question_type": qt,
                "hop": hop,
                "sessions": [],
                "successes": 0,
                "examples": [],
            }
        c = clusters[key]
        c["sessions"].append(node.source_sessions[0] if node.source_sessions else node.id)
        c["successes"] += int(m.get("correct", False))
        c["examples"].append({
            "question": m.get("question", ""),
            "queries": m.get("queries", []),
            "correct": m.get("correct", False),
            "support_found": 0,
            "support_total": 1,
        })

    logger.info(f"Reconstructed {len(clusters)} clusters")

    # Check which clusters already have F6/F7
    pool.load(formats=["F6_preference", "F7_correction"])
    existing_f6_types = {n.meta.get("question_type") for n in pool.nodes_for("F6_preference")}
    existing_f7_types = {n.meta.get("question_type") for n in pool.nodes_for("F7_correction")}
    logger.info(f"Existing F6 types: {existing_f6_types}")
    logger.info(f"Existing F7 types: {existing_f7_types}")

    new_pool = MemoryPool(pool_dir)
    generated = {"F6": 0, "F7": 0}

    for (qt, hop), cluster in clusters.items():
        sessions = cluster["sessions"]
        examples = cluster["examples"]
        success_rate = cluster["successes"] / max(len(sessions), 1)
        logger.info(f"\n[{qt}] hop={hop} n={len(sessions)} sr={success_rate:.0%}")

        # F6: preference
        if qt not in existing_f6_types:
            if args.no_llm:
                pref = {"approach": f"[mock preference for {qt}]", "avoid": ""}
            else:
                pref = extract_f6_preference(hop, qt, examples, model, url, timeout)

            if pref and pref.get("approach"):
                node = MemoryNode.new("F6_preference", hop,
                    f"{qt}: {pref['approach']}",
                    {
                        "question_type": qt,
                        "avoid": pref.get("avoid", ""),
                        "success_rate": success_rate,
                        "num_sessions": len(sessions),
                    },
                    sessions[0],
                )
                new_pool.add(node)
                generated["F6"] += 1
                logger.info(f"  F6: {pref['approach'][:80]}")
            else:
                logger.info(f"  F6: extraction failed")
        else:
            logger.info(f"  F6: already exists, skipping")

        # F7: correction (only if there are multi-query sessions)
        if qt not in existing_f7_types:
            pivots = [e for e in examples if len(e["queries"]) > 1]
            if not pivots:
                logger.info(f"  F7: no pivot sessions, skipping")
                continue

            if args.no_llm:
                corr = {"wrong_pattern": f"[mock mistake for {qt}]", "correction": "[mock fix]"}
            else:
                corr = extract_f7_correction(hop, qt, examples, model, url, timeout)

            if corr and corr.get("correction"):
                node = MemoryNode.new("F7_correction", hop,
                    f"{qt}: {corr['correction']}",
                    {
                        "question_type": qt,
                        "wrong_pattern": corr.get("wrong_pattern", ""),
                        "num_sessions": len(sessions),
                    },
                    sessions[0],
                )
                new_pool.add(node)
                generated["F7"] += 1
                logger.info(f"  F7: {corr['correction'][:80]}")
            else:
                logger.info(f"  F7: extraction failed")
        else:
            logger.info(f"  F7: already exists, skipping")

    new_pool.flush()
    logger.info(f"\nDone. Generated: F6={generated['F6']}, F7={generated['F7']}")


if __name__ == "__main__":
    main()
