"""
E1b: Ablation of four untested levers on top of E1's `graph_session` condition —
each isolates one item from the memory-controller's own "what else can we do"
list rather than bundling them all at once:

  graph_session_baseline  : control, identical behavior to E1's graph_session
  graph_session_embed     : + real sentence-embedding dedup/retrieval (TF-IDF was
                             the only tested option in E1; embed_fn existed but
                             was never wired to an actual model)
  graph_session_grounded  : + task_type/keyword prefilter before similarity scoring,
                             + reinforce_count-weighted ranking (a node independently
                             rediscovered across sessions outranks one seen once)
  graph_session_full      : all of the above + a periodic graph-wide recluster()
                             pass that catches near-duplicates the incremental
                             per-trajectory dedup missed

Same 2-session protocol as E1 (memory_controller/main.py): session 1 builds
memory, session 2 uses it — the session 1 -> session 2 success-rate delta is
the key metric, run on TravelPlanner (sole-planning mode).

Usage:
  python experiments/memory_controller_v2/main.py [options]

  --mock            Use MockEnv instead of TravelPlanner (fast structural test)
  --no-llm          Bypass Ollama entirely (structural harness test)
  --condition COND  Run a single condition (default: all)
  --num-tasks N     Override tasks_per_session * num_sessions
  --log-level LEVEL
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
MC_DIR = ROOT / "experiments" / "memory_controller"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MC_DIR))  # reuse travelplanner_env / env_adapter / prompt+agent helpers

from src.memory.graph_store import ProcedureGraph
from src.memory.consolidator import SessionConsolidator, ConsolidationLog
from src.memory.context_injector import ContextInjector
from src.memory.embeddings import try_get_embed_fn

import main as base_main  # experiments/memory_controller/main.py — reuse agent/env glue

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Per-condition session runner (mirrors memory_controller/main.py::_run_session,
# trimmed to graph-only conditions and threading task_type through)
# ---------------------------------------------------------------------------

def _run_session(
    session_num: int,
    task_ids: list[str],
    cond_name: str,
    env,
    is_travel: bool,
    graph: ProcedureGraph,
    injector: ContextInjector,
    model: str,
    ollama_url: str,
    timeout: int,
    logger: logging.Logger,
    no_llm: bool,
) -> tuple[list[dict], list[str], list[str]]:
    task_results, session_trajectories, session_task_types = [], [], []

    for task_id in task_ids:
        task_desc, obs = env.reset(task_id)
        total_tokens, retrieval_calls = 0, 0
        t_start = time.time()

        memory_block = injector.inject(task_desc, task_type=task_id)
        if memory_block:
            retrieval_calls += 1

        if no_llm:
            action = task_desc
            obs_out, success_final, _ = env.step(action)
            traj_text = f"TASK: {task_desc}\nACTION: {action}\nOBS: {obs_out}"
        elif is_travel:
            try:
                plan_text, tokens = base_main.run_travel_plan(
                    task_desc, obs, memory_block, model, ollama_url, timeout
                )
            except Exception as e:
                logger.warning(f"  [{cond_name}] Travel agent error: {e}")
                plan_text, tokens = "", 0
            total_tokens += tokens
            feedback, success_final, _ = env.step(plan_text)
            traj_text = f"QUERY: {task_desc}\n\nREFERENCE (excerpt):\n{obs[:2000]}\n\nPLAN:\n{plan_text}\n\nEVAL: {feedback}"
        else:
            history: list[dict] = []
            for _ in range(20):
                try:
                    action, tokens = base_main.run_agent_step(
                        task_desc, obs, history, memory_block, model, ollama_url, timeout
                    )
                except Exception as e:
                    logger.warning(f"  [{cond_name}] Agent error: {e}")
                    break
                total_tokens += tokens
                obs, success, done = env.step(action)
                history.append({"action": action, "obs": obs, "success": success})
                if done:
                    break
            success_final = any(h["success"] for h in history)
            traj_text = base_main.trajectory_to_text(task_desc, history)

        session_trajectories.append(traj_text)
        session_task_types.append(task_id)
        result = {
            "task_id": task_id,
            "session": session_num,
            "success": success_final,
            "tokens": total_tokens,
            "retrieval_calls": retrieval_calls,
            "elapsed_s": round(time.time() - t_start, 2),
        }
        task_results.append(result)
        logger.info(
            f"[{cond_name}] s{session_num} {task_id}: success={success_final}, "
            f"tokens={total_tokens}, retrieved={'yes' if retrieval_calls else 'no'}"
        )

    return task_results, session_trajectories, session_task_types


def run_condition(condition: dict, env, task_ids: list[str], cfg: dict, results_dir: Path,
                   logger: logging.Logger, no_llm: bool = False) -> dict:
    cond_name = condition["name"]
    model = cfg["model"]["name"]
    ollama_url = cfg["model"]["ollama_url"]
    timeout = cfg["model"]["timeout"]
    num_sessions = cfg["experiment"]["num_sessions"]
    tasks_per_session = cfg["experiment"].get("tasks_per_session", len(task_ids) // num_sessions)
    graph_path = ROOT / cfg["memory"]["graph_path"]

    embed_fn = None
    if condition["use_embed"] and not no_llm:
        embed_fn = try_get_embed_fn(cfg["memory"]["embed_model"])
        if embed_fn is None:
            logger.warning(
                f"[{cond_name}] sentence-transformers not installed — falling back to TF-IDF "
                f"(run `pip install sentence-transformers` to actually test this lever)"
            )

    graph = ProcedureGraph()
    consolidator = SessionConsolidator(
        graph=graph, model=cfg["model"].get("distil_model", model),
        dedup_threshold=cfg["memory"]["dedup_threshold"], embed_fn=embed_fn,
    )
    injector = ContextInjector(
        graph=graph, k_hops=cfg["memory"]["k_hops"], top_seeds=cfg["memory"]["top_seeds"],
        token_budget=cfg["memory"]["token_budget"], embed_fn=embed_fn,
        use_keyword_prefilter=condition["use_keyword_prefilter"],
        use_reinforcement_weighting=condition["use_reinforcement"],
    )

    from travelplanner_env import TravelPlannerEnv
    is_travel = isinstance(env, TravelPlannerEnv)

    all_task_results, session_summaries = [], []

    for session_num in range(1, num_sessions + 1):
        start = (session_num - 1) * tasks_per_session
        session_task_ids = task_ids[start: start + tasks_per_session]
        if not session_task_ids:
            break
        logger.info(f"\n[{cond_name}] -- Session {session_num}/{num_sessions} ({len(session_task_ids)} tasks) --")

        task_results, trajectories, task_types = _run_session(
            session_num, session_task_ids, cond_name, env, is_travel, graph, injector,
            model, ollama_url, timeout, logger, no_llm,
        )
        all_task_results.extend(task_results)

        session_sr = sum(r["success"] for r in task_results) / len(task_results) if task_results else 0
        session_summaries.append({
            "session": session_num,
            "num_tasks": len(task_results),
            "success_rate": session_sr,
            "mean_tokens": sum(r["tokens"] for r in task_results) / len(task_results) if task_results else 0,
            "retrieval_calls": sum(r["retrieval_calls"] for r in task_results),
            "graph_nodes_after": None,  # filled in after consolidation below
        })
        logger.info(f"[{cond_name}] Session {session_num} success rate: {session_sr:.1%}")

        # Session-end consolidation (this is the only update_policy in this ablation —
        # E1 already established graph_session >= graph_pertraj, so every condition
        # here inherits session_end and isolates a different downstream lever instead)
        try:
            cond_graph_path = graph_path.parent / f"procedure_graph_v2_{cond_name}.json"
            if no_llm:
                log = ConsolidationLog(session_id=f"session_{session_num}", source_agent=cond_name)
                for traj, ttype in zip(trajectories, task_types):
                    graph.add_procedure(traj[:500], level="script", source_agent=cond_name,
                                         session_id=f"session_{session_num}", task_type=ttype)
                    log.nodes_added += 1
                graph.save(cond_graph_path)
            else:
                log = consolidator.consolidate(
                    trajectories, session_id=f"session_{session_num}", source_agent=cond_name,
                    graph_path=cond_graph_path, task_types=task_types,
                )
                if condition["use_recluster"]:
                    recluster_log = consolidator.recluster()
                    logger.info(f"[{cond_name}] Recluster: merged {recluster_log.nodes_merged} additional nodes")
                    graph.save(cond_graph_path)
            session_summaries[-1]["graph_nodes_after"] = len(graph)
            logger.info(
                f"[{cond_name}] Consolidated session {session_num}: +{log.nodes_added} nodes, "
                f"{log.nodes_merged} merged -> graph now has {len(graph)} nodes"
            )
        except Exception as e:
            logger.warning(f"[{cond_name}] Consolidation error: {e}")

    delta = None
    if len(session_summaries) >= 2:
        delta = session_summaries[1]["success_rate"] - session_summaries[0]["success_rate"]

    out = {
        "condition": cond_name,
        "sessions": session_summaries,
        "tasks": all_task_results,
        "summary": {
            "session_1_success_rate": session_summaries[0]["success_rate"] if session_summaries else None,
            "session_2_success_rate": session_summaries[1]["success_rate"] if len(session_summaries) > 1 else None,
            "delta_s1_to_s2": delta,
            "mean_tokens_s2": session_summaries[1]["mean_tokens"] if len(session_summaries) > 1 else None,
            "final_graph_nodes": len(graph),
        },
    }
    out_path = results_dir / f"{cond_name}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    s1 = session_summaries[0]["success_rate"] if session_summaries else 0
    s2 = session_summaries[1]["success_rate"] if len(session_summaries) > 1 else 0
    logger.info(f"[{cond_name}] Done. S1={s1:.1%} -> S2={s2:.1%}" +
                (f" (D={delta:+.1%})" if delta is not None else ""))
    return out


def main():
    parser = argparse.ArgumentParser(description="E1b: memory-controller lever ablation")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--travelplanner", action="store_true", default=True)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = load_config()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                         format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    num_sessions = cfg["experiment"]["num_sessions"]
    tasks_per_session = cfg["experiment"]["tasks_per_session"]
    num_tasks = args.num_tasks or (num_sessions * tasks_per_session)

    if args.mock:
        from env_adapter import MockEnv
        env = MockEnv(seed=cfg["experiment"]["seed"])
        logger.info("Using MockEnv")
    else:
        from travelplanner_env import TravelPlannerEnv
        env = TravelPlannerEnv(splits=["validation"], subset=num_tasks)
        logger.info(f"Using TravelPlannerEnv ({num_tasks} tasks)")

    task_ids = env.task_ids()[:num_tasks]
    if not task_ids:
        logger.error("No task IDs available")
        sys.exit(1)

    results_dir = ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    conditions = cfg["conditions"]
    if args.condition:
        conditions = [c for c in conditions if c["name"] == args.condition]
        if not conditions:
            logger.error(f"Condition '{args.condition}' not found in config")
            sys.exit(1)

    all_results = []
    for condition in conditions:
        logger.info(f"\n{'='*60}\nRunning condition: {condition['name']}\n{'='*60}")
        all_results.append(run_condition(condition, env, task_ids, cfg, results_dir, logger, no_llm=args.no_llm))

    summary_path = results_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"conditions": [r["summary"] | {"name": r["condition"]} for r in all_results]}, indent=2),
        encoding="utf-8",
    )
    logger.info(f"\nResults written to {results_dir}")

    print("\n=== E1b RESULTS SUMMARY (2-session) ===")
    print(f"{'Condition':<26} {'S1 Success':>12} {'S2 Success':>12} {'D S1->S2':>10} {'S2 Tokens':>11} {'Graph Nodes':>12}")
    print("-" * 85)
    for r in all_results:
        s = r["summary"]
        s1, s2 = s.get("session_1_success_rate") or 0, s.get("session_2_success_rate") or 0
        d = s.get("delta_s1_to_s2")
        tok = s.get("mean_tokens_s2") or 0
        gn = s.get("final_graph_nodes") or "-"
        delta_str = f"{d:+.1%}" if d is not None else "  n/a"
        print(f"{r['condition']:<26} {s1:>11.1%} {s2:>12.1%} {delta_str:>10} {tok:>11.0f} {str(gn):>12}")


if __name__ == "__main__":
    main()
