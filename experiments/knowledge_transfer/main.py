"""
E2: Cross-agent knowledge transfer via shared procedure graph.

Protocol:
  Phase 1 (Teacher): Run `teacher.num_sessions` sessions of tasks.
                     After each session, consolidate trajectories into a shared graph.
  Phase 2 (Transfer): Export graph (or flat summary) to the student.
  Phase 3 (Student):  Run tasks using the transferred memory.
                     Measure sessions-to-target-success-rate.

Conditions:
  b_from_scratch  : student starts cold (no memory)
  b_from_a_flat   : student initialised from teacher's flat procedure dump
  b_from_a_graph  : student initialised from teacher's graph (proposed)

Usage:
    python experiments/knowledge_transfer/main.py [--mock] [--condition COND]
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
sys.path.insert(0, str(ROOT))

from src.memory.graph_store import ProcedureGraph
from src.memory.consolidator import SessionConsolidator
from src.memory.context_injector import ContextInjector

# Reuse helpers from E1
sys.path.insert(0, str(Path(__file__).parent.parent / "memory_controller"))
from main import (  # noqa: E402
    FlatMemoryStore,
    run_agent_step,
    trajectory_to_text,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Teacher phase
# ---------------------------------------------------------------------------

def run_teacher_sessions(env, cfg: dict, logger: logging.Logger) -> tuple[ProcedureGraph, FlatMemoryStore]:
    """
    Run teacher for num_sessions × tasks_per_session tasks.
    Returns (teacher_graph, teacher_flat_store) both populated.
    """
    t_cfg = cfg["teacher"]
    m_cfg = cfg["memory"]
    exp_cfg = cfg["experiment"]

    graph = ProcedureGraph()
    flat_store = FlatMemoryStore()
    consolidator = SessionConsolidator(
        graph=graph,
        model=m_cfg.get("distil_model", t_cfg["model"]),
        dedup_threshold=m_cfg["dedup_threshold"],
    )
    injector = ContextInjector(
        graph=graph,
        k_hops=m_cfg["k_hops"],
        top_seeds=m_cfg["top_seeds"],
        token_budget=m_cfg["token_budget"],
    )

    task_ids = env.task_ids()
    idx = 0

    for session_num in range(t_cfg["num_sessions"]):
        session_trajs: list[str] = []
        session_tasks = task_ids[idx: idx + t_cfg["tasks_per_session"]]
        idx = (idx + t_cfg["tasks_per_session"]) % len(task_ids)

        logger.info(f"\n[Teacher] Session {session_num + 1}/{t_cfg['num_sessions']}: {session_tasks}")

        for task_id in session_tasks:
            task_desc, obs = env.reset(task_id)
            history: list[dict] = []

            for _ in range(exp_cfg["max_steps"]):
                memory_block = injector.inject(task_desc)
                try:
                    action, _ = run_agent_step(
                        task_desc, obs, history, memory_block,
                        t_cfg["model"], t_cfg["ollama_url"], t_cfg["timeout"]
                    )
                except Exception as e:
                    logger.warning(f"  [Teacher] Agent error: {e}")
                    break

                obs, success, done = env.step(action)
                history.append({"action": action, "obs": obs, "success": success})
                if done:
                    break

            traj = trajectory_to_text(task_desc, history)
            session_trajs.append(traj)

            if any(h["success"] for h in history):
                flat_store.add(traj, task_type=task_id)

        # Session-end consolidation for graph
        try:
            log = consolidator.consolidate(
                session_trajs,
                session_id=f"teacher_session_{session_num}",
                source_agent="teacher",
                graph_path=None,
            )
            logger.info(f"  [Teacher] Consolidated: +{log.nodes_added} nodes, {len(log.conflicts_detected)} conflicts")
        except Exception as e:
            logger.warning(f"  [Teacher] Consolidation error: {e}")

    graph_path = ROOT / cfg["memory"]["graph_path"]
    graph.save(graph_path)
    logger.info(f"[Teacher] Done. Graph: {len(graph)} nodes, Flat: {len(flat_store)} entries")
    return graph, flat_store


# ---------------------------------------------------------------------------
# Student phase
# ---------------------------------------------------------------------------

def run_student(
    condition: dict,
    env,
    teacher_graph: ProcedureGraph,
    teacher_flat: FlatMemoryStore,
    cfg: dict,
    results_dir: Path,
    logger: logging.Logger,
) -> dict:
    cond_name = condition["name"]
    student_init = condition["student_init"]
    s_cfg = cfg["student"]
    m_cfg = cfg["memory"]
    exp_cfg = cfg["experiment"]

    # Initialise student memory from transfer
    injector: ContextInjector | None = None
    flat_store: FlatMemoryStore | None = None

    if student_init == "graph":
        injector = ContextInjector(
            graph=teacher_graph,
            k_hops=m_cfg["k_hops"],
            top_seeds=m_cfg["top_seeds"],
            token_budget=m_cfg["token_budget"],
        )
    elif student_init == "flat_export":
        flat_store = teacher_flat
    # else: no memory (from_scratch)

    task_ids = env.task_ids()
    target_sr = s_cfg["target_success_rate"]
    sessions_to_target: int | None = None
    session_results = []

    for session_num in range(s_cfg["max_student_sessions"]):
        session_tasks = task_ids[session_num % len(task_ids): (session_num % len(task_ids)) + 4]
        session_successes = []

        for task_id in session_tasks:
            task_desc, obs = env.reset(task_id)
            history: list[dict] = []

            for _ in range(exp_cfg["max_steps"]):
                memory_block = ""
                if injector:
                    memory_block = injector.inject(task_desc)
                elif flat_store:
                    memory_block = flat_store.retrieve(task_desc)

                try:
                    action, tokens = run_agent_step(
                        task_desc, obs, history, memory_block,
                        s_cfg["model"], s_cfg["ollama_url"], s_cfg["timeout"]
                    )
                except Exception as e:
                    logger.warning(f"  [Student/{cond_name}] Error: {e}")
                    break

                obs, success, done = env.step(action)
                history.append({"action": action, "obs": obs, "success": success})
                if done:
                    break

            task_success = any(h["success"] for h in history)
            session_successes.append(task_success)

        session_sr = sum(session_successes) / len(session_successes) if session_successes else 0
        session_results.append({"session": session_num + 1, "success_rate": session_sr})
        logger.info(f"  [Student/{cond_name}] Session {session_num + 1}: success_rate={session_sr:.2%}")

        if session_sr >= target_sr and sessions_to_target is None:
            sessions_to_target = session_num + 1
            logger.info(f"  [Student/{cond_name}] Reached target {target_sr:.0%} at session {sessions_to_target}")

    peak_sr = max(r["success_rate"] for r in session_results) if session_results else 0

    result = {
        "condition": cond_name,
        "student_init": student_init,
        "sessions_to_target": sessions_to_target,
        "target_success_rate": target_sr,
        "peak_success_rate": peak_sr,
        "session_results": session_results,
    }

    out_path = results_dir / f"{cond_name}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E2: Cross-agent knowledge transfer")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--condition", default=None)
    parser.add_argument("--skip-teacher", action="store_true", help="Load existing graph, skip teacher phase")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = load_config()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    if args.mock:
        sys.path.insert(0, str(Path(__file__).parent.parent / "memory_controller"))
        from env_adapter import MockEnv
        env = MockEnv(seed=cfg["experiment"]["seed"])
    else:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "memory_controller"))
            from env_adapter import ALFWorldAdapter
            alfworld_cfg = Path(__file__).parent.parent / "memory_controller" / "alfworld_config.yaml"
            env = ALFWorldAdapter(str(alfworld_cfg), split=cfg["experiment"]["task_split"])
        except ImportError:
            logger.warning("ALFWorld not available, falling back to MockEnv")
            from env_adapter import MockEnv
            env = MockEnv(seed=cfg["experiment"]["seed"])

    results_dir = ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # Teacher phase
    graph_path = ROOT / cfg["memory"]["graph_path"]
    if args.skip_teacher and graph_path.exists():
        teacher_graph = ProcedureGraph.load(graph_path)
        teacher_flat = FlatMemoryStore()
        logger.info(f"[Teacher] Loaded existing graph: {len(teacher_graph)} nodes")
    else:
        teacher_graph, teacher_flat = run_teacher_sessions(env, cfg, logger)

    # Student conditions
    conditions = cfg["conditions"]
    if args.condition:
        conditions = [c for c in conditions if c["name"] == args.condition]

    all_results = []
    for condition in conditions:
        logger.info(f"\n{'='*60}\nStudent condition: {condition['name']}\n{'='*60}")
        result = run_student(condition, env, teacher_graph, teacher_flat, cfg, results_dir, logger)
        all_results.append(result)

    # Print comparison
    print("\n=== E2: CROSS-AGENT TRANSFER RESULTS ===")
    print(f"{'Condition':<20} {'Sessions→Target':>16} {'Peak Success':>14}")
    print("-" * 54)
    for r in all_results:
        stt = r["sessions_to_target"] or "never"
        print(f"{r['condition']:<20} {str(stt):>16} {r['peak_success_rate']:>13.1%}")


if __name__ == "__main__":
    main()
