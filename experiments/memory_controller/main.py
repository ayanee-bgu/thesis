"""
E1: Graph Memory Controller vs. MemP flat store — 4-condition experiment.

Conditions:
  no_memory      : agent receives no procedural memory (pure baseline)
  memp_flat      : TF-IDF flat store, updated per trajectory (MemP baseline)
  graph_pertraj  : graph store, updated per trajectory
  graph_session  : graph store, updated at session end (proposed)

Usage:
  python experiments/memory_controller/main.py [--mock] [--condition COND] [--num-tasks N]

  --mock        Use MockEnv instead of ALFWorld (for testing without ALFWorld setup)
  --condition   Run only one condition (default: all)
  --num-tasks   Override num_tasks from config
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from pathlib import Path

import requests
import yaml

# ---------------------------------------------------------------------------
# Path setup so experiment can be run from the repo root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # allows `import env_adapter` from this package

from src.memory.graph_store import ProcedureGraph, DEFAULT_GRAPH_PATH
from src.memory.consolidator import SessionConsolidator
from src.memory.context_injector import ContextInjector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if overrides:
        for k, v in overrides.items():
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Flat TF-IDF memory store (MemP baseline)
# ---------------------------------------------------------------------------

class FlatMemoryStore:
    """Simple key-value store with TF-IDF retrieval — mimics MemP's flat repository."""

    def __init__(self):
        self._entries: list[dict] = []  # {content, task_type, tokens}

    def add(self, content: str, task_type: str = ""):
        tokens = re.findall(r"[a-z0-9]+", content.lower())
        self._entries.append({"content": content, "task_type": task_type, "tokens": tokens})

    def retrieve(self, query: str, top_k: int = 3) -> str:
        if not self._entries:
            return ""
        q_tokens = re.findall(r"[a-z0-9]+", query.lower())
        if not q_tokens:
            return ""

        all_toks = [e["tokens"] for e in self._entries]
        vocab = set(q_tokens)
        N = len(all_toks) + 1

        def idf(t):
            df = sum(1 for toks in all_toks if t in toks)
            return math.log((N + 1) / (df + 1)) + 1

        def tfidf(toks):
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            n = len(toks) or 1
            return {t: (tf.get(t, 0) / n) * idf(t) for t in vocab}

        def cos(a, b):
            dot = sum(a.get(t, 0) * b.get(t, 0) for t in b)
            na = math.sqrt(sum(v * v for v in a.values())) or 1
            nb = math.sqrt(sum(v * v for v in b.values())) or 1
            return dot / (na * nb)

        q_vec = tfidf(q_tokens)
        scored = [(e, cos(q_vec, tfidf(e["tokens"]))) for e in self._entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        parts = [f"- {e['content']}" for e, _ in scored[:top_k]]
        return "[PROCEDURAL MEMORY]\n" + "\n".join(parts) + "\n[END PROCEDURAL MEMORY]\n"

    def __len__(self):
        return len(self._entries)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

AGENT_SYSTEM_HOUSEHOLD = """You are a household task agent. Complete the task step by step.
{memory_block}
At each step, output exactly one action on a line starting with "ACTION: ".
If the task is complete, output "ACTION: done".
Be concise. One action per response."""

AGENT_SYSTEM = AGENT_SYSTEM_HOUSEHOLD  # default; overridden to TRAVEL for TravelPlanner tasks

TRAVEL_PLANNER_SYSTEM = """You are a proficient travel planner. Based on the provided information and query, give me a detailed plan including flight numbers, restaurant names, and accommodation names. All information must come from the provided reference data.

{memory_block}

**Format (follow exactly):**
Day 1:
Current City: from [Origin] to [Destination]
Transportation: Flight Number: FXXXXXXX, from [Origin] to [Destination], Departure Time: HH:MM, Arrival Time: HH:MM
Breakfast: [Restaurant Name], [City]
Attraction: [Attraction Name], [City]
Lunch: [Restaurant Name], [City]
Dinner: [Restaurant Name], [City]
Accommodation: [Hotel Name], [City]

Day 2:
Current City: [City]
Transportation: -
...

Use '-' when a field is not applicable (e.g., no accommodation on the return day).
Vary restaurants and attractions — do not repeat them across days."""


def run_agent_step(
    task_desc: str,
    observation: str,
    history: list[dict],
    memory_block: str,
    model: str,
    ollama_url: str,
    timeout: int,
) -> tuple[str, int]:
    """Call the LLM for one step. Returns (action_text, tokens_used)."""
    system = AGENT_SYSTEM.format(memory_block=memory_block)
    messages = [{"role": "system", "content": system}]

    # Prepend task and compressed history
    task_msg = f"TASK: {task_desc}\n\nCurrent observation: {observation}"
    if history:
        hist_lines = [f"Step {i+1}: {h['action']} → {h['obs'][:100]}" for i, h in enumerate(history[-5:])]
        task_msg = f"TASK: {task_desc}\n\nHistory:\n" + "\n".join(hist_lines) + f"\n\nCurrent observation: {observation}"
    messages.append({"role": "user", "content": task_msg})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_gpu": 0},
    }
    resp = requests.post(ollama_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    content = data["message"]["content"]
    tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)

    # Extract action
    match = re.search(r"ACTION:\s*(.+)", content, re.IGNORECASE)
    action = match.group(1).strip() if match else content.strip().split("\n")[0]
    return action, tokens


def run_travel_plan(
    query: str,
    reference_text: str,
    memory_block: str,
    model: str,
    ollama_url: str,
    timeout: int,
) -> tuple[str, int]:
    """Single-turn TravelPlanner call. Returns (plan_text, tokens_used)."""
    system = TRAVEL_PLANNER_SYSTEM.format(memory_block=memory_block)
    user_msg = f"Given information:\n{reference_text[:8000]}\n\nQuery: {query}\n\nTravel Plan:"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_gpu": 0},
    }
    resp = requests.post(ollama_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    plan = data["message"]["content"].strip()
    tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
    return plan, tokens


def trajectory_to_text(task_desc: str, history: list[dict]) -> str:
    lines = [f"TASK: {task_desc}"]
    for i, h in enumerate(history):
        lines.append(f"Step {i+1}: {h['action']}")
        lines.append(f"  Observation: {h['obs']}")
        if h.get("success"):
            lines.append("  [TASK COMPLETED]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def _run_session(
    session_num: int,
    task_ids: list[str],
    cond_name: str,
    memory_type: str | None,
    update_policy: str | None,
    env,
    is_travel: bool,
    graph,
    flat_store,
    consolidator,
    injector,
    model: str,
    ollama_url: str,
    timeout: int,
    max_steps: int,
    graph_path: Path,
    logger: logging.Logger,
    no_llm: bool = False,
) -> tuple[list[dict], list[str]]:
    """Run one session of tasks. Returns (task_results, trajectory_texts)."""
    task_results = []
    session_trajectories = []

    for task_id in task_ids:
        task_desc, obs = env.reset(task_id)
        total_tokens = 0
        retrieval_calls = 0
        t_start = time.time()

        # Retrieve from memory (only populated from session 1 onwards for graph_session)
        memory_block = ""
        if memory_type == "graph" and injector:
            memory_block = injector.inject(task_desc)
            if memory_block:
                retrieval_calls += 1
        elif memory_type == "flat" and flat_store:
            memory_block = flat_store.retrieve(task_desc)
            if memory_block:
                retrieval_calls += 1

        if no_llm:
            # Bypass Ollama: use the task description as the action (MockEnv accepts goal keywords).
            # Simulates a perfect agent — success rate = 100% — so any S1→S2 delta is
            # purely from memory injection, not model variance.
            action = task_desc
            obs_out, success_final, _ = env.step(action)
            history = [{"action": action, "obs": obs_out, "success": success_final}]
            mem_note = f" [memory injected: {len(memory_block)} chars]" if memory_block else ""
            traj_text = f"TASK: {task_desc}\nACTION: {action}\nOBS: {obs_out}{mem_note}"
            logger.debug(f"  [{cond_name}] s{session_num} {task_id}: no-llm, success={success_final}")
        elif is_travel:
            try:
                plan_text, tokens = run_travel_plan(
                    task_desc, obs, memory_block, model, ollama_url, timeout
                )
            except Exception as e:
                logger.warning(f"  [{cond_name}] Travel agent error: {e}")
                plan_text, tokens = "", 0
            total_tokens += tokens
            feedback, success_final, _ = env.step(plan_text)
            history = [{"action": plan_text, "obs": feedback, "success": success_final}]
            traj_text = f"QUERY: {task_desc}\n\nREFERENCE (excerpt):\n{obs[:2000]}\n\nPLAN:\n{plan_text}\n\nEVAL: {feedback}"
            logger.debug(f"  [{cond_name}] s{session_num} {task_id}: {feedback[:100]}")
        else:
            history: list[dict] = []
            for step_num in range(max_steps):
                try:
                    action, tokens = run_agent_step(
                        task_desc, obs, history, memory_block,
                        model, ollama_url, timeout
                    )
                except Exception as e:
                    logger.warning(f"  Agent error at step {step_num}: {e}")
                    break
                total_tokens += tokens
                obs, success, done = env.step(action)
                history.append({"action": action, "obs": obs, "success": success})
                if done:
                    break
            success_final = any(h["success"] for h in history)
            traj_text = trajectory_to_text(task_desc, history)

        # Per-trajectory memory update (graph_pertraj and memp_flat)
        if update_policy == "per_trajectory":
            if memory_type == "graph" and consolidator:
                if no_llm:
                    # Skip LLM distillation — write raw trajectory as a script node
                    graph.add_procedure(traj_text[:500], level="script",
                                        source_agent=cond_name, session_id=f"s{session_num}_{task_id}")
                    logger.debug(f"  [{cond_name}] +1 node (no-llm)")
                else:
                    try:
                        log = consolidator.consolidate(
                            [traj_text], session_id=f"s{session_num}_{task_id}",
                            source_agent=cond_name, graph_path=None
                        )
                        logger.debug(f"  [{cond_name}] +{log.nodes_added} nodes")
                    except Exception as e:
                        logger.warning(f"  [{cond_name}] Consolidation error: {e}")
            elif memory_type == "flat" and flat_store:
                if success_final:
                    flat_store.add(traj_text, task_type=task_id)

        session_trajectories.append(traj_text)
        result = {
            "task_id": task_id,
            "session": session_num,
            "success": success_final,
            "steps": len(history),
            "tokens": total_tokens,
            "retrieval_calls": retrieval_calls,
            "elapsed_s": round(time.time() - t_start, 2),
        }
        task_results.append(result)
        logger.info(
            f"[{cond_name}] s{session_num} {task_id}: "
            f"success={success_final}, tokens={total_tokens}, retrieved={'yes' if retrieval_calls else 'no'}"
        )

    return task_results, session_trajectories


def run_condition(
    condition: dict,
    env,
    task_ids: list[str],
    cfg: dict,
    results_dir: Path,
    logger: logging.Logger,
    no_llm: bool = False,
) -> dict:
    cond_name = condition["name"]
    memory_type = condition.get("memory_type")
    update_policy = condition.get("update_policy")

    model = cfg["model"]["name"]
    ollama_url = cfg["model"]["ollama_url"]
    timeout = cfg["model"]["timeout"]
    max_steps = cfg["experiment"]["max_steps"]
    num_sessions = cfg["experiment"].get("num_sessions", 2)
    tasks_per_session = cfg["experiment"].get("tasks_per_session", len(task_ids) // num_sessions)
    graph_path = ROOT / cfg["memory"]["graph_path"]

    # Initialise memory (persists across sessions — this is the point)
    graph: ProcedureGraph | None = None
    flat_store: FlatMemoryStore | None = None
    consolidator: SessionConsolidator | None = None
    injector: ContextInjector | None = None

    if memory_type == "graph":
        graph = ProcedureGraph()
        consolidator = SessionConsolidator(
            graph=graph,
            model=cfg["model"].get("distil_model", model),
            dedup_threshold=cfg["memory"]["dedup_threshold"],
        )
        injector = ContextInjector(
            graph=graph,
            k_hops=cfg["memory"]["k_hops"],
            top_seeds=cfg["memory"]["top_seeds"],
            token_budget=cfg["memory"]["token_budget"],
        )
    elif memory_type == "flat":
        flat_store = FlatMemoryStore()

    from travelplanner_env import TravelPlannerEnv
    is_travel = isinstance(env, TravelPlannerEnv)

    all_task_results = []
    session_summaries = []

    for session_num in range(1, num_sessions + 1):
        # Each session uses a different slice of task_ids
        start = (session_num - 1) * tasks_per_session
        session_task_ids = task_ids[start: start + tasks_per_session]
        if not session_task_ids:
            logger.warning(f"[{cond_name}] No tasks left for session {session_num}, stopping.")
            break

        logger.info(f"\n[{cond_name}] ── Session {session_num}/{num_sessions} ({len(session_task_ids)} tasks) ──")

        task_results, session_trajectories = _run_session(
            session_num=session_num,
            task_ids=session_task_ids,
            cond_name=cond_name,
            memory_type=memory_type,
            update_policy=update_policy,
            env=env,
            is_travel=is_travel,
            graph=graph,
            flat_store=flat_store,
            consolidator=consolidator,
            injector=injector,
            model=model,
            ollama_url=ollama_url,
            timeout=timeout,
            max_steps=max_steps,
            graph_path=graph_path,
            logger=logger,
            no_llm=no_llm,
        )
        all_task_results.extend(task_results)

        session_sr = sum(r["success"] for r in task_results) / len(task_results) if task_results else 0
        session_summaries.append({
            "session": session_num,
            "num_tasks": len(task_results),
            "success_rate": session_sr,
            "mean_tokens": sum(r["tokens"] for r in task_results) / len(task_results) if task_results else 0,
            "retrieval_calls": sum(r["retrieval_calls"] for r in task_results),
            "graph_nodes_after": len(graph) if graph else None,
            "flat_entries_after": len(flat_store) if flat_store else None,
        })
        logger.info(f"[{cond_name}] Session {session_num} success rate: {session_sr:.1%}")

        # Session-end consolidation: runs after session 1 so session 2 can retrieve from it
        if update_policy == "session_end" and memory_type == "graph" and consolidator:
            try:
                cond_graph_path = graph_path.parent / f"procedure_graph_{cond_name}.json"
                if no_llm:
                    # Skip LLM distillation: write each trajectory as a raw script node
                    from src.memory.consolidator import ConsolidationLog
                    log = ConsolidationLog(session_id=f"session_{session_num}", source_agent=cond_name)
                    for i, traj in enumerate(session_trajectories):
                        graph.add_procedure(traj[:500], level="script", source_agent=cond_name, session_id=f"session_{session_num}")
                        log.nodes_added += 1
                    log.trajectories_processed = len(session_trajectories)
                    graph.save(cond_graph_path)
                else:
                    log = consolidator.consolidate(
                        session_trajectories, session_id=f"session_{session_num}",
                        source_agent=cond_name, graph_path=cond_graph_path
                    )
                logger.info(
                    f"[{cond_name}] Consolidated session {session_num}: "
                    f"+{log.nodes_added} nodes, {log.nodes_merged} merged, "
                    f"{len(log.conflicts_detected)} conflicts → graph now has {len(graph)} nodes"
                )
            except Exception as e:
                logger.warning(f"[{cond_name}] Consolidation error: {e}")

    # Key metric: improvement from session 1 → session 2
    delta = None
    if len(session_summaries) >= 2:
        delta = session_summaries[1]["success_rate"] - session_summaries[0]["success_rate"]

    out = {
        "condition": cond_name,
        "memory_type": memory_type,
        "update_policy": update_policy,
        "task_offset": 0,  # overwritten by caller if --task-offset used
        "sessions": session_summaries,
        "tasks": all_task_results,
        "summary": {
            "session_1_success_rate": session_summaries[0]["success_rate"] if session_summaries else None,
            "session_2_success_rate": session_summaries[1]["success_rate"] if len(session_summaries) > 1 else None,
            "delta_s1_to_s2": delta,
            "mean_tokens_s2": session_summaries[1]["mean_tokens"] if len(session_summaries) > 1 else None,
            "total_retrieval_calls_s2": session_summaries[1]["retrieval_calls"] if len(session_summaries) > 1 else None,
            "final_graph_nodes": len(graph) if graph else None,
        },
    }

    # Suffix output file if using task offset (avoids overwriting the base run)
    _offset = out.get("task_offset", 0)
    suffix = f"_offset{_offset}" if _offset else ""
    out_path = results_dir / f"{cond_name}{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    s1 = session_summaries[0]["success_rate"] if session_summaries else 0
    s2 = session_summaries[1]["success_rate"] if len(session_summaries) > 1 else 0
    logger.info(
        f"[{cond_name}] Done. S1={s1:.1%} → S2={s2:.1%} "
        f"(Δ={delta:+.1%})" if delta is not None else f"[{cond_name}] Done. S1={s1:.1%}"
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="E1: Graph Memory Controller experiment")
    parser.add_argument("--mock", action="store_true", help="Use MockEnv (household tasks)")
    parser.add_argument("--travelplanner", action="store_true", help="Use TravelPlanner benchmark (recommended)")
    parser.add_argument("--split", nargs="+", default=["validation"],
                        choices=["validation", "test"],
                        help="TravelPlanner split(s) to run. Pass both to run simultaneously.")
    parser.add_argument("--levels", nargs="+", default=None,
                        choices=["easy", "medium", "hard"],
                        help="Filter TravelPlanner tasks by difficulty. Default: all levels.")
    parser.add_argument("--condition", default=None, help="Run only one condition by name")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--task-offset", type=int, default=0,
                        help="Skip first N task IDs. Use to add new tasks without re-running existing ones.")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip all Ollama calls: agent returns goal action directly, "
                             "consolidation writes raw text as nodes. Fast demo of 2-session logic.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = load_config()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Total tasks needed = sessions × tasks_per_session
    num_sessions = cfg["experiment"].get("num_sessions", 2)
    tasks_per_session = cfg["experiment"].get("tasks_per_session", 10)
    num_tasks = args.num_tasks or (num_sessions * tasks_per_session)
    offset = args.task_offset

    # Environment
    if args.travelplanner:
        from travelplanner_env import TravelPlannerEnv
        env = TravelPlannerEnv(splits=args.split, subset=offset + num_tasks, levels=args.levels)
        level_str = f", levels={args.levels}" if args.levels else ""
        logger.info(f"Using TravelPlannerEnv (splits={args.split}{level_str}, {num_tasks} tasks/split)")
    elif args.mock:
        from env_adapter import MockEnv
        env = MockEnv(seed=cfg["experiment"]["seed"])
        logger.info("Using MockEnv")
    else:
        try:
            from env_adapter import ALFWorldAdapter
            alfworld_cfg = Path(__file__).parent / "alfworld_config.yaml"
            env = ALFWorldAdapter(
                config_path=str(alfworld_cfg),
                split=cfg["experiment"]["task_split"],
                num_tasks=num_tasks,
            )
            logger.info("Using ALFWorldAdapter")
        except ImportError:
            logger.warning("ALFWorld not available, falling back to TravelPlannerEnv")
            from travelplanner_env import TravelPlannerEnv
            env = TravelPlannerEnv(split="validation", subset=num_tasks)

    all_task_ids = env.task_ids()
    offset = args.task_offset
    task_ids = all_task_ids[offset: offset + num_tasks]
    if not task_ids:
        logger.error(f"No task IDs at offset {offset} (total available: {len(all_task_ids)})")
        sys.exit(1)
    if offset > 0:
        logger.info(f"Task offset: {offset} — using tasks {task_ids[0]}…{task_ids[-1]}")

    results_dir = ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    conditions = cfg["conditions"]
    if args.condition:
        conditions = [c for c in conditions if c["name"] == args.condition]
        if not conditions:
            logger.error(f"Condition '{args.condition}' not found in config")
            sys.exit(1)

    if args.no_llm:
        logger.info("--no-llm mode: bypassing Ollama for agent steps and consolidation")

    all_results = []
    for condition in conditions:
        logger.info(f"\n{'='*60}\nRunning condition: {condition['name']}\n{'='*60}")
        result = run_condition(condition, env, task_ids, cfg, results_dir, logger, no_llm=args.no_llm)
        result["task_offset"] = offset
        all_results.append(result)

    # Combined summary
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps({"conditions": [r["summary"] | {"name": r["condition"]} for r in all_results]}, indent=2), encoding="utf-8")
    logger.info(f"\nResults written to {results_dir}")

    # Print comparison table — key metric is S1 → S2 delta
    print("\n=== RESULTS SUMMARY (2-session) ===")
    print(f"{'Condition':<22} {'S1 Success':>12} {'S2 Success':>12} {'D S1->S2':>10} {'S2 Tokens':>11} {'Graph Nodes':>12}")
    print("-" * 81)
    for r in all_results:
        s = r["summary"]
        s1  = s.get("session_1_success_rate") or 0
        s2  = s.get("session_2_success_rate") or 0
        d   = s.get("delta_s1_to_s2")
        tok = s.get("mean_tokens_s2") or 0
        gn  = s.get("final_graph_nodes") or "-"
        delta_str = f"{d:+.1%}" if d is not None else "  n/a"
        print(f"{r['condition']:<22} {s1:>11.1%} {s2:>12.1%} {delta_str:>10} {tok:>11.0f} {str(gn):>12}")


if __name__ == "__main__":
    main()
