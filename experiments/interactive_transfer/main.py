"""
E3: Interactive Session Querying for Knowledge Transfer.

Research question: Does demand-driven interactive querying of a previous session
achieve better cost-efficiency (tokens per unit of performance) than static
knowledge transfer?

Conditions:
  no_memory          — cold start (baseline)
  graph_session      — k-hop subgraph injection from E1 graph infrastructure
  full_handoff       — all teacher trajectories prepended to student context
  interactive_oracle — student asks up to K questions before planning

Two phases:
  Phase 1 (teacher): Run num_tasks_teacher tasks, record trajectories.
  Phase 2 (student): Run num_tasks_student NEW tasks with different knowledge access.

Usage:
  python experiments/interactive_transfer/main.py [options]

  --no-llm             Bypass Ollama (structural test only)
  --condition COND     Run one condition only
  --num-tasks N        Override num_tasks_teacher AND num_tasks_student
  --max-questions K    Override oracle.max_questions
  --split SPLIT        TravelPlanner split (default: validation)
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

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / "memory_controller"))

from src.memory.graph_store import ProcedureGraph
from src.memory.consolidator import SessionConsolidator
from src.memory.context_injector import ContextInjector
from src.memory.session_oracle import SessionOracle

CONFIG_PATH = Path(__file__).parent / "config.yaml"

CONDITIONS = ["no_memory", "graph_session", "full_handoff", "interactive_oracle"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if overrides:
        for k, v in overrides.items():
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TRAVEL_PLANNER_SYSTEM = """\
You are a proficient travel planner. Based on the provided information and query, \
give me a detailed plan including flight numbers, restaurant names, and accommodation names. \
All information must come from the provided reference data.

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

QUESTION_GENERATION_SYSTEM = """\
You are a travel planning agent about to solve a new task. You have access to an \
oracle that knows how a previous agent solved similar tasks. Ask one targeted, \
specific question that would most help you satisfy the task constraints. \
The oracle can only answer one question at a time — make it count.
Reply with just the question, nothing else."""

QA_BLOCK_HEADER = "[PRIOR KNOWLEDGE FROM ORACLE]\n"
QA_BLOCK_FOOTER = "\n[END PRIOR KNOWLEDGE]\n"


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_CONNECT_TIMEOUT = 15    # seconds to establish TCP connection to Ollama
_READ_TIMEOUT    = 900   # seconds to wait for response bytes — 15 min for CPU inference on long oracle prompts
_MAX_RETRIES     = 3
_RETRY_BACKOFF   = 30    # seconds between retries


def _llm_call(messages: list[dict], model: str, ollama_url: str, timeout: int) -> tuple[str, int, float]:
    """
    Single LLM call using the OpenAI-compatible /v1/chat/completions API.
    Retries up to _MAX_RETRIES times with backoff. Returns (content, tokens, elapsed_s).
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    t0 = time.time()
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(
                ollama_url, json=payload,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            elapsed = round(time.time() - t0, 2)
            return content, tokens, elapsed
        except Exception as exc:
            last_exc = exc
            elapsed_so_far = round(time.time() - t0, 1)
            if attempt < _MAX_RETRIES:
                import logging as _log
                _log.getLogger(__name__).warning(
                    f"LLM call failed (attempt {attempt}/{_MAX_RETRIES}, "
                    f"{elapsed_so_far}s elapsed): {exc}. "
                    f"Retrying in {_RETRY_BACKOFF}s..."
                )
                time.sleep(_RETRY_BACKOFF)
    raise last_exc


def run_travel_plan(
    query: str,
    reference_text: str,
    memory_block: str,
    model: str,
    ollama_url: str,
    timeout: int,
) -> tuple[str, int, float]:
    """Returns (plan_text, tokens, elapsed_s)."""
    system = TRAVEL_PLANNER_SYSTEM.format(memory_block=memory_block)
    user_msg = f"Given information:\n{reference_text[:8000]}\n\nQuery: {query}\n\nTravel Plan:"
    return _llm_call(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        model, ollama_url, timeout,
    )


def generate_question(
    task_desc: str,
    prior_qa: list[dict],
    model: str,
    ollama_url: str,
    timeout: int,
) -> tuple[str, int, float]:
    """Ask the student LLM to generate its next oracle question. Returns (question, tokens, elapsed_s)."""
    prior_text = ""
    if prior_qa:
        lines = []
        for i, pair in enumerate(prior_qa):
            lines.append(f"Q{i+1}: {pair['question']}")
            lines.append(f"A{i+1}: {pair['answer']}")
        prior_text = "\n\nPRIOR ANSWERS SO FAR:\n" + "\n".join(lines)

    user_msg = f"CURRENT TASK:\n{task_desc}{prior_text}\n\nWhat is your next question for the oracle?"
    return _llm_call(
        [{"role": "system", "content": QUESTION_GENERATION_SYSTEM}, {"role": "user", "content": user_msg}],
        model, ollama_url, timeout,
    )


def format_qa_block(qa_pairs: list[dict]) -> str:
    if not qa_pairs:
        return ""
    lines = [QA_BLOCK_HEADER]
    for i, pair in enumerate(qa_pairs):
        lines.append(f"Q{i+1}: {pair['question']}")
        lines.append(f"A{i+1}: {pair['answer']}")
        lines.append("")
    lines.append(QA_BLOCK_FOOTER)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 1: Teacher session
# ---------------------------------------------------------------------------

def run_teacher_session(
    env,
    task_ids: list[str],
    model: str,
    ollama_url: str,
    timeout: int,
    logger: logging.Logger,
    no_llm: bool = False,
) -> list[str]:
    """Run teacher on task_ids. Returns list of trajectory text strings."""
    trajectories = []
    for task_id in task_ids:
        task_desc, reference = env.reset(task_id)
        t0 = time.time()
        if no_llm:
            plan_text = f"[MOCK PLAN for {task_id}] Day 1: Current City: from A to B\nTransportation: -\n..."
            feedback, success, _ = env.step(plan_text)
            tokens = 0
        else:
            try:
                plan_text, tokens, _ = run_travel_plan(task_desc, reference, "", model, ollama_url, timeout)
                feedback, success, _ = env.step(plan_text)
            except Exception as e:
                logger.warning(f"[teacher] {task_id} error: {e}")
                plan_text, tokens, success, feedback = "", 0, False, str(e)

        traj = (
            f"QUERY: {task_desc}\n\n"
            f"REFERENCE (excerpt):\n{reference[:2000]}\n\n"
            f"PLAN:\n{plan_text}\n\n"
            f"EVAL: success={success}, feedback={feedback[:200]}"
        )
        trajectories.append(traj)
        elapsed = round(time.time() - t0, 2)
        logger.info(f"[teacher] {task_id}: success={success}, tokens={tokens}, elapsed={elapsed}s")

    logger.info(f"[teacher] session complete — {len(trajectories)} trajectories collected")
    return trajectories


# ---------------------------------------------------------------------------
# Phase 2: Oracle Q&A loop
# ---------------------------------------------------------------------------

def run_oracle_qa(
    oracle: SessionOracle,
    task_desc: str,
    k: int,
    model: str,
    ollama_url: str,
    timeout: int,
    no_llm: bool = False,
) -> tuple[str, int, list[str], list[float]]:
    """
    Student asks up to k questions, oracle answers.
    Returns (formatted_qa_block, total_oracle_tokens, list_of_questions, oracle_call_latencies_s).
    oracle_call_latencies_s contains one elapsed time per oracle answer call (not question generation).
    """
    qa_pairs = []
    total_tokens = 0
    log_start = len(oracle.log)

    for _ in range(k):
        if no_llm:
            question = f"[MOCK Q] What constraints did the teacher encounter in similar tasks?"
            q_tokens = 0
        else:
            try:
                question, q_tokens, _ = generate_question(task_desc, qa_pairs, model, ollama_url, timeout)
            except Exception as e:
                question = "What general strategies worked for travel planning tasks?"
                q_tokens = 0

        if no_llm:
            answer, a_tokens = oracle.ask_mock(question)
        else:
            try:
                answer, a_tokens = oracle.ask(question)
            except Exception as e:
                answer = "[oracle error]"
                a_tokens = 0

        qa_pairs.append({"question": question, "answer": answer})
        total_tokens += q_tokens + a_tokens

    # Pull per-call latencies from oracle log (entries appended during this call)
    oracle_call_latencies = [entry["elapsed_s"] for entry in oracle.log[log_start:]]
    questions = [p["question"] for p in qa_pairs]
    return format_qa_block(qa_pairs), total_tokens, questions, oracle_call_latencies


# ---------------------------------------------------------------------------
# Phase 2: Student session
# ---------------------------------------------------------------------------

def run_student_session(
    condition: str,
    env,
    task_ids: list[str],
    oracle: SessionOracle,
    injector: ContextInjector | None,
    cfg: dict,
    logger: logging.Logger,
    no_llm: bool = False,
) -> list[dict]:
    """Run student on task_ids with knowledge access based on condition."""
    model = cfg["model"]["name"]
    ollama_url = cfg["model"]["ollama_url"]
    timeout = cfg["model"]["timeout"]
    k = cfg["oracle"]["max_questions"]

    task_results = []

    for task_id in task_ids:
        task_desc, reference = env.reset(task_id)
        t0 = time.time()
        oracle_tokens = 0
        oracle_questions: list[str] = []
        oracle_call_latencies: list[float] = []
        oracle_elapsed_s = 0.0

        # --- Build memory block based on condition ---
        if condition == "no_memory":
            memory_block = ""

        elif condition == "full_handoff":
            memory_block = oracle.format_all_trajectories()

        elif condition == "graph_session":
            memory_block = injector.inject(task_desc) if injector else ""

        elif condition == "interactive_oracle":
            t_oracle = time.time()
            memory_block, oracle_tokens, oracle_questions, oracle_call_latencies = run_oracle_qa(
                oracle, task_desc, k, model, ollama_url, timeout, no_llm=no_llm
            )
            oracle_elapsed_s = round(time.time() - t_oracle, 2)

        else:
            memory_block = ""

        # --- Plan ---
        if no_llm:
            plan_text = f"[MOCK PLAN for {task_id}] Day 1: Current City: from A to B\nTransportation: -\n..."
            plan_tokens = 0
            plan_elapsed_s = 0.0
            feedback, success, _ = env.step(plan_text)
        else:
            try:
                plan_text, plan_tokens, plan_elapsed_s = run_travel_plan(
                    task_desc, reference, memory_block, model, ollama_url, timeout
                )
                feedback, success, _ = env.step(plan_text)
            except Exception as e:
                logger.warning(f"[{condition}] {task_id} plan error: {e}")
                plan_text, plan_tokens, plan_elapsed_s, success, feedback = "", 0, 0.0, False, str(e)

        total_tokens = plan_tokens + oracle_tokens
        total_elapsed_s = round(time.time() - t0, 2)
        retrieved = bool(memory_block)
        mean_oracle_call_s = (
            round(sum(oracle_call_latencies) / len(oracle_call_latencies), 2)
            if oracle_call_latencies else 0.0
        )

        result = {
            "task_id": task_id,
            "condition": condition,
            "success": success,
            "plan_tokens": plan_tokens,
            "oracle_tokens": oracle_tokens,
            "total_tokens": total_tokens,
            "oracle_questions_asked": len(oracle_questions),
            "oracle_questions": oracle_questions,
            "retrieved": retrieved,
            "memory_chars": len(memory_block),
            "plan_elapsed_s": plan_elapsed_s,
            "oracle_elapsed_s": oracle_elapsed_s,
            "mean_oracle_call_elapsed_s": mean_oracle_call_s,
            "total_elapsed_s": total_elapsed_s,
        }
        task_results.append(result)
        logger.info(
            f"[{condition}] {task_id}: success={success}, "
            f"plan_tok={plan_tokens}, oracle_tok={oracle_tokens}, "
            f"total_tok={total_tokens}, plan_s={plan_elapsed_s:.1f}, "
            f"oracle_s={oracle_elapsed_s:.1f}, total_s={total_elapsed_s:.1f}"
        )

    return task_results


# ---------------------------------------------------------------------------
# Model warmup
# ---------------------------------------------------------------------------

def warmup_model(model: str, ollama_url: str, timeout: int, logger: logging.Logger, max_attempts: int = 5) -> None:
    """Fire a cheap throwaway request and retry until the model responds successfully."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Ready?"}],
        "temperature": 0.0,
        "max_tokens": 1,
    }
    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.time()
            resp = requests.post(ollama_url, json=payload, timeout=timeout)
            elapsed = round(time.time() - t0, 1)
            if resp.status_code == 200:
                logger.info(f"[warmup] Model ready ({elapsed}s)")
                return
            logger.warning(f"[warmup] Attempt {attempt}/{max_attempts}: HTTP {resp.status_code} ({elapsed}s), retrying...")
        except Exception as e:
            logger.warning(f"[warmup] Attempt {attempt}/{max_attempts}: {e}, retrying...")
        time.sleep(5)
    logger.warning("[warmup] Model did not respond cleanly after all attempts — proceeding anyway.")


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def compute_summary(task_results: list[dict]) -> dict:
    if not task_results:
        return {}
    n = len(task_results)
    n_success = sum(r["success"] for r in task_results)
    total_tokens = [r["total_tokens"] for r in task_results]
    plan_tokens = [r["plan_tokens"] for r in task_results]
    oracle_tokens = [r["oracle_tokens"] for r in task_results]
    plan_elapsed = [r["plan_elapsed_s"] for r in task_results]
    oracle_elapsed = [r["oracle_elapsed_s"] for r in task_results]
    total_elapsed = [r["total_elapsed_s"] for r in task_results]
    oracle_call_elapsed = [r["mean_oracle_call_elapsed_s"] for r in task_results if r["mean_oracle_call_elapsed_s"] > 0]
    return {
        "num_tasks": n,
        "success_rate": n_success / n,
        "mean_total_tokens": sum(total_tokens) / n,
        "mean_plan_tokens": sum(plan_tokens) / n,
        "mean_oracle_tokens": sum(oracle_tokens) / n,
        "tokens_per_successful_task": sum(total_tokens) / max(n_success, 1),
        "mean_oracle_questions_asked": sum(r["oracle_questions_asked"] for r in task_results) / n,
        "mean_memory_chars": sum(r["memory_chars"] for r in task_results) / n,
        "retrieval_rate": sum(r["retrieved"] for r in task_results) / n,
        # Latency
        "mean_plan_elapsed_s": sum(plan_elapsed) / n,
        "mean_oracle_elapsed_s": sum(oracle_elapsed) / n,
        "mean_total_elapsed_s": sum(total_elapsed) / n,
        "mean_oracle_call_elapsed_s": sum(oracle_call_elapsed) / len(oracle_call_elapsed) if oracle_call_elapsed else 0.0,
    }


def save_condition_results(
    condition: str,
    task_results: list[dict],
    teacher_task_ids: list[str],
    results_dir: Path,
    cfg: dict,
) -> dict:
    summary = compute_summary(task_results)
    out = {
        "condition": condition,
        "config": {
            "num_tasks_teacher": cfg["experiment"]["num_tasks_teacher"],
            "num_tasks_student": cfg["experiment"]["num_tasks_student"],
            "max_questions": cfg["oracle"]["max_questions"],
            "task_split": cfg["experiment"]["task_split"],
        },
        "teacher_task_ids": teacher_task_ids,
        "tasks": task_results,
        "summary": summary,
    }
    out_path = results_dir / f"{condition}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E3: Interactive Session Querying")
    parser.add_argument("--condition", default=None,
                        help="Comma-separated conditions to run (default: all). "
                             f"Valid: {', '.join(CONDITIONS)}")
    parser.add_argument("--num-tasks", type=int, default=None,
                        help="Override num_tasks_teacher AND num_tasks_student")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Override oracle.max_questions (K ablation)")
    parser.add_argument("--split", default=None, choices=["validation", "test"],
                        help="TravelPlanner split (default: from config)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Bypass Ollama for structural testing")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="Load saved teacher trajectories and graph instead of re-running phase 1")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = load_config()
    if args.num_tasks:
        cfg["experiment"]["num_tasks_teacher"] = args.num_tasks
        cfg["experiment"]["num_tasks_student"] = args.num_tasks
    if args.max_questions is not None:
        cfg["oracle"]["max_questions"] = args.max_questions
    if args.split:
        cfg["experiment"]["task_split"] = args.split

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    model = cfg["model"]["name"]
    ollama_url = cfg["model"]["ollama_url"]
    timeout = cfg["model"]["timeout"]
    n_teacher = cfg["experiment"]["num_tasks_teacher"]
    n_student = cfg["experiment"]["num_tasks_student"]
    teacher_offset = cfg["experiment"]["teacher_offset"]
    student_offset = cfg["experiment"]["student_offset"]
    task_split = cfg["experiment"]["task_split"]
    task_levels = cfg["experiment"].get("task_levels") or None
    results_dir = ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # Environment
    from travelplanner_env import TravelPlannerEnv
    total_needed = max(teacher_offset + n_teacher, student_offset + n_student)
    env = TravelPlannerEnv(splits=[task_split], subset=total_needed, levels=task_levels)
    all_ids = env.task_ids()

    teacher_ids = all_ids[teacher_offset: teacher_offset + n_teacher]
    student_ids = all_ids[student_offset: student_offset + n_student]

    if not teacher_ids:
        logger.error(f"No teacher tasks at offset {teacher_offset}")
        sys.exit(1)
    if not student_ids:
        logger.error(f"No student tasks at offset {student_offset}")
        sys.exit(1)

    overlap = set(teacher_ids) & set(student_ids)
    if overlap:
        logger.warning(f"Teacher/student task overlap detected: {overlap}. Adjust offsets.")

    logger.info(
        f"Teacher tasks: {teacher_ids[0]}…{teacher_ids[-1]} ({len(teacher_ids)} tasks)\n"
        f"Student tasks: {student_ids[0]}…{student_ids[-1]} ({len(student_ids)} tasks)\n"
        f"Oracle K={cfg['oracle']['max_questions']}, "
        f"no_llm={args.no_llm}"
    )

    # Warmup: ensure the model is loaded before the timed experiment begins
    if not args.no_llm:
        warmup_model(model, ollama_url, timeout, logger)

    # -----------------------------------------------------------------------
    # Phase 1: Teacher session (or load saved trajectories)
    # -----------------------------------------------------------------------
    teacher_path = results_dir / "teacher_trajectories.json"
    graph_path = ROOT / cfg["memory"]["graph_path"]

    if args.skip_teacher and teacher_path.exists():
        logger.info("\n[skip-teacher] Loading saved teacher trajectories from disk...")
        saved = json.loads(teacher_path.read_text(encoding="utf-8"))
        teacher_trajectories = saved["trajectories"]
        teacher_ids = saved["task_ids"]
        logger.info(f"[skip-teacher] Loaded {len(teacher_trajectories)} trajectories")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: Teacher session")
        logger.info("=" * 60)
        teacher_trajectories = run_teacher_session(
            env, teacher_ids, model, ollama_url, timeout, logger, no_llm=args.no_llm
        )
        teacher_path.write_text(json.dumps({
            "task_ids": teacher_ids,
            "trajectories": teacher_trajectories,
        }, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Build shared knowledge artifacts (used by specific conditions)
    # -----------------------------------------------------------------------
    oracle = SessionOracle(
        trajectories=teacher_trajectories,
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
        context_budget=cfg["oracle"]["context_budget"],
    )

    graph = ProcedureGraph()
    injector: ContextInjector | None = None
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_teacher and graph_path.exists():
        logger.info("[skip-teacher] Loading saved graph from disk...")
        graph = ProcedureGraph.load(graph_path)
        logger.info(f"[skip-teacher] Loaded graph with {len(graph)} nodes")
    else:
        logger.info("\n[graph_session] Consolidating teacher trajectories into graph...")
        consolidator = SessionConsolidator(
            graph=graph,
            model=cfg["model"].get("distil_model", model),
            dedup_threshold=cfg["memory"]["dedup_threshold"],
        )
        if args.no_llm:
            from src.memory.consolidator import ConsolidationLog
            log = ConsolidationLog(session_id="teacher_session", source_agent="teacher")
            for i, traj in enumerate(teacher_trajectories):
                graph.add_procedure(traj[:500], level="script",
                                    source_agent="teacher", session_id=f"teacher_{i}")
                log.nodes_added += 1
            log.trajectories_processed = len(teacher_trajectories)
            graph.save(graph_path)
        else:
            try:
                log = consolidator.consolidate(
                    teacher_trajectories, session_id="teacher_session",
                    source_agent="teacher", graph_path=graph_path,
                )
                logger.info(f"[graph] +{log.nodes_added} nodes, {log.nodes_merged} merged")
            except Exception as e:
                logger.warning(f"[graph] Consolidation error: {e}")

    if len(graph) > 0:
        injector = ContextInjector(
            graph=graph,
            k_hops=cfg["memory"]["k_hops"],
            top_seeds=cfg["memory"]["top_seeds"],
            token_budget=cfg["memory"]["token_budget"],
        )

    # -----------------------------------------------------------------------
    # Phase 2: Student session — one run per condition
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Student session")
    logger.info("=" * 60)

    if args.condition:
        selected_conditions = [c.strip() for c in args.condition.split(",")]
        invalid = [c for c in selected_conditions if c not in CONDITIONS]
        if invalid:
            logger.error(f"Unknown conditions: {invalid}. Valid: {CONDITIONS}")
            sys.exit(1)
    else:
        selected_conditions = CONDITIONS
    all_results = []

    for condition in selected_conditions:
        logger.info(f"\n{'─'*50}\nCondition: {condition}\n{'─'*50}")
        task_results = run_student_session(
            condition=condition,
            env=env,
            task_ids=student_ids,
            oracle=oracle,
            injector=injector,
            cfg=cfg,
            logger=logger,
            no_llm=args.no_llm,
        )
        out = save_condition_results(condition, task_results, teacher_ids, results_dir, cfg)
        all_results.append(out)

    # -----------------------------------------------------------------------
    # Combined summary
    # -----------------------------------------------------------------------
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(
        {"conditions": [{"condition": r["condition"], **r["summary"]} for r in all_results]},
        indent=2,
    ), encoding="utf-8")
    logger.info(f"\nResults written to {results_dir}")

    # Print table
    print("\n=== E3 RESULTS SUMMARY ===")
    hdr = (f"{'Condition':<22} {'Success':>9} {'TotalTok':>10} {'PlanTok':>9} "
           f"{'OracleTok':>11} {'Tok/Succ':>10} {'PlanSec':>9} {'OracleSec':>11} {'TotalSec':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in all_results:
        s = r["summary"]
        print(
            f"{r['condition']:<22} "
            f"{s.get('success_rate', 0):>8.1%} "
            f"{s.get('mean_total_tokens', 0):>10.0f} "
            f"{s.get('mean_plan_tokens', 0):>9.0f} "
            f"{s.get('mean_oracle_tokens', 0):>11.0f} "
            f"{s.get('tokens_per_successful_task', 0):>10.0f} "
            f"{s.get('mean_plan_elapsed_s', 0):>9.1f} "
            f"{s.get('mean_oracle_elapsed_s', 0):>11.1f} "
            f"{s.get('mean_total_elapsed_s', 0):>10.1f}"
        )


if __name__ == "__main__":
    main()
