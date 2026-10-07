"""
E4: ALFWorld Plan Generation with Knowledge Transfer.

Research question: Does in-context knowledge transfer from teacher trajectories
help or hurt on structurally uniform household manipulation tasks?

This replicates the E3 knowledge-transfer conditions on ALFWorld data to test
whether the context pollution finding from TravelPlanner (E3) generalises.

ALFWorld tasks are loaded from traj_data.json without the TextWorld runtime
(textworld has no Windows wheel). The LLM generates high-level action plans
in response to natural-language task descriptions; we evaluate against
ground-truth sequences from expert demonstrations.

Task type: pick_and_place_simple
Expected structure: GotoLocation -> PickupObject -> GotoLocation -> PutObject
This is structurally IDENTICAL across all instances — any teacher trajectory
is procedurally relevant to any student task of the same type.

Conditions:
  no_memory          — cold start (baseline)
  graph_session      — k-hop subgraph injection from teacher trajectories
  full_handoff       — all teacher trajectories prepended to student context
  interactive_oracle — student asks up to K questions before planning

Usage:
  python experiments/alfworld_transfer/main.py [options]

  --no-llm           Bypass Ollama (structural test only)
  --condition COND   Run one condition only
  --num-tasks N      Override num_tasks for teacher and student
  --skip-teacher     Skip teacher phase, load saved trajectories
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.memory.graph_store import ProcedureGraph
from src.memory.consolidator import SessionConsolidator
from src.memory.context_injector import ContextInjector
from src.memory.session_oracle import SessionOracle

CONFIG_PATH = Path(__file__).parent / "config.yaml"
CONDITIONS = ["no_memory", "graph_session", "full_handoff", "interactive_oracle"]

_CONNECT_TIMEOUT = 15
_READ_TIMEOUT    = 900
_MAX_RETRIES     = 3
_RETRY_BACKOFF   = 30

# High-level actions present in pick_and_place_simple tasks
VALID_ACTIONS = {
    "GotoLocation", "PickupObject", "PutObject", "SliceObject",
    "CleanObject", "HeatObject", "CoolObject", "ToggleObject", "NoOp",
}

# ---------------------------------------------------------------------------
# Config / Logging
# ---------------------------------------------------------------------------

def load_config(overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if overrides:
        cfg.update(overrides)
    return cfg


def setup_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run.log"
    stream_handler = logging.StreamHandler(sys.stdout)
    # Avoid Windows cp1252 encode errors by replacing unencodable chars
    if hasattr(stream_handler.stream, "reconfigure"):
        try:
            stream_handler.stream.reconfigure(errors="replace")
        except Exception:
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            stream_handler,
        ],
    )


# ---------------------------------------------------------------------------
# ALFWorld data loading
# ---------------------------------------------------------------------------

ALFWORLD_DATA = os.path.expanduser("~/.cache/alfworld")


def load_tasks(task_type: str, split: str, offset: int, n: int, seed: int = 42) -> list[dict]:
    """Load n tasks of task_type from split, starting at offset (alphabetical order)."""
    base = Path(ALFWORLD_DATA) / "json_2.1.1" / split
    dirs = sorted([d for d in os.listdir(base) if d.startswith(task_type)])
    selected = dirs[offset : offset + n]
    tasks = []
    for d in selected:
        trial_dir = Path(base) / d
        trials = list(trial_dir.iterdir())
        if not trials:
            continue
        traj_path = trials[0] / "traj_data.json"
        with open(traj_path, encoding="utf-8") as f:
            data = json.load(f)
        ann = data["turk_annotations"]["anns"][0]
        tasks.append({
            "task_id": d,
            "task_desc": ann["task_desc"],
            "high_descs": ann["high_descs"],
            "object_target": data["pddl_params"].get("object_target", ""),
            "parent_target": data["pddl_params"].get("parent_target", ""),
            "toggle_target": data["pddl_params"].get("toggle_target", ""),
            "object_sliced": data["pddl_params"].get("object_sliced", False),
            "high_actions": [a["discrete_action"]["action"] for a in data["plan"]["high_pddl"]],
        })
    return tasks


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _llm_call(
    messages: list[dict],
    model: str,
    url: str,
    max_tokens: int = 512,
    no_llm: bool = False,
) -> tuple[str, int]:
    if no_llm:
        return "[MOCK PLAN] GotoLocation object\nPickupObject object\nGotoLocation destination\nPutObject object on destination", 0
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    last_exc: Exception | None = None
    t0 = time.time()
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            return text, tokens
        except Exception as exc:
            last_exc = exc
            elapsed = round(time.time() - t0, 2)
            logging.warning(f"LLM call failed (attempt {attempt}/{_MAX_RETRIES}, {elapsed}s): {exc}")
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF)
    raise last_exc


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ALFWORLD_SYSTEM = """\
You are a household robot planning agent. Given a task description, generate a \
step-by-step high-level action plan using only these actions:
- GotoLocation <location/object>
- PickupObject <object>
- PutObject <object> on/in <receptacle>
- CleanObject <object> (wash in sink)
- HeatObject <object> (use microwave)
- CoolObject <object> (use fridge)
- ToggleObject <lamp/appliance>

{memory_block}

**Rules:**
1. Use only the actions listed above.
2. Be specific about object names and locations.
3. Format as a numbered list, one action per line.
4. End with "DONE" on its own line.
5. Do not add explanations — only the action list."""

TEACHER_TRAJ_TEMPLATE = """\
TASK: {task_desc}
OBJECT: {object_target}{parent_suffix}
GROUND-TRUTH PLAN:
{steps}
"""


def format_teacher_trajectory(task: dict) -> str:
    parent = task.get("parent_target", "")
    parent_suffix = f", DESTINATION: {parent}" if parent else ""
    steps = "\n".join(
        f"{i+1}. {a}" for i, a in enumerate(task["high_actions"])
        if a != "NoOp"
    )
    return TEACHER_TRAJ_TEMPLATE.format(
        task_desc=task["task_desc"],
        object_target=task["object_target"],
        parent_suffix=parent_suffix,
        steps=steps,
    )


def format_memory_block(trajectories: list[str]) -> str:
    if not trajectories:
        return ""
    parts = ["[TEACHER TRAJECTORIES — examples of similar solved tasks]"]
    for i, t in enumerate(trajectories):
        parts.append(f"\n[Example {i+1}]\n{t}")
    parts.append("[END TEACHER TRAJECTORIES]\n")
    return "\n".join(parts)


STUDENT_TASK_PROMPT = """\
TASK: {task_desc}
OBJECT: {object_target}{parent_suffix}

Generate your action plan:"""


def format_student_prompt(task: dict) -> str:
    parent = task.get("parent_target", "")
    parent_suffix = f", DESTINATION: {parent}" if parent else ""
    return STUDENT_TASK_PROMPT.format(
        task_desc=task["task_desc"],
        object_target=task["object_target"],
        parent_suffix=parent_suffix,
    )


QUESTION_GEN_SYSTEM = """\
You are a household robot about to solve a task. You have access to an oracle that \
knows how a previous robot solved similar tasks. Ask one targeted question that would \
most help you generate the correct action sequence. Be specific. Reply with just the question."""

QUESTION_GEN_USER = """\
CURRENT TASK: {task_desc}
OBJECT: {object_target}{parent_suffix}

PRIOR ANSWERS:
{prior_qa}

What is your next question for the oracle? Reply with just the question."""


# ---------------------------------------------------------------------------
# Teacher phase
# ---------------------------------------------------------------------------

def run_teacher_phase(
    tasks: list[dict],
    model: str,
    url: str,
    no_llm: bool,
) -> tuple[list[str], list[dict]]:
    """
    Extract teacher trajectories directly from ground-truth data.
    No LLM needed — teacher trajectories are the expert demonstrations.
    """
    trajectories = []
    records = []
    for task in tasks:
        traj_text = format_teacher_trajectory(task)
        trajectories.append(traj_text)
        records.append({
            "task_id": task["task_id"],
            "task_desc": task["task_desc"],
            "trajectory": traj_text,
            "high_actions": task["high_actions"],
        })
        logging.info(f"Teacher task: {task['task_id']} | {len(task['high_actions'])} steps")
    return trajectories, records


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def parse_plan(plan_text: str) -> list[str]:
    """Extract action types from LLM-generated plan text."""
    action_types = []
    for line in plan_text.split("\n"):
        line = line.strip()
        # Remove leading numbering
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        if not line or line == "DONE":
            continue
        # Match action type at start of line (camel case)
        m = re.match(r"^([A-Z][a-zA-Z]+)", line)
        if m and m.group(1) in VALID_ACTIONS:
            action_types.append(m.group(1))
    return action_types


def evaluate_plan(generated_text: str, task: dict) -> dict:
    """
    Evaluate generated plan against ground-truth.
    Returns success metrics.
    """
    gt_actions = [a for a in task["high_actions"] if a != "NoOp"]
    generated_actions = parse_plan(generated_text)

    # Strict: exact action type sequence match
    sequence_correct = generated_actions == gt_actions

    # Loose: correct set of unique action types (order ignored)
    loose_correct = set(generated_actions) == set(gt_actions)

    # Object check: does generated plan mention the target object?
    obj = task["object_target"].lower()
    object_mentioned = obj in generated_text.lower() if obj else True

    # Destination check
    dest = task.get("parent_target", "").lower()
    dest_mentioned = dest in generated_text.lower() if dest else True

    # Composite success: correct sequence + object + destination mentioned
    success = sequence_correct and object_mentioned and dest_mentioned

    return {
        "success": success,
        "sequence_correct": sequence_correct,
        "loose_correct": loose_correct,
        "object_mentioned": object_mentioned,
        "dest_mentioned": dest_mentioned,
        "gt_actions": gt_actions,
        "generated_actions": generated_actions,
        "num_gt_steps": len(gt_actions),
        "num_generated_steps": len(generated_actions),
    }


# ---------------------------------------------------------------------------
# Student phase — oracle Q&A
# ---------------------------------------------------------------------------

def run_oracle_qa(
    oracle: SessionOracle,
    task: dict,
    k: int,
    model: str,
    url: str,
    no_llm: bool,
) -> tuple[str, int, list[str]]:
    """Run K rounds of oracle Q&A. Returns (qa_block, total_tokens, questions)."""
    parent = task.get("parent_target", "")
    parent_suffix = f", DESTINATION: {parent}" if parent else ""
    prior_qa: list[dict] = []
    total_tokens = 0

    for _ in range(k):
        qa_text = "\n".join(
            f"Q: {qa['question']}\nA: {qa['answer']}" for qa in prior_qa
        ) if prior_qa else "(none yet)"
        user_msg = QUESTION_GEN_USER.format(
            task_desc=task["task_desc"],
            object_target=task["object_target"],
            parent_suffix=parent_suffix,
            prior_qa=qa_text,
        )
        question, q_tokens = _llm_call(
            [{"role": "system", "content": QUESTION_GEN_SYSTEM},
             {"role": "user", "content": user_msg}],
            model=model, url=url, max_tokens=128, no_llm=no_llm,
        )
        total_tokens += q_tokens

        if no_llm:
            answer, a_tokens = oracle.ask_mock(question)
        else:
            answer, a_tokens = oracle.ask(question)
        total_tokens += a_tokens
        prior_qa.append({"question": question, "answer": answer})

    qa_block = "[ORACLE Q&A — answers from similar solved tasks]\n"
    for i, qa in enumerate(prior_qa):
        qa_block += f"\nQ{i+1}: {qa['question']}\nA{i+1}: {qa['answer']}\n"
    qa_block += "[END ORACLE Q&A]\n"
    questions = [qa["question"] for qa in prior_qa]
    return qa_block, total_tokens, questions


# ---------------------------------------------------------------------------
# Student phase — one condition
# ---------------------------------------------------------------------------

def run_student_condition(
    condition: str,
    tasks: list[dict],
    trajectories: list[str],
    oracle: SessionOracle,
    graph: ProcedureGraph | None,
    injector: ContextInjector | None,
    cfg: dict,
    no_llm: bool,
) -> list[dict]:
    model = cfg["model"]["name"]
    url = cfg["model"]["ollama_url"]
    k = cfg["oracle"]["max_questions"]
    results = []

    logging.info(f"\n{'='*60}\nCondition: {condition}\n{'='*60}")

    for task in tasks:
        t0 = time.time()
        oracle_tokens = 0
        oracle_questions: list[str] = []
        oracle_elapsed = 0.0

        # --- Build memory block ---
        if condition == "no_memory":
            memory_block = ""
        elif condition == "full_handoff":
            memory_block = format_memory_block(trajectories)
        elif condition == "graph_session":
            if injector:
                injected = injector.inject(task["task_desc"])
                memory_block = f"[GRAPH MEMORY]\n{injected}\n[END GRAPH MEMORY]\n" if injected else ""
            else:
                memory_block = ""
        elif condition == "interactive_oracle":
            t_oracle = time.time()
            memory_block, oracle_tokens, oracle_questions = run_oracle_qa(
                oracle, task, k=k, model=model, url=url, no_llm=no_llm,
            )
            oracle_elapsed = round(time.time() - t_oracle, 2)
        else:
            memory_block = ""

        oracle_phase_end = time.time()

        # --- Generate plan ---
        system = ALFWORLD_SYSTEM.format(memory_block=memory_block)
        user_msg = format_student_prompt(task)
        plan_text, plan_tokens = _llm_call(
            [{"role": "system", "content": system},
             {"role": "user", "content": user_msg}],
            model=model, url=url, max_tokens=400, no_llm=no_llm,
        )

        plan_elapsed = round(time.time() - oracle_phase_end, 2)
        total_elapsed = round(time.time() - t0, 2)
        total_tokens = plan_tokens + oracle_tokens

        eval_result = evaluate_plan(plan_text, task)

        result = {
            "task_id": task["task_id"],
            "condition": condition,
            "success": eval_result["success"],
            "sequence_correct": eval_result["sequence_correct"],
            "loose_correct": eval_result["loose_correct"],
            "object_mentioned": eval_result["object_mentioned"],
            "dest_mentioned": eval_result["dest_mentioned"],
            "plan_tokens": plan_tokens,
            "oracle_tokens": oracle_tokens,
            "total_tokens": total_tokens,
            "oracle_questions": oracle_questions,
            "oracle_elapsed_s": oracle_elapsed,
            "plan_elapsed_s": plan_elapsed,
            "total_elapsed_s": total_elapsed,
            "plan_text": plan_text,
            "gt_actions": eval_result["gt_actions"],
            "generated_actions": eval_result["generated_actions"],
            "memory_chars": len(memory_block),
        }
        results.append(result)

        logging.info(
            f"  {task['task_id']}: success={result['success']} "
            f"seq={result['sequence_correct']} "
            f"tokens={total_tokens} elapsed={total_elapsed:.0f}s"
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="E4: ALFWorld Knowledge Transfer")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--condition", default=None, help="Single condition or comma-separated list")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--skip-teacher", action="store_true", help="Load saved teacher trajectories")
    args = parser.parse_args()

    cfg = load_config()
    results_dir = ROOT / cfg["output"]["results_dir"]
    setup_logging(results_dir)
    logging.info("E4: ALFWorld Knowledge Transfer")
    logging.info(f"Results dir: {results_dir}")

    n = args.num_tasks or cfg["experiment"]["num_tasks_teacher"]
    if args.max_questions:
        cfg["oracle"]["max_questions"] = args.max_questions

    conditions = CONDITIONS
    if args.condition:
        conditions = [c.strip() for c in args.condition.split(",")]

    # --- Load tasks ---
    task_type = cfg["experiment"]["task_type"]
    split = cfg["experiment"]["data_split"]
    teacher_offset = cfg["experiment"]["teacher_offset"]
    student_offset = cfg["experiment"]["student_offset"]

    student_tasks = load_tasks(task_type, split, student_offset, n, cfg["experiment"]["seed"])
    logging.info(f"Loaded {len(student_tasks)} student tasks ({task_type}, {split})")

    teacher_traj_path = ROOT / cfg["output"]["teacher_trajectories_path"]

    # --- Teacher phase ---
    if args.skip_teacher and teacher_traj_path.exists():
        logging.info("Skipping teacher phase — loading saved trajectories")
        saved = json.loads(teacher_traj_path.read_text(encoding="utf-8"))
        trajectories = [r["trajectory"] for r in saved]
        logging.info(f"Loaded {len(trajectories)} teacher trajectories")
    else:
        teacher_tasks = load_tasks(task_type, split, teacher_offset, n, cfg["experiment"]["seed"])
        logging.info(f"Loaded {len(teacher_tasks)} teacher tasks")
        trajectories, teacher_records = run_teacher_phase(
            teacher_tasks, model=cfg["model"]["name"],
            url=cfg["model"]["ollama_url"], no_llm=args.no_llm,
        )
        teacher_traj_path.parent.mkdir(parents=True, exist_ok=True)
        teacher_traj_path.write_text(
            json.dumps(teacher_records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logging.info(f"Saved {len(trajectories)} teacher trajectories to {teacher_traj_path}")

    # --- Build oracle ---
    oracle = SessionOracle(
        trajectories=trajectories,
        model=cfg["model"]["name"],
        ollama_url=cfg["model"]["ollama_url"],
        timeout=cfg["model"]["timeout"],
        context_budget=cfg["oracle"]["context_budget"],
    )

    # --- Build graph ---
    graph_path = ROOT / cfg["memory"]["graph_path"]
    graph = ProcedureGraph()
    injector: ContextInjector | None = None
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    if any(c == "graph_session" for c in conditions):
        if args.skip_teacher and graph_path.exists():
            graph = ProcedureGraph.load(graph_path)
            logging.info(f"Loaded graph: {len(graph)} nodes")
        else:
            consolidator = SessionConsolidator(
                graph=graph,
                model=cfg["model"].get("distil_model", cfg["model"]["name"]),
                dedup_threshold=cfg["memory"]["dedup_threshold"],
            )
            if args.no_llm:
                from src.memory.consolidator import ConsolidationLog
                for i, traj in enumerate(trajectories):
                    graph.add_procedure(traj[:500], level="script",
                                        source_agent="teacher", session_id=f"teacher_{i}")
                graph.save(graph_path)
                logging.info(f"[no-llm] Built mock graph: {len(graph)} nodes")
            else:
                try:
                    log = consolidator.consolidate(
                        trajectories, session_id="teacher_session",
                        source_agent="teacher", graph_path=graph_path,
                    )
                    logging.info(f"Built graph: +{log.nodes_added} nodes, {log.nodes_merged} merged")
                except Exception as e:
                    logging.warning(f"Graph consolidation error: {e}")

    if len(graph) > 0:
        injector = ContextInjector(
            graph=graph,
            k_hops=cfg["memory"]["k_hops"],
            top_seeds=cfg["memory"]["top_seeds"],
            token_budget=cfg["memory"]["token_budget"],
        )

    # --- Student conditions ---
    all_results: list[dict] = []
    for condition in conditions:
        res = run_student_condition(
            condition=condition,
            tasks=student_tasks,
            trajectories=trajectories,
            oracle=oracle,
            graph=graph,
            injector=injector,
            cfg=cfg,
            no_llm=args.no_llm,
        )
        all_results.extend(res)
        # Save per-condition
        cond_path = results_dir / f"{condition}.json"
        cond_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info(f"Saved {condition} → {cond_path}")

    # --- Summary ---
    summary = compute_summary(all_results)
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(f"\nSummary saved to {summary_path}")
    print_summary(summary)


def compute_summary(all_results: list[dict]) -> dict:
    from collections import defaultdict
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_condition[r["condition"]].append(r)

    conditions_out = []
    for cond in CONDITIONS:
        if cond not in by_condition:
            continue
        rs = by_condition[cond]
        n = len(rs)
        success_rate = sum(r["success"] for r in rs) / n
        seq_rate = sum(r["sequence_correct"] for r in rs) / n
        loose_rate = sum(r["loose_correct"] for r in rs) / n
        mean_total = sum(r["total_tokens"] for r in rs) / n
        mean_plan = sum(r["plan_tokens"] for r in rs) / n
        mean_oracle = sum(r["oracle_tokens"] for r in rs) / n
        mean_total_s = sum(r["total_elapsed_s"] for r in rs) / n
        mean_oracle_s = sum(r["oracle_elapsed_s"] for r in rs) / n
        mean_plan_s = sum(r["plan_elapsed_s"] for r in rs) / n
        mean_q = sum(len(r["oracle_questions"]) for r in rs) / n
        mean_mem = sum(r["memory_chars"] for r in rs) / n
        successful = sum(r["success"] for r in rs)
        tok_per_success = (mean_total * n / successful) if successful > 0 else float("inf")
        conditions_out.append({
            "condition": cond,
            "num_tasks": n,
            "success_rate": round(success_rate, 3),
            "sequence_correct_rate": round(seq_rate, 3),
            "loose_correct_rate": round(loose_rate, 3),
            "mean_total_tokens": round(mean_total, 1),
            "mean_plan_tokens": round(mean_plan, 1),
            "mean_oracle_tokens": round(mean_oracle, 1),
            "tokens_per_successful_task": round(tok_per_success, 0),
            "mean_oracle_questions_asked": round(mean_q, 1),
            "mean_memory_chars": round(mean_mem, 0),
            "mean_plan_elapsed_s": round(mean_plan_s, 3),
            "mean_oracle_elapsed_s": round(mean_oracle_s, 3),
            "mean_total_elapsed_s": round(mean_total_s, 3),
        })
    return {"conditions": conditions_out}


def print_summary(summary: dict) -> None:
    print("\n" + "="*70)
    print("E4 RESULTS: ALFWorld Knowledge Transfer")
    print("="*70)
    hdr = f"{'Condition':<22} {'Success':>8} {'Seq.':>7} {'Loose':>7} {'Tokens':>8} {'Tok/Suc':>10}"
    print(hdr)
    print("-"*65)
    for c in summary["conditions"]:
        tok_suc = c["tokens_per_successful_task"]
        tok_suc_str = f"{tok_suc:,.0f}" if tok_suc != float("inf") else "inf"
        print(
            f"{c['condition']:<22} "
            f"{c['success_rate']:>7.1%} "
            f"{c['sequence_correct_rate']:>7.1%} "
            f"{c['loose_correct_rate']:>7.1%} "
            f"{c['mean_total_tokens']:>8,.0f} "
            f"{tok_suc_str:>10}"
        )
    print("="*70)


if __name__ == "__main__":
    main()
