"""
Session 2: MATH benchmark evaluation.

Runs the multi-agent group (message-passing, no memory retrieval) on MATH dataset problems
and compares accuracy to a single-agent baseline.

Usage (from thesis root):
    python -m experiments.multi_agent_cooperation.math_eval [OPTIONS]

Options:
    --n-problems  INT     Number of problems per subject (default: 5, total = 5 × subjects)
    --level       INT     MATH difficulty level 1-5 (default: 1)
    --subjects    STR     Comma-separated subjects (default: all)
    --max-rounds  INT     Max coordination rounds (default: 3)
    --baseline-only       Run only the single-agent baseline, skip group
    --group-only          Run only the group, skip baseline
    --quick               Use llama3:8b for all agents (fast smoke test)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

from datasets import load_dataset

from src.agents.cooperative_agent import CooperativeAgent
from src.protocols.round_loop import RoundLoop
from src.utils.answer_norm import answers_match, extract_boxed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUBJECTS = [
    "algebra",
    "geometry",
    "number_theory",
    "counting_and_probability",
    "prealgebra",
]

# Domain → agent role that leads on this subject
DOMAIN_LEAD = {
    "algebra": "reasoner",
    "geometry": "reasoner",
    "number_theory": "reasoner",
    "counting_and_probability": "coder",
    "prealgebra": "generalist",
    "intermediate_algebra": "coder",
    "precalculus": "coder",
}

DIFFERENTIATED_MODELS = {
    "reasoner": "deepseek-r1:32b",
    "coder": "qwen2.5-coder:14b",      # 14b for speed; swap to 32b for full runs
    "retriever": "command-r:35b",
    "generalist": "llama3:8b",
}

QUICK_MODELS = {role: "llama3:8b" for role in DIFFERENTIATED_MODELS}

AGENTS_DIR = Path(__file__).parent / "agents"
SHARED_DIR = Path(__file__).parent / "shared"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "math"

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_math_problems(subjects: list[str], level: int, n_per_subject: int) -> list[dict]:
    print(f"Loading MATH dataset (level={level}, n_per_subject={n_per_subject})...")
    problems = []
    for subject in subjects:
        # EleutherAI/hendrycks_math uses the subject name as the config
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        subject_problems = [
            p for p in ds
            if int(p["level"].replace("Level ", "")) == level
        ]
        sampled = subject_problems[:n_per_subject]
        for p in sampled:
            problems.append({
                "subject": subject,
                "level": level,
                "problem": p["problem"],
                "ground_truth": extract_boxed(p["solution"]),
                "full_solution": p["solution"],
            })
        print(f"  {subject}: {len(sampled)} problems")

    print(f"Total: {len(problems)} problems\n")
    return problems


# ---------------------------------------------------------------------------
# Single-agent baseline
# ---------------------------------------------------------------------------

def run_baseline(problems: list[dict], model: str) -> list[dict]:
    import requests

    results = []
    print(f"=== Baseline: {model} ===")

    for i, p in enumerate(problems):
        prompt = (
            f"Solve this math problem step by step. "
            f"At the end, state your answer inside \\boxed{{}}.\n\n{p['problem']}"
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        t0 = time.time()
        resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=1800)
        resp.raise_for_status()
        elapsed = time.time() - t0

        raw_answer = resp.json()["message"]["content"]
        predicted = extract_boxed(raw_answer) or raw_answer.split("\n")[-1]
        correct = answers_match(predicted, p["ground_truth"])

        results.append({
            "subject": p["subject"],
            "level": p["level"],
            "problem": p["problem"][:120] + "...",
            "ground_truth": p["ground_truth"],
            "predicted": predicted,
            "correct": correct,
            "elapsed_s": round(elapsed, 1),
        })
        status = "OK" if correct else "XX"
        print(f"  [{i+1}/{len(problems)}] {p['subject']} | {status} | pred='{predicted}' gt='{p['ground_truth']}'")

    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"  Baseline accuracy: {accuracy:.1%}\n")
    return results


# ---------------------------------------------------------------------------
# Group evaluation
# ---------------------------------------------------------------------------

def build_agents(models: dict) -> list[CooperativeAgent]:
    return [
        CooperativeAgent(
            role=role,
            model=model,
            memory_dir=AGENTS_DIR / f"agent_{role}" / "memory",
        )
        for role, model in models.items()
    ]


def run_group(problems: list[dict], models: dict, max_rounds: int) -> list[dict]:
    results = []
    print(f"=== Group ({', '.join(f'{r}={m}' for r, m in models.items())}) ===")

    for i, p in enumerate(problems):
        # Clear agent memories between problems
        agents = build_agents(models)

        # Annotate task with domain lead hint
        lead = DOMAIN_LEAD.get(p["subject"], "reasoner")
        task = (
            f"[Domain: {p['subject']} | Lead specialist: {lead}]\n\n"
            f"Solve this problem. State your final answer as a plain number or expression "
            f"(no LaTeX boxed notation needed — just the value).\n\n"
            f"{p['problem']}"
        )

        loop = RoundLoop(
            agents=agents,
            shared_dir=SHARED_DIR / f"problem_{i}",
            max_rounds=max_rounds,
            consensus_threshold=0.7,
        )

        t0 = time.time()
        run_result = loop.run(task)
        elapsed = time.time() - t0

        predicted = run_result["final_answer"] or ""
        correct = answers_match(predicted, p["ground_truth"])

        results.append({
            "subject": p["subject"],
            "level": p["level"],
            "problem": p["problem"][:120] + "...",
            "ground_truth": p["ground_truth"],
            "predicted": predicted,
            "correct": correct,
            "consensus_reached": run_result["consensus_reached"],
            "rounds_taken": run_result["total_rounds"],
            "elapsed_s": round(elapsed, 1),
        })
        status = "OK" if correct else "XX"
        print(
            f"  [{i+1}/{len(problems)}] {p['subject']} | {status} | "
            f"pred='{predicted}' gt='{p['ground_truth']}' | "
            f"rounds={run_result['total_rounds']} consensus={run_result['consensus_reached']}"
        )

    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_rounds = sum(r["rounds_taken"] for r in results) / len(results)
    print(f"  Group accuracy: {accuracy:.1%} | avg rounds: {avg_rounds:.1f}\n")
    return results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(run_id: str, baseline: list[dict] | None, group: list[dict] | None, args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "baseline": {
            "results": baseline,
            "accuracy": sum(r["correct"] for r in baseline) / len(baseline) if baseline else None,
        } if baseline else None,
        "group": {
            "results": group,
            "accuracy": sum(r["correct"] for r in group) / len(group) if group else None,
            "avg_rounds": sum(r["rounds_taken"] for r in group) / len(group) if group else None,
            "consensus_rate": sum(r["consensus_reached"] for r in group) / len(group) if group else None,
        } if group else None,
    }
    path = RESULTS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Results saved to {path}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-problems", type=int, default=5)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--subjects", type=str, default=",".join(SUBJECTS))
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--group-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use llama3:8b for all agents")
    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",")]
    models = QUICK_MODELS if args.quick else DIFFERENTIATED_MODELS
    baseline_model = "llama3:8b" if args.quick else "deepseek-r1:32b"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.quick:
        run_id += "_quick"

    problems = load_math_problems(subjects, args.level, args.n_problems)

    baseline_results = None
    group_results = None

    if not args.group_only:
        baseline_results = run_baseline(problems, baseline_model)

    if not args.baseline_only:
        group_results = run_group(problems, models, args.max_rounds)

    summary = save_results(run_id, baseline_results, group_results, args)

    # Print comparison if both ran
    if baseline_results and group_results:
        b_acc = summary["baseline"]["accuracy"]
        g_acc = summary["group"]["accuracy"]
        delta = g_acc - b_acc
        print(f"\n=== Summary ===")
        print(f"Baseline accuracy : {b_acc:.1%}")
        print(f"Group accuracy    : {g_acc:.1%}  ({'+' if delta >= 0 else ''}{delta:.1%})")
        print(f"Avg rounds        : {summary['group']['avg_rounds']:.1f}")
        print(f"Consensus rate    : {summary['group']['consensus_rate']:.1%}")


if __name__ == "__main__":
    main()
