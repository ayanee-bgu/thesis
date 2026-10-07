"""
Session 5: Multi-step reasoning benchmark evaluation (HotpotQA).

Uses HotpotQA (distractor setting) as a stand-in for GAIA until HF access is granted.
HotpotQA tests multi-hop reasoning: each question requires chaining 2+ facts across
different Wikipedia passages — exactly the kind of task where specialist retrieval and
reasoning agents should outperform single-model baselines.

To swap in GAIA once you have HF access:
  1. Set HF_TOKEN environment variable
  2. Run with --dataset gaia

Usage (from thesis root):
    python -m experiments.multi_agent_cooperation.gaia_eval [OPTIONS]

Options:
    --n-problems  INT         Number of problems (default: 10)
    --dataset     STR         "hotpotqa" or "gaia" (default: hotpotqa)
    --q-type      STR         HotpotQA type: "bridge", "comparison", or "all" (default: all)
    --max-rounds  INT         Max coordination rounds (default: 3)
    --condition   STR         flat_msgpass | flat_memory | hierarchical (default: flat_memory)
    --baseline-only           Run only single-agent baseline
    --group-only              Skip baseline
    --quick                   Use llama3:8b for all agents
"""

import argparse
import json
import os
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
from src.protocols.hierarchical_loop import HierarchicalLoop

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AGENTS_DIR = Path(__file__).parent / "agents"
SHARED_DIR = Path(__file__).parent / "shared"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "hotpotqa"
THESIS_ROOT = Path(__file__).parent.parent.parent

DIFFERENTIATED_MODELS = {
    "reasoner":   "deepseek-r1:32b",
    "coder":      "qwen2.5-coder:14b",
    "retriever":  "command-r:35b",
    "generalist": "llama3:8b",
}
QUICK_MODELS = {role: "llama3:8b" for role in DIFFERENTIATED_MODELS}

AGENT_TOOLS = {
    "coder":     ["python_exec"],
    "retriever": ["rag_retrieval"],
}

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_hotpotqa(n: int, q_type: str) -> list[dict]:
    print(f"Loading HotpotQA (n={n}, type={q_type})...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    filtered = [p for p in ds if q_type == "all" or p["type"] == q_type]
    sampled = filtered[:n]
    problems = []
    for p in sampled:
        problems.append({
            "id": p["id"],
            "question": p["question"],
            "answer": p["answer"],
            "type": p["type"],
            "context": _format_hotpotqa_context(p["context"]),
        })
    print(f"Loaded {len(problems)} problems\n")
    return problems


def _format_hotpotqa_context(context: dict) -> str:
    # context = {"title": [str, ...], "sentences": [[str, ...], ...]}
    titles = context.get("title", [])
    sentences_list = context.get("sentences", [])
    parts = []
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(sentences)
        parts.append(f"[{title}] {text}")
    return "\n\n".join(parts)


def load_gaia(n: int) -> list[dict]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN to load GAIA dataset.")
    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation", token=token)
    level1 = [p for p in ds if p.get("Level") == 1][:n]
    return [{"id": p["task_id"], "question": p["Question"], "answer": p["Final answer"], "type": "gaia", "context": ""} for p in level1]


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def answers_match(predicted: str, ground_truth: str) -> bool:
    def norm(s: str) -> str:
        return s.strip().lower().strip(".,!?")
    p, g = norm(predicted), norm(ground_truth)
    return p == g or g in p or p in g


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def run_baseline(problems: list[dict], model: str) -> list[dict]:
    import requests
    results = []
    print(f"=== Baseline: {model} ===")
    for i, p in enumerate(problems):
        prompt = (
            f"Answer the following question using the provided context. "
            f"Give a short direct answer (1-3 words or yes/no).\n\n"
            f"CONTEXT:\n{p['context']}\n\nQUESTION: {p['question']}"
        )
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"temperature": 0.1}}
        t0 = time.time()
        resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=1800)
        resp.raise_for_status()
        elapsed = time.time() - t0
        raw = resp.json()["message"]["content"].strip()
        # Extract short answer from response
        predicted = raw.split("\n")[0][:100]
        correct = answers_match(predicted, p["answer"])
        results.append({
            "id": p["id"], "type": p["type"], "question": p["question"][:100],
            "ground_truth": p["answer"], "predicted": predicted,
            "correct": correct, "elapsed_s": round(elapsed, 1),
        })
        status = "OK" if correct else "XX"
        print(f"  [{i+1}/{len(problems)}] {p['type']} | {status} | pred='{predicted}' gt='{p['answer']}'")
    acc = sum(r["correct"] for r in results) / len(results)
    print(f"  Baseline accuracy: {acc:.1%}\n")
    return results


# ---------------------------------------------------------------------------
# Agent builders
# ---------------------------------------------------------------------------

def build_agents(models: dict, use_tools: bool, suffix: str = "") -> list[CooperativeAgent]:
    agents = []
    for role, model in models.items():
        tools = AGENT_TOOLS.get(role, []) if use_tools else []
        agents.append(CooperativeAgent(
            role=role, model=model,
            memory_dir=AGENTS_DIR / f"agent_{role}{suffix}" / "memory",
            tools=tools, thesis_root=THESIS_ROOT,
        ))
    return agents


# ---------------------------------------------------------------------------
# Group evaluation (flat)
# ---------------------------------------------------------------------------

def run_flat_group(problems: list[dict], models: dict, max_rounds: int, memory_enabled: bool) -> list[dict]:
    label = "flat_memory" if memory_enabled else "flat_msgpass"
    print(f"=== Group [{label}] ===")
    results = []
    for i, p in enumerate(problems):
        suffix = f"_g{i}"
        agents = build_agents(models, use_tools=True, suffix=suffix)
        task = (
            f"Answer this question with a SHORT answer (1-3 words or yes/no).\n\n"
            f"CONTEXT (use this to answer):\n{p['context']}\n\n"
            f"QUESTION: {p['question']}"
        )
        loop = RoundLoop(agents=agents, shared_dir=SHARED_DIR / f"hotpotqa_{i}_{label}",
                         max_rounds=max_rounds, consensus_threshold=0.7, memory_enabled=memory_enabled)
        t0 = time.time()
        run_result = loop.run(task)
        elapsed = time.time() - t0
        predicted = str(run_result["final_answer"] or "")
        correct = answers_match(predicted, p["answer"])
        results.append({
            "id": p["id"], "type": p["type"], "question": p["question"][:100],
            "ground_truth": p["answer"], "predicted": predicted,
            "correct": correct, "consensus_reached": run_result["consensus_reached"],
            "rounds_taken": run_result["total_rounds"],
            "memory_transfers": run_result.get("memory_transfer_summary", {}).get("total_transfers", 0),
            "elapsed_s": round(elapsed, 1),
        })
        status = "OK" if correct else "XX"
        print(f"  [{i+1}/{len(problems)}] {p['type']} | {status} | pred='{predicted}' gt='{p['answer']}' | rounds={run_result['total_rounds']}")
    acc = sum(r["correct"] for r in results) / len(results)
    avg_rounds = sum(r["rounds_taken"] for r in results) / len(results)
    print(f"  Accuracy: {acc:.1%} | avg rounds: {avg_rounds:.1f}\n")
    return results


# ---------------------------------------------------------------------------
# Hierarchical evaluation
# ---------------------------------------------------------------------------

def run_hierarchical_group(problems: list[dict], models: dict, max_rounds: int) -> list[dict]:
    print("=== Group [hierarchical] ===")
    results = []
    for i, p in enumerate(problems):
        suffix = f"_h{i}"
        all_agents = build_agents(models, use_tools=True, suffix=suffix)
        coordinator = next(a for a in all_agents if a.role == "generalist")
        specialists = [a for a in all_agents if a.role != "generalist"]

        task = (
            f"Answer this question with a SHORT answer (1-3 words or yes/no).\n\n"
            f"CONTEXT:\n{p['context']}\n\nQUESTION: {p['question']}"
        )
        loop = HierarchicalLoop(coordinator=coordinator, specialists=specialists,
                                shared_dir=SHARED_DIR / f"hotpotqa_{i}_hierarchical",
                                max_rounds=max_rounds)
        t0 = time.time()
        run_result = loop.run(task)
        elapsed = time.time() - t0
        predicted = str(run_result["final_answer"] or "")
        correct = answers_match(predicted, p["answer"])
        results.append({
            "id": p["id"], "type": p["type"], "question": p["question"][:100],
            "ground_truth": p["answer"], "predicted": predicted,
            "correct": correct, "rounds_taken": run_result["total_rounds"],
            "elapsed_s": round(elapsed, 1),
        })
        status = "OK" if correct else "XX"
        print(f"  [{i+1}/{len(problems)}] {p['type']} | {status} | pred='{predicted}' gt='{p['answer']}'")
    acc = sum(r["correct"] for r in results) / len(results)
    print(f"  Accuracy: {acc:.1%}\n")
    return results


# ---------------------------------------------------------------------------
# Save & main
# ---------------------------------------------------------------------------

def save_results(run_id: str, baseline, flat_msgpass, flat_memory, hierarchical, args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def summary(results):
        if not results:
            return None
        return {
            "results": results,
            "accuracy": sum(r["correct"] for r in results) / len(results),
            "avg_rounds": sum(r.get("rounds_taken", 0) for r in results) / len(results),
            "consensus_rate": sum(r.get("consensus_reached", False) for r in results) / len(results),
        }

    out = {
        "run_id": run_id, "timestamp": datetime.now().isoformat(), "config": vars(args),
        "baseline": summary(baseline),
        "flat_msgpass": summary(flat_msgpass),
        "flat_memory": summary(flat_memory),
        "hierarchical": summary(hierarchical),
    }
    path = RESULTS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Results saved to {path}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-problems", type=int, default=10)
    parser.add_argument("--dataset", choices=["hotpotqa", "gaia"], default="hotpotqa")
    parser.add_argument("--q-type", choices=["bridge", "comparison", "all"], default="all")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--condition", choices=["flat_msgpass", "flat_memory", "hierarchical", "all"], default="flat_memory")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--group-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    models = QUICK_MODELS if args.quick else DIFFERENTIATED_MODELS
    baseline_model = "llama3:8b" if args.quick else "deepseek-r1:32b"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.quick:
        run_id += "_quick"

    if args.dataset == "hotpotqa":
        problems = load_hotpotqa(args.n_problems, args.q_type)
    else:
        problems = load_gaia(args.n_problems)

    baseline_results = flat_msgpass_results = flat_memory_results = hierarchical_results = None

    if not args.group_only:
        baseline_results = run_baseline(problems, baseline_model)

    if not args.baseline_only:
        run_all = args.condition == "all"
        if run_all or args.condition == "flat_msgpass":
            flat_msgpass_results = run_flat_group(problems, models, args.max_rounds, memory_enabled=False)
        if run_all or args.condition == "flat_memory":
            flat_memory_results = run_flat_group(problems, models, args.max_rounds, memory_enabled=True)
        if run_all or args.condition == "hierarchical":
            hierarchical_results = run_hierarchical_group(problems, models, args.max_rounds)

    out = save_results(run_id, baseline_results, flat_msgpass_results, flat_memory_results, hierarchical_results, args)

    print("\n=== Summary ===")
    for condition, key in [("Baseline", "baseline"), ("Flat (msgpass)", "flat_msgpass"),
                           ("Flat (memory)", "flat_memory"), ("Hierarchical", "hierarchical")]:
        s = out.get(key)
        if s:
            print(f"  {condition:<18}: {s['accuracy']:.1%}")


if __name__ == "__main__":
    main()
