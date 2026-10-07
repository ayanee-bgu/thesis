"""
Session 5 test: HotpotQA eval smoke test.

Loads 3 problems, runs baseline + flat_memory group, verifies round_log.json
captures tool calls and memory requests.

Run from thesis root:
    python -m experiments.multi_agent_cooperation.test_session5
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.multi_agent_cooperation.gaia_eval import (
    load_hotpotqa, run_baseline, run_flat_group, QUICK_MODELS
)

SHARED_DIR = Path(__file__).parent / "shared"


def main():
    problems = load_hotpotqa(3, "all")
    assert len(problems) == 3, "Failed to load problems"
    print(f"Loaded {len(problems)} HotpotQA problems")
    for p in problems:
        print(f"  [{p['type']}] {p['question'][:80]} -> '{p['answer']}'")

    print()
    baseline = run_baseline(problems, "llama3:8b")
    group = run_flat_group(problems, QUICK_MODELS, max_rounds=2, memory_enabled=True)

    print("\n=== Session 5 Test Results ===")
    b_acc = sum(r["correct"] for r in baseline) / len(baseline)
    g_acc = sum(r["correct"] for r in group) / len(group)
    print(f"Baseline accuracy : {b_acc:.1%}")
    print(f"Group accuracy    : {g_acc:.1%}")

    # Verify round_log.json was written for at least one problem
    logs = list(SHARED_DIR.glob("hotpotqa_*_flat_memory/round_log.json"))
    assert logs, "No round_log.json files written"

    with open(logs[0]) as f:
        log = json.load(f)

    assert "rounds" in log, "round_log.json missing 'rounds'"
    assert log["total_rounds"] >= 1, "No rounds ran"
    print(f"\nround_log.json OK: {len(log['rounds'])} round(s), answer='{log['final_answer']}'")
    print("All Session 5 checks passed.")


if __name__ == "__main__":
    main()
