"""
Session 1 test: 2 agents on a hardcoded MATH problem.
Verifies the round loop runs and round_log.json is written correctly.

Run from the thesis root:
    python -m experiments.multi_agent_cooperation.test_session1
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.cooperative_agent import CooperativeAgent
from src.protocols.round_loop import RoundLoop

TASK = """
Solve this algebra problem step by step:

A train travels from city A to city B at 60 km/h and returns at 40 km/h.
What is the average speed for the entire round trip?

Express your answer as a single number in km/h (e.g. "48").
"""

SHARED_DIR = Path(__file__).parent / "shared"
AGENTS_DIR = Path(__file__).parent / "agents"


def main():
    # Using llama3:8b for both in Session 1 to keep the smoke test fast.
    # Real differentiated models (deepseek-r1:32b, etc.) are used in math_eval.py / gaia_eval.py.
    agents = [
        CooperativeAgent(
            role="reasoner",
            model="llama3:8b",
            memory_dir=AGENTS_DIR / "agent_reasoner" / "memory",
        ),
        CooperativeAgent(
            role="generalist",
            model="llama3:8b",
            memory_dir=AGENTS_DIR / "agent_generalist" / "memory",
        ),
    ]

    loop = RoundLoop(
        agents=agents,
        shared_dir=SHARED_DIR,
        max_rounds=3,
        consensus_threshold=0.7,
    )

    result = loop.run(TASK)

    log_path = SHARED_DIR / "round_log.json"
    assert log_path.exists(), "round_log.json was not written"

    with open(log_path) as f:
        log = json.load(f)

    print("\n=== Test Results ===")
    print(f"Final answer : {result['final_answer']}")
    print(f"Consensus    : {result['consensus_reached']}")
    print(f"Rounds taken : {result['total_rounds']}")
    print(f"Log written  : {log_path}")
    print(f"Rounds in log: {len(log['rounds'])}")

    # Sanity checks
    assert result["final_answer"] is not None, "No final answer produced"
    assert result["total_rounds"] >= 1, "No rounds ran"
    assert len(log["rounds"]) == result["total_rounds"], "Log round count mismatch"

    # Check memory files written
    for agent in agents:
        obs_files = list(agent.memory_dir.glob("observations_round_*.md"))
        beliefs_file = agent.memory_dir / "beliefs.md"
        assert obs_files, f"{agent.role}: no observation files written"
        assert beliefs_file.exists(), f"{agent.role}: beliefs.md not written"
        print(f"  [{agent.role}] memory files: {[f.name for f in obs_files]} + beliefs.md")

    print("\nAll assertions passed.")
    # Correct answer is 48 km/h (harmonic mean)
    print(f"Expected answer: 48 — got: {result['final_answer']}")


if __name__ == "__main__":
    main()
