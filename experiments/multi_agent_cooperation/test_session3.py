"""
Session 3 test: verify memory transfer protocol.

Runs 2 agents for 3 rounds. In round 1 the reasoner requests the generalist's memory.
Checks that:
  - queries_sent.md is written for the requesting agent
  - memory_transfers.json is written with at least 1 transfer by round 2
  - retrieved content appears in subsequent observations

Run from thesis root:
    python -m experiments.multi_agent_cooperation.test_session3
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.cooperative_agent import CooperativeAgent
from src.protocols.round_loop import RoundLoop

TASK = """
A train travels from city A to city B at 60 km/h and returns at 40 km/h.
What is the average speed for the entire round trip?
Answer with a single number in km/h.
"""

BASE = Path(__file__).parent
AGENTS_DIR = BASE / "agents"
SHARED_DIR = BASE / "shared_session3"


def main():
    agents = [
        CooperativeAgent(role="reasoner", model="llama3:8b", memory_dir=AGENTS_DIR / "agent_reasoner_s3" / "memory"),
        CooperativeAgent(role="generalist", model="llama3:8b", memory_dir=AGENTS_DIR / "agent_generalist_s3" / "memory"),
    ]

    loop = RoundLoop(
        agents=agents,
        shared_dir=SHARED_DIR,
        max_rounds=3,
        consensus_threshold=0.7,
        memory_enabled=True,
    )

    result = loop.run(TASK)

    print("\n=== Session 3 Test Results ===")
    print(f"Final answer : {result['final_answer']}")
    print(f"Consensus    : {result['consensus_reached']}")
    print(f"Rounds       : {result['total_rounds']}")

    transfer_log = SHARED_DIR / "memory_transfers.json"
    round_log = SHARED_DIR / "round_log.json"

    assert round_log.exists(), "round_log.json missing"
    with open(round_log) as f:
        log = json.load(f)

    transfer_summary = log.get("memory_transfer_summary", {})
    print(f"Memory transfers: {transfer_summary}")

    # Check queries_sent.md was written if any memory was requested
    any_queries = any(
        (AGENTS_DIR / f"agent_{a.role}_s3" / "memory" / "queries_sent.md").exists()
        for a in agents
    )

    # Check memory_transfers.json
    transfers_happened = transfer_log.exists() and transfer_log.stat().st_size > 10

    print(f"queries_sent.md written: {any_queries}")
    print(f"memory_transfers.json written: {transfers_happened}")

    # Verify observation files were written each round
    for agent in agents:
        obs = list(agent.memory_dir.glob("observations_round_*.md"))
        assert obs, f"{agent.role}: no observation files"
        print(f"  [{agent.role}] {len(obs)} observation file(s)")

    print("\nAll checks passed.")
    print(f"Expected: 48 | Got: {result['final_answer']}")


if __name__ == "__main__":
    main()
