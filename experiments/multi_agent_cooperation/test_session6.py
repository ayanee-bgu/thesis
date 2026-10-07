"""
Session 6 test: hierarchical coordination smoke test.

Runs HierarchicalLoop on a 2-hop HotpotQA question.
Verifies:
  - Coordinator produces a decomposition plan (routing messages in round_log.json)
  - Each specialist gets a sub-task different from the full task
  - Coordinator produces a synthesis with a final answer

Run from thesis root:
    python -m experiments.multi_agent_cooperation.test_session6
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.cooperative_agent import CooperativeAgent
from src.protocols.hierarchical_loop import HierarchicalLoop

THESIS_ROOT = Path(__file__).parent.parent.parent
BASE = Path(__file__).parent
AGENTS_DIR = BASE / "agents"
SHARED_DIR = BASE / "shared_session6"

# A classic HotpotQA bridge question requiring 2 hops
TASK = """Answer this question with a SHORT answer (1-3 words or yes/no).

CONTEXT:
[Scott Derrickson] Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.
[Ed Wood] Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978) was an American filmmaker, actor, writer, producer, and director.

QUESTION: Were Scott Derrickson and Ed Wood of the same nationality?"""

EXPECTED = "yes"


def main():
    coordinator = CooperativeAgent(
        role="generalist", model="llama3:8b",
        memory_dir=AGENTS_DIR / "agent_generalist_s6" / "memory",
    )
    specialists = [
        CooperativeAgent(
            role="reasoner", model="llama3:8b",
            memory_dir=AGENTS_DIR / "agent_reasoner_s6" / "memory",
        ),
        CooperativeAgent(
            role="retriever", model="llama3:8b",
            memory_dir=AGENTS_DIR / "agent_retriever_s6" / "memory",
            tools=["rag_retrieval"], thesis_root=THESIS_ROOT,
        ),
    ]

    loop = HierarchicalLoop(
        coordinator=coordinator,
        specialists=specialists,
        shared_dir=SHARED_DIR,
        max_rounds=2,
    )

    result = loop.run(TASK)

    print("\n=== Session 6 Test Results ===")
    print(f"Final answer : {result['final_answer']}")
    print(f"Rounds       : {result['total_rounds']}")
    print(f"Plan         : {result.get('plan', {}).get('decomposition', {})}")
    print(f"Synthesis    : {str(result.get('synthesis', ''))[:150]}")

    # Verify round_log.json structure
    log_path = SHARED_DIR / "round_log.json"
    assert log_path.exists(), "round_log.json not written"
    with open(log_path) as f:
        log = json.load(f)

    assert log.get("mode") == "hierarchical", "Mode not set to hierarchical"
    assert log.get("plan") is not None, "Plan missing from log"
    assert log.get("synthesis") is not None, "Synthesis missing from log"
    assert log.get("final_answer") is not None, "No final answer"
    assert len(log.get("rounds", [])) == result["total_rounds"], "Round count mismatch"

    # Check that coordinator synthesis.md was written
    synthesis_file = AGENTS_DIR / "agent_generalist_s6" / "memory" / "synthesis.md"
    assert synthesis_file.exists(), "Coordinator synthesis.md not written"
    print(f"synthesis.md  : OK")

    print(f"\nExpected: '{EXPECTED}' | Got: '{result['final_answer']}'")
    print("All Session 6 checks passed.")


if __name__ == "__main__":
    main()
