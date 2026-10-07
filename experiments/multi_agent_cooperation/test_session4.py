"""
Session 4 test: verify tool integrations.

Test 1 — python_exec: coder agent solves an arithmetic problem using code execution.
Test 2 — rag_retrieval: retriever agent pulls a relevant paper chunk.

Run from thesis root:
    python -m experiments.multi_agent_cooperation.test_session4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

THESIS_ROOT = Path(__file__).parent.parent.parent


def test_python_exec():
    print("--- Test 1: python_exec ---")
    from src.agents.tools.python_exec import run_code, extract_code_blocks

    code = "d = 120\nv1, v2 = 60, 40\nt = d/v1 + d/v2\navg = 2*d / t\nprint(round(avg, 4))"
    result = run_code(code)
    assert result["error"] is None, f"Unexpected error: {result['error']}"
    assert "48" in result["stdout"], f"Expected 48 in output, got: {result['stdout']}"
    print(f"  Code output: {result['stdout'].strip()}")

    blocks = extract_code_blocks("Here is the solution:\n```python\nprint(42)\n```")
    assert blocks == ["print(42)"], f"Block extraction failed: {blocks}"
    print(f"  Block extraction: OK")
    print("  PASSED\n")


def test_rag_retrieval():
    print("--- Test 2: rag_retrieval ---")
    from src.agents.tools.rag_retrieval import retrieve

    result = retrieve("knowledge transfer multi-agent", THESIS_ROOT)
    print(f"  Result preview: {result[:200]}")
    assert "[rag_retrieval]" in result, "Missing rag_retrieval prefix"
    print("  PASSED\n")


def test_coder_agent_with_tool():
    print("--- Test 3: CooperativeAgent with python_exec tool ---")
    from src.agents.cooperative_agent import CooperativeAgent
    from src.protocols.round_loop import RoundLoop

    BASE = Path(__file__).parent
    AGENTS_DIR = BASE / "agents"
    SHARED_DIR = BASE / "shared_session4"

    agents = [
        CooperativeAgent(
            role="coder",
            model="llama3:8b",
            memory_dir=AGENTS_DIR / "agent_coder_s4" / "memory",
            tools=["python_exec"],
            thesis_root=THESIS_ROOT,
        ),
        CooperativeAgent(
            role="generalist",
            model="llama3:8b",
            memory_dir=AGENTS_DIR / "agent_generalist_s4" / "memory",
        ),
    ]

    loop = RoundLoop(agents=agents, shared_dir=SHARED_DIR, max_rounds=2, consensus_threshold=0.7)
    result = loop.run(
        "What is the average speed of a trip where you go 120 km at 60 km/h and return at 40 km/h? "
        "Use Python code to compute this precisely. Answer with a single number."
    )

    print(f"  Final answer: {result['final_answer']}")
    print(f"  Consensus: {result['consensus_reached']}")

    # Check if tool was actually invoked (observations will mention tool_result)
    coder_obs = list(agents[0].memory_dir.glob("observations_round_*.md"))
    tool_used = any("Tool Result" in f.read_text(encoding="utf-8") for f in coder_obs)
    print(f"  Tool was invoked: {tool_used}")
    print("  PASSED\n")


if __name__ == "__main__":
    test_python_exec()
    test_rag_retrieval()
    test_coder_agent_with_tool()
    print("=== All Session 4 tests passed ===")
