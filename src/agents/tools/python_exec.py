"""
Safe Python code execution tool for the Coder agent.
Runs code in a subprocess with a timeout and captures stdout.
"""

import re
import subprocess
import sys
import textwrap


TIMEOUT_S = 15
MAX_OUTPUT_CHARS = 2000


def extract_code_blocks(text: str) -> list[str]:
    """Pull all ```python ... ``` blocks from a model response."""
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text)
    # Also accept bare indented blocks if no fenced blocks found
    if not blocks:
        indented = re.findall(r"(?m)^((?:    .+\n?)+)", text)
        blocks = [textwrap.dedent(b) for b in indented]
    return [b.strip() for b in blocks if b.strip()]


def run_code(code: str) -> dict:
    """
    Execute code string and return:
      {"stdout": str, "stderr": str, "returncode": int, "error": str|None}
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        stdout = result.stdout[:MAX_OUTPUT_CHARS]
        stderr = result.stderr[:MAX_OUTPUT_CHARS]
        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "", "returncode": -1, "error": f"Timed out after {TIMEOUT_S}s"}
    except Exception as e:
        return {"stdout": "", "stderr": "", "returncode": -1, "error": str(e)}


def execute_from_response(response_text: str) -> str | None:
    """
    Find the first code block in response_text, execute it, and return a
    formatted result string to inject back into the model context.
    Returns None if no code block found.
    """
    blocks = extract_code_blocks(response_text)
    if not blocks:
        return None

    result = run_code(blocks[0])
    if result["error"]:
        return f"[python_exec] Error: {result['error']}"
    if result["returncode"] != 0:
        return f"[python_exec] Exit {result['returncode']}:\nstdout: {result['stdout']}\nstderr: {result['stderr']}"
    output = result["stdout"].strip() or "(no output)"
    return f"[python_exec] Output:\n{output}"
