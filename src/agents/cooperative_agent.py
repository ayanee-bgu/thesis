import json
import re
import requests
from pathlib import Path


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 1800  # 30 min — deepseek-r1:32b needs time on complex Level 3+ problems

SYSTEM_PROMPT = """You are a {role} agent in a cooperative reasoning group. Your speciality is {speciality}.

Each round you will receive a task and summaries from peer agents. You must:
1. Reason carefully using your speciality
2. State your best answer
3. Assign a confidence score (0.0-1.0)
4. Write a short summary to broadcast to peers

Always respond in this exact JSON format:
{{
  "reasoning": "<your step-by-step reasoning>",
  "answer": "<your final answer>",
  "confidence": <float 0.0-1.0>,
  "summary": "<2-3 sentences summarizing your key findings for peers>",
  "memory_requests": ["<role of agent whose memory you want>"]
}}

memory_requests should be an empty list [] if you don't need other agents' memories.
Only request memory when you are uncertain and believe another specialist has relevant knowledge."""

SYSTEM_PROMPT_WITH_TOOLS = SYSTEM_PROMPT + """

You have access to the following tools:
{tool_descriptions}

To use a tool, add a "tool_call" field to your JSON BEFORE providing your final answer:
{{
  "reasoning": "<your reasoning so far>",
  "tool_call": {{"name": "<tool_name>", "input": "<tool input>"}},
  "answer": "",
  "confidence": 0.0,
  "summary": "",
  "memory_requests": []
}}
Your answer and confidence will be filled in after seeing the tool result.
Only use a tool when it will genuinely help — do not call tools for simple reasoning."""

TOOL_DESCRIPTIONS = {
    "python_exec": "Execute Python code and get stdout. Input: a Python code string. Use for arithmetic, symbolic computation, or any calculation that benefits from precision.",
    "rag_retrieval": "Search the local knowledge base of paper summaries. Input: a natural language query. Use when you need to recall prior research relevant to the task.",
}

ROLES = {
    "reasoner": "multi-step logical and mathematical reasoning",
    "coder": "symbolic computation and code-based problem solving",
    "retriever": "knowledge lookup and information synthesis",
    "generalist": "synthesis, cross-domain reasoning, and consensus building",
}


class CooperativeAgent:
    def __init__(self, role: str, model: str, memory_dir: Path, tools: list[str] | None = None, thesis_root: Path | None = None):
        self.role = role
        self.model = model
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.tools = tools or []
        self.thesis_root = Path(thesis_root) if thesis_root else Path(__file__).parent.parent.parent
        self._current_answer = None
        self._current_confidence = 0.0

    def set_peer_roles(self, roles: list[str]):
        """Called by RoundLoop so agents know which peer roles exist for memory requests."""
        self._peer_roles = [r for r in roles if r != self.role]

    def think(self, task: str, round_num: int, peer_summaries: dict[str, str], peer_memories: dict[str, str]) -> dict:
        context_parts = [f"TASK:\n{task}"]

        if peer_summaries:
            summaries_text = "\n".join(f"[{role}]: {s}" for role, s in peer_summaries.items())
            context_parts.append(f"PEER SUMMARIES FROM PREVIOUS ROUND:\n{summaries_text}")

        if peer_memories:
            memories_text = "\n\n".join(f"=== {role.upper()} MEMORY ===\n{m}" for role, m in peer_memories.items())
            context_parts.append(f"RETRIEVED PEER MEMORIES:\n{memories_text}")

        user_message = "\n\n".join(context_parts)
        system = self._build_system_prompt()

        response = self._call_ollama(system, user_message)
        parsed = self._parse_response(response)

        # Tool-use loop: if agent requested a tool, execute it and re-call
        tool_result = self._maybe_execute_tool(parsed)
        if tool_result is not None:
            follow_up = user_message + f"\n\nTOOL RESULT:\n{tool_result}\n\nNow provide your final answer in the required JSON format."
            response2 = self._call_ollama(system, follow_up)
            parsed = self._parse_response(response2)
            parsed["tool_result"] = tool_result

        self._current_answer = parsed.get("answer", "")
        self._current_confidence = float(parsed.get("confidence", 0.0))
        self._write_memory(round_num, parsed)
        return parsed

    def vote(self) -> dict:
        return {"role": self.role, "answer": self._current_answer, "confidence": self._current_confidence}

    def read_own_memory(self) -> str:
        files = sorted(self.memory_dir.glob("*.md"))
        if not files:
            return ""
        return "\n\n".join(f.read_text(encoding="utf-8") for f in files)

    def clear_memory(self):
        for f in self.memory_dir.glob("*.md"):
            f.unlink()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        peer_note = ""
        peer_roles = getattr(self, "_peer_roles", [])
        if peer_roles:
            peer_note = f"\n\nAvailable peer roles for memory_requests: {peer_roles}. Only use these exact role names."

        if self.tools:
            descs = "\n".join(f"- {t}: {TOOL_DESCRIPTIONS.get(t, t)}" for t in self.tools)
            return SYSTEM_PROMPT_WITH_TOOLS.format(
                role=self.role,
                speciality=ROLES.get(self.role, self.role),
                tool_descriptions=descs,
            ) + peer_note
        return SYSTEM_PROMPT.format(role=self.role, speciality=ROLES.get(self.role, self.role)) + peer_note

    def _maybe_execute_tool(self, parsed: dict) -> str | None:
        tool_call = parsed.get("tool_call")
        if not tool_call or not isinstance(tool_call, dict):
            return None

        name = tool_call.get("name", "")
        inp = tool_call.get("input", "")

        if name == "python_exec" and "python_exec" in self.tools:
            from src.agents.tools.python_exec import run_code
            result = run_code(inp)
            if result["error"]:
                return f"[python_exec] Error: {result['error']}"
            output = result["stdout"].strip() or f"(exit code {result['returncode']})"
            return f"[python_exec] Output:\n{output}"

        if name == "rag_retrieval" and "rag_retrieval" in self.tools:
            from src.agents.tools.rag_retrieval import retrieve
            return retrieve(inp, self.thesis_root)

        return f"[tool] Unknown tool '{name}' or tool not available to this agent."

    def _write_memory(self, round_num: int, parsed: dict):
        obs_path = self.memory_dir / f"observations_round_{round_num}.md"
        tool_note = f"\n\n**Tool Result**: {parsed.get('tool_result', '')}" if parsed.get("tool_result") else ""
        obs_path.write_text(
            f"# Round {round_num} Observations\n\n"
            f"**Reasoning**: {parsed.get('reasoning', '')}\n\n"
            f"**Answer**: {parsed.get('answer', '')}\n\n"
            f"**Confidence**: {parsed.get('confidence', 0.0)}"
            f"{tool_note}\n",
            encoding="utf-8",
        )

        beliefs_path = self.memory_dir / "beliefs.md"
        beliefs_path.write_text(
            f"# Current Beliefs (updated round {round_num})\n\n"
            f"**Answer**: {parsed.get('answer', '')}\n\n"
            f"**Confidence**: {parsed.get('confidence', 0.0)}\n\n"
            f"**Summary**: {parsed.get('summary', '')}\n",
            encoding="utf-8",
        )

        requests_list = parsed.get("memory_requests", [])
        if requests_list:
            queries_path = self.memory_dir / "queries_sent.md"
            existing = queries_path.read_text(encoding="utf-8") if queries_path.exists() else "# Queries Sent\n\n"
            queries_path.write_text(
                existing + f"- Round {round_num}: requested memory from {requests_list}\n",
                encoding="utf-8",
            )

    def _call_ollama(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.3},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _parse_response(self, response: str) -> dict:
        # Try increasingly lenient JSON extraction
        parsed = self._try_parse_json(response)
        if parsed:
            # Normalise answer to string and confidence to float
            parsed["answer"] = str(parsed.get("answer", "")).strip()
            parsed["confidence"] = float(parsed.get("confidence", 0.3))
            parsed.setdefault("reasoning", "")
            parsed.setdefault("summary", parsed["answer"][:200])
            parsed.setdefault("memory_requests", [])
            return parsed

        # Regex fallback — handles both "answer": "text" and "answer": 42
        answer = re.search(r'"answer"\s*:\s*(?:"([^"]*)"|(-?[\d.]+))', response)
        confidence = re.search(r'"confidence"\s*:\s*([\d.]+)', response)
        summary = re.search(r'"summary"\s*:\s*"([^"]*)"', response)
        ans_val = (answer.group(1) or answer.group(2)) if answer else ""
        return {
            "reasoning": response,
            "answer": ans_val.strip() if ans_val else "",
            "confidence": float(confidence.group(1)) if confidence else 0.3,
            "summary": summary.group(1) if summary else response[:200],
            "memory_requests": [],
        }

    def _try_parse_json(self, text: str) -> dict | None:
        """Try to extract and parse JSON from model output, handling common formatting issues."""
        # Find the outermost { ... } block by tracking brace depth
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Model sometimes emits literal newlines inside strings — try sanitising
                        sanitised = re.sub(r'(?<!\\)\n', r'\\n', candidate)
                        try:
                            return json.loads(sanitised)
                        except json.JSONDecodeError:
                            return None
        return None
