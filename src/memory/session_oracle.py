"""
SessionOracle: an LLM loaded with teacher trajectories that answers
student agent questions on demand (stateless per question).
"""

from __future__ import annotations

import time
from typing import Optional

import requests

ORACLE_SYSTEM_TEMPLATE = """\
You are a session oracle. You have access to records of a previous agent's \
travel planning sessions. Answer the student agent's question based ONLY on \
those records. Be concise and specific — 2-5 sentences maximum. \
If the records do not contain relevant information, say so explicitly.

TEACHER SESSION RECORDS:
{records}"""


def _format_trajectories(trajectories: list[str], budget: int) -> str:
    parts = []
    used = 0
    for i, traj in enumerate(trajectories):
        header = f"\n[TEACHER TRAJECTORY {i+1}]\n"
        chunk = header + traj
        if used + len(chunk) > budget:
            remaining = budget - used - len(header)
            if remaining > 100:
                parts.append(header + traj[:remaining] + "\n...[truncated]")
            break
        parts.append(chunk)
        used += len(chunk)
    return "".join(parts)


class SessionOracle:
    """
    Wraps a set of teacher trajectories and answers student questions via LLM.

    Each ask() is a stateless Ollama call — the oracle has no memory of prior
    questions. This forces the student to ask well-formed, targeted questions
    and prevents gaming the question budget with one broad "tell me everything".
    """

    def __init__(
        self,
        trajectories: list[str],
        model: str,
        ollama_url: str,
        timeout: int = 120,
        context_budget: int = 8000,
    ):
        self._trajectories = trajectories
        self._model = model
        self._ollama_url = ollama_url
        self._timeout = timeout
        self._context_budget = context_budget
        self._log: list[dict] = []
        self._records = _format_trajectories(trajectories, context_budget)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ask(self, question: str) -> tuple[str, int]:
        """Ask the oracle one question. Returns (answer, tokens_used)."""
        system = ORACLE_SYSTEM_TEMPLATE.format(records=self._records)
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }
        _CONNECT_TIMEOUT = 15
        _READ_TIMEOUT    = 900
        _MAX_RETRIES     = 3
        _RETRY_BACKOFF   = 30

        t0 = time.time()
        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self._ollama_url, json=payload,
                    timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                elapsed = round(time.time() - t0, 2)
                self._log.append({"question": question, "answer": answer, "tokens": tokens, "elapsed_s": elapsed})
                return answer, tokens
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF)
        elapsed = round(time.time() - t0, 2)
        self._log.append({"question": question, "answer": "[oracle error]", "tokens": 0, "elapsed_s": elapsed})
        raise last_exc

    def ask_mock(self, question: str) -> tuple[str, int]:
        """Bypass Ollama — returns a stub answer for structural testing."""
        answer = f"[MOCK ORACLE] Based on teacher records: approach the task step by step. (Q: {question[:60]})"
        self._log.append({"question": question, "answer": answer, "tokens": 0, "elapsed_s": 0.0})
        return answer, 0

    def format_all_trajectories(self) -> str:
        """Return all trajectories formatted as a memory block (for full_handoff condition)."""
        if not self._records:
            return ""
        return f"[TEACHER SESSION RECORDS]\n{self._records}\n[END TEACHER SESSION RECORDS]\n"

    @property
    def log(self) -> list[dict]:
        return list(self._log)

    @property
    def num_trajectories(self) -> int:
        return len(self._trajectories)
