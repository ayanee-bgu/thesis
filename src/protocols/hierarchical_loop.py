"""
Hierarchical coordination protocol.

The Generalist acts as coordinator:
  Phase 0 (Planning):   Coordinator decomposes the task into per-specialist sub-tasks.
  Phase 1-N (Execution): Specialists work on their sub-tasks; coordinator synthesizes.
  Termination:          Coordinator declares a final answer OR max rounds hit.

Compared to RoundLoop (flat), this tests H2:
  hierarchical coordination reduces rounds-to-consensus on GAIA-style tasks
  (decomposable) but not on MATH (benefits from parallel reasoning).
"""

import json
from datetime import datetime
from pathlib import Path

import requests as http_requests

from src.agents.cooperative_agent import CooperativeAgent, OLLAMA_URL, OLLAMA_TIMEOUT

COORDINATOR_PLAN_PROMPT = """You are the coordinator of a multi-agent reasoning group.
Your specialists are: {specialist_roles}.

Analyse the task below and decompose it into one focused sub-task per specialist.
Each sub-task should play to that specialist's strengths:
  - reasoner: multi-step logical/mathematical reasoning
  - coder: computation, code execution, symbolic calculation
  - retriever: knowledge lookup, finding facts, information synthesis

Respond in this exact JSON format:
{{
  "decomposition": {{
    "<role>": "<sub-task description for that specialist>"
  }},
  "coordinator_note": "<brief plan summary>"
}}

Only include roles that have a meaningful sub-task. Omit roles if there is nothing relevant for them.
"""

COORDINATOR_SYNTHESIS_PROMPT = """You are the coordinator of a multi-agent reasoning group.
Based on the specialist findings below, produce the final answer.

Respond in this exact JSON format:
{{
  "synthesis": "<how you combined the specialist findings>",
  "answer": "<final answer>",
  "confidence": <float 0.0-1.0>
}}
"""


class HierarchicalLoop:
    def __init__(
        self,
        coordinator: CooperativeAgent,
        specialists: list[CooperativeAgent],
        shared_dir: Path,
        max_rounds: int = 3,
        memory_enabled: bool = True,
    ):
        self.coordinator = coordinator
        self.specialists = specialists
        self.shared_dir = Path(shared_dir)
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.max_rounds = max_rounds
        self.memory_enabled = memory_enabled
        self._agent_map = {a.role: a for a in [coordinator] + specialists}

        all_roles = list(self._agent_map.keys())
        for agent in self._agent_map.values():
            agent.set_peer_roles(all_roles)

    def run(self, task: str) -> dict:
        (self.shared_dir / "task.md").write_text(f"# Task\n\n{task}\n", encoding="utf-8")

        log = {
            "task": task,
            "started_at": datetime.now().isoformat(),
            "mode": "hierarchical",
            "memory_enabled": self.memory_enabled,
            "plan": None,
            "rounds": [],
            "synthesis": None,
            "final_answer": None,
            "total_rounds": 0,
        }

        # ── Phase 0: Planning ──────────────────────────────────────────────
        print("\n--- Planning Phase ---")
        plan = self._plan(task)
        log["plan"] = plan
        print(f"  Decomposition: {list(plan.get('decomposition', {}).keys())}")
        print(f"  Note: {plan.get('coordinator_note', '')[:100]}")
        self._flush(log)

        decomposition: dict[str, str] = plan.get("decomposition", {})

        # ── Phase 1-N: Specialist execution ───────────────────────────────
        peer_summaries: dict[str, str] = {}

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n--- Round {round_num} ---")
            round_results = []

            for specialist in self.specialists:
                sub_task = decomposition.get(specialist.role, task)
                peer_memories = self._fetch_memories(specialist, round_num) if self.memory_enabled else {}

                result = specialist.think(sub_task, round_num, peer_summaries, peer_memories)
                vote = specialist.vote()
                round_results.append({
                    **vote,
                    "sub_task": sub_task[:100],
                    "summary": result.get("summary", ""),
                    "used_tool": bool(result.get("tool_result")),
                })
                print(
                    f"  [{specialist.role}] answer='{vote['answer']}' conf={vote['confidence']:.2f}"
                    + (" [tool]" if result.get("tool_result") else "")
                )

            peer_summaries = {r["role"]: r["summary"] for r in round_results}

            log["rounds"].append({
                "round": round_num,
                "specialist_votes": [{k: v for k, v in r.items() if k != "sub_task"} for r in round_results],
            })
            self._flush(log)

        # ── Synthesis: Coordinator produces final answer ───────────────────
        print("\n--- Synthesis Phase ---")
        synthesis = self._synthesize(task, peer_summaries)
        log["synthesis"] = synthesis
        log["final_answer"] = synthesis.get("answer", "")
        log["total_rounds"] = self.max_rounds
        print(f"  Final answer: '{log['final_answer']}' (conf={synthesis.get('confidence', 0):.2f})")

        # Write coordinator's synthesis to its memory
        self.coordinator.memory_dir.mkdir(parents=True, exist_ok=True)
        (self.coordinator.memory_dir / "synthesis.md").write_text(
            f"# Final Synthesis\n\n{synthesis.get('synthesis', '')}\n\n"
            f"**Answer**: {synthesis.get('answer', '')}\n"
            f"**Confidence**: {synthesis.get('confidence', 0)}\n",
            encoding="utf-8",
        )
        self._flush(log)
        return log

    # ── Internal helpers ───────────────────────────────────────────────────

    def _plan(self, task: str) -> dict:
        specialist_roles = [s.role for s in self.specialists]
        system = COORDINATOR_PLAN_PROMPT.format(specialist_roles=specialist_roles)
        raw = self._call_ollama(self.coordinator.model, system, task)
        return self.coordinator._try_parse_json(raw) or {"decomposition": {}, "coordinator_note": raw[:200]}

    def _synthesize(self, task: str, specialist_summaries: dict[str, str]) -> dict:
        summaries_text = "\n".join(f"[{role}]: {s}" for role, s in specialist_summaries.items())
        user = f"ORIGINAL TASK:\n{task}\n\nSPECIALIST FINDINGS:\n{summaries_text}"
        raw = self._call_ollama(self.coordinator.model, COORDINATOR_SYNTHESIS_PROMPT, user)
        parsed = self.coordinator._try_parse_json(raw)
        if parsed:
            parsed["answer"] = str(parsed.get("answer", "")).strip()
            parsed["confidence"] = float(parsed.get("confidence", 0.5))
            return parsed
        return {"synthesis": raw, "answer": raw.split("\n")[-1][:100], "confidence": 0.3}

    def _fetch_memories(self, requesting: CooperativeAgent, round_num: int) -> dict[str, str]:
        memories = {}
        for role, agent in self._agent_map.items():
            if role != requesting.role:
                content = agent.read_own_memory()
                if content:
                    memories[role] = content
        return memories

    def _call_ollama(self, model: str, system: str, user: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        resp = http_requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _flush(self, log: dict):
        (self.shared_dir / "round_log.json").write_text(
            json.dumps(log, indent=2), encoding="utf-8"
        )
