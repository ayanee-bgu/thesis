import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.agents.cooperative_agent import CooperativeAgent
from src.protocols.memory_transfer import MemoryTransfer


class RoundLoop:
    def __init__(
        self,
        agents: list[CooperativeAgent],
        shared_dir: Path,
        max_rounds: int = 5,
        consensus_threshold: float = 0.7,
        memory_enabled: bool = True,   # False = message-passing only (Session 2 baseline condition)
    ):
        self.agents = agents
        self.shared_dir = Path(shared_dir)
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.memory_enabled = memory_enabled
        self._agent_map = {a.role: a for a in agents}
        self._transfer = MemoryTransfer(self.shared_dir / "memory_transfers.json")
        # Inform each agent of its peers so memory_requests use valid role names
        all_roles = [a.role for a in agents]
        for agent in agents:
            agent.set_peer_roles(all_roles)

    def run(self, task: str) -> dict:
        (self.shared_dir / "task.md").write_text(f"# Task\n\n{task}\n", encoding="utf-8")

        log = {
            "task": task,
            "started_at": datetime.now().isoformat(),
            "memory_enabled": self.memory_enabled,
            "rounds": [],
            "final_answer": None,
            "consensus_reached": False,
            "total_rounds": 0,
            "memory_transfer_summary": {},
        }

        peer_summaries: dict[str, str] = {}
        # Carry memory requests from the previous round so agents can prefetch
        pending_requests: dict[str, list[str]] = {a.role: [] for a in self.agents}

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n--- Round {round_num} ---")
            round_results = []
            next_requests: dict[str, list[str]] = {}

            for agent in self.agents:
                # Fetch peer memories requested last round (empty in round 1)
                peer_memories: dict[str, str] = {}
                if self.memory_enabled:
                    for requested_role in pending_requests.get(agent.role, []):
                        provider = self._agent_map.get(requested_role)
                        if provider and provider is not agent:
                            content = self._transfer.fetch(agent.role, provider, round_num)
                            if content:
                                peer_memories[requested_role] = content

                result = agent.think(task, round_num, peer_summaries, peer_memories)
                next_requests[agent.role] = result.get("memory_requests", [])

                vote = agent.vote()
                round_results.append({
                    **vote,
                    "summary": result.get("summary", ""),
                    "memory_requests": next_requests[agent.role],
                    "used_tool": bool(result.get("tool_result")),
                })
                print(
                    f"  [{agent.role}] answer='{vote['answer']}' "
                    f"conf={vote['confidence']:.2f}"
                    + (f" [tool]" if result.get("tool_result") else "")
                    + (f" [mem-req: {next_requests[agent.role]}]" if next_requests[agent.role] else "")
                )

            pending_requests = next_requests
            peer_summaries = {r["role"]: r["summary"] for r in round_results}

            votes = [{"role": r["role"], "answer": r["answer"], "confidence": r["confidence"]} for r in round_results]
            consensus, winning_answer = self._check_consensus(votes)

            log["rounds"].append({
                "round": round_num,
                "votes": votes,
                "consensus": consensus,
                "winning_answer": winning_answer,
                "memory_transfers": self._transfer.summary()["total_transfers"],
            })
            self._flush_log(log)

            if consensus:
                print(f"\nConsensus in round {round_num}: '{winning_answer}'")
                log["final_answer"] = winning_answer
                log["consensus_reached"] = True
                log["total_rounds"] = round_num
                log["memory_transfer_summary"] = self._transfer.summary()
                self._flush_log(log)
                return log

        final = self._tiebreak(log["rounds"][-1]["votes"])
        print(f"\nMax rounds reached. Tie-break: '{final}'")
        log["final_answer"] = final
        log["total_rounds"] = self.max_rounds
        log["memory_transfer_summary"] = self._transfer.summary()
        self._flush_log(log)
        return log

    def _check_consensus(self, votes: list[dict]) -> tuple[bool, str | None]:
        if not votes:
            return False, None
        # Never reach consensus on an empty answer
        valid = [v for v in votes if str(v.get("answer", "")).strip()]
        if not valid:
            return False, None
        counts = Counter(str(v["answer"]).strip().lower() for v in valid)
        most_common, count = counts.most_common(1)[0]
        if count / len(valid) <= 0.5:
            return False, None
        agreeing = [v for v in valid if str(v["answer"]).strip().lower() == most_common]
        avg_conf = sum(v["confidence"] for v in agreeing) / len(agreeing)
        if avg_conf >= self.consensus_threshold:
            return True, agreeing[0]["answer"]
        return False, None

    def _tiebreak(self, votes: list[dict]) -> str:
        for v in votes:
            if v["role"] == "generalist":
                return v["answer"]
        return max(votes, key=lambda v: v["confidence"])["answer"]

    def _flush_log(self, log: dict):
        (self.shared_dir / "round_log.json").write_text(
            json.dumps(log, indent=2), encoding="utf-8"
        )
