"""
Memory transfer protocol for inter-agent knowledge sharing.

Wraps the low-level file reads in cooperative_agent.py with transfer logging
so we can analyse when and how often agents request peer knowledge (H4).
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class TransferEvent:
    round_num: int
    requesting_role: str
    providing_role: str
    files_returned: list[str]
    byte_count: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MemoryTransfer:
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.events: list[TransferEvent] = []

    def fetch(
        self,
        requesting_role: str,
        providing_agent,       # CooperativeAgent
        round_num: int,
    ) -> str:
        """Read all memory files from providing_agent and log the transfer."""
        files = sorted(providing_agent.memory_dir.glob("*.md"))
        if not files:
            return ""

        content_parts = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            content_parts.append(f"### {f.name}\n{text}")

        content = "\n\n".join(content_parts)
        event = TransferEvent(
            round_num=round_num,
            requesting_role=requesting_role,
            providing_role=providing_agent.role,
            files_returned=[f.name for f in files],
            byte_count=len(content.encode()),
        )
        self.events.append(event)
        self._flush()
        return content

    def summary(self) -> dict:
        """Aggregate stats for the results log (tests H4: requests cluster in rounds 2-3)."""
        if not self.events:
            return {"total_transfers": 0, "by_round": {}, "by_pair": {}}

        by_round: dict[int, int] = {}
        by_pair: dict[str, int] = {}
        for e in self.events:
            by_round[e.round_num] = by_round.get(e.round_num, 0) + 1
            pair = f"{e.requesting_role}→{e.providing_role}"
            by_pair[pair] = by_pair.get(pair, 0) + 1

        return {
            "total_transfers": len(self.events),
            "total_bytes": sum(e.byte_count for e in self.events),
            "by_round": {str(k): v for k, v in sorted(by_round.items())},
            "by_pair": by_pair,
        }

    def _flush(self):
        self.log_path.write_text(
            json.dumps([asdict(e) for e in self.events], indent=2),
            encoding="utf-8",
        )
