"""
Task environment adapters.

ALFWorldAdapter wraps the alfworld library for household task evaluation.
MockEnv provides a lightweight synthetic environment for testing without ALFWorld.

Both expose the same interface:
    reset(task_id) -> (task_description: str, observation: str)
    step(action: str) -> (observation: str, success: bool, done: bool)
    task_ids() -> list[str]
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from typing import Optional


class TaskEnv(ABC):
    @abstractmethod
    def reset(self, task_id: str) -> tuple[str, str]:
        """Returns (task_description, initial_observation)."""

    @abstractmethod
    def step(self, action: str) -> tuple[str, bool, bool]:
        """Returns (observation, success, done)."""

    @abstractmethod
    def task_ids(self) -> list[str]:
        """Returns all available task IDs."""


# ---------------------------------------------------------------------------
# ALFWorld adapter
# ---------------------------------------------------------------------------

class ALFWorldAdapter(TaskEnv):
    """
    Wraps alfworld.agents.environment for text-based household tasks.
    Requires: pip install alfworld  (and ALFWORLD_DATA env var set)
    """

    def __init__(self, config_path: str, split: str = "eval_out_of_distribution", num_tasks: int = 134):
        try:
            import alfworld.agents.environment as alf_env
            import yaml
        except ImportError as e:
            raise ImportError(
                "ALFWorld not installed. Run: pip install alfworld\n"
                "Then download data: python -m alfworld.scripts.alfworld_download"
            ) from e

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # ALFWorld cycles through tasks on each reset() call — expose N sequential IDs
        self._alf_env = alf_env.AlfredTWEnv(cfg, train_eval=split)
        self._alf_env = self._alf_env.init_env(batch_size=1)
        self._task_ids: list[str] = [f"task_{i:03d}" for i in range(num_tasks)]
        self._last_obs: str = ""
        self._last_task_desc: str = ""

    def reset(self, task_id: str) -> tuple[str, str]:
        # ALFWorld ignores task_id and advances its internal task iterator
        obs, info = self._alf_env.reset()
        obs_text = str(obs[0] if isinstance(obs, list) else obs)

        # Extract a human-readable task description from the observation or admissible commands
        admissible = info.get("admissible_commands", [[]])[0] if isinstance(info.get("admissible_commands"), list) else []
        # First non-look line of obs usually contains the task goal
        task_desc = next(
            (line.strip() for line in obs_text.split("\n") if line.strip() and "Your task" in line),
            obs_text.split("\n")[0].strip(),
        )
        self._last_task_desc = task_desc
        self._last_obs = obs_text
        return task_desc, obs_text

    def step(self, action: str) -> tuple[str, bool, bool]:
        obs, scores, dones, info = self._alf_env.step([action])
        obs_text = str(obs[0] if isinstance(obs, list) else obs)
        score = float(scores[0] if isinstance(scores, list) else scores)
        done = bool(dones[0] if isinstance(dones, list) else dones)
        success = score >= 1.0
        return obs_text, success, done

    def task_ids(self) -> list[str]:
        return self._task_ids


# ---------------------------------------------------------------------------
# Mock environment (for unit tests and CI without ALFWorld)
# ---------------------------------------------------------------------------

MOCK_TASKS = {
    # Session 1 tasks
    "pick_cup": {
        "description": "Pick up the red cup from the counter and place it in the sink.",
        "goal_action": "put cup sink",
        "distractors": ["examine cup", "go to counter", "pick up cup", "go to sink"],
    },
    "heat_apple": {
        "description": "Find the apple, heat it in the microwave, and place it on the plate.",
        "goal_action": "put apple plate",
        "distractors": ["find apple", "open microwave", "put apple microwave", "heat apple", "go to plate"],
    },
    "clean_knife": {
        "description": "Clean the knife in the sink and put it on the countertop.",
        "goal_action": "put knife countertop",
        "distractors": ["find knife", "go to sink", "clean knife"],
    },
    "cool_mug": {
        "description": "Cool the mug in the fridge and place it on the coffee table.",
        "goal_action": "put mug coffeetable",
        "distractors": ["find mug", "open fridge", "cool mug", "go to coffeetable"],
    },
    "examine_book": {
        "description": "Find the book and examine it under the desklamp.",
        "goal_action": "examine book desklamp",
        "distractors": ["find book", "go to desk", "turn on lamp"],
    },
    # Session 2 tasks (structurally similar — tests whether S1 memory transfers)
    "pick_plate": {
        "description": "Pick up the plate from the table and place it in the dishwasher.",
        "goal_action": "put plate dishwasher",
        "distractors": ["go to table", "pick up plate", "go to dishwasher"],
    },
    "heat_egg": {
        "description": "Find the egg, heat it in the microwave, and place it on the counter.",
        "goal_action": "put egg counter",
        "distractors": ["find egg", "open microwave", "heat egg", "go to counter"],
    },
    "clean_pan": {
        "description": "Clean the pan in the sink and place it on the stove.",
        "goal_action": "put pan stove",
        "distractors": ["find pan", "go to sink", "clean pan", "go to stove"],
    },
    "cool_bottle": {
        "description": "Cool the bottle in the fridge and place it on the shelf.",
        "goal_action": "put bottle shelf",
        "distractors": ["find bottle", "open fridge", "cool bottle", "go to shelf"],
    },
    "examine_laptop": {
        "description": "Find the laptop and examine it under the desklamp.",
        "goal_action": "examine laptop desklamp",
        "distractors": ["find laptop", "go to desk", "turn on lamp"],
    },
}


class MockEnv(TaskEnv):
    """Deterministic mock environment for testing the experiment harness."""

    def __init__(self, seed: int = 42, success_after_n_steps: int = 3):
        self._rng = random.Random(seed)
        self._success_after = success_after_n_steps
        self._step_count = 0
        self._current: Optional[dict] = None
        self._task_id: Optional[str] = None

    def reset(self, task_id: str) -> tuple[str, str]:
        task = MOCK_TASKS.get(task_id, list(MOCK_TASKS.values())[0])
        self._current = task
        self._task_id = task_id
        self._step_count = 0
        obs = f"You are in a room. {task['description']}"
        return task["description"], obs

    def step(self, action: str) -> tuple[str, bool, bool]:
        assert self._current is not None, "Call reset() first"
        self._step_count += 1
        goal_tokens = set(self._current["goal_action"].lower().split())
        action_lower = action.lower().strip()
        action_tokens = set(re.findall(r"[a-z]+", action_lower))

        # Success if all goal keywords appear in the action
        if goal_tokens and goal_tokens.issubset(action_tokens):
            return "Task completed successfully.", True, True

        if self._step_count >= self._success_after + len(self._current["distractors"]):
            return "You ran out of steps.", False, True

        # Progress hints
        distractors = self._current["distractors"]
        idx = min(self._step_count - 1, len(distractors) - 1)
        hint = distractors[idx]
        return f"OK. You can try: {hint}", False, False

    def task_ids(self) -> list[str]:
        return list(MOCK_TASKS.keys())
