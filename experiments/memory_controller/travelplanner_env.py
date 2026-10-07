"""
TravelPlanner environment adapter — sole-planning mode.

The agent receives a travel query + pre-fetched reference information (flights,
hotels, restaurants, attractions for the relevant cities and dates). It produces
a day-by-day plan in the TravelPlanner text format. We evaluate the plan with a
lightweight parser + budget/format checker (proxy for the full HC/CS evaluation).

Metrics reported:
  delivery_rate     : plan produced with correct number of days
  format_complete   : all required fields present for every day
  budget_satisfied  : estimated total cost ≤ stated budget
  success           : format_complete AND budget_satisfied (proxy for HC pass)

For formal comparison with the MemP paper, run the official TravelPlanner eval
scripts (.travelplanner/evaluation/) on the saved plan JSON files.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from datasets import load_dataset

TRAVELPLANNER_ROOT = Path(__file__).parent.parent.parent / ".travelplanner"
_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


def _hf_token() -> Optional[str]:
    """Return HF_TOKEN from environment or .env file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None
REQUIRED_FIELDS = ["current_city", "transportation", "breakfast", "attraction",
                   "lunch", "dinner", "accommodation"]


# ---------------------------------------------------------------------------
# Plan parser
# ---------------------------------------------------------------------------

FIELD_ALIASES = {
    "current_city":    r"current[_ ]city",
    "transportation":  r"transportation",
    "breakfast":       r"breakfast",
    "attraction":      r"attraction",
    "lunch":           r"lunch",
    "dinner":          r"dinner",
    "accommodation":   r"accommodation",
}


def _has_fields(block: str) -> bool:
    """True if a block contains at least one recognisable field."""
    return any(re.search(alias + r"\s*:", block, re.IGNORECASE)
               for alias in FIELD_ALIASES.values())


def parse_plan(plan_text: str, num_days: int) -> list[dict] | None:
    """
    Parse a TravelPlanner plan text into a list of day dicts.
    Returns None if fewer than num_days parseable day sections are found.
    """
    # Split on "Day N:" markers; discard any leading preamble block
    raw_blocks = re.split(r"Day\s+\d+\s*:", plan_text, flags=re.IGNORECASE)
    day_blocks = [b.strip() for b in raw_blocks if b.strip() and _has_fields(b)]

    if len(day_blocks) < num_days:
        return None

    days = []
    for block in day_blocks[:num_days]:
        day = {}
        for field, alias in FIELD_ALIASES.items():
            pattern = rf"{alias}\s*:\s*(.+?)(?=\n\w|\Z)"
            m = re.search(pattern, block, re.IGNORECASE | re.DOTALL)
            value = m.group(1).strip() if m else ""
            # Normalise "- (...)" → "-"
            if value.startswith("-"):
                value = "-"
            day[field] = value
        days.append(day)
    return days


class LocalConstraintChecker:
    """
    Checks TravelPlanner hard constraints against the database CSVs.
    Loaded once and reused across tasks.
    """
    _instance: Optional["LocalConstraintChecker"] = None

    def __init__(self):
        try:
            import pandas as pd
            db = TRAVELPLANNER_ROOT / "database"
            self._acc = pd.read_csv(db / "accommodations" / "clean_accommodations_2022.csv")
            self._rest = pd.read_csv(db / "restaurants" / "clean_restaurant_2022.csv")
            # Normalise name columns for fuzzy matching
            self._acc["_name_lower"] = self._acc["NAME"].str.lower().str.strip()
            self._rest["_name_lower"] = self._rest["Name"].str.lower().str.strip()
            self._ok = True
        except Exception as e:
            print(f"[LocalConstraintChecker] Could not load DB: {e}")
            self._ok = False

    @classmethod
    def get(cls) -> "LocalConstraintChecker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def check(self, days: list[dict], constraints: dict) -> dict:
        """Return {constraint_name: bool, ..., 'all_satisfied': bool}."""
        if not constraints or not self._ok:
            return {"all_satisfied": True}

        results = {}

        transport_c = constraints.get("transportation")
        if transport_c:
            results["transportation"] = self._check_transport(days, transport_c)

        room_c = constraints.get("room type")
        if room_c:
            results["room_type"] = self._check_room_type(days, room_c)

        house_c = constraints.get("house rule")
        if house_c:
            results["house_rule"] = self._check_house_rule(days, house_c)

        cuisine_c = constraints.get("cuisine")
        if cuisine_c:
            results["cuisine"] = self._check_cuisine(days, cuisine_c)

        results["all_satisfied"] = all(v for v in results.values())
        return results

    # ------------------------------------------------------------------
    # Individual constraint checks
    # ------------------------------------------------------------------

    def _check_transport(self, days: list[dict], constraint: str) -> bool:
        """'no self-driving' → no day's transportation field contains 'self-driving'."""
        forbidden = constraint.lower().replace("no ", "").strip()
        for day in days:
            transport = day.get("transportation", "").lower()
            if transport and transport != "-" and forbidden in transport:
                return False
        return True

    def _check_room_type(self, days: list[dict], required: str) -> bool:
        """Check all chosen accommodations match the required room type."""
        required_l = required.lower()
        for day in days[:-1]:  # last day has no accommodation
            acc_field = day.get("accommodation", "")
            if not acc_field or acc_field == "-":
                continue
            acc_name = acc_field.split(",")[0].strip().lower()
            row = self._lookup_accommodation(acc_name)
            if row is None:
                continue  # can't verify, skip
            room_type = str(row.get("room type", "")).lower()
            if "not shared" in required_l:
                if "shared" in room_type:
                    return False
            elif "entire" in required_l:
                if "entire" not in room_type:
                    return False
            elif "private" in required_l:
                if "private" not in room_type:
                    return False
        return True

    def _check_house_rule(self, days: list[dict], constraint: str) -> bool:
        """
        constraint is a keyword e.g. 'pets', 'smoking', 'visitors', 'children under 10'.
        Traveler needs accommodation that ALLOWS this → house_rules must NOT forbid it.
        """
        forbidden_phrase = f"no {constraint.lower()}"
        for day in days[:-1]:
            acc_field = day.get("accommodation", "")
            if not acc_field or acc_field == "-":
                continue
            acc_name = acc_field.split(",")[0].strip().lower()
            row = self._lookup_accommodation(acc_name)
            if row is None:
                continue
            rules = str(row.get("house_rules", "")).lower()
            if forbidden_phrase in rules:
                return False
        return True

    def _check_cuisine(self, days: list[dict], required_cuisines: list | str) -> bool:
        """At least one meal per day must come from a restaurant serving a required cuisine."""
        if isinstance(required_cuisines, str):
            required_cuisines = [required_cuisines]
        required_lower = [c.lower() for c in required_cuisines]

        for day in days:
            meal_names = []
            for field in ["breakfast", "lunch", "dinner"]:
                meal = day.get(field, "")
                if meal and meal != "-":
                    meal_names.append(meal.split(",")[0].strip().lower())

            day_ok = False
            for name in meal_names:
                row = self._lookup_restaurant(name)
                if row is None:
                    continue
                cuisines = str(row.get("Cuisines", "")).lower()
                if any(req in cuisines for req in required_lower):
                    day_ok = True
                    break
            if meal_names and not day_ok:
                return False
        return True

    # ------------------------------------------------------------------
    # Lookup helpers (fuzzy name matching)
    # ------------------------------------------------------------------

    def _lookup_accommodation(self, name_lower: str) -> Optional[dict]:
        matches = self._acc[self._acc["_name_lower"].str.contains(name_lower[:20], regex=False, na=False)]
        if len(matches) == 0:
            for _, row in self._acc.iterrows():
                val = row["_name_lower"]
                if isinstance(val, str) and val[:20] in name_lower:
                    return row.to_dict()
            return None
        return matches.iloc[0].to_dict()

    def _lookup_restaurant(self, name_lower: str) -> Optional[dict]:
        matches = self._rest[self._rest["_name_lower"].str.contains(name_lower[:20], regex=False, na=False)]
        if len(matches) == 0:
            for _, row in self._rest.iterrows():
                val = row["_name_lower"]
                if isinstance(val, str) and val[:20] in name_lower:
                    return row.to_dict()
            return None
        return matches.iloc[0].to_dict()


def evaluate_plan(
    days: list[dict],
    query_info: dict,
    reference: dict,
) -> dict:
    """
    Evaluate a parsed plan against format, budget, and local constraints.
    Returns a dict of boolean flags; 'success' requires all checks to pass.
    """
    num_days = query_info["days"]
    budget = query_info.get("budget") or float("inf")
    people = query_info.get("people_number", 1)

    meal_fields = ["breakfast", "lunch", "dinner"]
    format_complete = len(days) == num_days and all(
        day.get(f, "") not in ("", None) for day in days for f in meal_fields
    ) and all(
        day.get("accommodation", "") not in ("", None)
        for day in days[:-1]  # last day accommodation is "-" (returning home)
    )

    estimated_cost = _estimate_cost(days, reference, people, num_days)
    budget_satisfied = estimated_cost <= budget if budget < float("inf") else True

    # Local constraint check (room type, cuisine, transportation, house rule)
    raw_constraints = query_info.get("local_constraint") or {}
    if isinstance(raw_constraints, str):
        try:
            raw_constraints = ast.literal_eval(raw_constraints)
        except Exception:
            raw_constraints = {}
    constraints = raw_constraints if isinstance(raw_constraints, dict) else {}
    constraint_results = LocalConstraintChecker.get().check(days, constraints)
    constraints_satisfied = constraint_results.get("all_satisfied", True)

    return {
        "format_complete": format_complete,
        "budget_satisfied": budget_satisfied,
        "estimated_cost": estimated_cost,
        "constraints_satisfied": constraints_satisfied,
        "constraint_detail": {k: v for k, v in constraint_results.items() if k != "all_satisfied"},
        "success": format_complete and constraints_satisfied,
    }


def _parse_ref(raw) -> list:
    """Parse reference_information into a list of {Description, Content} dicts."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.startswith("[{"):
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
    return []


def _ref_sections(reference) -> list[tuple[str, str]]:
    """Normalise reference into (description, content_text) pairs."""
    if isinstance(reference, list):
        return [(item.get("Description", ""), item.get("Content", ""))
                for item in reference if isinstance(item, dict)]
    if isinstance(reference, dict):
        return [(k, json.dumps(v)) for k, v in reference.items()]
    return []


def _estimate_cost(days: list[dict], reference, people: int, num_days: int) -> float:
    """
    Estimate total trip cost from plan days and reference data.
    Uses flight prices from reference content text.
    """
    total = 0.0
    sections = _ref_sections(reference)

    for day in days:
        transport = day.get("transportation", "")
        if transport and transport != "-":
            fn_match = re.search(r"F\d{7}", transport)
            if fn_match:
                fn = fn_match.group(0)
                for desc, content in sections:
                    if "Flight" in desc:
                        price_match = re.search(
                            rf"{fn}.*?Price.*?(\d+\.?\d*)|(\d+\.?\d*).*?{fn}", content
                        )
                        if price_match:
                            price = float(price_match.group(1) or price_match.group(2))
                            total += price * people
                            break
            sd_match = re.search(r"cost:\s*(\d+\.?\d*)", transport, re.IGNORECASE)
            if sd_match:
                total += float(sd_match.group(1)) * max(1, people // 5)

        accommodation = day.get("accommodation", "")
        if accommodation and accommodation not in ("-", ""):
            acc_name = accommodation.split(",")[0].strip().lower()
            for desc, content in sections:
                if "Accommodation" in desc and acc_name in content.lower():
                    # Try to extract price from content (tabular format)
                    price_match = re.search(rf"{re.escape(acc_name[:15])}.*?(\d{{2,5}}\.?\d*)", content, re.IGNORECASE)
                    if price_match:
                        total += float(price_match.group(1)) * max(1, people)
                    break

    return total


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TravelPlannerEnv:
    """
    Sole-planning mode TravelPlanner environment.

    Supports one or both HuggingFace splits simultaneously:
      splits=["validation"]        — 180 tasks, evaluable (budget + format proxy)
      splits=["test"]              — 1000 tasks, generate plans for leaderboard submission
      splits=["validation","test"] — both interleaved (val_XXX then test_XXX task IDs)

    Task IDs are prefixed with split name: val_000 … val_179, test_000 … test_999.
    The test split has no ground-truth answers; success is based on format + budget only.
    """

    def __init__(
        self,
        splits: list[str] | str = "validation",
        subset: Optional[int] = None,
        levels: Optional[list[str]] = None,
    ):
        """
        levels: filter to task difficulty levels, e.g. ["medium", "hard"].
                None = include all levels (easy + medium + hard).
                easy=3-day, medium=5-day, hard=7-day trips.
        """
        if isinstance(splits, str):
            splits = [splits]
        token = _hf_token()
        self._data: list[tuple[str, dict]] = []  # (task_id, example)
        for split in splits:
            print(f"[TravelPlannerEnv] Loading {split} split...")
            kwargs = {"token": token} if token else {}
            ds = load_dataset("osunlp/TravelPlanner", split, **kwargs)[split]
            prefix = "val" if split == "validation" else split[:4]
            rows = list(ds)
            if levels:
                rows = [r for r in rows if r.get("level") in levels]
                print(f"[TravelPlannerEnv] Filtered to {levels}: {len(rows)} tasks")
            if subset:
                rows = rows[:subset]
            self._data.extend((f"{prefix}_{i:03d}", ex) for i, ex in enumerate(rows))
        self._task_ids = [tid for tid, _ in self._data]
        self._id_to_ex = {tid: ex for tid, ex in self._data}
        self._current_id: Optional[str] = None
        self._current_query: Optional[dict] = None
        print(f"[TravelPlannerEnv] Loaded {len(self._data)} tasks total ({', '.join(splits)}).")

    def task_ids(self) -> list[str]:
        return self._task_ids

    def reset(self, task_id: str) -> tuple[str, str]:
        """Returns (query_text, reference_information_text)."""
        self._current_id = task_id
        self._current_query = self._id_to_ex[task_id]

        query = self._current_query["query"]
        ref = self._current_query.get("reference_information", "")

        # Dataset stores reference_information as a string repr of a Python list — parse it
        if isinstance(ref, str) and ref.startswith("[{"):
            try:
                ref = ast.literal_eval(ref)
            except Exception:
                pass

        # reference_information is a list of {Description, Content} dicts
        if isinstance(ref, list):
            parts = []
            for item in ref:
                if isinstance(item, dict):
                    desc = item.get("Description", "")
                    content = item.get("Content", "")
                    # Collapse excessive whitespace from pandas DataFrame string format
                    content = re.sub(r" {3,}", "  ", content)
                    content = re.sub(r"\n{3,}", "\n", content)
                    # Drop columns irrelevant to planning (lat, lng, address, phone, website)
                    content = re.sub(r"Latitude\s+Longitude\s+", "", content)
                    content = re.sub(r"\s+-?\d+\.\d+\s+-?\d+\.\d+", "", content)
                    parts.append(f"=== {desc} ===\n{content.strip()}")
                else:
                    parts.append(str(item))
            ref_text = "\n\n".join(parts)
        elif isinstance(ref, dict):
            ref_text = json.dumps(ref, indent=1)
        else:
            ref_text = str(ref)

        return query, ref_text

    def step(self, plan_text: str) -> tuple[str, bool, bool]:
        """
        Evaluate the plan. Always returns done=True (single-turn task).
        Returns (feedback, success, done=True).
        """
        assert self._current_query is not None, "Call reset() first"
        q = self._current_query
        num_days = q["days"]

        days = parse_plan(plan_text, num_days)
        if days is None:
            return (
                f"Plan could not be parsed. Expected {num_days} day sections.",
                False,
                True,
            )

        reference = _parse_ref(q.get("reference_information", ""))
        result = evaluate_plan(days, q, reference)
        feedback_parts = [
            f"Days parsed: {len(days)}/{num_days}",
            f"Format complete: {result['format_complete']}",
            f"Estimated cost: ${result['estimated_cost']:.0f} / Budget: ${q.get('budget','?')}",
            f"Budget satisfied: {result['budget_satisfied']}",
            f"Constraints satisfied: {result['constraints_satisfied']} {result.get('constraint_detail', {})}",
        ]
        return "\n".join(feedback_parts), result["success"], True

    def score(self, plan_text: str) -> dict:
        """Return full scoring dict (for analysis)."""
        assert self._current_query is not None
        q = self._current_query
        days = parse_plan(plan_text, q["days"])
        if days is None:
            return {"delivery_rate": False, "format_complete": False,
                    "budget_satisfied": False, "success": False, "estimated_cost": 0}
        ref = _parse_ref(q.get("reference_information", ""))
        result = evaluate_plan(days, q, ref)
        result["delivery_rate"] = days is not None
        return result

    def query_info(self) -> dict:
        """Return metadata for the current task (level, days, budget, etc.)."""
        if not self._current_query:
            return {}
        q = self._current_query
        return {
            "level": q.get("level"),
            "days": q.get("days"),
            "budget": q.get("budget"),
            "people_number": q.get("people_number"),
            "org": q.get("org"),
            "dest": q.get("dest"),
            "local_constraint": q.get("local_constraint"),
        }
