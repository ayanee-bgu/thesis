"""
MINT iterative educability experiment — hard BIG-Bench Hard style tasks.

  logical_deduction — 6-element ordering from constraint sets
  date_reasoning    — multi-step chained date arithmetic

Fixes vs previous run:
  - Score normalises leading zeros ("October 01" == "October 1")
  - Turn loop: if agent never writes FINAL ANSWER, last reply is used as fallback answer
  - API failure retried once before giving up
"""

import json
import logging
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import requests
import yaml

logging.basicConfig(
    filename=str(Path(__file__).parent / 'responses.log'),
    level=logging.INFO,
    format='%(asctime)s - MINT - %(message)s'
)

CONFIG_PATH = Path(__file__).parent / 'config.yaml'
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

OLLAMA_URL = config['model']['url']
MODEL      = config['model']['name']
MINT_CFG   = config.get('mint', {})


# ── Task Banks ─────────────────────────────────────────────────────────────────

@dataclass
class MintTask:
    task_type: str
    problem:   str
    expected:  str


# 6-element ordering puzzles — all solutions verified.
# Solution order written as comments; constraints uniquely determine it.
LOGICAL_DEDUCTION_TASKS: list[dict] = [
    {
        # Anna=1, Ben=2, Clara=3, Dan=4, Eva=5, Fred=6
        "problem": textwrap.dedent("""
            Six people stand in a queue (position 1 = front, position 6 = back).
            - Anna is at the very front.
            - Fred is at the very back.
            - Clara is directly behind Ben.
            - Dan is directly behind Clara.
            - Eva is between Dan and Fred.
            What position is Ben? Reply with just the number.
        """).strip(),
        "expected": "2",
    },
    {
        # gray=1, brown=2, red=3, blue=4, green=5, white=6
        "problem": textwrap.dedent("""
            Six books are stacked on a shelf (position 1 = bottom, position 6 = top).
            - White is at the very top.
            - Gray is at the very bottom.
            - Red is directly above brown.
            - Blue is directly above red.
            - Green is between blue and white.
            What position is blue? Reply with just the number.
        """).strip(),
        "expected": "4",
    },
    {
        # Sam=1, Tim=2, Uma=3, Val=4, Will=5, Xena=6
        "problem": textwrap.dedent("""
            Six runners finish a race (position 1 = first, position 6 = last).
            - Sam finishes first.
            - Xena finishes last.
            - Tim finishes immediately before Uma.
            - Val finishes immediately after Uma.
            - Will finishes between Val and Xena.
            What position does Uma finish? Reply with just the number.
        """).strip(),
        "expected": "3",
    },
    {
        # oval=1, square=2, round=3, diamond=4, triangle=5, star=6
        "problem": textwrap.dedent("""
            Six paintings hang on a gallery wall (position 1 = leftmost, position 6 = rightmost).
            - Oval is leftmost.
            - Star is rightmost.
            - Square is directly right of oval.
            - Round is directly right of square.
            - Diamond is directly right of round.
            - Triangle is between diamond and star.
            What position is round? Reply with just the number.
        """).strip(),
        "expected": "3",
    },
    {
        # ant=1, bee=2, cat=3, dog=4, elk=5, fox=6
        "problem": textwrap.dedent("""
            Six animals are placed in a row (position 1 = leftmost, position 6 = rightmost).
            - Ant is leftmost.
            - Fox is rightmost.
            - Bee is directly right of ant.
            - Cat is directly right of bee.
            - Dog is directly right of cat.
            - Elk is between dog and fox.
            What position is dog? Reply with just the number.
        """).strip(),
        "expected": "4",
    },
    {
        # parking=1, lobby=2, offices=3, labs=4, cafeteria=5, rooftop=6
        "problem": textwrap.dedent("""
            Six floors in a building (floor 1 = ground, floor 6 = top).
            - Parking is on the ground floor.
            - Rooftop garden is on the top floor.
            - Lobby is directly above parking.
            - Offices are directly above the lobby.
            - Labs are directly above the offices.
            - Cafeteria is between the labs and the rooftop garden.
            What floor is the cafeteria on? Reply with just the number.
        """).strip(),
        "expected": "5",
    },
    {
        # Paula=1, Quinn=2, Rosa=3, Steve=4, Tara=5, Uma=6
        "problem": textwrap.dedent("""
            Six competitors finish a race (position 1 = first, position 6 = last).
            - Paula finishes first.
            - Uma finishes last.
            - Quinn finishes immediately before Rosa.
            - Rosa finishes immediately before Steve.
            - Tara finishes between Steve and Uma.
            What position does Rosa finish? Reply with just the number.
        """).strip(),
        "expected": "3",
    },
    {
        # anchor=1, bolt=2, chain=3, drill=4, engine=5, filter=6
        "problem": textwrap.dedent("""
            Six items are arranged on a shelf (position 1 = leftmost, position 6 = rightmost).
            - Anchor is leftmost.
            - Filter is rightmost.
            - Bolt is directly right of anchor.
            - Chain is directly right of bolt.
            - Drill is directly right of chain.
            - Engine is between drill and filter.
            What position is chain? Reply with just the number.
        """).strip(),
        "expected": "3",
    },
    {
        # Arrow=1, Blaze=2, Cloud=3, Dust=4, Echo=5, Flash=6
        "problem": textwrap.dedent("""
            Six horses finish a race (position 1 = first, position 6 = last).
            - Arrow finishes first.
            - Flash finishes last.
            - Blaze finishes directly after Arrow.
            - Cloud finishes directly after Blaze.
            - Dust finishes directly after Cloud.
            - Echo finishes between Dust and Flash.
            What position does Cloud finish? Reply with just the number.
        """).strip(),
        "expected": "3",
    },
    {
        # Ava=1, Blake=2, Chloe=3, Dylan=4, Ella=5, Finn=6
        "problem": textwrap.dedent("""
            Six students wait in a queue (position 1 = front, position 6 = back).
            - Ava is at the front.
            - Finn is at the back.
            - Blake is directly behind Ava.
            - Chloe is directly behind Blake.
            - Dylan is directly behind Chloe.
            - Ella is between Dylan and Finn.
            What position is Dylan? Reply with just the number.
        """).strip(),
        "expected": "4",
    },
]


# Multi-step date problems — all answers verified with Python datetime.
DATE_REASONING_TASKS: list[dict] = [
    {
        # (date(2023,9,1).weekday()=4=Fri → Sep 4=Mon, Sep 11=Mon, Sep 18=Mon=3rd)
        # day_of_year(Sep 18) = 31+28+31+30+31+30+31+31+18 = 261
        "problem": (
            "The third Monday of September 2023 is a key date. "
            "September 1, 2023 is a Friday. "
            "What is the day-of-year number of that third Monday "
            "(counting January 1 as day 1)? Give just the number."
        ),
        "expected": "261",
    },
    {
        # date(2023,11,15) + timedelta(100) = date(2024,2,23)
        "problem": (
            "A lease signed on November 15, 2021 runs for exactly 2 years "
            "and then 100 more days. On what date does it expire? "
            "Give the answer as Month Day, Year."
        ),
        "expected": "February 23, 2024",
    },
    {
        # (date(2021,11,11)-date(2021,3,1)).days=255; 255%7=3; Mon+3=Thu
        "problem": (
            "March 1, 2021 is a Monday. "
            "What day of the week is November 11, 2021? "
            "Give just the day name."
        ),
        "expected": "Thursday",
    },
    {
        # Feb 1 2023 + 5*4 months = Feb 1 2023 + 20 months = Oct 1 2024
        "problem": (
            "A software subscription renews every 4 months, "
            "starting on February 1, 2023. "
            "What date is the 6th renewal? "
            "Give the answer as Month Day, Year."
        ),
        "expected": "October 1, 2024",
    },
    {
        # 31+29+31+30+31+30 = 182 (2024 is a leap year)
        "problem": (
            "2024 is a leap year. How many days are in the first half of 2024 "
            "(January 1 through June 30, inclusive)? Give just the number."
        ),
        "expected": "182",
    },
    {
        # date(2019,12,28)+timedelta(1500): 4y(+1leap)=1461 days to Dec28 2023;
        # 1500-1461=39 more; Dec 3 days + Jan 31 + Feb 5 = 39; Feb 5 2024
        "problem": (
            "A child is born on December 28, 2019. "
            "On what date does the child turn exactly 1500 days old? "
            "Give the answer as Month Day, Year."
        ),
        "expected": "February 5, 2024",
    },
    {
        # Jan1=Sun → first Fri=Jan6; (365-6)//7+1 = 359//7+1 = 51+1 = 52
        "problem": (
            "January 1, 2023 is a Sunday. "
            "How many Fridays are there in the entire year 2023? "
            "Give just the number."
        ),
        "expected": "52",
    },
    {
        # date(2024,1,15)+timedelta(225): Jan16+Feb29+Mar31+Apr30+May31+Jun30=167;
        # 225-167=58; Jul31 → 27 left → Aug27
        "problem": (
            "A project milestone recurs every 45 days, starting January 15, 2024 "
            "(2024 is a leap year). "
            "What date is the 6th milestone? "
            "Give the answer as Month Day, Year."
        ),
        "expected": "August 27, 2024",
    },
    {
        # A ends: date(2022,4,5)+timedelta(60)=Jun4; B starts Jun4-30=May5;
        # B ends: date(2022,5,5)+timedelta(45)=Jun19
        "problem": (
            "Event A starts April 5, 2022 and lasts 60 days. "
            "Event B starts 30 days before Event A ends and lasts 45 days. "
            "On what date does Event B end? "
            "Give the answer as Month Day, Year."
        ),
        "expected": "June 19, 2022",
    },
    {
        # date(2023,10,15)-timedelta(25)=Sep20
        "problem": (
            "What is the date exactly 3 weeks and 4 days before October 15, 2023? "
            "Give the answer as Month Day, Year."
        ),
        "expected": "September 20, 2023",
    },
]


def get_task(task_type: str, agent_idx: int) -> MintTask:
    bank = (
        LOGICAL_DEDUCTION_TASKS if task_type == 'logical_deduction'
        else DATE_REASONING_TASKS
    )
    t = bank[agent_idx % len(bank)]
    return MintTask(task_type=task_type, problem=t['problem'], expected=t['expected'])


# ── Tool: Python Executor ─────────────────────────────────────────────────────

def execute_python(code: str, timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True, timeout=timeout,
        )
        out = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            return f"Error:\n{result.stderr.strip()[:400]}"
        return out if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out"
    except Exception as e:
        return f"Error: {e}"


# ── Multi-turn Agent ──────────────────────────────────────────────────────────

_CODE_RE  = re.compile(r'```python\s*(.*?)```', re.DOTALL | re.IGNORECASE)
_FINAL_RE = re.compile(r'FINAL\s+ANSWER\s*:\s*(.+)', re.IGNORECASE)

SYSTEM_BASE = textwrap.dedent("""
    You are a problem-solving agent with access to a Python interpreter.

    TO USE THE INTERPRETER: wrap code in ```python ... ``` — you get the output back.
    TO SUBMIT: write  FINAL ANSWER: <value>  on its own line when confident.

    - Try reasoning first; use code when it helps compute or verify.
    - Submit FINAL ANSWER as soon as you are confident.
""").strip()


def _call_model(messages: list[dict]) -> Optional[str]:
    """Single model call with one retry on failure."""
    for attempt in range(2):
        try:
            r = requests.post(
                OLLAMA_URL,
                json={"model": MODEL, "messages": messages, "temperature": 0},
                timeout=90,
            )
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content']
        except Exception as e:
            logging.error("model call attempt %d failed: %s", attempt + 1, e)
            if attempt == 0:
                time.sleep(2)
    return None


def run_agent(
    task: MintTask, knowledge: list[str], max_turns: int
) -> tuple[Optional[str], int]:
    system = SYSTEM_BASE
    if knowledge:
        block = "\n".join(f"- {k}" for k in knowledge)
        system += f"\n\nStrategy notes from prior agents:\n{block}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Problem:\n{task.problem}"},
    ]

    final_answer: Optional[str] = None
    last_reply:   Optional[str] = None
    turns = 0

    for turn in range(max_turns):
        reply = _call_model(messages)
        if reply is None:
            break

        turns += 1
        last_reply = reply
        messages.append({"role": "assistant", "content": reply})

        fa = _FINAL_RE.search(reply)
        if fa:
            final_answer = fa.group(1).strip()
            break

        code_match = _CODE_RE.search(reply)
        if code_match:
            output = execute_python(code_match.group(1).strip())
            messages.append({"role": "user", "content": f"Code output:\n{output}"})
        elif turn == max_turns - 2:
            messages.append({
                "role": "user",
                "content": "Last turn. Write FINAL ANSWER: <value> now.",
            })

    # Fallback: if model never wrote FINAL ANSWER, use last reply
    if final_answer is None and last_reply:
        fa = _FINAL_RE.search(last_reply)
        if fa:
            final_answer = fa.group(1).strip()
        else:
            # Take the last non-empty line of the last reply
            lines = [l.strip() for l in last_reply.splitlines() if l.strip()]
            if lines:
                final_answer = lines[-1]

    return final_answer, turns


# ── Scoring ────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Lowercase, strip, remove leading zeros from day numbers."""
    s = s.lower().strip()
    # "October 01, 2023" → "october 1, 2023"
    s = re.sub(r'\b0+(\d)', r'\1', s)
    return s


def score(answer: Optional[str], expected: str, task_type: str) -> bool:
    if not answer:
        return False
    a = _normalize(answer)
    e = _normalize(expected)
    if e in a or a == e:
        return True
    # Numeric tolerance for counts and day-of-year numbers
    if task_type == 'date_reasoning':
        try:
            return abs(float(re.sub(r'[^\d.]', '', a.split()[0])) - float(e)) < 0.5
        except (ValueError, IndexError):
            pass
    return False


# ── Strategy Extraction ────────────────────────────────────────────────────────

TASK_DESCRIPTIONS_MINT = {
    'logical_deduction': (
        "determine the position of one item in a 6-element ordered sequence by "
        "systematically applying relational constraints — assign known anchor positions "
        "first, then propagate constraints to fill remaining slots"
    ),
    'date_reasoning': (
        "compute a calendar result through multiple chained arithmetic steps — "
        "Python's datetime and timedelta give exact answers; "
        "mental arithmetic fails on leap years, month boundaries, and day-of-week shifts"
    ),
}


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_duplicate(candidate: str, existing: list[str], threshold: float = 0.55) -> bool:
    return any(_jaccard(candidate, e) >= threshold for e in existing)


def extract_strategy(
    task_type: str, correct: bool, turns: int, prior: list[str],
) -> Optional[str]:
    prior_str = "\n".join(f"- {k}" for k in prior) if prior else "None yet."
    task_desc = TASK_DESCRIPTIONS_MINT.get(task_type, task_type)
    outcome   = "correctly" if correct else "incorrectly"

    if correct:
        directive = (
            "You answered CORRECTLY. ONE sentence (≤35 words): what specific "
            "approach or code pattern led to the right answer?"
        )
    else:
        directive = (
            "You answered INCORRECTLY. ONE sentence (≤35 words): what concrete "
            "step, code pattern, or verification would have fixed it?"
        )

    prompt = (
        f"Task: {task_type}\n"
        f"Requires: {task_desc}\n"
        f"Outcome: {outcome} in {turns} turn(s).\n"
        f"Prior notes: {prior_str}\n\n"
        f"{directive}\n"
        "Do NOT mention the specific answer. Focus on method."
    )

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You reflect on problem-solving strategies."},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": 0.3,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error("extract_strategy failed: %s", e)
        return None


# ── Experiment Loop ────────────────────────────────────────────────────────────

def run_mint_experiment():
    num_agents = MINT_CFG.get('num_agents', 10)
    max_turns  = MINT_CFG.get('max_turns', 6)
    task_types = MINT_CFG.get('task_types', ['logical_deduction', 'date_reasoning'])

    all_results: dict = {}

    for task_type in task_types:
        print(f"\n{'='*62}")
        print(f"MINT | {task_type.upper()} | Agents: {num_agents}  Max turns: {max_turns}")
        print('='*62)

        knowledge:     list[str]  = []
        agent_results: list[dict] = []

        for idx in range(num_agents):
            task = get_task(task_type, idx)
            print(f"\n  Agent {idx + 1}/{num_agents}")
            print(f"  Problem  : {task.problem[:100].strip()}...")

            answer, turns = run_agent(task, knowledge, max_turns)
            ok = score(answer, task.expected, task_type)

            print(f"  Expected : {task.expected}")
            print(f"  Got      : {answer}")
            print(f"  Correct  : {ok}  |  Turns: {turns}  |  K={len(knowledge)}")

            strategy = extract_strategy(task_type, ok, turns, knowledge)
            if strategy:
                if is_duplicate(strategy, knowledge):
                    print("  Strategy : [duplicate — skipped]")
                    strategy = None
                else:
                    knowledge.append(strategy)
                    print(f"  Strategy : {strategy}")

            logging.info(
                "task=%s agent=%d correct=%s turns=%d answer=%s expected=%s",
                task_type, idx + 1, ok, turns, answer, task.expected,
            )

            agent_results.append({
                'agent':           idx + 1,
                'correct':         ok,
                'turns':           turns,
                'final_answer':    answer,
                'expected':        task.expected,
                'knowledge_items': len(knowledge),
                'strategy':        strategy,
            })

        streak = convergence = 0
        for r in agent_results:
            if r['correct']:
                streak += 1
                if streak >= 3 and not convergence:
                    convergence = r['agent'] - 2
            else:
                streak = 0

        accuracy  = sum(r['correct'] for r in agent_results) / len(agent_results) if agent_results else 0
        avg_turns = sum(r['turns']   for r in agent_results) / len(agent_results) if agent_results else 0
        print(
            f"\n  Accuracy: {accuracy:.1%}  |  Avg turns: {avg_turns:.1f}  "
            f"|  Convergence: {convergence or 'not reached'}"
        )

        all_results[task_type] = {
            'accuracy':          round(accuracy, 3),
            'avg_turns':         round(avg_turns, 2),
            'convergence_agent': convergence or None,
            'agents':            agent_results,
        }

    out = Path(__file__).parent / 'mint_results.json'
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {out}")
    print("\n--- MINT SUMMARY ---")
    for t, r in all_results.items():
        print(
            f"  {t:22s}  accuracy={r['accuracy']:.1%}  "
            f"avg_turns={r['avg_turns']}  convergence={r['convergence_agent']}"
        )


if __name__ == '__main__':
    run_mint_experiment()
