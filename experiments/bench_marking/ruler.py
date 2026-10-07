"""
RULER iterative educability experiment.

Hard task variants:
  niah_multihop  — 3-hop chain: three linked needles, must traverse all to answer
  niah_multikey  — 5 key-value pairs + 1 distractor with a near-identical key
  vt             — 6 variables, 15 steps, includes multiplication
"""

import requests
import json
import time
import yaml
import logging
import random
import string
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(
    filename=str(Path(__file__).parent / 'responses.log'),
    level=logging.INFO,
    format='%(asctime)s - RULER - %(message)s'
)

CONFIG_PATH = Path(__file__).parent / 'config.yaml'
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

OLLAMA_URL = config['model']['url']
MODEL      = config['model']['name']
RULER_CFG  = config['ruler']


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class RulerTask:
    task_type: str
    context:   str
    question:  str
    expected:  str


FILLER_POOL = [
    "Scientists discovered new minerals on the ocean floor last year.",
    "The committee voted unanimously to approve the revised budget proposal.",
    "Children played happily near the fountain under the afternoon sun.",
    "The library catalogue contains thousands of volumes on philosophy.",
    "A freight train departed from the central station at exactly noon.",
    "Researchers published their findings in a peer-reviewed journal.",
    "The chef prepared a meal using locally sourced seasonal ingredients.",
    "Engineers completed the bridge inspection ahead of the scheduled date.",
    "The ambassador delivered a speech at the annual trade conference.",
    "Volunteers planted two hundred trees along the northern riverbank.",
    "The museum opened a new wing dedicated to modern sculpture.",
    "Astronomers observed an unusual comet approaching from the outer system.",
    "The marathon runners crossed the finish line after four gruelling hours.",
    "Teachers attended a professional development workshop on Friday morning.",
    "The hospital launched a new patient record management system.",
    "Geologists mapped a previously uncharted fault line in the mountain range.",
    "The parliament passed legislation on environmental protection standards.",
    "A software update was deployed to all company laptops overnight.",
    "The bakery introduced three new varieties of sourdough bread this week.",
    "Tourists queued outside the ancient cathedral from early morning.",
    "The orchestra rehearsed Beethoven's Ninth Symphony for three weeks.",
    "A new pedestrian crossing was installed near the primary school.",
    "The satellite successfully transmitted data from its polar orbit.",
    "Farmers reported an unusually early harvest due to the warm spring.",
    "The bookshop celebrated its fiftieth anniversary with a reading event.",
    "The city council approved plans for a new public transit line.",
    "Meteorologists warned of heavy rainfall across the northern region.",
    "The annual science fair attracted over two thousand student entries.",
    "Local fishermen reported unusual migration patterns along the coast.",
    "The university launched a new scholarship programme for first-generation students.",
]


def _make_filler(word_count: int, rng: random.Random) -> list[str]:
    words: list[str] = []
    while len(words) < word_count:
        words.extend(rng.choice(FILLER_POOL).split())
    return words[:word_count]


def _rand_word(length: int, rng: random.Random) -> str:
    return ''.join(rng.choices(string.ascii_uppercase, k=length))


def _mutate(word: str, rng: random.Random) -> str:
    """Return a word that differs from `word` by exactly one character."""
    idx = rng.randint(0, len(word) - 1)
    chars = list(string.ascii_uppercase)
    replacement = rng.choice([c for c in chars if c != word[idx]])
    return word[:idx] + replacement + word[idx + 1:]


# ── Task Generators ────────────────────────────────────────────────────────────

def generate_niah_multihop_task(seed: int, cfg: dict) -> RulerTask:
    """
    3-hop NIAH chain:
      Needle A: 'The master passphrase equals vault ALPHA's contents.'
      Needle B: 'Vault ALPHA mirrors vault BETA.'
      Needle C: 'Vault BETA holds value VALUE.'
    The model must traverse all three and chain A→B→C to answer.
    Needles placed in early / middle / late thirds of the document.
    """
    rng = random.Random(seed)
    context_length = rng.choice(cfg['context_lengths'])

    alpha = _rand_word(6, rng)
    beta  = _rand_word(6, rng)
    value = rng.randint(1000, 99999)

    needle_a = f"The master passphrase is defined as the contents of vault {alpha}."
    needle_b = f"Vault {alpha} is an exact mirror of vault {beta}."
    needle_c = f"Vault {beta} holds the value {value}."

    word_count = max(150, int(context_length * 0.75))
    words = _make_filler(word_count, rng)

    # Spread across thirds — intentionally far apart
    pos_a = int(len(words) * rng.uniform(0.05, 0.20))
    pos_b = int(len(words) * rng.uniform(0.40, 0.55))
    pos_c = int(len(words) * rng.uniform(0.78, 0.92))

    words.insert(pos_a, needle_a)
    words.insert(pos_b + 1, needle_b)
    words.insert(pos_c + 2, needle_c)

    return RulerTask(
        task_type='niah_multihop',
        context=' '.join(words),
        question=(
            "What is the master passphrase? "
            "Reply with just the numeric value."
        ),
        expected=str(value),
    )


def generate_multi_niah_task(seed: int, cfg: dict) -> RulerTask:
    """
    5 key-value pairs scattered across a long document, plus one distractor pair
    whose key differs from the target key by exactly one character.
    Ask about one specific key.
    """
    rng = random.Random(seed)
    context_length = rng.choice(cfg['context_lengths'])

    num_pairs = 5
    keys   = [_rand_word(6, rng) for _ in range(num_pairs)]
    values = [_rand_word(7, rng) for _ in range(num_pairs)]
    target = rng.randint(0, num_pairs - 1)

    # One distractor: same length as target key, one char different
    trap_key   = _mutate(keys[target], rng)
    trap_value = _rand_word(7, rng)

    real_needles = [
        f"Credential {keys[i]} unlocks access code {values[i]}."
        for i in range(num_pairs)
    ]
    trap_needle = f"Credential {trap_key} unlocks access code {trap_value}."

    word_count = max(150, int(context_length * 0.75))
    words = _make_filler(word_count, rng)

    # Spread real pairs evenly; place trap adjacent to the target pair
    positions = [int(len(words) * (i + 1) / (num_pairs + 2)) for i in range(num_pairs)]
    for i, needle in enumerate(real_needles):
        words.insert(positions[i], needle)

    # Trap placed 15-25 words after the target needle
    trap_pos = positions[target] + rng.randint(15, 25)
    words.insert(min(trap_pos, len(words)), trap_needle)

    return RulerTask(
        task_type='niah_multikey',
        context=' '.join(words),
        question=(
            f"What is the access code that credential {keys[target]} unlocks? "
            "Reply with just the access code."
        ),
        expected=values[target],
    )


def generate_vt_task(seed: int, cfg: dict) -> RulerTask:
    """6 variables, 15 steps, +/-/* (multiplication capped at 50)."""
    rng = random.Random(seed)
    num_steps = cfg.get('vt_steps', 15)
    num_vars  = cfg.get('vt_vars', 6)
    var_names = rng.sample(list(string.ascii_lowercase[:12]), num_vars)
    target    = var_names[0]

    initial = {v: rng.randint(1, 9) for v in var_names}
    vals    = dict(initial)
    assignments: list[str] = []

    for _ in range(num_steps):
        v  = rng.choice(var_names)
        op = rng.choice(['+', '-', '+', '-', '*'])

        if op == '*':
            factor  = rng.choice([2, 3])
            new_val = vals[v] * factor
            if abs(new_val) > 50:
                op = '+'
                operand = rng.randint(1, 5)
                vals[v] += operand
                assignments.append(f"Variable {v} is updated: {v} = {v} + {operand}.")
            else:
                vals[v] = new_val
                assignments.append(f"Variable {v} is updated: {v} = {v} * {factor}.")
        elif rng.random() < 0.55:
            operand = rng.randint(1, 7)
            vals[v] = vals[v] + operand if op == '+' else vals[v] - operand
            assignments.append(f"Variable {v} is updated: {v} = {v} {op} {operand}.")
        else:
            other = rng.choice([x for x in var_names if x != v])
            vals[v] = vals[v] + vals[other] if op == '+' else vals[v] - vals[other]
            assignments.append(f"Variable {v} is updated: {v} = {v} {op} {other}.")

    init_lines = [f"Initially: {v} = {initial[v]}." for v in var_names]

    filler = _make_filler(80, rng)
    tokens: list[str] = filler[:4]
    tokens.extend(init_lines)
    tokens.extend(filler[4:8])
    for i, asgn in enumerate(assignments):
        tokens.append(asgn)
        start = 8 + i * 4
        tokens.extend(filler[start:start + 4])
    tokens.extend(filler[70:])

    return RulerTask(
        task_type='vt',
        context=' '.join(tokens),
        question=(
            f"After all updates in the text, what is the final numeric value of "
            f"variable {target}? Reply with just the number."
        ),
        expected=str(vals[target]),
    )


TASK_GENERATORS = {
    'niah_multihop': generate_niah_multihop_task,
    'niah_multikey': generate_multi_niah_task,
    'vt':            generate_vt_task,
}


# ── LLM Calls ─────────────────────────────────────────────────────────────────

def query_model(
    context: str, question: str, knowledge: list[str]
) -> tuple[Optional[str], float]:
    messages = [
        {"role": "system", "content": "You are a careful analytical agent that reads text precisely."}
    ]
    if knowledge:
        block = "\n".join(f"- {k}" for k in knowledge)
        messages.append({
            "role": "system",
            "content": f"Strategy notes from prior agents (apply these):\n{block}",
        })
    messages.append({
        "role": "user",
        "content": f"Text:\n{context}\n\n{question}",
    })

    try:
        t0 = time.time()
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "messages": messages, "temperature": 0},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'], time.time() - t0
    except Exception as e:
        logging.error("query_model failed: %s", e)
        return None, 0.0


# ── Strategy Extraction ────────────────────────────────────────────────────────

TASK_DESCRIPTIONS = {
    'niah_multihop': (
        "find three linked sentences scattered across a long document and chain them: "
        "sentence A names vault X, sentence B says X mirrors vault Y, "
        "sentence C gives Y's value — all three must be found and connected"
    ),
    'niah_multikey': (
        "find the access code for a specific credential among 5 key-value pairs; "
        "a distractor pair with a near-identical key is planted nearby to mislead — "
        "read the key character-by-character before accepting a match"
    ),
    'vt': (
        "compute the final value of a target variable after 15 arithmetic updates "
        "(+, -, *) with 6 variables; initial values are stated at the start; "
        "cross-variable assignments must use the value of the other variable "
        "at the exact moment of the update, not its initial value"
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
    task_type: str, answer: str, correct: bool,
    prior: list[str],
) -> Optional[str]:
    prior_str = "\n".join(f"- {k}" for k in prior) if prior else "None yet."
    task_desc = TASK_DESCRIPTIONS.get(task_type, task_type)

    if correct:
        directive = (
            "You answered CORRECTLY. In ONE sentence (≤30 words), describe the "
            "SPECIFIC steps that worked — what you searched for and how you linked "
            "the relevant pieces."
        )
    else:
        directive = (
            "You answered INCORRECTLY. In ONE sentence (≤30 words), describe "
            "what you should do DIFFERENTLY — a concrete scanning, chaining, or "
            "arithmetic step you missed."
        )

    prompt = (
        f"Task: {task_type}\n"
        f"Requires: {task_desc}\n"
        f"Prior notes: {prior_str}\n\n"
        f"{directive}\n"
        "Do NOT mention the specific answer. Focus on method only."
    )

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a metacognitive agent."},
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


def score(answer: Optional[str], expected: str) -> bool:
    if not answer:
        return False
    return expected.lower() in answer.lower()


# ── Experiment Loop ────────────────────────────────────────────────────────────

def run_ruler_experiment():
    num_agents = RULER_CFG.get('num_agents', 10)
    task_types = RULER_CFG.get('task_types', ['niah_multihop', 'niah_multikey', 'vt'])

    all_results: dict = {}

    for task_type in task_types:
        print(f"\n{'='*62}")
        print(f"RULER | Task: {task_type.upper()} | Agents: {num_agents}")
        print('='*62)

        knowledge:     list[str]  = []
        agent_results: list[dict] = []

        for idx in range(num_agents):
            task = TASK_GENERATORS[task_type](seed=idx, cfg=RULER_CFG)

            print(f"\n  Agent {idx + 1}/{num_agents}")
            answer, latency = query_model(task.context, task.question, knowledge)

            if answer is None:
                print("    [ERROR] No response from model.")
                continue

            ok           = score(answer, task.expected)
            answer_short = answer.strip().splitlines()[0][:80]
            print(f"    Expected : {task.expected}")
            print(f"    Got      : {answer_short}")
            print(f"    Correct  : {ok}  |  {latency:.1f}s  |  K={len(knowledge)}")

            strategy = extract_strategy(task_type, answer, ok, knowledge)
            if strategy:
                if is_duplicate(strategy, knowledge):
                    print("    Strategy : [duplicate — skipped]")
                    strategy = None
                else:
                    knowledge.append(strategy)
                    print(f"    Strategy : {strategy}")

            logging.info(
                "task=%s agent=%d correct=%s expected=%s answer=%s",
                task_type, idx + 1, ok, task.expected, answer[:120],
            )

            agent_results.append({
                'agent':           idx + 1,
                'correct':         ok,
                'latency':         round(latency, 2),
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

        accuracy = (
            sum(r['correct'] for r in agent_results) / len(agent_results)
            if agent_results else 0
        )
        print(f"\n  Accuracy: {accuracy:.1%}  |  Convergence: {convergence or 'not reached'}")

        all_results[task_type] = {
            'accuracy':          round(accuracy, 3),
            'convergence_agent': convergence or None,
            'agents':            agent_results,
        }

    out = Path(__file__).parent / 'ruler_results.json'
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {out}")
    print("\n--- RULER SUMMARY ---")
    for t, r in all_results.items():
        print(f"  {t:20s}  accuracy={r['accuracy']:.1%}  convergence={r['convergence_agent']}")


if __name__ == '__main__':
    run_ruler_experiment()
