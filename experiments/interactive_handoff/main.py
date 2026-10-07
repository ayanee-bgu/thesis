"""
Interactive Handoff -- Multi-Agent Coordination Experiment

Tests whether interactive handoff (Worker B querying Worker A's reasoning)
outperforms static documentation and message-passing as a coordination primitive.

Conditions:
  none         -- Worker B receives no context from Worker A
  static       -- B receives A's verbatim output as context
  message_pass -- B receives structured JSON extracted from A's output
  interactive  -- B generates questions; A answers from full context; B uses Q&A

Tasks:
  coactive_pipeline          -- A identifies code bugs; B fixes the function
  hierarchical_orchestration -- A designs a data model; B implements a processing function

Metrics per agent pair:
  detail_preservation    -- fraction of A's key outputs B correctly incorporates (0.0-1.0)
  task_success           -- binary: did B complete the task correctly
  questions_asked        -- questions B generated (interactive only)
  clarification_requests -- times B escalated to supervisor (hierarchical only)
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests
import yaml

# -- Config --------------------------------------------------------------------

logging.basicConfig(
    filename=str(Path(__file__).parent / 'responses.log'),
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

CONFIG_PATH = Path(__file__).parent / 'config.yaml'
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

URL   = config['model']['url']
MODEL = config['model']['name']
CFG   = config['experiment']

NUM_AGENTS    = CFG['num_agents']
CONDITIONS    = CFG['conditions']
TASK_TYPES    = CFG['task_types']
MAX_QUESTIONS = CFG.get('interactive_max_questions', 2)


# -- LLM Utility ---------------------------------------------------------------

def call(messages: list[dict], temperature: float = 0.0, timeout: int = 200, retries: int = 2) -> Optional[str]:
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            r = requests.post(
                URL,
                json={"model": MODEL, "messages": messages, "temperature": temperature},
                timeout=timeout,
            )
            r.raise_for_status()
            text = r.json()['choices'][0]['message']['content']
            logging.info("call %.1fs | %s", time.time() - t0, text[:100].replace('\n', ' '))
            return text
        except Exception as e:
            logging.error("call attempt %d failed: %s", attempt + 1, e)
            if attempt < retries:
                print(f"      [RETRY {attempt + 1}] {e}")
                time.sleep(5)
    return None


def parse_json_from(text: str) -> Optional[dict]:
    """Extract first JSON object from text, tolerating markdown fences."""
    text = re.sub(r'```(?:json)?', '', text).strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


# -- Task Banks ----------------------------------------------------------------

COACTIVE_TASKS = [
    {
        "id": 0,
        "name": "calculate_discount",
        "code": (
            'def calculate_discount(price, quantity):\n'
            '    """15% off for qty >= 10, 8% off for qty >= 5, else no discount."""\n'
            '    if quantity > 10:\n'
            '        discount = 0.15\n'
            '    elif quantity > 5:\n'
            '        discount = 0.08\n'
            '    else:\n'
            '        discount = 0.0\n'
            '    return price * quantity * (1 - discount)'
        ),
        "known_bugs": [
            "First threshold uses > 10 instead of >= 10: customers buying exactly 10 items miss the bulk discount",
            "Second threshold uses > 5 instead of >= 5: customers buying exactly 5 items miss the mid-tier discount",
        ],
    },
    {
        "id": 1,
        "name": "find_second_largest",
        "code": (
            'def find_second_largest(numbers):\n'
            '    """Returns the second largest unique value in the list."""\n'
            '    if len(numbers) < 2:\n'
            '        return None\n'
            '    unique = list(set(numbers))\n'
            '    unique.sort()\n'
            '    return unique[1]'
        ),
        "known_bugs": [
            "After ascending sort, index [1] returns the second smallest, not second largest (should use unique[-2])",
            "If all elements are identical, unique has length 1 and unique[1] raises IndexError instead of returning None",
        ],
    },
    {
        "id": 2,
        "name": "rotate_list_left",
        "code": (
            'def rotate_list(lst, k):\n'
            '    """Left-rotate list by k positions: [1,2,3,4,5] by 2 -> [3,4,5,1,2]"""\n'
            '    if not lst:\n'
            '        return lst\n'
            '    k = k % len(lst)\n'
            '    return lst[-k:] + lst[:-k]'
        ),
        "known_bugs": [
            "lst[-k:] + lst[:-k] is a RIGHT rotation; left rotation requires lst[k:] + lst[:k]",
            "When k == 0 after modulo, lst[-0:] + lst[:-0] evaluates to lst + [] which accidentally returns the correct result, masking the bug for k=0",
        ],
    },
    {
        "id": 3,
        "name": "binary_search",
        "code": (
            'def binary_search(arr, target):\n'
            '    """Returns the index of target in a sorted array, or -1 if not found."""\n'
            '    low, high = 0, len(arr)\n'
            '    while low < high:\n'
            '        mid = (low + high) // 2\n'
            '        if arr[mid] == target:\n'
            '            return mid\n'
            '        elif arr[mid] < target:\n'
            '            low = mid + 1\n'
            '        else:\n'
            '            high = mid - 1\n'
            '    return -1'
        ),
        "known_bugs": [
            "high is initialised to len(arr) instead of len(arr) - 1; on an array whose last element equals target, mid can reach the correct index but the loop condition exits before checking it",
            "Loop condition 'low < high' should be 'low <= high'; when low == high, the single remaining candidate is never checked, causing misses",
        ],
    },
    {
        "id": 4,
        "name": "celsius_to_fahrenheit_filtered",
        "code": (
            'def celsius_to_fahrenheit_list(temps, min_celsius=None):\n'
            '    """Convert Celsius to Fahrenheit; skip temps strictly below min_celsius."""\n'
            '    result = []\n'
            '    for t in temps:\n'
            '        if min_celsius is not None and t < min_celsius:\n'
            '            continue\n'
            '        f = t * 9 + 32\n'
            '        result.append(f)\n'
            '    return result'
        ),
        "known_bugs": [
            "Formula t * 9 + 32 is missing division by 5; correct formula is t * 9/5 + 32",
            "Filter condition t < min_celsius uses strict less-than; the docstring says 'strictly below' but the intended behaviour (exclude boundary) requires t <= min_celsius",
        ],
    },
    {
        "id": 5,
        "name": "find_pair_with_sum",
        "code": (
            'def find_pair_with_sum(numbers, target):\n'
            '    """Returns (i, j) such that numbers[i] + numbers[j] == target."""\n'
            '    seen = {}\n'
            '    for i, num in enumerate(numbers):\n'
            '        complement = target - num\n'
            '        if complement in seen:\n'
            '            return (seen[complement], i)\n'
            '        seen[i] = num\n'
            '    return None'
        ),
        "known_bugs": [
            "Dictionary stores seen[i] = num (index as key, value as value); lookup checks 'complement in seen' which searches keys (indices), not values -- should store seen[num] = i",
        ],
    },
    {
        "id": 6,
        "name": "group_by_first_letter",
        "code": (
            'def group_by_first_letter(words):\n'
            '    """Groups words by first letter (lowercase). Returns {letter: [words]}."""\n'
            '    groups = {}\n'
            '    for word in words:\n'
            '        if not word:\n'
            '            continue\n'
            '        key = word[0]\n'
            '        if key in groups:\n'
            '            groups[key] = []\n'
            '        groups[key].append(word)\n'
            '    return groups'
        ),
        "known_bugs": [
            "key = word[0] is case-sensitive; 'Apple' and 'ant' go to separate groups (fix: word[0].lower())",
            "Branch 'if key in groups: groups[key] = []' resets an existing group to empty each time; should be 'else: groups[key] = []' to initialise only when absent",
        ],
    },
    {
        "id": 7,
        "name": "find_missing_number",
        "code": (
            'def find_missing_number(nums):\n'
            '    """Given n-1 unique integers from 1..n, find the missing one."""\n'
            '    n = len(nums) + 1\n'
            '    expected_sum = n * (n - 1) // 2\n'
            '    return expected_sum - sum(nums)'
        ),
        "known_bugs": [
            "Sum formula uses n*(n-1)//2 (sum of 0..n-1) instead of n*(n+1)//2 (sum of 1..n), producing a result that is n less than correct",
        ],
    },
]


_NAMING_INSTRUCTION = (
    "IMPORTANT: Worker A must choose non-obvious, domain-specific field names "
    "that Worker B would not conventionally guess. Avoid generic names like "
    "'name', 'value', 'data', 'items', 'list'. Use abbreviated or domain-coined "
    "terms (e.g., 'enrolment_units' instead of 'credits', 'ledger_rows' instead of 'entries'). "
    "Worker B must use A's exact field names -- not invent equivalent ones."
)

HIERARCHICAL_TASKS = [
    {
        "id": 0,
        "system": "student grade tracker",
        "brief": (
            "Build a student grade tracker. "
            "Worker A: design Python classes for Student and CourseGrade with relevant fields. "
            "Worker B: implement calculate_gpa(student) that computes the student's weighted GPA. "
            "The supervisor has NOT specified field names, GPA scale, or how credits are stored -- "
            "Worker A decides these; Worker B must match A's design exactly. "
            + _NAMING_INSTRUCTION
        ),
    },
    {
        "id": 1,
        "system": "library checkout system",
        "brief": (
            "Build a library book checkout system. "
            "Worker A: design Python classes for Book and Borrower. "
            "Worker B: implement checkout_book(book, borrower, library_inventory) "
            "that marks a book as checked out and sets a due date. "
            "The supervisor has NOT specified field names, due-date rules, or maximum books per borrower. "
            + _NAMING_INSTRUCTION
        ),
    },
    {
        "id": 2,
        "system": "restaurant order billing",
        "brief": (
            "Build a restaurant billing system. "
            "Worker A: design Python classes for MenuItem and Order. "
            "Worker B: implement generate_bill(order) that returns the total including tax. "
            "The supervisor has NOT specified field names, tax rate, or item structure. "
            + _NAMING_INSTRUCTION
        ),
    },
    {
        "id": 3,
        "system": "employee payroll",
        "brief": (
            "Build an employee payroll calculator. "
            "Worker A: design Python classes for Employee and TimeEntry. "
            "Worker B: implement calculate_weekly_pay(employee, entries) with overtime handling. "
            "The supervisor has NOT specified field names, overtime threshold, or overtime multiplier. "
            + _NAMING_INSTRUCTION
        ),
    },
    {
        "id": 4,
        "system": "event ticket booking",
        "brief": (
            "Build an event ticket booking system. "
            "Worker A: design Python classes for Event and Booking. "
            "Worker B: implement reserve_seats(event, num_seats, customer_name) "
            "that checks availability and returns a booking confirmation. "
            "The supervisor has NOT specified field names, seat numbering, or overbooking policy. "
            + _NAMING_INSTRUCTION
        ),
    },
    {
        "id": 5,
        "system": "inventory restock alerts",
        "brief": (
            "Build an inventory management system. "
            "Worker A: design Python classes for Product and StockLevel. "
            "Worker B: implement generate_restock_alerts(products) "
            "that returns a list of products needing restocking. "
            "The supervisor has NOT specified field names, threshold strategy, or alert format. "
            + _NAMING_INSTRUCTION
        ),
    },
    {
        "id": 6,
        "system": "task priority queue",
        "brief": (
            "Build a project task manager. "
            "Worker A: design Python classes for Task and Project. "
            "Worker B: implement get_priority_queue(project) "
            "that returns tasks sorted by priority, highest first. "
            "The supervisor has NOT specified the priority field type, sort criteria, or task structure. "
            + _NAMING_INSTRUCTION
        ),
    },
    {
        "id": 7,
        "system": "blog comment threading",
        "brief": (
            "Build a blog comment system. "
            "Worker A: design Python classes for Post and Comment (supporting replies). "
            "Worker B: implement get_comment_thread(post, all_comments) "
            "that returns comments as a nested tree structure. "
            "The supervisor has NOT specified field names, reply depth limit, or parent reference. "
            + _NAMING_INSTRUCTION
        ),
    },
]


# -- Worker A ------------------------------------------------------------------

def run_worker_a_coactive(task: dict) -> Optional[str]:
    return call([
        {"role": "system", "content": "You are a careful code analyst."},
        {"role": "user", "content": (
            f"Analyze this Python function and identify ALL bugs:\n\n"
            f"```python\n{task['code']}\n```\n\n"
            "For each bug: state exactly what is wrong, why it produces incorrect behaviour, "
            "and what the correct code should be. Be specific about which line or expression is faulty."
        )},
    ], temperature=0.1)


def run_worker_a_hierarchical(task: dict) -> Optional[str]:
    return call([
        {"role": "system", "content": "You are a software engineer (Worker A) designing data models."},
        {"role": "user", "content": (
            f"Supervisor brief: {task['brief']}\n\n"
            "Your job is Worker A. Design the Python data model classes described above. "
            "Define all classes with EXPLICIT, NON-OBVIOUS field names and types -- "
            "do NOT use generic names like 'name', 'value', 'data', or 'items'. "
            "Coin domain-specific abbreviated terms. "
            "Worker B will implement a function using your classes; they must ask you "
            "if they do not know your field names."
        )},
    ], temperature=0.3)


# -- Handoff Condition Helpers -------------------------------------------------

def extract_structured_bugs(a_output: str) -> str:
    """For message_pass: extract JSON bug list from A's analysis."""
    result = call([
        {"role": "system", "content": "Extract structured data. Respond ONLY with valid JSON."},
        {"role": "user", "content": (
            f"Extract a structured bug list from this analysis.\n\nAnalysis:\n{a_output}\n\n"
            'JSON format: {"bugs": [{"description": "...", "location": "...", "fix": "..."}]}'
        )},
    ], temperature=0)
    return result or '{"bugs": []}'


def extract_structured_model(a_output: str) -> str:
    """For message_pass: extract JSON class schema from A's design."""
    result = call([
        {"role": "system", "content": "Extract structured data. Respond ONLY with valid JSON."},
        {"role": "user", "content": (
            f"Extract a concise class schema from this design.\n\nDesign:\n{a_output}\n\n"
            'JSON format: {"classes": [{"name": "...", "fields": [{"name": "...", "type": "..."}]}]}'
        )},
    ], temperature=0)
    return result or '{"classes": []}'


def generate_b_questions(task_type: str, task: dict) -> list[str]:
    """B generates clarifying questions WITHOUT seeing A's output yet."""
    if task_type == 'coactive_pipeline':
        context = f"You need to fix bugs in this Python function:\n```python\n{task['code']}\n```"
        instruction = (
            f"A code analyst has reviewed this function. Before seeing their report, "
            f"generate exactly {MAX_QUESTIONS} specific questions you would ask them -- "
            "focus on code patterns you find suspicious or logic you find ambiguous. "
            f"Output exactly {MAX_QUESTIONS} numbered questions, nothing else."
        )
    else:
        context = f"Supervisor brief: {task['brief']}"
        instruction = (
            "You are Worker B. Worker A has designed the data model. "
            f"Before implementing your function, generate exactly {MAX_QUESTIONS} specific questions "
            "about Worker A's design -- field names, types, naming conventions, or structural decisions. "
            f"Output exactly {MAX_QUESTIONS} numbered questions, nothing else."
        )
    response = call([
        {"role": "system", "content": "You are a software engineer preparing to implement code."},
        {"role": "user", "content": f"{context}\n\n{instruction}"},
    ], temperature=0.3)
    if not response:
        return []
    questions = []
    for line in response.strip().split('\n'):
        q = re.sub(r'^\d+[.)]\s*', '', line.strip())
        if q and len(q) > 10:
            questions.append(q)
    return questions[:MAX_QUESTIONS]


def answer_question(question: str, a_output: str) -> str:
    """Simulate A answering B's question from A's full reasoning context."""
    result = call([
        {"role": "system", "content": "You are a software engineer who completed an analysis."},
        {"role": "user", "content": (
            f"Your analysis/design:\n{a_output}\n\n"
            f"A colleague asks: {question}\n\n"
            "Answer based strictly on your analysis above. Be specific and concise."
        )},
    ], temperature=0.1)
    return result or "No answer available."


def build_b_context(condition: str, task_type: str, task: dict, a_output: str) -> tuple[str, int]:
    """
    Returns (context_string_for_B, questions_asked_count).
    questions_asked_count is non-zero only for the interactive condition.
    """
    if condition == 'none':
        return "", 0

    if condition == 'static':
        if task_type == 'coactive_pipeline':
            return f"A code analyst reviewed this function and wrote:\n\n{a_output}", 0
        else:
            return f"Worker A designed the following data model:\n\n{a_output}", 0

    if condition == 'message_pass':
        if task_type == 'coactive_pipeline':
            structured = extract_structured_bugs(a_output)
            return f"Structured bug report from code analyst:\n{structured}", 0
        else:
            structured = extract_structured_model(a_output)
            return f"Worker A's class schema (JSON):\n{structured}", 0

    if condition == 'interactive':
        questions = generate_b_questions(task_type, task)
        if not questions:
            return f"(No questions generated. Worker A's notes:)\n{a_output}", 0
        qa_pairs = []
        for q in questions:
            answer = answer_question(q, a_output)
            qa_pairs.append(f"Q: {q}\nA: {answer}")
        context = "Worker A answered your questions:\n\n" + "\n\n".join(qa_pairs)
        return context, len(questions)

    return "", 0


# -- Worker B ------------------------------------------------------------------

def run_worker_b_coactive(task: dict, b_context: str) -> Optional[str]:
    prefix = f"{b_context}\n\n" if b_context else ""
    return call([
        {"role": "system", "content": "You are a Python developer. Fix bugs as instructed."},
        {"role": "user", "content": (
            f"{prefix}Fix ALL bugs in this Python function:\n```python\n{task['code']}\n```\n\n"
            "Return ONLY the corrected Python function. No explanation."
        )},
    ], temperature=0)


def run_worker_b_hierarchical(task: dict, b_context: str) -> Optional[str]:
    no_model_note = (
        "You have NOT seen Worker A's data model. Make reasonable assumptions about field names "
        "and types.\n" if not b_context else ""
    )
    prefix = f"{b_context}\n\n" if b_context else ""
    return call([
        {"role": "system", "content": "You are a software engineer (Worker B) implementing processing functions."},
        {"role": "user", "content": (
            f"Supervisor brief: {task['brief']}\n\n"
            f"{prefix}{no_model_note}"
            "Your job is Worker B. Implement the processing function described in the brief. "
            "If you genuinely cannot proceed without a specific design detail from Worker A, "
            "write 'SUPERVISOR CLARIFICATION NEEDED: <question>' then implement your best guess."
        )},
    ], temperature=0.2)


# -- Scoring -------------------------------------------------------------------

def score_coactive(task: dict, b_output: str) -> dict:
    """LLM judge: did B fix each of the known bugs?"""
    bugs_str = "\n".join(f"- {b}" for b in task['known_bugs'])
    result = call([
        {"role": "system", "content": "Code evaluator. Respond ONLY with valid JSON."},
        {"role": "user", "content": (
            f"These bugs were present in a Python function:\n{bugs_str}\n\n"
            f"A developer produced this fix:\n```python\n{b_output}\n```\n\n"
            "For each bug listed above (in order), was it correctly fixed in the code above? "
            '{"fixed": [true/false per bug], "overall_success": true/false}'
        )},
    ], temperature=0)
    data = parse_json_from(result or '')
    if not data or 'fixed' not in data:
        return {"detail_preservation": 0.0, "task_success": False}
    fixed = [bool(f) for f in data.get('fixed', [])]
    if not fixed:
        return {"detail_preservation": 0.0, "task_success": False}
    preservation = sum(fixed) / len(fixed)
    return {
        "detail_preservation": round(preservation, 3),
        "task_success": bool(data.get('overall_success', preservation >= 1.0)),
    }


def score_hierarchical(task: dict, a_output: str, b_output: str) -> dict:
    """LLM judge: does B use A's exact field names?"""
    clarification_count = b_output.lower().count('supervisor clarification needed') if b_output else 0
    result = call([
        {"role": "system", "content": "Code evaluator. Respond ONLY with valid JSON."},
        {"role": "user", "content": (
            f"System: {task['system']}\n\n"
            f"Worker A's data model (authoritative field names):\n{a_output}\n\n"
            f"Worker B's processing function:\n{b_output}\n\n"
            "Rate strictly:\n"
            "1. interface_score (0-10): Does B reference A's EXACT field names as defined above?\n"
            "   - Score 0-3: B invents its own field names not present in A's model\n"
            "   - Score 4-6: B uses some of A's field names but guesses or invents others\n"
            "   - Score 7-9: B uses most of A's field names correctly\n"
            "   - Score 10: B uses only A's exact field names throughout, no invented names\n"
            "2. completeness_score (0-10): Is B's function a complete working implementation "
            "(ignoring field name correctness)?\n"
            '{"interface_score": int, "completeness_score": int}'
        )},
    ], temperature=0)
    data = parse_json_from(result or '')
    if not data:
        return {"detail_preservation": 0.0, "task_success": False, "clarification_requests": clarification_count}
    interface    = min(max(int(data.get('interface_score', 0)), 0), 10) / 10
    completeness = min(max(int(data.get('completeness_score', 0)), 0), 10) / 10
    return {
        "detail_preservation": round(interface, 3),
        "task_success": interface >= 0.6 and completeness >= 0.6,
        "clarification_requests": clarification_count,
    }


# -- Experiment Loop -----------------------------------------------------------

OUT_PATH = Path(__file__).parent / 'handoff_results.json'


def save(results: dict):
    try:
        OUT_PATH.write_text(json.dumps(results, indent=2), encoding='utf-8')
    except Exception as e:
        logging.error("save failed: %s", e)


def load_prior() -> dict:
    if OUT_PATH.exists():
        try:
            return json.loads(OUT_PATH.read_text())
        except Exception:
            pass
    return {}


def _partial_key(task_type: str, condition: str) -> str:
    return f"{task_type}|{condition}"


def summarise(results: dict):
    print("\n--- CURRENT SUMMARY ---")
    for task_type, cond_results in results.items():
        print(f"\n  {task_type}")
        print(f"  {'Condition':<16} {'Status':<12} {'Detail Pres':>12} {'Success':>8} {'Avg Q':>7} {'Avg CR':>7}")
        for cond, r in cond_results.items():
            if not r.get('completed', False):
                n = len(r.get('agents', []))
                print(f"  {cond:<16} {'partial '+str(n)+'/8':<12}")
                continue
            q  = f"{r['avg_questions_asked']:.1f}"        if r.get('avg_questions_asked')        is not None else "--"
            cr = f"{r['avg_clarification_requests']:.1f}" if r.get('avg_clarification_requests') is not None else "--"
            print(f"  {cond:<16} {'done':<12} {r['avg_detail_preservation']:>12.2f} {r['avg_task_success']:>7.1%} {q:>7} {cr:>7}")


def run_experiment():
    all_results = load_prior()

    for task_type in TASK_TYPES:
        print(f"\n{'='*68}")
        print(f"TASK TYPE: {task_type.upper()}")
        print("="*68)
        task_bank = COACTIVE_TASKS if task_type == 'coactive_pipeline' else HIERARCHICAL_TASKS

        if task_type not in all_results:
            all_results[task_type] = {}

        for condition in CONDITIONS:
            prior_cond = all_results.get(task_type, {}).get(condition, {})
            if prior_cond.get('completed', False):
                print(f"\n  Condition: {condition.upper()} -- already complete, skipping.")
                continue

            print(f"\n  Condition: {condition.upper()}")
            print(f"  {'-'*62}")

            # Resume from any previously saved agent results for this condition
            prior_agents: list[dict] = prior_cond.get('agents', [])
            done_agent_nums = {r['agent'] for r in prior_agents}
            agent_results: list[dict] = list(prior_agents)

            for idx in range(NUM_AGENTS):
                agent_num = idx + 1
                if agent_num in done_agent_nums:
                    prior = next(r for r in prior_agents if r['agent'] == agent_num)
                    dp = prior['detail_preservation']
                    ts = 'Y' if prior['task_success'] else 'N'
                    print(f"\n    Agent {agent_num}/{NUM_AGENTS} -- already done (dp={dp:.2f} success={ts}), skipping.")
                    continue

                task = task_bank[idx % len(task_bank)]
                task_label = task.get('name', task.get('system', str(task['id'])))
                print(f"\n    Agent {agent_num}/{NUM_AGENTS} | {task_label}")

                t0 = time.time()

                # Worker A
                if task_type == 'coactive_pipeline':
                    a_output = run_worker_a_coactive(task)
                else:
                    a_output = run_worker_a_hierarchical(task)

                if not a_output:
                    print("      [ERROR] Worker A failed -- skipping agent.")
                    continue

                # Build B's context for this condition
                b_context, questions_asked = build_b_context(condition, task_type, task, a_output)

                # Worker B
                if task_type == 'coactive_pipeline':
                    b_output = run_worker_b_coactive(task, b_context)
                else:
                    b_output = run_worker_b_hierarchical(task, b_context)

                if not b_output:
                    print("      [ERROR] Worker B failed -- skipping agent.")
                    continue

                # Score
                if task_type == 'coactive_pipeline':
                    scores = score_coactive(task, b_output)
                else:
                    scores = score_hierarchical(task, a_output, b_output)

                latency = round(time.time() - t0, 1)
                entry = {
                    "agent": agent_num,
                    "condition": condition,
                    "task_id": task['id'],
                    "detail_preservation": scores["detail_preservation"],
                    "task_success": scores["task_success"],
                    "questions_asked": questions_asked,
                    "clarification_requests": scores.get("clarification_requests", 0),
                    "latency": latency,
                }
                agent_results.append(entry)

                dp = scores['detail_preservation']
                ts = 'Y' if scores['task_success'] else 'N'
                q_str  = f" | Q={questions_asked}" if condition == 'interactive' else ""
                cr_str = f" | CR={scores.get('clarification_requests', 0)}" if task_type == 'hierarchical_orchestration' else ""
                print(f"      dp={dp:.2f} | success={ts}{q_str}{cr_str} | {latency}s")

                logging.info("task=%s cond=%s agent=%d dp=%.3f success=%s",
                             task_type, condition, agent_num, dp, scores['task_success'])

                # Save partial progress after each agent
                all_results.setdefault(task_type, {})[condition] = {
                    "completed": False,
                    "agents": agent_results,
                }
                save(all_results)

            if not agent_results:
                continue

            n = len(agent_results)
            avg_dp      = sum(r['detail_preservation'] for r in agent_results) / n
            avg_success = sum(r['task_success'] for r in agent_results) / n
            avg_q  = sum(r['questions_asked'] for r in agent_results) / n if condition == 'interactive' else None
            avg_cr = sum(r['clarification_requests'] for r in agent_results) / n if task_type == 'hierarchical_orchestration' else None

            print(f"\n    Summary: dp={avg_dp:.2f} | success={avg_success:.1%}")

            all_results[task_type][condition] = {
                "completed": True,
                "avg_detail_preservation":    round(avg_dp, 3),
                "avg_task_success":           round(avg_success, 3),
                "avg_questions_asked":        round(avg_q, 2) if avg_q is not None else None,
                "avg_clarification_requests": round(avg_cr, 2) if avg_cr is not None else None,
                "agents": agent_results,
            }

            # Final save for completed condition
            save(all_results)
            print(f"    [saved -> {OUT_PATH.name}]")

    summarise(all_results)


if __name__ == '__main__':
    run_experiment()
