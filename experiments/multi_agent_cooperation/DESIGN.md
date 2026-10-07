# Multi-Agent Cooperation Experiment Design

## Research Goal

Design and evaluate a **generalized cooperation protocol** for groups of truly differentiated LLM agents solving multi-step reasoning tasks. The core question:

> Does explicit inter-agent knowledge transfer — via agent-scoped memory retrieval — improve accuracy and reduce coordination overhead compared to message-passing summaries alone?

This is distinct from prior multi-agent work (debate, mixture-of-agents) because:
1. Agents differ in **model weights** (different architectures/training objectives)
2. Agents differ in **tools** (calculator, code interpreter, retrieval, search)
3. Knowledge sharing is **explicit and asymmetric** — agents must request what they need from each other's memory, rather than broadcasting to a shared context

The protocol is designed to generalize: swapping in new agents, benchmarks, or memory backends should require minimal changes.

---

## Agents

Each agent maps to a distinct Ollama model and a corresponding tool set:

| Agent | Model | Speciality | Tools |
|---|---|---|---|
| Reasoner | `deepseek-r1:32b` | Multi-step logical/mathematical reasoning | None (pure chain-of-thought) |
| Coder | `qwen2.5-coder:32b` | Symbolic computation, code execution | Python interpreter, calculator |
| Retriever | `command-r:35b` | Knowledge lookup, RAG, citation | Local RAG over knowledge base |
| Generalist | `llama3:8b` | Synthesis, consensus, fast routing | None |

**Why these models:** deepseek-r1 has an explicit reasoning trace we can analyze; qwen2.5-coder is purpose-built for tool use; command-r was designed for retrieval-augmented generation; llama3 is the fastest and serves as a baseline/synthesis agent.

---

## Coordination Protocol

### Memory Structure

```
experiments/multi_agent_cooperation/
  agents/
    agent_<role>/
      memory/
        observations_round_N.md    ← what the agent learned this round
        beliefs.md                 ← current best answer + confidence (0-1)
        queries_sent.md            ← requests made to other agents
  shared/
    task.md                        ← problem statement
    round_log.json                 ← per-round votes, consensus state, timing
```

### Per-Round Flow

```
Round N:
  1. Each agent reads task.md + its own memory/
  2. Agent optionally requests files from another agent's memory/ (transfer primitive)
  3. Agent reasons using its tools, updates beliefs.md, writes observations_round_N.md
  4. Agent broadcasts a structured summary (message passing)
  5. All agents cast a vote: {answer, confidence}
  6. Check consensus: if majority agree above threshold → terminate
  7. Otherwise: next round (max rounds = configurable, default 5)
```

### Termination

- **Consensus**: majority vote agreement above confidence threshold (default 0.7)
- **Max rounds**: hard cap (default 5) to prevent runaway costs
- **Tie-break**: Generalist agent answer is final if no consensus after max rounds

### Structure Variants (both will be tested)

- **Flat**: all agents communicate peer-to-peer, no designated coordinator
- **Hierarchical**: Generalist acts as coordinator, routes sub-tasks, aggregates answers

---

## Benchmarks

### 1. MATH (Hendrycks et al., 2021)
**Paper**: *Measuring Mathematical Problem Solving With the MATH Dataset*

**Why**: Five subject domains (algebra, geometry, number theory, counting, calculus) map naturally to specialist roles. Problems have verifiable ground-truth answers. Difficulty levels 1–5 allow ablation by complexity.

**How we use it**: Assign each domain to the most relevant specialist. Measure whether cross-agent retrieval improves accuracy on problems that require combining domains (e.g., a geometry problem needing algebraic manipulation).

**Baseline**: Single deepseek-r1:32b agent (best single-model expected performance).

**Expected result**: Group outperforms baseline on Level 4–5 problems that cross domain boundaries. Flat coordination likely sufficient here.

---

### 2. GAIA (Mialon et al., 2023)
**Paper**: *GAIA: A Benchmark for General AI Assistants*

**Why**: Explicitly designed for multi-step tasks requiring heterogeneous capabilities — web search, file reading, calculator, code execution. The benchmark's three difficulty levels map cleanly to our coordination overhead hypothesis.

**How we use it**: Each task implicitly requires different tool combinations. Measure whether agents self-select the right specialist (via memory requests) vs. needing explicit routing.

**Baseline**: Single qwen2.5-coder:32b agent with all tools (strongest single-agent expected).

**Expected result**: Group with memory retrieval outperforms single agent on Level 2–3 tasks. Hierarchical coordination may outperform flat here due to task complexity.

---

## Experimental Conditions

| Condition | Description |
|---|---|
| `single_best` | Best single model with all tools (upper-single baseline) |
| `group_msgpass` | Group with message-passing summaries only (no memory retrieval) |
| `group_memory` | Group with message-passing + agent-scoped memory retrieval |
| `flat` | Group with peer-to-peer coordination |
| `hierarchical` | Group with Generalist as coordinator |

Full factorial: 2 benchmarks × 2 structures × 2 memory conditions = 8 primary conditions + 2 baselines.

---

## Metrics

| Metric | What it measures |
|---|---|
| **Accuracy** | Final answer correctness vs. ground truth |
| **Rounds to consensus** | Coordination efficiency |
| **Memory requests per round** | How much inter-agent transfer actually occurs |
| **Agent contribution rate** | Which agents drive correct answers (did specialization matter?) |
| **Confidence calibration** | Do agents know when they're right? |
| **Token cost per task** | Practical overhead of group vs. single agent |

---

## Expected Results & Hypotheses

**H1**: Group + memory retrieval outperforms group + message-passing alone on cross-domain tasks (MATH Level 4–5, GAIA Level 2–3).

**H2**: Hierarchical coordination reduces rounds-to-consensus on GAIA but not on MATH (GAIA tasks are more decomposable; MATH tasks benefit from parallel reasoning).

**H3**: Agent contribution rate will be uneven — deepseek-r1 will drive most correct answers on MATH; qwen2.5-coder on GAIA. This validates that model differentiation matters.

**H4**: Memory requests cluster in rounds 2–3, not round 1 (agents first form their own beliefs, then seek external knowledge to resolve uncertainty).

---

## Implementation Sessions

Each session is self-contained and ends with a working, testable artifact.

---

### Session 1 — Core Framework
**Goal**: Agent class, memory read/write, round loop, consensus vote, logging.
**Deliverable**: `src/agents/cooperative_agent.py` + `src/protocols/round_loop.py`
**Test**: Run 2 agents on a single hardcoded MATH problem, verify round_log.json is written.
**Does NOT include**: Real benchmarks, tool integrations, model switching.

---

### Session 2 — MATH Benchmark Integration
**Prerequisite**: Session 1 complete.
**Goal**: Load MATH dataset, assign problems to agents by domain, run full evaluation loop, compute accuracy.
**Deliverable**: `experiments/multi_agent_cooperation/math_eval.py` + results in `results/math/`
**Test**: Run on 20 Level-1 problems, verify accuracy > single-agent baseline (sanity check).
**Does NOT include**: GAIA, hierarchical mode, memory retrieval (uses message-passing only first).

---

### Session 3 — Agent-Scoped Memory Retrieval
**Prerequisite**: Session 2 complete.
**Goal**: Implement the memory request protocol — agents can read from each other's `memory/` folders. Add to round loop. Re-run MATH under `group_memory` condition.
**Deliverable**: `src/protocols/memory_transfer.py`, updated round loop, new results column.
**Test**: Verify that `queries_sent.md` is populated and that retrieved content appears in subsequent `observations_round_N.md`.

---

### Session 4 — Tool Integrations
**Prerequisite**: Session 1 complete (can run in parallel with Sessions 2–3).
**Goal**: Wire up tools for Coder agent (Python interpreter) and Retriever agent (RAG over local knowledge base).
**Deliverable**: `src/agents/tools/` with `python_exec.py`, `rag_retrieval.py`
**Test**: Coder agent solves a MATH algebra problem using code execution. Retriever agent pulls a relevant paper chunk.

---

### Session 5 — GAIA Benchmark Integration
**Prerequisite**: Sessions 3 and 4 complete.
**Goal**: Load GAIA dataset, run all experimental conditions, collect full metrics.
**Deliverable**: `experiments/multi_agent_cooperation/gaia_eval.py` + results in `results/gaia/`
**Test**: Run on 10 Level-1 GAIA tasks, verify round_log.json captures tool calls and memory requests.

---

### Session 6 — Hierarchical Coordination
**Prerequisite**: Session 5 complete.
**Goal**: Implement coordinator role for Generalist agent. Route sub-tasks, aggregate votes. Run flat vs. hierarchical comparison on both benchmarks.
**Deliverable**: `src/protocols/hierarchical_loop.py`, updated results with `flat` vs `hierarchical` column.
**Test**: Verify Generalist issues routing messages in round_log.json.

---

### Session 7 — Analysis & Visualization
**Prerequisite**: All results collected.
**Goal**: Compute all metrics, generate plots, test H1–H4.
**Deliverable**: `experiments/multi_agent_cooperation/analysis.ipynb` + `results/figures/`
**Plots**: Accuracy by condition, rounds-to-consensus distribution, agent contribution heatmap, memory request timing.

---

## File Structure (final state)

```
experiments/multi_agent_cooperation/
  DESIGN.md                  ← this file
  math_eval.py
  gaia_eval.py
  analysis.ipynb
  agents/
    agent_reasoner/memory/
    agent_coder/memory/
    agent_retriever/memory/
    agent_generalist/memory/
  shared/
    round_log.json
    task.md
src/
  agents/
    cooperative_agent.py
    tools/
      python_exec.py
      rag_retrieval.py
  protocols/
    round_loop.py
    memory_transfer.py
    hierarchical_loop.py
results/
  math/
  gaia/
  figures/
```

---

## References

- Hendrycks et al. (2021). *Measuring Mathematical Problem Solving With the MATH Dataset*. NeurIPS.
- Mialon et al. (2023). *GAIA: A Benchmark for General AI Assistants*. arXiv:2311.12983.
- Du et al. (2023). *Improving Factuality and Reasoning in Language Models through Multiagent Debate*. arXiv:2305.14325.
- Wu et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*. arXiv:2308.08155.
- Wang et al. (2024). *Mixture-of-Agents Enhances Large Language Model Capabilities*. arXiv:2406.04692.
