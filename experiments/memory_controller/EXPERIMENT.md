# E1: Graph-Based Memory Controller for Agent Procedural Memory

## Research Context

**Thesis topic:** Model educability and knowledge transfer between agents.

**This experiment** implements and evaluates a graph-structured procedural memory controller as an enhancement to [MemP (Fang et al., 2025)](https://arxiv.org/abs/2508.06433). MemP stores distilled agent trajectories in a flat key-value store; we propose replacing this with a typed directed graph (NetworkX) that encodes relationships between procedures, and consolidating memory at session end (batch) rather than per-trajectory.

**Primary benchmark:** TravelPlanner (validation + test splits), sole-planning mode — same as MemP paper. Direct comparison to MemP's reported numbers is the goal.

**Secondary benchmark:** ALFWorld (household tasks) — requires WSL + build tools to install; deferred.

---

## Goals

| # | Goal | Status |
|---|---|---|
| G1 | Replicate MemP's flat-store baseline on TravelPlanner | **In progress** |
| G2 | Show `graph_session` ≥ `memp_flat` on success rate | Pending results |
| G3 | Show `graph_session` ≤ `memp_flat` on token cost | Pending results |
| G4 | Show session-end consolidation > per-trajectory update (G2 ablation) | Pending results |
| G5 | Cross-agent graph transfer (E2 experiment) | Not started |

---

## Conditions (E1)

| Condition | Memory | Update | Hypothesis |
|---|---|---|---|
| `no_memory` | None | — | Baseline |
| `memp_flat` | TF-IDF flat store | Per-trajectory (success only) | MemP replication |
| `graph_pertraj` | Graph (NetworkX) | Per-trajectory | Ablation |
| `graph_session` | Graph (NetworkX) | Session-end batch | **Proposed** |

---

## Metrics

**Primary (from MemP paper):**
- `success_rate` — proxy HC: plan parses correctly AND budget satisfied
- `mean_tokens` — average tokens per plan generation

**Additional (our contribution):**
- `format_complete_rate` — all required day fields present
- `budget_satisfied_rate` — estimated cost ≤ stated budget
- `retrieval_calls` — how often memory was retrieved (injection rate)
- `graph_nodes` — size of procedure graph at session end
- `conflicts_detected` — contradictory procedures flagged

**Formal eval (run separately after getting plans):**  
Use `.travelplanner/evaluation/eval.py` with the saved plan JSONs for official CS + HC scores comparable to the MemP paper.

---

## What Results We Need

### Minimum publishable result (G1–G4)
Run all 4 conditions on the **validation** split (20–50 tasks each) and produce:
1. A table: `success_rate`, `mean_tokens`, `mean_steps` per condition
2. A learning curve: success rate vs. task index (does `graph_session` improve as graph grows?)
3. An ablation: `graph_session` vs `graph_pertraj` — isolates the value of session-end consolidation

Expected direction (to confirm hypothesis):
```
success_rate:  graph_session ≥ memp_flat > graph_pertraj > no_memory
mean_tokens:   graph_session ≤ memp_flat  (fewer, more relevant procedures retrieved)
```

### For E2 (knowledge transfer, separate experiment)
See `experiments/knowledge_transfer/` — teacher runs N sessions, student bootstrapped from graph.

---

## Architecture

```
src/memory/
  graph_store.py       — ProcedureGraph (NetworkX, typed edges, JSON serializable)
  consolidator.py      — SessionConsolidator (LLM distillation → graph at session end)
  context_injector.py  — ContextInjector (k-hop subgraph retrieval → prompt prefix)

experiments/memory_controller/
  main.py              — Experiment runner (4 conditions, TravelPlanner + MockEnv)
  travelplanner_env.py — TravelPlannerEnv (validation + test splits, sole-planning mode)
  env_adapter.py       — MockEnv (testing without TravelPlanner) + ALFWorldAdapter stub
  analyze.py           — Metrics tables + bar charts
  config.yaml          — All hyperparameters
  EXPERIMENT.md        — This file
```

**Graph edge types:**
- `generalizes` — script node is abstract form of step nodes
- `depends_on` — step A requires step B to be done first  
- `conflicts_with` — two procedures give contradictory advice for same precondition
- `derived_from` — step/script came from the same trajectory/session as another

---

## How to Run

### Prerequisites
```powershell
# From the thesis root
pip install networkx pyyaml requests matplotlib numpy datasets pandas
# HF_TOKEN is read automatically from .env (already set)
```

### Quick test (3 tasks, no_memory only)
```powershell
python experiments/memory_controller/main.py --travelplanner --num-tasks 3 --condition no_memory
```

### Full E1 run — validation split only
```powershell
python experiments/memory_controller/main.py --travelplanner --num-tasks 20
```

### Full E1 run — validation + test simultaneously
```powershell
python experiments/memory_controller/main.py --travelplanner --split validation test --num-tasks 20
```
This generates plans for both splits in one pass. Validation results are evaluable immediately; test plans are saved for leaderboard submission.

### Single condition
```powershell
python experiments/memory_controller/main.py --travelplanner --condition graph_session --num-tasks 20
```

### Analyze results
```powershell
python experiments/memory_controller/analyze.py
```

### Mock run (no HF/Ollama needed — just tests the harness)
```powershell
python experiments/memory_controller/main.py --mock --num-tasks 5
```

---

## Config Knobs (`config.yaml`)

| Key | Default | Notes |
|---|---|---|
| `model.name` | `llama3:8b` | Swap for `qwen2.5:32b` or stronger for better plans |
| `model.timeout` | `600` | TravelPlanner plans take 3–5 min on llama3:8b |
| `experiment.num_tasks` | `20` | Increase to 50–180 for publishable results |
| `memory.k_hops` | `2` | Subgraph depth for context injection |
| `memory.token_budget` | `2000` | Max chars injected into prompt |
| `memory.dedup_threshold` | `0.82` | TF-IDF similarity above this → merge nodes |

---

## Current Status

### Done ✅
- `ProcedureGraph`, `SessionConsolidator`, `ContextInjector` implemented and unit-tested
- `TravelPlannerEnv` set up (sole-planning mode, validation + test splits, HF_TOKEN from .env)
- All 4 conditions wired in `main.py`
- `analyze.py` ready
- Mock env smoke test passed (all 4 conditions ran end-to-end)
- TravelPlanner database downloaded to `.travelplanner/database/`

### In progress ⏳
- TravelPlanner smoke test running (3 tasks, `no_memory` condition) — waiting on Ollama
- Known issue: `llama3:8b` takes ~3–5 min per TravelPlanner plan (large context)

### TODO for next session 📋
1. **Confirm smoke test passed** — check `results/memory_controller/no_memory.json`
2. **Run full E1** — `python experiments/memory_controller/main.py --travelplanner --num-tasks 20`
3. **Run E1 on both splits** — add `--split validation test` once validation run looks good
4. **Analyze** — `python experiments/memory_controller/analyze.py`
5. **Compare to MemP paper** — MemP reports ~50% success improvement on TravelPlanner over no-memory baseline; see if `graph_session` beats `memp_flat`
6. **Upgrade model** — if llama3:8b results are low, swap `model.name` to a stronger model (qwen2.5:32b or claude via API) — see `src/agents/cooperative_agent.py` for Ollama integration pattern
7. **E2 (knowledge transfer)** — once E1 complete, run `experiments/knowledge_transfer/main.py`

---

## Key Files to Read Before Continuing

| File | Why |
|---|---|
| `papers/summaries/2508.06433v4.md` | MemP paper — our baseline, metrics to beat |
| `src/memory/graph_store.py` | Core data structure |
| `src/memory/consolidator.py` | How trajectories → graph nodes |
| `src/memory/context_injector.py` | How graph → prompt prefix |
| `experiments/memory_controller/travelplanner_env.py` | Task env + evaluation logic |
| `results/memory_controller/summary.json` | Latest results (if run) |

---

## MemP Baseline Numbers (from paper, TravelPlanner validation)

| Condition | CS Score | HC Score | Notes |
|---|---|---|---|
| No memory (ReAct) | ~30% | ~5% | Paper's no-memory baseline |
| MemP (GPT-4o) | ~55% | ~10% | Best MemP result |
| MemP cross-model transfer | ~48% | ~8% | GPT-4o memory → Qwen2.5-14B |

Our proxy metric (`format_complete AND budget_satisfied`) is not identical to HC score. **Do not directly compare numbers** — use relative ordering across conditions. For formal HC/CS comparison, run `.travelplanner/evaluation/eval.py` on saved plans.

---

## Leaderboard Submission (test split)

Plans generated for the test split should be saved to:
```
results/memory_controller/test_plans_{condition}.jsonl
```
Submit to: https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard
