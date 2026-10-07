# E1: Graph Memory Controller vs. MemP Flat Store

## Research Question

Does graph-structured procedural memory with session-end consolidation improve agent performance across sessions, compared to a flat store (MemP baseline) and no memory?

## Key Design: Two Sessions

The experiment runs **2 sessions** per condition with **different tasks each session**:

- **Session 1** — agent runs tasks cold; memory conditions build their store/graph
- **Session 2** — agent runs new tasks with memory from session 1 available

The primary metric is the **S1→S2 delta**: how much does success rate improve from session 1 to session 2? `no_memory` should show Δ≈0 (no learning). Memory conditions should show Δ>0.

> **Why 2 sessions?** In a single session, `graph_session` runs entirely cold (graph is empty until end of session 1). Without a second session, we can't test whether the graph actually helps — only that it was written. The S1→S2 delta is the honest test of whether memory transfers knowledge.

## Conditions

| Condition | Session 1 | Session 2 | What it tests |
|---|---|---|---|
| `no_memory` | Cold | Cold | True baseline — no learning across sessions |
| `memp_flat` | Builds flat store per successful task | Retrieves from S1 flat store | Does a list of past plans help? |
| `graph_pertraj` | Builds graph per task + retrieves mid-session | Retrieves from S1+S2 graph | Does early per-trajectory noise cancel graph benefit? |
| `graph_session` | Cold (graph built at S1 end) | Retrieves from clean S1 graph | Does batch-consolidated memory help? (proposed) |

## Expected Results

| Condition | S1→S2 Δ | Reasoning |
|---|---|---|
| `no_memory` | ~0% | No memory, no improvement |
| `memp_flat` | +5–15% | Successful past plans provide useful retrieval signal |
| `graph_pertraj` | Small or negative | Per-trajectory noise may outweigh graph structure benefit |
| `graph_session` | Highest | Clean session-consolidated graph, no early-trajectory noise |

## Setup

```bash
pip install -r experiments/memory_controller/requirements.txt
# HF_TOKEN is read automatically from .env
```

## Run

```bash
# Instant structural demo — no Ollama, shows 2-session logic and S1→S2 delta
python experiments/memory_controller/main.py --mock --no-llm

# Full mock run with real LLM (~15 min)
python experiments/memory_controller/main.py --mock

# Full TravelPlanner run — validation + test splits (~4-5 hours, run overnight)
python experiments/memory_controller/main.py --travelplanner --split validation test

# Single condition only
python experiments/memory_controller/main.py --mock --no-llm --condition graph_session
```

## Analyze

```bash
python experiments/memory_controller/analyze.py
```

## Config (`config.yaml`)

| Key | Default | Notes |
|---|---|---|
| `experiment.num_sessions` | 2 | Number of sessions per condition |
| `experiment.tasks_per_session` | 5 | Tasks per session (MockEnv has 10 total; TravelPlanner has 180) |
| `model.name` | `llama3:8b` | Swap for stronger model for better TravelPlanner results |
| `model.timeout` | 600 | TravelPlanner needs ~3–5 min per plan on llama3:8b |
| `memory.k_hops` | 2 | Subgraph depth for context injection |
| `memory.token_budget` | 2000 | Max chars of memory injected into prompt |

## Files

| File | Purpose |
|---|---|
| `config.yaml` | All hyperparameters |
| `main.py` | Experiment runner (4 conditions × 2 sessions) |
| `travelplanner_env.py` | TravelPlanner sole-planning mode (validation + test) |
| `env_adapter.py` | MockEnv (10 tasks) + ALFWorldAdapter stub |
| `analyze.py` | Metrics tables + bar charts |
| `EXPERIMENT.md` | Full research context, goals, and next steps |

## Output

Results written to `results/memory_controller/{condition}.json`.  
Key field: `summary.delta_s1_to_s2` — the S1→S2 success rate improvement.
