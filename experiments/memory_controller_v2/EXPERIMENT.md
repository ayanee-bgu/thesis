# E1b: Memory-Controller Lever Ablation

## Research Context

E1 (`experiments/memory_controller/`) established a graph-structured procedural
memory controller (`ProcedureGraph` + `SessionConsolidator` + `ContextInjector`)
and compared it to a flat MemP-style store. That work's own "what else can we
do" list named four levers it never tested:

1. Real sentence-embedding retrieval/dedup instead of TF-IDF (`embed_fn` was a
   wired parameter on both `SessionConsolidator` and `ContextInjector`, but
   `memory_controller/main.py` never passed it — every E1 run used TF-IDF only).
2. Coarse task_type/keyword tags for a cheap retrieval prefilter, mirroring the
   query-grounding fix identified in the interactive-oracle work
   (`advisor_report_interactive_oracle_synthesis.md` §4.6) — the graph path
   dropped `task_type` entirely even though the flat-store baseline captured it.
3. Reinforcement-weighted ranking — `_merge_content` picked whichever text was
   longer and threw away how many sessions had independently converged on the
   same procedure.
4. Chunked distillation instead of a hard 4000-char truncation before the LLM
   ever sees a long trajectory — the same truncation failure mode that made
   `full_context` lose information in the retrieval-oracle experiments (E7-E9),
   here applied to what the consolidator distills from.

This experiment isolates each lever with its own condition, all built on top of
`graph_session` (session-end consolidation — the update policy E1 already
favored), so any effect is attributable to the specific lever, not to bundling
everything at once.

## Conditions

| Condition | embed | keyword prefilter | reinforcement weighting | recluster |
|---|---|---|---|---|
| `graph_session_baseline` | – | – | – | – |
| `graph_session_embed` | ✓ | – | – | – |
| `graph_session_grounded` | – | ✓ | ✓ | – |
| `graph_session_full` | ✓ | ✓ | ✓ | ✓ |

`graph_session_baseline` should reproduce E1's `graph_session` behavior —
it's the control, not a new claim.

## Library changes (additive, backward compatible)

- `src/memory/embeddings.py` (new) — `get_embed_fn()` / `try_get_embed_fn()`,
  lazy-loads `sentence-transformers` (`all-MiniLM-L6-v2` by default). Every
  condition here falls back to TF-IDF automatically (with a logged warning) if
  the package isn't installed, so the harness never hard-fails on a missing
  optional dependency.
- `src/memory/graph_store.py` — `add_procedure()` gained `task_type` and
  `keywords` fields; nodes now track `reinforce_count`, bumped on every merge.
- `src/memory/consolidator.py` — `chunk_trajectory()` replaces the hard
  4000-char truncation with overlapping, step-boundary-aware windows, each
  distilled separately and merged; `extract_keywords()` is a cheap
  no-LLM-call TF top-N tagger; `SessionConsolidator.recluster()` is a new
  graph-wide re-consolidation pass (O(n²) in active node count — meant to run
  periodically, not after every trajectory).
- `src/memory/context_injector.py` — `inject()` takes an optional `task_type`
  and, when `use_keyword_prefilter=True`, narrows candidate nodes by
  task_type/keyword overlap *before* similarity scoring; `use_reinforcement_weighting=True`
  multiplies score by `log2(reinforce_count)`.
- `experiments/memory_controller/main.py` and `src/memory/consolidator.py` —
  added `"num_gpu": 0` to Ollama call options. Both were missing the CPU-forcing
  fix applied to later experiments after E5's GPU-offload crash discovery;
  without it, Ollama's backend process crashed outright on every call during
  this experiment's own smoke test (`llama-server process has terminated`).

## Metrics

Same as E1: `session_1_success_rate`, `session_2_success_rate`,
`delta_s1_to_s2` (the key metric — memory should help session 2), `mean_tokens_s2`,
`final_graph_nodes`. Worth adding per-condition once real runs are in: how often
the keyword prefilter actually changed which nodes got scored (log via
`ContextInjector._keyword_prefilter`'s return length vs the full active-node count).

## How to Run

```powershell
# Structural smoke test — no Ollama needed
python experiments/memory_controller_v2/main.py --mock --no-llm --num-tasks 4

# Real run, single condition, MockEnv (fast iteration)
python experiments/memory_controller_v2/main.py --mock --num-tasks 4 --condition graph_session_grounded

# Full ablation on TravelPlanner (drop --mock)
python experiments/memory_controller_v2/main.py --num-tasks 20
```

Install `sentence-transformers` before running `graph_session_embed` /
`graph_session_full` for real — without it they silently degrade to TF-IDF
and the ablation won't show anything for that lever.

## Status

- [x] Library changes implemented, backward compatible with E1's existing conditions.
- [x] Harness smoke-tested structurally (`--no-llm`, all 4 conditions, MockEnv).
- [x] Harness smoke-tested with real Ollama calls (`graph_session_grounded`,
      MockEnv, 4 tasks) — consolidation, chunked distillation, keyword
      extraction, and grounded retrieval all exercised successfully.
- [ ] `sentence-transformers` not installed in this environment — `graph_session_embed`
      and `graph_session_full` currently fall back to TF-IDF (with a warning).
      Install it to actually test the embedding lever.
- [ ] Full TravelPlanner run (all 4 conditions, ≥20 tasks/split) not yet done.
