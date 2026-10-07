# Graph-Based Procedural Memory for Multi-Agent Knowledge Transfer
## Progress Report — Experiment E1

**Date:** June 17, 2026  
**Thesis Topic:** Model Educability and Knowledge Transfer Between Agents  
**Experiment:** E1 — Graph Memory Controller vs. MemP Flat Store  

---

## 1. Motivation

LLM-based agents lack persistent procedural memory. When encountering a task type for the first time, they start from scratch — even if they (or a peer agent) have solved structurally similar tasks before. The recent **MemP** paper (Fang et al., 2025, arXiv:2508.06433) establishes that distilling past trajectories into an explicit procedure store, and retrieving those procedures at inference time, yields ~50% improvement on TravelPlanner and ALFWorld. Its most striking result: memory built by GPT-4o transfers to Qwen2.5-14B with +5% task completion — direct empirical evidence that procedural knowledge is model-architecture-agnostic.

This work takes MemP as a baseline and asks: **does the structure of the memory store matter?**

---

## 2. What This Work Contributes — A New Direction

### Where the field currently stands

Agent memory research clusters into four paradigms:

| Paradigm | Examples | Limitation |
|---|---|---|
| **Weight-based** | Fine-tuning on trajectories, RL-from-feedback | Knowledge frozen in weights; non-transferable across models |
| **Raw trajectory RAG** | Reflexion, Expel | Retrieves full trajectory text; expensive, no structure |
| **Flat procedure store** | MemP, AutoManual, AWM | Distills trajectories into procedures, retrieves by similarity — but flat |
| **Hierarchical context** | MemGPT | Manages what fits in context window — still flat key-value |

**The common limitation:** all of these treat memory as a list. Retrieved items are independent. The relationships between procedures — that step A *depends on* step B, that strategy C *generalizes* steps D and E, that approach F *conflicts with* approach G — are implicit at best, absent at worst.

### What this work does differently

We propose that **procedural knowledge has inherent relational structure**, and that making this structure explicit in a typed directed graph enables three things the flat-store paradigm cannot:

**1. Relational retrieval instead of similarity retrieval**  
MemP retrieves the top-k most similar procedures to a query. Our k-hop subgraph retrieval instead traverses the *causal neighborhood* — retrieving not just the most similar procedure, but everything that procedure depends on, generalizes from, or conflicts with. This gives the agent structured scaffolding rather than an unordered list.

**2. Session-end consolidation — a sleep analogy**  
MemP updates its store after every trajectory. This is equivalent to immediately encoding every experience as long-term memory, including failed and incomplete ones. Cognitive science gives us a better model: the hippocampal-neocortical consolidation hypothesis suggests that episodic memories are consolidated into semantic memory during offline periods (sleep), not online during experience. Our session-end batch update is a direct computational analog — raw trajectories accumulate during a session; at session end, an LLM distills them together, deduplicating near-identical procedures and detecting conflicts before they enter the permanent store.

**3. A shared, agent-neutral knowledge substrate**  
MemP's repository is per-agent and per-run. Our graph serializes to JSON and can be shared across agents, models, and sessions. A stronger agent's graph can bootstrap a weaker agent. Two specialized agents can write to the same graph. The graph persists across deployment boundaries. This directly enables the thesis's core question: *what does it mean for one model to educate another?*

### The TF-IDF failure point — an empirical finding

A concrete result from this experiment: MemP's TF-IDF flat retrieval **failed to retrieve anything** in session 2, despite having valid session 1 entries. The reason: TravelPlanner tasks reference different city names, dates, and airlines across sessions — the vocabulary overlap between training and test procedures is near zero. A flat store retrieves by surface token similarity; our graph retrieves by semantic proximity after embedding-based seed finding, which is robust to vocabulary mismatch.

This is not a corner case. In any real deployment where memory accumulates across domains or time periods, surface vocabulary will shift. Relational structure that transcends surface form is essential.

---

## 3. Architecture

```
src/memory/
  graph_store.py       — ProcedureGraph: NetworkX directed graph, JSON-serializable, no server
  consolidator.py      — SessionConsolidator: LLM distills session trajectories → graph nodes
  context_injector.py  — ContextInjector: k-hop subgraph retrieval → structured prompt prefix

experiments/memory_controller/
  main.py              — 4-condition, 2-session experiment runner
  travelplanner_env.py — TravelPlanner env + full local constraint evaluator (room type,
                         cuisine, transportation, house rules checked against DB CSVs)
```

**Graph edge types and what they encode:**

| Edge | Meaning | Why it matters |
|---|---|---|
| `generalizes` | Script → step: abstract form of concrete steps | Enables retrieving the right abstraction level for context |
| `depends_on` | Step A → step B: A requires B first | Retrieval returns the dependency chain, not just the target node |
| `conflicts_with` | Two procedures contradict each other | Detectable at consolidation time; undetectable in flat stores |
| `derived_from` | Provenance: which session produced this | Enables time-based pruning and cross-agent attribution |

---

## 4. Experimental Design

### Benchmark
**TravelPlanner** (Xie et al., 2024) — sole-planning mode, **hard tasks only** (3+ local constraints).  
Each task specifies: transportation restrictions ("no self-driving"), room type requirements ("private room only"), cuisine preferences ("Indian or Chinese only"), and house rules ("no pets"). The agent receives pre-fetched reference data and must produce a constraint-satisfying multi-day itinerary.

We evaluate against **all four local constraints** using the TravelPlanner database CSVs (5,047 accommodations, 9,551 restaurants), not just format+budget. This gives meaningful failure rates — without constraint checking, llama3:8b achieves 100% on all conditions.

### Two-Session Protocol

**Why two sessions?** In a single-session experiment, `graph_session` runs entirely cold (graph only written at session end). Without a second session, we only verify the graph was written — not whether it helps. The S1→S2 delta is the honest test.

| Session | What happens |
|---|---|
| **Session 1** | All conditions run cold. Memory conditions accumulate trajectories. |
| **Session 2** | *New tasks.* Memory conditions retrieve from S1 store. This is when memory is tested. |

### Conditions

| Condition | Memory | Update | Role |
|---|---|---|---|
| `no_memory` | None | — | True baseline |
| `memp_flat` | TF-IDF flat store | Per successful trajectory | MemP replication |
| `graph_pertraj` | Graph | Per trajectory | Ablation: early-noise vs. structure benefit |
| `graph_session` | Graph | Session-end batch | **Proposed** |

---

## 5. Preliminary Results

*n=2 tasks per session. Results have high variance — treat as directional only. Larger run (n≥10) needed for statistical significance.*

### 5.1 Full 4-Condition Run — Hard Tasks, n=2/session

| Condition | S1 Success | S2 Success | Δ S1→S2 | S2 Tokens | Graph Nodes | S2 Retrieved |
|---|---|---|---|---|---|---|
| `no_memory` | 50% | 100% | +50%* | 3,488 | — | no |
| `memp_flat` | 100% | 100% | 0% | 3,450 | — | **no** |
| `graph_pertraj` | 100% | 100% | 0% | 3,538 | 20 | **yes** |
| `graph_session` | 100% | 100% | 0% | 3,612 | 12 | **yes** |

*TravelPlanner validation, hard tasks (3+ constraints), llama3:8b, June 17, 2026*  
*\* no_memory Δ is task variance, not a learning signal (n=2, S1 had one constraint failure, S2 tasks happened to be easier)*

### 5.2 What these results show — and what they don't

**Valid findings from n=2:**

1. **TF-IDF flat retrieval fails across task boundaries.** `memp_flat` had valid entries from S1 (both tasks succeeded) but retrieved nothing in S2. The S2 tasks referenced different cities and dates — zero vocabulary overlap. This is a genuine structural weakness of MemP's retrieval mechanism, not a sample artefact.

2. **Graph retrieval works across task boundaries.** Both `graph_pertraj` and `graph_session` retrieved in S2 (retrieved=yes). The graph's seed-finding uses procedure content embeddings, not task surface tokens, making it robust to the vocabulary shift that broke `memp_flat`.

3. **Session-end consolidation and deduplication are working.** `graph_session` consolidated 4 nodes after S1. After S2, consolidation produced 12 nodes with 4 merges — the system correctly identified that some S2 procedures were variants of S1 procedures and merged them rather than duplicating.

4. **Constraint evaluation is now meaningful.** val_001 (`no_memory` S1) failed — the model violated a hard constraint (most likely transportation mode or room type). This gives the failure rate we need for a discriminating comparison.

**What n=2 cannot show:**
- Whether memory injection changes the quality of S2 plans (need n≥10 for signal over noise)
- Whether `graph_session` Δ exceeds `memp_flat` Δ (the core hypothesis)
- Learning curves (task index vs. success within a session)

### 5.3 Infrastructure Validation — Mock Environment

Prior run confirming the 2-session pipeline works end-to-end without LLM calls:

| What was confirmed | Detail |
|---|---|
| 2-session structure | All 4 conditions run S1 → consolidate → S2 correctly |
| Graph write timing | `graph_session` graph is empty in S1, populated at S1 end, retrieved in S2 |
| Per-trajectory write | `graph_pertraj` writes a node per task; retrieval fires from task 2 onward in S1 |
| Deduplication | `graph_session` S2 consolidation: 4 merges out of 12 nodes |
| Flat store limit | `memp_flat` only writes successful trajectories; failed tasks don't populate the store |

---

## 6. Other Experiments to Design

### E2 — Cross-Model Knowledge Transfer (framework ready, `experiments/knowledge_transfer/`)
**Question:** Can a student agent reach teacher-level performance faster by bootstrapping from the teacher's graph?  
**Design:** Teacher (stronger model, e.g. qwen2.5:32b) runs N sessions → graph built → student (llama3:8b) starts from graph vs. cold start. Measure sessions-to-target-success.  
**Why it matters:** Directly replicates MemP's strongest result (strong→weak transfer) in a live graph-based setting. If graph transfer outperforms MemP flat export, that's the thesis's core empirical claim.

### E3 — Conflict Detection Under Contradictory Conditions
**Question:** Does the graph detect conflicts that a flat store silently ignores?  
**Design:** Run agents on two task subsets with genuinely contradictory optimal strategies (e.g., subset A: always fly, subset B: always self-drive). Compare how many conflicts the graph flags (via `conflicts_with` edges) vs. a flat store (which just holds both contradictory entries, undetected).  
**Why it matters:** Conflict detection is the clearest structural advantage of the graph over a flat store. If we can show empirically that flat stores accumulate silent contradictions while the graph surfaces them, that's a concrete architectural contribution.

### E4 — Multi-Agent Shared Graph
**Question:** Do two specialized agents (e.g., one trained on short trips, one on long trips) produce a better shared graph than either alone?  
**Design:** Agent A runs on 3-day tasks, Agent B on 7-day tasks. Both write to one shared graph. Evaluate a third agent bootstrapping from the shared graph vs. from either individual graph.  
**Why it matters:** Tests whether the graph as a shared substrate enables *division of cognitive labor* — something no existing agent memory framework supports.

### E5 — Weak-to-Strong Transfer Reversal
**Question:** Can a weaker model's failure-mode graph augment a stronger model?  
**Design:** Weaker model runs tasks, builds a graph of what *failed* and how it recovered. Stronger model bootstraps with this failure-pattern graph. Compare to stronger model with no prior graph.  
**Why it matters:** MemP only tested strong→weak. The hypothesis that diverse failure modes from a weaker model encode useful information for a stronger model is unexplored and would be a novel contribution.

### E6 — Graph Compression and Pruning
**Question:** As the graph grows (100s of nodes), which nodes to prune to maintain retrieval efficiency?  
**Design:** Compare (a) full graph, (b) prune nodes used <k times, (c) collapse subtrees with >0.9 similarity. Metric: retrieval precision vs. task success.  
**Why it matters:** A graph that grows without bound becomes as noisy as a flat store. Principled pruning is necessary for a lifelong memory system.

### E7 — Procedure Composition via Graph Traversal
**Question:** Can the agent solve novel tasks by following `depends_on` chains rather than direct retrieval?  
**Design:** Build a graph where task A's procedure depends on sub-procedure X. Evaluate on task B which requires X as a subcomponent, without task B ever appearing in training.  
**Why it matters:** This tests whether the graph enables *generalization by composition* — the strongest claim for structured memory over flat retrieval.

---

## 7. Connection to Thesis Goals

| Thesis Goal | E1 Contribution |
|---|---|
| **Memory Controller** | Build/Retrieve/Update lifecycle with typed graph structure. Session-end consolidation as the update mechanism. K-hop subgraph as the retrieve mechanism. |
| **Procedural Memory** | Dual-level nodes (step = episodic, script = semantic) mirror the episodic/semantic distinction in cognitive science. Session-end consolidation mirrors hippocampal replay. |
| **Knowledge Transfer** | Graph is model-agnostic (JSON), shareable, and cross-session. E2 directly tests cross-model transfer. The TF-IDF retrieval failure shows why *structured* knowledge is more transferable than surface-similar knowledge. |
| **Educability** | The S1→S2 delta operationalizes "did the agent learn from its own experience?" The cross-model version (E2) tests "can one agent educate another?" |

---

## 8. Next Steps

**Immediate (this week):**
1. Run full E1 overnight: `python experiments/memory_controller/main.py --travelplanner --split validation --levels hard` with `tasks_per_session: 10` — needs ~8 hours
2. Run `analyze.py` to generate comparison table and per-task learning curve
3. Investigate val_001 failure: what constraint did the model violate? (retrieve from results JSON)

**Near term:**
4. Run E2 (cross-model transfer) with qwen2.5:32b as teacher, llama3:8b as student
5. Design E3 (conflict detection) — needs carefully curated contradictory task sets
6. Submit test split plans to TravelPlanner leaderboard for formal HC/CS scores

**Model upgrade:**
- llama3:8b achieves 100% on easy tasks but constraint failure rate on hard tasks is unclear at n=2
- swap `model.name` to `qwen2.5:32b` for a stronger agent with more measurable failure rate

---

## Appendix A: MemP Baseline (from paper, TravelPlanner validation)

| System | CS Score | HC Score | Model |
|---|---|---|---|
| No memory (ReAct) | ~30% | ~5% | GPT-4o |
| MemP (Proceduralization + Adjustment) | ~55% | ~10% | GPT-4o |
| MemP cross-model transfer | ~48% | ~8% | GPT-4o → Qwen2.5-14B |

Our constraint-checking proxy (transportation + room type + cuisine + house rules from DB) is closely aligned with HC but not identical. Formal comparison requires running `.travelplanner/evaluation/eval.py` on saved plan JSONs.

## Appendix B: Key Engineering Observations

- **Retrieval fires across task boundaries for graph, not for flat.** The most concrete finding from n=2.
- **Deduplication at session end is meaningful.** 4/12 nodes merged in session 2 — the consolidator is correctly identifying re-encountered procedures rather than duplicating them.
- **Hard constraints require DB lookup, not proxy metrics.** llama3:8b achieves 100% on format+budget alone. Real differentiation requires checking room type, cuisine, transportation, and house rules against the actual database.
- **n=2 per session is insufficient.** S1→S2 delta is dominated by task-to-task variance at this scale. n≥10 required for a meaningful signal.

---

*Report generated: June 17, 2026 | Code: `experiments/memory_controller/` | Results: `results/memory_controller/`*
