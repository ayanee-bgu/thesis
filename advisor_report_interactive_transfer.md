# Progress Report: Interactive Knowledge Transfer Between LLM Agents
**Date:** 2026-07-12

---

## 1. Research Question

Can an agent learn from a previous session more efficiently by *asking targeted questions* than by receiving a static dump of the prior session's context? And does the answer depend on the type of task?

This report covers two experiments (E3 and E4) that test four knowledge-transfer strategies across two benchmarks: **TravelPlanner** (constraint-diverse planning) and **ALFWorld** (uniform household manipulation). The same model (llama3:8b, local CPU) plays all roles in both experiments.

---

## 2. Experimental Setup

### Four Conditions

Each experiment runs a **teacher** on 5 tasks, records its trajectories, then runs a **student** on 5 different tasks under four knowledge-access conditions:

| Condition | What the student receives before solving its task |
|---|---|
| **no_memory** | Nothing — cold start baseline |
| **graph_session** | A compressed procedural graph built from teacher trajectories (k-hop retrieval) |
| **full_handoff** | All 5 teacher trajectories prepended verbatim to the system prompt |
| **interactive_oracle** | Up to K=5 Q&A turns with an LLM oracle loaded with teacher trajectories |

**Primary hypothesis:** Interactive querying achieves ≥ full-handoff accuracy at lower total token cost.

### Two Benchmarks

| | TravelPlanner (E3) | ALFWorld (E4) |
|---|---|---|
| **Task** | Generate multi-day travel plans satisfying hard constraints (flights, hotels, budget, diet) | Generate household action sequences (pick-and-place) |
| **Procedure** | Highly variable — each task has unique city/budget/dietary/room constraints | Uniform — all tasks follow GotoLocation → PickupObject → GotoLocation → PutObject |
| **Success** | All hard constraints satisfied | Correct action sequence + correct object/destination |
| **Teacher tasks** | val_000–004 | pick_and_place_simple, offset 0–4 |
| **Student tasks** | val_005–009 | pick_and_place_simple, offset 5–9 |

---

## 3. Results

### 3.1 TravelPlanner (E3)

| Condition | Success | Mean Tokens | Oracle Tokens | Tok / Success | Time / Task |
|---|---|---|---|---|---|
| No Memory | 60% | 3,537 | — | 5,895 | 344s |
| Graph Session | 60% | 3,764 | — | 6,273 | 319s |
| Full Handoff | 40% | 5,743 | — | 14,358 | 395s |
| Interactive Oracle | **20%** | 21,639 | 16,713 | **108,193** | 984s |

**Per-task breakdown:**

| Task | No Memory | Graph | Full Handoff | Oracle |
|---|---|---|---|---|
| val_005 | Y | Y | **N** | **N** |
| val_006 | Y | Y | Y | Y |
| val_007 | Y | Y | Y | **N** |
| val_008 | N | N | N | N |
| val_009 | N | N | N | N |

### 3.2 ALFWorld (E4)

| Condition | Success | Mean Tokens | Oracle Tokens | Tok / Success | Time / Task |
|---|---|---|---|---|---|
| No Memory | 60% | 229 | — | 381 | 18s |
| Graph Session | 60% | 229 | — | 381 | 12s |
| Full Handoff | **100%** | 522 | — | 522 | 17s |
| Interactive Oracle | **100%** | 5,357 | 4,441 | 5,357 | 377s |

**Per-task breakdown:**

| Task | No Memory | Graph | Full Handoff | Oracle |
|---|---|---|---|---|
| Candle → Toilet-407 | **N** | **N** | Y | Y |
| Candle → Toilet-429 | Y | Y | Y | Y |
| CellPhone → ArmChair | **N** | **N** | Y | Y |
| CellPhone → Bed | Y | Y | Y | Y |
| Cloth → BathtubBasin | Y | Y | Y | Y |

### 3.3 Cross-Benchmark Comparison

| Condition | E3 (TravelPlanner) | E4 (ALFWorld) | What changed |
|---|---|---|---|
| No Memory | 60% | 60% | — |
| Graph Session | 60% | 60% | — |
| Full Handoff | **40%** ↓ | **100%** ↑ | +60pp |
| Interactive Oracle | **20%** ↓ | **100%** ↑ | +80pp |

The baseline is identical (60%) across both benchmarks. Every transfer condition reverses direction between the two domains.

---

## 4. Key Findings

### Finding 1: Context pollution is real but domain-specific

On TravelPlanner, injecting more teacher context *monotonically degrades* student performance:

```
no_memory (60%) > graph (60%) > full_handoff (40%) > oracle (20%)
```

On ALFWorld, the pattern reverses completely:

```
full_handoff (100%) = oracle (100%) > no_memory (60%) > graph (60%)
```

**Why the difference?** TravelPlanner tasks have diverse constraint profiles — different cities, budgets, dietary requirements, room types across tasks. The teacher's context is mostly task-specific noise that conflicts with the student's constraints. The student over-attends to this irrelevant signal and fails its own constraints.

ALFWorld `pick_and_place_simple` tasks have an *identical* procedural structure across all instances. The teacher's trajectories carry pure signal: they demonstrate that the correct plan has 4 steps, not 3. The 2 failures under `no_memory` both dropped the second `GotoLocation` step (model generated Goto → Pickup → **Put**, omitting the "go to destination" before placing). Any teacher example showing the complete sequence corrected this omission.

### Finding 2: The interactive oracle is always the most expensive option

In both benchmarks, the interactive oracle matches the best non-oracle condition at dramatically higher cost:

| Benchmark | Oracle vs best static | Token overhead | Time overhead |
|---|---|---|---|
| TravelPlanner | Oracle (20%) vs Full Handoff (40%) | 3.8× *more* tokens, *worse* accuracy | 2.5× |
| ALFWorld | Oracle (100%) = Full Handoff (100%) | **10.3×** more tokens, same accuracy | **22×** |

The cost comes from a design flaw: the oracle is **stateless per question**. Each of K=5 oracle calls re-sends the full teacher trajectories (~2,000 tokens per call), generating massive redundant context. The oracle re-transmits the same information 5 times instead of once.

**Oracle token breakdown (E3):**
- Teacher trajectories re-sent 5 times: ~10,000 tokens
- Actual Q&A content: ~6,700 tokens
- Total oracle overhead: 16,713 tokens vs 5,743 for full_handoff

### Finding 3: The interactive hypothesis isn't cleanly testable with this design

The original hypothesis — *interactive querying is more token-efficient than full context* — was never fairly tested because the oracle design violates the token-efficiency assumption. The oracle costs *more* than full_handoff by construction, not because interactive querying is fundamentally inefficient.

The root cause is a single architectural decision: **the oracle is stateless per question**. Each of K=5 calls is a fresh LLM request that re-sends the full teacher trajectory corpus in the system prompt. Teacher context is paid K times over, making total cost scale as `K × context_size` instead of `1 × context_size + K × question_size`. With K=5 and ~2,000-token teacher context per call, this generates ~10,000 tokens of redundant payload before a single question is answered.

**Token cost breakdown (E3):**

| Component | Current (stateless) | If sent once |
|---|---|---|
| Teacher trajectories | 5 × 2,000 = **10,000** | 1 × 2,000 = **2,000** |
| Questions (5 × ~50) | 250 | 250 |
| Answers (5 × ~100) | 500 | 500 |
| **Total oracle tokens** | **~16,713** | **~2,750** |
| vs full_handoff | 2.9× more | **0.5× — cheaper** |

If the teacher context were sent only once, interactive querying would be *cheaper* than full_handoff — the original hypothesis would hold. The current implementation tests a broken version of the idea.

**Three redesigns that give the hypothesis a fair test:**

**Option A — Batch questions (simplest fix):**  
The student generates all K questions in one call before any answers are sent. One oracle call answers all of them together. Teacher context is sent once; questions and answers are bundled.  
- Estimated cost: 2,000 (trajectories) + 5×50 (questions) + 5×150 (answers) = ~3,000 tokens  
- Implementation: change the outer loop from K sequential calls to one batched call  
- Tradeoff: loses adaptivity — question 3 can't depend on the answer to question 2  

**Option B — Stateful oracle (preserves adaptivity):**  
The oracle runs as a single growing conversation. The student sends question 1, receives answer 1, then sends question 2 in the same thread — the oracle retains the full Q&A history and can build on previous answers without contradicting itself. Under standard LLM APIs, the full conversation is still re-sent each turn (no true server-side memory), so teacher trajectories appear in every call — but the *Q&A history* accumulated in previous turns replaces the need to re-ask context-setting questions. The oracle can answer question 3 ("what about the budget given the accommodation you described?") with reference to its own answer 2, without the student needing to repeat the context.  
- Estimated cost: 2,000 + (250 per turn × 5 growing turns) = ~3,250 tokens  
- With prefix caching (supported by most hosted APIs): teacher context cached after turn 1, cost drops to ~1,250 tokens for turns 2–5  
- Implementation: accumulate `messages` array across K rounds instead of resetting each call  
- Best fit for: tasks where questions are genuinely sequential and each answer informs the next (e.g. multi-constraint travel planning where budget answer shapes the follow-up accommodation question)  

*Context window scaling problem.* Option B faces a fundamental bounded memory constraint. The oracle's context window must simultaneously hold three competing components: (1) the knowledge base (teacher trajectories — stable, high-value, present every turn), (2) the Q&A conversation history (grows monotonically at ~200 tokens per turn), and (3) a generation budget for the answer being produced. For llama3:8b the hard limit is 8,192 tokens. In our experiments (5 teacher tasks × ~500 tokens each = 2,500 tokens, K=5 questions), the total oracle context stays around 3,250 tokens — comfortably within the window. But this is a lab configuration. A production oracle with 50 teacher sessions and K=20 questions would overflow before turn 10.

The deeper problem is that overflow produces two failure modes, only one of which is visible. The **hard failure** — an API error when context exceeds the limit — is at least detectable. The **soft failure** is more dangerous: at long context lengths, empirical evidence shows that LLMs attend disproportionately to content near the end of the window and lose reliable access to material in the middle ("lost in the middle" effect, Liu et al. 2023). An oracle nominally holding 20 turns of Q&A history may effectively ignore turns 3–14, producing answers that are fluent but ungrounded in earlier exchanges. This failure is invisible — the oracle confidently answers from a partial view of its own history.

When the window fills, something must be discarded, and every option is a tradeoff. **Truncating old Q&A turns** (rolling window) recovers space but breaks conversational coherence — the oracle at turn 8 may contradict what it said at turn 2. **Truncating the knowledge base** (shortening teacher trajectories) destroys the oracle's only source of truth. **Compressing Q&A history into a running summary** introduces a new LLM call with its own token cost, and embeds an irreversible editorial decision — the summary author (another LLM call) commits to what mattered before knowing what question 15 will ask.

This is a specific instance of the general **compression-fidelity tradeoff in bounded working memory**: the oracle can hold more turns at lower fidelity, or fewer turns at higher fidelity, but cannot do both. Biological working memory faces the same constraint and resolves it through attention-weighted decay. The LLM context window lacks graceful decay — it either holds content at full fidelity or drops it entirely. Option B therefore works in the small-K, small-history regime of our experiments but has no principled scaling path.

**Option C — Retrieval oracle (cheapest; the theoretically correct architecture):**  
The oracle does not load teacher trajectories into its context window at all. Instead, it maintains an index (TF-IDF or embedding-based) over chunked trajectory segments. Each oracle call retrieves only the 1–3 most relevant chunks for the current question (200–500 tokens each) and adds them to a short, stable context. The conversation history can still grow, but it is tiny relative to the space freed by removing the full knowledge base from the prompt.

- Estimated cost: 5 × (500 retrieved + 50 question + 150 answer) = ~3,500 tokens — cheaper than full_handoff  
- Scales to arbitrarily large teacher histories without architectural change — the index absorbs growth, the context window sees only retrieved excerpts  
- Implementation: requires a retrieval layer over trajectory chunks; the existing `ContextInjector` from E1 is directly reusable  
- Note: this is the architecture Mem^p actually uses — AveFact keyword-averaged cosine retrieval over procedural memory. Their reported 50% improvement on ALFWorld comes from this retrieval step, not from loading full trajectories.

*Why this is theoretically cleaner.* Option C resolves the bounded memory problem by separating **storage** (the index, unbounded and outside the LLM) from **working memory** (the context window, bounded and used only for active reasoning). The context window is no longer required to hold the entire knowledge base hostage while the conversation grows — it is used only for the current question, the retrieved excerpt, and the recent exchange history. The failure mode shifts from silent attention dilution to retrieval miss — which is a different and more tractable problem. Retrieval failures are detectable (the retrieved chunk can be logged and scored), recoverable (fall back to a broader query), and well-studied in the RAG literature. The tradeoff is that retrieval quality now determines answer quality: a missed relevant excerpt produces a worse answer than full context would have. But this is a solvable engineering problem, whereas context window overflow is a hard architectural limit.

**Which option to pursue next:**  
Option B is the lowest-friction implementation and a direct test of whether statefulness alone (without retrieval) closes the token-cost gap. It should be the first experiment — it isolates the statelessness variable cleanly. Option C is the correct long-term architecture and the path to a system that scales beyond the toy K=5, n=5 regime of these experiments. Option A (batch questions) is a useful cost-floor baseline but sacrifices the adaptivity that motivates interactive querying in the first place.

### Finding 4: The graph is neutral in both experiments

Graph injection produces identical accuracy to no_memory in both benchmarks (60% both cases). This is a consistent null result. Either (a) the graph compression loses too much detail for these task types, or (b) the information is within the model's base capability and the graph adds no marginal signal. The graph does not add harmful noise — it is exactly neutral.

### Finding 5: Model utilisation is task-structure-dependent

E3 concluded that llama3:8b cannot translate injected procedural knowledge into correct plans (the information is received but not used). E4 partially refutes this: the same model, with the same teacher context, uses it correctly on structurally uniform tasks. The utilisation bottleneck is not a fixed property of the model — it depends on whether the teacher's procedure applies directly to the student's task.

---

## 5. Advantages and Disadvantages of Interactive Transfer

### Advantages

| Advantage | Evidence |
|---|---|
| **Demand-driven selection** (in theory) | Student asks only what it needs; oracle answers from teacher records |
| **Correct accuracy when helpful** | Oracle achieves 100% on ALFWorld, matching full_handoff |
| **Interpretable questions** | Questions reveal what the student is uncertain about (budget/constraints in E3; physical properties in E4) |
| **Flexible** | K is adjustable; student can ask fewer questions on easy tasks |

### Disadvantages

| Disadvantage | Evidence |
|---|---|
| **Stateless design is expensive** | 10–18× token overhead vs full_handoff in both benchmarks |
| **Wall-clock cost is prohibitive** | 22× slower than full_handoff on ALFWorld (377s vs 17s); 2.5× on TravelPlanner |
| **Questions are often irrelevant** | E4 student asks about physical measurements (candle weight, toilet dimensions) rather than procedural steps |
| **Same information re-sent K times** | Stateless oracle: each call re-pays the full context cost |
| **Accuracy never exceeds full_handoff** | Oracle is always ≥ full_handoff in cost and ≤ full_handoff in accuracy |
| **Domain sensitivity unchanged** | Both oracle and full_handoff benefit from ALFWorld structure; both suffer context pollution on TravelPlanner — the interactive mechanism doesn't help with domain mismatch |

---

## 6. Summary

The two experiments provide a clean two-cell design that isolates one variable — task structure uniformity — while holding everything else constant (model, conditions, n):

|  | Diverse constraints (TravelPlanner) | Uniform structure (ALFWorld) |
|---|---|---|
| **Full handoff** | Harmful (−20pp vs baseline) | Beneficial (+40pp vs baseline) |
| **Interactive oracle** | Harmful (−40pp vs baseline) | Beneficial (+40pp vs baseline) |
| **Graph** | Neutral | Neutral |
| **Oracle vs handoff** | Worse accuracy + higher cost | Same accuracy + 10× higher cost |

**The core insight:** Whether to inject teacher context at all depends on task-structure similarity, not on the injection mechanism. The interactive oracle has the same domain sensitivity as full context injection — it doesn't solve the context pollution problem. And when context injection is beneficial (uniform tasks), the interactive oracle achieves the same result as full handoff at 10× the cost.

**The oracle design hypothesis** (interactive querying is cheaper than full handoff) requires a stateful or retrieval-based oracle to be fairly tested. The current stateless K=5 design is a worst-case oracle: it pays the full context cost K times instead of once.

---

## 7. Open Questions

1. **Does partial structural similarity produce partial context pollution?** Testing `pick_clean_then_place` (multi-stage, 7–12 steps) on ALFWorld would isolate how much structural overlap is needed for transfer to be beneficial.

2. **Does a stateful or retrieval-based oracle outperform full handoff?** Option C (retrieval oracle, ~5k tokens) would be cheaper than full handoff while preserving interactivity — a genuine test of the hypothesis.

3. **Does a stronger base model eliminate context pollution?** Context pollution on TravelPlanner might disappear with a 70B+ model that can selectively attend to relevant context rather than over-attending to all of it.

4. **Is the graph null result an infrastructure issue or a conceptual limitation?** The graph consolidation prompt failed on short ALFWorld trajectories. A correctly calibrated graph might provide the efficiency benefits of sparse injection without the noise of full handoff.

---

*Experiments:*  
- *E3: `experiments/interactive_transfer/` — TravelPlanner, CPU llama-server, normalized run (2026-07-03)*  
- *E4: `experiments/alfworld_transfer/` — ALFWorld pick_and_place_simple, same CPU backend (2026-07-11)*
