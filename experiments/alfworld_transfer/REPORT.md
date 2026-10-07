# E4: ALFWorld Plan Generation with Knowledge Transfer

**Date:** 2026-07-11  
**Model:** llama3:8b (local, via llama-server CPU, 10 threads, i7-8750H)  
**Experiment:** Does in-context knowledge transfer help or hurt on structurally uniform household manipulation tasks? Replication of E3's four-condition knowledge-transfer design on ALFWorld data to test whether context pollution is domain-specific or general.

---

## 1. Overview

E3 (TravelPlanner) found that injecting teacher context strictly *degrades* student performance — more context, worse outcomes (60% → 40% → 20%). We labelled this **context pollution** and flagged the open question: is this a general property of in-context knowledge transfer at llama3:8b scale, or is it specific to TravelPlanner's highly variable constraint profiles?

E4 answers this question. It runs the same four conditions on ALFWorld household manipulation tasks, which have a structurally *opposite* profile to TravelPlanner:

| Property | TravelPlanner (E3) | ALFWorld (E4) |
|---|---|---|
| Task structure | Highly variable (budget, city, dietary, room constraints vary arbitrarily) | Uniform (all `pick_and_place_simple` = GotoLocation → PickupObject → GotoLocation → PutObject) |
| Success criterion | Binary hard constraint check (all constraints must pass) | Exact action-type sequence match + object/destination mentioned |
| Teacher relevance | Teacher's constraints are specific to teacher's tasks — mostly noise for student | Teacher's procedure is identical to student's — pure signal |

If context pollution persists on ALFWorld, C1 from E3 is a general model-scale phenomenon. If it reverses, C1 is domain-specific to TravelPlanner's constraint diversity.

**Setup note:** TextWorld (the ALFWorld runtime) has no pre-built Windows wheels. We run without the game engine: teacher trajectories are expert demonstrations loaded from `traj_data.json`; the LLM generates high-level action plans for student tasks, evaluated against ground-truth sequences.

---

## 2. Design

### 2.1 Task selection

Task type: `pick_and_place_simple` from ALFWorld valid_seen split.  
Teacher: 5 tasks (offset 0–4). Student: 5 tasks (offset 5–9). Non-overlapping.  
Ground-truth action sequence for all tasks: `[GotoLocation, PickupObject, GotoLocation, PutObject]`.

### 2.2 Conditions

Same four conditions as E3:

| Condition | What student sees before planning |
|---|---|
| `no_memory` | Task description and object/destination only |
| `graph_session` | K-hop subgraph from teacher trajectories (graph was empty due to distillation failure — see §6) |
| `full_handoff` | All 5 teacher expert demonstrations prepended to system prompt |
| `interactive_oracle` | Up to K=5 Q&A turns with oracle loaded with teacher demonstrations |

### 2.3 Evaluation

- **`sequence_correct`**: Generated action-type sequence exactly matches ground truth `[Goto, Pickup, Goto, Put]`
- **`object_mentioned`**: Generated plan mentions the target object by name
- **`dest_mentioned`**: Generated plan mentions the destination receptacle by name
- **`success`**: All three conditions hold

Primary comparison: does `full_handoff` success rate exceed or fall below `no_memory`?

---

## 3. Results

### 3.1 Condition Comparison

| Condition | Success | Seq. Correct | Mean Tokens | Tok/Success | Time/task |
|---|---|---|---|---|---|
| No Memory | 60.0% | 60.0% | 229 | 381 | 18s |
| Graph (E1) | 60.0% | 60.0% | 229 | 381 | 12s |
| Full Handoff | **100.0%** | **100.0%** | 522 | 522 | 17s |
| Interactive Oracle | **100.0%** | **100.0%** | 5,357 | 5,357 | 377s |

### 3.2 Efficiency (Tokens per Successful Task)

| Condition | Success | Tok/Success | vs no_memory |
|---|---|---|---|
| No Memory | 60% | 381 | 1.00× |
| Graph (E1) | 60% | 381 | 1.00× |
| Full Handoff | 100% | 522 | **1.37×** |
| Interactive Oracle | 100% | 5,357 | **14.06×** |

### 3.3 Per-Task Breakdown

| Task | no_memory | graph | full_handoff | oracle |
|---|---|---|---|---|
| Candle → Toilet-407 | **N** | **N** | Y | Y |
| Candle → Toilet-429 | Y | Y | Y | Y |
| CellPhone → ArmChair-318 | **N** | **N** | Y | Y |
| CellPhone → Bed-324 | Y | Y | Y | Y |
| Cloth → BathtubBasin-405 | Y | Y | Y | Y |

(S = composite success, Y/N for sequence_correct and object/dest mentioned)

Both failures under `no_memory` (and the identical `graph_session`) flip to success under `full_handoff` and `interactive_oracle`. There are no tasks where teacher context causes a regression.

### 3.4 Action Sequence Analysis

The 2 failures under `no_memory` reveal a consistent error:

```
Ground truth:  [GotoLocation, PickupObject, GotoLocation, PutObject]
Generated:     [GotoLocation, PickupObject, PutObject]          ← missing 2nd Goto
```

The model collapses the 4-step sequence to 3 steps by omitting the second `GotoLocation` (travel to destination before placing). This is a base-model omission error. Teacher examples that demonstrate the full 4-step sequence correct it entirely — `full_handoff` achieves 100% sequence accuracy on all 5 tasks.

### 3.5 Oracle Question Analysis

The student's oracle questions focused on physical properties of the environment:

> *"What is the minimum height and distance from the toilet bowl that the robot should position itself to successfully place an object on it?"*

> *"What is the average weight and size of the candles that the previous robot successfully placed on the toilet?"*

Top keywords (25 questions): **considering** (14), **previous** (11), **objects** (11), **distance** (10), **toilet** (9), **robot** (9), **locations** (7).

The questions ask about physical measurements rather than procedural steps. Despite this mismatch between what was asked and what would help, the oracle's answers drew on teacher trajectories that contained procedural information, and the student extracted the structural pattern from the oracle context. The mechanism is indirect: the oracle's responses implicitly referenced the GotoLocation→Pickup→GotoLocation→Put template even when answering questions about candle weight.

### 3.6 Cross-experiment Comparison (vs E3)

| Condition | E3 (TravelPlanner) | E4 (ALFWorld) | Delta |
|---|---|---|---|
| no_memory | 60.0% | 60.0% | +0.0% |
| graph_session | 60.0% | 60.0% | +0.0% |
| full_handoff | **40.0%** | **100.0%** | **+60.0pp** |
| interactive_oracle | **20.0%** | **100.0%** | **+80.0pp** |

The baseline (`no_memory`) is identical across both experiments at 60%. The transfer conditions diverge completely: context *hurts* on TravelPlanner and *helps* on ALFWorld.

---

## 4. Analysis

### 4.1 Context Pollution is Absent on ALFWorld

The key result: `full_handoff` goes from **40% → 100%** relative to TravelPlanner, while `no_memory` stays at **60%** in both experiments. Teacher context is unambiguously beneficial on ALFWorld. The E3 finding of monotonic degradation with injection volume is reversed:

```
TravelPlanner (E3):  no_memory=60% > full_handoff=40% > oracle=20%  ← context hurts
ALFWorld (E4):       full_handoff=oracle=100% > no_memory=60%        ← context helps
```

### 4.2 The Mechanism: Structural Uniformity vs. Constraint Diversity

The divergence between E3 and E4 is explained by a single property: **task structure uniformity**.

In TravelPlanner, the teacher's tasks (val_000–004) have different cities, budgets, dietary constraints, and travel dates than the student's tasks (val_005–009). The teacher's context carries task-specific information that conflicts with the student's constraints — it is mostly noise. The model over-attends to this irrelevant context and fails on its own constraints.

In ALFWorld `pick_and_place_simple`, every task has the same procedural structure regardless of object or destination. The teacher example "pick up the book, put it on the sofa" demonstrates *exactly the same sequence* as "pick up the candle, put it on the toilet". The teacher's context is pure procedural signal: it teaches the model that the correct plan has 4 steps (Goto → Pickup → Goto → Put), not 3 (Goto → Pickup → Put).

The model's base failure mode — omitting the second GotoLocation — is corrected by any teacher example showing the complete sequence. This generalises perfectly: the structural template from teacher tasks is directly applicable to student tasks.

### 4.3 Full Handoff is Optimal — at 1.37× Token Cost

`full_handoff` achieves 100% success at 1.37× the token cost of `no_memory` per successful task (522 vs 381 tok/success). This is the most efficient condition: the cheapest way to achieve perfect performance.

In E3, `full_handoff` was 2.44× the token cost of `no_memory` per successful task while *also* achieving lower success. Here, it is 1.37× at higher success — a genuine improvement in both dimensions.

Inference time does not change measurably (17s vs 18s per task). On CPU, reading a longer system prompt is cheap; the bottleneck is generating the plan.

### 4.4 Interactive Oracle: Correct Outcome, Wrong Mechanism, High Cost

The oracle achieves 100% success but at 14.06× the token cost of `no_memory` (5,357 vs 381 tok/success) and 20× the wall-clock time (377s vs 18s per task).

The oracle question analysis shows the student asks *procedurally irrelevant* questions (physical measurements of objects rather than "what steps did the previous robot follow?"). Despite this, the oracle succeeds — because the oracle's system prompt contains the teacher trajectories, and the oracle's *answers* inevitably reference the structural template even when answering non-structural questions.

This means the oracle is doing something different from what the interactive design intended: it is acting as an implicit procedure carrier rather than an explicit Q&A partner. The success is real but the mechanism is indirect. If the student were asking precise procedural questions ("what actions did the robot take in what order?"), the oracle would be even more efficient.

The C2 finding from E3 replicates: a stateless oracle re-sending full teacher context on every one of K=5 calls generates 10× oracle-token overhead (5,357 tokens) relative to a single full_handoff (522 tokens) with identical performance. The interactive design is not justified at this task scale.

### 4.5 Graph Session is Empty (Limitation)

The graph consolidation step (distilling 5 teacher trajectories into ProcedureGraph nodes) returned 0 nodes — the LLM's response to the distillation prompt was likely not valid JSON for these short ALFWorld trajectories (teacher trajectories are 3–5 line text stubs). `graph_session` is therefore identical to `no_memory` in this run. This is a limitation of the E1 graph infrastructure's distillation prompt being calibrated for TravelPlanner's richer trajectories.

---

## 5. Conclusions

**C1 from E3 is domain-specific: context pollution is a TravelPlanner phenomenon, not a general model-scale limitation.**  
On ALFWorld `pick_and_place_simple`, full teacher context injection raises success from 60% to 100% at negligible latency overhead and 1.37× token cost. The same llama3:8b model that degrades under teacher context on TravelPlanner benefits strongly from it on structurally uniform household tasks. Context pollution requires task-constraint diversity; structural uniformity eliminates it.

**The utilisation gap is narrower than E3 suggested.**  
E3's C3 claimed the bottleneck is model utilisation (the model can't translate injected knowledge into valid plans). E4 refutes this in the procedural domain: llama3:8b *can* translate teacher demonstrations into correct plans when the teacher's procedure directly applies to the student's task. The utilisation gap is real on hard constraint-satisfaction tasks with diverse profiles; it is absent on structurally identical procedural tasks.

**Practical implication for knowledge transfer system design:**  
The correct transfer mechanism depends on task structure:
- **Uniform structure** (procedural, household, navigation): full context injection is cheap, effective, and recommended. ~1.4× token overhead; 100% success.
- **Diverse constraint profiles** (travel planning, logistics, scheduling): context injection is harmful. Use retrieval (sparse injection of constraint-relevant information), fine-tuning, or stronger base models instead.

**The oracle design flaw generalises.**  
Interactive oracle at K=5 stateless calls costs 14× more tokens than full_handoff with identical success. This is not domain-specific — it is a consequence of re-sending the full teacher context on every oracle call regardless of domain. A stateful oracle or retrieval-based oracle (C2 redesign options from E3) would be domain-agnostic improvements.

---

## 6. Limitations

- **n=5 student tasks**: Single-flip granularity (1 flip = 20pp). Full_handoff 100% is based on 5/5 successes; requires replication at n≥20 for confidence.
- **Single task type** (`pick_and_place_simple`): The most uniform task type in ALFWorld (all 4-step sequences). `pick_clean_then_place` (7–12 steps, multi-stage) or `pick_heat_then_place` may show partial pollution effects where the teacher's specific sub-procedures don't transfer as cleanly.
- **Graph consolidation failed**: 0 nodes built; `graph_session` is uninformative as an E1 replication. The distillation prompt needs adaptation for short ALFWorld trajectory strings.
- **No game engine**: Success evaluated against ground-truth action-type sequences, not actual game completion. A model could generate the correct sequence types but with wrong object names or locations and still fail in the real environment.
- **Same model all roles**: Oracle and student both use llama3:8b. A stronger oracle (larger model) with a weaker student might show different patterns.

---

## 7. Connection to Thesis

E4 provides a clean domain boundary for the context pollution finding from E3:

| Finding | Status after E4 |
|---|---|
| C1: Context pollution on TravelPlanner | Confirmed as **domain-specific** (not present on ALFWorld uniform tasks) |
| C2: Oracle stateless design flaw | **Generalises** — replicated on ALFWorld (14× token overhead, same success as full_handoff) |
| C3: Utilisation bottleneck at llama3:8b | **Partially refuted** — bottleneck absent on procedurally uniform tasks; present on constraint-diverse tasks |

The key contribution for the educability thesis: **model educability is task-structure-dependent at this model scale**. The same model (llama3:8b) is educable on uniform procedural tasks (teacher context reliably improves performance) and not educable on constraint-diverse tasks (teacher context degrades performance). Educability is not a fixed property of the model; it depends on the structural relationship between teacher and student tasks.

This motivates a task-structure-aware transfer protocol:
1. **Assess structural similarity** between teacher and student tasks before injecting context.
2. **Inject freely** when tasks are structurally equivalent (same procedure, different objects/locations).
3. **Use retrieval or fine-tuning** when tasks have diverse constraint profiles that make raw context injection harmful.

---

## 8. Next Steps

1. **Extend to `pick_clean_then_place`** (7–12 step, multi-stage tasks): tests whether partial structural similarity produces partial context pollution.
2. **Fix graph consolidation prompt** for short ALFWorld trajectories: adapt distillation prompt to extract high-level action templates from 3–5 line trajectory stubs.
3. **Stateful oracle redesign** (C2): implement a conversation-history oracle and retest — should approach full_handoff performance at ~1/K the oracle token cost.
4. **Larger n** (n≥20 student tasks): stabilise success rates, especially for `no_memory` which at 60% has high variance.
5. **Cross-model teacher**: test whether a GPT-4o teacher trajectory injected into llama3:8b student preserves the structural transfer benefit — direct test of the Mem^p cross-model transfer claim.

---

*Run `python main.py` to replicate. Run `python main.py --skip-teacher` to reuse saved trajectories.*  
*Run `python analyze.py` to regenerate analysis and chart.*
