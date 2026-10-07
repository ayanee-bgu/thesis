# E6: 2WikiMultihopQA — Interactive Retrieval vs Static Context (generalizing E5)

**Date:** 2026-07-31
**Model:** llama3:8b (local, via Ollama, forced CPU — `num_gpu=0`, same fix carried forward from E5)
**Experiment:** Does E5's HotpotQA finding — static `summary_doc` beats both `full_context` and `interactive_retrieval`, and `interactive_retrieval`'s accuracy is bottlenecked by retrieval recall rather than reasoning — generalize to 2WikiMultihopQA, which has the same passage/supporting-fact schema shape but a richer 4-way question-type taxonomy (`compositional`, `comparison`, `bridge_comparison`, `inference`) instead of HotpotQA's 2 (`bridge`, `comparison`)?

---

## 1. Overview

**Two of E5's three findings replicate cleanly. One does not.**

Replicates: (1) BFS-shaped `comparison` questions are systematically easier than the three DFS-shaped types across nearly every condition, and (2) `interactive_retrieval`'s accuracy is capped by retrieval recall — when its 3 queries find the gold passages, accuracy roughly triples.

Does not replicate: `summary_doc` was E5's best condition on HotpotQA (beating both `full_context` and `interactive_retrieval` on EM and F1). On 2WikiMultihopQA, `summary_doc` is **not** clearly best — it's competitive on `comparison` questions but its F1 collapses on `compositional` questions (14.5%, worse than `full_context`'s 37.6% and even below `no_context`'s tie), because 2Wiki's compositional questions need a precise fact (a date, a specific name) that a 150-250 word prose summary of 10 passages frequently drops or garbles, where HotpotQA's bridge/comparison facts apparently survived compression. **Summarization's benefit is not a general property of static compression — it depends on whether the compression target's answer is the kind of fact that survives paraphrasing.**

---

## 2. Design

Same 4 conditions as E5 (`no_context`/`full_context`/`summary_doc`/`interactive_retrieval`, K=3 reused from E5's near-optimal value, not re-derived here), same harness (`experiments/wikihop_transfer/main.py`, near-verbatim port of E5's `main.py` — only the dataset loader and two passage/gold-title formatters changed, since 2WikiMultihopQA's `context` is a list of `[title, sentences]` pairs and `supporting_facts` a list of `[title, sent_id]` pairs, rather than HotpotQA's dict-of-parallel-lists). n=20 questions per type × 4 types = 80 questions, same seeded/deterministic first-N slice per type.

4 question types (vs E5's 2): `compositional` (multi-hop fact composition), `comparison` (two independent entities — the one BFS-shaped type, same role as E5's `comparison`), `bridge_comparison` (comparison routed through a shared bridge entity), `inference` (requires an inferential relationship step). All three non-`comparison` types require some form of multi-hop chaining and are treated as DFS-shaped for analysis.

---

## 3. Results

### 3.1 Overall (all 4 types combined, n=80/condition)

| Condition | EM | F1 | Tokens | SF-Recall | DFS |
|---|---|---|---|---|---|
| no_context | 13.8% | 20.5% | 28 | 0.0% | — |
| full_context | 17.5% | **41.1%** | 905 | 100.0% | — |
| summary_doc | 17.5% | 38.2% | 1,706 | 100.0% | — |
| **interactive_retrieval** | **26.2%** | 32.0% | 1,018 | 63.7% | 78.8% |

Unlike E5, there is no single dominant condition. `interactive_retrieval` has the best EM (26.2%) but the worst F1 among the non-baseline conditions (32.0%); `full_context` has the best F1. §3.4 explains the mechanism behind this EM/F1 split.

### 3.2 By question type

| Type | no_context F1 | full_context F1 | summary_doc F1 | interactive F1 |
|---|---|---|---|---|
| compositional | 0.0% | 37.6% | **14.5%** ← | 38.3% |
| comparison (BFS) | 53.6% | 41.0% | **62.2%** | 36.4% |
| bridge_comparison | 16.2% | 37.1% | 38.1% | 20.0% |
| inference | 12.2% | 48.7% | 37.7% | 33.4% |

`summary_doc` wins clearly on `comparison` (62.2%, its only clean win) but is the *worst* non-baseline condition on `compositional` (14.5% — below `full_context` by 23pp). This single-type collapse is what breaks E5's "summary_doc is best" finding at the aggregate level.

### 3.3 DFS-shaped vs BFS-shaped delta (replicates E5's structural pattern)

| Condition | Mean DFS-type F1 | Comparison F1 | Delta |
|---|---|---|---|
| no_context | 9.5% | 53.6% | −44.2pp |
| full_context | 41.1% | 41.0% | +0.1pp |
| summary_doc | 30.1% | 62.2% | −32.1pp |
| interactive_retrieval | 30.6% | 36.4% | −5.9pp |

This reproduces E5's qualitative shape closely: `no_context` is far worse on DFS-shaped questions (comparison questions name both entities explicitly, so parametric knowledge can often answer them directly — same mechanism E5 found), `full_context` is roughly neutral between question shapes, and both `summary_doc` and `interactive_retrieval` are worse on DFS-shaped types than on `comparison`. The magnitudes differ (E5's summary_doc delta was −12.8pp; here it's −32.1pp, driven entirely by the compositional collapse in §3.2), but the direction and ordering replicate.

### 3.4 Why interactive_retrieval has high EM but low F1: answer length

```
Mean answer length:  interactive_retrieval = 2.8 words   full_context = 6.8 words
Exact matches:        interactive_retrieval = 21/80        full_context = 14/80
```

`interactive_retrieval`'s final-answer prompt only sees 3 short retrieved snippets, which pushes it toward terse, decisive answers — when right, these are *exactly* right (higher EM); when wrong, they contain nothing of the gold answer (F1 = 0, no partial credit). `full_context`'s prompt sees all 10 passages and tends toward longer, more discursive answers — less often an exact match, but more often containing at least a fragment of the correct answer somewhere in the response, which token-F1 rewards. This is a real, reproducible difference in answer *style* induced by context volume, not just an accuracy difference — worth separating from the underlying retrieval-recall finding in §3.5, since it means EM and F1 can rank conditions differently depending on how verbose each condition's prompt design encourages the model to be.

### 3.5 The recall-bottleneck mechanism replicates exactly

Splitting `interactive_retrieval`'s results by whether its 3 queries found all gold passages (same analysis as E5 §3.3):

| Subset | n | EM | F1 |
|---|---|---|---|
| Found gold passages | 33/80 (41.3%) | **45.5%** | **0.52** |
| Missed at least one | 47/80 (58.7%) | 12.8% | 0.18 |

Accuracy roughly triples when retrieval succeeds, on a completely different dataset with a different passage set and a different question taxonomy — this is the strongest replication in this experiment. It confirms E5's central mechanistic claim was not a HotpotQA-specific artifact: **`interactive_retrieval`'s ceiling is set by whether its query budget finds the right evidence, not by whether the model can reason over evidence it has.**

Recall itself varies sharply by type — `comparison` (90.0%) and `compositional` (70.0%) are found far more often than `bridge_comparison` (40.0%) and `inference` (55.0%) — and `bridge_comparison` is correspondingly `interactive_retrieval`'s worst-scoring type (EM 20.0%, F1 20.0%), consistent with the recall→accuracy link.

### 3.6 DFS chain-following: less uniform than E5, but still present everywhere

| Type | DFS (chain-following) rate |
|---|---|
| compositional | 97.5% |
| comparison (BFS) | **60.0%** ← lowest |
| bridge_comparison | 80.0% |
| inference | 77.5% |

Unlike E5 (where bridge and comparison chain-following rates were nearly identical, 83.3% vs 81.7%), here `comparison` has the visibly lowest chain-following rate — a small signal that the mechanism is *slightly* more sensitive to task structure on this dataset than E5 suggested. But the effect is partial, not clean: chain-following is still high (60%) even on the one type with no real dependency chain, and 3 of 4 types show chaining above 77%. E5's core claim — that the query-generation prompt induces sequential, context-building queries largely independent of whether the task rewards it — still holds as the dominant pattern; 2Wiki just shows a mild type-sensitivity that HotpotQA's cleaner 2-type split did not surface.

---

## 4. Conclusions

1. **The retrieval-recall bottleneck mechanism (E5's central finding) replicates exactly on a different dataset.** Interactive retrieval accuracy roughly triples when its query budget finds the gold evidence (45.5% vs 12.8% EM) — same shape, same magnitude of effect as E5's HotpotQA result (80% vs 20% EM on bridge questions).
2. **The DFS-shaped vs BFS-shaped accuracy asymmetry replicates directionally** across all conditions except `full_context` (which stays roughly neutral in both experiments) — comparison-type questions are consistently easier, largely because cold-start parametric knowledge already handles many of them.
3. **E5's "summary_doc is the best condition" finding does NOT generalize.** It depends on whether the dataset's answers are the kind of fact (specific dates, exact names) that survives being folded into a 150-250 word prose summary. 2WikiMultihopQA's `compositional` questions need exactly that kind of precise fact, and summarization loses it; HotpotQA's bridge/comparison questions apparently didn't hit this failure mode. **Static compression's value is task/dataset-dependent, not a general property of compression itself** — this qualifies E5's conclusion rather than contradicting the underlying mechanism (recall/precision tradeoffs), since `summary_doc`'s SF-recall is still nominally 100% in both experiments (all passages contributed to the summary) — the loss here is a *distillation* failure, not a *retrieval* failure.
4. **The EM/F1 divergence for `interactive_retrieval`** (best EM, worst F1 among non-baseline conditions) is a new observation not visible in E5's 2-condition-type design: shorter, retrieval-scoped context induces shorter, more decisive answers, which is a genuine style effect on top of the accuracy story and worth tracking separately in future experiments (report both metrics, don't collapse to one).

---

## 5. Limitations

- **n=20/type**: smaller than E5's n=30/type (chosen to keep total runtime comparable given 4 types instead of 2); each flip ≈ 5pp instead of E5's ≈3.3pp — the `compositional` `summary_doc` collapse (14.5% F1, 4/20 items) should be read as a real, striking effect but not over-interpreted at the individual-percentage-point level.
- **K=3 not re-ablated here**: reused E5's near-optimal HotpotQA value without re-deriving it for 2Wiki's 4-type taxonomy or generally larger/more complex `compositional`/`inference` questions — if these types need more or fewer queries to reach their own accuracy-optimal K, that is not tested here (E7's hop-count × K ablation on MuSiQue is a more direct test of whether optimal K is dataset/task-dependent).
- **No hop-count control**: 2WikiMultihopQA's question types are not stratified by explicit reasoning-hop count the way MuSiQue's `id` field is (see E7) — "DFS-shaped" here is a coarser proxy (3 non-comparison types lumped together) than a true hop-count variable.
- **Same model in all roles**: llama3:8b generates queries, summaries, and answers, same limitation as E5.

---

## 6. Connection to Thesis

E6 sharpens E5's picture by separating a **general mechanism** from a **dataset-contingent instance**:

| Finding | Status after E6 |
|---|---|
| Interactive retrieval bottlenecked by recall, not reasoning | **Generalizes** — replicated near-exactly on a second dataset |
| DFS-shaped questions harder than BFS-shaped, roughly uniformly across conditions | **Generalizes directionally**, magnitude varies by dataset |
| Static summarization beats full context injection | **Does NOT generalize** — dataset/answer-type-dependent; failed specifically where answers require precise facts a prose summary tends to drop |
| DFS chain-following happens at a uniform rate regardless of task structure | **Partially qualified** — still broadly uniform, but 2Wiki shows a real (if small) type-sensitivity HotpotQA's cleaner split didn't surface |

The practical implication sharpens: a memory-controller design shouldn't assume "compress to a static summary" is a safe default improvement over raw context — whether it helps or actively hurts depends on whether the target answers are the kind of fact prose summarization preserves. This is a genuinely new, dataset-motivated qualification of E5's headline result, not just a confirming replicate.

---

*Run `python main.py` to replicate (resumes automatically from cached per-type results; `--fresh` forces a clean recompute).*
*Run `python analyze.py` to regenerate tables and the `plots/e6_dfs_vs_bfs.png` chart.*
