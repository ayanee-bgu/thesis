# E7: MuSiQue — Hop-Count x K Ablation

**Date:** 2026-08-01
**Model:** llama3:8b (local, via Ollama, forced CPU — `num_gpu=0`, same fix carried forward from E5/E6)
**Experiment:** E5's K-ablation found K=3 near-optimal on HotpotQA and that raising K past 3 made things *worse* (context dilution outweighing recall gains). Was K=3 near-optimal because it suits HotpotQA's effectively-fixed 2-hop bridge structure, or does the accuracy-optimal K actually scale with how many hops a question requires? MuSiQue's `id` field directly encodes hop count (2/3/4-hop), turning this into a genuine hop-count × K ablation.

---

## 1. Overview

**Two findings, one confirming a mechanism from E5/E6, one new and unexpected:**

1. **The retrieval-recall bottleneck mechanism replicates a third time**, on a third dataset with a different passage count (20 vs 10) and a genuinely different task structure (explicit hop-count control): `interactive_retrieval` accuracy is 46.4% EM when its queries find all gold passages vs 16.8% when they don't — same shape as E5 (HotpotQA: 80% vs 20%) and E6 (2WikiMultihopQA: 45.5% vs 12.8%).

2. **The optimal K does appear to shift with hop count — but the more striking result is that `full_context` degrades sharply at MuSiQue's 20-passage scale**, badly enough that `interactive_retrieval` actually *beats* both static conditions in several hop×K cells — a reversal of every previous experiment (E5, E6), where full-context/summary conditions were always competitive with or better than interactive retrieval. This points to a context-pollution effect that scales with passage count, not just with information density.

---

## 2. Design

Same 4 mechanisms as E5/E6 (`no_context`/`full_context`/`summary_doc`/`interactive_retrieval`), new dataset: **MuSiQue** (`dgslibisey/MuSiQue`, validation split, `answerable=True` only). Each question has exactly 20 candidate passages (vs HotpotQA/2Wiki's 10) with a hop-count-equal number of gold "supporting" passages.

Hop buckets extracted directly from the `id` field prefix (`2hop__...`, `3hop1__...`, `4hop1__...`, etc.): **hop2, hop3, hop4**, n=15 questions/bucket, same seeded deterministic first-N slice.

- `experiments/musique_transfer/main.py` runs the 3 static conditions once per hop bucket (reference ceilings).
- `experiments/musique_transfer/k_ablation.py` runs `interactive_retrieval` at **K ∈ {3, 5, 7}** across all 3 hop buckets (9 hop×K cells total, 15 questions each = 135 interactive_retrieval runs).
- Both scripts are near-verbatim ports of E5's harness; only the dataset loader (`load_musique`, hop-bucket extraction) and passage/gold-title formatters changed (MuSiQue's `paragraphs` is a list of `{idx, title, paragraph_text, is_supporting}` dicts, vs HotpotQA's dict-of-parallel-lists), plus EM/F1 scoring was extended to check MuSiQue's `answer_aliases` field alongside the primary gold answer.
- `model.timeout` raised to 450s (from 300s) given 20-passage prompts roughly double HotpotQA's token count.

---

## 3. Results

### 3.1 Static baselines by hop bucket (n=15/bucket)

| Condition | hop2 F1 | hop3 F1 | hop4 F1 | Overall F1 |
|---|---|---|---|---|
| no_context | 11.2% | 0.0% | 8.0% | 6.4% |
| **full_context** | **19.4%** | 14.7% | **18.0%** | **17.4%** |
| summary_doc | 15.7% | 0.0% | 16.2% | 10.6% |

MuSiQue is substantially harder than HotpotQA/2Wiki for all conditions (F1 in the 0-20% range vs E5/E6's 30-60% range) — expected, since MuSiQue is explicitly designed to resist shortcut answering (more distractors, questions constructed so single-hop or partial-chain reasoning fails). `summary_doc` collapses to 0% F1 on `hop3` — a total failure, discussed in §3.4.

### 3.2 Hop-count x K ablation (interactive_retrieval, n=15/cell)

**EM:**

| Hop bucket | K=3 | K=5 | K=7 |
|---|---|---|---|
| hop2 | **26.7%** | 20.0% | 20.0% |
| hop3 | 40.0% | **53.3%** | 26.7% |
| hop4 | 6.7% | 6.7% | 6.7% |

**F1:**

| Hop bucket | K=3 | K=5 | K=7 |
|---|---|---|---|
| hop2 | **46.9%** | 35.4% | 35.4% |
| hop3 | 40.0% | **53.3%** | 26.7% |
| hop4 | 9.9% | 13.3% | 12.0% |

**Full-recall rate** (both/all gold passages found):

| Hop bucket | K=3 | K=5 | K=7 |
|---|---|---|---|
| hop2 | 13.3% | 40.0% | 46.7% |
| hop3 | 20.0% | 20.0% | 20.0% |
| hop4 | 0.0% | 6.7% | 20.0% |

**The optimal K shifts with hop count**: hop2 peaks at K=3 (replicating E5's HotpotQA finding almost exactly — HotpotQA's bridge/comparison questions are effectively 2-hop), but hop3 peaks at **K=5**, beating both K=3 and K=7 by a wide margin (53.3% vs 40.0% vs 26.7% EM). hop4 shows no usable peak — accuracy is flat and low (6.7% EM at every K), a floor effect discussed in §3.3.

Note hop2's pattern reproduces E5's core K-ablation finding exactly: full-recall rate rises monotonically with K (13.3%→40.0%→46.7%) while accuracy *falls* (26.7%→20.0%→20.0% EM) — more successful retrieval, worse accuracy, the same context-dilution signature from E5 §9.3.

### 3.3 Why hop4 shows no peak: recall never gets high enough to matter

hop4 needs 4 of 20 passages found within K≤7 queries — full-recall rate tops out at only 20% even at K=7 (vs hop2's 46.7%). The task is bottlenecked on recall so severely across the entire tested K range that no amount of query-budget variation within {3,5,7} produces a visible accuracy signal — EM stays pinned at 6.7% (functionally 1/15 correct) regardless of K. This isn't evidence that K doesn't matter for 4-hop questions; it's evidence that K=3-7 is nowhere near the recall regime where an optimum would become visible. A meaningfully larger K (or a non-lexical retrieval method — see Next Steps) would be needed to even test this hop bucket properly.

### 3.4 Unexpected: full_context degrades sharply at 20 passages, verbose hedging replaces confident answers

Comparing `full_context` and `interactive_retrieval` (K=5) on the same `hop3` questions:

```
gold='Denver'                pred='According to the reference passage, Trey Parker voices four ' f1=0.04   [full_context]
gold='Francisco Guterres'    pred='According to the reference passage "Indonesia-Timor Leste Co'  f1=0.10   [full_context]
gold='Francisco Guterres'    pred='Susilo Bambang Yudhoyono (Indonesian President)'               f1=0.00   [full_context — wrong entity: answered about Indonesia, not Timor-Leste]

gold='Francisco Guterres'    pred='Francisco Guterres'                                             f1=1.00   [interactive_retrieval K=5]
gold='Francisco Guterres'    pred='Francisco Guterres'                                             f1=1.00   [interactive_retrieval K=5]
gold='Francisco Guterres'    pred='Francisco Guterres'                                             f1=1.00   [interactive_retrieval K=5]
```

Mean answer length: `full_context` = **22.7 words**, `interactive_retrieval` = **2.2 words**. Given all 20 passages at once, the model frequently hedges ("According to the reference passage...") and — more importantly — picks the *wrong entity* among the 20 distractor-heavy passages (answering about Indonesia's president when the question needs Timor-Leste's). Given only its own 3-7 retrieved snippets, the model answers tersely and, when retrieval succeeds, correctly and confidently.

This is the same "more context, worse accuracy" mechanism as E3's original context-pollution finding and E5's K-ablation dilution effect — but here it appears in `full_context` itself, at 20 passages, where it didn't appear (or appeared much more mildly) at HotpotQA/2Wiki's 10 passages in E5/E6. **Context pollution severity appears to scale with the size of the context being dumped, not just with the presence of irrelevant content.** This directly explains why `interactive_retrieval` — despite being capped by its own recall bottleneck — can outscore `full_context` outright on MuSiQue (hop3 K=5: 53.3% vs 14.7% F1) in a way it never did on HotpotQA or 2Wiki: `full_context`'s disadvantage at 20 passages grew large enough to exceed `interactive_retrieval`'s recall-driven disadvantage.

`summary_doc`'s total collapse on `hop3` (0.0% F1, §3.1) is a related but distinct failure: summarizing 20 passages into 250-400 words apparently drops the specific facts hop3 questions need entirely (same distillation-loses-precision failure mode as E6's `compositional` collapse, worse here because there is more to compress and lose).

### 3.5 The recall-bottleneck mechanism, pooled across all 135 interactive_retrieval runs

| Subset | n | EM | F1 |
|---|---|---|---|
| Found all gold passages | 28/135 (20.7%) | **46.4%** | **0.56** |
| Missed at least one | 107/135 (79.3%) | 16.8% | 0.23 |

Consistent with E5 (80% vs 20% EM) and E6 (45.5% vs 12.8% EM) — a third independent confirmation that `interactive_retrieval`'s ceiling is set by whether its query budget finds the right evidence, holding across three datasets, two passage counts, and a fundamentally different question-generation methodology (MuSiQue's questions are algorithmically composed from single-hop sub-questions, unlike HotpotQA/2Wiki's crowd-sourced questions).

---

## 4. Conclusions

1. **The recall-bottleneck mechanism is now confirmed on three independent datasets** (HotpotQA, 2WikiMultihopQA, MuSiQue), with consistent magnitude (accuracy roughly 2.5-4x higher when retrieval succeeds vs when it doesn't). This is the most robust finding across E5-E7.
2. **The optimal K does shift with hop count, at least between 2-hop and 3-hop** (K=3 → K=5), directly refuting the possibility that E5's "K=3 is near-optimal, more hurts" finding was a fixed constant independent of task depth. It was, at least partly, an artifact of HotpotQA's effectively-2-hop structure. The 4-hop case remains untested in any informative sense (§3.3) — recall never rises far enough within K≤7 to reveal an optimum.
3. **A new, unplanned finding: `full_context`'s advantage over `interactive_retrieval` seen in E5/E6 reverses at MuSiQue's 20-passage scale.** Dumping more passages doesn't just add tokens — past some threshold it appears to actively degrade the model's ability to pick the right entity among distractors, badly enough that a recall-bottlenecked interactive mechanism can outscore it. This complicates any simple "static context is safer than interactive retrieval" heuristic drawn from E5/E6 alone — the right choice depends on how large the underlying context is, not just on task structure.
4. **`summary_doc`'s reliability continues to be dataset/task-dependent** (per E6 §4), now failing completely (0% F1) on one MuSiQue hop bucket — reinforcing that static compression is not a safe general-purpose default and needs task-aware validation before being assumed beneficial.

---

## 5. Limitations

- **n=15/cell is small**: single flip = 6.7pp EM. The hop3 "K=5 is optimal" result (53.3% vs 40.0% vs 26.7%) is a large, striking effect but rests on just 15 questions per K value — deserves replication at larger n before being treated as settled rather than a strong signal.
- **hop4 is uninformative by construction** (§3.3) — the K range tested (3-7) never reaches the recall regime where an optimum would be visible; this ablation cannot distinguish "4-hop has no K-dependent optimum" from "4-hop's optimum requires K > 7."
- **TF-IDF retrieval only**, same limitation as E5/E6 — the recall bottleneck driving all three experiments' central finding might be substantially alleviated by embedding-based retrieval, still untested (E5's Next Steps #2, now even more clearly the highest-value follow-up given it would let hop4 actually be tested).
- **The full_context degradation finding (§3.4) is observed, not yet isolated**: it's plausible but unconfirmed that the 20-vs-10 passage count is the causal factor rather than something specific to MuSiQue's distractor construction (e.g. MuSiQue may deliberately include more topically-similar distractors than HotpotQA's construction process). A controlled test — same questions, varying only passage count — would be needed to isolate this cleanly.
- **Same model in all roles**, same limitation as E5/E6.

---

## 6. Connection to Thesis

E7 both strengthens and complicates the emerging picture from E5-E6:

| Finding | Status after E7 |
|---|---|
| Interactive retrieval bottlenecked by recall, not reasoning | **Generalizes a third time** — same magnitude effect on a third dataset with different passage count and question-generation methodology |
| K=3 is a fixed near-optimal budget | **Refuted** — optimal K shifts with hop count (3→5 between hop2 and hop3); E5's K=3 finding was specific to HotpotQA's ~2-hop structure |
| Static context injection (full_context/summary_doc) is safer than interactive retrieval | **Reversed at scale** — at 20 passages, full_context's distractor-confusion cost can exceed interactive retrieval's recall-bottleneck cost; the E5/E6 ranking was not passage-count-invariant |

The practical implication sharpens further: neither "always inject full context" nor "always use interactive retrieval" is a safe default — the right choice is a function of *how much* context there is to inject, not just of task structure (E5/E6's finding) or of whether static compression preserves the needed facts (E6's finding). A production memory-controller design needs all three axes: task structure, answer-fact precision requirements, and raw context volume.

---

## 7. Next Steps

1. **Isolate the passage-count effect** (§3.4, Limitations): rerun `full_context` on MuSiQue questions but truncated to 10 (or fewer) passages, holding the question/gold-answer fixed, to test whether the full_context degradation is really about volume rather than MuSiQue-specific distractor construction.
2. **Larger K for hop4** (K=10, 15): §3.3 showed hop4's full-recall rate is still climbing at K=7 (0%→6.7%→20%) — a genuinely larger budget might finally reveal whether 4-hop has its own optimal K or whether the task is simply out of reach for TF-IDF retrieval at any reasonable K.
3. **Embedding-based retrieval** (carried over from E5/E6 Next Steps, now higher priority): would directly test whether the recall ceiling seen across all three experiments is a lexical-matching artifact or a more fundamental budget limitation.
4. **Larger n per hop×K cell**: n=15 leaves the headline hop3 result (K=5 optimal) under-replicated; n=30+ would tighten the confidence in exactly where the K-optimum sits.

---

*Run `python main.py` for the static baselines (resumes automatically from cached per-hop-bucket results; `--fresh` forces a clean recompute).*
*Run `python k_ablation.py` for the K sweep (reads K values from `config.yaml`'s `retrieval.k_values`, same resume/`--fresh` behavior).*
*Run `python analyze_k_ablation.py` to regenerate tables and the `plots/e7_hopcount_k_ablation.png` chart.*
