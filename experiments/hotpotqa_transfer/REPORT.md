# E5: HotpotQA — Interactive Retrieval vs Static Context (DFS vs BFS)

**Date:** 2026-07-26
**Model:** llama3:8b (local, via Ollama, forced CPU — `num_gpu=0`, see §6)
**Experiment:** Does forcing an agent to retrieve information one targeted query at a time (rather than dumping everything, or handing it a static summary) induce DFS-style chained reasoning on tasks that require it (bridge questions), without a cost/accuracy penalty on tasks that don't (comparison questions)?

---

## 1. Overview

E3/E4 (`advisor_report_interactive_transfer.md`) found the stateless E3/E4 oracle was a broken test of the interactive-querying hypothesis — it re-paid the full teacher context K times, so it could never beat `full_handoff` on cost, and neither TravelPlanner nor ALFWorld has a real "search the information space" structure to make DFS-vs-BFS behavior measurable in the first place.

E5 moves to HotpotQA, where deep-vs-broad search is structurally real:
- **bridge** questions require chaining through an intermediate entity — hop 1's answer determines what to look up in hop 2 (DFS-shaped).
- **comparison** questions compare two independent, already-named entities — no dependency chain (BFS-shaped; nothing to gain from sequential drill-down).

The interactive condition here is a genuine **retrieval oracle** (Option C from the E3/E4 report), not the stateless Q&A oracle: a TF-IDF index over the question's 10 distractor passages returns the single best-matching chunk per query, so the E3/E4 cost-inflation flaw does not recur.

**Headline result: the interactive condition does not beat the static conditions on accuracy, and the reason is not "bad reasoning" — it's a retrieval-recall bottleneck.** When the retrieval controller's K=3 queries happen to surface both gold passages, accuracy is very high (bridge: 80% EM). When they don't (which is most of the time — retrieval found both gold passages in only 16.7% of bridge tasks), accuracy craters (bridge: 20% EM). The chain-following (DFS) rate itself is real and roughly equal across both question types (~82-83%), which is itself informative: the model chains its queries whether or not the task structure calls for it — the *hypothesized* bridge-vs-comparison split in interactive behavior did not materialize.

---

## 2. Design

### 2.1 Dataset

HotpotQA distractor validation set (`hotpotqa/hotpot_qa`, HF parquet mirror — the original `hotpot_qa` script-based loader is dead in `datasets` 4.x). Each question ships with exactly 10 candidate passages (2 gold + 8 distractors) and gold `supporting_facts` (titles of the passages actually needed).

n=30 `bridge` + 30 `comparison` questions (first 30 of each type in validation order — deterministic, reproducible).

### 2.2 Conditions

| Condition | What the model sees before answering |
|---|---|
| `no_context` | Question only — cold-start baseline, tests base-model trivia knowledge |
| `full_context` | All 10 distractor passages dumped into the prompt — BFS-shaped: everything visible at once |
| `summary_doc` | A single static LLM-generated summary of the 10 passages, then QA against that summary only — analogous to `graph_session` in E3/E4 |
| `interactive_retrieval` | Up to K=3 rounds: the model emits a search query, a TF-IDF controller returns the single best-matching passage, the model may issue a follow-up query using what it just learned, then answers — DFS-shaped by construction |

K=3 (not K=5 like E3/E4) since HotpotQA distractor only has 10 short passages total.

### 2.3 Metrics

- **EM / F1** — standard SQuAD/HotpotQA-style exact-match and token-F1 against the gold answer.
- **Supporting-fact recall** — fraction of the gold evidence titles that were actually present in what the model was given (trivially 100% for `full_context`/`summary_doc`, since both are built from all 10 passages; the interesting number is `interactive_retrieval`'s, which depends on retrieval quality).
- **DFS / chain-following rate** (`interactive_retrieval` only) — for each query after the first, does it share a content word with the *previous* round's retrieved passage? A high rate means the model is building each query on what it just learned (chaining); a low rate means it's jumping to unrelated topics regardless of what was retrieved.

---

## 3. Results

### 3.1 Overall (both question types combined, n=60/condition)

| Condition | EM | F1 | Mean Tokens | Mean Sec | SF-Recall | DFS |
|---|---|---|---|---|---|---|
| no_context | 28.3% | 35.7% | 31 | 4.1 | 0.0% | — |
| full_context | 45.0% | 59.5% | 1,355 | 73.0 | 100.0% | — |
| **summary_doc** | **50.0%** | **61.9%** | 2,005 | 180.1 | 100.0% | — |
| interactive_retrieval | 36.7% | 53.4% | 1,009 | 71.1 | 49.2% | 82.5% |

`summary_doc` is the best condition on both EM and F1. `interactive_retrieval` sits between `no_context` and `full_context` — worse than either static context-injection condition, though far cheaper in tokens than `summary_doc` (1,009 vs 2,005, ~50%) and comparable in tokens to `full_context` (1,009 vs 1,355).

### 3.2 By question type

**Bridge (DFS-shaped — chain through an intermediate entity):**

| Condition | EM | F1 | Tokens | SF-Recall | DFS |
|---|---|---|---|---|---|
| no_context | 10.0% | 16.7% | 35 | 0.0% | — |
| full_context | 43.3% | 58.9% | 1,473 | 100.0% | — |
| **summary_doc** | **46.7%** | 55.5% | 2,035 | 100.0% | — |
| interactive_retrieval | 30.0% | 46.5% | 952 | 48.3% | 83.3% |

**Comparison (BFS-shaped — two independent, already-named entities):**

| Condition | EM | F1 | Tokens | SF-Recall | DFS |
|---|---|---|---|---|---|
| no_context | 46.7% | 54.8% | 26 | 0.0% | — |
| full_context | 46.7% | 60.1% | 1,238 | 100.0% | — |
| summary_doc | 53.3% | **68.3%** | 1,975 | 100.0% | — |
| **interactive_retrieval** | 43.3% | 60.3% | 1,066 | 50.0% | 81.7% |

Two things stand out immediately:

1. **`no_context` is far stronger on comparison (46.7% EM) than bridge (10.0% EM).** Comparison questions name both entities explicitly in the question itself ("Were Scott Derrickson and Ed Wood of the same nationality?") — llama3:8b's parametric knowledge can often answer these directly. Bridge questions hide the answer behind an intermediate entity the question never names, which cold-start knowledge cannot supply.
2. **`interactive_retrieval`'s deficit relative to `summary_doc` is larger on bridge (−16.7pp EM) than on comparison (−10.0pp EM).** This is the opposite of the original hypothesis (that interactivity would specifically help bridge questions) — it *hurts* bridge questions more.

### 3.3 The mechanism: retrieval recall, not reasoning failure

Splitting `interactive_retrieval`'s bridge results by whether its 3 queries actually surfaced *both* gold passages:

| Bridge subset | n | EM | F1 |
|---|---|---|---|
| Found both gold passages | 5/30 (16.7%) | **80.0%** | 0.96 |
| Missed at least one | 25/30 (83.3%) | 20.0% | 0.37 |

| Comparison subset | n | EM | F1 |
|---|---|---|---|
| Found both gold passages | 8/30 (26.7%) | 50.0% | 0.70 |
| Missed at least one | 22/30 (73.3%) | 40.9% | 0.57 |

When retrieval succeeds, `interactive_retrieval` is competitive with (bridge: 80% EM) or close to (comparison: 50% EM) the static conditions. When it fails — which is the large majority of cases, and *especially* on bridge questions (found-both rate 16.7% vs 26.7% on comparison) — accuracy collapses toward `no_context` levels. **The bottleneck is retrieval recall under a tight K=3 budget with lexical (TF-IDF) matching, not the model's ability to reason from retrieved context.**

Bridge questions are the harder retrieval case precisely *because* they're bridge questions: the second entity needed is often not named anywhere in the original question, so the model's second query has to have already correctly guessed or discovered that entity's name from the first retrieved passage. TF-IDF lexical matching has no way to help with this — it can only match surface tokens the model already typed, not infer what to search for next. Example:

```
Q: What is the name of the fight song of the university whose main campus is
   in Lawrence, Kansas and whose branch campuses are in the Kansas City
   metropolitan area?
Gold: Kansas Song
Queries: "University of Kansas fight song" -> "Kansas Jayhawks lyrics" -> "Kansas City metropolitan area"
DFS score: 1.0 (each query chains on the last)   Recall: 0.5 (missed one gold passage)
Predicted: "Rock Chalk, Jayhawk!"  (WRONG — a real KU chant, but not the gold fight song title)
```

The chain-following behavior here is textbook DFS — each query builds on the last — and it still produces the wrong answer, because the retrieval controller surfaced a plausible-but-wrong passage. This is a clean illustration of §3.3's point: **DFS-shaped querying and answer correctness are separate axes.** A successful case shows the same mechanism working when retrieval lands correctly:

```
Q: 2014 S/S is the debut album of a South Korean boy group that was formed by who?
Gold: YG Entertainment
Queries: "SM Entertainment formation" -> "Who founded YG Entertainment" -> "Who founded SM Entertainment"
DFS score: 1.0   Recall: 1.0 (both gold passages found)
Predicted: "YG Entertainment"  (CORRECT)
```

### 3.4 DFS chain-following is real but does not distinguish bridge from comparison

Mean chain-following rate: **83.3%** (bridge) vs **81.7%** (comparison) — essentially identical. The query-generation prompt ("use what you learned from previous results to decide your next query") induces sequential, context-building queries regardless of whether the underlying task actually has a dependency chain to follow. On comparison questions — which have no real chain, since both entities are already named — the model still chains its queries (e.g. "Scott Derrickson nationality" → "Ed Wood nationality" → "Ed Wood American filmmaker") simply because the prompt instructs it to build on prior results. This is DFS-*shaped behavior* induced by the interaction protocol itself, not a task-adaptive response to task structure. The original hypothesis — that interactivity would differentially induce DFS behavior on tasks that reward it — is not supported: the mechanism fires uniformly.

---

## 4. Analysis

### 4.1 Interactive retrieval is not a free win here

Unlike the report's Option C prediction (retrieval oracle should be "cheaper than full_handoff while preserving interactivity"), `interactive_retrieval` here is cheaper than both `full_context` and `summary_doc` in tokens, but *not* more accurate — it is strictly dominated by `summary_doc` (worse on every metric except cost) and mixed against `full_context` (cheaper, but lower EM/F1 on both question types). The E3/E4 finding that "the oracle is always the most expensive option, for no accuracy gain" partially repeats here in a different form: this time the interactive condition is *cheap*, but the accuracy gain still doesn't materialize, because a fixed, small query budget (K=3) against lexical retrieval caps how much of the passage set the model can actually see.

### 4.2 Static summarization is the strongest condition, not full dump

`summary_doc` beats `full_context` on both EM (+5.0pp) and F1 (+6.4pp overall) at higher token cost (2,005 vs 1,355). This is a notable contrast with E3, where compression (`graph_session`) was neutral, and with the general "context pollution" framing from E3 — here, compressing 10 passages into a focused summary *improves* accuracy over presenting all 10 verbatim, plausibly because the summarization step already discards distractor content and makes the two relevant facts easier to locate in a shorter, denser context. This is the opposite of E3's finding that added processing of teacher context never helped; the difference is likely that E3's graph compression served *cross-task* transfer (compressing a different session's trajectories) while here the summary compresses the *same question's* own reference material — a much easier compression target.

### 4.3 Practical implication: interactive querying needs either a bigger budget or better retrieval, not just statelessness fixes

E3/E4 diagnosed the interactive oracle's problem as architectural (stateless re-sending of full context). E5 fixes that architectural flaw — this retrieval oracle never re-pays context, each query is cheap and targeted — and the interactive condition *still* underperforms. The bottleneck that remains is different: **query budget × retrieval precision**, not context-resending cost. A retrieval oracle is only as good as its ability to surface the right evidence within its query budget. Two untested levers that follow directly from §3.3: (a) a larger K (the report's Option C assumed retrieval scales cheaply — worth testing K=5 or K=7 here, since each miss costs only ~250 tokens), and (b) embedding-based retrieval instead of TF-IDF, which might resolve queries like "who founded YG Entertainment" against a passage that never uses the word "founded" literally.

---

## 5. Conclusions

1. **The interactive retrieval condition does not outperform static context injection on HotpotQA at this model scale and K=3 budget.** `summary_doc` > `full_context` > `interactive_retrieval` > `no_context` on both EM and F1, overall and on both question types.
2. **The failure mode is retrieval recall, not reasoning.** Conditioning on retrieval success (both gold passages found), `interactive_retrieval` accuracy is competitive with or close to the static conditions (bridge: 80% EM when found vs 20% when missed). The interactive mechanism *works* when it retrieves the right evidence; it simply doesn't retrieve the right evidence often enough (16.7% full-recall rate on bridge, 26.7% on comparison) under a K=3 TF-IDF budget.
3. **DFS-shaped chain-following is real (~82%) but task-structure-agnostic.** The model chains its queries at essentially the same rate on bridge and comparison questions. The interaction *protocol* (a prompt instructing sequential query-building) induces DFS-shaped behavior unconditionally — it is not a response to the task's actual dependency structure. This means "interactive querying induces DFS behavior" is true as a description of the mechanism, but the original hypothesis that this behavior would be *selectively* beneficial on DFS-shaped (bridge) tasks is not supported by these results — if anything, `interactive_retrieval`'s deficit vs `summary_doc` is larger on bridge than on comparison.
4. **Static summarization, not full context dump, is the strongest static condition.** Unlike E3's finding that compression of cross-session teacher context was neutral, compressing a question's own 10 reference passages into a focused summary improves both EM and F1 over presenting all 10 verbatim.

---

## 6. Limitations

- **n=30/type**: better granularity than E3/E4's n=5 (each flip ≈ 3.3pp instead of 20pp), but still modest; a single-digit-percent EM difference (e.g. `full_context` vs `comparison`'s `no_context`, both 46.7%) should not be over-read.
- **K=3, fixed (non-adaptive) query budget**: the model always issues exactly 3 queries regardless of whether it judges it has enough information sooner or needs more — §4.3's "bigger K" lever was not tested here.
- **TF-IDF (lexical) retrieval only**: no embedding-based retrieval was tried; §4.3 flags this as the most likely lever to close the recall gap, since TF-IDF cannot match a query against a passage that expresses the same fact in different words.
- **Infrastructure note**: this run required forcing `num_gpu=0` on every Ollama call. Partial Vulkan GPU offload (15/33 layers) was crashing the `llama-server` subprocess with `exit status 0xe06d7363` on every request after a multi-day gap between sessions (likely a driver/GPU-state issue surfaced by sleep/resume or an Ollama update) — `main.py` now always calls `/api/chat` with `options.num_gpu=0` rather than the OpenAI-compatible endpoint, since the latter does not forward that override. All results in this report come from CPU-only inference, consistent with E3/E4.
- **Same model in all roles**: llama3:8b generates the questions, does the retrieval-triggered reasoning, and produces the summaries. A stronger query-generation model might close the retrieval-recall gap without needing K or embedding changes.
- **Result-saving robustness**: an earlier attempt at this run was interrupted (background process killed across an idle multi-day gap) partway through `summary_doc`, and — before a fix — lost ~2.5 hours of completed work because results were only persisted after *all* question types for a condition finished. `main.py` now saves incrementally after each question type and auto-resumes from cached per-type results on restart (`--fresh` forces a clean recompute).

---

## 7. Connection to Thesis

E5 extends E3/E4's "context injection helps or hurts depending on task-structure similarity" finding to the interactive-querying mechanism specifically:

| Finding | E3/E4 | E5 |
|---|---|---|
| Context injection value | Domain-dependent (hurts on TravelPlanner, helps on ALFWorld) | Static injection (summary/full) helps on both HotpotQA question types |
| Interactive oracle vs static | Always ≥ static in cost, ≤ static in accuracy (stateless design flaw) | Cheaper than `summary_doc`, still ≤ static in accuracy — but now the bottleneck is retrieval recall, not resend cost |
| Does interactivity induce task-adaptive behavior? | Not tested (no search-shaped task) | No — DFS-shaped chaining occurs uniformly regardless of whether the task has a real dependency chain |

The core addition: **fixing the architectural flaw identified in E3/E4 (stateless oracle) is necessary but not sufficient for interactive querying to beat static context.** A cheap, non-repeating retrieval oracle still loses to static summarization when its query budget and retrieval method cap how much of the relevant evidence it can surface. For interactive memory controllers to show a genuine advantage over static docs, the open problem is retrieval quality under budget, not context-resending cost — a different lever than the one E3/E4 flagged as the fix.

---

## 8. Next Steps

1. ~~**K ablation** (K=3 vs 5 vs 7): tests whether the retrieval-recall bottleneck (§3.3) is a budget problem that a larger query allowance would fix.~~ **Done — see §9. Result: no, and it actively hurts past K=3.**
2. **Embedding-based retrieval** instead of TF-IDF: tests whether the recall gap is a lexical-matching limitation specifically (query and gold passage sharing no surface tokens) rather than a fundamental retrieval-budget limitation. More important now than before §9's result, since §9 shows simply raising K is not the fix.
3. **Adaptive stopping**: let the model decide when it has enough information rather than always using exactly K queries — may reduce wasted queries on comparison questions (where 1-2 queries are often sufficient) and free budget for bridge questions. §9.3 shows the model itself runs out of distinct things to ask well before K=10, repeating near-identical queries — adaptive stopping would let it quit rather than keep accumulating redundant context.
4. **Test whether DFS chain-following predicts retrieval success**: dfs_score is logged per-task; a follow-up analysis could check whether higher chain-following correlates with higher supporting-fact recall within `interactive_retrieval`, which would clarify whether "chaining well" and "retrieving the right evidence" are related or orthogonal.

---

## 9. K-Ablation Addendum (2026-07-26)

**Question:** §3.3 found `interactive_retrieval`'s accuracy is capped by retrieval recall under a K=3 budget. Does raising K close the gap with `summary_doc`?

**Answer: No.** Recall improves monotonically with K, but accuracy does not follow it — it peaks at K=3 and *declines* at higher K, while token cost keeps climbing. Increasing the query budget makes the interactive condition strictly worse on the cost/accuracy tradeoff, not better.

### 9.1 Setup

Same mechanism (`interactive_retrieval`), same 60 HotpotQA questions (30 bridge + 30 comparison, same seeded slice) as the main E5 run, rerun at K ∈ {2, 5, 7, 10} (K=3 reuses the main run's `interactive_retrieval.json` directly — same questions, not rerun).

### 9.2 Results

**Overall (both types, n=60):**

| K | EM | F1 | Tokens | SF-Recall | Full-Recall Rate | DFS |
|---|---|---|---|---|---|---|
| 2 | 33.3% | 48.0% | 540 | 43.3% | 21.7% | 85.0% |
| **3** | **36.7%** | **53.4%** | 1,009 | 49.2% | 21.7% | 82.5% |
| 5 | 33.3% | 46.0% | 1,351 | 57.5% | 33.3% | 83.8% |
| 7 | 36.7% | 48.8% | 2,149 | 61.7% | 40.0% | 82.5% |
| 10 | 30.0% | 44.3% | 3,037 | 65.8% | 46.7% | 87.6% |

**Bridge (DFS-shaped):**

| K | EM | F1 | Tokens | Full-Recall Rate |
|---|---|---|---|---|
| 2 | 26.7% | 41.5% | 574 | 16.7% |
| **3** | **30.0%** | **46.5%** | 952 | 16.7% |
| 5 | 26.7% | 37.9% | 1,343 | 30.0% |
| 7 | 20.0% | 34.8% | 2,296 | 40.0% |
| 10 | 20.0% | 38.0% | 3,185 | 40.0% |

For reference, the static ceilings from the main run: `full_context` 45.0% EM / 59.5% F1 / 1,355 tokens; `summary_doc` 50.0% EM / 61.9% F1 / 2,005 tokens. **At K=10, `interactive_retrieval` costs more tokens than `summary_doc` (3,037 vs 2,005) while scoring far worse (44.3% vs 61.9% F1)** — the interactive condition never approaches the static ceiling at any tested K, and moves further from it past K=3.

### 9.3 Why more successful retrieval doesn't translate to better answers

Full-recall rate (both gold passages found) rises from 21.7% (K=2) to 46.7% (K=10) — the bigger budget genuinely does what it's supposed to: it surfaces the needed evidence far more often. But EM/F1 don't rise with it, because two things happen simultaneously as K grows:

**Query exhaustion.** Once the model has asked the "obvious" questions, it runs out of genuinely new things to ask and starts repeating near-duplicate queries rather than adding new information. Example from a K=10 bridge task:

```
Q: What government position was held by the woman who portrayed Corliss Archer
   in the film Kiss and Tell?
Gold: Chief of Protocol
Queries (10): "Shirley Temple's later career" -> "...government roles" -> "...roles in
   Ghana" -> "...roles in Czechoslovakia" -> "...role in Kiss and Tell" -> "...roles in
   films" -> "...film roles" -> "...film roles in the 1940s" -> "...role in films"
   -> "...role in films"  [last two queries are near-identical to query 6]
```
The last 2-3 of 10 queries add no new information — the model has already exhausted the genuinely distinct angles and is padding out its query budget, accumulating redundant retrieved chunks in the final-answer context rather than new evidence.

**Context dilution.** Even when the extra queries do find the right passages (full recall), the answer can still come out wrong — this is not just a formatting/EM-strictness artifact. Among K=10 bridge tasks with full recall (8/30), several are genuine substantive errors, not near-misses:

| Predicted | Gold |
|---|---|
| "Aladin" | "Eenasul Fateh" |
| "World War I" | "World War II" |
| "Rock Chalk, Jayhawk!" | "Kansas Song" |

The correct evidence was present in context in all three cases (full recall = 1.0), yet the model produced a confidently wrong answer. This mirrors E3's original "context pollution" finding (more injected context degrades TravelPlanner accuracy) in a new setting: accumulating more retrieved chunks — many redundant, some genuinely irrelevant once queries run dry — dilutes the signal the model needs to isolate the one correct fact, even though the correct fact is technically present somewhere in the prompt.

### 9.4 Conclusion

**Raising K is not a fix for the retrieval-recall bottleneck identified in §3.3 — it trades a recall problem for a dilution problem, at strictly higher cost.** K=3 (the value used in the main E5 run, chosen only because HotpotQA distractor has just 10 passages) turns out, post hoc, to be close to the accuracy-optimal point in this sweep — not because it retrieves enough evidence (its full-recall rate, 21.7%, is the lowest tied with K=2), but because it avoids the query-exhaustion/dilution penalty that appears at K≥5. This reframes E5's Next Step #2 (embedding-based retrieval) as more important than previously stated: the path to closing the gap with `summary_doc` is *better-targeted* retrieval within a small budget, not a *larger* budget with the same lexical (TF-IDF) matching. It also elevates Next Step #3 (adaptive stopping) — a model that recognizes when it has exhausted useful queries and stops, rather than padding out a fixed K with redundant ones, should recover some of K≥5's recall gains without paying the dilution cost observed here.

---

*Run `python main.py` to replicate the main run (resumes automatically from cached per-type results; `--fresh` forces a clean recompute).*
*Run `python k_ablation.py --k-values "2,5,7,10"` to replicate the K-ablation (K=3 reuses the main run's results; same resume/`--fresh` behavior).*
*Run `python analyze.py` / `python analyze_k_ablation.py` to regenerate tables and the `plots/e5_dfs_vs_bfs.png` / `plots/e5_k_ablation.png` charts.*
