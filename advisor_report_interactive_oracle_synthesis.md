# Interactive Oracle / Retrieval for Knowledge Transfer: Advantages, Challenges, and Evidence Across Nine Experiments

**Date:** 2026-08-02
**Author's note:** This report synthesizes E3-E9, seven experiments run over local llama3:8b (CPU, Ollama) on seven benchmarks, testing one recurring question in different forms: *can an agent that queries a knowledge source interactively — asking targeted questions rather than receiving everything at once — outperform static context-injection strategies (dumping full context, or handing it a pre-built summary)?*

**One-sentence answer:** Sometimes, and the conditions under which it wins are now fairly well characterized — but it never wins by a wide margin, it is always bounded by how well it can *find* the right information, and it loses outright to a genuinely good static summary whenever one is available.

---

## 1. Executive Summary

| # | Finding | Status |
|---|---|---|
| 1 | The original interactive-oracle design (E3/E4) was architecturally broken — it re-paid the full knowledge-base cost on every one of K questions, making it strictly worse than static handoff by construction | **Fixed** in E5 onward (retrieval-oracle redesign) |
| 2 | A properly designed interactive retrieval oracle's accuracy is **bottlenecked by retrieval recall, not reasoning quality** | **Confirmed on 4 independent datasets** (E5, E6, E7, E9) — the single most robust finding in this research program |
| 3 | Static context injection (`full_context`, dumping everything) **degrades as the amount of context grows**, even before any hard token-window limit is hit | **Confirmed and progressively sharpened** (E7 → E8 → E9) |
| 4 | Interactive retrieval can **beat** static context injection outright, specifically once the context is large enough that dumping it causes distractor confusion | **Demonstrated** (E7's MuSiQue, several hop×K cells; E9's NarrativeQA, vs full_context specifically) |
| 5 | The optimal query budget (K) is **not a fixed constant** — it scales with how many reasoning hops a task actually requires, and raising K past the task's real requirement **actively hurts** (context dilution) rather than helping | **Demonstrated** (E5 K-ablation; E7 hop-count × K ablation) |
| 6 | A **good static summary is the best condition in this entire research program**, when one is available or feasible to produce — it wins even in the regime designed to favor interactive retrieval | **Demonstrated most starkly in E9**; task-dependent elsewhere (E5, E8 favor it; E6 shows it can catastrophically fail) |
| 7 | Interactive retrieval's query-generation step has a **grounding gap**: without an anchor to what it's actually searching, it can hallucinate a plausible-sounding but entirely wrong target and confidently search for it | **Newly discovered** (E9) — not visible in earlier experiments because their entities were common-knowledge Wikipedia topics |

The research arc moved from *"is interactive querying worth it at all?"* (E3/E4: no, but the test was broken) to *"under a fair design, when specifically does it help?"* (E5-E9: a well-characterized but narrow set of conditions, always capped by recall and query-grounding quality).

---

## 2. The Nine Experiments at a Glance

| # | Dataset | Domain | What varies | Headline result |
|---|---|---|---|---|
| E3 | TravelPlanner | Constraint-diverse planning | Teacher→student knowledge transfer, 4 conditions | Stateless oracle costs 3.8× more than full handoff for *worse* accuracy — broken test |
| E4 | ALFWorld | Uniform household tasks | Same 4 conditions, uniform-structure task | Oracle matches full handoff (100%) at 10× the cost — same design flaw, different domain |
| E5 | HotpotQA | Multi-hop trivia QA (10 passages) | Redesigned retrieval oracle; bridge vs comparison questions | Recall bottleneck established; K=3 near-optimal, more hurts |
| E6 | 2WikiMultihopQA | Multi-hop trivia QA (10 passages, 4 types) | Generalization test across richer question taxonomy | Recall bottleneck replicates; summary_doc reliability does NOT generalize |
| E7 | MuSiQue | Multi-hop trivia QA (20 passages, controlled hop count) | Hop-count × K ablation | Optimal K shifts with hop count; full_context reverses ranking at 20-passage scale |
| E8 | QASPER | Single research papers (~2k-8k tokens) | Paper length (approaching context window) | full_context never wins outright, even within budget; hedging pattern replicates in new domain |
| E9 | NarrativeQA | Full books/scripts (60k-90k+ tokens) | Genuinely infeasible full_context (forced truncation) | Good static summary dominates everything; new query-grounding failure mode found |

All experiments after E4 share one harness design (four conditions — `no_context`/`no_memory`, `full_context`/`full_handoff`, `summary_doc`/`graph_session`, `interactive_retrieval`/`interactive_oracle` — evaluated with EM/F1 or task-success metrics, run on the same local llama3:8b model via Ollama, CPU-forced after a GPU-crash discovery in E5). Full technical detail, exact numbers, and per-experiment limitations live in each experiment's own `REPORT.md`; this document synthesizes across them.

---

## 3. Phase 1 (E3-E4): The Original Design Was Broken

The starting hypothesis — *interactive querying is more token-efficient than full context handoff* — was tested with a **stateless oracle**: each of the student's K questions triggered a fresh call that re-sent the entire teacher knowledge base in the system prompt. This makes total oracle cost scale as `K × knowledge_base_size` instead of `1 × knowledge_base_size + K × question_size`.

- **TravelPlanner (E3):** oracle achieved 20% success (worse than doing nothing, 60%) at 108,193 tokens per success — 3.8× more tokens than full handoff for worse accuracy. TravelPlanner's diverse, per-task constraints also meant *any* injected teacher context (not just interactive) hurt — a **context pollution** effect specific to task-structure diversity.
- **ALFWorld (E4):** oracle matched full handoff's 100% success rate — but at 14× the token cost and 22× the wall-clock time. The uniform procedural structure (Goto→Pickup→Goto→Put every time) meant injected context helped regardless of mechanism; interactivity added nothing but cost.

**Conclusion of Phase 1:** the interactive-querying hypothesis was never fairly tested — the stateless design guaranteed the oracle would lose on cost, independent of whether interactivity itself has value. This motivated a redesign.

---

## 4. Phase 2 (E5-E9): A Fair Test, and What It Actually Shows

E5 onward replaced the stateless oracle with a **retrieval oracle**: a TF-IDF index over the knowledge source (passages, paper sections, or document chunks), where each of the model's K queries retrieves only the 1 best-matching chunk. This is architecturally the "correct" design flagged in the E3/E4 postmortem — no repeated re-sending, cost scales with K×(small retrieval), not K×(whole knowledge base).

### 4.1 Advantage: Interactive retrieval CAN beat static context, in a specific regime

The clearest positive result is that once the static context gets large enough to actively confuse the model, a recall-bottlenecked interactive mechanism can outscore it outright:

- **E7 (MuSiQue, 20 passages):** on 3-hop questions at K=5, `interactive_retrieval` scored 53.3% F1 vs `full_context`'s 14.7% — full_context's answers were long (22.7 words average vs interactive's 2.2), hedging ("According to the reference passage...") and frequently picked the *wrong entity* among 20 distractor-heavy passages.
- **E9 (NarrativeQA, 60k-90k+ token documents):** `interactive_retrieval` beat `full_context` (15.8% vs 6.0% F1) even though `full_context` had to be truncated to fit and `interactive_retrieval`'s own recall was mediocre.
- **E8 (QASPER, papers approaching the context window):** `full_context` never won outright at any paper length, even short papers with room to spare — its advantage from earlier work (implicitly, E5's 10-passage HotpotQA setting) had already eroded before any hard limit was hit.

The common thread: **raw context volume is not free**, and past some (task-dependent, not yet precisely characterized) threshold, a mechanism that keeps the model's working context small — even at the cost of sometimes missing relevant information — outperforms a mechanism that shows the model everything.

### 4.2 Challenge: The retrieval-recall bottleneck (the most robust finding in this program)

Every multi-hop QA experiment that tested it found the same shape: `interactive_retrieval`'s accuracy is almost entirely explained by whether its query budget happened to surface the passages actually needed.

| Experiment | Found gold evidence | Missed gold evidence |
|---|---|---|
| E5 (HotpotQA, bridge questions) | 80.0% EM | 20.0% EM |
| E6 (2WikiMultihopQA) | 45.5% EM | 12.8% EM |
| E7 (MuSiQue, pooled across K/hop) | 46.4% EM | 16.8% EM |

Three independent datasets, three different passage counts (10/10/20), two different question-generation methodologies (crowd-sourced vs. algorithmically composed) — the same 2.5-4× accuracy gap between "retrieval succeeded" and "retrieval failed." **This is the load-bearing mechanism behind nearly every other result in this program.** When the interactive mechanism wins (§4.1), it's despite this bottleneck, not because it's been solved.

### 4.3 Challenge: More query budget (K) is not a free lever — and the optimum is task-dependent

The naive fix to the recall bottleneck — give it more queries — doesn't work as a blanket policy:

- **E5's K-ablation (HotpotQA):** raising K from 3 to 10 *increased* the rate of finding the right evidence (full-recall rate: 21.7% → 46.7%) but *decreased* F1 (53.4% → 44.3%). More queries meant more accumulated, partly redundant context, diluting the signal the model needed — the same mechanism as raw context-volume pollution (§4.1), just emerging from accumulated retrieval rather than a single big dump.
- **E7's hop-count × K ablation (MuSiQue):** the *optimal* K genuinely shifts with task depth — 2-hop questions peak at K=3 (replicating E5 exactly), but 3-hop questions peak at K=5 (53.3% EM, beating both K=3's 40.0% and K=7's 26.7%). E5's "K=3 is near-optimal" finding was specific to HotpotQA's effectively-2-hop structure, not a general constant.
- 4-hop questions in E7 showed no usable optimum within the tested range (K≤7) — recall never rose high enough to reveal one, a reminder that "give it more budget" has diminishing and eventually untestable returns without also improving retrieval precision.

### 4.4 Challenge: A good static summary is a very high bar — sometimes the best condition outright

Across the program, static compression (an LLM- or human-written summary, used instead of raw context) shows the widest swing in outcome of any condition:

- **E5 (HotpotQA):** `summary_doc` was the *best* condition overall (61.9% F1), beating both `full_context` (59.5%) and `interactive_retrieval` (53.4%).
- **E6 (2WikiMultihopQA):** the same mechanism **collapsed** on `compositional`-type questions (14.5% F1 — worse than the no-context baseline) because the summary dropped precise dates/names those questions needed — summarization's value is not a fixed property of the technique, it depends on whether the target answer is the kind of fact that survives paraphrasing.
- **E8 (QASPER):** `summary_doc` won again (30.1% F1, best condition), on a dataset that also demands precise facts (method names, numbers) — directly contradicting a naive "summarization loses precision" generalization from E6.
- **E9 (NarrativeQA):** `summary_doc`, using the dataset's own **human-written** summary (an LLM couldn't summarize a 90k-token document in one call), **dominated overwhelmingly** — 67.8% F1 vs `full_context`'s 6.0% and `interactive_retrieval`'s 15.8%, at a fraction of the token cost of either. This is the strongest result in the entire program, and it happened in the regime specifically designed to showcase interactive retrieval's advantage.

**The honest takeaway:** static summarization's reliability is genuinely dataset/task-dependent (E6 is a real counterexample, not noise), but *when it works*, it is very hard for either raw context injection or interactive retrieval to beat — and it is dramatically cheaper than both.

### 4.5 Challenge: DFS-shaped ("deep, chained") querying happens whether or not the task rewards it

A secondary research thread asked whether interactive querying induces *task-adaptive* search behavior — chaining queries specifically on tasks that have a real dependency structure (bridge questions), and not on tasks that don't (comparison questions).

- **E5:** chain-following rate was ~82-83% on both bridge and comparison questions — essentially identical. The interaction protocol ("build your next query on what you just learned") induces sequential, chained querying *unconditionally*, as an artifact of the prompt design, not as an adaptive response to task structure.
- **E6:** showed a small but real divergence (comparison: 60% chain-following vs the other three types: 78-98%) — a partial qualification, but the dominant pattern (high chaining even on non-chain-shaped tasks) held.

**Implication:** "interactive querying induces DFS-style behavior" is true as a mechanical description, but the original hope — that this behavior would be *selectively* triggered by tasks that actually need it — is not well supported. The mechanism fires from the prompt design, not from the model detecting task structure.

### 4.6 Newly discovered challenge: query-generation needs to be grounded (E9)

E9 (NarrativeQA) surfaced a failure mode invisible in every earlier experiment: when the interactive mechanism's query-generation step has no anchor to what it's actually researching (no document title, nothing beyond the question text), and the target document is obscure enough that llama3:8b has no strong prior about it, the model **confidently generates queries about a different, more famous, superficially-similar work**:

- Researching the 1601 Ben Jonson play *Cynthia's Revels*, the model asked about "Cynthia Rhodes actress filmography," "Vices in The Picture of Dorian Gray," and "Cynthia's sentence in American Gods" — none related to the actual play.
- Researching the boxing film *The Fighter*, the model asked about "Rocky IV Atlantic City fight" and "Mickey Rourke eye injury" — confusing it with the unrelated (but more famous) Rocky franchise and a different actor.

This explains why `interactive_retrieval` recovered only 1 of 6 (17%) cases where `full_context`'s answer was structurally unreachable (truncated away) but `interactive_retrieval` — in principle — had index access to the full document: the retrieval mechanism itself was often fine, but the query asked was about the wrong universe entirely. **This is upstream of, and distinct from, the recall bottleneck in §4.2** — no amount of better retrieval matching fixes a query aimed at the wrong target. It didn't appear in E5-E8 because those datasets' entities (HotpotQA/2Wiki/MuSiQue people and places, QASPER's own paper content) were either common-knowledge or the query-generation prompt had less room to substitute a wrong-but-plausible target.

---

## 5. A Decision Framework

Synthesizing across all seven Phase-2 experiments, the evidence supports a rough decision structure for when each mechanism is likely to help:

| Situation | Best mechanism | Why |
|---|---|---|
| Context is small and fits comfortably (≲10 passages / well under context window) | `full_context` or a good `summary_doc` | Static context isn't yet large enough to cause hedging/confusion; interactive retrieval's recall bottleneck has nothing to make up for |
| Context is large but still fits (≳20 passages, or a paper near the context window) | `summary_doc` if feasible, else `interactive_retrieval` | `full_context` starts hedging/confusing itself; interactive retrieval's recall cost is now worth paying |
| Context is genuinely infeasible (must truncate or can't process at all) | **A good summary, if one can be produced or is available** | E9: dominates by a wide margin. If no good summary is available, `interactive_retrieval` is the fallback, not `full_context` (truncated) |
| Task requires deep, multi-hop chaining AND retrieval can reliably find the chain | `interactive_retrieval`, tuned K to the task's actual hop depth | K=3 for ~2-hop tasks; higher K for deeper tasks, but not unboundedly — watch for dilution |
| The knowledge source is obscure / the model has no prior about it | Static context (grounded from token one) over interactive retrieval | Query-generation grounding gap (§4.6) — interactive retrieval risks confidently searching the wrong thing |

**The one condition that is never clearly wrong**: a well-constructed static summary. It is the best or tied-best condition in 4 of 5 datasets where it was tested (E5, E7 static baselines, E8, E9), and where it failed (E6), the failure was interpretable (precision-demanding facts lost to paraphrasing) rather than mysterious.

---

## 6. Limitations of This Research Program as a Whole

- **Single model, all roles, throughout**: llama3:8b generates questions, retrieves, summarizes, and answers in every experiment. A stronger model might close the recall bottleneck (§4.2) or the grounding gap (§4.6) without any architectural change — this is untested.
- **CPU-only, local inference**: all experiments ran on forced CPU inference (a GPU-offload crash was discovered and worked around during E5) — absolute timing numbers reflect this constraint, though the accuracy/mechanism findings should not depend on it.
- **TF-IDF (lexical) retrieval only, throughout**: every "interactive_retrieval" condition across E5-E9 uses the same TF-IDF cosine matching. The recall bottleneck (§4.2) may be substantially a lexical-matching limitation — an embedding-based retriever is the most consistently flagged, never-yet-tested lever across every individual report's "Next Steps."
- **Sample sizes are modest** (n=15-30 per condition/bucket in most experiments, n=10-20 in E8/E9) — individual percentage-point differences should not be over-read; the cross-experiment *replication* of directional findings (the recall bottleneck appearing 4 times, the hedging pattern appearing 3 times) is the stronger form of evidence in this program, not any single experiment's precision.
- **E9's `summary_doc` is human-written, not LLM-generated** — its dominance (§4.4) shows what a *good* summary can do, but doesn't establish that an LLM could produce one that good from a 90k-token document. This is flagged as E9's own top limitation.
- **The query-grounding fix (§4.6) is proposed, not yet tested** — including the document/topic title in the query-generation prompt is a one-line, low-cost change with clear motivating evidence, but its actual effect is not measured anywhere in this program yet.

---

## 7. Recommended Next Steps (Consolidated)

In rough priority order, combining every individual experiment's "Next Steps" section:

1. **Ground the query-generation prompt** (E9) — include the document/topic identity, not just the question. Highest expected value relative to implementation cost of any open item.
2. **Embedding-based retrieval** instead of TF-IDF (flagged since E5, repeated in E6/E7/E8/E9) — the most consistently identified lever for the recall bottleneck, never yet tested.
3. **Adaptive query stopping** — let the model decide when it has enough information rather than always spending exactly K queries; motivated by E5/E7's finding that excess queries dilute rather than help, and by E7's observation that models run out of genuinely new things to ask well before a fixed K is exhausted.
4. **Test whether the E6 vs E8 summarization split resolves** with a systematic study of *which kinds of questions* summarization is safe for (a precision/paraphrase-survivability axis), rather than treating it dataset-by-dataset.
5. **Stronger or larger base model**, to test whether the recall bottleneck and grounding gap are model-capability-limited (would shrink with a better model) or architecture-limited (would persist regardless).
6. **Larger sample sizes**, particularly for the K-ablation and hop-count results (E7), which currently rest on n=15/cell.

---

## 8. Source Material

Each experiment's full methodology, complete result tables, and individually-scoped limitations are documented in its own report:

- `advisor_report_interactive_transfer.md` — E3 (TravelPlanner) and E4 (ALFWorld), the original stateless-oracle design and its diagnosis
- `experiments/hotpotqa_transfer/REPORT.md` — E5 (HotpotQA) and its K-ablation addendum
- `experiments/wikihop_transfer/REPORT.md` — E6 (2WikiMultihopQA)
- `experiments/musique_transfer/REPORT.md` — E7 (MuSiQue) hop-count × K ablation
- `experiments/qasper_transfer/REPORT.md` — E8 (QASPER)
- `experiments/narrativeqa_transfer/REPORT.md` — E9 (NarrativeQA)

Two earlier, separate lines of work exist in this repository but are out of scope for this synthesis: `experiments/memory_controller/` (E1, a graph-structured procedural memory controller, a different research question about memory *representation* rather than *access mechanism*) and `experiments/interactive_handoff/` (an incomplete, partial-data experiment on agent-to-agent coordination rather than knowledge retrieval — results explicitly marked directional-only in that report).
