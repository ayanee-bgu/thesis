# E8: QASPER — Interactive Retrieval vs Static Context, by Paper Length

**Date:** 2026-08-01
**Model:** llama3:8b (local, via Ollama, forced CPU — `num_gpu=0`, same fix carried forward from E5-E7)
**Experiment:** E7 found `full_context`'s advantage over `interactive_retrieval` (seen in E5/E6 on 10-passage benchmarks) reverses once context gets large enough (MuSiQue's 20 passages). QASPER pushes into a new regime: single research papers, ~1,400-6,400 words (~2,000-8,300 tokens), right at the edge of llama3:8b's 8,192-token window. Does `full_context` degrade progressively as paper length approaches the window limit?

---

## 1. Overview

**Partial answer: yes, but noisily, and the more robust finding is a replication of E7's "full_context hedges and dilutes" mechanism, this time in a completely different domain (research papers instead of trivia QA).**

`full_context` F1 does not decline monotonically with paper length (24.5% short → 15.5% medium → 21.2% long — a dip, not a clean gradient) — at n=15/bucket this is noisy, not a clean confirmation of the hypothesized gradient. But the qualitative mechanism from E7 §3.4 shows up again unmistakably: `full_context` answers hedge with "According to the paper/text, ..." preambles and frequently pick a plausible-but-wrong specific claim from among many in the paper, exactly the same failure signature seen on MuSiQue's 20-passage `full_context`. **`summary_doc` is the best condition overall** (30.1% F1, beating `full_context`'s 20.4% and `interactive_retrieval`'s 26.4%) — replicating E5's/E7's "summary compression helps" finding, and notably NOT replicating E6's "summary compression collapses on precision-demanding questions" finding, even though QASPER questions also require precise facts (dataset names, specific methods). Dataset-dependence continues, but in QASPER's case it lands on the E5/E7 side, not E6's.

---

## 2. Design

Same 4 conditions and mechanisms as E5-E7. New: QASPER papers vary continuously in length, so instead of a fixed question-type taxonomy, examples are bucketed by **paper word-count tertile** (`short` / `medium` / `long`, cut points computed from the validation set's actual word-count distribution) — same bucketing pattern as E7's hop buckets, just continuous-variable-derived instead of dataset-labeled.

n=15 (paper, question) pairs per bucket (45 total), flattened from QASPER's per-paper multi-question structure. Retrieval chunks = one per paper **section** (already naturally segmented — no invented chunking, unlike E9's fixed-window chunking over raw prose). `model.timeout` = 450s (same scale as E7).

**Evidence recall changed from title-overlap to text-containment**: QASPER's gold `evidence` field is raw quoted text snippets (not passage titles like HotpotQA/2Wiki), so recall here is the fraction of gold evidence snippets that appear **verbatim** in whatever text the model was given. **This recall metric is much stricter than E5-E7's title-overlap version** — it requires an exact substring match, not just "the right passage was shown." Keep this in mind reading §3.3: `summary_doc`'s near-zero recall (6.7%) reflects paraphrasing, not retrieval failure — the summary almost never preserves the gold quote's exact wording, even when it correctly conveys the fact.

**Answer scoring**: QASPER answers are `unanswerable` / `yes_no` / `free_form_answer` / `extractive_spans` per annotator, with up to 2 annotators per question. Built one candidate gold-answer string per annotator, scored EM/F1 as max-over-candidates (same pattern as E7's `answer_aliases`).

---

## 3. Results

### 3.1 Overall (all 3 length buckets combined, n=45/condition)

| Condition | EM | F1 | Tokens | EvRecall (verbatim) |
|---|---|---|---|---|
| no_context | 4.4% | 7.2% | 23 | 6.7% |
| full_context | 0.0% | 20.4% | 3,095 | 100.0% |
| **summary_doc** | **11.1%** | **30.1%** | 3,794 | 6.7% |
| interactive_retrieval | 13.3% | 26.4% | 2,374 | 41.4% |

EM is near-zero for every context-bearing condition (0-13%) — QASPER's free-form/extractive answers rarely produce an exact string match against llama3:8b's own phrasing even when substantively correct (consistent with published QASPER work treating F1, not EM, as the primary metric). `summary_doc` wins on F1 despite having the worst verbatim-evidence recall by construction (paraphrasing destroys exact quotes) — **verbatim quote preservation and answer correctness are different things**, and QASPER's short-answer format rewards the latter.

### 3.2 By paper length — the hypothesized gradient, tested

| Bucket | Mean words | full_context F1 | interactive_retrieval F1 | summary_doc F1 |
|---|---|---|---|---|
| short | 1,962 | **24.5%** | 24.1% | 29.4% |
| medium | 3,343 | 15.5% | **32.3%** | 23.7% |
| long | 4,420 | 21.2% | 22.8% | **37.3%** |

`full_context`'s F1 is NOT a clean monotonic decline (24.5→15.5→21.2 — dips at medium, partially recovers at long). At n=15/bucket, single-item flips are ~6.7pp, so this shape should be read as noisy rather than a confirmed gradient. What IS consistent: **`full_context` never wins outright** — it's tied with `interactive_retrieval` on `short` (24.5% vs 24.1%) and loses clearly on `medium` (15.5% vs 32.3%) and `long` (21.2% vs 37.3% summary_doc). The hypothesized "progressive degradation" is not cleanly confirmed, but the related E7 claim — that `full_context`'s advantage erodes as context grows even before hitting a hard limit — is directionally supported: `full_context` never separates itself from the alternatives the way it did in E5/E6's 10-passage setting.

### 3.3 Important caveat: the truncation safety net never triggered

`full_context`'s evidence recall is 100.0% in every bucket — meaning the defensive 6,500-word truncation cap (`full_context_max_words`) never actually fired, since even the longest sampled papers (max 6,398 words in the full validation set) fit under it. **This experiment tests the "approaching the window" gradient in raw token volume, not literal truncation** — `full_context` always saw the complete paper. The genuinely infeasible case (forced truncation, guaranteed information loss) is E9's, not E8's; E8 is the "before it breaks" half of the picture E9 completes.

### 3.4 The `full_context` hedging pattern replicates from E7, in a new domain

```
gold='Europarl; MultiUN'
full_context: 'According to the paper, the au[thors evaluate on ...]'  f1=0.24
summary_doc:  'Europarl and MultiUN.'                                   f1=0.80

gold='Stanford NER; spaCy 2.0; recu[rrent...]'
full_context: 'According to the text, three N[LP tools...]'            f1=0.30
interactive:  'Stanford NER, spaCy 2.0, recur[rent...]'                 f1=0.63
summary_doc:  'Stanford NER, spaCy 2.0, and a[...]'                     f1=0.95
```

`full_context` answers consistently open with "According to the paper/text, ..." hedging preambles that dilute token-F1 even when the substantive content is on the right track — the identical pattern to E7 §3.4's MuSiQue `full_context` answers ("According to the reference passage, ..."), now observed in a structurally unrelated domain (academic papers vs. trivia/multi-hop QA). This strengthens the E7 finding from a single-dataset observation into a cross-domain pattern: **large raw context dumps induce verbose, hedging answer generation in llama3:8b, regardless of domain**, which costs F1 even independent of whether the underlying fact retrieval was correct.

---

## 4. Conclusions

1. **The hypothesized clean length-gradient in `full_context` degradation is not confirmed at n=15/bucket** — the F1 curve is noisy (dip-then-recover), not monotonic. A larger n would be needed to determine whether there's a real, smaller-magnitude gradient underneath the noise.
2. **`full_context` never wins outright at any length bucket** — this is the more robust finding. Even at QASPER's "short" bucket (~2,000 words, well within the context window with room to spare), `full_context` only ties `interactive_retrieval` rather than beating it, consistent with E7's finding that `full_context`'s E5/E6-era advantage doesn't hold once there's enough context to hedge/dilute over.
3. **The hedging-answer mechanism from E7 §3.4 replicates in a new domain** — verbose "According to the paper..." preambles that dilute F1, now confirmed on academic-paper QA as well as multi-hop trivia QA. This is now a 2-dataset-confirmed pattern, not a MuSiQue-specific quirk.
4. **`summary_doc` wins here, unlike E6's `compositional`-question collapse** — QASPER's precision-demanding short answers (method names, dataset names, numbers) still benefit from compression, unlike 2WikiMultihopQA's `compositional` questions. Whether static summarization helps or hurts precision-demanding QA remains dataset-specific, not resolved by a single general rule across E6 and E8.
5. **Evidence-containment recall and answer F1 are not the same signal** — `summary_doc`'s near-zero verbatim recall (6.7%, an artifact of paraphrasing) coexists with its best-in-class F1. Future experiments using text-containment-style recall metrics should report both, since a low verbatim-recall condition is not necessarily failing at the actual task.

---

## 5. Limitations

- **n=15/bucket (45 total) is small**; single-item flip ≈ 6.7pp F1/EM. The length-gradient question (§3.2) in particular needs a larger n to distinguish real signal from bucket-to-bucket noise.
- **The truncation safety net never fired** (§3.3) — this experiment does not test forced truncation/information loss, only "more raw context, still within budget." E9 tests the genuinely infeasible case directly.
- **QASPER's EM metric is close to uninformative** here (0-13% across all context-bearing conditions) — F1 is the load-bearing metric, consistent with the QASPER paper's own evaluation choices.
- **Evidence-containment recall is strict (exact substring)** — a looser fuzzy-match version might reveal that `summary_doc`'s "0% recall" actually captures the gist of most evidence snippets without the literal wording; not implemented here to keep the metric simple and comparable across E8/E9's shared containment-style design.
- **Same model in all roles**, same limitation as E5-E7.

---

## 6. Connection to Thesis

E8 extends E7's "more context isn't free" finding from a passage-count axis (10 vs 20 passages) to a raw-token-volume axis (papers ranging 2,000-8,300 tokens) in an entirely different domain:

| Finding | E7 (MuSiQue) | E8 (QASPER) |
|---|---|---|
| full_context hedges/dilutes with more context | Confirmed (verbose "According to..." answers, wrong-entity picks) | **Replicates in a new domain** — same hedging pattern, academic papers instead of trivia QA |
| full_context's E5/E6-era advantage erodes with scale | Confirmed (interactive_retrieval outscores it in several cells) | **Partially confirmed** — full_context never wins outright, but doesn't show a clean monotonic length-gradient at this n |
| Static summarization reliability | Task/dataset-dependent (E6: hurts compositional; E7: total collapse on one hop bucket) | **Dataset-dependent, lands on the "helps" side here** — best condition overall |

The practical takeaway sharpens further: the E5/E6 heuristic "static context beats interactive retrieval" was itself scale-dependent, and QASPER shows that erosion starting well before the context window is actually exceeded — `full_context` loses its edge as soon as there's enough content to hedge over, not only once truncation forces information loss. E9 tests what happens past that point entirely.

---

## 7. Next Steps

1. **Larger n** (30+/bucket) to resolve whether the length-gradient in `full_context` F1 is real-but-small or genuinely non-monotonic.
2. **A true forced-truncation condition on QASPER** (artificially cap at, say, 1,500 words even for papers that would fit) — would let the "before infeasible" (E8) and "after infeasible" (E9) regimes be compared on the same dataset rather than across two different ones.
3. **Fuzzy/paraphrase-aware evidence recall** — would clarify whether `summary_doc`'s near-zero verbatim recall masks substantial gist-level evidence preservation, which the current strict substring metric can't distinguish from genuine evidence loss.

---

*Run `python main.py` to replicate (resumes automatically from cached per-bucket results; `--fresh` forces a clean recompute).*
*Run `python analyze.py` to regenerate tables and the `plots/e8_length_degradation.png` chart.*
