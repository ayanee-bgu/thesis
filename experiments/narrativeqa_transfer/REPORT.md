# E9: NarrativeQA — When full_context Is Structurally Infeasible

**Date:** 2026-08-01
**Model:** llama3:8b (local, via Ollama, forced CPU — `num_gpu=0`, same fix carried forward from E5-E8)
**Experiment:** E7/E8 showed `full_context` degrades as context grows, even before hitting a hard limit. NarrativeQA tests the case where the limit is hit outright: documents (full books/movie scripts) run 60k-90k+ tokens, confirmed far beyond llama3:8b's 8,192-token window. `full_context` must be truncated to fit. Can `interactive_retrieval` — searching an index over the FULL untruncated document — reach answers that fall outside `full_context`'s truncated window and are therefore unreachable by construction?

---

## 1. Overview

**Three findings, in order of how much they complicate the story E5-E8 were building:**

1. **`summary_doc` wins overwhelmingly** (67.8% F1, 50% EM) — by a wide margin over both `full_context` (6.0% F1) and `interactive_retrieval` (15.8% F1). This condition uses NarrativeQA's own human-written summary rather than an LLM-generated one (the document is too long to summarize in one call), and a good static summary sidesteps every failure mode the other conditions hit.
2. **`full_context`'s failures are majority "visible but still wrong," not "truncated away."** Of 19/20 failures, 13 (68%) had the answer's content words present in the truncated text yet still got it wrong (the same hedging/confusion mechanism from E7 §3.4 and E8 §3.4) — only 6 (32%) were genuinely structurally unreachable.
3. **`interactive_retrieval` does NOT reliably recover the structurally-unreachable cases** — it recovered only 1 of 6 (17%). Digging into why surfaced a genuinely new failure mode not seen in E5-E8: **the query-generation step, given no anchor to the actual work's identity, hallucinates confident queries about a completely different, more famous work** (e.g. asking about "Cynthia Rhodes" and "The Picture of Dorian Gray" while researching the obscure 1601 Ben Jonson play *Cynthia's Revels*; asking about "Rocky IV" and "Mickey Rourke" while researching the boxing film *The Fighter*). This is upstream of the TF-IDF retrieval-precision bottleneck identified in E5-E7 — no amount of retrieval quality helps if the query itself targets the wrong fictional universe.

---

## 2. Design

Same 4 conditions and mechanisms as E5-E8, adapted for genuinely oversized documents:
- `full_context`: document text truncated to a 5,500-word budget (~7,000-7,500 estimated tokens) — always triggers here, unlike E8 where the defensive truncation cap never fired.
- `summary_doc`: uses `document.summary.text` (a human-written summary shipped with the dataset) directly — **no LLM summarization call**, since the document itself cannot be fed to an LLM for summarization in one call.
- `interactive_retrieval`: TF-IDF retrieval (K=3) over ~200-word chunks spanning the FULL document (~450 chunks for a 90k-word novel).
- Recall metric: **content-word-overlap containment** (≥50% of a gold answer's content words present in what the model was given), not exact-substring — NarrativeQA's reference answers are annotator-composed paraphrases (e.g. "THE ACTOR WEARING THE BLACK CLOAK"), not verbatim quotes, so exact-substring matching returned 0% for every condition in the initial smoke test, including `full_context` which had the text right there. This was caught and fixed before the full run (see §5).

n=10 questions per document `kind` (`gutenberg` = novels, `movie` = scripts), 20 total. `model.timeout` = 600s.

---

## 3. Results

### 3.1 Overall (n=20/condition)

| Condition | EM | F1 | Tokens | Containment |
|---|---|---|---|---|
| no_context | 0.0% | 2.5% | 29 | 0.0% |
| full_context | 0.0% | 6.0% | 4,120 | 65.0% |
| **summary_doc** | **50.0%** | **67.8%** | 30 | 100.0% |
| interactive_retrieval | 10.0% | 15.8% | 2,317 | 40.0% |

`summary_doc` is not just the best condition — it's in a different tier entirely, at a fraction of the token cost of `full_context` (30 vs 4,120 tokens) since it skips both the summarization call and the need to process a long truncated document at all.

### 3.2 By document kind

| Kind | full_context F1 | full_context Containment | summary_doc F1 |
|---|---|---|---|
| gutenberg (novels) | 8.2% | 40.0% | 64.3% |
| movie (scripts) | 3.7% | 90.0% | 71.4% |

Containment diverges sharply by kind — `full_context` sees a much smaller fraction of a novel's needed content (40%) than a movie script's (90%), consistent with novels typically being far longer relative to the 5,500-word truncation budget than scripts. Despite `full_context` seeing MORE of the relevant content on movies (90% containment), its F1 is actually *lower* on movies than novels (3.7% vs 8.2%) — reinforcing that containment and answer correctness are not the same thing (echoing E8 §4.5).

### 3.3 Decomposing full_context's failures: structural vs visible-but-wrong

| Failure type | n | Share |
|---|---|---|
| Structurally unreachable (containment=0 — answer truncated away) | 6 | 32% |
| Visible but still wrong (containment=1 — evidence present, answer still wrong) | 13 | 68% |

Both failure modes are real, but the majority is NOT truncation — it's the same "sees the evidence, still gets it wrong" pattern first identified in E7 §3.4 (MuSiQue, distractor confusion) and replicated in E8 §3.4 (QASPER, hedging preambles). At the scale of a 5,500-word truncated document, `full_context` is failing mostly because it can't use what it has, not because it doesn't have enough.

### 3.4 interactive_retrieval only weakly recovers the structurally-unreachable cases

Of the 6 structurally-unreachable `full_context` failures, `interactive_retrieval` recovered (F1 ≥ 0.3) only **1 (17%)**. This is well below what the "searches the full document" theoretical advantage would predict — `interactive_retrieval`'s own recall bottleneck (confirmed across E5-E8: it only found the needed evidence in 40% of all cases here) limits how much it can actually capitalize on having index access to content `full_context` structurally cannot see.

### 3.5 A new failure mode: query-generation hallucinates the wrong fictional universe

Investigating why recovery was so weak surfaced something not seen in any of E5-E8. Query sequences for *Cynthia's Revels* (a 1601 Ben Jonson play — obscure by modern standards):

```
"Cynthia Rhodes actress filmography"        <- confuses the play's title character
                                                 with an unrelated 1980s actress
"Vices in The Picture of Dorian Gray"       <- the play's vices/virtues theme
                                                 triggers a completely different, far more
                                                 famous novel
"Cynthia's sentence in American Gods"       <- "sentence" (the goddess's judgment)
                                                 triggers a modern fantasy novel
"Shakespeare's Court Jester character"      <- generic Renaissance-drama association,
                                                 not this specific non-Shakespeare play
```

And for *The Fighter* (2010, about boxer Micky Ward — a real but less iconic boxing film than the Rocky franchise):

```
"Rocky IV Atlantic City fight"              <- same director/genre confused for
                                                 a much more famous franchise
"Mickey Rourke eye injury film"             <- protagonist "Micky Ward" confused
                                                 with actor "Mickey Rourke"
"Rocky V manager name"                       <- same Rocky-franchise substitution
```

The query-generation prompt gives the model only the question text — never the document's actual title — so for any work obscure enough that llama3:8b has no strong prior, it fills the gap with surface-level keyword association to a more famous work sharing incidental vocabulary (a name, a theme, a genre). TF-IDF retrieval then searches the *real* document with queries about the *wrong* one, returning irrelevant chunks — the resulting wrong answers (e.g. "ARCHIE" for "who challenges the courtiers to a compliment competition") are essentially hallucinated from the wrong universe entirely. **This is upstream of and distinct from the retrieval-precision bottleneck driving E5-E8's central finding** — no amount of better TF-IDF matching fixes a query that's about the wrong story.

`full_context` and `summary_doc` don't hit this failure mode because they're grounded in the real document/summary text from their very first token — there's nothing to hallucinate around. This is a design gap specific to `interactive_retrieval`'s query-generation step, not a fundamental limitation of interactivity itself.

---

## 4. Conclusions

1. **The clearest, strongest finding of this experiment**: a good static summary (here, human-written, but the mechanism should hold for a sufficiently good LLM-generated one too) is dramatically better than either alternative when the source document is too large to process directly. This is the sharpest version yet of E6-E8's recurring "summarization quality matters more than the injection mechanism" theme.
2. **`full_context`'s failure under infeasibility is mostly not about truncation** — most of its failures happen even when the relevant content survived truncation, continuing the hedging/dilution pattern from E7-E8 rather than introducing a new one.
3. **`interactive_retrieval`'s "sees the full document" advantage is real but weak in practice** (17% recovery rate on structurally-unreachable cases) — both because of its established recall bottleneck (E5-E8) and because of the newly identified query-generation hallucination problem specific to under-known documents.
4. **The E5-E7 "interactive querying beats static context under certain conditions" thesis does not hold here** — this is the first experiment in the E5-E9 series where `interactive_retrieval` is clearly and unambiguously dominated by a static condition on every metric. The conditions that made interactive retrieval competitive in E7 (large context volume causing static-condition confusion) are present here too, but `summary_doc`'s quality ceiling this time is high enough (human-written) to beat both alternatives outright.

---

## 5. Limitations

- **n=10/kind (20 total) is small** — single-item flips are 10pp EM/5pp F1. The §3.3/§3.4 structural-vs-visible breakdown (6 vs 13 cases) is illustrative, not statistically robust.
- **`summary_doc` uses a human-written summary, not an LLM-generated one** — this was the right call for feasibility (the document can't be fed to an LLM for summarization in one call) but means this condition is not a clean comparison to E5-E8's LLM-summarized `summary_doc`. The finding "good summaries beat both alternatives" is well-supported; "LLM-generated summaries would do the same" is not tested here.
- **The containment metric was redesigned mid-experiment** (§2, exact-substring → content-word-overlap) after the initial smoke test showed the exact-substring version was uninformative for NarrativeQA's paraphrased answers. This was caught and fixed before the full run, but is a reminder that metric design needs to be re-validated per dataset, not assumed to transfer from prior experiments' schemas.
- **The query-generation hallucination finding (§3.5) is observational**, from a handful of qualitative examples, not a systematically measured rate across the full sample. A follow-up could measure it directly (e.g., check whether generated queries share more vocabulary with the actual document's chunks vs. with a general knowledge base) rather than relying on manual inspection.
- **Same model in all roles**, same limitation as E5-E8.

---

## 6. Connection to Thesis

E9 completes the "how much context is too much" arc started in E7 and continued in E8, and delivers the sharpest qualification yet of the E5-E7 "interactive retrieval can beat static context" narrative:

| Finding | E7/E8 | E9 |
|---|---|---|
| full_context degrades with scale | Confirmed, progressively | Confirmed, but MOSTLY not from truncation — the hedging/confusion mechanism dominates even under forced truncation |
| interactive_retrieval can beat full_context | Yes, in several E7 cells | **Yes here too** (15.8% vs 6.0% F1) — but neither approaches `summary_doc` |
| Static summarization is a reliable win | Task-dependent (E6: hurts; E7: mixed; E8: helps) | **Decisively helps**, when the summary itself is high-quality — the strongest version of this finding across E5-E9 |
| Interactive retrieval's unique structural advantage (reaching truncated-away content) | Not previously testable | **Real but weak** (17% recovery) — capped by its own recall bottleneck AND a newly identified query-generation grounding gap |

The overall arc across E5-E9: interactive retrieval's advantage over static context is real under specific conditions (large-enough raw context volume to cause static-condition confusion, as in E7 and partially E9), but it is never a free win — it is always bounded by retrieval recall, and E9 adds a second bound specific to under-known documents: the query-generation step itself needs grounding in what it's actually searching, or it will confidently search the wrong thing.

---

## 7. Next Steps

1. **Ground the query-generation prompt in the document's actual title/identity** (directly motivated by §3.5) — a one-line fix (include the book/movie title in the query-generation system prompt) that should eliminate the wrong-universe hallucination and is the most actionable, highest-expected-value fix identified across the entire E5-E9 series.
2. **Test LLM-generated (not human-written) summaries via hierarchical summarization** (chunk → per-chunk summary → summary-of-summaries) to see whether `summary_doc`'s dominance here depends on human-level summary quality or would hold with a realistic LLM-generated equivalent.
3. **Larger n** (30+) to firm up the §3.3/§3.4 structural-failure breakdown.
4. **Systematically measure the query-generation grounding problem** (§5) rather than relying on qualitative inspection — e.g. score each generated query's vocabulary overlap with the actual document vs. a general corpus, to quantify how often and how badly this occurs.

---

*Run `python main.py` to replicate (resumes automatically from cached per-kind results; `--fresh` forces a clean recompute).*
*Run `python analyze.py` to regenerate tables and the `plots/e9_infeasible_context.png` chart.*
