# Skill: Read and Summarize Research Papers (PDF)

## Trigger
When user provides a PDF paper or requests paper analysis.

## Workflow

### Phase 1: Ingest
1. **Initial scan**: Read pages 1-3 first (`pages: "1-3"`) to confirm this is the correct paper and identify its length
2. **Full read**: Read the complete PDF using the Read tool
   - Papers ≤ 20 pages: read in one call (no `pages` param needed)
   - Papers > 20 pages: read in chunks — `pages: "1-20"`, then `"21-40"`, etc. — the Read tool caps at 20 pages per call
3. **CRITICAL — Verify content was actually extracted**: The Read tool renders PDFs visually (multimodal). If the result is an error message or contains no paper text — no abstract, no authors, no section headings — **STOP** and inform the user: "PDF extraction failed. Please paste the PDF content as a document attachment in your next message." Do NOT proceed to synthesize a summary from zero content — that produces hallucinated summaries that look plausible but describe a completely different paper.
4. If content is successfully extracted: identify paper metadata (title, authors, venue, year, DOI/arxiv ID)
5. **Verify path handling**: Paths are relative to repo root (`papers/` not `thesis/papers/`)

### Phase 2: Analyze

**Initial Scan (first pass):**
- Abstract → problem statement & claimed contributions
- Introduction → motivation, context, summary of approach
- Conclusion → findings and future work
- Figures/diagrams → architecture, results visualization
- Skim section headings for structure

**Deep Read (second pass):**
- Method section → extract technical approach, architectures, algorithms
- Experiments → datasets, baselines, metrics, evaluation protocols
- Results → key numbers, statistical significance, ablation studies
- Limitations → acknowledged weaknesses
- References → identify foundational works and related research

**Critical Assessment:**
- What assumptions does this work make?
- Are the experiments sufficient to support the claims?
- What baselines were used? Are they appropriate?
- Is the code/data available and usable?
- What's novel vs. incremental?
- **Compute budget**: How much training/inference compute was used? Is this replicable?
- **Domain specificity**: Does this only work for math/games or generalizes to open-ended domains?

### Phase 3: Create Summary

**Location:** `papers/summaries/{paper_name}.md`

**Format:**

```markdown
# {Paper Title}

## Metadata
- **Authors:** {author list}
- **Venue:** {conference/journal}
- **Year:** {year}
- **Links:** [Paper]({url}) | [Code]({code_url}) | [Data]({data_url})
- **PDF:** `../papers/pdfs/{filename}.pdf`

## Summary

{Paragraph 1: Problem and approach}
Clear statement of what problem this paper addresses and the core idea of their solution.
Include the key insight that differentiates this from prior work.

{Paragraph 2: Results and contributions}
Main achievements: what they demonstrated, key metrics/improvements, what this enables.
Be specific about what was actually accomplished vs. claimed.

## Relevance to Thesis

{Connection to model educability, knowledge transfer, memory systems}
- How does this inform memory controller design?
- Does this enable knowledge transfer between models?
- What does this teach us about model educability?
- Is this a baseline we should compare against?

## Key Resources

| Type | Name | URL/Location | Notes |
|------|------|--------------|-------|
| Dataset | ... | ... | ... |
| Code | ... | ... | ... |
| Model | ... | ... | ... |

## Method Details

**Architecture:** {key components}
**Training:** {data, objectives, compute}
**Evaluation:** {benchmarks, metrics}

## Gaps & Open Questions

### Limitations (acknowledged)
- {what authors admit they didn't solve}

### Limitations (observed)
- {what you noticed they didn't address}
- {experimental gaps, missing baselines, narrow evaluation}

### Unexplored Directions
- {specific research questions this opens up}
- {combinations with other approaches}
- {scaling, generalization, different domains}
- {what to build as a POC based on this}

## Notes
- {additional observations, connections to other papers, implementation ideas}

## Citation
```bibtex
{full bibtex entry}
```
```

### Phase 4: Extract Structured Data

**Location:** `papers/summaries/{paper_name}.json`

**Format:**

```json
{
  "title": "",
  "authors": [],
  "venue": "",
  "year": "",
  "url": "",
  "arxiv_id": "",
  "doi": "",
  "code_url": "",
  "data_url": "",
  "problem_statement": "",
  "key_ideas": [],
  "method_summary": "",
  "datasets_used": [],
  "baselines": [],
  "metrics": [],
  "key_results": {},
  "contributions": [],
  "limitations_acknowledged": [],
  "limitations_observed": [],
  "relevance_to_thesis": "",
  "gaps_and_questions": [],
  "poc_ideas": [],
  "tags": [],
  "related_papers": [],
  "citations_count": null,
  "read_date": "",
  "confidence": "high|medium|low"
}
```

### Phase 5: Update Knowledge Graph

After summarizing:
1. Add entry to `knowledge_base/ontologies/knowledge_graph.json` linking this paper to:
   - Concepts it uses
   - Concepts it introduces
   - Related papers (citations and related work)
   - Potential POC experiments

2. If paper introduces key concepts, create `knowledge_base/schemas/concepts/{concept_name}.json`

3. **Tag for retrieval**: Add tags to enable semantic search later (e.g., #educability, #formal-methods, #value-model)

## Quality Checks

- [ ] Both paragraphs in summary are substantive (3-5 sentences each)
- [ ] All URLs are working and labeled
- [ ] Gaps section has at least 2 specific items
- [ ] Relevance section explicitly connects to thesis goals
- [ ] JSON summary is complete and valid
- [ ] BibTeX is properly formatted
- [ ] Path verification: Files in `papers/summaries/` (repo root relative)
- [ ] POC ideas are actionable (can describe experiment in <50 words)
- [ ] Distinguished between "what they claimed" vs "what they demonstrated"

## Tools Used

- `Read` - Parse PDF content
- `Write` - Create summary files
- `Bash` - Move files, check directories
- `Grep` - Search existing summaries for related work

## Example Invocation

```
User: "Read this paper: papers/pdfs/2403.12345.pdf"

Assistant actions:
1. Read the PDF
2. Analyze using workflow above
3. Create papers/summaries/2403.12345.md
4. Create papers/summaries/2403.12345.json
5. Report back with 2-sentence summary and key findings
```
