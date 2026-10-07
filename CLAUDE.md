# Thesis Project

## Goals
Research: **model educability** | **knowledge transfer between models**

### Focus Areas
- **Memory Controller**: Knowledge base (RAG/alternatives) | Procedural memory | Structured representation
- **Educability**: Learning from models | Distillation | Teaching effectiveness metrics
- **Transfer Protocols**: Inter-agent communication | Persistent storage/retrieval

## Structure
```
papers/ → PDFs + summaries (.md + .json)
experiments/ → POCs (README.md, config.yaml, main.py, requirements.txt)
knowledge_base/ → Schemas, embeddings, ontologies, concepts/
src/ → Libraries (memory/, agents/, protocols/)
configs/ → Experiment configs
datasets/ → Training/evaluation data
results/ → Outputs & metrics
```

## Skills (see `.claude/skills/*.md`)
- `read-paper.md` → PDF ingestion, analysis, dual-format summaries, knowledge graph updates
- `run-experiment.md` → POC execution, metrics collection, result documentation
- `build-knowledge.md` → Schema creation, embedding generation, concept linking
- `transfer-knowledge.md` → Agent-to-agent knowledge protocols, distillation, evaluation

## Agents
- **Reader** → Paper analysis, summary generation
- **Experimenter** → POC implementation, benchmarking
- **Architect** → Knowledge schema design, ontology management
- **Teacher** → Knowledge distillation, transfer evaluation

## Conventions
- JSON/YAML for programmatic data; include embeddings for search
- Track provenance (source paper/experiment)
- Paths: `papers/` not `thesis/papers/`
- Communication: concise, direct, implementation-focused, flag issues early
