# Skill: Build Knowledge Base Entries

## Trigger
When creating structured knowledge from papers, experiments, or concepts.

## Workflow

### Phase 1: Extract
1. Identify concept/entity from source (paper/experiment)
2. Determine type: concept, method, dataset, metric, or tool
3. Extract relationships to existing knowledge

### Phase 2: Structure
1. Create entry in `knowledge_base/schemas/concepts/{name}.json`:
   ```json
   {
     "name": "",
     "type": "",
     "definition": "",
     "related_concepts": [],
     "source_papers": [],
     "source_experiments": [],
     "embeddings": {"model": "", "vector": []},
     "tags": []
   }
   ```
2. Generate embedding for semantic search
3. Add tags for retrieval

### Phase 3: Link
1. Update `knowledge_base/ontologies/knowledge_graph.json`:
   - Add nodes for new concepts
   - Create edges for relationships (uses, introduces, extends, related_to)
2. Cross-reference with existing entries
3. Track provenance

## Quality Checks
- [ ] Entry has unique identifier
- [ ] Embedding generated and stored
- [ ] Relationships documented
- [ ] Tags enable discoverability
