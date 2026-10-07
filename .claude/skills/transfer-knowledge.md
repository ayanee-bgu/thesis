# Skill: Transfer Knowledge Between Agents

## Trigger
When implementing knowledge distillation, agent-to-agent teaching, or memory retrieval.

## Workflow

### Phase 1: Identify Transfer
1. Source: What knowledge to transfer? (trajectories, reasoning patterns, skills)
2. Target: Which agent receives knowledge?
3. Format: How is knowledge represented? (structured, embeddings, traces)

### Phase 2: Encode
1. Extract knowledge from source:
   - Reasoning traces → structured schemas
   - Successful trajectories → procedural memory
   - Skills → executable modules
2. Generate embeddings for retrieval
3. Validate completeness

### Phase 3: Transfer
1. Choose protocol:
   - Direct: Copy structured data
   - Distillation: Train target on source outputs
   - Retrieval: Enable target to query memory
2. Execute transfer
3. Measure fidelity

### Phase 4: Evaluate
1. Target performance on source tasks
2. Knowledge retention over time
3. Transfer efficiency (tokens, compute)
4. Generalization to new tasks

## Metrics
- Teaching effectiveness: Target improvement rate
- Knowledge fidelity: Output similarity
- Efficiency: Tokens saved vs. re-learning
- Retention: Performance decay over time

## Quality Checks
- [ ] Transfer protocol documented
- [ ] Before/after metrics captured
- [ ] Compute costs tracked
- [ ] Generalization tested
