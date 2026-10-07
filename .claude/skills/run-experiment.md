# Skill: Run and Document Experiments

## Trigger
When user requests POC implementation, experiment execution, or benchmarking.

## Workflow

### Phase 1: Setup
1. Check if experiment exists in `experiments/{name}/`
2. If new: create directory with `README.md`, `config.yaml`, `main.py`, `requirements.txt`
3. Verify dependencies match root or are specified
4. Document hypothesis and success criteria

### Phase 2: Execute
1. Run experiment with config parameters
2. Capture metrics, logs, outputs to `results/{experiment_name}/{timestamp}/`
3. Track compute budget (tokens, time, memory)
4. Note failures and recovery attempts

### Phase 3: Document
1. Update `experiments/{name}/README.md` with results
2. Add metrics to `results/{experiment_name}/metrics.json`
3. Link to related papers in `papers/summaries/`
4. Tag with concepts for knowledge base

## Quality Checks
- [ ] Config is reproducible (all parameters logged)
- [ ] Results include raw outputs and aggregated metrics
- [ ] Compute costs documented
- [ ] Success criteria evaluated
