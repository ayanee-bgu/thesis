# E2: Cross-Agent Knowledge Transfer via Shared Procedure Graph

## Research Question

Can Agent B reach Agent A's task performance faster by bootstrapping from A's session graph?
Does this generalise MemP's stronger→weaker transfer result to a live graph-based protocol?

## Protocol

**Phase 1 (Teacher):** Run `num_sessions` task sessions. After each session, consolidate
trajectories into a shared procedure graph (session-end batch update).

**Phase 2 (Transfer):** Export teacher's memory to student under one of three conditions.

**Phase 3 (Student):** Student runs tasks with transferred memory. Measure
`sessions_to_target` (sessions needed to reach 70% success rate) and `peak_success_rate`.

## Conditions

| Condition | Student Bootstrap | Description |
|---|---|---|
| `b_from_scratch` | None | Baseline — no prior knowledge |
| `b_from_a_flat` | Teacher's flat procedure dump | MemP-style export baseline |
| `b_from_a_graph` | Teacher's session graph | **Proposed**: graph-based transfer |

## Key Metric

**Sessions-to-target**: how many sessions does the student need to reach 70% success?
Lower = more efficient knowledge transfer.

## Setup

```bash
pip install -r experiments/memory_controller/requirements.txt
```

## Run

```bash
# Test without ALFWorld
python experiments/knowledge_transfer/main.py --mock

# Skip teacher if graph already built
python experiments/knowledge_transfer/main.py --mock --skip-teacher

# Single condition
python experiments/knowledge_transfer/main.py --mock --condition b_from_a_graph
```

## Expected Results

- `b_from_a_graph` reaches target in fewer sessions than `b_from_scratch`
- `b_from_a_graph` ≥ `b_from_a_flat` (graph retrieval is more precise than flat dump)
- Peak success rate: `b_from_a_graph` ≥ `b_from_a_flat` > `b_from_scratch`

## Connection to Thesis

Directly tests **model educability** and **knowledge transfer**:
- Is procedural knowledge model-agnostic? (transfer across agent boundaries)
- Does graph structure preserve more transferable knowledge than flat export?
- How many sessions does a student need when bootstrapped vs. cold-start?
