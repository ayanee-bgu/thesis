# VOYAGER: An Open-Ended Embodied Agent with Large Language Models

## Metadata
- **Authors:** Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi "Jim" Fan, Anima Anandkumar
- **Venue:** arXiv (NeurIPS 2023)
- **Year:** 2023
- **Links:** [Paper](https://arxiv.org/abs/2305.16291) | [Code](https://github.com/MineDojo/Voyager) | [Project](https://voyager.minedojo.org)
- **PDF:** `../pdfs/voyager_open_ended_agents_with_llms.pdf`

## Summary

VOYAGER addresses the problem of lifelong open-ended learning in embodied agents — specifically, how an agent can continuously acquire and accumulate skills without human intervention or model fine-tuning. Their key insight is that skills can be represented as executable code (JavaScript/Mineflayer API calls), stored in a vector database indexed by semantic embeddings, and retrieved compositionally to solve new tasks; combined with a GPT-4-driven automatic curriculum and an iterative feedback loop (environment feedback + execution errors + self-verification), the agent self-improves in a closed loop without ever updating model parameters.

VOYAGER demonstrated strong empirical results in Minecraft: it discovers 3.3× more unique items, traverses 2.3× longer distances, and unlocks tech tree milestones up to 15.3× faster than prior SOTA (ReAct, Reflexion, AutoGPT). Critically, the skill library is portable — AutoGPT equipped with VOYAGER's accumulated skill library also significantly improved, suggesting the skill library is a general plug-and-play knowledge asset, not just an artifact of VOYAGER's specific agent loop.

## Relevance to Thesis

VOYAGER is one of the most directly relevant papers to all three thesis focus areas:

- **Memory Controller**: The skill library is a concrete implementation of a procedural memory system — executable code indexed by semantic embeddings, enabling retrieval by similarity. It avoids catastrophic forgetting (a key continual learning failure mode) because memories are stored externally, not in weights. This is a strong baseline design for the memory controller's procedural memory component.
- **Knowledge Transfer**: The plug-and-play experiment (AutoGPT + VOYAGER skill library) is a direct demonstration of knowledge transfer between agents — the skill library built by one agent improves another. This is exactly the transfer protocol question: what representation enables skills to be reused across agents?
- **Educability**: VOYAGER learns without weight updates — all "learning" is via in-context retrieval. This raises a sharp question for the thesis: does educability require weight updates, or is accumulation of reusable external skills sufficient? VOYAGER represents the "no-weight-update" extreme of the educability spectrum.

Likely baseline for comparison in experiments on procedural memory and inter-agent transfer.

## Key Resources

| Type | Name | URL/Location | Notes |
|------|------|--------------|-------|
| Code | VOYAGER | https://github.com/MineDojo/Voyager | JavaScript + Python |
| Environment | MineDojo | https://github.com/MineDojo/MineDojo | Minecraft AI framework |
| Environment | Mineflayer | https://github.com/PrismarineJS/mineflayer | JS low-level Minecraft API |
| Model | GPT-4 (`gpt-4-0314`) | OpenAI API | Code generation + verification |
| Model | GPT-3.5 (`gpt-3.5-turbo-0301`) | OpenAI API | Embeddings, self-Q&A, budget tasks |
| Embeddings | `text-embedding-ada-002` | OpenAI API | Skill indexing |

## Method Details

**Architecture:**
- **Automatic Curriculum**: GPT-4 prompted with agent state (inventory, biome, health, position, previously completed/failed tasks) + self-generated Q&A (GPT-3.5). Goal: "discover as many diverse things as possible." Adapts bottom-up to agent's current capability frontier. Temperature 0.1 for task diversity.
- **Skill Library**: Vector DB where key = GPT-3.5 embedding of program description, value = JavaScript program. Retrieval: embed (task plan + environment feedback) → top-5 cosine-similar skills → inject as context for GPT-4 code generation. New skills added after self-verification confirms success.
- **Iterative Prompting Mechanism**: Up to 4 rounds per task. Each round: execute code → collect (a) environment feedback via `bot.chat()`, (b) execution errors from JS interpreter, (c) self-verification by a second GPT-4 instance acting as critic. If task fails after 4 rounds, curriculum proposes a different task.

**Training:** No gradient-based training. GPT-4/GPT-3.5 used as blackbox APIs (prompt-only). All "learning" is via accumulation in the external skill library.

**Evaluation:**
- *Exploration*: unique items discovered over 160 prompting iterations
- *Tech tree mastery*: iterations to unlock wooden/stone/iron/diamond tool tiers (3 runs each)
- *Map coverage*: total distance traversed
- *Zero-shot generalization*: success rate on 4 unseen tasks in a fresh world (50 iteration budget)
- Baselines: ReAct, Reflexion, AutoGPT (all re-implemented for MineDojo)

## Gaps & Open Questions

### Limitations (acknowledged)
- High API cost: GPT-4 is 15× more expensive than GPT-3.5; GPT-3.5 produces 5.7× fewer items, making cost a hard constraint for scaling
- Hallucinations: curriculum proposes non-existent items; code generation calls non-existent APIs
- Self-verification occasionally fails (e.g., misses spider string as zombie-kill evidence)
- Text-only: no visual perception; evaluation entirely in Minecraft

### Limitations (observed)
- Skill library has no forgetting or pruning mechanism — grows unboundedly; unclear how retrieval quality degrades at scale
- Zero-shot generalization tested on only 4 tasks, all craft-oriented; no evaluation on tasks requiring spatial reasoning or novel biomes
- No comparison against gradient-based continual learners (DreamerV3, VPT) on the same exploration metric — the "no-weight-update" framing sidesteps this comparison rather than engaging it
- Curriculum quality is entirely dependent on GPT-4's world knowledge of Minecraft; unclear if the approach generalizes to less well-documented domains

### Unexplored Directions
- **Cross-model skill transfer**: Does a skill library accumulated with GPT-4 transfer to Claude or Llama? If skill descriptions are natural language + code, there's no obvious reason it shouldn't work — but nobody has tested this
- **Domain generalization**: Can the same architecture work for research tasks, API workflows, or code assistance? Replace Mineflayer with tool-call APIs
- **Skill pruning / quality control**: When the library grows large, do redundant or incorrect skills degrade retrieval? A skill quality scoring mechanism (success rate, generalization count) could help
- **Educability measurement via skill acquisition rate**: The curriculum + skill library provides a natural educability metric — how many prompting iterations does a given base model need to unlock a fixed task tier? This could operationalize thesis metrics for model educability

## Notes
- The term "lifelong learning" here means in-context accumulation, not continual learning in the traditional sense. This distinction matters for the thesis: VOYAGER's approach doesn't update the base model, so it is fully "model-agnostic" — you can swap GPT-4 for any backend.
- Generative Agents [82] is the closest related work: stores experiences as memories, retrieves for planning. Key difference: VOYAGER's memories are *executable* (code), not natural language descriptions. This executable-vs-descriptive distinction is potentially important for the thesis's knowledge transfer protocol.
- The `bot.chat()` trick (injecting feedback as in-world chat messages) is an elegant low-overhead way to generate structured environment feedback without instrumenting the game — worth noting for POC design.

## Citation
```bibtex
@article{wang2023voyager,
  title={Voyager: An Open-Ended Embodied Agent with Large Language Models},
  author={Wang, Guanzhi and Xie, Yuqi and Jiang, Yunfan and Mandlekar, Ajay and Xiao, Chaowei and Zhu, Yuke and Fan, Linxi and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2305.16291},
  year={2023}
}
```
