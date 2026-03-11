# Architecture blueprint for a nanochat cognition layer

## Thesis
The objective is to improve capability through **system structure** while preserving nanochat's existing strengths: minimalism, hackability, and end-to-end clarity.

This is an extension architecture, not a replacement architecture.

## Core design principle
Wrap the current nanochat stack with an optional cognition layer rather than rewriting the repo.

Preferred shape:
- existing model
- existing tokenizer
- existing `Engine`
- cognition controller around them

## Integration stance
The first version should live as a clearly separated subsystem, ideally under `nanochat/cognition/`.

That subsystem should be able to:
- call the current generation stack through an adapter
- retrieve and store memories
- route tasks into different modes
- explore candidates
- verify and rank candidates
- run lightweight branch experiments
- consolidate repeated wins into reusable skills

## Two capability layers (wrapper vs model-core)
The repo now has two distinct capability layers, and they serve different purposes:

1. **Cognition wrapper layer** (`nanochat/cognition/`)
   - An external controller around model inference.
   - Handles episodic/semantic memory, routing, candidate generation, verification, sandbox trials, and consolidation into reusable skills.
   - Produces inspectable traces at the system level.

2. **Model-core local deliberation layer** (`nanochat/gpt.py`)
   - An internal, optional latent module that runs inside the transformer stack.
   - Adds token-local recurrent refinement without changing tokenizer/chat formatting or requiring an external planner.

The model-core layer is a **lightweight graph-of-thought approximation** implemented in latent space, not a full explicit graph executor.

### Local deliberation layer in plain language
When enabled, selected transformer layers can run small internal "thinking" updates per token:
- **Token-local recurrent micro-steps:** each token representation is refined for a small number of internal passes.
- **Phrase pooling / phrase consensus:** nearby tokens are pooled into short phrase chunks so the layer can blend local agreement signals.
- **Optional semantic top-k neighbor path:** a sparse semantic neighbor lookup path can be enabled to mix in top-k latent neighbors when available.
- **Adaptive per-token gating:** token-level gates modulate how much deliberation state should influence each token.
- **Decode-time cache support:** the path is designed to remain compatible with incremental decoding/cache-aware inference behavior when used in decode mode.

This keeps the mechanism small and inspectable: it improves local latent coordination, but does not attempt to execute a full symbolic or tool-driven reasoning graph.

### Local deliberation config surface (first-class plumbing)
The local deliberation interface now exposes explicit model/train config keys for advanced options, even when behaviors are still reserved for later milestones:
- `local_delib_semantic_topk`, `local_delib_semantic_lookback`, `local_delib_use_neighbor_graph`, `local_delib_use_phrase_consensus`
- `local_delib_adaptive_halt`
- `local_delib_branch_factor`, `local_delib_branch_every`, `local_delib_branch_dim`, `local_delib_debug_branch_stats`
- `local_delib_hierarchy_chunk_sizes`
- `local_delib_scratch_slots`, `local_delib_scratch_dim`

Defaults keep current behavior unchanged (`0`/`False`/empty-string off states), and training/config wiring should only provide plumbing unless a milestone explicitly turns on implementation work for these options.

## Core components

### 1. Backend adapter
Purpose:
- provide a stable interface between the cognition subsystem and the existing nanochat generation path

Responsibilities:
- wrap current model + tokenizer + `Engine`
- expose generation in a controlled, testable way
- return structured metadata useful for traces

Current implementation note:
- the cognition layer now includes an `EngineBackend` adapter that can call the checkpoint-backed `Engine` directly for opt-in chat and eval paths
- prompt construction should carry reusable context explicitly through sections such as relevant episodic memory, semantic memory, and skills

### 2. Episodic memory
Stores concrete experiences such as:
- task or prompt
- context
- retrieved memories
- generated candidates
- chosen action
- outcomes
- scores
- reflections

Use it for:
- recalling similar prior episodes
- learning from success and failure history
- grounding later routing decisions

### 3. Semantic memory
Stores distilled and more durable abstractions such as:
- reusable procedures
- lessons learned
- concepts
- trigger heuristics
- skill definitions

Use it for:
- compact reuse
- reducing repeated rediscovery
- supporting later routing and verification

### 4. Router
Chooses among modes such as:
- direct answer
- retrieve memory
- creative exploration
- verification / planning
- sandbox experimentation
- consolidation trigger

The router must be explicit and inspectable.
It should return not just a choice, but also rationale.

### 5. Creative workspace
Purpose:
- generate multiple possible ideas, plans, or framings
- preserve option breadth before collapsing too early

Outputs may include:
- hypotheses
- plan sketches
- alternate framings
- candidate responses

### 6. Verifier workspace
Purpose:
- critique, rank, and optionally repair candidate outputs
- apply constraints and select the best current option

Outputs may include:
- critiques
- rankings
- repair instructions
- confidence estimates
- chosen candidate

### 7. Lightweight sandbox
Purpose:
- allow branch-and-score experimentation without requiring a full world model

Capabilities:
- try multiple candidate actions or plans
- score outcomes using simple heuristics or small task-specific scorers
- record failures and successes
- feed evidence back into episodic memory

### 8. Consolidator
Purpose:
- detect repeated successful patterns
- distill them into reusable skills or concepts

Potential signals:
- repeated success across related contexts
- stable action pattern
- improved outcome after reuse
- lower retries or repair count

### 9. Skill registry
Stores reusable artifacts with:
- identifier
- description
- trigger conditions
- provenance
- supporting evidence
- usage notes

### 10. Trace layer
Every major decision should be inspectable.
Track at minimum:
- router choice and rationale
- retrieved memories
- generated candidates
- verifier rankings
- sandbox outcomes
- consolidation events

## Recommended v1 implementation strategy
- keep storage simple first
- keep tests fake-backed and CPU-only where possible
- keep each module replaceable
- keep integration shallow until the end-to-end loop is proven
- favor explicit data structures over hidden magic

## What not to do in v1
- do not rewrite the main training loop
- do not make the speedrun path depend on cognition modules
- do not introduce heavy persistence or service dependencies too early
- do not hide routing and reasoning decisions in opaque abstractions

## Example end-to-end flow
1. Receive a task
2. Router inspects novelty, ambiguity, and risk
3. Retrieve relevant episodic and semantic memory if needed
4. Generate multiple candidates in the creative workspace if needed
5. Rank or repair them in the verifier workspace
6. Run a lightweight sandbox if uncertainty remains and the task warrants it
7. Return the selected answer or plan
8. Record the full episode
9. Consolidate repeated successful patterns into skills over time
10. Reuse those skills on future related tasks

## Incremental target architecture: full latent hierarchical graph-of-thought (optional path)

This section defines a **forward plan** for evolving the existing model-core local deliberation path into a fuller latent hierarchical graph-of-thought subsystem.

Design constraints for this direction:
- keep the existing wrapper cognition layer (`nanochat/cognition/*`) as-is and compatible
- keep default runtime behavior close to today unless explicitly enabled
- keep speedrun/core pretraining flow untouched by default
- introduce changes as opt-in, milestone-scoped increments

### A. Adaptive per-token halting / variable compute
Target behavior:
- each token maintains a halting probability and can stop local micro-steps earlier than neighbors
- global bounds still apply (`min_steps <= token_steps <= max_steps`) for stability and reproducibility

Repo-native integration pattern:
- implement as an extension of current token state scalars (salience/uncertainty/halt)
- preserve fixed-step fallback and current cache-compatible path
- emit inspectable stats: mean token steps, halt distribution, late-halt ratio

### B. Dynamic latent nearest-neighbor graph / flocking
Target behavior:
- at each local deliberation iteration, token latents form a dynamic top-k neighbor graph in latent space
- optional flocking operators apply local alignment/cohesion/separation updates before residual merge

Repo-native integration pattern:
- keep graph building local to the existing deliberation block instead of adding a new global runtime service
- support cheap approximate selection first (top-k over current window)
- keep this path switchable so disabled mode equals current behavior

### C. Latent branch spawning and merge
Target behavior:
- uncertain/high-conflict regions can spawn small latent branches (alternative local trajectories)
- branches compete and merge back using bounded policies (gated weighted merge or consensus merge)

Repo-native integration pattern:
- no symbolic tree executor; branches are latent tensors with strict per-layer caps
- decode-time cache includes branch metadata only when branching is enabled
- branch provenance exposed in debug metadata for downstream cognition traces

### D. Multi-scale hierarchy beyond phrase chunks
Target behavior:
- extend from token/phrase to deeper scales, e.g. token -> phrase -> span/segment -> sequence summary nodes
- allow bidirectional message passing across adjacent scales each micro-step

Repo-native integration pattern:
- reuse existing phrase concepts as the first hierarchy level
- add levels incrementally with enable flags and conservative defaults
- keep each scale independently disable-able for ablations and rollback

### E. Latent creative scratchpad slots
Target behavior:
- add a tiny bank of latent scratchpad slots for speculative composition and recombination
- slots can ingest selected token/phrase/branch summaries and write back refined guidance

Repo-native integration pattern:
- keep scratchpad internal to model-core deliberation state
- optionally export compact scratchpad summaries to wrapper cognition traces
- preserve current prompt-level `CreativeWorkspace` as orchestrator; scratchpad augments rather than replaces it

### F. Optional auxiliary losses and evaluations
Target behavior:
- define auxiliary objectives that improve the new mechanisms without forcing them into baseline training
- evaluate quality/compute tradeoffs and mechanism utility with lightweight, repeatable metrics

Candidate auxiliary objectives (all optional):
- halting calibration loss (align halt confidence with utility)
- graph consistency loss (stability of useful neighbors across steps)
- branch utility loss (reward branches that improve final token decisions)
- scratchpad contribution loss (penalize unused/noisy slots)

Current minimal implemented auxiliary path (all default disabled with weight `0.0`):
- `local_delib_halt_sparsity_loss = mean(halt_gate)`
- `local_delib_branch_diversity_loss = mean_step(mean_offdiag(cosine_similarity(branch_proposals)^2))`
- `local_delib_branch_entropy_loss = mean_step(log(branch_factor) - entropy(softmax(branch_logits)))`
- `local_delib_consensus_agreement_loss = mean_step((1 - mean_agreement_score) / 2)`
- `local_delib_scratch_utilization_loss = 1 - mean_step((mean(uncertainty) + mean(salience * uncertainty)) / 2)`

Training-side composition in `scripts/base_train.py`:
- base objective stays cross-entropy;
- when any configured auxiliary weight is non-zero and `model.last_aux_losses` is present, training adds
  `sum(weight_i * aux_loss_i)`;
- when all weights are zero (default), behavior is unchanged.

Evaluation additions:
- variable compute efficiency (quality per micro-step)
- branch usefulness rate (spawned vs merged-helpful branches)
- hierarchy utilization metrics (cross-scale message usage)
- scratchpad win-rate on creative/divergent prompts

## Staging strategy for this target
1. Add instrumentation-first hooks and docs-level toggles.
2. Introduce adaptive halting with fixed-step fallback.
3. Add neighbor graph + flocking as optional updates.
4. Add bounded branch spawn/merge.
5. Add deeper hierarchy levels.
6. Add scratchpad slots and wrapper trace surfacing.
7. Add optional auxiliary losses/evals with explicit ablations.

This preserves nanochat's incremental philosophy: each mechanism lands as a small optional extension, is measurable, and can be disabled without disturbing established paths.


## Local deliberation multi-scale hierarchy (optional)

The model-core local deliberation block now supports an optional hierarchy stack configured by `GPTConfig.local_delib_hierarchy_chunk_sizes` (comma-separated string such as `"4,16"`). When enabled, each hierarchy level performs:
- chunk pooling from token states to latent nodes,
- lightweight node refinement,
- causal broadcast back to tokens via per-level prefix summaries.

Per-step level feedback is averaged across levels and passed through a bounded nonlinearity before being merged with other local deliberation signals. When disabled (empty config string), existing phrase-chunk behavior remains unchanged.

Debug stats include:
- `hierarchy_levels_used`
- `mean_hierarchy_feedback_norm`
- `hierarchy_level_chunk_counts`

## Local deliberation latent scratchpad slots (optional)

`LocalDeliberationBlock` now supports an internal latent scratchpad configured by:
- `GPTConfig.local_delib_scratch_slots`
- `GPTConfig.local_delib_scratch_dim`

Behavior:
- when `local_delib_scratch_slots == 0`, scratchpad logic is fully disabled and existing paths are unchanged;
- when enabled, per-token latent states read from and write to a tiny scratch slot bank using uncertainty/salience-gated attention;
- scratch feedback is merged into the internal deliberation update input only (never decoded as text tokens);
- scratch readout projection is zero-initialized so initialization remains near-identity.

Debug stats include:
- `scratch_slots_used`
- `mean_scratch_read_weight`
- `mean_scratch_write_weight`
- `mean_scratch_norm`

Decode-cache note: local deliberation cache support remains the existing safe strategy (re-deliberate over cached latent state + new token prefix). Scratch slots are recomputed from the cached latent token state each forward call and are not separately persisted as an external token stream.
