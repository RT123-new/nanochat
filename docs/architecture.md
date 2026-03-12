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
- **Optional explicit flocking terms:** the causal neighbor graph can also add bounded alignment, cohesion, and separation updates over the same local latent neighborhood.
- **Optional explicit thought nodes:** bounded latent thought nodes can be built from causal token chunks, updated through a small causal top-k graph loop, and read back into token states.
- **Adaptive per-token gating:** token-level gates modulate how much deliberation state should influence each token.
- **Decode-time cache support:** the path is designed to remain compatible with incremental decoding/cache-aware inference behavior when used in decode mode.

This keeps the mechanism small and inspectable: it improves local latent coordination, but does not attempt to execute a full symbolic or tool-driven reasoning graph.

### Local deliberation config surface (first-class plumbing)
The local deliberation interface now exposes explicit model/train config keys for advanced options, even when behaviors are still reserved for later milestones:
- `local_delib_semantic_topk`, `local_delib_semantic_lookback`, `local_delib_use_neighbor_graph`, `local_delib_use_phrase_consensus`
- `local_delib_use_flocking`, `local_delib_flocking_alignment_weight`, `local_delib_flocking_cohesion_weight`, `local_delib_flocking_separation_weight`, `local_delib_flocking_separation_margin`, `local_delib_flocking_radius_cap`
- `local_delib_adaptive_halt`
- `local_delib_branch_factor`, `local_delib_branch_every`, `local_delib_branch_dim`, `local_delib_branch_consensus`, `local_delib_branch_verifier`, `local_delib_branch_consensus_temp`, `local_delib_branch_max_active`, `local_delib_branch_disagreement_threshold`, `local_delib_debug_branch_stats`
- `local_delib_hierarchy_chunk_sizes`
- `local_delib_use_deep_hierarchy`, `local_delib_span_chunk_size`, `local_delib_sequence_summary`, `local_delib_hierarchy_bidirectional`, `local_delib_hierarchy_scale_gate`
- `local_delib_scratch_slots`, `local_delib_scratch_dim`, `local_delib_scratch_refine_steps`, `local_delib_scratch_use_branch_inputs`, `local_delib_scratch_use_hierarchy_inputs`, `local_delib_scratch_export_summary`, `local_delib_scratch_summary_dim`
- `local_delib_use_thought_graph`, `local_delib_thought_node_budget`, `local_delib_thought_node_dim`, `local_delib_thought_graph_steps`, `local_delib_thought_topk_edges`, `local_delib_thought_token_chunk_size`, `local_delib_thought_use_branch_inputs`, `local_delib_thought_use_hierarchy_inputs`, `local_delib_thought_use_scratch_inputs`
- `local_delib_global_anchor_count`, `local_delib_global_anchor_dim`, `local_delib_global_anchor_update`, `local_delib_global_anchor_temp`, `local_delib_global_anchor_use_hierarchy`, `local_delib_global_anchor_use_scratch`, `local_delib_global_anchor_use_thought`

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

### Wrapper trace surfacing for model-core summaries
The cognition wrapper now preserves the existing `model_local_delib.*` metadata and can additionally surface compact advanced-mechanism summaries under:
- `model_local_delib.thought_summaries.branch_consensus`
- `model_local_delib.thought_summaries.deep_hierarchy`
- `model_local_delib.thought_summaries.scratch`
- `model_local_delib.thought_summaries.thought_graph`
- `model_local_delib.thought_summaries.global_anchors`
- `model_local_delib.thought_summaries.flocking`

Design constraints:
- wrapper metadata stays namespaced and additive; generation return contracts do not change;
- summaries are compact numeric/statistical payloads only, not raw latent tensors;
- summary keys only appear when the underlying mechanism actually surfaces non-zero or exported summary data;
- the older aggregate keys such as `model_local_delib.branch`, `model_local_delib.hierarchy`, `model_local_delib.scratchpad`, `model_local_delib.adaptive_halt`, and `model_local_delib.scratchpad_summaries` remain available for backward compatibility.

## Prompt 12 hardening snapshot
Implemented and stable enough for targeted experiments:
- local recurrent deliberation with optional adaptive halting
- causal neighbor graph mixing with optional flocking
- bounded branch spawn plus optional branch-consensus / verifier merge
- legacy hierarchy stack plus opt-in deep phrase/span/sequence hierarchy
- persistent scratch slots with optional summary export
- explicit latent thought nodes with bounded message passing
- bounded global anchor bank
- Prompt 8 structured decode cache for the bounded scratch/anchor/hierarchy path
- Prompt 10 ablation harness and Prompt 11 wrapper-level compact summary surfacing

Still experimental:
- engine-backed variant hot-swapping in evals depends on backend support
- explicit thought-graph decode continuation still falls back to full local-deliberation recompute for correctness
- Prompt 10 quality and quality-per-compute numbers remain heuristic rather than benchmark-grade metrics
- broader non-Prompt-12 legacy local-deliberation pytest debt may still exist outside the targeted hardening slice

Enable major model-core features through `GPTConfig` or `scripts/base_train.py` flags:
- base local deliberation: `local_delib`, `local_delib_steps`
- adaptive compute: `local_delib_adaptive_halt`
- neighbor graph / flocking: `local_delib_use_neighbor_graph`, `local_delib_use_flocking`, and flocking weights
- branching: `local_delib_branch_factor`, `local_delib_branch_consensus`, `local_delib_branch_verifier`
- deep hierarchy: `local_delib_use_deep_hierarchy`, `local_delib_span_chunk_size`, `local_delib_sequence_summary`
- scratch: `local_delib_scratch_slots`, `local_delib_scratch_refine_steps`
- thought graph: `local_delib_use_thought_graph`, `local_delib_thought_node_budget`, `local_delib_thought_graph_steps`
- global anchors: `local_delib_global_anchor_count`, `local_delib_global_anchor_update`

Rollback guidance:
- disable the whole path by leaving `local_delib=False` or `local_delib_steps=0`
- disable individual mechanisms by leaving their booleans off, keeping counts at `0`, or keeping auxiliary weights at `0.0`
- prefer the lightweight `local-delib-ablation` suite first, then widen to `local-delib-ablation-advanced`

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

Current implementation note:
- the causal neighbor graph path is implemented and can now optionally emit explicit alignment/cohesion/separation feedback terms, with all flocking weights defaulting to `0.0` and the feature remaining off by default

### C. Latent branch spawning and merge
Target behavior:
- uncertain/high-conflict regions can spawn small latent branches (alternative local trajectories)
- branches compete and merge back using bounded policies (gated weighted merge or consensus merge)

Repo-native integration pattern:
- no symbolic tree executor; branches are latent tensors with strict per-layer caps
- decode-time cache includes branch metadata only when branching is enabled
- branch provenance exposed in debug metadata for downstream cognition traces

Current implementation note:
- the branch path still begins with bounded per-token latent proposals and a simple branch scorer;
- when `local_delib_branch_consensus=True`, the block computes pairwise latent agreement/disagreement between active branches, forms a lightweight consensus summary, and only lets that consensus influence merge when disagreement clears `local_delib_branch_disagreement_threshold`;
- when `local_delib_branch_verifier=True`, proposals receive an additional verifier-style latent rescore conditioned on token state plus the consensus/summary context before final branch weighting;
- `local_delib_branch_max_active` caps how many proposals participate in consensus/verifier comparisons, so pairwise compute stays bounded even when `branch_factor` is larger;
- Prompt 8 keeps branch metadata decode-cache bounded by reusing cached per-step token states instead of persisting a separate long-lived branch tree, because branch merge remains token-local once the prefix stage states are fixed.

Prompt 3 debug stats now include:
- `mean_branch_disagreement`
- `mean_branch_consensus_weight`
- `mean_branch_verifier_score`
- `mean_branch_entropy`
- `branch_consensus_used`

### D. Explicit latent thought graph runtime
Target behavior:
- promote the earlier local approximation into a bounded explicit latent graph with first-class thought nodes and stable graph-update steps
- let tokens write into those nodes, let nodes exchange causal top-k messages, and let later tokens read the updated node state back out

Repo-native integration pattern:
- thought nodes stay latent tensors, never symbolic text
- node count is capped by `local_delib_thought_node_budget`
- token-group chunking keeps node construction local and causal
- optional branch, hierarchy, and scratch summaries can be folded into node construction without changing the disabled path

Current implementation note:
- when `local_delib_use_thought_graph=True`, `LocalDeliberationBlock` now builds explicit thought nodes from causal token chunks and optional branch/hierarchy/scratch summaries;
- each thought node receives a token-write summary from its assigned chunk, then runs a bounded number of causal top-k message-passing steps;
- tokens can only read thought nodes whose anchor chunk has already completed, so token-prefix causality is preserved;
- Prompt 8 now stores bounded per-step decode-cache metadata for thought-node windows during prefill, but decode continuation still falls back to full local-deliberation recompute when the thought graph is enabled so graph-step semantics stay exact.

Prompt 4 debug stats now include:
- `thought_nodes_used`
- `mean_thought_degree`
- `mean_token_to_thought_weight`
- `mean_thought_to_token_weight`
- `mean_thought_update_norm`
- `thought_graph_steps_used`

### E. Multi-scale hierarchy beyond phrase chunks
Target behavior:
- extend from token/phrase to deeper scales, e.g. token -> phrase -> span/segment -> sequence summary nodes
- allow bidirectional message passing across adjacent scales each micro-step

Repo-native integration pattern:
- reuse existing phrase concepts as the first hierarchy level
- add levels incrementally with enable flags and conservative defaults
- keep each scale independently disable-able for ablations and rollback

### F. Latent creative scratchpad slots
Target behavior:
- add a tiny bank of latent scratchpad slots for speculative composition and recombination
- slots can ingest selected token/phrase/branch summaries and write back refined guidance

Repo-native integration pattern:
- keep scratchpad internal to model-core deliberation state
- optionally export compact scratchpad summaries to wrapper cognition traces
- preserve current prompt-level `CreativeWorkspace` as orchestrator; scratchpad augments rather than replaces it

### G. Global memory anchors
Target behavior:
- maintain a tiny bank of global latent anchor states that summarize broader prefix context without introducing full global attention
- let tokens read from those anchors each micro-step and optionally let anchors absorb hierarchy, scratch, or thought summaries

Repo-native integration pattern:
- keep the anchor bank small and fully optional
- preserve causality by only exposing prefix-complete anchor state to each token
- keep decode-time anchor state bounded to a tiny per-step anchor-bank cache rather than a separate unbounded token stream

Current implementation note:
- when `local_delib_global_anchor_count > 0`, `LocalDeliberationBlock` allocates a bounded anchor bank initialized from learned anchor states and maintains a causal per-prefix anchor state across micro-steps;
- tokens read anchors through a small attention-style lookup and write back only when `local_delib_global_anchor_update=True`, using causal prefix summaries plus optional hierarchy/scratch/thought summaries when those toggles are enabled;
- anchor feedback is merged into the internal deliberation update input only, and the anchor-to-token projection is zero-initialized so enabled-at-init behavior remains near-identity until the path learns or is explicitly configured.

Prompt 7 debug stats now include:
- `global_anchors_used`
- `mean_anchor_read_weight`
- `mean_anchor_write_weight`
- `mean_anchor_norm`

### H. Optional auxiliary losses and evaluations
Target behavior:
- define auxiliary objectives that improve the new mechanisms without forcing them into baseline training
- evaluate quality/compute tradeoffs and mechanism utility with lightweight, repeatable metrics

Candidate auxiliary objectives (all optional):
- halting calibration loss (align halt confidence with utility)
- graph consistency loss (stability of useful neighbors across steps)
- branch utility loss (reward branches that improve final token decisions)
- scratchpad contribution loss (penalize unused/noisy slots)

Current implemented auxiliary path (all default disabled with weight `0.0` and surfaced via `model.last_aux_losses`):
- `local_delib_halt_sparsity_loss = mean(halt_gate)`
- `local_delib_branch_diversity_loss = mean_step(mean_offdiag(cosine_similarity(branch_proposals)^2))`
- `local_delib_branch_entropy_loss = mean_step(log(branch_factor) - entropy(softmax(branch_logits)))`
- `local_delib_consensus_agreement_loss = mean_step((1 - mean_agreement_score) / 2)`
- `local_delib_scratch_utilization_loss = 1 - mean_step((mean(uncertainty) + mean(salience * uncertainty)) / 2)`
- `local_delib_flocking_stability_loss = mean_step(mean_token(((||a|| + ||c|| + ||s|| - ||a + c + s||)_+) / (||a|| + ||c|| + ||s|| + eps)))`
  where `a/c/s` are the alignment/cohesion/separation flocking deltas for the token.
- `local_delib_thought_edge_stability_loss = mean_step>1(0.5 * mean(abs(A_t - A_{t-1})))`
  where `A_t` is the bounded dense top-k thought-node adjacency reconstructed from step-`t` edge weights.
- `local_delib_thought_node_utilization_loss = 1 - mean_step((mean(token_to_thought_gate) + mean(thought_to_token_read_weight)) / 2)`
- `local_delib_hierarchy_agreement_loss = mean_step(((1 - mean_scale_gate) + mean_adjacent((1 - cosine(scale_i, scale_{i+1})) / 2)) / 2)`
- `local_delib_branch_usefulness_loss = mean_step(1 - mean(branch_support * branch_novelty))`
  where `branch_support` is the effective branch-score support (averaged with verifier support when enabled) and
  `branch_novelty = (1 - cosine(branch_summary, pre_merge_state)) / 2`.
- `local_delib_anchor_usage_loss = mean_step(1 - anchor_usage)`
  where `anchor_usage` is mean anchor read weight, or the mean of read/write weights when anchor updates are enabled.

Mechanism-disabled behavior:
- flocking/thought/hierarchy/branch/anchor Prompt 9 losses return `0` when the corresponding mechanism is not active in that block
- the older scratch utilization loss remains the original heuristic path, so it can still report a non-zero proxy even when scratch is disabled

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
5. Add bounded explicit thought nodes and causal graph message passing.
6. Add deeper hierarchy levels.
7. Add scratchpad slots and wrapper trace surfacing.
8. Add global anchor bank and long-range summary feedback.
9. Add optional auxiliary losses/evals with explicit ablations.

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

## Local deliberation deep hierarchy (optional)

Prompt 5 adds a second, more explicit hierarchy path alongside the legacy `local_delib_hierarchy_chunk_sizes` stack. The legacy path remains available as-is; the new path is opt-in and keeps causal behavior explicit.

Deep hierarchy config:
- `GPTConfig.local_delib_use_deep_hierarchy`
- `GPTConfig.local_delib_span_chunk_size`
- `GPTConfig.local_delib_sequence_summary`
- `GPTConfig.local_delib_hierarchy_bidirectional`
- `GPTConfig.local_delib_hierarchy_scale_gate`

Behavior:
- phrase scale always uses the existing `local_delib_phrase_chunk_size` as the first explicit level
- optional span scale adds a coarser causal chunk-prefix summary (`0` disables the span level)
- optional sequence-summary scale adds a causal whole-prefix summary
- upward message passing runs token -> phrase -> span -> sequence when the corresponding scales are enabled
- optional downward message passing runs back across the enabled adjacent scales each micro-step
- optional per-scale gates can attenuate each scale's token-facing contribution without changing disabled behavior

This differs from the older hierarchy chunk stack in two ways:
- it treats phrase/span/sequence as named scales instead of an anonymous list of chunk sizes
- all summaries are built from causal prefix means, so earlier tokens remain stable when later tokens change

Debug stats include:
- `phrase_nodes_used`
- `span_nodes_used`
- `sequence_summary_used`
- `mean_upward_message_norm`
- `mean_downward_message_norm`
- `mean_scale_gate`
- `hierarchy_depth_used`

## Local deliberation latent scratchpad slots (optional)

`LocalDeliberationBlock` now supports an internal latent scratchpad configured by:
- `GPTConfig.local_delib_scratch_slots`
- `GPTConfig.local_delib_scratch_dim`
- `GPTConfig.local_delib_scratch_refine_steps`
- `GPTConfig.local_delib_scratch_use_branch_inputs`
- `GPTConfig.local_delib_scratch_use_hierarchy_inputs`
- `GPTConfig.local_delib_scratch_export_summary`
- `GPTConfig.local_delib_scratch_summary_dim`

Behavior:
- when `local_delib_scratch_slots == 0`, scratchpad logic is fully disabled and existing paths are unchanged;
- when enabled, per-token latent states read from and write to a tiny scratch slot bank using uncertainty/salience-gated attention;
- scratch now persists across local-deliberation micro-steps through causal per-prefix scratch state rather than resetting inside each micro-step, so the workspace can accumulate speculative composition without letting future tokens leak backward;
- optional scratch refinement runs a bounded slot-only update after each token write;
- optional branch and hierarchy summaries can be projected into scratch writes, but those paths remain disabled by default;
- scratch feedback is merged into the internal deliberation update input only (never decoded as text tokens);
- optional summary export produces a compact per-layer scratch summary vector for debug metadata and wrapper traces only when explicitly enabled;
- scratch readout projection is zero-initialized so initialization remains near-identity;
- scratch always resets at the start of each new forward/request, while Prompt 8 decode-cache now persists only the bounded per-step scratch-slot bank needed to continue the next token.

Debug stats include:
- `scratch_slots_used`
- `mean_scratch_read_weight`
- `mean_scratch_write_weight`
- `mean_scratch_norm`
- `mean_scratch_refine_norm`
- `mean_scratch_summary_norm`
- `mean_branch_to_scratch_weight`
- `mean_hierarchy_to_scratch_weight`
- `scratch_reset_ok`
- optional `scratch_summary_vector`

Decode-cache note: Prompt 8 now stores structured per-step local-deliberation cache state:
- cached micro-step token states (`stage_states`) for the prefix
- bounded step caches for hierarchy summaries, scratch slots, and global anchors
- optional thought-node window metadata captured during prefill

Decode continuation uses those caches for the bounded scratch/anchor-oriented path and batch-expands them safely when a batch-1 prefill cache is cloned into a larger decode batch. Thought-graph decode still takes the correctness-first fallback to full local-deliberation recompute.

## Local deliberation explicit latent thought graph (optional)

`LocalDeliberationBlock` now also supports a bounded explicit latent thought graph configured by:
- `GPTConfig.local_delib_use_thought_graph`
- `GPTConfig.local_delib_thought_node_budget`
- `GPTConfig.local_delib_thought_node_dim`
- `GPTConfig.local_delib_thought_graph_steps`
- `GPTConfig.local_delib_thought_topk_edges`
- `GPTConfig.local_delib_thought_token_chunk_size`
- `GPTConfig.local_delib_thought_use_branch_inputs`
- `GPTConfig.local_delib_thought_use_hierarchy_inputs`
- `GPTConfig.local_delib_thought_use_scratch_inputs`

Behavior:
- when `local_delib_use_thought_graph` is false, no thought-node modules are instantiated into the local deliberation update path and behavior is unchanged;
- when enabled, the block builds a bounded set of explicit thought nodes from causal token chunks, optionally folding in branch, hierarchy, and scratch summaries that already exist inside the same micro-step;
- tokens write into their assigned chunk node, thought nodes run a small causal top-k message-passing loop, and tokens only read back nodes whose anchor chunk is already complete;
- this is more explicit than the earlier local approximation because the runtime now has named node state, node-to-node edges, token-to-node writes, and node-to-token reads rather than only implicit token mixing paths;
- Prompt 8 captures bounded thought-node decode metadata during prefill, but still preserves correctness-first decode continuation by falling back to full recompute whenever the explicit thought graph is active.

Debug stats include:
- `thought_nodes_used`
- `mean_thought_degree`
- `mean_token_to_thought_weight`
- `mean_thought_to_token_weight`
- `mean_thought_update_norm`
- `thought_graph_steps_used`

## Local deliberation global memory anchors (optional)

`LocalDeliberationBlock` now also supports a bounded global anchor bank configured by:
- `GPTConfig.local_delib_global_anchor_count`
- `GPTConfig.local_delib_global_anchor_dim`
- `GPTConfig.local_delib_global_anchor_update`
- `GPTConfig.local_delib_global_anchor_temp`
- `GPTConfig.local_delib_global_anchor_use_hierarchy`
- `GPTConfig.local_delib_global_anchor_use_scratch`
- `GPTConfig.local_delib_global_anchor_use_thought`

Behavior:
- when `local_delib_global_anchor_count == 0`, no anchor modules are instantiated and existing paths are unchanged;
- when enabled, the block maintains a tiny causal per-prefix anchor state across micro-steps, analogous to the scratch prefix-state approach but scoped to a much smaller long-range summary bank;
- tokens read anchor summaries through bounded attention over the anchor bank instead of full token-to-token global attention;
- when `local_delib_global_anchor_update=True`, anchors receive writes from causal prefix summaries of token state plus optional hierarchy, scratch, and thought feedback that already exists inside the same micro-step;
- anchor feedback is projected back into token-state space and merged into the local deliberation update input only;
- the anchor-to-token projection is zero-initialized, so simply turning the feature on does not force a behavior change before training or explicit test-time configuration;
- Prompt 8 decode-cache now persists a bounded per-step anchor bank plus prefix token-stage state so anchor reads/writes do not require rerunning the whole cached prefix for the scratch/anchor continuation path.

Debug stats include:
- `global_anchors_used`
- `mean_anchor_read_weight`
- `mean_anchor_write_weight`
- `mean_anchor_norm`
