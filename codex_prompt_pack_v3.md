 # codex_prompt_pack_v3.md

## Purpose

This prompt pack is designed for **Codex / repo agent work inside `RT123-new/nanochat`**.

It assumes the repo already contains:
- the cognition wrapper layer
- model-core local deliberation
- adaptive halting
- causal neighbor graph
- branch spawn/merge
- hierarchy
- latent scratchpad
- auxiliary losses
- lightweight evaluation hooks

The goal of this pack is to push the repo from a **latent local-deliberation approximation** toward a fuller **latent hierarchical graph-of-thought architecture**, while staying aligned with the repo’s own implementation discipline.

This pack is intentionally **sequenced**. Do not paste everything at once. Use one prompt at a time, in order.

---

## Repo-grounding notes

The repo’s own docs say to:
- follow `plans.md` milestone by milestone
- read core repo files before coding
- make the smallest clean slice
- run targeted validation first
- update `documentation.md` every time

Those constraints come directly from:
- `plans.md`
- `implement.md`
- `docs/architecture.md`
- `docs/evals.md`

---

# Prompt 0 — Master bootstrap prompt

```text
You are working inside the existing `RT123-new/nanochat` repository.

First, read these files before changing anything:
1. README.md
2. pyproject.toml
3. AGENTS.md
4. plans.md
5. implement.md
6. documentation.md
7. docs/architecture.md
8. docs/evals.md
9. nanochat/gpt.py
10. nanochat/local_deliberation.py
11. nanochat/cognition/backend.py
12. nanochat/cognition/agent.py
13. scripts/base_train.py
14. scripts/cognition_eval.py
15. relevant tests under `tests/`

Important constraints:
- Preserve repo layout.
- Do not refactor unrelated modules.
- Keep changes opt-in and off-by-default unless explicitly justified.
- Keep speedrun / pretraining flow intact by default.
- Prefer small milestone-scoped PRs.
- Run the lightest relevant validation first.
- Update `documentation.md` with milestone, files inspected, files changed, commands run, results, known issues, next step.

Current repo status:
- model-core local deliberation already exists
- adaptive halt, neighbor graph, latent branching, hierarchy, scratchpad, aux losses, and eval/tracing hooks already exist
- the remaining goal is to evolve this into a fuller latent token-localized graph-of-thought architecture without breaking current nanochat behavior

Before coding:
1. Audit the current implementation and summarize what already exists for:
   - local recurrence
   - adaptive halt
   - neighbor graph
   - phrase consensus
   - branch spawn/merge
   - hierarchy
   - scratchpad
   - aux losses
   - eval/tracing
2. Then propose the smallest next milestone that adds real capability rather than duplicate plumbing.
3. Then implement only that milestone.

Do not jump straight into a giant rewrite.
Do not remove existing local deliberation behavior.
Do not change public generation return contracts.
Prefer exact, inspectable stats and CPU-friendly tests.
```

---

# Prompt 1 — Audit + gap map

```text
Perform a repo-grounded audit of the current local deliberation architecture.

Tasks:
1. Inspect:
   - nanochat/local_deliberation.py
   - nanochat/gpt.py
   - scripts/base_train.py
   - scripts/cognition_eval.py
   - nanochat/cognition/backend.py
   - nanochat/cognition/agent.py
   - tests/test_local_deliberation.py
   - tests/test_gpt_local_deliberation.py
   - tests/test_engine_local_deliberation.py
   - tests/test_cognition_eval.py
   - docs/architecture.md
   - documentation.md

2. Produce a concise implementation matrix with columns:
   - capability
   - implemented / partial / missing
   - exact files
   - current limitations
   - highest-value next extension

Focus on:
- nested recurrent local thinking
- per-token adaptive compute
- local neighbor graph / flocking-like behavior
- phrase consensus
- branch spawn/merge
- multi-scale hierarchy
- latent scratchpad
- aux losses
- eval/trace surfacing
- decode-time cache efficiency
- explicit graph-of-thought behavior

3. Based on the audit, recommend the best next milestone that moves the repo from “latent local approximation” toward “fuller latent hierarchical graph-of-thought”.

Do not change code yet unless you find a tiny obvious bug.
Update documentation.md with the audit summary.
Run only lightweight validation for docs changes.
```

---

# Prompt 2 — Explicit flocking operators

```text
Implement an opt-in flocking operator layer inside model-core local deliberation.

Context:
The repo already has a causal neighbor graph mixer, but it does not yet expose explicit flocking-style update terms such as alignment, cohesion, and separation. Add these in a bounded, inspectable, off-by-default way.

Goal:
Extend `LocalDeliberationBlock` so that when enabled, token latent states can receive additional local update signals computed from their causal neighbor set:
- alignment: move toward neighbor direction / velocity
- cohesion: move toward local centroid
- separation: push away from overly similar or overly close neighbors

Implementation requirements:
- Keep all behavior disabled by default.
- Reuse the existing causal neighbor graph path rather than inventing a new global service.
- Add first-class config plumbing in `GPTConfig` and `scripts/base_train.py`.
- Keep decode-time cache compatibility intact.
- Expose per-layer debug stats for flocking activation and contribution magnitudes.

Suggested config additions:
- local_delib_use_flocking: bool = False
- local_delib_flocking_alignment_weight: float = 0.0
- local_delib_flocking_cohesion_weight: float = 0.0
- local_delib_flocking_separation_weight: float = 0.0
- local_delib_flocking_separation_margin: float = sensible default
- local_delib_flocking_radius_cap: int = 0 or optional bounded cap

Suggested files:
- nanochat/local_deliberation.py
- nanochat/gpt.py
- scripts/base_train.py
- tests/test_local_deliberation.py
- tests/test_gpt_local_deliberation.py
- docs/architecture.md
- documentation.md

Acceptance criteria:
- disabled path preserves current outputs
- enabled path adds explicit flocking update terms
- stats include at least:
  - mean_alignment_norm
  - mean_cohesion_norm
  - mean_separation_norm
  - mean_flocking_total_norm
  - flocking_neighbor_count
- tests cover disabled parity, enabled shapes/stats, strict causality, and locality
- docs updated

Validation:
- syntax checks on touched files
- targeted pytest only for touched local deliberation / GPT tests

Do not rewrite the existing neighbor graph mixer.
Do not enable flocking by default.
```

---

# Prompt 3 — Branch-to-branch consensus and verifier-guided merge

```text
Implement a stronger latent branch consensus / verifier-guided merge mechanism for local deliberation.

Context:
The repo already supports bounded latent branch spawn/score/merge. However, the current branch path is still shallow: branches are proposed, scored, and merged, but there is no richer branch-to-branch comparison or consensus signal.

Goal:
Extend branching so branch proposals can:
- compare against each other,
- produce a lightweight branch-consensus summary,
- optionally use a verifier-style latent scorer before final merge.

Design constraints:
- stay latent and bounded
- do not build a symbolic tree executor
- keep off-by-default
- preserve current branch path when the new mode is disabled

Suggested config additions:
- local_delib_branch_consensus: bool = False
- local_delib_branch_verifier: bool = False
- local_delib_branch_consensus_temp: float
- local_delib_branch_max_active: int (bounded)
- local_delib_branch_disagreement_threshold: float

Suggested implementation ideas:
- compute pairwise branch similarity / disagreement
- derive a branch-consensus latent summary
- optionally derive a verifier score using token state + proposal + consensus context
- merge via gated weighted combination of:
  - current token state
  - weighted branch summary
  - branch consensus summary

Expose stats such as:
- mean_branch_disagreement
- mean_branch_consensus_weight
- mean_branch_verifier_score
- mean_branch_entropy
- branch_consensus_used

Suggested files:
- nanochat/local_deliberation.py
- nanochat/gpt.py
- tests/test_local_deliberation.py
- tests/test_gpt_local_deliberation.py
- tests/test_engine_local_deliberation.py
- docs/architecture.md
- documentation.md

Acceptance criteria:
- old branch path unchanged when disabled
- new consensus/verifier branch mode works when enabled
- causality preserved
- decode cache path still functions
- bounded memory / compute behavior documented
- tests cover disabled parity and enabled stats

Validation:
- targeted syntax checks
- targeted local deliberation / GPT / engine tests only
```

---

# Prompt 4 — Explicit latent graph runtime

```text
Implement a bounded explicit latent graph runtime inside model-core local deliberation.

Context:
The current system is a latent local-deliberation architecture with neighbor graph, branches, hierarchy, and scratchpad. It is still only an approximation of graph-of-thought because there are no explicit latent thought nodes and edge types with a stable graph update loop.

Goal:
Add a small, optional latent graph runtime that creates and updates explicit latent thought nodes derived from token states, while keeping everything inside model-core and off-by-default.

Design:
- Thought nodes remain latent tensors, not symbolic text
- Graph is bounded and local
- Tokens can write to and read from thought nodes
- Thought nodes can connect to:
  - token groups
  - branch summaries
  - hierarchy summaries
  - scratch summaries
- Graph updates happen for a small fixed number of micro-iterations

Suggested components:
- LatentThoughtNodeBuilder
- LatentThoughtGraph
- LatentThoughtMessagePassing
- TokenToThoughtReadWrite
- ThoughtConsensusReducer

Suggested config:
- local_delib_use_thought_graph: bool = False
- local_delib_thought_node_budget: int
- local_delib_thought_node_dim: int
- local_delib_thought_graph_steps: int
- local_delib_thought_topk_edges: int
- local_delib_thought_token_chunk_size: int
- local_delib_thought_use_branch_inputs: bool = True/False
- local_delib_thought_use_hierarchy_inputs: bool = True/False
- local_delib_thought_use_scratch_inputs: bool = True/False

Implementation requirements:
- keep graph construction causal
- keep graph bounded and cheap
- graph must be opt-in
- preserve current paths when disabled
- expose compact per-layer graph stats:
  - thought_nodes_used
  - mean_thought_degree
  - mean_token_to_thought_weight
  - mean_thought_to_token_weight
  - mean_thought_update_norm
  - thought_graph_steps_used

Suggested files:
- nanochat/local_deliberation.py
- nanochat/gpt.py
- scripts/base_train.py
- tests/test_local_deliberation.py
- tests/test_gpt_local_deliberation.py
- docs/architecture.md
- documentation.md

Acceptance criteria:
- explicit latent thought nodes exist when enabled
- token states can write/read to/from thought nodes
- graph updates are bounded and inspectable
- disabled path preserves behavior
- tests cover enabled shapes/stats, causality, disabled parity, and bounded node budget
- docs explain how this differs from the existing lightweight local approximation

Validation:
- targeted syntax checks
- targeted local deliberation / GPT tests only
```

---

# Prompt 5 — Deeper hierarchy beyond chunk stack

```text
Extend local deliberation hierarchy from chunk-stack feedback to a fuller token -> phrase -> span -> sequence-summary hierarchy.

Context:
The repo already supports optional hierarchy chunk sizes and causal broadcast. Extend this into a more explicit multi-level hierarchy with clearer scale semantics and optional cross-scale message passing.

Goal:
Support at least these hierarchy levels when enabled:
- token
- phrase
- span / segment
- sequence summary

Requirements:
- keep levels modular and independently disable-able
- preserve current behavior when disabled
- keep all message passing causal
- keep implementation bounded and inspectable

Suggested additions:
- explicit sequence summary node
- optional upward pass: token -> phrase -> span -> sequence
- optional downward pass: sequence -> span -> phrase -> token
- optional adjacent-scale bidirectional updates per micro-step
- expose stats per scale and per direction

Suggested config:
- local_delib_use_deep_hierarchy: bool = False
- local_delib_span_chunk_size: int
- local_delib_sequence_summary: bool = False
- local_delib_hierarchy_bidirectional: bool = False
- local_delib_hierarchy_scale_gate: bool = False

Expose stats like:
- phrase_nodes_used
- span_nodes_used
- sequence_summary_used
- mean_upward_message_norm
- mean_downward_message_norm
- mean_scale_gate
- hierarchy_depth_used

Suggested files:
- nanochat/local_deliberation.py
- nanochat/gpt.py
- tests/test_local_deliberation.py
- tests/test_gpt_local_deliberation.py
- docs/architecture.md
- documentation.md

Acceptance criteria:
- existing hierarchy path remains available
- deeper hierarchy path is opt-in
- tests validate cross-scale information flow and causality
- stats make scale participation inspectable
```

---

# Prompt 6 — Persistent creative scratch orchestration

```text
Upgrade the latent scratchpad from a simple slot bank into a more structured creative workspace inside model-core local deliberation.

Context:
The repo already has optional scratch slots with gated token read/write. The next step is to make scratch behavior more like a bounded speculative composition workspace rather than just another side input.

Goal:
Add structured scratch lifecycle behavior:
- allocate
- write
- refine
- read
- optionally summarize
- reset cleanly between requests

Important:
Do not make scratch persistent across requests unless explicitly scoped and safe.
Do not emit scratch as text.
Keep it latent and bounded.

Suggested features:
- scratch refinement substep per micro-step
- optional branch-to-scratch writes
- optional hierarchy-to-scratch writes
- optional scratch summary vector exported to metadata
- optional wrapper-level conditioning hook so cognition traces can inspect compact scratch summaries

Suggested config:
- local_delib_scratch_refine_steps: int = 0
- local_delib_scratch_use_branch_inputs: bool = False
- local_delib_scratch_use_hierarchy_inputs: bool = False
- local_delib_scratch_export_summary: bool = False
- local_delib_scratch_summary_dim: int = 0

Expose stats:
- mean_scratch_refine_norm
- mean_scratch_summary_norm
- mean_branch_to_scratch_weight
- mean_hierarchy_to_scratch_weight
- scratch_reset_ok

Suggested files:
- nanochat/local_deliberation.py
- nanochat/gpt.py
- nanochat/cognition/backend.py
- nanochat/cognition/agent.py
- tests/test_local_deliberation.py
- tests/test_gpt_local_deliberation.py
- tests/test_cognition_backend.py
- tests/test_cognition_agent.py
- docs/architecture.md
- documentation.md

Acceptance criteria:
- scratch remains off-by-default
- existing scratch path remains valid
- reset semantics are tested
- optional summary export appears only when enabled
- metadata path remains backward-compatible
```

---

# Prompt 7 — Global memory tokens / long-range anchors

```text
Implement optional global memory tokens / anchor states for local deliberation.

Context:
The current architecture is mostly local: neighbor graph, branches, hierarchy, scratch. To reduce long-range coherence loss, add a tiny set of global anchor states that can summarize broader context and feed back into local token updates.

Goal:
Add a bounded global-memory path inside local deliberation that complements the local graph rather than replacing it.

Requirements:
- keep off-by-default
- keep bounded and causal
- do not introduce full global attention
- preserve disabled-path parity

Suggested design:
- a small bank of learned or dynamically pooled global anchor states
- anchors updated from current latent state summaries
- tokens can attend to anchors in a bounded way
- anchors can optionally receive hierarchy / thought-graph / scratch summaries

Suggested config:
- local_delib_global_anchor_count: int = 0
- local_delib_global_anchor_dim: int = 0
- local_delib_global_anchor_update: bool = False
- local_delib_global_anchor_temp: float
- local_delib_global_anchor_use_hierarchy: bool = False
- local_delib_global_anchor_use_scratch: bool = False

Expose stats:
- global_anchors_used
- mean_anchor_read_weight
- mean_anchor_write_weight
- mean_anchor_norm

Suggested files:
- nanochat/local_deliberation.py
- nanochat/gpt.py
- tests/test_local_deliberation.py
- tests/test_gpt_local_deliberation.py
- docs/architecture.md
- documentation.md

Acceptance criteria:
- no behavior change when disabled
- enabled path gives bounded global-summary feedback
- tests validate causality and shape/stats
```

---

# Prompt 8 — Decode-time cache optimization

```text
Optimize decode-time local deliberation cache behavior for advanced modes.

Context:
The repo already supports local deliberation in KV-cached decode mode, but the implementation is correctness-first and not yet optimized for advanced modes like branching, hierarchy, scratch, and any new thought-graph path.

Goal:
Refactor cached local deliberation state so decode-time behavior remains correct while reducing unnecessary recomputation and keeping per-feature cache metadata bounded.

Requirements:
- preserve current public interfaces
- preserve correctness first
- improve memory/compute behavior for enabled advanced paths
- document tradeoffs and limits

Focus on:
- branch metadata cache
- hierarchy summary cache
- scratch summary / slot cache
- optional thought-graph cache
- anchor/global-summary cache if present

Suggested files:
- nanochat/gpt.py
- nanochat/engine.py
- nanochat/local_deliberation.py
- tests/test_engine_local_deliberation.py
- tests/test_gpt_local_deliberation.py
- documentation.md
- docs/architecture.md

Acceptance criteria:
- decode path still passes existing tests
- advanced mode cache state is bounded and documented
- no change to return contracts
- new targeted tests cover cache population and decode continuation for enabled advanced modes

Validation:
- targeted engine/GPT tests only
```

---

# Prompt 9 — Second-wave auxiliary losses

```text
Implement a second wave of optional auxiliary local-deliberation losses for the fuller latent graph-of-thought direction.

Context:
The repo already includes minimal aux losses for halting sparsity, branch diversity, branch entropy, consensus agreement, and scratch utilization. Extend this carefully with additional opt-in losses that support explicit flocking, thought-graph stability, deeper hierarchy, and better branch usefulness.

Goal:
Add well-scoped, off-by-default auxiliary losses such as:
- flocking consistency / stability loss
- thought-graph edge stability loss
- thought-node utilization loss
- hierarchy utilization / cross-scale agreement loss
- branch usefulness proxy loss
- anchor/global-summary usage loss (if anchors exist)

Requirements:
- all losses default to zero-weight / off
- preserve training behavior when weights are zero
- surface aux values through existing `last_aux_losses`
- document formulas clearly
- add focused tests for numeric sanity and zero-weight parity

Suggested files:
- nanochat/local_deliberation.py
- nanochat/gpt.py
- scripts/base_train.py
- tests/test_local_deliberation.py
- tests/test_gpt_local_deliberation.py
- docs/architecture.md
- documentation.md

Acceptance criteria:
- new aux terms are surfaced but inert by default
- weighted composition works via existing training path
- tests cover zero-weight parity and finite numeric ranges
- docs updated with exact formulas
```

---

# Prompt 10 — Real ablation and evaluation suite

```text
Expand the current local deliberation evaluation harness into a fuller ablation suite for the latent hierarchical graph-of-thought architecture.

Context:
The repo already has a lightweight local-delib ablation eval with demo and engine-backed modes. Extend it so the architecture can be compared across more meaningful variants and telemetry can be analyzed cleanly.

Goal:
Add evaluation coverage for:
- basic local deliberation
- adaptive halt
- neighbor graph
- explicit flocking
- branch consensus / verifier merge
- hierarchy depth
- scratch refinement
- thought graph
- global anchors
- selected combined variants

Deliverables:
1. variant definitions
2. machine-readable artifact output
3. clear stats aggregation
4. interpretation guidance in docs

Suggested artifact fields:
- quality proxy scores
- variant_mean_scores
- compute proxy metrics
- mean_steps_taken
- branch stats
- hierarchy stats
- scratch stats
- thought-graph stats
- flocking stats
- anchor stats
- optional combined “quality per compute” estimate

Suggested files:
- nanochat/cognition/eval.py
- scripts/cognition_eval.py
- tests/test_cognition_eval.py
- docs/evals.md
- documentation.md

Acceptance criteria:
- current suite remains intact
- new suite is selectable via CLI
- demo backend remains CPU-friendly
- engine-backed run remains optional
- docs include concrete commands and interpretation notes
```

---

# Prompt 11 — Wrapper-level thought summaries

```text
Add optional wrapper-level surfacing of compact model-core thought summaries into cognition traces.

Context:
The repo already exposes model local deliberation stats into cognition metadata. Extend this so compact summaries from advanced mechanisms can be surfaced in a controlled, namespaced way without changing generation return contracts.

Goal:
When enabled, backend metadata should optionally include compact summaries for:
- branch consensus
- deep hierarchy
- scratch summary
- thought graph
- global anchors
- flocking

Requirements:
- keep namespaced metadata keys
- do not dump huge tensors
- summaries must be compact numeric/statistical payloads only
- preserve old metadata behavior

Suggested files:
- nanochat/cognition/backend.py
- nanochat/cognition/agent.py
- tests/test_cognition_backend.py
- tests/test_cognition_agent.py
- documentation.md
- docs/architecture.md

Acceptance criteria:
- old trace metadata still works
- new summaries appear only when available
- tests cover metadata plumbing
```

---

# Prompt 12 — Final hardening + docs + recovery card

```text
Perform a final hardening pass for the advanced local deliberation / latent graph-of-thought path.

Tasks:
1. Audit all new config fields in GPTConfig and scripts/base_train.py
2. Check disabled-path parity assumptions
3. Check docs consistency across:
   - plans.md
   - implement.md
   - docs/architecture.md
   - docs/evals.md
   - documentation.md
4. Add or update a compact context recovery section explaining:
   - what is implemented
   - what is still experimental
   - how to run the main ablations
   - how to enable major features
   - known risks / rollback switches
5. Add any missing targeted tests for regressions discovered during the audit

Constraints:
- no repo-wide refactor
- no new broad dependencies
- keep changes scoped to hardening / docs / tiny fixes

Deliver a concise summary of:
- completed advanced capabilities
- remaining gaps
- recommended next research directions
```

---

# Recommended execution order

1. Prompt 0 — Master bootstrap  
2. Prompt 1 — Audit + gap map  
3. Prompt 2 — Explicit flocking operators  
4. Prompt 3 — Branch consensus / verifier merge  
5. Prompt 4 — Explicit latent graph runtime  
6. Prompt 5 — Deeper hierarchy  
7. Prompt 6 — Persistent creative scratch orchestration  
8. Prompt 7 — Global memory anchors  
9. Prompt 8 — Decode cache optimization  
10. Prompt 9 — Second-wave auxiliary losses  
11. Prompt 10 — Evaluation suite  
12. Prompt 11 — Wrapper-level thought summaries  
13. Prompt 12 — Final hardening  

---

# Single highest-value prompt

If you only run one next prompt, use this:

```text
Implement a bounded explicit latent graph runtime inside model-core local deliberation.

The current repo already has:
- recurrent local deliberation
- adaptive halting
- causal neighbor graph
- phrase consensus
- latent branching
- hierarchy
- scratchpad
- aux losses
- eval/tracing hooks

The missing step is to move from a latent local approximation toward a fuller latent token-localized graph-of-thought architecture.

Read first:
- README.md
- pyproject.toml
- AGENTS.md
- plans.md
- implement.md
- documentation.md
- docs/architecture.md
- nanochat/gpt.py
- nanochat/local_deliberation.py
- scripts/base_train.py
- tests relevant to local deliberation / GPT / engine

Implement:
- opt-in explicit latent thought nodes
- bounded token-to-thought and thought-to-token message passing
- causal thought-node construction from token groups
- compact thought-graph stats
- disabled-path parity
- focused tests
- documentation update

Keep everything:
- off-by-default
- bounded
- inspectable
- compatible with existing generation interfaces
- scoped to the smallest clean slice

Run targeted validation only.
Update documentation.md in the required repo format.
```

---

# Practical usage notes

- Paste **one prompt at a time**.
- Wait for Codex to finish each milestone.
- Review diffs after every prompt.
- Do not let Codex batch unrelated milestones together.
- If a prompt starts drifting into rewrite mode, stop it and restate:
  - smallest clean slice
  - off-by-default
  - targeted validation
  - update documentation.md

---

# Suggested PR breakdown

- PR 1: audit + flocking
- PR 2: branch consensus / verifier merge
- PR 3: explicit latent thought graph
- PR 4: deeper hierarchy + scratch orchestration
- PR 5: global anchors + cache optimization
- PR 6: aux losses + eval suite + trace surfacing
- PR 7: hardening / docs / recovery card

This keeps review manageable and aligns with the repo’s milestone discipline.
