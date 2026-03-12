# Evaluation plan for the nanochat cognition layer

## Objective
Show that the cognition subsystem improves practical behavior over a simpler baseline while fitting naturally into the existing nanochat repo.

## Evaluation philosophy
The first evaluation pass should be cheap, controlled, and easy to interpret.
Avoid building a giant benchmark harness before the basic subsystem works.

## Baselines
1. Existing nanochat-style direct generation loop
2. Nanochat plus memory only
3. Full cognition subsystem

## Evaluation families

### A. Memory usefulness
Goal:
Show that retrieval improves behavior on tasks that benefit from prior episodes or distilled knowledge.

Possible measures:
- retrieval relevance
- retrieval usefulness
- improvement versus no retrieval
- reduced repeated rediscovery

### B. Repeated-task improvement
Goal:
Show that repeated related tasks get easier after consolidation and reuse.

Possible measures:
- success rate over repeated trials
- retries before success
- skill reuse count
- improvement after a skill artifact is formed

### C. Candidate generation and selection
Goal:
Show that creative generation plus verification produces better final choices than direct one-shot answering on suitable tasks.

Possible measures:
- candidate diversity
- verifier ranking quality
- final-task success
- repair count

### D. Sandbox benefit
Goal:
Show that lightweight branch-and-score experimentation can improve final choice quality on tasks where trying alternatives helps.

Possible measures:
- score before sandbox vs after sandbox
- branch efficiency
- final win rate

### E. Trace quality
Goal:
Show that the system is inspectable enough to explain why it improved or failed.

Possible measures:
- presence of structured route rationale
- readable trace artifacts
- ability to identify why a choice was made

## How to align with the existing repo
Use existing repo patterns where sensible.
For example:
- existing script-style entrypoints in `scripts/`
- existing tests in `tests/`
- existing report or artifact conventions if practical

The first cognition eval does not need to reuse every current evaluation mechanism, but it should not fight the repo's style.

## Minimal first deliverables
- one lightweight evaluation entrypoint or script
- one baseline vs cognition-enhanced comparison
- machine-readable result artifact if practical
- markdown notes on setup, commands, and limitations

## Success criteria
A promising result is not just a better answer once.
A promising result shows some combination of:
- better use of prior experience
- lower rediscovery cost
- improved repeated-task performance
- evidence of reuse through skill artifacts
- clearer explanation of why the system made a choice

## Milestone 8 implementation (lightweight harness)

A repo-native evaluation entrypoint now exists at `scripts/cognition_eval.py`.

### What it compares
- **Baseline**: direct backend generation through `BackendAdapter.run(...)`
- **Cognition-enhanced**: `CognitionAgent.run(...)` over the same prompts

### Backend modes
- `--backend demo` uses a deterministic context-aware backend that only improves when cognition injects episodic, semantic, or skill context into the prompt
- `--backend engine` optionally loads a real checkpoint-backed `EngineBackend` for end-to-end comparisons against live nanochat generation

### Built-in cases
The default harness now focuses on cases that should only improve if cognition changes the effective prompt:
- episodic recall
- semantic memory reuse
- skill reuse

### Scoring and artifacts
- Per-case keyword recall score in `[0, 1]`
- Aggregate `baseline_mean`, `cognition_mean`, and `delta`
- Route histogram (`route_counts`) for inspectability
- JSON artifact containing per-case rows and aggregate summary
- strict failure when a case marked as requiring cognition gain does not outperform baseline

### Run command
```bash
python -m scripts.cognition_eval --output artifacts/cognition_eval.json
```

Optional real-checkpoint comparison:
```bash
python -m scripts.cognition_eval --backend engine --source sft
```

### Current limitations
- Keyword scoring is intentionally simple and should be replaced with richer task metrics in future milestones.
- The engine-backed path remains opt-in because it depends on local checkpoints and runtime setup.


## Milestone 15+ local deliberation architecture ablation (lightweight)

A focused model-side ablation suite is now available through `scripts/cognition_eval.py`.

### What it compares
- `local_delib_off`
- `local_delib_basic`
- `local_delib_adaptive_halt`
- `local_delib_branch`
- `local_delib_hierarchy`
- `local_delib_scratchpad`

### Run command
```bash
python -m scripts.cognition_eval   --suite local-delib-ablation   --backend demo   --output artifacts/local_delib_ablation_eval.json
```

Optional checkpoint-backed run (requires local model + torch runtime):
```bash
python -m scripts.cognition_eval   --suite local-delib-ablation   --backend engine   --source sft   --output artifacts/local_delib_ablation_engine.json
```

### Artifact fields and interpretation
- `variant_mean_scores`: mean keyword score per variant across included prompts.
- Per-row advanced model metadata:
  - `model_local_delib_branch`
  - `model_local_delib_hierarchy`
  - `model_local_delib_scratchpad`
  - `model_local_delib_adaptive_halt`

Interpretation guidance:
- Use `branch_factor_used` and `mean_branch_score` to verify branch mode activation.
- Use `mean_branch_disagreement`, `mean_branch_consensus_weight`, `mean_branch_verifier_score`, and `branch_consensus_used` to verify Prompt 3 branch-consensus mode activation.
- Use `thought_nodes_used`, `mean_thought_degree`, `mean_token_to_thought_weight`, `mean_thought_to_token_weight`, `mean_thought_update_norm`, and `thought_graph_steps_used` to verify Prompt 4 explicit thought-graph activation.
- Use `hierarchy_levels_used` and hierarchy feedback norms to confirm hierarchy participation.
- Use `phrase_nodes_used`, `span_nodes_used`, `sequence_summary_used`, `mean_upward_message_norm`, `mean_downward_message_norm`, `mean_scale_gate`, and `hierarchy_depth_used` to verify Prompt 5 deep-hierarchy activation.
- Use `scratch_slots_used`, read/write weights, `mean_scratch_refine_norm`, `mean_scratch_summary_norm`, `mean_branch_to_scratch_weight`, `mean_hierarchy_to_scratch_weight`, and `scratch_reset_ok` to verify Prompt 6 structured scratch orchestration.
- When `model_local_delib.scratchpad_summaries` is present, inspect the exported compact summary vectors to compare scratch usage across layers without decoding latent scratch state as text.
- Prompt 7 does not yet have a dedicated eval variant in this harness, but anchor-enabled runs can still be verified from raw layer stats via `global_anchors_used`, `mean_anchor_read_weight`, `mean_anchor_write_weight`, and `mean_anchor_norm`.
- Use halt fraction / mean steps to compare fixed-step vs adaptive-halt compute behavior.

## Prompt 10 advanced local deliberation ablation suite

The lightweight `local-delib-ablation` suite remains available as a fast smoke path. Prompt 10 adds a broader, still CPU-friendly suite at the same entrypoint with richer variant coverage and machine-readable telemetry aggregation.

### Variant coverage
- `local_delib_off`
- `local_delib_basic`
- `local_delib_adaptive_halt`
- `local_delib_neighbor_graph`
- `local_delib_flocking`
- `local_delib_branch_consensus_verifier`
- `local_delib_deep_hierarchy`
- `local_delib_scratch_refine`
- `local_delib_thought_graph`
- `local_delib_global_anchors`
- `local_delib_combo_reasoner`
- `local_delib_combo_full_stack`

### Run commands

Demo backend:
```bash
python -m scripts.cognition_eval \
  --suite local-delib-ablation-advanced \
  --backend demo \
  --output artifacts/local_delib_ablation_advanced.json
```

Optional checkpoint-backed run:
```bash
python -m scripts.cognition_eval \
  --suite local-delib-ablation-advanced \
  --backend engine \
  --source sft \
  --output artifacts/local_delib_ablation_advanced_engine.json
```

### Artifact fields
- `quality_proxy_scores`: per-case keyword proxy scores grouped by variant.
- `variant_mean_scores`: mean quality proxy score per variant.
- `quality_per_compute`: mean quality proxy divided by a simple telemetry-derived compute estimate.
- `compute_proxy_metrics`: aggregated `mean_steps_taken`, `halted_token_fraction`, and `estimated_compute_cost`.
- `mean_steps_taken`: convenience view of the per-variant mean step count.
- `neighbor_graph_stats`: aggregated neighbor-graph metrics such as `mean_neighbor_count` and `semantic_topk_used`.
- `branch_stats`: branch activation, disagreement, consensus, and verifier metrics.
- `hierarchy_stats`: legacy/deep hierarchy usage plus phrase/span/sequence telemetry.
- `scratch_stats`: scratch slot usage and refinement telemetry.
- `thought_graph_stats`: thought-node usage, degree, and token-write/read telemetry.
- `flocking_stats`: alignment/cohesion/separation metrics and flocking activation rates.
- `anchor_stats`: global anchor usage and read/write telemetry.
- `runtime_variant_overrides_applied`: whether the backend actually honored per-variant runtime kwargs.

### Interpretation guidance
- Treat `quality_per_compute` as a heuristic, not a benchmark-grade efficiency metric. It is intended for quick ablation ranking when all variants use the same cheap harness.
- Use `runtime_variant_overrides_applied=true` to confirm the backend really switched features per variant. The demo backend does this. The engine-backed path only reports `true` when the backend supports runtime variant overrides directly.
- Compare `mean_steps_taken` and `halted_token_fraction` to see whether adaptive halting is actually reducing compute.
- Compare `neighbor_graph_stats.mean_neighbor_count` and `flocking_stats.fraction_flocking_tokens_active` to separate plain graph mixing from explicit flocking behavior.
- Use `branch_stats.branch_consensus_used`, `branch_stats.mean_branch_disagreement`, and `branch_stats.mean_branch_verifier_score` to judge whether branch consensus / verifier merge is doing meaningful work.
- Use `hierarchy_stats.hierarchy_depth_used`, `hierarchy_stats.phrase_nodes_used`, `hierarchy_stats.span_nodes_used`, and `hierarchy_stats.sequence_summary_used` to verify deep hierarchy participation rather than only legacy chunk summaries.
- Use `scratch_stats.mean_scratch_refine_norm` alongside `scratch_stats.scratch_slots_used` to check whether scratch refinement is active or merely allocated.
- Use `thought_graph_stats.thought_nodes_used` and `thought_graph_stats.thought_graph_steps_used` to verify explicit thought-graph engagement.
- Use `anchor_stats.global_anchors_used` and `anchor_stats.mean_anchor_read_weight` to confirm that global anchors are contributing long-range state instead of staying idle.

### Current limitation
- The engine-backed suite can be run, but runtime hot-swapping of local-deliberation architecture variants depends on backend support. When that support is unavailable, the artifact marks `runtime_variant_overrides_applied=false` so the run is interpreted as telemetry over the loaded checkpoint config rather than a true per-variant swap.

## Prompt 11/12 wrapper trace and hardening notes

When running cognition-wrapper traces on top of engine-backed generation, the wrapper can now surface compact advanced local-deliberation summaries under:
- `model_local_delib.thought_summaries.branch_consensus`
- `model_local_delib.thought_summaries.deep_hierarchy`
- `model_local_delib.thought_summaries.scratch`
- `model_local_delib.thought_summaries.thought_graph`
- `model_local_delib.thought_summaries.global_anchors`
- `model_local_delib.thought_summaries.flocking`

Use those keys as compact trace payloads alongside the older aggregate buckets (`model_local_delib.branch`, `model_local_delib.hierarchy`, `model_local_delib.scratchpad`, `model_local_delib.adaptive_halt`, and `model_local_delib.scratchpad_summaries`) rather than as replacements.

Recommended hardening run order:
- smoke ablation first: `python -m scripts.cognition_eval --suite local-delib-ablation --backend demo --output artifacts/local_delib_ablation_eval.json`
- broaden to the advanced suite next: `python -m scripts.cognition_eval --suite local-delib-ablation-advanced --backend demo --output artifacts/local_delib_ablation_advanced.json`
- only then compare against a real checkpoint-backed engine run if local model/runtime setup is available

Rollback guidance for eval interpretation:
- if `runtime_variant_overrides_applied=false`, treat engine-backed variant rows as telemetry over the loaded checkpoint config, not a true architecture swap
- if a mechanism destabilizes quality or compute, fall back by disabling its flag/count or by using the lighter smoke suite until the issue is isolated
