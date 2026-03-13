# Evaluation plan for the nanochat cognition layer

## Objective
Show that the cognition subsystem improves practical behavior over a simpler baseline while fitting naturally into the existing nanochat repo.

## Pre-training proof pack and runbook
The repo now includes docs-only pre-training validation assets for the added systems:
- [codex_test_prompt_pack.md](../codex_test_prompt_pack.md)
- [pretraining_validation_runbook.md](./pretraining_validation_runbook.md)

Use these when you want to generate or execute a comprehensive proof suite before a training run.

They define:
- `works` proofs: correctness, causality, parity, boundedness, and artifact-shape checks
- `useful` proofs: baseline deltas, activation evidence, compute accounting, trace quality, and truthful runtime-override interpretation
- a default CPU/mock-first gate through the wrapper cognition and model-core local-deliberation stack
- an optional checkpoint-backed engine smoke pass that is not part of the default gate

Recommended proof-pack artifact paths:
- `artifacts/pretraining_proofs/cpu_mock/cognition_eval.json`
- `artifacts/pretraining_proofs/cpu_mock/local_delib_ablation.json`
- `artifacts/pretraining_proofs/cpu_mock/local_delib_ablation_advanced.json`
- `artifacts/pretraining_proofs/cpu_mock/local_delib_research.json`
- `artifacts/pretraining_proofs/engine/cognition_eval.json`
- `artifacts/pretraining_proofs/engine/local_delib_ablation_advanced.json`
- `artifacts/pretraining_proofs/engine/local_delib_research.json`
- `artifacts/pretraining_proofs/engine/task_grounded.json`
- `artifacts/pretraining_proofs/engine/local_delib_natural.json`
- `artifacts/pretraining_proofs/engine/engine_smoke_manifest.json`

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

Prompt 3 note:
- the demo backend also understands the wrapper `Creative strategy:` section, so creative/verify/sandbox orchestration can be smoke-tested without a checkpoint-backed engine.

### Built-in cases
The default harness now focuses on cases that should only improve if cognition changes the effective prompt:
- episodic recall
- semantic memory reuse
- skill reuse

### Scoring and artifacts
- Per-case keyword recall score in `[0, 1]`
- Aggregate `baseline_mean`, `cognition_mean`, and `delta`
- Route histogram (`route_counts`) for inspectability
- Stable top-level JSON keys: `baseline_mean`, `cognition_mean`, `delta`, `route_counts`, and `rows`
- Per-row creative-path telemetry when the wrapper enters a creative route:
  - `creative_strategy_ids`
  - `creative_selected_strategy`
  - `creative_candidate_count`
  - `creative_handoff`
  - `creative_model_summary_used`
- JSON artifact containing per-case rows and aggregate summary
- strict failure when a case marked as requiring cognition gain does not outperform baseline

Interpretation guidance:
- `creative_strategy_ids` should show whether the wrapper actually explored divergence, memory grounding, branch resolution, or recombination instead of producing naive duplicate drafts.
- `creative_handoff` distinguishes pure verifier collapse from verifier-plus-sandbox exploration.
- `creative_model_summary_used=true` only appears when wrapper creativity adapted to surfaced `model_local_delib.*` summaries; demo backend runs without model metadata usually keep this `false`.
- For proof-pack claims, treat a default eval gain as meaningful only when the matching no-support path stays flat under the same backend. The demo harness is designed so episodic, semantic, and skill wins only appear when those sections are actually injected.

### Run command
```bash
python -m scripts.cognition_eval --output artifacts/cognition_eval.json
```

For the pre-training proof pack, prefer:

```bash
python -m scripts.cognition_eval --backend demo --output artifacts/pretraining_proofs/cpu_mock/cognition_eval.json
```

Optional real-checkpoint comparison:
```bash
python -m scripts.cognition_eval --backend engine --source sft
```

Proof-pack engine path:

```bash
python -m scripts.cognition_eval --backend engine --source sft --no-enforce-improvement --output artifacts/pretraining_proofs/engine/cognition_eval.json
```

Optional Prompt 9 smoke test:

```bash
python -m pytest -q tests/test_cognition_engine_smoke.py -m slow
```

Environment notes for the slow smoke path:
- set `NANOCHAT_BASE_DIR` if checkpoints/tokenizer do not live under the default `~/.cache/nanochat`
- use `NANOCHAT_SMOKE_SOURCE`, `NANOCHAT_SMOKE_MODEL_TAG`, and `NANOCHAT_SMOKE_STEP` to pin a specific checkpoint
- use `NANOCHAT_SMOKE_DEVICE_TYPE=cpu|cuda|mps` to choose the runtime
- use `NANOCHAT_SMOKE_ARTIFACT_DIR` if you want the slow smoke artifacts somewhere other than `artifacts/pretraining_proofs/engine/`
- use `--no-enforce-improvement` for engine smoke cognition runs when the goal is to validate checkpoint-backed artifact generation rather than prove support-sensitive gains
- the dedicated slow smoke test uses a built-in offline `SmokeTinyTask` for the task-grounded artifact so the engine audit does not depend on `datasets` downloads or external word lists
- the slow test skips cleanly when the checkpoint root, tokenizer, or requested checkpoint files are unavailable
- the smoke run now writes `engine_smoke_manifest.json` with checkpoint identity, commands, artifacts, runtime-override status coverage, and explicit skip/fail reason when available

### Current limitations
- Keyword scoring is intentionally simple and should be replaced with richer task metrics in future milestones.
- The engine-backed path remains opt-in because it depends on local checkpoints and runtime setup.
- For stronger evidence, prefer the task-grounded and natural local-delib suites below; the default cognition suite remains a lightweight support-injection harness, not a benchmark.

## Task-grounded benchmark suite

This opt-in suite reuses existing repo task graders instead of keyword proxies.

### What it compares
- **Baseline**: direct generation through the selected backend
- **Cognition-enhanced**: `CognitionAgent.run(...)` over the same prompt with the same backend underneath

### Initial task set
- `GSM8K`
- `SpellingBee`
- `HumanEval`

Comparability note:
- `MMLU` and `ARC` are intentionally excluded here because the repo currently evaluates them through categorical/logit selection, which is not directly comparable to cognition-wrapped text generation.

### Run command
```bash
python -m scripts.cognition_eval \
  --suite task-grounded \
  --backend engine \
  --source sft \
  --tasks GSM8K,SpellingBee,HumanEval \
  --max-problems 5 \
  --seed 42 \
  --output artifacts/pretraining_proofs/engine/task_grounded.json
```

### Artifact fields
- `backend_kind`
- `metric_tier`: always `task_grounded`
- `checkpoint_identity`
- `task_names`
- `baseline_mean`, `cognition_mean`, `delta`
- `proof_baseline_mean`, `proof_cognition_mean`, `proof_delta`
- `per_task`: per-task pass rates and proof-filtered pass rates
- Per-row fields:
  - `task_name`
  - `example_index`
  - `prompt`
  - `baseline_response`
  - `cognition_response`
  - `baseline_passed`
  - `cognition_passed`
  - `cognition_decision`
  - `benchmark_eligible`
  - baseline/cognition runtime-override status fields
  - `cognition_trace_metadata`

### Interpretation guidance
- Treat this suite as stronger than the default cognition harness because grading comes from the existing task evaluators.
- Use the `proof_*` fields for benchmark claims. They exclude rows that are not exact-row-equivalent evidence.
- A zero or negative delta is still useful: this suite is designed to detect regressions or parity on real tasks, not only cognition gains.


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

Proof-pack path:

```bash
python -m scripts.cognition_eval   --suite local-delib-ablation   --backend demo   --output artifacts/pretraining_proofs/cpu_mock/local_delib_ablation.json
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

Proof-pack path:

```bash
python -m scripts.cognition_eval \
  --suite local-delib-ablation-advanced \
  --backend demo \
  --output artifacts/pretraining_proofs/cpu_mock/local_delib_ablation_advanced.json
```

Optional checkpoint-backed run:
```bash
python -m scripts.cognition_eval \
  --suite local-delib-ablation-advanced \
  --backend engine \
  --source sft \
  --output artifacts/local_delib_ablation_advanced_engine.json
```

Proof-pack engine path:

```bash
python -m scripts.cognition_eval \
  --suite local-delib-ablation-advanced \
  --backend engine \
  --source sft \
  --output artifacts/pretraining_proofs/engine/local_delib_ablation_advanced.json
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
- `runtime_variant_overrides_applied`: `true` only when every requested variant row was applied exactly.
- `runtime_variant_override_statuses`: per-variant status map with `exact`, `approximated`, or `unsupported`.
- `runtime_variant_override_counts`: aggregate row counts by override status.
- Per-row override fields:
  - `runtime_override_applied`
  - `runtime_override_status`
  - `runtime_override_application_method`
  - `runtime_override_reason`

Hardening note:
- `runtime_variant_override_statuses` is conservative at the per-variant level: if any row for a variant is `unsupported`, the summary status is `unsupported`; otherwise `approximated` outranks `exact`. Use the row-level counts for exact detail.
- For the pre-training proof gate, do not accept a variant score change on its own. Read `quality_per_compute`, `compute_proxy_metrics`, the mechanism-specific stats buckets, and per-row `model_local_delib_graph_artifact` together to verify that the mechanism actually activated.
- The optional Prompt 9 smoke test will fall back to a tiny targeted audit artifact (`runtime_override_audit.json`) only when the broader engine-backed ablation/research runs do not already surface both an `exact` row and a non-exact (`approximated` or `unsupported`) row.

### Interpretation guidance
- Treat `quality_per_compute` as a heuristic, not a benchmark-grade efficiency metric. It is intended for quick ablation ranking when all variants use the same cheap harness.
- Use `runtime_override_status=exact` to confirm a row really ran against the requested variant.
- `runtime_override_status=approximated` means the backend fell back to the loaded checkpoint because the requested override was not strictly state-dict compatible; the row is useful only as baseline telemetry, not as a truthful variant result.
- `runtime_override_status=unsupported` means the backend could not apply the request and no approximation was allowed; these rows should be treated as explicit non-results.
- Compare `mean_steps_taken` and `halted_token_fraction` to see whether adaptive halting is actually reducing compute.
- Compare `neighbor_graph_stats.mean_neighbor_count` and `flocking_stats.fraction_flocking_tokens_active` to separate plain graph mixing from explicit flocking behavior.
- Use `branch_stats.branch_consensus_used`, `branch_stats.mean_branch_disagreement`, and `branch_stats.mean_branch_verifier_score` to judge whether branch consensus / verifier merge is doing meaningful work.
- Use `hierarchy_stats.hierarchy_depth_used`, `hierarchy_stats.phrase_nodes_used`, `hierarchy_stats.span_nodes_used`, and `hierarchy_stats.sequence_summary_used` to verify deep hierarchy participation rather than only legacy chunk summaries.
- Use `scratch_stats.mean_scratch_refine_norm` alongside `scratch_stats.scratch_slots_used` to check whether scratch refinement is active or merely allocated.
- Use `thought_graph_stats.thought_nodes_used` and `thought_graph_stats.thought_graph_steps_used` to verify explicit thought-graph engagement.
- Use `anchor_stats.global_anchors_used` and `anchor_stats.mean_anchor_read_weight` to confirm that global anchors are contributing long-range state instead of staying idle.

### Prompt 5 runtime override matrix
The engine-backed path now distinguishes three cases explicitly:
- `exact`: the backend rebuilt a temporary model with the requested `GPTConfig` overrides and loaded the current checkpoint weights with strict compatibility.
- `approximated`: the requested override was known but not strictly compatible with the loaded checkpoint, and `--allow-approximate-runtime-overrides` told the backend to run the base checkpoint anyway.
- `unsupported`: the override was not compatible and approximation was not allowed, or the backend did not support local-deliberation runtime overrides at all.

Safe exact overrides are the ones that keep the checkpoint state dict compatible with the requested config. In practice this usually means scalar/control changes such as:
- `local_delib_adaptive_halt`
- `local_delib_semantic_lookback`
- `local_delib_branch_every`
- `local_delib_branch_consensus_temp`
- `local_delib_branch_max_active`
- `local_delib_branch_disagreement_threshold`
- `local_delib_hierarchy_bidirectional`
- `local_delib_hierarchy_scale_gate`
- `local_delib_thought_graph_steps`
- `local_delib_thought_token_chunk_size`
- `local_delib_global_anchor_temp`

Overrides that add/remove modules or change parameter shapes are only exact when the rebuilt config stays strictly compatible with the loaded checkpoint. Examples that often become `approximated` or `unsupported` on a checkpoint trained without the same architecture include:
- toggling `local_delib` blocks on/off
- changing `local_delib_every`
- enabling/disabling neighbor graph, deep hierarchy, scratch slots, thought graph, or anchors when that changes module presence
- changing branch factor, scratch dimension, thought-node dimension, or anchor dimension

CLI controls:
- `--allow-approximate-runtime-overrides`: keep running unsupported engine-backed variants, but mark them as `approximated`.
- `--fail-on-unsupported-runtime-overrides`: raise immediately instead of emitting unsupported rows.

## Prompt 4 research local deliberation suite

Prompt 10 remains the cheap ablation path. Prompt 4 adds a separate, more explicit research suite that uses structured tasks and clearer pass/fail semantics instead of plain keyword recall.

### What it measures
- exact recall under long-range filler
- branch consensus usefulness
- deep-hierarchy usefulness
- scratch refinement usefulness on divergent prompts
- global-anchor usefulness on long-context summarization
- thought-graph usefulness on multi-step structured reasoning
- quality-per-compute deltas versus the `local_delib_off` baseline

### Run commands

Demo backend:
```bash
python -m scripts.cognition_eval \
  --suite local-delib-research \
  --backend demo \
  --output artifacts/local_delib_research_eval.json
```

Proof-pack path:

```bash
python -m scripts.cognition_eval \
  --suite local-delib-research \
  --backend demo \
  --output artifacts/pretraining_proofs/cpu_mock/local_delib_research.json
```

Optional checkpoint-backed run:
```bash
python -m scripts.cognition_eval \
  --suite local-delib-research \
  --backend engine \
  --source sft \
  --output artifacts/local_delib_research_eval_engine.json
```

Proof-pack engine path:

```bash
python -m scripts.cognition_eval \
  --suite local-delib-research \
  --backend engine \
  --source sft \
  --output artifacts/pretraining_proofs/engine/local_delib_research.json
```

### Artifact fields
- `backend_kind`: `demo`, `engine`, or `external`.
- `metric_tier`: `deterministic_structured` for the demo backend and `structured_prompt_proxy` for engine-backed runs.
- `baseline_variant_id`: current baseline variant used for delta calculations.
- `variant_mean_scores`: mean structured task score per variant.
- `variant_pass_rates`: fraction of cases meeting the case pass threshold per variant.
- `delta_vs_baseline`: mean score delta relative to `baseline_variant_id`.
- `case_scores`: per-case score map by variant.
- `case_deltas_vs_baseline`: per-case delta map by variant.
- `task_family_scores`: family-level means for exact recall, branch, hierarchy, scratch, anchors, and thought graph tasks.
- `quality_per_compute`: structured score divided by compute accounting cost.
- `compute_accounting`: mean executed-step and active-mechanism cost fields per variant.
- `activation_coverage`: expected mechanism activations plus activation/interpretable/format rates per variant.
- `runtime_variant_override_statuses`: per-variant `exact` / `approximated` / `unsupported` map.
- `runtime_variant_override_counts`: aggregate row counts by override status.
- Per-row fields now include:
  - `task_metrics`
  - `activation_checks`
  - `activation_ok`
  - `metrics_interpretable`
  - `response_format_ok`
  - `runtime_override_status`
  - `runtime_override_application_method`
  - `runtime_override_reason`
  - `active_mechanisms`
  - `compute_accounting`

Shared schema note:
- all local-deliberation eval suites now carry the same runtime-override row contract (`runtime_override_applied`, `runtime_override_status`, `runtime_override_application_method`, `runtime_override_reason`) and persist `model_local_delib_graph_artifact` when available, so downstream analysis can compare smoke, advanced, and research artifacts without per-suite override parsing.
- the optional slow smoke test reuses this same row contract when it audits engine-backed proof artifacts, so checkpoint-backed Prompt 9 results stay comparable with the default CPU/mock artifacts rather than inventing a separate schema.

### Interpretation guidance
- Treat `deterministic_structured` demo results as regression-safe harness checks: they tell you whether the eval plumbing, scoring, and activation sanity behave as expected.
- Treat `structured_prompt_proxy` engine results as stronger than keyword heuristics but still not benchmark-grade; the model must follow the requested `KEY=VALUE` format for full interpretability.
- Treat engine-backed rows with `runtime_override_status!=exact` as non-benchmark rows. They are useful for debugging override compatibility, not for claiming mechanism gains.
- Use `activation_coverage.*.activation_ok_rate` before trusting variant gains. A variant that fails to activate should not be interpreted as evidence for or against that mechanism.
- Use `metrics_interpretable_rate` to separate “the variant produced a score” from “the score was both parseable and activation-backed.”
- Read `delta_vs_baseline` together with `activation_coverage` and `runtime_variant_override_statuses`. A positive delta with `activation_ok_rate=0`, `metrics_interpretable_rate=0`, or `runtime_override_status!=exact` is a non-proof row, not evidence that the mechanism worked.
- `quality_per_compute` is still a repo-native proxy, but it is now tied to executed-step fields plus the count of active mechanisms rather than only a rough keyword score heuristic.
- The Prompt 4 suite is still synthetic and repo-native. It is intended to answer “does this mechanism help on the kind of structured task it claims to help on?” without pretending to be an external benchmark leaderboard.

## Natural local-delib benchmark suite

This opt-in suite complements `local-delib-research`. It keeps the same activation telemetry and runtime-truthfulness surfaces, but scores short natural-language answers with task-specific exact graders instead of `KEY=VALUE` response formatting.

### What it measures
- long-context needle recall
- branch consistency and merge selection
- deep hierarchy summarization
- scratch refinement
- anchor-based early/late fact retention
- two-hop thought-graph reasoning

### Run command
```bash
python -m scripts.cognition_eval \
  --suite local-delib-natural \
  --backend engine \
  --source sft \
  --max-problems 6 \
  --seed 42 \
  --output artifacts/pretraining_proofs/engine/local_delib_natural.json
```

### Artifact fields
- `backend_kind`
- `metric_tier`: always `natural_task_grounded`
- `checkpoint_identity`
- `variant_mean_scores`, `variant_pass_rates`
- `proof_variant_mean_scores`, `proof_pass_rates`
- `delta_vs_baseline`, `proof_delta_vs_baseline`
- `case_scores`, `task_family_scores`
- `quality_per_compute`, `compute_accounting`
- `activation_coverage`
- `runtime_variant_override_statuses`, `runtime_variant_override_counts`
- Per-row fields:
  - `grader_extractable`
  - `proof_eligible`
  - `proof_passed`
  - `task_metrics`
  - `activation_checks`
  - `metrics_interpretable`
  - `runtime_override_status`

### Interpretation guidance
- This suite is stronger than `structured_prompt_proxy` because it does not require `KEY=VALUE` formatting, but it is still a small repo-native benchmark pack rather than a broad external leaderboard.
- Use `proof_pass_rates` and `proof_variant_mean_scores` for benchmark claims. Non-exact rows remain in the artifact for debugging but do not count as evidence.
- Read `proof_delta_vs_baseline` together with `activation_coverage` and `runtime_variant_override_statuses`. Gains on non-exact or non-activating rows remain non-proof.

## Prompt 11/12 wrapper trace and hardening notes

When running cognition-wrapper traces on top of engine-backed generation, the wrapper can now surface compact advanced local-deliberation summaries under:
- `model_local_delib.graph_artifact`
- `model_local_delib.thought_summaries.branch_consensus`
- `model_local_delib.thought_summaries.deep_hierarchy`
- `model_local_delib.thought_summaries.scratch`
- `model_local_delib.thought_summaries.thought_graph`
- `model_local_delib.thought_summaries.global_anchors`
- `model_local_delib.thought_summaries.flocking`

`model_local_delib.graph_artifact` is the richest compact object and is organized into:
- `overview`
- `branch`
- `thought_graph`
- `hierarchy`
- `scratch`
- `anchors`
- `compute`
- `flocking`

Use the graph artifact and thought-summary keys as trace payloads alongside the older aggregate buckets (`model_local_delib.branch`, `model_local_delib.hierarchy`, `model_local_delib.scratchpad`, `model_local_delib.adaptive_halt`, and `model_local_delib.scratchpad_summaries`) rather than as replacements.

Eval artifact note:
- local-deliberation ablation rows now persist `model_local_delib_graph_artifact` so downstream JSON analysis can reconstruct compact mechanism activity per variant without reading raw `local_deliberation_stats`

Recommended hardening run order:
- smoke ablation first: `python -m scripts.cognition_eval --suite local-delib-ablation --backend demo --output artifacts/local_delib_ablation_eval.json`
- broaden to the advanced suite next: `python -m scripts.cognition_eval --suite local-delib-ablation-advanced --backend demo --output artifacts/local_delib_ablation_advanced.json`
- only then compare against a real checkpoint-backed engine run if local model/runtime setup is available

Rollback guidance for eval interpretation:
- if `runtime_variant_overrides_applied=false`, inspect `runtime_variant_override_statuses` before using any engine-backed comparison
- treat any row with `runtime_override_status=approximated` as loaded-checkpoint telemetry, not a true architecture swap
- treat any row with `runtime_override_status=unsupported` as an explicit non-result unless you rerun with `--allow-approximate-runtime-overrides`
- if a mechanism destabilizes quality or compute, fall back by disabling its flag/count or by using the lighter smoke suite until the issue is isolated
