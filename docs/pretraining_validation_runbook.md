# Pre-Training Validation Runbook for Added Systems

## Purpose

This runbook defines how to prove the **added systems** in this repository are ready for a training run or deeper research iteration.

Scope:
- wrapper cognition layer under `nanochat/cognition/`
- model-core local deliberation layer under `nanochat/local_deliberation.py` and `nanochat/gpt.py`
- eval/tracing/runtime-truthfulness surfaces tied to those added systems

Out of scope by default:
- tokenizer training/eval
- base model evals outside the added-system path
- web/UI ergonomics
- speedrun or base pretraining flow changes

Use this runbook together with:
- [codex_test_prompt_pack.md](../codex_test_prompt_pack.md)
- [docs/evals.md](./evals.md)

## Evidence Model

Each feature family needs two kinds of proof:

### `Works`

The feature is mechanically correct.

Typical signals:
- deterministic unit/integration tests pass
- causality/parity/boundedness/shape invariants hold
- negative cases fail explicitly instead of silently
- artifacts contain the expected keys and metadata

### `Useful`

The feature produces a meaningful benefit or at least reliable evidence of its own participation.

Typical signals:
- baseline vs enhanced deltas on deterministic eval cases
- activation telemetry shows the mechanism actually ran
- compute accounting changes when adaptive compute is enabled
- traces explain why the final choice improved
- runtime override status proves whether a claimed engine-backed variant was exact

Do not treat a feature as “ready” if it only has one of these.

## Standard Artifact Layout

Recommended output paths:

```text
artifacts/
  pretraining_proofs/
    cpu_mock/
      cognition_eval.json
      local_delib_ablation.json
      local_delib_ablation_advanced.json
      local_delib_research.json
    engine/
      cognition_eval.json
      local_delib_ablation_advanced.json
      local_delib_research.json
      task_grounded.json
      local_delib_natural.json
      engine_smoke_manifest.json
```

The repo does not need these directories checked in. Create them only when running the proof commands.

Recommended setup command:

```bash
mkdir -p artifacts/pretraining_proofs/cpu_mock artifacts/pretraining_proofs/engine
```

## Feature Inventory and Proof Mapping

| Feature family | Correctness proof | Useful/evidence proof | Prompt | Primary tests/artifacts |
|---|---|---|---|---|
| Schemas and timestamps | dataclass/default/timestamp tests | trace/routing payload realism | 1 | `tests/test_cognition_schemas.py` |
| Normalization helpers | term extraction and alias handling | prove retrieval inputs normalize consistently | 1 | `tests/test_cognition_normalize.py` |
| Episodic memory | write/search/retrieve/ranking tests | matched-term + score evidence | 1 | `tests/test_cognition_memory.py` |
| Semantic memory | semantic-only writes, replacement, ranking | relevance-over-recency proof | 1 | `tests/test_cognition_memory.py` |
| Router | explicit route trigger tests | prove no accidental over-routing | 1 | `tests/test_cognition_router.py` |
| Skill registry | discover/best_for ranking | later query reuse evidence | 1 | `tests/test_cognition_skills.py` |
| Consolidator | repeated-win skill creation | provenance survives into skill + semantic memory | 1 | `tests/test_cognition_consolidation.py` |
| Creative workspace | strategy planning/composition tests | diverse strategy ids and model-summary-aware planning | 2 | `tests/test_cognition_creative.py` |
| Verifier | ranking/repair tests | grounded candidates beat noisy ones | 2 | `tests/test_cognition_verifier.py` |
| Sandbox | branch scoring/selection tests | exploration can change the winner for justified reasons | 2 | `tests/test_cognition_sandbox.py` |
| Trace recorder | deep-copy and metadata persistence tests | compact evidence payloads survive mutation | 2 | `tests/test_cognition_traces.py` |
| Cognition agent | route/orchestration/prompt-injection tests | support injection changes output and traces explain why | 3 | `tests/test_cognition_agent.py`, `tests/test_cognition_prompt_composition.py` |
| Backend adapter / EngineBackend | prompt serialization and metadata contract tests | graph artifact + runtime truthfulness evidence | 4 | `tests/test_cognition_backend.py` |
| Graph artifact / thought summaries | compact section surfacing tests | enough metadata to reconstruct active mechanisms | 4, 7 | `tests/test_cognition_backend.py`, eval artifacts |
| Runtime override reporting | exact/approx/unsupported tests | disqualify non-exact engine rows from proof claims | 4, 5, 9 | backend/eval tests and engine artifacts |
| Default cognition eval | row/summary writer tests | cognition-vs-baseline gain on memory/skill cases | 5 | `tests/test_cognition_eval.py`, `tests/test_cognition_eval_usefulness.py`, `cognition_eval.json` |
| Local-delib core | causality/parity/shape tests | semantic/consensus/halt telemetry evidence | 6 | `tests/test_local_deliberation.py` |
| Neighbor graph + flocking | graph/flocking causal and stats tests | activation stats + compute evidence | 7 | `tests/test_local_deliberation.py`, advanced artifacts |
| Branching + consensus/verifier | branch path tests | disagreement/consensus/verifier telemetry | 7 | `tests/test_local_deliberation.py`, advanced/research artifacts |
| Deep hierarchy | cross-scale flow tests | hierarchy-depth and message stats evidence | 7 | `tests/test_local_deliberation.py`, advanced/research artifacts |
| Structured scratch | refine/reset/input-path tests | scratch summary/export evidence | 7 | `tests/test_local_deliberation.py`, backend/eval artifacts |
| Thought graph | budget/cache/fallback tests | thought-node telemetry + research-case evidence | 7, 8 | `tests/test_local_deliberation.py`, `tests/test_engine_local_deliberation.py`, research artifacts |
| Global anchors | anchor-path tests | long-range state evidence | 7 | `tests/test_local_deliberation.py`, advanced/research artifacts |
| Auxiliary losses | zero/finite/disabled behavior tests | prove metrics are meaningful only when mechanism is active | 7 | `tests/test_local_deliberation.py` |
| GPT wiring / config audit | GPTConfig and `base_train` wiring tests | prevent silent pre-training config drift | 8 | `tests/test_gpt_local_deliberation.py` |
| KV-cache / decode-cache | parity and bounded-cache tests | justify faster decode paths with correctness evidence | 8 | `tests/test_gpt_local_deliberation.py`, `tests/test_engine_local_deliberation.py` |
| Engine integration | engine batch/decode smoke tests | engine-level metadata and cache evidence | 8 | `tests/test_engine_local_deliberation.py` |
| Task-grounded benchmarks | task-native pass/fail aggregation and artifact tests | real-task baseline vs cognition parity/regression evidence | 17 | `tests/test_cognition_eval.py`, `task_grounded.json` |
| Natural local-delib benchmarks | exact grader + proof-filter tests | non-proxy mechanism evidence without `KEY=VALUE` formatting | 17 | `tests/test_cognition_eval.py`, `tests/test_cognition_eval_usefulness.py`, `local_delib_natural.json` |
| Optional checkpoint smoke | slow engine-backed artifact tests | exact/approx/unsupported statuses on real checkpoint runs | 9 | engine JSON artifacts, optional slow tests |

## Required Default Gate: CPU/Mock First

This is the default proof path to complete before a training run.

### Stage 0 — Audit

Prompt:
- `Prompt 0`

Output:
- gap map only

No tests should be written yet in this stage.

### Stage 1 — Foundational cognition contracts

Prompt:
- `Prompt 1`

Run:

```bash
python -m pytest -q \
  tests/test_cognition_schemas.py \
  tests/test_cognition_normalize.py \
  tests/test_cognition_memory.py \
  tests/test_cognition_router.py \
  tests/test_cognition_skills.py \
  tests/test_cognition_consolidation.py
```

Pass condition:
- foundational cognition contracts are directly covered
- retrieval/skill/consolidation evidence is explicit in assertions

### Stage 2 — Creative/verifier/sandbox/traces

Prompt:
- `Prompt 2`

Run:

```bash
python -m pytest -q \
  tests/test_cognition_creative.py \
  tests/test_cognition_verifier.py \
  tests/test_cognition_sandbox.py \
  tests/test_cognition_traces.py
```

Pass condition:
- helper subsystems have direct coverage
- usefulness heuristics are tested, not only happy-path construction

### Stage 3 — Cognition agent end-to-end

Prompt:
- `Prompt 3`

Run:

```bash
python -m pytest -q \
  tests/test_cognition_agent.py \
  tests/test_cognition_prompt_composition.py
```

Pass condition:
- prompt injection is direct and inspectable
- traces explain the selected path and outcome

### Stage 4 — Backend metadata and runtime truthfulness

Prompt:
- `Prompt 4`

Run:

```bash
python -m pytest -q tests/test_cognition_backend.py tests/test_cognition_eval.py -k 'backend or runtime_override or graph_artifact'
```

Pass condition:
- compact metadata contract is stable
- exact vs approximated vs unsupported is explicit

### Stage 5 — Eval usefulness proofs

Prompt:
- `Prompt 5`

Runs:

```bash
python -m pytest -q tests/test_cognition_eval.py tests/test_cognition_eval_usefulness.py
python -m scripts.cognition_eval --backend demo --output artifacts/pretraining_proofs/cpu_mock/cognition_eval.json
python -m scripts.cognition_eval --suite local-delib-ablation --backend demo --output artifacts/pretraining_proofs/cpu_mock/local_delib_ablation.json
python -m scripts.cognition_eval --suite local-delib-ablation-advanced --backend demo --output artifacts/pretraining_proofs/cpu_mock/local_delib_ablation_advanced.json
python -m scripts.cognition_eval --suite local-delib-research --backend demo --output artifacts/pretraining_proofs/cpu_mock/local_delib_research.json
```

Pass condition:
- default cognition eval shows the expected context-sensitive gains
- advanced/research artifacts contain activation and compute evidence
- no proof claim relies on ambiguous runtime-override rows

### Stage 6 — Core local-delib correctness

Prompt:
- `Prompt 6`

Run:

```bash
python -m pytest -q tests/test_local_deliberation.py -k 'semantic or consensus or halt or causal or identity'
```

Pass condition:
- causality/parity/core activation stats are covered

### Stage 7 — Advanced local-delib mechanisms

Prompt:
- `Prompt 7`

Run:

```bash
python -m pytest -q \
  tests/test_local_deliberation.py \
  tests/test_cognition_backend.py \
  tests/test_cognition_eval.py \
  -k 'flocking or branch or hierarchy or scratch or thought or anchor or aux'
python -m scripts.cognition_eval --suite local-delib-ablation-advanced --backend demo --output artifacts/pretraining_proofs/cpu_mock/local_delib_ablation_advanced.json
python -m scripts.cognition_eval --suite local-delib-research --backend demo --output artifacts/pretraining_proofs/cpu_mock/local_delib_research.json
```

Pass condition:
- every advanced mechanism has both direct correctness tests and artifact-level evidence

### Stage 8 — GPT and Engine integration

Prompt:
- `Prompt 8`

Run:

```bash
python -m pytest -q tests/test_gpt_local_deliberation.py tests/test_engine_local_deliberation.py
```

Pass condition:
- config wiring is audited
- decode-cache fast paths match full-forward expectations
- engine integration is proven on tiny deterministic setups

## Optional Engine/Checkpoint Smoke Gate

Run this only if local checkpoints and runtime are available.

Prompt:
- `Prompt 9`

Suggested runs:

```bash
python -m scripts.cognition_eval --backend engine --source sft --no-enforce-improvement --output artifacts/pretraining_proofs/engine/cognition_eval.json
python -m scripts.cognition_eval --suite local-delib-ablation-advanced --backend engine --source sft --output artifacts/pretraining_proofs/engine/local_delib_ablation_advanced.json
python -m scripts.cognition_eval --suite local-delib-research --backend engine --source sft --output artifacts/pretraining_proofs/engine/local_delib_research.json
python -m scripts.cognition_eval --suite task-grounded --backend engine --source sft --tasks GSM8K,SpellingBee,HumanEval --max-problems 5 --seed 42 --output artifacts/pretraining_proofs/engine/task_grounded.json
python -m scripts.cognition_eval --suite local-delib-natural --backend engine --source sft --max-problems 6 --seed 42 --output artifacts/pretraining_proofs/engine/local_delib_natural.json
python -m scripts.chat_cli --cognition --source sft --prompt "Summarize why exact runtime override reporting matters."
python -m pytest -q tests/test_cognition_engine_smoke.py -m slow
```

Rules:
- any slow smoke tests must be marked `slow`
- skip cleanly when checkpoint discovery fails
- engine-backed rows with `runtime_override_status!=exact` are debugging rows, not proof rows

Suggested environment overrides:
- set `NANOCHAT_BASE_DIR` if checkpoints/tokenizer are stored outside the default `~/.cache/nanochat`
- set `NANOCHAT_SMOKE_SOURCE`, `NANOCHAT_SMOKE_MODEL_TAG`, and `NANOCHAT_SMOKE_STEP` when you want the slow smoke test pinned to one exact checkpoint
- set `NANOCHAT_SMOKE_DEVICE_TYPE=cpu|cuda|mps` to match the runtime you actually want to smoke
- set `NANOCHAT_SMOKE_ARTIFACT_DIR` if you do not want the slow smoke artifacts under `artifacts/pretraining_proofs/engine/`
- use `--no-enforce-improvement` for the engine-backed cognition command when you need transport-smoke evidence from an arbitrary local checkpoint rather than a support-sensitive usefulness claim
- the dedicated slow smoke test uses a built-in offline `SmokeTinyTask` for its task-grounded artifact so the engine audit can run without task dataset downloads

Pass condition:
- optional engine-backed artifacts are present
- exact vs approximated vs unsupported rows are visible and auditable
- `engine_smoke_manifest.json` records the checkpoint identity, commands, artifacts, and skip/fail reason for the smoke run

## Stop/Go Criteria Before Training

Do not claim the added systems are ready for a training run until:
- the default CPU/mock gate passes through Stage 8
- each feature family has both a `works` proof and a `useful` or evidence proof
- eval artifacts exist under `artifacts/pretraining_proofs/cpu_mock/`
- no engine-backed claim depends on non-exact runtime overrides

It is acceptable to defer the optional engine smoke gate if:
- local checkpoints are unavailable
- the goal is only to prove repo-native correctness and deterministic usefulness before additional implementation
- stronger task-grounded and natural-language benchmark artifacts are not required for the default CPU/mock gate

## Documentation Discipline

After each prompt execution:
- update `documentation.md`
- record files changed
- record commands run
- record results
- record what remains

If a prompt reveals a runtime bug, fix the bug in the smallest clean slice before broadening scope.
