# Codex Test Prompt Pack — Pre-Training Proof for Added Systems

This pack is designed for the current state of the `nanochat` repository.

It assumes the repo already contains:
- the wrapper cognition subsystem under `nanochat/cognition/`
- the model-core local deliberation stack in `nanochat/local_deliberation.py` and `nanochat/gpt.py`
- eval entrypoints in `scripts/cognition_eval.py`
- existing lightweight tests in `tests/test_cognition_*.py`, `tests/test_local_deliberation.py`, `tests/test_gpt_local_deliberation.py`, and `tests/test_engine_local_deliberation.py`

The purpose of this pack is to create a **full pre-training proof suite** for the added systems only.

That means:
- prove each added feature works
- prove each added feature produces useful evidence or measurable benefit
- keep the default path CPU/mock-first
- keep checkpoint-backed engine smoke optional
- avoid changing training, speedrun, or base pretraining flow

This pack is for **creating tests and proof commands**, not for refactoring runtime behavior.

---

## How to use this pack

Recommended order:
1. Run **Prompt 0** in ask/planning mode first.
2. Execute **Prompts 1–8** one at a time in code mode.
3. Use **Prompt 9** only when you have a local checkpoint and want optional engine-backed smoke coverage.
4. After each prompt, update `documentation.md` with what changed, what passed, and what remains.

General rules for every prompt in this pack:
- preserve existing repo structure
- do not migrate to `src/`
- do not change `runs/`, `scripts/base_train.py`, or the default training path
- keep tests CPU-friendly by default
- prefer fake backends, tiny GPT configs, fake flash attention, and deterministic fixtures
- expand existing direct suites where they already fit
- add new focused suites where the current coverage is indirect
- every feature must get:
  - a correctness proof
  - a usefulness or evidence proof
- checkpoint-backed smoke is optional, slow, and must be clearly separated from the default gate

---

## Prompt 0 — Audit the current proof surface before writing tests

```text
You are working inside the existing nanochat repository.

Do not write code yet. Inspect and plan the proof surface first.

Read these files first:
- README.md
- documentation.md
- plans.md
- docs/architecture.md
- docs/evals.md
- nanochat/cognition/schemas.py
- nanochat/cognition/normalize.py
- nanochat/cognition/memory.py
- nanochat/cognition/router.py
- nanochat/cognition/creative.py
- nanochat/cognition/verifier.py
- nanochat/cognition/sandbox.py
- nanochat/cognition/consolidation.py
- nanochat/cognition/skills.py
- nanochat/cognition/traces.py
- nanochat/cognition/backend.py
- nanochat/cognition/agent.py
- nanochat/cognition/eval.py
- nanochat/local_deliberation.py
- nanochat/gpt.py
- nanochat/engine.py
- scripts/cognition_eval.py
- tests/test_cognition_*.py
- tests/test_local_deliberation.py
- tests/test_gpt_local_deliberation.py
- tests/test_engine_local_deliberation.py

Goal:
Produce a concrete proof-gap map before adding any tests.

Required output:
A. Current coverage by feature family and by existing test file
B. Missing direct test suites that should be added
C. Feature -> correctness proof -> usefulness/evidence proof mapping
D. CPU/mock-first command list and optional engine-smoke command list
E. Exact files to create or expand in Prompts 1–9

Rules:
- stay within the added systems only
- do not expand scope to tokenizer/base-eval/chat-web unless an added system depends on them directly
- prefer expanding current cognition/local-delib suites unless the current coverage is indirect or overloaded
- recommend new focused suites for creative/verifier/sandbox/skills/traces if direct coverage is still missing

Do not write code in this response.
```

---

## Prompt 1 — Foundational cognition contracts: schemas, normalization, memory, router, skills, consolidation

```text
Implement the foundational correctness and evidence tests for the wrapper cognition layer.

Files to inspect:
- nanochat/cognition/schemas.py
- nanochat/cognition/normalize.py
- nanochat/cognition/memory.py
- nanochat/cognition/router.py
- nanochat/cognition/skills.py
- nanochat/cognition/consolidation.py
- tests/test_cognition_schemas.py
- tests/test_cognition_memory.py
- tests/test_cognition_router.py
- tests/test_cognition_consolidation.py

Test files to create or expand:
- expand tests/test_cognition_schemas.py
- create tests/test_cognition_normalize.py
- expand tests/test_cognition_memory.py
- expand tests/test_cognition_router.py
- create tests/test_cognition_skills.py
- expand tests/test_cognition_consolidation.py

Preferred fixtures:
- synthetic Episode, MemoryItem, SkillArtifact instances only
- explicit created_at timestamps instead of real-time sleeps
- deterministic text fixtures with paraphrases, aliases, punctuation differences, and repeated wins

Required correctness cases:
1. Schemas
- dataclass fields exist and keep expected defaults
- timestamp helpers produce ISO timestamps
- trace/routing/verification/skill artifacts accept realistic metadata payloads

2. Normalization
- normalization is case-insensitive
- punctuation and repeated whitespace do not change core term extraction
- alias-like text variants still share overlapping normalized terms
- unique_terms removes duplicates while preserving useful tokens

3. Episodic memory
- write -> search -> ranked retrieve path works
- combined ranking reflects both relevance and recency
- paraphrased queries still retrieve the right episode
- empty or non-overlapping queries return no matches
- recent(limit=...) order is newest-first

4. Semantic memory
- only semantic items are accepted
- writes replace previous items with the same id
- retrieve prefers relevance over stale but weak matches
- retrieve returns ranked metadata with stable ordering

5. Router
- routes memory, creative, verify, sandbox, and consolidate triggers correctly
- defaults to direct_answer when no trigger matches
- empty query path stays explicit and low-confidence
- add negative tests to prove unrelated wording does not misroute into advanced modes

6. Skills
- discover() ranks skills by overlap with trigger/name/procedure terms
- best_for() returns the top match only when overlap exists
- no-match path returns None / empty cleanly
- ordering is deterministic for overlapping skills

7. Consolidation
- repeated successful episodes create one skill artifact
- semantic memory stores a semantic artifact with provenance
- insufficient repetition does not create a skill
- failed episodes are ignored
- missing trigger/strategy data prevents accidental consolidation

Required usefulness/evidence cases:
- memory ranking tests must prove later retrieval is not random: assert matched terms, relevance, recency, and combined score behave as expected
- skill tests must prove a later query can discover and reuse a previously consolidated pattern
- consolidation tests must prove provenance survives into both the skill artifact and semantic memory record

Validation commands after implementation:
- python -m pytest -q tests/test_cognition_schemas.py tests/test_cognition_normalize.py tests/test_cognition_memory.py tests/test_cognition_router.py tests/test_cognition_skills.py tests/test_cognition_consolidation.py

Deliverables:
- new/expanded foundational tests
- no runtime changes unless a test reveals a real bug
- documentation.md updated with commands and results
```

---

## Prompt 2 — Focused direct suites for creative workspace, verifier, sandbox, and traces

```text
Add direct tests for the cognition helper subsystems that are currently only partly exercised through end-to-end agent coverage.

Files to inspect:
- nanochat/cognition/creative.py
- nanochat/cognition/verifier.py
- nanochat/cognition/sandbox.py
- nanochat/cognition/traces.py
- nanochat/cognition/backend.py
- tests/test_cognition_agent.py

Test files to create:
- tests/test_cognition_creative.py
- tests/test_cognition_verifier.py
- tests/test_cognition_sandbox.py
- tests/test_cognition_traces.py

Preferred fixtures/fakes:
- a deterministic fake backend that returns a different response per strategy id
- an optional fake backend metadata payload through last_generation_metadata
- synthetic support_profile dicts
- a small set of candidate responses with intentionally different grounding/diversity properties

Required correctness cases:
1. Creative workspace planning
- route-only planning picks conservative + divergent strategies where appropriate
- memory-heavy support adds memory_grounded
- branch disagreement / branch consensus metadata adds branch_resolution
- scratch / hierarchy / thought / anchor summary signals add recombination
- strategy order is deduplicated and limited correctly

2. Creative prompt composition
- generated prompts contain the base prompt
- strategy section fields are explicit
- support-profile section only appears when relevant
- model-deliberation guidance only appears when model summary is present
- candidate metadata records model_summary_used and model_focus consistently

3. Verifier
- empty candidate is scored as unusable
- relevance/usefulness/diversity/repairability/strategy-fit all affect rank
- verify route can require repair when grounding is weak
- ranking order is deterministic for the same inputs

4. Sandbox
- sandbox scores branches using relevance, verifier score, support overlap, and branch bonus
- divergent/branch-resolution strategies can receive branch bonus
- selected outcome is the highest score
- empty shortlist returns no selected outcome cleanly

5. Trace recorder
- build() increments trace ids
- metadata is deep-copied so later mutation does not leak into stored traces
- nested dict/list payloads are copied safely

Required usefulness/evidence cases:
- creative tests must prove the workspace explores genuinely different strategy ids instead of emitting duplicate drafts with new labels
- verifier tests must prove a more grounded candidate outranks a merely verbose one
- sandbox tests must prove exploration can change the final choice relative to pure verifier ordering when branch bonus/support evidence makes that justified
- trace tests must prove all evidence payloads are preserved in a stable debugging form

Validation commands after implementation:
- python -m pytest -q tests/test_cognition_creative.py tests/test_cognition_verifier.py tests/test_cognition_sandbox.py tests/test_cognition_traces.py

Deliverables:
- four focused direct suites
- no broad agent-file overload for helper-only behavior
- documentation.md updated
```

---

## Prompt 3 — Cognition agent orchestration, prompt injection, and end-to-end traceability

```text
Expand the end-to-end cognition proof around the CognitionAgent and its prompt-composition path.

Files to inspect:
- nanochat/cognition/agent.py
- nanochat/cognition/backend.py
- nanochat/cognition/creative.py
- nanochat/cognition/verifier.py
- nanochat/cognition/sandbox.py
- nanochat/cognition/memory.py
- nanochat/cognition/skills.py
- nanochat/cognition/traces.py
- tests/test_cognition_agent.py

Test files to create or expand:
- expand tests/test_cognition_agent.py
- create tests/test_cognition_prompt_composition.py

Preferred fixtures/fakes:
- a capturing backend that stores the final prompt
- a strategy-aware backend that returns different outputs for different strategy sections
- a metadata-rich backend that exposes local_delib graph/thought summary/runtime-override payloads
- preseeded episodic memory, semantic memory, and skill registry state

Required correctness cases:
1. Prompt composition
- semantic memory section is injected when semantic hits exist
- episodic memory section is injected for paraphrased ordinary queries when support is selected
- relevant skill section is injected with id/name/trigger/procedure
- section ordering stays stable: skill/semantic/episodic support before the final user request
- omission rules hold when no support exists

2. Routing and orchestration
- direct_answer and retrieve_memory paths use backend.run directly
- creative path runs creative -> verifier
- sandbox path runs creative -> verifier -> sandbox and records shortlist/outcomes
- explicit consolidate path returns user-visible consolidation result when a repeated pattern exists
- auto-consolidation after successful episodes is recorded without breaking the response path

3. Traceability
- trace steps record route, creative strategies, candidate counts, verifier score, repair reason, sandbox branch count, episodic hit count, semantic hit count, and skill reuse when present
- trace metadata includes creative workspace, verifier, sandbox, retrieved ids, and confidence
- backend local deliberation metadata and runtime override metadata are preserved verbatim enough for downstream debugging

4. Negative cases
- empty query stays explicit and does not crash
- no repeated pattern returns a clean consolidation miss
- missing support sources do not create empty prompt sections
- creative path still behaves sensibly when backend metadata is absent

Required usefulness/evidence cases:
- prove support injection changes backend output deterministically by using a backend that only improves when episodic/semantic/skill context is actually inserted into the prompt
- prove traces explain why a final candidate won by asserting selected/rejected candidate ids, selected strategy id, handoff type, and ranked verifier data
- prove runtime-override metadata is visible end-to-end so later engine-backed gains can be judged truthfully

Validation commands after implementation:
- python -m pytest -q tests/test_cognition_agent.py tests/test_cognition_prompt_composition.py

Deliverables:
- stronger end-to-end agent proofs
- focused prompt-composition tests separated from the broad agent file
- documentation.md updated
```

---

## Prompt 4 — Backend adapter, graph artifacts, thought summaries, and runtime truthfulness

```text
Strengthen direct proof around the EngineBackend and metadata surfacing layer.

Files to inspect:
- nanochat/cognition/backend.py
- nanochat/cognition/agent.py
- nanochat/cognition/eval.py
- nanochat/gpt.py
- nanochat/engine.py
- tests/test_cognition_backend.py
- tests/test_cognition_eval.py

Test files to create or expand:
- expand tests/test_cognition_backend.py
- expand tests/test_cognition_eval.py only where eval rows depend on backend metadata contract

Preferred fixtures/fakes:
- fake tokenizer
- fake engine with prompt capture
- tiny GPT/Engine setup with fake flash attention when a real model path is needed
- fake runtime override backends covering exact, approximated, unsupported, and mixed rows

Required correctness cases:
1. BackendAdapter
- passes default kwargs and call-time kwargs correctly

2. EngineBackend prompt path
- chat serialization uses system + user messages correctly
- extra generation kwargs still pass through
- decoded response strips prompt prefix correctly

3. Metadata surfacing
- namespaced `model_local_delib.*` metadata appears only when local deliberation stats exist
- graph artifact includes only active sections
- compact thought summaries appear only for mechanisms with real activity
- scratchpad summary export remains backward compatible with older keys

4. Runtime override truthfulness
- exact runtime override returns `exact` status and applied_overrides metadata
- unsupported override raises or emits unsupported rows according to caller path
- approximation path stays explicit and never masquerades as exact
- unknown override keys fail clearly
- non-mutable engine model/config surfaces unsupported status rather than silent success

5. Creative-policy summary extraction
- branch/scratch/hierarchy/thought/anchor/compute signals collapse into the expected compact summary used by the wrapper creativity policy

Required usefulness/evidence cases:
- prove graph artifacts are compact but still reconstructable enough for debugging by asserting overview, per-section summaries, and layer rows
- prove runtime override metadata is sufficient to disqualify non-exact rows from benchmark claims
- prove inactive mechanisms do not create noisy empty sections that would confuse downstream evidence parsing

Validation commands after implementation:
- python -m pytest -q tests/test_cognition_backend.py tests/test_cognition_eval.py -k 'backend or runtime_override or graph_artifact'

Deliverables:
- stronger backend metadata and truthfulness coverage
- no change to generation return contracts
- documentation.md updated
```

---

## Prompt 5 — Cognition eval harness and usefulness proofs

```text
Turn the evaluation layer into a stronger proof surface for “useful before training” claims.

Files to inspect:
- nanochat/cognition/eval.py
- scripts/cognition_eval.py
- docs/evals.md
- tests/test_cognition_eval.py

Test files to create or expand:
- expand tests/test_cognition_eval.py
- create tests/test_cognition_eval_usefulness.py

Preferred fixtures/fakes:
- deterministic context-aware backend
- deterministic local-delib context backend
- mixed runtime-override backend
- fake rows/artifacts with exact and unsupported statuses

Required correctness cases:
1. Default cognition eval
- baseline vs cognition rows are produced
- route histogram is populated
- strict failure trips when a required cognition-gain case does not improve
- creative metadata fields are preserved in rows and summary artifacts

2. Local-delib ablation eval
- every advertised variant appears in summary rows
- branch/hierarchy/scratch/adaptive-halt buckets are populated correctly
- graph artifact is persisted in row artifacts

3. Advanced ablation eval
- all advanced variants appear
- compute proxy metrics, neighbor/branch/hierarchy/scratch/thought/flocking/anchor sections are written
- per-variant runtime override statuses aggregate conservatively

4. Research eval
- structured fields parse cleanly
- pass/fail and response_format_ok behave correctly
- activation checks and activation_ok are computed
- metrics_interpretable responds to parseability + activation evidence

5. Artifact writers
- default, local-delib, advanced, and research artifact writers emit stable JSON top-level keys

Required usefulness/evidence cases:
- default eval cases must prove cognition helps only when memory/semantic/skill context is injected, not because the baseline backend was already enough
- advanced ablation tests must prove the artifact contains enough compute and activation data to judge whether a mechanism was active, not just whether a score changed
- research suite tests must prove variant deltas are paired with activation_coverage so a “better score” without activation evidence is not accepted as real proof
- override tests must prove `approximated` and `unsupported` rows are visible as non-proof rows

Target artifact commands after implementation:
- python -m scripts.cognition_eval --backend demo --output artifacts/pretraining_proofs/cpu_mock/cognition_eval.json
- python -m scripts.cognition_eval --suite local-delib-ablation --backend demo --output artifacts/pretraining_proofs/cpu_mock/local_delib_ablation.json
- python -m scripts.cognition_eval --suite local-delib-ablation-advanced --backend demo --output artifacts/pretraining_proofs/cpu_mock/local_delib_ablation_advanced.json
- python -m scripts.cognition_eval --suite local-delib-research --backend demo --output artifacts/pretraining_proofs/cpu_mock/local_delib_research.json

Validation commands after implementation:
- python -m pytest -q tests/test_cognition_eval.py tests/test_cognition_eval_usefulness.py

Deliverables:
- stronger usefulness/evidence tests
- explicit artifact expectations for the pre-training gate
- documentation.md updated
```

---

## Prompt 6 — Local deliberation core correctness: causality, parity, semantic top-k, phrase consensus, adaptive halt

```text
Expand the model-core proof surface for the local deliberation block’s core mechanisms.

Files to inspect:
- nanochat/local_deliberation.py
- tests/test_local_deliberation.py

Test files to create or expand:
- expand tests/test_local_deliberation.py

Preferred fixtures:
- tiny deterministic tensors
- direct module construction with explicit weight overrides when needed
- helper configuration functions for branch/thought/hierarchy/scratch/anchor modules only when the case requires them

Required correctness cases:
1. Base structural behavior
- output shapes match input shapes
- causal depthwise mixer never leaks future tokens
- near-identity-at-init remains intact

2. Semantic top-k path
- disabled path matches default behavior
- enabled path surfaces stats and respects top-k + lookback bounds
- semantic neighbor search remains strictly causal

3. Phrase consensus path
- phrase pool shapes and broadcast are correct
- consensus score is surfaced when enabled
- local chunk effects stay more local than distant-token effects

4. Adaptive halt
- disabled path matches default behavior
- all-halt, no-halt, and mixed-halt token patterns are supported
- mean_steps_taken / halted_token_fraction / related compute stats stay finite and bounded

5. Negative/guard cases
- invalid kernel/chunk sizes raise clearly where constructors already enforce them
- no mechanism should add non-finite stats or shape drift

Required usefulness/evidence cases:
- each core mechanism must surface explicit numeric stats that later evals can use as activation evidence
- adaptive halt tests must prove compute-related fields actually move when token halting behavior changes
- semantic and consensus tests must prove the surfaced stats correspond to real activation rather than placeholder zeros

Validation commands after implementation:
- python -m pytest -q tests/test_local_deliberation.py -k 'semantic or consensus or halt or causal or identity'

Deliverables:
- stronger core local-delib correctness coverage
- no GPT/engine changes unless a real bug is revealed
- documentation.md updated
```

---

## Prompt 7 — Advanced local deliberation proofs: flocking, branch consensus/verifier, deep hierarchy, scratch, thought graph, anchors, aux losses

```text
Expand the proof surface for the advanced model-core mechanisms and their evidence hooks.

Files to inspect:
- nanochat/local_deliberation.py
- nanochat/cognition/backend.py
- nanochat/cognition/eval.py
- tests/test_local_deliberation.py
- tests/test_cognition_backend.py
- tests/test_cognition_eval.py

Test files to create or expand:
- expand tests/test_local_deliberation.py
- expand tests/test_cognition_backend.py where graph-artifact / thought-summary surfacing depends on advanced stats
- expand tests/test_cognition_eval.py where advanced eval rows depend on those stats

Required correctness cases:
1. Neighbor graph + flocking
- neighbor graph disabled path matches the simpler semantic path
- enabled graph surfaces neighbor counts and respects causal constraints
- flocking off matches graph-only behavior
- flocking on surfaces alignment/cohesion/separation stats and remains causal
- radius cap favors near neighbors over far ones

2. Branching + consensus + verifier
- branching off matches default path
- branching on surfaces branch stats and preserves shape
- consensus/verifier path activates only when configured
- disagreement threshold / max active controls are respected

3. Deep hierarchy
- disabled path matches legacy/default behavior
- enabled path surfaces phrase/span/sequence stats
- cross-scale information flow reaches far tokens without violating causality
- scale gates and feedback norms stay finite

4. Structured scratch
- disabled path matches default path
- enabled path surfaces read/write/refine/summary stats
- scratch can influence the internal update path when configured
- scratch resets cleanly between calls
- branch/hierarchy-to-scratch inputs stay causal

5. Thought graph
- disabled path matches default path
- enabled path surfaces node/degree/write/read/update stats
- node budget stays bounded
- incremental feedback matches full feedback for the new token
- decode cache matches full stack across multiple decode steps
- fallback only occurs for the documented budget-window slide case

6. Global anchors
- disabled path matches default path
- enabled path surfaces read/write stats
- optional hierarchy/scratch/thought summaries can feed anchors when configured
- anchor path stays causal

7. Auxiliary losses
- zero-weight paths behave like base loss
- branch/consensus/scratch/flocking/thought/hierarchy/anchor auxiliary losses remain finite when enabled
- aux-loss keys disappear or zero out when the mechanism is disabled

Required usefulness/evidence cases:
- graph artifact and thought-summary tests must prove advanced mechanisms appear in compact metadata only when active
- advanced eval rows must have enough telemetry to judge activation, compute cost, and mechanism participation
- adaptive-compute plus mechanism-count accounting must change in the expected direction for combo variants

Validation commands after implementation:
- python -m pytest -q tests/test_local_deliberation.py tests/test_cognition_backend.py tests/test_cognition_eval.py -k 'flocking or branch or hierarchy or scratch or thought or anchor or aux'
- python -m scripts.cognition_eval --suite local-delib-ablation-advanced --backend demo --output artifacts/pretraining_proofs/cpu_mock/local_delib_ablation_advanced.json
- python -m scripts.cognition_eval --suite local-delib-research --backend demo --output artifacts/pretraining_proofs/cpu_mock/local_delib_research.json

Deliverables:
- stronger advanced mechanism proofs
- explicit evidence that activation telemetry is meaningful
- documentation.md updated
```

---

## Prompt 8 — GPT wiring, base-train config surface, KV-cache, decode-cache, engine integration, batch expansion

```text
Prove the advanced local-deliberation stack stays wired correctly through GPT and Engine integration.

Files to inspect:
- nanochat/gpt.py
- nanochat/local_deliberation.py
- nanochat/engine.py
- scripts/base_train.py
- tests/test_gpt_local_deliberation.py
- tests/test_engine_local_deliberation.py

Test files to create or expand:
- expand tests/test_gpt_local_deliberation.py
- expand tests/test_engine_local_deliberation.py

Preferred fixtures/fakes:
- fake flash attention monkeypatches
- TinyTokenizer
- tiny GPTConfig instances
- KVCache prefill + decode scenarios, including batch expansion from prefill cache

Required correctness cases:
1. GPT config and wiring
- `GPTConfig.local_delib_*` surface matches `scripts/base_train.py` parser/build wiring
- advanced fields are passed into LocalDeliberationBlock construction correctly
- block creation rules only instantiate modules when the config requires them

2. GPT forward path
- forward works with local_delib disabled and enabled
- advanced fields surface stats through GPT metadata
- near-identity-at-init holds for both basic and advanced stacks
- no future-token influence in semantic, halt, branch, scratch, anchor, and combo paths

3. KV-cache and decode cache
- kv cache can bypass or continue local deliberation correctly
- decode cache is populated under local-delib paths
- decode cache parity matches full forward across multiple steps
- thought-graph continuation path matches full forward when the cached window stays valid
- documented fallback path is the only fallback
- prefill cache can expand into a larger decode batch while preserving local-delib extra state
- cache growth stays bounded by the configured budgets

4. Engine integration
- model.generate and Engine.generate_batch work with local_delib enabled
- engine decode path works with adaptive halt, branching, branch consensus, and thought graph enabled
- engine-backed cache state contains the expected extra_caches sections

Required usefulness/evidence cases:
- GPT/Engine tests must prove surfaced stats and cache metadata are trustworthy enough to interpret later training/eval artifacts
- base-train config audit must prevent silent plumbing drift before training runs
- decode-cache parity tests must prove fast paths are not sacrificing correctness for claimed compute wins

Validation commands after implementation:
- python -m pytest -q tests/test_gpt_local_deliberation.py tests/test_engine_local_deliberation.py

Deliverables:
- stronger GPT/engine integration proof
- no training-path refactor
- documentation.md updated
```

---

## Prompt 9 — Optional checkpoint-backed smoke and artifact audit

```text
Add optional slow proof steps for local checkpoints and engine-backed evals without making them part of the default CPU/mock gate.

Files to inspect:
- scripts/chat_cli.py
- scripts/cognition_eval.py
- nanochat/cognition/backend.py
- nanochat/checkpoint_manager.py
- docs/evals.md

Test files to create only if they add real value:
- optional tests/test_cognition_engine_smoke.py
- optional tests/test_local_delib_engine_smoke.py

Rules:
- mark all checkpoint-backed smoke tests with @pytest.mark.slow
- skip cleanly when the required checkpoint/runtime is unavailable
- do not make the default pre-training gate depend on these tests

Required smoke scenarios:
1. Engine-backed cognition eval
- default cognition eval runs with `--backend engine`
- advanced ablation runs with `--backend engine`
- research eval runs with `--backend engine`
- artifacts are written under `artifacts/pretraining_proofs/engine/`

2. Runtime override truthfulness
- at least one exact-compatible override is exercised
- at least one incompatible override is surfaced as `approximated` or `unsupported`
- artifact audit asserts those statuses are visible in JSON output

3. CLI smoke
- optional `scripts.chat_cli --cognition --prompt ...` run returns non-empty text
- if a slow pytest wrapper is added, keep it single-prompt and skip when checkpoint discovery fails

Suggested optional commands:
- python -m scripts.cognition_eval --backend engine --source sft --output artifacts/pretraining_proofs/engine/cognition_eval.json
- python -m scripts.cognition_eval --suite local-delib-ablation-advanced --backend engine --source sft --output artifacts/pretraining_proofs/engine/local_delib_ablation_advanced.json
- python -m scripts.cognition_eval --suite local-delib-research --backend engine --source sft --output artifacts/pretraining_proofs/engine/local_delib_research.json
- python -m scripts.chat_cli --cognition --source sft --prompt "Summarize why cached thought-graph continuation matters."

Acceptance criteria:
- optional smoke remains isolated from the default gate
- every engine-backed proof artifact clearly distinguishes exact, approximated, and unsupported runtime-override rows
- no checkpoint-backed proof is allowed to masquerade as deterministic CPU/mock proof

Deliverables:
- optional slow smoke tests only if justified
- artifact-audit coverage for engine-backed proof outputs
- documentation.md updated
```

---

## Recommended execution order

1. **Prompt 0** — audit the current proof surface
2. **Prompt 1** — foundational cognition contracts
3. **Prompt 2** — creative/verifier/sandbox/traces direct suites
4. **Prompt 3** — cognition agent orchestration and prompt injection
5. **Prompt 4** — backend metadata and runtime truthfulness
6. **Prompt 5** — eval harness and usefulness proofs
7. **Prompt 6** — core local-delib correctness
8. **Prompt 7** — advanced local-delib mechanisms and evidence
9. **Prompt 8** — GPT/Engine integration and cache proof
10. **Prompt 9** — optional checkpoint-backed smoke

The default pre-training gate should stop after Prompt 8 unless a checkpoint-backed smoke pass is explicitly required.
