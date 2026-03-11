# documentation.md

## Current status
- Active milestone: Milestone 9 integration polish and optional deeper chat integration
- Overall state: the live chat path now shares tokenizer-compatible request serialization, cognition can call the real checkpoint-backed `Engine` through an opt-in adapter, and the cognition reuse path now normalizes paraphrases so ordinary prompts can reuse episodic, semantic, and skill context without relying on exact wording.

## Repo constraints already identified
- Repository already exists and is functioning.
- Root package layout is already in use.
- Existing repo includes `nanochat/`, `scripts/`, `tasks/`, `tests/`, and `pyproject.toml`.
- Existing chat and evaluation flows already exist.
- The training / speedrun path should be treated as high-sensitivity code.
- New work should begin as an isolated subsystem, preferably under `nanochat/cognition/`.
- The repo currently targets Python 3.10+.

## Initial design stance
- Build a developmental cognition layer around the existing nanochat stack.
- Wrap existing model / tokenizer / `Engine` behavior rather than replacing it.
- Start with cheap CPU-friendly tests and fake backends.
- Add a dedicated demo path before considering deeper integration.

## Validation log
- No code changes yet.

## Change log
### Entry template
#### YYYY-MM-DD HH:MM
- Milestone:
- Repo files inspected:
- Files changed:
- Summary:
- Decisions made:
- Commands run:
- Results:
- Known issues:
- Next step:

## Known issues / risks
- It is easy for an agent to overreach and start restructuring the repo.
- It is easy to accidentally disturb speedrun or training-critical paths.
- A cognition layer can become too abstract too early if interfaces are not kept tight.
- Sandbox scope must stay intentionally lightweight in v1.

## How to run
- Populate after Milestone 0.

## Demo notes
- Populate after Milestone 0.

#### 2026-03-08 19:29
- Milestone: Milestone 0 (Task A scaffold)
- Repo files inspected: `documentation.md`, `plans.md`, `implement.md`, `AGENTS.md`, `docs/architecture.md`, `docs/evals.md`, `README.md`, `pyproject.toml`, `scripts/chat_cli.py`, `scripts/chat_eval.py`, `nanochat/engine.py`, `tests/test_engine.py`
- Files changed: `nanochat/cognition/__init__.py`, `tests/test_cognition_smoke.py`
- Summary: Added a minimal cognition package scaffold and a cheap smoke test to confirm import and baseline routing behavior.
- Decisions made: Kept the subsystem isolated under `nanochat/cognition/`; avoided touching training/speedrun/core paths.
- Commands run: `python -m compileall nanochat/cognition tests/test_cognition_smoke.py`; `python -m pytest -q tests/test_cognition_smoke.py`
- Results: Syntax check passed; smoke test passed.
- Known issues: None for Milestone 0 scaffold.
- Next step: Implement Milestone 1 typed schemas and backend contracts with focused tests.

#### 2026-03-08 19:29
- Milestone: Milestone 1 (Task B schemas/backend contracts)
- Repo files inspected: `plans.md`, `implement.md`, `docs/architecture.md`, `docs/evals.md`
- Files changed: `nanochat/cognition/schemas.py`, `nanochat/cognition/backend.py`, `tests/test_cognition_schemas.py`
- Summary: Added explicit dataclass schemas (`Episode`, `MemoryItem`, `Trace`, `RoutingDecision`, `Hypothesis`, `VerificationResult`, `SkillArtifact`) plus a minimal backend protocol and adapter contract.
- Decisions made: Used stdlib dataclasses/protocols only; no new dependencies.
- Commands run: `python -m compileall nanochat/cognition/schemas.py nanochat/cognition/backend.py tests/test_cognition_schemas.py`; `python -m pytest -q tests/test_cognition_schemas.py`
- Results: Syntax check passed; schema/contract tests passed.
- Known issues: Backend adapter remains intentionally minimal and not yet wired to `Engine` in this milestone.
- Next step: Implement Milestone 2 in-memory episodic and semantic memory with ranking tests.

#### 2026-03-08 19:30
- Milestone: Milestone 2 (Task C memory subsystem)
- Repo files inspected: `plans.md`, `docs/architecture.md`, `docs/evals.md`
- Files changed: `nanochat/cognition/memory.py`, `tests/test_cognition_memory.py`
- Summary: Implemented replaceable in-memory `EpisodicMemory` and `SemanticMemory` with write/retrieve helpers and simple relevance+recency ranking.
- Decisions made: Kept persistence out of scope; used transparent scoring and list-backed stores for easy replacement.
- Commands run: `python -m compileall nanochat/cognition/memory.py tests/test_cognition_memory.py`; `python -m pytest -q tests/test_cognition_memory.py`
- Results: Syntax check passed; memory behavior tests passed.
- Known issues: Ranking heuristics are intentionally simple and token-substring based.
- Next step: Implement Milestone 3 explicit router with structured decisions and edge-case tests.

#### 2026-03-08 19:30
- Milestone: Milestone 3 (Task D explicit router)
- Repo files inspected: `plans.md`, `docs/architecture.md`, `docs/evals.md`
- Files changed: `nanochat/cognition/router.py`, `tests/test_cognition_router.py`
- Summary: Added an explicit heuristic router that emits structured `RoutingDecision` values with rationale and confidence for direct/retrieve/creative/verify/sandbox/consolidate actions.
- Decisions made: Prioritized inspectable keyword heuristics over opaque logic to match milestone intent.
- Commands run: `python -m compileall nanochat/cognition/router.py tests/test_cognition_router.py`; `python -m pytest -q tests/test_cognition_router.py`
- Results: Syntax check passed; router scenario tests passed.
- Known issues: Heuristics are intentionally conservative and may need tuning with eval data in later milestones.
- Next step: Milestone 4+ workspace modules and optional cognition adapter integration around existing `Engine`.

#### 2026-03-08 22:27
- Milestone: Milestone 6 (Consolidation and skill reuse)
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `docs/architecture.md`, `docs/evals.md`, `nanochat/cognition/schemas.py`, `nanochat/cognition/memory.py`, `nanochat/cognition/router.py`, `nanochat/cognition/__init__.py`
- Files changed: `nanochat/cognition/consolidation.py`, `nanochat/cognition/skills.py`, `nanochat/cognition/__init__.py`, `tests/test_cognition_consolidation.py`, `documentation.md`
- Summary: Implemented consolidation logic that detects repeated successful episode patterns and emits reusable `SkillArtifact` records; added an in-memory skill registry with query-time discovery; and persisted consolidated skills into semantic memory with provenance metadata.
- Decisions made: Kept design isolated and replaceable with pure in-memory stores; required configurable repetition threshold before skill creation; used explicit trigger/strategy metadata for inspectable consolidation decisions.
- Commands run: `python -m compileall nanochat/cognition tests/test_cognition_consolidation.py`; `python -m pytest -q tests/test_cognition_consolidation.py tests/test_cognition_memory.py tests/test_cognition_router.py tests/test_cognition_schemas.py tests/test_cognition_smoke.py`
- Results: Syntax check passed; cognition regression suite passed (15 tests).
- Known issues: Consolidation heuristics are intentionally simple (metadata and substring matching) and may need refinement with real eval traces.
- Next step: Implement Milestone 7 end-to-end cognition loop wiring these modules through an adapter-driven demo path.

#### 2026-03-08 23:05
- Milestone: Milestone 7 (End-to-end cognition loop)
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `docs/architecture.md`, `docs/evals.md`, `nanochat/cognition/__init__.py`, `nanochat/cognition/backend.py`, `nanochat/cognition/memory.py`, `nanochat/cognition/router.py`, `nanochat/cognition/consolidation.py`, `nanochat/cognition/skills.py`
- Files changed: `nanochat/cognition/creative.py`, `nanochat/cognition/verifier.py`, `nanochat/cognition/sandbox.py`, `nanochat/cognition/traces.py`, `nanochat/cognition/agent.py`, `nanochat/cognition/__init__.py`, `scripts/cognition_demo.py`, `tests/test_cognition_agent.py`, `documentation.md`
- Summary: Added a full optional cognition loop controller (`CognitionAgent`) that wires backend calls, router decisions, memory retrieval, creative candidate generation, verifier ranking, sandbox branch scoring, consolidation, and trace recording; also added a runnable script demo and end-to-end tests.
- Decisions made: Kept implementation isolated under `nanochat/cognition/` and `scripts/cognition_demo.py`; used lightweight in-memory stores plus fake backend-driven behavior to stay CPU-cheap and deterministic for tests.
- Commands run: `python -m compileall nanochat/cognition scripts/cognition_demo.py tests/test_cognition_agent.py`; `python -m pytest -q tests/test_cognition_agent.py tests/test_cognition_consolidation.py tests/test_cognition_memory.py tests/test_cognition_router.py tests/test_cognition_schemas.py tests/test_cognition_smoke.py`; `python -m scripts.cognition_demo "Brainstorm three ideas for retrieval routing"`
- Results: Syntax check passed; cognition test suite passed; demo script runs and prints decision/response/trace.
- Known issues: Heuristics are intentionally simple and primarily inspectability-focused; backend integration with the full production `Engine` remains optional and can be added in a later milestone.
- Next step: Milestone 8 evaluation harness for baseline vs cognition-enhanced comparisons.


#### 2026-03-08 23:40
- Milestone: Milestone 8 (Evaluation harness)
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `docs/evals.md`, `scripts/chat_eval.py`, `tasks/common.py`, `nanochat/cognition/agent.py`, `scripts/cognition_demo.py`
- Files changed: `nanochat/cognition/eval.py`, `scripts/cognition_eval.py`, `tests/test_cognition_eval.py`, `docs/evals.md`, `documentation.md`
- Summary: Added a lightweight evaluation harness that compares direct baseline responses with cognition-enhanced responses on deterministic prompt cases, computes aggregate scores, and writes machine-readable JSON artifacts.
- Decisions made: Kept evaluation isolated and CPU-cheap with a fake deterministic backend; used simple keyword recall scoring for inspectable milestone-appropriate metrics.
- Commands run: `python -m compileall nanochat/cognition/eval.py scripts/cognition_eval.py tests/test_cognition_eval.py`; `python -m pytest -q tests/test_cognition_eval.py tests/test_cognition_agent.py tests/test_cognition_consolidation.py tests/test_cognition_memory.py tests/test_cognition_router.py tests/test_cognition_schemas.py tests/test_cognition_smoke.py`; `python -m scripts.cognition_eval --output artifacts/cognition_eval.json`
- Results: Syntax check passed; cognition regression tests passed; eval script produced comparison summary and JSON artifact.
- Known issues: Evaluation uses synthetic prompt cases and simple keyword metrics, so results are directional rather than benchmark-grade.
- Next step: Milestone 9 polish, docs cleanup, and optional deeper integration hooks.

#### 2026-03-09 00:10
- Milestone: Targeted mathematical consistency fixes for eval/train scripts.
- Repo files inspected: `scripts/base_eval.py`, `scripts/base_train.py`, `scripts/chat_sft.py`, `nanochat/gpt.py`, `nanochat/engine.py`.
- Files changed: `scripts/base_eval.py`, `scripts/base_train.py`, `scripts/chat_sft.py`, `tests/test_base_eval_baseline_normalization.py`, `documentation.md`.
- Summary: Added normalized CORE random-baseline handling (fraction in [0,1) with backward-compatible percent parsing), switched base train/SFT grad accumulation to ceil-based realization with explicit requested vs effective total batch logging, and made FLOPs/throughput/token accounting use the effective realized batch consistently.
- Decisions made: Kept `nanochat/gpt.py` initialization and optimizer grouping unchanged after audit; kept temperature==0 argmax sampling unchanged in `nanochat/engine.py` and `nanochat/gpt.py`.
- Commands run: `python -m py_compile scripts/base_eval.py scripts/base_train.py scripts/chat_sft.py nanochat/gpt.py nanochat/engine.py`; `python -m pytest -q tests/test_base_eval_baseline_normalization.py`; `rg -n "total_batch_size % world_tokens_per_fwdbwd|// world_tokens_per_fwdbwd|assert .*world_tokens_per_fwdbwd|assert .*total_batch_size" scripts/base_train.py scripts/chat_sft.py || true`.
- Results: Syntax checks passed; baseline normalization sanity tests passed (4/4); no remaining divisibility-based total-batch assumptions in `base_train.py` or `chat_sft.py`.
- Known issues: None found for this scoped change.
- Next step: None for this scoped patch.

#### 2026-03-09 09:02
- Milestone: Targeted hardening patch from latest review (RL LR clamp + BPB clarification).
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `scripts/chat_rl.py`, `nanochat/loss_eval.py`.
- Files changed: `nanochat/rl_schedule.py`, `scripts/chat_rl.py`, `nanochat/loss_eval.py`, `tests/test_chat_rl_lr_schedule.py`, `documentation.md`.
- Summary: Added a tiny import-safe RL schedule helper with a defensive non-negative clamp, wired `scripts/chat_rl.py` to use it without changing in-range behavior, added a focused regression test for normal and overshoot cases, and clarified BPB unit conversion comments (nats to bits via ln(2)) while keeping math unchanged.
- Decisions made: Kept `nanochat/gpt.py` untouched as requested; avoided optimizer/training-flow refactors; extracted only the schedule helper to enable side-effect-free unit testing.
- Commands run: `python -m py_compile scripts/chat_rl.py nanochat/loss_eval.py nanochat/rl_schedule.py tests/test_chat_rl_lr_schedule.py`; `python -m pytest -q tests/test_chat_rl_lr_schedule.py`.
- Results: Syntax/compile checks passed; targeted LR schedule regression test passed.
- Known issues: None for this scoped patch.
- Next step: No further changes required for this review scope.

#### 2026-03-09 11:45
- Milestone: Milestone 9 targeted integration patch (shared chat serialization + Engine-backed cognition).
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `docs/architecture.md`, `docs/evals.md`, `nanochat/tokenizer.py`, `scripts/chat_cli.py`, `scripts/chat_web.py`, `nanochat/cognition/backend.py`, `nanochat/cognition/agent.py`, `nanochat/cognition/eval.py`, `tests/test_cognition_agent.py`, `tests/test_cognition_eval.py`.
- Files changed: `nanochat/chat_format.py`, `nanochat/tokenizer.py`, `nanochat/cognition/backend.py`, `nanochat/cognition/agent.py`, `nanochat/cognition/eval.py`, `nanochat/cognition/memory.py`, `nanochat/cognition/__init__.py`, `scripts/chat_cli.py`, `scripts/chat_web.py`, `scripts/cognition_eval.py`, `tests/test_chat_web_serialization.py`, `tests/test_cognition_backend.py`, `tests/test_cognition_agent.py`, `tests/test_cognition_eval.py`, `docs/architecture.md`, `docs/evals.md`, `documentation.md`.
- Summary: Added shared chat validation/rendering helpers so the web API accepts system messages and serializes prompts with the same rules as tokenizer chat rendering; introduced an `EngineBackend` so cognition can call the real checkpoint-backed `Engine`; changed the cognition agent to inject retrieved semantic memory and reused skills into prompts and to expose retrieved ids in traces; and replaced the prompt-echo cognition eval with strict memory/skill-sensitive cases plus an optional real-engine backend mode.
- Decisions made: Kept the default chat generation path intact unless `--cognition` is set; used a per-request cognition agent in web mode to avoid cross-user memory leakage while seeding request-local episodic history; made semantic memory writes idempotent by `item_id` to avoid duplicated prompt context after repeated consolidation.
- Commands run: `python3 -m py_compile nanochat/chat_format.py nanochat/tokenizer.py nanochat/cognition/backend.py nanochat/cognition/agent.py nanochat/cognition/eval.py scripts/chat_cli.py scripts/chat_web.py scripts/cognition_eval.py tests/test_chat_web_serialization.py tests/test_cognition_backend.py tests/test_cognition_agent.py tests/test_cognition_eval.py`; `python3 - <<'PY' ... smoke-ok ... PY`; `python3 -m scripts.cognition_eval --output /tmp/cognition_eval.json`.
- Results: Syntax checks passed; pure-Python smoke checks passed; `scripts.cognition_eval` demo backend now reports a positive delta (`baseline_mean=0.000`, `cognition_mean=1.000`, `delta=1.000`) and writes an artifact.
- Known issues: `pytest` is still unavailable in the current local interpreter, so the added tests were not executed under pytest in this environment; web cognition mode currently returns a single SSE chunk for the final response instead of token-by-token streaming.
- Next step: If the runtime environment is provisioned, run the new pytest targets and optionally try `python -m scripts.cognition_eval --backend engine --source sft` against a local checkpoint.

#### 2026-03-09 11:58
- Milestone: Milestone 9 cognition quality patch (normalized reuse and scored episodic support).
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `nanochat/cognition/__init__.py`, `nanochat/cognition/agent.py`, `nanochat/cognition/memory.py`, `nanochat/cognition/router.py`, `nanochat/cognition/skills.py`, `nanochat/cognition/eval.py`, `tests/test_cognition_agent.py`, `tests/test_cognition_memory.py`, `tests/test_cognition_router.py`, `tests/test_cognition_eval.py`.
- Files changed: `nanochat/cognition/normalize.py`, `nanochat/cognition/memory.py`, `nanochat/cognition/skills.py`, `nanochat/cognition/router.py`, `nanochat/cognition/agent.py`, `nanochat/cognition/eval.py`, `nanochat/cognition/__init__.py`, `tests/test_cognition_memory.py`, `tests/test_cognition_router.py`, `tests/test_cognition_agent.py`, `tests/test_cognition_eval.py`, `documentation.md`.
- Summary: Added a shared normalization path for routing and retrieval, introduced scored episodic search, and changed the cognition agent to reuse episodic context for ordinary paraphrased prompts while keeping `direct_answer` as the default route.
- Decisions made: Kept the patch isolated to `nanochat/cognition/`; limited support injection to 1 skill, 2 semantic items, and 2 episodic items; used normalized overlap plus recency scoring for episodic search; filtered ordinary episodic support against already selected skill/semantic context to avoid obvious redundancy; preserved explicit override routes for memory retrieval, creative exploration, verification, sandboxing, and consolidation.
- Commands run: `python3 -m py_compile nanochat/cognition/normalize.py nanochat/cognition/memory.py nanochat/cognition/skills.py nanochat/cognition/router.py nanochat/cognition/agent.py nanochat/cognition/eval.py nanochat/cognition/__init__.py tests/test_cognition_memory.py tests/test_cognition_router.py tests/test_cognition_agent.py tests/test_cognition_eval.py`; `python3 -m scripts.cognition_eval --output /tmp/cognition_eval_quality.json`; `python3 - <<'PY' ... direct-answer episodic injection smoke ... PY`; `python3 - <<'PY' ... skill paraphrase reuse smoke ... PY`; `python3 - <<'PY' ... router alias smoke ... PY`; `python3 -m pytest -q tests/test_cognition_memory.py`
- Results: Syntax checks passed; cognition eval still reports a positive gain (`baseline_mean=0.000`, `cognition_mean=1.000`, `delta=1.000`) while now covering paraphrased episodic and skill reuse cases; direct `python3` smoke checks confirmed that `Please summarize this draft for me.` remains `direct_answer` while reusing episodic support, paraphrased skill lookup now works for `summarize`, and alias-based routing maps `validate` to `verify` and `recall`/`previous` to `retrieve_memory`.
- Known issues: `pytest` remains unavailable in the current `python3` interpreter (`python3 -m pytest -q tests/test_cognition_memory.py` fails with `No module named pytest`), so the updated pytest suite could not be executed in this environment.
- Next step: Once the runtime is provisioned with pytest, run the targeted cognition tests and optionally rerun `python3 -m scripts.cognition_eval --backend engine --source sft` against a local checkpoint to measure prompt-context gains on a real model.

#### 2026-03-11 00:00
- Milestone: Architecture documentation update for model-core local deliberation vs cognition wrapper.
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `docs/architecture.md`, `nanochat/gpt.py`, `scripts/base_train.py`.
- Files changed: `docs/architecture.md`, `documentation.md`.
- Summary: Documented that the repository now has two distinct capability layers: (1) the existing wrapper-style cognition subsystem under `nanochat/cognition/` and (2) the model-core latent local deliberation path in `nanochat/gpt.py`, including token-local micro-steps, phrase pooling/consensus, optional semantic top-k neighbor path, adaptive per-token gating, and decode-time cache compatibility notes.
- Decisions made: Kept the update docs-only and tightly scoped; explicitly clarified that local deliberation is a lightweight latent graph-of-thought approximation rather than a full explicit graph executor.
- Commands run: `rg -n "deliber|latent|micro|phrase|semantic|top-k|topk|gate|gating|cache|cognition|local" nanochat/gpt.py scripts/base_train.py docs/architecture.md documentation.md`; `sed -n '1,260p' docs/architecture.md`; `sed -n '1,220p' documentation.md`.
- Results: Documentation now distinguishes wrapper cognition behavior from model-core local deliberation behavior and points to train-script enablement flags.
- Known issues: The local deliberation path is off-by-default (`--local-delib` plus nonzero `--local-delib-steps` required), remains intentionally lightweight/approximate, and is not a full external reasoning graph.
- Next step: Enable from training via `scripts/base_train.py` flags (minimum: `--local-delib --local-delib-steps <N>`; optionally `--local-delib-every`, `--local-delib-state-dim`, `--local-delib-kernel-size`, `--local-delib-phrase-chunk-size`, `--local-delib-use-token-gate`, and `--local-delib-debug-stats`) and validate in targeted experiments.

#### 2026-03-11 00:20
- Milestone: Local deliberation agreement/consensus wiring (token proposals -> phrase consensus -> token feedback).
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `nanochat/local_deliberation.py`, `tests/test_local_deliberation.py`.
- Files changed: `nanochat/local_deliberation.py`, `tests/test_local_deliberation.py`, `documentation.md`.
- Summary: Added an explicit `PhraseConsensusHead` that builds token proposals, aggregates per-chunk phrase consensus nodes, broadcasts consensus to tokens with a learned acceptance gate, computes mean agreement via cosine similarity, and feeds consensus feedback into the local-deliberation update path when enabled.
- Decisions made: Kept consensus path off-by-default through a new `use_phrase_consensus=False` flag in `LocalDeliberationBlock`; initialized the agreement gate near-disabled (`weight=0`, `bias=-8`) so existing behavior is effectively preserved at init when enabled; kept output projection near-zero init unchanged.
- Commands run: `python -m py_compile nanochat/local_deliberation.py tests/test_local_deliberation.py`; `python -m pytest -q tests/test_local_deliberation.py`.
- Results: Syntax compilation passed; pytest could not run because the environment lacks torch (`ModuleNotFoundError: No module named 'torch'`).
- Next step: Re-run `python -m pytest -q tests/test_local_deliberation.py` in an environment with torch installed.

#### 2026-03-11 01:10
- Milestone: Local deliberation decode-time cache support in KV-cached generation.
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `nanochat/engine.py`, `nanochat/gpt.py`, `tests/test_engine_local_deliberation.py`.
- Files changed: `nanochat/engine.py`, `nanochat/gpt.py`, `tests/test_engine_local_deliberation.py`, `documentation.md`.
- Summary: Added minimal model-side cache storage alongside KV cache (`extra_caches`) and wired GPT local deliberation to use a decode-time cache path so local deliberation runs in kv-cache mode without re-running full-sequence deliberation from scratch.
- Decisions made: Kept the implementation bounded and minimal by caching only per-layer latent deliberation states plus token count; preserved existing KV tensor and flash-attention cache behavior; kept deliberation debug stats behavior unchanged for kv-cache mode.
- Commands run: `python -m py_compile nanochat/engine.py nanochat/gpt.py tests/test_engine_local_deliberation.py`; `python -m pytest -q tests/test_engine_local_deliberation.py`.
- Results: Syntax compilation passed; targeted pytest could not run in system Python due to missing torch and also failed under `uv run` due missing CUDA runtime libraries (`libcudart.so.12`/`libcublas`).
- Known issues: Decode-time deliberation cache currently stores full per-layer token states up to current decode length (bounded by cache sequence length); this is intentionally minimal/correctness-first and not yet optimized for more aggressive state compression.
- Next step: If needed in future milestones, optimize deliberation cache memory by retaining only the exact left-context slices required by each enabled subpath.

#### 2026-03-11 01:40
- Milestone: Expose model-side local deliberation debug stats to cognition tracing metadata without changing generation return contracts.
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `nanochat/cognition/backend.py`, `nanochat/cognition/agent.py`, `tests/test_cognition_backend.py`, `tests/test_cognition_agent.py`.
- Files changed: `nanochat/cognition/backend.py`, `nanochat/cognition/agent.py`, `tests/test_cognition_backend.py`, `tests/test_cognition_agent.py`, `documentation.md`.
- Summary: Added an optional `EngineBackend.last_generation_metadata` side-channel populated from `engine.model.last_deliberation_stats` (when present), kept `EngineBackend.generate()` return type as plain `str`, and plumbed that metadata into cognition traces under `trace.metadata["model_local_delib"]` only when available.
- Decisions made: Kept API backward-compatible and off-by-default by using a nullable side-channel; avoided any core model/cognition coupling beyond attribute introspection on the backend instance.
- Commands run: `python -m py_compile nanochat/cognition/backend.py nanochat/cognition/agent.py tests/test_cognition_backend.py tests/test_cognition_agent.py`; `python -m pytest -q tests/test_cognition_backend.py tests/test_cognition_agent.py`.
- Results: Syntax compilation passed; targeted cognition backend/agent tests passed (`7 passed`).
- Known issues: None for this scoped patch.
- Next step: If future milestones require richer diagnostics, extend side-channel keys in `last_generation_metadata` while preserving string-return generation APIs.

#### 2026-03-11 02:05
- Milestone: Post-M9 roadmap extension for full latent hierarchical graph-of-thought direction (docs-only planning pass).
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `docs/architecture.md`, `nanochat/local_deliberation.py`, `nanochat/gpt.py`, `nanochat/cognition/backend.py`, `nanochat/cognition/agent.py`, `nanochat/cognition/creative.py`.
- Files changed: `plans.md`, `docs/architecture.md`, `documentation.md`.
- Summary: Added a new milestone sequence after Milestone 9 covering adaptive per-token halting/variable compute, dynamic latent nearest-neighbor graph + flocking, latent branch spawn/merge, deeper multi-scale hierarchy beyond phrase chunks, latent creative scratchpad slots, and optional auxiliary losses/evals; expanded architecture docs with a repo-native staged target architecture for those mechanisms.
- Decisions made: Kept this update docs-only (no runtime code changes), preserved existing wrapper cognition + model-core layering, and maintained the incremental/opt-in integration stance so current training/speedrun paths remain untouched by default.
- Commands run: `sed -n '1,320p' README.md`; `sed -n '1,260p' pyproject.toml`; `sed -n '1,320p' AGENTS.md`; `sed -n '1,320p' plans.md`; `sed -n '1,300p' implement.md`; `sed -n '1,320p' documentation.md`; `sed -n '1,320p' docs/architecture.md`; `sed -n '1,260p' nanochat/local_deliberation.py`; `sed -n '1,260p' nanochat/gpt.py`; `sed -n '1,260p' nanochat/cognition/backend.py`; `sed -n '1,280p' nanochat/cognition/agent.py`; `sed -n '1,260p' nanochat/cognition/creative.py`; `rg -n "adaptive|halting|variable compute|nearest-neighbor|flocking|branch|merge|multi-scale|hierarchy|scratchpad|auxiliary|loss|eval" plans.md docs/architecture.md documentation.md`.
- Results: Sanity checks confirm the new architecture concepts are consistently present across plan and architecture docs.
- Known issues: None for this docs-only milestone extension.
- Next step: Execute Milestone 10 as the first implementation slice with instrumentation-first adaptive halting and fixed-step fallback tests.



#### 2026-03-11 02:30
- Milestone: Local deliberation advanced-option first-class config plumbing (no behavior changes).
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `docs/architecture.md`, `nanochat/gpt.py`, `nanochat/local_deliberation.py`, `scripts/base_train.py`, `tests/test_gpt_local_deliberation.py`, `tests/test_engine_local_deliberation.py`, `tests/test_local_deliberation.py`.
- Files changed: `nanochat/gpt.py`, `scripts/base_train.py`, `tests/test_gpt_local_deliberation.py`, `tests/test_engine_local_deliberation.py`, `docs/architecture.md`, `documentation.md`.
- Summary: Promoted advanced local-deliberation options to first-class `GPTConfig` fields and `base_train` CLI/config wiring; replaced `getattr` fallback usage in GPT local-deliberation block construction with direct config fields; added parser validation for new numeric knobs; expanded train-time local-deliberation config printing; and added focused tests for default stability and block wiring.
- Decisions made: Kept all new flags plumbing-only with defaults preserving existing behavior; did not implement adaptive-halt/branch/hierarchy/scratch runtime behavior in this patch.
- Commands run: `python -m py_compile nanochat/gpt.py scripts/base_train.py tests/test_gpt_local_deliberation.py tests/test_engine_local_deliberation.py`; `python -m pytest -q tests/test_gpt_local_deliberation.py tests/test_engine_local_deliberation.py`.
- Results: Syntax compilation passed; targeted pytest could not run in system Python due to missing torch (`ModuleNotFoundError: No module named 'torch'`).
- Known issues: New branch/hierarchy/scratch/adaptive options are intentionally accepted and validated but not consumed by runtime local-deliberation logic yet.
- Next step: Implement milestone-scoped runtime behavior for adaptive halt/branch/hierarchy/scratch in follow-on patches.

#### 2026-03-11 03:05
- Milestone: Milestone 10 adaptive per-token local-deliberation halting (runtime + tests).
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `docs/architecture.md`, `nanochat/local_deliberation.py`, `nanochat/gpt.py`, `nanochat/engine.py`, `tests/test_local_deliberation.py`, `tests/test_gpt_local_deliberation.py`, `tests/test_engine_local_deliberation.py`.
- Files changed: `nanochat/local_deliberation.py`, `nanochat/gpt.py`, `tests/test_local_deliberation.py`, `tests/test_gpt_local_deliberation.py`, `tests/test_engine_local_deliberation.py`, `documentation.md`.
- Summary: Implemented `local_delib_adaptive_halt` as an off-by-default adaptive compute path where tokens maintain an active mask across micro-steps, stop receiving updates after their halt confidence crosses a learned threshold, and preserve inactive-token latent state; added matching behavior to decode-time cached deliberation via shared block execution logic; and extended deliberation stats with `mean_executed_steps_per_token`, `max_executed_steps_any_token`, `fraction_halted_early`, and `mean_final_halt` while keeping fixed-step behavior unchanged when adaptive halt is disabled.
- Decisions made: Kept output projection near-identity defaults unchanged (zero-init) and initialized the halt threshold parameter to a conservative high-confidence default to avoid destabilizing early halting at init.
- Commands run: `python -m py_compile nanochat/local_deliberation.py nanochat/gpt.py tests/test_local_deliberation.py tests/test_gpt_local_deliberation.py tests/test_engine_local_deliberation.py`; `python -m pytest -q tests/test_local_deliberation.py tests/test_gpt_local_deliberation.py tests/test_engine_local_deliberation.py`.
- Results: Syntax compilation passed; targeted pytest could not execute in this environment because `torch` is not installed in the active interpreter (`ModuleNotFoundError: No module named 'torch'` during collection).
- Known issues: Runtime test execution remains blocked by missing `torch` in system Python.
- Next step: Re-run the targeted local-deliberation pytest set in an environment with `torch` available.

#### 2026-03-11 03:40
- Milestone: Milestone 11 dynamic latent neighbor graph/flocking-style mixer (scoped local-deliberation patch).
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `docs/architecture.md`, `nanochat/local_deliberation.py`, `nanochat/gpt.py`, `tests/test_local_deliberation.py`, `tests/test_gpt_local_deliberation.py`.
- Files changed: `nanochat/local_deliberation.py`, `nanochat/gpt.py`, `tests/test_local_deliberation.py`, `tests/test_gpt_local_deliberation.py`, `docs/architecture.md`, `documentation.md`.
- Summary: Added an off-by-default `local_delib_use_neighbor_graph` path that builds a bounded causal neighbor graph per token (sequence predecessor + semantic top-k within lookback + optional phrase-node link), aggregates messages through a new helper class, and surfaces graph stats (`mean_neighbor_count`, sequence/semantic/phrase mean neighbor weights, `semantic_topk_used`) while preserving prior behavior when disabled.
- Decisions made: Kept the legacy semantic summary path untouched for parity when graph mode is disabled; implemented phrase links causally using chunk-prefix means to avoid same-step future leakage; and kept decode-cache execution unchanged via existing `_run_local_delib_cached` block path.
- Commands run: `python -m py_compile nanochat/local_deliberation.py nanochat/gpt.py tests/test_local_deliberation.py tests/test_gpt_local_deliberation.py`; `python -m pytest -q tests/test_local_deliberation.py tests/test_gpt_local_deliberation.py`.
- Results: `py_compile` passed; pytest collection/execution is blocked in this interpreter because `torch` is unavailable (`ModuleNotFoundError: No module named 'torch'`).
- Known issues: Runtime pytest validation remains environment-blocked without torch.
- Next step: Re-run the targeted local deliberation pytest set in a torch-enabled environment.

#### 2026-03-11 04:05
- Milestone: Off-by-default latent branch spawn/score/merge path for local deliberation, with metadata plumbing to cognition traces.
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `docs/architecture.md`, `nanochat/local_deliberation.py`, `nanochat/gpt.py`, `nanochat/cognition/backend.py`, `nanochat/cognition/agent.py`, `tests/test_local_deliberation.py`, `tests/test_gpt_local_deliberation.py`, `tests/test_engine_local_deliberation.py`, `tests/test_cognition_backend.py`, `tests/test_cognition_agent.py`.
- Files changed: `nanochat/local_deliberation.py`, `nanochat/gpt.py`, `tests/test_local_deliberation.py`, `tests/test_gpt_local_deliberation.py`, `tests/test_engine_local_deliberation.py`, `tests/test_cognition_backend.py`, `tests/test_cognition_agent.py`, `documentation.md`.
- Summary: Added latent branching helper heads (`BranchProposalHead`, `BranchScorer`, `BranchMergeHead`) into `LocalDeliberationBlock`; enabled branch spawn/score/merge only when `branch_factor > 0` and at optional `branch_every` cadence; kept near-identity behavior at init using zero-initialized branch projections and strongly off merge gate bias; and surfaced branch stats (`mean_branch_score`, `max_branch_score`, `mean_merge_weight`, `branch_factor_used`, `fraction_tokens_branched`) into per-layer deliberation stats.
- Decisions made: Kept branch path internal-only (no text emission) and lightweight by reusing per-token latent states; preserved default parity when branching disabled; wired branch config fields from `GPTConfig` into local deliberation block creation without changing generation return contracts.
- Commands run: `python -m py_compile nanochat/local_deliberation.py nanochat/gpt.py nanochat/cognition/backend.py nanochat/cognition/agent.py tests/test_local_deliberation.py tests/test_gpt_local_deliberation.py tests/test_engine_local_deliberation.py tests/test_cognition_backend.py tests/test_cognition_agent.py`; `python -m pytest -q tests/test_local_deliberation.py tests/test_gpt_local_deliberation.py tests/test_engine_local_deliberation.py tests/test_cognition_backend.py tests/test_cognition_agent.py`; `python -m pytest -q tests/test_cognition_backend.py tests/test_cognition_agent.py`.
- Results: `py_compile` passed; local-deliberation and engine local-deliberation tests could not run in this interpreter due to missing `torch` (`ModuleNotFoundError` during collection); cognition backend/agent tests passed (`7 passed`).
- Known issues: Cached decode compatibility is functionally preserved via existing cached deliberation execution path, but branching currently reuses the same full cached latent-state deliberation update (correct but not additionally optimized for branch-specific incremental compute).
- Next step: Re-run local deliberation + GPT + engine local deliberation tests in a torch-enabled environment.

#### 2026-03-11 00:00
- Milestone: Model-core local deliberation hierarchy extension (multi-scale chunk stack, off-by-default).
- Repo files inspected: `README.md`, `pyproject.toml`, `AGENTS.md`, `plans.md`, `implement.md`, `documentation.md`, `docs/architecture.md`, `nanochat/local_deliberation.py`, `nanochat/gpt.py`, `tests/test_local_deliberation.py`, `tests/test_gpt_local_deliberation.py`.
- Files changed: `nanochat/local_deliberation.py`, `nanochat/gpt.py`, `tests/test_local_deliberation.py`, `tests/test_gpt_local_deliberation.py`, `documentation.md`, `docs/architecture.md`.
- Summary: Added an optional multi-scale hierarchy feedback path in `LocalDeliberationBlock` via per-level chunk pooling/refinement/broadcast and bounded averaged feedback; added hierarchy stats (`hierarchy_levels_used`, `mean_hierarchy_feedback_norm`, `hierarchy_level_chunk_counts`); and wired `GPTConfig.local_delib_hierarchy_chunk_sizes` parsing (`"4,16"`) into local deliberation block construction so stats surface through `last_deliberation_stats` when debug stats are enabled.
- Decisions made: Kept hierarchy disabled by default; preserved existing single phrase-chunk path behavior when no hierarchy chunk sizes are configured; kept hierarchy broadcasting causal by using prefix means of per-level chunk nodes.
- Commands run: `python -m py_compile nanochat/local_deliberation.py nanochat/gpt.py tests/test_local_deliberation.py tests/test_gpt_local_deliberation.py`; `pytest -q tests/test_local_deliberation.py tests/test_gpt_local_deliberation.py`.
- Results: `py_compile` passed. `pytest` collection failed in this environment due to missing `torch`.
- Known issues: Full targeted tests require a torch-enabled runtime.
- Next step: Re-run local deliberation and GPT local deliberation tests in an environment with `torch` installed.
