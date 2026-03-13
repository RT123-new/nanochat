# Codex Prompt Pack — Fullest Realistic Implementation for `RT123-new/nanochat`

This prompt pack is designed for the current state of the `RT123-new/nanochat` repository.

It assumes the repo already contains:
- model-core local deliberation
- adaptive halting
- neighbor graph + flocking
- branch spawn / consensus / verifier merge
- legacy and deep hierarchy
- latent scratch slots
- explicit latent thought nodes / graph message passing
- global anchors
- wrapper cognition modules
- ablation and trace plumbing

The purpose of this pack is to push the repo from its current advanced state toward the **fullest realistic implementation** without breaking the repo’s current training, engine, or evaluation flows.

---

## How to use this pack

Recommended order:
1. Run **Prompt 0** in Ask mode first.
2. Execute **Prompts 1–6** one at a time in code mode.
3. Use **Prompt 7** whenever Codex starts drifting.
4. Use **Prompt 8** after each milestone to generate the change log / PR summary.

General rules for every prompt in this pack:
- preserve existing repo structure
- avoid repo-wide rewrites
- keep new behavior opt-in by flags/config
- preserve disabled-path parity and near-identity-at-init behavior
- do not casually change generation return contracts
- prefer focused tests, then code, then docs
- keep speedrun / base training / engine paths safe

---

## Prompt 0 — Ask-mode bootstrap and gap plan

```text
You are working inside the RT123-new/nanochat repository.

Do not code yet. First inspect and plan.

Read these files first:
- README.md
- documentation.md
- docs/architecture.md
- docs/evals.md
- nanochat/gpt.py
- nanochat/local_deliberation.py
- nanochat/cognition/backend.py
- nanochat/cognition/agent.py
- nanochat/cognition/creative.py
- nanochat/cognition/eval.py
- scripts/base_train.py
- tests/test_local_deliberation.py
- tests/test_gpt_local_deliberation.py
- tests/test_engine_local_deliberation.py
- tests/test_cognition_backend.py
- tests/test_cognition_agent.py
- tests/test_cognition_eval.py

Goal:
Produce a concrete implementation plan for taking the repo from its current advanced local-deliberation state to the fullest realistic implementation of:
1. exact incremental thought-graph decode continuation
2. first-class inspectable latent graph artifacts and wrapper traces
3. full creative workspace integration with model-core scratch / thought summaries
4. stronger non-heuristic evaluation and benchmark-grade ablations
5. reliable runtime variant override / hot swapping in evals and engine-backed runs
6. end-to-end hardening and docs cleanup

Constraints:
- preserve existing repo structure
- do not rewrite the main training loop
- do not break speedrun / base train / engine behavior
- keep all new behavior opt-in by config/flags
- preserve disabled-path near-identity behavior
- keep generation return contracts backward compatible unless there is a very strong reason not to
- prefer small PR-sized milestones
- add focused tests first, then implementation, then docs

Output format:
A. exact gaps you confirm are still missing
B. milestone order with rationale
C. files to touch for each milestone
D. acceptance criteria for each milestone
E. risk list / rollback switches
F. recommended commands/tests to run after each milestone

Do not write code in this response.
```

---

## Prompt 1 — Exact incremental thought-graph decode continuation

```text
Implement the highest-value remaining model-core gap: exact incremental thought-graph decode continuation.

Current situation:
- explicit latent thought graph exists in nanochat/local_deliberation.py
- decode-time cache support exists for bounded scratch/anchor/hierarchy paths
- explicit thought-graph decode continuation still falls back to correctness-first full local-deliberation recompute
- I want that gap closed as fully as possible without breaking correctness

Primary goal:
Remove the thought-graph decode fallback path and replace it with a true bounded incremental continuation path for explicit latent thought nodes/edges during cached decoding.

Files to inspect and modify:
- nanochat/local_deliberation.py
- nanochat/gpt.py
- nanochat/engine.py
- tests/test_local_deliberation.py
- tests/test_gpt_local_deliberation.py
- tests/test_engine_local_deliberation.py
- docs/architecture.md
- documentation.md

Requirements:
1. Preserve exact causal semantics.
2. Keep the disabled path unchanged.
3. Keep memory bounded; no unbounded per-token graph objects.
4. Reuse existing cache structure where sensible.
5. Thought-node state, node-write state, node-read state, and any required edge/cache metadata must be stored in a structured bounded decode cache.
6. Support batch expansion when prefill cache is cloned into a larger decode batch.
7. Keep behavior near-identity at init.
8. Preserve existing GPT forward/generate contracts.
9. Keep any fallback only for clearly defined impossible states such as window-slide overflow, and document that explicitly.

Acceptance criteria:
- explicit thought-graph mode can continue incrementally in decode without full-prefix recompute
- new focused tests cover:
  - parity against full forward on short sequences
  - cache continuation across multiple decode steps
  - batch expansion from prefill to larger decode batch
  - interaction with branch/hierarchy/scratch/anchor inputs when enabled
  - bounded cache growth
- docs updated to describe exact supported incremental behavior and remaining limits
- add a short implementation note to documentation.md with what changed and why

Deliverables:
- code changes
- focused tests
- docs update
- concise summary of design choices and any remaining limitations
```

---

## Prompt 2 — First-class inspectable latent graph artifacts and traces

```text
Implement first-class inspectable latent graph artifacts and richer wrapper trace surfacing for the model-core local deliberation system.

Current situation:
- backend/agent already surface compact local_delib metadata
- there are summary buckets for branch/hierarchy/scratch/thought/anchors/flocking
- I want a fuller inspectable graph artifact layer without exposing raw latent tensors or breaking backward compatibility

Files to inspect and modify:
- nanochat/local_deliberation.py
- nanochat/cognition/backend.py
- nanochat/cognition/agent.py
- nanochat/cognition/traces.py
- nanochat/cognition/eval.py
- tests/test_cognition_backend.py
- tests/test_cognition_agent.py
- tests/test_cognition_eval.py
- docs/architecture.md
- docs/evals.md
- documentation.md

Goal:
Add a compact, structured, machine-readable graph artifact that captures the important dynamics of:
- branch activity and branch consensus
- thought node counts and degree patterns
- token-to-thought and thought-to-token interaction
- hierarchy usage by scale
- scratch usage summaries
- anchor usage summaries
- adaptive-halt compute behavior
- flocking/local-neighbor behavior

Requirements:
1. Do not expose raw latent tensors.
2. Keep existing namespaced metadata keys backward compatible.
3. Add new graph-artifact keys only when mechanisms are active.
4. Make artifacts small enough for traces and eval JSON output.
5. Include enough information to reconstruct “what happened” at a debugging level.
6. Preserve existing generation return contracts.

Suggested output shape:
- model_local_delib.graph_artifact.overview
- model_local_delib.graph_artifact.branch
- model_local_delib.graph_artifact.thought_graph
- model_local_delib.graph_artifact.hierarchy
- model_local_delib.graph_artifact.scratch
- model_local_delib.graph_artifact.anchors
- model_local_delib.graph_artifact.compute
- model_local_delib.graph_artifact.flocking

Acceptance criteria:
- backend captures the new artifact
- cognition traces include the new artifact when present
- eval artifacts can persist it in compact form
- focused tests cover empty/disabled, partial, and full-stack active cases
- docs clearly explain what is surfaced, what is intentionally omitted, and why

Please keep this implementation additive and safe.
```

---

## Prompt 3 — Upgrade the creative workspace to full-stack integration

```text
Upgrade the wrapper-level creative workspace so it is no longer just a naive multi-draft generator.

Current situation:
- model-core already has latent scratch, branching, hierarchy, thought graph, and anchor summaries
- wrapper-level CreativeWorkspace is still simple and mostly prompt-variation-based
- I want the fullest practical integration between wrapper cognition and model-core local-deliberation summaries

Files to inspect and modify:
- nanochat/cognition/creative.py
- nanochat/cognition/verifier.py
- nanochat/cognition/sandbox.py
- nanochat/cognition/agent.py
- nanochat/cognition/backend.py
- nanochat/cognition/eval.py
- tests/test_cognition_agent.py
- tests/test_cognition_backend.py
- tests/test_cognition_eval.py
- docs/architecture.md
- docs/evals.md
- documentation.md

Goal:
Turn the creative workspace into a structured multi-stage process that can:
1. generate diverse candidate directions
2. use model_local_delib summaries when available
3. explicitly encourage divergence before collapse
4. hand off candidates to verifier/sandbox more intelligently
5. record why a candidate was explored, rejected, repaired, or chosen

Requirements:
- preserve a demo-backend-friendly path
- preserve engine-backed compatibility
- remain opt-in and backward compatible
- keep the implementation lightweight and inspectable
- do not invent hidden magic or giant abstractions

Desired behavior:
- candidate generation can vary by strategy, not just suffix numbering
- strategies may include:
  - conservative answer
  - divergent idea generation
  - recombination / synthesis
  - long-range memory-grounded answer
  - branch-disagreement resolution
- if model_local_delib metadata is present, use it to adapt the creative policy
- verifier should score not just correctness but also diversity/usefulness/repairability
- sandbox should be able to run branch-and-score comparisons with trace output

Acceptance criteria:
- clearer separation between generation, exploration, ranking, and selection
- improved trace metadata explaining creative-path choices
- tests cover:
  - no local_delib metadata available
  - local_delib metadata available
  - memory-heavy query
  - divergent brainstorming query
  - verify/sandbox path
- docs explain the wrapper-vs-model-core creativity relationship
```

---

## Prompt 4 — Stronger evaluation suite, beyond heuristics

```text
Implement a stronger evaluation suite for the advanced local-deliberation architecture.

Current situation:
- existing evals and ablations are useful but still largely heuristic
- I want the repo to answer: which mechanisms actually help, on what tasks, and at what compute cost?

Files to inspect and modify:
- nanochat/cognition/eval.py
- scripts/cognition_eval.py
- tests/test_cognition_eval.py
- docs/evals.md
- documentation.md

Goal:
Add a more rigorous, still repo-native evaluation framework for:
- exact-recall / long-range dependency tasks
- branch usefulness
- hierarchy usefulness
- scratch usefulness on divergent/creative prompts
- anchor usefulness on long-context summarization/reasoning
- thought-graph usefulness on multi-step structured reasoning
- quality-per-compute tradeoffs

Requirements:
1. Keep the current cheap smoke suite.
2. Add a stronger “research” suite separately.
3. Make outputs machine-readable.
4. Distinguish demo-backend metrics from engine-backed metrics.
5. Include variant-level activation sanity checks so metrics are not interpreted without mechanism activation.
6. Avoid benchmark theater; be explicit where metrics are still proxies.

Add or improve:
- per-task metrics with clearer pass/fail meaning
- long-range context tests
- exact retrieval tests
- branch-consensus utility tests
- deep-hierarchy utility tests
- thought-graph utility tests
- compute accounting tied to executed steps / active mechanisms
- aggregate summary report with deltas vs baseline

Acceptance criteria:
- new suite names are clear and documented
- JSON artifacts include richer metrics and telemetry
- tests cover artifact schema and a few representative cases
- docs explain how to interpret results and what remains heuristic
```

---

## Prompt 5 — Reliable runtime variant override / hot-swapping

```text
Implement reliable runtime variant override / hot-swapping support for local-deliberation architecture variants in evals and engine-backed runs.

Current situation:
- eval docs mention that runtime overrides may depend on backend support
- I want this made real and dependable where possible, not just best-effort telemetry over a loaded checkpoint

Files to inspect and modify:
- nanochat/cognition/eval.py
- scripts/cognition_eval.py
- nanochat/cognition/backend.py
- nanochat/gpt.py
- any engine/checkpoint loading path you need to touch carefully
- tests/test_cognition_eval.py
- tests/test_cognition_backend.py
- docs/evals.md
- documentation.md

Goal:
Make per-variant evaluation actually instantiate/apply the requested local-deliberation config variant at runtime, or fail loudly and explicitly.

Requirements:
1. Do not silently pretend a variant was applied when it was not.
2. Preserve the old path where needed.
3. Reuse model weights safely where config compatibility allows.
4. Clearly define which overrides are safe at runtime and which require fresh instantiation.
5. Mark unsupported combinations explicitly in artifacts.

Acceptance criteria:
- engine-backed eval can truthfully report whether a variant was truly applied
- artifact fields clearly distinguish:
  - applied exactly
  - approximated
  - unsupported
- focused tests cover:
  - safe runtime override
  - unsupported override
  - explicit failure/reporting path
- docs explain the override matrix
```

---

## Prompt 6 — Final full-stack hardening pass

```text
Do a final full-stack hardening pass for the advanced local-deliberation / cognition system after the previous milestones.

Inspect:
- nanochat/local_deliberation.py
- nanochat/gpt.py
- nanochat/cognition/*
- scripts/base_train.py
- scripts/cognition_eval.py
- tests/*
- docs/architecture.md
- docs/evals.md
- documentation.md

Goal:
Stabilize the fullest implementation, reduce regression risk, and leave the repo in a clean research-ready state.

Tasks:
1. Audit all local_delib config fields for wiring consistency.
2. Audit disabled-path parity and near-identity-at-init behavior.
3. Audit cache behavior across prefill and decode.
4. Audit wrapper trace metadata consistency.
5. Audit eval artifact schema consistency.
6. Add missing targeted tests where regressions are most likely.
7. Tighten docs so the current architecture and remaining limitations are unambiguous.
8. Update documentation.md with:
   - implemented state
   - remaining limits
   - recommended ablation commands
   - rollback switches
   - next research priorities

Requirements:
- no repo-wide refactor
- prefer fixes over abstractions
- preserve backward compatibility unless clearly unsafe
- be explicit about anything still approximate or heuristic

Acceptance criteria:
- targeted test slices are green
- docs match code
- no stale config/documentation drift remains
- final summary includes:
  - what is now fully implemented
  - what is still partial
  - what should be tackled next
```

---

## Prompt 7 — Context recovery when Codex drifts

```text
You are working in RT123-new/nanochat.

Recover context before making changes.

Read these files first:
- documentation.md
- docs/architecture.md
- docs/evals.md
- nanochat/gpt.py
- nanochat/local_deliberation.py
- nanochat/cognition/backend.py
- nanochat/cognition/agent.py

Then answer briefly:
1. what advanced local-deliberation capabilities already exist
2. what remaining gap this task is supposed to close
3. which files are the smallest correct touch surface
4. what must not be broken
5. the exact tests/docs you will update

Do not start coding until you have written that short recovery summary.
```

---

## Prompt 8 — Post-milestone PR / changelog summary

```text
Write a concise implementation summary for this milestone in the style of documentation.md.

Include:
- milestone name
- files inspected
- files changed
- what was implemented
- what design choices were made
- commands/tests run
- results
- remaining known issues
- next step

Also write:
1. a short reviewer-facing PR summary
2. a rollback note
3. a one-paragraph explanation of how this changes the current architecture
```

---

## Suggested execution order

Use this exact sequence:

1. **Prompt 0** — confirm the latest gap map before coding
2. **Prompt 1** — close the exact incremental thought-graph decode gap
3. **Prompt 2** — add first-class graph artifacts and traces
4. **Prompt 3** — upgrade the creative workspace into a real orchestrator
5. **Prompt 4** — strengthen evaluation beyond heuristics
6. **Prompt 5** — make runtime variant overrides reliable and explicit
7. **Prompt 6** — harden the whole stack
8. **Prompt 8** — generate the changelog / PR summary after each milestone

Use **Prompt 7** any time Codex starts drifting or loses the repo state.

---

## One-shot master bootstrap prompt

Use this only if you want Codex to manage the milestone sequence itself, but still one milestone at a time.

```text
You are working inside RT123-new/nanochat.

Your job is to take the repository from its current advanced local-deliberation state to the fullest realistic implementation, but you must do it in tightly scoped milestones.

Before coding, read:
- documentation.md
- docs/architecture.md
- docs/evals.md
- nanochat/gpt.py
- nanochat/local_deliberation.py
- nanochat/cognition/backend.py
- nanochat/cognition/agent.py
- nanochat/cognition/creative.py
- nanochat/cognition/eval.py
- scripts/base_train.py
- tests/test_local_deliberation.py
- tests/test_gpt_local_deliberation.py
- tests/test_engine_local_deliberation.py
- tests/test_cognition_backend.py
- tests/test_cognition_agent.py
- tests/test_cognition_eval.py

Work in this order:
1. exact incremental thought-graph decode continuation
2. first-class latent graph artifacts and traces
3. full creative workspace integration with model-core summaries
4. stronger non-heuristic evaluation
5. reliable runtime variant overrides / hot swapping
6. final hardening and docs cleanup

Rules:
- do not attempt all milestones at once
- first write a short recovery summary
- then complete exactly one milestone
- then write a documentation.md-style implementation summary
- then stop

Constraints:
- preserve repo structure
- preserve disabled-path behavior
- preserve backward compatibility unless unsafe
- do not rewrite the main training loop
- do not break engine/speedrun/base train
- keep everything opt-in where possible
- tests first, then code, then docs

When responding, start with:
A. current repo understanding
B. selected milestone
C. exact files to touch
D. acceptance criteria

Then implement only that milestone.
```

---

## Final note

The most important first real implementation task is **Prompt 1**.

That closes the biggest remaining model-core gap and makes the thought-graph path much more complete. After that, the most important task is **Prompt 4**, because once the architecture is in place, the limiting factor becomes proof and measurement rather than more mechanisms.
