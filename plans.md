# plans.md

## Project
Nanochat cognition layer for the existing nanochat repository.

## Outcome
A repo-native experimental subsystem that wraps the existing nanochat stack with:
- episodic memory
- semantic memory
- router
- creative workspace
- verifier workspace
- lightweight sandbox
- consolidation / skill reuse
- traceable decision-making

The system should improve capability through structure while preserving nanochat's current minimal, hackable philosophy.

## Global constraints
- Preserve existing repo layout.
- Prefer `nanochat/cognition/` for new implementation.
- Prefer `scripts/cognition_demo.py` for first runnable entrypoint.
- Keep training and speedrun code untouched unless a milestone explicitly says otherwise.
- Keep first tests lightweight and CPU-friendly.
- Remain compatible with Python 3.10+.

## Suggested target file layout
This is a target, not a hard requirement:

```text
nanochat/
  cognition/
    __init__.py
    schemas.py
    backend.py
    memory.py
    router.py
    creative.py
    verifier.py
    sandbox.py
    consolidation.py
    skills.py
    traces.py
    agent.py
scripts/
  cognition_demo.py
  cognition_eval.py            # later milestone if needed
tests/
  test_cognition_schemas.py
  test_cognition_memory.py
  test_cognition_router.py
  test_cognition_agent.py
```

## Milestone 0 - Repo-native scaffold and integration plan
### Goals
- Confirm the existing repo structure and constraints
- Add only the minimum docs / module scaffold needed for the cognition subsystem
- Establish the first isolated entrypoint without disturbing current paths

### Acceptance criteria
- new work follows existing repo layout
- no `src/` migration or broad restructuring
- `nanochat/cognition/` exists with minimal scaffold
- at least one cheap smoke test exists for the new subsystem
- `documentation.md` records repo constraints and first design decisions

## Milestone 1 - Contracts and shared schemas
### Goals
- Define typed interfaces and shared data structures for the cognition subsystem
- Make integration with existing nanochat model / tokenizer / Engine explicit

### Acceptance criteria
- typed schemas exist for episodes, memories, traces, routing decisions, hypotheses, verifications, and skills
- backend interface can wrap existing nanochat generation stack
- focused unit tests validate basic contracts

## Milestone 2 - Memory subsystem
### Goals
- Implement episodic memory and semantic memory for the cognition layer
- Add simple retrieval and write policies

### Acceptance criteria
- memory write and retrieve path works
- relevance + recency strategy exists
- tests cover write -> retrieve -> rank behavior
- design is simple and replaceable

## Milestone 3 - Router
### Goals
- Decide when to answer directly, retrieve memory, explore creatively, verify, sandbox, or consolidate

### Acceptance criteria
- router emits structured decisions with rationale
- common scenarios are tested
- routing stays explicit and inspectable

## Milestone 4 - Creative and verifier workspaces
### Goals
- Add divergent idea generation and convergent critique / ranking
- Keep the first version inspectable rather than fancy

### Acceptance criteria
- creative workspace can produce multiple candidates
- verifier can critique, rank, and optionally repair candidates
- traces show why the final candidate was chosen
- tests validate candidate narrowing behavior

## Milestone 5 - Lightweight sandbox
### Goals
- Add a simple experimentation loop for branching and scoring candidate actions or plans

### Acceptance criteria
- multiple branches can be explored
- outcomes can be scored
- results are written back to episodic memory
- smoke tests prove the loop works without external infrastructure

## Milestone 6 - Consolidation and skill reuse
### Goals
- Distill repeated successful patterns into reusable skills or concepts
- Make future runs able to reuse them

### Acceptance criteria
- repeated wins can produce a skill artifact
- semantic memory and skill registry store provenance and trigger conditions
- later runs can discover and reuse the skill
- regression test proves reuse behavior

## Milestone 7 - End-to-end cognition loop
### Goals
- Connect backend wrapper, memory, router, workspaces, sandbox, and consolidation into a coherent loop
- Deliver a runnable demo path in the existing repo

### Acceptance criteria
- `scripts/cognition_demo.py` or equivalent runs
- end-to-end integration test passes
- output trace is inspectable
- existing nanochat paths remain intact

## Milestone 8 - Evaluation harness
### Goals
- Compare baseline nanochat behavior against cognition-enhanced behavior
- Reuse existing repo evaluation style where sensible

### Acceptance criteria
- at least one evaluation entrypoint exists for cognition experiments
- baseline vs enhanced comparisons are recorded
- evaluation artifacts are documented
- docs explain how to run the evals

## Milestone 9 - Polish and optional deeper integration
### Goals
- Improve docs, examples, config, and graceful failure behavior
- Optionally add carefully scoped integration hooks into chat workflows if justified

### Acceptance criteria
- docs are complete and repo-native
- quickstart for the cognition subsystem works
- optional integration points are documented and justified

## Milestone 10 - Adaptive token compute and halting policy
### Goals
- Extend model-core local deliberation with adaptive per-token halting / variable compute while preserving current default behavior
- Keep controls explicit and off-by-default behind config/flags first

### Acceptance criteria
- per-token halting policy is documented with bounded min/max micro-steps
- implementation plan preserves deterministic fallback to fixed-step behavior
- targeted tests cover halting edge cases (all-halt, no-halt, mixed-halt) without requiring checkpoints

## Milestone 11 - Dynamic latent neighbor graph and flocking
### Goals
- Introduce a dynamic latent nearest-neighbor graph over token states during local deliberation
- Add lightweight flocking-style update rules (align/cohere/separate) as optional latent operators

### Acceptance criteria
- neighbor graph construction strategy (top-k + optional radius cap) is specified and inspectable
- flocking operators are optional and can be disabled with no behavior change to existing path
- debug stats expose neighbor counts and flocking activation rates

## Milestone 12 - Latent branch spawn, competition, and merge
### Goals
- Add latent branch spawning for uncertain/high-salience regions without creating an explicit symbolic executor
- Define merge/reduction policies that collapse branches back into the main token stream

### Acceptance criteria
- spawn triggers are bounded and deterministic under fixed seeds
- merge policy (weighted vote, gating, or verifier-guided) is documented and testable
- cache behavior and memory bounds are defined for decode-time branch handling

## Milestone 13 - Multi-scale latent hierarchy beyond phrase chunks
### Goals
- Expand the current phrase-level grouping into a deeper hierarchy (token -> phrase -> span/segment -> sequence summary)
- Keep hierarchy operators modular so each scale can be enabled independently

### Acceptance criteria
- hierarchy levels and message-passing directions are documented
- at least one cheap synthetic test validates cross-scale information flow
- default configuration remains close to current phrase-centric behavior

## Milestone 14 - Latent creative scratchpad slots
### Goals
- Add persistent latent scratchpad slots for creative exploration inside the model-core deliberation loop
- Keep existing wrapper `CreativeWorkspace` intact as a higher-level orchestrator

### Acceptance criteria
- scratchpad slot lifecycle is defined (allocate, update, readout, release)
- wrapper-level creative generation can optionally condition on scratchpad summaries
- tests verify scratchpad isolation and reset semantics between requests

## Milestone 15 - Auxiliary objectives and hierarchical eval suite
### Goals
- Add optional auxiliary training losses and evals for halting quality, graph consistency, branch utility, and scratchpad usefulness
- Keep speedrun and default training path untouched unless an explicit opt-in flag is provided

### Acceptance criteria
- each auxiliary loss has a disable-by-default switch and clear weighting config
- `docs/evals.md` includes lightweight metrics for variable compute efficiency and quality tradeoffs
- documentation records ablation expectations and rollback criteria for each auxiliary path
