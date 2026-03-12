# Codex context-recovery card

Use this when Codex starts drifting, repeating itself, or forgetting repo constraints.

## Read order
1. `documentation.md`
2. `plans.md`
3. `implement.md`
4. `docs/architecture.md`
5. `docs/evals.md`
6. relevant existing repo files for the active milestone
7. any existing `nanochat/cognition/` files

## What Codex must reconstruct
- current milestone
- completed milestones
- remaining acceptance criteria
- repo constraints that cannot be violated
- latest design decisions
- validation status
- next smallest coding step

## Current advanced local-deliberation state
- Implemented: adaptive halting, neighbor graph, flocking, branch consensus / verifier merge, legacy hierarchy, deep hierarchy, persistent scratch slots, explicit thought graph, global anchors, structured decode-cache support for the bounded scratch/anchor/hierarchy path, advanced ablations, and wrapper-level compact thought summaries.
- Still experimental: exact incremental decode continuation for the explicit thought graph, engine-backed runtime variant hot-swapping, and the heuristic quality-per-compute scoring used in lightweight ablations.

## Hard rules
- trust repository files over chat memory
- do not assume unfinished work is complete
- do not widen scope
- do not restructure the repo
- do not disturb training / speedrun code unless explicitly required
- update `documentation.md` after every real change

## Main ablations
- Smoke suite: `python -m scripts.cognition_eval --suite local-delib-ablation --backend demo --output artifacts/local_delib_ablation_eval.json`
- Advanced suite: `python -m scripts.cognition_eval --suite local-delib-ablation-advanced --backend demo --output artifacts/local_delib_ablation_advanced.json`
- Optional checkpoint-backed comparison: `python -m scripts.cognition_eval --suite local-delib-ablation-advanced --backend engine --source sft --output artifacts/local_delib_ablation_advanced_engine.json`

## Major feature switches
- Base path: `--local-delib --local-delib-steps <N>`
- Adaptive compute: `--local-delib-adaptive-halt`
- Neighbor graph / flocking: `--local-delib-use-neighbor-graph`, `--local-delib-use-flocking`, and flocking weights
- Branching: `--local-delib-branch-factor`, `--local-delib-branch-consensus`, `--local-delib-branch-verifier`
- Deep hierarchy: `--local-delib-use-deep-hierarchy`, `--local-delib-span-chunk-size`, `--local-delib-sequence-summary`
- Scratch: `--local-delib-scratch-slots`, `--local-delib-scratch-refine-steps`
- Thought graph: `--local-delib-use-thought-graph`, `--local-delib-thought-node-budget`, `--local-delib-thought-graph-steps`
- Global anchors: `--local-delib-global-anchor-count`, `--local-delib-global-anchor-update`

## Known risks and rollback switches
- Disable the whole subsystem by omitting `--local-delib` or setting `local_delib_steps=0`.
- Disable individual mechanisms by leaving feature booleans off, keeping counts at `0`, or leaving auxiliary weights at `0.0`.
- If engine-backed ablations report `runtime_variant_overrides_applied=false`, do not treat the run as a true per-variant architecture swap.
- If decode-time behavior becomes suspect under explicit thought-graph runs, fall back to the bounded scratch/anchor/hierarchy path or disable `local_delib_use_thought_graph` first.

## Minimal recovery prompt
```text
Recover project state from repository files only.
Read documentation.md, plans.md, implement.md, docs/architecture.md, docs/evals.md, and the relevant current repo files.
Tell me the active milestone, remaining acceptance criteria, repo constraints, and the next smallest coding step.
Then implement only that step, run targeted validation, update documentation.md, and stop.
```
