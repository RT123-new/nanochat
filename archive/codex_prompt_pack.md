# Codex prompt pack for the existing nanochat repo

These prompts are rewritten for the current `nanochat` repository, not for a greenfield repo.

---

## 0. Bootstrap prompt

Use this first.

```text
This is an existing nanochat repository, not a new repo.

Read these files first:
- README.md
- pyproject.toml
- AGENTS.md
- plans.md
- implement.md
- documentation.md
- docs/architecture.md
- docs/evals.md
- scripts/chat_cli.py
- scripts/chat_eval.py
- tests/test_engine.py

Then inspect the current repo structure before making changes.

I want you to add the developmental cognition subsystem described in the docs, but do it in a repo-native way.

Rules:
1. Do not restructure the repo.
2. Do not introduce `src/` layout.
3. Do not casually modify training or speedrun code.
4. Prefer implementing new work under `nanochat/cognition/`.
5. Keep the first slice small and testable.
6. Stay compatible with Python 3.10+.

Work on Milestone 0 only.
Before changing anything, tell me:
- the active milestone
- the acceptance criteria
- the repo constraints you infer
- the exact files you plan to create or modify

Then implement the smallest clean slice for Milestone 0, run targeted validation, update documentation.md, and summarize what remains.
```

---

## 1. Continue-after-pause prompt

```text
Please resume work on the nanochat cognition subsystem.

First read:
- README.md
- pyproject.toml
- AGENTS.md
- plans.md
- implement.md
- documentation.md
- docs/architecture.md
- docs/evals.md

Then inspect any existing cognition files plus the relevant repo files for the active milestone.

Tell me:
1. which milestone is active
2. what is already done
3. what remains for the active milestone
4. the smallest next implementation step
5. the exact files you plan to touch

Then perform only that next step, run the relevant validation, update documentation.md, and summarize the new state.
```

---

## 2. Lost-context recovery prompt

```text
Recover project state from repository files only.
Do not rely on chat memory.

Read in this order:
1. documentation.md
2. plans.md
3. implement.md
4. docs/architecture.md
5. docs/evals.md
6. relevant source files for the active milestone

Then produce:
- current project status
- active milestone
- remaining acceptance criteria
- repo constraints that must be respected
- exact next coding step

Implement only that next step, run targeted validation, update documentation.md, and stop.
```

---

## 3. Milestone 0 prompt: repo-native scaffold

```text
Complete Milestone 0 only.

This is not a greenfield scaffold task. The repo already exists.

Your job is to add the minimum viable cognition scaffold in a repo-native way.

Required outcomes:
- confirm repo constraints in documentation.md
- add a minimal `nanochat/cognition/` scaffold
- add one cheap smoke test for the new subsystem
- avoid changes to training / speedrun paths
- keep diffs small and focused

Before coding, inspect the existing repo files that matter.
After coding, run only targeted validation unless there is a strong reason to run broader validation.
Update documentation.md with commands run and results.
```

---

## 4. Milestone 1 prompt: contracts and schemas

```text
Complete Milestone 1 only.

Build typed contracts and shared schemas for the cognition subsystem inside the existing repo.

At minimum, define explicit structures for:
- episodes
- memories
- traces
- routing decisions
- hypotheses
- verifications
- skill artifacts
- a backend adapter interface that can wrap nanochat's existing generation stack

Requirements:
- repo-native placement
- Python 3.10+ compatibility
- focused unit tests
- no heavy implementation yet beyond what is needed to validate the contracts

Run targeted tests and update documentation.md.
```

---

## 5. Milestone 2 prompt: memory subsystem

```text
Complete Milestone 2 only.

Implement simple episodic and semantic memory modules for the cognition layer.

Requirements:
- simple storage first
- relevance + recency retrieval
- clear write policy helpers
- tests for write -> retrieve -> rank
- architecture notes updated where needed

Do not overengineer persistence in v1.
Run targeted validation and update documentation.md.
```

---

## 6. Milestone 3 prompt: router

```text
Complete Milestone 3 only.

Implement an explicit router for the cognition subsystem that can choose among:
- direct answer
- retrieve memory
- creative exploration
- verification / planning
- sandbox experiment
- consolidation trigger

Requirements:
- structured decision output
- rationale included in traces
- configurable heuristics or thresholds
- tests for common scenarios and edge cases

Keep it inspectable, not magical.
Run targeted validation and update documentation.md.
```

---

## 7. Milestone 4 prompt: creative and verifier workspaces

```text
Complete Milestone 4 only.

Implement:
- a creative workspace that generates multiple candidate ideas or plans
- a verifier workspace that critiques, ranks, and optionally repairs them
- traces showing the narrowing process

Favor readability and inspectability over sophistication.
Add targeted tests and update documentation.md.
```

---

## 8. Milestone 5 prompt: lightweight sandbox

```text
Complete Milestone 5 only.

Implement a lightweight sandbox for branch-and-score experimentation.

Requirements:
- branch multiple candidate actions or plans
- score outcomes
- record failures and successes
- write outcomes to episodic memory
- keep the abstraction easy to swap later

Do not build a heavyweight world model.
Add smoke tests, run targeted validation, and update documentation.md.
```

---

## 9. Milestone 6 prompt: consolidation and skills

```text
Complete Milestone 6 only.

Implement a consolidator that:
- detects repeated successful patterns
- distills them into reusable skills or concepts
- stores provenance and trigger conditions
- writes them to semantic memory and a skill registry
- enables later reuse

Add regression tests for repeated-task improvement behavior.
Run targeted validation and update documentation.md.
```

---

## 10. Milestone 7 prompt: end-to-end loop

```text
Complete Milestone 7 only.

Connect the cognition subsystem into a coherent end-to-end loop.

Deliver:
- a runnable `scripts/cognition_demo.py` or justified equivalent
- at least one end-to-end integration test
- readable trace output
- updated run instructions

Keep existing nanochat paths intact.
Run the relevant validation and update documentation.md.
```

---

## 11. Evaluation prompt

```text
Complete Milestone 8 only.

Design and implement a lightweight evaluation harness for the cognition subsystem.

Use the existing repo's evaluation style where sensible, but keep the first version cheap and controlled.

Deliver:
- at least one baseline vs cognition-enhanced comparison
- documented evaluation commands
- machine-readable artifacts if practical
- notes on limitations of the first eval design

Run the relevant validation and update documentation.md.
```
