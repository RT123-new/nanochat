# Architecture blueprint for a nanochat cognition layer

## Thesis
The objective is to improve capability through **system structure** while preserving nanochat's existing strengths: minimalism, hackability, and end-to-end clarity.

This is an extension architecture, not a replacement architecture.

## Core design principle
Wrap the current nanochat stack with an optional cognition layer rather than rewriting the repo.

Preferred shape:
- existing model
- existing tokenizer
- existing `Engine`
- cognition controller around them

## Integration stance
The first version should live as a clearly separated subsystem, ideally under `nanochat/cognition/`.

That subsystem should be able to:
- call the current generation stack through an adapter
- retrieve and store memories
- route tasks into different modes
- explore candidates
- verify and rank candidates
- run lightweight branch experiments
- consolidate repeated wins into reusable skills

## Two capability layers (wrapper vs model-core)
The repo now has two distinct capability layers, and they serve different purposes:

1. **Cognition wrapper layer** (`nanochat/cognition/`)
   - An external controller around model inference.
   - Handles episodic/semantic memory, routing, candidate generation, verification, sandbox trials, and consolidation into reusable skills.
   - Produces inspectable traces at the system level.

2. **Model-core local deliberation layer** (`nanochat/gpt.py`)
   - An internal, optional latent module that runs inside the transformer stack.
   - Adds token-local recurrent refinement without changing tokenizer/chat formatting or requiring an external planner.

The model-core layer is a **lightweight graph-of-thought approximation** implemented in latent space, not a full explicit graph executor.

### Local deliberation layer in plain language
When enabled, selected transformer layers can run small internal "thinking" updates per token:
- **Token-local recurrent micro-steps:** each token representation is refined for a small number of internal passes.
- **Phrase pooling / phrase consensus:** nearby tokens are pooled into short phrase chunks so the layer can blend local agreement signals.
- **Optional semantic top-k neighbor path:** a sparse semantic neighbor lookup path can be enabled to mix in top-k latent neighbors when available.
- **Adaptive per-token gating:** token-level gates modulate how much deliberation state should influence each token.
- **Decode-time cache support:** the path is designed to remain compatible with incremental decoding/cache-aware inference behavior when used in decode mode.

This keeps the mechanism small and inspectable: it improves local latent coordination, but does not attempt to execute a full symbolic or tool-driven reasoning graph.

## Core components

### 1. Backend adapter
Purpose:
- provide a stable interface between the cognition subsystem and the existing nanochat generation path

Responsibilities:
- wrap current model + tokenizer + `Engine`
- expose generation in a controlled, testable way
- return structured metadata useful for traces

Current implementation note:
- the cognition layer now includes an `EngineBackend` adapter that can call the checkpoint-backed `Engine` directly for opt-in chat and eval paths
- prompt construction should carry reusable context explicitly through sections such as relevant episodic memory, semantic memory, and skills

### 2. Episodic memory
Stores concrete experiences such as:
- task or prompt
- context
- retrieved memories
- generated candidates
- chosen action
- outcomes
- scores
- reflections

Use it for:
- recalling similar prior episodes
- learning from success and failure history
- grounding later routing decisions

### 3. Semantic memory
Stores distilled and more durable abstractions such as:
- reusable procedures
- lessons learned
- concepts
- trigger heuristics
- skill definitions

Use it for:
- compact reuse
- reducing repeated rediscovery
- supporting later routing and verification

### 4. Router
Chooses among modes such as:
- direct answer
- retrieve memory
- creative exploration
- verification / planning
- sandbox experimentation
- consolidation trigger

The router must be explicit and inspectable.
It should return not just a choice, but also rationale.

### 5. Creative workspace
Purpose:
- generate multiple possible ideas, plans, or framings
- preserve option breadth before collapsing too early

Outputs may include:
- hypotheses
- plan sketches
- alternate framings
- candidate responses

### 6. Verifier workspace
Purpose:
- critique, rank, and optionally repair candidate outputs
- apply constraints and select the best current option

Outputs may include:
- critiques
- rankings
- repair instructions
- confidence estimates
- chosen candidate

### 7. Lightweight sandbox
Purpose:
- allow branch-and-score experimentation without requiring a full world model

Capabilities:
- try multiple candidate actions or plans
- score outcomes using simple heuristics or small task-specific scorers
- record failures and successes
- feed evidence back into episodic memory

### 8. Consolidator
Purpose:
- detect repeated successful patterns
- distill them into reusable skills or concepts

Potential signals:
- repeated success across related contexts
- stable action pattern
- improved outcome after reuse
- lower retries or repair count

### 9. Skill registry
Stores reusable artifacts with:
- identifier
- description
- trigger conditions
- provenance
- supporting evidence
- usage notes

### 10. Trace layer
Every major decision should be inspectable.
Track at minimum:
- router choice and rationale
- retrieved memories
- generated candidates
- verifier rankings
- sandbox outcomes
- consolidation events

## Recommended v1 implementation strategy
- keep storage simple first
- keep tests fake-backed and CPU-only where possible
- keep each module replaceable
- keep integration shallow until the end-to-end loop is proven
- favor explicit data structures over hidden magic

## What not to do in v1
- do not rewrite the main training loop
- do not make the speedrun path depend on cognition modules
- do not introduce heavy persistence or service dependencies too early
- do not hide routing and reasoning decisions in opaque abstractions

## Example end-to-end flow
1. Receive a task
2. Router inspects novelty, ambiguity, and risk
3. Retrieve relevant episodic and semantic memory if needed
4. Generate multiple candidates in the creative workspace if needed
5. Rank or repair them in the verifier workspace
6. Run a lightweight sandbox if uncertainty remains and the task warrants it
7. Return the selected answer or plan
8. Record the full episode
9. Consolidate repeated successful patterns into skills over time
10. Reuse those skills on future related tasks
