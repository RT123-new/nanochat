# Chat Quality Snapshot

## Run metadata
- base_dir: `/Users/RT1/.cache/nanochat`
- source: `sft`
- model_tag: `m1demo`
- step: `999`
- device_type: `mps`
- temperature: `0.0`
- top_k: `50`
- max_tokens: `192`
- output_dir: `artifacts/chat_quality/20260316_115453`

## Prompt comparisons

### explain_gradient_clipping

**Prompt**

```text
Explain gradient clipping in plain English and give one short example.
```

**Cognition route**: `direct_answer`
**Trace steps**: `route:direct_answer`

**Baseline**

```text
are
, is are
, is are
```

**Cognition**

```text
are
, is are
, is are
```

### debug_training_nan

**Prompt**

```text
Give me a five-step plan to debug training loss suddenly becoming NaN.
```

**Cognition route**: `direct_answer`
**Trace steps**: `route:direct_answer`

**Baseline**

```text
the: the, is not
, is not is not
, is not
```

**Cognition**

```text
the: the, is not
, is not is not
, is not
```

### python_fibonacci

**Prompt**

```text
Write a short Python function that returns Fibonacci numbers up to n.
```

**Cognition route**: `direct_answer`
**Trace steps**: `route:direct_answer`

**Baseline**

```text
the, is not be are
 are
 are
```

**Cognition**

```text
the, is not be are
 are
 are
```

### brainstorm_memory_features

**Prompt**

```text
Brainstorm four practical memory features for a tiny chatbot.
```

**Cognition route**: `creative_explore`
**Trace steps**: `route:creative_explore, creative_strategies:conservative_answer,divergent_ideas, candidates:2, verifier_score:0.17`

**Baseline**

```text
are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are are
```

**Cognition**

```text
We are
 are
 are
```

## Task-grounded eval
- status: `skipped`
