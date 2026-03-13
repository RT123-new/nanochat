"""Lightweight evaluation harness for baseline vs cognition comparisons."""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .agent import CognitionAgent
from .backend import (
    BackendAdapter,
    EngineBackend,
    LocalDelibRuntimeOverrideError,
    LocalDelibRuntimeOverrideReport,
    build_local_delib_namespaced_metadata,
)
from .schemas import Episode, MemoryItem, SkillArtifact


class PromptBackend(Protocol):
    """Protocol for simple prompt-to-text evaluators."""

    def generate(self, prompt: str, **kwargs: object) -> str:
        """Return one response for the provided prompt."""


@dataclass(slots=True)
class EvalCase:
    """Single benchmark-style prompt with expected keywords."""

    case_id: str
    prompt: str
    expected_keywords: list[str]
    seeded_episodes: list[Episode] = field(default_factory=list)
    seeded_semantic_items: list[MemoryItem] = field(default_factory=list)
    seeded_skills: list[SkillArtifact] = field(default_factory=list)
    requires_cognition_gain: bool = False


@dataclass(slots=True)
class EvalRow:
    """Result row for one eval case."""

    case_id: str
    baseline_response: str
    cognition_response: str
    baseline_score: float
    cognition_score: float
    cognition_decision: str
    creative_strategy_ids: list[str] = field(default_factory=list)
    creative_selected_strategy: str | None = None
    creative_candidate_count: int = 0
    creative_handoff: str | None = None
    creative_model_summary_used: bool = False


@dataclass(slots=True)
class EvalSummary:
    """Aggregate summary of baseline vs cognition runs."""

    baseline_mean: float
    cognition_mean: float
    delta: float
    route_counts: dict[str, int]
    rows: list[EvalRow]


@dataclass(slots=True)
class LocalDelibEvalCase:
    """Single prompt for model-side local deliberation ablations."""

    case_id: str
    prompt: str
    expected_keywords: list[str]


@dataclass(slots=True)
class LocalDelibVariant:
    """Configuration variant for local deliberation architecture."""

    variant_id: str
    generation_kwargs: dict[str, Any]


@dataclass(slots=True)
class LocalDelibEvalRow:
    """Per prompt/variant result row with advanced stats."""

    case_id: str
    variant_id: str
    response: str
    score: float
    runtime_override_applied: bool
    runtime_override_status: str
    runtime_override_application_method: str | None
    runtime_override_reason: str | None
    model_local_delib_branch: dict[str, float]
    model_local_delib_hierarchy: dict[str, float]
    model_local_delib_scratchpad: dict[str, float]
    model_local_delib_adaptive_halt: dict[str, float]
    model_local_delib_graph_artifact: dict[str, Any]


@dataclass(slots=True)
class LocalDelibEvalSummary:
    """Ablation summary for local deliberation architecture variants."""

    variant_mean_scores: dict[str, float]
    runtime_variant_overrides_applied: bool
    runtime_variant_override_statuses: dict[str, str]
    runtime_variant_override_counts: dict[str, int]
    rows: list[LocalDelibEvalRow]


@dataclass(slots=True)
class AdvancedLocalDelibEvalRow:
    """Per prompt/variant row for the Prompt 10 advanced ablation suite."""

    case_id: str
    variant_id: str
    response: str
    quality_proxy_score: float
    quality_per_compute: float
    runtime_override_applied: bool
    runtime_override_status: str
    runtime_override_application_method: str | None
    runtime_override_reason: str | None
    compute_proxy_metrics: dict[str, float]
    neighbor_graph_stats: dict[str, float]
    branch_stats: dict[str, float]
    hierarchy_stats: dict[str, float]
    scratch_stats: dict[str, float]
    thought_graph_stats: dict[str, float]
    flocking_stats: dict[str, float]
    anchor_stats: dict[str, float]
    model_local_delib_graph_artifact: dict[str, Any]


@dataclass(slots=True)
class AdvancedLocalDelibEvalSummary:
    """Aggregate Prompt 10 ablation summary with per-variant telemetry."""

    quality_proxy_scores: dict[str, dict[str, float]]
    variant_mean_scores: dict[str, float]
    quality_per_compute: dict[str, float]
    compute_proxy_metrics: dict[str, dict[str, float]]
    mean_steps_taken: dict[str, float]
    neighbor_graph_stats: dict[str, dict[str, float]]
    branch_stats: dict[str, dict[str, float]]
    hierarchy_stats: dict[str, dict[str, float]]
    scratch_stats: dict[str, dict[str, float]]
    thought_graph_stats: dict[str, dict[str, float]]
    flocking_stats: dict[str, dict[str, float]]
    anchor_stats: dict[str, dict[str, float]]
    runtime_variant_overrides_applied: bool
    runtime_variant_override_statuses: dict[str, str]
    runtime_variant_override_counts: dict[str, int]
    rows: list[AdvancedLocalDelibEvalRow]


@dataclass(slots=True)
class ResearchLocalDelibEvalCase:
    """Structured Prompt 4 research case with clearer pass/fail semantics."""

    case_id: str
    task_family: str
    prompt: str
    expected_fields: dict[str, str]
    pass_threshold: float = 1.0
    targeted_mechanisms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResearchLocalDelibEvalRow:
    """Per prompt/variant row for the Prompt 4 research suite."""

    case_id: str
    task_family: str
    variant_id: str
    response: str
    metric_score: float
    quality_per_compute: float
    pass_threshold: float
    passed: bool
    response_format_ok: bool
    runtime_override_applied: bool
    runtime_override_status: str
    runtime_override_application_method: str | None
    runtime_override_reason: str | None
    backend_kind: str
    metric_tier: str
    targeted_mechanisms: list[str]
    expected_activations: list[str]
    activation_checks: dict[str, bool]
    activation_ok: bool
    metrics_interpretable: bool
    task_metrics: dict[str, float]
    active_mechanisms: list[str]
    compute_accounting: dict[str, float]
    neighbor_graph_stats: dict[str, float]
    branch_stats: dict[str, float]
    hierarchy_stats: dict[str, float]
    scratch_stats: dict[str, float]
    thought_graph_stats: dict[str, float]
    flocking_stats: dict[str, float]
    anchor_stats: dict[str, float]
    model_local_delib_graph_artifact: dict[str, Any]


@dataclass(slots=True)
class ResearchLocalDelibEvalSummary:
    """Aggregate Prompt 4 research eval summary."""

    backend_kind: str
    metric_tier: str
    baseline_variant_id: str
    variant_mean_scores: dict[str, float]
    variant_pass_rates: dict[str, float]
    delta_vs_baseline: dict[str, float]
    case_scores: dict[str, dict[str, float]]
    case_deltas_vs_baseline: dict[str, dict[str, float]]
    task_family_scores: dict[str, dict[str, float]]
    quality_per_compute: dict[str, float]
    compute_accounting: dict[str, dict[str, float]]
    activation_coverage: dict[str, dict[str, Any]]
    runtime_variant_overrides_applied: bool
    runtime_variant_override_statuses: dict[str, str]
    runtime_variant_override_counts: dict[str, int]
    rows: list[ResearchLocalDelibEvalRow]


@dataclass(slots=True)
class TaskGroundedEvalRow:
    """Per-example row for real task-graded baseline vs cognition evals."""

    task_name: str
    example_index: int
    prompt: str
    baseline_response: str
    cognition_response: str
    baseline_score: float
    cognition_score: float
    baseline_passed: bool
    cognition_passed: bool
    cognition_decision: str
    benchmark_eligible: bool
    baseline_runtime_override_status: str
    baseline_runtime_override_application_method: str | None
    baseline_runtime_override_reason: str | None
    cognition_runtime_override_status: str
    cognition_runtime_override_application_method: str | None
    cognition_runtime_override_reason: str | None
    cognition_trace_metadata: dict[str, Any]


@dataclass(slots=True)
class TaskGroundedEvalSummary:
    """Aggregate summary for task-native graded baseline vs cognition comparisons."""

    backend_kind: str
    metric_tier: str
    checkpoint_identity: dict[str, Any]
    task_names: list[str]
    baseline_mean: float
    cognition_mean: float
    delta: float
    proof_baseline_mean: float
    proof_cognition_mean: float
    proof_delta: float
    per_task: dict[str, dict[str, float]]
    rows: list[TaskGroundedEvalRow]


@dataclass(slots=True)
class NaturalLocalDelibEvalCase:
    """Natural-language local-deliberation benchmark case with task-specific grading."""

    case_id: str
    task_family: str
    prompt: str
    expected_answers: dict[str, str]
    pass_threshold: float = 1.0
    targeted_mechanisms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NaturalLocalDelibEvalRow:
    """Per prompt/variant row for the natural-language local-deliberation suite."""

    case_id: str
    task_family: str
    variant_id: str
    response: str
    metric_score: float
    quality_per_compute: float
    pass_threshold: float
    passed: bool
    grader_extractable: bool
    proof_eligible: bool
    proof_passed: bool
    runtime_override_applied: bool
    runtime_override_status: str
    runtime_override_application_method: str | None
    runtime_override_reason: str | None
    backend_kind: str
    metric_tier: str
    targeted_mechanisms: list[str]
    expected_activations: list[str]
    activation_checks: dict[str, bool]
    activation_ok: bool
    metrics_interpretable: bool
    task_metrics: dict[str, float]
    active_mechanisms: list[str]
    compute_accounting: dict[str, float]
    neighbor_graph_stats: dict[str, float]
    branch_stats: dict[str, float]
    hierarchy_stats: dict[str, float]
    scratch_stats: dict[str, float]
    thought_graph_stats: dict[str, float]
    flocking_stats: dict[str, float]
    anchor_stats: dict[str, float]
    model_local_delib_graph_artifact: dict[str, Any]


@dataclass(slots=True)
class NaturalLocalDelibEvalSummary:
    """Aggregate summary for the natural-language local-deliberation suite."""

    backend_kind: str
    metric_tier: str
    baseline_variant_id: str
    checkpoint_identity: dict[str, Any]
    variant_mean_scores: dict[str, float]
    variant_pass_rates: dict[str, float]
    proof_variant_mean_scores: dict[str, float]
    proof_pass_rates: dict[str, float]
    delta_vs_baseline: dict[str, float]
    proof_delta_vs_baseline: dict[str, float]
    case_scores: dict[str, dict[str, float]]
    task_family_scores: dict[str, dict[str, float]]
    quality_per_compute: dict[str, float]
    compute_accounting: dict[str, dict[str, float]]
    activation_coverage: dict[str, dict[str, Any]]
    runtime_variant_overrides_applied: bool
    runtime_variant_override_statuses: dict[str, str]
    runtime_variant_override_counts: dict[str, int]
    rows: list[NaturalLocalDelibEvalRow]


@dataclass(slots=True)
class EngineSmokeCommandRecord:
    """Command executed by the optional engine smoke harness."""

    label: str
    argv: list[str]
    artifact_path: str | None = None
    exit_code: int | None = None


@dataclass(slots=True)
class EngineSmokeArtifactRecord:
    """Artifact emitted by the optional engine smoke harness."""

    label: str
    path: str
    row_count: int
    runtime_override_statuses: list[str] = field(default_factory=list)
    runtime_override_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class EngineSmokeManifest:
    """Machine-readable summary of an optional engine smoke run."""

    status: str
    strict_audit: bool
    checkpoint_identity: dict[str, Any]
    commands: list[EngineSmokeCommandRecord]
    artifacts: list[EngineSmokeArtifactRecord]
    observed_runtime_override_statuses: list[str]
    reason: str | None = None


DEFAULT_CASES: list[EvalCase] = [
    EvalCase(
        case_id="memory_recall",
        prompt="Can you recall the previous summarization guidance for this draft?",
        expected_keywords=["terse", "bullet", "citations"],
        seeded_episodes=[
            Episode(
                episode_id="episode-style",
                prompt="What is our summarization style?",
                response="Use terse bullet summaries with citations.",
                tags=["summarization"],
                metadata={"success": True, "trigger": "summarization"},
            )
        ],
        requires_cognition_gain=True,
    ),
    EvalCase(
        case_id="memory_reuse_paraphrase",
        prompt="Please summarize this draft for me.",
        expected_keywords=["terse", "bullet", "citations"],
        seeded_episodes=[
            Episode(
                episode_id="episode-style-paraphrase",
                prompt="What is our summarization style?",
                response="Use terse bullet summaries with citations.",
                tags=["summarization"],
                metadata={"success": True, "trigger": "summarization"},
            )
        ],
        requires_cognition_gain=True,
    ),
    EvalCase(
        case_id="semantic_house_style",
        prompt="Answer in our house style for this reply.",
        expected_keywords=["brief", "neutral", "numbered"],
        seeded_semantic_items=[
            MemoryItem(
                item_id="semantic-house-style",
                kind="semantic",
                content="House style: brief neutral numbered answers.",
            )
        ],
        requires_cognition_gain=True,
    ),
    EvalCase(
        case_id="skill_reuse_paraphrase",
        prompt="Please summarize this draft.",
        expected_keywords=["extract", "bullets", "condense"],
        seeded_skills=[
            SkillArtifact(
                skill_id="skill-summarization",
                name="Reusable summarization checklist",
                trigger="summarization",
                procedure=["extract bullets", "condense the bullets"],
            )
        ],
        requires_cognition_gain=True,
    ),
]


DEFAULT_LOCAL_DELIB_CASES: list[LocalDelibEvalCase] = [
    LocalDelibEvalCase(
        case_id="planning_prompt",
        prompt="Create a short plan for writing clean tests and verify edge cases.",
        expected_keywords=["plan", "tests", "verify"],
    ),
    LocalDelibEvalCase(
        case_id="creative_prompt",
        prompt="Brainstorm two concise ideas for memory-aware response improvements.",
        expected_keywords=["ideas", "memory", "improvements"],
    ),
]


def _build_research_prompt(task_id: str, *lines: str) -> str:
    return "\n".join((f"RESEARCH_TASK: {task_id}", *lines))


DEFAULT_LOCAL_DELIB_VARIANTS: list[LocalDelibVariant] = [
    LocalDelibVariant(variant_id="local_delib_off", generation_kwargs={"local_delib": False}),
    LocalDelibVariant(variant_id="local_delib_basic", generation_kwargs={"local_delib": True}),
    LocalDelibVariant(
        variant_id="local_delib_adaptive_halt",
        generation_kwargs={"local_delib": True, "local_delib_adaptive_halt": True},
    ),
    LocalDelibVariant(
        variant_id="local_delib_branch",
        generation_kwargs={"local_delib": True, "local_delib_branch_factor": 2, "local_delib_branch_every": 1},
    ),
    LocalDelibVariant(
        variant_id="local_delib_hierarchy",
        generation_kwargs={"local_delib": True, "local_delib_hierarchy_chunk_sizes": "4,8"},
    ),
    LocalDelibVariant(
        variant_id="local_delib_scratchpad",
        generation_kwargs={"local_delib": True, "local_delib_scratch_slots": 2, "local_delib_scratch_dim": 16},
    ),
]


ADVANCED_LOCAL_DELIB_CASES: list[LocalDelibEvalCase] = [
    LocalDelibEvalCase(
        case_id="basic_planning",
        prompt="Create a short plan and verify the edge cases before answering.",
        expected_keywords=["plan", "verify", "steps"],
    ),
    LocalDelibEvalCase(
        case_id="neighbor_graph",
        prompt="Explain how latent neighbors can align, cohere, and separate during reasoning.",
        expected_keywords=["neighbors", "align", "cohere", "separate"],
    ),
    LocalDelibEvalCase(
        case_id="branch_consensus",
        prompt="Compare candidate branches, build a consensus, and verify the final merge.",
        expected_keywords=["branches", "consensus", "verify", "merge"],
    ),
    LocalDelibEvalCase(
        case_id="deep_hierarchy",
        prompt="Summarize this sequence with phrase, span, and sequence-level structure.",
        expected_keywords=["phrase", "span", "sequence", "summary"],
    ),
    LocalDelibEvalCase(
        case_id="scratch_refine",
        prompt="Draft an answer, refine it in scratch space, and return a concise revision.",
        expected_keywords=["draft", "refine", "revision"],
    ),
    LocalDelibEvalCase(
        case_id="thought_anchor",
        prompt="Use thought nodes and global anchors to preserve long-range context.",
        expected_keywords=["thought", "anchor", "context", "long-range"],
    ),
]


RESEARCH_LOCAL_DELIB_CASES: list[ResearchLocalDelibEvalCase] = [
    ResearchLocalDelibEvalCase(
        case_id="exact_recall",
        task_family="exact_recall",
        prompt=_build_research_prompt(
            "exact_recall",
            "Structured exact-recall probe with long-range filler.",
            "Archive note at the start: codeword ivory-cobalt-7.",
            "Middle filler: latency audit, branch logs, verifier notes, and route traces.",
            "Late checksum note: 47.",
            "Return exactly two lines:",
            "NEEDLE=<codeword>",
            "CHECKSUM=<checksum>",
        ),
        expected_fields={"NEEDLE": "ivory-cobalt-7", "CHECKSUM": "47"},
        targeted_mechanisms=["exact_recall", "long_range_context"],
    ),
    ResearchLocalDelibEvalCase(
        case_id="branch_consensus",
        task_family="branch_utility",
        prompt=_build_research_prompt(
            "branch_consensus",
            "Candidate branch_a keeps alpha only.",
            "Candidate branch_b keeps alpha,beta,gamma and is the consistent choice.",
            "Candidate branch_c keeps alpha,gamma but drops beta.",
            "Return exactly two lines:",
            "WINNER=<best branch id>",
            "MERGED=<comma-separated kept facts>",
        ),
        expected_fields={"WINNER": "branch_b", "MERGED": "alpha,beta,gamma"},
        targeted_mechanisms=["branch_consensus", "verifier_merge"],
    ),
    ResearchLocalDelibEvalCase(
        case_id="deep_hierarchy",
        task_family="hierarchy_utility",
        prompt=_build_research_prompt(
            "deep_hierarchy",
            "Summarize the example at phrase, span, and sequence scales.",
            "Phrase evidence: cache | skills.",
            "Span evidence: memory -> routing.",
            "Sequence conclusion: stable-controller.",
            "Return exactly three lines:",
            "PHRASE=<phrase summary>",
            "SPAN=<span summary>",
            "SEQUENCE=<sequence summary>",
        ),
        expected_fields={
            "PHRASE": "cache|skills",
            "SPAN": "memory->routing",
            "SEQUENCE": "stable-controller",
        },
        targeted_mechanisms=["deep_hierarchy"],
    ),
    ResearchLocalDelibEvalCase(
        case_id="scratch_refine",
        task_family="scratch_utility",
        prompt=_build_research_prompt(
            "scratch_refine",
            "Generate two distinct ideas and then collapse them into one revision.",
            "Use these exact idea payloads: cache audit and memory routing.",
            "Return exactly three lines:",
            "IDEA_1=<first idea>",
            "IDEA_2=<second idea>",
            "REVISION=<combined revision>",
        ),
        expected_fields={
            "IDEA_1": "cache audit",
            "IDEA_2": "memory routing",
            "REVISION": "cache audit + memory routing",
        },
        targeted_mechanisms=["scratch_refinement", "creative_divergence"],
    ),
    ResearchLocalDelibEvalCase(
        case_id="anchor_long_context",
        task_family="anchor_utility",
        prompt=_build_research_prompt(
            "anchor_long_context",
            "Long-context summarization probe.",
            "Earliest fact: canal.",
            "Intervening filler: planning notes, test matrix, branch metrics, and scratch traces.",
            "Latest fact: harbor.",
            "Return exactly two lines:",
            "ANCHOR=<earliest|latest facts>",
            "SUMMARY=<single concise summary>",
        ),
        expected_fields={
            "ANCHOR": "canal|harbor",
            "SUMMARY": "canal linked to harbor",
        },
        targeted_mechanisms=["global_anchors", "long_context_summary"],
    ),
    ResearchLocalDelibEvalCase(
        case_id="thought_graph",
        task_family="thought_graph_utility",
        prompt=_build_research_prompt(
            "thought_graph",
            "Structured reasoning probe with two causal hops.",
            "alpha implies beta.",
            "beta implies gamma.",
            "Return exactly three lines:",
            "STEP_1=<first hop>",
            "STEP_2=<second hop>",
            "ANSWER=<final answer>",
        ),
        expected_fields={
            "STEP_1": "alpha->beta",
            "STEP_2": "beta->gamma",
            "ANSWER": "gamma",
        },
        targeted_mechanisms=["thought_graph", "multi_step_reasoning"],
    ),
]


ADVANCED_LOCAL_DELIB_VARIANTS: list[LocalDelibVariant] = [
    LocalDelibVariant(variant_id="local_delib_off", generation_kwargs={"local_delib": False}),
    LocalDelibVariant(variant_id="local_delib_basic", generation_kwargs={"local_delib": True}),
    LocalDelibVariant(
        variant_id="local_delib_adaptive_halt",
        generation_kwargs={"local_delib": True, "local_delib_adaptive_halt": True},
    ),
    LocalDelibVariant(
        variant_id="local_delib_neighbor_graph",
        generation_kwargs={
            "local_delib": True,
            "local_delib_use_neighbor_graph": True,
            "local_delib_use_phrase_consensus": True,
            "local_delib_semantic_topk": 4,
        },
    ),
    LocalDelibVariant(
        variant_id="local_delib_flocking",
        generation_kwargs={
            "local_delib": True,
            "local_delib_use_neighbor_graph": True,
            "local_delib_use_phrase_consensus": True,
            "local_delib_use_flocking": True,
            "local_delib_flocking_alignment_weight": 0.4,
            "local_delib_flocking_cohesion_weight": 0.3,
            "local_delib_flocking_separation_weight": 0.2,
        },
    ),
    LocalDelibVariant(
        variant_id="local_delib_branch_consensus_verifier",
        generation_kwargs={
            "local_delib": True,
            "local_delib_branch_factor": 3,
            "local_delib_branch_every": 1,
            "local_delib_branch_consensus": True,
            "local_delib_branch_verifier": True,
        },
    ),
    LocalDelibVariant(
        variant_id="local_delib_deep_hierarchy",
        generation_kwargs={
            "local_delib": True,
            "local_delib_use_deep_hierarchy": True,
            "local_delib_span_chunk_size": 8,
            "local_delib_sequence_summary": True,
            "local_delib_hierarchy_bidirectional": True,
        },
    ),
    LocalDelibVariant(
        variant_id="local_delib_scratch_refine",
        generation_kwargs={
            "local_delib": True,
            "local_delib_scratch_slots": 2,
            "local_delib_scratch_dim": 16,
            "local_delib_scratch_refine_steps": 2,
        },
    ),
    LocalDelibVariant(
        variant_id="local_delib_thought_graph",
        generation_kwargs={
            "local_delib": True,
            "local_delib_use_thought_graph": True,
            "local_delib_thought_node_budget": 6,
            "local_delib_thought_graph_steps": 2,
        },
    ),
    LocalDelibVariant(
        variant_id="local_delib_global_anchors",
        generation_kwargs={
            "local_delib": True,
            "local_delib_global_anchor_count": 3,
            "local_delib_global_anchor_update": True,
            "local_delib_global_anchor_use_hierarchy": True,
        },
    ),
    LocalDelibVariant(
        variant_id="local_delib_combo_reasoner",
        generation_kwargs={
            "local_delib": True,
            "local_delib_adaptive_halt": True,
            "local_delib_use_neighbor_graph": True,
            "local_delib_use_phrase_consensus": True,
            "local_delib_use_flocking": True,
            "local_delib_flocking_alignment_weight": 0.4,
            "local_delib_flocking_cohesion_weight": 0.3,
            "local_delib_flocking_separation_weight": 0.2,
            "local_delib_branch_factor": 3,
            "local_delib_branch_every": 1,
            "local_delib_branch_consensus": True,
            "local_delib_branch_verifier": True,
        },
    ),
    LocalDelibVariant(
        variant_id="local_delib_combo_full_stack",
        generation_kwargs={
            "local_delib": True,
            "local_delib_adaptive_halt": True,
            "local_delib_use_neighbor_graph": True,
            "local_delib_use_phrase_consensus": True,
            "local_delib_use_flocking": True,
            "local_delib_flocking_alignment_weight": 0.4,
            "local_delib_flocking_cohesion_weight": 0.3,
            "local_delib_flocking_separation_weight": 0.2,
            "local_delib_branch_factor": 3,
            "local_delib_branch_every": 1,
            "local_delib_branch_consensus": True,
            "local_delib_branch_verifier": True,
            "local_delib_use_deep_hierarchy": True,
            "local_delib_span_chunk_size": 8,
            "local_delib_sequence_summary": True,
            "local_delib_hierarchy_bidirectional": True,
            "local_delib_scratch_slots": 2,
            "local_delib_scratch_dim": 16,
            "local_delib_scratch_refine_steps": 2,
            "local_delib_use_thought_graph": True,
            "local_delib_thought_node_budget": 6,
            "local_delib_thought_graph_steps": 2,
            "local_delib_global_anchor_count": 3,
            "local_delib_global_anchor_update": True,
            "local_delib_global_anchor_use_hierarchy": True,
            "local_delib_global_anchor_use_scratch": True,
            "local_delib_global_anchor_use_thought": True,
        },
    ),
]


RESEARCH_LOCAL_DELIB_VARIANTS: list[LocalDelibVariant] = ADVANCED_LOCAL_DELIB_VARIANTS

TASK_GROUNDED_DEFAULT_TASKS: tuple[str, ...] = ("GSM8K", "SpellingBee", "HumanEval")
TASK_GROUNDED_SMOKE_TASK = "SmokeTinyTask"
BENCHMARK_ELIGIBLE_OVERRIDE_STATUSES = {"not_requested", "exact"}

NATURAL_LOCAL_DELIB_CASES: list[NaturalLocalDelibEvalCase] = [
    NaturalLocalDelibEvalCase(
        case_id="needle_recall",
        task_family="exact_recall",
        prompt=(
            "You are reviewing a long project note. Near the beginning, the archive code is "
            "ivory-cobalt-7 and the checksum is 47. After several unrelated notes about latency, "
            "branch logs, verifier scores, and scratch traces, answer the final question in one short sentence: "
            "what were the archive code and checksum?"
        ),
        expected_answers={"needle": "ivory-cobalt-7", "checksum": "47"},
        targeted_mechanisms=["exact_recall", "long_range_context"],
    ),
    NaturalLocalDelibEvalCase(
        case_id="branch_consistency",
        task_family="branch_utility",
        prompt=(
            "Three candidate branches disagree. Branch A keeps only alpha. "
            "Branch B keeps alpha, beta, and gamma and is the internally consistent choice. "
            "Branch C keeps alpha and gamma but drops beta. "
            "Reply in one short sentence saying which branch should survive and which facts should be kept."
        ),
        expected_answers={"winner": "branch_b", "facts": "alpha,beta,gamma"},
        targeted_mechanisms=["branch_consensus", "verifier_merge"],
    ),
    NaturalLocalDelibEvalCase(
        case_id="deep_hierarchy_natural",
        task_family="hierarchy_utility",
        prompt=(
            "Give one concise explanation that preserves three levels of detail: "
            "at the phrase level say 'cache and skills', "
            "at the span level say 'memory guides routing', "
            "and at the sequence level say 'stable controller'."
        ),
        expected_answers={
            "phrase": "cache and skills",
            "span": "memory guides routing",
            "sequence": "stable controller",
        },
        targeted_mechanisms=["deep_hierarchy"],
    ),
    NaturalLocalDelibEvalCase(
        case_id="scratch_refine_natural",
        task_family="scratch_utility",
        prompt=(
            "Start with two ideas, 'cache audit' and 'memory routing', then return one short refined answer "
            "that combines both ideas in plain English."
        ),
        expected_answers={
            "idea_1": "cache audit",
            "idea_2": "memory routing",
            "revision": "cache audit + memory routing",
        },
        targeted_mechanisms=["scratch_refinement", "creative_divergence"],
    ),
    NaturalLocalDelibEvalCase(
        case_id="anchor_long_context_natural",
        task_family="anchor_utility",
        prompt=(
            "A long note begins with the fact 'canal' and ends with the fact 'harbor'. "
            "After lots of unrelated detail, answer with one short sentence connecting the earliest and latest facts."
        ),
        expected_answers={"summary": "canal linked to harbor"},
        targeted_mechanisms=["global_anchors", "long_context_summary"],
    ),
    NaturalLocalDelibEvalCase(
        case_id="thought_graph_natural",
        task_family="thought_graph_utility",
        prompt=(
            "If alpha leads to beta and beta leads to gamma, what follows from alpha? "
            "Answer in one short sentence that makes the two-hop chain explicit."
        ),
        expected_answers={
            "step_1": "alpha leads to beta",
            "step_2": "beta leads to gamma",
            "answer": "gamma",
        },
        targeted_mechanisms=["thought_graph", "multi_step_reasoning"],
    ),
]

NATURAL_LOCAL_DELIB_VARIANTS: list[LocalDelibVariant] = ADVANCED_LOCAL_DELIB_VARIANTS


class ContextAwareEvalBackend:
    """Deterministic backend that only improves when support context is injected."""

    def generate(self, prompt: str, **kwargs: object) -> str:
        skill_lines = _extract_section(prompt, "Relevant skill:")
        semantic_lines = _extract_section(prompt, "Relevant semantic memory:")
        episodic_lines = _extract_section(prompt, "Relevant episodic memory:")
        strategy_lines = _extract_section(prompt, "Creative strategy:")
        strategy_id = _extract_prefixed_line(strategy_lines, "- id:")

        if strategy_id == "divergent_ideas":
            lower_prompt = prompt.lower()
            terms = ["ideas", "options", "angles"]
            if "memory" in lower_prompt:
                terms.append("memory")
            return "Explore " + " ".join(terms)

        if strategy_id == "memory_grounded":
            if skill_lines:
                procedure_lines = [line.removeprefix("- ").strip() for line in skill_lines if "extract" in line or "condense" in line]
                if procedure_lines:
                    return "Grounded by prior skill: " + "; ".join(procedure_lines)
            if semantic_lines:
                cleaned = [line.split(":", 1)[-1].strip() for line in semantic_lines if ":" in line]
                if cleaned:
                    return "Grounded by stored guidance: " + " ".join(cleaned)
            if episodic_lines:
                cleaned = [line.split("->", 1)[-1].strip() for line in episodic_lines if "->" in line]
                if cleaned:
                    return "Grounded by prior episode: " + " ".join(cleaned)
            return "Grounded answer with retrieved support."

        if strategy_id == "branch_resolution":
            return "Resolve branches into one consensus choice and verify the merge."

        if strategy_id == "recombination":
            return "Synthesize the strongest parts into one grounded revision."

        if strategy_id == "conservative_answer":
            return "Provide a grounded direct answer."

        if skill_lines:
            procedure_lines = [line.removeprefix("- ").strip() for line in skill_lines if "extract" in line or "condense" in line]
            if procedure_lines:
                return "Apply this skill: " + "; ".join(procedure_lines)

        if semantic_lines:
            cleaned = [line.split(":", 1)[-1].strip() for line in semantic_lines if ":" in line]
            if cleaned:
                return "Follow this stored guidance: " + " ".join(cleaned)

        if episodic_lines:
            cleaned = [line.split("->", 1)[-1].strip() for line in episodic_lines if "->" in line]
            if cleaned:
                return "Based on prior guidance: " + " ".join(cleaned)

        return "I can answer directly, but I do not have any prior guidance to reuse yet."


class LocalDelibContextEvalBackend:
    """Deterministic model-side backend that emits advanced local-delib metadata."""

    last_generation_metadata: dict[str, Any] | None
    supports_local_delib_runtime_overrides = True

    def __init__(self) -> None:
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        requested_overrides = {
            str(key): value
            for key, value in kwargs.items()
            if str(key) == "local_delib" or str(key).startswith("local_delib_")
        }
        local_delib = bool(kwargs.get("local_delib", False))
        adaptive_halt = bool(kwargs.get("local_delib_adaptive_halt", False))
        branch_factor = int(kwargs.get("local_delib_branch_factor", 1))
        branch_consensus = bool(kwargs.get("local_delib_branch_consensus", False))
        branch_verifier = bool(kwargs.get("local_delib_branch_verifier", False))
        hierarchy = bool(kwargs.get("local_delib_hierarchy_chunk_sizes"))
        use_deep_hierarchy = bool(kwargs.get("local_delib_use_deep_hierarchy", False))
        scratch_slots = int(kwargs.get("local_delib_scratch_slots", 0))
        scratch_refine_steps = int(kwargs.get("local_delib_scratch_refine_steps", 0))
        use_neighbor_graph = bool(kwargs.get("local_delib_use_neighbor_graph", False))
        use_flocking = bool(kwargs.get("local_delib_use_flocking", False))
        semantic_topk = int(kwargs.get("local_delib_semantic_topk", 0))
        use_thought_graph = bool(kwargs.get("local_delib_use_thought_graph", False))
        thought_graph_steps = int(kwargs.get("local_delib_thought_graph_steps", 0))
        thought_node_budget = int(kwargs.get("local_delib_thought_node_budget", 0))
        global_anchor_count = int(kwargs.get("local_delib_global_anchor_count", 0))

        hierarchy_active = hierarchy or use_deep_hierarchy
        hierarchy_levels_used = 3.0 if use_deep_hierarchy else (2.0 if hierarchy else 0.0)
        mean_steps_taken = 1.5 if adaptive_halt else (2.0 if local_delib else 0.0)
        executed_steps = 5.0 if adaptive_halt else (6.0 if local_delib else 0.0)
        neighbor_count = 4.0 if use_neighbor_graph else 0.0
        flocking_active = use_flocking and use_neighbor_graph
        thought_nodes_used = float(min(max(thought_node_budget, 0), 6)) if use_thought_graph else 0.0
        global_anchors_used = float(max(global_anchor_count, 0)) if global_anchor_count > 0 else 0.0

        stats = [{
            "layer_idx": 0,
            "agreement": 0.7 if local_delib else 0.0,
            "branch_factor_used": float(branch_factor if local_delib and branch_factor > 1 else 0),
            "mean_branch_score": 0.6 if branch_factor > 1 else 0.0,
            "mean_branch_disagreement": 0.35 if branch_consensus else 0.0,
            "mean_branch_consensus_weight": 0.55 if branch_consensus else 0.0,
            "mean_branch_verifier_score": 0.68 if branch_verifier else 0.0,
            "branch_consensus_used": 1.0 if branch_consensus else 0.0,
            "hierarchy_levels_used": hierarchy_levels_used,
            "mean_hierarchy_feedback_norm": 0.4 if hierarchy_active else 0.0,
            "phrase_nodes_used": 4.0 if use_deep_hierarchy else 0.0,
            "span_nodes_used": 2.0 if use_deep_hierarchy else 0.0,
            "sequence_summary_used": 1.0 if use_deep_hierarchy else 0.0,
            "mean_upward_message_norm": 0.31 if use_deep_hierarchy else 0.0,
            "mean_downward_message_norm": 0.22 if use_deep_hierarchy else 0.0,
            "mean_scale_gate": 0.58 if use_deep_hierarchy else 0.0,
            "hierarchy_depth_used": 3.0 if use_deep_hierarchy else 0.0,
            "scratch_slots_used": float(scratch_slots if local_delib else 0),
            "mean_scratch_read_weight": 0.35 if scratch_slots > 0 else 0.0,
            "mean_scratch_write_weight": 0.28 if scratch_slots > 0 else 0.0,
            "mean_scratch_refine_norm": 0.42 if scratch_refine_steps > 0 else 0.0,
            "mean_scratch_summary_norm": 0.25 if scratch_slots > 0 else 0.0,
            "scratch_reset_ok": 1.0,
            "halted_token_fraction": 0.5 if adaptive_halt else 0.0,
            "mean_steps_taken": mean_steps_taken,
            "executed_steps": executed_steps,
            "mean_executed_steps_per_token": mean_steps_taken,
            "max_executed_steps_any_token": 3.0 if adaptive_halt else (2.0 if local_delib else 0.0),
            "fraction_halted_early": 0.5 if adaptive_halt else 0.0,
            "mean_halt": 0.46 if adaptive_halt else 0.0,
            "mean_final_halt": 0.79 if adaptive_halt else 0.0,
            "mean_neighbor_count": neighbor_count,
            "mean_sequence_neighbor_weight": 0.24 if use_neighbor_graph else 0.0,
            "mean_semantic_neighbor_weight": 0.19 if use_neighbor_graph else 0.0,
            "mean_phrase_neighbor_weight": 0.21 if use_neighbor_graph else 0.0,
            "semantic_topk_used": float(semantic_topk if use_neighbor_graph else 0),
            "mean_alignment_norm": 0.26 if flocking_active else 0.0,
            "mean_cohesion_norm": 0.23 if flocking_active else 0.0,
            "mean_separation_norm": 0.18 if flocking_active else 0.0,
            "mean_flocking_total_norm": 0.67 if flocking_active else 0.0,
            "flocking_neighbor_count": 3.0 if flocking_active else 0.0,
            "fraction_flocking_tokens_active": 0.75 if flocking_active else 0.0,
            "thought_nodes_used": thought_nodes_used,
            "mean_thought_degree": 2.5 if use_thought_graph else 0.0,
            "mean_token_to_thought_weight": 0.29 if use_thought_graph else 0.0,
            "mean_thought_to_token_weight": 0.33 if use_thought_graph else 0.0,
            "mean_thought_update_norm": 0.37 if use_thought_graph else 0.0,
            "thought_graph_steps_used": float(thought_graph_steps if use_thought_graph else 0),
            "global_anchors_used": global_anchors_used,
            "mean_anchor_read_weight": 0.27 if global_anchor_count > 0 else 0.0,
            "mean_anchor_write_weight": 0.22 if global_anchor_count > 0 else 0.0,
            "mean_anchor_norm": 0.41 if global_anchor_count > 0 else 0.0,
        }]
        self.last_generation_metadata = {
            "local_deliberation_stats": stats,
            **build_local_delib_namespaced_metadata(stats),
            "local_delib_runtime_override": _report_to_metadata(
                LocalDelibRuntimeOverrideReport(
                    status="exact",
                    requested_overrides=requested_overrides,
                    applied_overrides=requested_overrides,
                    application_method="demo_backend",
                )
            ),
        }
        research_task = _extract_prefixed_line(prompt.splitlines(), "RESEARCH_TASK:")
        if research_task is not None:
            return self._research_response(
                task_id=research_task,
                local_delib=local_delib,
                branch_factor=branch_factor,
                branch_consensus=branch_consensus,
                branch_verifier=branch_verifier,
                hierarchy_active=hierarchy_active,
                use_deep_hierarchy=use_deep_hierarchy,
                scratch_slots=scratch_slots,
                scratch_refine_steps=scratch_refine_steps,
                use_thought_graph=use_thought_graph,
                global_anchor_count=global_anchor_count,
            )
        natural_task = _identify_natural_task(prompt)
        if natural_task is not None:
            return self._natural_response(
                task_id=natural_task,
                local_delib=local_delib,
                branch_factor=branch_factor,
                branch_consensus=branch_consensus,
                branch_verifier=branch_verifier,
                hierarchy_active=hierarchy_active,
                use_deep_hierarchy=use_deep_hierarchy,
                scratch_slots=scratch_slots,
                scratch_refine_steps=scratch_refine_steps,
                use_thought_graph=use_thought_graph,
                global_anchor_count=global_anchor_count,
            )
        response_terms = self._response_terms(
            prompt=prompt,
            local_delib=local_delib,
            adaptive_halt=adaptive_halt,
            use_neighbor_graph=use_neighbor_graph,
            use_flocking=flocking_active,
            branch_factor=branch_factor,
            branch_consensus=branch_consensus,
            branch_verifier=branch_verifier,
            hierarchy_active=hierarchy_active,
            use_deep_hierarchy=use_deep_hierarchy,
            scratch_slots=scratch_slots,
            scratch_refine_steps=scratch_refine_steps,
            use_thought_graph=use_thought_graph,
            global_anchor_count=global_anchor_count,
        )
        return " ".join(response_terms).strip()

    def _response_terms(
        self,
        *,
        prompt: str,
        local_delib: bool,
        adaptive_halt: bool,
        use_neighbor_graph: bool,
        use_flocking: bool,
        branch_factor: int,
        branch_consensus: bool,
        branch_verifier: bool,
        hierarchy_active: bool,
        use_deep_hierarchy: bool,
        scratch_slots: int,
        scratch_refine_steps: int,
        use_thought_graph: bool,
        global_anchor_count: int,
    ) -> list[str]:
        lower_prompt = prompt.lower()
        terms: list[str] = []
        if local_delib:
            terms.extend(["plan", "steps"])
        if adaptive_halt:
            terms.append("verify")
        if "neighbor" in lower_prompt or "align" in lower_prompt:
            if use_neighbor_graph:
                terms.extend(["neighbors", "align"])
            if use_flocking:
                terms.extend(["cohere", "separate"])
        if "branch" in lower_prompt or "consensus" in lower_prompt:
            if branch_factor > 1:
                terms.extend(["branches", "merge"])
            if branch_consensus:
                terms.append("consensus")
            if branch_verifier:
                terms.append("verify")
        if "phrase" in lower_prompt or "sequence" in lower_prompt:
            if hierarchy_active:
                terms.append("summary")
            if use_deep_hierarchy:
                terms.extend(["phrase", "span", "sequence"])
        if "scratch" in lower_prompt or "draft" in lower_prompt:
            if scratch_slots > 0:
                terms.append("draft")
            if scratch_refine_steps > 0:
                terms.extend(["refine", "revision"])
        if "thought" in lower_prompt or "anchor" in lower_prompt:
            if use_thought_graph:
                terms.append("thought")
            if global_anchor_count > 0:
                terms.extend(["anchor", "context", "long-range"])
        if not terms:
            return ["direct", "response", "without", "extra", "latent", "support"]
        return terms

    def _research_response(
        self,
        *,
        task_id: str,
        local_delib: bool,
        branch_factor: int,
        branch_consensus: bool,
        branch_verifier: bool,
        hierarchy_active: bool,
        use_deep_hierarchy: bool,
        scratch_slots: int,
        scratch_refine_steps: int,
        use_thought_graph: bool,
        global_anchor_count: int,
    ) -> str:
        if task_id == "exact_recall":
            if global_anchor_count > 0:
                return _format_structured_response({"NEEDLE": "ivory-cobalt-7", "CHECKSUM": "47"})
            if local_delib:
                return _format_structured_response({"NEEDLE": "ivory-cobalt-7", "CHECKSUM": "41"})
            return _format_structured_response({"NEEDLE": "ivory", "CHECKSUM": "0"})

        if task_id == "branch_consensus":
            if branch_consensus and branch_verifier:
                return _format_structured_response({"WINNER": "branch_b", "MERGED": "alpha,beta,gamma"})
            if branch_factor > 1:
                return _format_structured_response({"WINNER": "branch_b", "MERGED": "alpha,beta"})
            return _format_structured_response({"WINNER": "branch_a", "MERGED": "alpha"})

        if task_id == "deep_hierarchy":
            if use_deep_hierarchy:
                return _format_structured_response(
                    {
                        "PHRASE": "cache|skills",
                        "SPAN": "memory->routing",
                        "SEQUENCE": "stable-controller",
                    }
                )
            if hierarchy_active:
                return _format_structured_response(
                    {
                        "PHRASE": "cache|skills",
                        "SPAN": "memory",
                        "SEQUENCE": "stable-controller",
                    }
                )
            return _format_structured_response({"PHRASE": "cache", "SEQUENCE": "controller"})

        if task_id == "scratch_refine":
            if scratch_slots > 0 and scratch_refine_steps > 0:
                return _format_structured_response(
                    {
                        "IDEA_1": "cache audit",
                        "IDEA_2": "memory routing",
                        "REVISION": "cache audit + memory routing",
                    }
                )
            if scratch_slots > 0:
                return _format_structured_response(
                    {
                        "IDEA_1": "cache audit",
                        "IDEA_2": "cache audit",
                        "REVISION": "cache audit",
                    }
                )
            return _format_structured_response({"IDEA_1": "cache audit", "IDEA_2": "cache audit"})

        if task_id == "anchor_long_context":
            if global_anchor_count > 0:
                return _format_structured_response(
                    {
                        "ANCHOR": "canal|harbor",
                        "SUMMARY": "canal linked to harbor",
                    }
                )
            if use_thought_graph or use_deep_hierarchy:
                return _format_structured_response({"ANCHOR": "canal", "SUMMARY": "canal linked"})
            return _format_structured_response({"SUMMARY": "harbor"})

        if task_id == "thought_graph":
            if use_thought_graph:
                return _format_structured_response(
                    {
                        "STEP_1": "alpha->beta",
                        "STEP_2": "beta->gamma",
                        "ANSWER": "gamma",
                    }
                )
            if branch_consensus or use_deep_hierarchy:
                return _format_structured_response({"ANSWER": "gamma", "STEP_1": "alpha"})
            return _format_structured_response({"ANSWER": "beta"})

        return _format_structured_response({"STATUS": "unsupported"})

    def _natural_response(
        self,
        *,
        task_id: str,
        local_delib: bool,
        branch_factor: int,
        branch_consensus: bool,
        branch_verifier: bool,
        hierarchy_active: bool,
        use_deep_hierarchy: bool,
        scratch_slots: int,
        scratch_refine_steps: int,
        use_thought_graph: bool,
        global_anchor_count: int,
    ) -> str:
        if task_id == "needle_recall":
            if global_anchor_count > 0:
                return "The archive code was ivory-cobalt-7 and the checksum was 47."
            if local_delib:
                return "The archive code was ivory-cobalt-7 and the checksum was 41."
            return "I only remember ivory."

        if task_id == "branch_consistency":
            if branch_consensus and branch_verifier:
                return "Branch B should survive, keeping alpha, beta, and gamma."
            if branch_factor > 1:
                return "Branch B looks strongest, but I would only keep alpha and beta."
            return "Branch A survives with alpha only."

        if task_id == "deep_hierarchy_natural":
            if use_deep_hierarchy:
                return (
                    "At the phrase level I keep cache and skills, at the span level memory guides routing, "
                    "and overall this becomes a stable controller."
                )
            if hierarchy_active:
                return "I keep cache and skills, and overall it becomes a stable controller."
            return "This ends as a controller."

        if task_id == "scratch_refine_natural":
            if scratch_slots > 0 and scratch_refine_steps > 0:
                return (
                    "I would start with cache audit and memory routing, then combine them into one revision: "
                    "cache audit + memory routing."
                )
            if scratch_slots > 0:
                return "I would start with cache audit twice and keep cache audit."
            return "I would just do a cache audit."

        if task_id == "anchor_long_context_natural":
            if global_anchor_count > 0:
                return "The canal linked to the harbor."
            if use_thought_graph or use_deep_hierarchy:
                return "The canal eventually connects to the harbor."
            return "The harbor mattered most."

        if task_id == "thought_graph_natural":
            if use_thought_graph:
                return "Because alpha leads to beta and beta leads to gamma, the answer is gamma."
            if branch_consensus or use_deep_hierarchy:
                return "Alpha leads to beta, so I think the answer is gamma."
            return "Beta follows from alpha."

        return "I do not have a grounded answer."


def score_keywords(response: str, expected_keywords: list[str]) -> float:
    """Return simple keyword recall score in [0, 1]."""
    if not expected_keywords:
        return 0.0
    response_text = response.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in response_text)
    return hits / len(expected_keywords)


def _format_structured_response(fields: dict[str, str]) -> str:
    return "\n".join(f"{key}={value}" for key, value in fields.items())


def _parse_structured_response(response: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in response.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _csv_overlap_score(value: str | None, expected: str) -> float:
    if value is None:
        return 0.0
    actual_items = {item.strip() for item in value.split(",") if item.strip()}
    expected_items = {item.strip() for item in expected.split(",") if item.strip()}
    if not expected_items:
        return 0.0
    return len(actual_items & expected_items) / len(expected_items)


def _field_exact_score(parsed: dict[str, str], key: str, expected: str) -> float:
    return 1.0 if parsed.get(key) == expected else 0.0


def _identify_natural_task(prompt: str) -> str | None:
    lower_prompt = prompt.lower()
    if "archive code is ivory-cobalt-7" in lower_prompt:
        return "needle_recall"
    if "three candidate branches disagree" in lower_prompt:
        return "branch_consistency"
    if "phrase level say 'cache and skills'" in lower_prompt:
        return "deep_hierarchy_natural"
    if "start with two ideas, 'cache audit' and 'memory routing'" in lower_prompt:
        return "scratch_refine_natural"
    if "a long note begins with the fact 'canal' and ends with the fact 'harbor'" in lower_prompt:
        return "anchor_long_context_natural"
    if "if alpha leads to beta and beta leads to gamma" in lower_prompt:
        return "thought_graph_natural"
    return None


def _normalize_text(value: str) -> str:
    lowered = value.casefold()
    lowered = lowered.replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return " ".join(lowered.split())


def _contains_normalized_phrase(response: str, phrase: str) -> bool:
    normalized_response = f" {_normalize_text(response)} "
    normalized_phrase = f" {_normalize_text(phrase)} "
    return normalized_phrase in normalized_response


def _contains_all_normalized_phrases(response: str, phrases: list[str]) -> bool:
    return all(_contains_normalized_phrase(response, phrase) for phrase in phrases)


def _extract_branch_choice(response: str) -> str | None:
    match = re.search(r"\bbranch[\s_-]*([abc])\b", response, flags=re.IGNORECASE)
    if match is None:
        return None
    return f"branch_{match.group(1).lower()}"


def _extract_fact_coverage(response: str, expected_csv: str) -> float:
    expected = [item.strip() for item in expected_csv.split(",") if item.strip()]
    if not expected:
        return 0.0
    response_text = _normalize_text(response)
    hits = sum(1 for item in expected if _normalize_text(item) in response_text)
    return hits / len(expected)


def _score_task_pass(passed: bool) -> float:
    return 1.0 if passed else 0.0


def _score_research_case(
    case: ResearchLocalDelibEvalCase,
    response: str,
) -> tuple[float, dict[str, float], bool]:
    parsed = _parse_structured_response(response)
    response_format_ok = all(field in parsed for field in case.expected_fields)

    if case.case_id == "exact_recall":
        metrics = {
            "needle_exact": _field_exact_score(parsed, "NEEDLE", case.expected_fields["NEEDLE"]),
            "checksum_exact": _field_exact_score(parsed, "CHECKSUM", case.expected_fields["CHECKSUM"]),
        }
        score = 0.6 * metrics["needle_exact"] + 0.4 * metrics["checksum_exact"]
        return score, metrics, response_format_ok

    if case.case_id == "branch_consensus":
        metrics = {
            "winner_exact": _field_exact_score(parsed, "WINNER", case.expected_fields["WINNER"]),
            "merged_support": _csv_overlap_score(parsed.get("MERGED"), case.expected_fields["MERGED"]),
        }
        score = 0.5 * metrics["winner_exact"] + 0.5 * metrics["merged_support"]
        return score, metrics, response_format_ok

    if case.case_id == "deep_hierarchy":
        metrics = {
            "phrase_exact": _field_exact_score(parsed, "PHRASE", case.expected_fields["PHRASE"]),
            "span_exact": _field_exact_score(parsed, "SPAN", case.expected_fields["SPAN"]),
            "sequence_exact": _field_exact_score(parsed, "SEQUENCE", case.expected_fields["SEQUENCE"]),
        }
        score = (metrics["phrase_exact"] + metrics["span_exact"] + metrics["sequence_exact"]) / 3.0
        return score, metrics, response_format_ok

    if case.case_id == "scratch_refine":
        metrics = {
            "idea_1_exact": _field_exact_score(parsed, "IDEA_1", case.expected_fields["IDEA_1"]),
            "idea_2_exact": _field_exact_score(parsed, "IDEA_2", case.expected_fields["IDEA_2"]),
            "revision_exact": _field_exact_score(parsed, "REVISION", case.expected_fields["REVISION"]),
            "ideas_distinct": 1.0 if parsed.get("IDEA_1") and parsed.get("IDEA_1") != parsed.get("IDEA_2") else 0.0,
        }
        score = (
            metrics["idea_1_exact"]
            + metrics["idea_2_exact"]
            + metrics["revision_exact"]
            + metrics["ideas_distinct"]
        ) / 4.0
        return score, metrics, response_format_ok

    if case.case_id == "anchor_long_context":
        metrics = {
            "anchor_exact": _field_exact_score(parsed, "ANCHOR", case.expected_fields["ANCHOR"]),
            "summary_exact": _field_exact_score(parsed, "SUMMARY", case.expected_fields["SUMMARY"]),
        }
        score = 0.5 * metrics["anchor_exact"] + 0.5 * metrics["summary_exact"]
        return score, metrics, response_format_ok

    if case.case_id == "thought_graph":
        metrics = {
            "step_1_exact": _field_exact_score(parsed, "STEP_1", case.expected_fields["STEP_1"]),
            "step_2_exact": _field_exact_score(parsed, "STEP_2", case.expected_fields["STEP_2"]),
            "answer_exact": _field_exact_score(parsed, "ANSWER", case.expected_fields["ANSWER"]),
        }
        score = (
            metrics["step_1_exact"] + metrics["step_2_exact"] + metrics["answer_exact"]
        ) / 3.0
        return score, metrics, response_format_ok

    return 0.0, {}, response_format_ok


def _score_natural_case(
    case: NaturalLocalDelibEvalCase,
    response: str,
) -> tuple[float, dict[str, float], bool]:
    grader_extractable = bool(response.strip())

    if case.case_id == "needle_recall":
        metrics = {
            "needle_exact": 1.0 if _contains_normalized_phrase(response, case.expected_answers["needle"]) else 0.0,
            "checksum_exact": 1.0 if _contains_normalized_phrase(response, case.expected_answers["checksum"]) else 0.0,
        }
        score = 0.6 * metrics["needle_exact"] + 0.4 * metrics["checksum_exact"]
        return score, metrics, grader_extractable

    if case.case_id == "branch_consistency":
        metrics = {
            "winner_exact": 1.0 if _extract_branch_choice(response) == case.expected_answers["winner"] else 0.0,
            "fact_support": _extract_fact_coverage(response, case.expected_answers["facts"]),
        }
        score = 0.5 * metrics["winner_exact"] + 0.5 * metrics["fact_support"]
        return score, metrics, grader_extractable

    if case.case_id == "deep_hierarchy_natural":
        metrics = {
            "phrase_exact": 1.0 if _contains_normalized_phrase(response, case.expected_answers["phrase"]) else 0.0,
            "span_exact": 1.0 if _contains_normalized_phrase(response, case.expected_answers["span"]) else 0.0,
            "sequence_exact": 1.0 if _contains_normalized_phrase(response, case.expected_answers["sequence"]) else 0.0,
        }
        score = (metrics["phrase_exact"] + metrics["span_exact"] + metrics["sequence_exact"]) / 3.0
        return score, metrics, grader_extractable

    if case.case_id == "scratch_refine_natural":
        idea_1 = _contains_normalized_phrase(response, case.expected_answers["idea_1"])
        idea_2 = _contains_normalized_phrase(response, case.expected_answers["idea_2"])
        revision = _contains_normalized_phrase(response, case.expected_answers["revision"])
        metrics = {
            "idea_1_exact": 1.0 if idea_1 else 0.0,
            "idea_2_exact": 1.0 if idea_2 else 0.0,
            "revision_exact": 1.0 if revision else 0.0,
            "ideas_distinct": 1.0 if idea_1 and idea_2 else 0.0,
        }
        score = (
            metrics["idea_1_exact"]
            + metrics["idea_2_exact"]
            + metrics["revision_exact"]
            + metrics["ideas_distinct"]
        ) / 4.0
        return score, metrics, grader_extractable

    if case.case_id == "anchor_long_context_natural":
        metrics = {
            "summary_exact": 1.0 if _contains_normalized_phrase(response, case.expected_answers["summary"]) else 0.0,
            "anchor_support": 1.0 if _contains_all_normalized_phrases(response, ["canal", "harbor"]) else 0.0,
        }
        score = 0.7 * metrics["summary_exact"] + 0.3 * metrics["anchor_support"]
        return score, metrics, grader_extractable

    if case.case_id == "thought_graph_natural":
        metrics = {
            "step_1_exact": 1.0 if _contains_normalized_phrase(response, case.expected_answers["step_1"]) else 0.0,
            "step_2_exact": 1.0 if _contains_normalized_phrase(response, case.expected_answers["step_2"]) else 0.0,
            "answer_exact": 1.0 if _contains_normalized_phrase(response, case.expected_answers["answer"]) else 0.0,
        }
        score = (
            metrics["step_1_exact"] + metrics["step_2_exact"] + metrics["answer_exact"]
        ) / 3.0
        return score, metrics, grader_extractable

    return 0.0, {}, grader_extractable


class _TaskGroundedSmokeTask:
    """Offline task-grounded fixture used only by engine smoke validation."""

    def __init__(self) -> None:
        self.rows = [
            {
                "messages": [
                    {"role": "user", "content": "solve arithmetic task"},
                    {"role": "assistant", "content": "#### 2"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "write a small function"},
                    {"role": "assistant", "content": "def ok():\n    return 1"},
                ]
            },
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def evaluate(self, conversation: dict[str, Any], assistant_response: str) -> bool:
        prompt = conversation["messages"][0]["content"]
        if prompt == "solve arithmetic task":
            return assistant_response.strip() == "#### 2"
        return assistant_response.strip() == "def ok():\n    return 1"


def _resolve_task_grounded_task_names(task_names: list[str] | None) -> list[str]:
    resolved = list(task_names) if task_names else list(TASK_GROUNDED_DEFAULT_TASKS)
    supported = set(TASK_GROUNDED_DEFAULT_TASKS) | {TASK_GROUNDED_SMOKE_TASK}
    unknown = sorted(set(resolved) - supported)
    if unknown:
        raise ValueError(
            f"Unsupported task-grounded tasks: {', '.join(unknown)}. "
            f"Choose from {', '.join(sorted(supported))}."
        )
    return resolved


def _build_task_grounded_task(task_name: str, *, max_problems: int | None) -> Any:
    if task_name == TASK_GROUNDED_SMOKE_TASK:
        return _TaskGroundedSmokeTask()
    if task_name == "GSM8K":
        from tasks.gsm8k import GSM8K

        return GSM8K(subset="main", split="test")
    if task_name == "SpellingBee":
        from tasks.spellingbee import SpellingBee

        size = max(1, max_problems) if max_problems is not None else 1000
        return SpellingBee(size=size, split="test")
    if task_name == "HumanEval":
        from tasks.humaneval import HumanEval

        return HumanEval()
    raise ValueError(f"Unsupported task-grounded task: {task_name}")


def _select_indices(total: int, *, max_items: int | None, seed: int) -> list[int]:
    indices = list(range(total))
    if max_items is None or max_items >= total:
        return indices
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:max_items]


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content)


def _prompt_messages_from_conversation(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    messages = conversation.get("messages", [])
    if not isinstance(messages, list):
        return []
    if not messages:
        return []
    if isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
        return [dict(message) for message in messages[:-1] if isinstance(message, dict)]
    return [dict(message) for message in messages if isinstance(message, dict)]


def _prompt_text_from_conversation(conversation: dict[str, Any]) -> str:
    prompt_messages = _prompt_messages_from_conversation(conversation)
    if not prompt_messages:
        return ""
    return "\n".join(
        _message_content_to_text(message.get("content", ""))
        for message in prompt_messages
    ).strip()


def _checkpoint_identity_payload(checkpoint_identity: dict[str, Any] | None) -> dict[str, Any]:
    return dict(checkpoint_identity or {})


def _benchmark_eligible_from_statuses(*statuses: str) -> bool:
    return all(status in BENCHMARK_ELIGIBLE_OVERRIDE_STATUSES for status in statuses)


COMPUTE_PROXY_KEYS = (
    "mean_steps_taken",
    "halted_token_fraction",
)
NEIGHBOR_GRAPH_KEYS = (
    "mean_neighbor_count",
    "mean_sequence_neighbor_weight",
    "mean_semantic_neighbor_weight",
    "mean_phrase_neighbor_weight",
    "semantic_topk_used",
)
BRANCH_KEYS = (
    "agreement",
    "branch_factor_used",
    "mean_branch_score",
    "mean_branch_disagreement",
    "mean_branch_consensus_weight",
    "mean_branch_verifier_score",
    "branch_consensus_used",
)
HIERARCHY_KEYS = (
    "hierarchy_levels_used",
    "mean_hierarchy_feedback_norm",
    "phrase_nodes_used",
    "span_nodes_used",
    "sequence_summary_used",
    "mean_upward_message_norm",
    "mean_downward_message_norm",
    "mean_scale_gate",
    "hierarchy_depth_used",
)
SCRATCH_KEYS = (
    "scratch_slots_used",
    "mean_scratch_read_weight",
    "mean_scratch_write_weight",
    "mean_scratch_refine_norm",
    "mean_scratch_summary_norm",
    "scratch_reset_ok",
)
THOUGHT_GRAPH_KEYS = (
    "thought_nodes_used",
    "mean_thought_degree",
    "mean_token_to_thought_weight",
    "mean_thought_to_token_weight",
    "mean_thought_update_norm",
    "thought_graph_steps_used",
)
FLOCKING_KEYS = (
    "mean_alignment_norm",
    "mean_cohesion_norm",
    "mean_separation_norm",
    "mean_flocking_total_norm",
    "flocking_neighbor_count",
    "fraction_flocking_tokens_active",
)
ANCHOR_KEYS = (
    "global_anchors_used",
    "mean_anchor_read_weight",
    "mean_anchor_write_weight",
    "mean_anchor_norm",
)


RESEARCH_COMPUTE_KEYS = (
    "executed_steps",
    "mean_executed_steps_per_token",
    "max_executed_steps_any_token",
    "fraction_halted_early",
    "mean_steps_taken",
    "halted_token_fraction",
)

RUNTIME_OVERRIDE_STATUS_PRIORITY = {
    "not_requested": 0,
    "exact": 1,
    "approximated": 2,
    "unsupported": 3,
}


def _supports_local_delib_runtime_overrides(backend: PromptBackend) -> bool:
    return bool(getattr(backend, "supports_local_delib_runtime_overrides", False))


def _infer_backend_kind(backend: PromptBackend) -> str:
    if isinstance(backend, LocalDelibContextEvalBackend):
        return "demo"
    if isinstance(backend, EngineBackend) or hasattr(backend, "engine"):
        return "engine"
    return "external"


def _infer_metric_tier(backend_kind: str) -> str:
    if backend_kind == "demo":
        return "deterministic_structured"
    if backend_kind == "engine":
        return "structured_prompt_proxy"
    return "external_prompt_proxy"


def _report_to_metadata(report: LocalDelibRuntimeOverrideReport) -> dict[str, Any]:
    return report.to_metadata()


def _unsupported_runtime_override_report(
    requested_overrides: dict[str, Any],
    *,
    reason: str,
    application_method: str = "unsupported",
) -> dict[str, Any]:
    return _report_to_metadata(
        LocalDelibRuntimeOverrideReport(
            status="unsupported",
            requested_overrides=requested_overrides,
            applied_overrides={},
            application_method=application_method,
            reason=reason,
        )
    )


def _coerce_runtime_override_report(
    metadata: dict[str, Any] | None,
    *,
    requested_overrides: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(metadata, dict):
        raw = metadata.get("local_delib_runtime_override")
        if isinstance(raw, dict):
            return {
                "status": str(raw.get("status", "unsupported")),
                "requested_overrides": dict(raw.get("requested_overrides", requested_overrides)),
                "applied_overrides": dict(raw.get("applied_overrides", {})),
                "application_method": raw.get("application_method"),
                "reason": raw.get("reason"),
            }
    if requested_overrides:
        return _report_to_metadata(
            LocalDelibRuntimeOverrideReport(
                status="exact",
                requested_overrides=requested_overrides,
                applied_overrides=requested_overrides,
                application_method="backend_runtime_kwargs",
            )
        )
    return _report_to_metadata(
        LocalDelibRuntimeOverrideReport(
            status="not_requested",
            requested_overrides={},
            applied_overrides={},
            application_method="none",
        )
    )


def _variant_override_statuses(rows: list[Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    default_priority = max(RUNTIME_OVERRIDE_STATUS_PRIORITY.values(), default=0) + 1
    for row in rows:
        candidate = str(row.runtime_override_status)
        current = statuses.get(row.variant_id)
        candidate_priority = RUNTIME_OVERRIDE_STATUS_PRIORITY.get(candidate, default_priority)
        current_priority = RUNTIME_OVERRIDE_STATUS_PRIORITY.get(current, -1) if current is not None else -1
        if current is None or candidate_priority >= current_priority:
            statuses[row.variant_id] = candidate
    return statuses


def _count_override_statuses(rows: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.runtime_override_status] = counts.get(row.runtime_override_status, 0) + 1
    return counts


def _expected_variant_activations(variant: LocalDelibVariant) -> list[str]:
    kwargs = variant.generation_kwargs
    expected: list[str] = []
    if bool(kwargs.get("local_delib", False)):
        expected.append("local_delib")
    if bool(kwargs.get("local_delib_adaptive_halt", False)):
        expected.append("adaptive_halt")
    if bool(kwargs.get("local_delib_use_neighbor_graph", False)):
        expected.append("neighbor_graph")
    if bool(kwargs.get("local_delib_use_flocking", False)):
        expected.append("flocking")
    if int(kwargs.get("local_delib_branch_factor", 0)) > 1:
        expected.append("branching")
    if bool(kwargs.get("local_delib_branch_consensus", False)):
        expected.append("branch_consensus")
    if bool(kwargs.get("local_delib_use_deep_hierarchy", False)) or bool(kwargs.get("local_delib_hierarchy_chunk_sizes")):
        expected.append("hierarchy")
    if int(kwargs.get("local_delib_scratch_slots", 0)) > 0:
        expected.append("scratch")
    if bool(kwargs.get("local_delib_use_thought_graph", False)):
        expected.append("thought_graph")
    if int(kwargs.get("local_delib_global_anchor_count", 0)) > 0:
        expected.append("anchors")
    return expected


def _build_activation_checks(
    variant: LocalDelibVariant,
    *,
    compute_proxy_metrics: dict[str, float],
    neighbor_graph_stats: dict[str, float],
    branch_stats: dict[str, float],
    hierarchy_stats: dict[str, float],
    scratch_stats: dict[str, float],
    thought_graph_stats: dict[str, float],
    flocking_stats: dict[str, float],
    anchor_stats: dict[str, float],
) -> dict[str, bool]:
    kwargs = variant.generation_kwargs
    checks: dict[str, bool] = {}
    if bool(kwargs.get("local_delib", False)):
        checks["local_delib"] = (
            compute_proxy_metrics.get("mean_steps_taken", 0.0) > 0.0
            or compute_proxy_metrics.get("executed_steps", 0.0) > 0.0
        )
    if bool(kwargs.get("local_delib_adaptive_halt", False)):
        checks["adaptive_halt"] = (
            compute_proxy_metrics.get("halted_token_fraction", 0.0) > 0.0
            or compute_proxy_metrics.get("fraction_halted_early", 0.0) > 0.0
        )
    if bool(kwargs.get("local_delib_use_neighbor_graph", False)):
        checks["neighbor_graph"] = neighbor_graph_stats.get("mean_neighbor_count", 0.0) > 0.0
    if bool(kwargs.get("local_delib_use_flocking", False)):
        checks["flocking"] = flocking_stats.get("fraction_flocking_tokens_active", 0.0) > 0.0
    if int(kwargs.get("local_delib_branch_factor", 0)) > 1:
        checks["branching"] = branch_stats.get("branch_factor_used", 0.0) > 1.0
    if bool(kwargs.get("local_delib_branch_consensus", False)):
        checks["branch_consensus"] = branch_stats.get("branch_consensus_used", 0.0) > 0.0
    if bool(kwargs.get("local_delib_use_deep_hierarchy", False)) or bool(kwargs.get("local_delib_hierarchy_chunk_sizes")):
        checks["hierarchy"] = (
            hierarchy_stats.get("hierarchy_depth_used", 0.0) > 0.0
            or hierarchy_stats.get("hierarchy_levels_used", 0.0) > 0.0
        )
    if int(kwargs.get("local_delib_scratch_slots", 0)) > 0:
        checks["scratch"] = scratch_stats.get("scratch_slots_used", 0.0) > 0.0
    if bool(kwargs.get("local_delib_use_thought_graph", False)):
        checks["thought_graph"] = (
            thought_graph_stats.get("thought_nodes_used", 0.0) > 0.0
            and thought_graph_stats.get("thought_graph_steps_used", 0.0) > 0.0
        )
    if int(kwargs.get("local_delib_global_anchor_count", 0)) > 0:
        checks["anchors"] = anchor_stats.get("global_anchors_used", 0.0) > 0.0
    return checks


def _build_compute_accounting(
    *,
    compute_proxy_metrics: dict[str, float],
    neighbor_graph_stats: dict[str, float],
    branch_stats: dict[str, float],
    hierarchy_stats: dict[str, float],
    scratch_stats: dict[str, float],
    thought_graph_stats: dict[str, float],
    anchor_stats: dict[str, float],
    active_mechanisms: list[str],
) -> dict[str, float]:
    base_steps = max(
        compute_proxy_metrics.get("executed_steps", 0.0),
        compute_proxy_metrics.get("mean_executed_steps_per_token", 0.0),
        compute_proxy_metrics.get("mean_steps_taken", 0.0),
        1.0,
    )
    mechanism_cost = 0.25 * float(len(active_mechanisms))
    mechanism_cost += 0.05 * neighbor_graph_stats.get("mean_neighbor_count", 0.0)
    mechanism_cost += 0.10 * branch_stats.get("branch_factor_used", 0.0)
    mechanism_cost += 0.10 * max(
        hierarchy_stats.get("hierarchy_depth_used", 0.0),
        hierarchy_stats.get("hierarchy_levels_used", 0.0),
    )
    mechanism_cost += 0.10 * scratch_stats.get("scratch_slots_used", 0.0)
    mechanism_cost += 0.15 * thought_graph_stats.get("thought_graph_steps_used", 0.0)
    mechanism_cost += 0.10 * anchor_stats.get("global_anchors_used", 0.0)
    estimated_compute_cost = round(base_steps + mechanism_cost, 4)
    return {
        "executed_steps": compute_proxy_metrics.get("executed_steps", 0.0),
        "mean_executed_steps_per_token": compute_proxy_metrics.get("mean_executed_steps_per_token", 0.0),
        "max_executed_steps_any_token": compute_proxy_metrics.get("max_executed_steps_any_token", 0.0),
        "fraction_halted_early": compute_proxy_metrics.get("fraction_halted_early", 0.0),
        "mean_steps_taken": compute_proxy_metrics.get("mean_steps_taken", 0.0),
        "halted_token_fraction": compute_proxy_metrics.get("halted_token_fraction", 0.0),
        "active_mechanism_count": float(len(active_mechanisms)),
        "estimated_compute_cost": estimated_compute_cost,
    }


def _mean_stats_subset(
    stats_rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in stats_rows:
        if not isinstance(row, dict):
            continue
        for key in keys:
            value = row.get(key)
            if not isinstance(value, (int, float)):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {
        key: (sums[key] / counts[key]) if counts[key] else 0.0
        for key in keys
        if key in counts
    }


def _estimate_compute_cost(
    *,
    compute_proxy_metrics: dict[str, float],
    neighbor_graph_stats: dict[str, float],
    branch_stats: dict[str, float],
    hierarchy_stats: dict[str, float],
    scratch_stats: dict[str, float],
    thought_graph_stats: dict[str, float],
    flocking_stats: dict[str, float],
    anchor_stats: dict[str, float],
) -> float:
    base_steps = max(compute_proxy_metrics.get("mean_steps_taken", 0.0), 1.0)
    cost = base_steps
    cost += 0.10 * neighbor_graph_stats.get("mean_neighbor_count", 0.0)
    cost += 0.10 * neighbor_graph_stats.get("semantic_topk_used", 0.0)
    cost += 0.15 * flocking_stats.get("flocking_neighbor_count", 0.0)
    cost += 0.20 * branch_stats.get("branch_factor_used", 0.0)
    cost += 0.15 * hierarchy_stats.get("hierarchy_depth_used", 0.0)
    cost += 0.10 * scratch_stats.get("scratch_slots_used", 0.0)
    cost += 0.20 * thought_graph_stats.get("thought_nodes_used", 0.0)
    cost += 0.20 * thought_graph_stats.get("thought_graph_steps_used", 0.0)
    cost += 0.15 * anchor_stats.get("global_anchors_used", 0.0)
    return round(cost, 4)


def _mean_row_section(rows: list[Any], attr_name: str) -> dict[str, dict[str, float]]:
    variant_sums: dict[str, dict[str, float]] = {}
    variant_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        section = getattr(row, attr_name)
        variant_sums.setdefault(row.variant_id, {})
        variant_counts.setdefault(row.variant_id, {})
        for key, value in section.items():
            variant_sums[row.variant_id][key] = variant_sums[row.variant_id].get(key, 0.0) + float(value)
            variant_counts[row.variant_id][key] = variant_counts[row.variant_id].get(key, 0) + 1
    return {
        variant_id: {
            key: values[key] / variant_counts[variant_id][key]
            for key in sorted(values)
        }
        for variant_id, values in variant_sums.items()
    }


def run_eval(
    cases: list[EvalCase],
    backend: PromptBackend,
    *,
    enforce_improvement: bool = True,
) -> EvalSummary:
    """Evaluate direct baseline generation against cognition-enhanced generation."""
    adapter = BackendAdapter(backend=backend)

    route_counts: dict[str, int] = {}
    rows: list[EvalRow] = []

    for case in cases:
        agent = CognitionAgent(backend=adapter)
        for episode in case.seeded_episodes:
            agent.episodic.write(episode)
        for item in case.seeded_semantic_items:
            agent.semantic.write(item)
        for skill in case.seeded_skills:
            agent.registry.register(skill)

        baseline_response = adapter.run(case.prompt)
        cognition_result = agent.run(case.prompt)
        creative_metadata = cognition_result.trace.metadata.get("creative_workspace", {})
        creative_candidates = creative_metadata.get("candidates", []) if isinstance(creative_metadata, dict) else []
        creative_strategy_ids = creative_metadata.get("explored_strategy_ids", []) if isinstance(creative_metadata, dict) else []

        baseline_score = score_keywords(baseline_response, case.expected_keywords)
        cognition_score = score_keywords(cognition_result.response, case.expected_keywords)
        if enforce_improvement and case.requires_cognition_gain and cognition_score <= baseline_score:
            raise AssertionError(
                f"Case {case.case_id} did not improve under cognition. "
                f"baseline={baseline_score:.3f}, cognition={cognition_score:.3f}"
            )
        route_counts[cognition_result.decision] = route_counts.get(cognition_result.decision, 0) + 1

        rows.append(
            EvalRow(
                case_id=case.case_id,
                baseline_response=baseline_response,
                cognition_response=cognition_result.response,
                baseline_score=baseline_score,
                cognition_score=cognition_score,
                cognition_decision=cognition_result.decision,
                creative_strategy_ids=list(creative_strategy_ids) if isinstance(creative_strategy_ids, list) else [],
                creative_selected_strategy=creative_metadata.get("selected_strategy_id") if isinstance(creative_metadata, dict) else None,
                creative_candidate_count=len(creative_candidates) if isinstance(creative_candidates, list) else 0,
                creative_handoff=creative_metadata.get("handoff") if isinstance(creative_metadata, dict) else None,
                creative_model_summary_used=bool(creative_metadata.get("model_summary_used", False)) if isinstance(creative_metadata, dict) else False,
            )
        )

    baseline_mean = sum(row.baseline_score for row in rows) / len(rows)
    cognition_mean = sum(row.cognition_score for row in rows) / len(rows)
    return EvalSummary(
        baseline_mean=baseline_mean,
        cognition_mean=cognition_mean,
        delta=cognition_mean - baseline_mean,
        route_counts=route_counts,
        rows=rows,
    )


def run_task_grounded_eval(
    backend: PromptBackend,
    *,
    task_names: list[str] | None = None,
    tasks: dict[str, Any] | None = None,
    max_problems: int | None = None,
    seed: int = 42,
    checkpoint_identity: dict[str, Any] | None = None,
) -> TaskGroundedEvalSummary:
    """Evaluate baseline vs cognition on task-native graded generation benchmarks."""
    adapter = BackendAdapter(backend=backend)
    backend_kind = _infer_backend_kind(adapter.backend)
    resolved_task_names = list(tasks) if tasks is not None else _resolve_task_grounded_task_names(task_names)
    task_objects = (
        dict(tasks)
        if tasks is not None
        else {
            task_name: _build_task_grounded_task(task_name, max_problems=max_problems)
            for task_name in resolved_task_names
        }
    )
    rows: list[TaskGroundedEvalRow] = []

    for task_offset, task_name in enumerate(resolved_task_names):
        task_object = task_objects[task_name]
        indices = _select_indices(len(task_object), max_items=max_problems, seed=seed + task_offset)
        for example_index in indices:
            conversation = task_object[example_index]
            prompt = _prompt_text_from_conversation(conversation)
            prompt_messages = _prompt_messages_from_conversation(conversation)

            baseline_response = adapter.run(prompt, messages=prompt_messages)
            baseline_metadata = copy.deepcopy(getattr(adapter.backend, "last_generation_metadata", {}) or {})
            baseline_report = _coerce_runtime_override_report(baseline_metadata, requested_overrides={})
            baseline_passed = bool(task_object.evaluate(conversation, baseline_response))

            cognition_agent = CognitionAgent(backend=BackendAdapter(backend=backend))
            cognition_result = cognition_agent.run(prompt)
            cognition_metadata = copy.deepcopy(
                getattr(cognition_agent.backend.backend, "last_generation_metadata", {}) or {}
            )
            cognition_report = _coerce_runtime_override_report(cognition_metadata, requested_overrides={})
            cognition_passed = bool(task_object.evaluate(conversation, cognition_result.response))
            benchmark_eligible = _benchmark_eligible_from_statuses(
                str(baseline_report["status"]),
                str(cognition_report["status"]),
            )

            rows.append(
                TaskGroundedEvalRow(
                    task_name=task_name,
                    example_index=example_index,
                    prompt=prompt,
                    baseline_response=baseline_response,
                    cognition_response=cognition_result.response,
                    baseline_score=_score_task_pass(baseline_passed),
                    cognition_score=_score_task_pass(cognition_passed),
                    baseline_passed=baseline_passed,
                    cognition_passed=cognition_passed,
                    cognition_decision=cognition_result.decision,
                    benchmark_eligible=benchmark_eligible,
                    baseline_runtime_override_status=str(baseline_report["status"]),
                    baseline_runtime_override_application_method=baseline_report.get("application_method"),
                    baseline_runtime_override_reason=baseline_report.get("reason"),
                    cognition_runtime_override_status=str(cognition_report["status"]),
                    cognition_runtime_override_application_method=cognition_report.get("application_method"),
                    cognition_runtime_override_reason=cognition_report.get("reason"),
                    cognition_trace_metadata=copy.deepcopy(cognition_result.trace.metadata),
                )
            )

    baseline_mean = sum(row.baseline_score for row in rows) / len(rows) if rows else 0.0
    cognition_mean = sum(row.cognition_score for row in rows) / len(rows) if rows else 0.0
    proof_rows = [row for row in rows if row.benchmark_eligible]
    proof_baseline_mean = sum(row.baseline_score for row in proof_rows) / len(proof_rows) if proof_rows else 0.0
    proof_cognition_mean = sum(row.cognition_score for row in proof_rows) / len(proof_rows) if proof_rows else 0.0

    per_task: dict[str, dict[str, float]] = {}
    for task_name in resolved_task_names:
        task_rows = [row for row in rows if row.task_name == task_name]
        proof_task_rows = [row for row in task_rows if row.benchmark_eligible]
        baseline_task_mean = sum(row.baseline_score for row in task_rows) / len(task_rows) if task_rows else 0.0
        cognition_task_mean = sum(row.cognition_score for row in task_rows) / len(task_rows) if task_rows else 0.0
        proof_baseline_task_mean = (
            sum(row.baseline_score for row in proof_task_rows) / len(proof_task_rows)
            if proof_task_rows
            else 0.0
        )
        proof_cognition_task_mean = (
            sum(row.cognition_score for row in proof_task_rows) / len(proof_task_rows)
            if proof_task_rows
            else 0.0
        )
        per_task[task_name] = {
            "count": float(len(task_rows)),
            "baseline_pass_rate": baseline_task_mean,
            "cognition_pass_rate": cognition_task_mean,
            "delta": cognition_task_mean - baseline_task_mean,
            "proof_count": float(len(proof_task_rows)),
            "proof_baseline_pass_rate": proof_baseline_task_mean,
            "proof_cognition_pass_rate": proof_cognition_task_mean,
            "proof_delta": proof_cognition_task_mean - proof_baseline_task_mean,
        }

    return TaskGroundedEvalSummary(
        backend_kind=backend_kind,
        metric_tier="task_grounded",
        checkpoint_identity=_checkpoint_identity_payload(checkpoint_identity),
        task_names=resolved_task_names,
        baseline_mean=baseline_mean,
        cognition_mean=cognition_mean,
        delta=cognition_mean - baseline_mean,
        proof_baseline_mean=proof_baseline_mean,
        proof_cognition_mean=proof_cognition_mean,
        proof_delta=proof_cognition_mean - proof_baseline_mean,
        per_task=per_task,
        rows=rows,
    )


def _run_local_delib_variant_generation(
    adapter: BackendAdapter,
    *,
    prompt: str,
    variant: LocalDelibVariant,
    fail_on_unsupported_runtime_overrides: bool,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    requested_overrides = dict(variant.generation_kwargs)
    runtime_overrides_supported = _supports_local_delib_runtime_overrides(adapter.backend)
    if not runtime_overrides_supported:
        report = _unsupported_runtime_override_report(
            requested_overrides,
            reason="backend does not advertise local-deliberation runtime override support",
            application_method="backend_no_override_support",
        )
        if fail_on_unsupported_runtime_overrides:
            raise LocalDelibRuntimeOverrideError(
                LocalDelibRuntimeOverrideReport(
                    status=str(report["status"]),
                    requested_overrides=dict(report["requested_overrides"]),
                    applied_overrides=dict(report["applied_overrides"]),
                    application_method=str(report["application_method"]),
                    reason=report.get("reason"),
                )
            )
        return "", {}, report

    try:
        response = adapter.run(prompt, **requested_overrides)
        metadata = getattr(adapter.backend, "last_generation_metadata", {}) or {}
        report = _coerce_runtime_override_report(metadata, requested_overrides=requested_overrides)
        return response, metadata, report
    except LocalDelibRuntimeOverrideError as exc:
        if fail_on_unsupported_runtime_overrides:
            raise
        metadata = dict(getattr(adapter.backend, "last_generation_metadata", {}) or {})
        report = exc.report.to_metadata()
        metadata.setdefault("local_delib_runtime_override", report)
        return "", metadata, report


def run_local_delib_ablation_eval(
    cases: list[LocalDelibEvalCase],
    backend: PromptBackend,
    *,
    variants: list[LocalDelibVariant] | None = None,
    fail_on_unsupported_runtime_overrides: bool = False,
) -> LocalDelibEvalSummary:
    """Evaluate model-side local deliberation architecture variants."""
    adapter = BackendAdapter(backend=backend)
    resolved_variants = variants or DEFAULT_LOCAL_DELIB_VARIANTS
    rows: list[LocalDelibEvalRow] = []

    for case in cases:
        for variant in resolved_variants:
            response, metadata, override_report = _run_local_delib_variant_generation(
                adapter,
                prompt=case.prompt,
                variant=variant,
                fail_on_unsupported_runtime_overrides=fail_on_unsupported_runtime_overrides,
            )
            score = score_keywords(response, case.expected_keywords)
            rows.append(
                LocalDelibEvalRow(
                    case_id=case.case_id,
                    variant_id=variant.variant_id,
                    response=response,
                    score=score,
                    runtime_override_applied=override_report["status"] == "exact",
                    runtime_override_status=str(override_report["status"]),
                    runtime_override_application_method=override_report.get("application_method"),
                    runtime_override_reason=override_report.get("reason"),
                    model_local_delib_branch=dict(metadata.get("model_local_delib.branch", {})),
                    model_local_delib_hierarchy=dict(metadata.get("model_local_delib.hierarchy", {})),
                    model_local_delib_scratchpad=dict(metadata.get("model_local_delib.scratchpad", {})),
                    model_local_delib_adaptive_halt=dict(metadata.get("model_local_delib.adaptive_halt", {})),
                    model_local_delib_graph_artifact=dict(metadata.get("model_local_delib.graph_artifact", {})),
                )
            )

    variant_totals: dict[str, float] = {variant.variant_id: 0.0 for variant in resolved_variants}
    variant_counts: dict[str, int] = {variant.variant_id: 0 for variant in resolved_variants}
    for row in rows:
        variant_totals[row.variant_id] += row.score
        variant_counts[row.variant_id] += 1
    variant_mean_scores = {
        variant_id: (variant_totals[variant_id] / variant_counts[variant_id]) if variant_counts[variant_id] else 0.0
        for variant_id in variant_totals
    }
    runtime_variant_override_statuses = _variant_override_statuses(rows)
    return LocalDelibEvalSummary(
        variant_mean_scores=variant_mean_scores,
        runtime_variant_overrides_applied=all(status == "exact" for status in runtime_variant_override_statuses.values()),
        runtime_variant_override_statuses=runtime_variant_override_statuses,
        runtime_variant_override_counts=_count_override_statuses(rows),
        rows=rows,
    )


def run_advanced_local_delib_ablation_eval(
    cases: list[LocalDelibEvalCase],
    backend: PromptBackend,
    *,
    variants: list[LocalDelibVariant] | None = None,
    fail_on_unsupported_runtime_overrides: bool = False,
) -> AdvancedLocalDelibEvalSummary:
    """Run the Prompt 10 advanced local-deliberation ablation suite."""
    adapter = BackendAdapter(backend=backend)
    resolved_variants = variants or ADVANCED_LOCAL_DELIB_VARIANTS
    rows: list[AdvancedLocalDelibEvalRow] = []

    for case in cases:
        for variant in resolved_variants:
            response, metadata, override_report = _run_local_delib_variant_generation(
                adapter,
                prompt=case.prompt,
                variant=variant,
                fail_on_unsupported_runtime_overrides=fail_on_unsupported_runtime_overrides,
            )
            quality_proxy_score = score_keywords(response, case.expected_keywords)
            stats_rows = list(metadata.get("local_deliberation_stats", []))
            compute_proxy_metrics = _mean_stats_subset(stats_rows, COMPUTE_PROXY_KEYS)
            neighbor_graph_stats = _mean_stats_subset(stats_rows, NEIGHBOR_GRAPH_KEYS)
            branch_stats = _mean_stats_subset(stats_rows, BRANCH_KEYS)
            hierarchy_stats = _mean_stats_subset(stats_rows, HIERARCHY_KEYS)
            scratch_stats = _mean_stats_subset(stats_rows, SCRATCH_KEYS)
            thought_graph_stats = _mean_stats_subset(stats_rows, THOUGHT_GRAPH_KEYS)
            flocking_stats = _mean_stats_subset(stats_rows, FLOCKING_KEYS)
            anchor_stats = _mean_stats_subset(stats_rows, ANCHOR_KEYS)
            compute_cost = _estimate_compute_cost(
                compute_proxy_metrics=compute_proxy_metrics,
                neighbor_graph_stats=neighbor_graph_stats,
                branch_stats=branch_stats,
                hierarchy_stats=hierarchy_stats,
                scratch_stats=scratch_stats,
                thought_graph_stats=thought_graph_stats,
                flocking_stats=flocking_stats,
                anchor_stats=anchor_stats,
            )
            compute_proxy_metrics = {
                **compute_proxy_metrics,
                "estimated_compute_cost": compute_cost,
            }
            rows.append(
                AdvancedLocalDelibEvalRow(
                    case_id=case.case_id,
                    variant_id=variant.variant_id,
                    response=response,
                    quality_proxy_score=quality_proxy_score,
                    quality_per_compute=quality_proxy_score / compute_cost if compute_cost > 0 else 0.0,
                    runtime_override_applied=override_report["status"] == "exact",
                    runtime_override_status=str(override_report["status"]),
                    runtime_override_application_method=override_report.get("application_method"),
                    runtime_override_reason=override_report.get("reason"),
                    compute_proxy_metrics=compute_proxy_metrics,
                    neighbor_graph_stats=neighbor_graph_stats,
                    branch_stats=branch_stats,
                    hierarchy_stats=hierarchy_stats,
                    scratch_stats=scratch_stats,
                    thought_graph_stats=thought_graph_stats,
                    flocking_stats=flocking_stats,
                    anchor_stats=anchor_stats,
                    model_local_delib_graph_artifact=dict(metadata.get("model_local_delib.graph_artifact", {})),
                )
            )

    variant_totals: dict[str, float] = {variant.variant_id: 0.0 for variant in resolved_variants}
    variant_quality_per_compute: dict[str, float] = {variant.variant_id: 0.0 for variant in resolved_variants}
    variant_counts: dict[str, int] = {variant.variant_id: 0 for variant in resolved_variants}
    quality_proxy_scores: dict[str, dict[str, float]] = {variant.variant_id: {} for variant in resolved_variants}
    for row in rows:
        variant_totals[row.variant_id] += row.quality_proxy_score
        variant_quality_per_compute[row.variant_id] += row.quality_per_compute
        variant_counts[row.variant_id] += 1
        quality_proxy_scores[row.variant_id][row.case_id] = row.quality_proxy_score

    variant_mean_scores = {
        variant_id: (variant_totals[variant_id] / variant_counts[variant_id]) if variant_counts[variant_id] else 0.0
        for variant_id in variant_totals
    }
    quality_per_compute = {
        variant_id: (variant_quality_per_compute[variant_id] / variant_counts[variant_id]) if variant_counts[variant_id] else 0.0
        for variant_id in variant_quality_per_compute
    }
    compute_proxy_metrics = _mean_row_section(rows, "compute_proxy_metrics")
    mean_steps_taken = {
        variant_id: metrics.get("mean_steps_taken", 0.0)
        for variant_id, metrics in compute_proxy_metrics.items()
    }
    runtime_variant_override_statuses = _variant_override_statuses(rows)

    return AdvancedLocalDelibEvalSummary(
        quality_proxy_scores=quality_proxy_scores,
        variant_mean_scores=variant_mean_scores,
        quality_per_compute=quality_per_compute,
        compute_proxy_metrics=compute_proxy_metrics,
        mean_steps_taken=mean_steps_taken,
        neighbor_graph_stats=_mean_row_section(rows, "neighbor_graph_stats"),
        branch_stats=_mean_row_section(rows, "branch_stats"),
        hierarchy_stats=_mean_row_section(rows, "hierarchy_stats"),
        scratch_stats=_mean_row_section(rows, "scratch_stats"),
        thought_graph_stats=_mean_row_section(rows, "thought_graph_stats"),
        flocking_stats=_mean_row_section(rows, "flocking_stats"),
        anchor_stats=_mean_row_section(rows, "anchor_stats"),
        runtime_variant_overrides_applied=all(status == "exact" for status in runtime_variant_override_statuses.values()),
        runtime_variant_override_statuses=runtime_variant_override_statuses,
        runtime_variant_override_counts=_count_override_statuses(rows),
        rows=rows,
    )


def run_research_local_delib_eval(
    cases: list[ResearchLocalDelibEvalCase],
    backend: PromptBackend,
    *,
    variants: list[LocalDelibVariant] | None = None,
    baseline_variant_id: str = "local_delib_off",
    fail_on_unsupported_runtime_overrides: bool = False,
) -> ResearchLocalDelibEvalSummary:
    """Run the Prompt 4 structured research suite."""
    adapter = BackendAdapter(backend=backend)
    resolved_variants = variants or RESEARCH_LOCAL_DELIB_VARIANTS
    backend_kind = _infer_backend_kind(adapter.backend)
    metric_tier = _infer_metric_tier(backend_kind)
    rows: list[ResearchLocalDelibEvalRow] = []

    for case in cases:
        for variant in resolved_variants:
            response, metadata, override_report = _run_local_delib_variant_generation(
                adapter,
                prompt=case.prompt,
                variant=variant,
                fail_on_unsupported_runtime_overrides=fail_on_unsupported_runtime_overrides,
            )
            metric_score, task_metrics, response_format_ok = _score_research_case(case, response)
            stats_rows = list(metadata.get("local_deliberation_stats", []))
            compute_proxy_metrics = _mean_stats_subset(stats_rows, RESEARCH_COMPUTE_KEYS)
            neighbor_graph_stats = _mean_stats_subset(stats_rows, NEIGHBOR_GRAPH_KEYS)
            branch_stats = _mean_stats_subset(stats_rows, BRANCH_KEYS)
            hierarchy_stats = _mean_stats_subset(stats_rows, HIERARCHY_KEYS)
            scratch_stats = _mean_stats_subset(stats_rows, SCRATCH_KEYS)
            thought_graph_stats = _mean_stats_subset(stats_rows, THOUGHT_GRAPH_KEYS)
            flocking_stats = _mean_stats_subset(stats_rows, FLOCKING_KEYS)
            anchor_stats = _mean_stats_subset(stats_rows, ANCHOR_KEYS)
            activation_checks = _build_activation_checks(
                variant,
                compute_proxy_metrics=compute_proxy_metrics,
                neighbor_graph_stats=neighbor_graph_stats,
                branch_stats=branch_stats,
                hierarchy_stats=hierarchy_stats,
                scratch_stats=scratch_stats,
                thought_graph_stats=thought_graph_stats,
                flocking_stats=flocking_stats,
                anchor_stats=anchor_stats,
            )
            expected_activations = _expected_variant_activations(variant)
            override_exact = override_report["status"] == "exact"
            activation_ok = (all(activation_checks.values()) if activation_checks else True) and override_exact
            active_mechanisms = sorted(name for name, active in activation_checks.items() if active) if override_exact else []
            compute_accounting = _build_compute_accounting(
                compute_proxy_metrics=compute_proxy_metrics,
                neighbor_graph_stats=neighbor_graph_stats,
                branch_stats=branch_stats,
                hierarchy_stats=hierarchy_stats,
                scratch_stats=scratch_stats,
                thought_graph_stats=thought_graph_stats,
                anchor_stats=anchor_stats,
                active_mechanisms=active_mechanisms,
            )
            estimated_compute_cost = compute_accounting.get("estimated_compute_cost", 0.0)
            rows.append(
                ResearchLocalDelibEvalRow(
                    case_id=case.case_id,
                    task_family=case.task_family,
                    variant_id=variant.variant_id,
                    response=response,
                    metric_score=metric_score,
                    quality_per_compute=metric_score / estimated_compute_cost if estimated_compute_cost > 0 else 0.0,
                    pass_threshold=case.pass_threshold,
                    passed=metric_score >= case.pass_threshold,
                    response_format_ok=response_format_ok,
                    runtime_override_applied=override_exact,
                    runtime_override_status=str(override_report["status"]),
                    runtime_override_application_method=override_report.get("application_method"),
                    runtime_override_reason=override_report.get("reason"),
                    backend_kind=backend_kind,
                    metric_tier=metric_tier,
                    targeted_mechanisms=list(case.targeted_mechanisms),
                    expected_activations=expected_activations,
                    activation_checks=activation_checks,
                    activation_ok=activation_ok,
                    metrics_interpretable=response_format_ok and activation_ok and override_exact,
                    task_metrics=task_metrics,
                    active_mechanisms=active_mechanisms,
                    compute_accounting=compute_accounting,
                    neighbor_graph_stats=neighbor_graph_stats,
                    branch_stats=branch_stats,
                    hierarchy_stats=hierarchy_stats,
                    scratch_stats=scratch_stats,
                    thought_graph_stats=thought_graph_stats,
                    flocking_stats=flocking_stats,
                    anchor_stats=anchor_stats,
                    model_local_delib_graph_artifact=dict(metadata.get("model_local_delib.graph_artifact", {})),
                )
            )

    variant_totals: dict[str, float] = {variant.variant_id: 0.0 for variant in resolved_variants}
    variant_quality_per_compute: dict[str, float] = {variant.variant_id: 0.0 for variant in resolved_variants}
    variant_passes: dict[str, int] = {variant.variant_id: 0 for variant in resolved_variants}
    variant_counts: dict[str, int] = {variant.variant_id: 0 for variant in resolved_variants}
    case_scores: dict[str, dict[str, float]] = {case.case_id: {} for case in cases}
    task_family_scores: dict[str, dict[str, float]] = {}

    for row in rows:
        variant_totals[row.variant_id] += row.metric_score
        variant_quality_per_compute[row.variant_id] += row.quality_per_compute
        variant_passes[row.variant_id] += int(row.passed)
        variant_counts[row.variant_id] += 1
        case_scores.setdefault(row.case_id, {})[row.variant_id] = row.metric_score
        family_scores = task_family_scores.setdefault(row.task_family, {})
        family_scores[row.variant_id] = family_scores.get(row.variant_id, 0.0) + row.metric_score

    variant_mean_scores = {
        variant_id: (variant_totals[variant_id] / variant_counts[variant_id]) if variant_counts[variant_id] else 0.0
        for variant_id in variant_totals
    }
    variant_pass_rates = {
        variant_id: (variant_passes[variant_id] / variant_counts[variant_id]) if variant_counts[variant_id] else 0.0
        for variant_id in variant_passes
    }
    quality_per_compute = {
        variant_id: (variant_quality_per_compute[variant_id] / variant_counts[variant_id]) if variant_counts[variant_id] else 0.0
        for variant_id in variant_quality_per_compute
    }
    task_family_scores = {
        family: {
            variant_id: (total / sum(1 for row in rows if row.task_family == family and row.variant_id == variant_id))
            if any(row.task_family == family and row.variant_id == variant_id for row in rows)
            else 0.0
            for variant_id, total in family_totals.items()
        }
        for family, family_totals in task_family_scores.items()
    }
    baseline_score = variant_mean_scores.get(baseline_variant_id, 0.0)
    delta_vs_baseline = {
        variant_id: score - baseline_score
        for variant_id, score in variant_mean_scores.items()
    }
    case_deltas_vs_baseline = {
        case_id: {
            variant_id: score - scores.get(baseline_variant_id, 0.0)
            for variant_id, score in scores.items()
        }
        for case_id, scores in case_scores.items()
    }
    compute_accounting = _mean_row_section(rows, "compute_accounting")
    activation_coverage: dict[str, dict[str, Any]] = {}
    for variant in resolved_variants:
        variant_rows = [row for row in rows if row.variant_id == variant.variant_id]
        count = len(variant_rows)
        activation_coverage[variant.variant_id] = {
            "expected_mechanisms": _expected_variant_activations(variant),
            "activation_ok_rate": (
                sum(1 for row in variant_rows if row.activation_ok) / count if count else 0.0
            ),
            "metrics_interpretable_rate": (
                sum(1 for row in variant_rows if row.metrics_interpretable) / count if count else 0.0
            ),
            "response_format_rate": (
                sum(1 for row in variant_rows if row.response_format_ok) / count if count else 0.0
            ),
        }
    runtime_variant_override_statuses = _variant_override_statuses(rows)

    return ResearchLocalDelibEvalSummary(
        backend_kind=backend_kind,
        metric_tier=metric_tier,
        baseline_variant_id=baseline_variant_id,
        variant_mean_scores=variant_mean_scores,
        variant_pass_rates=variant_pass_rates,
        delta_vs_baseline=delta_vs_baseline,
        case_scores=case_scores,
        case_deltas_vs_baseline=case_deltas_vs_baseline,
        task_family_scores=task_family_scores,
        quality_per_compute=quality_per_compute,
        compute_accounting=compute_accounting,
        activation_coverage=activation_coverage,
        runtime_variant_overrides_applied=all(status == "exact" for status in runtime_variant_override_statuses.values()),
        runtime_variant_override_statuses=runtime_variant_override_statuses,
        runtime_variant_override_counts=_count_override_statuses(rows),
        rows=rows,
    )


def run_natural_local_delib_eval(
    cases: list[NaturalLocalDelibEvalCase],
    backend: PromptBackend,
    *,
    variants: list[LocalDelibVariant] | None = None,
    baseline_variant_id: str = "local_delib_off",
    fail_on_unsupported_runtime_overrides: bool = False,
    checkpoint_identity: dict[str, Any] | None = None,
) -> NaturalLocalDelibEvalSummary:
    """Run natural-language, task-specific local-deliberation benchmarks."""
    adapter = BackendAdapter(backend=backend)
    resolved_variants = variants or NATURAL_LOCAL_DELIB_VARIANTS
    backend_kind = _infer_backend_kind(adapter.backend)
    rows: list[NaturalLocalDelibEvalRow] = []

    for case in cases:
        for variant in resolved_variants:
            response, metadata, override_report = _run_local_delib_variant_generation(
                adapter,
                prompt=case.prompt,
                variant=variant,
                fail_on_unsupported_runtime_overrides=fail_on_unsupported_runtime_overrides,
            )
            metric_score, task_metrics, grader_extractable = _score_natural_case(case, response)
            stats_rows = list(metadata.get("local_deliberation_stats", []))
            compute_proxy_metrics = _mean_stats_subset(stats_rows, RESEARCH_COMPUTE_KEYS)
            neighbor_graph_stats = _mean_stats_subset(stats_rows, NEIGHBOR_GRAPH_KEYS)
            branch_stats = _mean_stats_subset(stats_rows, BRANCH_KEYS)
            hierarchy_stats = _mean_stats_subset(stats_rows, HIERARCHY_KEYS)
            scratch_stats = _mean_stats_subset(stats_rows, SCRATCH_KEYS)
            thought_graph_stats = _mean_stats_subset(stats_rows, THOUGHT_GRAPH_KEYS)
            flocking_stats = _mean_stats_subset(stats_rows, FLOCKING_KEYS)
            anchor_stats = _mean_stats_subset(stats_rows, ANCHOR_KEYS)
            activation_checks = _build_activation_checks(
                variant,
                compute_proxy_metrics=compute_proxy_metrics,
                neighbor_graph_stats=neighbor_graph_stats,
                branch_stats=branch_stats,
                hierarchy_stats=hierarchy_stats,
                scratch_stats=scratch_stats,
                thought_graph_stats=thought_graph_stats,
                flocking_stats=flocking_stats,
                anchor_stats=anchor_stats,
            )
            expected_activations = _expected_variant_activations(variant)
            override_exact = override_report["status"] == "exact"
            activation_ok = (all(activation_checks.values()) if activation_checks else True) and override_exact
            active_mechanisms = sorted(name for name, active in activation_checks.items() if active) if override_exact else []
            compute_accounting = _build_compute_accounting(
                compute_proxy_metrics=compute_proxy_metrics,
                neighbor_graph_stats=neighbor_graph_stats,
                branch_stats=branch_stats,
                hierarchy_stats=hierarchy_stats,
                scratch_stats=scratch_stats,
                thought_graph_stats=thought_graph_stats,
                anchor_stats=anchor_stats,
                active_mechanisms=active_mechanisms,
            )
            estimated_compute_cost = compute_accounting.get("estimated_compute_cost", 0.0)
            proof_eligible = grader_extractable and activation_ok and override_exact
            passed = metric_score >= case.pass_threshold
            rows.append(
                NaturalLocalDelibEvalRow(
                    case_id=case.case_id,
                    task_family=case.task_family,
                    variant_id=variant.variant_id,
                    response=response,
                    metric_score=metric_score,
                    quality_per_compute=metric_score / estimated_compute_cost if estimated_compute_cost > 0 else 0.0,
                    pass_threshold=case.pass_threshold,
                    passed=passed,
                    grader_extractable=grader_extractable,
                    proof_eligible=proof_eligible,
                    proof_passed=passed and proof_eligible,
                    runtime_override_applied=override_exact,
                    runtime_override_status=str(override_report["status"]),
                    runtime_override_application_method=override_report.get("application_method"),
                    runtime_override_reason=override_report.get("reason"),
                    backend_kind=backend_kind,
                    metric_tier="natural_task_grounded",
                    targeted_mechanisms=list(case.targeted_mechanisms),
                    expected_activations=expected_activations,
                    activation_checks=activation_checks,
                    activation_ok=activation_ok,
                    metrics_interpretable=grader_extractable and activation_ok and override_exact,
                    task_metrics=task_metrics,
                    active_mechanisms=active_mechanisms,
                    compute_accounting=compute_accounting,
                    neighbor_graph_stats=neighbor_graph_stats,
                    branch_stats=branch_stats,
                    hierarchy_stats=hierarchy_stats,
                    scratch_stats=scratch_stats,
                    thought_graph_stats=thought_graph_stats,
                    flocking_stats=flocking_stats,
                    anchor_stats=anchor_stats,
                    model_local_delib_graph_artifact=dict(metadata.get("model_local_delib.graph_artifact", {})),
                )
            )

    variant_totals: dict[str, float] = {variant.variant_id: 0.0 for variant in resolved_variants}
    variant_passes: dict[str, int] = {variant.variant_id: 0 for variant in resolved_variants}
    proof_variant_totals: dict[str, float] = {variant.variant_id: 0.0 for variant in resolved_variants}
    proof_variant_passes: dict[str, int] = {variant.variant_id: 0 for variant in resolved_variants}
    variant_quality_per_compute: dict[str, float] = {variant.variant_id: 0.0 for variant in resolved_variants}
    variant_counts: dict[str, int] = {variant.variant_id: 0 for variant in resolved_variants}
    case_scores: dict[str, dict[str, float]] = {case.case_id: {} for case in cases}
    task_family_scores: dict[str, dict[str, float]] = {}

    for row in rows:
        variant_totals[row.variant_id] += row.metric_score
        variant_passes[row.variant_id] += int(row.passed)
        proof_variant_totals[row.variant_id] += row.metric_score if row.proof_eligible else 0.0
        proof_variant_passes[row.variant_id] += int(row.proof_passed)
        variant_quality_per_compute[row.variant_id] += row.quality_per_compute
        variant_counts[row.variant_id] += 1
        case_scores.setdefault(row.case_id, {})[row.variant_id] = row.metric_score
        family_scores = task_family_scores.setdefault(row.task_family, {})
        family_scores[row.variant_id] = family_scores.get(row.variant_id, 0.0) + row.metric_score

    variant_mean_scores = {
        variant_id: (variant_totals[variant_id] / variant_counts[variant_id]) if variant_counts[variant_id] else 0.0
        for variant_id in variant_totals
    }
    variant_pass_rates = {
        variant_id: (variant_passes[variant_id] / variant_counts[variant_id]) if variant_counts[variant_id] else 0.0
        for variant_id in variant_passes
    }
    proof_variant_mean_scores = {
        variant_id: (proof_variant_totals[variant_id] / variant_counts[variant_id]) if variant_counts[variant_id] else 0.0
        for variant_id in proof_variant_totals
    }
    proof_pass_rates = {
        variant_id: (proof_variant_passes[variant_id] / variant_counts[variant_id]) if variant_counts[variant_id] else 0.0
        for variant_id in proof_variant_passes
    }
    quality_per_compute = {
        variant_id: (variant_quality_per_compute[variant_id] / variant_counts[variant_id]) if variant_counts[variant_id] else 0.0
        for variant_id in variant_quality_per_compute
    }
    task_family_scores = {
        family: {
            variant_id: (total / sum(1 for row in rows if row.task_family == family and row.variant_id == variant_id))
            if any(row.task_family == family and row.variant_id == variant_id for row in rows)
            else 0.0
            for variant_id, total in family_totals.items()
        }
        for family, family_totals in task_family_scores.items()
    }
    baseline_score = variant_mean_scores.get(baseline_variant_id, 0.0)
    proof_baseline_score = proof_variant_mean_scores.get(baseline_variant_id, 0.0)
    delta_vs_baseline = {
        variant_id: score - baseline_score
        for variant_id, score in variant_mean_scores.items()
    }
    proof_delta_vs_baseline = {
        variant_id: score - proof_baseline_score
        for variant_id, score in proof_variant_mean_scores.items()
    }
    compute_accounting = _mean_row_section(rows, "compute_accounting")
    activation_coverage: dict[str, dict[str, Any]] = {}
    for variant in resolved_variants:
        variant_rows = [row for row in rows if row.variant_id == variant.variant_id]
        count = len(variant_rows)
        activation_coverage[variant.variant_id] = {
            "expected_mechanisms": _expected_variant_activations(variant),
            "activation_ok_rate": (
                sum(1 for row in variant_rows if row.activation_ok) / count if count else 0.0
            ),
            "metrics_interpretable_rate": (
                sum(1 for row in variant_rows if row.metrics_interpretable) / count if count else 0.0
            ),
            "grader_extractable_rate": (
                sum(1 for row in variant_rows if row.grader_extractable) / count if count else 0.0
            ),
            "proof_eligible_rate": (
                sum(1 for row in variant_rows if row.proof_eligible) / count if count else 0.0
            ),
        }
    runtime_variant_override_statuses = _variant_override_statuses(rows)

    return NaturalLocalDelibEvalSummary(
        backend_kind=backend_kind,
        metric_tier="natural_task_grounded",
        baseline_variant_id=baseline_variant_id,
        checkpoint_identity=_checkpoint_identity_payload(checkpoint_identity),
        variant_mean_scores=variant_mean_scores,
        variant_pass_rates=variant_pass_rates,
        proof_variant_mean_scores=proof_variant_mean_scores,
        proof_pass_rates=proof_pass_rates,
        delta_vs_baseline=delta_vs_baseline,
        proof_delta_vs_baseline=proof_delta_vs_baseline,
        case_scores=case_scores,
        task_family_scores=task_family_scores,
        quality_per_compute=quality_per_compute,
        compute_accounting=compute_accounting,
        activation_coverage=activation_coverage,
        runtime_variant_overrides_applied=all(status == "exact" for status in runtime_variant_override_statuses.values()),
        runtime_variant_override_statuses=runtime_variant_override_statuses,
        runtime_variant_override_counts=_count_override_statuses(rows),
        rows=rows,
    )


def write_eval_artifact(summary: EvalSummary, output_path: str) -> Path:
    """Persist machine-readable eval artifact to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline_mean": summary.baseline_mean,
        "cognition_mean": summary.cognition_mean,
        "delta": summary.delta,
        "route_counts": summary.route_counts,
        "rows": [asdict(row) for row in summary.rows],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_task_grounded_eval_artifact(summary: TaskGroundedEvalSummary, output_path: str) -> Path:
    """Persist task-grounded benchmark results to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend_kind": summary.backend_kind,
        "metric_tier": summary.metric_tier,
        "checkpoint_identity": summary.checkpoint_identity,
        "task_names": summary.task_names,
        "baseline_mean": summary.baseline_mean,
        "cognition_mean": summary.cognition_mean,
        "delta": summary.delta,
        "proof_baseline_mean": summary.proof_baseline_mean,
        "proof_cognition_mean": summary.proof_cognition_mean,
        "proof_delta": summary.proof_delta,
        "per_task": summary.per_task,
        "rows": [asdict(row) for row in summary.rows],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_local_delib_eval_artifact(summary: LocalDelibEvalSummary, output_path: str) -> Path:
    """Persist machine-readable local deliberation ablation artifact to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant_mean_scores": summary.variant_mean_scores,
        "runtime_variant_overrides_applied": summary.runtime_variant_overrides_applied,
        "runtime_variant_override_statuses": summary.runtime_variant_override_statuses,
        "runtime_variant_override_counts": summary.runtime_variant_override_counts,
        "rows": [asdict(row) for row in summary.rows],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_advanced_local_delib_eval_artifact(summary: AdvancedLocalDelibEvalSummary, output_path: str) -> Path:
    """Persist the Prompt 10 advanced ablation artifact to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "quality_proxy_scores": summary.quality_proxy_scores,
        "variant_mean_scores": summary.variant_mean_scores,
        "quality_per_compute": summary.quality_per_compute,
        "compute_proxy_metrics": summary.compute_proxy_metrics,
        "mean_steps_taken": summary.mean_steps_taken,
        "neighbor_graph_stats": summary.neighbor_graph_stats,
        "branch_stats": summary.branch_stats,
        "hierarchy_stats": summary.hierarchy_stats,
        "scratch_stats": summary.scratch_stats,
        "thought_graph_stats": summary.thought_graph_stats,
        "flocking_stats": summary.flocking_stats,
        "anchor_stats": summary.anchor_stats,
        "runtime_variant_overrides_applied": summary.runtime_variant_overrides_applied,
        "runtime_variant_override_statuses": summary.runtime_variant_override_statuses,
        "runtime_variant_override_counts": summary.runtime_variant_override_counts,
        "rows": [asdict(row) for row in summary.rows],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_research_local_delib_eval_artifact(summary: ResearchLocalDelibEvalSummary, output_path: str) -> Path:
    """Persist the Prompt 4 research eval artifact to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend_kind": summary.backend_kind,
        "metric_tier": summary.metric_tier,
        "baseline_variant_id": summary.baseline_variant_id,
        "variant_mean_scores": summary.variant_mean_scores,
        "variant_pass_rates": summary.variant_pass_rates,
        "delta_vs_baseline": summary.delta_vs_baseline,
        "case_scores": summary.case_scores,
        "case_deltas_vs_baseline": summary.case_deltas_vs_baseline,
        "task_family_scores": summary.task_family_scores,
        "quality_per_compute": summary.quality_per_compute,
        "compute_accounting": summary.compute_accounting,
        "activation_coverage": summary.activation_coverage,
        "runtime_variant_overrides_applied": summary.runtime_variant_overrides_applied,
        "runtime_variant_override_statuses": summary.runtime_variant_override_statuses,
        "runtime_variant_override_counts": summary.runtime_variant_override_counts,
        "rows": [asdict(row) for row in summary.rows],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_natural_local_delib_eval_artifact(summary: NaturalLocalDelibEvalSummary, output_path: str) -> Path:
    """Persist the natural-language local-deliberation eval artifact to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend_kind": summary.backend_kind,
        "metric_tier": summary.metric_tier,
        "baseline_variant_id": summary.baseline_variant_id,
        "checkpoint_identity": summary.checkpoint_identity,
        "variant_mean_scores": summary.variant_mean_scores,
        "variant_pass_rates": summary.variant_pass_rates,
        "proof_variant_mean_scores": summary.proof_variant_mean_scores,
        "proof_pass_rates": summary.proof_pass_rates,
        "delta_vs_baseline": summary.delta_vs_baseline,
        "proof_delta_vs_baseline": summary.proof_delta_vs_baseline,
        "case_scores": summary.case_scores,
        "task_family_scores": summary.task_family_scores,
        "quality_per_compute": summary.quality_per_compute,
        "compute_accounting": summary.compute_accounting,
        "activation_coverage": summary.activation_coverage,
        "runtime_variant_overrides_applied": summary.runtime_variant_overrides_applied,
        "runtime_variant_override_statuses": summary.runtime_variant_override_statuses,
        "runtime_variant_override_counts": summary.runtime_variant_override_counts,
        "rows": [asdict(row) for row in summary.rows],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_engine_smoke_manifest(manifest: EngineSmokeManifest, output_path: str) -> Path:
    """Persist the optional engine smoke manifest to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return output


def _extract_section(prompt: str, title: str) -> list[str]:
    lines = prompt.splitlines()
    in_section = False
    captured: list[str] = []
    for line in lines:
        if line == title:
            in_section = True
            continue
        if in_section and not line.strip():
            break
        if in_section:
            captured.append(line)
    return captured


def _extract_prefixed_line(lines: list[str], prefix: str) -> str | None:
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None
