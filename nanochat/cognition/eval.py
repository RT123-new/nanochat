"""Lightweight evaluation harness for baseline vs cognition comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any, Protocol

from .agent import CognitionAgent
from .backend import BackendAdapter
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
    model_local_delib_branch: dict[str, float]
    model_local_delib_hierarchy: dict[str, float]
    model_local_delib_scratchpad: dict[str, float]
    model_local_delib_adaptive_halt: dict[str, float]


@dataclass(slots=True)
class LocalDelibEvalSummary:
    """Ablation summary for local deliberation architecture variants."""

    variant_mean_scores: dict[str, float]
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
    compute_proxy_metrics: dict[str, float]
    neighbor_graph_stats: dict[str, float]
    branch_stats: dict[str, float]
    hierarchy_stats: dict[str, float]
    scratch_stats: dict[str, float]
    thought_graph_stats: dict[str, float]
    flocking_stats: dict[str, float]
    anchor_stats: dict[str, float]


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
    rows: list[AdvancedLocalDelibEvalRow]


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


class ContextAwareEvalBackend:
    """Deterministic backend that only improves when support context is injected."""

    def generate(self, prompt: str, **kwargs: object) -> str:
        skill_lines = _extract_section(prompt, "Relevant skill:")
        semantic_lines = _extract_section(prompt, "Relevant semantic memory:")
        episodic_lines = _extract_section(prompt, "Relevant episodic memory:")

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
            "model_local_delib.stats": stats,
            "model_local_delib.branch": {
                "agreement": stats[0]["agreement"],
                "branch_factor_used": stats[0]["branch_factor_used"],
                "mean_branch_score": stats[0]["mean_branch_score"],
            },
            "model_local_delib.hierarchy": {
                "hierarchy_levels_used": stats[0]["hierarchy_levels_used"],
                "mean_hierarchy_feedback_norm": stats[0]["mean_hierarchy_feedback_norm"],
            },
            "model_local_delib.scratchpad": {
                "scratch_slots_used": stats[0]["scratch_slots_used"],
                "mean_scratch_read_weight": stats[0]["mean_scratch_read_weight"],
                "mean_scratch_write_weight": stats[0]["mean_scratch_write_weight"],
                "mean_scratch_refine_norm": stats[0]["mean_scratch_refine_norm"],
            },
            "model_local_delib.adaptive_halt": {
                "halted_token_fraction": stats[0]["halted_token_fraction"],
                "mean_steps_taken": stats[0]["mean_steps_taken"],
            },
        }
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


def score_keywords(response: str, expected_keywords: list[str]) -> float:
    """Return simple keyword recall score in [0, 1]."""
    if not expected_keywords:
        return 0.0
    response_text = response.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in response_text)
    return hits / len(expected_keywords)


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


def _supports_local_delib_runtime_overrides(backend: PromptBackend) -> bool:
    return bool(getattr(backend, "supports_local_delib_runtime_overrides", False))


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


def _mean_row_section(rows: list[AdvancedLocalDelibEvalRow], attr_name: str) -> dict[str, dict[str, float]]:
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


def run_local_delib_ablation_eval(
    cases: list[LocalDelibEvalCase],
    backend: PromptBackend,
    *,
    variants: list[LocalDelibVariant] | None = None,
) -> LocalDelibEvalSummary:
    """Evaluate model-side local deliberation architecture variants."""
    adapter = BackendAdapter(backend=backend)
    resolved_variants = variants or DEFAULT_LOCAL_DELIB_VARIANTS
    rows: list[LocalDelibEvalRow] = []
    runtime_overrides_supported = _supports_local_delib_runtime_overrides(adapter.backend)

    for case in cases:
        for variant in resolved_variants:
            generation_kwargs = variant.generation_kwargs if runtime_overrides_supported else {}
            response = adapter.run(case.prompt, **generation_kwargs)
            score = score_keywords(response, case.expected_keywords)
            metadata = getattr(adapter.backend, "last_generation_metadata", {}) or {}
            rows.append(
                LocalDelibEvalRow(
                    case_id=case.case_id,
                    variant_id=variant.variant_id,
                    response=response,
                    score=score,
                    model_local_delib_branch=dict(metadata.get("model_local_delib.branch", {})),
                    model_local_delib_hierarchy=dict(metadata.get("model_local_delib.hierarchy", {})),
                    model_local_delib_scratchpad=dict(metadata.get("model_local_delib.scratchpad", {})),
                    model_local_delib_adaptive_halt=dict(metadata.get("model_local_delib.adaptive_halt", {})),
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

    return LocalDelibEvalSummary(variant_mean_scores=variant_mean_scores, rows=rows)


def run_advanced_local_delib_ablation_eval(
    cases: list[LocalDelibEvalCase],
    backend: PromptBackend,
    *,
    variants: list[LocalDelibVariant] | None = None,
) -> AdvancedLocalDelibEvalSummary:
    """Run the Prompt 10 advanced local-deliberation ablation suite."""
    adapter = BackendAdapter(backend=backend)
    resolved_variants = variants or ADVANCED_LOCAL_DELIB_VARIANTS
    runtime_overrides_supported = _supports_local_delib_runtime_overrides(adapter.backend)
    rows: list[AdvancedLocalDelibEvalRow] = []

    for case in cases:
        for variant in resolved_variants:
            generation_kwargs = variant.generation_kwargs if runtime_overrides_supported else {}
            response = adapter.run(case.prompt, **generation_kwargs)
            quality_proxy_score = score_keywords(response, case.expected_keywords)
            metadata = getattr(adapter.backend, "last_generation_metadata", {}) or {}
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
                    runtime_override_applied=runtime_overrides_supported,
                    compute_proxy_metrics=compute_proxy_metrics,
                    neighbor_graph_stats=neighbor_graph_stats,
                    branch_stats=branch_stats,
                    hierarchy_stats=hierarchy_stats,
                    scratch_stats=scratch_stats,
                    thought_graph_stats=thought_graph_stats,
                    flocking_stats=flocking_stats,
                    anchor_stats=anchor_stats,
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
        runtime_variant_overrides_applied=runtime_overrides_supported,
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


def write_local_delib_eval_artifact(summary: LocalDelibEvalSummary, output_path: str) -> Path:
    """Persist machine-readable local deliberation ablation artifact to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant_mean_scores": summary.variant_mean_scores,
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
        "rows": [asdict(row) for row in summary.rows],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
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
