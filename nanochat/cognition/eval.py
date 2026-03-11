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

    def __init__(self) -> None:
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        local_delib = bool(kwargs.get("local_delib", False))
        adaptive_halt = bool(kwargs.get("local_delib_adaptive_halt", False))
        branch_factor = int(kwargs.get("local_delib_branch_factor", 1))
        hierarchy = bool(kwargs.get("local_delib_hierarchy_chunk_sizes"))
        scratch_slots = int(kwargs.get("local_delib_scratch_slots", 0))

        stats = [{
            "layer_idx": 0,
            "agreement": 0.7 if local_delib else 0.0,
            "branch_factor_used": float(branch_factor if local_delib else 1),
            "mean_branch_score": 0.6 if branch_factor > 1 else 0.0,
            "hierarchy_levels_used": 2.0 if hierarchy else 0.0,
            "mean_hierarchy_feedback_norm": 0.4 if hierarchy else 0.0,
            "scratch_slots_used": float(scratch_slots if local_delib else 0),
            "mean_scratch_read_weight": 0.35 if scratch_slots > 0 else 0.0,
            "mean_scratch_write_weight": 0.28 if scratch_slots > 0 else 0.0,
            "halted_token_fraction": 0.5 if adaptive_halt else 0.0,
            "mean_steps_taken": 1.5 if adaptive_halt else 2.0,
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
            },
            "model_local_delib.adaptive_halt": {
                "halted_token_fraction": stats[0]["halted_token_fraction"],
                "mean_steps_taken": stats[0]["mean_steps_taken"],
            },
        }
        modes: list[str] = []
        if local_delib:
            modes.append("basic")
        if adaptive_halt:
            modes.append("adaptive_halt")
        if branch_factor > 1:
            modes.append("branch")
        if hierarchy:
            modes.append("hierarchy")
        if scratch_slots > 0:
            modes.append("scratchpad")
        suffix = ",".join(modes) if modes else "off"
        return f"{prompt} :: local_delib={suffix}"


def score_keywords(response: str, expected_keywords: list[str]) -> float:
    """Return simple keyword recall score in [0, 1]."""
    if not expected_keywords:
        return 0.0
    response_text = response.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in response_text)
    return hits / len(expected_keywords)


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

    for case in cases:
        for variant in resolved_variants:
            response = adapter.run(case.prompt, **variant.generation_kwargs)
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
