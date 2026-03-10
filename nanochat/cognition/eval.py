"""Lightweight evaluation harness for baseline vs cognition comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Protocol

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
