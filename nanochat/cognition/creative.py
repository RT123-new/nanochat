"""Creative workspace for structured multi-strategy candidate generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .backend import BackendAdapter, summarize_local_delib_for_creative_policy
from .normalize import unique_terms


@dataclass(frozen=True, slots=True)
class CreativeStrategy:
    strategy_id: str
    label: str
    guidance: str
    rationale: str


@dataclass(slots=True)
class CreativePlan:
    route: str
    strategy_order: list[str]
    explored_strategy_ids: list[str] = field(default_factory=list)
    support_profile: dict[str, Any] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)
    model_summary_used: bool = False

    def as_trace_payload(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "strategy_order": list(self.strategy_order),
            "explored_strategy_ids": list(self.explored_strategy_ids),
            "support_profile": dict(self.support_profile),
            "signals": dict(self.signals),
            "model_summary_used": self.model_summary_used,
        }


@dataclass(slots=True)
class CreativeCandidate:
    candidate_id: str
    strategy_id: str
    strategy_label: str
    response: str
    prompt: str
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_trace_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "strategy_id": self.strategy_id,
            "strategy_label": self.strategy_label,
            "response": self.response,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class CreativeRun:
    plan: CreativePlan
    candidates: list[CreativeCandidate]
    model_summary: dict[str, Any] = field(default_factory=dict)


class CreativeWorkspace:
    """Generate inspectable candidates using explicit creative strategies."""

    STRATEGIES: dict[str, CreativeStrategy] = {
        "conservative_answer": CreativeStrategy(
            strategy_id="conservative_answer",
            label="Conservative answer",
            guidance="Stay close to the query and preserve retrieved support without unnecessary novelty.",
            rationale="Ground the first draft so later stages have a stable anchor.",
        ),
        "divergent_ideas": CreativeStrategy(
            strategy_id="divergent_ideas",
            label="Divergent ideas",
            guidance="Push for distinct angles, alternatives, or framings before collapsing to one answer.",
            rationale="Encourage exploration instead of early convergence.",
        ),
        "memory_grounded": CreativeStrategy(
            strategy_id="memory_grounded",
            label="Memory-grounded answer",
            guidance="Lean on retrieved episodes, semantic memory, and reusable skills when they are available.",
            rationale="Use long-range support when the query looks memory-heavy.",
        ),
        "branch_resolution": CreativeStrategy(
            strategy_id="branch_resolution",
            label="Branch disagreement resolution",
            guidance="Resolve competing directions, keep the strongest branch, and explain what was discarded.",
            rationale="Use model-core disagreement and consensus summaries to collapse branches deliberately.",
        ),
        "recombination": CreativeStrategy(
            strategy_id="recombination",
            label="Recombination and synthesis",
            guidance="Recombine the strongest pieces into one compact answer and favor coherence over breadth.",
            rationale="Translate scratch, hierarchy, thought, and anchor summaries into one inspectable synthesis pass.",
        ),
    }

    def __init__(self, backend: BackendAdapter) -> None:
        self.backend = backend

    def plan(
        self,
        *,
        query: str,
        route: str,
        support_profile: dict[str, Any] | None = None,
        model_summary: dict[str, Any] | None = None,
        limit: int = 3,
        explored_strategy_ids: list[str] | None = None,
    ) -> CreativePlan:
        support_profile = dict(support_profile or {})
        model_summary = dict(model_summary or {})
        terms = set(unique_terms(query))
        brainstorming = route == "creative_explore" or "creative_explore" in terms
        memory_heavy = bool(support_profile.get("memory_heavy", False))
        active_sections = [str(section) for section in model_summary.get("active_sections", []) if isinstance(section, str)]
        branch_pressure = (
            model_summary.get("branch_consensus_used", 0.0) > 0.0
            or model_summary.get("branch_disagreement", 0.0) >= 0.2
            or "branch" in active_sections
        )
        synthesis_pressure = any(
            model_summary.get(key, 0.0) > 0.0
            for key in ("scratch_slots_used", "thought_nodes_used", "hierarchy_depth_used", "global_anchors_used")
        )

        strategy_order = ["conservative_answer"]
        if brainstorming or route == "sandbox":
            strategy_order.append("divergent_ideas")
        if memory_heavy:
            strategy_order.append("memory_grounded")
        if branch_pressure:
            strategy_order.append("branch_resolution")
        if synthesis_pressure or route == "verify":
            strategy_order.append("recombination")

        strategy_order = _dedupe(strategy_order)[: max(limit, 1)]
        signals = {
            "brainstorming": brainstorming,
            "memory_heavy": memory_heavy,
            "branch_pressure": branch_pressure,
            "synthesis_pressure": synthesis_pressure,
            "active_sections": active_sections,
        }
        support_snapshot = {
            "memory_heavy": memory_heavy,
            "episodic_count": int(support_profile.get("episodic_count", 0)),
            "semantic_count": int(support_profile.get("semantic_count", 0)),
            "skill_count": int(support_profile.get("skill_count", 0)),
            "support_terms": list(support_profile.get("support_terms", [])),
        }
        return CreativePlan(
            route=route,
            strategy_order=strategy_order,
            explored_strategy_ids=list(explored_strategy_ids or []),
            support_profile=support_snapshot,
            signals=signals,
            model_summary_used=bool(model_summary),
        )

    def generate_candidates(
        self,
        *,
        query: str,
        base_prompt: str,
        route: str,
        support_profile: dict[str, Any] | None = None,
        initial_model_summary: dict[str, Any] | None = None,
        limit: int = 3,
    ) -> CreativeRun:
        support_profile = dict(support_profile or {})
        active_model_summary = dict(initial_model_summary or self._current_model_summary())
        used_strategy_ids: list[str] = []
        candidates: list[CreativeCandidate] = []

        while len(candidates) < max(limit, 1):
            plan = self.plan(
                query=query,
                route=route,
                support_profile=support_profile,
                model_summary=active_model_summary,
                limit=limit,
                explored_strategy_ids=used_strategy_ids,
            )
            next_strategy_id = next(
                (strategy_id for strategy_id in plan.strategy_order if strategy_id not in used_strategy_ids),
                None,
            )
            if next_strategy_id is None:
                break

            strategy = self.STRATEGIES[next_strategy_id]
            used_strategy_ids.append(strategy.strategy_id)
            candidate_prompt = self._compose_prompt(
                base_prompt=base_prompt,
                strategy=strategy,
                support_profile=support_profile,
                model_summary=active_model_summary,
            )
            response = self.backend.run(candidate_prompt).strip()
            latest_model_summary = self._current_model_summary()
            if latest_model_summary:
                active_model_summary = latest_model_summary
            candidates.append(
                CreativeCandidate(
                    candidate_id=f"candidate-{len(candidates) + 1}",
                    strategy_id=strategy.strategy_id,
                    strategy_label=strategy.label,
                    response=response,
                    prompt=candidate_prompt,
                    rationale=strategy.rationale,
                    metadata={
                        "model_summary_used": bool(active_model_summary),
                        "model_focus": self._model_focus(active_model_summary),
                        "support_terms": list(support_profile.get("support_terms", [])),
                    },
                )
            )

        final_plan = self.plan(
            query=query,
            route=route,
            support_profile=support_profile,
            model_summary=active_model_summary,
            limit=max(limit, len(used_strategy_ids), 1),
            explored_strategy_ids=used_strategy_ids,
        )
        return CreativeRun(
            plan=final_plan,
            candidates=candidates,
            model_summary=active_model_summary,
        )

    def _compose_prompt(
        self,
        *,
        base_prompt: str,
        strategy: CreativeStrategy,
        support_profile: dict[str, Any],
        model_summary: dict[str, Any],
    ) -> str:
        sections = [
            base_prompt,
            "",
            "Creative strategy:",
            f"- id: {strategy.strategy_id}",
            f"- label: {strategy.label}",
            f"- guidance: {strategy.guidance}",
            f"- rationale: {strategy.rationale}",
        ]
        support_lines = self._support_lines(support_profile)
        if support_lines:
            sections.extend(["", "Creative support profile:", *support_lines])
        model_lines = self._model_lines(model_summary)
        if model_lines:
            sections.extend(["", "Model deliberation guidance:", *model_lines])
        sections.extend(["", "Return one concise candidate response."])
        return "\n".join(sections)

    def _current_model_summary(self) -> dict[str, Any]:
        metadata = getattr(self.backend.backend, "last_generation_metadata", None)
        return summarize_local_delib_for_creative_policy(metadata)

    def _model_focus(self, model_summary: dict[str, Any]) -> list[str]:
        focus: list[str] = []
        if model_summary.get("branch_disagreement", 0.0) > 0.0:
            focus.append("branch_disagreement")
        if model_summary.get("scratch_summary_dim", 0.0) > 0.0:
            focus.append("scratch_summary")
        if model_summary.get("thought_nodes_used", 0.0) > 0.0:
            focus.append("thought_graph")
        if model_summary.get("hierarchy_depth_used", 0.0) > 0.0:
            focus.append("hierarchy")
        if model_summary.get("global_anchors_used", 0.0) > 0.0:
            focus.append("anchors")
        return focus

    def _support_lines(self, support_profile: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        if support_profile.get("memory_heavy", False):
            lines.append("- memory_heavy: true")
        for key in ("episodic_count", "semantic_count", "skill_count"):
            value = int(support_profile.get(key, 0))
            if value > 0:
                lines.append(f"- {key}: {value}")
        support_terms = list(support_profile.get("support_terms", []))
        if support_terms:
            lines.append("- support_terms: " + ", ".join(support_terms))
        return lines

    def _model_lines(self, model_summary: dict[str, Any]) -> list[str]:
        if not model_summary:
            return []
        lines: list[str] = []
        active_sections = model_summary.get("active_sections", [])
        if active_sections:
            lines.append("- active_sections: " + ", ".join(str(section) for section in active_sections))
        for key in (
            "branch_disagreement",
            "branch_consensus_used",
            "branch_verifier_score",
            "scratch_summary_dim",
            "thought_nodes_used",
            "hierarchy_depth_used",
            "global_anchors_used",
            "mean_steps_taken",
        ):
            value = model_summary.get(key)
            if isinstance(value, (int, float)) and value > 0.0:
                lines.append(f"- {key}: {float(value):.2f}")
        return lines


def _dedupe(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
