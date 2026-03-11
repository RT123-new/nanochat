"""End-to-end cognition controller that wires module interactions."""

from __future__ import annotations

from dataclasses import dataclass

from .backend import BackendAdapter
from .consolidation import Consolidator
from .creative import CreativeWorkspace
from .memory import EpisodicMemory, RankedEpisode, RankedMemory, SemanticMemory
from .normalize import term_set
from .router import ExplicitRouter
from .sandbox import LightweightSandbox
from .schemas import Episode, SkillArtifact, Trace
from .skills import SkillRegistry
from .traces import TraceRecorder
from .verifier import VerifierWorkspace


@dataclass(slots=True)
class CognitionResult:
    response: str
    trace: Trace
    decision: str
    reused_skill_id: str | None = None
    consolidated_skill: SkillArtifact | None = None


class CognitionAgent:
    EPISODIC_SEARCH_LIMIT = 6
    EPISODIC_SUPPORT_LIMIT = 2
    EPISODIC_SUPPORT_MIN_SCORE = 0.35
    EXPLICIT_MEMORY_MIN_SCORE = 0.15
    SEMANTIC_SUPPORT_LIMIT = 2

    def __init__(self, backend: BackendAdapter, min_skill_repetitions: int = 2) -> None:
        self.backend = backend
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.router = ExplicitRouter()
        self.registry = SkillRegistry()
        self.creative = CreativeWorkspace(backend=backend)
        self.verifier = VerifierWorkspace()
        self.sandbox = LightweightSandbox()
        self.traces = TraceRecorder()
        self.consolidator = Consolidator(
            semantic_memory=self.semantic,
            skill_registry=self.registry,
            min_repetitions=min_skill_repetitions,
        )

    def run(self, query: str) -> CognitionResult:
        decision = self.router.decide(query)
        steps = [f"route:{decision.action}"]
        semantic_hits = self.semantic.retrieve(query, limit=self.SEMANTIC_SUPPORT_LIMIT)
        reused_skill = self.registry.best_for(query)
        response = ""
        episodic_hits = self._select_episodic_support(
            query=query,
            decision=decision.action,
            semantic_hits=semantic_hits,
            reused_skill=reused_skill,
        )
        selected_episodes = [match.episode for match in episodic_hits]

        if decision.action == "retrieve_memory":
            prompt = _compose_prompt(
                query=query,
                episodes=selected_episodes,
                semantic_hits=semantic_hits,
                reused_skill=reused_skill,
            )
            response = self.backend.run(prompt)
        elif decision.action in {"creative_explore", "verify", "sandbox"}:
            prompt = _compose_prompt(
                query=query,
                episodes=selected_episodes,
                semantic_hits=semantic_hits,
                reused_skill=reused_skill,
            )
            candidates = self.creative.generate_candidates(prompt, limit=3)
            steps.append(f"candidates:{len(candidates)}")
            if decision.action == "sandbox":
                outcomes = self.sandbox.explore(query, branches=candidates)
                steps.append(f"sandbox_branches:{len(outcomes)}")
                response = outcomes[0].branch if outcomes else ""
            else:
                best = self.verifier.choose(query=query, candidates=candidates)
                steps.append(f"verifier_score:{best.score:.2f}")
                response = best.candidate
        elif decision.action == "consolidate":
            skill = self.consolidator.consolidate(self.episodic.recent(limit=50))
            if skill is None:
                response = "No repeated successful pattern was found yet."
                steps.append("consolidated:false")
            else:
                response = f"Consolidated skill: {skill.name} ({skill.skill_id})"
                steps.append("consolidated:true")
        else:
            prompt = _compose_prompt(
                query=query,
                episodes=selected_episodes,
                semantic_hits=semantic_hits,
                reused_skill=reused_skill,
            )
            response = self.backend.run(prompt)

        if episodic_hits:
            steps.append(f"episodic_hits:{len(episodic_hits)}")
        if semantic_hits:
            steps.append(f"semantic_hits:{len(semantic_hits)}")
        if reused_skill is not None:
            steps.append(f"skill_reused:{reused_skill.skill_id}")

        episode = Episode(
            episode_id=f"ep-{len(self.episodic.recent(limit=10_000)) + 1}",
            prompt=query,
            response=response,
            tags=[decision.action],
            metadata={"success": bool(response.strip()), "decision": decision.action},
        )
        self.episodic.write(episode)

        consolidated_skill = self.consolidator.consolidate(self.episodic.recent(limit=50))
        if consolidated_skill is not None:
            steps.append(f"auto_consolidated:{consolidated_skill.skill_id}")

        metadata = {
            "confidence": decision.confidence,
            "retrieved_episode_ids": [episode.episode_id for episode in selected_episodes],
            "retrieved_semantic_ids": [match.item.item_id for match in semantic_hits],
            "reused_skill_ids": [reused_skill.skill_id] if reused_skill else [],
        }
        backend_metadata = getattr(self.backend.backend, "last_generation_metadata", None)
        if backend_metadata:
            if backend_metadata.get("local_deliberation_stats") is not None:
                metadata["model_local_delib"] = backend_metadata["local_deliberation_stats"]
            for key, value in backend_metadata.items():
                if key.startswith("model_local_delib."):
                    metadata[key] = value

        trace = self.traces.build(
            query=query,
            decision=decision.action,
            rationale=decision.rationale,
            steps=steps,
            metadata=metadata,
        )
        return CognitionResult(
            response=response,
            trace=trace,
            decision=decision.action,
            reused_skill_id=reused_skill.skill_id if reused_skill else None,
            consolidated_skill=consolidated_skill,
        )

    def _select_episodic_support(
        self,
        *,
        query: str,
        decision: str,
        semantic_hits: list[RankedMemory],
        reused_skill: SkillArtifact | None,
    ) -> list[RankedEpisode]:
        if not query.strip():
            return []

        ranked = self.episodic.search(query=query, limit=self.EPISODIC_SEARCH_LIMIT)
        if decision == "retrieve_memory":
            return [match for match in ranked if match.combined_score >= self.EXPLICIT_MEMORY_MIN_SCORE][: self.EPISODIC_SUPPORT_LIMIT]

        support_terms = set()
        if reused_skill is not None:
            support_terms.update(_skill_terms(reused_skill))
        for match in semantic_hits:
            support_terms.update(_semantic_terms(match))

        selected: list[RankedEpisode] = []
        covered_terms = set(support_terms)
        for match in ranked:
            if match.combined_score < self.EPISODIC_SUPPORT_MIN_SCORE:
                continue
            episode_terms = _episode_terms(match)
            if episode_terms and covered_terms:
                shared_ratio = len(episode_terms & covered_terms) / len(episode_terms)
                if shared_ratio >= 0.75:
                    continue
            selected.append(match)
            covered_terms.update(episode_terms)
            if len(selected) >= self.EPISODIC_SUPPORT_LIMIT:
                break
        return selected


def _compose_prompt(
    *,
    query: str,
    episodes: list[Episode] | None = None,
    semantic_hits: list[RankedMemory] | None = None,
    reused_skill: SkillArtifact | None = None,
) -> str:
    sections: list[str] = []

    if reused_skill is not None:
        procedure = "\n".join(f"- {step}" for step in reused_skill.procedure)
        sections.append(
            "\n".join(
                [
                    "Relevant skill:",
                    f"- skill_id: {reused_skill.skill_id}",
                    f"- name: {reused_skill.name}",
                    f"- trigger: {reused_skill.trigger}",
                    "- procedure:",
                    procedure or "- none",
                ]
            )
        )

    if semantic_hits:
        lines = [f"- {match.item.item_id}: {match.item.content}" for match in semantic_hits]
        sections.append("Relevant semantic memory:\n" + "\n".join(lines))

    if episodes:
        lines = [f"- {episode.episode_id}: {episode.prompt} -> {episode.response}" for episode in episodes]
        sections.append("Relevant episodic memory:\n" + "\n".join(lines))

    if not sections:
        return query
    return f"{query}\n\n" + "\n\n".join(sections)


def _episode_terms(match: RankedEpisode) -> set[str]:
    episode = match.episode
    return term_set(episode.prompt, episode.response, episode.tags, episode.metadata)


def _semantic_terms(match: RankedMemory) -> set[str]:
    return term_set(match.item.content, match.item.metadata)


def _skill_terms(skill: SkillArtifact) -> set[str]:
    return term_set(skill.name, skill.trigger, skill.procedure, skill.metadata)
