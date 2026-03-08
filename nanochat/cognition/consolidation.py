"""Consolidation flow for turning repeated wins into reusable skills."""

from __future__ import annotations

from collections import defaultdict

from .memory import SemanticMemory
from .schemas import Episode, MemoryItem, SkillArtifact
from .skills import SkillRegistry


class Consolidator:
    """Detect repeated successful patterns and emit skill artifacts."""

    def __init__(self, semantic_memory: SemanticMemory, skill_registry: SkillRegistry, min_repetitions: int = 2) -> None:
        self.semantic_memory = semantic_memory
        self.skill_registry = skill_registry
        self.min_repetitions = min_repetitions

    def consolidate(self, episodes: list[Episode]) -> SkillArtifact | None:
        buckets: dict[tuple[str, str], list[Episode]] = defaultdict(list)

        for episode in episodes:
            if not _is_success(episode):
                continue
            trigger = _trigger_for(episode)
            strategy = _strategy_for(episode)
            if not trigger or not strategy:
                continue
            buckets[(trigger, strategy)].append(episode)

        best_key: tuple[str, str] | None = None
        best_count = 0
        for key, grouped in buckets.items():
            if len(grouped) >= self.min_repetitions and len(grouped) > best_count:
                best_key = key
                best_count = len(grouped)

        if best_key is None:
            return None

        trigger, strategy = best_key
        supporting = buckets[best_key]
        skill = SkillArtifact(
            skill_id=f"skill-{_slug(trigger)}-{best_count}",
            name=f"Reusable pattern for {trigger}",
            trigger=trigger,
            procedure=[strategy],
            success_signals=[f"repeated_successes:{best_count}"],
            metadata={
                "provenance_episode_ids": [ep.episode_id for ep in supporting],
                "pattern_count": best_count,
                "source": "consolidator",
            },
        )

        self.skill_registry.register(skill)
        self.semantic_memory.write(
            MemoryItem(
                item_id=f"semantic-{skill.skill_id}",
                kind="semantic",
                content=f"{skill.name}. Trigger: {skill.trigger}. Procedure: {'; '.join(skill.procedure)}",
                source="consolidation",
                metadata={
                    "skill_id": skill.skill_id,
                    "trigger": skill.trigger,
                    "provenance_episode_ids": skill.metadata["provenance_episode_ids"],
                },
            )
        )
        return skill


def _is_success(episode: Episode) -> bool:
    if episode.metadata.get("success") is True:
        return True
    return episode.metadata.get("outcome") == "success"


def _trigger_for(episode: Episode) -> str:
    if "trigger" in episode.metadata:
        return str(episode.metadata["trigger"]).strip().lower()
    if episode.tags:
        return episode.tags[0].strip().lower()
    return ""


def _strategy_for(episode: Episode) -> str:
    if "strategy" in episode.metadata:
        return str(episode.metadata["strategy"]).strip().lower()
    return episode.response.strip().lower()


def _slug(text: str) -> str:
    return "-".join(part for part in text.lower().split() if part)
