from __future__ import annotations

from nanochat.cognition.consolidation import Consolidator
from nanochat.cognition.memory import SemanticMemory
from nanochat.cognition.schemas import Episode
from nanochat.cognition.skills import SkillRegistry


def test_consolidator_emits_skill_for_repeated_wins_and_stores_provenance() -> None:
    semantic = SemanticMemory()
    registry = SkillRegistry()
    consolidator = Consolidator(semantic_memory=semantic, skill_registry=registry, min_repetitions=2)

    episodes = [
        Episode(
            episode_id="e1",
            prompt="Help me summarize this report",
            response="first extract bullets, then condense",
            tags=["summarization"],
            metadata={"outcome": "success", "trigger": "summarization", "strategy": "extract bullets then condense"},
        ),
        Episode(
            episode_id="e2",
            prompt="Summarize meeting notes",
            response="first extract bullets, then condense",
            tags=["summarization"],
            metadata={"success": True, "trigger": "summarization", "strategy": "extract bullets then condense"},
        ),
    ]

    skill = consolidator.consolidate(episodes)

    assert skill is not None
    assert skill.trigger == "summarization"
    assert skill.metadata["provenance_episode_ids"] == ["e1", "e2"]
    assert registry.best_for("Can you do summarization for me?") is not None

    retrieved = semantic.retrieve("summarization bullets condense", limit=1)
    assert retrieved
    assert retrieved[0].item.metadata["skill_id"] == skill.skill_id
    assert retrieved[0].item.metadata["trigger"] == "summarization"


def test_consolidator_requires_repetition_before_creating_skill() -> None:
    semantic = SemanticMemory()
    registry = SkillRegistry()
    consolidator = Consolidator(semantic_memory=semantic, skill_registry=registry, min_repetitions=2)

    single = [
        Episode(
            episode_id="e1",
            prompt="Summarize once",
            response="extract bullets then condense",
            tags=["summarization"],
            metadata={"outcome": "success", "trigger": "summarization", "strategy": "extract bullets then condense"},
        )
    ]

    assert consolidator.consolidate(single) is None
    assert registry.best_for("summarization") is None
    assert semantic.retrieve("summarization") == []
