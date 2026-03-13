from datetime import datetime, timedelta, timezone

import pytest

import nanochat.cognition.memory as memory_module
from nanochat.cognition.memory import EpisodicMemory, SemanticMemory
from nanochat.cognition.schemas import Episode, MemoryItem


FIXED_NOW = datetime(2026, 3, 13, 12, 0, tzinfo=timezone.utc)


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        if tz is None:
            return FIXED_NOW.replace(tzinfo=None)
        return FIXED_NOW.astimezone(tz)

    @classmethod
    def fromisoformat(cls, value: str):  # type: ignore[override]
        return datetime.fromisoformat(value)


def _freeze_memory_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_module, "datetime", FixedDateTime)


def _iso(hours_ago: float) -> str:
    return (FIXED_NOW - timedelta(hours=hours_ago)).isoformat()


def test_episodic_memory_write_search_and_retrieve_are_ranked(monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze_memory_now(monkeypatch)
    mem = EpisodicMemory()
    mem.write(
        Episode(
            episode_id="exact-new",
            prompt="Summarize the project update",
            response="Use terse bullet summaries with citations",
            tags=["summarization"],
            created_at=_iso(1),
        )
    )
    mem.write(
        Episode(
            episode_id="exact-old",
            prompt="Summarize the project update",
            response="Keep the answer grounded in citations",
            tags=["summarization"],
            created_at=_iso(10),
        )
    )
    mem.write(
        Episode(
            episode_id="partial-recent",
            prompt="Summarize the update",
            response="Use terse bullets",
            tags=["summarization"],
            created_at=_iso(0.5),
        )
    )

    ranked = mem.search("Please summarize the project update", limit=3)

    assert [match.episode.episode_id for match in ranked] == ["exact-new", "exact-old", "partial-recent"]
    assert ranked[0].matched_terms == ["summarization", "project", "update"]
    assert ranked[0].relevance == pytest.approx(1.0)
    assert ranked[0].recency == pytest.approx(0.5)
    assert ranked[0].combined_score == pytest.approx(0.9)
    assert ranked[1].relevance == pytest.approx(1.0)
    assert ranked[1].recency == pytest.approx(1 / 11)
    assert ranked[1].combined_score > ranked[2].combined_score
    assert ranked[2].relevance == pytest.approx(2 / 3)
    assert [episode.episode_id for episode in mem.retrieve("Please summarize the project update", limit=2)] == [
        "exact-new",
        "exact-old",
    ]


def test_episodic_memory_search_handles_paraphrase_and_empty_or_non_overlapping_queries() -> None:
    mem = EpisodicMemory()
    mem.write(
        Episode(
            episode_id="style",
            prompt="Summarize the project update",
            response="Use terse bullet summaries with citations",
            tags=["summarization"],
        )
    )

    found = mem.search("Please summarize this draft for me")

    assert [match.episode.episode_id for match in found] == ["style"]
    assert found[0].combined_score > 0.0
    assert found[0].matched_terms == ["summarization"]
    assert mem.search("") == []
    assert mem.search("unrelated database migration") == []


def test_episodic_memory_recent_is_newest_first() -> None:
    mem = EpisodicMemory()
    for i in range(4):
        mem.write(Episode(episode_id=f"e{i}", prompt="p", response="r"))

    assert [episode.episode_id for episode in mem.recent(limit=3)] == ["e3", "e2", "e1"]


def test_semantic_memory_write_requires_semantic_kind() -> None:
    mem = SemanticMemory()

    with pytest.raises(ValueError):
        mem.write(MemoryItem(item_id="x", content="bad", kind="episodic"))


def test_semantic_memory_write_replaces_existing_item_id() -> None:
    mem = SemanticMemory()
    mem.write(MemoryItem(item_id="style", content="Old terse style", kind="semantic"))
    mem.write(MemoryItem(item_id="style", content="Updated terse bullet style", kind="semantic"))

    ranked = mem.retrieve("terse bullet style", limit=3)

    assert [match.item.item_id for match in ranked] == ["style"]
    assert ranked[0].item.content == "Updated terse bullet style"


def test_semantic_memory_retrieve_prefers_relevance_over_stale_but_weak_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze_memory_now(monkeypatch)
    mem = SemanticMemory()
    mem.write(
        MemoryItem(
            item_id="strong-old",
            content="router memory retrieval strategy",
            kind="semantic",
            created_at=_iso(12),
        )
    )
    mem.write(
        MemoryItem(
            item_id="weak-new",
            content="router notes",
            kind="semantic",
            created_at=_iso(0.25),
        )
    )

    ranked = mem.retrieve("router retrieval strategy", limit=2)

    assert [match.item.item_id for match in ranked] == ["strong-old", "weak-new"]
    assert ranked[0].relevance == pytest.approx(1.0)
    assert ranked[0].combined_score > ranked[1].combined_score
    assert ranked[1].relevance == pytest.approx(1 / 3)


def test_semantic_memory_retrieve_returns_ranked_metadata_with_stable_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze_memory_now(monkeypatch)
    mem = SemanticMemory()
    created_at = _iso(2)
    mem.write(MemoryItem(item_id="tie-a", content="summarization checklist bullets", kind="semantic", created_at=created_at))
    mem.write(MemoryItem(item_id="tie-b", content="summarization checklist bullets", kind="semantic", created_at=created_at))

    ranked = mem.retrieve("summarization checklist", limit=2)

    assert [match.item.item_id for match in ranked] == ["tie-a", "tie-b"]
    assert ranked[0].relevance == pytest.approx(ranked[1].relevance)
    assert ranked[0].recency == pytest.approx(ranked[1].recency)
    assert ranked[0].combined_score == pytest.approx(ranked[1].combined_score)


def test_semantic_memory_retrieve_normalizes_aliases() -> None:
    mem = SemanticMemory()
    mem.write(
        MemoryItem(
            item_id="style",
            content="House style: brief neutral numbered answers.",
            kind="semantic",
        )
    )

    ranked = mem.retrieve("Answer in our house style for this reply.")

    assert [match.item.item_id for match in ranked] == ["style"]
