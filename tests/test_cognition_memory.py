from datetime import datetime, timedelta

import pytest

from nanochat.cognition.memory import EpisodicMemory, SemanticMemory
from nanochat.cognition.schemas import Episode, MemoryItem


def test_episodic_memory_write_and_retrieve_ranked() -> None:
    mem = EpisodicMemory()
    mem.write(Episode(episode_id="e1", prompt="How do I train?", response="Use batches", tags=["train"]))
    mem.write(Episode(episode_id="e2", prompt="How do I eval?", response="Use eval loop", tags=["eval"]))
    mem.write(Episode(episode_id="e3", prompt="How do I train faster?", response="Tune batch size", tags=["train"]))

    found = mem.retrieve("train")
    assert [ep.episode_id for ep in found] == ["e3", "e1"]


def test_episodic_memory_recent() -> None:
    mem = EpisodicMemory()
    for i in range(4):
        mem.write(Episode(episode_id=f"e{i}", prompt="p", response="r"))
    assert [e.episode_id for e in mem.recent(limit=2)] == ["e3", "e2"]


def test_semantic_memory_write_requires_semantic_kind() -> None:
    mem = SemanticMemory()
    with pytest.raises(ValueError):
        mem.write(MemoryItem(item_id="x", content="bad", kind="episodic"))


def test_semantic_memory_retrieve_prefers_relevance_and_recency() -> None:
    mem = SemanticMemory()
    old_time = (datetime.utcnow() - timedelta(hours=10)).isoformat()
    new_time = datetime.utcnow().isoformat()
    mem.write(MemoryItem(item_id="m1", content="router memory retrieval strategy", kind="semantic", created_at=old_time))
    mem.write(MemoryItem(item_id="m2", content="router memory retrieval strategy", kind="semantic", created_at=new_time))
    mem.write(MemoryItem(item_id="m3", content="unrelated topic", kind="semantic", created_at=new_time))

    ranked = mem.retrieve("router retrieval")
    assert [r.item.item_id for r in ranked][:2] == ["m2", "m1"]
    assert all(r.combined_score > 0 for r in ranked)
