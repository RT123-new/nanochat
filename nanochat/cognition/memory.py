"""Minimal in-memory episodic and semantic memory stores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .schemas import Episode, MemoryItem


@dataclass(slots=True)
class RankedMemory:
    item: MemoryItem
    relevance: float
    recency: float
    combined_score: float


class EpisodicMemory:
    def __init__(self) -> None:
        self._episodes: list[Episode] = []

    def write(self, episode: Episode) -> None:
        self._episodes.append(episode)

    def recent(self, limit: int = 5) -> list[Episode]:
        return list(reversed(self._episodes[-limit:]))

    def retrieve(self, query: str, limit: int = 5) -> list[Episode]:
        terms = _terms(query)
        scored: list[tuple[int, Episode]] = []
        for idx, ep in enumerate(self._episodes):
            haystack = f"{ep.prompt} {ep.response} {' '.join(ep.tags)}".lower()
            score = sum(1 for t in terms if t in haystack)
            if score > 0:
                scored.append((score * 1000 + idx, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]


class SemanticMemory:
    def __init__(self) -> None:
        self._items: list[MemoryItem] = []

    def write(self, item: MemoryItem) -> None:
        if item.kind != "semantic":
            raise ValueError("SemanticMemory accepts only semantic MemoryItem entries")
        self._items.append(item)

    def retrieve(self, query: str, limit: int = 5) -> list[RankedMemory]:
        terms = _terms(query)
        now = datetime.utcnow().timestamp()
        ranked: list[RankedMemory] = []
        for item in self._items:
            relevance = _relevance(item.content, terms)
            if relevance <= 0:
                continue
            created_ts = datetime.fromisoformat(item.created_at).timestamp()
            age_secs = max(now - created_ts, 0.0)
            recency = 1.0 / (1.0 + age_secs / 3600.0)
            combined = relevance * 0.7 + recency * 0.3
            ranked.append(RankedMemory(item=item, relevance=relevance, recency=recency, combined_score=combined))
        ranked.sort(key=lambda x: x.combined_score, reverse=True)
        return ranked[:limit]


def _terms(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _relevance(content: str, terms: list[str]) -> float:
    haystack = content.lower()
    if not terms:
        return 0.0
    hits = sum(1 for t in terms if t in haystack)
    return hits / len(terms)
