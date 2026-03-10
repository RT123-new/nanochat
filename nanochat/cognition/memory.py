"""Minimal in-memory episodic and semantic memory stores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .normalize import overlap_score, term_set, unique_terms
from .schemas import Episode, MemoryItem


@dataclass(slots=True)
class RankedMemory:
    item: MemoryItem
    relevance: float
    recency: float
    combined_score: float


@dataclass(slots=True)
class RankedEpisode:
    episode: Episode
    relevance: float
    recency: float
    combined_score: float
    matched_terms: list[str]


class EpisodicMemory:
    def __init__(self) -> None:
        self._episodes: list[Episode] = []

    def write(self, episode: Episode) -> None:
        self._episodes.append(episode)

    def recent(self, limit: int = 5) -> list[Episode]:
        return list(reversed(self._episodes[-limit:]))

    def search(self, query: str, limit: int = 5, min_score: float = 0.0) -> list[RankedEpisode]:
        query_terms = unique_terms(query)
        if not query_terms:
            return []

        now = datetime.now(timezone.utc).timestamp()
        ranked: list[RankedEpisode] = []
        for episode in self._episodes:
            episode_terms = _episode_terms(episode)
            relevance = overlap_score(query_terms, episode_terms)
            if relevance <= 0.0:
                continue
            recency = _recency_score(episode.created_at, now)
            combined = relevance * 0.8 + recency * 0.2
            if combined < min_score:
                continue
            matched_terms = [term for term in query_terms if term in episode_terms]
            ranked.append(
                RankedEpisode(
                    episode=episode,
                    relevance=relevance,
                    recency=recency,
                    combined_score=combined,
                    matched_terms=matched_terms,
                )
            )

        ranked.sort(
            key=lambda match: (match.combined_score, match.relevance, match.recency, match.episode.created_at),
            reverse=True,
        )
        return ranked[:limit]

    def retrieve(self, query: str, limit: int = 5) -> list[Episode]:
        return [match.episode for match in self.search(query=query, limit=limit)]


class SemanticMemory:
    def __init__(self) -> None:
        self._items: list[MemoryItem] = []

    def write(self, item: MemoryItem) -> None:
        if item.kind != "semantic":
            raise ValueError("SemanticMemory accepts only semantic MemoryItem entries")
        for idx, existing in enumerate(self._items):
            if existing.item_id == item.item_id:
                self._items[idx] = item
                return
        self._items.append(item)

    def retrieve(self, query: str, limit: int = 5) -> list[RankedMemory]:
        query_terms = unique_terms(query)
        if not query_terms:
            return []

        now = datetime.now(timezone.utc).timestamp()
        ranked: list[RankedMemory] = []
        for item in self._items:
            relevance = overlap_score(query_terms, _memory_item_terms(item))
            if relevance <= 0:
                continue
            recency = _recency_score(item.created_at, now)
            combined = relevance * 0.7 + recency * 0.3
            ranked.append(RankedMemory(item=item, relevance=relevance, recency=recency, combined_score=combined))
        ranked.sort(key=lambda x: x.combined_score, reverse=True)
        return ranked[:limit]


def _episode_terms(episode: Episode) -> set[str]:
    return term_set(episode.prompt, episode.response, episode.tags, episode.metadata)


def _memory_item_terms(item: MemoryItem) -> set[str]:
    return term_set(item.content, item.metadata)


def _recency_score(created_at: str, now_ts: float) -> float:
    created = datetime.fromisoformat(created_at)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    created_ts = created.timestamp()
    age_secs = max(now_ts - created_ts, 0.0)
    return 1.0 / (1.0 + age_secs / 3600.0)
