"""Skill registry and retrieval helpers for cognition skill reuse."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import SkillArtifact


@dataclass(slots=True)
class SkillMatch:
    skill: SkillArtifact
    score: float


class SkillRegistry:
    """In-memory registry for reusable skill artifacts."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillArtifact] = {}

    def register(self, skill: SkillArtifact) -> None:
        self._skills[skill.skill_id] = skill

    def all(self) -> list[SkillArtifact]:
        return list(self._skills.values())

    def discover(self, query: str, limit: int = 3) -> list[SkillMatch]:
        terms = _terms(query)
        matches: list[SkillMatch] = []
        for skill in self._skills.values():
            score = _score_skill(skill, terms)
            if score <= 0:
                continue
            matches.append(SkillMatch(skill=skill, score=score))
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:limit]

    def best_for(self, query: str) -> SkillArtifact | None:
        matches = self.discover(query=query, limit=1)
        if not matches:
            return None
        return matches[0].skill


def _score_skill(skill: SkillArtifact, terms: list[str]) -> float:
    if not terms:
        return 0.0
    text = " ".join([skill.name, skill.trigger, " ".join(skill.procedure)]).lower()
    hits = sum(1 for term in terms if term in text)
    return hits / len(terms)


def _terms(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]
