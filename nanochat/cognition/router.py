"""Explicit inspectable routing for cognition behaviors."""

from __future__ import annotations

from .normalize import unique_terms
from .schemas import RoutingDecision


class ExplicitRouter:
    def decide(self, query: str) -> RoutingDecision:
        q = query.strip().lower()
        terms = set(unique_terms(query))
        if not q:
            return RoutingDecision(action="direct_answer", rationale="Empty query; defaulting to direct answer", confidence=0.2)

        if _memory_requested(q):
            return RoutingDecision(action="retrieve_memory", rationale="User references past context or memory", confidence=0.8)

        if _creative_requested(q, terms):
            return RoutingDecision(action="creative_explore", rationale="User asks for divergent/creative exploration", confidence=0.8)

        if _verify_requested(q):
            return RoutingDecision(action="verify", rationale="User requests validation or correctness checking", confidence=0.85)

        if _sandbox_requested(q):
            return RoutingDecision(action="sandbox", rationale="User asks to safely explore hypothetical branches", confidence=0.75)

        if _consolidation_requested(q, terms):
            return RoutingDecision(action="consolidate", rationale="User asks to capture recurring pattern into reusable skill", confidence=0.7)

        return RoutingDecision(action="direct_answer", rationale="No special routing trigger matched", confidence=0.6)


def _memory_requested(query: str) -> bool:
    return any(token in query for token in ("remember", "recall", "previous", "prior", "earlier"))


def _creative_requested(query: str, terms: set[str]) -> bool:
    if any(token in query for token in ("brainstorm", "creative", "alternative", "alternatives", "divergent")):
        return True
    if "creative_explore" in terms and "ideas" in query and any(token in query for token in ("brainstorm", "generate", "explore", "list")):
        return True
    return False


def _verify_requested(query: str) -> bool:
    return any(token in query for token in ("verify", "validate", "validation", "verifying", "validated", "prove"))


def _sandbox_requested(query: str) -> bool:
    return "what if" in query or "simulate" in query or "sandbox" in query


def _consolidation_requested(query: str, terms: set[str]) -> bool:
    if "consolidate" in query or "repeated" in query:
        return True
    if "pattern" in terms and any(token in query for token in ("reuse", "reusable", "skill")):
        return True
    return False
