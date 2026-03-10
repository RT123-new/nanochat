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

        if "memory_reuse" in terms:
            return RoutingDecision(action="retrieve_memory", rationale="User references past context or memory", confidence=0.8)

        if "creative_explore" in terms or "creative" in terms:
            return RoutingDecision(action="creative_explore", rationale="User asks for divergent/creative exploration", confidence=0.8)

        if "verify" in terms or "prove" in terms:
            return RoutingDecision(action="verify", rationale="User requests validation or correctness checking", confidence=0.85)

        if "simulate" in terms or "sandbox" in terms or "what if" in q:
            return RoutingDecision(action="sandbox", rationale="User asks to safely explore hypothetical branches", confidence=0.75)

        if "pattern" in terms or "repeated" in terms or "consolidate" in terms:
            return RoutingDecision(action="consolidate", rationale="User asks to capture recurring pattern into reusable skill", confidence=0.7)

        return RoutingDecision(action="direct_answer", rationale="No special routing trigger matched", confidence=0.6)
