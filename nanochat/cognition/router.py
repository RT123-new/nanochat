"""Explicit inspectable routing for cognition behaviors."""

from __future__ import annotations

from .schemas import RoutingDecision


class ExplicitRouter:
    def decide(self, query: str) -> RoutingDecision:
        q = query.strip().lower()
        if not q:
            return RoutingDecision(action="direct_answer", rationale="Empty query; defaulting to direct answer", confidence=0.2)

        if "remember" in q or "recall" in q or "previous" in q:
            return RoutingDecision(action="retrieve_memory", rationale="User references past context or memory", confidence=0.8)

        if "brainstorm" in q or "creative" in q or "ideas" in q:
            return RoutingDecision(action="creative_explore", rationale="User asks for divergent/creative exploration", confidence=0.8)

        if "verify" in q or "check" in q or "prove" in q:
            return RoutingDecision(action="verify", rationale="User requests validation or correctness checking", confidence=0.85)

        if "simulate" in q or "sandbox" in q or "what if" in q:
            return RoutingDecision(action="sandbox", rationale="User asks to safely explore hypothetical branches", confidence=0.75)

        if "pattern" in q or "repeated" in q or "consolidate" in q:
            return RoutingDecision(action="consolidate", rationale="User asks to capture recurring pattern into reusable skill", confidence=0.7)

        return RoutingDecision(action="direct_answer", rationale="No special routing trigger matched", confidence=0.6)
