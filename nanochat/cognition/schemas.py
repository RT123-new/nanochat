"""Shared typed schemas for the cognition subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Episode:
    episode_id: str
    prompt: str
    response: str
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryItem:
    item_id: str
    content: str
    kind: Literal["episodic", "semantic"]
    source: str = "cognition"
    score: float = 0.0
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trace:
    trace_id: str
    query: str
    decision: str
    rationale: str
    steps: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoutingDecision:
    action: Literal[
        "direct_answer",
        "retrieve_memory",
        "creative_explore",
        "verify",
        "sandbox",
        "consolidate",
    ]
    rationale: str
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.5
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationResult:
    verified: bool
    verdict: str
    issues: list[str] = field(default_factory=list)
    score: float = 0.0
    checked_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SkillArtifact:
    skill_id: str
    name: str
    trigger: str
    procedure: list[str]
    success_signals: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)
