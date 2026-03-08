"""Experimental developmental cognition layer for nanochat.

This package is intentionally lightweight and optional.
"""

from .backend import BackendAdapter, GenerationBackend
from .memory import EpisodicMemory, SemanticMemory
from .router import ExplicitRouter
from .schemas import (
    Episode,
    Hypothesis,
    MemoryItem,
    RoutingDecision,
    SkillArtifact,
    Trace,
    VerificationResult,
)

__all__ = [
    "BackendAdapter",
    "GenerationBackend",
    "EpisodicMemory",
    "SemanticMemory",
    "ExplicitRouter",
    "Episode",
    "Hypothesis",
    "MemoryItem",
    "RoutingDecision",
    "SkillArtifact",
    "Trace",
    "VerificationResult",
]
