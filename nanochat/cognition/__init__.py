"""Experimental developmental cognition layer for nanochat.

This package is intentionally lightweight and optional.
"""

from .backend import BackendAdapter, GenerationBackend
from .consolidation import Consolidator
from .memory import EpisodicMemory, SemanticMemory
from .router import ExplicitRouter
from .skills import SkillMatch, SkillRegistry
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
    "Consolidator",
    "GenerationBackend",
    "EpisodicMemory",
    "SemanticMemory",
    "ExplicitRouter",
    "SkillMatch",
    "SkillRegistry",
    "Episode",
    "Hypothesis",
    "MemoryItem",
    "RoutingDecision",
    "SkillArtifact",
    "Trace",
    "VerificationResult",
]
