"""Experimental developmental cognition layer for nanochat.

This package is intentionally lightweight and optional.
"""

from .backend import BackendAdapter, EngineBackend, GenerationBackend
from .agent import CognitionAgent, CognitionResult
from .consolidation import Consolidator
from .creative import CreativeWorkspace
from .memory import EpisodicMemory, RankedEpisode, SemanticMemory
from .router import ExplicitRouter
from .sandbox import LightweightSandbox
from .skills import SkillMatch, SkillRegistry
from .traces import TraceRecorder
from .verifier import RankedCandidate, VerifierWorkspace
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
    "CognitionAgent",
    "CognitionResult",
    "Consolidator",
    "CreativeWorkspace",
    "EngineBackend",
    "GenerationBackend",
    "EpisodicMemory",
    "RankedEpisode",
    "SemanticMemory",
    "ExplicitRouter",
    "LightweightSandbox",
    "SkillMatch",
    "TraceRecorder",
    "SkillRegistry",
    "RankedCandidate",
    "VerifierWorkspace",
    "Episode",
    "Hypothesis",
    "MemoryItem",
    "RoutingDecision",
    "SkillArtifact",
    "Trace",
    "VerificationResult",
]
