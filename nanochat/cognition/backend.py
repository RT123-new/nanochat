"""Backend contracts for cognition-layer generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class GenerationBackend(Protocol):
    """Minimal protocol for text generation backends."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate one text response for a prompt."""


@dataclass(slots=True)
class BackendAdapter:
    """Thin adapter around a generation backend.

    This keeps cognition modules decoupled from the existing Engine while making
    integration straightforward later.
    """

    backend: GenerationBackend
    default_kwargs: dict[str, Any] = field(default_factory=dict)

    def run(self, prompt: str, **kwargs: Any) -> str:
        merged = {**self.default_kwargs, **kwargs}
        return self.backend.generate(prompt, **merged)
