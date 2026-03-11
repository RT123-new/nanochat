"""Backend contracts for cognition-layer generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from nanochat.chat_format import render_messages_for_completion


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


@dataclass(slots=True)
class EngineBackend:
    """Checkpoint-backed generation adapter using nanochat's Engine."""

    engine: Any
    tokenizer: Any
    system_prompt: str | None = None
    prompt_max_tokens: int | None = None
    max_tokens: int = 256
    temperature: float = 0.6
    top_k: int | None = 50
    seed: int = 42
    last_generation_metadata: dict[str, Any] | None = None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.last_generation_metadata = None
        system_prompt = kwargs.pop("system_prompt", self.system_prompt)
        messages = kwargs.pop("messages", None)
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        prompt_max_tokens = kwargs.pop("prompt_max_tokens", self._prompt_max_tokens())
        tokens = render_messages_for_completion(
            self.tokenizer,
            messages,
            max_tokens=prompt_max_tokens,
        )
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)
        top_k = kwargs.pop("top_k", self.top_k)
        seed = kwargs.pop("seed", self.seed)

        results, _ = self.engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
        self._capture_generation_metadata()
        return self.tokenizer.decode(results[0][len(tokens):]).strip()

    def _capture_generation_metadata(self) -> None:
        model = getattr(self.engine, "model", None)
        local_delib_stats = getattr(model, "last_deliberation_stats", None)
        if local_delib_stats is None:
            self.last_generation_metadata = None
            return
        self.last_generation_metadata = {"local_deliberation_stats": local_delib_stats}

    def _prompt_max_tokens(self) -> int | None:
        if self.prompt_max_tokens is not None:
            return self.prompt_max_tokens
        model = getattr(self.engine, "model", None)
        config = getattr(model, "config", None)
        return getattr(config, "sequence_len", None)
