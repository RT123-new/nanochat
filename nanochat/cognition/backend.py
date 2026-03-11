"""Backend contracts for cognition-layer generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

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
            **kwargs,
        )
        self._capture_generation_metadata()
        return self.tokenizer.decode(results[0][len(tokens):]).strip()

    def _capture_generation_metadata(self) -> None:
        model = getattr(self.engine, "model", None)
        local_delib_stats = getattr(model, "last_deliberation_stats", None)
        if local_delib_stats is None:
            self.last_generation_metadata = None
            return
        namespaced = _build_local_delib_namespaced_metadata(local_delib_stats)
        self.last_generation_metadata = {
            "local_deliberation_stats": local_delib_stats,
            **namespaced,
        }

    def _prompt_max_tokens(self) -> int | None:
        if self.prompt_max_tokens is not None:
            return self.prompt_max_tokens
        model = getattr(self.engine, "model", None)
        config = getattr(model, "config", None)
        return getattr(config, "sequence_len", None)


def _build_local_delib_namespaced_metadata(local_delib_stats: list[dict[str, Any]]) -> dict[str, Any]:
    branch = _mean_numeric_fields(local_delib_stats, include=lambda key: key.startswith("branch_") or key == "agreement")
    hierarchy = _mean_numeric_fields(local_delib_stats, include=lambda key: key.startswith("hierarchy_"))
    scratchpad = _mean_numeric_fields(local_delib_stats, include=lambda key: key.startswith("scratch_"))
    adaptive_halt = _mean_numeric_fields(
        local_delib_stats,
        include=lambda key: key.startswith("halt_") or key.startswith("halted_") or "_halt_" in key or key in {"mean_steps_taken", "steps_taken"},
    )
    return {
        "model_local_delib.stats": local_delib_stats,
        "model_local_delib.branch": branch,
        "model_local_delib.hierarchy": hierarchy,
        "model_local_delib.scratchpad": scratchpad,
        "model_local_delib.adaptive_halt": adaptive_halt,
    }


def _mean_numeric_fields(
    local_delib_stats: list[dict[str, Any]],
    *,
    include: Callable[[str], bool],
) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in local_delib_stats:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if not include(key) or not isinstance(value, (int, float)):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sorted(sums)}
