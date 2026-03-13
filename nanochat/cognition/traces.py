"""Trace assembly helpers for cognition loop inspection."""

from __future__ import annotations

from typing import Any

from .schemas import Trace


class TraceRecorder:
    def __init__(self) -> None:
        self._counter = 0

    def build(self, query: str, decision: str, rationale: str, steps: list[str], metadata: dict[str, object] | None = None) -> Trace:
        self._counter += 1
        return Trace(
            trace_id=f"trace-{self._counter}",
            query=query,
            decision=decision,
            rationale=rationale,
            steps=list(steps),
            metadata=_trace_safe_copy(metadata or {}),
        )


def _trace_safe_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _trace_safe_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_trace_safe_copy(item) for item in value]
    return value
