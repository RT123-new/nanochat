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
    scratchpad = _mean_numeric_fields(
        local_delib_stats,
        include=lambda key: key.startswith("scratch_") or key.startswith("mean_scratch_") or key.endswith("_to_scratch_weight"),
    )
    adaptive_halt = _mean_numeric_fields(
        local_delib_stats,
        include=lambda key: key.startswith("halt_") or key.startswith("halted_") or "_halt_" in key or key in {"mean_steps_taken", "steps_taken"},
    )
    metadata = {
        "model_local_delib.stats": local_delib_stats,
        "model_local_delib.branch": branch,
        "model_local_delib.hierarchy": hierarchy,
        "model_local_delib.scratchpad": scratchpad,
        "model_local_delib.adaptive_halt": adaptive_halt,
    }
    scratchpad_summaries = _scratchpad_summary_metadata(local_delib_stats)
    if scratchpad_summaries:
        metadata["model_local_delib.scratchpad_summaries"] = scratchpad_summaries
    metadata.update(
        _build_compact_thought_summaries(
            local_delib_stats,
            scratchpad_summaries=scratchpad_summaries,
        )
    )
    return metadata


def _build_compact_thought_summaries(
    local_delib_stats: list[dict[str, Any]],
    *,
    scratchpad_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}

    branch_consensus = _compact_numeric_summary(
        local_delib_stats,
        fields=(
            "branch_factor_used",
            "fraction_tokens_branched",
            "mean_branch_disagreement",
            "mean_branch_consensus_weight",
            "mean_branch_verifier_score",
            "mean_branch_entropy",
            "branch_consensus_used",
        ),
        include_row=lambda row: _has_nonzero_numeric_fields(
            row,
            (
                "branch_consensus_used",
                "mean_branch_consensus_weight",
                "mean_branch_verifier_score",
                "mean_branch_disagreement",
            ),
        ),
    )
    if branch_consensus is not None:
        summaries["model_local_delib.thought_summaries.branch_consensus"] = branch_consensus

    deep_hierarchy = _compact_numeric_summary(
        local_delib_stats,
        fields=(
            "phrase_nodes_used",
            "span_nodes_used",
            "sequence_summary_used",
            "hierarchy_depth_used",
            "mean_upward_message_norm",
            "mean_downward_message_norm",
            "mean_scale_gate",
        ),
        include_row=lambda row: _has_nonzero_numeric_fields(
            row,
            (
                "phrase_nodes_used",
                "span_nodes_used",
                "sequence_summary_used",
                "hierarchy_depth_used",
                "mean_upward_message_norm",
                "mean_downward_message_norm",
            ),
        ),
    )
    if deep_hierarchy is not None:
        summaries["model_local_delib.thought_summaries.deep_hierarchy"] = deep_hierarchy

    scratch_summary = _scratch_summary_metadata(local_delib_stats, scratchpad_summaries=scratchpad_summaries)
    if scratch_summary is not None:
        summaries["model_local_delib.thought_summaries.scratch"] = scratch_summary

    thought_graph = _compact_numeric_summary(
        local_delib_stats,
        fields=(
            "thought_nodes_used",
            "mean_thought_degree",
            "mean_token_to_thought_weight",
            "mean_thought_to_token_weight",
            "mean_thought_update_norm",
            "thought_graph_steps_used",
        ),
        include_row=lambda row: _has_nonzero_numeric_fields(
            row,
            (
                "thought_nodes_used",
                "mean_thought_degree",
                "mean_token_to_thought_weight",
                "mean_thought_to_token_weight",
                "mean_thought_update_norm",
                "thought_graph_steps_used",
            ),
        ),
    )
    if thought_graph is not None:
        summaries["model_local_delib.thought_summaries.thought_graph"] = thought_graph

    global_anchors = _compact_numeric_summary(
        local_delib_stats,
        fields=(
            "global_anchors_used",
            "mean_anchor_read_weight",
            "mean_anchor_write_weight",
            "mean_anchor_norm",
        ),
        include_row=lambda row: _has_nonzero_numeric_fields(
            row,
            (
                "global_anchors_used",
                "mean_anchor_read_weight",
                "mean_anchor_write_weight",
                "mean_anchor_norm",
            ),
        ),
    )
    if global_anchors is not None:
        summaries["model_local_delib.thought_summaries.global_anchors"] = global_anchors

    flocking = _compact_numeric_summary(
        local_delib_stats,
        fields=(
            "mean_alignment_norm",
            "mean_cohesion_norm",
            "mean_separation_norm",
            "mean_flocking_total_norm",
            "flocking_neighbor_count",
            "fraction_flocking_tokens_active",
        ),
        include_row=lambda row: _has_nonzero_numeric_fields(
            row,
            (
                "mean_alignment_norm",
                "mean_cohesion_norm",
                "mean_separation_norm",
                "mean_flocking_total_norm",
                "fraction_flocking_tokens_active",
            ),
        ),
    )
    if flocking is not None:
        summaries["model_local_delib.thought_summaries.flocking"] = flocking

    return summaries


def _scratchpad_summary_metadata(local_delib_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in local_delib_stats:
        if not isinstance(row, dict):
            continue
        vector = row.get("scratch_summary_vector")
        if not isinstance(vector, list) or not all(isinstance(value, (int, float)) for value in vector):
            continue
        summaries.append(
            {
                "layer_idx": int(row.get("layer_idx", len(summaries))),
                "summary": [float(value) for value in vector],
            }
        )
    return summaries


def _scratch_summary_metadata(
    local_delib_stats: list[dict[str, Any]],
    *,
    scratchpad_summaries: list[dict[str, Any]],
) -> dict[str, float | int] | None:
    if not scratchpad_summaries:
        return None

    flattened: list[float] = []
    norms: list[float] = []
    max_dim = 0
    for row in scratchpad_summaries:
        summary = row["summary"]
        flattened.extend(summary)
        norms.append(sum(value * value for value in summary) ** 0.5)
        max_dim = max(max_dim, len(summary))

    numeric = _mean_numeric_fields(local_delib_stats, include=lambda key: key in {"scratch_slots_used", "mean_scratch_summary_norm"})
    result: dict[str, float | int] = {
        "layer_count": len(scratchpad_summaries),
        "summary_dim": max_dim,
        "mean_summary_value": sum(flattened) / float(len(flattened)),
        "mean_summary_abs": sum(abs(value) for value in flattened) / float(len(flattened)),
        "max_summary_abs": max(abs(value) for value in flattened),
        "mean_summary_norm": sum(norms) / float(len(norms)),
    }
    result.update(numeric)
    return result


def _compact_numeric_summary(
    local_delib_stats: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    include_row: Callable[[dict[str, Any]], bool],
) -> dict[str, float | int] | None:
    matching_rows = [row for row in local_delib_stats if isinstance(row, dict) and include_row(row)]
    if not matching_rows:
        return None

    summary = _mean_numeric_fields(matching_rows, include=lambda key: key in fields)
    if not summary:
        return None
    return {
        "layer_count": len(matching_rows),
        **summary,
    }


def _has_nonzero_numeric_fields(row: dict[str, Any], fields: tuple[str, ...]) -> bool:
    for key in fields:
        value = row.get(key)
        if isinstance(value, (int, float)) and abs(float(value)) > 1e-12:
            return True
    return False


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
