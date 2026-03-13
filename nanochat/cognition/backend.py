"""Backend contracts for cognition-layer generation."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch

from nanochat.chat_format import render_messages_for_completion

GRAPH_ARTIFACT_VERSION = 1
BRANCH_GRAPH_FIELDS = (
    "agreement",
    "branch_factor_used",
    "fraction_tokens_branched",
    "mean_branch_score",
    "max_branch_score",
    "mean_merge_weight",
    "mean_branch_disagreement",
    "mean_branch_consensus_weight",
    "mean_branch_verifier_score",
    "mean_branch_entropy",
    "branch_consensus_used",
)
THOUGHT_GRAPH_FIELDS = (
    "thought_nodes_used",
    "mean_thought_degree",
    "mean_token_to_thought_weight",
    "mean_thought_to_token_weight",
    "mean_thought_update_norm",
    "thought_graph_steps_used",
)
HIERARCHY_GRAPH_FIELDS = (
    "hierarchy_levels_used",
    "mean_hierarchy_feedback_norm",
    "phrase_nodes_used",
    "span_nodes_used",
    "sequence_summary_used",
    "mean_upward_message_norm",
    "mean_downward_message_norm",
    "mean_scale_gate",
    "hierarchy_depth_used",
)
SCRATCH_GRAPH_FIELDS = (
    "scratch_slots_used",
    "mean_scratch_read_weight",
    "mean_scratch_write_weight",
    "mean_scratch_norm",
    "mean_scratch_refine_norm",
    "mean_scratch_summary_norm",
    "mean_branch_to_scratch_weight",
    "mean_hierarchy_to_scratch_weight",
    "scratch_reset_ok",
)
ANCHOR_GRAPH_FIELDS = (
    "global_anchors_used",
    "mean_anchor_read_weight",
    "mean_anchor_write_weight",
    "mean_anchor_norm",
)
COMPUTE_GRAPH_FIELDS = (
    "executed_steps",
    "mean_executed_steps_per_token",
    "max_executed_steps_any_token",
    "fraction_halted_early",
    "mean_halt",
    "mean_final_halt",
    "mean_steps_taken",
    "halted_token_fraction",
)
FLOCKING_GRAPH_FIELDS = (
    "mean_neighbor_count",
    "mean_sequence_neighbor_weight",
    "mean_semantic_neighbor_weight",
    "mean_phrase_neighbor_weight",
    "semantic_topk_used",
    "mean_alignment_norm",
    "mean_cohesion_norm",
    "mean_separation_norm",
    "mean_flocking_total_norm",
    "flocking_neighbor_count",
    "fraction_flocking_tokens_active",
)
LOCAL_DELIB_RUNTIME_OVERRIDE_KEYS = ("local_delib",)


@dataclass(slots=True)
class LocalDelibRuntimeOverrideReport:
    """Structured report describing how local-deliberation overrides were handled."""

    status: str
    requested_overrides: dict[str, Any]
    applied_overrides: dict[str, Any]
    application_method: str
    reason: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "status": self.status,
            "requested_overrides": dict(self.requested_overrides),
            "applied_overrides": dict(self.applied_overrides),
            "application_method": self.application_method,
        }
        if self.reason:
            metadata["reason"] = self.reason
        return metadata


class LocalDelibRuntimeOverrideError(RuntimeError):
    """Raised when an exact runtime override cannot be applied."""

    def __init__(self, report: LocalDelibRuntimeOverrideReport) -> None:
        self.report = report
        super().__init__(report.reason or "local-deliberation runtime override could not be applied")


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
    allow_approximate_local_delib_overrides: bool = False
    supports_local_delib_runtime_overrides: bool = True
    last_generation_metadata: dict[str, Any] | None = None
    last_local_delib_runtime_override_report: LocalDelibRuntimeOverrideReport | None = None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.last_generation_metadata = None
        self.last_local_delib_runtime_override_report = None
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
        runtime_override_kwargs, generation_kwargs = self._split_local_delib_runtime_kwargs(kwargs)
        runtime_override_report, runtime_override_model = self._resolve_local_delib_runtime_overrides(runtime_override_kwargs)
        self.last_local_delib_runtime_override_report = runtime_override_report

        with self._temporary_engine_model(runtime_override_model):
            results, _ = self.engine.generate_batch(
                tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
                **generation_kwargs,
            )
        self._capture_generation_metadata(runtime_override_report=runtime_override_report)
        return self.tokenizer.decode(results[0][len(tokens):]).strip()

    def _capture_generation_metadata(
        self,
        *,
        runtime_override_report: LocalDelibRuntimeOverrideReport | None = None,
    ) -> None:
        model = getattr(self.engine, "model", None)
        local_delib_stats = getattr(model, "last_deliberation_stats", None)
        if local_delib_stats is None:
            self.last_generation_metadata = {}
        else:
            namespaced = build_local_delib_namespaced_metadata(local_delib_stats)
            self.last_generation_metadata = {
                "local_deliberation_stats": local_delib_stats,
                **namespaced,
            }
        if runtime_override_report is not None:
            if self.last_generation_metadata is None:
                self.last_generation_metadata = {}
            self.last_generation_metadata["local_delib_runtime_override"] = runtime_override_report.to_metadata()
        if self.last_generation_metadata == {}:
            self.last_generation_metadata = None
            return

    def _prompt_max_tokens(self) -> int | None:
        if self.prompt_max_tokens is not None:
            return self.prompt_max_tokens
        model = getattr(self.engine, "model", None)
        config = getattr(model, "config", None)
        return getattr(config, "sequence_len", None)

    def _split_local_delib_runtime_kwargs(self, kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        runtime_override_kwargs: dict[str, Any] = {}
        generation_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if _is_local_delib_runtime_override_key(key):
                runtime_override_kwargs[key] = value
            else:
                generation_kwargs[key] = value
        return runtime_override_kwargs, generation_kwargs

    def _resolve_local_delib_runtime_overrides(
        self,
        runtime_override_kwargs: dict[str, Any],
    ) -> tuple[LocalDelibRuntimeOverrideReport | None, Any | None]:
        if not runtime_override_kwargs:
            return None, None

        model = getattr(self.engine, "model", None)
        config = getattr(model, "config", None)
        requested_overrides = dict(runtime_override_kwargs)
        if model is None or config is None:
            report = LocalDelibRuntimeOverrideReport(
                status="unsupported",
                requested_overrides=requested_overrides,
                applied_overrides={},
                application_method="unsupported",
                reason="engine model does not expose a mutable GPTConfig",
            )
            self.last_generation_metadata = {"local_delib_runtime_override": report.to_metadata()}
            raise LocalDelibRuntimeOverrideError(report)

        unknown_keys = sorted(key for key in requested_overrides if not hasattr(config, key))
        if unknown_keys:
            report = LocalDelibRuntimeOverrideReport(
                status="unsupported",
                requested_overrides=requested_overrides,
                applied_overrides={},
                application_method="unsupported",
                reason=f"unknown local-deliberation override keys: {', '.join(unknown_keys)}",
            )
            self.last_generation_metadata = {"local_delib_runtime_override": report.to_metadata()}
            raise LocalDelibRuntimeOverrideError(report)

        changed_overrides = {
            key: value
            for key, value in requested_overrides.items()
            if getattr(config, key) != value
        }
        if not changed_overrides:
            report = LocalDelibRuntimeOverrideReport(
                status="exact",
                requested_overrides=requested_overrides,
                applied_overrides=requested_overrides,
                application_method="loaded_checkpoint",
                reason="requested overrides already match the loaded checkpoint config",
            )
            return report, None

        try:
            target_config = copy.deepcopy(config)
            for key, value in changed_overrides.items():
                setattr(target_config, key, value)
        except Exception as exc:
            report = LocalDelibRuntimeOverrideReport(
                status="unsupported",
                requested_overrides=requested_overrides,
                applied_overrides={},
                application_method="unsupported",
                reason=f"engine model config does not allow runtime override mutation: {exc}",
            )
            self.last_generation_metadata = {"local_delib_runtime_override": report.to_metadata()}
            raise LocalDelibRuntimeOverrideError(report) from exc

        try:
            runtime_override_model = self._instantiate_runtime_override_model(target_config)
        except Exception as exc:
            reason = (
                "runtime override requires a model shape/module layout that is not strictly "
                f"state-dict compatible with the loaded checkpoint: {exc}"
            )
            if self.allow_approximate_local_delib_overrides:
                report = LocalDelibRuntimeOverrideReport(
                    status="approximated",
                    requested_overrides=requested_overrides,
                    applied_overrides={},
                    application_method="loaded_checkpoint_fallback",
                    reason=reason,
                )
                return report, None
            report = LocalDelibRuntimeOverrideReport(
                status="unsupported",
                requested_overrides=requested_overrides,
                applied_overrides={},
                application_method="unsupported",
                reason=reason,
            )
            self.last_generation_metadata = {"local_delib_runtime_override": report.to_metadata()}
            raise LocalDelibRuntimeOverrideError(report) from exc

        report = LocalDelibRuntimeOverrideReport(
            status="exact",
            requested_overrides=requested_overrides,
            applied_overrides=requested_overrides,
            application_method="reinstantiated_model",
        )
        return report, runtime_override_model

    def _instantiate_runtime_override_model(self, target_config: Any) -> Any:
        base_model = getattr(self.engine, "model", None)
        if base_model is None:
            raise ValueError("engine model is unavailable")

        model_cls = base_model.__class__
        state_dict = base_model.state_dict()
        with torch.device("meta"):
            runtime_override_model = model_cls(target_config)
        runtime_override_model.to_empty(device=base_model.get_device())
        if hasattr(runtime_override_model, "init_weights"):
            runtime_override_model.init_weights()
        runtime_override_model.load_state_dict(state_dict, strict=True, assign=True)
        if base_model.training:
            runtime_override_model.train()
        else:
            runtime_override_model.eval()
        return runtime_override_model

    @contextmanager
    def _temporary_engine_model(self, runtime_override_model: Any | None):
        if runtime_override_model is None:
            yield
            return
        original_model = getattr(self.engine, "model", None)
        self.engine.model = runtime_override_model
        try:
            yield
        finally:
            self.engine.model = original_model


def _is_local_delib_runtime_override_key(key: str) -> bool:
    return key in LOCAL_DELIB_RUNTIME_OVERRIDE_KEYS or key.startswith("local_delib_")


def build_local_delib_namespaced_metadata(local_delib_stats: list[dict[str, Any]]) -> dict[str, Any]:
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
    graph_artifact = build_local_delib_graph_artifact(
        local_delib_stats,
        scratchpad_summaries=scratchpad_summaries,
    )
    if graph_artifact is not None:
        metadata["model_local_delib.graph_artifact"] = graph_artifact
    metadata.update(
        _build_compact_thought_summaries(
            local_delib_stats,
            scratchpad_summaries=scratchpad_summaries,
        )
    )
    return metadata


def summarize_local_delib_for_creative_policy(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Extract lightweight model-summary signals for wrapper creativity policy."""
    if not isinstance(metadata, dict):
        return {}

    graph_artifact = metadata.get("model_local_delib.graph_artifact")
    overview = graph_artifact.get("overview", {}) if isinstance(graph_artifact, dict) else {}
    branch_summary = _artifact_section_summary(graph_artifact, "branch") or metadata.get("model_local_delib.thought_summaries.branch_consensus", {})
    hierarchy_summary = _artifact_section_summary(graph_artifact, "hierarchy") or metadata.get("model_local_delib.thought_summaries.deep_hierarchy", {})
    scratch_summary = _artifact_section_summary(graph_artifact, "scratch") or metadata.get("model_local_delib.thought_summaries.scratch", {})
    thought_summary = _artifact_section_summary(graph_artifact, "thought_graph") or metadata.get("model_local_delib.thought_summaries.thought_graph", {})
    anchor_summary = _artifact_section_summary(graph_artifact, "anchors") or metadata.get("model_local_delib.thought_summaries.global_anchors", {})
    compute_summary = _artifact_section_summary(graph_artifact, "compute") or metadata.get("model_local_delib.adaptive_halt", {})

    summary: dict[str, Any] = {}
    active_sections = overview.get("active_sections")
    if isinstance(active_sections, list) and active_sections:
        summary["active_sections"] = [str(section) for section in active_sections]

    numeric_fields = {
        "branch_disagreement": _numeric_lookup(branch_summary, "mean_branch_disagreement"),
        "branch_consensus_weight": _numeric_lookup(branch_summary, "mean_branch_consensus_weight"),
        "branch_consensus_used": _numeric_lookup(branch_summary, "branch_consensus_used"),
        "branch_verifier_score": _numeric_lookup(branch_summary, "mean_branch_verifier_score"),
        "scratch_slots_used": _numeric_lookup(scratch_summary, "scratch_slots_used"),
        "scratch_summary_dim": _numeric_lookup(scratch_summary, "summary_dim"),
        "thought_nodes_used": _numeric_lookup(thought_summary, "thought_nodes_used"),
        "hierarchy_depth_used": _numeric_lookup(hierarchy_summary, "hierarchy_depth_used"),
        "global_anchors_used": _numeric_lookup(anchor_summary, "global_anchors_used"),
        "mean_anchor_read_weight": _numeric_lookup(anchor_summary, "mean_anchor_read_weight"),
        "mean_steps_taken": _numeric_lookup(compute_summary, "mean_steps_taken"),
    }
    for key, value in numeric_fields.items():
        if value is not None:
            summary[key] = value

    if "scratch_summary_dim" not in summary:
        scratchpad_summaries = metadata.get("model_local_delib.scratchpad_summaries", [])
        if isinstance(scratchpad_summaries, list) and scratchpad_summaries:
            first_row = scratchpad_summaries[0]
            if isinstance(first_row, dict) and isinstance(first_row.get("summary"), list):
                summary["scratch_summary_dim"] = float(len(first_row["summary"]))

    return summary


def build_local_delib_graph_artifact(
    local_delib_stats: list[dict[str, Any]],
    *,
    scratchpad_summaries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    artifact: dict[str, Any] = {}

    branch = _build_graph_section(
        local_delib_stats,
        fields=BRANCH_GRAPH_FIELDS,
        include_row=lambda row: _has_nonzero_numeric_fields(
            row,
            (
                "branch_factor_used",
                "fraction_tokens_branched",
                "mean_branch_score",
                "mean_branch_disagreement",
                "mean_branch_consensus_weight",
                "mean_branch_verifier_score",
                "branch_consensus_used",
            ),
        ),
        extra_summary=lambda rows: {
            "consensus_active_layers": _count_rows_with_nonzero_fields(rows, ("branch_consensus_used", "mean_branch_consensus_weight")),
            "verifier_active_layers": _count_rows_with_nonzero_fields(rows, ("mean_branch_verifier_score",)),
        },
    )
    if branch is not None:
        artifact["branch"] = branch

    thought_graph = _build_graph_section(
        local_delib_stats,
        fields=THOUGHT_GRAPH_FIELDS,
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
        extra_summary=lambda rows: {
            "degree_pattern": _degree_pattern(
                _mean_numeric_fields(rows, include=lambda key: key == "mean_thought_degree").get("mean_thought_degree", 0.0)
            ),
        },
    )
    if thought_graph is not None:
        artifact["thought_graph"] = thought_graph

    hierarchy = _build_graph_section(
        local_delib_stats,
        fields=HIERARCHY_GRAPH_FIELDS,
        include_row=lambda row: _has_nonzero_numeric_fields(
            row,
            (
                "hierarchy_levels_used",
                "mean_hierarchy_feedback_norm",
                "phrase_nodes_used",
                "span_nodes_used",
                "sequence_summary_used",
                "hierarchy_depth_used",
            ),
        ),
        list_fields=("hierarchy_level_chunk_counts",),
        extra_summary=lambda rows: {
            "scale_layers": {
                "phrase": _count_rows_with_nonzero_fields(rows, ("phrase_nodes_used",)),
                "span": _count_rows_with_nonzero_fields(rows, ("span_nodes_used",)),
                "sequence": _count_rows_with_nonzero_fields(rows, ("sequence_summary_used",)),
            }
        },
    )
    if hierarchy is not None:
        artifact["hierarchy"] = hierarchy

    scratch = _build_graph_section(
        local_delib_stats,
        fields=SCRATCH_GRAPH_FIELDS,
        include_row=lambda row: _has_nonzero_numeric_fields(
            row,
            (
                "scratch_slots_used",
                "mean_scratch_read_weight",
                "mean_scratch_write_weight",
                "mean_scratch_refine_norm",
                "mean_scratch_summary_norm",
                "mean_branch_to_scratch_weight",
                "mean_hierarchy_to_scratch_weight",
            ),
        ) or isinstance(row.get("scratch_summary_vector"), list),
    )
    if scratch is not None:
        scratch_summary = _scratch_summary_metadata(
            local_delib_stats,
            scratchpad_summaries=scratchpad_summaries,
        )
        if scratch_summary is not None:
            scratch["summary"].update(scratch_summary)
            scratch["summary"]["has_exported_summaries"] = True
            summary_dims = {int(row["layer_idx"]): len(row["summary"]) for row in scratchpad_summaries}
            for layer in scratch["layers"]:
                dim = summary_dims.get(int(layer["layer_idx"]))
                if dim is not None:
                    layer["summary_exported"] = True
                    layer["summary_dim"] = dim
        artifact["scratch"] = scratch

    anchors = _build_graph_section(
        local_delib_stats,
        fields=ANCHOR_GRAPH_FIELDS,
        include_row=lambda row: _has_nonzero_numeric_fields(
            row,
            (
                "global_anchors_used",
                "mean_anchor_read_weight",
                "mean_anchor_write_weight",
                "mean_anchor_norm",
            ),
        ),
        extra_summary=lambda rows: {
            "write_active_layers": _count_rows_with_nonzero_fields(rows, ("mean_anchor_write_weight",)),
        },
    )
    if anchors is not None:
        artifact["anchors"] = anchors

    compute = _build_graph_section(
        local_delib_stats,
        fields=COMPUTE_GRAPH_FIELDS,
        include_row=lambda row: _has_positive_numeric_fields(
            row,
            (
                "executed_steps",
                "mean_executed_steps_per_token",
                "max_executed_steps_any_token",
                "fraction_halted_early",
                "mean_halt",
                "mean_final_halt",
                "mean_steps_taken",
                "halted_token_fraction",
            ),
        ),
        extra_summary=lambda rows: {
            "adaptive_halt_active_layers": _count_rows_with_nonzero_fields(
                rows,
                ("halted_token_fraction", "fraction_halted_early"),
            ),
        },
    )
    if compute is not None:
        artifact["compute"] = compute

    flocking = _build_graph_section(
        local_delib_stats,
        fields=FLOCKING_GRAPH_FIELDS,
        include_row=lambda row: _has_nonzero_numeric_fields(
            row,
            (
                "mean_neighbor_count",
                "mean_sequence_neighbor_weight",
                "mean_semantic_neighbor_weight",
                "mean_phrase_neighbor_weight",
                "semantic_topk_used",
                "mean_alignment_norm",
                "mean_cohesion_norm",
                "mean_separation_norm",
                "mean_flocking_total_norm",
                "flocking_neighbor_count",
                "fraction_flocking_tokens_active",
            ),
        ),
        extra_summary=lambda rows: {
            "neighbor_graph_active_layers": _count_rows_with_nonzero_fields(
                rows,
                (
                    "mean_neighbor_count",
                    "mean_sequence_neighbor_weight",
                    "mean_semantic_neighbor_weight",
                    "mean_phrase_neighbor_weight",
                    "semantic_topk_used",
                ),
            ),
            "flocking_active_layers": _count_rows_with_nonzero_fields(
                rows,
                (
                    "mean_alignment_norm",
                    "mean_cohesion_norm",
                    "mean_separation_norm",
                    "mean_flocking_total_norm",
                    "fraction_flocking_tokens_active",
                ),
            ),
        },
    )
    if flocking is not None:
        artifact["flocking"] = flocking

    if not artifact:
        return None

    active_layers = sorted(
        {
            int(layer)
            for section in artifact.values()
            for layer in section["summary"]["active_layers"]
        }
    )
    overview: dict[str, Any] = {
        "trace_version": GRAPH_ARTIFACT_VERSION,
        "layer_count": len([row for row in local_delib_stats if isinstance(row, dict)]),
        "active_layer_count": len(active_layers),
        "active_layers": active_layers,
        "active_sections": list(artifact.keys()),
        "section_layer_counts": {
            name: int(section["summary"]["layer_count"])
            for name, section in artifact.items()
        },
    }
    if scratchpad_summaries:
        overview["scratch_summary_layers"] = len(scratchpad_summaries)
    return {
        "overview": overview,
        **artifact,
    }


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


def _build_graph_section(
    local_delib_stats: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    include_row: Callable[[dict[str, Any]], bool],
    list_fields: tuple[str, ...] = (),
    extra_summary: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    matching_rows = [row for row in local_delib_stats if isinstance(row, dict) and include_row(row)]
    if not matching_rows:
        return None

    summary: dict[str, Any] = {
        "layer_count": len(matching_rows),
        "active_layers": [_layer_idx(row, fallback_idx) for fallback_idx, row in enumerate(matching_rows)],
    }
    summary.update(_mean_numeric_fields(matching_rows, include=lambda key: key in fields))
    if extra_summary is not None:
        summary.update(extra_summary(matching_rows))

    return {
        "summary": summary,
        "layers": [
            _compact_layer_row(
                row,
                fields=fields,
                list_fields=list_fields,
                fallback_idx=fallback_idx,
            )
            for fallback_idx, row in enumerate(matching_rows)
        ],
    }


def _artifact_section_summary(graph_artifact: Any, section: str) -> dict[str, Any]:
    if not isinstance(graph_artifact, dict):
        return {}
    section_value = graph_artifact.get(section)
    if not isinstance(section_value, dict):
        return {}
    summary = section_value.get("summary")
    return summary if isinstance(summary, dict) else {}


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


def _compact_layer_row(
    row: dict[str, Any],
    *,
    fields: tuple[str, ...],
    list_fields: tuple[str, ...],
    fallback_idx: int,
) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "layer_idx": _layer_idx(row, fallback_idx),
    }
    for key in fields:
        value = _normalize_graph_value(row.get(key))
        if value is not None:
            compact[key] = value
    for key in list_fields:
        value = row.get(key)
        if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
            compact[key] = [float(item) for item in value]
    return compact


def _layer_idx(row: dict[str, Any], fallback_idx: int) -> int:
    raw_idx = row.get("layer_idx", fallback_idx)
    if isinstance(raw_idx, bool):
        return fallback_idx
    if isinstance(raw_idx, (int, float)):
        return int(raw_idx)
    return fallback_idx


def _normalize_graph_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _numeric_lookup(values: Any, key: str) -> float | None:
    if not isinstance(values, dict):
        return None
    return _normalize_graph_value(values.get(key))


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


def _has_positive_numeric_fields(row: dict[str, Any], fields: tuple[str, ...]) -> bool:
    for key in fields:
        value = row.get(key)
        if isinstance(value, (int, float)) and float(value) > 0.0:
            return True
    return False


def _count_rows_with_nonzero_fields(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> int:
    return sum(1 for row in rows if _has_nonzero_numeric_fields(row, fields))


def _degree_pattern(mean_degree: float) -> str:
    if mean_degree <= 0.0:
        return "inactive"
    if mean_degree < 1.5:
        return "sparse"
    if mean_degree < 3.0:
        return "moderate"
    return "dense"


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
