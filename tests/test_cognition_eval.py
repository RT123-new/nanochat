from __future__ import annotations

import json

import pytest

from nanochat.cognition.backend import LocalDelibRuntimeOverrideError, LocalDelibRuntimeOverrideReport
from nanochat.cognition.eval import (
    ADVANCED_LOCAL_DELIB_CASES,
    ADVANCED_LOCAL_DELIB_VARIANTS,
    ContextAwareEvalBackend,
    DEFAULT_CASES,
    DEFAULT_LOCAL_DELIB_CASES,
    DEFAULT_LOCAL_DELIB_VARIANTS,
    EngineSmokeArtifactRecord,
    EngineSmokeCommandRecord,
    EngineSmokeManifest,
    EvalCase,
    LocalDelibVariant,
    LocalDelibContextEvalBackend,
    NATURAL_LOCAL_DELIB_CASES,
    NATURAL_LOCAL_DELIB_VARIANTS,
    RESEARCH_LOCAL_DELIB_CASES,
    RESEARCH_LOCAL_DELIB_VARIANTS,
    run_natural_local_delib_eval,
    run_advanced_local_delib_ablation_eval,
    run_eval,
    run_local_delib_ablation_eval,
    run_research_local_delib_eval,
    run_task_grounded_eval,
    write_advanced_local_delib_eval_artifact,
    write_engine_smoke_manifest,
    write_eval_artifact,
    write_local_delib_eval_artifact,
    write_natural_local_delib_eval_artifact,
    write_research_local_delib_eval_artifact,
    write_task_grounded_eval_artifact,
)


class FakeEvalBackend:
    def generate(self, prompt: str, **kwargs: object) -> str:
        return "same response"


class UnsupportedRuntimeOverrideBackend:
    supports_local_delib_runtime_overrides = True

    def __init__(self) -> None:
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        requested_overrides = {
            str(key): value
            for key, value in kwargs.items()
            if str(key) == "local_delib" or str(key).startswith("local_delib_")
        }
        report = LocalDelibRuntimeOverrideReport(
            status="unsupported",
            requested_overrides=requested_overrides,
            applied_overrides={},
            application_method="unsupported_backend",
            reason="backend cannot apply requested local-deliberation variant",
        )
        self.last_generation_metadata = {"local_delib_runtime_override": report.to_metadata()}
        raise LocalDelibRuntimeOverrideError(report)


class MixedRuntimeOverrideBackend:
    supports_local_delib_runtime_overrides = True

    def __init__(self) -> None:
        self.call_count = 0
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.call_count += 1
        requested_overrides = {
            str(key): value
            for key, value in kwargs.items()
            if str(key) == "local_delib" or str(key).startswith("local_delib_")
        }
        status = "unsupported" if self.call_count == 1 else "exact"
        report = LocalDelibRuntimeOverrideReport(
            status=status,
            requested_overrides=requested_overrides,
            applied_overrides=requested_overrides if status == "exact" else {},
            application_method="mixed_backend",
            reason="flaky test backend" if status == "unsupported" else None,
        )
        self.last_generation_metadata = {"local_delib_runtime_override": report.to_metadata()}
        return "alpha" if status == "exact" else ""


class ApproximateRuntimeOverrideBackend:
    supports_local_delib_runtime_overrides = True

    def __init__(self) -> None:
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        requested_overrides = {
            str(key): value
            for key, value in kwargs.items()
            if str(key) == "local_delib" or str(key).startswith("local_delib_")
        }
        report = LocalDelibRuntimeOverrideReport(
            status="approximated",
            requested_overrides=requested_overrides,
            applied_overrides={},
            application_method="loaded_checkpoint_fallback",
            reason="checkpoint-compatible approximation only",
        )
        self.last_generation_metadata = {"local_delib_runtime_override": report.to_metadata()}
        return "approximate fallback response"


class TinyTask:
    def __init__(self) -> None:
        self.rows = [
            {
                "messages": [
                    {"role": "user", "content": "solve arithmetic task"},
                    {"role": "assistant", "content": "#### 2"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "write a small function"},
                    {"role": "assistant", "content": "def ok():\n    return 1"},
                ]
            },
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]

    def evaluate(self, conversation: dict[str, object], assistant_response: str) -> bool:
        prompt = conversation["messages"][0]["content"]
        if prompt == "solve arithmetic task":
            return assistant_response.strip() == "#### 2"
        return assistant_response.strip() == "def ok():\n    return 1"


class TaskGroundedBackend:
    def __init__(self) -> None:
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.last_generation_metadata = {}
        if "Relevant episodic memory:" in prompt or "Relevant semantic memory:" in prompt or "Relevant skill:" in prompt:
            return "#### 2" if "solve arithmetic task" in prompt else "def ok():\n    return 1"
        if "solve arithmetic task" in prompt:
            return "#### 0"
        return "def bad():\n    return 0"


def test_run_eval_produces_comparison_rows_and_route_counts() -> None:
    summary = run_eval(DEFAULT_CASES, backend=ContextAwareEvalBackend())

    assert len(summary.rows) == len(DEFAULT_CASES)
    assert set(summary.route_counts)
    assert summary.delta > 0.0
    rows_by_id = {row.case_id: row for row in summary.rows}
    assert rows_by_id["memory_reuse_paraphrase"].cognition_score > rows_by_id["memory_reuse_paraphrase"].baseline_score
    assert rows_by_id["skill_reuse_paraphrase"].cognition_score > rows_by_id["skill_reuse_paraphrase"].baseline_score


def test_run_eval_raises_when_required_cases_do_not_improve() -> None:
    with pytest.raises(AssertionError):
        run_eval(DEFAULT_CASES, backend=FakeEvalBackend())


def test_run_eval_can_disable_required_improvement_enforcement() -> None:
    summary = run_eval(DEFAULT_CASES, backend=FakeEvalBackend(), enforce_improvement=False)

    assert len(summary.rows) == len(DEFAULT_CASES)
    assert summary.delta == 0.0


def test_write_eval_artifact_writes_json_payload(tmp_path) -> None:
    summary = run_eval(DEFAULT_CASES, backend=ContextAwareEvalBackend())

    artifact_path = write_eval_artifact(summary, str(tmp_path / "cognition_eval.json"))

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(payload) == {"baseline_mean", "cognition_mean", "delta", "route_counts", "rows"}
    assert payload["baseline_mean"] == summary.baseline_mean
    assert payload["cognition_mean"] == summary.cognition_mean
    assert len(payload["rows"]) == len(DEFAULT_CASES)


def test_run_eval_surfaces_prompt3_creative_metadata() -> None:
    cases = [
        EvalCase(
            case_id="creative_brainstorm",
            prompt="Brainstorm ideas for memory routing.",
            expected_keywords=["ideas", "memory", "options"],
            requires_cognition_gain=True,
        )
    ]

    summary = run_eval(cases, backend=ContextAwareEvalBackend())

    row = summary.rows[0]
    assert row.cognition_decision == "creative_explore"
    assert row.creative_strategy_ids == ["conservative_answer", "divergent_ideas"]
    assert row.creative_selected_strategy == "divergent_ideas"
    assert row.creative_candidate_count == 2
    assert row.creative_handoff == "verifier"
    assert row.creative_model_summary_used is False


def test_write_eval_artifact_preserves_creative_metadata_fields(tmp_path) -> None:
    cases = [
        EvalCase(
            case_id="creative_brainstorm",
            prompt="Brainstorm ideas for memory routing.",
            expected_keywords=["ideas", "memory", "options"],
            requires_cognition_gain=True,
        )
    ]

    summary = run_eval(cases, backend=ContextAwareEvalBackend())
    artifact_path = write_eval_artifact(summary, str(tmp_path / "creative_eval.json"))

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert set(row) == {
        "case_id",
        "baseline_response",
        "cognition_response",
        "baseline_score",
        "cognition_score",
        "cognition_decision",
        "creative_strategy_ids",
        "creative_selected_strategy",
        "creative_candidate_count",
        "creative_handoff",
        "creative_model_summary_used",
    }
    assert row["creative_strategy_ids"] == ["conservative_answer", "divergent_ideas"]
    assert row["creative_selected_strategy"] == "divergent_ideas"
    assert row["creative_candidate_count"] == 2
    assert row["creative_handoff"] == "verifier"
    assert row["creative_model_summary_used"] is False


def test_run_local_delib_ablation_eval_covers_all_variants() -> None:
    summary = run_local_delib_ablation_eval(DEFAULT_LOCAL_DELIB_CASES, backend=LocalDelibContextEvalBackend())

    expected_rows = len(DEFAULT_LOCAL_DELIB_CASES) * len(DEFAULT_LOCAL_DELIB_VARIANTS)
    assert len(summary.rows) == expected_rows
    assert summary.runtime_variant_overrides_applied is True
    assert summary.runtime_variant_override_statuses["local_delib_branch"] == "exact"
    assert set(summary.variant_mean_scores) == {variant.variant_id for variant in DEFAULT_LOCAL_DELIB_VARIANTS}

    rows_by_variant = {row.variant_id: row for row in summary.rows if row.case_id == DEFAULT_LOCAL_DELIB_CASES[0].case_id}
    assert rows_by_variant["local_delib_branch"].model_local_delib_branch["branch_factor_used"] == 2.0
    assert rows_by_variant["local_delib_hierarchy"].model_local_delib_hierarchy["hierarchy_levels_used"] == 2.0
    assert rows_by_variant["local_delib_scratchpad"].model_local_delib_scratchpad["scratch_slots_used"] == 2.0
    assert rows_by_variant["local_delib_adaptive_halt"].model_local_delib_adaptive_halt["halted_token_fraction"] == 0.5
    assert rows_by_variant["local_delib_branch"].model_local_delib_graph_artifact["branch"]["summary"]["branch_factor_used"] == 2.0
    assert rows_by_variant["local_delib_adaptive_halt"].model_local_delib_graph_artifact["compute"]["summary"]["halted_token_fraction"] == 0.5


def test_write_local_delib_eval_artifact_writes_advanced_stats(tmp_path) -> None:
    summary = run_local_delib_ablation_eval(DEFAULT_LOCAL_DELIB_CASES, backend=LocalDelibContextEvalBackend())

    artifact_path = write_local_delib_eval_artifact(summary, str(tmp_path / "local_delib_eval.json"))

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "variant_mean_scores",
        "runtime_variant_overrides_applied",
        "runtime_variant_override_statuses",
        "runtime_variant_override_counts",
        "rows",
    }
    assert set(payload["variant_mean_scores"]) == {variant.variant_id for variant in DEFAULT_LOCAL_DELIB_VARIANTS}
    assert payload["runtime_variant_overrides_applied"] is True
    assert payload["runtime_variant_override_statuses"]["local_delib_off"] == "exact"
    assert payload["rows"]
    first_row = payload["rows"][0]
    assert "runtime_override_status" in first_row
    assert "model_local_delib_branch" in first_row
    assert "model_local_delib_hierarchy" in first_row
    assert "model_local_delib_scratchpad" in first_row
    assert "model_local_delib_adaptive_halt" in first_row
    assert "model_local_delib_graph_artifact" in first_row


def test_run_advanced_local_delib_ablation_eval_covers_prompt10_variants() -> None:
    summary = run_advanced_local_delib_ablation_eval(
        ADVANCED_LOCAL_DELIB_CASES,
        backend=LocalDelibContextEvalBackend(),
    )

    expected_rows = len(ADVANCED_LOCAL_DELIB_CASES) * len(ADVANCED_LOCAL_DELIB_VARIANTS)
    assert len(summary.rows) == expected_rows
    assert summary.runtime_variant_overrides_applied is True
    assert set(summary.variant_mean_scores) == {variant.variant_id for variant in ADVANCED_LOCAL_DELIB_VARIANTS}
    assert set(summary.quality_per_compute) == {variant.variant_id for variant in ADVANCED_LOCAL_DELIB_VARIANTS}

    rows_by_variant = {row.variant_id: row for row in summary.rows if row.case_id == ADVANCED_LOCAL_DELIB_CASES[1].case_id}
    assert rows_by_variant["local_delib_neighbor_graph"].neighbor_graph_stats["mean_neighbor_count"] == 4.0
    assert rows_by_variant["local_delib_flocking"].flocking_stats["fraction_flocking_tokens_active"] == 0.75

    branch_case_rows = {row.variant_id: row for row in summary.rows if row.case_id == "branch_consensus"}
    assert branch_case_rows["local_delib_branch_consensus_verifier"].branch_stats["branch_consensus_used"] == 1.0
    assert branch_case_rows["local_delib_branch_consensus_verifier"].branch_stats["mean_branch_verifier_score"] == 0.68

    hierarchy_case_rows = {row.variant_id: row for row in summary.rows if row.case_id == "deep_hierarchy"}
    assert hierarchy_case_rows["local_delib_deep_hierarchy"].hierarchy_stats["hierarchy_depth_used"] == 3.0

    scratch_case_rows = {row.variant_id: row for row in summary.rows if row.case_id == "scratch_refine"}
    assert scratch_case_rows["local_delib_scratch_refine"].scratch_stats["mean_scratch_refine_norm"] == 0.42

    thought_case_rows = {row.variant_id: row for row in summary.rows if row.case_id == "thought_anchor"}
    assert thought_case_rows["local_delib_thought_graph"].thought_graph_stats["thought_nodes_used"] == 6.0
    assert thought_case_rows["local_delib_global_anchors"].anchor_stats["global_anchors_used"] == 3.0
    assert "thought_graph" in thought_case_rows["local_delib_combo_full_stack"].model_local_delib_graph_artifact
    assert "anchors" in thought_case_rows["local_delib_combo_full_stack"].model_local_delib_graph_artifact
    assert "flocking" in thought_case_rows["local_delib_combo_full_stack"].model_local_delib_graph_artifact
    assert thought_case_rows["local_delib_combo_full_stack"].quality_per_compute > 0.0


def test_advanced_ablation_eval_tracks_combo_compute_pressure_and_graph_participation() -> None:
    summary = run_advanced_local_delib_ablation_eval(
        [next(case for case in ADVANCED_LOCAL_DELIB_CASES if case.case_id == "thought_anchor")],
        backend=LocalDelibContextEvalBackend(),
        variants=[
            next(variant for variant in ADVANCED_LOCAL_DELIB_VARIANTS if variant.variant_id == "local_delib_off"),
            next(variant for variant in ADVANCED_LOCAL_DELIB_VARIANTS if variant.variant_id == "local_delib_basic"),
            next(variant for variant in ADVANCED_LOCAL_DELIB_VARIANTS if variant.variant_id == "local_delib_adaptive_halt"),
            next(variant for variant in ADVANCED_LOCAL_DELIB_VARIANTS if variant.variant_id == "local_delib_flocking"),
            next(variant for variant in ADVANCED_LOCAL_DELIB_VARIANTS if variant.variant_id == "local_delib_combo_full_stack"),
        ],
    )

    rows = {row.variant_id: row for row in summary.rows}
    assert rows["local_delib_off"].model_local_delib_graph_artifact == {}
    assert rows["local_delib_flocking"].model_local_delib_graph_artifact["overview"]["active_sections"] == [
        "compute",
        "flocking",
    ]
    assert rows["local_delib_combo_full_stack"].model_local_delib_graph_artifact["overview"]["active_sections"] == [
        "branch",
        "thought_graph",
        "hierarchy",
        "scratch",
        "anchors",
        "compute",
        "flocking",
    ]
    assert summary.mean_steps_taken["local_delib_adaptive_halt"] < summary.mean_steps_taken["local_delib_basic"]
    assert summary.mean_steps_taken["local_delib_combo_full_stack"] == summary.mean_steps_taken["local_delib_adaptive_halt"]
    assert (
        summary.compute_proxy_metrics["local_delib_combo_full_stack"]["estimated_compute_cost"]
        > summary.compute_proxy_metrics["local_delib_adaptive_halt"]["estimated_compute_cost"]
    )
    assert (
        summary.compute_proxy_metrics["local_delib_combo_full_stack"]["estimated_compute_cost"]
        > summary.compute_proxy_metrics["local_delib_flocking"]["estimated_compute_cost"]
    )


def test_write_advanced_local_delib_eval_artifact_writes_prompt10_fields(tmp_path) -> None:
    summary = run_advanced_local_delib_ablation_eval(
        ADVANCED_LOCAL_DELIB_CASES,
        backend=LocalDelibContextEvalBackend(),
    )

    artifact_path = write_advanced_local_delib_eval_artifact(
        summary,
        str(tmp_path / "local_delib_advanced_eval.json"),
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "quality_proxy_scores",
        "variant_mean_scores",
        "quality_per_compute",
        "compute_proxy_metrics",
        "mean_steps_taken",
        "neighbor_graph_stats",
        "branch_stats",
        "hierarchy_stats",
        "scratch_stats",
        "thought_graph_stats",
        "flocking_stats",
        "anchor_stats",
        "runtime_variant_overrides_applied",
        "runtime_variant_override_statuses",
        "runtime_variant_override_counts",
        "rows",
    }
    assert set(payload["variant_mean_scores"]) == {variant.variant_id for variant in ADVANCED_LOCAL_DELIB_VARIANTS}
    assert set(payload["quality_proxy_scores"]) == {variant.variant_id for variant in ADVANCED_LOCAL_DELIB_VARIANTS}
    assert set(payload["quality_per_compute"]) == {variant.variant_id for variant in ADVANCED_LOCAL_DELIB_VARIANTS}
    assert set(payload["compute_proxy_metrics"]) == {variant.variant_id for variant in ADVANCED_LOCAL_DELIB_VARIANTS}
    assert "neighbor_graph_stats" in payload
    assert "branch_stats" in payload
    assert "hierarchy_stats" in payload
    assert "scratch_stats" in payload
    assert "thought_graph_stats" in payload
    assert "flocking_stats" in payload
    assert "anchor_stats" in payload
    assert payload["runtime_variant_overrides_applied"] is True
    assert payload["runtime_variant_override_statuses"]["local_delib_off"] == "exact"
    assert payload["rows"]
    assert "model_local_delib_graph_artifact" in payload["rows"][0]
    assert "runtime_override_status" in payload["rows"][0]


def test_run_research_local_delib_eval_produces_prompt4_metrics_and_activation_checks() -> None:
    summary = run_research_local_delib_eval(
        RESEARCH_LOCAL_DELIB_CASES,
        backend=LocalDelibContextEvalBackend(),
    )

    expected_rows = len(RESEARCH_LOCAL_DELIB_CASES) * len(RESEARCH_LOCAL_DELIB_VARIANTS)
    assert len(summary.rows) == expected_rows
    assert summary.backend_kind == "demo"
    assert summary.metric_tier == "deterministic_structured"
    assert summary.runtime_variant_overrides_applied is True
    assert summary.baseline_variant_id == "local_delib_off"
    assert summary.variant_pass_rates["local_delib_off"] < summary.variant_pass_rates["local_delib_combo_full_stack"]
    assert summary.delta_vs_baseline["local_delib_combo_full_stack"] > 0.0
    assert summary.activation_coverage["local_delib_branch_consensus_verifier"]["activation_ok_rate"] == 1.0

    branch_rows = {row.variant_id: row for row in summary.rows if row.case_id == "branch_consensus"}
    assert branch_rows["local_delib_branch_consensus_verifier"].passed is True
    assert branch_rows["local_delib_branch_consensus_verifier"].task_metrics["winner_exact"] == 1.0
    assert branch_rows["local_delib_branch_consensus_verifier"].activation_checks["branch_consensus"] is True
    assert branch_rows["local_delib_off"].passed is False

    hierarchy_rows = {row.variant_id: row for row in summary.rows if row.case_id == "deep_hierarchy"}
    assert hierarchy_rows["local_delib_deep_hierarchy"].passed is True
    assert hierarchy_rows["local_delib_deep_hierarchy"].activation_checks["hierarchy"] is True

    thought_rows = {row.variant_id: row for row in summary.rows if row.case_id == "thought_graph"}
    assert thought_rows["local_delib_thought_graph"].passed is True
    assert thought_rows["local_delib_thought_graph"].activation_checks["thought_graph"] is True
    assert thought_rows["local_delib_thought_graph"].quality_per_compute > 0.0

    anchor_rows = {row.variant_id: row for row in summary.rows if row.case_id == "anchor_long_context"}
    assert anchor_rows["local_delib_global_anchors"].passed is True
    assert anchor_rows["local_delib_global_anchors"].activation_checks["anchors"] is True
    assert anchor_rows["local_delib_global_anchors"].compute_accounting["estimated_compute_cost"] > 0.0


def test_research_eval_tracks_active_mechanism_count_for_combo_variants() -> None:
    summary = run_research_local_delib_eval(
        [next(case for case in RESEARCH_LOCAL_DELIB_CASES if case.case_id == "thought_graph")],
        backend=LocalDelibContextEvalBackend(),
        variants=[
            next(variant for variant in RESEARCH_LOCAL_DELIB_VARIANTS if variant.variant_id == "local_delib_off"),
            next(variant for variant in RESEARCH_LOCAL_DELIB_VARIANTS if variant.variant_id == "local_delib_basic"),
            next(variant for variant in RESEARCH_LOCAL_DELIB_VARIANTS if variant.variant_id == "local_delib_adaptive_halt"),
            next(
                variant
                for variant in RESEARCH_LOCAL_DELIB_VARIANTS
                if variant.variant_id == "local_delib_branch_consensus_verifier"
            ),
            next(variant for variant in RESEARCH_LOCAL_DELIB_VARIANTS if variant.variant_id == "local_delib_combo_full_stack"),
        ],
    )

    rows = {row.variant_id: row for row in summary.rows}
    assert rows["local_delib_off"].active_mechanisms == []
    assert rows["local_delib_off"].compute_accounting["active_mechanism_count"] == 0.0
    assert rows["local_delib_basic"].active_mechanisms == ["local_delib"]
    assert rows["local_delib_basic"].compute_accounting["active_mechanism_count"] == 1.0
    assert rows["local_delib_adaptive_halt"].active_mechanisms == ["adaptive_halt", "local_delib"]
    assert rows["local_delib_adaptive_halt"].compute_accounting["active_mechanism_count"] == 2.0
    assert set(rows["local_delib_branch_consensus_verifier"].active_mechanisms) == {
        "branch_consensus",
        "branching",
        "local_delib",
    }

    combo_row = rows["local_delib_combo_full_stack"]
    assert combo_row.activation_ok is True
    assert combo_row.metrics_interpretable is True
    assert combo_row.active_mechanisms == sorted(combo_row.expected_activations)
    assert combo_row.compute_accounting["active_mechanism_count"] == float(len(combo_row.active_mechanisms))
    assert combo_row.model_local_delib_graph_artifact["overview"]["active_sections"] == [
        "branch",
        "thought_graph",
        "hierarchy",
        "scratch",
        "anchors",
        "compute",
        "flocking",
    ]
    assert (
        combo_row.compute_accounting["active_mechanism_count"]
        > rows["local_delib_adaptive_halt"].compute_accounting["active_mechanism_count"]
    )
    assert (
        combo_row.compute_accounting["estimated_compute_cost"]
        > rows["local_delib_branch_consensus_verifier"].compute_accounting["estimated_compute_cost"]
    )
    assert (
        summary.compute_accounting["local_delib_combo_full_stack"]["active_mechanism_count"]
        == float(len(combo_row.active_mechanisms))
    )
    assert summary.activation_coverage["local_delib_combo_full_stack"]["expected_mechanisms"] == combo_row.expected_activations


def test_write_research_local_delib_eval_artifact_writes_prompt4_fields(tmp_path) -> None:
    summary = run_research_local_delib_eval(
        RESEARCH_LOCAL_DELIB_CASES,
        backend=LocalDelibContextEvalBackend(),
    )

    artifact_path = write_research_local_delib_eval_artifact(
        summary,
        str(tmp_path / "local_delib_research_eval.json"),
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "backend_kind",
        "metric_tier",
        "baseline_variant_id",
        "variant_mean_scores",
        "variant_pass_rates",
        "delta_vs_baseline",
        "case_scores",
        "case_deltas_vs_baseline",
        "task_family_scores",
        "quality_per_compute",
        "compute_accounting",
        "activation_coverage",
        "runtime_variant_overrides_applied",
        "runtime_variant_override_statuses",
        "runtime_variant_override_counts",
        "rows",
    }
    assert payload["backend_kind"] == "demo"
    assert payload["metric_tier"] == "deterministic_structured"
    assert payload["baseline_variant_id"] == "local_delib_off"
    assert set(payload["variant_mean_scores"]) == {variant.variant_id for variant in RESEARCH_LOCAL_DELIB_VARIANTS}
    assert set(payload["variant_pass_rates"]) == {variant.variant_id for variant in RESEARCH_LOCAL_DELIB_VARIANTS}
    assert "activation_coverage" in payload
    assert "case_scores" in payload
    assert "case_deltas_vs_baseline" in payload
    assert "task_family_scores" in payload
    assert "compute_accounting" in payload
    assert payload["runtime_variant_override_statuses"]["local_delib_off"] == "exact"
    assert payload["rows"]
    first_row = payload["rows"][0]
    assert "activation_checks" in first_row
    assert "task_metrics" in first_row
    assert "compute_accounting" in first_row
    assert "metric_tier" in first_row
    assert "runtime_override_status" in first_row


def test_run_advanced_local_delib_ablation_eval_reports_unsupported_runtime_overrides() -> None:
    summary = run_advanced_local_delib_ablation_eval(
        ADVANCED_LOCAL_DELIB_CASES[:1],
        backend=UnsupportedRuntimeOverrideBackend(),
        variants=ADVANCED_LOCAL_DELIB_VARIANTS[:2],
    )

    assert summary.runtime_variant_overrides_applied is False
    assert summary.runtime_variant_override_counts["unsupported"] == len(summary.rows)
    assert summary.runtime_variant_override_statuses == {
        "local_delib_off": "unsupported",
        "local_delib_basic": "unsupported",
    }
    assert all(row.runtime_override_status == "unsupported" for row in summary.rows)
    assert all(row.response == "" for row in summary.rows)


def test_local_delib_eval_aggregates_variant_override_statuses_conservatively() -> None:
    summary = run_local_delib_ablation_eval(
        DEFAULT_LOCAL_DELIB_CASES[:2],
        backend=MixedRuntimeOverrideBackend(),
        variants=[LocalDelibVariant("flaky_variant", {"local_delib": True, "local_delib_steps": 1})],
    )

    assert [row.runtime_override_status for row in summary.rows] == ["unsupported", "exact"]
    assert summary.runtime_variant_overrides_applied is False
    assert summary.runtime_variant_override_statuses == {"flaky_variant": "unsupported"}
    assert summary.runtime_variant_override_counts == {"unsupported": 1, "exact": 1}


def test_run_research_local_delib_eval_can_fail_on_unsupported_runtime_overrides() -> None:
    with pytest.raises(LocalDelibRuntimeOverrideError):
        run_research_local_delib_eval(
            RESEARCH_LOCAL_DELIB_CASES[:1],
            backend=UnsupportedRuntimeOverrideBackend(),
            variants=RESEARCH_LOCAL_DELIB_VARIANTS[:1],
            fail_on_unsupported_runtime_overrides=True,
        )


def test_run_research_local_delib_eval_marks_approximated_rows_as_non_exact() -> None:
    summary = run_research_local_delib_eval(
        RESEARCH_LOCAL_DELIB_CASES[:1],
        backend=ApproximateRuntimeOverrideBackend(),
        variants=[LocalDelibVariant("approx_variant", {"local_delib": True, "local_delib_steps": 1})],
    )

    assert [row.runtime_override_status for row in summary.rows] == ["approximated"]
    assert [row.runtime_override_applied for row in summary.rows] == [False]
    assert summary.runtime_variant_overrides_applied is False
    assert summary.runtime_variant_override_statuses == {"approx_variant": "approximated"}
    assert summary.runtime_variant_override_counts == {"approximated": 1}
    assert summary.rows[0].runtime_override_application_method == "loaded_checkpoint_fallback"


def test_run_task_grounded_eval_writes_task_native_artifact(tmp_path) -> None:
    summary = run_task_grounded_eval(
        backend=TaskGroundedBackend(),
        tasks={"TinyTask": TinyTask()},
        checkpoint_identity={"source": "sft", "model_tag": "d1", "step": 7, "device_type": "cpu"},
    )

    assert summary.backend_kind == "external"
    assert summary.metric_tier == "task_grounded"
    assert summary.task_names == ["TinyTask"]
    assert len(summary.rows) == 2
    assert summary.per_task["TinyTask"]["count"] == 2.0
    assert summary.rows[0].benchmark_eligible is True
    assert summary.rows[0].baseline_runtime_override_status == "not_requested"
    assert summary.rows[0].cognition_runtime_override_status == "not_requested"

    artifact_path = write_task_grounded_eval_artifact(summary, str(tmp_path / "task_grounded.json"))
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["metric_tier"] == "task_grounded"
    assert payload["checkpoint_identity"]["source"] == "sft"
    assert payload["task_names"] == ["TinyTask"]
    assert set(payload["per_task"]["TinyTask"]) == {
        "count",
        "baseline_pass_rate",
        "cognition_pass_rate",
        "delta",
        "proof_count",
        "proof_baseline_pass_rate",
        "proof_cognition_pass_rate",
        "proof_delta",
    }
    assert len(payload["rows"]) == 2


def test_run_task_grounded_eval_supports_builtin_smoke_task() -> None:
    summary = run_task_grounded_eval(
        backend=TaskGroundedBackend(),
        task_names=["SmokeTinyTask"],
        max_problems=2,
        checkpoint_identity={"source": "sft", "model_tag": "d1", "step": 7, "device_type": "cpu"},
    )

    assert summary.task_names == ["SmokeTinyTask"]
    assert len(summary.rows) == 2
    assert summary.per_task["SmokeTinyTask"]["count"] == 2.0


def test_run_natural_local_delib_eval_writes_proof_filtered_fields(tmp_path) -> None:
    summary = run_natural_local_delib_eval(
        NATURAL_LOCAL_DELIB_CASES,
        backend=LocalDelibContextEvalBackend(),
        variants=NATURAL_LOCAL_DELIB_VARIANTS,
        checkpoint_identity={"source": "demo"},
    )

    assert summary.backend_kind == "demo"
    assert summary.metric_tier == "natural_task_grounded"
    assert summary.variant_pass_rates["local_delib_off"] < summary.variant_pass_rates["local_delib_combo_full_stack"]
    assert summary.proof_pass_rates["local_delib_combo_full_stack"] > 0.0
    assert summary.proof_delta_vs_baseline["local_delib_combo_full_stack"] > 0.0
    thought_rows = {row.variant_id: row for row in summary.rows if row.case_id == "thought_graph_natural"}
    assert thought_rows["local_delib_thought_graph"].passed is True
    assert thought_rows["local_delib_thought_graph"].proof_eligible is True
    assert thought_rows["local_delib_off"].passed is False

    artifact_path = write_natural_local_delib_eval_artifact(summary, str(tmp_path / "local_delib_natural.json"))
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["metric_tier"] == "natural_task_grounded"
    assert payload["checkpoint_identity"]["source"] == "demo"
    assert "proof_variant_mean_scores" in payload
    assert "proof_pass_rates" in payload
    assert "proof_delta_vs_baseline" in payload
    assert "grader_extractable_rate" in payload["activation_coverage"]["local_delib_combo_full_stack"]


def test_write_engine_smoke_manifest_writes_strict_audit_payload(tmp_path) -> None:
    manifest = EngineSmokeManifest(
        status="passed",
        strict_audit=True,
        checkpoint_identity={"source": "sft", "model_tag": "d1", "step": 7, "device_type": "cpu"},
        commands=[
            EngineSmokeCommandRecord(
                label="task-grounded",
                argv=["python", "-m", "scripts.cognition_eval", "--suite", "task-grounded"],
                artifact_path="artifacts/pretraining_proofs/engine/task_grounded.json",
                exit_code=0,
            )
        ],
        artifacts=[
            EngineSmokeArtifactRecord(
                label="task-grounded",
                path="artifacts/pretraining_proofs/engine/task_grounded.json",
                row_count=3,
                runtime_override_statuses=[],
                runtime_override_counts={},
            )
        ],
        observed_runtime_override_statuses=["exact", "unsupported"],
    )

    artifact_path = write_engine_smoke_manifest(manifest, str(tmp_path / "engine_smoke_manifest.json"))
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["strict_audit"] is True
    assert payload["checkpoint_identity"]["source"] == "sft"
    assert payload["commands"][0]["label"] == "task-grounded"
    assert payload["artifacts"][0]["path"].endswith("task_grounded.json")
