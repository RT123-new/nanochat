from __future__ import annotations

import json

import pytest

from nanochat.cognition.eval import (
    ADVANCED_LOCAL_DELIB_CASES,
    ADVANCED_LOCAL_DELIB_VARIANTS,
    ContextAwareEvalBackend,
    DEFAULT_CASES,
    DEFAULT_LOCAL_DELIB_CASES,
    DEFAULT_LOCAL_DELIB_VARIANTS,
    LocalDelibContextEvalBackend,
    run_advanced_local_delib_ablation_eval,
    run_eval,
    run_local_delib_ablation_eval,
    write_advanced_local_delib_eval_artifact,
    write_eval_artifact,
    write_local_delib_eval_artifact,
)


class FakeEvalBackend:
    def generate(self, prompt: str, **kwargs: object) -> str:
        return "same response"


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


def test_write_eval_artifact_writes_json_payload(tmp_path) -> None:
    summary = run_eval(DEFAULT_CASES, backend=ContextAwareEvalBackend())

    artifact_path = write_eval_artifact(summary, str(tmp_path / "cognition_eval.json"))

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["baseline_mean"] == summary.baseline_mean
    assert payload["cognition_mean"] == summary.cognition_mean
    assert len(payload["rows"]) == len(DEFAULT_CASES)


def test_run_local_delib_ablation_eval_covers_all_variants() -> None:
    summary = run_local_delib_ablation_eval(DEFAULT_LOCAL_DELIB_CASES, backend=LocalDelibContextEvalBackend())

    expected_rows = len(DEFAULT_LOCAL_DELIB_CASES) * len(DEFAULT_LOCAL_DELIB_VARIANTS)
    assert len(summary.rows) == expected_rows
    assert set(summary.variant_mean_scores) == {variant.variant_id for variant in DEFAULT_LOCAL_DELIB_VARIANTS}

    rows_by_variant = {row.variant_id: row for row in summary.rows if row.case_id == DEFAULT_LOCAL_DELIB_CASES[0].case_id}
    assert rows_by_variant["local_delib_branch"].model_local_delib_branch["branch_factor_used"] == 2.0
    assert rows_by_variant["local_delib_hierarchy"].model_local_delib_hierarchy["hierarchy_levels_used"] == 2.0
    assert rows_by_variant["local_delib_scratchpad"].model_local_delib_scratchpad["scratch_slots_used"] == 2.0
    assert rows_by_variant["local_delib_adaptive_halt"].model_local_delib_adaptive_halt["halted_token_fraction"] == 0.5


def test_write_local_delib_eval_artifact_writes_advanced_stats(tmp_path) -> None:
    summary = run_local_delib_ablation_eval(DEFAULT_LOCAL_DELIB_CASES, backend=LocalDelibContextEvalBackend())

    artifact_path = write_local_delib_eval_artifact(summary, str(tmp_path / "local_delib_eval.json"))

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(payload["variant_mean_scores"]) == {variant.variant_id for variant in DEFAULT_LOCAL_DELIB_VARIANTS}
    assert payload["rows"]
    first_row = payload["rows"][0]
    assert "model_local_delib_branch" in first_row
    assert "model_local_delib_hierarchy" in first_row
    assert "model_local_delib_scratchpad" in first_row
    assert "model_local_delib_adaptive_halt" in first_row


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
    assert thought_case_rows["local_delib_combo_full_stack"].quality_per_compute > 0.0


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
    assert payload["rows"]
