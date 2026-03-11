from __future__ import annotations

import json

import pytest

from nanochat.cognition.eval import (
    ContextAwareEvalBackend,
    DEFAULT_CASES,
    DEFAULT_LOCAL_DELIB_CASES,
    DEFAULT_LOCAL_DELIB_VARIANTS,
    LocalDelibContextEvalBackend,
    run_eval,
    run_local_delib_ablation_eval,
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
