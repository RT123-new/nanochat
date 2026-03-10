from __future__ import annotations

import json

import pytest

from nanochat.cognition.eval import (
    ContextAwareEvalBackend,
    DEFAULT_CASES,
    run_eval,
    write_eval_artifact,
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
