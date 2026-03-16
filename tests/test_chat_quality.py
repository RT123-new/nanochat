from __future__ import annotations

from scripts.chat_quality import (
    PromptComparisonRow,
    PromptSpec,
    SnapshotMetadata,
    _resolve_checkpoint_identity,
    build_snapshot_markdown,
    run_prompt_comparisons,
)


class FakeQualityBackend:
    def __init__(self) -> None:
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.last_generation_metadata = {}
        if "Creative strategy:" in prompt:
            return "memory ideas: retrieval, summary, profile, reflection"
        return f"baseline: {prompt.splitlines()[0]}"


def test_run_prompt_comparisons_collects_baseline_and_cognition_rows() -> None:
    rows = run_prompt_comparisons(
        backend=FakeQualityBackend(),
        prompts=[
            PromptSpec("direct", "Explain gradient clipping simply."),
            PromptSpec("creative", "Brainstorm four practical memory features for a tiny chatbot."),
        ],
    )

    assert [row.prompt_id for row in rows] == ["direct", "creative"]
    assert rows[0].baseline_response.startswith("baseline:")
    assert rows[0].cognition_response
    assert rows[0].trace_steps[0].startswith("route:")
    assert rows[1].cognition_decision == "creative_explore"
    assert "memory ideas" in rows[1].cognition_response


def test_resolve_checkpoint_identity_prefers_largest_tag_and_latest_step(tmp_path) -> None:
    base_dir = tmp_path / "nanochat"
    d1 = base_dir / "chatsft_checkpoints" / "d1"
    d3 = base_dir / "chatsft_checkpoints" / "d3"
    d1.mkdir(parents=True)
    d3.mkdir(parents=True)
    (d1 / "model_000001.pt").write_text("", encoding="utf-8")
    (d1 / "meta_000001.json").write_text("{}", encoding="utf-8")
    (d3 / "model_000002.pt").write_text("", encoding="utf-8")
    (d3 / "meta_000002.json").write_text("{}", encoding="utf-8")

    identity = _resolve_checkpoint_identity(
        base_dir=base_dir,
        source="sft",
        model_tag=None,
        step=None,
    )

    assert identity["model_tag"] == "d3"
    assert identity["step"] == 2
    assert identity["checkpoint_dir"].endswith("chatsft_checkpoints/d3")


def test_build_snapshot_markdown_includes_prompt_outputs_and_task_summary() -> None:
    metadata = SnapshotMetadata(
        base_dir="/tmp/nanochat",
        source="sft",
        model_tag="d3",
        step=10,
        device_type="mps",
        output_dir="/tmp/out",
        temperature=0.0,
        top_k=50,
        max_tokens=192,
        prompt_count=1,
        task_names=["GSM8K", "SpellingBee"],
        task_grounded_max_problems=3,
    )
    prompt_rows = [
        PromptComparisonRow(
            prompt_id="direct",
            prompt="Explain gradient clipping simply.",
            baseline_response="It stops huge updates.",
            cognition_response="It stops huge updates by capping the gradient norm.",
            cognition_decision="direct_answer",
            trace_steps=["route:direct_answer"],
        )
    ]
    task_summary = {
        "task_names": ["GSM8K", "SpellingBee"],
        "baseline_mean": 0.25,
        "cognition_mean": 0.50,
        "delta": 0.25,
        "proof_delta": 0.25,
        "row_count": 6,
        "per_task": {
            "GSM8K": {"baseline_pass_rate": 0.0, "cognition_pass_rate": 0.5},
            "SpellingBee": {"baseline_pass_rate": 0.5, "cognition_pass_rate": 0.5},
        },
    }

    report = build_snapshot_markdown(
        metadata=metadata,
        prompt_rows=prompt_rows,
        task_summary=task_summary,
        task_artifact_path="/tmp/out/task_grounded.json",
        task_error=None,
    )

    assert "# Chat Quality Snapshot" in report
    assert "Explain gradient clipping simply." in report
    assert "capping the gradient norm" in report
    assert "`GSM8K, SpellingBee`" in report
    assert "/tmp/out/task_grounded.json" in report


def test_build_snapshot_markdown_records_task_failure() -> None:
    metadata = SnapshotMetadata(
        base_dir="/tmp/nanochat",
        source="sft",
        model_tag="d3",
        step=10,
        device_type="mps",
        output_dir="/tmp/out",
        temperature=0.0,
        top_k=50,
        max_tokens=192,
        prompt_count=0,
        task_names=["GSM8K"],
        task_grounded_max_problems=3,
    )

    report = build_snapshot_markdown(
        metadata=metadata,
        prompt_rows=[],
        task_summary=None,
        task_artifact_path=None,
        task_error="dataset download failed",
    )

    assert "## Task-grounded eval" in report
    assert "`failed`" in report
    assert "dataset download failed" in report
