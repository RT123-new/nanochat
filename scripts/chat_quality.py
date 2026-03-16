#!/usr/bin/env python3
"""Automate chat-style quality inspection and small benchmark runs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

from nanochat.cognition.agent import CognitionAgent
from nanochat.cognition.backend import BackendAdapter, EngineBackend
from nanochat.cognition.eval import (
    TaskGroundedEvalSummary,
    run_task_grounded_eval,
    write_task_grounded_eval_artifact,
)

SOURCE_DIR_NAMES = {
    "base": "base_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}
DEFAULT_TASK_NAMES = ["GSM8K", "SpellingBee"]


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """Single chat-style prompt to compare baseline vs cognition outputs."""

    prompt_id: str
    prompt: str


@dataclass(slots=True)
class PromptComparisonRow:
    """Baseline vs cognition output for one prompt."""

    prompt_id: str
    prompt: str
    baseline_response: str
    cognition_response: str
    cognition_decision: str
    trace_steps: list[str]


@dataclass(slots=True)
class SnapshotMetadata:
    """Top-level metadata for a saved quality snapshot."""

    base_dir: str
    source: str
    model_tag: str
    step: int
    device_type: str
    output_dir: str
    temperature: float
    top_k: int | None
    max_tokens: int
    prompt_count: int
    task_names: list[str]
    task_grounded_max_problems: int


DEFAULT_PROMPTS = [
    PromptSpec(
        prompt_id="explain_gradient_clipping",
        prompt="Explain gradient clipping in plain English and give one short example.",
    ),
    PromptSpec(
        prompt_id="debug_training_nan",
        prompt="Give me a five-step plan to debug training loss suddenly becoming NaN.",
    ),
    PromptSpec(
        prompt_id="python_fibonacci",
        prompt="Write a short Python function that returns Fibonacci numbers up to n.",
    ),
    PromptSpec(
        prompt_id="brainstorm_memory_features",
        prompt="Brainstorm four practical memory features for a tiny chatbot.",
    ),
]


def _resolve_prompts(raw_prompts: list[str] | None) -> list[PromptSpec]:
    if not raw_prompts:
        return list(DEFAULT_PROMPTS)
    return [
        PromptSpec(prompt_id=f"user_prompt_{index + 1}", prompt=prompt)
        for index, prompt in enumerate(raw_prompts)
    ]


def _resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "chat_quality" / timestamp


def _resolve_base_dir(base_dir: str | None) -> Path:
    from nanochat.common import get_base_dir

    if base_dir:
        resolved = Path(base_dir).expanduser()
        os.environ["NANOCHAT_BASE_DIR"] = str(resolved)
        return resolved
    return Path(get_base_dir())


def _resolve_checkpoint_identity(
    *,
    base_dir: Path,
    source: str,
    model_tag: str | None,
    step: int | None,
) -> dict[str, Any]:
    source_dir = base_dir / SOURCE_DIR_NAMES[source]
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint source directory not found: {source_dir}. "
            "Set --base-dir or NANOCHAT_BASE_DIR to your checkpoint root."
        )

    resolved_model_tag = model_tag or _find_largest_model(source_dir)
    checkpoint_dir = source_dir / resolved_model_tag
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint model directory not found: {checkpoint_dir}")

    resolved_step = step if step is not None else _find_last_step(checkpoint_dir)
    model_path = checkpoint_dir / f"model_{resolved_step:06d}.pt"
    meta_path = checkpoint_dir / f"meta_{resolved_step:06d}.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Checkpoint files for step {resolved_step} are missing in {checkpoint_dir}"
        )

    return {
        "base_dir": str(base_dir),
        "source": source,
        "model_tag": resolved_model_tag,
        "step": resolved_step,
        "checkpoint_dir": str(checkpoint_dir),
    }


def run_prompt_comparisons(
    *,
    backend: EngineBackend,
    prompts: list[PromptSpec],
) -> list[PromptComparisonRow]:
    """Generate baseline and cognition responses for a small prompt set."""
    adapter = BackendAdapter(backend=backend)
    rows: list[PromptComparisonRow] = []

    for prompt_spec in prompts:
        baseline_response = adapter.run(prompt_spec.prompt)
        agent = CognitionAgent(backend=BackendAdapter(backend=backend))
        cognition_result = agent.run(prompt_spec.prompt)
        rows.append(
            PromptComparisonRow(
                prompt_id=prompt_spec.prompt_id,
                prompt=prompt_spec.prompt,
                baseline_response=baseline_response,
                cognition_response=cognition_result.response,
                cognition_decision=cognition_result.decision,
                trace_steps=list(cognition_result.trace.steps),
            )
        )

    return rows


def _find_largest_model(source_dir: Path) -> str:
    candidates = [path for path in source_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint model directories found in {source_dir}")

    depth_candidates: list[tuple[int, str]] = []
    for path in candidates:
        if path.name.startswith("d") and path.name[1:].isdigit():
            depth_candidates.append((int(path.name[1:]), path.name))
    if depth_candidates:
        depth_candidates.sort(reverse=True)
        return depth_candidates[0][1]

    return max(candidates, key=lambda path: path.stat().st_mtime).name


def _find_last_step(checkpoint_dir: Path) -> int:
    steps = []
    for model_path in glob(str(checkpoint_dir / "model_*.pt")):
        stem = Path(model_path).stem
        _, _, suffix = stem.partition("_")
        if suffix.isdigit():
            steps.append(int(suffix))
    if not steps:
        raise FileNotFoundError(f"No model_*.pt files found in {checkpoint_dir}")
    return max(steps)


def _task_summary_excerpt(summary: TaskGroundedEvalSummary) -> dict[str, Any]:
    return {
        "backend_kind": summary.backend_kind,
        "metric_tier": summary.metric_tier,
        "checkpoint_identity": dict(summary.checkpoint_identity),
        "task_names": list(summary.task_names),
        "baseline_mean": summary.baseline_mean,
        "cognition_mean": summary.cognition_mean,
        "delta": summary.delta,
        "proof_baseline_mean": summary.proof_baseline_mean,
        "proof_cognition_mean": summary.proof_cognition_mean,
        "proof_delta": summary.proof_delta,
        "per_task": dict(summary.per_task),
        "row_count": len(summary.rows),
    }


def build_snapshot_markdown(
    *,
    metadata: SnapshotMetadata,
    prompt_rows: list[PromptComparisonRow],
    task_summary: dict[str, Any] | None,
    task_artifact_path: str | None,
    task_error: str | None,
) -> str:
    """Render a compact human-readable report for one snapshot run."""
    lines = [
        "# Chat Quality Snapshot",
        "",
        "## Run metadata",
        f"- base_dir: `{metadata.base_dir}`",
        f"- source: `{metadata.source}`",
        f"- model_tag: `{metadata.model_tag}`",
        f"- step: `{metadata.step}`",
        f"- device_type: `{metadata.device_type}`",
        f"- temperature: `{metadata.temperature}`",
        f"- top_k: `{metadata.top_k}`",
        f"- max_tokens: `{metadata.max_tokens}`",
        f"- output_dir: `{metadata.output_dir}`",
        "",
        "## Prompt comparisons",
    ]

    for row in prompt_rows:
        lines.extend(
            [
                "",
                f"### {row.prompt_id}",
                "",
                "**Prompt**",
                "",
                "```text",
                row.prompt,
                "```",
                "",
                f"**Cognition route**: `{row.cognition_decision}`",
                f"**Trace steps**: `{', '.join(row.trace_steps)}`",
                "",
                "**Baseline**",
                "",
                "```text",
                row.baseline_response,
                "```",
                "",
                "**Cognition**",
                "",
                "```text",
                row.cognition_response,
                "```",
            ]
        )

    lines.extend(["", "## Task-grounded eval"])
    if task_summary is not None:
        lines.extend(
            [
                f"- tasks: `{', '.join(task_summary['task_names'])}`",
                f"- baseline_mean: `{task_summary['baseline_mean']:.3f}`",
                f"- cognition_mean: `{task_summary['cognition_mean']:.3f}`",
                f"- delta: `{task_summary['delta']:.3f}`",
                f"- proof_delta: `{task_summary['proof_delta']:.3f}`",
                f"- row_count: `{task_summary['row_count']}`",
            ]
        )
        if task_artifact_path:
            lines.append(f"- artifact: `{task_artifact_path}`")
        lines.extend(["", "### Per-task summary", "", "```json", json.dumps(task_summary["per_task"], indent=2), "```"])
    elif task_error is not None:
        lines.extend(
            [
                "- status: `failed`",
                f"- reason: `{task_error}`",
            ]
        )
    else:
        lines.extend(["- status: `skipped`"])

    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_text(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Automate local chat-quality inspection for a checkpoint.")
    parser.add_argument("-i", "--source", default="sft", choices=["base", "sft", "rl"], help="Checkpoint source")
    parser.add_argument("-g", "--model-tag", default=None, help="Checkpoint model tag to load")
    parser.add_argument("-s", "--step", type=int, default=None, help="Checkpoint step to load")
    parser.add_argument("--base-dir", default=None, help="Checkpoint root; overrides NANOCHAT_BASE_DIR")
    parser.add_argument(
        "--device-type",
        default="",
        choices=["", "cuda", "cpu", "mps"],
        help="Device type; empty means autodetect",
    )
    parser.add_argument("-t", "--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("-k", "--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("-m", "--max-tokens", type=int, default=192, help="Max generation tokens")
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to compare. Pass multiple times to replace the built-in prompt set.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASK_NAMES),
        help="Comma-separated task-grounded task names",
    )
    parser.add_argument(
        "--task-grounded-max-problems",
        type=int,
        default=3,
        help="How many examples per task to evaluate",
    )
    parser.add_argument(
        "--skip-task-grounded",
        action="store_true",
        help="Skip task-grounded evaluation and only save prompt comparisons",
    )
    parser.add_argument(
        "--strict-task-grounded",
        action="store_true",
        help="Fail the whole run if task-grounded evaluation errors",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: artifacts/chat_quality/<timestamp>",
    )
    args = parser.parse_args()

    prompts = _resolve_prompts(args.prompt)
    output_dir = _resolve_output_dir(args.output_dir)
    base_dir = _resolve_base_dir(args.base_dir)
    checkpoint_identity = _resolve_checkpoint_identity(
        base_dir=base_dir,
        source=args.source,
        model_tag=args.model_tag,
        step=args.step,
    )

    from nanochat.checkpoint_manager import load_model
    from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
    from nanochat.engine import Engine

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    cleanup_required = False
    try:
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
        cleanup_required = True
        model, tokenizer, meta = load_model(
            checkpoint_identity["source"],
            device,
            phase="eval",
            model_tag=checkpoint_identity["model_tag"],
            step=checkpoint_identity["step"],
        )
        engine = Engine(model, tokenizer)
        backend = EngineBackend(
            engine=engine,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        prompt_rows = run_prompt_comparisons(backend=backend, prompts=prompts)
        prompt_payload = [asdict(row) for row in prompt_rows]
        prompt_json_path = _write_json(output_dir / "chat_prompts.json", prompt_payload)

        task_summary: TaskGroundedEvalSummary | None = None
        task_summary_excerpt: dict[str, Any] | None = None
        task_artifact_path: Path | None = None
        task_error: str | None = None
        task_names = [task_name.strip() for task_name in args.tasks.split(",") if task_name.strip()]
        if not args.skip_task_grounded:
            try:
                task_summary = run_task_grounded_eval(
                    backend=backend,
                    task_names=task_names,
                    max_problems=args.task_grounded_max_problems,
                    checkpoint_identity=checkpoint_identity,
                )
                task_artifact_path = write_task_grounded_eval_artifact(
                    task_summary,
                    str(output_dir / "task_grounded.json"),
                )
                task_summary_excerpt = _task_summary_excerpt(task_summary)
            except Exception as exc:
                task_error = str(exc)
                if args.strict_task_grounded:
                    raise

        metadata = SnapshotMetadata(
            base_dir=str(base_dir),
            source=checkpoint_identity["source"],
            model_tag=str(checkpoint_identity["model_tag"]),
            step=int(checkpoint_identity["step"]),
            device_type=device_type,
            output_dir=str(output_dir),
            temperature=args.temperature,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            prompt_count=len(prompts),
            task_names=task_names,
            task_grounded_max_problems=args.task_grounded_max_problems,
        )
        snapshot_payload = {
            "metadata": asdict(metadata),
            "prompt_rows": prompt_payload,
            "task_grounded": task_summary_excerpt,
            "task_grounded_error": task_error,
            "artifacts": {
                "prompt_json": str(prompt_json_path),
                "task_grounded_json": str(task_artifact_path) if task_artifact_path else None,
            },
        }
        summary_json_path = _write_json(output_dir / "snapshot_summary.json", snapshot_payload)
        report_text = build_snapshot_markdown(
            metadata=metadata,
            prompt_rows=prompt_rows,
            task_summary=task_summary_excerpt,
            task_artifact_path=str(task_artifact_path) if task_artifact_path else None,
            task_error=task_error,
        )
        report_path = _write_text(output_dir / "REPORT.md", report_text)

        print("Chat quality snapshot complete")
        print(f"- checkpoint: {checkpoint_identity['source']}/{checkpoint_identity['model_tag']} step {checkpoint_identity['step']}")
        print(f"- device: {device_type}")
        print(f"- prompt artifact: {prompt_json_path}")
        print(f"- summary artifact: {summary_json_path}")
        print(f"- report: {report_path}")
        if task_artifact_path is not None:
            print(f"- task-grounded artifact: {task_artifact_path}")
        elif task_error is not None:
            print(f"- task-grounded eval failed: {task_error}")
        else:
            print("- task-grounded eval skipped")
    finally:
        if cleanup_required:
            compute_cleanup()


if __name__ == "__main__":
    main()
