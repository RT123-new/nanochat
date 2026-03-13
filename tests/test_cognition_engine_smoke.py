from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from nanochat.checkpoint_manager import load_model
from nanochat.cognition.backend import EngineBackend
from nanochat.cognition.eval import (
    ADVANCED_LOCAL_DELIB_CASES,
    EngineSmokeArtifactRecord,
    EngineSmokeCommandRecord,
    EngineSmokeManifest,
    LocalDelibVariant,
    run_advanced_local_delib_ablation_eval,
    write_advanced_local_delib_eval_artifact,
    write_engine_smoke_manifest,
)
from nanochat.engine import Engine

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR_NAMES = {
    "base": "base_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}
ROW_STATUS_VALUES = {"exact", "approximated", "unsupported"}
ENGINE_SMOKE_MANIFEST_NAME = "engine_smoke_manifest.json"


@dataclass(frozen=True)
class SmokeCheckpoint:
    base_dir: Path
    source: str
    model_tag: str
    step: int
    device_type: str
    artifact_dir: Path


def _default_artifact_dir() -> Path:
    return Path(
        os.environ.get(
            "NANOCHAT_SMOKE_ARTIFACT_DIR",
            str(REPO_ROOT / "artifacts" / "pretraining_proofs" / "engine"),
        )
    ).expanduser()


def _write_smoke_manifest(
    *,
    status: str,
    smoke: SmokeCheckpoint | None = None,
    commands: list[EngineSmokeCommandRecord] | None = None,
    artifacts: list[EngineSmokeArtifactRecord] | None = None,
    observed_statuses: set[str] | None = None,
    reason: str | None = None,
) -> Path:
    artifact_dir = smoke.artifact_dir if smoke is not None else _default_artifact_dir()
    checkpoint_identity = (
        {
            "base_dir": str(smoke.base_dir),
            "source": smoke.source,
            "model_tag": smoke.model_tag,
            "step": smoke.step,
            "device_type": smoke.device_type,
        }
        if smoke is not None
        else {}
    )
    manifest = EngineSmokeManifest(
        status=status,
        strict_audit=True,
        checkpoint_identity=checkpoint_identity,
        commands=list(commands or []),
        artifacts=list(artifacts or []),
        observed_runtime_override_statuses=sorted(observed_statuses or set()),
        reason=reason,
    )
    return write_engine_smoke_manifest(manifest, str(artifact_dir / ENGINE_SMOKE_MANIFEST_NAME))


def _skip_smoke(reason: str) -> None:
    _write_smoke_manifest(status="skipped", reason=reason)
    pytest.skip(reason)


def _default_base_dir() -> Path:
    env_value = os.environ.get("NANOCHAT_BASE_DIR")
    if env_value:
        return Path(env_value).expanduser()
    return Path.home() / ".cache" / "nanochat"


def _resolve_sources() -> list[str]:
    override = os.environ.get("NANOCHAT_SMOKE_SOURCE")
    if override:
        if override not in SOURCE_DIR_NAMES:
            _skip_smoke(f"NANOCHAT_SMOKE_SOURCE must be one of {sorted(SOURCE_DIR_NAMES)}, got {override!r}")
        return [override]
    return ["sft", "base", "rl"]


def _select_model_tag(source_dir: Path) -> str:
    override = os.environ.get("NANOCHAT_SMOKE_MODEL_TAG")
    if override:
        if not (source_dir / override).is_dir():
            _skip_smoke(f"checkpoint model tag {override!r} was requested but not found in {source_dir}")
        return override

    candidates = [path for path in source_dir.iterdir() if path.is_dir()]
    if not candidates:
        _skip_smoke(f"no checkpoint model directories found in {source_dir}")

    depth_matches: list[tuple[int, str]] = []
    for path in candidates:
        match = re.fullmatch(r"d(\d+)", path.name)
        if match:
            depth_matches.append((int(match.group(1)), path.name))
    if depth_matches:
        depth_matches.sort(reverse=True)
        return depth_matches[0][1]

    return max(candidates, key=lambda path: path.stat().st_mtime).name


def _select_step(model_dir: Path) -> int:
    override = os.environ.get("NANOCHAT_SMOKE_STEP")
    if override:
        try:
            step = int(override)
        except ValueError:
            _skip_smoke(f"NANOCHAT_SMOKE_STEP must be an integer, got {override!r}")
        expected_model = model_dir / f"model_{step:06d}.pt"
        expected_meta = model_dir / f"meta_{step:06d}.json"
        if not expected_model.exists() or not expected_meta.exists():
            _skip_smoke(f"checkpoint step {step} was requested but files are missing in {model_dir}")
        return step

    steps = []
    for path in model_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        _skip_smoke(f"no model_*.pt files found in {model_dir}")
    return max(steps)


def _resolve_smoke_checkpoint() -> SmokeCheckpoint:
    base_dir = _default_base_dir()
    if not base_dir.exists():
        _skip_smoke(
            "optional engine smoke requires a checkpoint root; "
            f"{base_dir} does not exist. Set NANOCHAT_BASE_DIR to a local checkpoint root."
        )

    tokenizer_path = base_dir / "tokenizer" / "tokenizer.pkl"
    if not tokenizer_path.exists():
        _skip_smoke(
            "optional engine smoke requires a local tokenizer at "
            f"{tokenizer_path}. Set NANOCHAT_BASE_DIR to a prepared nanochat checkpoint root."
        )

    for source in _resolve_sources():
        source_dir = base_dir / SOURCE_DIR_NAMES[source]
        if not source_dir.is_dir():
            continue
        candidates = [path for path in source_dir.iterdir() if path.is_dir()]
        if not candidates:
            continue
        model_tag = _select_model_tag(source_dir)
        model_dir = source_dir / model_tag
        try:
            step = _select_step(model_dir)
        except pytest.skip.Exception:
            continue
        device_type = os.environ.get("NANOCHAT_SMOKE_DEVICE_TYPE", "cpu")
        if device_type not in {"cpu", "cuda", "mps"}:
            _skip_smoke(f"NANOCHAT_SMOKE_DEVICE_TYPE must be cpu/cuda/mps, got {device_type!r}")
        if device_type == "cuda" and not torch.cuda.is_available():
            _skip_smoke("optional engine smoke requested CUDA but torch.cuda.is_available() is false")
        if device_type == "mps" and not torch.backends.mps.is_available():
            _skip_smoke("optional engine smoke requested MPS but torch.backends.mps.is_available() is false")
        artifact_dir = _default_artifact_dir()
        return SmokeCheckpoint(
            base_dir=base_dir,
            source=source,
            model_tag=model_tag,
            step=step,
            device_type=device_type,
            artifact_dir=artifact_dir,
        )

    _skip_smoke(
        "optional engine smoke requires a checkpoint under one of "
        f"{[SOURCE_DIR_NAMES[source] for source in _resolve_sources()]} inside {base_dir}"
    )


def _ensure_runtime_is_loadable(smoke: SmokeCheckpoint) -> None:
    previous_base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    model = None
    tokenizer = None
    try:
        os.environ["NANOCHAT_BASE_DIR"] = str(smoke.base_dir)
        model, tokenizer, _ = load_model(
            smoke.source,
            torch.device(smoke.device_type),
            phase="eval",
            model_tag=smoke.model_tag,
            step=smoke.step,
        )
    except Exception as exc:  # pragma: no cover - exercised only in local smoke environments
        _skip_smoke(f"optional engine smoke could not load the checkpoint runtime: {exc}")
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if previous_base_dir is None:
            os.environ.pop("NANOCHAT_BASE_DIR", None)
        else:
            os.environ["NANOCHAT_BASE_DIR"] = previous_base_dir


def _build_command_env(smoke: SmokeCheckpoint) -> dict[str, str]:
    env = os.environ.copy()
    env["NANOCHAT_BASE_DIR"] = str(smoke.base_dir)
    if smoke.device_type in {"cpu", "mps"}:
        env.setdefault("NANOCHAT_DTYPE", "float32")
    return env


def _run_command(args: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        command = " ".join(args)
        raise AssertionError(
            f"command failed: {command}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _load_payload(path: Path) -> dict[str, object]:
    assert path.exists(), f"expected artifact {path} to exist"
    return json.loads(path.read_text(encoding="utf-8"))


def _row_statuses(payload: dict[str, object]) -> set[str]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return set()
    return {
        str(row.get("runtime_override_status"))
        for row in rows
        if isinstance(row, dict) and "runtime_override_status" in row
    }


def _artifact_record(label: str, path: Path, payload: dict[str, object]) -> EngineSmokeArtifactRecord:
    rows = payload.get("rows", [])
    row_count = len(rows) if isinstance(rows, list) else 0
    statuses = _row_statuses(payload)
    runtime_override_counts = payload.get("runtime_variant_override_counts", {})
    if not isinstance(runtime_override_counts, dict):
        runtime_override_counts = {}
    return EngineSmokeArtifactRecord(
        label=label,
        path=str(path),
        row_count=row_count,
        runtime_override_statuses=sorted(statuses),
        runtime_override_counts={str(key): int(value) for key, value in runtime_override_counts.items()},
    )


def _build_truthfulness_variants(config: object) -> list[LocalDelibVariant]:
    local_delib_enabled = bool(getattr(config, "local_delib", False))
    exact_kwargs = {"local_delib": local_delib_enabled}

    incompatible_kwargs = {
        "local_delib": True,
        "local_delib_steps": max(1, int(getattr(config, "local_delib_steps", 0) or 0)),
    }
    current_scratch_slots = int(getattr(config, "local_delib_scratch_slots", 0) or 0)
    current_scratch_dim = int(getattr(config, "local_delib_scratch_dim", 0) or 0)
    if current_scratch_slots > 0 and current_scratch_dim > 0:
        incompatible_kwargs["local_delib_scratch_slots"] = current_scratch_slots
        incompatible_kwargs["local_delib_scratch_dim"] = current_scratch_dim + 1
    else:
        incompatible_kwargs["local_delib_scratch_slots"] = max(1, current_scratch_slots)
        incompatible_kwargs["local_delib_scratch_dim"] = max(4, current_scratch_dim)

    return [
        LocalDelibVariant("checkpoint_config_exact", exact_kwargs),
        LocalDelibVariant("shape_incompatible_override", incompatible_kwargs),
    ]


def _write_targeted_runtime_override_audit(smoke: SmokeCheckpoint) -> dict[str, object]:
    previous_base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    audit_path: Path | None = None
    model = None
    tokenizer = None
    try:
        os.environ["NANOCHAT_BASE_DIR"] = str(smoke.base_dir)
        model, tokenizer, _ = load_model(
            smoke.source,
            torch.device(smoke.device_type),
            phase="eval",
            model_tag=smoke.model_tag,
            step=smoke.step,
        )
        engine = Engine(model, tokenizer)
        backend = EngineBackend(
            engine=engine,
            tokenizer=tokenizer,
            max_tokens=32,
            temperature=0.0,
            top_k=1,
        )
        summary = run_advanced_local_delib_ablation_eval(
            [ADVANCED_LOCAL_DELIB_CASES[0]],
            backend=backend,
            variants=_build_truthfulness_variants(model.config),
        )
        audit_path = write_advanced_local_delib_eval_artifact(
            summary,
            str(smoke.artifact_dir / "runtime_override_audit.json"),
        )
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if previous_base_dir is None:
            os.environ.pop("NANOCHAT_BASE_DIR", None)
        else:
            os.environ["NANOCHAT_BASE_DIR"] = previous_base_dir

    assert audit_path is not None
    payload = _load_payload(audit_path)
    statuses = payload.get("runtime_variant_override_statuses", {})
    assert statuses.get("checkpoint_config_exact") == "exact"
    assert statuses.get("shape_incompatible_override") in {"approximated", "unsupported"}
    return payload


def test_engine_eval_cli_smoke_writes_artifacts_and_audits_runtime_override_statuses() -> None:
    smoke = _resolve_smoke_checkpoint()
    _ensure_runtime_is_loadable(smoke)
    smoke.artifact_dir.mkdir(parents=True, exist_ok=True)
    env = _build_command_env(smoke)

    cognition_path = smoke.artifact_dir / "cognition_eval.json"
    advanced_path = smoke.artifact_dir / "local_delib_ablation_advanced.json"
    research_path = smoke.artifact_dir / "local_delib_research.json"
    task_grounded_path = smoke.artifact_dir / "task_grounded.json"
    natural_path = smoke.artifact_dir / "local_delib_natural.json"

    common_args = [
        "--backend",
        "engine",
        "--source",
        smoke.source,
        "--model-tag",
        smoke.model_tag,
        "--step",
        str(smoke.step),
        "--device-type",
        smoke.device_type,
        "--temperature",
        "0.0",
        "--top-k",
        "1",
        "--max-tokens",
        "64",
    ]
    commands: list[EngineSmokeCommandRecord] = []
    artifacts: list[EngineSmokeArtifactRecord] = []

    def run_and_record(label: str, argv: list[str], artifact_path: Path) -> dict[str, object]:
        command = EngineSmokeCommandRecord(
            label=label,
            argv=[str(item) for item in argv],
            artifact_path=str(artifact_path),
        )
        commands.append(command)
        result = _run_command(argv, env=env)
        command.exit_code = result.returncode
        payload = _load_payload(artifact_path)
        artifacts.append(_artifact_record(label, artifact_path, payload))
        return payload

    try:
        cognition_payload = run_and_record(
            "cognition",
            [
                sys.executable,
                "-m",
                "scripts.cognition_eval",
                *common_args,
                "--no-enforce-improvement",
                "--output",
                str(cognition_path),
            ],
            cognition_path,
        )
        advanced_payload = run_and_record(
            "local-delib-ablation-advanced",
            [
                sys.executable,
                "-m",
                "scripts.cognition_eval",
                "--suite",
                "local-delib-ablation-advanced",
                *common_args,
                "--output",
                str(advanced_path),
            ],
            advanced_path,
        )
        research_payload = run_and_record(
            "local-delib-research",
            [
                sys.executable,
                "-m",
                "scripts.cognition_eval",
                "--suite",
                "local-delib-research",
                *common_args,
                "--output",
                str(research_path),
            ],
            research_path,
        )
        task_grounded_payload = run_and_record(
            "task-grounded",
            [
                sys.executable,
                "-m",
                "scripts.cognition_eval",
                "--suite",
                "task-grounded",
                *common_args,
                "--tasks",
                "SmokeTinyTask",
                "--max-problems",
                "2",
                "--seed",
                "42",
                "--output",
                str(task_grounded_path),
            ],
            task_grounded_path,
        )
        natural_payload = run_and_record(
            "local-delib-natural",
            [
                sys.executable,
                "-m",
                "scripts.cognition_eval",
                "--suite",
                "local-delib-natural",
                *common_args,
                "--max-problems",
                "2",
                "--seed",
                "42",
                "--output",
                str(natural_path),
            ],
            natural_path,
        )
    except Exception as exc:
        _write_smoke_manifest(
            status="failed",
            smoke=smoke,
            commands=commands,
            artifacts=artifacts,
            observed_statuses={
                status
                for artifact in artifacts
                for status in artifact.runtime_override_statuses
            },
            reason=str(exc),
        )
        raise

    assert isinstance(cognition_payload.get("rows"), list)
    assert cognition_payload["rows"]
    assert "route_counts" in cognition_payload

    assert task_grounded_payload["metric_tier"] == "task_grounded"
    assert task_grounded_payload["checkpoint_identity"]["source"] == smoke.source
    assert isinstance(task_grounded_payload.get("rows"), list)
    assert task_grounded_payload["rows"]

    for payload in (advanced_payload, research_payload, natural_payload):
        assert isinstance(payload.get("rows"), list)
        assert payload["rows"]
        assert "runtime_variant_override_statuses" in payload
        assert "runtime_variant_override_counts" in payload
        row_statuses = _row_statuses(payload)
        assert row_statuses <= ROW_STATUS_VALUES
        assert row_statuses

    assert natural_payload["metric_tier"] == "natural_task_grounded"
    assert "proof_pass_rates" in natural_payload

    observed_statuses = _row_statuses(advanced_payload) | _row_statuses(research_payload) | _row_statuses(natural_payload)
    if "exact" not in observed_statuses or observed_statuses.isdisjoint({"approximated", "unsupported"}):
        audit_payload = _write_targeted_runtime_override_audit(smoke)
        artifacts.append(_artifact_record("runtime-override-audit", smoke.artifact_dir / "runtime_override_audit.json", audit_payload))
        observed_statuses |= _row_statuses(audit_payload)

    manifest_path = _write_smoke_manifest(
        status="passed",
        smoke=smoke,
        commands=commands,
        artifacts=artifacts,
        observed_statuses=observed_statuses,
    )

    manifest_payload = _load_payload(manifest_path)
    assert manifest_payload["status"] == "passed"
    assert manifest_payload["strict_audit"] is True
    assert manifest_payload["checkpoint_identity"]["source"] == smoke.source
    assert len(manifest_payload["commands"]) >= 5
    assert len(manifest_payload["artifacts"]) >= 5
    assert "exact" in manifest_payload["observed_runtime_override_statuses"]
    assert set(manifest_payload["observed_runtime_override_statuses"]) & {"approximated", "unsupported"}

    assert "exact" in observed_statuses
    assert observed_statuses & {"approximated", "unsupported"}


def test_chat_cli_cognition_engine_smoke_returns_non_empty_text() -> None:
    smoke = _resolve_smoke_checkpoint()
    _ensure_runtime_is_loadable(smoke)
    env = _build_command_env(smoke)
    result = _run_command(
        [
            sys.executable,
            "-m",
            "scripts.chat_cli",
            "--cognition",
            "--source",
            smoke.source,
            "--model-tag",
            smoke.model_tag,
            "--step",
            str(smoke.step),
            "--device-type",
            smoke.device_type,
            "--temperature",
            "0.0",
            "--top-k",
            "1",
            "--prompt",
            "Summarize why exact runtime override reporting matters.",
        ],
        env=env,
    )

    marker = "Assistant:"
    assert marker in result.stdout
    assistant_text = result.stdout.rsplit(marker, 1)[-1].strip()
    assert assistant_text
