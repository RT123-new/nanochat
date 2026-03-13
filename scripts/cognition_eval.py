#!/usr/bin/env python3
"""Evaluate cognition and local-deliberation architecture variants."""

from __future__ import annotations

import argparse
import random

from nanochat.cognition.backend import EngineBackend
from nanochat.cognition.eval import (
    ADVANCED_LOCAL_DELIB_CASES,
    NATURAL_LOCAL_DELIB_CASES,
    ContextAwareEvalBackend,
    DEFAULT_CASES,
    DEFAULT_LOCAL_DELIB_CASES,
    LocalDelibContextEvalBackend,
    RESEARCH_LOCAL_DELIB_CASES,
    run_natural_local_delib_eval,
    run_research_local_delib_eval,
    run_advanced_local_delib_ablation_eval,
    run_eval,
    run_local_delib_ablation_eval,
    run_task_grounded_eval,
    write_research_local_delib_eval_artifact,
    write_advanced_local_delib_eval_artifact,
    write_eval_artifact,
    write_local_delib_eval_artifact,
    write_natural_local_delib_eval_artifact,
    write_task_grounded_eval_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cognition and local-deliberation eval suites")
    parser.add_argument(
        "--suite",
        choices=[
            "cognition",
            "local-delib-ablation",
            "local-delib-ablation-advanced",
            "local-delib-research",
            "task-grounded",
            "local-delib-natural",
        ],
        default="cognition",
        help="Eval suite to run",
    )
    parser.add_argument(
        "--backend",
        choices=["demo", "engine"],
        default="demo",
        help="Backend to compare against cognition: deterministic demo or checkpoint-backed engine",
    )
    parser.add_argument(
        "--output",
        default="artifacts/cognition_eval.json",
        help="Path to JSON artifact for eval results",
    )
    parser.add_argument("-i", "--source", default="sft", help="Checkpoint source when --backend=engine")
    parser.add_argument("-g", "--model-tag", default=None, help="Model tag to load when --backend=engine")
    parser.add_argument("-s", "--step", type=int, default=None, help="Checkpoint step to load when --backend=engine")
    parser.add_argument("--device-type", default="", choices=["cuda", "cpu", "mps"], help="Device type when --backend=engine")
    parser.add_argument("-t", "--temperature", type=float, default=0.6, help="Sampling temperature for engine backend")
    parser.add_argument("-k", "--top-k", type=int, default=50, help="Top-k sampling for engine backend")
    parser.add_argument("-m", "--max-tokens", type=int, default=192, help="Max generation tokens for engine backend")
    parser.add_argument("--tasks", default="", help="Comma-separated task names for --suite task-grounded")
    parser.add_argument("--max-problems", type=int, default=None, help="Limit the number of task examples or natural cases")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when sampling task examples or natural cases")
    parser.add_argument(
        "--allow-approximate-runtime-overrides",
        action="store_true",
        help="When an engine-backed local-delib variant cannot be applied exactly, run the loaded checkpoint anyway and mark the row as approximated",
    )
    parser.add_argument(
        "--fail-on-unsupported-runtime-overrides",
        action="store_true",
        help="Raise instead of emitting unsupported rows when a backend cannot apply a requested local-delib variant exactly",
    )
    parser.add_argument(
        "--no-enforce-improvement",
        action="store_true",
        help="Allow the cognition suite to write artifacts even when required gain rows do not beat baseline; intended for engine smoke validation rather than usefulness proofs",
    )
    args = parser.parse_args()

    backend = LocalDelibContextEvalBackend() if args.suite.startswith("local-delib") else ContextAwareEvalBackend()
    cleanup = None
    try:
        if args.backend == "engine":
            from nanochat.checkpoint_manager import load_model
            from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
            from nanochat.engine import Engine

            device_type = autodetect_device_type() if args.device_type == "" else args.device_type
            ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
            model, tokenizer, meta = load_model(
                args.source,
                device,
                phase="eval",
                model_tag=args.model_tag,
                step=args.step,
            )
            engine = Engine(model, tokenizer)
            backend = EngineBackend(
                engine=engine,
                tokenizer=tokenizer,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                allow_approximate_local_delib_overrides=args.allow_approximate_runtime_overrides,
            )
            cleanup = compute_cleanup

        checkpoint_identity = (
            {
                "source": args.source,
                "model_tag": args.model_tag,
                "step": args.step,
                "device_type": device_type,
            }
            if args.backend == "engine"
            else {}
        )

        if args.suite == "cognition":
            summary = run_eval(
                DEFAULT_CASES,
                backend=backend,
                enforce_improvement=not args.no_enforce_improvement,
            )
            artifact_path = write_eval_artifact(summary, args.output)
            print("Cognition eval summary")
            print(f"- backend: {args.backend}")
            print(f"- enforce_improvement: {not args.no_enforce_improvement}")
            print(f"- baseline_mean: {summary.baseline_mean:.3f}")
            print(f"- cognition_mean: {summary.cognition_mean:.3f}")
            print(f"- delta: {summary.delta:.3f}")
            print(f"- route_counts: {summary.route_counts}")
            print(f"- artifact: {artifact_path}")
        elif args.suite == "local-delib-ablation":
            summary = run_local_delib_ablation_eval(
                DEFAULT_LOCAL_DELIB_CASES,
                backend=backend,
                fail_on_unsupported_runtime_overrides=args.fail_on_unsupported_runtime_overrides,
            )
            artifact_path = write_local_delib_eval_artifact(summary, args.output)
            print("Local deliberation ablation eval summary")
            print(f"- backend: {args.backend}")
            print(f"- runtime_variant_overrides_applied: {summary.runtime_variant_overrides_applied}")
            print(f"- runtime_variant_override_statuses: {summary.runtime_variant_override_statuses}")
            print(f"- variant_mean_scores: {summary.variant_mean_scores}")
            print(f"- rows: {len(summary.rows)}")
            print(f"- artifact: {artifact_path}")
        elif args.suite == "local-delib-ablation-advanced":
            summary = run_advanced_local_delib_ablation_eval(
                ADVANCED_LOCAL_DELIB_CASES,
                backend=backend,
                fail_on_unsupported_runtime_overrides=args.fail_on_unsupported_runtime_overrides,
            )
            artifact_path = write_advanced_local_delib_eval_artifact(summary, args.output)
            print("Advanced local deliberation ablation eval summary")
            print(f"- backend: {args.backend}")
            print(f"- runtime_variant_overrides_applied: {summary.runtime_variant_overrides_applied}")
            print(f"- runtime_variant_override_statuses: {summary.runtime_variant_override_statuses}")
            print(f"- variant_mean_scores: {summary.variant_mean_scores}")
            print(f"- quality_per_compute: {summary.quality_per_compute}")
            print(f"- mean_steps_taken: {summary.mean_steps_taken}")
            print(f"- rows: {len(summary.rows)}")
            print(f"- artifact: {artifact_path}")
        elif args.suite == "task-grounded":
            task_names = [name.strip() for name in args.tasks.split(",") if name.strip()] or None
            summary = run_task_grounded_eval(
                backend=backend,
                task_names=task_names,
                max_problems=args.max_problems,
                seed=args.seed,
                checkpoint_identity=checkpoint_identity,
            )
            artifact_path = write_task_grounded_eval_artifact(summary, args.output)
            print("Task-grounded eval summary")
            print(f"- backend: {args.backend}")
            print(f"- backend_kind: {summary.backend_kind}")
            print(f"- metric_tier: {summary.metric_tier}")
            print(f"- task_names: {summary.task_names}")
            print(f"- baseline_mean: {summary.baseline_mean:.3f}")
            print(f"- cognition_mean: {summary.cognition_mean:.3f}")
            print(f"- proof_delta: {summary.proof_delta:.3f}")
            print(f"- artifact: {artifact_path}")
        elif args.suite == "local-delib-natural":
            cases = list(NATURAL_LOCAL_DELIB_CASES)
            if args.max_problems is not None and args.max_problems < len(cases):
                indices = list(range(len(cases)))
                random.Random(args.seed).shuffle(indices)
                cases = [cases[index] for index in indices[: args.max_problems]]
            summary = run_natural_local_delib_eval(
                cases,
                backend=backend,
                fail_on_unsupported_runtime_overrides=args.fail_on_unsupported_runtime_overrides,
                checkpoint_identity=checkpoint_identity,
            )
            artifact_path = write_natural_local_delib_eval_artifact(summary, args.output)
            print("Natural local deliberation eval summary")
            print(f"- backend: {args.backend}")
            print(f"- backend_kind: {summary.backend_kind}")
            print(f"- metric_tier: {summary.metric_tier}")
            print(f"- runtime_variant_overrides_applied: {summary.runtime_variant_overrides_applied}")
            print(f"- runtime_variant_override_statuses: {summary.runtime_variant_override_statuses}")
            print(f"- variant_mean_scores: {summary.variant_mean_scores}")
            print(f"- proof_pass_rates: {summary.proof_pass_rates}")
            print(f"- proof_delta_vs_baseline: {summary.proof_delta_vs_baseline}")
            print(f"- rows: {len(summary.rows)}")
            print(f"- artifact: {artifact_path}")
        else:
            summary = run_research_local_delib_eval(
                RESEARCH_LOCAL_DELIB_CASES,
                backend=backend,
                fail_on_unsupported_runtime_overrides=args.fail_on_unsupported_runtime_overrides,
            )
            artifact_path = write_research_local_delib_eval_artifact(summary, args.output)
            print("Prompt 4 research local deliberation eval summary")
            print(f"- backend: {args.backend}")
            print(f"- backend_kind: {summary.backend_kind}")
            print(f"- metric_tier: {summary.metric_tier}")
            print(f"- runtime_variant_overrides_applied: {summary.runtime_variant_overrides_applied}")
            print(f"- runtime_variant_override_statuses: {summary.runtime_variant_override_statuses}")
            print(f"- variant_mean_scores: {summary.variant_mean_scores}")
            print(f"- variant_pass_rates: {summary.variant_pass_rates}")
            print(f"- delta_vs_baseline: {summary.delta_vs_baseline}")
            print(f"- quality_per_compute: {summary.quality_per_compute}")
            print(f"- rows: {len(summary.rows)}")
            print(f"- artifact: {artifact_path}")
    finally:
        if cleanup is not None:
            cleanup()


if __name__ == "__main__":
    main()
