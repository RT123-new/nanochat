#!/usr/bin/env python3
"""Evaluate cognition and local-deliberation architecture variants."""

from __future__ import annotations

import argparse

from nanochat.cognition.backend import EngineBackend
from nanochat.cognition.eval import (
    ADVANCED_LOCAL_DELIB_CASES,
    ContextAwareEvalBackend,
    DEFAULT_CASES,
    DEFAULT_LOCAL_DELIB_CASES,
    LocalDelibContextEvalBackend,
    run_advanced_local_delib_ablation_eval,
    run_eval,
    run_local_delib_ablation_eval,
    write_advanced_local_delib_eval_artifact,
    write_eval_artifact,
    write_local_delib_eval_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cognition and local-deliberation eval suites")
    parser.add_argument(
        "--suite",
        choices=["cognition", "local-delib-ablation", "local-delib-ablation-advanced"],
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
    args = parser.parse_args()

    backend = LocalDelibContextEvalBackend() if args.suite.startswith("local-delib-ablation") else ContextAwareEvalBackend()
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
            )
            cleanup = compute_cleanup

        if args.suite == "cognition":
            summary = run_eval(DEFAULT_CASES, backend=backend)
            artifact_path = write_eval_artifact(summary, args.output)
            print("Cognition eval summary")
            print(f"- backend: {args.backend}")
            print(f"- baseline_mean: {summary.baseline_mean:.3f}")
            print(f"- cognition_mean: {summary.cognition_mean:.3f}")
            print(f"- delta: {summary.delta:.3f}")
            print(f"- route_counts: {summary.route_counts}")
            print(f"- artifact: {artifact_path}")
        elif args.suite == "local-delib-ablation":
            summary = run_local_delib_ablation_eval(DEFAULT_LOCAL_DELIB_CASES, backend=backend)
            artifact_path = write_local_delib_eval_artifact(summary, args.output)
            print("Local deliberation ablation eval summary")
            print(f"- backend: {args.backend}")
            print(f"- variant_mean_scores: {summary.variant_mean_scores}")
            print(f"- rows: {len(summary.rows)}")
            print(f"- artifact: {artifact_path}")
        else:
            summary = run_advanced_local_delib_ablation_eval(ADVANCED_LOCAL_DELIB_CASES, backend=backend)
            artifact_path = write_advanced_local_delib_eval_artifact(summary, args.output)
            print("Advanced local deliberation ablation eval summary")
            print(f"- backend: {args.backend}")
            print(f"- runtime_variant_overrides_applied: {summary.runtime_variant_overrides_applied}")
            print(f"- variant_mean_scores: {summary.variant_mean_scores}")
            print(f"- quality_per_compute: {summary.quality_per_compute}")
            print(f"- mean_steps_taken: {summary.mean_steps_taken}")
            print(f"- rows: {len(summary.rows)}")
            print(f"- artifact: {artifact_path}")
    finally:
        if cleanup is not None:
            cleanup()


if __name__ == "__main__":
    main()
