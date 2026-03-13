from __future__ import annotations

import json

from nanochat.cognition.backend import LocalDelibRuntimeOverrideError, LocalDelibRuntimeOverrideReport
from nanochat.cognition.eval import (
    ADVANCED_LOCAL_DELIB_CASES,
    ADVANCED_LOCAL_DELIB_VARIANTS,
    ContextAwareEvalBackend,
    EvalCase,
    LocalDelibContextEvalBackend,
    LocalDelibVariant,
    NATURAL_LOCAL_DELIB_CASES,
    RESEARCH_LOCAL_DELIB_CASES,
    run_natural_local_delib_eval,
    run_advanced_local_delib_ablation_eval,
    run_eval,
    run_research_local_delib_eval,
    write_research_local_delib_eval_artifact,
)
from nanochat.cognition.schemas import Episode, MemoryItem, SkillArtifact


def _advanced_variant(variant_id: str) -> LocalDelibVariant:
    return next(variant for variant in ADVANCED_LOCAL_DELIB_VARIANTS if variant.variant_id == variant_id)


def _research_case(case_id: str):
    return next(case for case in RESEARCH_LOCAL_DELIB_CASES if case.case_id == case_id)


class UninstrumentedResearchGainBackend:
    supports_local_delib_runtime_overrides = True

    def __init__(self) -> None:
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        requested_overrides = {
            str(key): value
            for key, value in kwargs.items()
            if str(key) == "local_delib" or str(key).startswith("local_delib_")
        }
        self.last_generation_metadata = {
            "local_delib_runtime_override": LocalDelibRuntimeOverrideReport(
                status="exact",
                requested_overrides=requested_overrides,
                applied_overrides=requested_overrides,
                application_method="uninstrumented_backend",
            ).to_metadata()
        }
        if "RESEARCH_TASK: branch_consensus" in prompt:
            if bool(kwargs.get("local_delib_branch_consensus", False)) and bool(kwargs.get("local_delib_branch_verifier", False)):
                return "WINNER=branch_b\nMERGED=alpha,beta,gamma"
            return "WINNER=branch_a\nMERGED=alpha"
        return "STATUS=unsupported"


class RuntimeStatusProofBackend:
    supports_local_delib_runtime_overrides = True

    def __init__(self) -> None:
        self._exact_backend = LocalDelibContextEvalBackend()
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        requested_overrides = {
            str(key): value
            for key, value in kwargs.items()
            if str(key) == "local_delib" or str(key).startswith("local_delib_")
        }
        if bool(kwargs.get("local_delib_use_thought_graph", False)):
            report = LocalDelibRuntimeOverrideReport(
                status="unsupported",
                requested_overrides=requested_overrides,
                applied_overrides={},
                application_method="unsupported_backend",
                reason="thought-graph variant disabled in this proof backend",
            )
            self.last_generation_metadata = {"local_delib_runtime_override": report.to_metadata()}
            raise LocalDelibRuntimeOverrideError(report)

        response = self._exact_backend.generate(prompt, **kwargs)
        metadata = dict(self._exact_backend.last_generation_metadata or {})
        if bool(kwargs.get("local_delib_branch_consensus", False)) and bool(kwargs.get("local_delib_branch_verifier", False)):
            metadata["local_delib_runtime_override"] = LocalDelibRuntimeOverrideReport(
                status="approximated",
                requested_overrides=requested_overrides,
                applied_overrides={},
                application_method="loaded_checkpoint_fallback",
                reason="used checkpoint-compatible fallback for branch proof",
            ).to_metadata()
        self.last_generation_metadata = metadata
        return response


def test_default_eval_gains_come_from_support_injection_not_backend_strength() -> None:
    cases = [
        EvalCase(
            case_id="episodic_supported",
            prompt="Can you recall the previous summarization guidance for this draft?",
            expected_keywords=["terse", "bullet", "citations"],
            seeded_episodes=[
                Episode(
                    episode_id="episode-style",
                    prompt="What is our summarization style?",
                    response="Use terse bullet summaries with citations.",
                    tags=["summarization"],
                    metadata={"success": True, "trigger": "summarization"},
                )
            ],
        ),
        EvalCase(
            case_id="episodic_missing",
            prompt="Can you recall the previous summarization guidance for this draft?",
            expected_keywords=["terse", "bullet", "citations"],
        ),
        EvalCase(
            case_id="semantic_supported",
            prompt="Answer in our house style for this reply.",
            expected_keywords=["brief", "neutral", "numbered"],
            seeded_semantic_items=[
                MemoryItem(
                    item_id="semantic-house-style",
                    kind="semantic",
                    content="House style: brief neutral numbered answers.",
                )
            ],
        ),
        EvalCase(
            case_id="semantic_missing",
            prompt="Answer in our house style for this reply.",
            expected_keywords=["brief", "neutral", "numbered"],
        ),
        EvalCase(
            case_id="skill_supported",
            prompt="Please summarize this draft.",
            expected_keywords=["extract", "bullets", "condense"],
            seeded_skills=[
                SkillArtifact(
                    skill_id="skill-summarization",
                    name="Reusable summarization checklist",
                    trigger="summarization",
                    procedure=["extract bullets", "condense the bullets"],
                )
            ],
        ),
        EvalCase(
            case_id="skill_missing",
            prompt="Please summarize this draft.",
            expected_keywords=["extract", "bullets", "condense"],
        ),
    ]

    summary = run_eval(cases, backend=ContextAwareEvalBackend(), enforce_improvement=False)
    rows = {row.case_id: row for row in summary.rows}

    assert rows["episodic_supported"].baseline_response == rows["episodic_missing"].baseline_response
    assert rows["episodic_supported"].cognition_score > rows["episodic_supported"].baseline_score
    assert rows["episodic_missing"].cognition_response == rows["episodic_missing"].baseline_response
    assert rows["episodic_missing"].cognition_score == rows["episodic_missing"].baseline_score == 0.0

    assert rows["semantic_supported"].baseline_response == rows["semantic_missing"].baseline_response
    assert rows["semantic_supported"].cognition_score > rows["semantic_supported"].baseline_score
    assert rows["semantic_missing"].cognition_response == rows["semantic_missing"].baseline_response
    assert rows["semantic_missing"].cognition_score == rows["semantic_missing"].baseline_score == 0.0

    assert rows["skill_supported"].baseline_response == rows["skill_missing"].baseline_response
    assert rows["skill_supported"].cognition_score > rows["skill_supported"].baseline_score
    assert rows["skill_missing"].cognition_response == rows["skill_missing"].baseline_response
    assert rows["skill_missing"].cognition_score == rows["skill_missing"].baseline_score == 0.0


def test_advanced_ablation_summary_contains_activation_and_compute_evidence() -> None:
    summary = run_advanced_local_delib_ablation_eval(
        [next(case for case in ADVANCED_LOCAL_DELIB_CASES if case.case_id == "thought_anchor")],
        backend=LocalDelibContextEvalBackend(),
        variants=[
            _advanced_variant("local_delib_off"),
            _advanced_variant("local_delib_thought_graph"),
            _advanced_variant("local_delib_global_anchors"),
            _advanced_variant("local_delib_combo_full_stack"),
        ],
    )

    rows = {row.variant_id: row for row in summary.rows}
    assert rows["local_delib_off"].quality_proxy_score < rows["local_delib_thought_graph"].quality_proxy_score
    assert rows["local_delib_thought_graph"].compute_proxy_metrics["estimated_compute_cost"] > rows["local_delib_off"].compute_proxy_metrics["estimated_compute_cost"]
    assert rows["local_delib_thought_graph"].thought_graph_stats["thought_nodes_used"] > 0.0
    assert rows["local_delib_global_anchors"].anchor_stats["global_anchors_used"] > 0.0
    assert rows["local_delib_combo_full_stack"].compute_proxy_metrics["estimated_compute_cost"] > 0.0
    assert "thought_graph" in rows["local_delib_combo_full_stack"].model_local_delib_graph_artifact
    assert "anchors" in rows["local_delib_combo_full_stack"].model_local_delib_graph_artifact

    assert summary.compute_proxy_metrics["local_delib_thought_graph"]["estimated_compute_cost"] > 0.0
    assert summary.thought_graph_stats["local_delib_thought_graph"]["thought_nodes_used"] > 0.0
    assert summary.anchor_stats["local_delib_global_anchors"]["global_anchors_used"] > 0.0


def test_research_eval_pairs_positive_delta_with_activation_coverage_in_artifact(tmp_path) -> None:
    summary = run_research_local_delib_eval(
        [_research_case("branch_consensus")],
        backend=UninstrumentedResearchGainBackend(),
        variants=[
            LocalDelibVariant("local_delib_off", {"local_delib": False}),
            LocalDelibVariant(
                "claimed_branch_consensus",
                {
                    "local_delib": True,
                    "local_delib_branch_factor": 3,
                    "local_delib_branch_every": 1,
                    "local_delib_branch_consensus": True,
                    "local_delib_branch_verifier": True,
                },
            ),
        ],
    )

    claimed_row = next(row for row in summary.rows if row.variant_id == "claimed_branch_consensus")
    assert claimed_row.metric_score == 1.0
    assert claimed_row.passed is True
    assert claimed_row.response_format_ok is True
    assert claimed_row.activation_checks["branch_consensus"] is False
    assert claimed_row.activation_ok is False
    assert claimed_row.metrics_interpretable is False
    assert summary.delta_vs_baseline["claimed_branch_consensus"] > 0.0
    assert summary.activation_coverage["claimed_branch_consensus"]["activation_ok_rate"] == 0.0
    assert summary.activation_coverage["claimed_branch_consensus"]["metrics_interpretable_rate"] == 0.0

    artifact_path = write_research_local_delib_eval_artifact(summary, str(tmp_path / "research_proof.json"))
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["delta_vs_baseline"]["claimed_branch_consensus"] > 0.0
    assert payload["activation_coverage"]["claimed_branch_consensus"]["activation_ok_rate"] == 0.0
    assert payload["activation_coverage"]["claimed_branch_consensus"]["metrics_interpretable_rate"] == 0.0


def test_research_eval_marks_approximated_and_unsupported_rows_as_non_proof() -> None:
    summary = run_research_local_delib_eval(
        [_research_case("branch_consensus")],
        backend=RuntimeStatusProofBackend(),
        variants=[
            LocalDelibVariant("local_delib_off", {"local_delib": False}),
            LocalDelibVariant(
                "approx_branch_consensus",
                {
                    "local_delib": True,
                    "local_delib_branch_factor": 3,
                    "local_delib_branch_every": 1,
                    "local_delib_branch_consensus": True,
                    "local_delib_branch_verifier": True,
                },
            ),
            LocalDelibVariant(
                "unsupported_thought_graph",
                {
                    "local_delib": True,
                    "local_delib_use_thought_graph": True,
                    "local_delib_thought_node_budget": 6,
                    "local_delib_thought_graph_steps": 2,
                },
            ),
        ],
    )

    rows = {row.variant_id: row for row in summary.rows}
    approx_row = rows["approx_branch_consensus"]
    unsupported_row = rows["unsupported_thought_graph"]

    assert approx_row.metric_score == 1.0
    assert approx_row.runtime_override_status == "approximated"
    assert approx_row.runtime_override_applied is False
    assert approx_row.activation_checks["branch_consensus"] is True
    assert approx_row.activation_ok is False
    assert approx_row.metrics_interpretable is False

    assert unsupported_row.runtime_override_status == "unsupported"
    assert unsupported_row.runtime_override_applied is False
    assert unsupported_row.response == ""
    assert unsupported_row.activation_ok is False
    assert unsupported_row.metrics_interpretable is False

    assert summary.runtime_variant_overrides_applied is False
    assert summary.runtime_variant_override_statuses == {
        "local_delib_off": "exact",
        "approx_branch_consensus": "approximated",
        "unsupported_thought_graph": "unsupported",
    }
    assert summary.runtime_variant_override_counts == {"exact": 1, "approximated": 1, "unsupported": 1}
    assert summary.activation_coverage["approx_branch_consensus"]["activation_ok_rate"] == 0.0
    assert summary.activation_coverage["approx_branch_consensus"]["metrics_interpretable_rate"] == 0.0
    assert summary.activation_coverage["unsupported_thought_graph"]["metrics_interpretable_rate"] == 0.0


def test_natural_eval_never_counts_non_exact_rows_as_benchmark_evidence() -> None:
    summary = run_natural_local_delib_eval(
        [next(case for case in NATURAL_LOCAL_DELIB_CASES if case.case_id == "branch_consistency")],
        backend=RuntimeStatusProofBackend(),
        variants=[
            LocalDelibVariant("local_delib_off", {"local_delib": False}),
            LocalDelibVariant(
                "approx_branch_consensus",
                {
                    "local_delib": True,
                    "local_delib_branch_factor": 3,
                    "local_delib_branch_every": 1,
                    "local_delib_branch_consensus": True,
                    "local_delib_branch_verifier": True,
                },
            ),
            LocalDelibVariant(
                "unsupported_thought_graph",
                {
                    "local_delib": True,
                    "local_delib_use_thought_graph": True,
                    "local_delib_thought_node_budget": 6,
                    "local_delib_thought_graph_steps": 2,
                },
            ),
        ],
    )

    rows = {row.variant_id: row for row in summary.rows}
    assert rows["approx_branch_consensus"].passed is True
    assert rows["approx_branch_consensus"].proof_eligible is False
    assert rows["approx_branch_consensus"].proof_passed is False
    assert rows["unsupported_thought_graph"].proof_eligible is False
    assert rows["unsupported_thought_graph"].proof_passed is False

    assert summary.proof_pass_rates == {
        "local_delib_off": 0.0,
        "approx_branch_consensus": 0.0,
        "unsupported_thought_graph": 0.0,
    }
    assert summary.proof_variant_mean_scores["approx_branch_consensus"] == 0.0
    assert summary.runtime_variant_override_statuses == {
        "local_delib_off": "exact",
        "approx_branch_consensus": "approximated",
        "unsupported_thought_graph": "unsupported",
    }
    assert summary.activation_coverage["approx_branch_consensus"]["proof_eligible_rate"] == 0.0
