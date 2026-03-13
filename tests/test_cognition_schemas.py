from datetime import datetime, timezone

from nanochat.cognition.backend import BackendAdapter
from nanochat.cognition.schemas import (
    Episode,
    Hypothesis,
    MemoryItem,
    RoutingDecision,
    SkillArtifact,
    Trace,
    VerificationResult,
    utc_now_iso,
)


class FakeBackend:
    def generate(self, prompt: str, **kwargs: object) -> str:
        suffix = kwargs.get("suffix", "")
        return f"{prompt}{suffix}"


def test_utc_now_iso_returns_timezone_aware_iso_timestamp() -> None:
    timestamp = utc_now_iso()

    parsed = datetime.fromisoformat(timestamp)

    assert parsed.tzinfo == timezone.utc


def test_schema_instances_have_expected_fields_and_defaults() -> None:
    episode_a = Episode(episode_id="e1", prompt="p", response="r")
    episode_b = Episode(episode_id="e2", prompt="p2", response="r2")
    memory = MemoryItem(item_id="m1", content="c", kind="semantic")
    trace = Trace(trace_id="t1", query="q", decision="direct_answer", rationale="default")
    decision = RoutingDecision(action="direct_answer", rationale="simple")
    hypothesis = Hypothesis(hypothesis_id="h1", statement="s")
    verification = VerificationResult(verified=True, verdict="ok")
    skill = SkillArtifact(skill_id="sk1", name="N", trigger="when x", procedure=["step"])

    episode_a.tags.append("tagged")
    trace.steps.append("route:direct_answer")

    assert episode_a.episode_id == "e1"
    assert episode_b.tags == []
    assert memory.kind == "semantic"
    assert memory.source == "cognition"
    assert memory.score == 0.0
    assert trace.trace_id == "t1"
    assert trace.steps == ["route:direct_answer"]
    assert decision.action == "direct_answer"
    assert decision.confidence == 0.5
    assert hypothesis.confidence == 0.5
    assert verification.verified is True
    assert verification.issues == []
    assert skill.procedure == ["step"]
    assert skill.success_signals == []


def test_trace_routing_verification_and_skill_accept_realistic_metadata_payloads() -> None:
    trace = Trace(
        trace_id="trace-7",
        query="Verify this plan",
        decision="verify",
        rationale="verification requested",
        metadata={
            "model_local_delib": [{"layer_idx": 1, "agreement": 0.77}],
            "creative_workspace": {"selected_strategy_id": "branch_resolution"},
        },
    )
    decision = RoutingDecision(
        action="verify",
        rationale="User asked for validation",
        confidence=0.9,
        metadata={"matched_terms": ["verify", "proof"], "route_family": "verification"},
    )
    verification = VerificationResult(
        verified=False,
        verdict="needs repair",
        issues=["missing evidence", "weak grounding"],
        score=0.42,
        metadata={"subscores": {"grounding": 0.2, "usefulness": 0.7}},
    )
    skill = SkillArtifact(
        skill_id="skill-summary",
        name="Summarization checklist",
        trigger="summarization",
        procedure=["extract bullets", "condense with citations"],
        metadata={
            "provenance_episode_ids": ["ep-1", "ep-3"],
            "supporting_scores": {"reuse_rate": 0.8},
        },
    )

    assert trace.metadata["model_local_delib"][0]["agreement"] == 0.77
    assert decision.metadata["matched_terms"] == ["verify", "proof"]
    assert verification.metadata["subscores"]["grounding"] == 0.2
    assert skill.metadata["provenance_episode_ids"] == ["ep-1", "ep-3"]


def test_backend_adapter_passes_kwargs() -> None:
    adapter = BackendAdapter(backend=FakeBackend(), default_kwargs={"suffix": "!"})
    out = adapter.run("hello")
    assert out == "hello!"

    out2 = adapter.run("hello", suffix="?")
    assert out2 == "hello?"
