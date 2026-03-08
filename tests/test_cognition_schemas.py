from nanochat.cognition.backend import BackendAdapter
from nanochat.cognition.schemas import (
    Episode,
    Hypothesis,
    MemoryItem,
    RoutingDecision,
    SkillArtifact,
    Trace,
    VerificationResult,
)


class FakeBackend:
    def generate(self, prompt: str, **kwargs: object) -> str:
        suffix = kwargs.get("suffix", "")
        return f"{prompt}{suffix}"


def test_schema_instances_have_expected_fields() -> None:
    episode = Episode(episode_id="e1", prompt="p", response="r")
    memory = MemoryItem(item_id="m1", content="c", kind="semantic")
    trace = Trace(trace_id="t1", query="q", decision="direct_answer", rationale="default")
    decision = RoutingDecision(action="direct_answer", rationale="simple")
    hypothesis = Hypothesis(hypothesis_id="h1", statement="s")
    verification = VerificationResult(verified=True, verdict="ok")
    skill = SkillArtifact(skill_id="sk1", name="N", trigger="when x", procedure=["step"])

    assert episode.episode_id == "e1"
    assert memory.kind == "semantic"
    assert trace.trace_id == "t1"
    assert decision.action == "direct_answer"
    assert hypothesis.confidence == 0.5
    assert verification.verified is True
    assert skill.procedure == ["step"]


def test_backend_adapter_passes_kwargs() -> None:
    adapter = BackendAdapter(backend=FakeBackend(), default_kwargs={"suffix": "!"})
    out = adapter.run("hello")
    assert out == "hello!"

    out2 = adapter.run("hello", suffix="?")
    assert out2 == "hello?"
