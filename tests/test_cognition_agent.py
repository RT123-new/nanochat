from __future__ import annotations

from nanochat.cognition.agent import CognitionAgent
from nanochat.cognition.backend import BackendAdapter
from nanochat.cognition.schemas import Episode, MemoryItem, SkillArtifact


class FakeBackend:
    def generate(self, prompt: str, **kwargs: object) -> str:
        return f"generated::{prompt.splitlines()[0]}"


class CapturingBackend:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return "captured"


class DelibMetadataBackend(CapturingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        self.last_generation_metadata = {
            "local_deliberation_stats": [{"layer_idx": 1, "agreement": 0.75, "mean_branch_score": 0.62, "branch_factor_used": 2}],
            "model_local_delib.branch": {"agreement": 0.75, "mean_branch_score": 0.62, "branch_factor_used": 2.0},
            "model_local_delib.scratchpad_summaries": [{"layer_idx": 1, "summary": [0.1, 0.2, 0.3]}],
            "model_local_delib.adaptive_halt": {"halted_token_fraction": 0.4},
            "model_local_delib.thought_summaries.branch_consensus": {
                "layer_count": 1,
                "branch_consensus_used": 1.0,
                "mean_branch_consensus_weight": 0.7,
            },
            "model_local_delib.thought_summaries.scratch": {
                "layer_count": 1,
                "summary_dim": 3,
                "mean_summary_norm": (0.1**2 + 0.2**2 + 0.3**2) ** 0.5,
            },
            "model_local_delib.thought_summaries.thought_graph": {
                "layer_count": 1,
                "thought_nodes_used": 4.0,
                "thought_graph_steps_used": 2.0,
            },
        }
        return "captured"


def test_end_to_end_cognition_loop_produces_trace_and_records_episode() -> None:
    agent = CognitionAgent(backend=BackendAdapter(backend=FakeBackend()))

    result = agent.run("Brainstorm ideas for memory routing")

    assert result.response
    assert result.trace.trace_id
    assert result.trace.decision == "creative_explore"
    assert result.trace.steps
    hits = agent.episodic.retrieve("memory routing", limit=5)
    assert hits


def test_end_to_end_retrieval_and_consolidation_paths() -> None:
    agent = CognitionAgent(backend=BackendAdapter(backend=FakeBackend()), min_skill_repetitions=2)
    agent.episodic.write(
        Episode(
            episode_id="e1",
            prompt="Summarize notes",
            response="extract bullets then condense",
            tags=["summarization"],
            metadata={"success": True, "trigger": "summarization", "strategy": "extract bullets then condense"},
        )
    )
    agent.episodic.write(
        Episode(
            episode_id="e2",
            prompt="Summarize report",
            response="extract bullets then condense",
            tags=["summarization"],
            metadata={"success": True, "trigger": "summarization", "strategy": "extract bullets then condense"},
        )
    )

    consolidated = agent.run("Please consolidate repeated summarization pattern")
    assert consolidated.decision == "consolidate"
    assert "Consolidated skill" in consolidated.response

    reused = agent.run("Can you help with summarization?")
    assert reused.reused_skill_id is not None
    assert reused.trace.metadata["reused_skill_ids"] == [reused.reused_skill_id]

    retrieval = agent.run("Can you recall previous summarization guidance?")
    assert retrieval.decision == "retrieve_memory"
    assert retrieval.trace.metadata["retrieved_episode_ids"]


def test_agent_injects_semantic_memory_and_skill_into_prompt() -> None:
    backend = CapturingBackend()
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))
    agent.semantic.write(
        MemoryItem(
            item_id="semantic-summarization",
            kind="semantic",
            content="Summarization style: terse bullet answers.",
        )
    )
    agent.registry.register(
        SkillArtifact(
            skill_id="skill-summarization",
            name="Summarization checklist",
            trigger="summarization",
            procedure=["extract bullets", "condense the bullets"],
        )
    )

    result = agent.run("Please summarize this draft.")

    assert backend.prompts
    assert "Relevant semantic memory:" in backend.prompts[-1]
    assert "Relevant skill:" in backend.prompts[-1]
    assert result.trace.metadata["retrieved_semantic_ids"] == ["semantic-summarization"]
    assert result.trace.metadata["reused_skill_ids"] == ["skill-summarization"]


def test_agent_injects_episodic_memory_for_ordinary_paraphrased_query() -> None:
    backend = CapturingBackend()
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))
    agent.episodic.write(
        Episode(
            episode_id="ep-style",
            prompt="Summarize the project update",
            response="Use terse bullet summaries with citations.",
            tags=["summarization"],
            metadata={"success": True, "trigger": "summarization"},
        )
    )

    result = agent.run("Please summarize this draft for me.")

    assert result.decision == "direct_answer"
    assert backend.prompts
    assert "Relevant episodic memory:" in backend.prompts[-1]
    assert result.trace.metadata["retrieved_episode_ids"] == ["ep-style"]


def test_agent_trace_includes_model_local_delib_metadata_when_available() -> None:
    backend = DelibMetadataBackend()
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))

    result = agent.run("Please summarize this draft for me.")

    assert result.response == "captured"
    assert result.trace.metadata["model_local_delib"] == [{"layer_idx": 1, "agreement": 0.75, "mean_branch_score": 0.62, "branch_factor_used": 2}]
    assert result.trace.metadata["model_local_delib.branch"]["branch_factor_used"] == 2.0
    assert result.trace.metadata["model_local_delib.scratchpad_summaries"] == [{"layer_idx": 1, "summary": [0.1, 0.2, 0.3]}]
    assert result.trace.metadata["model_local_delib.adaptive_halt"]["halted_token_fraction"] == 0.4
    assert result.trace.metadata["model_local_delib.thought_summaries.branch_consensus"]["branch_consensus_used"] == 1.0
    assert result.trace.metadata["model_local_delib.thought_summaries.scratch"]["summary_dim"] == 3
    assert result.trace.metadata["model_local_delib.thought_summaries.thought_graph"]["thought_nodes_used"] == 4.0
