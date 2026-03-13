from __future__ import annotations

from nanochat.cognition.agent import CognitionAgent
from nanochat.cognition.backend import BackendAdapter, build_local_delib_namespaced_metadata
from nanochat.cognition.schemas import Episode, MemoryItem, SkillArtifact


class FakeBackend:
    def generate(self, prompt: str, **kwargs: object) -> str:
        return f"generated::{prompt.splitlines()[0]}"


class ConstantBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return self.response


class CapturingBackend:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return "captured"


class DelibMetadataBackend(CapturingBackend):
    def __init__(self, *, include_runtime_override: bool = False) -> None:
        super().__init__()
        self.include_runtime_override = include_runtime_override
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        stats = [{
            "layer_idx": 1,
            "agreement": 0.75,
            "mean_branch_score": 0.62,
            "branch_factor_used": 2,
            "scratch_summary_vector": [0.1, 0.2, 0.3],
            "mean_scratch_summary_norm": 0.25,
            "scratch_slots_used": 3,
            "halted_token_fraction": 0.4,
            "mean_steps_taken": 1.6,
            "mean_branch_consensus_weight": 0.7,
            "branch_consensus_used": 1.0,
            "thought_nodes_used": 4.0,
            "thought_graph_steps_used": 2.0,
        }]
        self.last_generation_metadata = {
            "local_deliberation_stats": stats,
            **build_local_delib_namespaced_metadata(stats),
        }
        if self.include_runtime_override:
            self.last_generation_metadata["local_delib_runtime_override"] = {
                "status": "exact",
                "requested_overrides": {"local_delib": True, "local_delib_use_thought_graph": True},
                "applied_overrides": {"local_delib": True, "local_delib_use_thought_graph": True},
                "application_method": "reinstantiated_model",
            }
        return "captured"


class StrategyAwareBackend(CapturingBackend):
    def __init__(self, *, include_metadata: bool = False) -> None:
        super().__init__()
        self.include_metadata = include_metadata

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        strategy_id = _extract_prefixed_line(_extract_section(prompt, "Creative strategy:"), "- id:")
        if self.include_metadata:
            stats = [{
                "layer_idx": 1,
                "agreement": 0.81,
                "branch_factor_used": 3,
                "mean_branch_disagreement": 0.44,
                "mean_branch_consensus_weight": 0.61,
                "mean_branch_verifier_score": 0.66,
                "branch_consensus_used": 1.0,
                "scratch_slots_used": 2,
                "mean_scratch_summary_norm": 0.21,
                "scratch_summary_vector": [0.2, 0.3],
                "thought_nodes_used": 5.0,
                "thought_graph_steps_used": 2.0,
                "halted_token_fraction": 0.3,
                "mean_steps_taken": 1.7,
            }]
            self.last_generation_metadata = {
                "local_deliberation_stats": stats,
                **build_local_delib_namespaced_metadata(stats),
            }
        else:
            self.last_generation_metadata = None

        if strategy_id == "divergent_ideas":
            return "ideas options memory experiments"
        if strategy_id == "memory_grounded":
            return "grounded terse bullet guidance citations"
        if strategy_id == "branch_resolution":
            return "resolve branches with consensus and verify the merge"
        if strategy_id == "recombination":
            return "synthesized grounded revision with verify support"
        if strategy_id == "conservative_answer":
            return "grounded direct answer with memory support"
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


def test_direct_answer_and_retrieve_memory_paths_use_backend_without_workspace_handoffs() -> None:
    direct_backend = CapturingBackend()
    direct_agent = CognitionAgent(backend=BackendAdapter(backend=direct_backend))

    direct_result = direct_agent.run("Explain the current summary style.")

    assert direct_result.decision == "direct_answer"
    assert direct_backend.prompts == ["Explain the current summary style."]
    assert "creative_workspace" not in direct_result.trace.metadata
    assert "verifier" not in direct_result.trace.metadata
    assert "sandbox" not in direct_result.trace.metadata

    retrieve_backend = CapturingBackend()
    retrieve_agent = CognitionAgent(backend=BackendAdapter(backend=retrieve_backend))
    retrieve_agent.episodic.write(
        Episode(
            episode_id="ep-style",
            prompt="Summarize the project update",
            response="Use terse bullet summaries with citations.",
            tags=["summarization"],
            metadata={"success": True, "trigger": "summarization"},
        )
    )

    retrieve_result = retrieve_agent.run("Can you recall previous summarization guidance?")

    assert retrieve_result.decision == "retrieve_memory"
    assert len(retrieve_backend.prompts) == 1
    assert "Relevant episodic memory:" in retrieve_backend.prompts[0]
    assert "creative_workspace" not in retrieve_result.trace.metadata
    assert "verifier" not in retrieve_result.trace.metadata
    assert "sandbox" not in retrieve_result.trace.metadata


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
    assert "skill_reused:skill-summarization" in result.trace.steps


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
    assert "episodic_hits:1" in result.trace.steps


def test_agent_trace_includes_model_local_delib_metadata_when_available() -> None:
    backend = DelibMetadataBackend()
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))

    result = agent.run("Please summarize this draft for me.")

    assert result.response == "captured"
    assert result.trace.metadata["model_local_delib"][0]["layer_idx"] == 1
    assert result.trace.metadata["model_local_delib"][0]["agreement"] == 0.75
    assert result.trace.metadata["model_local_delib"][0]["branch_factor_used"] == 2
    assert result.trace.metadata["model_local_delib"][0]["thought_nodes_used"] == 4.0
    assert result.trace.metadata["model_local_delib.branch"]["branch_factor_used"] == 2.0
    assert result.trace.metadata["model_local_delib.scratchpad_summaries"] == [{"layer_idx": 1, "summary": [0.1, 0.2, 0.3]}]
    assert result.trace.metadata["model_local_delib.adaptive_halt"]["halted_token_fraction"] == 0.4
    assert result.trace.metadata["model_local_delib.thought_summaries.branch_consensus"]["branch_consensus_used"] == 1.0
    assert result.trace.metadata["model_local_delib.thought_summaries.scratch"]["summary_dim"] == 3
    assert result.trace.metadata["model_local_delib.thought_summaries.thought_graph"]["thought_nodes_used"] == 4.0
    graph_artifact = result.trace.metadata["model_local_delib.graph_artifact"]
    assert graph_artifact["overview"]["active_sections"] == ["branch", "thought_graph", "scratch", "compute"]
    assert graph_artifact["scratch"]["summary"]["has_exported_summaries"] is True
    assert graph_artifact["thought_graph"]["summary"]["thought_nodes_used"] == 4.0


def test_agent_trace_preserves_runtime_override_metadata_when_available() -> None:
    backend = DelibMetadataBackend(include_runtime_override=True)
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))

    result = agent.run("Please summarize this draft for me.")

    assert result.trace.metadata["local_delib_runtime_override"] == {
        "status": "exact",
        "requested_overrides": {"local_delib": True, "local_delib_use_thought_graph": True},
        "applied_overrides": {"local_delib": True, "local_delib_use_thought_graph": True},
        "application_method": "reinstantiated_model",
    }

    backend.last_generation_metadata["local_delib_runtime_override"]["status"] = "unsupported"
    assert result.trace.metadata["local_delib_runtime_override"]["status"] == "exact"


def test_creative_workspace_without_local_delib_metadata_uses_divergent_strategy_trace() -> None:
    backend = StrategyAwareBackend()
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))

    result = agent.run("Brainstorm ideas for memory routing")

    creative = result.trace.metadata["creative_workspace"]
    assert result.decision == "creative_explore"
    assert creative["model_summary_used"] is False
    assert creative["explored_strategy_ids"] == ["conservative_answer", "divergent_ideas"]
    assert creative["handoff"] == "verifier"
    assert creative["selected_candidate_id"] == "candidate-2"
    assert creative["rejected_candidate_ids"] == ["candidate-1"]
    assert creative["selected_strategy_id"] == "divergent_ideas"
    ranked = result.trace.metadata["verifier"]["ranked_candidates"]
    assert ranked
    assert ranked[0]["candidate_id"] == "candidate-2"
    assert ranked[0]["strategy_id"] == "divergent_ideas"
    assert any(step.startswith("verifier_score:") for step in result.trace.steps)


def test_creative_workspace_uses_local_delib_metadata_to_add_branch_resolution() -> None:
    backend = StrategyAwareBackend(include_metadata=True)
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))

    result = agent.run("Verify the branching plan before answering")

    creative = result.trace.metadata["creative_workspace"]
    assert result.decision == "verify"
    assert creative["model_summary_used"] is True
    assert "branch_resolution" in creative["explored_strategy_ids"]
    assert "recombination" in creative["explored_strategy_ids"]
    assert creative["model_summary"]["branch_disagreement"] == 0.44
    assert result.trace.metadata["verifier"]["chosen_strategy_id"] in {"branch_resolution", "recombination"}


def test_memory_heavy_creative_query_uses_memory_grounded_strategy() -> None:
    backend = StrategyAwareBackend()
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))
    agent.episodic.write(
        Episode(
            episode_id="ep-style",
            prompt="Summarize the design update",
            response="Use terse bullet summaries with citations.",
            tags=["summarization"],
            metadata={"success": True, "trigger": "summarization"},
        )
    )
    agent.semantic.write(
        MemoryItem(
            item_id="semantic-style",
            kind="semantic",
            content="Preferred style: terse bullet guidance with citations.",
        )
    )
    agent.registry.register(
        SkillArtifact(
            skill_id="skill-summarization",
            name="Summarization checklist",
            trigger="summarization",
            procedure=["extract bullets", "condense with citations"],
        )
    )

    result = agent.run("Brainstorm summarization ideas with stored guidance")

    assert result.decision == "creative_explore"
    creative = result.trace.metadata["creative_workspace"]
    assert creative["support_profile"]["memory_heavy"] is True
    assert "memory_grounded" in creative["explored_strategy_ids"]
    assert creative["selected_strategy_id"] == "memory_grounded"
    assert "Relevant semantic memory:" in backend.prompts[0]


def test_verify_path_records_verifier_handoff_metadata() -> None:
    backend = StrategyAwareBackend()
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))

    result = agent.run("Verify the answer plan for this response")

    assert result.decision == "verify"
    assert "verifier_score:" in " ".join(result.trace.steps)
    assert result.trace.metadata["creative_workspace"]["handoff"] == "verifier"
    assert result.trace.metadata["creative_workspace"]["selected_candidate_id"]
    assert result.trace.metadata["creative_workspace"]["rejected_candidate_ids"]
    assert result.trace.metadata["verifier"]["chosen_candidate_id"]
    assert result.trace.metadata["verifier"]["ranked_candidates"]
    assert result.trace.metadata["verifier"]["repair_reason"] in {"not_needed", "insufficient_grounding"}


def test_sandbox_path_records_shortlist_and_outcomes() -> None:
    backend = StrategyAwareBackend()
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))

    result = agent.run("What if we choose between two response plans?")

    assert result.decision == "sandbox"
    assert any(step.startswith("verifier_score:") for step in result.trace.steps)
    assert any(step.startswith("sandbox_shortlist:") for step in result.trace.steps)
    assert any(step.startswith("sandbox_branches:") for step in result.trace.steps)
    sandbox = result.trace.metadata["sandbox"]
    assert sandbox["shortlist_candidate_ids"]
    assert sandbox["selected_candidate_id"]
    assert sandbox["selected_candidate_id"] in sandbox["shortlist_candidate_ids"]
    assert sandbox["selected_strategy_id"]
    assert sandbox["outcomes"]
    assert result.trace.metadata["creative_workspace"]["selected_candidate_id"] == sandbox["selected_candidate_id"]
    assert result.trace.metadata["creative_workspace"]["rejected_candidate_ids"]
    assert result.trace.metadata["verifier"]["ranked_candidates"]
    assert result.trace.metadata["creative_workspace"]["handoff"] == "sandbox"


def test_explicit_consolidate_path_returns_clean_miss_without_repeated_pattern() -> None:
    agent = CognitionAgent(backend=BackendAdapter(backend=CapturingBackend()))

    result = agent.run("Please consolidate this pattern into a reusable skill.")

    assert result.decision == "consolidate"
    assert result.response == "No repeated successful pattern was found yet."
    assert "consolidated:false" in result.trace.steps
    assert result.consolidated_skill is None


def test_auto_consolidation_records_skill_without_breaking_response_path() -> None:
    backend = ConstantBackend("repeatable direct answer")
    agent = CognitionAgent(backend=BackendAdapter(backend=backend), min_skill_repetitions=2)
    agent.episodic.write(
        Episode(
            episode_id="ep-previous",
            prompt="Explain the style guidance.",
            response="repeatable direct answer",
            tags=["direct_answer"],
            metadata={"success": True},
        )
    )

    result = agent.run("Explain the style guidance again.")

    assert result.decision == "direct_answer"
    assert result.response == "repeatable direct answer"
    assert result.consolidated_skill is not None
    assert any(step.startswith("auto_consolidated:skill-direct_answer-2") for step in result.trace.steps)
    assert result.consolidated_skill.skill_id == "skill-direct_answer-2"


def test_empty_query_stays_explicit_and_does_not_crash() -> None:
    backend = CapturingBackend()
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))

    result = agent.run("")

    assert result.decision == "direct_answer"
    assert result.response == "captured"
    assert backend.prompts == [""]
    assert result.trace.rationale == "Empty query; defaulting to direct answer"
    assert result.trace.steps == ["route:direct_answer"]
    assert "creative_workspace" not in result.trace.metadata


def _extract_section(prompt: str, title: str) -> list[str]:
    lines = prompt.splitlines()
    in_section = False
    captured: list[str] = []
    for line in lines:
        if line == title:
            in_section = True
            continue
        if in_section and not line.strip():
            break
        if in_section:
            captured.append(line)
    return captured


def _extract_prefixed_line(lines: list[str], prefix: str) -> str | None:
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None
