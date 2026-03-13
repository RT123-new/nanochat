from __future__ import annotations

from nanochat.cognition.backend import BackendAdapter, build_local_delib_namespaced_metadata, summarize_local_delib_for_creative_policy
from nanochat.cognition.creative import CreativeWorkspace


class StrategyBackend:
    def __init__(self, *, metadata_by_strategy: dict[str, dict[str, object]] | None = None) -> None:
        self.prompts: list[str] = []
        self.last_generation_metadata: dict[str, object] | None = None
        self.metadata_by_strategy = dict(metadata_by_strategy or {})

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        strategy_id = _extract_prefixed_line(_extract_section(prompt, "Creative strategy:"), "- id:")
        self.last_generation_metadata = self.metadata_by_strategy.get(strategy_id or "")
        return f"draft::{strategy_id}"


def _metadata_from_stats(stats: list[dict[str, object]]) -> dict[str, object]:
    return {
        "local_deliberation_stats": stats,
        **build_local_delib_namespaced_metadata(stats),  # type: ignore[arg-type]
    }


def test_creative_plan_route_only_picks_conservative_and_divergent_strategies() -> None:
    workspace = CreativeWorkspace(BackendAdapter(StrategyBackend()))

    plan = workspace.plan(query="Brainstorm ideas for memory routing", route="creative_explore", limit=4)

    assert plan.strategy_order == ["conservative_answer", "divergent_ideas"]
    assert plan.signals["brainstorming"] is True
    assert plan.support_profile == {
        "memory_heavy": False,
        "episodic_count": 0,
        "semantic_count": 0,
        "skill_count": 0,
        "support_terms": [],
    }


def test_creative_plan_adds_memory_branch_and_recombination_signals_and_limits_cleanly() -> None:
    workspace = CreativeWorkspace(BackendAdapter(StrategyBackend()))
    model_summary = summarize_local_delib_for_creative_policy(
        _metadata_from_stats(
            [
                {
                    "layer_idx": 1,
                    "branch_factor_used": 2,
                    "mean_branch_disagreement": 0.44,
                    "mean_branch_consensus_weight": 0.61,
                    "branch_consensus_used": 1.0,
                    "scratch_slots_used": 2,
                    "scratch_summary_vector": [0.2, 0.4],
                    "mean_scratch_summary_norm": 0.31,
                    "thought_nodes_used": 5.0,
                    "thought_graph_steps_used": 2.0,
                    "phrase_nodes_used": 3,
                    "span_nodes_used": 2,
                    "sequence_summary_used": 1,
                    "hierarchy_depth_used": 3,
                    "global_anchors_used": 2,
                }
            ]
        )
    )

    full = workspace.plan(
        query="Brainstorm ideas for summarization",
        route="creative_explore",
        support_profile={"memory_heavy": True, "episodic_count": 2, "semantic_count": 1, "skill_count": 1},
        model_summary=model_summary,
        limit=5,
    )
    limited = workspace.plan(
        query="Brainstorm ideas for summarization",
        route="creative_explore",
        support_profile={"memory_heavy": True},
        model_summary=model_summary,
        limit=3,
    )

    assert full.strategy_order == [
        "conservative_answer",
        "divergent_ideas",
        "memory_grounded",
        "branch_resolution",
        "recombination",
    ]
    assert limited.strategy_order == ["conservative_answer", "divergent_ideas", "memory_grounded"]
    assert len(set(full.strategy_order)) == len(full.strategy_order)
    assert full.signals["branch_pressure"] is True
    assert full.signals["synthesis_pressure"] is True


def test_generate_candidates_includes_explicit_prompt_sections_only_when_relevant() -> None:
    backend = StrategyBackend()
    workspace = CreativeWorkspace(BackendAdapter(backend))

    run = workspace.generate_candidates(
        query="Brainstorm ideas for memory routing",
        base_prompt="BASE PROMPT",
        route="creative_explore",
        limit=2,
    )

    assert [candidate.strategy_id for candidate in run.candidates] == ["conservative_answer", "divergent_ideas"]
    assert [candidate.response for candidate in run.candidates] == [
        "draft::conservative_answer",
        "draft::divergent_ideas",
    ]
    assert "BASE PROMPT" in backend.prompts[0]
    assert "Creative strategy:" in backend.prompts[0]
    assert "- id: conservative_answer" in backend.prompts[0]
    assert "- label: Conservative answer" in backend.prompts[0]
    assert "- guidance:" in backend.prompts[0]
    assert "- rationale:" in backend.prompts[0]
    assert "Creative support profile:" not in backend.prompts[0]
    assert "Model deliberation guidance:" not in backend.prompts[0]
    assert all(candidate.metadata["model_summary_used"] is False for candidate in run.candidates)
    assert all(candidate.metadata["model_focus"] == [] for candidate in run.candidates)


def test_generate_candidates_records_model_guidance_focus_and_unique_strategy_exploration() -> None:
    stats = [
        {
            "layer_idx": 1,
            "branch_factor_used": 3,
            "mean_branch_disagreement": 0.52,
            "mean_branch_consensus_weight": 0.7,
            "branch_consensus_used": 1.0,
            "scratch_slots_used": 2,
            "scratch_summary_vector": [0.1, 0.2, 0.3],
            "mean_scratch_summary_norm": 0.25,
            "thought_nodes_used": 4.0,
            "thought_graph_steps_used": 2.0,
            "phrase_nodes_used": 3,
            "span_nodes_used": 2,
            "sequence_summary_used": 1,
            "hierarchy_depth_used": 3,
            "global_anchors_used": 2,
            "mean_anchor_read_weight": 0.4,
        }
    ]
    metadata = _metadata_from_stats(stats)
    backend = StrategyBackend(
        metadata_by_strategy={
            "conservative_answer": metadata,
            "memory_grounded": metadata,
            "branch_resolution": metadata,
            "recombination": metadata,
        }
    )
    workspace = CreativeWorkspace(BackendAdapter(backend))
    initial_summary = summarize_local_delib_for_creative_policy(metadata)

    run = workspace.generate_candidates(
        query="Verify the summarization answer",
        base_prompt="VERIFY BASE",
        route="verify",
        support_profile={
            "memory_heavy": True,
            "episodic_count": 1,
            "semantic_count": 1,
            "skill_count": 1,
            "support_terms": ["support", "memory"],
        },
        initial_model_summary=initial_summary,
        limit=4,
    )

    assert [candidate.strategy_id for candidate in run.candidates] == [
        "conservative_answer",
        "memory_grounded",
        "branch_resolution",
        "recombination",
    ]
    assert len({candidate.strategy_id for candidate in run.candidates}) == len(run.candidates)
    assert backend.prompts[0].startswith("VERIFY BASE")
    assert "Creative support profile:" in backend.prompts[0]
    assert "- memory_heavy: true" in backend.prompts[0]
    assert "- episodic_count: 1" in backend.prompts[0]
    assert "- semantic_count: 1" in backend.prompts[0]
    assert "- skill_count: 1" in backend.prompts[0]
    assert "- support_terms: support, memory" in backend.prompts[0]
    assert "Model deliberation guidance:" in backend.prompts[0]
    assert "- branch_disagreement: 0.52" in backend.prompts[0]
    assert "- branch_consensus_used: 1.00" in backend.prompts[0]
    assert "- scratch_summary_dim: 3.00" in backend.prompts[0]
    assert "- thought_nodes_used: 4.00" in backend.prompts[0]
    assert "- hierarchy_depth_used: 3.00" in backend.prompts[0]
    assert "- global_anchors_used: 2.00" in backend.prompts[0]
    assert all(candidate.metadata["model_summary_used"] is True for candidate in run.candidates)
    assert all(
        candidate.metadata["model_focus"] == [
            "branch_disagreement",
            "scratch_summary",
            "thought_graph",
            "hierarchy",
            "anchors",
        ]
        for candidate in run.candidates
    )


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
