from __future__ import annotations

from nanochat.cognition.agent import CognitionAgent, _compose_prompt
from nanochat.cognition.backend import BackendAdapter
from nanochat.cognition.memory import RankedMemory
from nanochat.cognition.schemas import Episode, MemoryItem, SkillArtifact


class SupportSensitiveBackend:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.last_generation_metadata = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        if "Relevant skill:" in prompt:
            return "skill-guided answer"
        if "Relevant semantic memory:" in prompt:
            return "semantic-guided answer"
        if "Relevant episodic memory:" in prompt:
            return "episodic-guided answer"
        return "plain answer"


def test_compose_prompt_orders_support_sections_before_final_user_request() -> None:
    skill = SkillArtifact(
        skill_id="skill-summarization",
        name="Summarization checklist",
        trigger="summarization",
        procedure=["extract bullets", "condense with citations"],
    )
    semantic_hit = RankedMemory(
        item=MemoryItem(
            item_id="semantic-style",
            kind="semantic",
            content="Summarization style: terse bullet answers with citations.",
        ),
        relevance=1.0,
        recency=1.0,
        combined_score=1.0,
    )
    episode = Episode(
        episode_id="ep-style",
        prompt="Summarize the project update",
        response="Use terse bullet summaries with citations.",
        tags=["summarization"],
        metadata={"success": True, "trigger": "summarization"},
    )

    prompt = _compose_prompt(
        query="Please summarize this draft.",
        episodes=[episode],
        semantic_hits=[semantic_hit],
        reused_skill=skill,
    )

    skill_idx = prompt.index("Relevant skill:")
    semantic_idx = prompt.index("Relevant semantic memory:")
    episodic_idx = prompt.index("Relevant episodic memory:")
    request_idx = prompt.index("User request:\nPlease summarize this draft.")

    assert skill_idx < semantic_idx < episodic_idx < request_idx
    assert "- skill_id: skill-summarization" in prompt
    assert "- name: Summarization checklist" in prompt
    assert "- trigger: summarization" in prompt
    assert "- procedure:" in prompt
    assert "- extract bullets" in prompt
    assert "- condense with citations" in prompt


def test_compose_prompt_returns_plain_query_when_no_support_exists() -> None:
    assert _compose_prompt(query="Explain the project status.") == "Explain the project status."


def test_agent_prompt_omits_empty_support_sections_when_none_available() -> None:
    backend = SupportSensitiveBackend()
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))

    result = agent.run("Explain the project status.")

    assert result.response == "plain answer"
    assert backend.prompts == ["Explain the project status."]
    assert "Relevant skill:" not in backend.prompts[0]
    assert "Relevant semantic memory:" not in backend.prompts[0]
    assert "Relevant episodic memory:" not in backend.prompts[0]


def test_skill_support_injection_changes_backend_output() -> None:
    backend = SupportSensitiveBackend()
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))
    agent.registry.register(
        SkillArtifact(
            skill_id="skill-summarization",
            name="Summarization checklist",
            trigger="summarization",
            procedure=["extract bullets", "condense with citations"],
        )
    )

    result = agent.run("Please summarize this draft.")

    assert result.response == "skill-guided answer"
    assert "Relevant skill:" in backend.prompts[-1]
    assert backend.prompts[-1].endswith("User request:\nPlease summarize this draft.")


def test_semantic_support_injection_changes_backend_output() -> None:
    backend = SupportSensitiveBackend()
    agent = CognitionAgent(backend=BackendAdapter(backend=backend))
    agent.semantic.write(
        MemoryItem(
            item_id="semantic-style",
            kind="semantic",
            content="Summarization style: terse bullet answers with citations.",
        )
    )

    result = agent.run("Please summarize this draft.")

    assert result.response == "semantic-guided answer"
    assert "Relevant semantic memory:" in backend.prompts[-1]
    assert backend.prompts[-1].endswith("User request:\nPlease summarize this draft.")


def test_episodic_support_injection_changes_backend_output_for_paraphrase() -> None:
    backend = SupportSensitiveBackend()
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
    assert result.response == "episodic-guided answer"
    assert "Relevant episodic memory:" in backend.prompts[-1]
    assert backend.prompts[-1].endswith("User request:\nPlease summarize this draft for me.")
