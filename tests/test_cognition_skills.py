from nanochat.cognition.schemas import SkillArtifact
from nanochat.cognition.skills import SkillRegistry


def _skill(skill_id: str, *, name: str, trigger: str, procedure: list[str]) -> SkillArtifact:
    return SkillArtifact(
        skill_id=skill_id,
        name=name,
        trigger=trigger,
        procedure=procedure,
    )


def test_discover_ranks_skills_by_overlap_with_name_trigger_and_procedure_terms() -> None:
    registry = SkillRegistry()
    registry.register(
        _skill(
            "summary-full",
            name="Summarization checklist",
            trigger="summarization",
            procedure=["extract bullets", "condense with citations"],
        )
    )
    registry.register(
        _skill(
            "summary-lite",
            name="Summary shortcut",
            trigger="summarization",
            procedure=["condense"],
        )
    )
    registry.register(
        _skill(
            "router-debug",
            name="Router debugging guide",
            trigger="routing",
            procedure=["inspect route rationale"],
        )
    )

    matches = registry.discover("Need a summarization checklist to extract bullets and condense.", limit=3)

    assert [match.skill.skill_id for match in matches] == ["summary-full", "summary-lite"]
    assert matches[0].score > matches[1].score


def test_best_for_returns_top_match_only_when_overlap_exists() -> None:
    registry = SkillRegistry()
    registry.register(
        _skill(
            "summary-full",
            name="Summarization checklist",
            trigger="summarization",
            procedure=["extract bullets", "condense with citations"],
        )
    )
    registry.register(
        _skill(
            "router-debug",
            name="Router debugging guide",
            trigger="routing",
            procedure=["inspect route rationale"],
        )
    )

    best = registry.best_for("Can you use the summarization checklist to extract bullets?")

    assert best is not None
    assert best.skill_id == "summary-full"


def test_discover_returns_empty_and_best_for_none_when_no_match_exists() -> None:
    registry = SkillRegistry()
    registry.register(
        _skill(
            "summary-full",
            name="Summarization checklist",
            trigger="summarization",
            procedure=["extract bullets", "condense with citations"],
        )
    )

    assert registry.discover("gpu kernel launch tuning") == []
    assert registry.best_for("gpu kernel launch tuning") is None


def test_discover_ordering_is_deterministic_for_equal_overlap() -> None:
    registry = SkillRegistry()
    registry.register(
        _skill(
            "skill-b",
            name="Summarization checklist",
            trigger="summarization",
            procedure=["extract bullets", "condense"],
        )
    )
    registry.register(
        _skill(
            "skill-a",
            name="Summarization checklist",
            trigger="summarization",
            procedure=["extract bullets", "condense"],
        )
    )

    matches = registry.discover("summarization checklist extract bullets condense", limit=2)

    assert [match.skill.skill_id for match in matches] == ["skill-a", "skill-b"]
