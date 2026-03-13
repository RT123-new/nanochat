from nanochat.cognition.creative import CreativeCandidate
from nanochat.cognition.verifier import VerifierWorkspace


def _candidate(candidate_id: str, *, strategy_id: str, response: str) -> CreativeCandidate:
    return CreativeCandidate(
        candidate_id=candidate_id,
        strategy_id=strategy_id,
        strategy_label=strategy_id,
        response=response,
        prompt="",
        rationale="",
    )


def test_verifier_scores_empty_candidate_as_unusable() -> None:
    verifier = VerifierWorkspace()

    ranked = verifier.rank("verify this answer", [_candidate("empty", strategy_id="divergent_ideas", response="")])

    assert len(ranked) == 1
    assert ranked[0].candidate_id == "empty"
    assert ranked[0].total_score == 0.0
    assert ranked[0].relevance_score == 0.0
    assert ranked[0].usefulness_score == 0.0
    assert ranked[0].diversity_score == 0.0
    assert ranked[0].repairability_score == 0.0
    assert ranked[0].strategy_fit_score == 0.0
    assert ranked[0].rationale == "empty candidate"
    assert ranked[0].repair_hint == "regenerate with clearer grounding"


def test_verifier_grounded_candidate_outranks_verbose_one_and_surfaces_subscores() -> None:
    verifier = VerifierWorkspace()
    candidates = [
        _candidate(
            "grounded",
            strategy_id="memory_grounded",
            response="answer with support evidence and citations",
        ),
        _candidate(
            "verbose",
            strategy_id="divergent_ideas",
            response="this is a very verbose response with many extra words but little grounding",
        ),
        _candidate(
            "repairable",
            strategy_id="conservative_answer",
            response="answer support",
        ),
        _candidate(
            "verify-fit",
            strategy_id="branch_resolution",
            response="answer support evidence branch comparison",
        ),
    ]

    ranked = verifier.rank(
        "answer with support evidence",
        candidates,
        route="verify",
        support_profile={"support_terms": ["support", "evidence", "citations"]},
    )
    ranked_by_id = {item.candidate_id: item for item in ranked}

    assert ranked[0].candidate_id == "grounded"
    assert ranked_by_id["grounded"].relevance_score > ranked_by_id["verbose"].relevance_score
    assert ranked_by_id["grounded"].usefulness_score > ranked_by_id["verbose"].usefulness_score
    assert ranked_by_id["verbose"].diversity_score >= ranked_by_id["grounded"].diversity_score
    assert ranked_by_id["repairable"].repairability_score == 0.4
    assert ranked_by_id["verify-fit"].strategy_fit_score == 1.0
    assert "relevance=" in ranked_by_id["grounded"].rationale
    assert "usefulness=" in ranked_by_id["grounded"].rationale
    assert "diversity=" in ranked_by_id["grounded"].rationale
    assert "repairability=" in ranked_by_id["grounded"].rationale
    assert "strategy_fit=" in ranked_by_id["grounded"].rationale


def test_verifier_select_can_require_repair_when_grounding_is_weak() -> None:
    verifier = VerifierWorkspace()

    selection = verifier.select(
        "verify the answer with evidence",
        [_candidate("weak", strategy_id="divergent_ideas", response="loose option text")],
        route="verify",
    )

    assert selection.chosen.candidate_id == "weak"
    assert selection.repair_required is True
    assert selection.repair_reason == "insufficient_grounding"
    assert selection.chosen.repair_hint in {
        "tighten the answer around the request",
        "ground the answer in retrieved support",
    }


def test_verifier_ranking_is_deterministic_for_the_same_inputs() -> None:
    verifier = VerifierWorkspace()
    candidates = [
        _candidate("a", strategy_id="branch_resolution", response="answer support branch evidence"),
        _candidate("b", strategy_id="memory_grounded", response="answer support evidence citations"),
        _candidate("c", strategy_id="divergent_ideas", response="answer alternative framing"),
    ]

    first = verifier.rank(
        "answer support evidence",
        candidates,
        route="verify",
        support_profile={"support_terms": ["support", "evidence"]},
    )
    second = verifier.rank(
        "answer support evidence",
        candidates,
        route="verify",
        support_profile={"support_terms": ["support", "evidence"]},
    )

    assert [(item.candidate_id, item.total_score) for item in first] == [
        (item.candidate_id, item.total_score) for item in second
    ]
