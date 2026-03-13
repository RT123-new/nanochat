from nanochat.cognition.creative import CreativeCandidate
from nanochat.cognition.sandbox import LightweightSandbox
from nanochat.cognition.verifier import RankedCandidate


def _candidate(candidate_id: str, *, strategy_id: str, response: str) -> CreativeCandidate:
    return CreativeCandidate(
        candidate_id=candidate_id,
        strategy_id=strategy_id,
        strategy_label=strategy_id,
        response=response,
        prompt="",
        rationale="",
    )


def _ranked(candidate_id: str, *, strategy_id: str, total_score: float, candidate: str) -> RankedCandidate:
    return RankedCandidate(
        candidate_id=candidate_id,
        strategy_id=strategy_id,
        candidate=candidate,
        total_score=total_score,
        relevance_score=0.0,
        usefulness_score=0.0,
        diversity_score=0.0,
        repairability_score=0.0,
        strategy_fit_score=0.0,
        rationale="",
        repair_hint="",
    )


def test_sandbox_scores_use_relevance_verifier_support_and_branch_bonus() -> None:
    sandbox = LightweightSandbox()
    candidates = [
        _candidate("safe", strategy_id="conservative_answer", response="grounded support answer"),
        _candidate("branch", strategy_id="branch_resolution", response="support answer branch option"),
    ]
    verifier_ranked = [
        _ranked("safe", strategy_id="conservative_answer", total_score=0.8, candidate="grounded support answer"),
        _ranked("branch", strategy_id="branch_resolution", total_score=0.6, candidate="support answer branch option"),
    ]

    report = sandbox.explore(
        "support answer",
        candidates,
        verifier_ranked=verifier_ranked,
        support_profile={"support_terms": ["support", "answer", "branch"]},
    )
    outcomes_by_id = {outcome.candidate_id: outcome for outcome in report.outcomes}

    assert outcomes_by_id["safe"].score == 0.79
    assert outcomes_by_id["safe"].rationale == "relevance=1.00; verifier=0.80; support=0.67; branch_bonus=0.00"
    assert outcomes_by_id["branch"].score == 0.93
    assert outcomes_by_id["branch"].rationale == "relevance=1.00; verifier=0.60; support=1.00; branch_bonus=0.15"


def test_sandbox_can_flip_pure_verifier_order_when_branch_bonus_and_support_justify_it() -> None:
    sandbox = LightweightSandbox()
    candidates = [
        _candidate("safe", strategy_id="conservative_answer", response="grounded support answer"),
        _candidate("branch", strategy_id="branch_resolution", response="support answer branch option"),
    ]
    verifier_ranked = [
        _ranked("safe", strategy_id="conservative_answer", total_score=0.8, candidate="grounded support answer"),
        _ranked("branch", strategy_id="branch_resolution", total_score=0.6, candidate="support answer branch option"),
    ]

    report = sandbox.explore(
        "support answer",
        candidates,
        verifier_ranked=verifier_ranked,
        support_profile={"support_terms": ["support", "answer", "branch"]},
    )

    assert verifier_ranked[0].candidate_id == "safe"
    assert report.selected is not None
    assert report.selected.candidate_id == "branch"
    assert report.outcomes[0].score > report.outcomes[1].score


def test_sandbox_branch_bonus_applies_to_divergent_and_branch_resolution_strategies() -> None:
    sandbox = LightweightSandbox()
    candidates = [
        _candidate("divergent", strategy_id="divergent_ideas", response="support answer alternatives"),
        _candidate("branch", strategy_id="branch_resolution", response="support answer branch option"),
        _candidate("plain", strategy_id="conservative_answer", response="support answer"),
    ]

    report = sandbox.explore(
        "support answer",
        candidates,
        verifier_ranked=[
            _ranked("divergent", strategy_id="divergent_ideas", total_score=0.5, candidate="support answer alternatives"),
            _ranked("branch", strategy_id="branch_resolution", total_score=0.5, candidate="support answer branch option"),
            _ranked("plain", strategy_id="conservative_answer", total_score=0.5, candidate="support answer"),
        ],
        support_profile={"support_terms": ["support", "answer"]},
    )
    outcomes_by_id = {outcome.candidate_id: outcome for outcome in report.outcomes}

    assert "branch_bonus=0.15" in outcomes_by_id["divergent"].rationale
    assert "branch_bonus=0.15" in outcomes_by_id["branch"].rationale
    assert "branch_bonus=0.00" in outcomes_by_id["plain"].rationale


def test_sandbox_empty_shortlist_returns_no_selected_outcome() -> None:
    sandbox = LightweightSandbox()

    report = sandbox.explore("support answer", [])

    assert report.outcomes == []
    assert report.selected is None
