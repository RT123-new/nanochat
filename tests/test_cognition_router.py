from nanochat.cognition.router import ExplicitRouter


def test_router_routes_memory_queries() -> None:
    router = ExplicitRouter()
    decision = router.decide("Can you recall what I said previously?")
    assert decision.action == "retrieve_memory"
    assert decision.rationale


def test_router_routes_creative_queries() -> None:
    router = ExplicitRouter()
    assert router.decide("Brainstorm ideas for a mascot").action == "creative_explore"


def test_router_routes_verify_queries() -> None:
    router = ExplicitRouter()
    assert router.decide("Please verify this proof").action == "verify"


def test_router_routes_sandbox_queries() -> None:
    router = ExplicitRouter()
    assert router.decide("What if we simulate this in a sandbox?").action == "sandbox"


def test_router_routes_consolidate_queries() -> None:
    router = ExplicitRouter()
    assert router.decide("Find repeated pattern and consolidate").action == "consolidate"


def test_router_defaults_direct_answer_and_handles_empty() -> None:
    router = ExplicitRouter()
    assert router.decide("What is 2 + 2?").action == "direct_answer"
    empty = router.decide("   ")
    assert empty.action == "direct_answer"
    assert empty.confidence < 0.5
