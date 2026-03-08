from nanochat.cognition import ExplicitRouter


def test_cognition_smoke_import_and_route() -> None:
    router = ExplicitRouter()
    decision = router.decide("What is 2 + 2?")
    assert decision.action in {
        "direct_answer",
        "retrieve_memory",
        "creative_explore",
        "verify",
        "sandbox",
        "consolidate",
    }
    assert decision.rationale
