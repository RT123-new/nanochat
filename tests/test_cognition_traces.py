from nanochat.cognition.traces import TraceRecorder


def test_trace_recorder_build_increments_trace_ids_and_copies_steps() -> None:
    recorder = TraceRecorder()
    steps = ["route:verify"]

    first = recorder.build("first query", "verify", "requested verification", steps, {"confidence": 0.7})
    steps.append("after-build")
    second = recorder.build("second query", "direct_answer", "default", ["route:direct_answer"], {})

    assert first.trace_id == "trace-1"
    assert second.trace_id == "trace-2"
    assert first.steps == ["route:verify"]


def test_trace_recorder_deep_copies_nested_metadata() -> None:
    recorder = TraceRecorder()
    metadata = {
        "creative_workspace": {
            "candidates": [{"candidate_id": "c1", "strategy_id": "divergent_ideas"}],
            "model_summary": {"branch_disagreement": 0.44},
        }
    }

    trace = recorder.build("query", "creative_explore", "brainstorm requested", ["route:creative_explore"], metadata)
    metadata["creative_workspace"]["candidates"][0]["strategy_id"] = "changed"
    metadata["creative_workspace"]["model_summary"]["branch_disagreement"] = 0.99

    assert trace.metadata["creative_workspace"]["candidates"][0]["strategy_id"] == "divergent_ideas"
    assert trace.metadata["creative_workspace"]["model_summary"]["branch_disagreement"] == 0.44


def test_trace_recorder_preserves_nested_debug_payloads_in_stable_form() -> None:
    recorder = TraceRecorder()
    metadata = {
        "model_local_delib.graph_artifact": {
            "overview": {"active_sections": ["branch", "thought_graph"]},
            "branch": {"summary": {"branch_factor_used": 2.0}},
            "thought_graph": {"summary": {"thought_nodes_used": 4.0}},
        },
        "verifier": {
            "ranked_candidates": [
                {"candidate_id": "c1", "total_score": 0.91},
                {"candidate_id": "c2", "total_score": 0.42},
            ]
        },
    }

    trace = recorder.build("query", "verify", "verification requested", ["route:verify"], metadata)

    assert trace.metadata["model_local_delib.graph_artifact"]["overview"]["active_sections"] == [
        "branch",
        "thought_graph",
    ]
    assert trace.metadata["model_local_delib.graph_artifact"]["branch"]["summary"]["branch_factor_used"] == 2.0
    assert trace.metadata["verifier"]["ranked_candidates"][1]["candidate_id"] == "c2"
