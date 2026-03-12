import pytest

from nanochat.chat_format import render_messages_for_completion
from nanochat.cognition.backend import EngineBackend


class FakeTokenizer:
    def __init__(self) -> None:
        self._special_tokens = {
            "<|bos|>": 256,
            "<|user_start|>": 257,
            "<|user_end|>": 258,
            "<|assistant_start|>": 259,
            "<|assistant_end|>": 260,
            "<|python_start|>": 261,
            "<|python_end|>": 262,
            "<|output_start|>": 263,
            "<|output_end|>": 264,
        }

    def get_bos_token_id(self):
        return self._special_tokens["<|bos|>"]

    def encode_special(self, text):
        return self._special_tokens[text]

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        byte_ids = [token for token in ids if token < 256]
        return bytes(byte_ids).decode("utf-8")


class FakeEngine:
    def __init__(self) -> None:
        self.prompt_tokens = None
        self.last_generate_kwargs = None
        self.model = type(
            "FakeModel",
            (),
            {"config": type("FakeConfig", (), {"sequence_len": 64})()},
        )()

    def generate_batch(self, tokens, **kwargs):
        self.prompt_tokens = tokens
        self.last_generate_kwargs = kwargs
        response = list("ok".encode("utf-8"))
        return [tokens + response], [[0] * len(tokens) + [1] * len(response)]


def test_engine_backend_uses_chat_serialization_and_decodes_response() -> None:
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    backend = EngineBackend(
        engine=engine,
        tokenizer=tokenizer,
        system_prompt="Be terse.",
        max_tokens=8,
        temperature=0.0,
        top_k=1,
    )

    response = backend.generate("Hello there")

    expected_tokens = render_messages_for_completion(
        tokenizer,
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello there"},
        ],
        max_tokens=64,
    )

    assert engine.prompt_tokens == expected_tokens
    assert response == "ok"


def test_engine_backend_captures_local_deliberation_stats_metadata() -> None:
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    engine.model.last_deliberation_stats = [
        {
            "layer_idx": 0,
            "agreement": 0.8,
            "mean_branch_score": 0.5,
            "branch_factor_used": 2,
            "fraction_tokens_branched": 0.5,
            "hierarchy_levels_used": 2,
            "mean_hierarchy_feedback_norm": 0.4,
            "scratch_slots_used": 3,
            "mean_scratch_read_weight": 0.2,
            "mean_scratch_refine_norm": 0.1,
            "mean_branch_to_scratch_weight": 0.3,
            "scratch_reset_ok": 1.0,
            "scratch_summary_vector": [0.1, 0.2, 0.3],
            "mean_scratch_summary_norm": 0.25,
            "halted_token_fraction": 0.6,
            "mean_steps_taken": 1.5,
            "mean_branch_disagreement": 0.4,
            "mean_branch_consensus_weight": 0.7,
            "mean_branch_verifier_score": 0.6,
            "mean_branch_entropy": 0.2,
            "branch_consensus_used": 1.0,
            "phrase_nodes_used": 3,
            "span_nodes_used": 2,
            "sequence_summary_used": 1,
            "mean_upward_message_norm": 0.9,
            "mean_downward_message_norm": 0.8,
            "mean_scale_gate": 0.95,
            "hierarchy_depth_used": 3,
            "thought_nodes_used": 4,
            "mean_thought_degree": 1.2,
            "mean_token_to_thought_weight": 0.33,
            "mean_thought_to_token_weight": 0.44,
            "mean_thought_update_norm": 0.55,
            "thought_graph_steps_used": 2,
            "global_anchors_used": 2,
            "mean_anchor_read_weight": 0.11,
            "mean_anchor_write_weight": 0.22,
            "mean_anchor_norm": 0.66,
            "mean_alignment_norm": 0.12,
            "mean_cohesion_norm": 0.23,
            "mean_separation_norm": 0.34,
            "mean_flocking_total_norm": 0.69,
            "flocking_neighbor_count": 5,
            "fraction_flocking_tokens_active": 0.8,
        }
    ]
    backend = EngineBackend(engine=engine, tokenizer=tokenizer)

    response = backend.generate("Hello there")

    assert isinstance(response, str)
    assert response == "ok"
    assert backend.last_generation_metadata is not None
    assert backend.last_generation_metadata["local_deliberation_stats"][0]["agreement"] == 0.8
    assert backend.last_generation_metadata["model_local_delib.branch"]["branch_factor_used"] == 2.0
    assert backend.last_generation_metadata["model_local_delib.hierarchy"]["hierarchy_levels_used"] == 2.0
    assert backend.last_generation_metadata["model_local_delib.scratchpad"]["scratch_slots_used"] == 3.0
    assert backend.last_generation_metadata["model_local_delib.scratchpad"]["mean_scratch_refine_norm"] == 0.1
    assert backend.last_generation_metadata["model_local_delib.scratchpad"]["mean_branch_to_scratch_weight"] == 0.3
    assert backend.last_generation_metadata["model_local_delib.scratchpad_summaries"] == [{"layer_idx": 0, "summary": [0.1, 0.2, 0.3]}]
    assert backend.last_generation_metadata["model_local_delib.adaptive_halt"]["halted_token_fraction"] == 0.6
    assert backend.last_generation_metadata["model_local_delib.thought_summaries.branch_consensus"] == {
        "layer_count": 1,
        "branch_consensus_used": 1.0,
        "branch_factor_used": 2.0,
        "fraction_tokens_branched": 0.5,
        "mean_branch_consensus_weight": 0.7,
        "mean_branch_disagreement": 0.4,
        "mean_branch_entropy": 0.2,
        "mean_branch_verifier_score": 0.6,
    }
    assert backend.last_generation_metadata["model_local_delib.thought_summaries.deep_hierarchy"] == {
        "layer_count": 1,
        "hierarchy_depth_used": 3.0,
        "mean_downward_message_norm": 0.8,
        "mean_scale_gate": 0.95,
        "mean_upward_message_norm": 0.9,
        "phrase_nodes_used": 3.0,
        "sequence_summary_used": 1.0,
        "span_nodes_used": 2.0,
    }
    scratch_summary = backend.last_generation_metadata["model_local_delib.thought_summaries.scratch"]
    assert scratch_summary["layer_count"] == 1
    assert scratch_summary["summary_dim"] == 3
    assert scratch_summary["mean_summary_value"] == pytest.approx((0.1 + 0.2 + 0.3) / 3)
    assert scratch_summary["mean_summary_abs"] == pytest.approx((abs(0.1) + abs(0.2) + abs(0.3)) / 3)
    assert scratch_summary["max_summary_abs"] == 0.3
    assert scratch_summary["mean_summary_norm"] == pytest.approx((0.1**2 + 0.2**2 + 0.3**2) ** 0.5)
    assert scratch_summary["mean_scratch_summary_norm"] == 0.25
    assert scratch_summary["scratch_slots_used"] == 3.0
    assert backend.last_generation_metadata["model_local_delib.thought_summaries.thought_graph"] == {
        "layer_count": 1,
        "mean_thought_degree": 1.2,
        "mean_thought_to_token_weight": 0.44,
        "mean_thought_update_norm": 0.55,
        "mean_token_to_thought_weight": 0.33,
        "thought_graph_steps_used": 2.0,
        "thought_nodes_used": 4.0,
    }
    assert backend.last_generation_metadata["model_local_delib.thought_summaries.global_anchors"] == {
        "layer_count": 1,
        "global_anchors_used": 2.0,
        "mean_anchor_norm": 0.66,
        "mean_anchor_read_weight": 0.11,
        "mean_anchor_write_weight": 0.22,
    }
    assert backend.last_generation_metadata["model_local_delib.thought_summaries.flocking"] == {
        "layer_count": 1,
        "flocking_neighbor_count": 5.0,
        "fraction_flocking_tokens_active": 0.8,
        "mean_alignment_norm": 0.12,
        "mean_cohesion_norm": 0.23,
        "mean_flocking_total_norm": 0.69,
        "mean_separation_norm": 0.34,
    }


def test_engine_backend_omits_thought_summaries_when_mechanisms_are_unavailable() -> None:
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    engine.model.last_deliberation_stats = [
        {
            "layer_idx": 0,
            "agreement": 0.8,
            "branch_factor_used": 2,
            "hierarchy_levels_used": 2,
            "scratch_slots_used": 3,
            "halted_token_fraction": 0.6,
            "mean_steps_taken": 1.5,
            "mean_branch_consensus_weight": 0.0,
            "branch_consensus_used": 0.0,
            "phrase_nodes_used": 0,
            "span_nodes_used": 0,
            "sequence_summary_used": 0,
            "hierarchy_depth_used": 0,
            "thought_nodes_used": 0,
            "thought_graph_steps_used": 0,
            "global_anchors_used": 0,
            "mean_alignment_norm": 0.0,
            "mean_cohesion_norm": 0.0,
            "mean_separation_norm": 0.0,
            "mean_flocking_total_norm": 0.0,
            "fraction_flocking_tokens_active": 0.0,
        }
    ]
    backend = EngineBackend(engine=engine, tokenizer=tokenizer)

    _ = backend.generate("Hello there")

    assert backend.last_generation_metadata is not None
    assert "model_local_delib.thought_summaries.branch_consensus" not in backend.last_generation_metadata
    assert "model_local_delib.thought_summaries.deep_hierarchy" not in backend.last_generation_metadata
    assert "model_local_delib.thought_summaries.scratch" not in backend.last_generation_metadata
    assert "model_local_delib.thought_summaries.thought_graph" not in backend.last_generation_metadata
    assert "model_local_delib.thought_summaries.global_anchors" not in backend.last_generation_metadata
    assert "model_local_delib.thought_summaries.flocking" not in backend.last_generation_metadata


def test_engine_backend_generate_accepts_extra_generation_kwargs() -> None:
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    backend = EngineBackend(engine=engine, tokenizer=tokenizer)

    _ = backend.generate("Hello", local_delib=True, local_delib_branch_factor=2)

    assert engine.last_generate_kwargs is not None
    assert engine.last_generate_kwargs["local_delib"] is True
    assert engine.last_generate_kwargs["local_delib_branch_factor"] == 2
