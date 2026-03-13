from dataclasses import dataclass

import pytest
import torch

from nanochat.chat_format import render_messages_for_completion
from nanochat.cognition.backend import (
    BackendAdapter,
    EngineBackend,
    LocalDelibRuntimeOverrideError,
    summarize_local_delib_for_creative_policy,
)
from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig


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


def _patch_flash_attention(monkeypatch):
    def fake_flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
        return torch.zeros_like(q)

    def fake_flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None, causal=False, window_size=(-1, -1)):
        if k is not None and v is not None and cache_seqlens is not None:
            pos = int(cache_seqlens[0].item())
            t = q.size(1)
            k_cache[:, pos:pos+t, :, :] = k
            v_cache[:, pos:pos+t, :, :] = v
        return torch.zeros_like(q)

    monkeypatch.setattr(
        "nanochat.gpt.flash_attn",
        type(
            "FakeFlashAttention",
            (),
            {
                "flash_attn_func": staticmethod(fake_flash_attn_func),
                "flash_attn_with_kvcache": staticmethod(fake_flash_attn_with_kvcache),
            },
        )(),
    )


class TinyTokenizer:
    def __init__(self):
        self._special = {
            "<|bos|>": 16,
            "<|user_start|>": 17,
            "<|user_end|>": 18,
            "<|assistant_start|>": 19,
            "<|assistant_end|>": 20,
            "<|python_start|>": 21,
            "<|python_end|>": 22,
            "<|output_start|>": 23,
            "<|output_end|>": 24,
        }

    def encode_special(self, s):
        return self._special[s]

    def get_bos_token_id(self):
        return self._special["<|bos|>"]

    def encode(self, s):
        return [ord(c) % 16 for c in s]

    def decode(self, tokens):
        return "".join(chr((t % 26) + 97) for t in tokens if t < 16)


def _tiny_config(**kwargs):
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=8,
        window_pattern="L",
    )
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


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


class CapturingBackend:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.calls.append((prompt, kwargs))
        return "captured"


@dataclass(frozen=True)
class FrozenConfig:
    sequence_len: int = 64
    local_delib: bool = False


def test_backend_adapter_merges_default_and_call_time_kwargs() -> None:
    backend = CapturingBackend()
    adapter = BackendAdapter(
        backend=backend,
        default_kwargs={"temperature": 0.1, "seed": 7, "top_k": 20},
    )

    response = adapter.run("Prompt", temperature=0.5, foo="bar")

    assert response == "captured"
    assert backend.calls == [
        (
            "Prompt",
            {"temperature": 0.5, "seed": 7, "top_k": 20, "foo": "bar"},
        )
    ]


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


def test_engine_backend_leaves_metadata_empty_when_no_local_delib_stats_exist() -> None:
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    backend = EngineBackend(engine=engine, tokenizer=tokenizer)

    response = backend.generate("Hello there")

    assert response == "ok"
    assert backend.last_generation_metadata is None


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
    graph_artifact = backend.last_generation_metadata["model_local_delib.graph_artifact"]
    assert graph_artifact["overview"] == {
        "trace_version": 1,
        "layer_count": 1,
        "active_layer_count": 1,
        "active_layers": [0],
        "active_sections": ["branch", "thought_graph", "hierarchy", "scratch", "anchors", "compute", "flocking"],
        "section_layer_counts": {
            "branch": 1,
            "thought_graph": 1,
            "hierarchy": 1,
            "scratch": 1,
            "anchors": 1,
            "compute": 1,
            "flocking": 1,
        },
        "scratch_summary_layers": 1,
    }
    assert graph_artifact["branch"]["summary"]["consensus_active_layers"] == 1
    assert graph_artifact["branch"]["summary"]["verifier_active_layers"] == 1
    assert graph_artifact["thought_graph"]["summary"]["degree_pattern"] == "sparse"
    assert graph_artifact["hierarchy"]["summary"]["scale_layers"] == {"phrase": 1, "span": 1, "sequence": 1}
    assert graph_artifact["scratch"]["summary"]["has_exported_summaries"] is True
    assert graph_artifact["scratch"]["layers"][0]["summary_dim"] == 3
    assert graph_artifact["anchors"]["summary"]["write_active_layers"] == 1
    assert graph_artifact["compute"]["summary"]["adaptive_halt_active_layers"] == 1
    assert graph_artifact["compute"]["layers"][0]["mean_steps_taken"] == 1.5
    assert graph_artifact["flocking"]["summary"]["flocking_active_layers"] == 1
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
    assert backend.last_generation_metadata["model_local_delib.graph_artifact"]["overview"]["active_sections"] == [
        "branch",
        "hierarchy",
        "scratch",
        "compute",
    ]
    assert "thought_graph" not in backend.last_generation_metadata["model_local_delib.graph_artifact"]
    assert "anchors" not in backend.last_generation_metadata["model_local_delib.graph_artifact"]
    assert "flocking" not in backend.last_generation_metadata["model_local_delib.graph_artifact"]
    assert "model_local_delib.thought_summaries.branch_consensus" not in backend.last_generation_metadata
    assert "model_local_delib.thought_summaries.deep_hierarchy" not in backend.last_generation_metadata
    assert "model_local_delib.thought_summaries.scratch" not in backend.last_generation_metadata
    assert "model_local_delib.thought_summaries.thought_graph" not in backend.last_generation_metadata
    assert "model_local_delib.thought_summaries.global_anchors" not in backend.last_generation_metadata
    assert "model_local_delib.thought_summaries.flocking" not in backend.last_generation_metadata


def test_engine_backend_selectively_surfaces_only_active_advanced_sections() -> None:
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    engine.model.last_deliberation_stats = [
        {
            "layer_idx": 0,
            "mean_steps_taken": 2.0,
            "executed_steps": 6.0,
            "mean_executed_steps_per_token": 2.0,
            "max_executed_steps_any_token": 2.0,
            "mean_neighbor_count": 4.0,
            "mean_sequence_neighbor_weight": 0.24,
            "mean_semantic_neighbor_weight": 0.19,
            "mean_phrase_neighbor_weight": 0.21,
            "semantic_topk_used": 4.0,
            "mean_alignment_norm": 0.26,
            "mean_cohesion_norm": 0.23,
            "mean_separation_norm": 0.18,
            "mean_flocking_total_norm": 0.67,
            "flocking_neighbor_count": 3.0,
            "fraction_flocking_tokens_active": 0.75,
            "global_anchors_used": 3.0,
            "mean_anchor_read_weight": 0.27,
            "mean_anchor_write_weight": 0.22,
            "mean_anchor_norm": 0.41,
        }
    ]
    backend = EngineBackend(engine=engine, tokenizer=tokenizer)

    _ = backend.generate("Hello there")

    assert backend.last_generation_metadata is not None
    graph_artifact = backend.last_generation_metadata["model_local_delib.graph_artifact"]
    assert set(graph_artifact) == {"overview", "anchors", "compute", "flocking"}
    assert graph_artifact["overview"]["active_sections"] == ["anchors", "compute", "flocking"]
    assert graph_artifact["anchors"]["summary"]["write_active_layers"] == 1
    assert graph_artifact["compute"]["summary"]["adaptive_halt_active_layers"] == 0
    assert graph_artifact["flocking"]["summary"]["neighbor_graph_active_layers"] == 1
    assert graph_artifact["flocking"]["summary"]["flocking_active_layers"] == 1
    assert backend.last_generation_metadata["model_local_delib.thought_summaries.global_anchors"] == {
        "layer_count": 1,
        "global_anchors_used": 3.0,
        "mean_anchor_norm": 0.41,
        "mean_anchor_read_weight": 0.27,
        "mean_anchor_write_weight": 0.22,
    }
    assert backend.last_generation_metadata["model_local_delib.thought_summaries.flocking"] == {
        "layer_count": 1,
        "flocking_neighbor_count": 3.0,
        "fraction_flocking_tokens_active": 0.75,
        "mean_alignment_norm": 0.26,
        "mean_cohesion_norm": 0.23,
        "mean_flocking_total_norm": 0.67,
        "mean_separation_norm": 0.18,
    }
    assert "model_local_delib.thought_summaries.branch_consensus" not in backend.last_generation_metadata
    assert "model_local_delib.thought_summaries.deep_hierarchy" not in backend.last_generation_metadata
    assert "model_local_delib.thought_summaries.scratch" not in backend.last_generation_metadata
    assert "model_local_delib.thought_summaries.thought_graph" not in backend.last_generation_metadata


def test_creative_policy_summary_extracts_prompt3_signals() -> None:
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    engine.model.last_deliberation_stats = [
        {
            "layer_idx": 0,
            "branch_factor_used": 3,
            "mean_branch_disagreement": 0.45,
            "mean_branch_consensus_weight": 0.7,
            "mean_branch_verifier_score": 0.6,
            "branch_consensus_used": 1.0,
            "scratch_slots_used": 2,
            "scratch_summary_vector": [0.1, 0.2, 0.3],
            "mean_scratch_summary_norm": 0.2,
            "thought_nodes_used": 4,
            "thought_graph_steps_used": 2,
            "hierarchy_depth_used": 3,
            "global_anchors_used": 2,
            "mean_anchor_read_weight": 0.3,
            "mean_steps_taken": 1.4,
        }
    ]
    backend = EngineBackend(engine=engine, tokenizer=tokenizer)

    _ = backend.generate("Hello there")

    summary = summarize_local_delib_for_creative_policy(backend.last_generation_metadata)

    assert summary["active_sections"] == ["branch", "thought_graph", "hierarchy", "scratch", "anchors", "compute"]
    assert summary["branch_disagreement"] == 0.45
    assert summary["branch_consensus_used"] == 1.0
    assert summary["scratch_summary_dim"] == 3.0
    assert summary["thought_nodes_used"] == 4.0
    assert summary["hierarchy_depth_used"] == 3.0
    assert summary["global_anchors_used"] == 2.0
    assert summary["mean_anchor_read_weight"] == 0.3
    assert summary["mean_steps_taken"] == 1.4


def test_engine_backend_generate_accepts_extra_generation_kwargs() -> None:
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    backend = EngineBackend(engine=engine, tokenizer=tokenizer)

    _ = backend.generate("Hello", foo="bar")

    assert engine.last_generate_kwargs is not None
    assert engine.last_generate_kwargs["foo"] == "bar"


def test_engine_backend_applies_exact_runtime_override_with_compatible_checkpoint(monkeypatch) -> None:
    _patch_flash_attention(monkeypatch)
    tokenizer = TinyTokenizer()
    model = GPT(_tiny_config(local_delib=True, local_delib_steps=2))
    model.eval()
    engine = Engine(model, tokenizer)
    backend = EngineBackend(
        engine=engine,
        tokenizer=tokenizer,
        max_tokens=2,
        temperature=0.0,
        top_k=1,
    )

    response = backend.generate("Hello", local_delib=True, local_delib_adaptive_halt=True)

    assert isinstance(response, str)
    assert backend.last_local_delib_runtime_override_report is not None
    assert backend.last_local_delib_runtime_override_report.status == "exact"
    assert backend.last_local_delib_runtime_override_report.application_method == "reinstantiated_model"
    assert backend.last_generation_metadata is not None
    assert backend.last_generation_metadata["local_delib_runtime_override"]["status"] == "exact"
    assert backend.last_generation_metadata["local_delib_runtime_override"]["requested_overrides"] == {
        "local_delib": True,
        "local_delib_adaptive_halt": True,
    }


def test_engine_backend_raises_for_unsupported_runtime_override(monkeypatch) -> None:
    _patch_flash_attention(monkeypatch)
    tokenizer = TinyTokenizer()
    model = GPT(_tiny_config(local_delib=False, local_delib_steps=0))
    model.eval()
    engine = Engine(model, tokenizer)
    backend = EngineBackend(
        engine=engine,
        tokenizer=tokenizer,
        max_tokens=2,
        temperature=0.0,
        top_k=1,
    )

    with pytest.raises(LocalDelibRuntimeOverrideError) as excinfo:
        backend.generate("Hello", local_delib=True, local_delib_steps=2)

    assert excinfo.value.report.status == "unsupported"
    assert "state-dict compatible" in (excinfo.value.report.reason or "")


def test_engine_backend_can_fall_back_to_approximate_runtime_override(monkeypatch) -> None:
    _patch_flash_attention(monkeypatch)
    tokenizer = TinyTokenizer()
    model = GPT(_tiny_config(local_delib=False, local_delib_steps=0))
    model.eval()
    engine = Engine(model, tokenizer)
    backend = EngineBackend(
        engine=engine,
        tokenizer=tokenizer,
        max_tokens=2,
        temperature=0.0,
        top_k=1,
        allow_approximate_local_delib_overrides=True,
    )

    response = backend.generate("Hello", local_delib=True, local_delib_steps=2)

    assert isinstance(response, str)
    assert backend.last_local_delib_runtime_override_report is not None
    assert backend.last_local_delib_runtime_override_report.status == "approximated"
    assert backend.last_local_delib_runtime_override_report.application_method == "loaded_checkpoint_fallback"
    assert backend.last_generation_metadata is not None
    assert backend.last_generation_metadata["local_delib_runtime_override"]["status"] == "approximated"


def test_engine_backend_raises_for_unknown_runtime_override_key() -> None:
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    backend = EngineBackend(engine=engine, tokenizer=tokenizer)

    with pytest.raises(LocalDelibRuntimeOverrideError) as excinfo:
        backend.generate("Hello", local_delib_unknown=True)

    assert excinfo.value.report.status == "unsupported"
    assert excinfo.value.report.reason == "unknown local-deliberation override keys: local_delib_unknown"
    assert backend.last_generation_metadata is not None
    assert backend.last_generation_metadata["local_delib_runtime_override"]["status"] == "unsupported"


def test_engine_backend_reports_unsupported_for_non_mutable_runtime_override_config() -> None:
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    engine.model = type("FrozenModel", (), {"config": FrozenConfig()})()
    backend = EngineBackend(
        engine=engine,
        tokenizer=tokenizer,
        prompt_max_tokens=64,
    )

    with pytest.raises(LocalDelibRuntimeOverrideError) as excinfo:
        backend.generate("Hello", local_delib=True)

    assert excinfo.value.report.status == "unsupported"
    assert "does not allow runtime override mutation" in (excinfo.value.report.reason or "")
    assert backend.last_generation_metadata is not None
    assert backend.last_generation_metadata["local_delib_runtime_override"]["status"] == "unsupported"
    assert engine.last_generate_kwargs is None
