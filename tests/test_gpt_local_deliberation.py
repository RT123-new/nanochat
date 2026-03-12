from types import SimpleNamespace

import torch

from nanochat.gpt import GPT, GPTConfig


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
        SimpleNamespace(
            flash_attn_func=fake_flash_attn_func,
            flash_attn_with_kvcache=fake_flash_attn_with_kvcache,
        ),
    )


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
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _configure_global_anchor_modules(block) -> None:
    with torch.no_grad():
        block.global_anchor_query.weight.fill_(0.1)
        block.global_anchor_query.bias.zero_()
        block.global_anchor_key.weight.fill_(0.1)
        block.global_anchor_key.bias.zero_()
        block.global_anchor_value.weight.fill_(0.1)
        block.global_anchor_value.bias.zero_()
        block.global_anchor_to_state.weight.fill_(0.1)
        block.global_anchor_to_state.bias.zero_()
        block.global_anchor_token_write.weight.fill_(0.1)
        block.global_anchor_token_write.bias.zero_()
        block.global_anchor_prefix_write.weight.fill_(0.1)
        block.global_anchor_prefix_write.bias.zero_()
        block.global_anchor_write_query.weight.fill_(0.1)
        block.global_anchor_write_query.bias.zero_()
        block.global_anchor_init.fill_(1.0)
        block.global_anchor_persist_gate.weight.zero_()
        block.global_anchor_persist_gate.bias.fill_(2.0)
        if hasattr(block, "global_anchor_hierarchy_write"):
            block.global_anchor_hierarchy_write.weight.fill_(0.1)
            block.global_anchor_hierarchy_write.bias.zero_()
            block.global_anchor_hierarchy_gate.weight.fill_(0.1)
            block.global_anchor_hierarchy_gate.bias.zero_()
        if hasattr(block, "global_anchor_scratch_write"):
            block.global_anchor_scratch_write.weight.fill_(0.1)
            block.global_anchor_scratch_write.bias.zero_()
            block.global_anchor_scratch_gate.weight.fill_(0.1)
            block.global_anchor_scratch_gate.bias.zero_()
        if hasattr(block, "global_anchor_thought_write"):
            block.global_anchor_thought_write.weight.fill_(0.1)
            block.global_anchor_thought_write.bias.zero_()
            block.global_anchor_thought_gate.weight.fill_(0.1)
            block.global_anchor_thought_gate.bias.zero_()


def _configure_deep_hierarchy_modules(block) -> None:
    state_dim = block.in_proj.out_features
    eye = torch.eye(state_dim)

    with torch.no_grad():
        for scale in (block.deep_phrase_scale, block.deep_span_scale, block.deep_sequence_scale):
            if scale is None:
                continue
            scale.summary_proj.weight.copy_(eye)
            scale.summary_proj.bias.zero_()
            scale.up_proj.weight.copy_(eye)
            scale.up_proj.bias.zero_()
            scale.down_proj.weight.copy_(eye)
            scale.down_proj.bias.zero_()
            scale.to_token_proj.weight.copy_(eye)
            scale.to_token_proj.bias.zero_()
            scale.gate.weight.zero_()
            scale.gate.bias.fill_(2.0)


def _configure_scratch_modules(block) -> None:
    with torch.no_grad():
        block.scratch_query.weight.fill_(0.1)
        block.scratch_query.bias.zero_()
        block.scratch_write_value.weight.fill_(0.1)
        block.scratch_write_value.bias.zero_()
        block.scratch_read_mix.weight.fill_(0.1)
        block.scratch_read_mix.bias.zero_()
        block.scratch_to_state.weight.fill_(0.1)
        block.scratch_to_state.bias.zero_()
        block.scratch_init.fill_(1.0)
        block.scratch_persist_gate.weight.zero_()
        block.scratch_persist_gate.bias.fill_(2.0)
        if getattr(block, "scratch_refine", None) is not None:
            block.scratch_refine[0].weight.fill_(0.1)
            block.scratch_refine[0].bias.zero_()
            block.scratch_refine[2].weight.fill_(0.1)
            block.scratch_refine[2].bias.zero_()
        if hasattr(block, "scratch_branch_write"):
            block.scratch_branch_write.weight.fill_(0.1)
            block.scratch_branch_write.bias.zero_()
            block.scratch_branch_gate.weight.fill_(0.1)
            block.scratch_branch_gate.bias.zero_()
        if hasattr(block, "scratch_hierarchy_write"):
            block.scratch_hierarchy_write.weight.fill_(0.1)
            block.scratch_hierarchy_write.bias.zero_()
            block.scratch_hierarchy_gate.weight.fill_(0.1)
            block.scratch_hierarchy_gate.bias.zero_()


def _configure_thought_modules(block) -> None:
    with torch.no_grad():
        block.thought_node_builder.token_proj.weight.fill_(0.2)
        block.thought_node_builder.token_proj.bias.zero_()
        block.thought_node_builder.branch_proj.weight.fill_(0.1)
        block.thought_node_builder.branch_proj.bias.zero_()
        block.thought_node_builder.hierarchy_proj.weight.fill_(0.1)
        block.thought_node_builder.hierarchy_proj.bias.zero_()
        block.thought_node_builder.scratch_proj.weight.fill_(0.1)
        block.thought_node_builder.scratch_proj.bias.zero_()
        block.token_to_thought.write_value.weight.fill_(0.2)
        block.token_to_thought.write_value.bias.zero_()
        block.token_to_thought.write_gate.weight.zero_()
        block.token_to_thought.write_gate.bias.fill_(2.0)
        block.token_to_thought.read_q.weight.fill_(0.2)
        block.token_to_thought.read_q.bias.zero_()
        block.token_to_thought.read_k.weight.fill_(0.2)
        block.token_to_thought.read_k.bias.zero_()
        block.token_to_thought.read_v.weight.fill_(0.2)
        block.token_to_thought.read_v.bias.zero_()
        block.thought_graph.q_proj.weight.fill_(0.2)
        block.thought_graph.q_proj.bias.zero_()
        block.thought_graph.k_proj.weight.fill_(0.2)
        block.thought_graph.k_proj.bias.zero_()
        block.thought_graph.v_proj.weight.fill_(0.2)
        block.thought_graph.v_proj.bias.zero_()
        block.thought_message_passing.update[0].weight.fill_(0.1)
        block.thought_message_passing.update[0].bias.zero_()
        block.thought_message_passing.update[2].weight.fill_(0.1)
        block.thought_message_passing.update[2].bias.zero_()
        block.thought_consensus_reducer.read_proj.weight.fill_(0.2)
        block.thought_consensus_reducer.read_proj.bias.zero_()
        block.thought_consensus_reducer.consensus_proj.weight.fill_(0.2)
        block.thought_consensus_reducer.consensus_proj.bias.zero_()
        block.thought_consensus_reducer.mix.weight.fill_(0.1)
        block.thought_consensus_reducer.mix.bias.zero_()


def test_local_delib_advanced_config_defaults_are_stable():
    cfg = GPTConfig()

    assert cfg.local_delib_semantic_topk == 0
    assert cfg.local_delib_semantic_lookback == 64
    assert cfg.local_delib_use_neighbor_graph is False
    assert cfg.local_delib_use_phrase_consensus is False
    assert cfg.local_delib_use_flocking is False
    assert cfg.local_delib_flocking_alignment_weight == 0.0
    assert cfg.local_delib_flocking_cohesion_weight == 0.0
    assert cfg.local_delib_flocking_separation_weight == 0.0
    assert cfg.local_delib_flocking_separation_margin == 1.0
    assert cfg.local_delib_flocking_radius_cap == 0
    assert cfg.local_delib_adaptive_halt is False
    assert cfg.local_delib_branch_factor == 0
    assert cfg.local_delib_branch_every == 1
    assert cfg.local_delib_branch_dim == 0
    assert cfg.local_delib_branch_consensus is False
    assert cfg.local_delib_branch_verifier is False
    assert cfg.local_delib_branch_consensus_temp == 1.0
    assert cfg.local_delib_branch_max_active == 0
    assert cfg.local_delib_branch_disagreement_threshold == 0.1
    assert cfg.local_delib_hierarchy_chunk_sizes == ""
    assert cfg.local_delib_use_deep_hierarchy is False
    assert cfg.local_delib_span_chunk_size == 0
    assert cfg.local_delib_sequence_summary is False
    assert cfg.local_delib_hierarchy_bidirectional is False
    assert cfg.local_delib_hierarchy_scale_gate is False
    assert cfg.local_delib_scratch_slots == 0
    assert cfg.local_delib_scratch_dim == 0
    assert cfg.local_delib_scratch_refine_steps == 0
    assert cfg.local_delib_scratch_use_branch_inputs is False
    assert cfg.local_delib_scratch_use_hierarchy_inputs is False
    assert cfg.local_delib_scratch_export_summary is False
    assert cfg.local_delib_scratch_summary_dim == 0
    assert cfg.local_delib_use_thought_graph is False
    assert cfg.local_delib_thought_node_budget == 8
    assert cfg.local_delib_thought_node_dim == 0
    assert cfg.local_delib_thought_graph_steps == 1
    assert cfg.local_delib_thought_topk_edges == 2
    assert cfg.local_delib_thought_token_chunk_size == 4
    assert cfg.local_delib_thought_use_branch_inputs is True
    assert cfg.local_delib_thought_use_hierarchy_inputs is True
    assert cfg.local_delib_thought_use_scratch_inputs is True
    assert cfg.local_delib_global_anchor_count == 0
    assert cfg.local_delib_global_anchor_dim == 0
    assert cfg.local_delib_global_anchor_update is False
    assert cfg.local_delib_global_anchor_temp == 1.0
    assert cfg.local_delib_global_anchor_use_hierarchy is False
    assert cfg.local_delib_global_anchor_use_scratch is False
    assert cfg.local_delib_global_anchor_use_thought is False
    assert cfg.local_delib_debug_branch_stats is False
    assert cfg.local_delib_halt_sparsity_weight == 0.0
    assert cfg.local_delib_branch_diversity_weight == 0.0
    assert cfg.local_delib_branch_entropy_weight == 0.0
    assert cfg.local_delib_consensus_agreement_weight == 0.0
    assert cfg.local_delib_scratch_utilization_weight == 0.0
    assert cfg.local_delib_flocking_stability_weight == 0.0
    assert cfg.local_delib_thought_edge_stability_weight == 0.0
    assert cfg.local_delib_thought_node_utilization_weight == 0.0
    assert cfg.local_delib_hierarchy_agreement_weight == 0.0
    assert cfg.local_delib_branch_usefulness_weight == 0.0
    assert cfg.local_delib_anchor_usage_weight == 0.0


def test_forward_works_with_local_delib_disabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(_tiny_config(local_delib=False, local_delib_steps=2))
    idx = torch.randint(0, model.config.vocab_size, (2, 6))
    targets = torch.randint(0, model.config.vocab_size, (2, 6))

    logits = model(idx)
    loss = model(idx, targets=targets)

    assert logits.shape == (2, 6, model.config.vocab_size)
    assert loss.ndim == 0
    assert len(model.local_delib_blocks) == 0
    assert model.last_deliberation_stats is None


def test_forward_works_with_local_delib_enabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(_tiny_config(local_delib=True, local_delib_steps=2, local_delib_debug_stats=True))
    idx = torch.randint(0, model.config.vocab_size, (2, 5))
    targets = torch.randint(0, model.config.vocab_size, (2, 5))

    logits = model(idx)
    loss = model(idx, targets=targets)

    assert logits.shape == (2, 5, model.config.vocab_size)
    assert loss.ndim == 0
    assert isinstance(model.last_deliberation_stats, list)
    assert len(model.last_deliberation_stats) == len(model.local_delib_blocks)
    assert "mean_branch_score" in model.last_deliberation_stats[0]
    assert "mean_merge_weight" in model.last_deliberation_stats[0]
    assert "mean_branch_disagreement" in model.last_deliberation_stats[0]


def test_forward_works_with_semantic_neighbor_config_enabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(_tiny_config(local_delib=True, local_delib_steps=2, local_delib_semantic_topk=2))
    idx = torch.randint(0, model.config.vocab_size, (2, 4))

    logits = model(idx)

    assert logits.shape == (2, 4, model.config.vocab_size)


def test_local_delib_advanced_fields_are_wired_into_block(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=1,
            local_delib_semantic_topk=3,
            local_delib_semantic_lookback=7,
            local_delib_use_neighbor_graph=True,
            local_delib_use_phrase_consensus=True,
            local_delib_adaptive_halt=True,
        )
    )

    block = model.local_delib_blocks["0"]
    assert block.semantic_topk == 3
    assert block.semantic_lookback == 7
    assert block.use_neighbor_graph is True
    assert block.use_phrase_consensus is True
    assert block.adaptive_halt is True


def test_local_delib_flocking_fields_are_wired_into_block(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=1,
            local_delib_use_neighbor_graph=True,
            local_delib_use_flocking=True,
            local_delib_flocking_alignment_weight=0.4,
            local_delib_flocking_cohesion_weight=0.3,
            local_delib_flocking_separation_weight=0.2,
            local_delib_flocking_separation_margin=1.5,
            local_delib_flocking_radius_cap=3,
        )
    )

    block = model.local_delib_blocks["0"]
    assert block.use_flocking is True
    assert block.flocking_alignment_weight == 0.4
    assert block.flocking_cohesion_weight == 0.3
    assert block.flocking_separation_weight == 0.2
    assert block.flocking_separation_margin == 1.5
    assert block.flocking_radius_cap == 3


def test_local_delib_branch_fields_are_wired_into_block(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_branch_factor=3,
            local_delib_branch_every=2,
            local_delib_branch_dim=5,
        )
    )

    block = model.local_delib_blocks["0"]
    assert block.branch_factor == 3
    assert block.branch_every == 2
    assert block.branch_dim == 5


def test_local_delib_branch_consensus_fields_are_wired_into_block(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_branch_factor=3,
            local_delib_branch_consensus=True,
            local_delib_branch_verifier=True,
            local_delib_branch_consensus_temp=0.7,
            local_delib_branch_max_active=2,
            local_delib_branch_disagreement_threshold=0.05,
        )
    )

    block = model.local_delib_blocks["0"]
    assert block.branch_consensus is True
    assert block.branch_verifier is True
    assert block.branch_consensus_temp == 0.7
    assert block.branch_max_active == 2
    assert block.branch_disagreement_threshold == 0.05


def test_local_delib_thought_graph_fields_are_wired_into_block(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_use_thought_graph=True,
            local_delib_thought_node_budget=3,
            local_delib_thought_node_dim=5,
            local_delib_thought_graph_steps=2,
            local_delib_thought_topk_edges=2,
            local_delib_thought_token_chunk_size=2,
            local_delib_thought_use_branch_inputs=False,
            local_delib_thought_use_hierarchy_inputs=True,
            local_delib_thought_use_scratch_inputs=False,
        )
    )

    block = model.local_delib_blocks["0"]
    assert block.use_thought_graph is True
    assert block.thought_node_budget == 3
    assert block.thought_node_dim == 5
    assert block.thought_graph_steps == 2
    assert block.thought_topk_edges == 2
    assert block.thought_token_chunk_size == 2
    assert block.thought_use_branch_inputs is False
    assert block.thought_use_hierarchy_inputs is True
    assert block.thought_use_scratch_inputs is False


def test_local_delib_global_anchor_fields_are_wired_into_block(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_global_anchor_count=3,
            local_delib_global_anchor_dim=5,
            local_delib_global_anchor_update=True,
            local_delib_global_anchor_temp=0.7,
            local_delib_global_anchor_use_hierarchy=True,
            local_delib_global_anchor_use_scratch=True,
            local_delib_global_anchor_use_thought=True,
        )
    )

    block = model.local_delib_blocks["0"]
    assert block.global_anchor_count == 3
    assert block.global_anchor_dim == 5
    assert block.global_anchor_update is True
    assert block.global_anchor_temp == 0.7
    assert block.global_anchor_use_hierarchy is True
    assert block.global_anchor_use_scratch is True
    assert block.global_anchor_use_thought is True


def test_local_delib_module_creation_rules(monkeypatch):
    _patch_flash_attention(monkeypatch)

    disabled = GPT(_tiny_config(local_delib=False, local_delib_steps=3))
    assert len(disabled.local_delib_blocks) == 0

    enabled = GPT(_tiny_config(local_delib=True, local_delib_steps=2, local_delib_every=2, n_layer=5))
    # layers: 0, 2, 4
    assert sorted(enabled.local_delib_blocks.keys()) == ["0", "2", "4"]




def test_local_delib_hierarchy_chunk_sizes_parse_and_wire(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=1,
            local_delib_hierarchy_chunk_sizes="4,16",
        )
    )

    block = model.local_delib_blocks["0"]
    assert block.hierarchy_chunk_sizes == (4, 16)
    assert len(block.hierarchy_levels) == 2


def test_local_delib_deep_hierarchy_fields_are_wired_into_block(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=1,
            local_delib_use_deep_hierarchy=True,
            local_delib_span_chunk_size=8,
            local_delib_sequence_summary=True,
            local_delib_hierarchy_bidirectional=True,
            local_delib_hierarchy_scale_gate=True,
        )
    )

    block = model.local_delib_blocks["0"]
    assert block.use_deep_hierarchy is True
    assert block.span_chunk_size == 8
    assert block.sequence_summary is True
    assert block.hierarchy_bidirectional is True
    assert block.hierarchy_scale_gate is True
    assert block.deep_phrase_scale is not None
    assert block.deep_span_scale is not None
    assert block.deep_sequence_scale is not None


def test_gpt_surfaces_hierarchy_stats(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_debug_stats=True,
            local_delib_hierarchy_chunk_sizes="2,4",
        )
    )
    idx = torch.randint(0, model.config.vocab_size, (1, 7))

    _ = model(idx)

    assert isinstance(model.last_deliberation_stats, list)
    layer_stats = model.last_deliberation_stats[0]
    assert layer_stats["hierarchy_levels_used"] == 2
    assert layer_stats["mean_hierarchy_feedback_norm"] >= 0.0
    assert len(layer_stats["hierarchy_level_chunk_counts"]) == 2


def test_gpt_surfaces_deep_hierarchy_stats(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_debug_stats=True,
            local_delib_phrase_chunk_size=2,
            local_delib_use_deep_hierarchy=True,
            local_delib_span_chunk_size=4,
            local_delib_sequence_summary=True,
            local_delib_hierarchy_bidirectional=True,
            local_delib_hierarchy_scale_gate=True,
        )
    )
    idx = torch.randint(0, model.config.vocab_size, (1, 7))

    _ = model(idx)

    layer_stats = model.last_deliberation_stats[0]
    assert layer_stats["phrase_nodes_used"] == 4
    assert layer_stats["span_nodes_used"] == 2
    assert layer_stats["sequence_summary_used"] == 1
    assert layer_stats["hierarchy_depth_used"] == 3
    assert layer_stats["mean_upward_message_norm"] >= 0.0
    assert layer_stats["mean_downward_message_norm"] >= 0.0
    assert 0.0 <= layer_stats["mean_scale_gate"] <= 1.0


def test_gpt_surfaces_flocking_stats(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_debug_stats=True,
            local_delib_semantic_topk=2,
            local_delib_use_neighbor_graph=True,
            local_delib_use_flocking=True,
            local_delib_flocking_alignment_weight=0.4,
            local_delib_flocking_cohesion_weight=0.3,
            local_delib_flocking_separation_weight=0.2,
            local_delib_flocking_separation_margin=1.25,
            local_delib_flocking_radius_cap=2,
        )
    )
    idx = torch.randint(0, model.config.vocab_size, (1, 6))

    _ = model(idx)

    layer_stats = model.last_deliberation_stats[0]
    assert layer_stats["mean_alignment_norm"] >= 0.0
    assert layer_stats["mean_cohesion_norm"] >= 0.0
    assert layer_stats["mean_separation_norm"] >= 0.0
    assert layer_stats["mean_flocking_total_norm"] >= 0.0
    assert layer_stats["flocking_neighbor_count"] >= 0.0
    assert 0.0 <= layer_stats["fraction_flocking_tokens_active"] <= 1.0


def test_kv_cache_bypasses_local_delib(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(_tiny_config(local_delib=True, local_delib_steps=2, local_delib_debug_stats=True))
    idx = torch.randint(0, model.config.vocab_size, (1, 3))

    class DummyKVCache:
        def __init__(self, n_layers, batch_size, seq_len, n_heads, head_dim):
            self.n_layers = n_layers
            self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32)
            self.k_cache = torch.zeros(n_layers, batch_size, seq_len, n_heads, head_dim)
            self.v_cache = torch.zeros(n_layers, batch_size, seq_len, n_heads, head_dim)

        def get_pos(self):
            return int(self.cache_seqlens[0].item())

        def get_layer_cache(self, layer_idx):
            return self.k_cache[layer_idx], self.v_cache[layer_idx]

        def advance(self, num_tokens):
            self.cache_seqlens += num_tokens

    head_dim = model.config.n_embd // model.config.n_head
    kv_cache = DummyKVCache(
        n_layers=model.config.n_layer,
        batch_size=1,
        seq_len=model.config.sequence_len,
        n_heads=model.config.n_kv_head,
        head_dim=head_dim,
    )

    logits = model(idx, kv_cache=kv_cache)

    assert logits.shape == (1, 3, model.config.vocab_size)
    assert kv_cache.get_pos() == 3
    assert model.last_deliberation_stats is None


def test_local_delib_is_near_identity_at_init(monkeypatch):
    _patch_flash_attention(monkeypatch)
    idx = torch.randint(0, 32, (2, 6))

    disabled = GPT(_tiny_config(local_delib=False, local_delib_steps=2))
    enabled = GPT(_tiny_config(local_delib=True, local_delib_steps=2))
    torch.manual_seed(0)
    disabled.init_weights()
    torch.manual_seed(0)
    enabled.init_weights()

    logits_disabled = disabled(idx)
    logits_enabled = enabled(idx)

    assert torch.allclose(logits_enabled, logits_disabled, atol=1e-6, rtol=1e-6)


def test_advanced_local_delib_stack_is_near_identity_at_init(monkeypatch):
    _patch_flash_attention(monkeypatch)
    idx = torch.randint(0, 32, (2, 6))

    baseline = GPT(_tiny_config(local_delib=False, local_delib_steps=2))
    advanced = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_semantic_topk=2,
            local_delib_use_neighbor_graph=True,
            local_delib_use_phrase_consensus=True,
            local_delib_use_flocking=True,
            local_delib_flocking_alignment_weight=0.4,
            local_delib_flocking_cohesion_weight=0.3,
            local_delib_flocking_separation_weight=0.2,
            local_delib_adaptive_halt=True,
            local_delib_branch_factor=3,
            local_delib_branch_consensus=True,
            local_delib_branch_verifier=True,
            local_delib_branch_max_active=2,
            local_delib_hierarchy_chunk_sizes="2,4",
            local_delib_use_deep_hierarchy=True,
            local_delib_phrase_chunk_size=2,
            local_delib_span_chunk_size=4,
            local_delib_sequence_summary=True,
            local_delib_hierarchy_bidirectional=True,
            local_delib_hierarchy_scale_gate=True,
            local_delib_scratch_slots=2,
            local_delib_scratch_dim=4,
            local_delib_scratch_refine_steps=1,
            local_delib_scratch_use_branch_inputs=True,
            local_delib_scratch_use_hierarchy_inputs=True,
            local_delib_use_thought_graph=True,
            local_delib_thought_node_budget=2,
            local_delib_thought_graph_steps=2,
            local_delib_thought_topk_edges=2,
            local_delib_thought_token_chunk_size=2,
            local_delib_global_anchor_count=2,
            local_delib_global_anchor_dim=4,
            local_delib_global_anchor_update=True,
            local_delib_global_anchor_use_hierarchy=True,
            local_delib_global_anchor_use_scratch=True,
            local_delib_global_anchor_use_thought=True,
        )
    )
    torch.manual_seed(0)
    baseline.init_weights()
    torch.manual_seed(0)
    advanced.init_weights()

    logits_baseline = baseline(idx)
    logits_advanced = advanced(idx)

    assert torch.allclose(logits_advanced, logits_baseline, atol=1e-6, rtol=1e-6)


def test_no_future_token_influence_in_local_delib_path(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_phrase_chunk_size=1,
            local_delib_semantic_topk=2,
        )
    )

    x1 = torch.randint(0, model.config.vocab_size, (1, 8))
    x2 = x1.clone()
    x2[:, 5:] = torch.randint(0, model.config.vocab_size, (1, 3))

    y1 = model(x1)
    y2 = model(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_no_future_token_influence_in_adaptive_halt_path(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=3,
            local_delib_adaptive_halt=True,
            local_delib_phrase_chunk_size=1,
            local_delib_semantic_topk=2,
        )
    )

    x1 = torch.randint(0, model.config.vocab_size, (1, 8))
    x2 = x1.clone()
    x2[:, 5:] = torch.randint(0, model.config.vocab_size, (1, 3))

    y1 = model(x1)
    y2 = model(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_kv_cache_works_with_neighbor_graph_enabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_semantic_topk=2,
            local_delib_use_neighbor_graph=True,
            local_delib_use_phrase_consensus=True,
        )
    )
    idx = torch.randint(0, model.config.vocab_size, (1, 4))

    class DummyKVCache:
        def __init__(self, n_layers, batch_size, seq_len, n_heads, head_dim):
            self.n_layers = n_layers
            self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32)
            self.k_cache = torch.zeros(n_layers, batch_size, seq_len, n_heads, head_dim)
            self.v_cache = torch.zeros(n_layers, batch_size, seq_len, n_heads, head_dim)
            self.extra_caches = {}

        def get_pos(self):
            return int(self.cache_seqlens[0].item())

        def get_layer_cache(self, layer_idx):
            return self.k_cache[layer_idx], self.v_cache[layer_idx]

        def advance(self, num_tokens):
            self.cache_seqlens += num_tokens

    head_dim = model.config.n_embd // model.config.n_head
    kv_cache = DummyKVCache(
        n_layers=model.config.n_layer,
        batch_size=1,
        seq_len=model.config.sequence_len,
        n_heads=model.config.n_kv_head,
        head_dim=head_dim,
    )

    logits = model(idx, kv_cache=kv_cache)

    assert logits.shape == (1, 4, model.config.vocab_size)
    assert kv_cache.get_pos() == 4


def test_no_future_token_influence_in_branch_path(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_phrase_chunk_size=1,
            local_delib_semantic_topk=2,
            local_delib_branch_factor=2,
        )
    )

    x1 = torch.randint(0, model.config.vocab_size, (1, 8))
    x2 = x1.clone()
    x2[:, 5:] = torch.randint(0, model.config.vocab_size, (1, 3))

    y1 = model(x1)
    y2 = model(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_gpt_surfaces_branch_consensus_stats(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_debug_stats=True,
            local_delib_branch_factor=3,
            local_delib_branch_consensus=True,
            local_delib_branch_verifier=True,
            local_delib_branch_max_active=2,
            local_delib_branch_disagreement_threshold=0.0,
        )
    )
    idx = torch.randint(0, model.config.vocab_size, (1, 6))

    _ = model(idx)

    layer_stats = model.last_deliberation_stats[0]
    assert layer_stats["mean_branch_disagreement"] >= 0.0
    assert 0.0 <= layer_stats["mean_branch_consensus_weight"] <= 1.0
    assert 0.0 <= layer_stats["mean_branch_verifier_score"] <= 1.0
    assert layer_stats["mean_branch_entropy"] >= 0.0
    assert 0.0 <= layer_stats["branch_consensus_used"] <= 1.0


def test_no_future_token_influence_in_branch_consensus_path(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_phrase_chunk_size=1,
            local_delib_semantic_topk=2,
            local_delib_branch_factor=3,
            local_delib_branch_consensus=True,
            local_delib_branch_verifier=True,
            local_delib_branch_max_active=2,
            local_delib_branch_disagreement_threshold=0.0,
        )
    )

    x1 = torch.randint(0, model.config.vocab_size, (1, 8))
    x2 = x1.clone()
    x2[:, 5:] = torch.randint(0, model.config.vocab_size, (1, 3))

    y1 = model(x1)
    y2 = model(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_local_delib_scratch_fields_are_wired_into_block(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_scratch_slots=4,
            local_delib_scratch_dim=6,
        )
    )

    block = model.local_delib_blocks["0"]
    assert block.scratch_slots == 4
    assert block.scratch_dim == 6


def test_local_delib_structured_scratch_fields_are_wired_into_block(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_scratch_slots=4,
            local_delib_scratch_dim=6,
            local_delib_scratch_refine_steps=1,
            local_delib_scratch_use_branch_inputs=True,
            local_delib_scratch_use_hierarchy_inputs=True,
            local_delib_scratch_export_summary=True,
            local_delib_scratch_summary_dim=5,
            local_delib_branch_factor=2,
            local_delib_use_deep_hierarchy=True,
            local_delib_phrase_chunk_size=2,
            local_delib_span_chunk_size=4,
            local_delib_sequence_summary=True,
        )
    )

    block = model.local_delib_blocks["0"]
    assert block.scratch_refine_steps == 1
    assert block.scratch_use_branch_inputs is True
    assert block.scratch_use_hierarchy_inputs is True
    assert block.scratch_export_summary is True
    assert block.scratch_summary_dim == 5


def test_gpt_local_delib_scratch_stats_and_sequence_shape(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_debug_stats=True,
            local_delib_scratch_slots=3,
            local_delib_scratch_dim=5,
        )
    )
    idx = torch.randint(0, model.config.vocab_size, (2, 5))

    logits = model(idx)

    assert logits.shape == (2, 5, model.config.vocab_size)
    layer_stats = model.last_deliberation_stats[0]
    assert "scratch_slots_used" in layer_stats
    assert "mean_scratch_read_weight" in layer_stats
    assert "mean_scratch_write_weight" in layer_stats
    assert "mean_scratch_norm" in layer_stats
    assert "mean_scratch_refine_norm" in layer_stats
    assert "mean_scratch_summary_norm" in layer_stats
    assert "mean_branch_to_scratch_weight" in layer_stats
    assert "mean_hierarchy_to_scratch_weight" in layer_stats
    assert "scratch_reset_ok" in layer_stats
    assert "scratch_summary_vector" not in layer_stats


def test_gpt_surfaces_structured_scratch_summary_only_when_enabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_debug_stats=True,
            local_delib_scratch_slots=2,
            local_delib_scratch_dim=4,
            local_delib_scratch_refine_steps=1,
            local_delib_scratch_export_summary=True,
            local_delib_scratch_summary_dim=3,
        )
    )
    for block in model.local_delib_blocks.values():
        with torch.no_grad():
            block.scratch_to_state.weight.fill_(0.1)
            block.scratch_init.fill_(1.0)
            block.scratch_persist_gate.weight.zero_()
            block.scratch_persist_gate.bias.fill_(2.0)
            block.scratch_refine[0].weight.fill_(0.1)
            block.scratch_refine[0].bias.zero_()
            block.scratch_refine[2].weight.fill_(0.1)
            block.scratch_refine[2].bias.zero_()
            block.scratch_summary_proj.weight.fill_(0.1)
            block.scratch_summary_proj.bias.zero_()
    idx = torch.randint(0, model.config.vocab_size, (1, 5))

    logits = model(idx)

    assert logits.shape == (1, 5, model.config.vocab_size)
    layer_stats = model.last_deliberation_stats[0]
    assert len(layer_stats["scratch_summary_vector"]) == 3
    assert layer_stats["mean_scratch_refine_norm"] > 0.0
    assert layer_stats["mean_scratch_summary_norm"] > 0.0


def test_gpt_surfaces_thought_graph_stats(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_debug_stats=True,
            local_delib_use_thought_graph=True,
            local_delib_thought_node_budget=2,
            local_delib_thought_graph_steps=2,
            local_delib_thought_topk_edges=2,
            local_delib_thought_token_chunk_size=2,
            local_delib_branch_factor=2,
            local_delib_hierarchy_chunk_sizes="2,4",
        )
    )
    idx = torch.randint(0, model.config.vocab_size, (1, 6))

    logits = model(idx)

    assert logits.shape == (1, 6, model.config.vocab_size)
    layer_stats = model.last_deliberation_stats[0]
    assert 0 < layer_stats["thought_nodes_used"] <= 2
    assert layer_stats["mean_thought_degree"] >= 1.0
    assert 0.0 <= layer_stats["mean_token_to_thought_weight"] <= 1.0
    assert 0.0 <= layer_stats["mean_thought_to_token_weight"] <= 1.0
    assert layer_stats["mean_thought_update_norm"] >= 0.0
    assert layer_stats["thought_graph_steps_used"] == 2


def test_gpt_surfaces_global_anchor_stats(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_debug_stats=True,
            local_delib_global_anchor_count=2,
            local_delib_global_anchor_dim=4,
            local_delib_global_anchor_update=True,
            local_delib_global_anchor_use_hierarchy=True,
            local_delib_global_anchor_use_scratch=True,
            local_delib_global_anchor_use_thought=True,
            local_delib_use_deep_hierarchy=True,
            local_delib_phrase_chunk_size=2,
            local_delib_span_chunk_size=4,
            local_delib_sequence_summary=True,
            local_delib_scratch_slots=2,
            local_delib_scratch_dim=4,
            local_delib_use_thought_graph=True,
            local_delib_thought_node_budget=2,
            local_delib_thought_graph_steps=1,
            local_delib_thought_token_chunk_size=2,
        )
    )
    for block in model.local_delib_blocks.values():
        _configure_global_anchor_modules(block)
        with torch.no_grad():
            block.scratch_to_state.weight.fill_(0.1)
    idx = torch.randint(0, model.config.vocab_size, (1, 6))

    logits = model(idx)

    assert logits.shape == (1, 6, model.config.vocab_size)
    layer_stats = model.last_deliberation_stats[0]
    assert 0 < layer_stats["global_anchors_used"] <= 2
    assert layer_stats["mean_anchor_read_weight"] > 0.0
    assert layer_stats["mean_anchor_write_weight"] > 0.0
    assert layer_stats["mean_anchor_norm"] > 0.0


def test_scratch_enabled_changes_logits_when_projection_enabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    idx = torch.randint(0, 32, (1, 6))

    torch.manual_seed(0)
    base = GPT(_tiny_config(local_delib=True, local_delib_steps=2, local_delib_scratch_slots=0, local_delib_scratch_dim=0))
    torch.manual_seed(0)
    scratch = GPT(_tiny_config(local_delib=True, local_delib_steps=2, local_delib_scratch_slots=2, local_delib_scratch_dim=4))

    for block in scratch.local_delib_blocks.values():
        with torch.no_grad():
            block.scratch_to_state.weight.fill_(0.1)
            block.scratch_init.fill_(2.0)

    logits_base = base(idx)
    logits_scratch = scratch(idx)

    assert not torch.allclose(logits_base, logits_scratch)


def test_global_anchors_enabled_change_logits_when_projection_enabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    idx = torch.randint(0, 32, (1, 6))

    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_global_anchor_count=2,
            local_delib_global_anchor_dim=4,
            local_delib_global_anchor_update=True,
        )
    )

    logits_base = model(idx)
    for block in model.local_delib_blocks.values():
        _configure_global_anchor_modules(block)
        with torch.no_grad():
            block.out_proj.weight.fill_(0.1)
            block.global_anchor_init.fill_(2.0)
    logits_anchor = model(idx)

    assert not torch.allclose(logits_base, logits_anchor)


def test_no_future_token_influence_in_scratch_path(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_phrase_chunk_size=1,
            local_delib_scratch_slots=3,
            local_delib_scratch_dim=6,
        )
    )

    for block in model.local_delib_blocks.values():
        with torch.no_grad():
            block.scratch_to_state.weight.fill_(0.1)

    x1 = torch.randint(0, model.config.vocab_size, (1, 8))
    x2 = x1.clone()
    x2[:, 5:] = torch.randint(0, model.config.vocab_size, (1, 3))

    y1 = model(x1)
    y2 = model(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_no_future_token_influence_in_global_anchor_path(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_phrase_chunk_size=1,
            local_delib_global_anchor_count=2,
            local_delib_global_anchor_dim=4,
            local_delib_global_anchor_update=True,
        )
    )

    for block in model.local_delib_blocks.values():
        _configure_global_anchor_modules(block)
        with torch.no_grad():
            block.out_proj.weight.fill_(0.1)

    x1 = torch.randint(0, model.config.vocab_size, (1, 8))
    x2 = x1.clone()
    x2[:, 5:] = torch.randint(0, model.config.vocab_size, (1, 3))

    y1 = model(x1)
    y2 = model(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_kv_cache_works_with_thought_graph_enabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_use_thought_graph=True,
            local_delib_thought_node_budget=2,
            local_delib_thought_graph_steps=2,
            local_delib_thought_topk_edges=2,
            local_delib_thought_token_chunk_size=2,
        )
    )
    idx = torch.randint(0, model.config.vocab_size, (1, 4))

    class DummyKVCache:
        def __init__(self, n_layers, batch_size, seq_len, n_heads, head_dim):
            self.n_layers = n_layers
            self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32)
            self.k_cache = torch.zeros(n_layers, batch_size, seq_len, n_heads, head_dim)
            self.v_cache = torch.zeros(n_layers, batch_size, seq_len, n_heads, head_dim)
            self.extra_caches = {}

        def get_pos(self):
            return int(self.cache_seqlens[0].item())

        def get_layer_cache(self, layer_idx):
            return self.k_cache[layer_idx], self.v_cache[layer_idx]

        def advance(self, num_tokens):
            self.cache_seqlens += num_tokens

    head_dim = model.config.n_embd // model.config.n_head
    kv_cache = DummyKVCache(
        n_layers=model.config.n_layer,
        batch_size=1,
        seq_len=model.config.sequence_len,
        n_heads=model.config.n_kv_head,
        head_dim=head_dim,
    )

    logits = model(idx, kv_cache=kv_cache)

    assert logits.shape == (1, 4, model.config.vocab_size)
    assert kv_cache.get_pos() == 4
    assert "local_delib" in kv_cache.extra_caches


def test_kv_cache_works_with_global_anchors_enabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_global_anchor_count=2,
            local_delib_global_anchor_dim=4,
            local_delib_global_anchor_update=True,
        )
    )
    idx = torch.randint(0, model.config.vocab_size, (1, 4))

    class DummyKVCache:
        def __init__(self, n_layers, batch_size, seq_len, n_heads, head_dim):
            self.n_layers = n_layers
            self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32)
            self.k_cache = torch.zeros(n_layers, batch_size, seq_len, n_heads, head_dim)
            self.v_cache = torch.zeros(n_layers, batch_size, seq_len, n_heads, head_dim)
            self.extra_caches = {}

        def get_pos(self):
            return int(self.cache_seqlens[0].item())

        def get_layer_cache(self, layer_idx):
            return self.k_cache[layer_idx], self.v_cache[layer_idx]

        def advance(self, num_tokens):
            self.cache_seqlens += num_tokens

    head_dim = model.config.n_embd // model.config.n_head
    kv_cache = DummyKVCache(
        n_layers=model.config.n_layer,
        batch_size=1,
        seq_len=model.config.sequence_len,
        n_heads=model.config.n_kv_head,
        head_dim=head_dim,
    )

    logits = model(idx, kv_cache=kv_cache)

    assert logits.shape == (1, 4, model.config.vocab_size)
    assert kv_cache.get_pos() == 4
    assert "local_delib" in kv_cache.extra_caches


def test_decode_cache_matches_full_forward_for_advanced_local_delib(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_phrase_chunk_size=2,
            local_delib_scratch_slots=2,
            local_delib_scratch_dim=4,
            local_delib_scratch_refine_steps=1,
            local_delib_global_anchor_count=2,
            local_delib_global_anchor_dim=4,
            local_delib_global_anchor_update=True,
            local_delib_global_anchor_use_scratch=True,
        )
    )
    for block in model.local_delib_blocks.values():
        _configure_scratch_modules(block)
        _configure_global_anchor_modules(block)
        with torch.no_grad():
            block.out_proj.weight.fill_(0.1)

    class DummyKVCache:
        def __init__(self, n_layers, batch_size, seq_len, n_heads, head_dim):
            self.n_layers = n_layers
            self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32)
            self.k_cache = torch.zeros(n_layers, batch_size, seq_len, n_heads, head_dim)
            self.v_cache = torch.zeros(n_layers, batch_size, seq_len, n_heads, head_dim)
            self.extra_caches = {}

        def get_pos(self):
            return int(self.cache_seqlens[0].item())

        def get_layer_cache(self, layer_idx):
            return self.k_cache[layer_idx], self.v_cache[layer_idx]

        def advance(self, num_tokens):
            self.cache_seqlens += num_tokens

    idx_prefill = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    idx_decode = torch.tensor([[5]], dtype=torch.long)
    idx_full = torch.cat([idx_prefill, idx_decode], dim=1)
    head_dim = model.config.n_embd // model.config.n_head
    kv_cache = DummyKVCache(
        n_layers=model.config.n_layer,
        batch_size=1,
        seq_len=model.config.sequence_len,
        n_heads=model.config.n_kv_head,
        head_dim=head_dim,
    )

    _ = model(idx_prefill, kv_cache=kv_cache)
    decode_logits = model(idx_decode, kv_cache=kv_cache)
    full_logits = model(idx_full)

    assert torch.allclose(decode_logits[:, -1, :], full_logits[:, -1, :], atol=1e-2, rtol=1e-2)
    layer_cache = kv_cache.extra_caches["local_delib"]["0"]
    assert len(layer_cache["stage_states"]) == model.local_delib_blocks["0"].micro_steps + 1
    assert len(layer_cache["step_caches"]) == model.local_delib_blocks["0"].micro_steps
    assert layer_cache["step_caches"][0]["scratch"]["slots"].shape[1] == 2
    assert layer_cache["step_caches"][0]["anchors"]["anchors"].shape[1] == 2
    assert layer_cache["step_caches"][0]["thought"] is None


def test_gpt_surfaces_aux_loss_dict_when_local_delib_enabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_use_phrase_consensus=True,
            local_delib_branch_factor=2,
            local_delib_scratch_slots=2,
            local_delib_scratch_dim=4,
        )
    )
    idx = torch.randint(0, model.config.vocab_size, (2, 6))
    targets = torch.randint(0, model.config.vocab_size, (2, 6))

    _ = model(idx, targets=targets)

    assert isinstance(model.last_aux_losses, dict)
    assert "local_delib_halt_sparsity_loss" in model.last_aux_losses
    assert "local_delib_branch_diversity_loss" in model.last_aux_losses
    assert "local_delib_branch_entropy_loss" in model.last_aux_losses
    assert "local_delib_consensus_agreement_loss" in model.last_aux_losses
    assert "local_delib_scratch_utilization_loss" in model.last_aux_losses
    assert "local_delib_flocking_stability_loss" in model.last_aux_losses
    assert "local_delib_thought_edge_stability_loss" in model.last_aux_losses
    assert "local_delib_thought_node_utilization_loss" in model.last_aux_losses
    assert "local_delib_hierarchy_agreement_loss" in model.last_aux_losses
    assert "local_delib_branch_usefulness_loss" in model.last_aux_losses
    assert "local_delib_anchor_usage_loss" in model.last_aux_losses


def test_zero_aux_weight_path_matches_base_loss(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(_tiny_config(local_delib=True, local_delib_steps=2))
    idx = torch.randint(0, model.config.vocab_size, (2, 5))
    targets = torch.randint(0, model.config.vocab_size, (2, 5))

    base_loss = model(idx, targets=targets)
    weighted_loss = base_loss
    for aux_name in (
        "local_delib_halt_sparsity_loss",
        "local_delib_branch_diversity_loss",
        "local_delib_branch_entropy_loss",
        "local_delib_consensus_agreement_loss",
        "local_delib_scratch_utilization_loss",
        "local_delib_flocking_stability_loss",
        "local_delib_thought_edge_stability_loss",
        "local_delib_thought_node_utilization_loss",
        "local_delib_hierarchy_agreement_loss",
        "local_delib_branch_usefulness_loss",
        "local_delib_anchor_usage_loss",
    ):
        weighted_loss = weighted_loss + (0.0 * model.last_aux_losses[aux_name])

    assert torch.allclose(weighted_loss, base_loss)


def test_nonzero_aux_weight_path_can_consume_model_aux_losses(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_use_phrase_consensus=True,
            local_delib_branch_factor=2,
            local_delib_scratch_slots=2,
            local_delib_scratch_dim=4,
            local_delib_use_neighbor_graph=True,
            local_delib_use_flocking=True,
            local_delib_flocking_alignment_weight=0.4,
            local_delib_flocking_cohesion_weight=0.3,
            local_delib_flocking_separation_weight=0.2,
            local_delib_semantic_topk=1,
            local_delib_use_deep_hierarchy=True,
            local_delib_phrase_chunk_size=2,
            local_delib_span_chunk_size=2,
            local_delib_sequence_summary=True,
            local_delib_hierarchy_scale_gate=True,
            local_delib_use_thought_graph=True,
            local_delib_thought_node_budget=3,
            local_delib_thought_graph_steps=2,
            local_delib_global_anchor_count=2,
            local_delib_global_anchor_update=True,
        )
    )
    idx = torch.randint(0, model.config.vocab_size, (2, 5))
    targets = torch.randint(0, model.config.vocab_size, (2, 5))

    loss = model(idx, targets=targets)
    weights = {
        "local_delib_halt_sparsity_loss": 0.1,
        "local_delib_branch_diversity_loss": 0.2,
        "local_delib_branch_entropy_loss": 0.3,
        "local_delib_consensus_agreement_loss": 0.4,
        "local_delib_scratch_utilization_loss": 0.5,
        "local_delib_flocking_stability_loss": 0.6,
        "local_delib_thought_edge_stability_loss": 0.7,
        "local_delib_thought_node_utilization_loss": 0.8,
        "local_delib_hierarchy_agreement_loss": 0.9,
        "local_delib_branch_usefulness_loss": 1.0,
        "local_delib_anchor_usage_loss": 1.1,
    }
    for name, weight in weights.items():
        loss = loss + weight * model.last_aux_losses[name]

    assert torch.isfinite(loss)
