import copy
import math

import pytest
import torch

from nanochat.local_deliberation import (
    CausalDepthwiseMixer,
    LocalDeliberationBlock,
    PhraseConsensusHead,
    PhrasePool,
)


def _configure_branch_modules(block: LocalDeliberationBlock, *, use_consensus: bool) -> None:
    state_dim = block.in_proj.out_features
    assert block.branch_dim == state_dim
    branch_biases = [
        torch.ones(state_dim),
        -torch.ones(state_dim),
        torch.full((state_dim,), 0.5),
    ]
    bias = torch.cat([branch_biases[idx % len(branch_biases)] for idx in range(block.branch_factor)])

    with torch.no_grad():
        block.branch_proposal.proj.weight.zero_()
        block.branch_proposal.proj.bias.copy_(bias)
        block.branch_proposal.back_proj.weight.copy_(torch.eye(state_dim))
        block.branch_proposal.back_proj.bias.zero_()
        block.branch_scorer.proj.weight.zero_()
        block.branch_scorer.proj.bias.zero_()
        block.branch_scorer.proj.weight[:, state_dim:].fill_(0.1)
        if use_consensus:
            block.branch_verifier_head.proj.weight.fill_(0.05)
            block.branch_verifier_head.proj.bias.zero_()
            block.branch_consensus_merge.gates.weight.fill_(0.2)
            block.branch_consensus_merge.gates.bias.zero_()
        else:
            block.branch_merge.gate.weight.fill_(0.2)
            block.branch_merge.gate.bias.zero_()


def _configure_thought_graph_modules(block: LocalDeliberationBlock) -> None:
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


def _configure_deep_hierarchy_modules(block: LocalDeliberationBlock) -> None:
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


def _configure_scratch_workspace_modules(block: LocalDeliberationBlock) -> None:
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
        if hasattr(block, "scratch_summary_proj"):
            block.scratch_summary_proj.weight.fill_(0.1)
            block.scratch_summary_proj.bias.zero_()


def _configure_global_anchor_modules(block: LocalDeliberationBlock) -> None:
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


def _override_state_head_with_halt_pattern(
    block: LocalDeliberationBlock,
    halt_values: list[float],
) -> None:
    pattern = torch.tensor(halt_values, dtype=torch.float32).view(1, -1, 1)

    def fake_state_head(h):
        assert pattern.shape[1] == h.shape[1]
        halt = pattern.to(device=h.device, dtype=h.dtype).expand(h.shape[0], -1, -1)
        return {"salience": halt, "uncertainty": 1.0 - halt, "halt_gate": halt}

    block.state_head.forward = fake_state_head


def _configure_phrase_consensus_identity(block: LocalDeliberationBlock) -> None:
    state_dim = block.in_proj.out_features
    eye = torch.eye(state_dim)

    with torch.no_grad():
        block.in_proj.weight.copy_(eye)
        block.in_proj.bias.zero_()
        block.phrase_consensus.proposal_proj.weight.copy_(eye)
        block.phrase_consensus.proposal_proj.bias.zero_()
        block.phrase_consensus.consensus_proj.weight.copy_(eye)
        block.phrase_consensus.consensus_proj.bias.zero_()
        block.phrase_consensus.agreement_gate.weight.zero_()
        block.phrase_consensus.agreement_gate.bias.fill_(2.0)


def _assert_finite_stats(stats: dict[str, object]) -> None:
    for key, value in stats.items():
        if isinstance(value, float):
            assert math.isfinite(value), key
        elif isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key


def test_local_deliberation_block_shapes():
    block = LocalDeliberationBlock(
        model_dim=16,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=4,
        micro_steps=2,
        use_token_gate=True,
    )
    x = torch.randn(2, 11, 16)

    y, stats = block(x)

    assert y.shape == x.shape
    assert isinstance(stats["mean_salience"], float)
    assert isinstance(stats["mean_uncertainty"], float)
    assert isinstance(stats["mean_halt"], float)
    assert isinstance(stats["mean_neighbor_count"], float)
    assert isinstance(stats["mean_sequence_neighbor_weight"], float)
    assert isinstance(stats["mean_semantic_neighbor_weight"], float)
    assert isinstance(stats["mean_phrase_neighbor_weight"], float)
    assert isinstance(stats["mean_alignment_norm"], float)
    assert isinstance(stats["mean_cohesion_norm"], float)
    assert isinstance(stats["mean_separation_norm"], float)
    assert isinstance(stats["mean_flocking_total_norm"], float)
    assert isinstance(stats["flocking_neighbor_count"], float)
    assert isinstance(stats["fraction_flocking_tokens_active"], float)
    assert isinstance(stats["mean_agreement_score"], float)
    assert isinstance(stats["mean_executed_steps_per_token"], float)
    assert isinstance(stats["max_executed_steps_any_token"], int)
    assert isinstance(stats["fraction_halted_early"], float)
    assert isinstance(stats["mean_final_halt"], float)
    assert isinstance(stats["mean_branch_score"], float)
    assert isinstance(stats["max_branch_score"], float)
    assert isinstance(stats["mean_merge_weight"], float)
    assert isinstance(stats["branch_factor_used"], int)
    assert isinstance(stats["fraction_tokens_branched"], float)
    assert isinstance(stats["mean_branch_disagreement"], float)
    assert isinstance(stats["mean_branch_consensus_weight"], float)
    assert isinstance(stats["mean_branch_verifier_score"], float)
    assert isinstance(stats["mean_branch_entropy"], float)
    assert isinstance(stats["branch_consensus_used"], float)
    assert isinstance(stats["phrase_nodes_used"], int)
    assert isinstance(stats["span_nodes_used"], int)
    assert isinstance(stats["sequence_summary_used"], int)
    assert isinstance(stats["mean_upward_message_norm"], float)
    assert isinstance(stats["mean_downward_message_norm"], float)
    assert isinstance(stats["mean_scale_gate"], float)
    assert isinstance(stats["hierarchy_depth_used"], int)
    assert isinstance(stats["thought_nodes_used"], int)
    assert isinstance(stats["mean_thought_degree"], float)
    assert isinstance(stats["mean_token_to_thought_weight"], float)
    assert isinstance(stats["mean_thought_to_token_weight"], float)
    assert isinstance(stats["mean_thought_update_norm"], float)
    assert isinstance(stats["thought_graph_steps_used"], int)
    assert isinstance(stats["global_anchors_used"], int)
    assert isinstance(stats["mean_anchor_read_weight"], float)
    assert isinstance(stats["mean_anchor_write_weight"], float)
    assert isinstance(stats["mean_anchor_norm"], float)
    assert isinstance(stats["mean_scratch_refine_norm"], float)
    assert isinstance(stats["mean_scratch_summary_norm"], float)
    assert isinstance(stats["mean_branch_to_scratch_weight"], float)
    assert isinstance(stats["mean_hierarchy_to_scratch_weight"], float)
    assert isinstance(stats["scratch_reset_ok"], float)
    assert stats["executed_steps"] == 2
    assert stats["semantic_topk_used"] == 0
    assert stats["mean_agreement_score"] == 0.0


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: CausalDepthwiseMixer(model_dim=4, kernel_size=0),
            "kernel_size must be odd and >= 1",
        ),
        (
            lambda: CausalDepthwiseMixer(model_dim=4, kernel_size=2),
            "kernel_size must be odd and >= 1",
        ),
        (
            lambda: PhrasePool(model_dim=4, chunk_size=0),
            "chunk_size must be >= 1",
        ),
        (
            lambda: PhraseConsensusHead(model_dim=4, chunk_size=0),
            "chunk_size must be >= 1",
        ),
        (
            lambda: LocalDeliberationBlock(
                model_dim=8,
                state_dim=8,
                kernel_size=3,
                phrase_chunk_size=2,
                micro_steps=1,
                use_token_gate=False,
                hierarchy_chunk_sizes=[0],
            ),
            "hierarchy chunk sizes must be >= 1",
        ),
        (
            lambda: LocalDeliberationBlock(
                model_dim=8,
                state_dim=8,
                kernel_size=3,
                phrase_chunk_size=4,
                micro_steps=1,
                use_token_gate=False,
                use_deep_hierarchy=True,
                span_chunk_size=2,
            ),
            "span_chunk_size must be >= phrase_chunk_size when deep hierarchy is enabled",
        ),
    ],
)
def test_causal_constructor_guards_invalid_kernel_and_chunk_sizes(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_causal_depthwise_mixer_no_lookahead():
    torch.manual_seed(0)
    mixer = CausalDepthwiseMixer(model_dim=4, kernel_size=5)

    x1 = torch.randn(1, 10, 4)
    x2 = x1.clone()
    x2[:, 6:, :] = torch.randn(1, 4, 4)

    y1 = mixer(x1)
    y2 = mixer(x2)

    assert torch.allclose(y1[:, :6, :], y2[:, :6, :], atol=1e-6)


def test_local_deliberation_near_identity_at_init():
    torch.manual_seed(0)
    block = LocalDeliberationBlock(
        model_dim=12,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=3,
        use_token_gate=False,
    )
    x = torch.randn(2, 7, 12)

    y, _ = block(x)

    max_diff = (y - x).abs().max().item()
    assert max_diff < 1e-8


def test_phrase_pool_shapes_and_broadcast():
    pool = PhrasePool(model_dim=5, chunk_size=3)
    x = torch.randn(2, 8, 5)

    phrase_states, token_broadcast = pool(x)

    assert phrase_states.shape == (2, 3, 5)
    assert token_broadcast.shape == x.shape

    for start, end in ((0, 3), (3, 6), (6, 8)):
        chunk_broadcast = token_broadcast[:, start:end, :]
        reference = chunk_broadcast[:, :1, :].expand_as(chunk_broadcast)
        assert torch.allclose(chunk_broadcast, reference, atol=1e-6)


def test_semantic_topk_disabled_preserves_forward_path():
    torch.manual_seed(123)
    x = torch.randn(2, 9, 10)

    block_off = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=True,
        semantic_topk=0,
        semantic_lookback=4,
    )
    y_off, stats_off = block_off(x)

    torch.manual_seed(123)
    block_default = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=True,
    )
    y_default, stats_default = block_default(x)

    assert torch.allclose(y_off, y_default, atol=1e-8)
    assert stats_off["executed_steps"] == stats_default["executed_steps"]
    assert stats_off["semantic_topk_used"] == 0
    assert stats_default["semantic_topk_used"] == 0


def test_adaptive_halt_disabled_matches_default_behavior():
    torch.manual_seed(202)
    x = torch.randn(2, 6, 10)

    block_default = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=3,
        use_token_gate=True,
    )
    y_default, stats_default = block_default(x)

    torch.manual_seed(202)
    block_explicit = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=3,
        use_token_gate=True,
        adaptive_halt=False,
    )
    y_explicit, stats_explicit = block_explicit(x)

    assert torch.allclose(y_default, y_explicit, atol=1e-8)
    assert stats_default["executed_steps"] == stats_explicit["executed_steps"]
    assert stats_default["mean_executed_steps_per_token"] == stats_explicit["mean_executed_steps_per_token"]


def test_adaptive_halt_can_execute_different_depths_per_token():
    block = LocalDeliberationBlock(
        model_dim=8,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=3,
        use_token_gate=False,
        adaptive_halt=True,
    )
    block.halt_threshold_logit.data.fill_(0.0)

    def fake_state_head(h):
        halt = torch.zeros(h.shape[0], h.shape[1], 1, device=h.device, dtype=h.dtype)
        halt[:, 0, :] = 0.9
        halt[:, 1, :] = 0.1
        halt[:, 2:, :] = 0.9
        return {"salience": halt, "uncertainty": 1.0 - halt, "halt_gate": halt}

    block.state_head.forward = fake_state_head
    x = torch.randn(1, 4, 8)

    y, stats = block(x)

    assert y.shape == x.shape
    assert stats["max_executed_steps_any_token"] == 3
    assert 1.0 < stats["mean_executed_steps_per_token"] < 3.0
    assert 0.0 < stats["fraction_halted_early"] < 1.0


def test_adaptive_halt_supports_all_halt_no_halt_and_mixed_patterns():
    torch.manual_seed(203)
    x = torch.randn(1, 4, 8)

    stats_by_pattern = {}
    for label, halt_values in {
        "all_halt": [0.9, 0.9, 0.9, 0.9],
        "mixed": [0.9, 0.1, 0.9, 0.9],
        "no_halt": [0.1, 0.1, 0.1, 0.1],
    }.items():
        block = LocalDeliberationBlock(
            model_dim=8,
            state_dim=8,
            kernel_size=3,
            phrase_chunk_size=2,
            micro_steps=4,
            use_token_gate=False,
            adaptive_halt=True,
        )
        block.halt_threshold_logit.data.fill_(0.0)
        _override_state_head_with_halt_pattern(block, halt_values)
        _, stats_by_pattern[label] = block(x)

    all_halt = stats_by_pattern["all_halt"]
    mixed = stats_by_pattern["mixed"]
    no_halt = stats_by_pattern["no_halt"]

    assert all_halt["mean_executed_steps_per_token"] == 1.0
    assert all_halt["max_executed_steps_any_token"] == 1
    assert all_halt["fraction_halted_early"] == 1.0

    assert no_halt["mean_executed_steps_per_token"] == 4.0
    assert no_halt["max_executed_steps_any_token"] == 4
    assert no_halt["fraction_halted_early"] == 0.0

    assert all_halt["mean_executed_steps_per_token"] < mixed["mean_executed_steps_per_token"] < no_halt["mean_executed_steps_per_token"]
    assert all_halt["max_executed_steps_any_token"] < mixed["max_executed_steps_any_token"] == no_halt["max_executed_steps_any_token"]
    assert no_halt["fraction_halted_early"] < mixed["fraction_halted_early"] < all_halt["fraction_halted_early"]

    for stats in stats_by_pattern.values():
        assert 1.0 <= stats["mean_executed_steps_per_token"] <= 4.0
        assert 0.0 <= stats["fraction_halted_early"] <= 1.0
        assert 0.0 <= stats["mean_final_halt"] <= 1.0
        _assert_finite_stats(stats)


def test_semantic_topk_enabled_shapes_and_stats():
    torch.manual_seed(0)
    block = LocalDeliberationBlock(
        model_dim=14,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        semantic_topk=3,
        semantic_lookback=5,
    )
    x = torch.randn(3, 6, 14)

    y, stats = block(x)

    assert y.shape == x.shape
    assert stats["semantic_topk_used"] == 3
    assert 0.0 <= stats["mean_semantic_neighbor_weight"] <= 1.0
    assert stats["mean_agreement_score"] == 0.0


def test_semantic_topk_stats_prove_real_activation():
    torch.manual_seed(2)
    x = torch.randn(1, 7, 10)

    torch.manual_seed(2)
    block_off = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        semantic_topk=0,
        semantic_lookback=4,
    )
    _, stats_off = block_off(x)

    torch.manual_seed(2)
    block_on = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        semantic_topk=2,
        semantic_lookback=4,
    )
    _, stats_on = block_on(x)

    assert stats_off["semantic_topk_used"] == 0
    assert stats_off["mean_neighbor_count"] == 0.0
    assert stats_off["mean_semantic_neighbor_weight"] == 0.0

    assert stats_on["semantic_topk_used"] == 2
    assert stats_on["mean_neighbor_count"] == 2.0
    assert 0.0 < stats_on["mean_semantic_neighbor_weight"] <= 1.0


def test_semantic_neighbors_strictly_causal():
    torch.manual_seed(0)
    block = LocalDeliberationBlock(
        model_dim=8,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        semantic_topk=3,
        semantic_lookback=10,
    )
    x = torch.randn(2, 7, 8)

    h = block.in_proj(x)
    _, topk_indices, _, _ = block._semantic_neighbor_summary(h)

    for token_idx in range(x.shape[1]):
        indices = topk_indices[:, token_idx, :]
        used = indices >= 0
        if used.any():
            assert torch.all(indices[used] < token_idx)


def test_semantic_topk_cap_respected_with_short_lookback():
    torch.manual_seed(1)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        semantic_topk=4,
        semantic_lookback=2,
    )
    x = torch.randn(1, 6, 10)

    h = block.in_proj(x)
    _, topk_indices, _, _ = block._semantic_neighbor_summary(h)

    for token_idx in range(x.shape[1]):
        used = (topk_indices[0, token_idx] >= 0).sum().item()
        expected = min(4, min(2, token_idx))
        assert used == expected


def test_phrase_consensus_shapes_and_agreement_score():
    torch.manual_seed(0)
    head = PhraseConsensusHead(model_dim=6, chunk_size=3)
    x = torch.randn(2, 7, 6)

    phrase_consensus, feedback, mean_agreement_score, token_proposals = head(x)

    assert phrase_consensus.shape == (2, 3, 6)
    assert feedback.shape == x.shape
    assert token_proposals.shape == x.shape
    assert mean_agreement_score.ndim == 0


def test_consensus_chunk_locality():
    torch.manual_seed(1)
    head = PhraseConsensusHead(model_dim=8, chunk_size=3)

    x_base = torch.randn(1, 6, 8)
    x_chunk0 = x_base.clone()
    x_chunk1 = x_base.clone()
    x_chunk0[:, 0:3, :] += 1.5
    x_chunk1[:, 3:6, :] += 1.5

    consensus_base, _, _, _ = head(x_base)
    consensus_chunk0, _, _, _ = head(x_chunk0)
    consensus_chunk1, _, _, _ = head(x_chunk1)

    delta_chunk0_self = (consensus_chunk0[:, 0, :] - consensus_base[:, 0, :]).norm().item()
    delta_chunk0_other = (consensus_chunk0[:, 1, :] - consensus_base[:, 1, :]).norm().item()
    delta_chunk1_self = (consensus_chunk1[:, 1, :] - consensus_base[:, 1, :]).norm().item()
    delta_chunk1_other = (consensus_chunk1[:, 0, :] - consensus_base[:, 0, :]).norm().item()

    assert delta_chunk0_self > delta_chunk0_other
    assert delta_chunk1_self > delta_chunk1_other


def test_local_deliberation_reports_mean_agreement_score_when_enabled():
    torch.manual_seed(3)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=True,
        use_phrase_consensus=True,
    )
    x = torch.randn(2, 8, 10)

    _, stats = block(x)

    assert isinstance(stats["mean_agreement_score"], float)
    assert -1.0 <= stats["mean_agreement_score"] <= 1.0


def test_phrase_consensus_agreement_stat_tracks_token_alignment():
    block = LocalDeliberationBlock(
        model_dim=4,
        state_dim=4,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=1,
        use_token_gate=False,
        use_phrase_consensus=True,
    )
    _configure_phrase_consensus_identity(block)

    aligned = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ]
    )
    mixed = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    _, aligned_stats = block(aligned)
    _, mixed_stats = block(mixed)

    assert aligned_stats["mean_agreement_score"] > mixed_stats["mean_agreement_score"]
    assert aligned_stats["mean_agreement_score"] > 0.95
    assert mixed_stats["mean_agreement_score"] < 0.8


def test_semantic_consensus_halt_stats_stay_finite_and_bounded():
    torch.manual_seed(21)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=3,
        use_token_gate=False,
        semantic_topk=2,
        semantic_lookback=4,
        use_phrase_consensus=True,
        adaptive_halt=True,
    )
    block.halt_threshold_logit.data.fill_(0.0)
    _override_state_head_with_halt_pattern(block, [0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    x = torch.randn(1, 6, 10)

    y, stats = block(x)

    assert y.shape == x.shape
    assert 0 < stats["semantic_topk_used"] <= 2
    assert 0.0 < stats["mean_semantic_neighbor_weight"] <= 1.0
    assert -1.0 <= stats["mean_agreement_score"] <= 1.0
    assert 1.0 <= stats["mean_executed_steps_per_token"] <= 3.0
    assert 0.0 <= stats["fraction_halted_early"] <= 1.0
    _assert_finite_stats(stats)


def test_neighbor_graph_disabled_matches_semantic_path():
    torch.manual_seed(11)
    x = torch.randn(2, 7, 10)

    block_sem = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        semantic_topk=2,
        semantic_lookback=4,
        use_neighbor_graph=False,
    )
    y_sem, stats_sem = block_sem(x)

    torch.manual_seed(11)
    block_same = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        semantic_topk=2,
        semantic_lookback=4,
    )
    y_same, stats_same = block_same(x)

    assert torch.allclose(y_sem, y_same, atol=1e-8)
    assert stats_sem["semantic_topk_used"] == stats_same["semantic_topk_used"]


def test_neighbor_graph_enabled_reports_graph_stats_and_bounds():
    torch.manual_seed(0)
    block = LocalDeliberationBlock(
        model_dim=12,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        semantic_topk=2,
        semantic_lookback=3,
        use_neighbor_graph=True,
        use_phrase_consensus=True,
    )
    x = torch.randn(2, 6, 12)

    y, stats = block(x)

    assert y.shape == x.shape
    assert stats["semantic_topk_used"] <= 2
    assert stats["mean_neighbor_count"] <= 1.0 + 2.0 + 1.0
    assert 0.0 <= stats["mean_sequence_neighbor_weight"] <= 1.0
    assert 0.0 <= stats["mean_semantic_neighbor_weight"] <= 1.0
    assert 0.0 <= stats["mean_phrase_neighbor_weight"] <= 1.0


def test_neighbor_graph_strict_causality():
    torch.manual_seed(4)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=1,
        use_token_gate=False,
        semantic_topk=3,
        semantic_lookback=5,
        use_neighbor_graph=True,
        use_phrase_consensus=True,
    )

    x1 = torch.randn(1, 8, 10)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn(1, 3, 10)

    y1, _ = block(x1)
    y2, _ = block(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_neighbor_graph_phrase_locality_stronger_than_distant_effects():
    torch.manual_seed(7)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=1,
        use_token_gate=False,
        semantic_topk=0,
        use_neighbor_graph=True,
        use_phrase_consensus=True,
    )

    base = torch.randn(1, 9, 10)
    local = base.clone()
    far = base.clone()
    local[:, 0:3, :] += 2.0
    far[:, 6:9, :] += 2.0

    y_base, _ = block(base)
    y_local, _ = block(local)
    y_far, _ = block(far)

    local_self = (y_local[:, 0:3, :] - y_base[:, 0:3, :]).norm().item()
    local_far = (y_local[:, 6:9, :] - y_base[:, 6:9, :]).norm().item()

    far_self = (y_far[:, 6:9, :] - y_base[:, 6:9, :]).norm().item()
    far_local = (y_far[:, 0:3, :] - y_base[:, 0:3, :]).norm().item()

    assert local_self > local_far
    assert far_self > far_local


def test_flocking_disabled_matches_neighbor_graph_path():
    torch.manual_seed(17)
    x = torch.randn(2, 7, 10)

    torch.manual_seed(17)
    base = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        semantic_topk=2,
        semantic_lookback=4,
        use_neighbor_graph=True,
    )
    h_base = base.in_proj(x)
    summary_base, stats_base, feedback_base, _ = base.neighbor_graph_mixer.summarize(h_base)

    torch.manual_seed(17)
    disabled = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        semantic_topk=2,
        semantic_lookback=4,
        use_neighbor_graph=True,
        use_flocking=False,
        flocking_alignment_weight=0.7,
        flocking_cohesion_weight=0.5,
        flocking_separation_weight=0.3,
    )
    h_disabled = disabled.in_proj(x)
    summary_disabled, stats_disabled, feedback_disabled, _ = disabled.neighbor_graph_mixer.summarize(h_disabled)

    assert torch.allclose(summary_base, summary_disabled, atol=1e-8)
    assert torch.allclose(feedback_base, feedback_disabled, atol=1e-8)
    assert stats_base["mean_flocking_total_norm"] == 0.0
    assert stats_disabled["mean_flocking_total_norm"] == 0.0


def test_flocking_enabled_surfaces_stats_and_feedback_shape():
    torch.manual_seed(18)
    block = LocalDeliberationBlock(
        model_dim=12,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        semantic_topk=3,
        semantic_lookback=4,
        use_neighbor_graph=True,
        use_phrase_consensus=True,
        use_flocking=True,
        flocking_alignment_weight=0.4,
        flocking_cohesion_weight=0.3,
        flocking_separation_weight=0.2,
        flocking_separation_margin=1.5,
    )
    x = torch.randn(2, 6, 12)

    h = block.in_proj(x)
    summary, graph_stats, feedback, _ = block.neighbor_graph_mixer.summarize(h)
    y, stats = block(x)

    assert summary.shape == h.shape
    assert feedback.shape == h.shape
    assert y.shape == x.shape
    assert graph_stats["mean_alignment_norm"] >= 0.0
    assert graph_stats["mean_cohesion_norm"] >= 0.0
    assert graph_stats["mean_separation_norm"] >= 0.0
    assert graph_stats["mean_flocking_total_norm"] > 0.0
    assert stats["mean_flocking_total_norm"] > 0.0
    assert stats["flocking_neighbor_count"] >= 1.0
    assert 0.0 <= stats["fraction_flocking_tokens_active"] <= 1.0


def test_flocking_path_is_strictly_causal():
    torch.manual_seed(19)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        semantic_topk=3,
        semantic_lookback=6,
        use_neighbor_graph=True,
        use_flocking=True,
        flocking_alignment_weight=0.5,
        flocking_cohesion_weight=0.5,
        flocking_separation_weight=0.2,
    )

    x1 = torch.randn(1, 8, 10)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn(1, 3, 10)

    _, _, feedback1, _ = block.neighbor_graph_mixer.summarize(block.in_proj(x1))
    _, _, feedback2, _ = block.neighbor_graph_mixer.summarize(block.in_proj(x2))

    assert torch.allclose(feedback1[:, :5, :], feedback2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_flocking_radius_cap_prefers_near_neighbors():
    torch.manual_seed(20)
    block = LocalDeliberationBlock(
        model_dim=8,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        semantic_topk=4,
        semantic_lookback=8,
        use_neighbor_graph=True,
        use_flocking=True,
        flocking_alignment_weight=0.0,
        flocking_cohesion_weight=1.0,
        flocking_separation_weight=0.0,
        flocking_radius_cap=2,
    )

    base = torch.randn(1, 8, 8)
    near = base.clone()
    far = base.clone()
    near[:, 5:7, :] += 2.0
    far[:, 0:2, :] += 2.0

    _, _, feedback_base, _ = block.neighbor_graph_mixer.summarize(block.in_proj(base))
    _, _, feedback_near, _ = block.neighbor_graph_mixer.summarize(block.in_proj(near))
    _, _, feedback_far, _ = block.neighbor_graph_mixer.summarize(block.in_proj(far))

    near_effect = (feedback_near[:, 7, :] - feedback_base[:, 7, :]).norm().item()
    far_effect = (feedback_far[:, 7, :] - feedback_base[:, 7, :]).norm().item()

    assert near_effect > far_effect


def test_branching_disabled_matches_default_behavior():
    torch.manual_seed(77)
    x = torch.randn(2, 6, 10)

    block_default = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=True,
    )
    y_default, stats_default = block_default(x)

    torch.manual_seed(77)
    block_disabled = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=True,
        branch_factor=0,
        branch_every=2,
        branch_dim=4,
    )
    y_disabled, stats_disabled = block_disabled(x)

    assert torch.allclose(y_default, y_disabled, atol=1e-8)
    assert stats_default["branch_factor_used"] == 0
    assert stats_disabled["branch_factor_used"] == 0


def test_branching_enabled_surfaces_branch_stats_and_shape():
    torch.manual_seed(88)
    block = LocalDeliberationBlock(
        model_dim=12,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=3,
        use_token_gate=False,
        branch_factor=2,
        branch_every=1,
        branch_dim=4,
    )
    x = torch.randn(1, 5, 12)

    y, stats = block(x)

    assert y.shape == x.shape
    assert stats["branch_factor_used"] == 2
    assert 0.0 <= stats["mean_branch_score"] <= 1.0
    assert 0.0 <= stats["max_branch_score"] <= 1.0
    assert 0.0 <= stats["mean_merge_weight"] <= 1.0
    assert 0.0 <= stats["fraction_tokens_branched"] <= 1.0
    assert stats["mean_branch_disagreement"] == 0.0
    assert stats["mean_branch_consensus_weight"] == 0.0
    assert stats["mean_branch_verifier_score"] == 0.0
    assert stats["mean_branch_entropy"] >= 0.0
    assert stats["branch_consensus_used"] == 0.0


def test_branch_consensus_disabled_matches_existing_branch_path():
    torch.manual_seed(89)
    x = torch.randn(2, 6, 10)

    baseline = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=False,
        branch_factor=3,
        branch_every=1,
    )
    y_baseline, stats_baseline = baseline(x)

    torch.manual_seed(89)
    explicit_disabled = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=False,
        branch_factor=3,
        branch_every=1,
        branch_consensus=False,
        branch_verifier=False,
        branch_consensus_temp=0.7,
        branch_max_active=2,
        branch_disagreement_threshold=0.0,
    )
    y_disabled, stats_disabled = explicit_disabled(x)

    assert torch.allclose(y_baseline, y_disabled, atol=1e-8)
    assert stats_baseline["mean_branch_score"] == stats_disabled["mean_branch_score"]
    assert stats_baseline["mean_merge_weight"] == stats_disabled["mean_merge_weight"]
    assert stats_baseline["branch_factor_used"] == stats_disabled["branch_factor_used"] == 3
    assert stats_baseline["mean_branch_disagreement"] == stats_disabled["mean_branch_disagreement"] == 0.0
    assert stats_baseline["branch_consensus_used"] == stats_disabled["branch_consensus_used"] == 0.0


def test_branch_consensus_and_verifier_enabled_surface_stats_and_effect():
    torch.manual_seed(90)
    x = torch.randn(1, 5, 8)

    torch.manual_seed(90)
    baseline = LocalDeliberationBlock(
        model_dim=8,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        branch_factor=3,
        branch_every=1,
    )
    torch.manual_seed(90)
    consensus = LocalDeliberationBlock(
        model_dim=8,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        branch_factor=3,
        branch_every=1,
        branch_consensus=True,
        branch_verifier=True,
        branch_consensus_temp=0.7,
        branch_max_active=2,
        branch_disagreement_threshold=0.0,
    )

    _configure_branch_modules(baseline, use_consensus=False)
    _configure_branch_modules(consensus, use_consensus=True)
    with torch.no_grad():
        baseline.out_proj.weight.fill_(0.1)
        consensus.out_proj.weight.fill_(0.1)

    y_baseline, _ = baseline(x)
    y_consensus, stats = consensus(x)

    assert y_consensus.shape == x.shape
    assert not torch.allclose(y_baseline, y_consensus)
    assert stats["branch_factor_used"] == 2
    assert stats["mean_branch_disagreement"] > 0.0
    assert 0.0 <= stats["mean_branch_consensus_weight"] <= 1.0
    assert 0.0 <= stats["mean_branch_verifier_score"] <= 1.0
    assert stats["mean_branch_entropy"] >= 0.0
    assert stats["branch_consensus_used"] > 0.0


def test_hierarchy_disabled_parity_with_default():
    torch.manual_seed(91)
    x = torch.randn(2, 7, 10)

    base = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=True,
    )
    y_base, stats_base = base(x)

    torch.manual_seed(91)
    explicit_disabled = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=True,
        hierarchy_chunk_sizes=[],
    )
    y_disabled, stats_disabled = explicit_disabled(x)

    assert torch.allclose(y_base, y_disabled, atol=1e-8)
    assert stats_base["hierarchy_levels_used"] == 0
    assert stats_disabled["hierarchy_levels_used"] == 0


def test_hierarchy_enabled_shapes_and_stats():
    torch.manual_seed(92)
    block = LocalDeliberationBlock(
        model_dim=12,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=False,
        hierarchy_chunk_sizes=[2, 4],
    )
    x = torch.randn(2, 9, 12)

    y, stats = block(x)

    assert y.shape == x.shape
    assert stats["hierarchy_levels_used"] == 2
    assert stats["mean_hierarchy_feedback_norm"] >= 0.0
    assert len(stats["hierarchy_level_chunk_counts"]) == 2
    assert stats["hierarchy_level_chunk_counts"][0] == 5
    assert stats["hierarchy_level_chunk_counts"][1] == 3


def test_hierarchy_locality_prefers_nearer_scale():
    torch.manual_seed(93)
    block_small = LocalDeliberationBlock(
        model_dim=8,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        hierarchy_chunk_sizes=[2],
    )
    block_large = LocalDeliberationBlock(
        model_dim=8,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        hierarchy_chunk_sizes=[8],
    )

    base = torch.randn(1, 8, 8)
    perturbed = base.clone()
    perturbed[:, 0:2, :] += 3.0

    out_small_base, _ = block_small._compute_legacy_hierarchy_feedback(block_small.in_proj(base))
    out_small_perturbed, _ = block_small._compute_legacy_hierarchy_feedback(block_small.in_proj(perturbed))
    out_large_base, _ = block_large._compute_legacy_hierarchy_feedback(block_large.in_proj(base))
    out_large_perturbed, _ = block_large._compute_legacy_hierarchy_feedback(block_large.in_proj(perturbed))

    near_small = (out_small_perturbed[:, 0:2, :] - out_small_base[:, 0:2, :]).norm().item()
    far_small = (out_small_perturbed[:, 6:8, :] - out_small_base[:, 6:8, :]).norm().item()
    near_large = (out_large_perturbed[:, 0:2, :] - out_large_base[:, 0:2, :]).norm().item()
    far_large = (out_large_perturbed[:, 6:8, :] - out_large_base[:, 6:8, :]).norm().item()

    assert near_small > far_small
    assert (near_small - far_small) > (near_large - far_large)


def test_hierarchy_path_remains_strictly_causal():
    torch.manual_seed(94)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        hierarchy_chunk_sizes=[2, 4],
    )

    x1 = torch.randn(1, 8, 10)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn(1, 3, 10)

    y1, _ = block(x1)
    y2, _ = block(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_deep_hierarchy_disabled_matches_default_path():
    torch.manual_seed(95)
    x = torch.randn(2, 7, 10)

    block_default = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
    )
    y_default, stats_default = block_default(x)

    torch.manual_seed(95)
    block_disabled = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        use_deep_hierarchy=False,
        span_chunk_size=4,
        sequence_summary=True,
        hierarchy_bidirectional=True,
        hierarchy_scale_gate=True,
    )
    y_disabled, stats_disabled = block_disabled(x)

    assert torch.allclose(y_default, y_disabled, atol=1e-8)
    assert stats_default["hierarchy_depth_used"] == 0
    assert stats_disabled["hierarchy_depth_used"] == 0
    assert stats_default["phrase_nodes_used"] == 0
    assert stats_disabled["phrase_nodes_used"] == 0
    assert stats_default["span_nodes_used"] == 0
    assert stats_disabled["span_nodes_used"] == 0
    assert stats_default["sequence_summary_used"] == 0
    assert stats_disabled["sequence_summary_used"] == 0


def test_deep_hierarchy_enabled_surfaces_stats_and_shape():
    torch.manual_seed(96)
    block = LocalDeliberationBlock(
        model_dim=12,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        use_deep_hierarchy=True,
        span_chunk_size=4,
        sequence_summary=True,
        hierarchy_bidirectional=True,
        hierarchy_scale_gate=True,
    )
    x = torch.randn(1, 9, 12)

    y, stats = block(x)

    assert y.shape == x.shape
    assert stats["phrase_nodes_used"] == 5
    assert stats["span_nodes_used"] == 3
    assert stats["sequence_summary_used"] == 1
    assert stats["hierarchy_depth_used"] == 3
    assert stats["mean_upward_message_norm"] >= 0.0
    assert stats["mean_downward_message_norm"] >= 0.0
    assert 0.0 <= stats["mean_scale_gate"] <= 1.0


def test_deep_hierarchy_cross_scale_flow_reaches_far_tokens():
    torch.manual_seed(97)
    phrase_only = LocalDeliberationBlock(
        model_dim=8,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        use_deep_hierarchy=True,
        span_chunk_size=0,
        sequence_summary=False,
    )
    torch.manual_seed(97)
    deep = LocalDeliberationBlock(
        model_dim=8,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        use_deep_hierarchy=True,
        span_chunk_size=8,
        sequence_summary=True,
        hierarchy_bidirectional=True,
    )
    _configure_deep_hierarchy_modules(phrase_only)
    _configure_deep_hierarchy_modules(deep)

    base = torch.randn(1, 8, 8)
    perturbed = base.clone()
    perturbed[:, 0:1, :] += 3.0

    phrase_base, _, _ = phrase_only._compute_deep_hierarchy_feedback(phrase_only.in_proj(base))
    phrase_perturbed, _, _ = phrase_only._compute_deep_hierarchy_feedback(phrase_only.in_proj(perturbed))
    deep_base, _, _ = deep._compute_deep_hierarchy_feedback(deep.in_proj(base))
    deep_perturbed, _, _ = deep._compute_deep_hierarchy_feedback(deep.in_proj(perturbed))

    far_phrase = (phrase_perturbed[:, 6:8, :] - phrase_base[:, 6:8, :]).norm().item()
    far_deep = (deep_perturbed[:, 6:8, :] - deep_base[:, 6:8, :]).norm().item()

    assert far_phrase < 1e-6
    assert far_deep > 0.0
    assert far_deep > far_phrase + 1e-3


def test_deep_hierarchy_feedback_is_strictly_causal():
    torch.manual_seed(98)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        use_deep_hierarchy=True,
        span_chunk_size=4,
        sequence_summary=True,
        hierarchy_bidirectional=True,
        hierarchy_scale_gate=True,
    )
    _configure_deep_hierarchy_modules(block)

    x1 = torch.randn(1, 8, 10)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn(1, 3, 10)

    feedback1, stats1, _ = block._compute_deep_hierarchy_feedback(block.in_proj(x1))
    feedback2, stats2, _ = block._compute_deep_hierarchy_feedback(block.in_proj(x2))

    assert stats1["hierarchy_depth_used"] == 3
    assert stats2["hierarchy_depth_used"] == 3
    assert torch.allclose(feedback1[:, :5, :], feedback2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_scratch_disabled_parity_with_default_path():
    torch.manual_seed(111)
    x = torch.randn(2, 7, 10)

    block_default = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=False,
    )
    y_default, stats_default = block_default(x)

    torch.manual_seed(111)
    block_disabled = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=False,
        scratch_slots=0,
        scratch_dim=4,
    )
    y_disabled, stats_disabled = block_disabled(x)

    assert torch.allclose(y_default, y_disabled, atol=1e-8)
    assert stats_default["scratch_slots_used"] == 0
    assert stats_disabled["scratch_slots_used"] == 0


def test_scratch_enabled_surfaces_stats_and_shape():
    torch.manual_seed(112)
    block = LocalDeliberationBlock(
        model_dim=12,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        scratch_slots=3,
        scratch_dim=5,
    )
    x = torch.randn(1, 6, 12)

    y, stats = block(x)

    assert y.shape == x.shape
    assert 0 <= stats["scratch_slots_used"] <= 3
    assert stats["mean_scratch_read_weight"] >= 0.0
    assert stats["mean_scratch_write_weight"] >= 0.0
    assert stats["mean_scratch_norm"] >= 0.0
    assert stats["mean_scratch_refine_norm"] == 0.0
    assert stats["mean_scratch_summary_norm"] >= 0.0
    assert stats["mean_branch_to_scratch_weight"] == 0.0
    assert stats["mean_hierarchy_to_scratch_weight"] == 0.0
    assert stats["scratch_reset_ok"] == 1.0
    assert "scratch_summary_vector" not in stats


def test_scratch_can_influence_internal_update_path_when_enabled():
    torch.manual_seed(113)
    x = torch.randn(1, 5, 8)

    block = LocalDeliberationBlock(
        model_dim=8,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        scratch_slots=2,
        scratch_dim=4,
    )

    with torch.no_grad():
        block.scratch_to_state.weight.fill_(0.1)
        block.out_proj.weight.fill_(0.1)

    baseline, _ = block(x)
    with torch.no_grad():
        block.scratch_init.fill_(2.0)
    changed, _ = block(x)

    assert not torch.allclose(baseline, changed)


def test_scratch_path_is_strictly_causal_prefix_stable():
    torch.manual_seed(114)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        scratch_slots=3,
        scratch_dim=6,
    )

    with torch.no_grad():
        block.scratch_to_state.weight.fill_(0.1)

    x1 = torch.randn(1, 8, 10)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn(1, 3, 10)

    y1, _ = block(x1)
    y2, _ = block(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_structured_scratch_surfaces_refine_summary_and_input_stats():
    torch.manual_seed(115)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        branch_factor=3,
        branch_every=1,
        branch_consensus=True,
        use_deep_hierarchy=True,
        span_chunk_size=4,
        sequence_summary=True,
        hierarchy_bidirectional=True,
        hierarchy_scale_gate=True,
        scratch_slots=2,
        scratch_dim=6,
        scratch_refine_steps=1,
        scratch_use_branch_inputs=True,
        scratch_use_hierarchy_inputs=True,
        scratch_export_summary=True,
        scratch_summary_dim=4,
    )
    _configure_branch_modules(block, use_consensus=True)
    _configure_deep_hierarchy_modules(block)
    _configure_scratch_workspace_modules(block)
    x = torch.randn(1, 6, 10)

    y, stats = block(x)

    assert y.shape == x.shape
    assert stats["mean_scratch_refine_norm"] > 0.0
    assert stats["mean_scratch_summary_norm"] > 0.0
    assert stats["mean_branch_to_scratch_weight"] > 0.0
    assert stats["mean_hierarchy_to_scratch_weight"] > 0.0
    assert stats["scratch_reset_ok"] == 1.0
    assert len(stats["scratch_summary_vector"]) == 4


def test_structured_scratch_resets_cleanly_between_calls():
    torch.manual_seed(116)
    block = LocalDeliberationBlock(
        model_dim=8,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        scratch_slots=2,
        scratch_dim=4,
        scratch_refine_steps=1,
        scratch_export_summary=True,
        scratch_summary_dim=3,
    )
    _configure_scratch_workspace_modules(block)
    fresh = copy.deepcopy(block)
    x1 = torch.randn(1, 5, 8)
    x2 = torch.randn(1, 5, 8)

    _ = block(x1)
    y_after, stats_after = block(x2)
    y_fresh, stats_fresh = fresh(x2)

    assert torch.allclose(y_after, y_fresh, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        torch.tensor(stats_after["scratch_summary_vector"]),
        torch.tensor(stats_fresh["scratch_summary_vector"]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert stats_after["scratch_reset_ok"] == 1.0


def test_structured_scratch_path_remains_causal_with_refinement_and_inputs():
    torch.manual_seed(117)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        branch_factor=2,
        branch_every=1,
        use_deep_hierarchy=True,
        span_chunk_size=4,
        sequence_summary=True,
        scratch_slots=2,
        scratch_dim=6,
        scratch_refine_steps=1,
        scratch_use_branch_inputs=True,
        scratch_use_hierarchy_inputs=True,
    )
    _configure_branch_modules(block, use_consensus=False)
    _configure_deep_hierarchy_modules(block)
    _configure_scratch_workspace_modules(block)

    x1 = torch.randn(1, 8, 10)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn(1, 3, 10)

    y1, _ = block(x1)
    y2, _ = block(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_thought_graph_disabled_matches_default_path():
    torch.manual_seed(120)
    x = torch.randn(2, 7, 10)

    block_default = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=False,
    )
    y_default, stats_default = block_default(x)

    torch.manual_seed(120)
    block_disabled = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=3,
        micro_steps=2,
        use_token_gate=False,
        use_thought_graph=False,
        thought_node_budget=2,
        thought_node_dim=4,
        thought_graph_steps=2,
        thought_topk_edges=2,
        thought_token_chunk_size=2,
    )
    y_disabled, stats_disabled = block_disabled(x)

    assert torch.allclose(y_default, y_disabled, atol=1e-8)
    assert stats_default["thought_nodes_used"] == 0
    assert stats_disabled["thought_nodes_used"] == 0
    assert stats_disabled["thought_graph_steps_used"] == 0


def test_thought_graph_enabled_surfaces_stats_and_budget():
    torch.manual_seed(121)
    block = LocalDeliberationBlock(
        model_dim=12,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        branch_factor=2,
        branch_every=1,
        hierarchy_chunk_sizes=[2, 4],
        use_thought_graph=True,
        thought_node_budget=2,
        thought_graph_steps=2,
        thought_topk_edges=2,
        thought_token_chunk_size=2,
    )
    x = torch.randn(1, 6, 12)

    _configure_branch_modules(block, use_consensus=False)
    _configure_thought_graph_modules(block)
    with torch.no_grad():
        block.out_proj.weight.fill_(0.1)

    y, stats = block(x)

    assert y.shape == x.shape
    assert 0 < stats["thought_nodes_used"] <= 2
    assert stats["mean_thought_degree"] >= 1.0
    assert 0.0 <= stats["mean_token_to_thought_weight"] <= 1.0
    assert 0.0 <= stats["mean_thought_to_token_weight"] <= 1.0
    assert stats["mean_thought_update_norm"] > 0.0
    assert stats["thought_graph_steps_used"] == 2


def test_thought_graph_path_is_strictly_causal():
    torch.manual_seed(122)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        use_thought_graph=True,
        thought_node_budget=4,
        thought_graph_steps=2,
        thought_topk_edges=2,
        thought_token_chunk_size=2,
    )
    _configure_thought_graph_modules(block)

    x1 = torch.randn(1, 8, 10)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn(1, 3, 10)

    feedback1, stats1, _ = block._compute_thought_feedback(block.in_proj(x1))
    feedback2, stats2, _ = block._compute_thought_feedback(block.in_proj(x2))

    assert stats1["thought_nodes_used"] > 0
    assert stats2["thought_nodes_used"] > 0
    assert feedback1.abs().sum().item() > 0.0
    assert torch.allclose(feedback1[:, :5, :], feedback2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_thought_graph_node_budget_is_bounded():
    torch.manual_seed(123)
    block = LocalDeliberationBlock(
        model_dim=8,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        use_thought_graph=True,
        thought_node_budget=2,
        thought_graph_steps=1,
        thought_topk_edges=2,
        thought_token_chunk_size=2,
    )
    x = torch.randn(1, 10, 8)

    _, stats = block(x)

    assert stats["thought_nodes_used"] == 2


def test_incremental_thought_feedback_matches_full_feedback_for_new_token():
    torch.manual_seed(1233)
    block = LocalDeliberationBlock(
        model_dim=8,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        use_thought_graph=True,
        thought_node_budget=3,
        thought_graph_steps=2,
        thought_topk_edges=2,
        thought_token_chunk_size=2,
    )
    _configure_thought_graph_modules(block)

    x_prefill = torch.randn(1, 3, 8)
    x_decode = torch.randn(1, 1, 8)
    h_prefill = block.in_proj(x_prefill)
    h_decode = block.in_proj(x_decode)
    h_full = torch.cat([h_prefill, h_decode], dim=1)

    full_feedback, _, _ = block._compute_thought_feedback(h_full)
    step_cache = block._build_thought_step_cache(h_prefill)
    incremental_feedback = block._incremental_thought_feedback(h_decode, step_cache, prefix_len=3)

    assert torch.allclose(incremental_feedback, full_feedback[:, -1:, :], atol=1e-6, rtol=1e-6)


def test_thought_graph_decode_cache_matches_full_stack_across_multiple_decode_steps():
    torch.manual_seed(1234)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        branch_factor=3,
        branch_every=1,
        branch_consensus=True,
        branch_verifier=True,
        branch_max_active=2,
        branch_disagreement_threshold=0.0,
        hierarchy_chunk_sizes=[2],
        use_deep_hierarchy=True,
        span_chunk_size=2,
        sequence_summary=True,
        hierarchy_bidirectional=True,
        hierarchy_scale_gate=True,
        scratch_slots=2,
        scratch_dim=4,
        scratch_refine_steps=1,
        scratch_use_branch_inputs=True,
        scratch_use_hierarchy_inputs=True,
        use_thought_graph=True,
        thought_node_budget=3,
        thought_graph_steps=2,
        thought_topk_edges=2,
        thought_token_chunk_size=2,
        global_anchor_count=2,
        global_anchor_dim=4,
        global_anchor_update=True,
        global_anchor_use_hierarchy=True,
        global_anchor_use_scratch=True,
        global_anchor_use_thought=True,
    )
    _configure_branch_modules(block, use_consensus=True)
    _configure_deep_hierarchy_modules(block)
    _configure_scratch_workspace_modules(block)
    _configure_thought_graph_modules(block)
    _configure_global_anchor_modules(block)

    x_prefill = torch.randn(1, 3, 10)
    x_decode_1 = torch.randn(1, 1, 10)
    x_decode_2 = torch.randn(1, 1, 10)
    x_full = torch.cat([x_prefill, x_decode_1, x_decode_2], dim=1)

    h_full, _ = block.deliberate_state(block.in_proj(x_full))

    _, _, cache = block.deliberate_state_cached(block.in_proj(x_prefill), None)
    h_decode_1, _, cache = block.deliberate_state_cached(block.in_proj(x_decode_1), cache)
    h_decode_2, _, cache = block.deliberate_state_cached(block.in_proj(x_decode_2), cache)

    assert torch.allclose(h_decode_1, h_full[:, 3:4, :], atol=1e-3, rtol=1e-3)
    assert torch.allclose(h_decode_2, h_full[:, 4:5, :], atol=1e-3, rtol=1e-3)

    thought_cache = cache["step_caches"][0]["thought"]
    assert thought_cache is not None
    assert len(thought_cache["prev_nodes_by_step"]) == block.thought_graph_steps + 1
    assert thought_cache["prev_nodes_by_step"][-1].shape[1] <= block.thought_node_budget
    assert int(thought_cache["current_count"]) <= block.thought_token_chunk_size
    assert cache["token_count"] == x_full.shape[1]


def test_thought_graph_decode_cache_falls_back_only_when_node_budget_window_would_slide():
    torch.manual_seed(1235)
    block = LocalDeliberationBlock(
        model_dim=8,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        use_thought_graph=True,
        thought_node_budget=2,
        thought_graph_steps=2,
        thought_topk_edges=2,
        thought_token_chunk_size=2,
    )
    _configure_thought_graph_modules(block)

    x_prefill = torch.randn(1, 4, 8)
    x_decode = torch.randn(1, 1, 8)
    x_full = torch.cat([x_prefill, x_decode], dim=1)

    _, _, cache = block.deliberate_state_cached(block.in_proj(x_prefill), None)
    thought_cache = cache["step_caches"][0]["thought"]
    assert thought_cache is not None
    assert thought_cache["will_slide_budget"] is True

    h_decode, _, new_cache = block.deliberate_state_cached(block.in_proj(x_decode), cache)
    h_full, _ = block.deliberate_state(block.in_proj(x_full))

    assert torch.allclose(h_decode, h_full[:, -1:, :], atol=1e-6, rtol=1e-6)
    assert new_cache["token_count"] == x_full.shape[1]


def test_global_anchors_disabled_parity_with_default_path():
    torch.manual_seed(124)
    x = torch.randn(2, 6, 10)

    block_default = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
    )
    y_default, stats_default = block_default(x)

    torch.manual_seed(124)
    block_disabled = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        global_anchor_count=0,
        global_anchor_dim=4,
        global_anchor_update=False,
        global_anchor_temp=0.7,
    )
    y_disabled, stats_disabled = block_disabled(x)

    assert torch.allclose(y_default, y_disabled, atol=1e-8)
    assert stats_default["global_anchors_used"] == 0
    assert stats_disabled["global_anchors_used"] == 0
    assert stats_default["mean_anchor_write_weight"] == 0.0
    assert stats_disabled["mean_anchor_write_weight"] == 0.0


def test_global_anchors_enabled_surface_stats_and_shape():
    torch.manual_seed(125)
    block = LocalDeliberationBlock(
        model_dim=12,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        global_anchor_count=3,
        global_anchor_dim=6,
        global_anchor_update=True,
    )
    _configure_global_anchor_modules(block)
    x = torch.randn(1, 6, 12)

    y, stats = block(x)

    assert y.shape == x.shape
    assert 0 < stats["global_anchors_used"] <= 3
    assert stats["mean_anchor_read_weight"] > 0.0
    assert stats["mean_anchor_write_weight"] > 0.0
    assert stats["mean_anchor_norm"] > 0.0


def test_global_anchors_can_ingest_optional_summaries():
    torch.manual_seed(126)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=1,
        use_token_gate=False,
        global_anchor_count=2,
        global_anchor_dim=6,
        global_anchor_update=True,
        global_anchor_use_hierarchy=True,
        global_anchor_use_scratch=True,
        global_anchor_use_thought=True,
    )
    _configure_global_anchor_modules(block)
    x = torch.randn(1, 5, 10)
    h = block.in_proj(x)
    head_states = block.state_head(h)
    prefix = block._allocate_global_anchor_prefix_state(
        bsz=h.shape[0],
        seq_len=h.shape[1],
        device=h.device,
        dtype=h.dtype,
    )

    feedback_base, _, _, _ = block._compute_global_anchor_feedback(
        h,
        head_states,
        global_anchor_prefix_state=prefix,
    )
    feedback_rich, _, stats, _ = block._compute_global_anchor_feedback(
        h,
        head_states,
        global_anchor_prefix_state=prefix,
        hierarchy_summary=torch.ones_like(h),
        scratch_summary=torch.full_like(h, 0.5),
        thought_summary=torch.full_like(h, 0.25),
    )

    assert stats["mean_anchor_write_weight"] > 0.0
    assert not torch.allclose(feedback_base, feedback_rich)


def test_global_anchor_path_is_strictly_causal():
    torch.manual_seed(127)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=1,
        micro_steps=2,
        use_token_gate=False,
        global_anchor_count=2,
        global_anchor_dim=6,
        global_anchor_update=True,
    )
    _configure_global_anchor_modules(block)
    with torch.no_grad():
        block.out_proj.weight.fill_(0.1)

    x1 = torch.randn(1, 8, 10)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn(1, 3, 10)

    y1, _ = block(x1)
    y2, _ = block(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-6)


def test_local_deliberation_surfaces_aux_losses_and_numeric_ranges():
    torch.manual_seed(200)
    block = LocalDeliberationBlock(
        model_dim=12,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=True,
        use_phrase_consensus=True,
        branch_factor=2,
        branch_every=1,
        scratch_slots=3,
        scratch_dim=5,
    )
    x = torch.randn(2, 6, 12)

    _, _ = block(x)

    aux = block.last_aux_losses
    assert isinstance(aux, dict)
    assert set(aux.keys()) == {
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
    }
    assert 0.0 <= aux["local_delib_halt_sparsity_loss"].item() <= 1.0
    assert aux["local_delib_branch_diversity_loss"].item() >= 0.0
    assert aux["local_delib_branch_entropy_loss"].item() >= 0.0
    assert 0.0 <= aux["local_delib_consensus_agreement_loss"].item() <= 1.0
    assert aux["local_delib_scratch_utilization_loss"].item() >= 0.0
    assert aux["local_delib_flocking_stability_loss"].item() == 0.0
    assert aux["local_delib_thought_edge_stability_loss"].item() == 0.0
    assert aux["local_delib_thought_node_utilization_loss"].item() == 0.0
    assert aux["local_delib_hierarchy_agreement_loss"].item() == 0.0
    assert 0.0 <= aux["local_delib_branch_usefulness_loss"].item() <= 1.0
    assert aux["local_delib_anchor_usage_loss"].item() == 0.0


def test_aux_branch_losses_are_zero_when_branching_disabled():
    torch.manual_seed(201)
    block = LocalDeliberationBlock(
        model_dim=10,
        state_dim=6,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=False,
        branch_factor=0,
    )
    x = torch.randn(1, 5, 10)

    _, _ = block(x)

    assert block.last_aux_losses["local_delib_branch_diversity_loss"].item() == 0.0
    assert block.last_aux_losses["local_delib_branch_entropy_loss"].item() == 0.0
    assert block.last_aux_losses["local_delib_branch_usefulness_loss"].item() == 0.0


def test_second_wave_aux_losses_are_finite_when_enabled():
    torch.manual_seed(202)
    block = LocalDeliberationBlock(
        model_dim=12,
        state_dim=8,
        kernel_size=3,
        phrase_chunk_size=2,
        micro_steps=2,
        use_token_gate=True,
        semantic_topk=2,
        semantic_lookback=4,
        use_neighbor_graph=True,
        use_flocking=True,
        flocking_alignment_weight=0.4,
        flocking_cohesion_weight=0.3,
        flocking_separation_weight=0.2,
        branch_factor=2,
        branch_every=1,
        use_deep_hierarchy=True,
        span_chunk_size=4,
        sequence_summary=True,
        hierarchy_bidirectional=True,
        hierarchy_scale_gate=True,
        use_thought_graph=True,
        thought_node_budget=3,
        thought_graph_steps=2,
        thought_topk_edges=2,
        global_anchor_count=2,
        global_anchor_dim=6,
        global_anchor_update=True,
    )
    _configure_branch_modules(block, use_consensus=False)
    _configure_deep_hierarchy_modules(block)
    _configure_thought_graph_modules(block)
    _configure_global_anchor_modules(block)
    x = torch.randn(2, 6, 12)

    _, _ = block(x)

    aux = block.last_aux_losses
    for name in (
        "local_delib_flocking_stability_loss",
        "local_delib_thought_edge_stability_loss",
        "local_delib_thought_node_utilization_loss",
        "local_delib_hierarchy_agreement_loss",
        "local_delib_branch_usefulness_loss",
        "local_delib_anchor_usage_loss",
    ):
        assert torch.isfinite(aux[name])
        assert 0.0 <= aux[name].item() <= 1.0
