import torch

from nanochat.local_deliberation import (
    CausalDepthwiseMixer,
    LocalDeliberationBlock,
    PhraseConsensusHead,
    PhrasePool,
)


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
    assert stats["executed_steps"] == 2
    assert stats["semantic_topk_used"] == 0
    assert stats["mean_agreement_score"] == 0.0


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

    out_small_base, _ = block_small(base)
    out_small_perturbed, _ = block_small(perturbed)
    out_large_base, _ = block_large(base)
    out_large_perturbed, _ = block_large(perturbed)

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
