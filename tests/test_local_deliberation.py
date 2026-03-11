import torch

from nanochat.local_deliberation import (
    CausalDepthwiseMixer,
    LocalDeliberationBlock,
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
    assert isinstance(stats["mean_semantic_neighbor_weight"], float)
    assert stats["executed_steps"] == 2
    assert stats["semantic_topk_used"] == 0


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
