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
    assert stats["executed_steps"] == 2


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
