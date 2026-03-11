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


def test_local_delib_advanced_config_defaults_are_stable():
    cfg = GPTConfig()

    assert cfg.local_delib_semantic_topk == 0
    assert cfg.local_delib_semantic_lookback == 64
    assert cfg.local_delib_use_phrase_consensus is False
    assert cfg.local_delib_adaptive_halt is False
    assert cfg.local_delib_branch_factor == 0
    assert cfg.local_delib_branch_every == 1
    assert cfg.local_delib_branch_dim == 0
    assert cfg.local_delib_hierarchy_chunk_sizes == ""
    assert cfg.local_delib_scratch_slots == 0
    assert cfg.local_delib_scratch_dim == 0
    assert cfg.local_delib_debug_branch_stats is False


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
            local_delib_use_phrase_consensus=True,
            local_delib_adaptive_halt=True,
        )
    )

    block = model.local_delib_blocks["0"]
    assert block.semantic_topk == 3
    assert block.semantic_lookback == 7
    assert block.use_phrase_consensus is True
    assert block.adaptive_halt is True


def test_local_delib_module_creation_rules(monkeypatch):
    _patch_flash_attention(monkeypatch)

    disabled = GPT(_tiny_config(local_delib=False, local_delib_steps=3))
    assert len(disabled.local_delib_blocks) == 0

    enabled = GPT(_tiny_config(local_delib=True, local_delib_steps=2, local_delib_every=2, n_layer=5))
    # layers: 0, 2, 4
    assert sorted(enabled.local_delib_blocks.keys()) == ["0", "2", "4"]


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

    torch.manual_seed(0)
    disabled = GPT(_tiny_config(local_delib=False, local_delib_steps=2))
    torch.manual_seed(0)
    enabled = GPT(_tiny_config(local_delib=True, local_delib_steps=2))

    logits_disabled = disabled(idx)
    logits_enabled = enabled(idx)

    assert torch.allclose(logits_enabled, logits_disabled, atol=1e-6, rtol=1e-6)


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
