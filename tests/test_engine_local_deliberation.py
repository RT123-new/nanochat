from types import SimpleNamespace

import torch

from nanochat.engine import Engine, KVCache
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


class TinyTokenizer:
    def __init__(self):
        self._special = {
            "<|assistant_end|>": 16,
            "<|bos|>": 17,
            "<|python_start|>": 18,
            "<|python_end|>": 19,
            "<|output_start|>": 20,
            "<|output_end|>": 21,
        }

    def encode_special(self, s):
        return self._special[s]

    def get_bos_token_id(self):
        return self._special["<|bos|>"]

    def encode(self, s, prepend=None):
        tokens = [ord(c) % 16 for c in s]
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

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
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def test_model_generate_runs_with_local_delib_enabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(_tiny_config(local_delib=True, local_delib_steps=2))

    prompt = [1, 2, 3]
    generated = list(model.generate(prompt, max_tokens=4, temperature=0.0))

    assert len(generated) == 4
    assert all(isinstance(token, int) for token in generated)


def test_engine_generate_batch_runs_with_local_delib_enabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(_tiny_config(local_delib=True, local_delib_steps=2, local_delib_semantic_topk=2))
    engine = Engine(model, TinyTokenizer())

    prompt = [1, 2, 3]
    results, masks = engine.generate_batch(prompt, num_samples=2, max_tokens=4, temperature=0.0)

    assert len(results) == 2
    assert len(masks) == 2
    for row, row_mask in zip(results, masks):
        assert len(row) >= len(prompt)
        assert len(row_mask) == len(row)


def test_decode_cache_path_populates_deliberation_cache(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(_tiny_config(local_delib=True, local_delib_steps=2, local_delib_semantic_topk=2))

    idx_prefill = torch.tensor([[1, 2, 3]], dtype=torch.long)
    idx_decode = torch.tensor([[4]], dtype=torch.long)
    head_dim = model.config.n_embd // model.config.n_head
    kv_cache = KVCache(
        batch_size=1,
        num_heads=model.config.n_kv_head,
        seq_len=model.config.sequence_len,
        head_dim=head_dim,
        num_layers=model.config.n_layer,
        device=idx_prefill.device,
        dtype=torch.float32,
    )

    prefill_logits = model(idx_prefill, kv_cache=kv_cache)
    decode_logits = model(idx_decode, kv_cache=kv_cache)

    assert prefill_logits.shape == (1, 3, model.config.vocab_size)
    assert decode_logits.shape == (1, 1, model.config.vocab_size)
    assert kv_cache.get_pos() == 4
    assert "local_delib" in kv_cache.extra_caches
    assert len(kv_cache.extra_caches["local_delib"]) == len(model.local_delib_blocks)


def test_decode_cache_prefill_expands_local_delib_batch_state(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_phrase_chunk_size=2,
            local_delib_use_deep_hierarchy=True,
            local_delib_span_chunk_size=4,
            local_delib_sequence_summary=True,
            local_delib_scratch_slots=2,
            local_delib_scratch_dim=4,
            local_delib_scratch_refine_steps=1,
            local_delib_use_thought_graph=True,
            local_delib_thought_node_budget=3,
            local_delib_thought_graph_steps=2,
            local_delib_thought_topk_edges=2,
            local_delib_thought_token_chunk_size=2,
            local_delib_global_anchor_count=2,
            local_delib_global_anchor_dim=4,
            local_delib_global_anchor_update=True,
        )
    )

    idx_prefill = torch.tensor([[1, 2, 3]], dtype=torch.long)
    idx_decode = torch.tensor([[4], [5]], dtype=torch.long)
    head_dim = model.config.n_embd // model.config.n_head
    kv_prefill = KVCache(
        batch_size=1,
        num_heads=model.config.n_kv_head,
        seq_len=model.config.sequence_len,
        head_dim=head_dim,
        num_layers=model.config.n_layer,
        device=idx_prefill.device,
        dtype=torch.float32,
    )
    kv_decode = KVCache(
        batch_size=2,
        num_heads=model.config.n_kv_head,
        seq_len=model.config.sequence_len,
        head_dim=head_dim,
        num_layers=model.config.n_layer,
        device=idx_prefill.device,
        dtype=torch.float32,
    )

    _ = model(idx_prefill, kv_cache=kv_prefill)
    kv_decode.prefill(kv_prefill)
    logits = model(idx_decode, kv_cache=kv_decode)

    assert logits.shape == (2, 1, model.config.vocab_size)
    layer_cache = kv_decode.extra_caches["local_delib"]["0"]
    assert layer_cache["stage_states"][0].shape[0] == 2
    assert layer_cache["step_caches"][0]["anchors"]["anchors"].shape[0] == 2


def test_decode_cache_path_works_with_adaptive_halt(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(_tiny_config(local_delib=True, local_delib_steps=3, local_delib_adaptive_halt=True, local_delib_semantic_topk=2))

    idx_prefill = torch.tensor([[1, 2, 3]], dtype=torch.long)
    idx_decode = torch.tensor([[4]], dtype=torch.long)
    head_dim = model.config.n_embd // model.config.n_head
    kv_cache = KVCache(
        batch_size=1,
        num_heads=model.config.n_kv_head,
        seq_len=model.config.sequence_len,
        head_dim=head_dim,
        num_layers=model.config.n_layer,
        device=idx_prefill.device,
        dtype=torch.float32,
    )

    _ = model(idx_prefill, kv_cache=kv_cache)
    decode_logits = model(idx_decode, kv_cache=kv_cache)

    assert decode_logits.shape == (1, 1, model.config.vocab_size)
    assert kv_cache.get_pos() == 4


def test_decode_cache_path_works_with_branching_enabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_semantic_topk=2,
            local_delib_branch_factor=2,
            local_delib_branch_every=1,
        )
    )

    idx_prefill = torch.tensor([[1, 2, 3]], dtype=torch.long)
    idx_decode = torch.tensor([[4]], dtype=torch.long)
    head_dim = model.config.n_embd // model.config.n_head
    kv_cache = KVCache(
        batch_size=1,
        num_heads=model.config.n_kv_head,
        seq_len=model.config.sequence_len,
        head_dim=head_dim,
        num_layers=model.config.n_layer,
        device=idx_prefill.device,
        dtype=torch.float32,
    )

    _ = model(idx_prefill, kv_cache=kv_cache)
    decode_logits = model(idx_decode, kv_cache=kv_cache)

    assert decode_logits.shape == (1, 1, model.config.vocab_size)
    assert kv_cache.get_pos() == 4


def test_decode_cache_path_works_with_branch_consensus_enabled(monkeypatch):
    _patch_flash_attention(monkeypatch)
    model = GPT(
        _tiny_config(
            local_delib=True,
            local_delib_steps=2,
            local_delib_semantic_topk=2,
            local_delib_branch_factor=3,
            local_delib_branch_every=1,
            local_delib_branch_consensus=True,
            local_delib_branch_verifier=True,
            local_delib_branch_max_active=2,
            local_delib_branch_disagreement_threshold=0.0,
        )
    )

    idx_prefill = torch.tensor([[1, 2, 3]], dtype=torch.long)
    idx_decode = torch.tensor([[4]], dtype=torch.long)
    head_dim = model.config.n_embd // model.config.n_head
    kv_cache = KVCache(
        batch_size=1,
        num_heads=model.config.n_kv_head,
        seq_len=model.config.sequence_len,
        head_dim=head_dim,
        num_layers=model.config.n_layer,
        device=idx_prefill.device,
        dtype=torch.float32,
    )

    _ = model(idx_prefill, kv_cache=kv_cache)
    decode_logits = model(idx_decode, kv_cache=kv_cache)

    assert decode_logits.shape == (1, 1, model.config.vocab_size)
    assert kv_cache.get_pos() == 4
