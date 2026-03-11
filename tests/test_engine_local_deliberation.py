from types import SimpleNamespace

import torch

from nanochat.engine import Engine
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
    model = GPT(_tiny_config(local_delib=True, local_delib_steps=2, semantic_topk=2))
    engine = Engine(model, TinyTokenizer())

    prompt = [1, 2, 3]
    results, masks = engine.generate_batch(prompt, num_samples=2, max_tokens=4, temperature=0.0)

    assert len(results) == 2
    assert len(masks) == 2
    for row, row_mask in zip(results, masks):
        assert len(row) >= len(prompt)
        assert len(row_mask) == len(row)
