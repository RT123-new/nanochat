from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalDepthwiseMixer(nn.Module):
    """Depthwise causal 1D convolution over token states."""

    def __init__(self, model_dim: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 1")
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=kernel_size,
            groups=model_dim,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x_channels_first = x.transpose(1, 2)
        padded = F.pad(x_channels_first, (self.kernel_size - 1, 0))
        mixed = self.conv(padded)
        return mixed.transpose(1, 2)


class PhrasePool(nn.Module):
    """Pools token chunks into phrase states and broadcasts phrase summaries."""

    def __init__(self, model_dim: int, chunk_size: int) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.chunk_size = chunk_size
        self.proj = nn.Linear(model_dim, model_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C)
        bsz, seq_len, dim = x.shape
        num_chunks = math.ceil(seq_len / self.chunk_size)

        phrase_states = []
        token_broadcast = torch.empty_like(x)

        for idx in range(num_chunks):
            start = idx * self.chunk_size
            end = min((idx + 1) * self.chunk_size, seq_len)
            chunk = x[:, start:end, :]
            phrase = chunk.mean(dim=1)
            projected_phrase = self.proj(phrase)

            phrase_states.append(phrase)
            token_broadcast[:, start:end, :] = projected_phrase.unsqueeze(1)

        phrase_states_t = torch.stack(phrase_states, dim=1)
        return phrase_states_t, token_broadcast


class TokenStateHead(nn.Module):
    """Computes per-token scalar control states."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(model_dim, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        values = torch.sigmoid(self.proj(x))
        salience, uncertainty, halt_gate = torch.chunk(values, chunks=3, dim=-1)
        return {
            "salience": salience,
            "uncertainty": uncertainty,
            "halt_gate": halt_gate,
        }


class LocalDeliberationBlock(nn.Module):
    """Token-local latent deliberation with causal mixing."""

    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        kernel_size: int,
        phrase_chunk_size: int,
        micro_steps: int,
        use_token_gate: bool,
        semantic_topk: int = 0,
        semantic_lookback: int = 64,
    ) -> None:
        super().__init__()
        if micro_steps < 1:
            raise ValueError("micro_steps must be >= 1")
        if semantic_topk < 0:
            raise ValueError("semantic_topk must be >= 0")
        if semantic_lookback < 1:
            raise ValueError("semantic_lookback must be >= 1")

        self.micro_steps = micro_steps
        self.use_token_gate = use_token_gate
        self.semantic_topk = semantic_topk
        self.semantic_lookback = semantic_lookback

        self.in_proj = nn.Linear(model_dim, state_dim)
        self.mixer = CausalDepthwiseMixer(state_dim, kernel_size=kernel_size)
        self.phrase_pool = PhrasePool(state_dim, chunk_size=phrase_chunk_size)
        self.state_head = TokenStateHead(state_dim)

        if self.semantic_topk > 0:
            self.semantic_q = nn.Linear(state_dim, state_dim)
            self.semantic_k = nn.Linear(state_dim, state_dim)
            self.semantic_v = nn.Linear(state_dim, state_dim)
            update_in_dim = state_dim * 4
        else:
            update_in_dim = state_dim * 3

        self.update = nn.Sequential(
            nn.Linear(update_in_dim, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, state_dim),
        )
        self.out_proj = nn.Linear(state_dim, model_dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _semantic_neighbor_summary(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # h: (B, T, C)
        bsz, seq_len, dim = h.shape
        q = self.semantic_q(h)
        k = self.semantic_k(h)
        v = self.semantic_v(h)
        scale = 1.0 / math.sqrt(dim)

        summary = torch.zeros_like(h)
        topk_indices = torch.full((bsz, seq_len, self.semantic_topk), -1, dtype=torch.long, device=h.device)
        topk_weights = torch.zeros((bsz, seq_len, self.semantic_topk), dtype=h.dtype, device=h.device)

        for token_idx in range(seq_len):
            start = max(0, token_idx - self.semantic_lookback)
            if start >= token_idx:
                continue

            window_keys = k[:, start:token_idx, :]
            window_vals = v[:, start:token_idx, :]
            scores = (q[:, token_idx : token_idx + 1, :] * window_keys).sum(dim=-1) * scale

            used_topk = min(self.semantic_topk, window_keys.shape[1])
            topk_scores, local_indices = torch.topk(scores, k=used_topk, dim=-1)
            weights = torch.softmax(topk_scores, dim=-1)

            gathered_vals = torch.gather(
                window_vals,
                dim=1,
                index=local_indices.unsqueeze(-1).expand(-1, -1, dim),
            )
            summary[:, token_idx, :] = (weights.unsqueeze(-1) * gathered_vals).sum(dim=1)

            topk_indices[:, token_idx, :used_topk] = local_indices + start
            topk_weights[:, token_idx, :used_topk] = weights

        return summary, topk_indices, topk_weights, min(self.semantic_topk, max(seq_len - 1, 0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | int]]:
        h = self.in_proj(x)
        head_states: dict[str, torch.Tensor] = self.state_head(h)

        semantic_topk_used = 0
        mean_semantic_neighbor_weight = 0.0

        for _ in range(self.micro_steps):
            mixed = self.mixer(h)
            _, phrase_broadcast = self.phrase_pool(h)
            if self.semantic_topk > 0:
                semantic_summary, _, semantic_weights, semantic_topk_used = self._semantic_neighbor_summary(h)
                used_weight_mask = (semantic_weights > 0).to(semantic_weights.dtype)
                used_weight_count = used_weight_mask.sum().clamp_min(1.0)
                mean_semantic_neighbor_weight = float((semantic_weights * used_weight_mask).sum().item() / used_weight_count.item())
                update_inputs = torch.cat([h, mixed, phrase_broadcast, semantic_summary], dim=-1)
            else:
                update_inputs = torch.cat([h, mixed, phrase_broadcast], dim=-1)

            delta = self.update(update_inputs)

            if self.use_token_gate:
                head_states = self.state_head(h)
                delta = delta * head_states["halt_gate"]

            h = h + delta

        output = x + self.out_proj(h)
        stats: dict[str, torch.Tensor | float | int] = {
            "mean_salience": float(head_states["salience"].mean().item()),
            "mean_uncertainty": float(head_states["uncertainty"].mean().item()),
            "mean_halt": float(head_states["halt_gate"].mean().item()),
            "executed_steps": self.micro_steps,
            "mean_semantic_neighbor_weight": mean_semantic_neighbor_weight,
            "semantic_topk_used": semantic_topk_used,
        }
        return output, stats
