from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class IncrementalCacheFallback(RuntimeError):
    """Raised when cached continuation should fall back to full recomputation."""


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


class HierarchyPoolBroadcast(nn.Module):
    """Pools chunk-level latent nodes and broadcasts causal level summaries."""

    def __init__(self, state_dim: int, chunk_size: int) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.chunk_size = chunk_size
        self.refine = nn.Linear(state_dim, state_dim)
        self.broadcast_proj = nn.Linear(state_dim, state_dim)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, C)
        _, seq_len, _ = h.shape
        num_chunks = math.ceil(seq_len / self.chunk_size)

        nodes = []
        token_broadcast = torch.empty_like(h)
        for idx in range(num_chunks):
            start = idx * self.chunk_size
            end = min((idx + 1) * self.chunk_size, seq_len)
            chunk = h[:, start:end, :]
            node = torch.tanh(self.refine(chunk.mean(dim=1)))
            nodes.append(node)

        nodes_t = torch.stack(nodes, dim=1)
        node_prefix = torch.cumsum(nodes_t, dim=1)
        for idx in range(num_chunks):
            start = idx * self.chunk_size
            end = min((idx + 1) * self.chunk_size, seq_len)
            prefix_mean = node_prefix[:, idx, :] / float(idx + 1)
            token_broadcast[:, start:end, :] = self.broadcast_proj(prefix_mean).unsqueeze(1)

        return nodes_t, token_broadcast


class CausalHierarchyScale(nn.Module):
    """Computes causal same-length summaries for an explicit hierarchy scale."""

    def __init__(
        self,
        state_dim: int,
        chunk_size: int | None = None,
        *,
        sequence_summary: bool = False,
    ) -> None:
        super().__init__()
        if sequence_summary:
            if chunk_size is not None:
                raise ValueError("sequence_summary scale does not take a chunk_size")
        elif chunk_size is None or chunk_size < 1:
            raise ValueError("chunk_size must be >= 1 for non-sequence hierarchy scales")
        self.chunk_size = chunk_size
        self.sequence_summary = sequence_summary
        self.summary_proj = nn.Linear(state_dim, state_dim)
        self.up_proj = nn.Linear(state_dim, state_dim)
        self.down_proj = nn.Linear(state_dim, state_dim)
        self.to_token_proj = nn.Linear(state_dim, state_dim)
        self.gate = nn.Linear(state_dim * 2, 1)

        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)

    def node_count(self, seq_len: int) -> int:
        if self.sequence_summary:
            return 1 if seq_len > 0 else 0
        return math.ceil(seq_len / self.chunk_size)

    def _causal_prefix_summary(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        if self.sequence_summary:
            prefix = torch.cumsum(x, dim=1)
            denom = torch.arange(1, seq_len + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
            return prefix / denom

        summary = torch.zeros_like(x)
        for start in range(0, seq_len, self.chunk_size):
            end = min(seq_len, start + self.chunk_size)
            chunk = x[:, start:end, :]
            prefix = torch.cumsum(chunk, dim=1)
            denom = torch.arange(1, chunk.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
            summary[:, start:end, :] = prefix / denom
        return summary

    def summarize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.summary_proj(self._causal_prefix_summary(x)))


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


class BranchProposalHead(nn.Module):
    """Produces latent branch proposals per token."""

    def __init__(self, state_dim: int, branch_factor: int, branch_dim: int) -> None:
        super().__init__()
        self.branch_factor = branch_factor
        self.branch_dim = branch_dim
        self.proj = nn.Linear(state_dim, branch_factor * branch_dim)
        self.back_proj = nn.Linear(branch_dim, state_dim)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.back_proj.weight)
        nn.init.zeros_(self.back_proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = h.shape
        proposals = self.proj(h).view(bsz, seq_len, self.branch_factor, self.branch_dim)
        return self.back_proj(proposals)


class BranchScorer(nn.Module):
    """Scores branch proposals against token states."""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(state_dim * 2, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        # h: (B, T, C), proposals: (B, T, K, C)
        token_context = h.unsqueeze(2).expand_as(proposals)
        return self.proj(torch.cat([token_context, proposals], dim=-1)).squeeze(-1)


class BranchVerifierHead(nn.Module):
    """Optionally rescoring proposals using token and consensus context."""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(state_dim * 3, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        h: torch.Tensor,
        proposals: torch.Tensor,
        consensus_summary: torch.Tensor,
    ) -> torch.Tensor:
        token_context = h.unsqueeze(2).expand_as(proposals)
        consensus_context = consensus_summary.unsqueeze(2).expand_as(proposals)
        inputs = torch.cat([token_context, proposals, consensus_context], dim=-1)
        return self.proj(inputs).squeeze(-1)


class BranchMergeHead(nn.Module):
    """Merges scored branch proposals back into the latent token state."""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(state_dim * 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -8.0)

    def forward(self, h: torch.Tensor, branch_summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        merge_weight = torch.sigmoid(self.gate(torch.cat([h, branch_summary], dim=-1)))
        merged = h + merge_weight * (branch_summary - h)
        return merged, merge_weight


class BranchConsensusMergeHead(nn.Module):
    """Blends current, branch-summary, and consensus-summary states."""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.gates = nn.Linear(state_dim * 3, 2)
        nn.init.zeros_(self.gates.weight)
        nn.init.constant_(self.gates.bias, -8.0)

    def forward(
        self,
        h: torch.Tensor,
        branch_summary: torch.Tensor,
        consensus_summary: torch.Tensor,
        consensus_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_inputs = torch.cat([h, branch_summary, consensus_summary], dim=-1)
        branch_gate, consensus_gate = torch.chunk(torch.sigmoid(self.gates(gate_inputs)), chunks=2, dim=-1)
        consensus_gate = consensus_gate * consensus_mask
        total = 1.0 + branch_gate + consensus_gate
        merged = (h + branch_gate * branch_summary + consensus_gate * consensus_summary) / total
        merge_weight = (branch_gate + consensus_gate) / total
        consensus_weight = consensus_gate / total
        return merged, merge_weight, consensus_weight


class PhraseConsensusHead(nn.Module):
    """Builds per-chunk phrase consensus from token-level proposals."""

    def __init__(self, model_dim: int, chunk_size: int) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.chunk_size = chunk_size
        self.proposal_proj = nn.Linear(model_dim, model_dim)
        self.consensus_proj = nn.Linear(model_dim, model_dim)
        self.agreement_gate = nn.Linear(model_dim, 1)

        # Near-disabled at init to preserve prior behavior unless this path learns to engage.
        nn.init.zeros_(self.agreement_gate.weight)
        nn.init.constant_(self.agreement_gate.bias, -8.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, C)
        bsz, seq_len, _ = x.shape
        num_chunks = math.ceil(seq_len / self.chunk_size)

        token_proposals = self.proposal_proj(x)
        phrase_consensus_nodes = []
        consensus_broadcast = torch.empty_like(token_proposals)

        for idx in range(num_chunks):
            start = idx * self.chunk_size
            end = min((idx + 1) * self.chunk_size, seq_len)
            chunk = token_proposals[:, start:end, :]
            consensus = self.consensus_proj(chunk.mean(dim=1))
            phrase_consensus_nodes.append(consensus)
            consensus_broadcast[:, start:end, :] = consensus.unsqueeze(1)

        phrase_consensus = torch.stack(phrase_consensus_nodes, dim=1)
        accept_gate = torch.sigmoid(self.agreement_gate(x))
        feedback = accept_gate * consensus_broadcast

        agreement = F.cosine_similarity(token_proposals, consensus_broadcast, dim=-1)
        mean_agreement_score = agreement.mean()
        return phrase_consensus, feedback, mean_agreement_score, token_proposals


class CausalNeighborGraphMixer(nn.Module):
    """Builds a bounded causal token-neighbor graph and aggregates local messages."""

    def __init__(
        self,
        state_dim: int,
        phrase_chunk_size: int,
        semantic_topk: int,
        semantic_lookback: int,
        use_phrase_nodes: bool,
        use_flocking: bool = False,
        flocking_alignment_weight: float = 0.0,
        flocking_cohesion_weight: float = 0.0,
        flocking_separation_weight: float = 0.0,
        flocking_separation_margin: float = 1.0,
        flocking_radius_cap: int = 0,
    ) -> None:
        super().__init__()
        self.semantic_topk = semantic_topk
        self.semantic_lookback = semantic_lookback
        self.use_phrase_nodes = use_phrase_nodes
        self.phrase_chunk_size = phrase_chunk_size
        self.use_flocking = use_flocking
        self.flocking_alignment_weight = flocking_alignment_weight
        self.flocking_cohesion_weight = flocking_cohesion_weight
        self.flocking_separation_weight = flocking_separation_weight
        self.flocking_separation_margin = flocking_separation_margin
        self.flocking_radius_cap = flocking_radius_cap

        self.q_proj = nn.Linear(state_dim, state_dim)
        self.k_proj = nn.Linear(state_dim, state_dim)
        self.v_proj = nn.Linear(state_dim, state_dim)
        if use_phrase_nodes:
            self.phrase_k_proj = nn.Linear(state_dim, state_dim)
            self.phrase_v_proj = nn.Linear(state_dim, state_dim)

    def _causal_phrase_states(self, h: torch.Tensor) -> torch.Tensor:
        # For token t in a chunk, phrase state is mean(chunk_start:t), preserving causality.
        bsz, seq_len, dim = h.shape
        phrase_states = torch.zeros_like(h)
        for start in range(0, seq_len, self.phrase_chunk_size):
            end = min(seq_len, start + self.phrase_chunk_size)
            chunk = h[:, start:end, :]
            prefix = torch.cumsum(chunk, dim=1)
            denom = torch.arange(1, chunk.shape[1] + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
            phrase_states[:, start:end, :] = prefix / denom
        return phrase_states

    def _apply_flocking_radius_cap(
        self,
        token_idx: int,
        neighbor_states: torch.Tensor,
        neighbor_weights: torch.Tensor,
        neighbor_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        if self.flocking_radius_cap <= 0:
            return neighbor_states, neighbor_weights, float(neighbor_states.shape[1])

        keep_mask = (token_idx - neighbor_positions) <= self.flocking_radius_cap
        masked_weights = neighbor_weights * keep_mask.to(neighbor_weights.dtype)
        masked_weights = masked_weights / masked_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mean_neighbor_count = float(keep_mask.to(torch.float32).sum(dim=-1).mean().item())
        return neighbor_states, masked_weights, mean_neighbor_count

    def _compute_flocking_update(
        self,
        token_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        neighbor_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], float, torch.Tensor]:
        if neighbor_states.shape[1] == 0:
            zero_update = torch.zeros_like(token_state)
            zero_loss = token_state.new_zeros(())
            return zero_update, {
                "mean_alignment_norm": 0.0,
                "mean_cohesion_norm": 0.0,
                "mean_separation_norm": 0.0,
                "mean_flocking_total_norm": 0.0,
            }, 0.0, zero_loss

        weight_mass = neighbor_weights.sum(dim=-1, keepdim=True)
        active_mask = weight_mass.squeeze(-1) > 0
        normalized_weights = neighbor_weights / weight_mass.clamp_min(1e-8)

        token_expanded = token_state.unsqueeze(1)
        deltas = neighbor_states - token_expanded
        delta_norms = deltas.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        alignment_dir = (normalized_weights.unsqueeze(-1) * (deltas / delta_norms)).sum(dim=1)
        centroid = (normalized_weights.unsqueeze(-1) * neighbor_states).sum(dim=1)
        cohesion_delta = centroid - token_state

        if self.flocking_separation_margin > 0.0:
            closeness = torch.clamp(
                self.flocking_separation_margin - delta_norms.squeeze(-1),
                min=0.0,
            ) / self.flocking_separation_margin
        else:
            closeness = torch.zeros_like(neighbor_weights)
        separation_dir = (token_expanded - neighbor_states) / delta_norms
        separation = (normalized_weights * closeness).unsqueeze(-1) * separation_dir
        separation_delta = separation.sum(dim=1)

        alignment = self.flocking_alignment_weight * alignment_dir
        cohesion = self.flocking_cohesion_weight * cohesion_delta
        separation = self.flocking_separation_weight * separation_delta
        total = alignment + cohesion + separation
        alignment_norm = alignment.norm(dim=-1)
        cohesion_norm = cohesion.norm(dim=-1)
        separation_norm = separation.norm(dim=-1)
        total_norm = total.norm(dim=-1)
        component_mass = alignment_norm + cohesion_norm + separation_norm
        stability = (component_mass - total_norm).clamp_min(0.0) / component_mass.clamp_min(1e-8)
        active_mask_f = active_mask.to(stability.dtype)
        stability_loss = (stability * active_mask_f).sum() / active_mask_f.sum().clamp_min(1.0)

        stats = {
            "mean_alignment_norm": float(alignment_norm.mean().item()),
            "mean_cohesion_norm": float(cohesion_norm.mean().item()),
            "mean_separation_norm": float(separation_norm.mean().item()),
            "mean_flocking_total_norm": float(total_norm.mean().item()),
        }
        active_fraction = float(active_mask.to(torch.float32).mean().item())
        return total, stats, active_fraction, stability_loss

    def summarize(
        self,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | int], torch.Tensor, dict[str, torch.Tensor]]:
        bsz, seq_len, dim = h.shape
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        scale = 1.0 / math.sqrt(dim)

        if self.use_phrase_nodes:
            phrase_states = self._causal_phrase_states(h)
            phrase_k = self.phrase_k_proj(phrase_states)
            phrase_v = self.phrase_v_proj(phrase_states)
        else:
            phrase_k = None
            phrase_v = None

        summary = torch.zeros_like(h)
        flocking_feedback = torch.zeros_like(h)
        semantic_topk_used = 0
        total_neighbor_count = 0.0
        total_sequence_weight = 0.0
        total_semantic_weight = 0.0
        total_phrase_weight = 0.0
        total_alignment_norm = 0.0
        total_cohesion_norm = 0.0
        total_separation_norm = 0.0
        total_flocking_norm = 0.0
        total_flocking_neighbor_count = 0.0
        total_flocking_active_fraction = 0.0
        flocking_stability_loss = h.new_zeros(())

        for token_idx in range(seq_len):
            if token_idx == 0:
                continue

            token_q = q[:, token_idx, :]
            score_chunks = []
            value_chunks = []
            type_chunks = []
            neighbor_state_chunks = []
            neighbor_position_chunks = []

            # Immediate sequence predecessor.
            seq_score = (token_q * k[:, token_idx - 1, :]).sum(dim=-1, keepdim=True) * scale
            seq_value = v[:, token_idx - 1 : token_idx, :]
            score_chunks.append(seq_score)
            value_chunks.append(seq_value)
            type_chunks.append("sequence")
            if self.use_flocking:
                neighbor_state_chunks.append(h[:, token_idx - 1 : token_idx, :])
                neighbor_position_chunks.append(
                    torch.full((bsz, 1), token_idx - 1, dtype=torch.long, device=h.device)
                )

            # Causal semantic top-k from bounded lookback.
            start = max(0, token_idx - self.semantic_lookback)
            window_keys = k[:, start:token_idx, :]
            if window_keys.shape[1] > 0 and self.semantic_topk > 0:
                scores = (token_q.unsqueeze(1) * window_keys).sum(dim=-1) * scale
                used_topk = min(self.semantic_topk, window_keys.shape[1])
                semantic_topk_used = max(semantic_topk_used, used_topk)
                topk_scores, local_indices = torch.topk(scores, k=used_topk, dim=-1)
                semantic_values = torch.gather(
                    v[:, start:token_idx, :],
                    dim=1,
                    index=local_indices.unsqueeze(-1).expand(-1, -1, dim),
                )
                score_chunks.append(topk_scores)
                value_chunks.append(semantic_values)
                type_chunks.extend(["semantic"] * used_topk)
                if self.use_flocking:
                    semantic_states = torch.gather(
                        h[:, start:token_idx, :],
                        dim=1,
                        index=local_indices.unsqueeze(-1).expand(-1, -1, dim),
                    )
                    neighbor_state_chunks.append(semantic_states)
                    neighbor_position_chunks.append(local_indices + start)

            # Optional phrase-node link (causal chunk-prefix summary).
            if phrase_k is not None and phrase_v is not None:
                phrase_score = (token_q * phrase_k[:, token_idx - 1, :]).sum(dim=-1, keepdim=True) * scale
                phrase_value = phrase_v[:, token_idx - 1 : token_idx, :]
                score_chunks.append(phrase_score)
                value_chunks.append(phrase_value)
                type_chunks.append("phrase")
                if self.use_flocking:
                    neighbor_state_chunks.append(phrase_states[:, token_idx - 1 : token_idx, :])
                    neighbor_position_chunks.append(
                        torch.full((bsz, 1), token_idx - 1, dtype=torch.long, device=h.device)
                    )

            all_scores = torch.cat(score_chunks, dim=-1)
            all_values = torch.cat(value_chunks, dim=1)
            all_weights = torch.softmax(all_scores, dim=-1)
            summary[:, token_idx, :] = (all_weights.unsqueeze(-1) * all_values).sum(dim=1)

            total_neighbor_count += float(len(type_chunks))
            type_offset = 0
            for t in type_chunks:
                w = all_weights[:, type_offset]
                mean_w = float(w.mean().item())
                if t == "sequence":
                    total_sequence_weight += mean_w
                elif t == "semantic":
                    total_semantic_weight += mean_w
                else:
                    total_phrase_weight += mean_w
                type_offset += 1

            if self.use_flocking:
                neighbor_states = torch.cat(neighbor_state_chunks, dim=1)
                neighbor_positions = torch.cat(neighbor_position_chunks, dim=1)
                filtered_states, filtered_weights, mean_filtered_neighbor_count = self._apply_flocking_radius_cap(
                    token_idx,
                    neighbor_states,
                    all_weights,
                    neighbor_positions,
                )
                flocking_delta, flocking_stats, active_fraction, stability_loss = self._compute_flocking_update(
                    h[:, token_idx, :],
                    filtered_states,
                    filtered_weights,
                )
                flocking_feedback[:, token_idx, :] = flocking_delta
                total_alignment_norm += flocking_stats["mean_alignment_norm"]
                total_cohesion_norm += flocking_stats["mean_cohesion_norm"]
                total_separation_norm += flocking_stats["mean_separation_norm"]
                total_flocking_norm += flocking_stats["mean_flocking_total_norm"]
                total_flocking_neighbor_count += mean_filtered_neighbor_count
                total_flocking_active_fraction += active_fraction
                flocking_stability_loss = flocking_stability_loss + stability_loss

        denom = max(float(seq_len - 1), 1.0)
        stats: dict[str, float | int] = {
            "mean_neighbor_count": total_neighbor_count / denom,
            "mean_sequence_neighbor_weight": total_sequence_weight / denom,
            "mean_semantic_neighbor_weight": total_semantic_weight / denom,
            "mean_phrase_neighbor_weight": total_phrase_weight / denom,
            "semantic_topk_used": semantic_topk_used,
            "mean_alignment_norm": total_alignment_norm / denom,
            "mean_cohesion_norm": total_cohesion_norm / denom,
            "mean_separation_norm": total_separation_norm / denom,
            "mean_flocking_total_norm": total_flocking_norm / denom,
            "flocking_neighbor_count": total_flocking_neighbor_count / denom,
            "fraction_flocking_tokens_active": total_flocking_active_fraction / denom,
        }
        aux_losses = {
            "local_delib_flocking_stability_loss": flocking_stability_loss / denom,
        }
        return summary, stats, flocking_feedback, aux_losses


class LatentThoughtNodeBuilder(nn.Module):
    """Builds bounded latent thought nodes from causal token chunks and optional summaries."""

    def __init__(self, state_dim: int, node_dim: int) -> None:
        super().__init__()
        self.token_proj = nn.Linear(state_dim, node_dim)
        self.branch_proj = nn.Linear(state_dim, node_dim)
        self.hierarchy_proj = nn.Linear(state_dim, node_dim)
        self.scratch_proj = nn.Linear(state_dim, node_dim)

    def forward(
        self,
        h: torch.Tensor,
        *,
        token_chunk_size: int,
        node_budget: int,
        branch_summary: torch.Tensor | None = None,
        hierarchy_summary: torch.Tensor | None = None,
        scratch_summary: torch.Tensor | None = None,
        use_branch_inputs: bool = True,
        use_hierarchy_inputs: bool = True,
        use_scratch_inputs: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = h.shape
        chunk_ranges = [
            (start, min(start + token_chunk_size, seq_len))
            for start in range(0, seq_len, token_chunk_size)
        ]
        if node_budget > 0 and len(chunk_ranges) > node_budget:
            chunk_ranges = chunk_ranges[-node_budget:]

        token_to_node = torch.full((seq_len,), -1, dtype=torch.long, device=h.device)
        nodes = []
        node_limits = []

        for node_idx, (start, end) in enumerate(chunk_ranges):
            pooled = self.token_proj(h[:, start:end, :].mean(dim=1))
            if use_branch_inputs and branch_summary is not None:
                pooled = pooled + self.branch_proj(branch_summary[:, start:end, :].mean(dim=1))
            if use_hierarchy_inputs and hierarchy_summary is not None:
                pooled = pooled + self.hierarchy_proj(hierarchy_summary[:, start:end, :].mean(dim=1))
            if use_scratch_inputs and scratch_summary is not None:
                pooled = pooled + self.scratch_proj(scratch_summary[:, start:end, :].mean(dim=1))
            nodes.append(torch.tanh(pooled))
            node_limits.append(end - 1)
            token_to_node[start:end] = node_idx

        if not nodes:
            return (
                torch.zeros(bsz, 0, self.token_proj.out_features, dtype=h.dtype, device=h.device),
                torch.zeros(0, dtype=torch.long, device=h.device),
                token_to_node,
            )

        return torch.stack(nodes, dim=1), torch.tensor(node_limits, dtype=torch.long, device=h.device), token_to_node


class LatentThoughtGraph(nn.Module):
    """Constructs bounded causal top-k edges between latent thought nodes."""

    def __init__(self, node_dim: int, topk_edges: int) -> None:
        super().__init__()
        self.topk_edges = topk_edges
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)

    def summarize(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        bsz, num_nodes, dim = nodes.shape
        if num_nodes == 0:
            empty_edges = torch.zeros(bsz, 0, 0, dtype=nodes.dtype, device=nodes.device)
            return torch.zeros_like(nodes), empty_edges, 0.0

        q = self.q_proj(nodes)
        k = self.k_proj(nodes)
        v = self.v_proj(nodes)
        scale = 1.0 / math.sqrt(dim)

        summary = torch.zeros_like(nodes)
        edge_weights = torch.zeros(bsz, num_nodes, num_nodes, dtype=nodes.dtype, device=nodes.device)
        degree_sum = 0.0
        for node_idx in range(num_nodes):
            usable = node_idx + 1
            edge_count = min(self.topk_edges, usable)
            scores = (q[:, node_idx : node_idx + 1, :] * k[:, :usable, :]).sum(dim=-1) * scale
            top_scores, top_indices = torch.topk(scores, k=edge_count, dim=-1)
            weights = torch.softmax(top_scores, dim=-1)
            gathered = torch.gather(
                v[:, :usable, :],
                dim=1,
                index=top_indices.unsqueeze(-1).expand(-1, -1, dim),
            )
            summary[:, node_idx, :] = (weights.unsqueeze(-1) * gathered).sum(dim=1)
            edge_weights[:, node_idx, :usable].scatter_(dim=-1, index=top_indices, src=weights)
            degree_sum += float(edge_count)

        return summary, edge_weights, degree_sum / float(num_nodes)


class LatentThoughtMessagePassing(nn.Module):
    """Runs bounded message passing updates over explicit latent thought nodes."""

    def __init__(self, node_dim: int) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(node_dim * 3, node_dim),
            nn.Tanh(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        node_writes: torch.Tensor,
        graph: LatentThoughtGraph,
        steps: int,
    ) -> tuple[torch.Tensor, dict[str, float | int], dict[str, torch.Tensor]]:
        if nodes.shape[1] == 0 or steps <= 0:
            return nodes, {
                "mean_thought_degree": 0.0,
                "mean_thought_update_norm": 0.0,
                "thought_graph_steps_used": 0,
            }, {
                "local_delib_thought_edge_stability_loss": nodes.new_zeros(()),
            }

        current = nodes
        degree_accum = 0.0
        update_norm_accum = 0.0
        edge_stability_accum = nodes.new_zeros(())
        edge_stability_steps = 0
        prev_edge_weights = None
        for _ in range(steps):
            neighbor_summary, edge_weights, mean_degree = graph.summarize(current)
            delta = self.update(torch.cat([current, neighbor_summary, node_writes], dim=-1))
            current = current + delta
            degree_accum += mean_degree
            update_norm_accum += float(delta.norm(dim=-1).mean().item())
            if prev_edge_weights is not None:
                edge_stability_accum = edge_stability_accum + (0.5 * (edge_weights - prev_edge_weights).abs().mean())
                edge_stability_steps += 1
            prev_edge_weights = edge_weights

        return current, {
            "mean_thought_degree": degree_accum / float(steps),
            "mean_thought_update_norm": update_norm_accum / float(steps),
            "thought_graph_steps_used": steps,
        }, {
            "local_delib_thought_edge_stability_loss": edge_stability_accum / float(max(edge_stability_steps, 1)),
        }


class TokenToThoughtReadWrite(nn.Module):
    """Handles causal token writes into thought nodes and token reads back out."""

    def __init__(self, state_dim: int, node_dim: int, topk_edges: int) -> None:
        super().__init__()
        self.topk_edges = topk_edges
        self.write_value = nn.Linear(state_dim, node_dim)
        self.write_gate = nn.Linear(state_dim, 1)
        self.read_q = nn.Linear(state_dim, node_dim)
        self.read_k = nn.Linear(node_dim, node_dim)
        self.read_v = nn.Linear(node_dim, node_dim)

    def write(
        self,
        h: torch.Tensor,
        token_to_node: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        bsz, _, _ = h.shape
        node_dim = self.write_value.out_features
        if num_nodes == 0:
            zero = torch.zeros(bsz, 0, node_dim, dtype=h.dtype, device=h.device)
            return zero, 0.0, h.new_zeros(())

        values = self.write_value(h)
        gates = torch.sigmoid(self.write_gate(h))
        node_writes = torch.zeros(bsz, num_nodes, node_dim, dtype=h.dtype, device=h.device)
        denom = torch.zeros(bsz, num_nodes, 1, dtype=h.dtype, device=h.device)

        for node_idx in range(num_nodes):
            token_mask = token_to_node == node_idx
            if not token_mask.any():
                continue
            weights = gates[:, token_mask, :]
            node_writes[:, node_idx, :] = (weights * values[:, token_mask, :]).sum(dim=1)
            denom[:, node_idx, :] = weights.sum(dim=1)

        assigned_tokens = token_to_node >= 0
        mean_weight = 0.0
        mean_weight_tensor = h.new_zeros(())
        if assigned_tokens.any():
            mean_weight_tensor = gates[:, assigned_tokens, :].mean()
            mean_weight = float(mean_weight_tensor.item())
        return node_writes / denom.clamp_min(1e-8), mean_weight, mean_weight_tensor

    def read(
        self,
        h: torch.Tensor,
        nodes: torch.Tensor,
        node_limits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        bsz, seq_len, _ = h.shape
        node_dim = nodes.shape[-1]
        if nodes.shape[1] == 0:
            zeros = torch.zeros(bsz, seq_len, node_dim, dtype=h.dtype, device=h.device)
            return zeros, zeros, 0.0, h.new_zeros(())

        q = self.read_q(h)
        k = self.read_k(nodes)
        v = self.read_v(nodes)
        prefix_nodes = torch.cumsum(nodes, dim=1)
        read_summary = torch.zeros(bsz, seq_len, node_dim, dtype=h.dtype, device=h.device)
        consensus_summary = torch.zeros_like(read_summary)

        weight_accum = 0.0
        weight_tensor_accum = h.new_zeros(())
        active_tokens = 0
        for token_idx in range(seq_len):
            accessible = node_limits <= token_idx
            accessible_count = int(accessible.sum().item())
            if accessible_count == 0:
                continue

            usable_k = min(self.topk_edges, accessible_count)
            scores = (q[:, token_idx : token_idx + 1, :] * k[:, :accessible_count, :]).sum(dim=-1) / math.sqrt(node_dim)
            top_scores, top_indices = torch.topk(scores, k=usable_k, dim=-1)
            weights = torch.softmax(top_scores, dim=-1)
            gathered = torch.gather(
                v[:, :accessible_count, :],
                dim=1,
                index=top_indices.unsqueeze(-1).expand(-1, -1, node_dim),
            )
            read_summary[:, token_idx, :] = (weights.unsqueeze(-1) * gathered).sum(dim=1)
            consensus_summary[:, token_idx, :] = prefix_nodes[:, accessible_count - 1, :] / float(accessible_count)
            weight_accum += float(weights.mean().item())
            weight_tensor_accum = weight_tensor_accum + weights.mean()
            active_tokens += 1

        mean_weight = weight_accum / float(max(active_tokens, 1))
        mean_weight_tensor = weight_tensor_accum / float(max(active_tokens, 1))
        return read_summary, consensus_summary, mean_weight, mean_weight_tensor


class ThoughtConsensusReducer(nn.Module):
    """Reduces thought readouts and prefix consensus back into token-state space."""

    def __init__(self, node_dim: int, state_dim: int) -> None:
        super().__init__()
        self.read_proj = nn.Linear(node_dim, state_dim)
        self.consensus_proj = nn.Linear(node_dim, state_dim)
        self.mix = nn.Linear(state_dim * 2, state_dim)

    def forward(self, read_summary: torch.Tensor, consensus_summary: torch.Tensor) -> torch.Tensor:
        return self.mix(
            torch.cat(
                [
                    self.read_proj(read_summary),
                    self.consensus_proj(consensus_summary),
                ],
                dim=-1,
            )
        )


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
        use_neighbor_graph: bool = False,
        use_phrase_consensus: bool = False,
        adaptive_halt: bool = False,
        branch_factor: int = 0,
        branch_every: int = 1,
        branch_dim: int = 0,
        branch_consensus: bool = False,
        branch_verifier: bool = False,
        branch_consensus_temp: float = 1.0,
        branch_max_active: int = 0,
        branch_disagreement_threshold: float = 0.1,
        use_flocking: bool = False,
        flocking_alignment_weight: float = 0.0,
        flocking_cohesion_weight: float = 0.0,
        flocking_separation_weight: float = 0.0,
        flocking_separation_margin: float = 1.0,
        flocking_radius_cap: int = 0,
        hierarchy_chunk_sizes: list[int] | None = None,
        use_deep_hierarchy: bool = False,
        span_chunk_size: int = 0,
        sequence_summary: bool = False,
        hierarchy_bidirectional: bool = False,
        hierarchy_scale_gate: bool = False,
        scratch_slots: int = 0,
        scratch_dim: int = 0,
        scratch_refine_steps: int = 0,
        scratch_use_branch_inputs: bool = False,
        scratch_use_hierarchy_inputs: bool = False,
        scratch_export_summary: bool = False,
        scratch_summary_dim: int = 0,
        use_thought_graph: bool = False,
        thought_node_budget: int = 8,
        thought_node_dim: int = 0,
        thought_graph_steps: int = 1,
        thought_topk_edges: int = 2,
        thought_token_chunk_size: int = 4,
        thought_use_branch_inputs: bool = True,
        thought_use_hierarchy_inputs: bool = True,
        thought_use_scratch_inputs: bool = True,
        global_anchor_count: int = 0,
        global_anchor_dim: int = 0,
        global_anchor_update: bool = False,
        global_anchor_temp: float = 1.0,
        global_anchor_use_hierarchy: bool = False,
        global_anchor_use_scratch: bool = False,
        global_anchor_use_thought: bool = False,
    ) -> None:
        super().__init__()
        if micro_steps < 1:
            raise ValueError("micro_steps must be >= 1")
        if semantic_topk < 0:
            raise ValueError("semantic_topk must be >= 0")
        if semantic_lookback < 1:
            raise ValueError("semantic_lookback must be >= 1")
        if branch_factor < 0:
            raise ValueError("branch_factor must be >= 0")
        if branch_every < 1:
            raise ValueError("branch_every must be >= 1")
        if (branch_consensus or branch_verifier) and branch_factor < 1:
            raise ValueError("branch_consensus/branch_verifier requires branch_factor > 0")
        if branch_consensus_temp <= 0.0:
            raise ValueError("branch_consensus_temp must be > 0")
        if branch_max_active < 0:
            raise ValueError("branch_max_active must be >= 0")
        if branch_disagreement_threshold < 0.0:
            raise ValueError("branch_disagreement_threshold must be >= 0")
        if use_flocking and not use_neighbor_graph:
            raise ValueError("use_flocking requires use_neighbor_graph")
        if flocking_alignment_weight < 0.0:
            raise ValueError("flocking_alignment_weight must be >= 0")
        if flocking_cohesion_weight < 0.0:
            raise ValueError("flocking_cohesion_weight must be >= 0")
        if flocking_separation_weight < 0.0:
            raise ValueError("flocking_separation_weight must be >= 0")
        if flocking_separation_margin < 0.0:
            raise ValueError("flocking_separation_margin must be >= 0")
        if flocking_radius_cap < 0:
            raise ValueError("flocking_radius_cap must be >= 0")
        if span_chunk_size < 0:
            raise ValueError("span_chunk_size must be >= 0")
        if use_deep_hierarchy and span_chunk_size > 0 and span_chunk_size < phrase_chunk_size:
            raise ValueError("span_chunk_size must be >= phrase_chunk_size when deep hierarchy is enabled")
        if scratch_slots < 0:
            raise ValueError("scratch_slots must be >= 0")
        if scratch_dim < 0:
            raise ValueError("scratch_dim must be >= 0")
        if scratch_slots > 0 and scratch_dim < 1:
            raise ValueError("scratch_dim must be >= 1 when scratch_slots > 0")
        if scratch_refine_steps < 0:
            raise ValueError("scratch_refine_steps must be >= 0")
        if scratch_summary_dim < 0:
            raise ValueError("scratch_summary_dim must be >= 0")
        if scratch_export_summary and scratch_slots < 1:
            raise ValueError("scratch_export_summary requires scratch_slots > 0")
        if thought_node_budget < 0:
            raise ValueError("thought_node_budget must be >= 0")
        if thought_node_dim < 0:
            raise ValueError("thought_node_dim must be >= 0")
        if thought_graph_steps < 0:
            raise ValueError("thought_graph_steps must be >= 0")
        if thought_topk_edges < 1:
            raise ValueError("thought_topk_edges must be >= 1")
        if thought_token_chunk_size < 1:
            raise ValueError("thought_token_chunk_size must be >= 1")
        if use_thought_graph and thought_node_budget < 1:
            raise ValueError("use_thought_graph requires thought_node_budget >= 1")
        if use_thought_graph and thought_graph_steps < 1:
            raise ValueError("use_thought_graph requires thought_graph_steps >= 1")
        if global_anchor_count < 0:
            raise ValueError("global_anchor_count must be >= 0")
        if global_anchor_dim < 0:
            raise ValueError("global_anchor_dim must be >= 0")
        if global_anchor_update and global_anchor_count < 1:
            raise ValueError("global_anchor_update requires global_anchor_count > 0")
        if global_anchor_temp <= 0.0:
            raise ValueError("global_anchor_temp must be > 0")
        if (
            global_anchor_use_hierarchy
            or global_anchor_use_scratch
            or global_anchor_use_thought
        ) and global_anchor_count < 1:
            raise ValueError("global anchor input projections require global_anchor_count > 0")

        self.micro_steps = micro_steps
        self.use_token_gate = use_token_gate
        self.semantic_topk = semantic_topk
        self.semantic_lookback = semantic_lookback
        self.use_neighbor_graph = use_neighbor_graph
        self.use_phrase_consensus = use_phrase_consensus
        self.adaptive_halt = adaptive_halt
        self.branch_factor = branch_factor
        self.branch_every = branch_every
        self.branch_dim = branch_dim if branch_dim > 0 else state_dim
        self.branch_consensus = branch_consensus
        self.branch_verifier = branch_verifier
        self.branch_consensus_temp = branch_consensus_temp
        self.branch_max_active = branch_max_active
        self.branch_disagreement_threshold = branch_disagreement_threshold
        self.use_flocking = use_flocking
        self.flocking_alignment_weight = flocking_alignment_weight
        self.flocking_cohesion_weight = flocking_cohesion_weight
        self.flocking_separation_weight = flocking_separation_weight
        self.flocking_separation_margin = flocking_separation_margin
        self.flocking_radius_cap = flocking_radius_cap
        self.hierarchy_chunk_sizes = tuple(hierarchy_chunk_sizes or [])
        self.use_deep_hierarchy = use_deep_hierarchy
        self.span_chunk_size = span_chunk_size
        self.sequence_summary = sequence_summary
        self.hierarchy_bidirectional = hierarchy_bidirectional
        self.hierarchy_scale_gate = hierarchy_scale_gate
        self.scratch_slots = scratch_slots
        self.scratch_dim = scratch_dim if scratch_slots > 0 else 0
        self.scratch_refine_steps = scratch_refine_steps
        self.scratch_use_branch_inputs = scratch_use_branch_inputs
        self.scratch_use_hierarchy_inputs = scratch_use_hierarchy_inputs
        self.scratch_export_summary = scratch_export_summary
        self.scratch_summary_dim = (
            (scratch_summary_dim if scratch_summary_dim > 0 else state_dim)
            if scratch_export_summary and scratch_slots > 0
            else 0
        )
        self.use_thought_graph = use_thought_graph
        self.thought_node_budget = thought_node_budget
        self.thought_node_dim = thought_node_dim if thought_node_dim > 0 else state_dim
        self.thought_graph_steps = thought_graph_steps
        self.thought_topk_edges = thought_topk_edges
        self.thought_token_chunk_size = thought_token_chunk_size
        self.thought_use_branch_inputs = thought_use_branch_inputs
        self.thought_use_hierarchy_inputs = thought_use_hierarchy_inputs
        self.thought_use_scratch_inputs = thought_use_scratch_inputs
        self.global_anchor_count = global_anchor_count
        self.global_anchor_dim = global_anchor_dim if global_anchor_count > 0 and global_anchor_dim > 0 else state_dim
        self.global_anchor_update = global_anchor_update
        self.global_anchor_temp = global_anchor_temp
        self.global_anchor_use_hierarchy = global_anchor_use_hierarchy
        self.global_anchor_use_scratch = global_anchor_use_scratch
        self.global_anchor_use_thought = global_anchor_use_thought
        for chunk_size in self.hierarchy_chunk_sizes:
            if chunk_size < 1:
                raise ValueError("hierarchy chunk sizes must be >= 1")

        self.in_proj = nn.Linear(model_dim, state_dim)
        self.mixer = CausalDepthwiseMixer(state_dim, kernel_size=kernel_size)
        self.phrase_pool = PhrasePool(state_dim, chunk_size=phrase_chunk_size)
        self.phrase_consensus = PhraseConsensusHead(state_dim, chunk_size=phrase_chunk_size)
        self.state_head = TokenStateHead(state_dim)
        self.halt_threshold_logit = nn.Parameter(torch.tensor(4.59511985013459))
        self.hierarchy_levels = nn.ModuleList(
            [HierarchyPoolBroadcast(state_dim=state_dim, chunk_size=chunk_size) for chunk_size in self.hierarchy_chunk_sizes]
        )
        self.deep_phrase_scale: CausalHierarchyScale | None = None
        self.deep_span_scale: CausalHierarchyScale | None = None
        self.deep_sequence_scale: CausalHierarchyScale | None = None

        update_in_dim = state_dim * 3
        if self.hierarchy_levels:
            update_in_dim += state_dim
        if self.use_deep_hierarchy:
            self.deep_phrase_scale = CausalHierarchyScale(state_dim=state_dim, chunk_size=phrase_chunk_size)
            if self.span_chunk_size > 0:
                self.deep_span_scale = CausalHierarchyScale(state_dim=state_dim, chunk_size=self.span_chunk_size)
            if self.sequence_summary:
                self.deep_sequence_scale = CausalHierarchyScale(state_dim=state_dim, sequence_summary=True)
            update_in_dim += state_dim
        if self.use_phrase_consensus:
            update_in_dim += state_dim
        if self.semantic_topk > 0:
            self.semantic_q = nn.Linear(state_dim, state_dim)
            self.semantic_k = nn.Linear(state_dim, state_dim)
            self.semantic_v = nn.Linear(state_dim, state_dim)
        if self.semantic_topk > 0 or self.use_neighbor_graph:
            update_in_dim += state_dim
        if self.use_neighbor_graph and self.use_flocking:
            update_in_dim += state_dim
        if self.use_neighbor_graph:
            self.neighbor_graph_mixer = CausalNeighborGraphMixer(
                state_dim=state_dim,
                phrase_chunk_size=phrase_chunk_size,
                semantic_topk=semantic_topk,
                semantic_lookback=semantic_lookback,
                use_phrase_nodes=use_phrase_consensus,
                use_flocking=use_flocking,
                flocking_alignment_weight=flocking_alignment_weight,
                flocking_cohesion_weight=flocking_cohesion_weight,
                flocking_separation_weight=flocking_separation_weight,
                flocking_separation_margin=flocking_separation_margin,
                flocking_radius_cap=flocking_radius_cap,
            )
        if self.scratch_slots > 0:
            self.scratch_query = nn.Linear(state_dim, self.scratch_dim)
            self.scratch_write_value = nn.Linear(state_dim, self.scratch_dim)
            self.scratch_read_mix = nn.Linear(self.scratch_dim, self.scratch_dim)
            self.scratch_to_state = nn.Linear(self.scratch_dim, state_dim)
            self.scratch_init = nn.Parameter(torch.zeros(self.scratch_slots, self.scratch_dim))
            self.scratch_read_temp = nn.Parameter(torch.tensor(1.0))
            self.scratch_persist_gate = nn.Linear(state_dim, 1)
            if self.scratch_refine_steps > 0:
                self.scratch_refine = nn.Sequential(
                    nn.Linear(self.scratch_dim * 2, self.scratch_dim),
                    nn.Tanh(),
                    nn.Linear(self.scratch_dim, self.scratch_dim),
                )
            else:
                self.scratch_refine = None
            if self.scratch_use_branch_inputs:
                self.scratch_branch_write = nn.Linear(state_dim, self.scratch_dim)
                self.scratch_branch_gate = nn.Linear(state_dim * 2, 1)
            if self.scratch_use_hierarchy_inputs:
                self.scratch_hierarchy_write = nn.Linear(state_dim, self.scratch_dim)
                self.scratch_hierarchy_gate = nn.Linear(state_dim * 2, 1)
            if self.scratch_export_summary:
                self.scratch_summary_proj = nn.Linear(self.scratch_dim, self.scratch_summary_dim)
            update_in_dim += state_dim
        if self.use_thought_graph:
            self.thought_node_builder = LatentThoughtNodeBuilder(state_dim=state_dim, node_dim=self.thought_node_dim)
            self.thought_graph = LatentThoughtGraph(node_dim=self.thought_node_dim, topk_edges=self.thought_topk_edges)
            self.thought_message_passing = LatentThoughtMessagePassing(node_dim=self.thought_node_dim)
            self.token_to_thought = TokenToThoughtReadWrite(
                state_dim=state_dim,
                node_dim=self.thought_node_dim,
                topk_edges=self.thought_topk_edges,
            )
            self.thought_consensus_reducer = ThoughtConsensusReducer(
                node_dim=self.thought_node_dim,
                state_dim=state_dim,
            )
            update_in_dim += state_dim
        if self.global_anchor_count > 0:
            self.global_anchor_query = nn.Linear(state_dim, self.global_anchor_dim)
            self.global_anchor_key = nn.Linear(self.global_anchor_dim, self.global_anchor_dim)
            self.global_anchor_value = nn.Linear(self.global_anchor_dim, self.global_anchor_dim)
            self.global_anchor_to_state = nn.Linear(self.global_anchor_dim, state_dim)
            self.global_anchor_token_write = nn.Linear(state_dim, self.global_anchor_dim)
            self.global_anchor_prefix_write = nn.Linear(state_dim, self.global_anchor_dim)
            self.global_anchor_write_query = nn.Linear(state_dim, self.global_anchor_dim)
            self.global_anchor_init = nn.Parameter(torch.zeros(self.global_anchor_count, self.global_anchor_dim))
            self.global_anchor_persist_gate = nn.Linear(state_dim, 1)
            if self.global_anchor_use_hierarchy:
                self.global_anchor_hierarchy_write = nn.Linear(state_dim, self.global_anchor_dim)
                self.global_anchor_hierarchy_gate = nn.Linear(state_dim * 2, 1)
            if self.global_anchor_use_scratch:
                self.global_anchor_scratch_write = nn.Linear(state_dim, self.global_anchor_dim)
                self.global_anchor_scratch_gate = nn.Linear(state_dim * 2, 1)
            if self.global_anchor_use_thought:
                self.global_anchor_thought_write = nn.Linear(state_dim, self.global_anchor_dim)
                self.global_anchor_thought_gate = nn.Linear(state_dim * 2, 1)
            update_in_dim += state_dim

        self.update = nn.Sequential(
            nn.Linear(update_in_dim, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, state_dim),
        )
        self.out_proj = nn.Linear(state_dim, model_dim)

        if self.branch_factor > 0:
            self.branch_proposal = BranchProposalHead(
                state_dim=state_dim,
                branch_factor=self.branch_factor,
                branch_dim=self.branch_dim,
            )
            self.branch_scorer = BranchScorer(state_dim=state_dim)
            self.branch_merge = BranchMergeHead(state_dim=state_dim)
            self.branch_verifier_head = BranchVerifierHead(state_dim=state_dim)
            self.branch_consensus_merge = BranchConsensusMergeHead(state_dim=state_dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        if self.scratch_slots > 0:
            nn.init.zeros_(self.scratch_to_state.weight)
            nn.init.zeros_(self.scratch_to_state.bias)
        if self.global_anchor_count > 0:
            nn.init.zeros_(self.global_anchor_to_state.weight)
            nn.init.zeros_(self.global_anchor_to_state.bias)
        self.last_aux_losses: dict[str, torch.Tensor] | None = None

    def _allocate_scratch_prefix_state(
        self,
        *,
        bsz: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        scratch = self.scratch_init.unsqueeze(0).to(device=device, dtype=dtype).expand(bsz, -1, -1).clone()
        return scratch.unsqueeze(1).expand(-1, seq_len + 1, -1, -1).clone()

    def _compute_scratch_feedback(
        self,
        h: torch.Tensor,
        head_states: dict[str, torch.Tensor],
        *,
        scratch_prefix_state: torch.Tensor,
        branch_summary: torch.Tensor | None = None,
        hierarchy_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int], list[float] | None]:
        bsz, seq_len, _ = h.shape
        scratch = scratch_prefix_state[:, 0, :, :].clone()
        scratch_feedback = torch.zeros(bsz, seq_len, h.shape[-1], device=h.device, dtype=h.dtype)
        next_prefix_states = [scratch.clone()]

        read_weight_sum = 0.0
        write_weight_sum = 0.0
        refine_norm_sum = 0.0
        summary_norm_sum = 0.0
        branch_weight_sum = 0.0
        hierarchy_weight_sum = 0.0
        slot_write_mass = torch.zeros(bsz, self.scratch_slots, device=h.device, dtype=h.dtype)

        read_temp = torch.clamp(self.scratch_read_temp, min=0.1)
        for token_idx in range(seq_len):
            token_state = h[:, token_idx, :]
            persist_gate = torch.sigmoid(self.scratch_persist_gate(token_state)).unsqueeze(-1)
            scratch = scratch + persist_gate * (scratch_prefix_state[:, token_idx, :, :] - scratch)
            query = self.scratch_query(token_state)
            slot_logits = torch.einsum("bd,bsd->bs", query, scratch) / math.sqrt(self.scratch_dim)
            slot_attn = torch.softmax(slot_logits * read_temp, dim=-1)

            read_gate = head_states["uncertainty"][:, token_idx, :]
            read_summary = torch.einsum("bs,bsd->bd", slot_attn, scratch)
            scratch_feedback[:, token_idx, :] = self.scratch_to_state(read_summary * read_gate)
            read_weight_sum += float((slot_attn.mean(dim=-1) * read_gate.squeeze(-1)).mean().item())

            write_gate = (head_states["salience"][:, token_idx, :] * head_states["uncertainty"][:, token_idx, :]).squeeze(-1)
            write_value = self.scratch_write_value(token_state) + self.scratch_read_mix(read_summary)
            if self.scratch_use_branch_inputs and branch_summary is not None:
                branch_state = branch_summary[:, token_idx, :]
                branch_gate = torch.sigmoid(self.scratch_branch_gate(torch.cat([token_state, branch_state], dim=-1)))
                write_value = write_value + branch_gate * self.scratch_branch_write(branch_state)
                branch_weight_sum += float(branch_gate.mean().item())
            if self.scratch_use_hierarchy_inputs and hierarchy_summary is not None:
                hierarchy_state = hierarchy_summary[:, token_idx, :]
                hierarchy_gate = torch.sigmoid(self.scratch_hierarchy_gate(torch.cat([token_state, hierarchy_state], dim=-1)))
                write_value = write_value + hierarchy_gate * self.scratch_hierarchy_write(hierarchy_state)
                hierarchy_weight_sum += float(hierarchy_gate.mean().item())
            write_weights = slot_attn * write_gate.unsqueeze(-1)
            scratch = scratch + write_weights.unsqueeze(-1) * write_value.unsqueeze(1)
            if self.scratch_refine is not None:
                for _ in range(self.scratch_refine_steps):
                    scratch_summary = scratch.mean(dim=1, keepdim=True).expand_as(scratch)
                    refine_delta = self.scratch_refine(torch.cat([scratch, scratch_summary], dim=-1))
                    scratch = scratch + torch.tanh(refine_delta)
                    refine_norm_sum += float(refine_delta.norm(dim=-1).mean().item())
            summary_norm_sum += float(scratch.mean(dim=1).norm(dim=-1).mean().item())
            slot_write_mass += write_weights
            write_weight_sum += float(write_weights.mean().item())
            next_prefix_states.append(scratch.clone())

        steps = float(max(seq_len, 1))
        next_prefix = torch.stack(next_prefix_states, dim=1)
        scratch_summary_vector = None
        mean_scratch_summary_norm = summary_norm_sum / steps
        if self.scratch_export_summary:
            exported = self.scratch_summary_proj(next_prefix[:, -1, :, :].mean(dim=1))
            scratch_summary_vector = [float(v) for v in exported.mean(dim=0).detach().cpu().tolist()]
            mean_scratch_summary_norm = float(exported.norm(dim=-1).mean().item())
        scratch_slots_used = int((slot_write_mass > 1e-4).any(dim=0).sum().item())
        mean_scratch_norm = float(next_prefix[:, -1, :, :].norm(dim=-1).mean().item())
        stats: dict[str, float | int] = {
            "mean_scratch_read_weight": read_weight_sum / steps,
            "mean_scratch_write_weight": write_weight_sum / steps,
            "scratch_slots_used": scratch_slots_used,
            "mean_scratch_norm": mean_scratch_norm,
            "mean_scratch_refine_norm": refine_norm_sum / float(max(seq_len * max(self.scratch_refine_steps, 1), 1)),
            "mean_scratch_summary_norm": mean_scratch_summary_norm,
            "mean_branch_to_scratch_weight": branch_weight_sum / steps,
            "mean_hierarchy_to_scratch_weight": hierarchy_weight_sum / steps,
            "scratch_reset_ok": 1.0,
        }
        return scratch_feedback, next_prefix, stats, scratch_summary_vector

    def _allocate_global_anchor_prefix_state(
        self,
        *,
        bsz: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        anchors = self.global_anchor_init.unsqueeze(0).to(device=device, dtype=dtype).expand(bsz, -1, -1).clone()
        return anchors.unsqueeze(1).expand(-1, seq_len + 1, -1, -1).clone()

    def _compute_global_anchor_feedback(
        self,
        h: torch.Tensor,
        head_states: dict[str, torch.Tensor],
        *,
        global_anchor_prefix_state: torch.Tensor,
        hierarchy_summary: torch.Tensor | None = None,
        scratch_summary: torch.Tensor | None = None,
        thought_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int], dict[str, torch.Tensor]]:
        bsz, seq_len, _ = h.shape
        anchors = global_anchor_prefix_state[:, 0, :, :].clone()
        anchor_feedback = torch.zeros(bsz, seq_len, h.shape[-1], device=h.device, dtype=h.dtype)
        next_prefix_states = [anchors.clone()]

        prefix_sum = torch.zeros(bsz, h.shape[-1], device=h.device, dtype=h.dtype)
        read_weight_sum = 0.0
        write_weight_sum = 0.0
        read_weight_tensor = h.new_zeros(())
        write_weight_tensor = h.new_zeros(())
        anchor_usage_mass = torch.zeros(bsz, self.global_anchor_count, device=h.device, dtype=h.dtype)
        scale = math.sqrt(self.global_anchor_dim)

        for token_idx in range(seq_len):
            token_state = h[:, token_idx, :]
            persist_gate = torch.sigmoid(self.global_anchor_persist_gate(token_state)).unsqueeze(-1)
            anchors = anchors + persist_gate * (global_anchor_prefix_state[:, token_idx, :, :] - anchors)

            anchor_keys = self.global_anchor_key(anchors)
            anchor_values = self.global_anchor_value(anchors)
            query = self.global_anchor_query(token_state)
            read_logits = torch.einsum("bd,bad->ba", query, anchor_keys) / scale
            read_attn = torch.softmax(read_logits / self.global_anchor_temp, dim=-1)
            read_gate = head_states["uncertainty"][:, token_idx, :].squeeze(-1)
            read_summary = torch.einsum("ba,bad->bd", read_attn, anchor_values)
            anchor_feedback[:, token_idx, :] = self.global_anchor_to_state(read_summary * read_gate.unsqueeze(-1))
            read_weights = read_attn * read_gate.unsqueeze(-1)
            read_weight_sum += float(read_weights.mean().item())
            read_weight_tensor = read_weight_tensor + read_weights.mean()
            anchor_usage_mass = anchor_usage_mass + read_weights

            prefix_sum = prefix_sum + token_state
            if self.global_anchor_update:
                prefix_mean = prefix_sum / float(token_idx + 1)
                write_query = self.global_anchor_write_query(prefix_mean)
                write_logits = torch.einsum("bd,bad->ba", write_query, anchor_keys) / scale
                write_attn = torch.softmax(write_logits / self.global_anchor_temp, dim=-1)
                write_gate = (
                    head_states["salience"][:, token_idx, :] * head_states["uncertainty"][:, token_idx, :]
                ).squeeze(-1)
                write_value = self.global_anchor_token_write(token_state) + self.global_anchor_prefix_write(prefix_mean)
                if self.global_anchor_use_hierarchy and hierarchy_summary is not None:
                    hierarchy_state = hierarchy_summary[:, token_idx, :]
                    hierarchy_gate = torch.sigmoid(
                        self.global_anchor_hierarchy_gate(torch.cat([token_state, hierarchy_state], dim=-1))
                    )
                    write_value = write_value + hierarchy_gate * self.global_anchor_hierarchy_write(hierarchy_state)
                if self.global_anchor_use_scratch and scratch_summary is not None:
                    scratch_state = scratch_summary[:, token_idx, :]
                    scratch_gate = torch.sigmoid(
                        self.global_anchor_scratch_gate(torch.cat([token_state, scratch_state], dim=-1))
                    )
                    write_value = write_value + scratch_gate * self.global_anchor_scratch_write(scratch_state)
                if self.global_anchor_use_thought and thought_summary is not None:
                    thought_state = thought_summary[:, token_idx, :]
                    thought_gate = torch.sigmoid(
                        self.global_anchor_thought_gate(torch.cat([token_state, thought_state], dim=-1))
                    )
                    write_value = write_value + thought_gate * self.global_anchor_thought_write(thought_state)
                write_weights = write_attn * write_gate.unsqueeze(-1)
                anchors = anchors + write_weights.unsqueeze(-1) * torch.tanh(write_value).unsqueeze(1)
                write_weight_sum += float(write_weights.mean().item())
                write_weight_tensor = write_weight_tensor + write_weights.mean()
                anchor_usage_mass = anchor_usage_mass + write_weights

            next_prefix_states.append(anchors.clone())

        next_prefix = torch.stack(next_prefix_states, dim=1)
        mean_read_weight_tensor = read_weight_tensor / float(max(seq_len, 1))
        mean_write_weight_tensor = write_weight_tensor / float(max(seq_len, 1))
        if self.global_anchor_update:
            anchor_usage = 0.5 * (mean_read_weight_tensor + mean_write_weight_tensor)
        else:
            anchor_usage = mean_read_weight_tensor
        stats: dict[str, float | int] = {
            "global_anchors_used": int((anchor_usage_mass > 1e-4).any(dim=0).sum().item()),
            "mean_anchor_read_weight": read_weight_sum / float(max(seq_len, 1)),
            "mean_anchor_write_weight": write_weight_sum / float(max(seq_len, 1)),
            "mean_anchor_norm": float(next_prefix[:, -1, :, :].norm(dim=-1).mean().item()),
        }
        aux_losses = {
            "local_delib_anchor_usage_loss": 1.0 - anchor_usage.clamp(0.0, 1.0),
        }
        return anchor_feedback, next_prefix, stats, aux_losses

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

    def _active_branch_mask(self, branch_logits: torch.Tensor) -> tuple[torch.Tensor, int]:
        active_count = self.branch_factor
        if (self.branch_consensus or self.branch_verifier) and self.branch_max_active > 0:
            active_count = min(self.branch_factor, self.branch_max_active)
        active_mask = torch.ones_like(branch_logits, dtype=torch.bool)
        if active_count < self.branch_factor:
            top_indices = torch.topk(branch_logits, k=active_count, dim=-1).indices
            active_mask = torch.zeros_like(branch_logits, dtype=torch.bool)
            active_mask.scatter_(dim=-1, index=top_indices, value=True)
        return active_mask, active_count

    def _masked_branch_softmax(
        self,
        branch_logits: torch.Tensor,
        active_mask: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        masked_logits = branch_logits / temperature
        min_value = torch.finfo(branch_logits.dtype).min
        masked_logits = masked_logits.masked_fill(~active_mask, min_value)
        weights = torch.softmax(masked_logits, dim=-1)
        weights = weights * active_mask.to(weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _branch_consensus_summary(
        self,
        branch_proposals: torch.Tensor,
        branch_logits: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active_weights = self._masked_branch_softmax(branch_logits, active_mask, temperature=1.0)
        active_mask_f = active_mask.to(branch_proposals.dtype)
        pairwise_delta = branch_proposals.unsqueeze(3) - branch_proposals.unsqueeze(2)
        pairwise_distance = pairwise_delta.norm(dim=-1)
        pairwise_agreement = torch.exp(-pairwise_distance)

        eye = torch.eye(self.branch_factor, device=branch_logits.device, dtype=branch_proposals.dtype).view(
            1, 1, self.branch_factor, self.branch_factor
        )
        off_diag = active_mask_f.unsqueeze(-1) * active_mask_f.unsqueeze(-2) * (1.0 - eye)

        pair_weights = active_weights.unsqueeze(-1) * active_weights.unsqueeze(-2) * off_diag
        disagreement = 1.0 - pairwise_agreement
        token_disagreement = (disagreement * pair_weights).sum(dim=(-1, -2)) / pair_weights.sum(dim=(-1, -2)).clamp_min(1e-8)

        support_weights = active_weights.unsqueeze(-2) * off_diag
        consensus_support = (pairwise_agreement * support_weights).sum(dim=-1) / support_weights.sum(dim=-1).clamp_min(1e-8)

        consensus_logits = branch_logits + consensus_support
        consensus_weights = self._masked_branch_softmax(
            consensus_logits,
            active_mask,
            temperature=self.branch_consensus_temp,
        )
        consensus_summary = (consensus_weights.unsqueeze(-1) * branch_proposals).sum(dim=2)
        return consensus_summary, token_disagreement, consensus_weights

    def _compute_branch_context(self, h: torch.Tensor) -> dict[str, torch.Tensor | float | int]:
        branch_proposals = self.branch_proposal(h)
        branch_logits = self.branch_scorer(h, branch_proposals)
        branch_scores = torch.sigmoid(branch_logits)
        active_mask, active_count = self._active_branch_mask(branch_logits)
        active_mask_f = active_mask.to(branch_logits.dtype)
        branch_weights = self._masked_branch_softmax(branch_logits, active_mask, temperature=1.0)
        branch_summary = (branch_weights.unsqueeze(-1) * branch_proposals).sum(dim=2)

        token_disagreement = torch.zeros(
            branch_logits.shape[:2],
            dtype=branch_logits.dtype,
            device=branch_logits.device,
        )
        consensus_summary = branch_summary
        consensus_mask = torch.zeros_like(token_disagreement)
        if self.branch_consensus:
            consensus_summary, token_disagreement, _ = self._branch_consensus_summary(
                branch_proposals,
                branch_logits,
                active_mask,
            )
            consensus_mask = (token_disagreement > self.branch_disagreement_threshold).to(branch_logits.dtype)

        verifier_scores = torch.zeros_like(branch_logits)
        effective_branch_weights = branch_weights
        if self.branch_verifier:
            verifier_context = consensus_summary if self.branch_consensus else branch_summary
            verifier_logits = self.branch_verifier_head(h, branch_proposals, verifier_context)
            verifier_scores = torch.sigmoid(verifier_logits)
            effective_branch_weights = self._masked_branch_softmax(
                branch_logits + verifier_logits,
                active_mask,
                temperature=1.0,
            )
            branch_summary = (effective_branch_weights.unsqueeze(-1) * branch_proposals).sum(dim=2)

        entropy = -(effective_branch_weights * torch.log(effective_branch_weights.clamp_min(1e-8))).sum(dim=-1)
        max_entropy = math.log(float(active_count)) if active_count > 1 else 0.0
        return {
            "branch_proposals": branch_proposals,
            "branch_logits": branch_logits,
            "branch_scores": branch_scores,
            "active_mask": active_mask,
            "active_mask_f": active_mask_f,
            "active_count": active_count,
            "branch_summary": branch_summary,
            "consensus_summary": consensus_summary,
            "consensus_mask": consensus_mask,
            "token_disagreement": token_disagreement,
            "verifier_scores": verifier_scores,
            "effective_branch_weights": effective_branch_weights,
            "entropy": entropy,
            "max_entropy": max_entropy,
        }

    def _compute_thought_feedback(
        self,
        h: torch.Tensor,
        *,
        branch_summary: torch.Tensor | None = None,
        hierarchy_summary: torch.Tensor | None = None,
        scratch_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | int], dict[str, torch.Tensor]]:
        zero_feedback = torch.zeros_like(h)
        if not self.use_thought_graph:
            return zero_feedback, {
                "thought_nodes_used": 0,
                "mean_thought_degree": 0.0,
                "mean_token_to_thought_weight": 0.0,
                "mean_thought_to_token_weight": 0.0,
                "mean_thought_update_norm": 0.0,
                "thought_graph_steps_used": 0,
            }, {
                "local_delib_thought_edge_stability_loss": h.new_zeros(()),
                "local_delib_thought_node_utilization_loss": h.new_zeros(()),
            }

        nodes, node_limits, token_to_node = self.thought_node_builder(
            h,
            token_chunk_size=self.thought_token_chunk_size,
            node_budget=self.thought_node_budget,
            branch_summary=branch_summary,
            hierarchy_summary=hierarchy_summary,
            scratch_summary=scratch_summary,
            use_branch_inputs=self.thought_use_branch_inputs,
            use_hierarchy_inputs=self.thought_use_hierarchy_inputs,
            use_scratch_inputs=self.thought_use_scratch_inputs,
        )
        thought_nodes_used = int(nodes.shape[1])
        if thought_nodes_used == 0:
            return zero_feedback, {
                "thought_nodes_used": 0,
                "mean_thought_degree": 0.0,
                "mean_token_to_thought_weight": 0.0,
                "mean_thought_to_token_weight": 0.0,
                "mean_thought_update_norm": 0.0,
                "thought_graph_steps_used": 0,
            }, {
                "local_delib_thought_edge_stability_loss": h.new_zeros(()),
                "local_delib_thought_node_utilization_loss": h.new_zeros(()),
            }

        node_writes, mean_token_to_thought_weight, token_to_thought_weight = self.token_to_thought.write(
            h,
            token_to_node=token_to_node,
            num_nodes=thought_nodes_used,
        )
        nodes, graph_stats, graph_aux = self.thought_message_passing(
            nodes,
            node_writes=node_writes,
            graph=self.thought_graph,
            steps=self.thought_graph_steps,
        )
        read_summary, consensus_summary, mean_thought_to_token_weight, thought_to_token_weight = self.token_to_thought.read(
            h,
            nodes=nodes,
            node_limits=node_limits,
        )
        thought_utilization = 0.5 * (token_to_thought_weight + thought_to_token_weight).clamp(0.0, 1.0)
        return self.thought_consensus_reducer(read_summary, consensus_summary), {
            "thought_nodes_used": thought_nodes_used,
            "mean_thought_degree": float(graph_stats["mean_thought_degree"]),
            "mean_token_to_thought_weight": mean_token_to_thought_weight,
            "mean_thought_to_token_weight": mean_thought_to_token_weight,
            "mean_thought_update_norm": float(graph_stats["mean_thought_update_norm"]),
            "thought_graph_steps_used": int(graph_stats["thought_graph_steps_used"]),
        }, {
            "local_delib_thought_edge_stability_loss": graph_aux["local_delib_thought_edge_stability_loss"],
            "local_delib_thought_node_utilization_loss": 1.0 - thought_utilization,
        }

    def _compute_deep_hierarchy_feedback(
        self,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | int], dict[str, torch.Tensor]]:
        zero_feedback = torch.zeros_like(h)
        if not self.use_deep_hierarchy or self.deep_phrase_scale is None:
            return zero_feedback, {
                "phrase_nodes_used": 0,
                "span_nodes_used": 0,
                "sequence_summary_used": 0,
                "mean_upward_message_norm": 0.0,
                "mean_downward_message_norm": 0.0,
                "mean_scale_gate": 0.0,
                "hierarchy_depth_used": 0,
            }, {
                "local_delib_hierarchy_agreement_loss": h.new_zeros(()),
            }

        levels: list[tuple[str, CausalHierarchyScale]] = [("phrase", self.deep_phrase_scale)]
        if self.deep_span_scale is not None:
            levels.append(("span", self.deep_span_scale))
        if self.deep_sequence_scale is not None:
            levels.append(("sequence", self.deep_sequence_scale))

        summaries: list[torch.Tensor] = []
        upward_norm_sum = 0.0
        upward_steps = 0
        source = h
        for idx, (_, level) in enumerate(levels):
            summary = level.summarize(source)
            summaries.append(summary)
            if idx < len(levels) - 1:
                source = torch.tanh(level.up_proj(summary))
                upward_norm_sum += float(source.norm(dim=-1).mean().item())
                upward_steps += 1

        refined = list(summaries)
        downward_norm_sum = 0.0
        downward_steps = 0
        if self.hierarchy_bidirectional and len(refined) > 1:
            for idx in range(len(refined) - 1, 0, -1):
                parent = refined[idx]
                child_level = levels[idx - 1][1]
                downward = torch.tanh(child_level.down_proj(parent))
                refined[idx - 1] = refined[idx - 1] + downward
                downward_norm_sum += float(downward.norm(dim=-1).mean().item())
                downward_steps += 1

        token_messages = []
        mean_scale_gate = 0.0
        mean_scale_gate_tensor = h.new_zeros(())
        for summary, (_, level) in zip(refined, levels):
            token_message = torch.tanh(level.to_token_proj(summary))
            if self.hierarchy_scale_gate:
                gate = torch.sigmoid(level.gate(torch.cat([h, summary], dim=-1)))
            else:
                gate = torch.ones(h.shape[0], h.shape[1], 1, device=h.device, dtype=h.dtype)
            token_messages.append(token_message * gate)
            gate_mean = gate.mean()
            mean_scale_gate += float(gate_mean.item())
            mean_scale_gate_tensor = mean_scale_gate_tensor + gate_mean

        adjacent_disagreement = h.new_zeros(())
        adjacent_pairs = 0
        if len(refined) > 1:
            for idx in range(len(refined) - 1):
                adjacent_disagreement = adjacent_disagreement + (
                    0.5 * (1.0 - F.cosine_similarity(refined[idx], refined[idx + 1], dim=-1))
                ).mean()
                adjacent_pairs += 1

        feedback = torch.tanh(torch.stack(token_messages, dim=0).mean(dim=0))
        seq_len = h.shape[1]
        mean_gate_tensor = mean_scale_gate_tensor / float(max(len(token_messages), 1))
        cross_scale_disagreement = adjacent_disagreement / float(max(adjacent_pairs, 1))
        return feedback, {
            "phrase_nodes_used": self.deep_phrase_scale.node_count(seq_len),
            "span_nodes_used": self.deep_span_scale.node_count(seq_len) if self.deep_span_scale is not None else 0,
            "sequence_summary_used": 1 if self.deep_sequence_scale is not None else 0,
            "mean_upward_message_norm": upward_norm_sum / float(max(upward_steps, 1)),
            "mean_downward_message_norm": downward_norm_sum / float(max(downward_steps, 1)),
            "mean_scale_gate": mean_scale_gate / float(max(len(token_messages), 1)),
            "hierarchy_depth_used": len(levels),
        }, {
            "local_delib_hierarchy_agreement_loss": 0.5 * ((1.0 - mean_gate_tensor) + cross_scale_disagreement),
        }

    def _compute_legacy_hierarchy_feedback(
        self,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        if not self.hierarchy_levels:
            return torch.zeros_like(h), []

        level_feedbacks = []
        chunk_counts = []
        for level in self.hierarchy_levels:
            nodes, level_feedback = level(h)
            chunk_counts.append(int(nodes.shape[1]))
            level_feedbacks.append(level_feedback)
        return torch.tanh(torch.stack(level_feedbacks, dim=0).mean(dim=0)), chunk_counts

    def _combine_hierarchy_summaries(
        self,
        legacy_feedback: torch.Tensor | None,
        deep_feedback: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if legacy_feedback is not None and deep_feedback is not None:
            return torch.tanh(0.5 * (legacy_feedback + deep_feedback))
        if legacy_feedback is not None:
            return legacy_feedback
        if deep_feedback is not None:
            return deep_feedback
        return None

    def _next_chunk_prefix_cache(
        self,
        source: torch.Tensor,
        chunk_size: int,
    ) -> dict[str, torch.Tensor | int]:
        seq_len = source.shape[1]
        start = (seq_len // chunk_size) * chunk_size
        prefix = source[:, start:seq_len, :]
        return {
            "sum": prefix.sum(dim=1).detach(),
            "count": int(prefix.shape[1]),
        }

    def _build_legacy_hierarchy_step_cache(
        self,
        h: torch.Tensor,
    ) -> list[dict[str, torch.Tensor | int]]:
        caches: list[dict[str, torch.Tensor | int]] = []
        seq_len = h.shape[1]
        for level in self.hierarchy_levels:
            chunk_size = level.chunk_size
            current_start = (seq_len // chunk_size) * chunk_size
            prev_node_sum = torch.zeros(
                h.shape[0],
                h.shape[-1],
                device=h.device,
                dtype=h.dtype,
            )
            prev_node_count = 0
            for start in range(0, current_start, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk = h[:, start:end, :]
                node = torch.tanh(level.refine(chunk.mean(dim=1)))
                prev_node_sum = prev_node_sum + node
                prev_node_count += 1
            caches.append(
                {
                    "prev_node_sum": prev_node_sum.detach(),
                    "prev_node_count": prev_node_count,
                    "current_chunk_sum": h[:, current_start:seq_len, :].sum(dim=1).detach(),
                    "current_chunk_count": int(seq_len - current_start),
                }
            )
        return caches

    def _incremental_legacy_hierarchy_feedback(
        self,
        token_h: torch.Tensor,
        step_cache: list[dict[str, torch.Tensor | int]],
    ) -> torch.Tensor:
        if not self.hierarchy_levels:
            return torch.zeros_like(token_h)

        token_state = token_h[:, 0, :]
        feedbacks = []
        for level, level_cache in zip(self.hierarchy_levels, step_cache):
            current_chunk_sum = level_cache["current_chunk_sum"] + token_state
            current_chunk_count = int(level_cache["current_chunk_count"]) + 1
            current_node = torch.tanh(level.refine(current_chunk_sum / float(current_chunk_count)))
            prefix_mean = current_node
            if int(level_cache["prev_node_count"]) > 0:
                prefix_mean = (
                    level_cache["prev_node_sum"] + current_node
                ) / float(int(level_cache["prev_node_count"]) + 1)
            feedbacks.append(level.broadcast_proj(prefix_mean))
        return torch.tanh(torch.stack(feedbacks, dim=0).mean(dim=0)).unsqueeze(1)

    def _build_deep_hierarchy_step_cache(
        self,
        h: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor | int]] | None:
        if not self.use_deep_hierarchy or self.deep_phrase_scale is None:
            return None

        cache: dict[str, dict[str, torch.Tensor | int]] = {}
        source = h
        cache["phrase"] = self._next_chunk_prefix_cache(source, self.phrase_pool.chunk_size)
        phrase_summary = self.deep_phrase_scale.summarize(source)
        needs_higher = self.deep_span_scale is not None or self.deep_sequence_scale is not None
        if needs_higher:
            source = torch.tanh(self.deep_phrase_scale.up_proj(phrase_summary))
        if self.deep_span_scale is not None:
            cache["span"] = self._next_chunk_prefix_cache(source, self.deep_span_scale.chunk_size)
            span_summary = self.deep_span_scale.summarize(source)
            if self.deep_sequence_scale is not None:
                source = torch.tanh(self.deep_span_scale.up_proj(span_summary))
        if self.deep_sequence_scale is not None:
            cache["sequence"] = {
                "sum": source.sum(dim=1).detach(),
                "count": int(source.shape[1]),
            }
        return cache

    def _incremental_deep_hierarchy_feedback(
        self,
        token_h: torch.Tensor,
        step_cache: dict[str, dict[str, torch.Tensor | int]] | None,
    ) -> torch.Tensor:
        zero_feedback = torch.zeros_like(token_h)
        if not self.use_deep_hierarchy or self.deep_phrase_scale is None or step_cache is None:
            return zero_feedback

        token_state = token_h[:, 0, :]
        levels: list[tuple[CausalHierarchyScale, torch.Tensor]] = []

        phrase_cache = step_cache["phrase"]
        phrase_mean = (phrase_cache["sum"] + token_state) / float(int(phrase_cache["count"]) + 1)
        phrase_summary = torch.tanh(self.deep_phrase_scale.summary_proj(phrase_mean))
        levels.append((self.deep_phrase_scale, phrase_summary))

        propagated = None
        if self.deep_span_scale is not None or self.deep_sequence_scale is not None:
            propagated = torch.tanh(self.deep_phrase_scale.up_proj(phrase_summary))

        if self.deep_span_scale is not None and propagated is not None:
            span_cache = step_cache["span"]
            span_mean = (span_cache["sum"] + propagated) / float(int(span_cache["count"]) + 1)
            span_summary = torch.tanh(self.deep_span_scale.summary_proj(span_mean))
            levels.append((self.deep_span_scale, span_summary))
            if self.deep_sequence_scale is not None:
                propagated = torch.tanh(self.deep_span_scale.up_proj(span_summary))

        if self.deep_sequence_scale is not None and propagated is not None:
            sequence_cache = step_cache["sequence"]
            sequence_mean = (sequence_cache["sum"] + propagated) / float(int(sequence_cache["count"]) + 1)
            sequence_summary = torch.tanh(self.deep_sequence_scale.summary_proj(sequence_mean))
            levels.append((self.deep_sequence_scale, sequence_summary))

        refined = [summary for _, summary in levels]
        if self.hierarchy_bidirectional and len(refined) > 1:
            for idx in range(len(refined) - 1, 0, -1):
                child_level = levels[idx - 1][0]
                downward = torch.tanh(child_level.down_proj(refined[idx]))
                refined[idx - 1] = refined[idx - 1] + downward

        token_messages = []
        for summary, (level, _) in zip(refined, levels):
            message = torch.tanh(level.to_token_proj(summary))
            if self.hierarchy_scale_gate:
                gate = torch.sigmoid(level.gate(torch.cat([token_state, summary], dim=-1)))
            else:
                gate = torch.ones(token_state.shape[0], 1, device=token_state.device, dtype=token_state.dtype)
            token_messages.append(message * gate)
        return torch.tanh(torch.stack(token_messages, dim=0).mean(dim=0)).unsqueeze(1)

    def _current_chunk_slice(
        self,
        prefix_h: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        seq_len = prefix_h.shape[1]
        start = (seq_len // chunk_size) * chunk_size
        return prefix_h[:, start:seq_len, :]

    def _incremental_phrase_feedback(
        self,
        prefix_h: torch.Tensor,
        token_h: torch.Tensor,
    ) -> torch.Tensor:
        current_chunk = self._current_chunk_slice(prefix_h, self.phrase_pool.chunk_size)
        chunk = torch.cat([current_chunk, token_h], dim=1)
        phrase = chunk.mean(dim=1)
        return self.phrase_pool.proj(phrase).unsqueeze(1)

    def _incremental_phrase_consensus_feedback(
        self,
        prefix_h: torch.Tensor,
        token_h: torch.Tensor,
    ) -> torch.Tensor:
        current_chunk = self._current_chunk_slice(prefix_h, self.phrase_consensus.chunk_size)
        chunk = torch.cat([current_chunk, token_h], dim=1)
        token_proposals = self.phrase_consensus.proposal_proj(chunk)
        consensus = self.phrase_consensus.consensus_proj(token_proposals.mean(dim=1))
        accept_gate = torch.sigmoid(self.phrase_consensus.agreement_gate(token_h))
        return accept_gate * consensus.unsqueeze(1)

    def _build_scratch_step_cache(
        self,
        h: torch.Tensor,
        head_states: dict[str, torch.Tensor],
        branch_summary: torch.Tensor | None,
        hierarchy_summary: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor | None]:
        if self.scratch_slots < 1:
            return None, None

        scratch_prefix_state = self._allocate_scratch_prefix_state(
            bsz=h.shape[0],
            seq_len=h.shape[1],
            device=h.device,
            dtype=h.dtype,
        )
        scratch_feedback, next_prefix, _, _ = self._compute_scratch_feedback(
            h,
            head_states,
            scratch_prefix_state=scratch_prefix_state,
            branch_summary=branch_summary,
            hierarchy_summary=hierarchy_summary,
        )
        return {"slots": next_prefix[:, -1, :, :].detach()}, scratch_feedback

    def _incremental_scratch_feedback(
        self,
        token_h: torch.Tensor,
        head_states: dict[str, torch.Tensor],
        step_cache: dict[str, torch.Tensor] | None,
        branch_summary: torch.Tensor | None = None,
        hierarchy_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.scratch_slots < 1 or step_cache is None:
            return torch.zeros_like(token_h)

        scratch = step_cache["slots"].clone()
        token_state = token_h[:, 0, :]
        query = self.scratch_query(token_state)
        slot_logits = torch.einsum("bd,bsd->bs", query, scratch) / math.sqrt(self.scratch_dim)
        slot_attn = torch.softmax(slot_logits * torch.clamp(self.scratch_read_temp, min=0.1), dim=-1)
        read_gate = head_states["uncertainty"][:, 0, :]
        read_summary = torch.einsum("bs,bsd->bd", slot_attn, scratch)
        feedback = self.scratch_to_state(read_summary * read_gate).unsqueeze(1)

        write_gate = (
            head_states["salience"][:, 0, :] * head_states["uncertainty"][:, 0, :]
        ).squeeze(-1)
        write_value = self.scratch_write_value(token_state) + self.scratch_read_mix(read_summary)
        if self.scratch_use_branch_inputs and branch_summary is not None:
            branch_state = branch_summary[:, 0, :]
            branch_gate = torch.sigmoid(self.scratch_branch_gate(torch.cat([token_state, branch_state], dim=-1)))
            write_value = write_value + branch_gate * self.scratch_branch_write(branch_state)
        if self.scratch_use_hierarchy_inputs and hierarchy_summary is not None:
            hierarchy_state = hierarchy_summary[:, 0, :]
            hierarchy_gate = torch.sigmoid(self.scratch_hierarchy_gate(torch.cat([token_state, hierarchy_state], dim=-1)))
            write_value = write_value + hierarchy_gate * self.scratch_hierarchy_write(hierarchy_state)
        scratch = scratch + (slot_attn * write_gate.unsqueeze(-1)).unsqueeze(-1) * write_value.unsqueeze(1)
        if self.scratch_refine is not None:
            for _ in range(self.scratch_refine_steps):
                scratch_summary = scratch.mean(dim=1, keepdim=True).expand_as(scratch)
                refine_delta = self.scratch_refine(torch.cat([scratch, scratch_summary], dim=-1))
                scratch = scratch + torch.tanh(refine_delta)
        return feedback

    def _build_global_anchor_step_cache(
        self,
        h: torch.Tensor,
        head_states: dict[str, torch.Tensor],
        hierarchy_summary: torch.Tensor | None,
        scratch_summary: torch.Tensor | None,
        thought_summary: torch.Tensor | None,
    ) -> dict[str, torch.Tensor] | None:
        if self.global_anchor_count < 1:
            return None

        anchor_prefix_state = self._allocate_global_anchor_prefix_state(
            bsz=h.shape[0],
            seq_len=h.shape[1],
            device=h.device,
            dtype=h.dtype,
        )
        _, next_prefix, _, _ = self._compute_global_anchor_feedback(
            h,
            head_states,
            global_anchor_prefix_state=anchor_prefix_state,
            hierarchy_summary=hierarchy_summary,
            scratch_summary=scratch_summary,
            thought_summary=thought_summary,
        )
        return {
            "anchors": next_prefix[:, -1, :, :].detach(),
            "prefix_sum": h.sum(dim=1).detach(),
            "prefix_count": torch.full(
                (h.shape[0], 1),
                float(h.shape[1]),
                device=h.device,
                dtype=h.dtype,
            ),
        }

    def _incremental_global_anchor_feedback(
        self,
        token_h: torch.Tensor,
        head_states: dict[str, torch.Tensor],
        step_cache: dict[str, torch.Tensor] | None,
        hierarchy_summary: torch.Tensor | None = None,
        scratch_summary: torch.Tensor | None = None,
        thought_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.global_anchor_count < 1 or step_cache is None:
            return torch.zeros_like(token_h)

        anchors = step_cache["anchors"]
        token_state = token_h[:, 0, :]
        anchor_keys = self.global_anchor_key(anchors)
        anchor_values = self.global_anchor_value(anchors)
        query = self.global_anchor_query(token_state)
        scale = math.sqrt(self.global_anchor_dim)
        read_logits = torch.einsum("bd,bad->ba", query, anchor_keys) / scale
        read_attn = torch.softmax(read_logits / self.global_anchor_temp, dim=-1)
        read_gate = head_states["uncertainty"][:, 0, :]
        read_summary = torch.einsum("ba,bad->bd", read_attn, anchor_values)
        feedback = self.global_anchor_to_state(read_summary * read_gate).unsqueeze(1)

        if self.global_anchor_update:
            prefix_sum = step_cache["prefix_sum"] + token_state
            prefix_mean = prefix_sum / (step_cache["prefix_count"] + 1.0)
            write_query = self.global_anchor_write_query(prefix_mean)
            write_logits = torch.einsum("bd,bad->ba", write_query, anchor_keys) / scale
            write_attn = torch.softmax(write_logits / self.global_anchor_temp, dim=-1)
            write_value = self.global_anchor_token_write(token_state) + self.global_anchor_prefix_write(prefix_mean)
            if self.global_anchor_use_hierarchy and hierarchy_summary is not None:
                hierarchy_state = hierarchy_summary[:, 0, :]
                hierarchy_gate = torch.sigmoid(
                    self.global_anchor_hierarchy_gate(torch.cat([token_state, hierarchy_state], dim=-1))
                )
                write_value = write_value + hierarchy_gate * self.global_anchor_hierarchy_write(hierarchy_state)
            if self.global_anchor_use_scratch and scratch_summary is not None:
                scratch_state = scratch_summary[:, 0, :]
                scratch_gate = torch.sigmoid(
                    self.global_anchor_scratch_gate(torch.cat([token_state, scratch_state], dim=-1))
                )
                write_value = write_value + scratch_gate * self.global_anchor_scratch_write(scratch_state)
            if self.global_anchor_use_thought and thought_summary is not None:
                thought_state = thought_summary[:, 0, :]
                thought_gate = torch.sigmoid(
                    self.global_anchor_thought_gate(torch.cat([token_state, thought_state], dim=-1))
                )
                write_value = write_value + thought_gate * self.global_anchor_thought_write(thought_state)
            _ = anchors + (write_attn * (head_states["salience"][:, 0, :] * head_states["uncertainty"][:, 0, :]).squeeze(-1).unsqueeze(-1)).unsqueeze(-1) * torch.tanh(write_value).unsqueeze(1)
        return feedback

    def _build_thought_step_cache(
        self,
        h: torch.Tensor,
        *,
        branch_summary: torch.Tensor | None = None,
        hierarchy_summary: torch.Tensor | None = None,
        scratch_summary: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | int | bool] | None:
        if not self.use_thought_graph:
            return None

        bsz, seq_len, _ = h.shape
        chunk_size = self.thought_token_chunk_size
        chunk_ranges = [
            (start, min(start + chunk_size, seq_len))
            for start in range(0, seq_len, chunk_size)
        ]
        if self.thought_node_budget > 0 and len(chunk_ranges) > self.thought_node_budget:
            chunk_ranges = chunk_ranges[-self.thought_node_budget:]

        current_chunk_start = (seq_len // chunk_size) * chunk_size
        current_chunk_count = seq_len - current_chunk_start
        prev_ranges = [chunk for chunk in chunk_ranges if chunk[0] < current_chunk_start]
        will_slide_budget = (
            self.thought_node_budget > 0
            and len(chunk_ranges) >= self.thought_node_budget
            and current_chunk_count == 0
            and seq_len > 0
        )

        def _node_base_for_chunk(start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
            chunk = h[:, start:end, :]
            token_proj = self.thought_node_builder.token_proj(chunk).sum(dim=1)
            count = float(max(end - start, 1))
            pooled = token_proj / count
            if self.thought_use_branch_inputs and branch_summary is not None:
                pooled = pooled + self.thought_node_builder.branch_proj(branch_summary[:, start:end, :]).sum(dim=1) / count
            if self.thought_use_hierarchy_inputs and hierarchy_summary is not None:
                pooled = pooled + self.thought_node_builder.hierarchy_proj(hierarchy_summary[:, start:end, :]).sum(dim=1) / count
            if self.thought_use_scratch_inputs and scratch_summary is not None:
                pooled = pooled + self.thought_node_builder.scratch_proj(scratch_summary[:, start:end, :]).sum(dim=1) / count

            values = self.token_to_thought.write_value(chunk)
            gates = torch.sigmoid(self.token_to_thought.write_gate(chunk))
            node_write = (gates * values).sum(dim=1) / gates.sum(dim=1).clamp_min(1e-8)
            return torch.tanh(pooled), node_write

        prev_node_limits: list[int] = []
        prev_bases: list[torch.Tensor] = []
        prev_writes: list[torch.Tensor] = []
        for start, end in prev_ranges:
            node_base, node_write = _node_base_for_chunk(start, end)
            prev_bases.append(node_base)
            prev_writes.append(node_write)
            prev_node_limits.append(end - 1)

        if prev_bases:
            prev_nodes = torch.stack(prev_bases, dim=1)
            prev_node_writes = torch.stack(prev_writes, dim=1)
            prev_nodes_by_step = [prev_nodes.detach()]
            current_prev = prev_nodes
            for _ in range(self.thought_graph_steps):
                neighbor_summary, _, _ = self.thought_graph.summarize(current_prev)
                delta = self.thought_message_passing.update(
                    torch.cat([current_prev, neighbor_summary, prev_node_writes], dim=-1)
                )
                current_prev = current_prev + delta
                prev_nodes_by_step.append(current_prev.detach())
        else:
            prev_nodes_by_step = [
                torch.zeros(bsz, 0, self.thought_node_dim, device=h.device, dtype=h.dtype)
                for _ in range(self.thought_graph_steps + 1)
            ]

        current_proj_sum = torch.zeros(bsz, self.thought_node_dim, device=h.device, dtype=h.dtype)
        current_write_num = torch.zeros(bsz, self.thought_node_dim, device=h.device, dtype=h.dtype)
        current_write_den = torch.zeros(bsz, 1, device=h.device, dtype=h.dtype)
        if current_chunk_count > 0:
            start, end = current_chunk_start, seq_len
            chunk = h[:, start:end, :]
            current_proj_sum = current_proj_sum + self.thought_node_builder.token_proj(chunk).sum(dim=1)
            if self.thought_use_branch_inputs and branch_summary is not None:
                current_proj_sum = current_proj_sum + self.thought_node_builder.branch_proj(branch_summary[:, start:end, :]).sum(dim=1)
            if self.thought_use_hierarchy_inputs and hierarchy_summary is not None:
                current_proj_sum = current_proj_sum + self.thought_node_builder.hierarchy_proj(hierarchy_summary[:, start:end, :]).sum(dim=1)
            if self.thought_use_scratch_inputs and scratch_summary is not None:
                current_proj_sum = current_proj_sum + self.thought_node_builder.scratch_proj(scratch_summary[:, start:end, :]).sum(dim=1)
            chunk_values = self.token_to_thought.write_value(chunk)
            chunk_gates = torch.sigmoid(self.token_to_thought.write_gate(chunk))
            current_write_num = current_write_num + (chunk_gates * chunk_values).sum(dim=1)
            current_write_den = current_write_den + chunk_gates.sum(dim=1)

        return {
            "prev_nodes_by_step": prev_nodes_by_step,
            "prev_node_limits": torch.tensor(prev_node_limits, dtype=torch.long, device=h.device),
            "current_proj_sum": current_proj_sum.detach(),
            "current_count": current_chunk_count,
            "current_write_num": current_write_num.detach(),
            "current_write_den": current_write_den.detach(),
            "will_slide_budget": will_slide_budget,
        }

    def _incremental_thought_feedback(
        self,
        token_h: torch.Tensor,
        step_cache: dict[str, torch.Tensor | list[torch.Tensor] | int | bool] | None,
        *,
        prefix_len: int,
        branch_summary: torch.Tensor | None = None,
        hierarchy_summary: torch.Tensor | None = None,
        scratch_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_thought_graph:
            return torch.zeros_like(token_h)
        if step_cache is None or bool(step_cache["will_slide_budget"]):
            raise IncrementalCacheFallback("thought node cache window would slide")

        token_state = token_h[:, 0, :]
        proj_sum = step_cache["current_proj_sum"] + self.thought_node_builder.token_proj(token_state)
        count = int(step_cache["current_count"]) + 1
        if self.thought_use_branch_inputs and branch_summary is not None:
            proj_sum = proj_sum + self.thought_node_builder.branch_proj(branch_summary[:, 0, :])
        if self.thought_use_hierarchy_inputs and hierarchy_summary is not None:
            proj_sum = proj_sum + self.thought_node_builder.hierarchy_proj(hierarchy_summary[:, 0, :])
        if self.thought_use_scratch_inputs and scratch_summary is not None:
            proj_sum = proj_sum + self.thought_node_builder.scratch_proj(scratch_summary[:, 0, :])
        current_node = torch.tanh(proj_sum / float(count))

        write_num = step_cache["current_write_num"] + (
            torch.sigmoid(self.token_to_thought.write_gate(token_state)) * self.token_to_thought.write_value(token_state)
        )
        write_den = step_cache["current_write_den"] + torch.sigmoid(self.token_to_thought.write_gate(token_state))
        current_node_write = write_num / write_den.clamp_min(1e-8)

        prev_nodes_by_step = step_cache["prev_nodes_by_step"]
        for graph_step in range(self.thought_graph_steps):
            prev_nodes = prev_nodes_by_step[graph_step]
            all_nodes = torch.cat([prev_nodes, current_node.unsqueeze(1)], dim=1)
            neighbor_summary, _, _ = self.thought_graph.summarize(all_nodes)
            current_neighbor = neighbor_summary[:, -1, :]
            delta = self.thought_message_passing.update(
                torch.cat([current_node, current_neighbor, current_node_write], dim=-1)
            )
            current_node = current_node + delta

        final_nodes = torch.cat([prev_nodes_by_step[-1], current_node.unsqueeze(1)], dim=1)
        node_limits = torch.cat(
            [
                step_cache["prev_node_limits"],
                torch.tensor([prefix_len], dtype=torch.long, device=token_h.device),
            ],
            dim=0,
        )
        read_summary, consensus_summary, _, _ = self.token_to_thought.read(
            token_h,
            nodes=final_nodes,
            node_limits=node_limits,
        )
        return self.thought_consensus_reducer(read_summary, consensus_summary)

    def build_decode_cache(
        self,
        stage_states: list[torch.Tensor],
    ) -> dict[str, object]:
        step_caches = []
        for h in stage_states[:-1]:
            head_states = self.state_head(h)
            legacy_feedback, _ = self._compute_legacy_hierarchy_feedback(h)
            deep_feedback = self._compute_deep_hierarchy_feedback(h)[0] if self.use_deep_hierarchy else None
            hierarchy_summary = self._combine_hierarchy_summaries(
                legacy_feedback if self.hierarchy_levels else None,
                deep_feedback if self.use_deep_hierarchy else None,
            )
            branch_summary = None
            if self.branch_factor > 0 and (self.scratch_use_branch_inputs or self.thought_use_branch_inputs):
                branch_context = self._compute_branch_context(h)
                branch_summary = (
                    branch_context["consensus_summary"] if self.branch_consensus else branch_context["branch_summary"]
                )

            scratch_cache, scratch_feedback = self._build_scratch_step_cache(
                h,
                head_states,
                branch_summary,
                hierarchy_summary,
            )
            thought_cache = self._build_thought_step_cache(
                h,
                branch_summary=branch_summary,
                hierarchy_summary=hierarchy_summary,
                scratch_summary=scratch_feedback,
            )
            thought_feedback = None
            if self.use_thought_graph:
                thought_feedback = self._compute_thought_feedback(
                    h,
                    branch_summary=branch_summary,
                    hierarchy_summary=hierarchy_summary,
                    scratch_summary=scratch_feedback,
                )[0]
            anchor_cache = self._build_global_anchor_step_cache(
                h,
                head_states,
                hierarchy_summary if self.global_anchor_use_hierarchy else None,
                scratch_feedback if self.global_anchor_use_scratch else None,
                thought_feedback if self.global_anchor_use_thought else None,
            )
            step_caches.append(
                {
                    "legacy_hierarchy": self._build_legacy_hierarchy_step_cache(h),
                    "deep_hierarchy": self._build_deep_hierarchy_step_cache(h),
                    "scratch": scratch_cache,
                    "thought": thought_cache,
                    "anchors": anchor_cache,
                }
            )
        return {
            "stage_states": [state.detach() for state in stage_states],
            "step_caches": step_caches,
            "token_count": int(stage_states[-1].shape[1]) if stage_states else 0,
        }

    def deliberate_state_cached(
        self,
        h_new: torch.Tensor,
        cache: dict[str, object] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | int], dict[str, object]]:
        if cache is None or int(cache.get("token_count", 0)) == 0:
            h_full, stats, stage_states = self.deliberate_state(h_new, capture_stage_states=True)
            return h_full, stats, self.build_decode_cache(stage_states)

        stage_states: list[torch.Tensor] = cache["stage_states"]  # type: ignore[assignment]
        if h_new.shape[1] != 1 or self.use_thought_graph:
            h_full = torch.cat([stage_states[0], h_new], dim=1)
            h_full, stats, new_stage_states = self.deliberate_state(h_full, capture_stage_states=True)
            return h_full[:, -h_new.shape[1]:, :], stats, self.build_decode_cache(new_stage_states)

        prefix_len = int(cache["token_count"])
        token_h = h_new
        updated_stage_states = [torch.cat([stage_states[0], token_h.detach()], dim=1)]
        executed_steps = 0.0
        active_mask = torch.ones(token_h.shape[0], 1, 1, device=token_h.device, dtype=torch.bool)
        halt_threshold = torch.sigmoid(self.halt_threshold_logit)

        try:
            for step_idx in range(self.micro_steps):
                prefix_state = stage_states[step_idx]
                step_cache = cache["step_caches"][step_idx]  # type: ignore[index]
                head_states = self.state_head(token_h)

                tail_width = self.mixer.kernel_size - 1
                tail_context = prefix_state[:, -tail_width:, :] if tail_width > 0 else prefix_state[:, :0, :]
                mixed = self.mixer(torch.cat([tail_context, token_h], dim=1))[:, -1:, :]
                phrase_broadcast = self._incremental_phrase_feedback(prefix_state, token_h)
                update_parts = [token_h, mixed, phrase_broadcast]

                if self.use_neighbor_graph:
                    semantic_summary, _, flocking_feedback, _ = self.neighbor_graph_mixer.summarize(torch.cat([prefix_state, token_h], dim=1))
                    update_parts.append(semantic_summary[:, -1:, :])
                    if self.use_flocking:
                        update_parts.append(flocking_feedback[:, -1:, :])
                elif self.semantic_topk > 0:
                    semantic_summary, _, _, _ = self._semantic_neighbor_summary(torch.cat([prefix_state, token_h], dim=1))
                    update_parts.append(semantic_summary[:, -1:, :])

                if self.use_phrase_consensus:
                    update_parts.append(self._incremental_phrase_consensus_feedback(prefix_state, token_h))

                legacy_hierarchy_feedback = None
                if self.hierarchy_levels:
                    legacy_hierarchy_feedback = self._incremental_legacy_hierarchy_feedback(
                        token_h,
                        step_cache["legacy_hierarchy"],
                    )
                    update_parts.append(legacy_hierarchy_feedback)

                deep_hierarchy_feedback = None
                if self.use_deep_hierarchy:
                    deep_hierarchy_feedback = self._incremental_deep_hierarchy_feedback(
                        token_h,
                        step_cache["deep_hierarchy"],
                    )
                    update_parts.append(deep_hierarchy_feedback)

                hierarchy_summary = self._combine_hierarchy_summaries(
                    legacy_hierarchy_feedback,
                    deep_hierarchy_feedback,
                )
                pre_branch_summary = None
                if self.branch_factor > 0 and (self.scratch_use_branch_inputs or self.thought_use_branch_inputs):
                    pre_branch_context = self._compute_branch_context(token_h)
                    pre_branch_summary = (
                        pre_branch_context["consensus_summary"] if self.branch_consensus else pre_branch_context["branch_summary"]
                    )

                scratch_feedback = None
                if self.scratch_slots > 0:
                    scratch_feedback = self._incremental_scratch_feedback(
                        token_h,
                        head_states,
                        step_cache["scratch"],
                        branch_summary=pre_branch_summary,
                        hierarchy_summary=hierarchy_summary,
                    )
                    update_parts.append(scratch_feedback)

                thought_feedback = None
                if self.use_thought_graph:
                    thought_feedback = self._incremental_thought_feedback(
                        token_h,
                        step_cache["thought"],
                        prefix_len=prefix_len,
                        branch_summary=pre_branch_summary,
                        hierarchy_summary=hierarchy_summary,
                        scratch_summary=scratch_feedback,
                    )
                    update_parts.append(thought_feedback)

                if self.global_anchor_count > 0:
                    anchor_feedback = self._incremental_global_anchor_feedback(
                        token_h,
                        head_states,
                        step_cache["anchors"],
                        hierarchy_summary=hierarchy_summary if self.global_anchor_use_hierarchy else None,
                        scratch_summary=scratch_feedback if self.global_anchor_use_scratch else None,
                        thought_summary=thought_feedback if self.global_anchor_use_thought else None,
                    )
                    update_parts.append(anchor_feedback)

                delta = self.update(torch.cat(update_parts, dim=-1))
                if self.use_token_gate or self.adaptive_halt:
                    head_states = self.state_head(token_h)
                if self.use_token_gate:
                    delta = delta * head_states["halt_gate"]

                if self.adaptive_halt:
                    executed_steps += float(active_mask.to(torch.float32).mean().item())
                    token_h = token_h + delta * active_mask.to(token_h.dtype)
                    active_mask = active_mask & (~(head_states["halt_gate"] >= halt_threshold))
                else:
                    token_h = token_h + delta
                    executed_steps += 1.0

                if self.branch_factor > 0 and (step_idx % self.branch_every == 0):
                    branch_context = self._compute_branch_context(token_h)
                    branch_summary = branch_context["branch_summary"]
                    if self.branch_consensus or self.branch_verifier:
                        token_h, _, _ = self.branch_consensus_merge(
                            token_h,
                            branch_summary,
                            branch_context["consensus_summary"] if self.branch_consensus else torch.zeros_like(branch_summary),
                            branch_context["consensus_mask"].unsqueeze(-1),
                        )
                    else:
                        token_h, _ = self.branch_merge(token_h, branch_summary)

                updated_stage_states.append(torch.cat([stage_states[step_idx + 1], token_h.detach()], dim=1))
        except IncrementalCacheFallback:
            h_full = torch.cat([stage_states[0], h_new], dim=1)
            h_full, stats, new_stage_states = self.deliberate_state(h_full, capture_stage_states=True)
            return h_full[:, -1:, :], stats, self.build_decode_cache(new_stage_states)

        final_head_states = self.state_head(token_h)
        self.last_aux_losses = {
            "local_delib_halt_sparsity_loss": final_head_states["halt_gate"].mean(),
            "local_delib_branch_diversity_loss": token_h.new_zeros(()),
            "local_delib_branch_entropy_loss": token_h.new_zeros(()),
            "local_delib_consensus_agreement_loss": token_h.new_zeros(()),
            "local_delib_scratch_utilization_loss": token_h.new_ones(()),
            "local_delib_flocking_stability_loss": token_h.new_zeros(()),
            "local_delib_thought_edge_stability_loss": token_h.new_zeros(()),
            "local_delib_thought_node_utilization_loss": token_h.new_zeros(()),
            "local_delib_hierarchy_agreement_loss": token_h.new_zeros(()),
            "local_delib_branch_usefulness_loss": token_h.new_zeros(()),
            "local_delib_anchor_usage_loss": token_h.new_zeros(()),
        }
        stats: dict[str, torch.Tensor | float | int] = {
            "mean_salience": float(final_head_states["salience"].mean().item()),
            "mean_uncertainty": float(final_head_states["uncertainty"].mean().item()),
            "mean_halt": float(final_head_states["halt_gate"].mean().item()),
            "mean_final_halt": float(final_head_states["halt_gate"].mean().item()),
            "executed_steps": self.micro_steps,
            "mean_executed_steps_per_token": executed_steps,
            "max_executed_steps_any_token": int(math.ceil(executed_steps)),
            "fraction_halted_early": float(executed_steps < float(self.micro_steps)),
        }
        return token_h, stats, self.build_decode_cache(updated_stage_states)

    def deliberate_state(
        self,
        h: torch.Tensor,
        capture_stage_states: bool = False,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor | float | int],
    ] | tuple[
        torch.Tensor,
        dict[str, torch.Tensor | float | int],
        list[torch.Tensor],
    ]:
        head_states: dict[str, torch.Tensor] = self.state_head(h)
        self.last_aux_losses = None
        stage_states = [h.detach()] if capture_stage_states else None

        semantic_topk_used = 0
        mean_neighbor_count = 0.0
        mean_sequence_neighbor_weight = 0.0
        mean_semantic_neighbor_weight = 0.0
        mean_phrase_neighbor_weight = 0.0
        mean_agreement_score = 0.0
        mean_alignment_norm_accum = 0.0
        mean_cohesion_norm_accum = 0.0
        mean_separation_norm_accum = 0.0
        mean_flocking_total_norm_accum = 0.0
        flocking_neighbor_count_accum = 0.0
        fraction_flocking_tokens_active_accum = 0.0
        flocking_steps = 0
        mean_branch_score_accum = 0.0
        max_branch_score = 0.0
        mean_merge_weight_accum = 0.0
        fraction_tokens_branched_accum = 0.0
        branch_steps = 0
        branch_factor_used = 0
        mean_branch_disagreement_accum = 0.0
        mean_branch_consensus_weight_accum = 0.0
        mean_branch_verifier_score_accum = 0.0
        mean_branch_entropy_accum = 0.0
        branch_consensus_used_accum = 0.0
        hierarchy_feedback_norm_accum = 0.0
        hierarchy_feedback_steps = 0
        hierarchy_level_chunk_counts = [0 for _ in self.hierarchy_levels]
        phrase_nodes_used = 0
        span_nodes_used = 0
        sequence_summary_used = 0
        mean_upward_message_norm_accum = 0.0
        mean_downward_message_norm_accum = 0.0
        mean_scale_gate_accum = 0.0
        hierarchy_depth_used = 0
        deep_hierarchy_steps = 0
        scratch_slots_used = 0
        mean_scratch_read_weight = 0.0
        mean_scratch_write_weight = 0.0
        mean_scratch_norm = 0.0
        mean_scratch_refine_norm = 0.0
        mean_scratch_summary_norm = 0.0
        mean_branch_to_scratch_weight = 0.0
        mean_hierarchy_to_scratch_weight = 0.0
        scratch_reset_ok = 1.0
        scratch_summary_vector = None
        thought_nodes_used = 0
        mean_thought_degree_accum = 0.0
        mean_token_to_thought_weight_accum = 0.0
        mean_thought_to_token_weight_accum = 0.0
        mean_thought_update_norm_accum = 0.0
        thought_graph_steps_used = 0
        thought_steps = 0
        global_anchors_used = 0
        mean_anchor_read_weight = 0.0
        mean_anchor_write_weight = 0.0
        mean_anchor_norm = 0.0
        anchor_steps = 0
        consensus_disagreement_accum = h.new_zeros(())
        branch_entropy_accum = h.new_zeros(())
        branch_diversity_accum = h.new_zeros(())
        scratch_utilization_accum = h.new_zeros(())
        flocking_stability_accum = h.new_zeros(())
        hierarchy_agreement_accum = h.new_zeros(())
        thought_edge_stability_accum = h.new_zeros(())
        thought_node_utilization_accum = h.new_zeros(())
        branch_usefulness_accum = h.new_zeros(())
        anchor_usage_accum = h.new_zeros(())
        executed_steps_per_token = torch.zeros(h.shape[0], h.shape[1], 1, device=h.device, dtype=torch.float32)
        active_mask = torch.ones(h.shape[0], h.shape[1], 1, device=h.device, dtype=torch.bool)
        halt_threshold = torch.sigmoid(self.halt_threshold_logit)
        scratch_prefix_state = None
        if self.scratch_slots > 0:
            scratch_prefix_state = self._allocate_scratch_prefix_state(
                bsz=h.shape[0],
                seq_len=h.shape[1],
                device=h.device,
                dtype=h.dtype,
            )
        global_anchor_prefix_state = None
        if self.global_anchor_count > 0:
            global_anchor_prefix_state = self._allocate_global_anchor_prefix_state(
                bsz=h.shape[0],
                seq_len=h.shape[1],
                device=h.device,
                dtype=h.dtype,
            )

        for step_idx in range(self.micro_steps):
            mixed = self.mixer(h)
            _, phrase_broadcast = self.phrase_pool(h)
            hierarchy_feedback = torch.zeros_like(h)
            semantic_summary = None
            flocking_feedback = None
            if self.hierarchy_levels:
                level_feedbacks = []
                for level_idx, level in enumerate(self.hierarchy_levels):
                    nodes, level_feedback = level(h)
                    hierarchy_level_chunk_counts[level_idx] = nodes.shape[1]
                    level_feedbacks.append(level_feedback)
                hierarchy_feedback = torch.tanh(torch.stack(level_feedbacks, dim=0).mean(dim=0))
                hierarchy_feedback_norm_accum += float(hierarchy_feedback.norm(dim=-1).mean().item())
                hierarchy_feedback_steps += 1
            if self.use_phrase_consensus:
                _, consensus_feedback, step_agreement_score, _ = self.phrase_consensus(h)
                mean_agreement_score += float(step_agreement_score.item())
                consensus_disagreement_accum = consensus_disagreement_accum + (1.0 - step_agreement_score) * 0.5
            else:
                consensus_feedback = torch.zeros_like(h)
            update_parts = [h, mixed, phrase_broadcast]
            if self.use_neighbor_graph:
                semantic_summary, graph_stats, flocking_feedback, graph_aux = self.neighbor_graph_mixer.summarize(h)
                semantic_topk_used = int(graph_stats["semantic_topk_used"])
                mean_neighbor_count = float(graph_stats["mean_neighbor_count"])
                mean_sequence_neighbor_weight = float(graph_stats["mean_sequence_neighbor_weight"])
                mean_semantic_neighbor_weight = float(graph_stats["mean_semantic_neighbor_weight"])
                mean_phrase_neighbor_weight = float(graph_stats["mean_phrase_neighbor_weight"])
                update_parts.append(semantic_summary)
                if self.use_flocking and flocking_feedback is not None:
                    update_parts.append(flocking_feedback)
                    mean_alignment_norm_accum += float(graph_stats["mean_alignment_norm"])
                    mean_cohesion_norm_accum += float(graph_stats["mean_cohesion_norm"])
                    mean_separation_norm_accum += float(graph_stats["mean_separation_norm"])
                    mean_flocking_total_norm_accum += float(graph_stats["mean_flocking_total_norm"])
                    flocking_neighbor_count_accum += float(graph_stats["flocking_neighbor_count"])
                    fraction_flocking_tokens_active_accum += float(graph_stats["fraction_flocking_tokens_active"])
                    flocking_stability_accum = flocking_stability_accum + graph_aux["local_delib_flocking_stability_loss"]
                    flocking_steps += 1
            elif self.semantic_topk > 0:
                semantic_summary, _, semantic_weights, semantic_topk_used = self._semantic_neighbor_summary(h)
                used_weight_mask = (semantic_weights > 0).to(semantic_weights.dtype)
                used_weight_count = used_weight_mask.sum().clamp_min(1.0)
                mean_semantic_neighbor_weight = float((semantic_weights * used_weight_mask).sum().item() / used_weight_count.item())
                mean_neighbor_count = float(semantic_topk_used)
                update_parts.append(semantic_summary)
            if self.use_phrase_consensus:
                update_parts.append(consensus_feedback)
            if self.hierarchy_levels:
                update_parts.append(hierarchy_feedback)
            deep_hierarchy_feedback, deep_hierarchy_stats, deep_hierarchy_aux = self._compute_deep_hierarchy_feedback(h)
            if self.use_deep_hierarchy:
                update_parts.append(deep_hierarchy_feedback)
                phrase_nodes_used = max(phrase_nodes_used, int(deep_hierarchy_stats["phrase_nodes_used"]))
                span_nodes_used = max(span_nodes_used, int(deep_hierarchy_stats["span_nodes_used"]))
                sequence_summary_used = max(sequence_summary_used, int(deep_hierarchy_stats["sequence_summary_used"]))
                mean_upward_message_norm_accum += float(deep_hierarchy_stats["mean_upward_message_norm"])
                mean_downward_message_norm_accum += float(deep_hierarchy_stats["mean_downward_message_norm"])
                mean_scale_gate_accum += float(deep_hierarchy_stats["mean_scale_gate"])
                hierarchy_depth_used = max(hierarchy_depth_used, int(deep_hierarchy_stats["hierarchy_depth_used"]))
                hierarchy_agreement_accum = hierarchy_agreement_accum + deep_hierarchy_aux["local_delib_hierarchy_agreement_loss"]
                deep_hierarchy_steps += 1
            update_inputs = torch.cat(update_parts, dim=-1)
            scratch_hierarchy_summary = (
                torch.tanh(0.5 * (hierarchy_feedback + deep_hierarchy_feedback))
                if self.hierarchy_levels and self.use_deep_hierarchy
                else hierarchy_feedback
                if self.hierarchy_levels
                else deep_hierarchy_feedback
                if self.use_deep_hierarchy
                else None
            )

            if self.scratch_slots > 0 and not (self.use_token_gate or self.adaptive_halt):
                head_states = self.state_head(h)

            if self.scratch_slots > 0:
                scratch_branch_summary = None
                if self.branch_factor > 0 and self.scratch_use_branch_inputs:
                    scratch_branch_context = self._compute_branch_context(h)
                    scratch_branch_summary = (
                        scratch_branch_context["consensus_summary"]
                        if self.branch_consensus
                        else scratch_branch_context["branch_summary"]
                    )
                (
                    scratch_feedback,
                    scratch_prefix_state,
                    scratch_stats,
                    step_scratch_summary_vector,
                ) = self._compute_scratch_feedback(
                    h,
                    head_states,
                    scratch_prefix_state=scratch_prefix_state,
                    branch_summary=scratch_branch_summary,
                    hierarchy_summary=scratch_hierarchy_summary,
                )
                update_inputs = torch.cat([update_inputs, scratch_feedback], dim=-1)
                mean_scratch_read_weight += float(scratch_stats["mean_scratch_read_weight"])
                mean_scratch_write_weight += float(scratch_stats["mean_scratch_write_weight"])
                mean_scratch_norm += float(scratch_stats["mean_scratch_norm"])
                mean_scratch_refine_norm += float(scratch_stats["mean_scratch_refine_norm"])
                mean_scratch_summary_norm += float(scratch_stats["mean_scratch_summary_norm"])
                mean_branch_to_scratch_weight += float(scratch_stats["mean_branch_to_scratch_weight"])
                mean_hierarchy_to_scratch_weight += float(scratch_stats["mean_hierarchy_to_scratch_weight"])
                scratch_reset_ok = min(scratch_reset_ok, float(scratch_stats["scratch_reset_ok"]))
                scratch_slots_used = max(scratch_slots_used, int(scratch_stats["scratch_slots_used"]))
                if step_scratch_summary_vector is not None:
                    scratch_summary_vector = step_scratch_summary_vector
                scratch_utilization_accum = scratch_utilization_accum + 0.5 * (
                    head_states["uncertainty"].mean() + (head_states["salience"] * head_states["uncertainty"]).mean()
                )
            else:
                scratch_feedback = None

            thought_branch_summary = None
            if self.use_thought_graph and self.branch_factor > 0 and self.thought_use_branch_inputs:
                thought_branch_context = self._compute_branch_context(h)
                thought_branch_summary = thought_branch_context["consensus_summary"] if self.branch_consensus else thought_branch_context["branch_summary"]

            thought_feedback, thought_stats, thought_aux = self._compute_thought_feedback(
                h,
                branch_summary=thought_branch_summary,
                hierarchy_summary=scratch_hierarchy_summary,
                scratch_summary=scratch_feedback,
            )
            if self.use_thought_graph:
                update_inputs = torch.cat([update_inputs, thought_feedback], dim=-1)
                thought_nodes_used = max(thought_nodes_used, int(thought_stats["thought_nodes_used"]))
                mean_thought_degree_accum += float(thought_stats["mean_thought_degree"])
                mean_token_to_thought_weight_accum += float(thought_stats["mean_token_to_thought_weight"])
                mean_thought_to_token_weight_accum += float(thought_stats["mean_thought_to_token_weight"])
                mean_thought_update_norm_accum += float(thought_stats["mean_thought_update_norm"])
                thought_graph_steps_used = max(thought_graph_steps_used, int(thought_stats["thought_graph_steps_used"]))
                thought_edge_stability_accum = thought_edge_stability_accum + thought_aux["local_delib_thought_edge_stability_loss"]
                thought_node_utilization_accum = thought_node_utilization_accum + thought_aux["local_delib_thought_node_utilization_loss"]
                thought_steps += 1

            if self.global_anchor_count > 0:
                anchor_feedback, global_anchor_prefix_state, anchor_stats, anchor_aux = self._compute_global_anchor_feedback(
                    h,
                    head_states,
                    global_anchor_prefix_state=global_anchor_prefix_state,
                    hierarchy_summary=scratch_hierarchy_summary if self.global_anchor_use_hierarchy else None,
                    scratch_summary=scratch_feedback if self.global_anchor_use_scratch else None,
                    thought_summary=thought_feedback if self.global_anchor_use_thought and self.use_thought_graph else None,
                )
                update_inputs = torch.cat([update_inputs, anchor_feedback], dim=-1)
                global_anchors_used = max(global_anchors_used, int(anchor_stats["global_anchors_used"]))
                mean_anchor_read_weight += float(anchor_stats["mean_anchor_read_weight"])
                mean_anchor_write_weight += float(anchor_stats["mean_anchor_write_weight"])
                mean_anchor_norm += float(anchor_stats["mean_anchor_norm"])
                anchor_usage_accum = anchor_usage_accum + anchor_aux["local_delib_anchor_usage_loss"]
                anchor_steps += 1

            delta = self.update(update_inputs)

            if self.use_token_gate or self.adaptive_halt:
                head_states = self.state_head(h)
            if self.use_token_gate:
                delta = delta * head_states["halt_gate"]

            if self.adaptive_halt:
                executed_steps_per_token += active_mask.to(executed_steps_per_token.dtype)
                active_float = active_mask.to(h.dtype)
                h = h + delta * active_float
                halted_now = head_states["halt_gate"] >= halt_threshold
                active_mask = active_mask & (~halted_now)
            else:
                h = h + delta
                executed_steps_per_token += 1.0

            if self.branch_factor > 0 and (step_idx % self.branch_every == 0):
                branch_context = self._compute_branch_context(h)
                branch_proposals = branch_context["branch_proposals"]
                branch_scores = branch_context["branch_scores"]
                branch_active_mask = branch_context["active_mask"]
                branch_active_mask_f = branch_context["active_mask_f"]
                active_count = int(branch_context["active_count"])
                branch_summary = branch_context["branch_summary"]
                consensus_summary = branch_context["consensus_summary"]
                consensus_mask = branch_context["consensus_mask"]
                token_disagreement = branch_context["token_disagreement"]
                verifier_scores = branch_context["verifier_scores"]
                effective_branch_weights = branch_context["effective_branch_weights"]
                entropy = branch_context["entropy"]
                max_entropy = float(branch_context["max_entropy"])
                pre_merge_state = h

                if self.branch_consensus or self.branch_verifier:
                    consensus_summary_for_merge = consensus_summary if self.branch_consensus else torch.zeros_like(branch_summary)
                    h, merge_weight, consensus_merge_weight = self.branch_consensus_merge(
                        h,
                        branch_summary,
                        consensus_summary_for_merge,
                        consensus_mask.unsqueeze(-1),
                    )
                else:
                    h, merge_weight = self.branch_merge(h, branch_summary)
                    consensus_merge_weight = torch.zeros_like(merge_weight)

                # Encourage branch-score entropy and proposal diversity only when branching is active.
                branch_entropy_accum = branch_entropy_accum + (max_entropy - entropy).mean()
                branch_support = (effective_branch_weights * branch_scores).sum(dim=-1)
                if self.branch_verifier:
                    branch_support = 0.5 * (branch_support + (effective_branch_weights * verifier_scores).sum(dim=-1))
                branch_novelty = 0.5 * (1.0 - F.cosine_similarity(branch_summary, pre_merge_state, dim=-1))
                branch_usefulness_accum = branch_usefulness_accum + (1.0 - (branch_support * branch_novelty)).mean()
                if active_count > 1:
                    normalized = F.normalize(branch_proposals, dim=-1)
                    pairwise_sim = torch.matmul(normalized, normalized.transpose(-2, -1))
                    eye = torch.eye(self.branch_factor, device=h.device, dtype=pairwise_sim.dtype).view(
                        1, 1, self.branch_factor, self.branch_factor
                    )
                    off_diag = branch_active_mask_f.unsqueeze(-1) * branch_active_mask_f.unsqueeze(-2) * (1.0 - eye)
                    off_diag_mean = (
                        (pairwise_sim.square() * off_diag).sum(dim=(-1, -2))
                        / off_diag.sum(dim=(-1, -2)).clamp_min(1.0)
                    ).mean()
                    branch_diversity_accum = branch_diversity_accum + off_diag_mean

                mean_branch_score_accum += float(
                    (
                        (branch_scores * branch_active_mask_f).sum(dim=-1)
                        / branch_active_mask_f.sum(dim=-1).clamp_min(1.0)
                    ).mean().item()
                )
                max_branch_score = max(
                    max_branch_score,
                    float(branch_scores.masked_fill(~branch_active_mask, 0.0).max().item()),
                )
                mean_merge_weight_accum += float(merge_weight.mean().item())
                mean_branch_disagreement_accum += float(token_disagreement.mean().item())
                mean_branch_consensus_weight_accum += float(consensus_merge_weight.mean().item())
                mean_branch_verifier_score_accum += float(
                    (
                        (verifier_scores * branch_active_mask_f).sum(dim=-1)
                        / branch_active_mask_f.sum(dim=-1).clamp_min(1.0)
                    ).mean().item()
                )
                mean_branch_entropy_accum += float(entropy.mean().item())
                branch_consensus_used_accum += float(consensus_mask.mean().item())
                token_branch_mask = (head_states["salience"] * head_states["uncertainty"]) > 0.25
                fraction_tokens_branched_accum += float(token_branch_mask.to(torch.float32).mean().item())
                branch_steps += 1
                branch_factor_used = active_count
            if stage_states is not None:
                stage_states.append(h.detach())

        head_states = self.state_head(h)
        mean_final_halt = float(head_states["halt_gate"].mean().item())
        mean_executed_steps_per_token = float(executed_steps_per_token.mean().item())
        max_executed_steps_any_token = int(executed_steps_per_token.max().item())
        fraction_halted_early = float((executed_steps_per_token < float(self.micro_steps)).to(torch.float32).mean().item())

        if self.use_phrase_consensus:
            mean_agreement_score /= float(self.micro_steps)
        mean_branch_score = mean_branch_score_accum / float(max(branch_steps, 1))
        mean_merge_weight = mean_merge_weight_accum / float(max(branch_steps, 1))
        fraction_tokens_branched = fraction_tokens_branched_accum / float(max(branch_steps, 1))
        mean_branch_disagreement = mean_branch_disagreement_accum / float(max(branch_steps, 1))
        mean_branch_consensus_weight = mean_branch_consensus_weight_accum / float(max(branch_steps, 1))
        mean_branch_verifier_score = mean_branch_verifier_score_accum / float(max(branch_steps, 1))
        mean_branch_entropy = mean_branch_entropy_accum / float(max(branch_steps, 1))
        branch_consensus_used = branch_consensus_used_accum / float(max(branch_steps, 1))
        mean_hierarchy_feedback_norm = hierarchy_feedback_norm_accum / float(max(hierarchy_feedback_steps, 1))
        mean_upward_message_norm = mean_upward_message_norm_accum / float(max(deep_hierarchy_steps, 1))
        mean_downward_message_norm = mean_downward_message_norm_accum / float(max(deep_hierarchy_steps, 1))
        mean_scale_gate = mean_scale_gate_accum / float(max(deep_hierarchy_steps, 1))
        mean_alignment_norm = mean_alignment_norm_accum / float(max(flocking_steps, 1))
        mean_cohesion_norm = mean_cohesion_norm_accum / float(max(flocking_steps, 1))
        mean_separation_norm = mean_separation_norm_accum / float(max(flocking_steps, 1))
        mean_flocking_total_norm = mean_flocking_total_norm_accum / float(max(flocking_steps, 1))
        flocking_neighbor_count = flocking_neighbor_count_accum / float(max(flocking_steps, 1))
        fraction_flocking_tokens_active = fraction_flocking_tokens_active_accum / float(max(flocking_steps, 1))
        mean_scratch_read_weight = mean_scratch_read_weight / float(max(self.micro_steps, 1))
        mean_scratch_write_weight = mean_scratch_write_weight / float(max(self.micro_steps, 1))
        mean_scratch_norm = mean_scratch_norm / float(max(self.micro_steps, 1))
        mean_scratch_refine_norm = mean_scratch_refine_norm / float(max(self.micro_steps, 1))
        mean_scratch_summary_norm = mean_scratch_summary_norm / float(max(self.micro_steps, 1))
        mean_branch_to_scratch_weight = mean_branch_to_scratch_weight / float(max(self.micro_steps, 1))
        mean_hierarchy_to_scratch_weight = mean_hierarchy_to_scratch_weight / float(max(self.micro_steps, 1))
        mean_thought_degree = mean_thought_degree_accum / float(max(thought_steps, 1))
        mean_token_to_thought_weight = mean_token_to_thought_weight_accum / float(max(thought_steps, 1))
        mean_thought_to_token_weight = mean_thought_to_token_weight_accum / float(max(thought_steps, 1))
        mean_thought_update_norm = mean_thought_update_norm_accum / float(max(thought_steps, 1))
        mean_anchor_read_weight = mean_anchor_read_weight / float(max(anchor_steps, 1))
        mean_anchor_write_weight = mean_anchor_write_weight / float(max(anchor_steps, 1))
        mean_anchor_norm = mean_anchor_norm / float(max(anchor_steps, 1))
        steps = float(max(self.micro_steps, 1))
        aux_losses = {
            "local_delib_halt_sparsity_loss": head_states["halt_gate"].mean(),
            "local_delib_branch_diversity_loss": branch_diversity_accum / float(max(branch_steps, 1)),
            "local_delib_branch_entropy_loss": branch_entropy_accum / float(max(branch_steps, 1)),
            "local_delib_consensus_agreement_loss": consensus_disagreement_accum / steps,
            "local_delib_scratch_utilization_loss": 1.0 - (scratch_utilization_accum / steps),
            "local_delib_flocking_stability_loss": flocking_stability_accum / float(max(flocking_steps, 1)),
            "local_delib_thought_edge_stability_loss": thought_edge_stability_accum / float(max(thought_steps, 1)),
            "local_delib_thought_node_utilization_loss": thought_node_utilization_accum / float(max(thought_steps, 1)),
            "local_delib_hierarchy_agreement_loss": hierarchy_agreement_accum / float(max(deep_hierarchy_steps, 1)),
            "local_delib_branch_usefulness_loss": branch_usefulness_accum / float(max(branch_steps, 1)),
            "local_delib_anchor_usage_loss": anchor_usage_accum / float(max(anchor_steps, 1)),
        }
        self.last_aux_losses = aux_losses
        stats: dict[str, torch.Tensor | float | int] = {
            "mean_salience": float(head_states["salience"].mean().item()),
            "mean_uncertainty": float(head_states["uncertainty"].mean().item()),
            "mean_halt": mean_final_halt,
            "mean_final_halt": mean_final_halt,
            "executed_steps": self.micro_steps,
            "mean_executed_steps_per_token": mean_executed_steps_per_token,
            "max_executed_steps_any_token": max_executed_steps_any_token,
            "fraction_halted_early": fraction_halted_early,
            "mean_neighbor_count": mean_neighbor_count,
            "mean_sequence_neighbor_weight": mean_sequence_neighbor_weight,
            "mean_semantic_neighbor_weight": mean_semantic_neighbor_weight,
            "mean_phrase_neighbor_weight": mean_phrase_neighbor_weight,
            "semantic_topk_used": semantic_topk_used,
            "mean_agreement_score": mean_agreement_score,
            "mean_alignment_norm": mean_alignment_norm,
            "mean_cohesion_norm": mean_cohesion_norm,
            "mean_separation_norm": mean_separation_norm,
            "mean_flocking_total_norm": mean_flocking_total_norm,
            "flocking_neighbor_count": flocking_neighbor_count,
            "fraction_flocking_tokens_active": fraction_flocking_tokens_active,
            "mean_branch_score": mean_branch_score,
            "max_branch_score": max_branch_score,
            "mean_merge_weight": mean_merge_weight,
            "branch_factor_used": branch_factor_used,
            "fraction_tokens_branched": fraction_tokens_branched,
            "mean_branch_disagreement": mean_branch_disagreement,
            "mean_branch_consensus_weight": mean_branch_consensus_weight,
            "mean_branch_verifier_score": mean_branch_verifier_score,
            "mean_branch_entropy": mean_branch_entropy,
            "branch_consensus_used": branch_consensus_used,
            "hierarchy_levels_used": len(self.hierarchy_levels),
            "mean_hierarchy_feedback_norm": mean_hierarchy_feedback_norm,
            "hierarchy_level_chunk_counts": hierarchy_level_chunk_counts,
            "phrase_nodes_used": phrase_nodes_used,
            "span_nodes_used": span_nodes_used,
            "sequence_summary_used": sequence_summary_used,
            "mean_upward_message_norm": mean_upward_message_norm,
            "mean_downward_message_norm": mean_downward_message_norm,
            "mean_scale_gate": mean_scale_gate,
            "hierarchy_depth_used": hierarchy_depth_used,
            "scratch_slots_used": scratch_slots_used,
            "mean_scratch_read_weight": mean_scratch_read_weight,
            "mean_scratch_write_weight": mean_scratch_write_weight,
            "mean_scratch_norm": mean_scratch_norm,
            "mean_scratch_refine_norm": mean_scratch_refine_norm,
            "mean_scratch_summary_norm": mean_scratch_summary_norm,
            "mean_branch_to_scratch_weight": mean_branch_to_scratch_weight,
            "mean_hierarchy_to_scratch_weight": mean_hierarchy_to_scratch_weight,
            "scratch_reset_ok": scratch_reset_ok,
            "thought_nodes_used": thought_nodes_used,
            "mean_thought_degree": mean_thought_degree,
            "mean_token_to_thought_weight": mean_token_to_thought_weight,
            "mean_thought_to_token_weight": mean_thought_to_token_weight,
            "mean_thought_update_norm": mean_thought_update_norm,
            "thought_graph_steps_used": thought_graph_steps_used,
            "global_anchors_used": global_anchors_used,
            "mean_anchor_read_weight": mean_anchor_read_weight,
            "mean_anchor_write_weight": mean_anchor_write_weight,
            "mean_anchor_norm": mean_anchor_norm,
        }
        if scratch_summary_vector is not None:
            stats["scratch_summary_vector"] = scratch_summary_vector
        if stage_states is not None:
            return h, stats, stage_states
        return h, stats

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | int]]:
        h = self.in_proj(x)
        h, stats = self.deliberate_state(h)
        output = x + self.out_proj(h)
        return output, stats
