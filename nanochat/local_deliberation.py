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
    ) -> None:
        super().__init__()
        self.semantic_topk = semantic_topk
        self.semantic_lookback = semantic_lookback
        self.use_phrase_nodes = use_phrase_nodes
        self.phrase_chunk_size = phrase_chunk_size

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

    def summarize(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | int]]:
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
        semantic_topk_used = 0
        total_neighbor_count = 0.0
        total_sequence_weight = 0.0
        total_semantic_weight = 0.0
        total_phrase_weight = 0.0

        for token_idx in range(seq_len):
            if token_idx == 0:
                continue

            token_q = q[:, token_idx, :]
            score_chunks = []
            value_chunks = []
            type_chunks = []

            # Immediate sequence predecessor.
            seq_score = (token_q * k[:, token_idx - 1, :]).sum(dim=-1, keepdim=True) * scale
            seq_value = v[:, token_idx - 1 : token_idx, :]
            score_chunks.append(seq_score)
            value_chunks.append(seq_value)
            type_chunks.append("sequence")

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

            # Optional phrase-node link (causal chunk-prefix summary).
            if phrase_k is not None and phrase_v is not None:
                phrase_score = (token_q * phrase_k[:, token_idx - 1, :]).sum(dim=-1, keepdim=True) * scale
                phrase_value = phrase_v[:, token_idx - 1 : token_idx, :]
                score_chunks.append(phrase_score)
                value_chunks.append(phrase_value)
                type_chunks.append("phrase")

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

        denom = max(float(seq_len - 1), 1.0)
        stats: dict[str, float | int] = {
            "mean_neighbor_count": total_neighbor_count / denom,
            "mean_sequence_neighbor_weight": total_sequence_weight / denom,
            "mean_semantic_neighbor_weight": total_semantic_weight / denom,
            "mean_phrase_neighbor_weight": total_phrase_weight / denom,
            "semantic_topk_used": semantic_topk_used,
        }
        return summary, stats


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
        hierarchy_chunk_sizes: list[int] | None = None,
        scratch_slots: int = 0,
        scratch_dim: int = 0,
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
        if scratch_slots < 0:
            raise ValueError("scratch_slots must be >= 0")
        if scratch_dim < 0:
            raise ValueError("scratch_dim must be >= 0")
        if scratch_slots > 0 and scratch_dim < 1:
            raise ValueError("scratch_dim must be >= 1 when scratch_slots > 0")

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
        self.hierarchy_chunk_sizes = tuple(hierarchy_chunk_sizes or [])
        self.scratch_slots = scratch_slots
        self.scratch_dim = scratch_dim if scratch_slots > 0 else 0
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

        update_in_dim = state_dim * 3
        if self.hierarchy_levels:
            update_in_dim += state_dim
        if self.use_phrase_consensus:
            update_in_dim += state_dim
        if self.semantic_topk > 0:
            self.semantic_q = nn.Linear(state_dim, state_dim)
            self.semantic_k = nn.Linear(state_dim, state_dim)
            self.semantic_v = nn.Linear(state_dim, state_dim)
            update_in_dim += state_dim
        if self.use_neighbor_graph:
            self.neighbor_graph_mixer = CausalNeighborGraphMixer(
                state_dim=state_dim,
                phrase_chunk_size=phrase_chunk_size,
                semantic_topk=semantic_topk,
                semantic_lookback=semantic_lookback,
                use_phrase_nodes=use_phrase_consensus,
            )
        if self.scratch_slots > 0:
            self.scratch_query = nn.Linear(state_dim, self.scratch_dim)
            self.scratch_write_value = nn.Linear(state_dim, self.scratch_dim)
            self.scratch_read_mix = nn.Linear(state_dim, self.scratch_dim)
            self.scratch_to_state = nn.Linear(self.scratch_dim, state_dim)
            self.scratch_init = nn.Parameter(torch.zeros(self.scratch_slots, self.scratch_dim))
            self.scratch_read_temp = nn.Parameter(torch.tensor(1.0))
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

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        if self.scratch_slots > 0:
            nn.init.zeros_(self.scratch_to_state.weight)
            nn.init.zeros_(self.scratch_to_state.bias)
        self.last_aux_losses: dict[str, torch.Tensor] | None = None

    def _compute_scratch_feedback(
        self,
        h: torch.Tensor,
        head_states: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, float, float, int, float]:
        bsz, seq_len, _ = h.shape
        scratch = self.scratch_init.unsqueeze(0).expand(bsz, -1, -1).clone()
        scratch_feedback = torch.zeros(bsz, seq_len, h.shape[-1], device=h.device, dtype=h.dtype)

        read_weight_sum = 0.0
        write_weight_sum = 0.0
        slot_write_mass = torch.zeros(bsz, self.scratch_slots, device=h.device, dtype=h.dtype)

        read_temp = torch.clamp(self.scratch_read_temp, min=0.1)
        for token_idx in range(seq_len):
            token_state = h[:, token_idx, :]
            query = self.scratch_query(token_state)
            slot_logits = torch.einsum("bd,bsd->bs", query, scratch) / math.sqrt(self.scratch_dim)
            slot_attn = torch.softmax(slot_logits * read_temp, dim=-1)

            read_gate = head_states["uncertainty"][:, token_idx, :]
            read_summary = torch.einsum("bs,bsd->bd", slot_attn, scratch)
            scratch_feedback[:, token_idx, :] = self.scratch_to_state(read_summary * read_gate)
            read_weight_sum += float((slot_attn.mean(dim=-1) * read_gate.squeeze(-1)).mean().item())

            write_gate = (head_states["salience"][:, token_idx, :] * head_states["uncertainty"][:, token_idx, :]).squeeze(-1)
            write_value = self.scratch_write_value(token_state) + self.scratch_read_mix(read_summary)
            write_weights = slot_attn * write_gate.unsqueeze(-1)
            scratch = scratch + write_weights.unsqueeze(-1) * write_value.unsqueeze(1)
            slot_write_mass += write_weights
            write_weight_sum += float(write_weights.mean().item())

        steps = float(max(seq_len, 1))
        mean_read_weight = read_weight_sum / steps
        mean_write_weight = write_weight_sum / steps
        scratch_slots_used = int((slot_write_mass > 1e-4).any(dim=0).sum().item())
        mean_scratch_norm = float(scratch.norm(dim=-1).mean().item())
        return scratch_feedback, mean_read_weight, mean_write_weight, scratch_slots_used, mean_scratch_norm

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

    def deliberate_state(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | int]]:
        head_states: dict[str, torch.Tensor] = self.state_head(h)
        self.last_aux_losses = None

        semantic_topk_used = 0
        mean_neighbor_count = 0.0
        mean_sequence_neighbor_weight = 0.0
        mean_semantic_neighbor_weight = 0.0
        mean_phrase_neighbor_weight = 0.0
        mean_agreement_score = 0.0
        mean_branch_score_accum = 0.0
        max_branch_score = 0.0
        mean_merge_weight_accum = 0.0
        fraction_tokens_branched_accum = 0.0
        branch_steps = 0
        branch_factor_used = 0
        hierarchy_feedback_norm_accum = 0.0
        hierarchy_feedback_steps = 0
        hierarchy_level_chunk_counts = [0 for _ in self.hierarchy_levels]
        scratch_slots_used = 0
        mean_scratch_read_weight = 0.0
        mean_scratch_write_weight = 0.0
        mean_scratch_norm = 0.0
        consensus_disagreement_accum = h.new_zeros(())
        branch_entropy_accum = h.new_zeros(())
        branch_diversity_accum = h.new_zeros(())
        scratch_utilization_accum = h.new_zeros(())
        executed_steps_per_token = torch.zeros(h.shape[0], h.shape[1], 1, device=h.device, dtype=torch.float32)
        active_mask = torch.ones(h.shape[0], h.shape[1], 1, device=h.device, dtype=torch.bool)
        halt_threshold = torch.sigmoid(self.halt_threshold_logit)

        for step_idx in range(self.micro_steps):
            mixed = self.mixer(h)
            _, phrase_broadcast = self.phrase_pool(h)
            hierarchy_feedback = torch.zeros_like(h)
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
            if self.use_neighbor_graph:
                semantic_summary, graph_stats = self.neighbor_graph_mixer.summarize(h)
                semantic_topk_used = int(graph_stats["semantic_topk_used"])
                mean_neighbor_count = float(graph_stats["mean_neighbor_count"])
                mean_sequence_neighbor_weight = float(graph_stats["mean_sequence_neighbor_weight"])
                mean_semantic_neighbor_weight = float(graph_stats["mean_semantic_neighbor_weight"])
                mean_phrase_neighbor_weight = float(graph_stats["mean_phrase_neighbor_weight"])
                if self.use_phrase_consensus:
                    update_inputs = torch.cat(
                        [h, mixed, phrase_broadcast, semantic_summary, consensus_feedback, hierarchy_feedback],
                        dim=-1,
                    ) if self.hierarchy_levels else torch.cat([h, mixed, phrase_broadcast, semantic_summary, consensus_feedback], dim=-1)
                else:
                    update_inputs = torch.cat([h, mixed, phrase_broadcast, semantic_summary, hierarchy_feedback], dim=-1) if self.hierarchy_levels else torch.cat([h, mixed, phrase_broadcast, semantic_summary], dim=-1)
            elif self.semantic_topk > 0:
                semantic_summary, _, semantic_weights, semantic_topk_used = self._semantic_neighbor_summary(h)
                used_weight_mask = (semantic_weights > 0).to(semantic_weights.dtype)
                used_weight_count = used_weight_mask.sum().clamp_min(1.0)
                mean_semantic_neighbor_weight = float((semantic_weights * used_weight_mask).sum().item() / used_weight_count.item())
                mean_neighbor_count = float(semantic_topk_used)
                if self.use_phrase_consensus:
                    update_inputs = torch.cat(
                        [h, mixed, phrase_broadcast, semantic_summary, consensus_feedback, hierarchy_feedback],
                        dim=-1,
                    ) if self.hierarchy_levels else torch.cat([h, mixed, phrase_broadcast, semantic_summary, consensus_feedback], dim=-1)
                else:
                    update_inputs = torch.cat([h, mixed, phrase_broadcast, semantic_summary, hierarchy_feedback], dim=-1) if self.hierarchy_levels else torch.cat([h, mixed, phrase_broadcast, semantic_summary], dim=-1)
            else:
                if self.use_phrase_consensus:
                    update_inputs = torch.cat([h, mixed, phrase_broadcast, consensus_feedback, hierarchy_feedback], dim=-1) if self.hierarchy_levels else torch.cat([h, mixed, phrase_broadcast, consensus_feedback], dim=-1)
                else:
                    update_inputs = torch.cat([h, mixed, phrase_broadcast, hierarchy_feedback], dim=-1) if self.hierarchy_levels else torch.cat([h, mixed, phrase_broadcast], dim=-1)

            if self.scratch_slots > 0 and not (self.use_token_gate or self.adaptive_halt):
                head_states = self.state_head(h)

            if self.scratch_slots > 0:
                (
                    scratch_feedback,
                    step_scratch_read_weight,
                    step_scratch_write_weight,
                    step_scratch_slots_used,
                    step_scratch_norm,
                ) = self._compute_scratch_feedback(h, head_states)
                update_inputs = torch.cat([update_inputs, scratch_feedback], dim=-1)
                mean_scratch_read_weight += step_scratch_read_weight
                mean_scratch_write_weight += step_scratch_write_weight
                mean_scratch_norm += step_scratch_norm
                scratch_slots_used = max(scratch_slots_used, step_scratch_slots_used)
                scratch_utilization_accum = scratch_utilization_accum + 0.5 * (
                    head_states["uncertainty"].mean() + (head_states["salience"] * head_states["uncertainty"]).mean()
                )

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
                branch_proposals = self.branch_proposal(h)
                branch_logits = self.branch_scorer(h, branch_proposals)
                branch_scores = torch.sigmoid(branch_logits)
                branch_weights = torch.softmax(branch_logits, dim=-1)
                branch_summary = (branch_weights.unsqueeze(-1) * branch_proposals).sum(dim=2)
                h, merge_weight = self.branch_merge(h, branch_summary)

                # Encourage branch-score entropy and proposal diversity only when branching is active.
                entropy = -(branch_weights * torch.log(branch_weights.clamp_min(1e-8))).sum(dim=-1)
                max_entropy = math.log(float(self.branch_factor))
                branch_entropy_accum = branch_entropy_accum + (max_entropy - entropy).mean()
                if self.branch_factor > 1:
                    normalized = F.normalize(branch_proposals, dim=-1)
                    pairwise_sim = torch.matmul(normalized, normalized.transpose(-2, -1))
                    eye = torch.eye(self.branch_factor, device=h.device, dtype=pairwise_sim.dtype).view(1, 1, self.branch_factor, self.branch_factor)
                    off_diag = (1.0 - eye)
                    off_diag_mean = ((pairwise_sim.square() * off_diag).sum(dim=(-1, -2)) / float(self.branch_factor * (self.branch_factor - 1))).mean()
                    branch_diversity_accum = branch_diversity_accum + off_diag_mean

                mean_branch_score_accum += float(branch_scores.mean().item())
                max_branch_score = max(max_branch_score, float(branch_scores.max().item()))
                mean_merge_weight_accum += float(merge_weight.mean().item())
                token_branch_mask = (head_states["salience"] * head_states["uncertainty"]) > 0.25
                fraction_tokens_branched_accum += float(token_branch_mask.to(torch.float32).mean().item())
                branch_steps += 1
                branch_factor_used = self.branch_factor

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
        mean_hierarchy_feedback_norm = hierarchy_feedback_norm_accum / float(max(hierarchy_feedback_steps, 1))
        mean_scratch_read_weight = mean_scratch_read_weight / float(max(self.micro_steps, 1))
        mean_scratch_write_weight = mean_scratch_write_weight / float(max(self.micro_steps, 1))
        mean_scratch_norm = mean_scratch_norm / float(max(self.micro_steps, 1))
        steps = float(max(self.micro_steps, 1))
        aux_losses = {
            "local_delib_halt_sparsity_loss": head_states["halt_gate"].mean(),
            "local_delib_branch_diversity_loss": branch_diversity_accum / float(max(branch_steps, 1)),
            "local_delib_branch_entropy_loss": branch_entropy_accum / float(max(branch_steps, 1)),
            "local_delib_consensus_agreement_loss": consensus_disagreement_accum / steps,
            "local_delib_scratch_utilization_loss": 1.0 - (scratch_utilization_accum / steps),
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
            "mean_branch_score": mean_branch_score,
            "max_branch_score": max_branch_score,
            "mean_merge_weight": mean_merge_weight,
            "branch_factor_used": branch_factor_used,
            "fraction_tokens_branched": fraction_tokens_branched,
            "hierarchy_levels_used": len(self.hierarchy_levels),
            "mean_hierarchy_feedback_norm": mean_hierarchy_feedback_norm,
            "hierarchy_level_chunk_counts": hierarchy_level_chunk_counts,
            "scratch_slots_used": scratch_slots_used,
            "mean_scratch_read_weight": mean_scratch_read_weight,
            "mean_scratch_write_weight": mean_scratch_write_weight,
            "mean_scratch_norm": mean_scratch_norm,
        }
        return h, stats

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | int]]:
        h = self.in_proj(x)
        h, stats = self.deliberate_state(h)
        output = x + self.out_proj(h)
        return output, stats
