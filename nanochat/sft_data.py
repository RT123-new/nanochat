"""Helpers for building stable supervised-finetuning batches."""

from __future__ import annotations

from typing import Any


def prepare_packed_conversation(
    tokenizer: Any,
    conversation: Any,
    row_capacity: int,
) -> tuple[list[int], list[int]] | None:
    """Render one conversation so it can safely participate in row packing.

    Conversations are truncated to the row capacity up front so they cannot
    poison the packing buffer by never fitting. If truncation removes all
    supervised assistant tokens, the example is skipped.
    """

    ids, mask = tokenizer.render_conversation(conversation, max_tokens=row_capacity)
    if len(ids) != len(mask):
        raise ValueError("render_conversation returned mismatched ids/mask lengths")
    if len(ids) == 0:
        return None
    if len(ids) > row_capacity:
        ids = ids[:row_capacity]
        mask = mask[:row_capacity]
    if not any(mask):
        return None
    return ids, mask
