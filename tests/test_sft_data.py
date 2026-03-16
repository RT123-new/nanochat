from __future__ import annotations

from nanochat.sft_data import prepare_packed_conversation


class FakeTokenizer:
    def render_conversation(self, conversation, max_tokens=None):
        ids = list(conversation["ids"])
        mask = list(conversation["mask"])
        if max_tokens is not None:
            ids = ids[:max_tokens]
            mask = mask[:max_tokens]
        return ids, mask


def test_prepare_packed_conversation_truncates_to_row_capacity() -> None:
    tokenizer = FakeTokenizer()
    conversation = {
        "ids": [10, 11, 12, 13, 14, 15],
        "mask": [0, 0, 1, 1, 1, 1],
    }

    prepared = prepare_packed_conversation(tokenizer, conversation, row_capacity=4)

    assert prepared == ([10, 11, 12, 13], [0, 0, 1, 1])


def test_prepare_packed_conversation_skips_examples_without_supervision() -> None:
    tokenizer = FakeTokenizer()
    conversation = {
        "ids": [10, 11, 12, 13],
        "mask": [0, 0, 0, 0],
    }

    prepared = prepare_packed_conversation(tokenizer, conversation, row_capacity=4)

    assert prepared is None


def test_prepare_packed_conversation_skips_when_truncation_removes_supervision() -> None:
    tokenizer = FakeTokenizer()
    conversation = {
        "ids": [10, 11, 12, 13, 14],
        "mask": [0, 0, 0, 0, 1],
    }

    prepared = prepare_packed_conversation(tokenizer, conversation, row_capacity=4)

    assert prepared is None
