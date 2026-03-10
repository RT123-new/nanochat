from dataclasses import dataclass

from nanochat.chat_format import (
    render_chat_messages,
    render_messages_for_completion,
    validate_chat_messages,
)


@dataclass
class ChatMessage:
    role: str
    content: str


class FakeTokenizer:
    def __init__(self) -> None:
        self._special_tokens = {
            "<|bos|>": 256,
            "<|user_start|>": 257,
            "<|user_end|>": 258,
            "<|assistant_start|>": 259,
            "<|assistant_end|>": 260,
            "<|python_start|>": 261,
            "<|python_end|>": 262,
            "<|output_start|>": 263,
            "<|output_end|>": 264,
        }

    def get_bos_token_id(self):
        return self._special_tokens["<|bos|>"]

    def encode_special(self, text):
        return self._special_tokens[text]

    def encode(self, text):
        return list(text.encode("utf-8"))


def test_validate_chat_messages_accepts_system_role() -> None:
    messages = [
        ChatMessage(role="system", content="Be terse."),
        ChatMessage(role="user", content="Hello"),
    ]

    validate_chat_messages(
        messages,
        max_messages=10,
        max_message_length=100,
        max_total_conversation_length=1000,
    )


def test_render_messages_for_completion_matches_chat_serializer_with_system_message() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        ChatMessage(role="system", content="Be terse."),
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi"),
        ChatMessage(role="user", content="What did I say earlier?"),
    ]

    prefix_ids, _ = render_chat_messages(tokenizer, messages, max_tokens=127)
    expected = prefix_ids + [tokenizer.encode_special("<|assistant_start|>")]

    actual = render_messages_for_completion(tokenizer, messages, max_tokens=128)

    assert actual == expected
