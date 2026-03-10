"""Shared chat validation and token rendering helpers."""

from __future__ import annotations

import copy
from typing import Any


ALLOWED_CHAT_ROLES = ("system", "user", "assistant")


def _field(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message[name]
    return getattr(message, name)


def _message_to_dict(message: Any) -> dict[str, Any]:
    return {
        "role": _field(message, "role"),
        "content": _field(message, "content"),
    }


def validate_chat_messages(
    messages: list[Any],
    *,
    max_messages: int,
    max_message_length: int,
    max_total_conversation_length: int,
) -> None:
    """Validate a web chat request payload."""
    if len(messages) == 0:
        raise ValueError("At least one message is required")
    if len(messages) > max_messages:
        raise ValueError(f"Too many messages. Maximum {max_messages} messages allowed per request")

    total_length = 0
    for idx, message in enumerate(messages):
        role = _field(message, "role")
        content = _field(message, "content")
        if role not in ALLOWED_CHAT_ROLES:
            raise ValueError(
                f"Message {idx} has invalid role. Must be 'system', 'user', or 'assistant'"
            )
        if not isinstance(content, str):
            raise ValueError(f"Message {idx} content must be a string")
        if not content:
            raise ValueError(f"Message {idx} has empty content")
        if len(content) > max_message_length:
            raise ValueError(
                f"Message {idx} is too long. Maximum {max_message_length} characters allowed per message"
            )
        total_length += len(content)

    if total_length > max_total_conversation_length:
        raise ValueError(
            f"Total conversation is too long. Maximum {max_total_conversation_length} characters allowed"
        )


def normalize_chat_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Return a validated, tokenizer-ready message list."""
    normalized = [_message_to_dict(message) for message in messages]
    if not normalized:
        raise ValueError("Conversation has no messages")

    if normalized[0]["role"] == "system":
        if len(normalized) < 2:
            raise ValueError("System message must be followed by a user message")
        normalized = copy.deepcopy(normalized)
        if normalized[1]["role"] != "user":
            raise ValueError("System message must be followed by a user message")
        system_content = normalized[0]["content"]
        normalized[1]["content"] = f"{system_content}\n\n{normalized[1]['content']}"
        normalized = normalized[1:]

    for idx, message in enumerate(normalized):
        expected_role = "user" if idx % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            raise ValueError(
                f"Message {idx} has role {message['role']} but should be {expected_role}"
            )

    return normalized


def flatten_message_content(content: Any) -> str:
    """Convert tokenizer/training message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content)
    return str(content)


def render_chat_messages(
    tokenizer: Any,
    messages: list[Any],
    *,
    max_tokens: int | None = 2048,
) -> tuple[list[int], list[int]]:
    """Render a chat conversation into ids and supervision mask."""
    normalized = normalize_chat_messages(messages)
    ids: list[int] = []
    mask: list[int] = []

    def add_tokens(token_ids: int | list[int], mask_value: int) -> None:
        values = [token_ids] if isinstance(token_ids, int) else token_ids
        ids.extend(values)
        mask.extend([mask_value] * len(values))

    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    python_start = tokenizer.encode_special("<|python_start|>")
    python_end = tokenizer.encode_special("<|python_end|>")
    output_start = tokenizer.encode_special("<|output_start|>")
    output_end = tokenizer.encode_special("<|output_end|>")

    add_tokens(bos, 0)
    for message in normalized:
        content = message["content"]
        if message["role"] == "user":
            if not isinstance(content, str):
                raise ValueError("User messages must be strings")
            add_tokens(user_start, 0)
            add_tokens(tokenizer.encode(content), 0)
            add_tokens(user_end, 0)
            continue

        add_tokens(assistant_start, 0)
        if isinstance(content, str):
            add_tokens(tokenizer.encode(content), 1)
        elif isinstance(content, list):
            for part in content:
                value_ids = tokenizer.encode(part["text"])
                if part["type"] == "text":
                    add_tokens(value_ids, 1)
                elif part["type"] == "python":
                    add_tokens(python_start, 1)
                    add_tokens(value_ids, 1)
                    add_tokens(python_end, 1)
                elif part["type"] == "python_output":
                    add_tokens(output_start, 0)
                    add_tokens(value_ids, 0)
                    add_tokens(output_end, 0)
                else:
                    raise ValueError(f"Unknown part type: {part['type']}")
        else:
            raise ValueError(f"Unknown assistant content type: {type(content)}")
        add_tokens(assistant_end, 1)

    if max_tokens is not None:
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
    return ids, mask


def render_messages_for_completion(
    tokenizer: Any,
    messages: list[Any],
    *,
    max_tokens: int | None = 2048,
) -> list[int]:
    """Render chat history and prime the assistant for the next completion."""
    prompt_messages = copy.deepcopy(list(messages))
    if prompt_messages and _field(prompt_messages[-1], "role") == "assistant":
        prompt_messages.pop()

    reserved_budget = None if max_tokens is None else max(max_tokens - 1, 1)
    ids, _ = render_chat_messages(tokenizer, prompt_messages, max_tokens=reserved_budget)
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    if max_tokens is not None and len(ids) >= max_tokens:
        ids = ids[: max_tokens - 1]
    ids.append(assistant_start)
    return ids
