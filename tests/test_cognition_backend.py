from nanochat.chat_format import render_messages_for_completion
from nanochat.cognition.backend import EngineBackend


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

    def decode(self, ids):
        byte_ids = [token for token in ids if token < 256]
        return bytes(byte_ids).decode("utf-8")


class FakeEngine:
    def __init__(self) -> None:
        self.prompt_tokens = None
        self.last_generate_kwargs = None
        self.model = type(
            "FakeModel",
            (),
            {"config": type("FakeConfig", (), {"sequence_len": 64})()},
        )()

    def generate_batch(self, tokens, **kwargs):
        self.prompt_tokens = tokens
        self.last_generate_kwargs = kwargs
        response = list("ok".encode("utf-8"))
        return [tokens + response], [[0] * len(tokens) + [1] * len(response)]


def test_engine_backend_uses_chat_serialization_and_decodes_response() -> None:
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    backend = EngineBackend(
        engine=engine,
        tokenizer=tokenizer,
        system_prompt="Be terse.",
        max_tokens=8,
        temperature=0.0,
        top_k=1,
    )

    response = backend.generate("Hello there")

    expected_tokens = render_messages_for_completion(
        tokenizer,
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello there"},
        ],
        max_tokens=64,
    )

    assert engine.prompt_tokens == expected_tokens
    assert response == "ok"


def test_engine_backend_captures_local_deliberation_stats_metadata() -> None:
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    engine.model.last_deliberation_stats = [
        {
            "layer_idx": 0,
            "agreement": 0.8,
            "mean_branch_score": 0.5,
            "branch_factor_used": 2,
            "hierarchy_levels_used": 2,
            "mean_hierarchy_feedback_norm": 0.4,
            "scratch_slots_used": 3,
            "mean_scratch_read_weight": 0.2,
            "halted_token_fraction": 0.6,
            "mean_steps_taken": 1.5,
        }
    ]
    backend = EngineBackend(engine=engine, tokenizer=tokenizer)

    response = backend.generate("Hello there")

    assert isinstance(response, str)
    assert response == "ok"
    assert backend.last_generation_metadata is not None
    assert backend.last_generation_metadata["local_deliberation_stats"][0]["agreement"] == 0.8
    assert backend.last_generation_metadata["model_local_delib.branch"]["branch_factor_used"] == 2.0
    assert backend.last_generation_metadata["model_local_delib.hierarchy"]["hierarchy_levels_used"] == 2.0
    assert backend.last_generation_metadata["model_local_delib.scratchpad"]["scratch_slots_used"] == 3.0
    assert backend.last_generation_metadata["model_local_delib.adaptive_halt"]["halted_token_fraction"] == 0.6


def test_engine_backend_generate_accepts_extra_generation_kwargs() -> None:
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    backend = EngineBackend(engine=engine, tokenizer=tokenizer)

    _ = backend.generate("Hello", local_delib=True, local_delib_branch_factor=2)

    assert engine.last_generate_kwargs is not None
    assert engine.last_generate_kwargs["local_delib"] is True
    assert engine.last_generate_kwargs["local_delib_branch_factor"] == 2
