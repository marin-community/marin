# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from levanter import eval_harness
from levanter.eval_harness import LevanterHarnessLM


class _MarinStyleTokenizer:
    pad_token_id = None
    eos_token_id = 2

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]:
        offset = 100 if add_special_tokens else 0
        return [[offset + len(text)] for text in texts]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, int):
            raise TypeError("argument 'ids': 'int' object cannot be converted to 'Sequence'")
        if skip_special_tokens:
            ids = [token_id for token_id in ids if token_id != self.eos_token_id]
        return ",".join(str(token_id) for token_id in ids)


def test_tok_encode_supports_non_callable_marin_tokenizer_batch():
    harness = object.__new__(LevanterHarnessLM)
    harness.leader = SimpleNamespace(tokenizer=_MarinStyleTokenizer())

    assert harness.tok_encode(("ab", "hello"), add_special_tokens=False) == [[2], [5]]


def test_tok_encode_supports_non_callable_marin_tokenizer_single_string():
    harness = object.__new__(LevanterHarnessLM)
    harness.leader = SimpleNamespace(tokenizer=_MarinStyleTokenizer())

    assert harness.tok_encode("hello", add_special_tokens=True) == [105]


def test_tok_decode_wraps_scalar_token_for_non_callable_marin_tokenizer():
    harness = object.__new__(LevanterHarnessLM)
    harness.leader = SimpleNamespace(tokenizer=_MarinStyleTokenizer())

    assert harness.tok_decode(2) == "2"
    assert harness.tok_decode([1, 2, 3], skip_special_tokens=True) == "1,3"


def test_generate_until_uses_leader_axis_resources(monkeypatch):
    captured: dict[str, object] = {}

    class FakeEngine:
        def generate(self, requests, *, step_callback=None):
            return SimpleNamespace(tokens=[[3]])

    def fake_from_model_with_config(*, model, tokenizer, config, axis_resources):
        captured["axis_resources"] = axis_resources
        return FakeEngine()

    monkeypatch.setattr(
        eval_harness.InferenceEngine,
        "from_model_with_config",
        staticmethod(fake_from_model_with_config),
    )

    leader = SimpleNamespace(
        axis_resources={"batch": "data"},
        tokenizer=_MarinStyleTokenizer(),
        EvalBatch=SimpleNamespace(size=1),
        EvalPos=SimpleNamespace(size=8),
        _generation_kwargs={},
        model=SimpleNamespace(initial_cache=lambda: None, decode=lambda: None),
        sample_logging_config=SimpleNamespace(should_log=lambda: False),
        profiler_config=SimpleNamespace(enabled=False),
    )
    harness = LevanterHarnessLM(leader)

    outputs = harness.generate_until(
        [SimpleNamespace(args=("abc", {"max_gen_toks": 1, "temperature": 0.0, "n": 1, "until": []}))]
    )

    assert captured["axis_resources"] == {"batch": "data"}
    assert outputs == ["3"]
