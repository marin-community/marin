# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.rl.decoding import DecodingConfig, DecodingStrategy, default_eval_decoding


def test_greedy_decoding_accepts_deterministic_generation_fields():
    decoding = DecodingConfig(
        strategy=DecodingStrategy.GREEDY,
        temperature=0.0,
        repetition_penalty=1.1,
        presence_penalty=0.2,
        frequency_penalty=0.3,
        max_output_tokens=64,
        min_output_tokens=4,
        stop_strings=["<stop>"],
        ignore_eos=True,
    )

    trace = decoding.as_trace()

    assert trace.strategy == "greedy"
    assert trace.temperature == 0.0
    assert trace.top_k is None
    assert trace.top_p is None
    assert trace.min_p is None
    assert trace.seed is None
    assert trace.repetition_penalty == 1.1
    assert trace.presence_penalty == 0.2
    assert trace.frequency_penalty == 0.3
    assert trace.max_output_tokens == 64
    assert trace.min_output_tokens == 4
    assert trace.stop_strings == ("<stop>",)
    assert trace.ignore_eos is True


@pytest.mark.parametrize(
    ("overrides", "expected_field"),
    [
        ({"temperature": 0.7}, "temperature"),
        ({"top_k": 8}, "top_k"),
        ({"top_p": 0.91}, "top_p"),
        ({"min_p": 0.05}, "min_p"),
        ({"seed": 123}, "seed"),
    ],
)
def test_greedy_decoding_rejects_sampling_only_fields(overrides, expected_field):
    kwargs = {
        "strategy": DecodingStrategy.GREEDY,
        "temperature": 0.0,
        **overrides,
    }

    with pytest.raises(ValueError, match=expected_field):
        DecodingConfig(**kwargs)


def test_greedy_decoding_reports_all_sampling_only_fields():
    with pytest.raises(ValueError) as exc_info:
        DecodingConfig(
            strategy=DecodingStrategy.GREEDY,
            temperature=0.7,
            top_k=8,
            top_p=0.91,
            min_p=0.05,
            seed=123,
        )

    assert str(exc_info.value).endswith("temperature, top_k, top_p, min_p, seed")


def test_default_eval_decoding_strips_sampling_fields_before_switching_to_greedy():
    train_decoding = DecodingConfig(
        temperature=0.7,
        top_k=8,
        top_p=0.91,
        min_p=0.05,
        repetition_penalty=1.1,
        presence_penalty=0.2,
        frequency_penalty=0.3,
        max_output_tokens=64,
        min_output_tokens=4,
        stop_strings=["<stop>"],
        ignore_eos=True,
        seed=123,
    )

    eval_decoding = default_eval_decoding(train_decoding)

    assert eval_decoding.strategy == DecodingStrategy.GREEDY
    assert eval_decoding.temperature == 0.0
    assert eval_decoding.top_k is None
    assert eval_decoding.top_p is None
    assert eval_decoding.min_p is None
    assert eval_decoding.repetition_penalty is None
    assert eval_decoding.presence_penalty is None
    assert eval_decoding.frequency_penalty is None
    assert eval_decoding.seed is None
    assert eval_decoding.max_output_tokens == 64
    assert eval_decoding.min_output_tokens == 4
    assert eval_decoding.stop_strings == ["<stop>"]
    assert eval_decoding.ignore_eos is True
