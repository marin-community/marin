# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

import pytest

from experiments.post_training.async_rl_stop_diagnostics import (
    parser_inside_thinking,
    repeated_window_fraction,
    summarize,
)


class Decoder:
    pieces: ClassVar[dict[int, str]] = {1: "<think>", 2: "</think>", 3: "#### 12", 4: " filler "}

    def decode(self, ids, **unused):
        return "".join(self.pieces[i] for i in ids)


@pytest.mark.parametrize(
    "prompt,response,expected",
    [
        ([1], [3], True),
        ([], [3], False),
        ([1], [2, 3], False),
        ([1], [3, 2], True),
        ([1, 2], [1, 3], True),
        ([1], [4, 2, 1, 3], True),
        ([1], [4], None),
    ],
)
def test_actual_token_state_including_reopened_thinking(prompt, response, expected):
    decoder = Decoder()
    assert parser_inside_thinking(prompt, response, decoder.decode(response), decoder, 1, 2) is expected


def test_decode_mismatch_rejects():
    with pytest.raises(ValueError, match="retained response"):
        parser_inside_thinking([1], [3], "other text", Decoder(), 1, 2)


def test_repetition_window_denominator_and_short_response():
    assert repeated_window_fraction([1] * 15) == 0
    assert repeated_window_fraction([1] * 16) == 0
    assert repeated_window_fraction([1] * 17) == 0.5
    assert repeated_window_fraction(list(range(100))) == 0
    assert repeated_window_fraction([1] * 19) == 0.75


def test_nonexclusive_fractions_keep_unknown_eos_and_missing_marker_denominators():
    rows = [
        dict(
            stop_reason="length",
            reward=1,
            parser_inside_thinking=True,
            no_effective_eos=False,
            repeated_16gram_fraction=0.75,
        ),
        dict(
            stop_reason="length",
            reward=0,
            parser_inside_thinking=None,
            no_effective_eos=None,
            repeated_16gram_fraction=0,
        ),
    ]
    result = summarize(rows)
    assert result["all"]["parser_inside_thinking_among_rows"]["fraction"] == 0.5
    assert result["all"]["parser_inside_thinking_among_marker_rows"]["fraction"] == 1
    assert result["all"]["no_effective_eos"] == dict(numerator=0, denominator=1, fraction=0)
    assert result["all"]["effective_eos_unknown_rows"] == 1
    assert result["rewarded_length"]["parser_and_repetition_ge_half"]["fraction"] == 1
    assert summarize([])["all"]["no_effective_eos"]["fraction"] is None
