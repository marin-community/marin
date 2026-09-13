# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.models import WordLevel

from experiments.post_training.math_eval.rendering import render_serving_response


def byte_decoder():
    tokens = ["A", "Ã", "©", "Ġ", "!", "ï", "¿", "½", "<eos>", "Ā", "ĉ", "Ċ"]
    decoder = Tokenizer(WordLevel({token: index for index, token in enumerate(tokens)}, unk_token="A"))
    decoder.decoder = ByteLevel()
    decoder.add_special_tokens([AddedToken("<eos>", special=True)])
    return decoder


@pytest.mark.parametrize(
    "prompt,tokens,expected",
    [
        ([0], [3, 0, 4, 8], " A!"),
        ([0], [1], ""),
        ([0], [1, 8], ""),
        ([0], [1, 2, 8], "é"),
        ([0], [5, 6, 7, 0, 8], "\ufffdA"),
        ([1], [2, 8], "é"),
        ([0], [9, 10, 11, 8], "\x00\t\n"),
        ([0], [8], ""),
        ([0], [], ""),
    ],
)
def test_incremental_rendering_preserves_complete_bytes_and_buffers_partial_utf8(prompt, tokens, expected):
    decoder = byte_decoder()
    assert render_serving_response(decoder, prompt, tokens) == expected


def test_truncated_utf8_is_not_a_whole_decode_replacement_character():
    decoder = byte_decoder()
    assert decoder.decode([0, 1], skip_special_tokens=True) == "A\ufffd"
    assert render_serving_response(decoder, [0], [0, 1]) == "A"
    assert render_serving_response(decoder, [0], [0, 5, 6, 7, 0]) == "A\ufffdA"
