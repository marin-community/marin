# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for tree diffusion tokenizer with position tokens."""

import pytest

from experiments.kelp.tree.tokenizer import TreeDiffusionTokenizer


@pytest.fixture
def tok():
    return TreeDiffusionTokenizer(max_seq_len=512)


def test_special_token_ids_are_distinct(tok):
    ids = {tok.pad_token_id, tok.sos_token_id, tok.eos_token_id}
    assert len(ids) == 3


def test_position_tokens_dont_overlap_base(tok):
    """Position tokens and base vocab tokens should not overlap."""
    pos_range = set(range(tok.position_token_offset, tok.position_token_offset + tok.num_position_tokens))
    base_range = set(range(tok.base_token_offset, tok.base_token_offset + tok.base_vocab_size))
    assert not pos_range & base_range


def test_vocab_size(tok):
    expected = 3 + tok.max_seq_len + tok.base_vocab_size
    assert tok.vocab_size == expected


def test_position_token_id_roundtrip(tok):
    for pos in [0, 1, 100, 511]:
        tid = tok.position_token_id(pos)
        assert tok.is_position_token(tid)
        assert tok.position_from_token(tid) == pos


def test_position_token_id_out_of_range(tok):
    with pytest.raises(ValueError):
        tok.position_token_id(-1)
    with pytest.raises(ValueError):
        tok.position_token_id(512)


def test_position_from_token_non_position(tok):
    with pytest.raises(ValueError):
        tok.position_from_token(tok.sos_token_id)


def test_is_position_token_false_for_specials(tok):
    assert not tok.is_position_token(tok.pad_token_id)
    assert not tok.is_position_token(tok.sos_token_id)
    assert not tok.is_position_token(tok.eos_token_id)


def test_is_position_token_false_for_base(tok):
    base_token = tok.encode_char("a")
    assert not tok.is_position_token(base_token)


def test_encode_decode_source_roundtrip(tok):
    source = "x = 1 + 2\n"
    ids = tok.encode_source(source)
    decoded = tok.decode_source(ids)
    assert decoded == source


def test_encode_source_ascii(tok):
    source = "abc"
    ids = tok.encode_source(source)
    assert len(ids) == 3
    # Each should be in the base token range.
    for tid in ids:
        assert tid >= tok.base_token_offset
        assert tid < tok.base_token_offset + tok.base_vocab_size


def test_decode_token_specials(tok):
    assert tok.decode_token(tok.pad_token_id) == ""
    assert tok.decode_token(tok.sos_token_id) == "<SOS>"
    assert tok.decode_token(tok.eos_token_id) == "<EOS>"


def test_decode_token_position(tok):
    tid = tok.position_token_id(42)
    assert tok.decode_token(tid) == "<POS 42>"


def test_decode_token_base(tok):
    tid = tok.encode_char("Z")
    assert tok.decode_token(tid) == "Z"


def test_decode_source_skips_specials(tok):
    ids = [tok.sos_token_id, tok.encode_char("h"), tok.encode_char("i"), tok.eos_token_id]
    assert tok.decode_source(ids) == "hi"


def test_encode_training_example(tok):
    context = "x = 1\n"
    edit_pos = 4  # Points to "1" in "x = 1"
    replacement = "2"

    token_ids, loss_mask = tok.encode_training_example(context, edit_pos, replacement)

    # Length: context(6) + SOS(1) + POS(1) + replacement(1) + EOS(1) = 10
    assert len(token_ids) == 10
    assert len(loss_mask) == 10

    # Context + SOS have mask=0.
    assert loss_mask[:7] == [0, 0, 0, 0, 0, 0, 0]

    # POS + replacement + EOS have mask=1.
    assert loss_mask[7:] == [1, 1, 1]

    # Check SOS and EOS are in the right places.
    assert token_ids[6] == tok.sos_token_id
    assert token_ids[7] == tok.position_token_id(edit_pos)
    assert token_ids[-1] == tok.eos_token_id


def test_char_offset_to_token_index(tok):
    source = "hello"
    assert tok.char_offset_to_token_index(source, 0) == 0
    assert tok.char_offset_to_token_index(source, 3) == 3
    assert tok.char_offset_to_token_index(source, 10) == 4  # Clamps to len-1.


def test_valid_position_mask_has_true_entries(tok):
    source = "def f(x):\n    return x + 1\n"
    mask = tok.valid_position_mask(source)

    assert len(mask) == tok.num_position_tokens
    assert any(mask), "Should have at least one valid edit position"

    # Position 0 should be valid (start of FunctionDef).
    assert mask[0] is True


def test_valid_position_mask_invalid_source(tok):
    mask = tok.valid_position_mask("not valid{{{")
    assert not any(mask)


def test_valid_position_mask_simple_assignment(tok):
    source = "x = 1 + 2\n"
    mask = tok.valid_position_mask(source)
    assert any(mask)
