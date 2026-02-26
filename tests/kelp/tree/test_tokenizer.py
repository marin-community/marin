# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

from experiments.kelp.corpus import extract_docstring
from experiments.kelp.tree.tokenizer import TreeDiffusionTokenizer


@pytest.fixture
def tok():
    return TreeDiffusionTokenizer(max_seq_len=512)


@pytest.fixture
def tok_prompt():
    return TreeDiffusionTokenizer(max_seq_len=512, prompt_tokens=True)


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


# --- Prompt token tests ---


def test_prompt_vocab_size(tok, tok_prompt):
    """prompt_tokens=True adds 2 special tokens, increasing vocab_size by 2."""
    assert tok_prompt.vocab_size == tok.vocab_size + 2
    assert tok_prompt.num_special_tokens == 5
    assert tok.num_special_tokens == 3


def test_prompt_token_ids_are_distinct(tok_prompt):
    ids = {
        tok_prompt.pad_token_id,
        tok_prompt.sos_token_id,
        tok_prompt.eos_token_id,
        tok_prompt.prompt_start_token_id,
        tok_prompt.prompt_end_token_id,
    }
    assert len(ids) == 5


def test_prompt_token_ids_not_available_without_flag(tok):
    with pytest.raises(ValueError):
        tok.prompt_start_token_id
    with pytest.raises(ValueError):
        tok.prompt_end_token_id


def test_prompt_position_offset_shifted(tok, tok_prompt):
    """Position tokens start at 5 with prompt_tokens, vs 3 without."""
    assert tok.position_token_offset == 3
    assert tok_prompt.position_token_offset == 5


def test_decode_token_prompt_specials(tok_prompt):
    assert tok_prompt.decode_token(3) == "<PROMPT_START>"
    assert tok_prompt.decode_token(4) == "<PROMPT_END>"


def test_encode_decode_roundtrip_with_prompt(tok_prompt):
    """Source encoding/decoding works the same with prompt_tokens=True."""
    source = "x = 1 + 2\n"
    ids = tok_prompt.encode_source(source)
    decoded = tok_prompt.decode_source(ids)
    assert decoded == source


def test_encode_training_example_with_prompt(tok_prompt):
    context = "x = 1\n"
    edit_pos = 4
    replacement = "2"
    prompt = "Replace the number"

    token_ids, loss_mask = tok_prompt.encode_training_example(
        context,
        edit_pos,
        replacement,
        prompt_source=prompt,
    )

    prompt_len = 1 + len(prompt) + 1  # PROMPT_START + bytes + PROMPT_END
    context_len = len(context)

    # Total length: prompt_prefix + context + SOS + POS + replacement + EOS
    expected_len = prompt_len + context_len + 1 + 1 + len(replacement) + 1
    assert len(token_ids) == expected_len
    assert len(loss_mask) == expected_len

    # Prompt + context + SOS all have loss_mask=0.
    no_loss_len = prompt_len + context_len + 1
    assert loss_mask[:no_loss_len] == [0] * no_loss_len

    # POS + replacement + EOS have loss_mask=1.
    assert loss_mask[no_loss_len:] == [1] * (1 + len(replacement) + 1)

    # First and last prompt tokens.
    assert token_ids[0] == tok_prompt.prompt_start_token_id
    assert token_ids[prompt_len - 1] == tok_prompt.prompt_end_token_id

    # EOS at end.
    assert token_ids[-1] == tok_prompt.eos_token_id


def test_encode_training_example_without_prompt_unchanged(tok_prompt):
    """When prompt_source=None, sequence is the same as legacy (just shifted offsets)."""
    context = "x = 1\n"
    edit_pos = 4
    replacement = "2"

    token_ids, loss_mask = tok_prompt.encode_training_example(context, edit_pos, replacement)

    # No prompt prefix: context(6) + SOS + POS + replacement(1) + EOS = 10
    assert len(token_ids) == 10
    assert loss_mask[:7] == [0] * 7
    assert loss_mask[7:] == [1, 1, 1]


def test_encode_training_example_prompt_requires_flag(tok):
    """Passing prompt_source to a tokenizer without prompt_tokens raises."""
    with pytest.raises(ValueError, match="prompt_tokens=True"):
        tok.encode_training_example("x = 1\n", 4, "2", prompt_source="hello")


def test_encode_prompt_prefix(tok_prompt):
    prefix = tok_prompt.encode_prompt_prefix("hello")
    assert prefix[0] == tok_prompt.prompt_start_token_id
    assert prefix[-1] == tok_prompt.prompt_end_token_id
    assert len(prefix) == 1 + len("hello") + 1


def test_encode_prompt_prefix_requires_flag(tok):
    with pytest.raises(ValueError):
        tok.encode_prompt_prefix("hello")


def test_decode_source_skips_prompt_tokens(tok_prompt):
    ids = [
        tok_prompt.prompt_start_token_id,
        tok_prompt.encode_char("p"),
        tok_prompt.prompt_end_token_id,
        tok_prompt.encode_char("x"),
    ]
    assert tok_prompt.decode_source(ids) == "px"


def test_backward_compat_legacy_tokenizer():
    """Legacy tokenizer (prompt_tokens=False) has unchanged behavior."""
    tok = TreeDiffusionTokenizer(max_seq_len=128)
    assert tok.num_special_tokens == 3
    assert tok.position_token_offset == 3
    assert tok.vocab_size == 3 + 128 + 256


# --- extract_docstring tests ---


def test_extract_docstring_present():
    source = 'def f(x):\n    """Add one."""\n    return x + 1\n'
    assert extract_docstring(source) == "Add one."


def test_extract_docstring_absent():
    source = "def f(x):\n    return x + 1\n"
    assert extract_docstring(source) is None


def test_extract_docstring_syntax_error():
    assert extract_docstring("def f({{") is None


def test_extract_docstring_no_function():
    assert extract_docstring("x = 1\n") is None
