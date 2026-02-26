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

"""Tests for grammar-constrained decoding."""

import jax.numpy as jnp
import pytest

from experiments.kelp.tree.constrained_decoding import (
    apply_bracket_constraints,
    brackets_balanced,
    compute_bracket_mask,
    sample_edit_with_validation,
    validate_edit,
)
from experiments.kelp.tree.mutation import Mutation
from experiments.kelp.tree.tokenizer import TreeDiffusionTokenizer


@pytest.fixture
def tokenizer():
    return TreeDiffusionTokenizer(max_seq_len=64)


def test_brackets_balanced_empty():
    assert brackets_balanced("")


def test_brackets_balanced_simple():
    assert brackets_balanced("(a + b)")
    assert brackets_balanced("[1, 2, 3]")
    assert brackets_balanced("{x: y}")


def test_brackets_balanced_nested():
    assert brackets_balanced("f(g[h({x})])")


def test_brackets_unbalanced():
    assert not brackets_balanced("(a + b")
    assert not brackets_balanced("a + b)")
    assert not brackets_balanced("[1, 2)")


def test_brackets_in_strings_ignored():
    assert brackets_balanced('"(not a bracket"')
    assert brackets_balanced("'[still balanced'")


def test_validate_edit_valid():
    source = "x = 1 + 2\n"
    mutation = Mutation(start=4, end=9, replacement="3 * 4", node_type="BinOp", original="1 + 2")
    assert validate_edit(source, mutation)


def test_validate_edit_invalid():
    source = "x = 1 + 2\n"
    mutation = Mutation(start=4, end=9, replacement="if :", node_type="BinOp", original="1 + 2")
    assert not validate_edit(source, mutation)


def test_compute_bracket_mask_no_open_brackets(tokenizer):
    mask = compute_bracket_mask("", tokenizer)
    assert mask.shape == (tokenizer.vocab_size,)
    # Close brackets should be blocked.
    for ch in ")]}":
        tid = tokenizer.encode_char(ch)
        assert float(mask[tid]) == 0.0
    # Open brackets and regular chars should be allowed.
    for ch in "([{abc":
        tid = tokenizer.encode_char(ch)
        assert float(mask[tid]) == 1.0


def test_compute_bracket_mask_open_paren(tokenizer):
    mask = compute_bracket_mask("f(", tokenizer)
    # ) should be allowed (matches open paren).
    assert float(mask[tokenizer.encode_char(")")]) == 1.0
    # ] and } should be blocked (wrong bracket type).
    assert float(mask[tokenizer.encode_char("]")]) == 0.0
    assert float(mask[tokenizer.encode_char("}")]) == 0.0


def test_compute_bracket_mask_nested(tokenizer):
    mask = compute_bracket_mask("f([", tokenizer)
    # ] should be allowed (matches open bracket).
    assert float(mask[tokenizer.encode_char("]")]) == 1.0
    # ) should be blocked (wrong bracket type -- innermost is [).
    assert float(mask[tokenizer.encode_char(")")]) == 0.0


def test_apply_bracket_constraints(tokenizer):
    logits = jnp.ones(tokenizer.vocab_size)
    constrained = apply_bracket_constraints(logits, "f(x", tokenizer)

    # ) should be allowed.
    assert float(constrained[tokenizer.encode_char(")")]) == 1.0
    # ] should be masked to -inf.
    assert float(constrained[tokenizer.encode_char("]")]) < -1e8


def test_sample_edit_with_validation_valid(tokenizer):
    source = "x = 1 + 2\n"
    replacement_tokens = tokenizer.encode_source("3 * 4")

    mutation = sample_edit_with_validation(
        source=source,
        edit_position=4,
        original_span_end=9,
        replacement_tokens=replacement_tokens,
        tokenizer=tokenizer,
    )
    assert mutation is not None
    assert mutation.apply(source) == "x = 3 * 4\n"


def test_sample_edit_with_validation_invalid(tokenizer):
    source = "x = 1 + 2\n"
    replacement_tokens = tokenizer.encode_source("if :")

    mutation = sample_edit_with_validation(
        source=source,
        edit_position=4,
        original_span_end=9,
        replacement_tokens=replacement_tokens,
        tokenizer=tokenizer,
    )
    assert mutation is None
