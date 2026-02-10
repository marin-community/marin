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

"""Tests for e-graph-based subtree bank augmentation."""

import ast

import pytest

from experiments.kelp.tree.egraph_augmentation import (
    augment_bank_with_egraph,
    generate_expression_variants,
)
from experiments.kelp.tree.subtree_bank import SubtreeBank


@pytest.fixture
def toy_corpus():
    return [
        "def add(a, b):\n    return a + b\n",
        "def sub(a, b):\n    return a - b\n",
        "def neg(x):\n    return -x\n",
        "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n",
        "def abs_val(x):\n    if x < 0:\n        return -x\n    return x\n",
    ]


@pytest.fixture
def bank(toy_corpus):
    return SubtreeBank.from_corpus(toy_corpus)


# --- generate_expression_variants ---


def test_addition_commutativity():
    variants = generate_expression_variants("a + b")
    assert "b + a" in variants


def test_addition_generates_near_miss():
    variants = generate_expression_variants("a + b")
    assert "a - b" in variants


def test_multiplication_commutativity():
    variants = generate_expression_variants("a * b")
    assert "b * a" in variants


def test_comparison_flip():
    variants = generate_expression_variants("x < y")
    assert "y > x" in variants


def test_comparison_boundary_shift():
    variants = generate_expression_variants("x < y")
    assert "x <= y" in variants


def test_boolean_commutativity():
    variants = generate_expression_variants("a and b")
    assert "b and a" in variants


def test_boolean_op_swap():
    variants = generate_expression_variants("a and b")
    assert "a or b" in variants


def test_sign_flip_on_binop():
    variants = generate_expression_variants("a + b")
    assert "-(a + b)" in variants


def test_original_excluded():
    variants = generate_expression_variants("a + b")
    assert "a + b" not in variants


def test_all_variants_valid_python():
    exprs = ["a + b", "x < 0", "a and b", "a + b * c", "x > y"]
    for expr in exprs:
        variants = generate_expression_variants(expr)
        for v in variants:
            try:
                ast.parse(v, mode="eval")
            except SyntaxError:
                pytest.fail(f"Invalid variant {v!r} from {expr!r}")


def test_unparseable_input_returns_empty():
    assert generate_expression_variants("if x:") == []


def test_unsupported_node_returns_empty():
    # A function call is not in our PyExpr sort.
    assert generate_expression_variants("f(x)") == []


def test_max_variants_limit():
    variants = generate_expression_variants("a + b", max_variants=2)
    assert len(variants) <= 2


def test_no_duplicates():
    variants = generate_expression_variants("a + b", max_variants=20)
    assert len(variants) == len(set(variants))


# --- augment_bank_with_egraph ---


def test_augment_bank_increases_size(bank):
    augmented, added = augment_bank_with_egraph(bank)
    assert added > 0
    assert augmented.total_entries > bank.total_entries


def test_augment_bank_preserves_originals(bank):
    original_entries: dict[str, set[str]] = {}
    for node_type, entries in bank.entries.items():
        original_entries[node_type] = {e.source for e in entries}

    augmented, _ = augment_bank_with_egraph(bank)

    for node_type, sources in original_entries.items():
        augmented_sources = {e.source for e in augmented.entries.get(node_type, [])}
        for s in sources:
            assert s in augmented_sources, f"Original entry missing: {s!r}"


def test_augment_bank_only_adds_expression_types(bank):
    original_non_expr = {}
    for node_type, entries in bank.entries.items():
        if node_type not in ("BinOp", "Compare", "BoolOp", "UnaryOp"):
            original_non_expr[node_type] = len(entries)

    augmented, _ = augment_bank_with_egraph(bank)

    for node_type, count in original_non_expr.items():
        assert len(augmented.entries.get(node_type, [])) == count, (
            f"Non-expression type {node_type} should not gain entries"
        )


def test_augment_bank_all_entries_valid_python(bank):
    augmented, _ = augment_bank_with_egraph(bank)
    for node_type, entries in augmented.entries.items():
        for entry in entries:
            try:
                ast.parse(entry.source, mode="eval")
            except SyntaxError:
                try:
                    ast.parse(entry.source)
                except SyntaxError:
                    pytest.fail(f"Invalid entry [{node_type}]: {entry.source!r}")


def test_augment_bank_no_duplicates(bank):
    augmented, _ = augment_bank_with_egraph(bank)
    for node_type, entries in augmented.entries.items():
        sources = [e.source for e in entries]
        assert len(sources) == len(set(sources)), f"Duplicates in {node_type}"
