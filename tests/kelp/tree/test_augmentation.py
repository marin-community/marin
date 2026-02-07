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

"""Tests for subtree bank augmentation strategies."""

import ast
import random

import pytest

from experiments.kelp.tree.augmentation import (
    augment_bank,
    generate_synthetic_subtrees,
    perturb_operators,
    rename_variables,
)
from experiments.kelp.tree.subtree_bank import SubtreeBank


@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def toy_corpus():
    return [
        "def add(a, b):\n    return a + b\n",
        "def sub(a, b):\n    return a - b\n",
        "def mul(a, b):\n    return a * b\n",
        "def neg(x):\n    return -x\n",
        "def is_positive(x):\n    return x > 0\n",
        "def max_val(a, b):\n    if a > b:\n        return a\n    return b\n",
        "def abs_val(x):\n    if x < 0:\n        return -x\n    return x\n",
    ]


@pytest.fixture
def bank(toy_corpus):
    return SubtreeBank.from_corpus(toy_corpus)


# --- Variable renaming ---


def test_rename_variables_produces_valid_python(rng):
    source = "def add(a, b):\n    return a + b"
    result = rename_variables(source, rng)
    assert result is not None
    ast.parse(result)


def test_rename_variables_changes_names(rng):
    source = "def add(a, b):\n    return a + b"
    result = rename_variables(source, rng)
    assert result is not None
    assert result != source


def test_rename_variables_preserves_structure(rng):
    source = "def foo(x):\n    return x + 1"
    result = rename_variables(source, rng)
    assert result is not None
    # Should still have a function with a return and addition.
    tree = ast.parse(result)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    ret = func.body[0]
    assert isinstance(ret, ast.Return)
    assert isinstance(ret.value, ast.BinOp)
    assert isinstance(ret.value.op, ast.Add)


def test_rename_variables_no_names_returns_none(rng):
    # Only builtins and constants.
    source = "print(42)"
    result = rename_variables(source, rng)
    # "print" is protected, "42" is a constant — no names to rename.
    assert result is None


def test_rename_variables_deterministic_with_same_seed():
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    source = "def compute(x, y):\n    result = x * y\n    return result"
    r1 = rename_variables(source, rng1)
    r2 = rename_variables(source, rng2)
    assert r1 == r2


# --- Operator perturbation ---


def test_perturb_operators_arithmetic(rng):
    source = "x + y"
    # Try many times since perturbation is probabilistic.
    found_different = False
    for seed in range(50):
        r = random.Random(seed)
        result = perturb_operators(source, r, swap_prob=1.0)
        if result is not None:
            found_different = True
            ast.parse(result)
            assert result != source
            break
    assert found_different, "Expected at least one operator perturbation"


def test_perturb_operators_comparison(rng):
    source = "a > b"
    found = False
    for seed in range(50):
        r = random.Random(seed)
        result = perturb_operators(source, r, swap_prob=1.0)
        if result is not None:
            found = True
            ast.parse(result)
            break
    assert found


def test_perturb_operators_no_ops_returns_none(rng):
    source = "x = 42"
    result = perturb_operators(source, rng, swap_prob=1.0)
    assert result is None


def test_perturb_operators_valid_python(rng):
    source = "def f(a, b):\n    if a > b:\n        return a + b\n    return a - b"
    for seed in range(20):
        r = random.Random(seed)
        result = perturb_operators(source, r, swap_prob=0.5)
        if result is not None:
            ast.parse(result)


def test_perturb_operators_swap_prob_zero_returns_none(rng):
    source = "a + b"
    result = perturb_operators(source, rng, swap_prob=0.0)
    assert result is None


# --- Synthetic templates ---


def test_synthetic_subtrees_nonempty(rng):
    entries = generate_synthetic_subtrees(rng, count_per_category=10)
    assert len(entries) > 0


def test_synthetic_subtrees_valid_python(rng):
    entries = generate_synthetic_subtrees(rng, count_per_category=20)
    for entry in entries:
        try:
            ast.parse(entry.source)
        except SyntaxError:
            pytest.fail(f"Synthetic subtree is not valid Python: {entry.source!r}")


def test_synthetic_subtrees_have_correct_types(rng):
    entries = generate_synthetic_subtrees(rng, count_per_category=10)
    for entry in entries:
        assert entry.node_type in ("Return", "If", "For", "Assign", "BinOp", "Call", "Compare", "UnaryOp")


def test_synthetic_subtrees_no_duplicates(rng):
    entries = generate_synthetic_subtrees(rng, count_per_category=20)
    sources = [e.source for e in entries]
    assert len(sources) == len(set(sources))


def test_synthetic_subtrees_deterministic():
    r1 = random.Random(99)
    r2 = random.Random(99)
    e1 = generate_synthetic_subtrees(r1, count_per_category=10)
    e2 = generate_synthetic_subtrees(r2, count_per_category=10)
    assert [e.source for e in e1] == [e.source for e in e2]


# --- Bank augmentation ---


def test_augment_bank_increases_size(bank, rng):
    original_size = bank.total_entries
    augmented = augment_bank(bank, rng, n_renamed=1, n_perturbed=1, synthetic_count=10)
    assert augmented.total_entries > original_size


def test_augment_bank_preserves_originals(bank, rng):
    original_entries = {}
    for node_type, entries in bank.entries.items():
        original_entries[node_type] = {e.source for e in entries}

    augmented = augment_bank(bank, rng, n_renamed=1, n_perturbed=1, synthetic_count=5)

    for node_type, sources in original_entries.items():
        augmented_sources = {e.source for e in augmented.entries.get(node_type, [])}
        for s in sources:
            assert s in augmented_sources, f"Original entry missing: {s!r}"


def test_augment_bank_all_entries_valid_python(bank, rng):
    augmented = augment_bank(bank, rng, n_renamed=2, n_perturbed=2, synthetic_count=20)
    for node_type, entries in augmented.entries.items():
        for entry in entries:
            try:
                ast.parse(entry.source)
            except SyntaxError:
                # Expression entries need a wrapper.
                try:
                    ast.parse(f"__x = {entry.source}")
                except SyntaxError:
                    pytest.fail(f"Invalid entry [{node_type}]: {entry.source!r}")


def test_augment_bank_no_duplicates(bank, rng):
    augmented = augment_bank(bank, rng, n_renamed=2, n_perturbed=2, synthetic_count=20)
    for node_type, entries in augmented.entries.items():
        sources = [e.source for e in entries]
        assert len(sources) == len(set(sources)), f"Duplicates in {node_type}"


def test_augment_bank_zero_augmentation_preserves_bank(bank, rng):
    augmented = augment_bank(bank, rng, n_renamed=0, n_perturbed=0, synthetic_count=0)
    assert augmented.total_entries == bank.total_entries
