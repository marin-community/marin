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

"""Tests for AST-based mutation (tree diffusion forward process)."""

import ast
import random

import pytest

from experiments.kelp.tree.mutation import (
    Mutation,
    _linecol_to_offset,
    _node_source_span,
    corrupt_program,
    random_mutation,
)
from experiments.kelp.tree.subtree_bank import SubtreeBank


CORPUS = [
    """\
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b
""",
    """\
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
""",
    """\
def greet(name):
    msg = f"Hello, {name}!"
    print(msg)
    return msg
""",
    """\
def add(a, b):
    return a + b
""",
    """\
def maximum(lst):
    result = lst[0]
    for x in lst[1:]:
        if x > result:
            result = x
    return result
""",
    """\
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
""",
]


@pytest.fixture
def bank():
    return SubtreeBank.from_corpus(CORPUS)


def test_linecol_to_offset_first_line():
    source = "hello world\nsecond line\n"
    assert _linecol_to_offset(source, 1, 0) == 0
    assert _linecol_to_offset(source, 1, 6) == 6


def test_linecol_to_offset_second_line():
    source = "hello world\nsecond line\n"
    assert _linecol_to_offset(source, 2, 0) == 12
    assert _linecol_to_offset(source, 2, 7) == 19


def test_node_source_span():
    source = "x = 1 + 2\n"
    tree = ast.parse(source)
    # The BinOp node should span "1 + 2".
    assign = tree.body[0]
    binop = assign.value
    span = _node_source_span(source, binop)
    assert span is not None
    start, end = span
    assert source[start:end] == "1 + 2"


def test_mutation_apply():
    m = Mutation(start=4, end=9, replacement="world", node_type="Name", original="hello")
    result = m.apply("say hello there")
    assert result == "say world there"


def test_random_mutation_produces_valid_python(bank):
    source = CORPUS[0]  # fibonacci
    rng = random.Random(42)

    mutation = random_mutation(source, bank, rng=rng)
    assert mutation is not None

    mutated = mutation.apply(source)
    # The result must be valid Python.
    try:
        ast.parse(mutated)
    except SyntaxError:
        pytest.fail(f"Mutation produced invalid Python:\n{mutated}")


def test_random_mutation_changes_source(bank):
    source = CORPUS[0]  # fibonacci
    rng = random.Random(42)

    mutation = random_mutation(source, bank, rng=rng)
    assert mutation is not None
    assert mutation.replacement != mutation.original


def test_random_mutation_returns_none_for_invalid_source(bank):
    mutation = random_mutation("this is not python{{{", bank)
    assert mutation is None


def test_random_mutation_returns_none_for_empty_bank():
    bank = SubtreeBank()
    mutation = random_mutation("x = 1\n", bank)
    assert mutation is None


def test_random_mutation_multiple_seeds_produce_different_results(bank):
    source = CORPUS[4]  # maximum -- has multiple candidate nodes
    results = set()
    for seed in range(20):
        mutation = random_mutation(source, bank, rng=random.Random(seed))
        if mutation is not None:
            results.add((mutation.start, mutation.end, mutation.replacement))

    # With 20 different seeds, we should get at least 2 distinct mutations.
    assert len(results) >= 2, "Expected diverse mutations across seeds"


def test_corrupt_program_produces_valid_python(bank):
    source = CORPUS[0]  # fibonacci
    rng = random.Random(42)

    corrupted, mutations = corrupt_program(source, num_steps=3, bank=bank, rng=rng)

    assert len(mutations) > 0
    assert corrupted != source

    try:
        ast.parse(corrupted)
    except SyntaxError:
        pytest.fail(f"corrupt_program produced invalid Python:\n{corrupted}")


def test_corrupt_program_returns_mutations_in_order(bank):
    source = CORPUS[0]
    rng = random.Random(42)

    corrupted, mutations = corrupt_program(source, num_steps=3, bank=bank, rng=rng)

    # Replay the mutations to verify they produce the same result.
    current = source
    for m in mutations:
        current = m.apply(current)
    assert current == corrupted


def test_corrupt_program_single_step(bank):
    source = CORPUS[3]  # add -- very short
    rng = random.Random(42)

    corrupted, mutations = corrupt_program(source, num_steps=1, bank=bank, rng=rng)

    assert len(mutations) <= 1
    if mutations:
        assert corrupted != source


def test_corrupt_program_zero_steps(bank):
    source = CORPUS[0]
    corrupted, mutations = corrupt_program(source, num_steps=0, bank=bank)
    assert corrupted == source
    assert mutations == []


def test_corrupt_program_graceful_when_no_mutations_possible():
    """If the bank has no matching types, corruption returns the original."""
    bank = SubtreeBank()  # empty bank
    source = "x = 1\n"
    corrupted, mutations = corrupt_program(source, num_steps=5, bank=bank)
    assert corrupted == source
    assert mutations == []


def test_corrupt_many_programs_all_valid(bank):
    """Smoke test: corrupt every corpus program and check validity."""
    rng = random.Random(123)
    for source in CORPUS:
        corrupted, _mutations = corrupt_program(
            source, num_steps=2, bank=bank, rng=rng
        )
        try:
            ast.parse(corrupted)
        except SyntaxError:
            pytest.fail(
                f"corrupt_program produced invalid Python for:\n"
                f"--- original ---\n{source}\n"
                f"--- corrupted ---\n{corrupted}"
            )
