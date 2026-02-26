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

"""Tests for the AST subtree bank."""

import ast
import random
import tempfile
from pathlib import Path

import pytest

from experiments.kelp.tree.subtree_bank import (
    EXPRESSION_TYPES,
    STATEMENT_TYPES,
    SubtreeBank,
    SubtreeEntry,
    count_statements,
)

SAMPLE_PROGRAMS = [
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
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""",
    """\
def count_vowels(s):
    vowels = set('aeiouAEIOU')
    return sum(1 for c in s if c in vowels)
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
    return SubtreeBank.from_corpus(SAMPLE_PROGRAMS)


def test_count_statements_simple():
    tree = ast.parse("x = 1")
    # Module body has one Assign statement.
    assert count_statements(tree) == 1


def test_count_statements_nested():
    tree = ast.parse("if True:\n    x = 1\n    y = 2\n")
    # If + Assign + Assign = 3 statements.
    assert count_statements(tree) == 3


def test_count_statements_function():
    tree = ast.parse("def f():\n    return 1\n")
    # FunctionDef + Return = 2.
    assert count_statements(tree) == 2


def test_from_corpus_builds_nonempty_bank(bank):
    assert bank.total_entries > 0
    assert len(bank.entries) > 0


def test_bank_contains_expected_types(bank):
    # Our sample programs should produce at least some statements and expressions.
    type_names = set(bank.node_types)
    assert type_names & STATEMENT_TYPES, "Expected some statement types"
    assert type_names & EXPRESSION_TYPES, "Expected some expression types"


def test_bank_has_if_and_for(bank):
    assert bank.has_type("If"), "Should extract If statements"
    assert bank.has_type("For"), "Should extract For loops"


def test_bank_has_return(bank):
    assert bank.has_type("Return"), "Should extract Return statements"


def test_bank_has_call(bank):
    assert bank.has_type("Call"), "Should extract Call expressions"


def test_sample_returns_entry(bank):
    rng = random.Random(42)
    entry = bank.sample("If", rng)
    assert entry is not None
    assert entry.node_type == "If"
    assert len(entry.source) > 0


def test_sample_nonexistent_type_returns_none(bank):
    rng = random.Random(42)
    assert bank.sample("NonexistentType", rng) is None


def test_sample_with_size_respects_limit(bank):
    rng = random.Random(42)
    entry = bank.sample_with_size("If", max_stmts=2, rng=rng)
    if entry is not None:
        assert entry.stmt_count <= 2


def test_all_entries_are_parseable(bank):
    """Every extracted subtree should parse back to valid Python."""
    for node_type, entries in bank.entries.items():
        for entry in entries:
            if node_type in STATEMENT_TYPES:
                try:
                    ast.parse(entry.source)
                except SyntaxError:
                    pytest.fail(f"Statement entry failed to parse: " f"type={node_type}, source={entry.source!r}")
            else:
                try:
                    ast.parse(f"__x = {entry.source}")
                except SyntaxError:
                    pytest.fail(f"Expression entry failed to parse: " f"type={node_type}, source={entry.source!r}")


def test_no_trivially_short_entries(bank):
    """Entries shorter than 5 chars should be filtered out."""
    for entries in bank.entries.values():
        for entry in entries:
            assert len(entry.source) >= 5, f"Too short: {entry.source!r}"


def test_deduplication(bank):
    """Identical subtrees from different programs should be deduplicated."""
    for node_type, entries in bank.entries.items():
        sources = [e.source for e in entries]
        assert len(sources) == len(set(sources)), (
            f"Duplicates found in {node_type}: " f"{len(sources)} entries but {len(set(sources))} unique"
        )


def test_max_entries_per_type():
    """Cap on entries per type should be respected."""
    # Generate a lot of unique programs.
    programs = [f"def f{i}(x):\n    return x + {i}\n" for i in range(200)]
    bank = SubtreeBank.from_corpus(programs, max_entries_per_type=50)
    for entries in bank.entries.values():
        assert len(entries) <= 50


def test_max_subtree_stmts():
    """Large subtrees should be excluded."""
    programs = [SAMPLE_PROGRAMS[2]]  # merge_sort has many statements.
    bank = SubtreeBank.from_corpus(programs, max_subtree_stmts=2)
    for entries in bank.entries.values():
        for entry in entries:
            assert entry.stmt_count <= 2


def test_save_and_load(bank):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bank.json"
        bank.save(path)

        loaded = SubtreeBank.load(path)
        assert loaded.total_entries == bank.total_entries
        assert set(loaded.node_types) == set(bank.node_types)

        for node_type in bank.node_types:
            orig_sources = {e.source for e in bank.entries[node_type]}
            loaded_sources = {e.source for e in loaded.entries[node_type]}
            assert orig_sources == loaded_sources


def test_from_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, program in enumerate(SAMPLE_PROGRAMS[:2]):
            (Path(tmpdir) / f"prog_{i}.py").write_text(program)

        # Also write a non-Python file that should be skipped.
        (Path(tmpdir) / "readme.txt").write_text("not python")

        paths = list(Path(tmpdir).glob("*"))
        bank = SubtreeBank.from_files(paths)
        assert bank.total_entries > 0


def test_invalid_programs_are_skipped():
    """Programs with syntax errors should be skipped, not crash."""
    programs = [
        "def good():\n    return 1\n",
        "def bad(\n",  # syntax error
        "also bad{{{",  # syntax error
        "def also_good(x):\n    return x + 1\n",
    ]
    bank = SubtreeBank.from_corpus(programs)
    assert bank.total_entries > 0


def test_summary(bank):
    summary = bank.summary()
    assert "SubtreeBank" in summary
    assert "entries" in summary


def test_add_entry():
    bank = SubtreeBank()
    entry = SubtreeEntry(source="x + y", node_type="BinOp", stmt_count=0)
    bank.add(entry)
    assert bank.total_entries == 1
    assert bank.has_type("BinOp")
    assert not bank.has_type("Call")
