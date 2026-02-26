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

"""Tests for TreeDiff algorithm."""

import ast


from experiments.kelp.tree.tree_diff import (
    Edit,
    find_path,
    one_step_edit,
    tree_diff,
)


def test_identical_programs_no_edits():
    source = "x = 1\n"
    assert tree_diff(source, source) == []


def test_single_expression_change():
    source = "x = 1 + 2\n"
    target = "x = 3 + 4\n"
    edits = tree_diff(source, target)
    assert len(edits) >= 1

    # Applying all edits should produce valid Python.
    current = source
    for edit in edits:
        m = edit.to_mutation(current)
        current = m.apply(current)
    ast.parse(current)


def test_single_statement_replacement():
    source = "return a + b\n"
    target = "return a * b\n"
    edits = tree_diff(source, target)
    assert len(edits) >= 1
    assert edits[0].node_type in ("BinOp", "Return")


def test_if_body_change():
    source = "if x > 0:\n    return x\n"
    target = "if x > 0:\n    return -x\n"
    edits = tree_diff(source, target)
    assert len(edits) >= 1


def test_completely_different_programs():
    source = "x = 1\n"
    target = "y = 'hello'\n"
    edits = tree_diff(source, target)
    assert len(edits) >= 1


def test_function_body_difference():
    source = "def f(x):\n    return x + 1\n"
    target = "def f(x):\n    return x * 2\n"
    edits = tree_diff(source, target)
    assert len(edits) >= 1

    # The edit should target the body, not the whole function.
    for edit in edits:
        assert edit.node_type != "FunctionDef", "Should target a specific sub-expression, not the whole function"


def test_multiple_statement_differences():
    source = "a = 1\nb = 2\n"
    target = "a = 10\nb = 20\n"
    edits = tree_diff(source, target)
    assert len(edits) >= 2


def test_one_step_edit_returns_first_edit():
    source = "a = 1\nb = 2\n"
    target = "a = 10\nb = 20\n"
    mutation = one_step_edit(source, target)
    assert mutation is not None

    # Applying should produce valid Python.
    result = mutation.apply(source)
    ast.parse(result)


def test_one_step_edit_identical_returns_none():
    source = "x = 1\n"
    assert one_step_edit(source, source) is None


def test_one_step_edit_invalid_source_returns_none():
    assert one_step_edit("not valid{{{", "x = 1\n") is None


def test_find_path_simple():
    source = "x = 1 + 2\n"
    target = "x = 3 + 4\n"
    path = find_path(source, target)
    assert len(path) >= 1

    # Replaying the path should produce valid Python.
    current = source
    for mutation in path:
        current = mutation.apply(current)
    ast.parse(current)


def test_find_path_converges_to_target():
    source = "return a + b\n"
    target = "return a * b\n"
    path = find_path(source, target)
    assert len(path) >= 1

    current = source
    for mutation in path:
        current = mutation.apply(current)

    # The result should structurally match the target.
    assert ast.dump(ast.parse(current)) == ast.dump(ast.parse(target))


def test_find_path_identical_empty():
    source = "x = 1\n"
    path = find_path(source, source)
    assert path == []


def test_find_path_multi_step():
    source = "a = 1\nb = 2\nc = 3\n"
    target = "a = 10\nb = 20\nc = 30\n"
    path = find_path(source, target)
    assert len(path) >= 1

    current = source
    for mutation in path:
        current = mutation.apply(current)
    ast.parse(current)


def test_find_path_respects_max_steps():
    source = "a = 1\nb = 2\nc = 3\n"
    target = "a = 10\nb = 20\nc = 30\n"
    path = find_path(source, target, max_steps=1)
    assert len(path) <= 1


def test_edit_to_mutation_roundtrip():
    edit = Edit(
        source_start=4,
        source_end=9,
        target_fragment="world",
        node_type="Name",
        stmt_count=0,
    )
    source = "say hello there"
    mutation = edit.to_mutation(source)
    assert mutation.apply(source) == "say world there"
    assert mutation.original == "hello"


def test_tree_diff_with_real_functions():
    source = """\
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
"""
    target = """\
def fib(n):
    if n <= 0:
        return 0
    return fib(n - 1) + fib(n - 2)
"""
    edits = tree_diff(source, target)
    assert len(edits) >= 1

    # At least one edit should target the changed part (the if condition
    # or the return value inside the if).
    edit_types = {e.node_type for e in edits}
    assert edit_types & {
        "Compare",
        "Constant",
        "Return",
        "If",
    }, f"Expected edits to target changed nodes, got types: {edit_types}"


def test_tree_diff_syntax_error_returns_empty():
    assert tree_diff("invalid{{{", "x = 1\n") == []
    assert tree_diff("x = 1\n", "invalid{{{") == []


def test_find_path_function_body_converges():
    """End-to-end: corrupt a function body and find a path back."""
    source = """\
def add(a, b):
    return a + b
"""
    target = """\
def add(a, b):
    return a * b
"""
    path = find_path(source, target)
    assert len(path) >= 1

    current = source
    for mutation in path:
        current = mutation.apply(current)

    assert ast.dump(ast.parse(current)) == ast.dump(ast.parse(target))
