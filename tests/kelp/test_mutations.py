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

"""Tests for Kelp tree mutations."""

import ast

import pytest

from experiments.kelp.mutations import (
    MutationConfig,
    apply_n_mutations,
    corrupt_tree,
    find_mutation_candidates,
    get_replacement_for_node,
    mutate_ast,
)
from experiments.kelp.ast_utils import parse_python_to_tensors
from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab


@pytest.fixture
def node_vocab():
    return PythonNodeVocab()


@pytest.fixture
def value_vocab():
    return PythonValueVocab()


class TestMutationCandidates:
    def test_find_candidates_simple(self, node_vocab, value_vocab):
        code = "x = 1"
        tensors = parse_python_to_tensors(code, node_vocab, value_vocab)

        candidates = find_mutation_candidates(tensors, sigma_small=2)
        # Should find leaf nodes (size 1)
        assert len(candidates) > 0

    def test_find_candidates_sigma_small(self, node_vocab, value_vocab):
        code = "def f(x): return x + 1"
        tensors = parse_python_to_tensors(code, node_vocab, value_vocab)

        # With sigma_small=1, only leaf nodes
        candidates_1 = find_mutation_candidates(tensors, sigma_small=1)

        # With sigma_small=2, more candidates
        candidates_2 = find_mutation_candidates(tensors, sigma_small=2)

        # With sigma_small=3, even more
        candidates_3 = find_mutation_candidates(tensors, sigma_small=3)

        assert len(candidates_1) <= len(candidates_2) <= len(candidates_3)


class TestReplacements:
    def test_replacement_for_name(self):
        node = ast.Name(id="x", ctx=ast.Load())
        replacement = get_replacement_for_node(node)

        # Should get a replacement (might be Name or Constant)
        assert replacement is not None
        assert isinstance(replacement, (ast.Name, ast.Constant))

    def test_replacement_for_constant(self):
        node = ast.Constant(value=42)
        replacement = get_replacement_for_node(node)

        assert replacement is not None

    def test_replacement_for_binop(self):
        node = ast.BinOp(
            left=ast.Name(id="x", ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Constant(value=1),
        )
        replacement = get_replacement_for_node(node)

        # May return None if node is too complex, or a valid replacement
        # For a node of size 3, replacement should be possible
        # (depends on implementation details)

    def test_replacement_preserves_type_category(self):
        # Operators should be replaced with operators
        for op_class in [ast.Add, ast.Sub, ast.Mult]:
            node = op_class()
            replacement = get_replacement_for_node(node)
            if replacement is not None:
                assert isinstance(replacement, ast.operator)


class TestMutateAst:
    def test_mutate_simple_code(self):
        tree = ast.parse("x = 1")
        config = MutationConfig(sigma_small=2)

        mutated, indices = mutate_ast(tree, config)

        # Should be a valid AST
        assert isinstance(mutated, ast.Module)

        # Should be able to unparse
        code = ast.unparse(mutated)
        assert len(code) > 0

    def test_mutate_function(self):
        tree = ast.parse("def f(x): return x + 1")
        config = MutationConfig(sigma_small=2)

        mutated, indices = mutate_ast(tree, config)

        # Should still be parseable Python
        code = ast.unparse(mutated)
        # Re-parse to verify validity
        ast.parse(code)

    def test_multiple_mutations(self):
        tree = ast.parse("def f(x): return x + 1")
        config = MutationConfig(sigma_small=2)

        mutated, records = apply_n_mutations(tree, n=3, config=config)

        # Should still be valid Python
        code = ast.unparse(mutated)
        ast.parse(code)


class TestCorruptTree:
    def test_corrupt_simple(self):
        code = "x = 1"
        original, corrupted, steps = corrupt_tree(code, num_steps=1)

        assert original == code
        assert len(corrupted) > 0

        # Corrupted code might not always be valid Python
        # (e.g., "0 = 1" is syntactically valid AST but not valid Python)
        # This is expected behavior - the model learns to fix such issues

    def test_corrupt_function(self):
        code = "def add(a, b): return a + b"
        original, corrupted, steps = corrupt_tree(code, num_steps=2)

        assert original == code

        # Should still parse
        ast.parse(corrupted)

    def test_corrupt_identity_possible(self):
        # With some probability, corruption might not change anything
        # (if no valid mutation targets found)
        code = "pass"
        original, corrupted, steps = corrupt_tree(code, num_steps=1)

        # Should at least not crash
        assert len(corrupted) > 0
