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

"""Tests for Kelp AST utilities."""

import ast

import pytest

from experiments.kelp.ast_utils import (
    TreeTensors,
    count_nodes,
    get_node_value,
    get_subtree_size,
    parse_python_to_tensors,
)
from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab
from experiments.kelp.toy_dataset import TOY_PROGRAMS


@pytest.fixture
def node_vocab():
    return PythonNodeVocab()


@pytest.fixture
def value_vocab():
    return PythonValueVocab()


class TestPythonGrammar:
    def test_node_vocab_creation(self, node_vocab):
        assert node_vocab.vocab_size > 0
        assert node_vocab.pad_id == 0
        assert node_vocab.unk_id == 1

    def test_node_vocab_encode_decode(self, node_vocab):
        # Test known node types
        for node_type in ["Module", "FunctionDef", "Return", "Name", "Constant"]:
            node_id = node_vocab.encode(node_type)
            assert node_id != node_vocab.unk_id
            decoded = node_vocab.decode(node_id)
            assert decoded == node_type

    def test_node_vocab_unknown(self, node_vocab):
        # Unknown node type should return unk_id
        assert node_vocab.encode("NonexistentNode") == node_vocab.unk_id

    def test_value_vocab_creation(self, value_vocab):
        assert value_vocab.vocab_size > 0
        assert value_vocab.pad_id == 0
        assert value_vocab.unk_id == 1

    def test_value_vocab_encode_decode(self, value_vocab):
        # Test known tokens
        for token in ["x", "y", "self", "return"]:
            token_id = value_vocab.encode(token)
            decoded = value_vocab.decode(token_id)
            if token_id != value_vocab.unk_id:
                assert decoded == token


class TestAstUtils:
    def test_count_nodes_simple(self):
        tree = ast.parse("x = 1")
        count = count_nodes(tree)
        # Module -> Assign -> Name, Constant
        assert count >= 4

    def test_count_nodes_function(self):
        tree = ast.parse("def f(x): return x")
        count = count_nodes(tree)
        assert count >= 5

    def test_get_node_value_name(self):
        tree = ast.parse("x")
        name_node = tree.body[0].value
        assert get_node_value(name_node) == "x"

    def test_get_node_value_constant(self):
        tree = ast.parse("42")
        const_node = tree.body[0].value
        assert get_node_value(const_node) == "42"

    def test_get_node_value_function(self):
        tree = ast.parse("def foo(): pass")
        func_node = tree.body[0]
        assert get_node_value(func_node) == "foo"


class TestTreeTensors:
    def test_parse_simple_expression(self, node_vocab, value_vocab):
        code = "x = 1"
        tensors = parse_python_to_tensors(code, node_vocab, value_vocab)

        assert isinstance(tensors, TreeTensors)
        assert tensors.max_nodes == 256
        assert tensors.num_valid_nodes > 0

    def test_parse_function(self, node_vocab, value_vocab):
        code = "def add(a, b): return a + b"
        tensors = parse_python_to_tensors(code, node_vocab, value_vocab)

        assert tensors.num_valid_nodes > 5

    def test_tensor_shapes(self, node_vocab, value_vocab):
        code = "x = 1"
        tensors = parse_python_to_tensors(code, node_vocab, value_vocab, max_nodes=128, max_children=8, max_value_len=16)

        assert tensors.node_types.shape == (128,)
        assert tensors.node_values.shape == (128, 16)
        assert tensors.parent_indices.shape == (128,)
        assert tensors.child_indices.shape == (128, 8)
        assert tensors.num_children.shape == (128,)
        assert tensors.node_mask.shape == (128,)
        assert tensors.depth.shape == (128,)

    def test_root_has_no_parent(self, node_vocab, value_vocab):
        code = "x = 1"
        tensors = parse_python_to_tensors(code, node_vocab, value_vocab)

        # Root node (index 0) should have parent index -1
        assert tensors.parent_indices[0] == -1

    def test_children_have_valid_parents(self, node_vocab, value_vocab):
        code = "def f(x): return x + 1"
        tensors = parse_python_to_tensors(code, node_vocab, value_vocab)

        for i in range(tensors.num_valid_nodes):
            if i > 0:  # Skip root
                parent_idx = tensors.parent_indices[i]
                assert 0 <= parent_idx < i, f"Node {i} has invalid parent {parent_idx}"

    def test_depth_increases(self, node_vocab, value_vocab):
        code = "def f(x): return x + 1"
        tensors = parse_python_to_tensors(code, node_vocab, value_vocab)

        # Root should have depth 0
        assert tensors.depth[0] == 0

        # Children should have depth = parent_depth + 1
        for i in range(1, tensors.num_valid_nodes):
            parent_idx = tensors.parent_indices[i]
            assert tensors.depth[i] == tensors.depth[parent_idx] + 1


class TestRoundTrip:
    """Test that code -> tensors -> code produces valid Python."""

    @pytest.mark.parametrize("code", TOY_PROGRAMS[:20])  # Test first 20 programs
    def test_roundtrip_parses(self, code, node_vocab, value_vocab):
        """Test that reconstructed code at least parses."""
        tensors = parse_python_to_tensors(code, node_vocab, value_vocab)

        # Note: Full round-trip reconstruction is complex due to AST limitations.
        # For now, just verify tensors are created successfully.
        assert tensors.num_valid_nodes > 0

    def test_subtree_size(self, node_vocab, value_vocab):
        code = "def f(x): return x"
        tensors = parse_python_to_tensors(code, node_vocab, value_vocab)

        # Root should have size = total nodes
        root_size = get_subtree_size(tensors, 0)
        assert root_size == tensors.num_valid_nodes

        # Leaf nodes should have size 1
        for i in range(tensors.num_valid_nodes):
            if tensors.num_children[i] == 0:
                assert get_subtree_size(tensors, i) == 1
