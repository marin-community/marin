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

"""Tests for Kelp S-expression serialization."""

import pytest

from experiments.kelp.tree.sexp import (
    parse_sexp,
    sexp_to_code,
    sexp_to_tokens,
    tokens_to_sexp,
)


def _tree_sitter_available():
    try:
        import tree_sitter
        import tree_sitter_python

        return True
    except ImportError:
        return False


class TestSexpTokenization:
    def test_tokenize_simple(self):
        sexp = "(module (expression_statement LEAF:x))"
        tokens = sexp_to_tokens(sexp)
        assert tokens[0] == "("
        assert tokens[1] == "module"
        assert tokens[-1] == ")"

    def test_tokens_roundtrip(self):
        sexp = "(identifier LEAF:hello)"
        tokens = sexp_to_tokens(sexp)
        reconstructed = tokens_to_sexp(tokens)
        assert "identifier" in reconstructed
        assert "LEAF:hello" in reconstructed

    def test_parse_sexp(self):
        sexp = "(function_definition (name LEAF:foo) (parameters LEAF:paren))"
        node = parse_sexp(sexp)
        assert node is not None
        assert node.node_type == "function_definition"
        assert len(node.children) == 2


@pytest.mark.skipif(not _tree_sitter_available(), reason="tree-sitter not installed")
class TestCodeToSexp:
    def test_simple_code_to_sexp(self):
        from experiments.kelp.tree.sexp import code_to_sexp

        code = "x = 1"
        sexp = code_to_sexp(code)
        assert sexp is not None
        assert sexp.startswith("(module")
        assert "identifier" in sexp or "name" in sexp

    def test_invalid_code_returns_none(self):
        from experiments.kelp.tree.sexp import code_to_sexp

        code = "def add(a, b"  # Invalid
        sexp = code_to_sexp(code)
        assert sexp is None


@pytest.mark.skipif(not _tree_sitter_available(), reason="tree-sitter not installed")
class TestRoundTrip:
    def test_simple_roundtrip(self):
        from experiments.kelp.tree.sexp import code_to_sexp

        code = "x = 1"
        sexp = code_to_sexp(code)
        assert sexp is not None
        reconstructed = sexp_to_code(sexp)
        assert "x" in reconstructed
        assert "1" in reconstructed

    def test_function_roundtrip(self):
        from experiments.kelp.tree.sexp import code_to_sexp

        code = """def add(a, b):
    return a + b"""
        sexp = code_to_sexp(code)
        assert sexp is not None
        reconstructed = sexp_to_code(sexp)
        assert "def" in reconstructed
        assert "add" in reconstructed
        assert "return" in reconstructed
