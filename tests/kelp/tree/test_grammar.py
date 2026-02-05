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

"""Tests for Kelp grammar constraints."""

import pytest

from experiments.kelp.tree.grammar import GrammarState, PythonGrammarConstraints


class TestGrammarState:
    def test_initial_state(self):
        state = GrammarState()
        assert state.open_parens == 0
        assert state.node_stack == []
        assert state.expect_node_type is True

    def test_update_on_open(self):
        gc = PythonGrammarConstraints()
        state = GrammarState()
        new_state = gc.update_state(state, "(")
        assert new_state.open_parens == 1
        assert new_state.expect_node_type is True

    def test_update_on_node_type(self):
        gc = PythonGrammarConstraints()
        state = GrammarState(open_parens=1, expect_node_type=True)
        new_state = gc.update_state(state, "identifier")
        assert new_state.expect_node_type is False
        assert "identifier" in new_state.node_stack

    def test_update_on_close(self):
        gc = PythonGrammarConstraints()
        state = GrammarState(open_parens=1, node_stack=["module"])
        new_state = gc.update_state(state, ")")
        assert new_state.open_parens == 0
        assert len(new_state.node_stack) == 0


class TestPythonGrammarConstraints:
    def test_valid_tokens_empty(self):
        gc = PythonGrammarConstraints()
        valid = gc.valid_next_tokens("")
        assert "(" in valid
        assert ")" not in valid

    def test_valid_tokens_after_open(self):
        gc = PythonGrammarConstraints()
        valid = gc.valid_next_tokens("(")
        assert "module" in valid
        assert "identifier" in valid

    def test_valid_tokens_after_node_type(self):
        gc = PythonGrammarConstraints()
        valid = gc.valid_next_tokens("(module")
        assert "(" in valid
        assert ")" in valid

    def test_is_complete(self):
        gc = PythonGrammarConstraints()
        assert gc.is_complete("(module)")
        assert gc.is_complete("(module (identifier LEAF:x))")
        assert not gc.is_complete("(module")
        assert not gc.is_complete("(module (identifier")

    def test_is_valid_sexp(self):
        gc = PythonGrammarConstraints()
        assert gc.is_valid_sexp("(module)")
        assert gc.is_valid_sexp("(identifier LEAF:x)")
        assert not gc.is_valid_sexp("(module")
