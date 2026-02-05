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

"""Tests for Kelp tree parser."""

import pytest


def _tree_sitter_available():
    try:
        import tree_sitter
        import tree_sitter_python

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _tree_sitter_available(), reason="tree-sitter not installed")
class TestPythonTreeParser:
    def test_parse_simple_function(self):
        from experiments.kelp.tree.parser import PythonTreeParser

        parser = PythonTreeParser()
        code = """
def add(a, b):
    return a + b
"""
        result = parser.parse(code)
        assert result.is_valid
        assert result.root.type == "module"

    def test_parse_invalid_code(self):
        from experiments.kelp.tree.parser import PythonTreeParser

        parser = PythonTreeParser()
        code = "def add(a, b"  # Missing closing paren
        result = parser.parse(code)
        assert not result.is_valid
        assert len(result.error_nodes) > 0

    def test_is_valid(self):
        from experiments.kelp.tree.parser import PythonTreeParser

        parser = PythonTreeParser()
        assert parser.is_valid("x = 1")
        assert not parser.is_valid("x = ")


@pytest.mark.skipif(not _tree_sitter_available(), reason="tree-sitter not installed")
class TestExtractFunctions:
    def test_extract_function_with_docstring(self):
        from experiments.kelp.tree.parser import extract_functions

        code = '''
def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"
'''
        functions = extract_functions(code)
        assert len(functions) == 1
        func = functions[0]
        assert func["name"] == "greet"
        assert func["docstring"] == "Return a greeting message."
        assert "def greet" in func["signature"]

    def test_extract_multiple_functions(self):
        from experiments.kelp.tree.parser import extract_functions

        code = '''
def add(a, b):
    """Add two numbers."""
    return a + b

def sub(a, b):
    """Subtract two numbers."""
    return a - b
'''
        functions = extract_functions(code)
        assert len(functions) == 2
        assert functions[0]["name"] == "add"
        assert functions[1]["name"] == "sub"

    def test_skip_function_without_docstring(self):
        from experiments.kelp.tree.parser import extract_functions

        code = """
def add(a, b):
    return a + b
"""
        functions = extract_functions(code)
        assert len(functions) == 0 or functions[0]["docstring"] == ""
