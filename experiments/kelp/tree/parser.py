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

"""Tree-sitter integration for Python AST parsing."""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

logger = logging.getLogger(__name__)

# Lazy import tree-sitter to avoid import errors when not installed
_TREE_SITTER_AVAILABLE = None


def _check_tree_sitter() -> bool:
    """Check if tree-sitter and tree-sitter-python are available."""
    global _TREE_SITTER_AVAILABLE
    if _TREE_SITTER_AVAILABLE is None:
        try:
            import tree_sitter
            import tree_sitter_python

            _TREE_SITTER_AVAILABLE = True
        except ImportError:
            _TREE_SITTER_AVAILABLE = False
            logger.warning(
                "tree-sitter or tree-sitter-python not installed. "
                "Install with: pip install tree-sitter tree-sitter-python"
            )
    return _TREE_SITTER_AVAILABLE


@lru_cache(maxsize=1)
def get_python_parser():
    """Get a cached Python tree-sitter parser.

    Returns:
        tree_sitter.Parser configured for Python.

    Raises:
        ImportError: If tree-sitter is not available.
    """
    if not _check_tree_sitter():
        raise ImportError("tree-sitter not available")

    import tree_sitter
    import tree_sitter_python

    parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_python.language()))
    return parser


@lru_cache(maxsize=1)
def get_python_language():
    """Get the Python tree-sitter language.

    Returns:
        tree_sitter.Language for Python.
    """
    if not _check_tree_sitter():
        raise ImportError("tree-sitter not available")

    import tree_sitter
    import tree_sitter_python

    return tree_sitter.Language(tree_sitter_python.language())


@dataclass
class ParseResult:
    """Result of parsing Python code."""

    tree: object  # tree_sitter.Tree
    source: bytes
    is_valid: bool
    error_nodes: list[tuple[int, int, int, int]]  # (start_row, start_col, end_row, end_col)

    @property
    def root(self):
        """Get the root node of the parse tree."""
        return self.tree.root_node


class PythonTreeParser:
    """Parser for Python code using tree-sitter.

    Provides methods for parsing Python code into ASTs and checking validity.
    """

    def __init__(self):
        self.parser = get_python_parser()

    def parse(self, code: str) -> ParseResult:
        """Parse Python code into a tree-sitter tree.

        Args:
            code: Python source code.

        Returns:
            ParseResult containing the parse tree and validity info.
        """
        source = code.encode("utf-8")
        tree = self.parser.parse(source)

        error_nodes = []
        self._find_errors(tree.root_node, error_nodes)

        return ParseResult(
            tree=tree,
            source=source,
            is_valid=len(error_nodes) == 0 and not tree.root_node.has_error,
            error_nodes=error_nodes,
        )

    def _find_errors(
        self,
        node,
        errors: list[tuple[int, int, int, int]],
    ) -> None:
        """Recursively find error nodes in the parse tree."""
        if node.type == "ERROR" or node.is_missing:
            errors.append((
                node.start_point[0],
                node.start_point[1],
                node.end_point[0],
                node.end_point[1],
            ))
        for child in node.children:
            self._find_errors(child, errors)

    def is_valid(self, code: str) -> bool:
        """Check if code is syntactically valid Python.

        Args:
            code: Python source code.

        Returns:
            True if the code parses without errors.
        """
        result = self.parse(code)
        return result.is_valid

    def parse_incremental(self, code: str, old_tree=None) -> ParseResult:
        """Parse code incrementally using an old tree.

        This is more efficient when making small changes to existing code.

        Args:
            code: Python source code.
            old_tree: Previous parse tree to use for incremental parsing.

        Returns:
            ParseResult containing the new parse tree.
        """
        source = code.encode("utf-8")
        tree = self.parser.parse(source, old_tree)

        error_nodes = []
        self._find_errors(tree.root_node, error_nodes)

        return ParseResult(
            tree=tree,
            source=source,
            is_valid=len(error_nodes) == 0 and not tree.root_node.has_error,
            error_nodes=error_nodes,
        )


def iter_tree(node, depth: int = 0) -> Iterator[tuple[object, int]]:
    """Iterate over all nodes in a tree in pre-order.

    Args:
        node: Root node to start iteration.
        depth: Current depth (for tracking).

    Yields:
        Tuples of (node, depth).
    """
    yield node, depth
    for child in node.children:
        yield from iter_tree(child, depth + 1)


def node_text(node, source: bytes) -> str:
    """Get the source text for a node.

    Args:
        node: Tree-sitter node.
        source: Original source bytes.

    Returns:
        Source text as a string.
    """
    return source[node.start_byte : node.end_byte].decode("utf-8")


def extract_functions(code: str) -> list[dict]:
    """Extract function definitions from Python code.

    Args:
        code: Python source code.

    Returns:
        List of dicts with 'name', 'docstring', 'signature', 'body', 'full_code' keys.
    """
    parser = PythonTreeParser()
    result = parser.parse(code)

    if not result.is_valid:
        return []

    functions = []
    source = result.source

    for node, _ in iter_tree(result.root):
        if node.type == "function_definition":
            func_info = _extract_function_info(node, source)
            if func_info:
                functions.append(func_info)

    return functions


def _extract_function_info(node, source: bytes) -> dict | None:
    """Extract information from a function_definition node.

    Args:
        node: tree-sitter node of type function_definition.
        source: Original source bytes.

    Returns:
        Dict with function info, or None if extraction fails.
    """
    name = None
    parameters = None
    return_type = None
    docstring = None
    body_nodes = []

    for child in node.children:
        if child.type == "name":
            name = node_text(child, source)
        elif child.type == "parameters":
            parameters = node_text(child, source)
        elif child.type == "type":
            return_type = node_text(child, source)
        elif child.type == "block":
            body_start_idx = 0
            for i, block_child in enumerate(child.children):
                if block_child.type == "expression_statement":
                    first_child = block_child.children[0] if block_child.children else None
                    if first_child and first_child.type == "string":
                        docstring = node_text(first_child, source)
                        docstring = _clean_docstring(docstring)
                        body_start_idx = i + 1
                        break
                elif block_child.type != ":":
                    break

            for i in range(body_start_idx, len(child.children)):
                body_nodes.append(child.children[i])

    if name is None or parameters is None:
        return None

    signature = f"def {name}{parameters}"
    if return_type:
        signature += f" -> {return_type}"
    signature += ":"

    body_text = ""
    if body_nodes:
        start_byte = body_nodes[0].start_byte
        end_byte = body_nodes[-1].end_byte
        body_text = source[start_byte:end_byte].decode("utf-8")

    full_code = node_text(node, source)

    return {
        "name": name,
        "docstring": docstring or "",
        "signature": signature,
        "body": body_text,
        "full_code": full_code,
    }


def _clean_docstring(docstring: str) -> str:
    """Clean a docstring by removing quotes and normalizing whitespace.

    Args:
        docstring: Raw docstring with quotes.

    Returns:
        Cleaned docstring content.
    """
    if docstring.startswith('"""') and docstring.endswith('"""'):
        docstring = docstring[3:-3]
    elif docstring.startswith("'''") and docstring.endswith("'''"):
        docstring = docstring[3:-3]
    elif docstring.startswith('"') and docstring.endswith('"'):
        docstring = docstring[1:-1]
    elif docstring.startswith("'") and docstring.endswith("'"):
        docstring = docstring[1:-1]
    return docstring.strip()
