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

"""S-expression serialization for tree-sitter ASTs.

Converts between tree-sitter parse trees and S-expression token sequences
that can be used for tree diffusion training.
"""

import logging
import re
from dataclasses import dataclass
from typing import Iterator

from experiments.kelp.tree.parser import PythonTreeParser, node_text

logger = logging.getLogger(__name__)

# Special tokens for S-expression format
SEXP_OPEN = "("
SEXP_CLOSE = ")"
SEXP_SPACE = " "
LEAF_PREFIX = "LEAF:"
MASK_TOKEN = "[MASK]"
PREFIX_END_TOKEN = "[PREFIX_END]"


@dataclass
class SexpToken:
    """A token in the S-expression representation."""

    text: str
    is_node_type: bool = False
    is_leaf: bool = False
    is_structural: bool = False  # ( or )


def tree_to_sexp(node, source: bytes, include_leaves: bool = True) -> str:
    """Convert a tree-sitter node to S-expression string.

    The S-expression format is:
        (node_type child1 child2 ...)

    For leaf nodes (nodes with no children), we include the source text:
        (node_type LEAF:text)

    Args:
        node: tree-sitter node to convert.
        source: Original source bytes.
        include_leaves: Whether to include leaf text content.

    Returns:
        S-expression string representation.
    """
    parts = [SEXP_OPEN, node.type]

    if node.child_count == 0:
        if include_leaves:
            text = node_text(node, source)
            escaped = _escape_leaf_text(text)
            parts.append(SEXP_SPACE)
            parts.append(LEAF_PREFIX + escaped)
    else:
        for child in node.children:
            parts.append(SEXP_SPACE)
            parts.append(tree_to_sexp(child, source, include_leaves))

    parts.append(SEXP_CLOSE)
    return "".join(parts)


def _escape_leaf_text(text: str) -> str:
    """Escape special characters in leaf text for S-expression encoding.

    Args:
        text: Raw text to escape.

    Returns:
        Escaped text safe for S-expression.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace("(", "\\(")
    text = text.replace(")", "\\)")
    text = text.replace(" ", "\\s")
    text = text.replace("\n", "\\n")
    text = text.replace("\t", "\\t")
    return text


def _unescape_leaf_text(text: str) -> str:
    """Unescape special characters in leaf text.

    Args:
        text: Escaped text.

    Returns:
        Original unescaped text.
    """
    result = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            next_char = text[i + 1]
            if next_char == "\\":
                result.append("\\")
            elif next_char == "(":
                result.append("(")
            elif next_char == ")":
                result.append(")")
            elif next_char == "s":
                result.append(" ")
            elif next_char == "n":
                result.append("\n")
            elif next_char == "t":
                result.append("\t")
            else:
                result.append(text[i : i + 2])
            i += 2
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def sexp_to_tokens(sexp: str) -> list[str]:
    """Tokenize an S-expression string.

    Args:
        sexp: S-expression string.

    Returns:
        List of tokens.
    """
    tokens = []
    current = []

    i = 0
    while i < len(sexp):
        c = sexp[i]

        if c == "(":
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append("(")
        elif c == ")":
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(")")
        elif c == " ":
            if current:
                tokens.append("".join(current))
                current = []
        elif c == "\\":
            current.append(c)
            if i + 1 < len(sexp):
                current.append(sexp[i + 1])
                i += 1
        else:
            current.append(c)

        i += 1

    if current:
        tokens.append("".join(current))

    return tokens


def tokens_to_sexp(tokens: list[str]) -> str:
    """Convert tokens back to S-expression string.

    Args:
        tokens: List of S-expression tokens.

    Returns:
        S-expression string.
    """
    result = []
    prev_was_open = False

    for token in tokens:
        if token == "(":
            result.append("(")
            prev_was_open = True
        elif token == ")":
            result.append(")")
            prev_was_open = False
        else:
            if result and not prev_was_open:
                result.append(" ")
            result.append(token)
            prev_was_open = False

    return "".join(result)


@dataclass
class SexpNode:
    """Parsed S-expression node."""

    node_type: str
    children: list["SexpNode | str"]  # Children can be SexpNodes or leaf strings


def parse_sexp(sexp: str) -> SexpNode | None:
    """Parse an S-expression string into a tree structure.

    Args:
        sexp: S-expression string.

    Returns:
        Root SexpNode, or None if parsing fails.
    """
    tokens = sexp_to_tokens(sexp)
    if not tokens:
        return None

    try:
        node, _ = _parse_sexp_tokens(tokens, 0)
        return node
    except (IndexError, ValueError) as e:
        logger.debug(f"Failed to parse S-expression: {e}")
        return None


def _parse_sexp_tokens(tokens: list[str], idx: int) -> tuple[SexpNode, int]:
    """Parse tokens starting at idx.

    Returns:
        Tuple of (parsed_node, next_idx).
    """
    if tokens[idx] != "(":
        raise ValueError(f"Expected '(' at position {idx}, got {tokens[idx]}")

    idx += 1
    if idx >= len(tokens):
        raise ValueError("Unexpected end of tokens")

    node_type = tokens[idx]
    idx += 1

    children: list[SexpNode | str] = []

    while idx < len(tokens) and tokens[idx] != ")":
        if tokens[idx] == "(":
            child, idx = _parse_sexp_tokens(tokens, idx)
            children.append(child)
        elif tokens[idx].startswith(LEAF_PREFIX):
            leaf_text = tokens[idx][len(LEAF_PREFIX) :]
            children.append(_unescape_leaf_text(leaf_text))
            idx += 1
        else:
            children.append(tokens[idx])
            idx += 1

    if idx >= len(tokens):
        raise ValueError("Unexpected end of tokens - missing ')'")

    idx += 1  # consume ')'
    return SexpNode(node_type=node_type, children=children), idx


def sexp_to_code(sexp: str) -> str:
    """Convert S-expression back to Python code.

    This reconstructs the source code from the leaf tokens.

    Args:
        sexp: S-expression string.

    Returns:
        Reconstructed Python code.
    """
    node = parse_sexp(sexp)
    if node is None:
        return ""

    leaves = []
    _collect_leaves(node, leaves)
    return "".join(leaves)


def _collect_leaves(node: SexpNode | str, leaves: list[str]) -> None:
    """Recursively collect leaf text from S-expression tree."""
    if isinstance(node, str):
        leaves.append(node)
    else:
        for child in node.children:
            _collect_leaves(child, leaves)


def code_to_sexp(code: str) -> str | None:
    """Convert Python code to S-expression.

    Args:
        code: Python source code.

    Returns:
        S-expression string, or None if parsing fails.
    """
    parser = PythonTreeParser()
    result = parser.parse(code)

    if not result.is_valid:
        return None

    return tree_to_sexp(result.root, result.source)


def create_training_example(
    docstring: str,
    signature: str,
    body: str,
    tokenizer=None,
) -> dict:
    """Create a training example for tree diffusion.

    Args:
        docstring: Function docstring.
        signature: Function signature (def name(args) -> type:).
        body: Function body.
        tokenizer: Optional tokenizer for converting to token IDs.

    Returns:
        Dict with 'prefix', 'target', 'prefix_sexp', 'target_sexp' keys.
    """
    prefix_code = f'"""{docstring}"""\n{signature}'
    full_code = f'{signature}\n    """{docstring}"""\n{body}'

    prefix_sexp = code_to_sexp(prefix_code)
    target_sexp = code_to_sexp(full_code)

    result = {
        "prefix": prefix_code,
        "target": full_code,
        "prefix_sexp": prefix_sexp,
        "target_sexp": target_sexp,
    }

    if tokenizer is not None:
        prefix_tokens = sexp_to_tokens(prefix_sexp) if prefix_sexp else []
        target_tokens = sexp_to_tokens(target_sexp) if target_sexp else []

        result["prefix_token_ids"] = tokenizer.encode(prefix_tokens)
        result["target_token_ids"] = tokenizer.encode(target_tokens)

    return result
