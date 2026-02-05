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

"""Grammar constraints for logit masking during tree diffusion generation.

Provides constraint checking to ensure generated S-expressions conform to
Python grammar rules, enabling syntactically valid code generation.
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache

import jax.numpy as jnp
from jaxtyping import Array, Float

from experiments.kelp.tree.sexp import SEXP_CLOSE, SEXP_OPEN, parse_sexp, sexp_to_tokens

logger = logging.getLogger(__name__)


# Python tree-sitter node types that can have children
COMPOUND_NODE_TYPES = frozenset(
    [
        "module",
        "function_definition",
        "class_definition",
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
        "with_statement",
        "match_statement",
        "decorated_definition",
        "block",
        "expression_statement",
        "return_statement",
        "raise_statement",
        "assert_statement",
        "delete_statement",
        "global_statement",
        "nonlocal_statement",
        "import_statement",
        "import_from_statement",
        "future_import_statement",
        "parameters",
        "typed_parameter",
        "default_parameter",
        "typed_default_parameter",
        "list_splat_pattern",
        "dictionary_splat_pattern",
        "argument_list",
        "keyword_argument",
        "list",
        "set",
        "tuple",
        "dictionary",
        "pair",
        "list_comprehension",
        "set_comprehension",
        "dictionary_comprehension",
        "generator_expression",
        "for_in_clause",
        "if_clause",
        "binary_operator",
        "unary_operator",
        "not_operator",
        "boolean_operator",
        "comparison_operator",
        "lambda",
        "conditional_expression",
        "named_expression",
        "assignment",
        "augmented_assignment",
        "pattern_list",
        "subscript",
        "slice",
        "call",
        "attribute",
        "parenthesized_expression",
        "concatenated_string",
        "string",
        "interpolation",
        "format_expression",
        "await",
        "yield",
        "type",
        "dotted_name",
        "aliased_import",
        "wildcard_import",
        "except_clause",
        "finally_clause",
        "with_clause",
        "with_item",
        "case_clause",
        "case_pattern",
    ]
)

# Leaf node types (no children)
LEAF_NODE_TYPES = frozenset(
    [
        "identifier",
        "integer",
        "float",
        "string_content",
        "string_start",
        "string_end",
        "escape_sequence",
        "true",
        "false",
        "none",
        "comment",
        "line_continuation",
        "ellipsis",
        "type_conversion",
        # Operators and punctuation
        ":",
        ",",
        ".",
        ";",
        "->",
        "=",
        "+=",
        "-=",
        "*=",
        "/=",
        "//=",
        "%=",
        "**=",
        "&=",
        "|=",
        "^=",
        "<<=",
        ">>=",
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "+",
        "-",
        "*",
        "/",
        "//",
        "%",
        "**",
        "&",
        "|",
        "^",
        "~",
        "<<",
        ">>",
        "@",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "def",
        "class",
        "if",
        "elif",
        "else",
        "for",
        "while",
        "try",
        "except",
        "finally",
        "with",
        "as",
        "import",
        "from",
        "return",
        "yield",
        "raise",
        "pass",
        "break",
        "continue",
        "global",
        "nonlocal",
        "assert",
        "del",
        "lambda",
        "and",
        "or",
        "not",
        "in",
        "is",
        "await",
        "async",
        "match",
        "case",
        "_",
    ]
)


@dataclass
class GrammarState:
    """State for tracking grammar constraints during generation."""

    open_parens: int = 0
    """Number of unclosed parentheses."""

    node_stack: list[str] = field(default_factory=list)
    """Stack of open node types."""

    expect_node_type: bool = True
    """Whether we expect a node type next (after opening paren)."""

    last_token: str | None = None
    """The last token that was added."""


class PythonGrammarConstraints:
    """Grammar constraint checker for Python S-expression generation.

    Provides methods to determine valid next tokens based on the current
    partial S-expression, enabling grammar-constrained decoding.
    """

    def __init__(self, vocab: dict[str, int] | None = None):
        """Initialize grammar constraints.

        Args:
            vocab: Optional vocabulary mapping token strings to IDs.
        """
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self._all_node_types = COMPOUND_NODE_TYPES | LEAF_NODE_TYPES

    def update_state(self, state: GrammarState, token: str) -> GrammarState:
        """Update grammar state after adding a token.

        Args:
            state: Current grammar state.
            token: Token being added.

        Returns:
            Updated grammar state.
        """
        new_state = GrammarState(
            open_parens=state.open_parens,
            node_stack=state.node_stack.copy(),
            expect_node_type=state.expect_node_type,
            last_token=token,
        )

        if token == SEXP_OPEN:
            new_state.open_parens += 1
            new_state.expect_node_type = True
        elif token == SEXP_CLOSE:
            new_state.open_parens = max(0, new_state.open_parens - 1)
            if new_state.node_stack:
                new_state.node_stack.pop()
            new_state.expect_node_type = False
        elif new_state.expect_node_type:
            new_state.node_stack.append(token)
            new_state.expect_node_type = False

        return new_state

    def valid_next_tokens(self, partial_sexp: str) -> set[str]:
        """Get the set of valid next tokens for a partial S-expression.

        Args:
            partial_sexp: The S-expression generated so far.

        Returns:
            Set of valid next token strings.
        """
        tokens = sexp_to_tokens(partial_sexp) if partial_sexp else []

        state = GrammarState()
        for token in tokens:
            state = self.update_state(state, token)

        return self._valid_tokens_for_state(state)

    def _valid_tokens_for_state(self, state: GrammarState) -> set[str]:
        """Get valid tokens for a given grammar state.

        Args:
            state: Current grammar state.

        Returns:
            Set of valid token strings.
        """
        valid = set()

        if state.open_parens == 0:
            valid.add(SEXP_OPEN)
            return valid

        if state.expect_node_type:
            valid.update(self._all_node_types)
            return valid

        valid.add(SEXP_OPEN)

        if state.open_parens > 0:
            valid.add(SEXP_CLOSE)

        current_node = state.node_stack[-1] if state.node_stack else None
        if current_node in LEAF_NODE_TYPES:
            valid.discard(SEXP_OPEN)

        if self.vocab:
            for token in self.vocab:
                if token.startswith("LEAF:"):
                    valid.add(token)

        return valid

    def create_logit_mask(
        self,
        partial_sexp: str,
        vocab_size: int,
    ) -> Float[Array, "V"]:
        """Create a logit mask for valid next tokens.

        Args:
            partial_sexp: The S-expression generated so far.
            vocab_size: Size of the vocabulary.

        Returns:
            JAX array of shape (vocab_size,) with -inf for invalid tokens and 0 for valid.
        """
        valid_tokens = self.valid_next_tokens(partial_sexp)

        mask = jnp.full((vocab_size,), float("-inf"))

        if self.vocab:
            for token in valid_tokens:
                if token in self.vocab:
                    idx = self.vocab[token]
                    mask = mask.at[idx].set(0.0)
        else:
            mask = jnp.zeros((vocab_size,))

        return mask

    def is_valid_sexp(self, sexp: str) -> bool:
        """Check if an S-expression is grammatically valid.

        Args:
            sexp: S-expression string to check.

        Returns:
            True if the S-expression is valid.
        """
        node = parse_sexp(sexp)
        if node is None:
            return False

        tokens = sexp_to_tokens(sexp)
        state = GrammarState()
        for token in tokens:
            state = self.update_state(state, token)

        return state.open_parens == 0

    def is_complete(self, partial_sexp: str) -> bool:
        """Check if a partial S-expression is complete.

        Args:
            partial_sexp: Partial S-expression string.

        Returns:
            True if all parentheses are balanced.
        """
        tokens = sexp_to_tokens(partial_sexp) if partial_sexp else []
        state = GrammarState()
        for token in tokens:
            state = self.update_state(state, token)
        return state.open_parens == 0


@lru_cache(maxsize=1)
def get_default_grammar_constraints() -> PythonGrammarConstraints:
    """Get a cached default grammar constraints instance.

    Returns:
        PythonGrammarConstraints with no vocabulary.
    """
    return PythonGrammarConstraints()
