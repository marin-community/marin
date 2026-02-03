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

"""
Python AST node vocabulary for tree diffusion.

This module defines the vocabulary mapping between Python AST node types
and integer IDs for use in the tree diffusion model.
"""

import ast
from dataclasses import dataclass, field

# All Python AST node types (Python 3.11+)
# Organized by category for clarity
PYTHON_AST_NODE_TYPES: list[str] = [
    # Special tokens
    "PAD",  # Padding token
    "UNK",  # Unknown token
    # Module
    "Module",
    "Interactive",
    "Expression",
    "FunctionType",
    # Statements
    "FunctionDef",
    "AsyncFunctionDef",
    "ClassDef",
    "Return",
    "Delete",
    "Assign",
    "TypeAlias",
    "AugAssign",
    "AnnAssign",
    "For",
    "AsyncFor",
    "While",
    "If",
    "With",
    "AsyncWith",
    "Match",
    "Raise",
    "Try",
    "TryStar",
    "Assert",
    "Import",
    "ImportFrom",
    "Global",
    "Nonlocal",
    "Expr",
    "Pass",
    "Break",
    "Continue",
    # Expressions
    "BoolOp",
    "NamedExpr",
    "BinOp",
    "UnaryOp",
    "Lambda",
    "IfExp",
    "Dict",
    "Set",
    "ListComp",
    "SetComp",
    "DictComp",
    "GeneratorExp",
    "Await",
    "Yield",
    "YieldFrom",
    "Compare",
    "Call",
    "FormattedValue",
    "JoinedStr",
    "Constant",
    "Attribute",
    "Subscript",
    "Starred",
    "Name",
    "List",
    "Tuple",
    "Slice",
    # Expression context
    "Load",
    "Store",
    "Del",
    # Boolean operators
    "And",
    "Or",
    # Binary operators
    "Add",
    "Sub",
    "Mult",
    "MatMult",
    "Div",
    "Mod",
    "Pow",
    "LShift",
    "RShift",
    "BitOr",
    "BitXor",
    "BitAnd",
    "FloorDiv",
    # Unary operators
    "Invert",
    "Not",
    "UAdd",
    "USub",
    # Comparison operators
    "Eq",
    "NotEq",
    "Lt",
    "LtE",
    "Gt",
    "GtE",
    "Is",
    "IsNot",
    "In",
    "NotIn",
    # Comprehension
    "comprehension",
    # Exception handlers
    "ExceptHandler",
    # Arguments
    "arguments",
    "arg",
    "keyword",
    # Aliases
    "alias",
    # With items
    "withitem",
    # Match patterns
    "MatchValue",
    "MatchSingleton",
    "MatchSequence",
    "MatchMapping",
    "MatchClass",
    "MatchStar",
    "MatchAs",
    "MatchOr",
    "match_case",
    # Type parameters (Python 3.12+)
    "TypeVar",
    "ParamSpec",
    "TypeVarTuple",
    # Type ignore
    "type_ignore",
]


@dataclass
class PythonNodeVocab:
    """Vocabulary for Python AST node types.

    Maps between node type names (strings) and integer IDs.
    """

    node_types: list[str] = field(default_factory=lambda: list(PYTHON_AST_NODE_TYPES))
    _name_to_id: dict[str, int] = field(default_factory=dict, repr=False)
    _id_to_name: dict[int, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._name_to_id = {name: i for i, name in enumerate(self.node_types)}
        self._id_to_name = {i: name for i, name in enumerate(self.node_types)}

    @property
    def pad_id(self) -> int:
        return self._name_to_id["PAD"]

    @property
    def unk_id(self) -> int:
        return self._name_to_id["UNK"]

    @property
    def vocab_size(self) -> int:
        return len(self.node_types)

    def encode(self, node_type: str) -> int:
        """Convert a node type name to its ID."""
        return self._name_to_id.get(node_type, self.unk_id)

    def decode(self, node_id: int) -> str:
        """Convert a node ID to its type name."""
        return self._id_to_name.get(node_id, "UNK")

    def encode_node(self, node: ast.AST) -> int:
        """Get the ID for an AST node."""
        return self.encode(type(node).__name__)


# Common Python value tokens for identifiers, operators, and literals
# These are used for the node value vocabulary
COMMON_PYTHON_TOKENS: list[str] = [
    # Special tokens
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    # Common identifiers
    "self",
    "cls",
    "x",
    "y",
    "z",
    "i",
    "j",
    "k",
    "n",
    "m",
    "a",
    "b",
    "c",
    "args",
    "kwargs",
    "result",
    "value",
    "data",
    "item",
    "items",
    "key",
    "keys",
    "val",
    "values",
    "func",
    "fn",
    "f",
    "g",
    "h",
    "lst",
    "arr",
    "s",
    "t",
    "p",
    "q",
    "r",
    "idx",
    "index",
    "count",
    "total",
    "sum",
    "max",
    "min",
    "len",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sorted",
    "reversed",
    "list",
    "dict",
    "set",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "None",
    "True",
    "False",
    "print",
    "input",
    "open",
    "return",
    "if",
    "else",
    "elif",
    "for",
    "while",
    "in",
    "not",
    "and",
    "or",
    "is",
    "def",
    "class",
    "import",
    "from",
    "as",
    "with",
    "try",
    "except",
    "finally",
    "raise",
    "assert",
    "pass",
    "break",
    "continue",
    "lambda",
    "yield",
    "global",
    "nonlocal",
    "async",
    "await",
    # Common numeric literals
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "10",
    "100",
    "-1",
    "0.0",
    "1.0",
    "0.5",
    # Common string literals
    '""',
    "''",
    '" "',
    "' '",
]


@dataclass
class PythonValueVocab:
    """Vocabulary for Python value tokens (identifiers, literals, etc).

    This is a simple vocabulary for common tokens. In practice, you might
    want to use a subword tokenizer or character-level encoding for
    arbitrary identifiers and literals.
    """

    tokens: list[str] = field(default_factory=lambda: list(COMMON_PYTHON_TOKENS))
    _token_to_id: dict[str, int] = field(default_factory=dict, repr=False)
    _id_to_token: dict[int, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._token_to_id = {tok: i for i, tok in enumerate(self.tokens)}
        self._id_to_token = {i: tok for i, tok in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return self._token_to_id["<PAD>"]

    @property
    def unk_id(self) -> int:
        return self._token_to_id["<UNK>"]

    @property
    def bos_id(self) -> int:
        return self._token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self._token_to_id["<EOS>"]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, token: str) -> int:
        """Convert a token to its ID."""
        return self._token_to_id.get(token, self.unk_id)

    def decode(self, token_id: int) -> str:
        """Convert a token ID back to a string."""
        return self._id_to_token.get(token_id, "<UNK>")

    def encode_value(self, value: str, max_len: int = 32) -> list[int]:
        """Encode a value string to a list of token IDs.

        For now, we just do character-level encoding for unknown tokens.
        This is a simple fallback; a better approach would use subword tokenization.
        """
        if value in self._token_to_id:
            # Known token - pad to max_len
            ids = [self.encode(value)]
        else:
            # Character-level fallback for unknown tokens
            ids = [self.encode(c) if c in self._token_to_id else self.unk_id for c in value]

        # Pad or truncate to max_len
        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        return ids

    def decode_value(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back to a value string."""
        tokens = []
        for tid in token_ids:
            if tid == self.pad_id:
                break
            tokens.append(self.decode(tid))
        return "".join(tokens)
