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

"""Tokenizer for tree diffusion with position tokens.

Extends a base tokenizer with special tokens required by the tree diffusion
edit-prediction model:

- <PAD>: Padding token (ID 0)
- <SOS>: Start-of-sequence, marks the boundary between context and edit output
- <EOS>: End-of-sequence, marks the end of the edit
- <POS 0>, <POS 1>, ..., <POS N-1>: Position tokens that reference specific
  token positions in the context, identifying which AST node to edit

The model's output sequence format is:
  [context_tok_1, ..., context_tok_N, <SOS>, <POS k>, repl_1, ..., repl_M, <EOS>]

Following Tree Diffusion (Kapur et al., 2024), position tokens let the model
point to specific AST node boundaries in the input program.
"""

import ast
import logging
from dataclasses import dataclass

from experiments.kelp.tree.mutation import _linecol_to_offset

logger = logging.getLogger(__name__)


@dataclass
class TreeDiffusionTokenizer:
    """Tokenizer for tree diffusion edit prediction.

    Wraps a base character/byte vocabulary and adds special tokens for
    edit prediction (SOS, EOS, position tokens).

    Token ID layout:
        0              : <PAD>
        1              : <SOS>
        2              : <EOS>
        3 .. 3+N-1     : <POS 0> .. <POS N-1>  (position tokens)
        3+N .. 3+N+B-1 : base vocabulary tokens (characters/bytes)
    """

    max_seq_len: int
    """Maximum sequence length, determines the number of position tokens."""

    base_vocab_size: int = 256
    """Size of the base character vocabulary (before special tokens)."""

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def sos_token_id(self) -> int:
        return 1

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def num_position_tokens(self) -> int:
        return self.max_seq_len

    @property
    def position_token_offset(self) -> int:
        """First position token ID."""
        return 3

    @property
    def base_token_offset(self) -> int:
        """First base vocabulary token ID."""
        return 3 + self.num_position_tokens

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including all special tokens."""
        return self.base_token_offset + self.base_vocab_size

    def position_token_id(self, pos: int) -> int:
        """Get the token ID for <POS pos>."""
        if pos < 0 or pos >= self.num_position_tokens:
            raise ValueError(f"Position {pos} out of range [0, {self.num_position_tokens})")
        return self.position_token_offset + pos

    def is_position_token(self, token_id: int) -> bool:
        """Check if a token ID is a position token."""
        return self.position_token_offset <= token_id < self.position_token_offset + self.num_position_tokens

    def position_from_token(self, token_id: int) -> int:
        """Extract the position index from a position token ID.

        Raises ValueError if the token is not a position token.
        """
        if not self.is_position_token(token_id):
            raise ValueError(f"Token {token_id} is not a position token")
        return token_id - self.position_token_offset

    def encode_char(self, char: str) -> int:
        """Encode a single character to a base vocabulary token ID."""
        code = ord(char)
        if code < self.base_vocab_size:
            return self.base_token_offset + code
        return self.base_token_offset  # Map out-of-range to 0th base token.

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to its string representation."""
        if token_id == self.pad_token_id:
            return ""
        if token_id == self.sos_token_id:
            return "<SOS>"
        if token_id == self.eos_token_id:
            return "<EOS>"
        if self.is_position_token(token_id):
            pos = self.position_from_token(token_id)
            return f"<POS {pos}>"
        base_idx = token_id - self.base_token_offset
        if 0 <= base_idx < self.base_vocab_size:
            return chr(base_idx)
        return "?"

    def encode_source(self, source: str) -> list[int]:
        """Encode a Python source string to token IDs.

        Returns a list of base vocabulary token IDs (no special tokens).
        """
        return [self.encode_char(c) for c in source]

    def decode_source(self, token_ids: list[int]) -> str:
        """Decode base vocabulary token IDs back to a source string.

        Skips special tokens (PAD, SOS, EOS, POS).
        """
        chars = []
        for tid in token_ids:
            if tid == self.pad_token_id:
                continue
            if tid in (self.sos_token_id, self.eos_token_id):
                continue
            if self.is_position_token(tid):
                continue
            base_idx = tid - self.base_token_offset
            if 0 <= base_idx < self.base_vocab_size:
                chars.append(chr(base_idx))
        return "".join(chars)

    def encode_training_example(
        self,
        context_source: str,
        edit_position_token_idx: int,
        replacement_source: str,
    ) -> tuple[list[int], list[int]]:
        """Encode a complete training example.

        Args:
            context_source: The current (corrupted) program source.
            edit_position_token_idx: Token index in the context where the
                edit starts (will be encoded as a <POS> token).
            replacement_source: The replacement source code string.

        Returns:
            Tuple of (token_ids, loss_mask) where:
            - token_ids: Full sequence [context, SOS, POS, replacement, EOS]
            - loss_mask: 0 for context+SOS, 1 for POS+replacement+EOS
        """
        context_tokens = self.encode_source(context_source)
        replacement_tokens = self.encode_source(replacement_source)
        pos_token = self.position_token_id(edit_position_token_idx)

        token_ids = context_tokens + [self.sos_token_id, pos_token] + replacement_tokens + [self.eos_token_id]

        loss_mask = (
            [0] * len(context_tokens)  # Context: no loss.
            + [0]  # SOS: no loss.
            + [1]  # POS: loss.
            + [1] * len(replacement_tokens)  # Replacement: loss.
            + [1]  # EOS: loss.
        )

        return token_ids, loss_mask

    def char_offset_to_token_index(
        self,
        source: str,
        char_offset: int,
    ) -> int:
        """Convert a character offset in source to a token index.

        For this byte-level tokenizer, the mapping is 1:1 -- character i
        maps to token i. For subword tokenizers, this would need a mapping
        table.
        """
        return min(char_offset, len(source) - 1)

    def valid_position_mask(
        self,
        source: str,
        max_edit_stmts: int = 3,
    ) -> list[bool]:
        """Compute which position tokens are valid edit points.

        Returns a boolean list of length num_position_tokens where True
        means that token position corresponds to the start of an AST node
        that can be edited.

        Args:
            source: Python source code (already tokenized as context).
            max_edit_stmts: Maximum statement count for editable nodes.

        Returns:
            Boolean mask over position token indices.
        """
        from experiments.kelp.tree.subtree_bank import (
            EXTRACTABLE_TYPES,
            count_statements,
        )

        mask = [False] * self.num_position_tokens
        num_context_tokens = len(source)

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return mask

        for node in ast.walk(tree):
            type_name = type(node).__name__
            if type_name not in EXTRACTABLE_TYPES:
                continue
            if not hasattr(node, "lineno") or node.end_lineno is None:
                continue

            stmt_count = count_statements(node)
            if stmt_count > max_edit_stmts:
                continue

            char_offset = _linecol_to_offset(source, node.lineno, node.col_offset)
            token_idx = self.char_offset_to_token_index(source, char_offset)

            if 0 <= token_idx < min(num_context_tokens, self.num_position_tokens):
                mask[token_idx] = True

        return mask
