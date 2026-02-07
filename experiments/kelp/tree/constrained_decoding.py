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

"""Grammar-constrained decoding for tree diffusion.

Ensures that predicted edits produce syntactically valid Python. Following
the spirit of Tree Diffusion (Kapur et al., 2024), which uses an interactive
LALR parser to mask invalid tokens at each decoding step.

For Python (which has a much larger grammar than the paper's small DSLs),
we use a pragmatic multi-level approach:

1. **Structural constraints**: Enforce bracket/paren/brace balancing and
   indentation consistency during token generation.
2. **Post-hoc validation**: After generating a complete edit, validate the
   result with ast.parse() and retry if invalid.
3. **Node-type scoping**: Use the AST node type at the edit position to
   constrain what kind of code can be generated (e.g., if replacing an
   expression, only generate valid expressions).
"""

import ast
import logging

import jax.numpy as jnp
from jaxtyping import Float, Array

from experiments.kelp.tree.mutation import Mutation
from experiments.kelp.tree.tokenizer import TreeDiffusionTokenizer

logger = logging.getLogger(__name__)

# Characters that must be balanced in valid Python.
_OPEN_BRACKETS = frozenset("([{")
_CLOSE_BRACKETS = frozenset(")]}")
_BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
_CLOSE_TO_OPEN = {v: k for k, v in _BRACKET_PAIRS.items()}


def validate_edit(source: str, mutation: Mutation) -> bool:
    """Check whether applying a mutation produces valid Python.

    Args:
        source: Current program source.
        mutation: Proposed edit.

    Returns:
        True if the edited program parses successfully.
    """
    edited = mutation.apply(source)
    try:
        ast.parse(edited)
        return True
    except SyntaxError:
        return False


def brackets_balanced(text: str) -> bool:
    """Check whether brackets, parens, and braces are balanced in text."""
    stack: list[str] = []
    in_string = False
    string_char = ""

    for i, ch in enumerate(text):
        # Simple string detection (doesn't handle triple-quotes or escapes
        # perfectly, but good enough for bracket balancing).
        if ch in ('"', "'") and not in_string:
            in_string = True
            string_char = ch
        elif ch == string_char and in_string:
            in_string = False
        elif not in_string:
            if ch in _OPEN_BRACKETS:
                stack.append(ch)
            elif ch in _CLOSE_BRACKETS:
                if not stack or stack[-1] != _CLOSE_TO_OPEN[ch]:
                    return False
                stack.pop()

    return len(stack) == 0


def compute_bracket_mask(
    partial_replacement: str,
    tokenizer: TreeDiffusionTokenizer,
) -> Float[Array, "V"]:
    """Compute a soft mask that encourages bracket balancing.

    Returns a mask of shape (vocab_size,) where:
    - 1.0 for tokens that maintain or improve bracket balance
    - 0.0 for tokens that would create an unrecoverable imbalance
      (closing a bracket that was never opened)

    This is a lightweight structural constraint -- not full grammar
    enforcement, but catches the most common structural errors.
    """
    # Count current open brackets.
    stack: list[str] = []
    for ch in partial_replacement:
        if ch in _OPEN_BRACKETS:
            stack.append(ch)
        elif ch in _CLOSE_BRACKETS:
            if stack and stack[-1] == _CLOSE_TO_OPEN.get(ch):
                stack.pop()

    mask = jnp.ones(tokenizer.vocab_size, dtype=jnp.float32)

    # If no open brackets, block close-bracket characters.
    if not stack:
        for close_ch in _CLOSE_BRACKETS:
            token_id = tokenizer.encode_char(close_ch)
            mask = mask.at[token_id].set(0.0)
    else:
        # Only allow the matching close bracket (or keep generating).
        expected_close = _BRACKET_PAIRS[stack[-1]]
        for close_ch in _CLOSE_BRACKETS:
            if close_ch != expected_close:
                token_id = tokenizer.encode_char(close_ch)
                mask = mask.at[token_id].set(0.0)

    return mask


def sample_edit_with_validation(
    source: str,
    edit_position: int,
    original_span_end: int,
    replacement_tokens: list[int],
    tokenizer: TreeDiffusionTokenizer,
    max_retries: int = 5,
) -> Mutation | None:
    """Decode replacement tokens and validate the resulting edit.

    Decodes the replacement token IDs to a source string, constructs a
    Mutation, and validates that applying it produces valid Python. If
    validation fails, returns None (the caller should resample).

    Args:
        source: Current program source.
        edit_position: Character offset where the edit starts.
        original_span_end: Character offset where the original span ends.
        replacement_tokens: Token IDs for the replacement (excluding POS/EOS).
        tokenizer: Tokenizer for decoding.
        max_retries: Not used here (validation is a single check), but
            kept for API compatibility with retry-based approaches.

    Returns:
        A valid Mutation, or None if the edit produces invalid Python.
    """
    replacement_source = tokenizer.decode_source(replacement_tokens)

    mutation = Mutation(
        start=edit_position,
        end=original_span_end,
        replacement=replacement_source,
        node_type="unknown",
        original=source[edit_position:original_span_end],
    )

    if validate_edit(source, mutation):
        return mutation

    return None


def apply_bracket_constraints(
    logits: Float[Array, "V"],
    partial_replacement: str,
    tokenizer: TreeDiffusionTokenizer,
) -> Float[Array, "V"]:
    """Apply bracket-balancing constraints to logits.

    Masks out tokens that would create unrecoverable bracket imbalances
    by setting their logits to -inf.

    Args:
        logits: Raw logits from the model, shape (vocab_size,).
        partial_replacement: Tokens generated so far for this replacement.
        tokenizer: Tokenizer for mapping chars to token IDs.

    Returns:
        Constrained logits with invalid tokens set to -inf.
    """
    mask = compute_bracket_mask(partial_replacement, tokenizer)
    return jnp.where(mask > 0, logits, jnp.float32(-1e9))
