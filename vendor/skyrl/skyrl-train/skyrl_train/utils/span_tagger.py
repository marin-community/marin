"""Tag generated-token spans; the loss weights THINK tokens."""

from __future__ import annotations

import re
from typing import List

from loguru import logger

SPAN_OTHER: int = 0
SPAN_THINK: int = 1

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def tag_generated_tokens(
    generated_token_ids: List[int],
    tokenizer,
) -> List[int]:
    """Tag tokens within complete thinking spans as THINK and all other tokens as OTHER."""
    n = len(generated_token_ids)
    tags = [SPAN_OTHER] * n
    if n == 0:
        return tags

    # Decode the full generated span and each token incrementally so we can map a
    # character offset to the token that covers it. tokenizer.decode on a growing
    # prefix is monotonic in the produced string length for normal BPE tokenizers.
    try:
        full_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("span_tagger: decode failed ({}); tagging turn as OTHER", e)
        return tags

    # Build per-token cumulative char lengths via incremental decode.
    char_ends: List[int] = []
    for k in range(1, n + 1):
        try:
            prefix_text = tokenizer.decode(generated_token_ids[:k], skip_special_tokens=False)
        except Exception:  # pragma: no cover
            prefix_text = full_text[: char_ends[-1] if char_ends else 0]
        char_ends.append(len(prefix_text))

    def _char_to_token(char_idx: int) -> int:
        # First token whose decoded prefix reaches char_idx.
        for k, end in enumerate(char_ends):
            if end > char_idx:
                return k
        return n

    for m in re.finditer(re.escape(_THINK_OPEN) + r"(.*?)" + re.escape(_THINK_CLOSE), full_text, re.DOTALL):
        start_tok = _char_to_token(m.start())
        end_tok = _char_to_token(max(m.end() - 1, m.start()))
        for k in range(start_tok, min(end_tok + 1, n)):
            tags[k] = SPAN_THINK
    return tags
