"""Tag exact generated-token spans for reward shaping."""

from __future__ import annotations

import re
from typing import Dict, List

from loguru import logger

# Tag constants. OTHER=0 so an all-zeros tag vector == "everything is OTHER",
# which is the safe default (Stage C/D only act on non-OTHER spans).
SPAN_OTHER: int = 0
SPAN_THINK: int = 1
SPAN_ACTION: int = 2
SPAN_EDIT: int = 3

SPAN_TAG_NAMES: Dict[int, str] = {
    SPAN_OTHER: "other",
    SPAN_THINK: "think",
    SPAN_ACTION: "action",
    SPAN_EDIT: "edit",
}

# Markers used to decide ACTION vs EDIT within an assistant turn's non-think text.
# These are intentionally broad and content-based (mirrors reward_shaping's action
# payload extraction). The token-level boundary is taken from the <think> delimiter
# tokens; ACTION-vs-EDIT is a turn-level classification applied to the post-think
# generated span (sub-think granularity for EDIT is a Stage-C/D refinement, not B).
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
# An "edit" turn is one whose generated text writes file content: heredocs, file
# write tool calls, apply_patch / str_replace style payloads. Heuristic, refined
# in Stage C against the real test-result parser.
_EDIT_PATTERN = re.compile(
    r"(<<\s*['\"]?EOF|>\s*\S+\.\w+|apply_patch|str_replace|create_file|write_file|\bcat\s*>)",
    re.IGNORECASE,
)


def tag_generated_tokens(
    generated_token_ids: List[int],
    tokenizer,
) -> List[int]:
    """Tag a single assistant turn's generated-token span.

    Strategy (exact, token-id based for the THINK boundary):
      - Decode the generated tokens once to locate <think>…</think> by string,
        then map the think character span back to a token span by decoding a
        growing prefix (monotonic, no re-tokenization of sub-strings).
      - Tokens inside <think>…</think> -> THINK.
      - Remaining generated tokens -> ACTION, upgraded to EDIT for the whole
        turn's non-think tokens if the non-think text matches the edit heuristic.

    Returns a tag list of length len(generated_token_ids).
    """
    n = len(generated_token_ids)
    tags = [SPAN_ACTION] * n
    if n == 0:
        return tags

    # Decode the full generated span and each token incrementally so we can map a
    # character offset to the token that covers it. tokenizer.decode on a growing
    # prefix is monotonic in the produced string length for normal BPE tokenizers.
    try:
        full_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("span_tagger: decode failed ({}); tagging turn as ACTION", e)
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

    # Determine non-think text (for ACTION vs EDIT) and tag THINK token spans.
    non_think_text = full_text
    for m in re.finditer(re.escape(_THINK_OPEN) + r"(.*?)" + re.escape(_THINK_CLOSE), full_text, re.DOTALL):
        start_tok = _char_to_token(m.start())
        end_tok = _char_to_token(max(m.end() - 1, m.start()))
        for k in range(start_tok, min(end_tok + 1, n)):
            tags[k] = SPAN_THINK
        non_think_text = non_think_text.replace(m.group(0), " ")

    is_edit_turn = bool(_EDIT_PATTERN.search(non_think_text))
    for k in range(n):
        if tags[k] != SPAN_THINK:
            tags[k] = SPAN_EDIT if is_edit_turn else SPAN_ACTION
    return tags
