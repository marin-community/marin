# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token counting for enforcing per-doc budgets.

Uses ``tiktoken`` (the ``cl100k_base`` encoding) when it is importable, and
otherwise falls back to a characters-divided-by-four heuristic. The heuristic is
the standard rough English approximation (~4 characters per token); it
over-counts code slightly but is conservative enough to keep drafts under
budget without pulling in a hard dependency.
"""

import logging

logger = logging.getLogger(__name__)

# Average English characters per token; used only when tiktoken is unavailable.
CHARS_PER_TOKEN = 4

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODING = None


def count_tokens(text: str) -> int:
    """Count tokens in ``text``, preferring tiktoken over the chars/4 heuristic."""
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return len(text) // CHARS_PER_TOKEN


def within_budget(text: str, budget: int) -> bool:
    """Return whether ``text`` fits within ``budget`` tokens."""
    return count_tokens(text) <= budget
