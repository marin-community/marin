# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for extracting boxed answers from LaTeX math solutions.

The grading logic (grade_answer, normalize_answer, etc.) lives in
marin.rl.environments.tinker_environments.math_grading. Do not
duplicate it here.
"""


def find_matching_brace(text: str, start: int, open_b: str = "{", close_b: str = "}") -> int | None:
    """Find position of matching closing brace. Returns None if not found."""
    if start >= len(text) or not text[start:].startswith(open_b):
        return None

    nesting = 0
    i = start
    while i < len(text):
        if text[i : i + len(open_b)] == open_b:
            nesting += 1
            i += len(open_b)
        elif text[i : i + len(close_b)] == close_b:
            nesting -= 1
            if nesting == 0:
                return i + len(close_b) - 1
            i += len(close_b)
        else:
            i += 1
    return None


def last_boxed_only_string(value):
    """Extract the last \\boxed{} content as a full string."""
    idx = value.rfind("\\boxed")
    if idx == -1:
        return value

    # Handle \\boxed content (space-separated)
    if value[idx:].startswith("\\boxed "):
        content = value[idx + 7 :]
        dollar_pos = content.find("$")
        content = content[:dollar_pos] if dollar_pos >= 0 else content
        return "\\boxed " + content

    # Handle \\boxed{content}
    brace_pos = idx + 6  # len("\\boxed")
    while brace_pos < len(value) and value[brace_pos] in " \t":
        brace_pos += 1

    if brace_pos < len(value) and value[brace_pos] == "{":
        end_pos = find_matching_brace(value, brace_pos)
        if end_pos is not None:
            return value[idx : end_pos + 1]

    return None
