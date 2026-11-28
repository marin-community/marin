# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native abstention parsing from NeMo Gym e597c962; function bodies unchanged."""

import re
from pathlib import Path

_DEFAULT_JUDGE_PROMPT_TEMPLATE = Path(__file__).with_name("qa_abstention_prompt.txt").read_text().removesuffix("\n")


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)


def _strip_thinking_traces(text: str) -> str:
    """Remove <think>...</think> and <thinking>...</thinking> blocks from text."""
    text = _THINK_TAG_RE.sub("", text)
    text = _THINKING_TAG_RE.sub("", text)
    # Fallback: the opening <think>/<thinking> tag may have been part of
    # the prompt template rather than the model's generation, so the text
    # starts with CoT reasoning followed by </think> without a matching
    # opening tag. Strip everything up to and including the unpaired closing tag.
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} in *text*, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1]


def normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison (SQuAD-style)."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def parse_judge_grade(judge_text: str) -> str:
    """Parse the single-letter grade (A/B/C) from the judge's response.

    Returns "A", "B", or "C". Falls back to "B" (INCORRECT) when the
    output cannot be reliably parsed.
    """
    cleaned = judge_text.strip()
    if cleaned in ("A", "B", "C"):
        return cleaned

    last_line = cleaned.rsplit("\n", 1)[-1].strip()
    for letter in ("A", "B", "C"):
        if last_line == letter:
            return letter

    for letter in ("A", "B", "C"):
        if letter in cleaned:
            return letter

    return "B"
