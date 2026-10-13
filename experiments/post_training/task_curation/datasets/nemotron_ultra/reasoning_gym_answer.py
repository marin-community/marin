# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract explicit Markdown answer sections from Reasoning Gym replies."""

import re

ANSWER_HEADING = re.compile(r"(?im)^#{1,6}\s*answer\s*$")
NEXT_SECTION = re.compile(r"(?m)^(?:#{1,6}\s|---\s*$|```)")
NEGATION = re.compile(r"(?i)\b(?:no|not|isn't|isnt|never|neither)\b")


def markdown_answer(text: str, expected: str | None = None) -> str | None:
    """Return the last explicit Markdown answer section, or its asserted terminal reference."""
    headings = list(ANSWER_HEADING.finditer(text))
    if not headings:
        return None
    section = text[headings[-1].end() :].strip()
    boundary = NEXT_SECTION.search(section)
    section = section[: boundary.start()].strip() if boundary else section
    if expected is not None:
        match = re.search(rf"(?<!\w){re.escape(expected)}(?!\w)(?:[.!?]|[*_`])*$", section)
        if match is not None and not NEGATION.search(section[max(0, match.start() - 32) : match.start()]):
            return expected
    return section or None
