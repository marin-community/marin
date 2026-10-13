# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract explicit Markdown answer sections from Reasoning Gym replies."""

import re

ANSWER_HEADING = re.compile(r"(?im)^#{1,6}\s*answer\s*$")
NEXT_SECTION = re.compile(r"(?m)^(?:#{1,6}\s|---\s*$|```)")
ANSWER_ASSERTION = re.compile(r"(?is)^(?:the\s+)?[^.;:!?=\n]{1,80}?\s+(?:is|equals)\s+((?:(?![.!?]\s|;).)+?)[.!?]?$")
SECOND_ASSERTION = re.compile(r"(?i)\b(?:is|was|equals)\b")
EMPHASIS = ("**", "__", "*", "_", "`")


def markdown_answer(text: str) -> str | None:
    """Return the last explicit Markdown answer section or its single asserted value."""
    headings = list(ANSWER_HEADING.finditer(text))
    if not headings:
        return None
    section = text[headings[-1].end() :].strip()
    boundary = NEXT_SECTION.search(section)
    section = section[: boundary.start()].strip() if boundary else section
    if not section:
        return None
    plain = section
    for marker in EMPHASIS:
        if plain.startswith(marker) and plain.endswith(marker) and len(plain) > 2 * len(marker):
            plain = plain[len(marker) : -len(marker)].strip()
            break
    assertion = ANSWER_ASSERTION.fullmatch(plain)
    if assertion is None:
        return plain
    value = assertion.group(1).strip()
    return plain if SECOND_ASSERTION.search(value) else value
