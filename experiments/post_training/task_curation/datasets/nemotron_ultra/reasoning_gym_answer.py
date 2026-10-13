# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract explicit Markdown answer sections from Reasoning Gym replies."""

import re

ANSWER_HEADING = re.compile(r"(?im)^#{1,6}\s*answer\s*$")
NEXT_SECTION = re.compile(r"(?m)^(?:#{1,6}\s|---\s*$|```)")
EMPHASIS = ("**", "__", "*", "_", "`")


def markdown_answer(text: str) -> str | None:
    """Return the last explicit Markdown answer section."""
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
    return plain
