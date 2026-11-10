# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Split a proposal body into its level-two sections, as Taskforge's proposal parser does."""

import re

CODE_FENCES = ("```", "~~~")


def body_sections(body: str) -> list[dict[str, str]]:
    """Each ``## `` heading of ``body`` in order with the text under it, ignoring headings inside fences."""
    sections: list[dict[str, str]] = []
    lines: list[str] = []
    fence: str | None = None
    for line in (body or "").split("\n"):
        marker = line.lstrip(" \t")[:3]
        if fence is None and marker in CODE_FENCES:
            fence = marker
        elif fence is not None and marker == fence:
            fence = None
        elif fence is None and line.startswith("## "):
            if sections:
                sections[-1]["text"] = "\n".join(lines).strip()
            sections.append({"heading": line[3:].strip(), "text": ""})
            lines = []
            continue
        lines.append(line)
    if sections:
        sections[-1]["text"] = "\n".join(lines).strip()
    elif body and body.strip():
        sections.append({"heading": "", "text": body.strip()})
    return sections


def first_sentence(text: str, limit: int = 280) -> str:
    """The first sentence of Markdown ``text`` as plain text, at most ``limit`` characters.

    A sentence must run at least 20 characters before its closing punctuation, so an abbreviation or a short
    lead-in ("e.g.", "Step 1.") joins the text after it instead of ending the summary.
    """
    plain = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text or "")
    plain = re.sub(r"[*_`#>|]", "", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    match = re.match(r"^(.{20,}?[.!?])\s", plain)
    sentence = match.group(1) if match else plain
    return sentence if len(sentence) <= limit else sentence[: limit - 1] + "…"
