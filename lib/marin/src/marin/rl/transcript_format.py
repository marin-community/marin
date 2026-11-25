# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serialize transcript document blocks (see marin.rl.transcripts) to Markdown and HTML.

A document is a sequence of tagged tuples: ("heading", level, text),
("paragraph", text), ("fields", entries), ("code", text, language) and
("details", summary, inner blocks).
"""

import html
from collections.abc import Sequence


def to_markdown(blocks: Sequence[Sequence]) -> str:
    """Serialize document blocks to Markdown."""
    lines: list[str] = []
    for block in blocks:
        match block:
            case ("heading", level, text):
                lines += [f"{'#' * level} {text}", ""]
            case ("paragraph", text):
                lines += [text, ""]
            case ("fields", entries):
                lines += [f"- **{name}:** {value}" for name, value in entries if value != ""]
                lines.append("")
            case ("code", text, language):
                lines += [f"```{language}", text, "```", ""]
            case ("details", summary, inner):
                lines += ["<details>", f"<summary>{summary}</summary>", ""]
                lines.append(to_markdown(inner))
                lines += ["</details>", ""]
    return "\n".join(lines).replace("\n\n\n", "\n\n").strip() + "\n"


def to_html(blocks: Sequence[Sequence]) -> str:
    """Serialize document blocks to a standalone HTML page."""

    def render(inner: Sequence[Sequence]) -> str:
        parts: list[str] = []
        for block in inner:
            match block:
                case ("heading", level, text):
                    parts.append(f"<h{level}>{html.escape(text)}</h{level}>")
                case ("paragraph", text):
                    parts.append(f"<p>{html.escape(text)}</p>")
                case ("fields", entries):
                    rows = "".join(
                        f"<tr><th>{html.escape(name)}</th><td>{html.escape(value)}</td></tr>" for name, value in entries
                    )
                    parts.append(f"<table>{rows}</table>")
                case ("code", text, _):
                    parts.append(f"<pre><code>{html.escape(text)}</code></pre>")
                case ("details", summary, nested):
                    parts.append(f"<details><summary>{html.escape(summary)}</summary>{render(nested)}</details>")
        return "\n".join(parts)

    return (
        '<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
        "<title>SkyRL rollout transcript</title></head>\n<body>\n" + render(blocks) + "\n</body></html>\n"
    )
