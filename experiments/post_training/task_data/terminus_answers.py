# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the answer text out of a reply written as a Terminus-2 agent turn.

Snowball's Stage 0 ``B`` ends most single-turn replies, after its thinking, as a
Terminus-2 turn (``{"analysis": ..., "plan": ..., "commands": [{"keystrokes": ...}],
"task_complete": ...}``) with the answer inside the prose fields or a command such as
``echo 'f(1000)' >> solution.txt`` or a heredoc (decision log 2026-10-24). The X and R
graders read answer text; this module turns such a turn into that text: its
``analysis``, ``plan`` and keystrokes in order, then what the commands write to a
solution file (a one-line ``echo``/``printf`` as ``\\boxed{...}``, a heredoc body as it
is). The graders take the last answer, so the written solution wins. A reply that is not
a Terminus turn is used as it is.

String values are read by scanning, not ``json.loads``: models write ``\\boxed`` with a
single backslash, which JSON would decode as a backspace. Only the escapes that cannot
start a LaTeX command (quote, backslash, slash, ``\\n``, ``\\t``) are decoded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

TERMINUS_FIELDS = ("analysis", "plan", "keystrokes")
_TERMINUS_MARKER = re.compile(r'"(?:commands|task_complete)"\s*:')
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "n": "\n", "t": "\t"}


class AnswerFormat(StrEnum):
    TERMINUS = "terminus"
    PLAIN = "plain"


@dataclass(frozen=True)
class AnswerText:
    text: str
    format: AnswerFormat


def json_string_spans(content: str, key: str) -> list[tuple[int, int]]:
    """Character spans of every ``"key": "<value>"`` string value in ``content``, quotes excluded.

    A value still open at the end of the content (a reply cut at the token cap) runs to
    the end.
    """
    spans = []
    for match in re.finditer(rf'"{re.escape(key)}"\s*:\s*"', content):
        start = index = match.end()
        while index < len(content) and content[index] != '"':
            index += 2 if content[index] == "\\" else 1
        spans.append((start, min(index, len(content))))
    return spans


_SOLUTION_PATH = r"\S*solution\S*"
_ECHO_WRITE = re.compile(rf"(?:echo|printf)\s+(?:-[a-zA-Z]+\s+)*(['\"]?)(.+?)\1\s*>>?\s*{_SOLUTION_PATH}")
_HEREDOC_WRITE = re.compile(
    rf"cat\s*>>?\s*{_SOLUTION_PATH}\s*<<-?\s*(['\"]?)(\w+)\1[^\n]*\n(.*?)\n\2(?:\n|$)", re.DOTALL
)


def solution_writes(keystrokes: str) -> list[str]:
    """What the commands write to a solution file, in order: echoed values boxed, heredoc bodies as they are."""
    writes = [(match.start(), match.group(3)) for match in _HEREDOC_WRITE.finditer(keystrokes)]
    for line_start, line in _lines_with_offsets(keystrokes):
        for match in _ECHO_WRITE.finditer(line):
            value = match.group(2).strip()
            writes.append((line_start + match.start(), value if "\\boxed" in value else f"\\boxed{{{value}}}"))
    return [text for _, text in sorted(writes)]


def _lines_with_offsets(text: str) -> list[tuple[int, str]]:
    lines, offset = [], 0
    for line in text.split("\n"):
        lines.append((offset, line))
        offset += len(line) + 1
    return lines


def _unescape(value: str) -> str:
    return re.sub(r'\\(["\\/nt])', lambda match: _ESCAPES[match.group(1)], value)


def answer_text(reply: str, think_close: str) -> AnswerText:
    """The text a grader should read from ``reply``: the part after the last think close,
    flattened from a Terminus-2 turn when it is one."""
    content = reply.rsplit(think_close, 1)[-1]
    if not _TERMINUS_MARKER.search(content):
        return AnswerText(content, AnswerFormat.PLAIN)
    spans = sorted((start, end) for key in TERMINUS_FIELDS for start, end in json_string_spans(content, key))
    fields = [_unescape(content[start:end]) for start, end in spans]
    keystrokes = "\n".join(_unescape(content[start:end]) for start, end in json_string_spans(content, "keystrokes"))
    return AnswerText("\n".join([*fields, *solution_writes(keystrokes)]), AnswerFormat.TERMINUS)
