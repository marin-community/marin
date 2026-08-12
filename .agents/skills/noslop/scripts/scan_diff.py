#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print high-signal test and prose candidates from added diff lines."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROSE_SUFFIXES = frozenset({".md", ".rst", ".txt", ".py", ".rs", ".go", ".ts", ".tsx", ".js", ".jsx"})
TEST_PATH = re.compile(r"(^|/)(tests?|test_[^/]+|[^/]+_test\.[^/]+)(/|$)")
PATTERN_CATALOGS = (
    ".agents/skills/noslop/",
    ".agents/skills/writing-style/ai-writing-donts.md",
)

PROSE_PATTERNS = (
    (
        "stock-contrast",
        re.compile(
            r"(,\s*not\s+\w+|\bnot (?:just|only)\b.{0,80}\bbut|not\b.{0,80}\bbut|more than just|"
            r"the question is no longer|the real story is|what matters (?:most )?is)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "bridge-prose",
        re.compile(
            r"\b(the main change (?:is|here)|what this means is|stepping back|"
            r"it is worth noting|importantly|notably|in this section|let'?s dive in)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ai-vocabulary",
        re.compile(
            r"\b(crucial|pivotal|groundbreaking|seamless|tapestry|testament|"
            r"interplay|landscape|delve|showcase|underscore|foster)\b",
            re.IGNORECASE,
        ),
    ),
)

TEST_PATTERNS = (
    ("weak-assertion", re.compile(r"\bassert\s+.+\s+is\s+not\s+None\b")),
    ("type-assertion", re.compile(r"\bassert\s+isinstance\(")),
    ("mock-dispatch", re.compile(r"\bassert_(?:called(?:_once)?(?:_with)?|any_call)\b")),
    ("private-state", re.compile(r"\bassert\s+[^#\n]*\._[A-Za-z]")),
    ("log-text", re.compile(r"\b(?:caplog\.text|record\.message)\b")),
)


@dataclass(frozen=True)
class AddedLine:
    path: str
    line: int
    text: str


def added_lines(base: str) -> list[AddedLine]:
    result = subprocess.run(
        ["git", "diff", "--unified=0", "--no-ext-diff", base, "--"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    path = ""
    new_line = 0
    lines: list[AddedLine] = []
    for raw in result.stdout.splitlines():
        if raw.startswith("+++ b/"):
            path = raw[6:]
            continue
        if raw.startswith("@@"):
            match = re.search(r"\+(\d+)", raw)
            assert match is not None
            new_line = int(match.group(1))
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            lines.append(AddedLine(path, new_line, raw[1:]))
            new_line += 1
        elif not raw.startswith("-") and not raw.startswith("\\"):
            new_line += 1
    return lines


def untracked_lines() -> list[AddedLine]:
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    paths = result.stdout.decode().split("\0")
    lines: list[AddedLine] = []
    for path in filter(None, paths):
        if suffix(path) not in PROSE_SUFFIXES and not TEST_PATH.search(path):
            continue
        lines.extend(AddedLine(path, index, text) for index, text in enumerate(Path(path).read_text().splitlines(), 1))
    return lines


def suffix(path: str) -> str:
    dot = path.rfind(".")
    return path[dot:] if dot >= 0 else ""


def is_pattern_catalog(path: str) -> bool:
    return any(path == candidate or path.startswith(candidate) for candidate in PATTERN_CATALOGS)


def main() -> int:
    base = sys.argv[1] if len(sys.argv) == 2 else "origin/main"
    findings: list[tuple[AddedLine, str]] = []
    for line in [*added_lines(base), *untracked_lines()]:
        if suffix(line.path) in PROSE_SUFFIXES and not is_pattern_catalog(line.path):
            findings.extend((line, name) for name, pattern in PROSE_PATTERNS if pattern.search(line.text))
        if TEST_PATH.search(line.path):
            findings.extend((line, name) for name, pattern in TEST_PATTERNS if pattern.search(line.text))

    for line, name in findings:
        print(f"{line.path}:{line.line}: {name}: {line.text.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
