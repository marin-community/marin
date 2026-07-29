#!/usr/bin/env python3
"""Format `pulumi preview --diff` text for a GitHub ```diff PR comment.

Moves Pulumi's indented ``+/-/~`` markers to column 0 so GitHub's diff
highlighter applies (``~`` becomes ``!``). Also parses change counts into
meta.json for the aggregate comment's status icons.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_NO_CHANGES_RE = re.compile(r"Resources:\s*no changes", re.IGNORECASE)
_COUNT_RE = re.compile(
    r"(\d+)\s+to\s+(create|update|delete|replace|import)",
    re.IGNORECASE,
)
_UNCHANGED_RE = re.compile(r"(\d+)\s+unchanged", re.IGNORECASE)


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def transform_line(line: str) -> str:
    """Put +/-/! at column 0 when the first non-space char is +/-/~."""
    if not line or line[0] not in " \t":
        return line
    stripped = line.lstrip(" \t")
    if not stripped:
        return line
    marker = stripped[0]
    if marker == "-":
        return "-" + line[1:]
    if marker == "+":
        return "+" + line[1:]
    if marker == "~":
        return "!" + line[1:]
    return line


def transform_preview(text: str) -> str:
    return "\n".join(transform_line(line) for line in text.splitlines())


def parse_counts(text: str) -> dict[str, int]:
    counts = {
        "create": 0,
        "update": 0,
        "delete": 0,
        "replace": 0,
        "import": 0,
        "same": 0,
    }
    if _NO_CHANGES_RE.search(text):
        return counts
    for match in _COUNT_RE.finditer(text):
        counts[match.group(2).lower()] = int(match.group(1))
    unchanged = _UNCHANGED_RE.findall(text)
    if unchanged:
        counts["same"] = int(unchanged[-1])
    return counts


def severity(ok: bool, counts: dict[str, int]) -> str:
    if not ok:
        return "error"
    if counts["delete"] > 0:
        return "delete"
    if counts["create"] + counts["update"] + counts["replace"] + counts["import"] > 0:
        return "change"
    return "none"


def has_resource_changes(counts: dict[str, int]) -> bool:
    return (
        counts["create"]
        + counts["update"]
        + counts["delete"]
        + counts["replace"]
        + counts["import"]
    ) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", required=True)
    parser.add_argument(
        "--ok",
        required=True,
        choices=("true", "false"),
        help="Whether the pulumi preview step succeeded.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw preview text.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    ok = args.ok == "true"
    raw = strip_ansi(args.input.read_text(encoding="utf-8", errors="replace"))
    counts = parse_counts(raw)
    sev = severity(ok, counts)
    transformed = transform_preview(raw)

    if not ok:
        diff_body = transformed.strip() or "pulumi preview failed (no output captured)."
    elif has_resource_changes(counts):
        diff_body = transformed.strip()
    else:
        diff_body = ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "diff.txt").write_text(diff_body, encoding="utf-8")
    meta = {
        "stack": args.stack,
        "ok": ok,
        "severity": sev,
        **counts,
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
