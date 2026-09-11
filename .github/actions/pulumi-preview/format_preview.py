#!/usr/bin/env python3
"""Format `pulumi preview --diff` text for a GitHub ```diff PR comment.

Moves Pulumi's indented ``+/-/~`` markers to column 0 so GitHub's diff
highlighter applies (``~`` becomes ``!``). Also parses change counts into
meta.json for the aggregate comment's status icons, and redacts human
`user:<email>` principals so personal addresses never reach the public PR
comment (`serviceAccount:`/`group:`/other pseudo-principals are left alone).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_NO_CHANGES_RE = re.compile(r"Resources:\s*no changes", re.IGNORECASE)
_COUNT_RE = re.compile(
    r"(\d+)\s+to\s+(create|update|delete|replace|import)",
    re.IGNORECASE,
)
_UNCHANGED_RE = re.compile(r"(\d+)\s+unchanged", re.IGNORECASE)
_USER_MEMBER_RE = re.compile(r"user:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_REDACTED_MEMBER = "user:[redacted]"


@dataclass
class ResourceCounts:
    create: int = 0
    update: int = 0
    delete: int = 0
    replace: int = 0
    import_: int = 0
    same: int = 0

    def as_meta(self) -> dict[str, int]:
        """JSON field names match Pulumi's summary verbs (import, not import_)."""
        return {
            "create": self.create,
            "update": self.update,
            "delete": self.delete,
            "replace": self.replace,
            "import": self.import_,
            "same": self.same,
        }


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _resource_slug(identifier: str) -> str:
    """Mirrors infra/pulumi/src/iac/gcp/cloud_run.py's `resource_slug`, duplicated here
    (not imported) so this formatter stays a standalone, dependency-free script."""
    return re.sub(r"[^a-z0-9]+", "-", identifier.lower()).strip("-")


def redact_user_emails(text: str) -> str:
    """Redact human `user:<email>` principals in both their literal and Pulumi
    resource-name-slugged forms (e.g. `user:a@b.com` and `...-user-a-b-com`).
    `serviceAccount:`/`group:`/other pseudo-principals are never matched."""
    members = sorted(set(_USER_MEMBER_RE.findall(text)), key=len, reverse=True)
    for member in members:
        text = text.replace(member, _REDACTED_MEMBER)
        text = text.replace(_resource_slug(member), _resource_slug(_REDACTED_MEMBER))
    return text


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


def parse_counts(text: str) -> ResourceCounts:
    counts = ResourceCounts()
    if _NO_CHANGES_RE.search(text):
        return counts
    for match in _COUNT_RE.finditer(text):
        kind = match.group(2).lower()
        value = int(match.group(1))
        if kind == "import":
            counts.import_ = value
        else:
            setattr(counts, kind, value)
    unchanged = _UNCHANGED_RE.findall(text)
    if unchanged:
        counts.same = int(unchanged[-1])
    return counts


def severity(ok: bool, counts: ResourceCounts) -> str:
    if not ok:
        return "error"
    if counts.delete > 0:
        return "delete"
    if counts.create + counts.update + counts.replace + counts.import_ > 0:
        return "change"
    return "none"


def has_resource_changes(counts: ResourceCounts) -> bool:
    return (
        counts.create
        + counts.update
        + counts.delete
        + counts.replace
        + counts.import_
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
    raw = redact_user_emails(raw)
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
        **counts.as_meta(),
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
