# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn refreshed `pulumi preview` artifacts into a drift report.

A plain `pulumi preview` compares the program against the state file and never asks the provider
what is actually deployed, so an out-of-band edit to a resource Pulumi already tracks — a
permission added to a custom role, a retargeted static IP — produces no diff. Previewing with
`--refresh` reconciles each resource against the live provider first, which covers every resource
in the stack's state and extends to new ones the moment they land there, with no per-resource
code here.

This module only renders and fingerprints; the per-stack severity and counts are parsed upstream
by `.github/actions/pulumi-preview/format_preview.py`, whose `diff.txt` is already ANSI-stripped
and has human `user:<email>` principals redacted.

`iam_scan` covers the complementary half: a refresh can only reconcile resources Pulumi already
tracks, so a live IAM binding that was never imported stays invisible to it.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

MARKER = "<!-- marin-stack-drift -->"

# format_preview.py's severity vocabulary. "none" means the refreshed preview planned no
# operations, i.e. live matches state matches program.
SEVERITY_ICON = {
    "none": "✅",
    "change": "⚠️",
    "delete": "🚨",
    "error": "❌",
}
_CLEAN_SEVERITY = "none"

# One stack's diff embedded in the issue body, capped so several stacks stay under GitHub's
# 65536-character body limit. The full plan stays in the run's uploaded artifact.
_MAX_DIFF_CHARS = 10_000


@dataclass(frozen=True)
class StackDrift:
    """One stack's refreshed-preview outcome."""

    stack: str
    severity: str
    counts: dict[str, int]
    diff: str

    @property
    def clean(self) -> bool:
        return self.severity == _CLEAN_SEVERITY


def load_stack_drifts(previews_dir: Path) -> list[StackDrift]:
    """Read every `meta.json`/`diff.txt` pair the preview action uploaded, sorted by stack."""
    drifts: list[StackDrift] = []
    for meta_path in sorted(previews_dir.glob("**/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        diff_path = meta_path.with_name("diff.txt")
        diff = diff_path.read_text(encoding="utf-8").strip() if diff_path.is_file() else ""
        drifts.append(
            StackDrift(
                stack=meta["stack"],
                severity=meta.get("severity", "error"),
                counts={key: meta.get(key, 0) for key in ("create", "update", "delete", "replace", "import")},
                diff=diff,
            )
        )
    return drifts


def fingerprint(drifts: list[StackDrift]) -> str:
    """Hash the per-stack severity and operation counts, deliberately not the diff text.

    Refreshed diffs carry provider-side churn (an `etag` changes whenever any binding on the
    same resource changes), so hashing the text would re-notify on drift that has not actually
    moved. Counts change whenever the shape of the drift changes, which is the signal worth
    interrupting someone for.
    """
    canonical = "\n".join(
        f"{d.stack}|{d.severity}|" + ",".join(f"{k}={d.counts.get(k, 0)}" for k in sorted(d.counts))
        for d in sorted(drifts, key=lambda d: d.stack)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _summarize(counts: dict[str, int]) -> str:
    parts = [f"{value} to {verb}" for verb, value in sorted(counts.items()) if value]
    return ", ".join(parts) if parts else "no changes"


def _truncate(diff: str, run_url: str | None) -> str:
    if len(diff) <= _MAX_DIFF_CHARS:
        return diff
    omitted = len(diff) - _MAX_DIFF_CHARS
    where = f" — full plan in the `iac-preview-*` artifact on {run_url}" if run_url else ""
    return f"{diff[:_MAX_DIFF_CHARS]}\n... ({omitted} more characters truncated{where})"


def render_markdown(drifts: list[StackDrift], *, run_url: str | None) -> str:
    """The tracking-issue body. Callers pass this only when some stack is not clean."""
    lines = [
        MARKER,
        f"<!-- fingerprint:{fingerprint(drifts)} -->",
        "# Pulumi stack drift",
        "",
        "A `pulumi preview --refresh` reconciled each stack against its live provider and found "
        "resources whose deployed state no longer matches Pulumi. Unlike the PR preview, this "
        "compares against the cloud rather than only against the state file, so these are "
        "out-of-band changes. Re-apply with `pulumi up`, or update the program if the live value "
        "is the intended one.",
        "",
    ]

    for drift in sorted(drifts, key=lambda d: d.stack):
        icon = SEVERITY_ICON.get(drift.severity, "❌")
        lines += [f"- {icon} `{drift.stack}` — {_summarize(drift.counts)}"]
    lines += [""]

    for drift in sorted(drifts, key=lambda d: d.stack):
        if drift.clean or not drift.diff:
            continue
        lines += [f"## `{drift.stack}`", "", "```diff", _truncate(drift.diff, run_url), "```", ""]

    if run_url:
        lines += [f"_Refresh run: {run_url}_"]
    return "\n".join(lines).rstrip() + "\n"
