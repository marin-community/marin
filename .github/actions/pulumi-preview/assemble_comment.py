#!/usr/bin/env python3
"""Assemble one IaC preview PR comment body from per-stack preview artifacts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

MARKER = "<!-- iac-preview -->"
METADATA_PREFIX = "<!-- iac-preview-metadata: "
METADATA_SUFFIX = " -->"
_HEAD_SHA_RE = re.compile(r"[0-9a-f]{40}")

# GitHub rejects an issue/PR comment body over 65536 characters. This repo previews at
# most 5 stacks in one comment, so capping each stack's embedded diff at 10k chars keeps
# the total body safely under the limit even if every stack's diff were maxed out at
# once; the full plan always remains available in the uploaded artifact.
_MAX_DIFF_CHARS = 10_000

_SEVERITY_ICON = {
    "none": "✅",
    "change": "⚠️",
    "delete": "🚨",
    "error": "❌",
}


@dataclass(frozen=True)
class StackPreview:
    stack: str
    severity: str
    diff: str


def load_stacks(previews_dir: Path) -> list[StackPreview]:
    stacks: list[StackPreview] = []
    for meta_path in sorted(previews_dir.glob("**/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        diff_path = meta_path.with_name("diff.txt")
        diff = diff_path.read_text(encoding="utf-8") if diff_path.is_file() else ""
        stacks.append(
            StackPreview(
                stack=meta["stack"],
                severity=meta.get("severity", "error"),
                diff=diff.strip(),
            )
        )
    stacks.sort(key=lambda item: item.stack)
    return stacks


def _truncate_diff(diff: str, run_url: str | None) -> str:
    if len(diff) <= _MAX_DIFF_CHARS:
        return diff
    omitted = len(diff) - _MAX_DIFF_CHARS
    where = f" — full plan in the `iac-preview-*` artifact on {run_url}" if run_url else " — see the `iac-preview-*` artifact for the full plan"
    return f"{diff[:_MAX_DIFF_CHARS]}\n... ({omitted} more characters truncated{where})"


def render_comment(
    stacks: list[StackPreview],
    run_url: str | None,
    head_sha: str,
    workflow_ok: bool,
) -> str:
    if _HEAD_SHA_RE.fullmatch(head_sha) is None:
        raise ValueError(f"invalid preview head SHA: {head_sha}")
    preview_ok = (
        workflow_ok
        and len({stack.stack for stack in stacks}) == len(stacks)
        and all(stack.severity in _SEVERITY_ICON and stack.severity != "error" for stack in stacks)
    )
    affected_stacks = sorted(
        stack.stack for stack in stacks if preview_ok and stack.severity in {"change", "delete"}
    )
    metadata = json.dumps(
        {
            "affected_stacks": affected_stacks,
            "head_sha": head_sha,
            "ok": preview_ok,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    lines = [MARKER, f"{METADATA_PREFIX}{metadata}{METADATA_SUFFIX}", "## IaC preview", ""]
    for item in stacks:
        icon = _SEVERITY_ICON.get(item.severity, "🚨")
        lines.append(f"- {icon} `{item.stack}`")

    for item in stacks:
        if not item.diff:
            continue
        lines.append("")
        lines.append("<details>")
        lines.append(f"<summary>{item.stack}</summary>")
        lines.append("")
        lines.append("```diff")
        lines.append(_truncate_diff(item.diff, run_url))
        lines.append("```")
        lines.append("")
        lines.append("</details>")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previews-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded iac-preview-* artifacts.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Write comment body here.")
    parser.add_argument("--head-sha", required=True, help="Commit SHA represented by the preview.")
    parser.add_argument(
        "--workflow-ok",
        required=True,
        choices=("true", "false"),
        help="Whether every preview producer job succeeded.",
    )
    parser.add_argument(
        "--run-url",
        default=None,
        help="Workflow run URL, linked when a stack's diff is truncated for size.",
    )
    args = parser.parse_args()

    stacks = load_stacks(args.previews_dir)
    if not stacks:
        raise SystemExit(f"no preview artifacts under {args.previews_dir}")
    args.out.write_text(
        render_comment(stacks, args.run_url, args.head_sha, args.workflow_ok == "true"),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
