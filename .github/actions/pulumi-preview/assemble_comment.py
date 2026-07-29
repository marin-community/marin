#!/usr/bin/env python3
"""Assemble one IaC preview PR comment body from per-stack preview artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MARKER = "<!-- iac-preview -->"

_SEVERITY_ICON = {
    "none": "✅",
    "change": "⚠️",
    "delete": "🚨",
    "error": "❌",
}


def load_stacks(previews_dir: Path) -> list[dict]:
    stacks: list[dict] = []
    for meta_path in sorted(previews_dir.glob("**/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        diff_path = meta_path.with_name("diff.txt")
        diff = diff_path.read_text(encoding="utf-8") if diff_path.is_file() else ""
        stacks.append({"meta": meta, "diff": diff.strip(), "stack": meta["stack"]})
    stacks.sort(key=lambda item: item["stack"])
    return stacks


def render_comment(stacks: list[dict]) -> str:
    lines = [MARKER, "## IaC preview", ""]
    for item in stacks:
        severity = item["meta"].get("severity", "error")
        icon = _SEVERITY_ICON.get(severity, "🚨")
        lines.append(f"- {icon} `{item['stack']}`")

    for item in stacks:
        if not item["diff"]:
            continue
        lines.append("")
        lines.append("<details>")
        lines.append(f"<summary>{item['stack']}</summary>")
        lines.append("")
        lines.append("```diff")
        lines.append(item["diff"])
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
    args = parser.parse_args()

    stacks = load_stacks(args.previews_dir)
    if not stacks:
        raise SystemExit(f"no preview artifacts under {args.previews_dir}")
    args.out.write_text(render_comment(stacks), encoding="utf-8")


if __name__ == "__main__":
    main()
