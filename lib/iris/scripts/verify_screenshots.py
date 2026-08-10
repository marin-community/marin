# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify E2E screenshots against their text descriptions using claude CLI."""

import argparse
import sys
from pathlib import Path

from scripts.ci.claude_runner import ClaudeRunStatus, report_rate_limit, run_claude


def verify_screenshots(pairs: list[tuple[Path, str]]) -> tuple[ClaudeRunStatus, str]:
    """Pass all screenshot+description pairs to claude in one call."""
    lines = []
    for i, (png, desc) in enumerate(pairs, 1):
        lines.append(f"{i}. Read the screenshot at {png} — expected: {desc}")

    descriptions = "\n".join(lines)
    prompt = (
        "You are verifying E2E test screenshots. For each numbered item below, "
        "read the screenshot file and determine if it matches the expected description.\n\n"
        f"{descriptions}\n\n"
        "Be lenient about exact text but verify structural elements "
        "(badges, tables, cards, charts) are present. We're testing for big failures, "
        "not minor text differences.\n\n"
        "End your reply with a verdict line on its own: 'OK' if ALL screenshots match "
        "their descriptions, or 'NOT_OK' if any fail. When NOT_OK, follow it with one "
        "line per failing screenshot in the format:\n"
        "  - <filename>: <brief reason>"
    )
    result = run_claude(
        prompt,
        ["--model=sonnet", "--dangerously-skip-permissions", "--tools=Read"],
        timeout=180,
    )
    if result.status == ClaudeRunStatus.RATE_LIMITED:
        return result.status, result.output
    text = result.output.strip()
    # Claude often prepends reasoning despite instructions, so we cannot require the
    # response to *start* with the verdict. Decide on the verdict token: NOT_OK marks a
    # failure (checked first since it contains the substring "OK"); a bare OK means pass.
    # Absence of either token is ambiguous and treated as a failure.
    if "NOT_OK" in text:
        return ClaudeRunStatus.FAILED, text
    if "OK" in text:
        return ClaudeRunStatus.SUCCESS, text
    return ClaudeRunStatus.FAILED, f"No OK/NOT_OK verdict found in claude response:\n{text}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.screenshot_dir.exists():
        print(f"Screenshot dir does not exist: {args.screenshot_dir}")
        sys.exit(0)

    pairs = []
    for png in sorted(args.screenshot_dir.glob("smoke-*.png")):
        txt = png.with_suffix(".txt")
        if txt.exists():
            pairs.append((png, txt.read_text().strip()))

    if not pairs:
        print("No screenshot+description pairs found, skipping")
        sys.exit(0)

    print(f"Verifying {len(pairs)} screenshots in one batch...")
    status, explanation = verify_screenshots(pairs)
    if status == ClaudeRunStatus.RATE_LIMITED:
        report_rate_limit()
        return
    print(explanation)

    if status == ClaudeRunStatus.FAILED:
        sys.exit(1)
    print(f"\nAll {len(pairs)} screenshots verified OK")


if __name__ == "__main__":
    main()
