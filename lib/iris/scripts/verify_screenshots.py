# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify E2E screenshots against their text descriptions using claude CLI."""

import argparse
import subprocess
import sys
from pathlib import Path


def verify_screenshots(pairs: list[tuple[Path, str]]) -> tuple[bool, str]:
    """Pass all screenshot+description pairs to claude in one call."""
    lines = []
    for i, (png, desc) in enumerate(pairs, 1):
        lines.append(f"{i}. Read the screenshot at {png} — expected: {desc}")

    descriptions = "\n".join(lines)
    prompt = (
        "You are verifying E2E test screenshots. For each numbered item below, "
        "read the screenshot file and determine if it matches the expected description.\n\n"
        f"{descriptions}\n\n"
        "Reply with a single line: OK if ALL screenshots match their descriptions.\n"
        "Otherwise reply NOT_OK followed by one line per failing screenshot in the format:\n"
        "  - <filename>: <brief reason>\n\n"
        "Be lenient about exact text but verify structural elements "
        "(badges, tables, cards, charts) are present. We're testing for big failures, "
        "not minor text differences."
    )
    result = subprocess.run(
        ["claude", "--model=sonnet", "--print", "--dangerously-skip-permissions", "--tools=Read", "--", prompt],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        return False, f"claude CLI failed: {result.stderr.strip()} {result.stdout.strip()}"
    text = result.stdout.strip()
    return text.startswith("OK"), text


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
    passed, explanation = verify_screenshots(pairs)
    print(explanation)

    if not passed:
        sys.exit(1)
    print(f"\nAll {len(pairs)} screenshots verified OK")


if __name__ == "__main__":
    main()
