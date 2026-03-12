# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify E2E screenshots against their text descriptions using claude CLI."""

import argparse
import subprocess
import sys
from pathlib import Path


def verify_screenshot(image_path: Path, description: str) -> tuple[bool, str]:
    prompt = (
        f"Read the screenshot at {image_path} and determine if it matches "
        f"the following description.\n\n"
        f"Description: {description}\n\n"
        f"Reply with exactly OK if the screenshot matches, or NOT_OK followed by "
        f"a brief explanation. Be lenient about exact text but verify structural "
        f"elements (badges, tables, cards, charts) are present."
    )
    result = subprocess.run(
        ["claude", "--print", prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return False, f"claude CLI failed: {result.stderr.strip()}"
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

    failures = []
    for png_path, desc in pairs:
        label = png_path.stem
        print(f"Verifying {label}...")
        passed, explanation = verify_screenshot(png_path, desc)
        print(f"  {'OK' if passed else 'NOT_OK'}: {explanation}")
        if not passed:
            failures.append(label)

    if failures:
        print(f"\nFailed: {', '.join(failures)}")
        sys.exit(1)
    print(f"\nAll {len(pairs)} screenshots verified OK")


if __name__ == "__main__":
    main()
