# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the Nsight Systems profile option used by H100 evidence runs."""

import argparse
import re
import sys
from pathlib import Path

MAX_HELP_BYTES = 1 << 20
OPTION_DECLARATION = re.compile(r"^\s*(?:-[A-Za-z0-9],\s+)?--(?P<name>[a-z0-9][a-z0-9-]*)(?:[ =].*)?$")
POSSIBLE_VALUES = re.compile(
    r"Possible values\s*(?:are|:)\s*(?P<values>.*?)(?:\.\s|\.$)",
    re.IGNORECASE | re.DOTALL,
)
QUOTED_VALUE = re.compile(r"['\"](?P<value>[^'\"]+)['\"]")
REQUIRED_VALUES = {"none", "stop", "stop-shutdown"}


def _capture_range_end_block(text: str) -> str:
    lines = text.splitlines()
    declarations = []
    for index, line in enumerate(lines):
        match = OPTION_DECLARATION.fullmatch(line)
        if match is not None:
            declarations.append((index, match.group("name")))

    target_indices = [index for index, name in declarations if name == "capture-range-end"]
    if len(target_indices) != 1:
        raise ValueError("help must contain one exact --capture-range-end option declaration")
    start = target_indices[0]
    end = next((index for index, _ in declarations if index > start), len(lines))
    return "\n".join(lines[start:end])


def validate_nsys_profile_help(text: str) -> tuple[str, ...]:
    """Return the closed capture-range-end enum from bounded nsys profile help."""
    if not text.strip():
        raise ValueError("help output is empty")
    if "\x00" in text:
        raise ValueError("help output contains NUL")
    if "--stop-on-range-end" in text:
        raise ValueError("help exposes obsolete --stop-on-range-end")

    block = _capture_range_end_block(text)
    value_lists = tuple(POSSIBLE_VALUES.finditer(block))
    if len(value_lists) != 1:
        raise ValueError("--capture-range-end must contain one recognizable possible-values list")
    values = tuple(match.group("value") for match in QUOTED_VALUE.finditer(value_lists[0].group("values")))
    if len(values) != len(set(values)):
        raise ValueError("--capture-range-end possible values are duplicated")
    if not REQUIRED_VALUES.issubset(values) or values.count("stop") != 1:
        raise ValueError("--capture-range-end does not expose the exact stop policy")
    return values


def validate_file(path: Path) -> tuple[str, ...]:
    """Read and validate one bounded UTF-8 nsys profile help artifact."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"help artifact must be a regular file: {path}")
    if path.stat().st_size > MAX_HELP_BYTES:
        raise ValueError(f"help artifact exceeds {MAX_HELP_BYTES} bytes")
    with path.open("rb") as artifact:
        payload = artifact.read(MAX_HELP_BYTES + 1)
    if len(payload) > MAX_HELP_BYTES:
        raise ValueError(f"help artifact exceeds {MAX_HELP_BYTES} bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("help artifact is not UTF-8") from error
    return validate_nsys_profile_help(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    try:
        values = validate_file(args.artifact)
    except (OSError, ValueError) as error:
        print(f"nsys profile help validation failed: {error}", file=sys.stderr)
        return 1
    print(f"nsys profile help validation passed: capture-range-end={','.join(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
