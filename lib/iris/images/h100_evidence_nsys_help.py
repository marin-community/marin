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
QUOTED_TOKEN = r"(?:'[^']+'|\"[^\"]+\")"
CLOSED_VALUE_LIST = re.compile(
    rf"^\s*{QUOTED_TOKEN}(?:\s*,\s*{QUOTED_TOKEN})*(?:\s*,?\s+and\s+{QUOTED_TOKEN})?\s*$",
    re.IGNORECASE,
)
CAPTURE_RANGE_END_VALUES = {
    "none",
    "stop",
    "stop-shutdown",
    "repeat[:N][:mode]",
    "repeat-shutdown:N[:mode]",
}
CUDA_GRAPH_TRACE_VALUES = {"graph", "node"}


def _option_block(text: str, option: str) -> str:
    lines = text.splitlines()
    declarations = []
    for index, line in enumerate(lines):
        match = OPTION_DECLARATION.fullmatch(line)
        if match is not None:
            declarations.append((index, match.group("name")))

    target_indices = [index for index, name in declarations if name == option]
    if len(target_indices) != 1:
        raise ValueError(f"help must contain one exact --{option} option declaration")
    start = target_indices[0]
    end = next((index for index, _ in declarations if index > start), len(lines))
    return "\n".join(lines[start:end])


def _option_values(text: str, option: str) -> tuple[str, ...]:
    block = _option_block(text, option)
    value_lists = tuple(POSSIBLE_VALUES.finditer(block))
    if len(value_lists) != 1:
        raise ValueError(f"--{option} must contain one recognizable possible-values list")
    serialized_values = value_lists[0].group("values")
    if CLOSED_VALUE_LIST.fullmatch(serialized_values) is None:
        raise ValueError(f"--{option} possible values contain unrecognized syntax")
    values = tuple(match.group("value") for match in QUOTED_VALUE.finditer(serialized_values))
    if not values or len(values) != len(set(values)):
        raise ValueError(f"--{option} possible values are empty or duplicated")
    return values


def validate_nsys_profile_help(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the closed capture-end and CUDA-graph enums from nsys help."""
    if not text.strip():
        raise ValueError("help output is empty")
    if "\x00" in text:
        raise ValueError("help output contains NUL")
    if "--stop-on-range-end" in text:
        raise ValueError("help exposes obsolete --stop-on-range-end")

    capture_values = _option_values(text, "capture-range-end")
    if set(capture_values) != CAPTURE_RANGE_END_VALUES or capture_values.count("stop") != 1:
        raise ValueError("--capture-range-end does not expose the exact closed stop policy")
    graph_values = _option_values(text, "cuda-graph-trace")
    if set(graph_values) != CUDA_GRAPH_TRACE_VALUES or graph_values.count("node") != 1:
        raise ValueError("--cuda-graph-trace does not expose exactly graph and node")
    return capture_values, graph_values


def validate_file(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
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
        capture_values, graph_values = validate_file(args.artifact)
    except (OSError, ValueError) as error:
        print(f"nsys profile help validation failed: {error}", file=sys.stderr)
        return 1
    print(
        "nsys profile help validation passed: "
        f"capture-range-end={','.join(capture_values)};cuda-graph-trace={','.join(graph_values)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
