# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the CUDA disassembler output used by the H100 image build."""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_SASS_BYTES = 1 << 20
KERNEL_NAME = re.compile(r"[A-Za-z_.$][A-Za-z0-9_.$]*")
DIAGNOSTIC = re.compile(r"\b(?:error|fatal|warning)\b", re.IGNORECASE)
INSTRUCTION = re.compile(
    r"^\s*/\*(?P<address>[0-9A-Fa-f]{4,16})\*/\s+"
    r"(?:(?:@!?[A-Z][A-Z0-9.]*)\s+)?"
    r"(?P<mnemonic>[A-Z][A-Z0-9]*(?:\.[A-Z0-9]+)*)\b"
    r".*;\s*(?:/\*\s*0x[0-9A-Fa-f]+\s*\*/)?\s*$"
)
COMMENT_PREFIX = re.compile(r"^\s*/\*")
ENCODING_CONTINUATION = re.compile(r"^\s*/\*\s*0x[0-9A-Fa-f]{16,32}\s*\*/\s*$")
CUOBJDUMP_FUNCTION = re.compile(r"^\s*Function\s*:\s*(?P<name>[A-Za-z_.$][A-Za-z0-9_.$]*)\s*$")
NVDISASM_GLOBAL = re.compile(r"^\s*\.global\s+(?P<name>[A-Za-z_.$][A-Za-z0-9_.$]*)\s*$")


@dataclass(frozen=True)
class InstructionRecord:
    address: int
    mnemonic: str


def validate_sass(text: str, *, output_format: str, expected_kernel: str) -> tuple[InstructionRecord, ...]:
    """Return validated instruction records for one expected CUDA kernel."""
    if KERNEL_NAME.fullmatch(expected_kernel) is None:
        raise ValueError("expected kernel is not a canonical CUDA symbol")
    if not text.strip():
        raise ValueError("output is empty")
    if "\x00" in text:
        raise ValueError("output contains NUL")
    if DIAGNOSTIC.search(text) is not None:
        raise ValueError("output contains a warning or error diagnostic")

    lines = text.splitlines()
    if output_format == "cuobjdump":
        names = tuple(match.group("name") for line in lines if (match := CUOBJDUMP_FUNCTION.fullmatch(line)))
        if names != (expected_kernel,):
            raise ValueError(f"cuobjdump functions differ from expected kernel: {sorted(names)!r}")
    elif output_format == "nvdisasm":
        names = tuple(match.group("name") for line in lines if (match := NVDISASM_GLOBAL.fullmatch(line)))
        if names != (expected_kernel,) or lines.count(f"{expected_kernel}:") != 1:
            raise ValueError(f"nvdisasm symbols differ from expected kernel: {sorted(names)!r}")
    else:
        raise ValueError(f"unsupported SASS output format: {output_format}")

    records = []
    previous_was_instruction = False
    for line in lines:
        match = INSTRUCTION.fullmatch(line)
        if match is not None:
            records.append(InstructionRecord(address=int(match.group("address"), 16), mnemonic=match.group("mnemonic")))
            previous_was_instruction = True
            continue
        if ENCODING_CONTINUATION.fullmatch(line) is not None:
            if output_format != "cuobjdump" or not previous_was_instruction:
                raise ValueError("unexpected standalone instruction encoding")
            previous_was_instruction = False
            continue
        if COMMENT_PREFIX.match(line) is not None:
            raise ValueError(f"malformed address-bearing instruction record: {line!r}")
        previous_was_instruction = False

    records = tuple(records)
    if not records:
        raise ValueError("output contains no address-bearing instruction records")
    addresses = tuple(record.address for record in records)
    if addresses != tuple(sorted(set(addresses))):
        raise ValueError("instruction addresses must be unique and strictly increasing")
    return records


def validate_file(path: Path, *, output_format: str, expected_kernel: str) -> tuple[InstructionRecord, ...]:
    """Read and validate one bounded UTF-8 SASS artifact."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"SASS artifact must be a regular file: {path}")
    if path.stat().st_size > MAX_SASS_BYTES:
        raise ValueError(f"SASS artifact exceeds {MAX_SASS_BYTES} bytes")
    with path.open("rb") as artifact:
        payload = artifact.read(MAX_SASS_BYTES + 1)
    if len(payload) > MAX_SASS_BYTES:
        raise ValueError(f"SASS artifact exceeds {MAX_SASS_BYTES} bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("SASS artifact is not UTF-8") from error
    return validate_sass(text, output_format=output_format, expected_kernel=expected_kernel)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("cuobjdump", "nvdisasm"), required=True)
    parser.add_argument("--expected-kernel", required=True)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    try:
        records = validate_file(args.artifact, output_format=args.format, expected_kernel=args.expected_kernel)
    except (OSError, ValueError) as error:
        print(f"{args.format} SASS validation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "format": args.format,
                "instruction_count": len(records),
                "kernel": args.expected_kernel,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
