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
NVDISASM_SECTION = re.compile(r"^\s*\.section\s+(?P<name>\.[A-Za-z0-9_.$]+)(?:\s*,.*)?$")
NVDISASM_FUNCTION_TYPE = re.compile(r"^\s*\.type\s+(?P<name>[A-Za-z_.$][A-Za-z0-9_.$]*)\s*,\s*@function\s*$")
NVDISASM_TEXT_LABEL = re.compile(r"^\s*\.text\.(?P<name>[A-Za-z_.$][A-Za-z0-9_.$]*):\s*$")


@dataclass(frozen=True)
class InstructionRecord:
    address: int
    mnemonic: str


def _nvdisasm_function_body(lines: tuple[str, ...], expected_kernel: str) -> tuple[str, ...]:
    global_indices = tuple(
        index
        for index, line in enumerate(lines)
        if (match := NVDISASM_GLOBAL.fullmatch(line)) is not None and match.group("name") == expected_kernel
    )
    label = re.compile(rf"^\s*{re.escape(expected_kernel)}:\s*$")
    label_indices = tuple(index for index, line in enumerate(lines) if label.fullmatch(line) is not None)
    if len(global_indices) != 1 or len(label_indices) != 1:
        raise ValueError("nvdisasm output must contain one exact expected global and function label")

    global_index = global_indices[0]
    label_index = label_indices[0]
    if label_index <= global_index:
        raise ValueError("nvdisasm expected function label does not follow its global")

    sections = tuple(
        (index, match.group("name"))
        for index, line in enumerate(lines[:global_index])
        if (match := NVDISASM_SECTION.fullmatch(line)) is not None
    )
    expected_section = f".text.{expected_kernel}"
    if not sections or sections[-1][1] != expected_section:
        raise ValueError(f"nvdisasm expected global is not in section {expected_section}")

    for line in lines[global_index + 1 : label_index]:
        if NVDISASM_SECTION.fullmatch(line) is not None or NVDISASM_GLOBAL.fullmatch(line) is not None:
            raise ValueError("nvdisasm expected global and function label are in different scopes")
        function_type = NVDISASM_FUNCTION_TYPE.fullmatch(line)
        if function_type is not None and function_type.group("name") != expected_kernel:
            raise ValueError("nvdisasm found another function before the expected label")

    text_label_index = label_index + 1
    if text_label_index >= len(lines):
        raise ValueError("nvdisasm expected function has no text label")
    text_label = NVDISASM_TEXT_LABEL.fullmatch(lines[text_label_index])
    if text_label is None or text_label.group("name") != expected_kernel:
        raise ValueError("nvdisasm expected text label does not immediately follow its function label")

    body = []
    for line in lines[text_label_index + 1 :]:
        if NVDISASM_SECTION.fullmatch(line) is not None or NVDISASM_GLOBAL.fullmatch(line) is not None:
            break
        if NVDISASM_FUNCTION_TYPE.fullmatch(line) is not None:
            break
        if NVDISASM_TEXT_LABEL.fullmatch(line) is not None:
            raise ValueError("nvdisasm function body contains an unexpected text label")
        body.append(line)
    return tuple(body)


def _instruction_records(lines: tuple[str, ...], *, allow_encoding_continuations: bool) -> tuple[InstructionRecord, ...]:
    records = []
    previous_was_instruction = False
    for line in lines:
        match = INSTRUCTION.fullmatch(line)
        if match is not None:
            records.append(InstructionRecord(address=int(match.group("address"), 16), mnemonic=match.group("mnemonic")))
            previous_was_instruction = True
            continue
        if ENCODING_CONTINUATION.fullmatch(line) is not None:
            if not allow_encoding_continuations or not previous_was_instruction:
                raise ValueError("unexpected standalone instruction encoding")
            previous_was_instruction = False
            continue
        if COMMENT_PREFIX.match(line) is not None:
            raise ValueError(f"malformed address-bearing instruction record: {line!r}")
        previous_was_instruction = False

    return tuple(records)


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

    lines = tuple(text.splitlines())
    if output_format == "cuobjdump":
        names = tuple(match.group("name") for line in lines if (match := CUOBJDUMP_FUNCTION.fullmatch(line)))
        if names != (expected_kernel,):
            raise ValueError(f"cuobjdump functions differ from expected kernel: {sorted(names)!r}")
        records = _instruction_records(lines, allow_encoding_continuations=True)
    elif output_format == "nvdisasm":
        body = _nvdisasm_function_body(lines, expected_kernel)
        records = _instruction_records(body, allow_encoding_continuations=False)
    else:
        raise ValueError(f"unsupported SASS output format: {output_format}")
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
