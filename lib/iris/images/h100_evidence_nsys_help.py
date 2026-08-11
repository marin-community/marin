# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the Nsight Systems profile option used by H100 evidence runs."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MAX_HELP_BYTES = 1 << 20
MAX_POSSIBLE_VALUES_CLAUSE_BYTES = 1024
MAX_GRAPH_DIAGNOSTIC_LINE_BYTES = 512
MAX_GRAPH_DIAGNOSTIC_BYTES = 1536
MAX_FAILURE_MESSAGE_CHARS = 4096
FAILURE_DIAGNOSTIC_SCHEMA = "iris.h100_evidence_nsys_help_failure.v3"
OPTION_DECLARATION = re.compile(r"^\s*(?:-[A-Za-z0-9],\s+)?--(?P<name>[a-z0-9][a-z0-9-]*)(?:[ =].*)?$")
POSSIBLE_VALUES = re.compile(
    r"Possible values\s*(?:are|:)\s*(?P<values>.*?)\.(?=\s|$)",
    re.IGNORECASE | re.DOTALL,
)
QUOTED_VALUE = re.compile(r"['\"](?P<value>[^'\"]+)['\"]")
QUOTED_TOKEN = r"(?:'[^']+'|\"[^\"]+\")"
CLOSED_VALUE_LIST = re.compile(
    rf"^\s*{QUOTED_TOKEN}(?:\s*,\s*{QUOTED_TOKEN})*(?:\s*,?\s+and\s+{QUOTED_TOKEN})?\s*$",
    re.IGNORECASE,
)
CAPTURE_RANGE_END_LIST = re.compile(
    r"^\s*'none'\s*,\s*'stop'\s*,\s*'stop-shutdown'\s*,\s*"
    r"'repeat\[:N\]\[:mode\]'\s+or\s+'repeat-shutdown:N'\[:mode\]\s*$"
)
CAPTURE_RANGE_END_VALUES = (
    "none",
    "stop",
    "stop-shutdown",
    "repeat[:N][:mode]",
    "repeat-shutdown:N[:mode]",
)
CUDA_GRAPH_TRACE_VALUES = {"graph", "node"}
GRAPH_DIAGNOSTIC_TOKEN = re.compile(r"(?<![a-z0-9_-])(?P<token>graph|node)(?![a-z0-9_-])")


def _option_block_lines(text: str, option: str) -> list[str]:
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
    return lines[start:end]


def _option_block(text: str, option: str) -> str:
    return "\n".join(_option_block_lines(text, option))


def _option_values(text: str, option: str) -> tuple[str, ...]:
    block = _option_block(text, option)
    value_lists = tuple(POSSIBLE_VALUES.finditer(block))
    if len(value_lists) != 1:
        raise ValueError(f"--{option} must contain one recognizable possible-values list")
    serialized_values = value_lists[0].group("values")
    if option == "capture-range-end":
        if CAPTURE_RANGE_END_LIST.fullmatch(serialized_values) is None:
            raise ValueError(f"--{option} possible values contain unrecognized syntax")
        return CAPTURE_RANGE_END_VALUES
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
    if capture_values != CAPTURE_RANGE_END_VALUES:
        raise ValueError("--capture-range-end does not expose the exact closed stop policy")
    graph_values = _option_values(text, "cuda-graph-trace")
    if set(graph_values) != CUDA_GRAPH_TRACE_VALUES or graph_values.count("node") != 1:
        raise ValueError("--cuda-graph-trace does not expose exactly graph and node")
    return capture_values, graph_values


def _read_help_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"help artifact must be a regular file: {path}")
    if path.stat().st_size > MAX_HELP_BYTES:
        raise ValueError(f"help artifact exceeds {MAX_HELP_BYTES} bytes")
    with path.open("rb") as artifact:
        payload = artifact.read(MAX_HELP_BYTES + 1)
    if len(payload) > MAX_HELP_BYTES:
        raise ValueError(f"help artifact exceeds {MAX_HELP_BYTES} bytes")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("help artifact is not UTF-8") from error


def validate_file(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read and validate one bounded UTF-8 nsys profile help artifact."""
    return validate_nsys_profile_help(_read_help_file(path))


def _diagnostic_possible_values_clause(text: str, option: str) -> dict[str, object]:
    try:
        block = _option_block(text, option)
    except ValueError:
        return {
            "available": False,
            "bytes": 0,
            "reason": "exact_option_anchor_unavailable",
            "sha256": None,
        }
    clauses = tuple(POSSIBLE_VALUES.finditer(block))
    if len(clauses) != 1:
        return {
            "available": False,
            "bytes": 0,
            "reason": "unique_possible_values_clause_unavailable",
            "sha256": None,
        }
    clause = clauses[0].group(0)
    payload = clause.encode("utf-8")
    record: dict[str, object] = {
        "available": len(payload) <= MAX_POSSIBLE_VALUES_CLAUSE_BYTES,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if len(payload) <= MAX_POSSIBLE_VALUES_CLAUSE_BYTES:
        record["text"] = clause
    else:
        record["reason"] = f"exceeds_{MAX_POSSIBLE_VALUES_CLAUSE_BYTES}_byte_bound"
    return record


def _bounded_text_record(text: str, limit: int) -> dict[str, object]:
    payload = text.encode("utf-8")
    record: dict[str, object] = {
        "available": len(payload) <= limit,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if len(payload) <= limit:
        record["text"] = text
    else:
        record["reason"] = f"exceeds_{limit}_byte_bound"
    return record


def _diagnostic_cuda_graph_context(text: str) -> dict[str, object]:
    try:
        block = _option_block_lines(text, "cuda-graph-trace")
    except ValueError:
        return {
            "available": False,
            "bytes": 0,
            "declaration": None,
            "graph_occurrences": 0,
            "node_occurrences": 0,
            "reason": "exact_option_anchor_unavailable",
            "sha256": None,
            "token_line_count": 0,
        }

    declaration = block[0]
    token_lines: list[str] = []
    token_sequence: list[str] = []
    for line in block[1:]:
        tokens = [match.group("token") for match in GRAPH_DIAGNOSTIC_TOKEN.finditer(line)]
        if tokens:
            token_lines.append(line)
            token_sequence.extend(tokens)

    context = "\n".join((declaration, *token_lines))
    context_payload = context.encode("utf-8")
    record: dict[str, object] = {
        "available": False,
        "bytes": len(context_payload),
        "declaration": _bounded_text_record(declaration, MAX_GRAPH_DIAGNOSTIC_LINE_BYTES),
        "graph_occurrences": token_sequence.count("graph"),
        "node_occurrences": token_sequence.count("node"),
        "sha256": hashlib.sha256(context_payload).hexdigest(),
        "token_line_count": len(token_lines),
    }
    if token_sequence != ["graph", "node"]:
        record["reason"] = "exact_graph_node_token_sequence_unavailable"
        return record
    if len(declaration.encode("utf-8")) > MAX_GRAPH_DIAGNOSTIC_LINE_BYTES:
        record["reason"] = f"declaration_line_exceeds_{MAX_GRAPH_DIAGNOSTIC_LINE_BYTES}_byte_bound"
        return record
    if any(len(line.encode("utf-8")) > MAX_GRAPH_DIAGNOSTIC_LINE_BYTES for line in token_lines):
        record["reason"] = f"token_line_exceeds_{MAX_GRAPH_DIAGNOSTIC_LINE_BYTES}_byte_bound"
        return record
    if len(context_payload) > MAX_GRAPH_DIAGNOSTIC_BYTES:
        record["reason"] = f"context_exceeds_{MAX_GRAPH_DIAGNOSTIC_BYTES}_byte_bound"
        return record

    record["available"] = True
    record["token_lines"] = [_bounded_text_record(line, MAX_GRAPH_DIAGNOSTIC_LINE_BYTES) for line in token_lines]
    return record


def _failure_diagnostic(text: str) -> dict[str, object]:
    return {
        "clauses": {
            option: _diagnostic_possible_values_clause(text, option)
            for option in ("capture-range-end", "cuda-graph-trace")
        },
        "cuda_graph_context": _diagnostic_cuda_graph_context(text),
        "schema": FAILURE_DIAGNOSTIC_SCHEMA,
    }


def _without_diagnostic_text(diagnostic: dict[str, object]) -> dict[str, object]:
    clauses = diagnostic["clauses"]
    assert isinstance(clauses, dict)
    bounded_clauses: dict[str, object] = {}
    for option, value in clauses.items():
        assert isinstance(value, dict)
        record = {key: field for key, field in value.items() if key != "text"}
        if "text" in value:
            record["available"] = False
            record["reason"] = f"omitted_to_fit_{MAX_FAILURE_MESSAGE_CHARS}_character_bound"
        bounded_clauses[option] = record
    context = diagnostic["cuda_graph_context"]
    assert isinstance(context, dict)
    bounded_context = {key: field for key, field in context.items() if key != "token_lines"}
    declaration = bounded_context.get("declaration")
    if isinstance(declaration, dict):
        bounded_context["declaration"] = {key: field for key, field in declaration.items() if key != "text"}
    if "token_lines" in context:
        bounded_context["available"] = False
        bounded_context["reason"] = f"omitted_to_fit_{MAX_FAILURE_MESSAGE_CHARS}_character_bound"
    return {
        "clauses": bounded_clauses,
        "cuda_graph_context": bounded_context,
        "schema": FAILURE_DIAGNOSTIC_SCHEMA,
    }


def _validation_failure_message(text: str, error: ValueError) -> str:
    prefix = f"nsys profile help validation failed: {error}"
    diagnostic = _failure_diagnostic(text)
    serialized = json.dumps(diagnostic, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    message = f"{prefix} diagnostic={serialized}"
    if len(f"{message}\n") <= MAX_FAILURE_MESSAGE_CHARS:
        return message

    serialized = json.dumps(
        _without_diagnostic_text(diagnostic),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    message = f"{prefix} diagnostic={serialized}"
    if len(f"{message}\n") <= MAX_FAILURE_MESSAGE_CHARS:
        return message
    return "nsys profile help validation failed: bounded diagnostic could not be serialized"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    try:
        text = _read_help_file(args.artifact)
    except (OSError, ValueError) as error:
        print(f"nsys profile help validation failed: {error}", file=sys.stderr)
        return 1
    try:
        capture_values, graph_values = validate_nsys_profile_help(text)
    except ValueError as error:
        print(_validation_failure_message(text, error), file=sys.stderr)
        return 1
    print(
        "nsys profile help validation passed: "
        f"capture-range-end={','.join(capture_values)};cuda-graph-trace={','.join(graph_values)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
