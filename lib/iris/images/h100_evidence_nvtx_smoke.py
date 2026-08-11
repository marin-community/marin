# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise and validate the H100 runner's exact NVTX range implementation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

MAX_RESULT_BYTES = 1 << 20
MAX_SQLITE_BYTES = 64 << 20
EXPECTED_RESULT_KEYS = frozenset(
    {
        "dladdr_library",
        "name",
        "pop_result_kind",
        "pop_return_code",
        "push_result_kind",
        "push_return_code",
        "requested_library",
        "resolved_library",
    }
)


def _load_nvtx_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("h100_evidence_nvtx_range", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load NVTX module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def emit_range(module_path: Path, library_path: Path, output_path: Path, name: str) -> None:
    """Run one exact production NVTX range and persist its closed result."""

    module = _load_nvtx_module(module_path)
    with module.NvtxRange(name, library_path) as nvtx_range:
        pass
    push_result = nvtx_range.push_result
    pop_result = nvtx_range.pop_result
    if push_result is None or pop_result is None:
        raise RuntimeError("NVTX range did not produce balanced push/pop results")
    identity = nvtx_range.library_identity
    output_path.write_text(
        json.dumps(
            {
                "dladdr_library": identity.dladdr_path,
                "name": name,
                "pop_result_kind": pop_result.kind.value,
                "pop_return_code": pop_result.return_code,
                "push_result_kind": push_result.kind.value,
                "push_return_code": push_result.return_code,
                "requested_library": identity.requested_path,
                "resolved_library": identity.resolved_path,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _require_bounded_regular_file(path: Path, maximum_bytes: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"NVTX smoke artifact must be a regular file: {path}")
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f"NVTX smoke artifact exceeds {maximum_bytes} bytes: {path}")


def _bounded_result_bytes(path: Path) -> bytes:
    _require_bounded_regular_file(path, MAX_RESULT_BYTES)
    with path.open("rb") as artifact:
        payload = artifact.read(MAX_RESULT_BYTES + 1)
    if len(payload) > MAX_RESULT_BYTES:
        raise ValueError(f"NVTX smoke artifact exceeds {MAX_RESULT_BYTES} bytes: {path}")
    return payload


def _closed_result(path: Path, expected_name: str) -> dict[str, Any]:
    payload = _bounded_result_bytes(path)
    try:
        result = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("NVTX smoke result is not canonical JSON") from error
    if not isinstance(result, dict) or frozenset(result) != EXPECTED_RESULT_KEYS:
        raise ValueError("NVTX smoke result has an unexpected schema")
    canonical = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if payload != canonical:
        raise ValueError("NVTX smoke result is not canonical JSON")
    if result["name"] != expected_name:
        raise ValueError("NVTX smoke result has the wrong range name")
    push_return_code = result["push_return_code"]
    pop_return_code = result["pop_return_code"]
    push_result_kind = result["push_result_kind"]
    pop_result_kind = result["pop_result_kind"]
    if type(push_return_code) is not int or type(pop_return_code) is not int:
        raise ValueError("NVTX smoke result has noninteger return codes")
    tracked = (
        push_result_kind == "tracked_level"
        and pop_result_kind == "tracked_level"
        and push_return_code >= 0
        and pop_return_code == push_return_code
    )
    untracked = (
        push_result_kind == "untracked_success"
        and pop_result_kind == "untracked_success"
        and push_return_code == -1
        and pop_return_code == -1
    )
    if not tracked and not untracked:
        raise ValueError("NVTX smoke result does not contain a balanced accepted range")
    for key in ("requested_library", "resolved_library", "dladdr_library"):
        if not isinstance(result[key], str) or not result[key]:
            raise ValueError(f"NVTX smoke result has an invalid {key}")
    return result


def validate_report(result_path: Path, sqlite_path: Path, expected_name: str) -> dict[str, Any]:
    """Validate the balanced result and exact exported Nsight Systems event."""

    result = _closed_result(result_path, expected_name)
    _require_bounded_regular_file(sqlite_path, MAX_SQLITE_BYTES)
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as database:
        tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "NVTX_EVENTS" not in tables:
            raise ValueError("Nsight Systems export omits NVTX_EVENTS")
        columns = {row[1] for row in database.execute("PRAGMA table_info(NVTX_EVENTS)")}
        if not {"start", "end", "text"} <= columns:
            raise ValueError("Nsight Systems NVTX_EVENTS has an unsupported schema")
        rows = tuple(database.execute("SELECT start, end, text FROM NVTX_EVENTS WHERE text = ?", (expected_name,)))
    if len(rows) != 1:
        raise ValueError("Nsight Systems export must contain exactly one expected NVTX range")
    start, end, text = rows[0]
    if type(start) is not int or type(end) is not int or end <= start or text != expected_name:
        raise ValueError("Nsight Systems export contains a malformed expected NVTX range")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit")
    emit.add_argument("--module", type=Path, required=True)
    emit.add_argument("--library", type=Path, required=True)
    emit.add_argument("--output", type=Path, required=True)
    emit.add_argument("--name", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--result", type=Path, required=True)
    validate.add_argument("--sqlite", type=Path, required=True)
    validate.add_argument("--name", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "emit":
            emit_range(args.module, args.library, args.output, args.name)
            return
        result = validate_report(args.result, args.sqlite, args.name)
    except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError) as error:
        print(f"NVTX smoke validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
