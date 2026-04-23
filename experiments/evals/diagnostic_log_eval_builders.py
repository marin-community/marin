# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample-capped builders for diagnostic-log eval slices.

These helpers intentionally require pre-staged sample inputs (local paths or
fsspec URLs). They never fetch full public corpora directly.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import posixpath
from dataclasses import dataclass
from json import JSONDecodeError
from collections.abc import Callable

import fsspec
from rigging.filesystem import url_to_fs

GHALOGS_TEXT_FIELDS = ("message", "log", "text", "line", "content")
GHALOGS_ALLOWED_SUFFIXES = (".jsonl", ".json", ".ndjson", ".log", ".txt", ".jsonl.gz", ".json.gz", ".log.gz")
LOGHUB_ALLOWED_SUFFIXES = (".log", ".txt")
DIAGNOSTIC_LOG_EVAL_OUTPUTS = {
    "ghalogs": "diagnostic_logs/ghalogs/runs.jsonl.gz",
    "loghub_apache": "diagnostic_logs/loghub/apache.jsonl.gz",
}


@dataclass(frozen=True)
class DiagnosticLogMaterializationStats:
    source_name: str
    files_seen: int
    files_used: int
    rows_written: int
    bytes_written: int
    max_files: int
    max_rows: int
    max_bytes: int


def _validate_cap(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _protocol_path(fs: fsspec.AbstractFileSystem, relative_path: str) -> str:
    protocol = fs.protocol
    if isinstance(protocol, tuple):
        protocol = protocol[0]
    if protocol in ("", None, "file"):
        return relative_path
    return f"{protocol}://{relative_path}"


def _list_source_files(
    source_path: str, *, allowed_suffixes: tuple[str, ...]
) -> tuple[fsspec.AbstractFileSystem, list[str]]:
    fs, source_root = url_to_fs(source_path)
    if fs.exists(source_root) and fs.isfile(source_root):
        return fs, [source_root]

    if not fs.exists(source_root) or not fs.isdir(source_root):
        raise ValueError(f"Expected a pre-staged file or directory at {source_path}")

    matches: set[str] = set()
    root = source_root.rstrip("/")
    for suffix in allowed_suffixes:
        pattern = os.path.join(root, "**", f"*{suffix}")
        for match in fs.glob(pattern):
            if fs.isfile(match):
                matches.add(match)

    files = sorted(matches)
    if not files:
        raise ValueError(f"No supported input files found at {source_path}")
    return fs, files


def _write_capped_jsonl(
    *,
    source_name: str,
    source_path: str,
    output_path: str,
    max_files: int,
    max_rows: int,
    max_bytes: int,
    allowed_suffixes: tuple[str, ...],
    line_parser: Callable[[str], str | None],
) -> DiagnosticLogMaterializationStats:
    _validate_cap("max_files", max_files)
    _validate_cap("max_rows", max_rows)
    _validate_cap("max_bytes", max_bytes)

    source_fs, source_files = _list_source_files(source_path, allowed_suffixes=allowed_suffixes)
    selected_files = source_files[:max_files]

    output_fs, output_file = url_to_fs(output_path)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        output_fs.makedirs(output_dir, exist_ok=True)

    rows_written = 0
    bytes_written = 0
    files_used = 0
    stop = False

    with output_fs.open(output_file, "wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb") as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8") as writer:
                for source_file in selected_files:
                    wrote_from_file = False
                    source_file_path = _protocol_path(source_fs, source_file)
                    with fsspec.open(
                        source_file_path,
                        mode="rt",
                        compression="infer",
                        encoding="utf-8",
                        errors="replace",
                    ) as reader:
                        for raw_line in reader:
                            text = line_parser(raw_line)
                            if text is None:
                                continue

                            record = {"text": text}
                            payload = json.dumps(record, ensure_ascii=False) + "\n"
                            payload_bytes = len(payload.encode("utf-8"))

                            if bytes_written + payload_bytes > max_bytes:
                                stop = True
                                break

                            writer.write(payload)
                            rows_written += 1
                            bytes_written += payload_bytes
                            wrote_from_file = True

                            if rows_written >= max_rows:
                                stop = True
                                break

                    if wrote_from_file:
                        files_used += 1
                    if stop:
                        break

    return DiagnosticLogMaterializationStats(
        source_name=source_name,
        files_seen=len(source_files),
        files_used=files_used,
        rows_written=rows_written,
        bytes_written=bytes_written,
        max_files=max_files,
        max_rows=max_rows,
        max_bytes=max_bytes,
    )


def _plain_log_line(line: str) -> str | None:
    text = line.strip()
    if not text:
        return None
    return text


def _ghalogs_line(line: str) -> str | None:
    text = line.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except JSONDecodeError:
        return text

    if isinstance(parsed, str):
        stripped = parsed.strip()
        if not stripped:
            return None
        return stripped

    if isinstance(parsed, dict):
        for field in GHALOGS_TEXT_FIELDS:
            value = parsed.get(field)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)

    return json.dumps(parsed, ensure_ascii=False)


def materialize_ghalogs_eval_sample(
    *,
    source_path: str,
    output_path: str,
    max_files: int = 4,
    max_rows: int = 4_000,
    max_bytes: int = 8_000_000,
) -> DiagnosticLogMaterializationStats:
    """Build a capped `ghalogs` eval slice from pre-staged sample files."""
    return _write_capped_jsonl(
        source_name="ghalogs",
        source_path=source_path,
        output_path=output_path,
        max_files=max_files,
        max_rows=max_rows,
        max_bytes=max_bytes,
        allowed_suffixes=GHALOGS_ALLOWED_SUFFIXES,
        line_parser=_ghalogs_line,
    )


def materialize_loghub_eval_sample(
    *,
    source_path: str,
    output_path: str,
    max_files: int = 4,
    max_rows: int = 4_000,
    max_bytes: int = 8_000_000,
) -> DiagnosticLogMaterializationStats:
    """Build a capped `loghub_apache` eval slice from pre-staged sample files."""
    return _write_capped_jsonl(
        source_name="loghub_apache",
        source_path=source_path,
        output_path=output_path,
        max_files=max_files,
        max_rows=max_rows,
        max_bytes=max_bytes,
        allowed_suffixes=LOGHUB_ALLOWED_SUFFIXES,
        line_parser=_plain_log_line,
    )


def diagnostic_log_eval_output_path(raw_root: str, *, slice_name: str) -> str:
    """Return the long-tail raw path for a supported diagnostic eval slice."""
    if slice_name not in DIAGNOSTIC_LOG_EVAL_OUTPUTS:
        raise ValueError(f"Unsupported slice_name {slice_name!r}. Expected one of {sorted(DIAGNOSTIC_LOG_EVAL_OUTPUTS)}")
    return posixpath.join(raw_root, DIAGNOSTIC_LOG_EVAL_OUTPUTS[slice_name])


def materialize_ghalogs_eval_slice(
    *,
    raw_root: str,
    source_path: str,
    max_files: int = 4,
    max_rows: int = 4_000,
    max_bytes: int = 8_000_000,
) -> DiagnosticLogMaterializationStats:
    """Build the `ghalogs` long-tail eval slice under ``raw_root``."""
    return materialize_ghalogs_eval_sample(
        source_path=source_path,
        output_path=diagnostic_log_eval_output_path(raw_root, slice_name="ghalogs"),
        max_files=max_files,
        max_rows=max_rows,
        max_bytes=max_bytes,
    )


def materialize_loghub_apache_eval_slice(
    *,
    raw_root: str,
    source_path: str,
    max_files: int = 4,
    max_rows: int = 4_000,
    max_bytes: int = 8_000_000,
) -> DiagnosticLogMaterializationStats:
    """Build the `loghub_apache` long-tail eval slice under ``raw_root``."""
    return materialize_loghub_eval_sample(
        source_path=source_path,
        output_path=diagnostic_log_eval_output_path(raw_root, slice_name="loghub_apache"),
        max_files=max_files,
        max_rows=max_rows,
        max_bytes=max_bytes,
    )
