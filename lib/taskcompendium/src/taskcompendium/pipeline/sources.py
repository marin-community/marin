# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode records from pinned source files staged by an acquisition artifact."""

import csv
import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, is_dataclass
from typing import Any, cast

from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from zephyr.readers import load_jsonl, load_parquet

from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat


def _callable_identity(fn: Callable[..., Any] | None) -> dict[str, Any] | None:
    if fn is None:
        return None
    configured = is_dataclass(fn) and not isinstance(fn, type)
    target = type(fn) if configured else fn
    identity: dict[str, Any] = {"module": target.__module__, "name": target.__qualname__}
    if configured:
        identity["parameters"] = asdict(cast(Any, fn))
    return identity


def source_files_identity(spec: SourceFiles) -> dict[str, Any]:
    """Stable artifact identity for a staged reader and its selection rules."""
    return {
        "revision": "1",
        "patterns": spec.patterns,
        "format": spec.format.value,
        "selector": _callable_identity(spec.selector),
        "decoder": _callable_identity(spec.decoder),
        "reader": _callable_identity(spec.reader),
    }


def staged_files(path: str, spec: SourceFiles) -> tuple[str, ...]:
    """List selected files by pinned relative path, rejecting missing declarations."""
    root = StoragePath(path)
    _, root_path = url_to_fs(path)
    filesystem_root = StoragePath(root_path)
    files = set()
    for pattern in spec.patterns:
        for file in (root / pattern).glob():
            _, file_path = url_to_fs(str(file))
            relative = StoragePath(file_path).relative_to(filesystem_root)
            if not any(part.startswith(".") for part in relative.split("/")):
                files.add(relative)
    if not files:
        raise FileNotFoundError(f"No staged files match {spec.patterns} under {path}")
    return tuple(sorted(files))


def _decoded_rows(path: StoragePath, source_format: SourceFormat) -> Iterator[dict[str, Any]]:
    if source_format == SourceFormat.PARQUET:
        yield from load_parquet(str(path))
    elif source_format == SourceFormat.JSONL:
        yield from load_jsonl(str(path))
    elif source_format == SourceFormat.JSON:
        with path.open("rt") as stream:
            payload = json.load(stream)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON record array in {path}")
        yield from payload
    elif source_format == SourceFormat.CSV:
        with path.open("rt", encoding="utf-8-sig") as stream:
            yield from csv.DictReader(stream)
    else:
        raise ValueError(f"Unsupported staged source format: {source_format}")


def staged_raw_file_rows(path: str, relative_file: str, spec: SourceFiles) -> Iterator[dict[str, Any]]:
    """Yield selected source rows before decoding with their original stable locators."""
    if relative_file.startswith("/") or ".." in relative_file.split("/"):
        raise ValueError(f"Source file must be relative to its staged root: {relative_file}")
    root = StoragePath(path)
    file = root / relative_file
    records = spec.reader(file) if spec.reader is not None else _decoded_rows(file, spec.format)
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {relative_file}:{index}")
        if spec.selector is not None and not spec.selector(row, root):
            continue
        yield {"index": index, "locator": f"{relative_file}:{index}", "data": row}


def decode_staged_row(record: dict[str, Any], path: str, spec: SourceFiles) -> dict[str, Any]:
    """Apply a source decoder to an already selected row without changing its locator."""
    data = spec.decoder(record["data"], StoragePath(path)) if spec.decoder is not None else record["data"]
    return {**record, "data": data}


def staged_file_rows(path: str, relative_file: str, spec: SourceFiles) -> Iterator[dict[str, Any]]:
    """Yield selected, decoded records with their original file and row locators."""
    for record in staged_raw_file_rows(path, relative_file, spec):
        yield decode_staged_row(record, path, spec)
