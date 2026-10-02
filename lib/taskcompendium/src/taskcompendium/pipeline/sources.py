# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode records from pinned source files staged by an acquisition artifact."""

import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import asdict, is_dataclass
from tempfile import TemporaryDirectory, TemporaryFile
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


def _generated_rows(archive_path: StoragePath) -> Iterator[dict[str, Any]]:
    """Run the pinned generator checkout without importing it into the audit worker."""
    with TemporaryDirectory() as directory:
        local_archive = os.path.join(directory, "generator.tar.gz")
        with archive_path.open("rb") as source, open(local_archive, "wb") as destination:
            shutil.copyfileobj(source, destination)
        with tarfile.open(local_archive, mode="r:gz") as archive:
            roots = {member.name.split("/", 1)[0] for member in archive if member.name}
            if len(roots) != 1:
                raise ValueError("Pinned generator archive has multiple roots")
            archive.extractall(directory, filter="data")
        root = os.path.join(directory, roots.pop())
        environment = {
            **os.environ,
            "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "MPLCONFIGDIR": directory,
        }
        with TemporaryFile(mode="w+t") as errors:
            process = subprocess.Popen(
                [sys.executable, "-m", "taskcompendium.pipeline.reasoning_gym_source"],
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                env=environment,
            )
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    yield json.loads(line)
                if process.wait() != 0:
                    errors.seek(0)
                    raise RuntimeError(f"Pinned reasoning-gym generator failed: {errors.read()}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()


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
    elif source_format == SourceFormat.XML:
        with path.open("rb") as stream:
            root = ET.parse(stream).getroot()
        yield from ({**item.attrib, **{child.tag: child.text or "" for child in item}} for item in root.iter("Problem"))
    elif source_format == SourceFormat.GENERATED:
        yield from _generated_rows(path)
    else:
        raise ValueError(f"Unsupported staged source format: {source_format}")


def staged_file_rows(path: str, relative_file: str, spec: SourceFiles) -> Iterator[dict[str, Any]]:
    """Yield selected records with a stable original file and row locator."""
    if relative_file.startswith("/") or ".." in relative_file.split("/"):
        raise ValueError(f"Source file must be relative to its staged root: {relative_file}")
    root = StoragePath(path)
    file = root / relative_file
    for index, row in enumerate(_decoded_rows(file, spec.format)):
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {relative_file}:{index}")
        if spec.selector is not None and not spec.selector(row, root):
            continue
        data = spec.decoder(row, root) if spec.decoder is not None else row
        yield {"index": index, "locator": f"{relative_file}:{index}", "data": data}


def unpack_task_binary(row: dict[str, Any], _staged_root: StoragePath) -> dict[str, Any]:
    """Expose Harbor task files in the form consumed by task converters."""
    blob = row["task_binary"]
    if not isinstance(blob, bytes):
        raise ValueError("Task binary must contain archived bytes")
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
        for member in archive:
            if member.isfile():
                handle = archive.extractfile(member)
                if handle is not None:
                    files[member.name.removeprefix("./")] = handle.read()
    prepared = {
        "path": row["path"],
        "instruction": files["instruction.md"].decode(),
        "files": {name: base64.b64encode(data).decode() for name, data in files.items()},
        "archive_sha256": hashlib.sha256(blob).hexdigest(),
    }
    if "tests/verifier_data.json" in files:
        prepared["verifier_data"] = json.loads(files["tests/verifier_data.json"])
    return prepared
