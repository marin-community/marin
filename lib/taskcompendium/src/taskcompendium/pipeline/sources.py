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
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from functools import lru_cache
from tempfile import TemporaryDirectory, TemporaryFile
from typing import Any, cast

from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_jsonl, load_parquet


class SourceFormat(StrEnum):
    PARQUET = "parquet"
    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    GENERATED = "generated"


@dataclass(frozen=True)
class SourceFiles:
    """Relative file selection and record decoding for a staged release."""

    patterns: tuple[str, ...]
    format: SourceFormat
    selector: Callable[[dict[str, Any]], bool] | None = None
    decoder: Callable[[dict[str, Any], StoragePath], dict[str, Any]] | None = None


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
    identity = {
        "revision": "1",
        "patterns": spec.patterns,
        "format": spec.format.value,
        "selector": _callable_identity(spec.selector),
        "decoder": _callable_identity(spec.decoder),
    }
    if spec.decoder is resolve_ultra_placeholder:
        identity["placeholder_pins"] = PLACEHOLDER_PINS
        identity["placeholder_files"] = PLACEHOLDER_FILES
        identity["placeholder_splits"] = PLACEHOLDER_SPLITS
    return identity


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
    gym_ids = None
    if isinstance(spec.selector, UltraComponentSelector) and spec.selector.family == "swe-repo":
        membership = root / SWE_GYM_MEMBERSHIP_FILE
        gym_ids = frozenset(row["instance_id"] for row in load_parquet(str(membership)))
    for index, row in enumerate(_decoded_rows(file, spec.format)):
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {relative_file}:{index}")
        if isinstance(spec.selector, UltraComponentSelector):
            if not spec.selector.matches(row, gym_ids):
                continue
        elif spec.selector is not None and not spec.selector(row):
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


def select_eurus_code(row: dict[str, Any]) -> bool:
    return row["ability"] == "code"


@dataclass(frozen=True)
class UltraComponentSelector:
    """Select one Ultra blend component using its release row descriptor."""

    selector: str
    component: str
    family: str

    def __call__(self, row: dict[str, Any]) -> bool:
        return self.matches(row, None)

    def matches(self, row: dict[str, Any], gym_ids: frozenset[str] | None) -> bool:
        identity = row.get("dataset") or "agent:" + row["agent_ref"]["name"]
        if identity != self.selector:
            return False
        if self.family == "swe-repo":
            if gym_ids is None:
                raise ValueError("SWE component selection requires the pinned SWE-Gym instance inventory")
            gym_member = row["metadata"]["instance_id"] in gym_ids
            return gym_member == self.component.endswith("/SWE-Gym/SWE-Gym")
        return True


SWE_GYM_MEMBERSHIP_FILE = "swe-gym-membership/data/train-00000-of-00001.parquet"

PLACEHOLDER_PINS = {
    "BytedTsinghua-SIA/DAPO-Math-17k": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
    "Skywork/Skywork-OR1-RL-Data": "1cdedc52e0e2db85fdf252f9be682e63a5a38c33",
}
PLACEHOLDER_FILES = {
    "BytedTsinghua-SIA/DAPO-Math-17k": "placeholder-dapo/data/dapo-math-17k.parquet",
    "Skywork/Skywork-OR1-RL-Data": "placeholder-skywork/data/math-00000-of-00001.parquet",
}
PLACEHOLDER_SPLITS = {
    "BytedTsinghua-SIA/DAPO-Math-17k": "train",
    "Skywork/Skywork-OR1-RL-Data": "math",
}


@lru_cache(maxsize=1024)
def _placeholder_source(root: StoragePath, dataset: str, split: str, index: int) -> dict[str, Any]:
    pin = PLACEHOLDER_PINS[dataset]
    if split != PLACEHOLDER_SPLITS[dataset]:
        raise ValueError(f"Unsupported placeholder split {dataset}/{split}")
    path = root / PLACEHOLDER_FILES[dataset]
    records = list(load_parquet(InputFileSpec(path=str(path), row_start=index, row_end=index + 1)))
    if len(records) != 1:
        raise ValueError(f"Pinned placeholder file omitted {dataset}/{split}/{index}")
    record = records[0]
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {
        "dataset": dataset,
        "revision": pin,
        "split": split,
        "row_index": index,
        "record_sha256": digest,
        "record": record,
    }


def resolve_ultra_placeholder(row: dict[str, Any], staged_root: StoragePath) -> dict[str, Any]:
    """Retain pinned upstream evidence needed to reconstruct Ultra math placeholders."""
    placeholder = row.get("_hf_question_placeholder")
    if placeholder is None:
        return row
    source = _placeholder_source(staged_root, placeholder["dataset"], placeholder["split"], int(placeholder["row"]))
    return {**row, "placeholder_source": source}
