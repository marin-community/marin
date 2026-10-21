# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream frozen selection records and hash their complete artifact inventories."""

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

POOL_SCHEMA = pa.schema(
    [
        (
            name,
            (
                pa.float64()
                if name in {"inclusion_probability", "previous_pool_inclusion_probability", "retention_probability"}
                else pa.string()
            ),
        )
        for name in (
            "source_id",
            "task_id",
            "contract_id",
            "public_content_id",
            "source_contract_id",
            "source_public_content_id",
            "difficulty",
            "intended_use",
            "population",
            "prior_admission_status",
            "task_json",
            "lowered_task_spec",
            "contract_kind",
            "family",
            "inclusion_probability",
            "previous_pool_inclusion_probability",
            "retention_probability",
            "sampling_stratum",
            "native_format",
            "native_family",
            "native_control",
            "native_prefix",
            "tokenizer_model",
            "tokenizer_revision",
            "runtime_compatibility",
            "teacher_route",
        )
    ]
)
BATCH_ROWS = 128
ARTIFACT_MANIFEST = "artifact-manifest.json"


def file_sha256(path: StoragePath) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_pool(records: Iterable[Mapping[str, Any]], root: StoragePath) -> int:
    """Write identical JSONL/Parquet rows with explicit nullable native columns."""
    count = 0
    pending = []
    with (root / "tasks.jsonl").open("w") as text, (root / "train.parquet").open("wb") as binary:
        with pq.ParquetWriter(binary, POOL_SCHEMA, compression="zstd") as writer:
            for record in records:
                if set(record) - set(POOL_SCHEMA.names):
                    raise ValueError("Selection record has fields outside the frozen pool schema")
                row = {name: record.get(name) for name in POOL_SCHEMA.names}
                text.write(json.dumps(row, sort_keys=True) + "\n")
                pending.append(row)
                count += 1
                if len(pending) == BATCH_ROWS:
                    writer.write_table(pa.Table.from_pylist(pending, schema=POOL_SCHEMA))
                    pending.clear()
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=POOL_SCHEMA))
    return count


def _manifest_root(root: StoragePath) -> StoragePath:
    if root.is_remote:
        return root
    local = replace(root, scheme=None, netloc="")
    return StoragePath(str(Path(str(local)).resolve()))


def write_manifest(root: StoragePath) -> dict:
    """Write the complete file inventory last; it is not a remote publication receipt."""
    root = _manifest_root(root)
    files = []
    paths = [directory / name for directory, _, names in root.walk() for name in names]
    for path in sorted(paths, key=str):
        if path == root / ARTIFACT_MANIFEST:
            continue
        with path.open("rb") as stream:
            size = 0
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
        files.append({"path": path.relative_to(root), "size": size, "sha256": digest.hexdigest()})
    manifest = {"format": 1, "files": files}
    (root / ARTIFACT_MANIFEST).write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def verify_manifest(root: StoragePath, expected_sha256: str) -> None:
    """Verify a pinned artifact inventory before consuming its payloads."""
    root = _manifest_root(root)
    path = root / ARTIFACT_MANIFEST
    if file_sha256(path) != expected_sha256:
        raise ValueError("Artifact manifest differs from its immutable pin")
    manifest = json.loads(path.read_bytes())
    if manifest["format"] != 1:
        raise ValueError("Unknown artifact manifest format")
    listed = [entry["path"] for entry in manifest["files"]]
    actual = {(directory / name).relative_to(root) for directory, _, names in root.walk() for name in names}
    actual.discard(ARTIFACT_MANIFEST)
    if len(listed) != len(set(listed)) or set(listed) != actual:
        raise ValueError("Artifact inventory differs from its complete manifest")
    for entry in manifest["files"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact payload must remain inside its source root")
        payload = root / entry["path"]
        if payload.size() != entry["size"] or file_sha256(payload) != entry["sha256"]:
            raise ValueError(f"Artifact payload differs from its manifest: {entry['path']}")
