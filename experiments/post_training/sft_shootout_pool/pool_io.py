# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reading and writing a pool prefix, shared by the build (pool), the exclusions (exclude) and the top-ups (topup).

A pool prefix holds manifest.json, whose file_sha256 lists every other file with its hash. A pass that changes the
pool keeps the manifest it replaces as manifest-v<N>.json.
"""

import hashlib
import io
import json
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

MANIFEST = "manifest.json"
SWESMITH_PACKAGES = "swesmith-packages.parquet"


def kept_manifest(version: int) -> str:
    """File name that keeps pool v`version`'s manifest once a later version replaces it."""
    return f"manifest-v{version}.json"


def parquet_bytes(rows: list[dict]) -> bytes:
    """Parquet in the Harbor package schema (path, task_binary)."""
    buffer = io.BytesIO()
    schema = pa.schema([pa.field("path", pa.string()), pa.field("task_binary", pa.binary())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), buffer)
    return buffer.getvalue()


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def read_jsonl(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode().splitlines()]


def swesmith_revision(prefix: str, files: dict[str, bytes]) -> str:
    """The package_revision of the SWE-smith packages in `files`, to be stored under `prefix`."""
    return f"{prefix.rstrip('/')}/{SWESMITH_PACKAGES}@sha256:{hashlib.sha256(files[SWESMITH_PACKAGES]).hexdigest()}"


class Pool(NamedTuple):
    manifest_bytes: bytes
    manifest: dict
    files: dict[str, bytes]


def read_pool(pool: StoragePath, since: str) -> Pool:
    """Load the manifest and every file it lists, checking each file's sha256; `since` names the build it came from."""
    manifest_bytes = (pool / MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    files = {name: (pool / name).read_bytes() for name in manifest["file_sha256"]}
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != manifest["file_sha256"][name]:
            raise ValueError(f"{name} changed since {since}")
    return Pool(manifest_bytes, manifest, files)


def write_pool(pool: StoragePath, files: dict[str, bytes], manifest: dict) -> None:
    """Write every file, re-hash each upload against the manifest, then write the manifest last."""
    for name, data in files.items():
        (pool / name).write_bytes(data)
        if hashlib.sha256((pool / name).read_bytes()).hexdigest() != manifest["file_sha256"][name]:
            raise ValueError(f"Upload hash mismatch: {name}")
    (pool / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
