# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atomic records and checksum-addressed bundles for preparation and result delivery."""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import zstandard
from fsspec.implementations.local import LocalFileSystem
from rigging.filesystem import atomic
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

SERVER_METRICS_RELATIVE_PATH = Path("aiperf_artifacts/server_metrics_export.json")


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_record(uri: str) -> dict:
    # Bucket routing supplies CoreWeave's endpoint and virtual-host addressing.
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "rt") as stream:
        return json.load(stream)


def write_record(uri: str, record: dict) -> None:
    filesystem, path = filesystem_for(uri)
    if isinstance(filesystem, LocalFileSystem):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with atomic.atomic_rename(path, filesystem=filesystem) as temporary:
        # fsspec.open's automatic mkdir issues CreateBucket on S3. The existing
        # bucket is infrastructure; writing an object must not try to create it.
        with filesystem.open(temporary, "wt") as stream:
            json.dump(record, stream, indent=2)


def copy_file(source: str, destination: str) -> None:
    """Commit a complete file before publishing a receipt that points to it."""
    source_fs, source_path = filesystem_for(source)
    destination_fs, destination_path = filesystem_for(destination)
    if isinstance(destination_fs, LocalFileSystem):
        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
    with atomic.atomic_rename(destination_path, filesystem=destination_fs) as temporary:
        with source_fs.open(source_path, "rb") as reader, destination_fs.open(temporary, "wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)


def unpack(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with archive.open("rb") as encoded, zstandard.ZstdDecompressor().stream_reader(encoded) as decoded:
        with tarfile.open(fileobj=decoded, mode="r|") as contents:
            contents.extractall(destination, filter="data")


def retain_artifacts(artifacts: Path, bundle: Path, prefix: str, *, name: str) -> tuple[str, str]:
    """Archive output under its checksum so earlier result descriptors remain readable."""
    with bundle.open("wb") as encoded, zstandard.ZstdCompressor().stream_writer(encoded) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as contents:
            contents.add(artifacts, arcname=".")
    digest = file_digest(bundle)
    uri = prefix_join(prefix, f"{name}-{digest}.tar.zst")
    copy_file(str(bundle), uri)
    return uri, digest


def pack_server_metrics(point: Path) -> None:
    """Keep lossless server metrics within Iris's uncompressed task-output limit."""
    source = point / SERVER_METRICS_RELATIVE_PATH
    if not source.is_file():
        return
    encoded = source.with_suffix(".json.zst")
    expected = file_digest(source)
    with source.open("rb") as reader, encoded.open("wb") as writer:
        with zstandard.ZstdCompressor().stream_writer(writer) as compressed:
            shutil.copyfileobj(reader, compressed, length=8 * 1024**2)
    with encoded.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as decoded:
        if hashlib.file_digest(decoded, "sha256").hexdigest() != expected:
            raise ValueError("Compressed server metrics differ from the original bytes")
    receipt = {
        "raw_sha256": expected,
        "raw_bytes": source.stat().st_size,
        "compressed_sha256": file_digest(encoded),
        "compressed_bytes": encoded.stat().st_size,
        "codec": "zstandard",
    }
    encoded.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    source.unlink()


def restore_server_metrics(point: Path) -> None:
    """Restore the original metrics bytes for the unchanged result processor."""
    source = point / SERVER_METRICS_RELATIVE_PATH
    encoded = source.with_suffix(".json.zst")
    if not encoded.exists():
        return
    receipt = json.loads(encoded.with_suffix(".receipt.json").read_text())
    if file_digest(encoded) != receipt["compressed_sha256"]:
        raise ValueError("Compressed server metrics checksum mismatch")
    if source.exists():
        if source.stat().st_size != receipt["raw_bytes"] or file_digest(source) != receipt["raw_sha256"]:
            raise ValueError("Restored server metrics checksum mismatch")
        return
    temporary = source.with_suffix(".json.tmp")
    with encoded.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as decoded:
        with temporary.open("wb") as writer:
            shutil.copyfileobj(decoded, writer, length=8 * 1024**2)
    if temporary.stat().st_size != receipt["raw_bytes"] or file_digest(temporary) != receipt["raw_sha256"]:
        raise ValueError("Restored server metrics checksum mismatch")
    temporary.replace(source)
