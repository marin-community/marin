# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable archives compatible with the tool-conversion receipt index."""

import gzip
import io
import json
import subprocess
import tarfile
import time
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.tool_conversion.export import digest


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def provenance(catalog_sha256: str, tokenizer_sha256: str) -> dict:
    directory = Path(__file__).parent
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True, check=False)
    return {
        "git_head": git.stdout.strip() if git.returncode == 0 else None,
        "git_head_source": "git" if git.returncode == 0 else "unavailable",
        "files_sha256": {p.name: digest(p.read_bytes()) for p in sorted(directory.iterdir()) if p.is_file()},
        "missing_required_files": [],
        "catalog_sha256": catalog_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "generated_at": time.time(),
        "note": "Per-file hashes are authoritative. Historical observations were not reexecuted.",
    }


def read_json(url: str) -> dict:
    return json.loads(StoragePath(url).read_bytes())


def write_immutable(url: str, content: bytes) -> None:
    path = StoragePath(url)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing immutable artifact differs: {url}")
        return
    path.parent.mkdirs()
    path.write_bytes(content)


def archive_bytes(files: dict[str, bytes]) -> bytes:
    """Pack stable members without retaining local copies or deleting artifacts."""
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", compresslevel=1, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, content in sorted(files.items()):
                member = tarfile.TarInfo(name)
                member.size = len(content)
                member.mtime = 0
                archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


def commit_batch(
    prefix: str, name: str, files: dict[str, bytes], catalog_sha256: str, cursor: dict, run_provenance: dict
) -> dict:
    """Upload an archive, then its receipt; consumers admit only receipted batches."""
    data = archive_bytes(files)
    archive_uri = prefix_join(prefix, f"batches/{name}.tar.gz")
    write_immutable(archive_uri, data)
    cursor = {**cursor, "artifact_bytes": cursor.get("artifact_bytes_before_batch", 0) + len(data)}
    receipt = {
        "batch": name,
        "archive_uri": archive_uri,
        "archive_sha256": digest(data),
        "archive_bytes": len(data),
        "catalog_sha256": catalog_sha256,
        "post_commit_cursor": cursor,
        "provenance": run_provenance,
        "at": time.time(),
    }
    write_immutable(prefix_join(prefix, f"receipts/{name}.json"), json_bytes(receipt))
    StoragePath(prefix_join(prefix, "cursor.json")).write_bytes(json_bytes(cursor))
    return receipt
