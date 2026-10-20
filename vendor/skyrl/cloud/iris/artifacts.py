"""Artifact storage and materialization."""

from __future__ import annotations

import json
import os
import posixpath
import shutil
import tempfile
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from fsspec.spec import AbstractFileSystem
from marinskyrl.remote_io import filesystem_and_path, open_output_stream
from marinskyrl.resource_locator import join_resource_path, relative_resource_path

CHECKPOINT_MARKER_FILENAME = "latest_ckpt_global_step.txt"
SOURCE_MANIFEST_FILENAME = ".marinskyrl-source.json"
FILE_COPY_WORKERS = 16


@dataclass(frozen=True)
class ArtifactSource:
    uri: str
    identity: str
    local_path: str


@dataclass(frozen=True)
class FileEntry:
    path: str
    size: int


@dataclass(frozen=True)
class MaterializedArtifact:
    source: ArtifactSource
    files: tuple[FileEntry, ...]


def fs_and_path(uri: str) -> tuple[AbstractFileSystem, str]:
    """Resolve a URI through the shared guarded remote-I/O factory."""
    return filesystem_and_path(uri)


def write_json(uri: str, value: dict[str, Any], *, overwrite: bool = True) -> None:
    """Write JSON to a local or object-store URI."""
    filesystem, path = fs_and_path(uri)
    if not overwrite and filesystem.exists(path):
        raise ValueError(f"JSON artifact already exists: {uri}")
    parent = posixpath.dirname(path)
    if parent:
        filesystem.makedirs(parent, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with open_output_stream(filesystem, path) as destination:
        destination.write(payload)


def read_json(uri: str) -> dict[str, Any] | None:
    """Read a JSON object, returning ``None`` when the URI does not exist."""
    filesystem, path = fs_and_path(uri)
    if not filesystem.exists(path):
        return None
    with filesystem.open(path) as source:
        return json.load(source)


def terminal_checkpoint_step(checkpoint_root: str) -> int:
    """Return the latest committed checkpoint step."""
    marker_uri = join_resource_path(checkpoint_root, CHECKPOINT_MARKER_FILENAME)
    filesystem, marker_path = fs_and_path(marker_uri)
    if not filesystem.exists(marker_path):
        raise ValueError(f"Successful Iris job did not commit a checkpoint marker: {marker_uri}")
    with filesystem.open(marker_path, "r") as source:
        return int(source.read().strip())


def file_inventory(filesystem: AbstractFileSystem, root: str) -> tuple[tuple[str, FileEntry], ...]:
    """List files below a storage root using metadata returned by the listing."""
    files = filesystem.find(root, detail=True)
    return tuple(
        sorted(
            (
                path,
                FileEntry(path=relative_resource_path(root, path), size=int(info["size"])),
            )
            for path, info in files.items()
            if info["type"] == "file"
        )
    )


def _source_inventory(uri: str) -> tuple[AbstractFileSystem, tuple[tuple[str, FileEntry], ...]]:
    filesystem, source_path = fs_and_path(uri)
    source_info = filesystem.info(source_path)
    if source_info["type"] == "file":
        entry = FileEntry(path=posixpath.basename(source_path), size=int(source_info["size"]))
        return filesystem, ((source_path, entry),)
    inventory = file_inventory(filesystem, source_path)
    if not inventory:
        raise ValueError(f"Artifact source contains no files: {uri}")
    return filesystem, inventory


def copy_file_inventory(
    filesystem: AbstractFileSystem,
    inventory: tuple[tuple[str, FileEntry], ...],
    destination: Path,
) -> tuple[FileEntry, ...]:
    """Copy a selected remote file inventory and verify each recorded size."""
    destination.mkdir(parents=True, exist_ok=False)
    if not inventory:
        return ()

    def copy_file(item: tuple[str, FileEntry]) -> None:
        source_path, entry = item
        local_path = destination / entry.path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        filesystem.get_file(source_path, str(local_path))
        actual_size = local_path.stat().st_size
        if actual_size != entry.size:
            raise ValueError(f"Staging size mismatch for {entry.path}: expected {entry.size}, found {actual_size}")

    with ThreadPoolExecutor(max_workers=min(FILE_COPY_WORKERS, len(inventory))) as executor:
        tuple(executor.map(copy_file, inventory))
    return tuple(entry for _, entry in inventory)


def copy_tree(source_uri: str, destination: Path) -> tuple[FileEntry, ...]:
    """Copy an object-store tree to an empty local directory and verify file sizes."""
    filesystem, inventory = _source_inventory(source_uri)
    return copy_file_inventory(filesystem, inventory, destination)


@contextmanager
def atomic_directory_update(target: Path, *, staging_prefix: str) -> Iterator[Path]:
    """Build and atomically install a replacement directory, restoring failures."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=staging_prefix, dir=target.parent))
    staging.rmdir()
    backup: Path | None = None
    try:
        yield staging
        if target.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.old-", dir=target.parent))
            backup.rmdir()
            os.replace(target, backup)
        os.replace(staging, target)
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def _materialization_matches(target: Path, source: ArtifactSource, inventory: tuple[FileEntry, ...]) -> bool:
    manifest_path = target / SOURCE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    expected = {
        "source_uri": source.uri,
        "source_identity": source.identity,
        "files": [asdict(entry) for entry in inventory],
    }
    if manifest != expected:
        return False
    expected_paths = {entry.path for entry in inventory}
    local_paths = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file() and path.name != SOURCE_MANIFEST_FILENAME
    }
    return local_paths == expected_paths and all(
        (target / entry.path).stat().st_size == entry.size for entry in inventory
    )


def materialize(source: ArtifactSource) -> MaterializedArtifact:
    """Materialize one immutable artifact into its declared node-local path."""
    filesystem, remote_inventory = _source_inventory(source.uri)
    return materialize_inventory(source, filesystem, remote_inventory)


def materialize_inventory(
    source: ArtifactSource,
    filesystem: AbstractFileSystem,
    remote_inventory: tuple[tuple[str, FileEntry], ...],
) -> MaterializedArtifact:
    """Materialize a selected remote inventory into its declared node-local path."""
    inventory = tuple(entry for _, entry in remote_inventory)
    target = Path(source.local_path).resolve()
    if _materialization_matches(target, source, inventory):
        return MaterializedArtifact(source=source, files=inventory)

    with atomic_directory_update(target, staging_prefix=f".{target.name}.staging-") as staging:
        copied_inventory = copy_file_inventory(filesystem, remote_inventory, staging)
        if copied_inventory != inventory:
            raise ValueError(f"Artifact source changed while it was being staged: {source.uri}")
        manifest = {
            "source_uri": source.uri,
            "source_identity": source.identity,
            "files": [asdict(entry) for entry in inventory],
        }
        (staging / SOURCE_MANIFEST_FILENAME).write_text(json.dumps(manifest, sort_keys=True))
    return MaterializedArtifact(source=source, files=inventory)
