# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verified, resumable experiment archives and manifest-scoped checkpoint cleanup."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.transfer import copy


@dataclass(frozen=True)
class ArchiveInput:
    source: str
    name: str
    category: str


@dataclass(frozen=True)
class ArchivedObject:
    source: str
    destination: str
    category: str
    size: int
    sha256: str


def digest(uri: str) -> tuple[int, str]:
    fs, key = filesystem_for(uri)
    hashed = hashlib.sha256()
    size = 0
    with fs.open(key, "rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hashed.update(chunk)
            size += len(chunk)
    return size, hashed.hexdigest()


def write_once(uri: str, content: bytes) -> None:
    fs, key = filesystem_for(uri)
    if fs.exists(key):
        if fs.cat_file(key) != content:
            raise ValueError(f"Immutable archive collision: {uri}")
        return
    parent = key.rsplit("/", 1)[0]
    fs.makedirs(parent, exist_ok=True)
    fs.pipe_file(key, content)


def archive_object(item: ArchiveInput, root: str) -> ArchivedObject:
    name = PurePosixPath(item.name)
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise ValueError("Archive names must be relative and cannot traverse parents")
    destination = root.rstrip("/") + "/objects/" + item.name
    size, checksum = digest(item.source)
    fs, key = filesystem_for(destination)
    if fs.exists(key):
        if digest(destination) != (size, checksum):
            raise ValueError(f"Archive payload differs: {destination}")
    else:
        copy(item.source, destination, no_clobber=True)
        if digest(destination) != (size, checksum):
            raise ValueError(f"Archive verification failed: {destination}")
    return ArchivedObject(item.source, destination, item.category, size, checksum)


def seal_archive(root: str, inputs: list[ArchiveInput], provenance: dict) -> dict:
    """Copy exact source bytes, verify every payload, then publish the completion marker."""
    names = [str(PurePosixPath(item.name)) for item in inputs]
    if len(names) != len(set(names)):
        raise ValueError("Archive inputs have duplicate destination names")
    objects = [archive_object(item, root) for item in inputs]
    manifest = {"schema": 1, "provenance": provenance, "objects": [asdict(obj) for obj in objects]}
    raw = json.dumps(manifest, sort_keys=True, indent=2).encode()
    write_once(root + "/manifest.json", raw)
    token = hashlib.sha256(raw).hexdigest()
    write_once(root + "/VERIFIED.json", json.dumps({"manifest_sha256": token}, sort_keys=True).encode())
    return manifest


def verify_archive(root: str) -> dict:
    fs, key = filesystem_for(root)
    raw = fs.cat_file(key + "/manifest.json")
    marker = json.loads(fs.cat_file(key + "/VERIFIED.json"))
    if hashlib.sha256(raw).hexdigest() != marker["manifest_sha256"]:
        raise ValueError("Archive manifest does not match completion marker")
    manifest = json.loads(raw)
    for obj in manifest["objects"]:
        if digest(obj["destination"]) != (obj["size"], obj["sha256"]):
            raise ValueError(f"Archived object failed verification: {obj['destination']}")
    return manifest


def checkpoint_cleanup(root: str, checkpoint_root: str, selected: str, candidates: list[str]) -> None:
    """Delete inventoried superseded weights after verifying the archived selected checkpoint."""
    manifest = verify_archive(root)
    archived = {obj["source"]: obj for obj in manifest["objects"]}
    inventory = manifest["provenance"]["checkpoint_inventory"]
    selected_files = inventory["selected_files"]
    disposable = {obj["source"]: obj for obj in inventory["disposable_files"]}
    if inventory["selected"] != selected or inventory["root"] != checkpoint_root:
        raise ValueError("Cleanup scope differs from sealed inventory")
    prefix = checkpoint_root.rstrip("/") + "/"
    keep = selected.rstrip("/") + "/"
    if not keep.startswith(prefix) or keep == prefix:
        raise ValueError("Selected checkpoint must be a child of this experiment's checkpoint root")
    if not selected_files:
        raise ValueError("Selected checkpoint inventory is empty")
    selected_fs, selected_key = filesystem_for(selected)
    observed_keys = set(selected_fs.find(selected_key))
    expected_keys = {filesystem_for(obj["source"])[1] for obj in selected_files}
    if observed_keys != expected_keys:
        raise ValueError("Selected checkpoint inventory does not cover its complete payload")
    for obj in selected_files:
        uri = obj["source"]
        if not uri.startswith(keep) or uri not in archived:
            raise ValueError("Selected checkpoint is not completely archived")
        if (obj["size"], obj["sha256"]) != (archived[uri]["size"], archived[uri]["sha256"]):
            raise ValueError("Selected checkpoint differs from sealed inventory")
    planned = []
    for uri in candidates:
        if ".." in PurePosixPath(uri).parts or not uri.startswith(prefix) or uri.startswith(keep) or uri == selected:
            raise ValueError(f"Cleanup target is outside scope or selected: {uri}")
        if uri not in disposable:
            raise ValueError(f"Cleanup requires a sealed checkpoint inventory: {uri}")
        fs, key = filesystem_for(uri)
        if fs.exists(key) and digest(uri) != (disposable[uri]["size"], disposable[uri]["sha256"]):
            raise ValueError(f"Checkpoint changed since archive: {uri}")
        planned.append((uri, fs, key))
    write_once(
        root + "/cleanup-plan.json", json.dumps({"selected": selected, "objects": candidates}, sort_keys=True).encode()
    )
    for _, fs, key in planned:
        if fs.exists(key):
            fs.rm(key)
    fs, key = filesystem_for(root + "/cleanup-completed.json")
    fs.pipe_file(
        key,
        json.dumps(
            {"selected": selected, "removed": candidates, "completed_at": datetime.now(UTC).isoformat()}
        ).encode(),
    )
