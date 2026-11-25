# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verified, resumable experiment archives and manifest-scoped checkpoint cleanup."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import repeat
from pathlib import PurePosixPath

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.filesystem.transfer import copy

from experiments.post_training.long_loop.hf_archive import file_identity, verify_hf_checkpoint
from experiments.post_training.long_loop.verify_final_sft_checkpoint import verify_final_checkpoint

ARCHIVE_COPY_WORKERS = 16


@dataclass(frozen=True)
class ArchiveInput:
    source: str
    name: str
    category: str


PERMANENT_TMP_SEGMENT = "tmp"


@dataclass(frozen=True)
class ArchivedObject:
    source: str
    destination: str
    category: str
    size: int
    sha256: str


def digest(uri: str) -> tuple[int, str]:
    hashed = hashlib.sha256()
    size = 0
    with StoragePath(uri).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hashed.update(chunk)
            size += len(chunk)
    return size, hashed.hexdigest()


def write_once(uri: str, content: bytes) -> None:
    path = StoragePath(uri)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Immutable archive collision: {uri}")
        return
    path.parent.mkdirs(exist_ok=True)
    path.write_bytes(content)


def archive_object(item: ArchiveInput, root: str) -> ArchivedObject:
    name = PurePosixPath(item.name)
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise ValueError("Archive names must be relative and cannot traverse parents")
    destination = prefix_join(prefix_join(root, "objects"), item.name)
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
    if any(item.category == "checkpoint" or item.source.endswith((".safetensors", ".distcp")) for item in inputs):
        raise ValueError("Checkpoint payloads belong in Hugging Face, not the evidence archive")
    with ThreadPoolExecutor(max_workers=ARCHIVE_COPY_WORKERS) as pool:
        objects = list(pool.map(archive_object, inputs, repeat(root)))
    manifest = {"schema": 1, "provenance": provenance, "objects": [asdict(obj) for obj in objects]}
    raw = json.dumps(manifest, sort_keys=True, indent=2).encode()
    write_once(prefix_join(root, "manifest.json"), raw)
    token = hashlib.sha256(raw).hexdigest()
    write_once(prefix_join(root, "VERIFIED.json"), json.dumps({"manifest_sha256": token}, sort_keys=True).encode())
    return manifest


def assert_final_sft_checkpoint(output: str, final_sft: dict) -> dict:
    """After archival, the final SFT checkpoint must exist in full at a permanent path."""
    uri = final_sft.get("uri") if isinstance(final_sft, dict) else None
    if not uri:
        raise ValueError("Archive plan must record the final SFT checkpoint")
    receipt = verify_final_checkpoint(uri, expect_step=final_sft.get("step"))
    write_once(
        prefix_join(output, "final-sft-checkpoint.json"),
        json.dumps(receipt, sort_keys=True, indent=2).encode(),
    )
    return receipt


def assert_best_rl_checkpoint(output: str, inventory: dict) -> dict:
    """Check committed trainer state and a nonempty native payload at permanent best/final paths."""
    checked = {}
    for role in ("selected", "final"):
        uri = inventory.get(role)
        if not uri:
            raise ValueError(f"Archive inventory must record the {role} RL checkpoint")
        path = StoragePath(uri)
        if PERMANENT_TMP_SEGMENT in path.segments:
            raise ValueError(f"The {role} RL checkpoint must live at a permanent path, not scratch storage: {uri}")
        step = str(uri).rsplit("global_step_", 1)[-1].split("/", 1)[0]
        if not (path / "trainer_state.pt").exists():
            raise ValueError(f"The {role} RL checkpoint has no committed trainer_state.pt: {uri}")
        fs, key = filesystem_for(uri)
        objects = fs.find(key)
        payload = [item for item in objects if not item.rstrip("/").endswith("trainer_state.pt")]
        if not payload:
            raise ValueError(f"The {role} RL checkpoint holds no weights payload: {uri}")
        checked[role] = {"uri": uri, "step": int(step), "objects": len(objects)}
    receipt = {"verdict": "full", **checked}
    write_once(
        prefix_join(output, "best-rl-checkpoint.json"),
        json.dumps(receipt, sort_keys=True, indent=2).encode(),
    )
    return receipt


def verify_archive(root: str) -> dict:
    raw = (StoragePath(root) / "manifest.json").read_bytes()
    marker = json.loads((StoragePath(root) / "VERIFIED.json").read_bytes())
    if hashlib.sha256(raw).hexdigest() != marker["manifest_sha256"]:
        raise ValueError("Archive manifest does not match completion marker")
    manifest = json.loads(raw)
    for obj in manifest["objects"]:
        if digest(obj["destination"]) != (obj["size"], obj["sha256"]):
            raise ValueError(f"Archived object failed verification: {obj['destination']}")
    return manifest


def checkpoint_cleanup(
    root: str, checkpoint_root: str, selected: str, candidates: list[str], *, protected: list[str] | None = None
) -> None:
    """Delete inventoried superseded weights after verifying HF publication and evidence.

    ``protected`` names checkpoints retained regardless of inventory: the
    final SFT checkpoint is never a cleanup target.
    """
    manifest = verify_archive(root)
    reference = verify_hf_checkpoint(manifest["provenance"]["hf_checkpoint"])
    inventory = manifest["provenance"]["checkpoint_inventory"]
    disposable = {obj["source"]: obj for obj in inventory["disposable_files"]}
    if inventory["selected"] != selected or inventory["root"] != checkpoint_root:
        raise ValueError("Cleanup scope differs from sealed inventory")
    if reference["native_checkpoint"] != selected:
        raise ValueError("HF publication does not match the selected native checkpoint")
    if not inventory["ttl"]:
        raise ValueError("Native checkpoint location and TTL must remain in provenance")
    root_path = StoragePath(checkpoint_root)
    keep_path = StoragePath(selected)
    retained_paths = [StoragePath(path) for path in inventory["retained"]]
    if str(root_path) != checkpoint_root or str(keep_path) != selected:
        raise ValueError("Checkpoint cleanup requires canonical storage paths")
    if not keep_path.relative_to(root_path):
        raise ValueError("Selected checkpoint must be a child of this experiment's checkpoint root")
    guarded = [StoragePath(uri) for uri in protected or []]
    planned = []
    for uri in candidates:
        path = StoragePath(uri)
        if str(path) != uri or ".." in path.segments or not path.relative_to(root_path):
            raise ValueError(f"Cleanup target is outside scope: {uri}")
        if path == keep_path or path.segments[: len(keep_path.segments)] == keep_path.segments:
            raise ValueError(f"Cleanup target is selected: {uri}")
        if any(path.segments[: len(keep.segments)] == keep.segments for keep in retained_paths):
            raise ValueError(f"Cleanup target is retained: {uri}")
        if any(path.segments[: len(guard.segments)] == guard.segments for guard in guarded):
            raise ValueError(f"Cleanup target is a protected checkpoint: {uri}")
        if uri not in disposable:
            raise ValueError(f"Cleanup requires a sealed checkpoint inventory: {uri}")
        fs, key = filesystem_for(uri)
        if fs.exists(key) and file_identity(fs.info(key)) != disposable[uri]["storage_identity"]:
            raise ValueError(f"Checkpoint changed since archive: {uri}")
        planned.append((uri, fs, key))
    write_once(
        prefix_join(root, "cleanup-plan.json"),
        json.dumps({"selected": selected, "objects": candidates}, sort_keys=True).encode(),
    )
    for _, fs, key in planned:
        if fs.exists(key):
            fs.rm(key)
    StoragePath(prefix_join(root, "cleanup-completed.json")).write_bytes(
        json.dumps(
            {"selected": selected, "removed": candidates, "completed_at": datetime.now(UTC).isoformat()}
        ).encode(),
    )
