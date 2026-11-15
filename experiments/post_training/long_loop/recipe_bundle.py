# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture the shipped runtime and verify an immutable archive-stage recipe bundle."""

import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

from iris.cluster.client.bundle import collect_workspace_files
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.long_loop.archive import ArchiveInput, digest, write_once

RECIPE_CATEGORIES = {"source", "config", "dataset"}


def require_clean_revision(workspace: Path, revision: str) -> None:
    """Refuse production submission when tracked or untracked source differs."""
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()
    if actual != revision:
        raise ValueError("Production source revision differs from the checked-out commit")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=workspace, text=True
    ).strip()
    if dirty:
        raise ValueError("Production submission requires a clean source tree, including untracked files")


def runtime_source_manifest(workspace: Path, revision: str) -> dict:
    """Pin every file shipped by the canonical Iris workspace collector."""
    files = []
    for path in collect_workspace_files(workspace):
        size, sha = digest(str(path))
        files.append({"path": path.relative_to(workspace).as_posix(), "size": size, "sha256": sha})
    if not files:
        raise ValueError("Runtime source snapshot is empty")
    return {"schema": 1, "marin_revision": revision, "files": files}


def _member_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name or not path.parts:
        raise ValueError(f"Invalid recipe member path: {name}")
    return name


def write_recipe_bundle(
    output: str, inputs: list[ArchiveInput], source_manifest: dict, workspace: Path
) -> tuple[list[ArchiveInput], dict]:
    """Write and verify source bytes plus every consumed config and dataset referent."""
    entries = []
    payloads = []
    if not source_manifest["files"]:
        raise ValueError("Recipe bundles require a nonempty runtime source snapshot")
    for item in source_manifest["files"]:
        relative = _member_name(item["path"])
        source = workspace / relative
        if digest(str(source)) != (item["size"], item["sha256"]):
            raise ValueError(f"Runtime source changed since graph construction: {relative}")
        entries.append({"name": f"source/{relative}", "category": "source", **item})
        payloads.append(str(source))
    for item in sorted(inputs, key=lambda item: item.name):
        if item.category not in RECIPE_CATEGORIES:
            continue
        name = "evidence/" + _member_name(item.name)
        size, sha = digest(item.source)
        entries.append({"name": name, "category": item.category, "source": item.source, "size": size, "sha256": sha})
        payloads.append(item.source)
    names = [item["name"] for item in entries]
    if len(names) != len(set(names)):
        raise ValueError("Recipe bundle contains duplicate member paths")
    manifest = {"schema": 1, "marin_revision": source_manifest["marin_revision"], "files": entries}
    raw = json.dumps(manifest, sort_keys=True, indent=2).encode()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    bundle = StoragePath(output) / "recipe.tar.gz"
    # A local temporary byte buffer is bounded by the shipped source plus small
    # config/catalog referents. Checkpoint and trajectory payloads are excluded.
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for entry, source in zip(entries, payloads, strict=True):
                content = StoragePath(source).read_bytes()
                if (len(content), hashlib.sha256(content).hexdigest()) != (entry["size"], entry["sha256"]):
                    raise ValueError(f"Recipe input changed while bundling: {source}")
                member = tarfile.TarInfo(entry["name"])
                member.size = len(content)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(content))
    write_once(str(bundle), buffer.getvalue())
    write_once(prefix_join(output, "manifest.json"), raw)
    receipt = verify_recipe_bundle(output, manifest_sha)
    write_once(prefix_join(output, "VERIFIED.json"), json.dumps(receipt, sort_keys=True, indent=2).encode())
    return [
        ArchiveInput(prefix_join(output, name), prefix_join("recipe", name), "recipe")
        for name in ("recipe.tar.gz", "manifest.json", "VERIFIED.json")
    ], receipt


def verify_recipe_bundle(root: str, manifest_sha256: str) -> dict:
    """Verify the pinned manifest, exact member set, and every member's bytes."""
    raw = (StoragePath(root) / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("Recipe manifest SHA256 mismatch")
    manifest = json.loads(raw)
    expected = {_member_name(item["name"]): item for item in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("Recipe manifest has duplicate member names")
    seen = set()
    with (StoragePath(root) / "recipe.tar.gz").open("rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                name = _member_name(member.name)
                if not member.isfile() or name not in expected or name in seen:
                    raise ValueError(f"Unexpected recipe member: {name}")
                content = archive.extractfile(member).read()
                item = expected[name]
                if (len(content), hashlib.sha256(content).hexdigest()) != (item["size"], item["sha256"]):
                    raise ValueError(f"Recipe member failed verification: {name}")
                seen.add(name)
    if seen != set(expected):
        raise ValueError("Recipe bundle is missing manifest members")
    size, checksum = digest(prefix_join(root, "recipe.tar.gz"))
    return {
        "schema": 1,
        "manifest_sha256": manifest_sha256,
        "bundle_sha256": checksum,
        "bundle_bytes": size,
        "files": len(seen),
        "marin_revision": manifest["marin_revision"],
    }
