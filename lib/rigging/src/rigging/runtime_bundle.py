# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install a pinned machine runtime from a verified archive."""

import fcntl
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_EXTRACTED_BYTES = 4 * 1024**3
MAX_ARCHIVE_FILES = 32768


@dataclass(frozen=True)
class RuntimeBundle:
    manifest_uri: str
    manifest_sha256: str
    archive_uri: str
    archive_sha256: str
    installation_parent: str = "/opt"


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _verify_files(root: Path, files: dict[str, str]) -> None:
    for relative, expected in files.items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Runtime file is missing or outside its bundle: {relative}")
        if _sha256(path) != expected:
            raise ValueError(f"Runtime file hash mismatch: {relative}")


def install_runtime_bundle(config: RuntimeBundle) -> dict:
    """Install a verified runtime and add its executable directories to this process PATH."""
    manifest_bytes = StoragePath(config.manifest_uri).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != config.manifest_sha256:
        raise ValueError("Runtime manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest["archive_sha256"] != config.archive_sha256:
        raise ValueError("Runtime archive identity differs from the manifest")
    parent = Path(config.installation_parent)
    parent.mkdir(parents=True, exist_ok=True)
    name = manifest["directory_name"]
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Runtime directory name must be a single path component")
    target = parent / name
    with (parent / f".{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = target / ".installed-sha256"
        if not marker.exists() or marker.read_text() != config.archive_sha256:
            if target.exists():
                raise ValueError(f"A different or incomplete runtime exists at {target}")
            with tempfile.TemporaryDirectory(prefix=".runtime-", dir=parent) as directory:
                staging = Path(directory)
                archive = staging / "runtime.tar.gz"
                storage = StoragePath(config.archive_uri)
                if storage.size() > MAX_ARCHIVE_BYTES:
                    raise ValueError("Runtime archive exceeds its byte budget")
                storage.download_to(str(archive))
                if _sha256(archive) != config.archive_sha256:
                    raise ValueError("Runtime archive hash mismatch")
                with tarfile.open(archive, "r:gz") as source:
                    members = []
                    extracted_bytes = 0
                    for member in source:
                        members.append(member)
                        extracted_bytes += member.size
                        if len(members) > MAX_ARCHIVE_FILES or extracted_bytes > MAX_EXTRACTED_BYTES:
                            raise ValueError("Runtime archive exceeds its extraction budget")
                        path = Path(member.name)
                        if not path.parts or path.is_absolute() or ".." in path.parts or path.parts[0] != target.name:
                            raise ValueError(f"Runtime archive has an invalid path: {member.name}")
                    source.extractall(staging, members=members, filter="data")
                extracted = staging / target.name
                _verify_files(extracted, manifest["files"])
                (extracted / marker.name).write_text(config.archive_sha256)
                extracted.rename(target)
        else:
            _verify_files(target, manifest["files"])
        packages = manifest["host_packages"]
        if packages:
            query = subprocess.run(
                ["dpkg-query", "-W", "-f=${Package}=${Version}\n", *packages],
                text=True,
                capture_output=True,
                check=False,
            )
            installed = dict(line.split("=", 1) for line in query.stdout.splitlines())
            if query.returncode != 0 or installed != packages:
                debs = sorted((target / "debs").glob("*.deb"))
                if not debs:
                    raise ValueError("Runtime host packages differ and the bundle has no offline packages")
                subprocess.run(["dpkg", "--install", *(str(path) for path in debs)], check=True)
                query = subprocess.run(
                    ["dpkg-query", "-W", "-f=${Package}=${Version}\n", *packages],
                    text=True,
                    capture_output=True,
                    check=True,
                )
                installed = dict(line.split("=", 1) for line in query.stdout.splitlines())
            if installed != packages:
                raise ValueError(f"Runtime package versions differ: expected {packages}, found {installed}")
        tool_directories = []
        for name, value in manifest["tools"].items():
            path = Path(value)
            if not path.is_file() or not os.access(path, os.X_OK):
                raise ValueError(f"Runtime tool is not executable: {name} at {path}")
            tool_directories.append(str(path.parent))
        os.environ["PATH"] = os.pathsep.join(dict.fromkeys((*tool_directories, os.environ.get("PATH", ""))))
        return manifest
