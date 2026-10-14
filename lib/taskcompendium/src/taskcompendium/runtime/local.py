# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build lock-only grading environments shared by curation and rollouts.

Each runtime contains a managed CPython, locked packages, verifyit, and the files the environment
places at absolute paths; it also carries the environment's variables. Bubblewrap mounts the built
root read-only and each file at its path. System packages come from the host; environments needing
other system packages require an image.
"""

import fcntl
import hashlib
import io
import json
import shutil
import stat
import subprocess
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache, cached_property
from pathlib import Path, PurePosixPath
from typing import Any

import verifyit
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.local.python_environment import (
    DEFAULT_PYTHON_VERSION,
    PythonEnvironment,
    build_python_environment,
)
from shellbox.machine import Backend, HostImage, MachineFactory, MachineSpec, Mount, NetworkPolicy

from taskcompendium.models import DEFAULT_WORKSPACE, EnvironmentRequirements, require_resolved_environment

RUNTIME_PACKAGES = (Path(verifyit.__file__).parent,)
LOCK_FILE = "requirements.lock"
FILES_ARCHIVE = "files.tar"
"""The environment's placed files, stored beside the lock as a tar of paths relative to ``/``."""
ENVIRONMENT_MANIFEST = "environment.json"
"""An ``EnvironmentManifest`` as JSON, stored beside the lock when the environment places files or sets variables."""
RUNTIME_PTH = "task-curation-runtime.pth"
IDENTITY_CHARS = 16
REGULAR_MODE = "100644"
EXECUTABLE_MODE = "100755"


@dataclass(frozen=True)
class EnvironmentManifest:
    """What a worker needs beyond the lock: the absolute paths the environment places files at and its variables."""

    files: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


def dump_manifest(manifest: EnvironmentManifest) -> str:
    return json.dumps({"files": sorted(manifest.files), "env": dict(sorted(manifest.env.items()))})


def load_manifest(text: str) -> EnvironmentManifest:
    declared = json.loads(text)
    return EnvironmentManifest(tuple(declared["files"]), declared["env"])


class ContextFile(BaseModel):
    """A file in a package directory, by path relative to the directory, git-style mode and content digest."""

    model_config = ConfigDict(frozen=True)

    path: str
    mode: str
    sha256: str


def context_paths(root: Path) -> list[Path]:
    """Every file below ``root`` in path order, skipping bytecode caches."""
    return sorted(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)


def _file_mode(path: Path) -> str:
    # Iris workspace bundles drop exec bits, so a consumer sees only these two modes.
    return EXECUTABLE_MODE if path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) else REGULAR_MODE


def context_files(root: Path) -> list[ContextFile]:
    return [
        ContextFile(
            path=path.relative_to(root).as_posix(),
            mode=_file_mode(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in context_paths(root)
    ]


def runtime_files() -> dict[str, list[dict[str, str]]]:
    """The files of every runtime package, by package name."""
    return {package.name: [file.model_dump() for file in context_files(package)] for package in RUNTIME_PACKAGES}


COMPLETE_MARKER = ".complete"
ENVIRONMENT_DIRECTORY = "env"
RUNTIME_DIRECTORY = "runtime"
FILES_DIRECTORY = "files"


def _runtime_parent() -> Path:
    """Use a mounted node cache for executable environments; local hosts keep the /tmp default."""
    cache = Path("/cache")
    return cache / "task-curation" if cache.is_mount() else Path("/tmp")


def extract_files(archive: bytes, directory: Path) -> None:
    """Unpack the placed files' tar under ``directory``; its members are paths relative to ``/``."""
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"{FILES_ARCHIVE} member escapes the files directory: {member.name}")
        tar.extractall(directory, filter="data")


@dataclass(frozen=True)
class LocalRuntime:
    """A Python environment under ``parent`` built from the lock at ``lock_url``.

    ``lock_sha256`` is the digest of the lock, ``files`` the absolute paths the environment places files
    at, from the ``files.tar`` beside the lock, and ``env`` the variables graders run with. The
    environment's directory is named by its identity, so a changed lock builds a new one beside the old
    and environments with the same packages, files and env share one.
    """

    lock_url: str
    lock_sha256: str
    files: tuple[str, ...]
    env: Mapping[str, str]
    parent: Path = field(default_factory=_runtime_parent)

    @cached_property
    def identity(self) -> str:
        """The SHA-256 of the lock, the placed files' paths, the env, the Python version and the runtime files."""
        return hashlib.sha256(
            json.dumps(
                {
                    "lock_sha256": self.lock_sha256,
                    "files": sorted(self.files),
                    "env": dict(self.env),
                    "python": DEFAULT_PYTHON_VERSION,
                    "runtime": runtime_files(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    @property
    def root(self) -> Path:
        return self.parent / f"task-curation-env-{self.identity[:IDENTITY_CHARS]}"

    @property
    def environment(self) -> PythonEnvironment:
        return PythonEnvironment(self.root / ENVIRONMENT_DIRECTORY)

    @property
    def bin_dir(self) -> Path:
        return self.environment.bin_dir

    @property
    def mounts(self) -> tuple[Mount, ...]:
        """Each placed file, from its copy under the root to the path the environment declares."""
        files = self.root / FILES_DIRECTORY
        return tuple(Mount(files / path.removeprefix("/"), PurePosixPath(path)) for path in self.files)

    def ensure_built(self) -> None:
        """Build the environment unless a complete one exists; concurrent callers on one host build it once.

        An exclusive ``flock`` beside the environment serializes builders across processes and threads.
        A directory without the completion marker is a failed or interrupted build and is rebuilt.
        """
        self.parent.mkdir(parents=True, exist_ok=True)
        with (self.parent / f"{self.root.name}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (self.root / COMPLETE_MARKER).exists():
                return
            if self.root.exists():
                shutil.rmtree(self.root)
            self._build()
            (self.root / COMPLETE_MARKER).write_text(self.identity)

    def _build(self) -> None:
        self.root.mkdir(parents=True)
        lock = self.root / LOCK_FILE
        lock.write_bytes(StoragePath(self.lock_url).read_bytes())
        python = str(build_python_environment(self.environment.root, lock, DEFAULT_PYTHON_VERSION).python)
        # A built image copies each runtime package into its runtime directory and puts that on sys.path.
        runtime = self.root / RUNTIME_DIRECTORY
        for package in RUNTIME_PACKAGES:
            shutil.copytree(package, runtime / package.name, ignore=shutil.ignore_patterns("__pycache__"))
        purelib = subprocess.run(
            [python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        (Path(purelib) / RUNTIME_PTH).write_text(f"{runtime}\n")
        if self.files:
            archive = (StoragePath(self.lock_url).parent / FILES_ARCHIVE).read_bytes()
            extract_files(archive, self.root / FILES_DIRECTORY)


@cache
def local_runtime(lock_url: str) -> LocalRuntime:
    """The runtime for a built lock and the environment manifest beside it, if any."""
    lock = StoragePath(lock_url)
    stored = lock.parent / ENVIRONMENT_MANIFEST
    manifest = load_manifest(stored.read_text()) if stored.exists() else EnvironmentManifest()
    return LocalRuntime(lock_url, hashlib.sha256(lock.read_bytes()).hexdigest(), manifest.files, manifest.env)


@cache
def local_factory() -> MachineFactory:
    """Return the shared bubblewrap machine factory."""
    from shellbox.backends.local.machine import LocalMachineFactory  # noqa: PLC0415

    return LocalMachineFactory()


@dataclass(frozen=True)
class LocalGraderMachines:
    """Unbuilt local grading machines with declared packages or a lock."""

    def identity(self) -> dict[str, Any]:
        return {"backend": Backend.LOCAL.value, "python": DEFAULT_PYTHON_VERSION}

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        """Return an unprepared bubblewrap spec; ``memory_mb`` is advisory."""
        require_resolved_environment(environment)
        if environment.packages_lock is None and environment.package_spec is None:
            raise ValueError("A local grader requires a package spec or packages lock")
        spec = MachineSpec(HostImage(), network=NetworkPolicy.DENY, workdir=DEFAULT_WORKSPACE)
        return local_factory(), spec
