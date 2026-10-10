# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build lock-only grading environments shared by curation and rollouts.

Each runtime contains a managed CPython, locked packages, verifyit, and declared NLTK data.
Bubblewrap mounts the built root read-only. System packages come from the host;
environments needing other system packages require an image.
"""

import fcntl
import hashlib
import json
import shutil
import stat
import subprocess
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Any

import verifyit
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.local.python_environment import PythonEnvironment, build_python_environment
from shellbox.machine import Backend, HostImage, MachineFactory, MachineSpec, NetworkPolicy

from taskcompendium.models import DEFAULT_WORKSPACE, EnvironmentRequirements, require_resolved_environment

RUNTIME_PACKAGES = (Path(verifyit.__file__).parent,)
LOCK_FILE = "requirements.lock"
RUNTIME_PTH = "task-curation-runtime.pth"
IDENTITY_CHARS = 16
REGULAR_MODE = "100644"
EXECUTABLE_MODE = "100755"


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


def nltk_packages(data: tuple[str, ...]) -> tuple[str, ...]:
    """The NLTK packages named by an environment's data entries."""
    return tuple(entry.removeprefix("nltk:") for entry in data)


class LockEnvironment(BaseModel):
    """Runtime fields from a curation environment artifact's result."""

    lock_sha256: str
    data: tuple[str, ...]


class _EnvironmentRecord(BaseModel):
    result: LockEnvironment


LOCAL_PYTHON_VERSION = "3.12.12"
"""The CPython installed for local graders; the host's uv must be able to download it."""

COMPLETE_MARKER = ".complete"
ENVIRONMENT_DIRECTORY = "env"
RUNTIME_DIRECTORY = "runtime"
NLTK_DATA_DIRECTORY = Path("share") / "nltk_data"
RUNTIME_PARENT = Path("/tmp")


@dataclass(frozen=True)
class LocalRuntime:
    """A Python environment under ``parent`` built from the lock at ``lock_url``.

    ``lock_sha256`` is the digest the environment's artifact records for the lock and ``data`` the
    downloads it names. The environment's directory is named by its identity, so a changed lock builds
    a new one beside the old and environments with the same packages share one.
    """

    lock_url: str
    lock_sha256: str
    data: tuple[str, ...]
    parent: Path = RUNTIME_PARENT

    @cached_property
    def identity(self) -> str:
        """The SHA-256 of the lock, the data, the Python version and the runtime packages' files."""
        return hashlib.sha256(
            json.dumps(
                {
                    "lock_sha256": self.lock_sha256,
                    "data": sorted(self.data),
                    "python": LOCAL_PYTHON_VERSION,
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
    def variables(self) -> dict[str, str]:
        """Environment variables graders need to find the environment's data."""
        return {"NLTK_DATA": str(self.root / NLTK_DATA_DIRECTORY)} if nltk_packages(self.data) else {}

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
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
        if digest != self.lock_sha256:
            raise RuntimeError(f"{self.lock_url} has SHA-256 {digest}; its artifact records {self.lock_sha256}")
        python = str(build_python_environment(self.environment.root, lock, LOCAL_PYTHON_VERSION).python)
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
        packages = nltk_packages(self.data)
        if packages:
            data = self.root / NLTK_DATA_DIRECTORY
            subprocess.run([python, "-m", "nltk.downloader", "-d", str(data), *packages], check=True)


@cache
def local_runtime(lock_url: str) -> LocalRuntime:
    """The runtime for a built lock, from the environment artifact the lock belongs to."""
    record = StoragePath(lock_url).parent / ".artifact.json"
    built = _EnvironmentRecord.model_validate_json(record.read_text()).result
    return LocalRuntime(lock_url, built.lock_sha256, tuple(built.data))


@cache
def local_factory() -> MachineFactory:
    """Return the shared bubblewrap machine factory."""
    from shellbox.backends.local.machine import LocalMachineFactory  # noqa: PLC0415

    return LocalMachineFactory()


@dataclass(frozen=True)
class LocalGraderMachines:
    """Unbuilt local machines for lock-backed grading environments."""

    def identity(self) -> dict[str, Any]:
        return {"backend": Backend.LOCAL.value, "python": LOCAL_PYTHON_VERSION}

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        """A bubblewrap sandbox with the environment's packages; ``memory_mb`` is not enforced."""
        require_resolved_environment(environment)
        if environment.packages_lock is None:
            raise ValueError("A local grader requires a packages lock")
        spec = MachineSpec(HostImage(), network=NetworkPolicy.DENY, workdir=DEFAULT_WORKSPACE)
        return local_factory(), spec
