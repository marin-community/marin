# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run environments that declare ``Backend.LOCAL`` as locked-down subprocesses of the Zephyr worker.

A local environment carries the storage URL of its built hash lock (``packages_lock``). The worker
builds a uv virtual environment from it once, with the NLTK data the environment's artifact names and
the runtime packages (verifyit) on its import path, so graders find there what a built image gives
them. Apt packages come from the worker image itself; ``placement`` in ``environment.py`` sends an
environment that needs others to a sandbox of an image built for it.
"""

import fcntl
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Any

from marin.execution.fingerprint import canonical_json
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import Backend, HostImage, MachineFactory, MachineSpec, NetworkPolicy
from taskcompendium.models import DEFAULT_WORKSPACE, EnvironmentRequirements

from experiments.post_training.task_curation.environment import nltk_packages
from experiments.post_training.task_curation.images.build import (
    IDENTITY_CHARS,
    LOCK_FILE,
    PYTHON_VERSION,
    RUNTIME_PACKAGES,
    RUNTIME_PTH,
    EnvironmentArtifact,
    runtime_files,
)

COMPLETE_MARKER = ".complete"
RUNTIME_DIRECTORY = "runtime"
NLTK_DATA_DIRECTORY = Path("share") / "nltk_data"
RUNTIME_PARENT = Path("/tmp")


@dataclass(frozen=True)
class LocalRuntime:
    """A uv virtual environment under ``parent`` built from the lock at ``lock_url``.

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
            canonical_json(
                {
                    "lock_sha256": self.lock_sha256,
                    "data": sorted(self.data),
                    "python": PYTHON_VERSION,
                    "runtime": runtime_files(),
                }
            ).encode()
        ).hexdigest()

    @property
    def root(self) -> Path:
        return self.parent / f"task-curation-env-{self.identity[:IDENTITY_CHARS]}"

    @property
    def bin_dir(self) -> Path:
        return self.root / "bin"

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
        python = str(self.bin_dir / "python")
        # Running from the parent keeps uv from reading the settings of a project in the working directory.
        subprocess.run(
            ["uv", "venv", "--no-project", "--python", PYTHON_VERSION, str(self.root)], check=True, cwd=self.parent
        )
        lock = self.root / LOCK_FILE
        lock.write_bytes(StoragePath(self.lock_url).read_bytes())
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
        if digest != self.lock_sha256:
            raise RuntimeError(f"{self.lock_url} has SHA-256 {digest}; its artifact records {self.lock_sha256}")
        subprocess.run(
            ["uv", "pip", "sync", "--python", python, "--require-hashes", str(lock)], check=True, cwd=self.parent
        )
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
    built = EnvironmentArtifact.raw_load(str(StoragePath(lock_url).parent))
    return LocalRuntime(lock_url, built.lock_sha256, tuple(built.data))


OWNED_ROOTS = ("/tests", "/logs", "/output", "/solution", "/controls")
"""Directories grading stages hidden tests, verdicts, captures, oracles and controls in; emptied per machine."""
SHARED_ROOTS = (DEFAULT_WORKSPACE,)
"""The grading workspace, which an Iris task also runs from, so it is written to but never emptied."""


@cache
def _local_factory(bin_dir: Path) -> LocalMachineFactory:
    return LocalMachineFactory(OWNED_ROOTS, shared_roots=SHARED_ROOTS, bin_dirs=(bin_dir,))


@dataclass(frozen=True)
class LocalGraderMachines:
    """Grading machines for environments that declare ``Backend.LOCAL``: subprocesses of this worker."""

    def identity(self) -> dict[str, Any]:
        return {"backend": Backend.LOCAL.value, "python": PYTHON_VERSION}

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        """A locked-down host machine with the environment's packages; ``memory_mb`` is not enforced."""
        if Backend.LOCAL not in environment.compatible_backends:
            raise ValueError("Only environments that declare the local backend grade in the worker")
        assert environment.packages_lock is not None
        runtime = local_runtime(environment.packages_lock)
        runtime.ensure_built()
        spec = MachineSpec(HostImage(), network=NetworkPolicy.DENY, workdir=DEFAULT_WORKSPACE, env=runtime.variables)
        return _local_factory(runtime.bin_dir), spec
