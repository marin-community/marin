# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A uv virtual environment that gives graders running in the worker what the grader image gives them.

Graders that declare ``Backend.LOCAL`` run as subprocesses of the Zephyr worker through Shellbox's
local backend. Their dependencies come from a virtual environment built on the worker from the same
recipe as the grader image: its locked requirements, its package directories, and its NLTK data. The
recipe's apt packages are not reproduced; graders that need them declare the sandbox instead.
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
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import Backend, HostImage, MachineFactory, MachineSpec, NetworkPolicy
from taskcompendium.models import DEFAULT_WORKSPACE, EnvironmentRequirements

from experiments.post_training.task_curation.images.build import IDENTITY_CHARS, LOCK_FILE, recipe_identity
from experiments.post_training.task_curation.images.recipes import GRADER, ImageRecipe

COMPLETE_MARKER = ".complete"
RUNTIME_DIRECTORY = "runtime"
RUNTIME_PTH = "task-curation-runtime.pth"
NLTK_DATA_DIRECTORY = Path("share") / "nltk_data"


@dataclass(frozen=True)
class GraderRuntime:
    """A virtual environment under ``parent`` holding ``recipe``'s locked requirements and packages.

    ``python`` is the interpreter request passed to ``uv venv``, and ``nltk_packages`` the NLTK data
    the recipe downloads. The environment's directory is named by its identity, so a changed recipe
    builds a new one beside the old.
    """

    recipe: ImageRecipe
    python: str
    nltk_packages: tuple[str, ...]
    parent: Path

    @cached_property
    def identity(self) -> str:
        """The SHA-256 of the recipe's identity, the interpreter request and the NLTK packages."""
        return hashlib.sha256(
            canonical_json(
                {"recipe": recipe_identity(self.recipe), "python": self.python, "nltk": list(self.nltk_packages)}
            ).encode()
        ).hexdigest()

    @property
    def root(self) -> Path:
        return self.parent / f"task-curation-{self.recipe.name}-{self.identity[:IDENTITY_CHARS]}"

    @property
    def bin_dir(self) -> Path:
        return self.root / "bin"

    @property
    def variables(self) -> dict[str, str]:
        """Environment variables graders need to find the environment's data."""
        return {"NLTK_DATA": str(self.root / NLTK_DATA_DIRECTORY)} if self.nltk_packages else {}

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
        lock = str(self.recipe.context / LOCK_FILE)
        # Running from the parent keeps uv from reading the settings of a project in the working directory.
        subprocess.run(
            ["uv", "venv", "--no-project", "--python", self.python, str(self.root)], check=True, cwd=self.parent
        )
        subprocess.run(["uv", "pip", "sync", "--python", python, "--require-hashes", lock], check=True, cwd=self.parent)
        # The Dockerfile copies each package into the image's runtime directory and puts that on sys.path.
        runtime = self.root / RUNTIME_DIRECTORY
        for package in self.recipe.packages:
            shutil.copytree(package, runtime / package.name, ignore=shutil.ignore_patterns("__pycache__"))
        purelib = subprocess.run(
            [python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        (Path(purelib) / RUNTIME_PTH).write_text(f"{runtime}\n")
        if self.nltk_packages:
            data = self.root / NLTK_DATA_DIRECTORY
            subprocess.run([python, "-m", "nltk.downloader", "-d", str(data), *self.nltk_packages], check=True)


GRADER_RUNTIME = GraderRuntime(
    GRADER,
    # The grader image's base runs Python 3.12, and its lock is compiled for 3.12.
    python="3.12",
    # The NLTK data the grader Dockerfile downloads.
    nltk_packages=("punkt_tab", "wordnet"),
    parent=Path("/tmp"),
)
"""The environment local graders run in: the grader recipe, built in the worker's /tmp."""

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

    runtime: GraderRuntime = GRADER_RUNTIME

    def identity(self) -> dict[str, Any]:
        return {"backend": Backend.LOCAL.value, "runtime": self.runtime.identity}

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        """A locked-down host machine with the grader recipe's packages; ``memory_mb`` is not enforced."""
        if Backend.LOCAL not in environment.compatible_backends:
            raise ValueError("Only environments that declare the local backend grade in the worker")
        self.runtime.ensure_built()
        spec = MachineSpec(
            HostImage(), network=NetworkPolicy.DENY, workdir=DEFAULT_WORKSPACE, env=self.runtime.variables
        )
        return _local_factory(self.runtime.bin_dir), spec
