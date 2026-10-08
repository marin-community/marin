# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build self-contained Python environments for local machines to mount read-only.

An environment root holds a uv-managed CPython and a venv made from it, so every file and symlink the
interpreter needs lies under the root. A ``LocalMachineFactory`` given the root in ``read_only`` and
the venv's ``bin`` directory in ``bin_dirs`` runs that interpreter with no other host directory mounted.
"""

import fcntl
import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PYTHON_DIRECTORY = "python"
VENV_DIRECTORY = "venv"
COMPLETE_MARKER = ".complete"


@dataclass(frozen=True)
class PythonEnvironment:
    """A venv at ``root/venv`` whose interpreter is installed at ``root/python``.

    The venv links to its interpreter by absolute path, so the root must stay where it was built.
    """

    root: Path

    @property
    def bin_dir(self) -> Path:
        return self.root / VENV_DIRECTORY / "bin"

    @property
    def python(self) -> Path:
        return self.bin_dir / "python3"


def _marker_text(lock: Path, python_version: str) -> str:
    return f"python {python_version}\nlock sha256 {hashlib.sha256(lock.read_bytes()).hexdigest()}\n"


def _uv(*args: str, environment: dict[str, str] | None = None) -> None:
    subprocess.run(["uv", *args], check=True, env=None if environment is None else {**os.environ, **environment})


def _build(environment: PythonEnvironment, lock: Path, python_version: str) -> None:
    install_dir = {"UV_PYTHON_INSTALL_DIR": str(environment.root / PYTHON_DIRECTORY)}
    # --no-bin keeps uv from linking the interpreter into the user's executable directory.
    _uv("python", "install", "--no-config", "--no-bin", python_version, environment=install_dir)
    _uv(
        "venv",
        "--no-config",
        "--managed-python",
        "--no-python-downloads",
        "--python",
        python_version,
        str(environment.root / VENV_DIRECTORY),
        environment=install_dir,
    )
    # Copying overrides a UV_LINK_MODE of symlink, which would point installed files into uv's cache.
    _uv(
        "pip",
        "sync",
        "--no-config",
        "--python",
        str(environment.python),
        "--require-hashes",
        "--link-mode",
        "copy",
        str(lock),
    )


def build_python_environment(root: Path, lock: Path, python_version: str) -> PythonEnvironment:
    """Build a venv under ``root`` with CPython ``python_version`` and the packages of the hash lock ``lock``.

    ``python_version`` is a full version such as ``3.12.13``, which uv downloads into ``root``. ``lock`` is a
    requirements file with hashes for every package, as ``uv pip compile --generate-hashes`` writes.

    Concurrent callers on one host build the root once: an exclusive ``flock`` beside it serializes
    builders, and a completion marker records a finished build. A root without the marker is a failed
    or interrupted build and is rebuilt. A finished root built from another lock or version raises
    ``ValueError``.
    """
    root = root.absolute()
    environment = PythonEnvironment(root)
    expected = _marker_text(lock, python_version)
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f"{root.name}.lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        marker = root / COMPLETE_MARKER
        if marker.exists():
            if marker.read_text() != expected:
                raise ValueError(f"{root} holds an environment built from other inputs:\n{marker.read_text()}")
            return environment
        if root.exists():
            shutil.rmtree(root)
        _build(environment, lock, python_version)
        marker.write_text(expected)
    return environment
