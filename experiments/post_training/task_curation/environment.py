# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grader build inputs and their packaging in an image or worker package lock."""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from taskcompendium.models import DOCKER_IMAGE_PATTERN

PINNED_IMAGE = re.compile(DOCKER_IMAGE_PATTERN)
PYPI_PIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?==[^\s=;]+")
NLTK_DATA = re.compile(r"nltk:[a-z0-9_]+")
APT_PACKAGE = re.compile(r"[a-z0-9][a-z0-9+.-]+")

FRAGMENT_INSTRUCTIONS = frozenset({"ARG", "ENV", "RUN"})
"""The Dockerfile instructions a fragment may use.

A fragment is spliced between the generated apt step and the lock sync, so an instruction that
changes the base image, the user, the working directory, the shell, the entrypoint, or the build
context would break or redirect the generated steps that follow it. ``ENV`` persists on purpose: it
is how a fragment puts its toolchain on ``PATH`` for the grader.
"""

WORKER_IMAGE_APT = frozenset(
    {
        "build-essential",
        "ca-certificates",
        "curl",
        "ffmpeg",
        "git",
        "libjemalloc2",
        "lldb",
        "openssh-client",
        "unzip",
    }
)
"""Debian packages the Zephyr worker image installs: the ``task`` stage of lib/iris/Dockerfile.

An environment that needs only these runs in the worker; update this set when that stage changes.
"""


@dataclass(frozen=True)
class Environment:
    """What a machine must provide; the pipeline decides where to run it.

    ``pypi`` holds exact ``name==version`` pins, compiled to a hash lock at build time; ``lock`` is a
    uv-compiled requirements lock with hashes, used verbatim instead. ``apt`` names Debian packages and
    ``data`` downloads such as ``nltk:punkt_tab``. ``image`` is a digest-pinned image used as-is and
    excludes every other field. ``dockerfile`` is a checked-in, self-contained Dockerfile fragment that
    installs a toolchain the packages cannot provide, such as Lean and Mathlib; the environment's image
    build runs it after the apt step, so it excludes ``image``. It may use only ``FRAGMENT_INSTRUCTIONS``.
    """

    pypi: tuple[str, ...] = ()
    lock: Path | None = None
    apt: tuple[str, ...] = ()
    image: str | None = None
    data: tuple[str, ...] = ()
    dockerfile: Path | None = None

    def __post_init__(self) -> None:
        if self.pypi and self.lock is not None:
            raise ValueError("An environment names its packages as pypi pins or as a lock, not both")
        if self.image is not None:
            if PINNED_IMAGE.fullmatch(self.image) is None:
                raise ValueError(f"An environment image must be pinned by digest: {self.image}")
            if self.pypi or self.lock is not None or self.apt or self.data or self.dockerfile is not None:
                raise ValueError(
                    f"An environment image is used as-is; declare no packages, data or dockerfile: {self.image}"
                )
        unpinned = [requirement for requirement in self.pypi if PYPI_PIN.fullmatch(requirement) is None]
        if unpinned:
            raise ValueError(f"pypi requirements must be exact name==version pins: {unpinned}")
        invalid = [package for package in self.apt if APT_PACKAGE.fullmatch(package) is None]
        if invalid:
            raise ValueError(f"apt entries must be Debian package names: {invalid}")
        if self.lock is not None and not self.lock.is_file():
            raise ValueError(f"Lock file does not exist: {self.lock}")
        if self.dockerfile is not None:
            if not self.dockerfile.is_file():
                raise ValueError(f"Dockerfile fragment does not exist: {self.dockerfile}")
            forbidden = sorted(set(dockerfile_instructions(self.dockerfile.read_text())) - FRAGMENT_INSTRUCTIONS)
            if forbidden:
                raise ValueError(
                    f"A Dockerfile fragment may use only {sorted(FRAGMENT_INSTRUCTIONS)}, not {forbidden}:"
                    f" {self.dockerfile}"
                )
        unsupported = [entry for entry in self.data if NLTK_DATA.fullmatch(entry) is None]
        if unsupported:
            raise ValueError(f"Data entries must be nltk:<package>: {unsupported}")


def dockerfile_instructions(text: str) -> list[str]:
    """The instruction keyword of each logical line of a Dockerfile, skipping comments and blank lines."""
    instructions = []
    continued = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue  # Docker drops comment lines even inside a continued instruction.
        if not continued:
            instructions.append(line.split(maxsplit=1)[0].upper())
        continued = line.endswith("\\")
    return instructions


class Placement(StrEnum):
    """Where the pipeline runs an environment."""

    IMAGE = "image"
    """A sandbox of the declared image."""
    BUILT_IMAGE = "built_image"
    """A sandbox of the image built for a toolchain fragment or packages the worker image lacks."""
    WORKER = "worker"
    """A bubblewrap sandbox on the Zephyr worker, in a uv environment built from the lock."""


def placement(environment: Environment) -> Placement:
    """Use an explicit image, build for a toolchain or missing system packages, or prepare a lock in Local.

    Local assumes the worker provides WORKER_IMAGE_APT; it does not probe the host.
    """
    if environment.image is not None:
        return Placement.IMAGE
    if environment.dockerfile is not None or set(environment.apt) - WORKER_IMAGE_APT:
        return Placement.BUILT_IMAGE
    return Placement.WORKER
