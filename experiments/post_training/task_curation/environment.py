# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grader environment declarations and where the pipeline runs them."""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath

from shellbox.backends.local.machine import SYSTEM_DIRECTORIES
from taskcompendium.models import DOCKER_IMAGE_PATTERN

PINNED_IMAGE = re.compile(DOCKER_IMAGE_PATTERN)
REQUIREMENT_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?"
PINNED_REQUIREMENT = re.compile(REQUIREMENT_NAME + r"==[^\s=;@]+")
"""An exact ``name==version`` pin, optionally with extras."""
ARCHIVE_REQUIREMENT = re.compile(REQUIREMENT_NAME + r" @ https://\S*(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])\S*")
"""A direct URL requirement whose URL names the commit it archives."""
HASHED_ARCHIVE_REQUIREMENT = re.compile(REQUIREMENT_NAME + r" @ https://[^\s#]+#sha256=[0-9a-f]{64}")
"""A direct URL requirement pinned by its content digest."""
SHA256 = re.compile(r"[0-9a-f]{64}")
VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
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
class Archive:
    """A pinned archive whose ``member`` (a file or directory inside it) is placed at an absolute path.

    ``url`` is an https URL of a zip or tar archive and ``sha256`` the digest of the archive's bytes.
    """

    url: str
    sha256: str
    member: str

    def __post_init__(self) -> None:
        if not self.url.startswith("https://"):
            raise ValueError(f"An archive URL must be https: {self.url}")
        if SHA256.fullmatch(self.sha256) is None:
            raise ValueError(f"An archive digest must be 64 lowercase hex digits: {self.sha256}")
        member = PurePosixPath(self.member)
        if not self.member or member.is_absolute() or ".." in member.parts:
            raise ValueError(f"An archive member must be a relative path inside the archive: {self.member}")


def _relative_or_escaping(path: str) -> bool:
    parts = PurePosixPath(path)
    return not parts.is_absolute() or ".." in parts.parts


def _under_another(path: str, paths: Iterable[str]) -> bool:
    return any(other != path and PurePosixPath(path).is_relative_to(other) for other in paths)


def _within_system_directories(path: str) -> bool:
    return any(PurePosixPath(path).is_relative_to(directory) for directory in SYSTEM_DIRECTORIES)


@dataclass(frozen=True)
class Environment:
    """What a grader machine must provide; the build resolves it and the pipeline decides where to run it.

    ``requirements`` are the direct Python requirements, each an exact ``name==version`` pin or a direct
    URL requirement pinned by a commit or a SHA-256 fragment; the build compiles them to a hash lock. ``apt`` names
    Debian packages. ``files`` places pinned archive members at absolute paths: the same path in a built
    image and in the worker, where bubblewrap binds each read-only, so a path cannot lie under the host
    system directories (``/usr``, ``/etc`` and the rest of ``SYSTEM_DIRECTORIES``), which are
    already bound read-only. ``env`` sets environment variables for graders in both placements. ``image``
    is a digest-pinned image used as-is and excludes every other field. ``dockerfile`` is a checked-in,
    self-contained Dockerfile fragment that installs a toolchain the packages cannot provide, such as
    Lean and Mathlib; the environment's image build runs it after the apt step. It may use only
    ``FRAGMENT_INSTRUCTIONS``.
    """

    requirements: tuple[str, ...] = ()
    apt: tuple[str, ...] = ()
    files: Mapping[str, Archive] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    image: str | None = None
    dockerfile: Path | None = None

    def __post_init__(self) -> None:
        if self.image is not None:
            if PINNED_IMAGE.fullmatch(self.image) is None:
                raise ValueError(f"An environment image must be pinned by digest: {self.image}")
            if self.requirements or self.apt or self.files or self.env or self.dockerfile is not None:
                raise ValueError(
                    "An environment image is used as-is; declare no requirements, apt, files, env or dockerfile:"
                    f" {self.image}"
                )
        unpinned = [
            requirement
            for requirement in self.requirements
            if not any(
                pattern.fullmatch(requirement)
                for pattern in (PINNED_REQUIREMENT, ARCHIVE_REQUIREMENT, HASHED_ARCHIVE_REQUIREMENT)
            )
        ]
        if unpinned:
            raise ValueError(
                "Requirements must be exact name==version pins or HTTPS URLs pinned by a commit or SHA-256: "
                f"{unpinned}"
            )
        invalid = [package for package in self.apt if APT_PACKAGE.fullmatch(package) is None]
        if invalid:
            raise ValueError(f"apt entries must be Debian package names: {invalid}")
        misplaced = [path for path in self.files if _relative_or_escaping(path)]
        if misplaced:
            raise ValueError(f"files are placed at absolute paths: {misplaced}")
        system = [path for path in self.files if _within_system_directories(path)]
        if system:
            raise ValueError(f"files cannot be placed under the worker's read-only system directories: {system}")
        nested = [path for path in self.files if _under_another(path, self.files)]
        if nested:
            raise ValueError(f"files cannot be placed inside another placed file; the worker mounts each one: {nested}")
        invalid_names = [name for name in self.env if VARIABLE_NAME.fullmatch(name) is None]
        if invalid_names:
            raise ValueError(f"env names must be environment variable names: {invalid_names}")
        if self.dockerfile is not None:
            if not self.dockerfile.is_file():
                raise ValueError(f"Dockerfile fragment does not exist: {self.dockerfile}")
            forbidden = sorted(set(dockerfile_instructions(self.dockerfile.read_text())) - FRAGMENT_INSTRUCTIONS)
            if forbidden:
                raise ValueError(
                    f"A Dockerfile fragment may use only {sorted(FRAGMENT_INSTRUCTIONS)}, not {forbidden}:"
                    f" {self.dockerfile}"
                )


def dockerfile_instructions(text: str) -> list[str]:
    """The instruction keyword of each logical line of a Dockerfile, skipping comments and blank lines."""
    return [line.split(maxsplit=1)[0].upper() for line in dockerfile_lines(text)]


def _heredoc_delimiter(argument: str) -> str | None:
    """The heredoc delimiter an unquoted ``<<`` in the command opens, honoring quoting.

    ``cat > f <<'EOF'`` opens a heredoc whose body is shell data; ``echo "a << b"``
    does not (the operator is inside quotes). Mirrors the OpenSWE profile's
    scanner; the two must agree on what counts as a heredoc.
    """
    quote: str | None = None
    index = 0
    while index < len(argument):
        char = argument[index]
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif argument.startswith("<<", index):
            index += 2
            if index < len(argument) and argument[index] == "-":
                index += 1
            while index < len(argument) and argument[index].isspace():
                index += 1
            if index < len(argument) and argument[index] in "'\"":
                quote_char = argument[index]
                index += 1
                end = argument.find(quote_char, index)
                if end == -1:
                    return None
                return argument[index:end]
            match = re.match(r"[\w.-]+", argument[index:])
            if match:
                return match.group(0)
            return None
        index += 1
    return None


def dockerfile_lines(dockerfile: str) -> list[str]:
    """Recipe lines with backslash continuations joined, comments and blanks dropped.

    Heredoc bodies are shell data: their bytes never join continuations or
    surface as instruction lines (machine-authored recipes embed
    instruction-looking lines — even backslash-continued ones — inside RUN
    heredocs).
    """
    lines: list[str] = []
    buffer: str | None = None
    heredoc_delimiter: str | None = None
    for raw in dockerfile.splitlines():
        line = raw.strip()
        if heredoc_delimiter is not None:
            if line == heredoc_delimiter:
                heredoc_delimiter = None
            continue
        if not line or line.startswith("#"):
            continue
        if buffer is None:
            buffer = ""
        if line.endswith("\\"):
            buffer += line[:-1].rstrip() + " "
            continue
        buffer += line
        joined = buffer.strip()
        lines.append(joined)
        buffer = None
        heredoc_delimiter = _heredoc_delimiter(joined)
    if buffer:
        raise ValueError("unterminated Dockerfile continuation")
    return lines


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
