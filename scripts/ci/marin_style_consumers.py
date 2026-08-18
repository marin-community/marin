# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registered repositories updated from marin-style releases."""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType

UV_LOCK_FILE = "uv.lock"


class LockMode(StrEnum):
    NONE = "none"
    UV = "uv"


@dataclass(frozen=True)
class MarinStyleConsumer:
    name: str
    repository: str
    base_branch: str
    revision_file: str
    pin_files: tuple[str, ...]
    required_checks: tuple[str, ...]
    lock_mode: LockMode = LockMode.NONE

    @property
    def lock_files(self) -> tuple[str, ...]:
        if self.lock_mode is LockMode.UV:
            return (UV_LOCK_FILE,)
        return ()


def _consumer(
    name: str,
    repository: str,
    *,
    revision_file: str,
    pin_files: tuple[str, ...],
    required_checks: tuple[str, ...],
    lock_mode: LockMode = LockMode.NONE,
) -> MarinStyleConsumer:
    owner, separator, repository_name = repository.partition("/")
    if not name or owner != "marin-community" or separator != "/" or not repository_name or "/" in repository_name:
        raise ValueError(f"invalid marin-style consumer: {name!r}, {repository!r}")
    if not pin_files or len(pin_files) != len(set(pin_files)):
        raise ValueError(f"consumer {name!r} must have unique pin files")
    if revision_file not in pin_files:
        raise ValueError(f"consumer {name!r} revision file must be a registered pin")
    for path in pin_files:
        relative = PurePosixPath(path)
        if relative.is_absolute() or relative.as_posix() != path or ".." in relative.parts or "*" in path:
            raise ValueError(f"consumer {name!r} has invalid pin file {path!r}")
        if path == UV_LOCK_FILE:
            raise ValueError(f"consumer {name!r} must declare {UV_LOCK_FILE} through lock_mode")
    if not required_checks or len(required_checks) != len(set(required_checks)):
        raise ValueError(f"consumer {name!r} must have unique required checks")
    return MarinStyleConsumer(
        name=name,
        repository=repository,
        base_branch="main",
        revision_file=revision_file,
        pin_files=pin_files,
        required_checks=required_checks,
        lock_mode=lock_mode,
    )


_CONSUMERS = (
    _consumer(
        "harbor",
        "marin-community/harbor",
        revision_file="infra/pre-commit.py",
        pin_files=(
            ".github/workflows/marin-ci.yaml",
            ".github/workflows/marin-nightly.yaml",
            "infra/pre-commit.py",
        ),
        required_checks=("harbor-config", "marin-precommit", "marin-style-sync", "tests"),
    ),
    _consumer(
        "tpu-inference",
        "marin-community/tpu-inference",
        revision_file="infra/pre-commit.py",
        pin_files=(
            ".github/workflows/marin-ci.yaml",
            ".github/workflows/marin-e2e-nightly.yaml",
            "infra/pre-commit.py",
        ),
        required_checks=("cpu-tests", "lint"),
    ),
    _consumer(
        "vllm",
        "marin-community/vllm",
        revision_file="infra/pre-commit.py",
        pin_files=(
            ".github/workflows/marin-ci.yaml",
            ".github/workflows/marin-nightly.yaml",
            "infra/pre-commit.py",
        ),
        required_checks=("delta-smoke", "marin-precommit"),
    ),
    _consumer(
        "evalchemy",
        "marin-community/evalchemy",
        revision_file="infra/pre-commit.py",
        pin_files=(
            ".github/workflows/e2e-nightly.yaml",
            ".github/workflows/marin-ci.yaml",
            "infra/pre-commit.py",
        ),
        required_checks=("harness", "marin-precommit", "marin-style-sync"),
    ),
    _consumer(
        "axolotl",
        "marin-community/axolotl",
        revision_file="infra/pre-commit.py",
        pin_files=(".github/workflows/marin-ci.yaml", "infra/pre-commit.py"),
        required_checks=("marin-style", "tests"),
    ),
    _consumer(
        "marinskyrl",
        "marin-community/MarinSkyRL",
        revision_file="infra/pre-commit.py",
        pin_files=(
            ".github/workflows/cpu_ci.yaml",
            ".github/workflows/marin-nightly.yaml",
            "infra/pre-commit.py",
            "pyproject.toml",
        ),
        required_checks=("lint", "skyrl_gym_tests", "skyrl_train_tests"),
        lock_mode=LockMode.UV,
    ),
)

MARIN_STYLE_CONSUMERS = MappingProxyType({consumer.name: consumer for consumer in _CONSUMERS})

LEGACY_MANAGED_FILES = frozenset(
    {
        ".agents/marin-style/AGENTS-core.md",
        ".agents/marin-style/TESTING-core.md",
        ".agents/skills/commit/SKILL.md",
        ".agents/skills/consult-echo/SKILL.md",
        ".agents/skills/consult-echo/scripts/echo.py",
        ".agents/skills/debug/SKILL.md",
        ".agents/skills/file-issue/SKILL.md",
        ".agents/skills/task-logbook/SKILL.md",
        ".agents/skills/write-design-doc/SKILL.md",
        ".agents/skills/write-ops-log/SKILL.md",
        ".agents/skills/write-tests/SKILL.md",
        ".agents/skills/writing-style/SKILL.md",
        ".agents/skills/writing-style/ai-writing-donts.md",
        ".agents/skills/writing-style/blog-posts.md",
        ".agents/skills/writing-style/discord.md",
        ".agents/skills/writing-style/issues.md",
        ".agents/skills/writing-style/pull-requests.md",
        ".agents/skills/writing-style/reference-docs.md",
        ".agents/skills/writing-style/reports.md",
        ".agents/skills/writing-style/tutorials.md",
    }
)


def marin_style_consumer(name: str) -> MarinStyleConsumer:
    """Return a registered consumer or raise ValueError for an unknown name."""
    try:
        return MARIN_STYLE_CONSUMERS[name]
    except KeyError as error:
        raise ValueError(f"unknown marin-style consumer: {name!r}") from error


def marin_style_consumer_matrix(name: str = "") -> str:
    """Return compact JSON for the registered GitHub Actions matrix."""
    consumers = (marin_style_consumer(name),) if name else _CONSUMERS
    return json.dumps(
        {
            "include": [
                {
                    "name": consumer.name,
                    "repository": consumer.repository,
                    "repository_name": consumer.repository.removeprefix("marin-community/"),
                    "base_branch": consumer.base_branch,
                }
                for consumer in consumers
            ]
        },
        separators=(",", ":"),
    )
