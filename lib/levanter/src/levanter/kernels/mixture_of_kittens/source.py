# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned source configuration for Marin's MoK-like kernel."""

from __future__ import annotations

import shutil
import subprocess
import uuid
import fcntl
import os
from dataclasses import dataclass
from pathlib import Path

MOK_KNOWN_GOOD_COMMIT = "6438bf48f88094d305972fbe0fa6deba0f7d4d1a"
MOK_REPOSITORY = "https://github.com/cursor/mixture-of-kittens.git"
THUNDERKITTENS_KNOWN_GOOD_COMMIT = "1c3920d993404dd49a6d4c7267ea11d583bd5c68"

_REQUIRED_FILES = (
    "csrc/mok_megakernel.cuh",
    "csrc/mxfp8.cuh",
    "csrc/utils.cuh",
    "third_party/ThunderKittens/include/kittens.cuh",
)
_SUPPORTED_ARCHES = ("sm_100a", "sm_103a")


@dataclass(frozen=True)
class MokLikeBuildConfig:
    """Explicit source and build-cache locations for the native adapter."""

    source_root: str
    cache_root: str
    cuda_arch: str
    clone_if_missing: bool = False

    def __post_init__(self) -> None:
        if not self.source_root:
            raise ValueError("source_root must be explicit")
        if not self.cache_root:
            raise ValueError("cache_root must be explicit")
        if self.cuda_arch not in _SUPPORTED_ARCHES:
            supported = ", ".join(_SUPPORTED_ARCHES)
            raise ValueError(f"cuda_arch must be one of {supported}, got {self.cuda_arch!r}")

    @property
    def resolved_source_root(self) -> Path:
        return Path(self.source_root).expanduser().resolve()

    @property
    def resolved_cache_root(self) -> Path:
        return Path(self.cache_root).expanduser().resolve()


def mok_cache_root(config: MokLikeBuildConfig, component: str) -> Path:
    return config.resolved_cache_root / component


def mok_cuda_arch_flag(config: MokLikeBuildConfig) -> str:
    compute = config.cuda_arch.replace("sm_", "compute_", 1)
    return f"-gencode=arch={compute},code={config.cuda_arch}"


def missing_source_files(root: Path) -> tuple[Path, ...]:
    return tuple(root / relative for relative in _REQUIRED_FILES if not (root / relative).is_file())


def source_revision(root: Path) -> str | None:
    """Return the Git revision for root without modifying it."""

    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    revision = result.stdout.strip()
    return revision or None


def source_is_clean(root: Path) -> bool | None:
    """Return whether a Git checkout has no tracked, untracked, or submodule changes."""

    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return not result.stdout.strip()


def _clone_source(root: Path) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is required to get the pinned Mixture-of-Kittens source")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = root.with_name(f".{root.name}.{uuid.uuid4().hex}.tmp")
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", MOK_REPOSITORY, str(temporary_root)],
            check=True,
        )
        subprocess.run(["git", "-C", str(temporary_root), "checkout", "--detach", MOK_KNOWN_GOOD_COMMIT], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(temporary_root),
                "submodule",
                "update",
                "--init",
                "--depth",
                "1",
                "third_party/ThunderKittens",
            ],
            check=True,
        )
        os.replace(temporary_root, root)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def mok_source_root(config: MokLikeBuildConfig) -> Path:
    """Materialize and validate the explicitly configured pinned source."""

    root = config.resolved_source_root
    if not root.exists():
        if not config.clone_if_missing:
            raise RuntimeError(
                f"MoK-like source does not exist at {root}; provide it or set clone_if_missing=True explicitly"
            )
        root.parent.mkdir(parents=True, exist_ok=True)
        lock_path = root.with_name(f".{root.name}.materialize.lock")
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not root.exists():
                _clone_source(root)

    missing = missing_source_files(root)
    if missing:
        names = ", ".join(str(path.relative_to(root)) for path in missing)
        raise RuntimeError(f"MoK-like source at {root} is missing required files: {names}")
    revision = source_revision(root)
    if revision is None:
        raise RuntimeError(f"MoK-like source at {root} must be the pinned Git checkout")
    if revision != MOK_KNOWN_GOOD_COMMIT:
        raise RuntimeError(f"MoK-like source at {root} is at {revision}; expected {MOK_KNOWN_GOOD_COMMIT}")
    if source_is_clean(root) is not True:
        raise RuntimeError(f"MoK-like source at {root} must be a clean pinned checkout")
    thunderkittens_revision = source_revision(root / "third_party" / "ThunderKittens")
    if thunderkittens_revision is None:
        raise RuntimeError("ThunderKittens source must be the pinned Git checkout")
    if thunderkittens_revision != THUNDERKITTENS_KNOWN_GOOD_COMMIT:
        raise RuntimeError(
            "ThunderKittens is at " f"{thunderkittens_revision}; expected {THUNDERKITTENS_KNOWN_GOOD_COMMIT}"
        )
    if source_is_clean(root / "third_party" / "ThunderKittens") is not True:
        raise RuntimeError("ThunderKittens source must be a clean pinned checkout")
    return root
