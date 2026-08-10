# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Source and build-environment helpers for Mixture-of-Kittens."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


MOK_SRC_ENV = "MOK_SRC_ROOT"
MOK_CACHE_ENV = "MARIN_MOK_CACHE_DIR"
MOK_CUDA_ARCH_ENV = "MOK_CUDA_ARCH"
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


def mok_cache_root(component: str) -> Path:
    """Return the persistent build cache for one MoK component."""
    base = Path(os.environ.get(MOK_CACHE_ENV, Path.home() / ".cache" / "marin")).expanduser()
    return base.resolve() / component


def mok_cuda_arch() -> str:
    """Return the CUDA architecture used for the MoK build."""
    arch = os.environ.get(MOK_CUDA_ARCH_ENV, "sm_100a").strip()
    if arch not in _SUPPORTED_ARCHES:
        supported = ", ".join(_SUPPORTED_ARCHES)
        raise RuntimeError(f"{MOK_CUDA_ARCH_ENV} must be one of {supported}, got {arch!r}")
    return arch


def mok_cuda_arch_flag() -> str:
    """Return the NVCC target flag for the selected architecture."""
    arch = mok_cuda_arch()
    compute = arch.replace("sm_", "compute_", 1)
    return f"-gencode=arch={compute},code={arch}"


def _missing_files(root: Path) -> tuple[Path, ...]:
    return tuple(root / relative for relative in _REQUIRED_FILES if not (root / relative).is_file())


def _source_revision(root: Path) -> str | None:
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


def _clone_source(root: Path) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is required to get the Mixture-of-Kittens source")
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            MOK_REPOSITORY,
            str(root),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "checkout", "--detach", MOK_KNOWN_GOOD_COMMIT], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "submodule",
            "update",
            "--init",
            "--depth",
            "1",
            "third_party/ThunderKittens",
        ],
        check=True,
    )


def mok_source_root() -> Path:
    """Return the validated source checkout, and get the pinned source when required."""
    raw = os.environ.get(MOK_SRC_ENV)
    if raw:
        root = Path(raw).expanduser().resolve()
    else:
        root = mok_cache_root("mixture_of_kittens_source") / MOK_KNOWN_GOOD_COMMIT
        if not root.exists():
            _clone_source(root)

    missing = _missing_files(root)
    if missing:
        names = ", ".join(str(path.relative_to(root)) for path in missing)
        raise RuntimeError(f"{MOK_SRC_ENV}={root} is missing required files: {names}")
    revision = _source_revision(root)
    if revision is not None and revision != MOK_KNOWN_GOOD_COMMIT:
        raise RuntimeError(
            f"{MOK_SRC_ENV}={root} is at {revision}; expected {MOK_KNOWN_GOOD_COMMIT}"
        )
    thunderkittens_revision = _source_revision(root / "third_party" / "ThunderKittens")
    if thunderkittens_revision is not None and thunderkittens_revision != THUNDERKITTENS_KNOWN_GOOD_COMMIT:
        raise RuntimeError(
            "ThunderKittens is at "
            f"{thunderkittens_revision}; expected {THUNDERKITTENS_KNOWN_GOOD_COMMIT}"
        )
    return root
