# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only availability checks for Marin's MoK-like native adapter."""

from __future__ import annotations

import importlib.metadata
import sys
from dataclasses import dataclass

from levanter.kernels.mixture_of_kittens.source import (
    MOK_KNOWN_GOOD_COMMIT,
    THUNDERKITTENS_KNOWN_GOOD_COMMIT,
    MokLikeBuildConfig,
    missing_source_files,
    source_is_clean,
    source_revision,
)


_CUDA_DISTRIBUTIONS = (
    "nvidia-cuda-runtime",
    "nvidia-cuda-nvcc",
    "nvidia-cuda-crt",
    "nvidia-cuda-cccl",
    "nvidia-nvvm",
)


@dataclass(frozen=True)
class MokLikePreflightStatus:
    """Result of probing source and build prerequisites without modifying them."""

    source_root: str
    source_revision: str | None
    thunderkittens_revision: str | None
    cuda_arch: str
    cuda_distributions: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def mok_like_preflight_status(build_config: MokLikeBuildConfig) -> MokLikePreflightStatus:
    """Check local prerequisites without cloning, compiling, or initializing CUDA."""

    errors: list[str] = []
    root = build_config.resolved_source_root
    revision: str | None = None
    thunderkittens_revision: str | None = None
    if sys.platform != "linux":
        errors.append(f"mok_like requires Linux, found {sys.platform}")
    if not root.exists():
        errors.append(f"configured source_root does not exist: {root}")
    else:
        missing = missing_source_files(root)
        if missing:
            errors.append(
                "configured source_root is missing: " + ", ".join(str(path.relative_to(root)) for path in missing)
            )
        revision = source_revision(root)
        if revision is None:
            errors.append("configured source_root must be the pinned Git checkout")
        elif revision != MOK_KNOWN_GOOD_COMMIT:
            errors.append(f"MoK source is at {revision}; expected {MOK_KNOWN_GOOD_COMMIT}")
        elif source_is_clean(root) is not True:
            errors.append("configured source_root must be a clean pinned checkout")
        thunderkittens_revision = source_revision(root / "third_party" / "ThunderKittens")
        if thunderkittens_revision is None:
            errors.append("ThunderKittens source must be the pinned Git checkout")
        elif thunderkittens_revision != THUNDERKITTENS_KNOWN_GOOD_COMMIT:
            errors.append(
                f"ThunderKittens is at {thunderkittens_revision}; expected {THUNDERKITTENS_KNOWN_GOOD_COMMIT}"
            )
        elif source_is_clean(root / "third_party" / "ThunderKittens") is not True:
            errors.append("ThunderKittens source must be a clean pinned checkout")

    found_distributions: list[str] = []
    for name in _CUDA_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"required CUDA build distribution is not installed: {name}")
        else:
            found_distributions.append(f"{name}=={version}")
            if not version.startswith("13."):
                errors.append(f"mok_like requires CUDA 13, found {name}=={version}")

    return MokLikePreflightStatus(
        source_root=str(root),
        source_revision=revision,
        thunderkittens_revision=thunderkittens_revision,
        cuda_arch=build_config.cuda_arch,
        cuda_distributions=tuple(found_distributions),
        errors=tuple(errors),
    )


def require_mok_like_available(build_config: MokLikeBuildConfig) -> MokLikePreflightStatus:
    """Return a successful read-only preflight status or raise one actionable error."""

    status = mok_like_preflight_status(build_config)
    if status.errors:
        details = "\n  - ".join(status.errors)
        raise RuntimeError(f"mok_like preflight failed:\n  - {details}")
    return status
