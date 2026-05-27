# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepEP source-tree and build-environment helpers."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

DEEPEP_SRC_ENV = "DEEPEP_SRC_ROOT"
DEEPEP_CACHE_ENV = "MARIN_DEEPEP_CACHE_DIR"
DEEPEP_CUDA_ARCH_ENV = "DEEPEP_CUDA_ARCH"
DISABLE_SM90_ENV = "DISABLE_SM90_FEATURES"
BUILD_WITH_TORCH_EXTENSION_ENV = "DEEPEP_BUILD_WITH_TORCH_EXTENSION"
LOAD_AS_PYTHON_MODULE_ENV = "DEEPEP_LOAD_AS_PYTHON_MODULE"

LAYOUT_REQUIRED_FILES = ("csrc/kernels/layout.cu",)
TRANSPORT_REQUIRED_FILES = (
    "csrc/config.hpp",
    "csrc/kernels/api.cuh",
    "csrc/kernels/configs.cuh",
    "csrc/kernels/layout.cu",
    "csrc/kernels/runtime.cu",
    "csrc/kernels/intranode.cu",
)

_SUPPORTED_ARCHES = ("sm_90", "sm_90a", "sm_100")


@dataclass(frozen=True)
class DeepEPPreflightStatus:
    """Result of a cheap DeepEP environment check."""

    source_root: Path | None
    cache_root: Path
    cuda_arch: str
    nvcc_path: str | None
    missing_source_files: tuple[Path, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def deepep_install_help() -> str:
    return (
        "DeepEP support expects an external DeepEP source checkout. Set "
        f"{DEEPEP_SRC_ENV}=/path/to/DeepEP and, on B200/GB200, set {DEEPEP_CUDA_ARCH_ENV}=sm_100. "
        f"Compiled JAX FFI objects are cached under {DEEPEP_CACHE_ENV} when set, otherwise under "
        "~/.cache/marin."
    )


def deepep_cache_root(component: str) -> Path:
    base = Path(os.environ.get(DEEPEP_CACHE_ENV, Path.home() / ".cache" / "marin")).expanduser()
    return base.resolve() / component


def deepep_cuda_arch() -> str:
    arch = os.environ.get(DEEPEP_CUDA_ARCH_ENV, "sm_90").strip()
    if arch not in _SUPPORTED_ARCHES:
        supported = ", ".join(_SUPPORTED_ARCHES)
        raise RuntimeError(f"{DEEPEP_CUDA_ARCH_ENV} must be one of {supported}, got {arch!r}")
    return arch


def deepep_cuda_arch_flag() -> list[str]:
    arch = deepep_cuda_arch()
    compute = arch.replace("sm_", "compute_", 1)
    return [f"-gencode=arch={compute},code={arch}"]


def deepep_torch_cuda_arch_list() -> str:
    if env_flag(DISABLE_SM90_ENV):
        return "8.0"
    arch = deepep_cuda_arch()
    if arch == "sm_90":
        return "9.0"
    if arch == "sm_90a":
        return "9.0a"
    if arch == "sm_100":
        return "10.0"
    raise AssertionError(f"Unhandled DeepEP CUDA architecture {arch!r}")


def env_flag(name: str) -> bool:
    return bool(int(os.environ.get(name, "0")))


def missing_deepep_source_files(root: Path, required_files: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(root / relative for relative in required_files if not (root / relative).is_file())


def deepep_source_root(
    *,
    required_files: tuple[str, ...],
    purpose: str,
) -> Path:
    raw = os.environ.get(DEEPEP_SRC_ENV)
    if not raw:
        raise RuntimeError(f"{DEEPEP_SRC_ENV} must point at a DeepEP checkout before using {purpose}.")
    root = Path(raw).expanduser().resolve()
    missing = missing_deepep_source_files(root, required_files)
    if missing:
        missing_text = ", ".join(str(path.relative_to(root)) for path in missing)
        raise RuntimeError(f"{DEEPEP_SRC_ENV}={root} is missing DeepEP files required for {purpose}: {missing_text}")
    return root


def deepep_preflight_status(
    *,
    required_files: tuple[str, ...] = TRANSPORT_REQUIRED_FILES,
    component: str = "deepep_transport_ffi",
) -> DeepEPPreflightStatus:
    errors: list[str] = []
    warnings: list[str] = []
    root: Path | None = None
    missing: tuple[Path, ...] = ()

    raw_root = os.environ.get(DEEPEP_SRC_ENV)
    if raw_root:
        root = Path(raw_root).expanduser().resolve()
        missing = missing_deepep_source_files(root, required_files)
        if missing:
            relative_missing = []
            for path in missing:
                try:
                    relative_missing.append(str(path.relative_to(root)))
                except ValueError:
                    relative_missing.append(str(path))
            errors.append(f"{DEEPEP_SRC_ENV} is missing required files: {', '.join(relative_missing)}")
    else:
        errors.append(f"{DEEPEP_SRC_ENV} is not set")

    try:
        arch = deepep_cuda_arch()
    except RuntimeError as exc:
        arch = os.environ.get(DEEPEP_CUDA_ARCH_ENV, "sm_90").strip()
        errors.append(str(exc))

    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        errors.append("nvcc is not on PATH")

    if env_flag(BUILD_WITH_TORCH_EXTENSION_ENV) and env_flag(LOAD_AS_PYTHON_MODULE_ENV):
        errors.append(
            f"{BUILD_WITH_TORCH_EXTENSION_ENV}=1 is not supported together with {LOAD_AS_PYTHON_MODULE_ENV}=1"
        )

    if arch != "sm_100":
        warnings.append(f"{DEEPEP_CUDA_ARCH_ENV}={arch}; use sm_100 for B200/GB200.")

    return DeepEPPreflightStatus(
        source_root=root,
        cache_root=deepep_cache_root(component),
        cuda_arch=arch,
        nvcc_path=nvcc_path,
        missing_source_files=missing,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
