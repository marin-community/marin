# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepEP source-tree and build-environment helpers."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEEPEP_SRC_ENV = "DEEPEP_SRC_ROOT"
DEEPEP_CACHE_ENV = "MARIN_DEEPEP_CACHE_DIR"
DEEPEP_CUDA_ARCH_ENV = "DEEPEP_CUDA_ARCH"
DISABLE_SM90_ENV = "DISABLE_SM90_FEATURES"
BUILD_WITH_TORCH_EXTENSION_ENV = "DEEPEP_BUILD_WITH_TORCH_EXTENSION"
LOAD_AS_PYTHON_MODULE_ENV = "DEEPEP_LOAD_AS_PYTHON_MODULE"
DEEPEP_RDMA_INCLUDE_DIR_ENV = "DEEPEP_RDMA_INCLUDE_DIR"
DEEPEP_KNOWN_GOOD_COMMIT = "7febc6e25660af0f54d95dd781ecdcd62265ecca"

LAYOUT_SOURCE_CANDIDATES = (
    "csrc/kernels/layout.cu",
    "csrc/kernels/legacy/layout.cu",
)
LAYOUT_REQUIRED_FILES: tuple[str, ...] = ()
TRANSPORT_REQUIRED_FILES = (
    "csrc/config.hpp",
    "csrc/kernels/api.cuh",
    "csrc/kernels/configs.cuh",
    "csrc/kernels/runtime.cu",
    "csrc/kernels/intranode.cu",
)
INTERNODE_TRANSPORT_REQUIRED_FILES = (
    *TRANSPORT_REQUIRED_FILES,
    "csrc/kernels/internode.cu",
    "csrc/kernels/internode_ll.cu",
    "csrc/kernels/pcie.cu",
)
INTERNODE_RDMA_REQUIRED_HEADERS = ("infiniband/mlx5dv.h",)

_SUPPORTED_ARCHES = ("sm_90", "sm_90a", "sm_100")
_PREFERRED_NVSHMEM_DISTRIBUTIONS = ("nvidia-nvshmem-cu13", "nvidia-nvshmem-cu12")
_NVSHMEM_CUDA13_MARKERS = ("cu13", "cuda13", "cuda-13", "r13.")
_NVSHMEM_CUDA12_MARKERS = ("cu12", "cuda12", "cuda-12", "r12.")


@dataclass(frozen=True)
class DeepEPPreflightStatus:
    """Result of a cheap DeepEP environment check."""

    source_root: Path | None
    source_revision: str | None
    cache_root: Path
    cuda_arch: str
    nvcc_path: str | None
    nvshmem_dir: Path | None
    nvshmem_host_lib: str | None
    nvshmem_device_lib: str | None
    rdma_include_dirs: tuple[Path, ...]
    missing_rdma_headers: tuple[str, ...]
    missing_source_files: tuple[Path, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class DeepEPNVSHMEMConfig:
    """Resolved NVSHMEM include/library paths for DeepEP internode builds."""

    root: Path
    host_library_path: Path
    device_library_path: Path

    @property
    def host_library_name(self) -> str:
        return self.host_library_path.name

    @property
    def device_library_name(self) -> str:
        return self.device_library_path.name

    @property
    def include_dirs(self) -> tuple[Path, ...]:
        candidates = (
            self.root / "include",
            self.root / "include" / "nvshmem",
        )
        return tuple(path for path in candidates if path.is_dir())

    @property
    def library_dirs(self) -> tuple[Path, ...]:
        return (self.host_library_path.parent,)


def deepep_install_help() -> str:
    return (
        "DeepEP support expects an external DeepEP source checkout. Set "
        f"{DEEPEP_SRC_ENV}=/path/to/DeepEP and, on B200/GB200, set {DEEPEP_CUDA_ARCH_ENV}=sm_100. "
        f"The validated DeepEP revision is {DEEPEP_KNOWN_GOOD_COMMIT}. "
        "Internode transport also requires NVSHMEM; install levanter[deepep] or set "
        "NVSHMEM_DIR to a valid NVSHMEM install. Internode builds also require RDMA development headers "
        f"such as infiniband/mlx5dv.h; install libibverbs-dev or set {DEEPEP_RDMA_INCLUDE_DIR_ENV}. "
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


def _dedupe_paths(paths: tuple[Path, ...] | list[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return tuple(deduped)


def _nvshmem_cuda_sort_key(path: Path) -> tuple[int, str]:
    text = str(path).lower()
    if any(marker in text for marker in _NVSHMEM_CUDA13_MARKERS):
        return (0, text)
    if any(marker in text for marker in _NVSHMEM_CUDA12_MARKERS):
        return (2, text)
    return (1, text)


def _nvshmem_python_package_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for distribution_name in _PREFERRED_NVSHMEM_DISTRIBUTIONS:
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        package_root = Path(distribution.locate_file("nvidia/nvshmem"))
        if package_root.exists():
            roots.append(package_root)

    fallback_roots: list[Path] = []
    try:
        spec = importlib.util.find_spec("nvidia.nvshmem")
    except ModuleNotFoundError:
        spec = None
    if spec is not None and spec.submodule_search_locations:
        fallback_roots.extend(Path(location) for location in spec.submodule_search_locations)

    return _dedupe_paths([*roots, *sorted(fallback_roots, key=_nvshmem_cuda_sort_key)])


def _nvshmem_host_lib_candidates(base_dir: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            _dedupe_paths([base_dir / "lib" / "libnvshmem_host.so", *base_dir.rglob("libnvshmem_host.so.*")]),
            key=_nvshmem_cuda_sort_key,
        )
    )


def _nvshmem_device_lib_candidates(base_dir: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            _dedupe_paths([base_dir / "lib" / "libnvshmem_device.a", *base_dir.rglob("libnvshmem_device.a")]),
            key=_nvshmem_cuda_sort_key,
        )
    )


def _first_existing_path(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _nvshmem_host_lib_path(base_dir: Path) -> Path | None:
    return _first_existing_path(_nvshmem_host_lib_candidates(base_dir))


def _nvshmem_device_lib_path(base_dir: Path) -> Path | None:
    return _first_existing_path(_nvshmem_device_lib_candidates(base_dir))


def _nvshmem_host_lib_name(base_dir: Path) -> str | None:
    library_path = _nvshmem_host_lib_path(base_dir)
    return library_path.name if library_path is not None else None


def _nvshmem_device_lib_name(base_dir: Path) -> str | None:
    library_path = _nvshmem_device_lib_path(base_dir)
    return library_path.name if library_path is not None else None


def deepep_nvshmem_status() -> tuple[Path | None, str | None, tuple[str, ...]]:
    """Return DeepEP NVSHMEM discovery status for internode transport builds."""
    raw_dir = os.environ.get("NVSHMEM_DIR")
    if raw_dir:
        nvshmem_dir = Path(raw_dir).expanduser().resolve()
        if not nvshmem_dir.exists():
            return nvshmem_dir, None, (f"NVSHMEM_DIR={nvshmem_dir} does not exist",)
        host_lib = _nvshmem_host_lib_name(nvshmem_dir)
        errors = (
            () if host_lib is not None else (f"Could not find libnvshmem_host.so under NVSHMEM_DIR={nvshmem_dir}",)
        )
        return nvshmem_dir, host_lib, errors

    candidate_roots = _nvshmem_python_package_roots()
    if not candidate_roots:
        return None, None, ("NVSHMEM_DIR is unset and Python package nvidia.nvshmem is not installed",)
    nvshmem_dir = next(
        (root for root in candidate_roots if _nvshmem_host_lib_path(root) is not None), candidate_roots[0]
    )
    host_lib = _nvshmem_host_lib_name(nvshmem_dir)
    errors = (
        ()
        if host_lib is not None
        else (
            "Could not find libnvshmem_host.so under nvidia.nvshmem at any candidate root: "
            + ", ".join(str(root) for root in candidate_roots),
        )
    )
    return nvshmem_dir, host_lib, errors


def deepep_nvshmem_config() -> DeepEPNVSHMEMConfig:
    """Return resolved NVSHMEM build paths or raise with install guidance."""
    nvshmem_dir, _host_lib, errors = deepep_nvshmem_status()
    if errors or nvshmem_dir is None:
        detail = "; ".join(errors) if errors else "NVSHMEM root is unavailable"
        raise RuntimeError(f"DeepEP internode transport requires NVSHMEM: {detail}. {deepep_install_help()}")
    host_library_path = _nvshmem_host_lib_path(nvshmem_dir)
    if host_library_path is None:
        raise RuntimeError(f"DeepEP internode transport requires libnvshmem_host.so under {nvshmem_dir}.")
    device_library_path = _nvshmem_device_lib_path(nvshmem_dir)
    if device_library_path is None:
        raise RuntimeError(f"DeepEP internode transport requires libnvshmem_device.a under {nvshmem_dir}.")
    config = DeepEPNVSHMEMConfig(
        root=nvshmem_dir,
        host_library_path=host_library_path.resolve(),
        device_library_path=device_library_path.resolve(),
    )
    if not config.include_dirs:
        raise RuntimeError(f"DeepEP internode transport could not find NVSHMEM headers under {nvshmem_dir}.")
    return config


def deepep_rdma_include_dirs() -> tuple[Path, ...]:
    """Return include roots to search for RDMA headers needed by DeepEP internode kernels."""
    dirs: list[Path] = []
    raw_dir = os.environ.get(DEEPEP_RDMA_INCLUDE_DIR_ENV)
    if raw_dir:
        dirs.append(Path(raw_dir).expanduser().resolve())
    for path in (Path("/usr/local/include"), Path("/usr/include")):
        if path not in dirs:
            dirs.append(path)
    return tuple(path for path in dirs if path.is_dir())


def missing_deepep_rdma_headers(
    include_dirs: tuple[Path, ...] | None = None,
    required_headers: tuple[str, ...] = INTERNODE_RDMA_REQUIRED_HEADERS,
) -> tuple[str, ...]:
    """Return required RDMA headers not found under any candidate include root."""
    roots = include_dirs if include_dirs is not None else deepep_rdma_include_dirs()
    missing = []
    for header in required_headers:
        if not any((root / header).is_file() for root in roots):
            missing.append(header)
    return tuple(missing)


def env_flag(name: str) -> bool:
    return bool(int(os.environ.get(name, "0")))


def missing_deepep_source_files(root: Path, required_files: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(root / relative for relative in required_files if not (root / relative).is_file())


def deepep_layout_source(root: Path) -> Path:
    """Return the DeepEP dispatch-layout CUDA source path for supported source layouts."""
    for relative in LAYOUT_SOURCE_CANDIDATES:
        path = root / relative
        if path.is_file():
            return path
    candidates = ", ".join(LAYOUT_SOURCE_CANDIDATES)
    raise RuntimeError(f"DeepEP layout source is missing; expected one of: {candidates}")


def deepep_source_revision(root: Path) -> str | None:
    """Return the DeepEP git revision when the source root is a git checkout."""
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def deepep_source_root(
    *,
    required_files: tuple[str, ...],
    purpose: str,
    requires_layout_source: bool = False,
) -> Path:
    raw = os.environ.get(DEEPEP_SRC_ENV)
    if not raw:
        raise RuntimeError(f"{DEEPEP_SRC_ENV} must point at a DeepEP checkout before using {purpose}.")
    root = Path(raw).expanduser().resolve()
    missing = missing_deepep_source_files(root, required_files)
    if missing:
        missing_text = ", ".join(str(path.relative_to(root)) for path in missing)
        raise RuntimeError(f"{DEEPEP_SRC_ENV}={root} is missing DeepEP files required for {purpose}: {missing_text}")
    if requires_layout_source:
        try:
            deepep_layout_source(root)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{DEEPEP_SRC_ENV}={root} is missing DeepEP files required for {purpose}: {exc}"
            ) from exc
    return root


def deepep_preflight_status(
    *,
    required_files: tuple[str, ...] = TRANSPORT_REQUIRED_FILES,
    component: str = "deepep_transport_ffi",
    requires_layout_source: bool = True,
    requires_nvshmem: bool = False,
    requires_rdma: bool = False,
) -> DeepEPPreflightStatus:
    errors: list[str] = []
    warnings: list[str] = []
    root: Path | None = None
    source_revision: str | None = None
    missing: tuple[Path, ...] = ()

    raw_root = os.environ.get(DEEPEP_SRC_ENV)
    if raw_root:
        root = Path(raw_root).expanduser().resolve()
        source_revision = deepep_source_revision(root)
        if source_revision is None:
            warnings.append(
                f"Could not verify DeepEP source revision; validated revision is {DEEPEP_KNOWN_GOOD_COMMIT}."
            )
        elif source_revision != DEEPEP_KNOWN_GOOD_COMMIT:
            warnings.append(
                f"{DEEPEP_SRC_ENV} is at {source_revision}; validated revision is {DEEPEP_KNOWN_GOOD_COMMIT}."
            )
        missing = missing_deepep_source_files(root, required_files)
        if missing:
            relative_missing = []
            for path in missing:
                try:
                    relative_missing.append(str(path.relative_to(root)))
                except ValueError:
                    relative_missing.append(str(path))
            errors.append(f"{DEEPEP_SRC_ENV} is missing required files: {', '.join(relative_missing)}")
        if requires_layout_source:
            try:
                deepep_layout_source(root)
            except RuntimeError as exc:
                errors.append(f"{DEEPEP_SRC_ENV} is missing required files: {exc}")
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

    nvshmem_dir, nvshmem_host_lib, nvshmem_errors = deepep_nvshmem_status()
    nvshmem_device_lib = _nvshmem_device_lib_name(nvshmem_dir) if nvshmem_dir is not None else None
    if requires_nvshmem:
        errors.extend(nvshmem_errors)
        if not nvshmem_errors and nvshmem_device_lib is None:
            errors.append(f"DeepEP internode transport requires libnvshmem_device.a under {nvshmem_dir}")
    elif nvshmem_errors:
        warnings.extend(nvshmem_errors)

    rdma_include_dirs = deepep_rdma_include_dirs()
    missing_rdma_headers = missing_deepep_rdma_headers(rdma_include_dirs)
    if requires_rdma and missing_rdma_headers:
        missing_text = ", ".join(missing_rdma_headers)
        search_text = ", ".join(str(path) for path in rdma_include_dirs) or "<none>"
        errors.append(
            f"DeepEP internode transport is missing RDMA development headers: {missing_text}; "
            f"searched include roots: {search_text}"
        )

    if env_flag(BUILD_WITH_TORCH_EXTENSION_ENV) and env_flag(LOAD_AS_PYTHON_MODULE_ENV):
        errors.append(
            f"{BUILD_WITH_TORCH_EXTENSION_ENV}=1 is not supported together with {LOAD_AS_PYTHON_MODULE_ENV}=1"
        )

    if arch != "sm_100":
        warnings.append(f"{DEEPEP_CUDA_ARCH_ENV}={arch}; use sm_100 for B200/GB200.")

    return DeepEPPreflightStatus(
        source_root=root,
        source_revision=source_revision,
        cache_root=deepep_cache_root(component),
        cuda_arch=arch,
        nvcc_path=nvcc_path,
        nvshmem_dir=nvshmem_dir,
        nvshmem_host_lib=nvshmem_host_lib,
        nvshmem_device_lib=nvshmem_device_lib,
        rdma_include_dirs=rdma_include_dirs,
        missing_rdma_headers=missing_rdma_headers,
        missing_source_files=missing,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
