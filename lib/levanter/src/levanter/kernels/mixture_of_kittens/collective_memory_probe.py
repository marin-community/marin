# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated XLA collective-memory probe for peer-visible typed FFI buffers."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
from jax._src import dispatch
from jax.extend import core
from jax.interpreters import mlir
from jaxlib.mlir import ir

from levanter.kernels.mixture_of_kittens.build import (
    _CUDA_DISTRIBUTIONS,
    _cuda_include_dirs,
    _cuda_toolchain_root,
    _distribution_version,
    _jaxlib_include_dir,
)

COLLECTIVE_MEMORY_PROBE_TARGET = "levanter_collective_memory_ring_u32"
_BUILD_SCHEMA = "collective_memory_probe_v1"
_SUPPORTED_ARCHES = ("sm_100a", "sm_103a")
_REGISTERED_LIBRARY_PATH: Path | None = None


@dataclass(frozen=True)
class CollectiveMemoryProbeBuildConfig:
    """Explicit build cache and CUDA architecture for the isolated probe."""

    cache_root: str
    cuda_arch: str

    def __post_init__(self) -> None:
        if not self.cache_root:
            raise ValueError("cache_root must be explicit")
        if self.cuda_arch not in _SUPPORTED_ARCHES:
            supported = ", ".join(_SUPPORTED_ARCHES)
            raise ValueError(f"cuda_arch must be one of {supported}, got {self.cuda_arch!r}")

    @property
    def resolved_cache_root(self) -> Path:
        return Path(self.cache_root).expanduser().resolve()


@dataclass(frozen=True)
class CollectiveMemoryProbeHandle:
    """Loaded native library that owns the registered probe target."""

    library_path: Path
    _cuda_driver: ctypes.CDLL
    _library: ctypes.CDLL


def _probe_source() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "collective_memory_probe.cu"


def _cuda_arch_flag(cuda_arch: str) -> str:
    compute = cuda_arch.replace("sm_", "compute_", 1)
    return f"-gencode=arch={compute},code={cuda_arch}"


def _build_path(config: CollectiveMemoryProbeBuildConfig) -> tuple[Path, Path]:
    source = _probe_source()
    if not source.is_file():
        raise RuntimeError(f"collective-memory probe source is missing at {source}")
    key = hashlib.sha256()
    key.update(source.read_bytes())
    key.update(
        repr(
            (
                _BUILD_SCHEMA,
                jax.__version__,
                jaxlib.__version__,
                config.cuda_arch,
                tuple(
                    (name, version)
                    for name in (*_CUDA_DISTRIBUTIONS, "nvidia-nvvm")
                    if (version := _distribution_version(name)) is not None
                ),
            )
        ).encode()
    )
    for include_dir in _cuda_include_dirs():
        key.update(str(include_dir).encode())
    build_dir = config.resolved_cache_root / "collective_memory_probe" / key.hexdigest()[:16]
    return build_dir, build_dir / "libcollective_memory_probe.so"


def build_collective_memory_probe_library(config: CollectiveMemoryProbeBuildConfig) -> Path:
    """Build and return the cached typed-FFI probe library."""

    build_dir, library_path = _build_path(config)
    build_dir.mkdir(parents=True, exist_ok=True)
    with (build_dir / ".build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if library_path.is_file():
            return library_path
        toolchain_root = _cuda_toolchain_root(build_dir)
        temporary_library = library_path.with_name(f"{library_path.name}.{uuid.uuid4().hex}.tmp")
        command = [
            str(toolchain_root / "bin" / "nvcc"),
            str(_probe_source()),
            "-o",
            str(temporary_library),
            "-std=c++20",
            "-shared",
            "-Xcompiler=-fPIC",
            "--cudart=shared",
            "-O3",
            "-lineinfo",
            "-DNDEBUG",
            _cuda_arch_flag(config.cuda_arch),
            "-I",
            str(_jaxlib_include_dir()),
            "-L",
            str(toolchain_root / "lib"),
        ]
        for include_dir in _cuda_include_dirs():
            command.extend(("-I", str(include_dir)))
        try:
            subprocess.run(command, check=True)
            os.replace(temporary_library, library_path)
        finally:
            temporary_library.unlink(missing_ok=True)
    return library_path


def initialize_collective_memory_probe(
    config: CollectiveMemoryProbeBuildConfig,
) -> CollectiveMemoryProbeHandle:
    """Build, load, and register the probe without allocating runtime buffers."""

    global _REGISTERED_LIBRARY_PATH
    global_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    cuda_driver = ctypes.CDLL("libcuda.so.1", mode=global_mode)
    library_path = build_collective_memory_probe_library(config)
    library = ctypes.CDLL(str(library_path), mode=global_mode)
    if _REGISTERED_LIBRARY_PATH is None:
        handler = getattr(library, COLLECTIVE_MEMORY_PROBE_TARGET)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(
            COLLECTIVE_MEMORY_PROBE_TARGET,
            jax.ffi.pycapsule(handler),
            platform="CUDA",
            api_version=1,
        )
        jax.ffi.register_ffi_target_as_batch_partitionable(COLLECTIVE_MEMORY_PROBE_TARGET)
        _REGISTERED_LIBRARY_PATH = library_path
    elif _REGISTERED_LIBRARY_PATH != library_path:
        raise RuntimeError(
            f"collective-memory probe target is already registered from {_REGISTERED_LIBRARY_PATH}, "
            f"not {library_path}"
        )
    return CollectiveMemoryProbeHandle(library_path, cuda_driver, library)


def memory_space_frontend_attributes(memory_space: int) -> dict[str, str]:
    """Return the exact OpenXLA frontend-attribute wire values."""

    if memory_space < 0:
        raise ValueError("memory_space must be non-negative")
    return {
        "operands_memory_spaces": f"{{0:{memory_space}}}",
        "results_memory_spaces": f"{{0:{memory_space},1:{memory_space}}}",
    }


def _probe_abstract_eval(x: core.AbstractValue, *, memory_space: int) -> tuple[core.AbstractValue, ...]:
    del memory_space
    return x, x


def _probe_lowering(
    ctx: mlir.LoweringRuleContext, x: ir.Value, *, memory_space: int
) -> Sequence[Sequence[ir.Value] | ir.Value]:
    attributes = ir.DictAttr.get(
        {name: ir.StringAttr.get(value) for name, value in memory_space_frontend_attributes(memory_space).items()}
    )
    lowering = jax.ffi.ffi_lowering(
        COLLECTIVE_MEMORY_PROBE_TARGET,
        operand_layouts=((0,),),
        result_layouts=((0,), (0,)),
        extra_attributes={"mhlo.frontend_attributes": attributes},
    )
    return lowering(ctx, x)


_collective_memory_probe_p = core.Primitive("levanter_collective_memory_probe")
_collective_memory_probe_p.multiple_results = True
_collective_memory_probe_p.def_impl(partial(dispatch.apply_primitive, _collective_memory_probe_p))
_collective_memory_probe_p.def_abstract_eval(_probe_abstract_eval)
mlir.register_lowering(_collective_memory_probe_p, _probe_lowering, platform="cuda")


def collective_memory_ring_u32(x: jax.Array, *, memory_space: int = 1) -> tuple[jax.Array, jax.Array]:
    """Remotely read and write peer shards using XLA-colored FFI buffers."""

    if x.ndim != 1:
        raise ValueError(f"collective-memory probe input must be rank one, got {x.shape}")
    if x.dtype != jnp.uint32:
        raise TypeError(f"collective-memory probe input must have dtype uint32, got {x.dtype}")
    if x.shape[0] <= 0:
        raise ValueError("collective-memory probe input must not be empty")
    if memory_space < 0:
        raise ValueError("memory_space must be non-negative")
    read_result, write_result = _collective_memory_probe_p.bind(x, memory_space=memory_space)
    return read_result, write_result
