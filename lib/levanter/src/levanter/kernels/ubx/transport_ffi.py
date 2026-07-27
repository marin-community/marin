# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX FFI wrappers for raw BF16 NVIDIA NCCL UB-X MoE transport."""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import logging
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np


PINNED_NCCL_COMMIT = "db0c814185a0415cc2e23dca387fecb9282de551"
PINNED_NCCL_VERSION = 23007
_DISPATCH_TARGET = "levanter_ubx_dispatch_topk_bf16"
_COMBINE_TARGET = "levanter_ubx_combine_push3_bf16"
_INIT_SYMBOL = "levanter_ubx_init_local_runtime"
_SHUTDOWN_SYMBOL = "levanter_ubx_shutdown_local_runtime"
_LAST_ERROR_SYMBOL = "levanter_ubx_last_error"
_BUILD_CACHE_SCHEMA_VERSION = "ubx_transport_ffi_v1"
_REG0_BYTES = 4096
_NCCL_ALLOCATION_ALIGNMENT = 2 * 1024 * 1024
_PAYLOAD_ALIGNMENT = 256
_LIBRARY_DLOPEN_MODE = getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_GLOBAL", 0)
logger = logging.getLogger(__name__)


class ArraySpec(Protocol):
    shape: tuple[int, ...]
    dtype: np.dtype


class LocalDevice(Protocol):
    local_hardware_id: int


@dataclass(frozen=True)
class UbxRuntimeConfig:
    """Static shape and launch contract for one local UB-X expert group."""

    num_ranks: int
    max_tokens_per_rank: int
    max_local_tokens: int
    hidden_size: int
    top_k: int
    experts_per_rank: int
    default_sms: int = 0
    sm_limit: int = 0
    timeout_clocks: int = 2_000_000_000

    def validate(self) -> None:
        if self.num_ranks != 8:
            raise ValueError(f"UB-X FFI currently supports exactly 8 local ranks, got {self.num_ranks}")
        for name in (
            "max_tokens_per_rank",
            "max_local_tokens",
            "hidden_size",
            "top_k",
            "experts_per_rank",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_size % 32 != 0:
            raise ValueError(f"hidden_size must be divisible by 32 for UB-X BF16 kernels, got {self.hidden_size}")
        if self.default_sms < 0 or self.sm_limit < 0:
            raise ValueError("default_sms and sm_limit must be nonnegative")
        if self.timeout_clocks <= 0:
            raise ValueError(f"timeout_clocks must be positive, got {self.timeout_clocks}")

    @property
    def total_experts(self) -> int:
        return self.num_ranks * self.experts_per_rank


@dataclass(frozen=True)
class PoolLayout:
    """Byte layout of each rank's NCCL symmetric pool."""

    reg0_bytes: int
    dispatch_offsets: tuple[int, int]
    dispatch_bytes: int
    combine_offsets: tuple[int, int]
    combine_bytes: int
    pool_bytes: int


@dataclass(frozen=True)
class BuildPlan:
    """Resolved source-built NCCL/UB-X compilation inputs."""

    source_root: Path
    cuda_home: Path
    nccl_include_dir: Path
    nccl_library_path: Path
    ubx_source: Path
    ffi_source: Path
    library_path: Path
    command: tuple[str, ...]


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def pool_layout(config: UbxRuntimeConfig) -> PoolLayout:
    """Return the fixed symmetric-pool layout for ``config``."""
    config.validate()
    dispatch_bytes = config.max_tokens_per_rank * config.hidden_size * np.dtype(jnp.bfloat16).itemsize
    combine_bytes = config.max_local_tokens * config.top_k * config.hidden_size * np.dtype(jnp.bfloat16).itemsize
    dispatch0 = _align(_REG0_BYTES, _PAYLOAD_ALIGNMENT)
    dispatch1 = _align(dispatch0 + dispatch_bytes, _PAYLOAD_ALIGNMENT)
    combine0 = _align(dispatch1 + dispatch_bytes, _PAYLOAD_ALIGNMENT)
    combine1 = _align(combine0 + combine_bytes, _PAYLOAD_ALIGNMENT)
    pool_bytes = _align(combine1 + combine_bytes, _NCCL_ALLOCATION_ALIGNMENT)
    return PoolLayout(
        reg0_bytes=_REG0_BYTES,
        dispatch_offsets=(dispatch0, dispatch1),
        dispatch_bytes=dispatch_bytes,
        combine_offsets=(combine0, combine1),
        combine_bytes=combine_bytes,
        pool_bytes=pool_bytes,
    )


def _jaxlib_include_dir() -> Path:
    return Path(jaxlib.__file__).resolve().parent / "include"


def _ffi_source() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "ubx_transport_ffi.cu"


def _git_commit(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_source_tree(source_root: Path) -> tuple[Path, Path, Path]:
    source_root = source_root.resolve()
    actual_commit = _git_commit(source_root)
    if actual_commit != PINNED_NCCL_COMMIT:
        raise RuntimeError(
            f"UB-X FFI requires NVIDIA/nccl commit {PINNED_NCCL_COMMIT}, " f"got {actual_commit} at {source_root}"
        )
    ubx_source = source_root / "contrib" / "nccl_ubx" / "csrc" / "ubx.cu"
    nccl_include_dir = source_root / "build" / "include"
    nccl_library_dir = source_root / "build" / "lib"
    required = (
        ubx_source,
        source_root / "contrib" / "nccl_ubx" / "csrc" / "include" / "ubx" / "ubx.h",
        nccl_include_dir / "nccl.h",
        nccl_include_dir / "nccl_device.h",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Pinned NCCL/UB-X source tree is not built; missing: {', '.join(missing)}")
    candidates = sorted(nccl_library_dir.glob("libnccl.so.2.30.7"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one source-built libnccl.so.2.30.7 under {nccl_library_dir}, got {candidates}"
        )
    return ubx_source, nccl_include_dir, candidates[0].resolve()


def _build_key(
    *,
    source_root: Path,
    cuda_home: Path,
    ubx_source: Path,
    nccl_library_path: Path,
) -> str:
    digest = hashlib.sha256()
    for path in (
        _ffi_source(),
        ubx_source,
        source_root / "contrib" / "nccl_ubx" / "csrc" / "include" / "ubx" / "ubx.h",
        source_root / "contrib" / "nccl_ubx" / "csrc" / "include" / "ubx" / "common.h",
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    digest.update(PINNED_NCCL_COMMIT.encode())
    digest.update(str(cuda_home).encode())
    digest.update(
        subprocess.run(
            [str(cuda_home / "bin" / "nvcc"), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.encode()
    )
    digest.update(str(_jaxlib_include_dir()).encode())
    digest.update(jaxlib.__version__.encode())
    digest.update(str(nccl_library_path).encode())
    digest.update(_BUILD_CACHE_SCHEMA_VERSION.encode())
    return digest.hexdigest()[:16]


def build_plan(
    *,
    source_root: Path,
    cuda_home: Path,
    cache_root: Path | None = None,
) -> BuildPlan:
    """Resolve and validate a direct NVCC build against pinned source artifacts."""
    source_root = source_root.resolve()
    cuda_home = cuda_home.resolve()
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA compiler not found at {nvcc}")
    ubx_source, nccl_include_dir, nccl_library_path = _validate_source_tree(source_root)
    key = _build_key(
        source_root=source_root,
        cuda_home=cuda_home,
        ubx_source=ubx_source,
        nccl_library_path=nccl_library_path,
    )
    root = (cache_root or Path.home() / ".cache" / "levanter" / "ubx_transport_ffi").resolve()
    library_path = root / key / "liblevanter_ubx_transport_ffi.so"
    ubx_include_dir = source_root / "contrib" / "nccl_ubx" / "csrc" / "include"
    command = (
        str(nvcc),
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-DUB_TIMEOUT_ENABLED",
        "-gencode=arch=compute_90a,code=sm_90a",
        f"-I{_jaxlib_include_dir()}",
        f"-I{ubx_include_dir}",
        f"-I{nccl_include_dir}",
        str(_ffi_source()),
        str(ubx_source),
        f"-L{nccl_library_path.parent}",
        "-lnccl",
        "-lcudart",
        "-ldl",
        "-pthread",
        "-Xlinker=-rpath",
        f"-Xlinker={nccl_library_path.parent}",
        "-o",
        str(library_path),
    )
    return BuildPlan(
        source_root=source_root,
        cuda_home=cuda_home,
        nccl_include_dir=nccl_include_dir,
        nccl_library_path=nccl_library_path,
        ubx_source=ubx_source,
        ffi_source=_ffi_source(),
        library_path=library_path,
        command=command,
    )


def _build_shared_library(plan: BuildPlan) -> None:
    if plan.library_path.is_file():
        return
    plan.library_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan.library_path.with_suffix(".tmp.so")
    command = tuple(str(temporary) if arg == str(plan.library_path) else arg for arg in plan.command)
    subprocess.run(command, check=True)
    temporary.replace(plan.library_path)


def _load_library(plan: BuildPlan) -> ctypes.CDLL:
    cached = getattr(_load_library, "_cached", None)
    cached_path = getattr(_load_library, "_path", None)
    if cached is not None and cached_path == plan.library_path:
        return cached
    _build_shared_library(plan)
    nccl_library = ctypes.CDLL(str(plan.nccl_library_path), mode=_LIBRARY_DLOPEN_MODE)
    library = ctypes.CDLL(str(plan.library_path), mode=_LIBRARY_DLOPEN_MODE)
    _load_library._nccl = nccl_library
    _load_library._cached = library
    _load_library._path = plan.library_path
    return library


def _register_targets(plan: BuildPlan) -> ctypes.CDLL:
    library = _load_library(plan)
    if getattr(_register_targets, "_path", None) == plan.library_path:
        return library
    if getattr(_register_targets, "_path", None) is not None:
        raise RuntimeError("UB-X FFI targets are already registered from a different build artifact")
    for target in (_DISPATCH_TARGET, _COMBINE_TARGET):
        handler = getattr(library, target)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(
            target,
            jax.ffi.pycapsule(handler),
            platform="CUDA",
            api_version=1,
        )
        jax.ffi.register_ffi_target_as_batch_partitionable(target)
    _register_targets._path = plan.library_path
    return library


def _last_error(library: ctypes.CDLL, default: str = "") -> str:
    function = getattr(library, _LAST_ERROR_SYMBOL)
    function.argtypes = []
    function.restype = ctypes.c_char_p
    message = function()
    return message.decode() if message else default


def validate_local_hardware_ordinals(devices: Sequence[LocalDevice], num_ranks: int) -> None:
    """Require one process-local CUDA ordinal per rank, independent of global JAX IDs."""
    ordinals = [device.local_hardware_id for device in devices]
    if len(ordinals) != num_ranks or ordinals != list(range(num_ranks)):
        raise RuntimeError(
            "UB-X FFI requires all eight process-local CUDA ordinals in order; "
            f"got local_hardware_id values {ordinals}"
        )


def ensure_local_runtime(
    config: UbxRuntimeConfig,
    *,
    source_root: Path,
    cuda_home: Path,
    cache_root: Path | None = None,
) -> None:
    """Build, register, and initialize one eight-GPU local UB-X runtime."""
    config.validate()
    plan = build_plan(source_root=source_root, cuda_home=cuda_home, cache_root=cache_root)
    signature = (config, plan.library_path)
    if getattr(ensure_local_runtime, "_signature", None) == signature:
        return
    library = _register_targets(plan)
    local_gpus = [device for device in jax.local_devices() if device.platform == "gpu"]
    validate_local_hardware_ordinals(local_gpus, config.num_ranks)
    init = getattr(library, _INIT_SYMBOL)
    init.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_ulonglong,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    init.restype = ctypes.c_int
    ensure_local_runtime._signature = None
    ensure_local_runtime._config = None
    status = init(
        config.num_ranks,
        config.max_tokens_per_rank,
        config.max_local_tokens,
        config.hidden_size,
        config.top_k,
        config.experts_per_rank,
        config.default_sms,
        config.sm_limit,
        config.timeout_clocks,
        os.fsencode(plan.nccl_library_path),
        PINNED_NCCL_VERSION,
    )
    if status != 0:
        raise RuntimeError(f"Failed to initialize UB-X local runtime: {_last_error(library, 'unknown error')}")
    ensure_local_runtime._signature = signature
    ensure_local_runtime._config = config


def shutdown_local_runtime() -> None:
    """Synchronize and release the process-local UB-X runtime."""
    if getattr(ensure_local_runtime, "_signature", None) is None:
        return
    library = getattr(_load_library, "_cached", None)
    if library is None:
        return
    shutdown = getattr(library, _SHUTDOWN_SYMBOL)
    shutdown.argtypes = []
    shutdown.restype = None
    shutdown()
    error = _last_error(library)
    ensure_local_runtime._signature = None
    ensure_local_runtime._config = None
    if error:
        raise RuntimeError(f"Failed to shut down UB-X local runtime: {error}")


def _shutdown_at_exit() -> None:
    try:
        shutdown_local_runtime()
    except Exception:
        logger.exception("Failed to shut down the UB-X local runtime during interpreter exit")


atexit.register(_shutdown_at_exit)


def _runtime_config() -> UbxRuntimeConfig:
    config = getattr(ensure_local_runtime, "_config", None)
    if config is None:
        raise RuntimeError("Call ensure_local_runtime before tracing UB-X FFI operations")
    return config


def _shape_dtype(value: ArraySpec) -> tuple[tuple[int, ...], np.dtype]:
    return tuple(value.shape), np.dtype(value.dtype)


def validate_dispatch_inputs(
    x: ArraySpec,
    dispatch_topk_expert: ArraySpec,
    dispatch_topk_slot: ArraySpec,
    dispatch_valid: ArraySpec,
    config: UbxRuntimeConfig,
) -> None:
    """Validate the static raw-dispatch contract without touching a GPU."""
    config.validate()
    x_shape, x_dtype = _shape_dtype(x)
    expert_shape, expert_dtype = _shape_dtype(dispatch_topk_expert)
    slot_shape, slot_dtype = _shape_dtype(dispatch_topk_slot)
    valid_shape, valid_dtype = _shape_dtype(dispatch_valid)
    expected_map = (config.max_local_tokens, config.top_k)
    if x_shape != (config.max_local_tokens, config.hidden_size) or x_dtype != np.dtype(jnp.bfloat16):
        raise ValueError(f"x must be BF16 [{config.max_local_tokens}, {config.hidden_size}], got {x_shape} {x_dtype}")
    if expert_shape != expected_map or slot_shape != expected_map:
        raise ValueError(f"dispatch maps must both have shape {expected_map}")
    if expert_dtype != np.dtype(np.int32) or slot_dtype != np.dtype(np.int32):
        raise ValueError("dispatch maps must both have dtype int32")
    if valid_shape != (config.max_tokens_per_rank,) or valid_dtype != np.dtype(np.bool_):
        raise ValueError(f"dispatch_valid must be bool [{config.max_tokens_per_rank}]")


def validate_combine_inputs(
    expert_outputs: ArraySpec,
    inverse_map: ArraySpec,
    topk_idx: ArraySpec,
    gate_weights: ArraySpec,
    config: UbxRuntimeConfig,
) -> None:
    """Validate the static push3-combine contract without touching a GPU."""
    config.validate()
    outputs_shape, outputs_dtype = _shape_dtype(expert_outputs)
    inverse_shape, inverse_dtype = _shape_dtype(inverse_map)
    topk_shape, topk_dtype = _shape_dtype(topk_idx)
    gates_shape, gates_dtype = _shape_dtype(gate_weights)
    if outputs_shape != (config.max_tokens_per_rank, config.hidden_size) or outputs_dtype != np.dtype(jnp.bfloat16):
        raise ValueError(
            "expert_outputs must be BF16 "
            f"[{config.max_tokens_per_rank}, {config.hidden_size}], got {outputs_shape} {outputs_dtype}"
        )
    if inverse_shape != (config.max_tokens_per_rank, 4) or inverse_dtype != np.dtype(np.int32):
        raise ValueError(f"inverse_map must be int32 [{config.max_tokens_per_rank}, 4]")
    if topk_shape != (config.max_local_tokens, config.top_k) or topk_dtype != np.dtype(np.int32):
        raise ValueError(f"topk_idx must be int32 [{config.max_local_tokens}, {config.top_k}]")
    if gates_shape != (config.max_local_tokens, config.total_experts) or gates_dtype != np.dtype(np.float32):
        raise ValueError(f"gate_weights must be float32 [{config.max_local_tokens}, {config.total_experts}]")


def dispatch_topk_bf16(
    x: jax.Array,
    dispatch_topk_expert: jax.Array,
    dispatch_topk_slot: jax.Array,
    dispatch_valid: jax.Array,
) -> jax.Array:
    """Dispatch rank-local BF16 tokens into compact expert-major capacity rows.

    This raw primitive has no autodiff rule. Callers must supply precomputed
    global expert IDs, arbitrary compact destination slots, and the local
    destination validity mask.
    """
    config = _runtime_config()
    validate_dispatch_inputs(x, dispatch_topk_expert, dispatch_topk_slot, dispatch_valid, config)
    result = jax.ffi.ffi_call(
        _DISPATCH_TARGET,
        jax.ShapeDtypeStruct((config.max_tokens_per_rank, config.hidden_size), jnp.bfloat16),
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        jnp.asarray(x, dtype=jnp.bfloat16),
        jnp.asarray(dispatch_topk_expert, dtype=jnp.int32),
        jnp.asarray(dispatch_topk_slot, dtype=jnp.int32),
        jnp.asarray(dispatch_valid, dtype=jnp.bool_),
    )
    return result


def combine_push3_bf16(
    expert_outputs: jax.Array,
    inverse_map: jax.Array,
    topk_idx: jax.Array,
    gate_weights: jax.Array,
) -> jax.Array:
    """Push and combine compact BF16 expert outputs with dense FP32 gates.

    ``inverse_map`` is scanned over exactly ``max_tokens_per_rank`` rows and
    carries validity in column 3. This raw primitive has no autodiff rule.
    """
    config = _runtime_config()
    validate_combine_inputs(expert_outputs, inverse_map, topk_idx, gate_weights, config)
    result = jax.ffi.ffi_call(
        _COMBINE_TARGET,
        jax.ShapeDtypeStruct((config.max_local_tokens, config.hidden_size), jnp.bfloat16),
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        jnp.asarray(expert_outputs, dtype=jnp.bfloat16),
        jnp.asarray(inverse_map, dtype=jnp.int32),
        jnp.asarray(topk_idx, dtype=jnp.int32),
        jnp.asarray(gate_weights, dtype=jnp.float32),
    )
    return result
