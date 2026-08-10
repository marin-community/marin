# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX FFI adapter for the Mixture-of-Kittens BF16 forward kernel."""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import importlib.metadata
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from levanter.kernels.mixture_of_kittens.source import (
    mok_cache_root,
    mok_cuda_arch,
    mok_cuda_arch_flag,
    mok_source_root,
)


_TARGET = "levanter_mok_forward_bf16_4"
_INIT_SYMBOL = "levanter_mok_init_runtime"
_SHUTDOWN_SYMBOL = "levanter_mok_shutdown_runtime"
_LAST_ERROR_SYMBOL = "levanter_mok_last_error"
_BUILD_SCHEMA = "mok_forward_ffi_v1"
_NUM_DEVICES = 4
_TILE_ROWS = 256
_CLUSTER_SIZE = 2


@dataclass(frozen=True)
class MoKForwardConfig:
    """Static controls for one fused forward call."""

    num_comm_sms: int = 40
    minibatch_size: int = 4096
    schedule_capacity_factor: float = 1.1

    def __post_init__(self) -> None:
        if self.num_comm_sms < _CLUSTER_SIZE or self.num_comm_sms % _CLUSTER_SIZE != 0:
            raise ValueError("num_comm_sms must be a positive multiple of the cluster size")
        if self.minibatch_size < _TILE_ROWS or self.minibatch_size % _TILE_ROWS != 0:
            raise ValueError("minibatch_size must be a positive multiple of 256")
        if self.schedule_capacity_factor < 1.0:
            raise ValueError("schedule_capacity_factor must be at least one")


def _jaxlib_include_dir() -> Path:
    return Path(jaxlib.__file__).resolve().parent / "include"


def _cuda_include_dirs() -> tuple[Path, ...]:
    include_dirs: list[Path] = []
    for distribution_name in (
        "nvidia-cuda-runtime",
        "nvidia-cuda-nvcc",
        "nvidia-cuda-crt",
        "nvidia-cuda-cccl",
        "nvidia-cuda-runtime-cu13",
        "nvidia-cuda-nvcc-cu13",
    ):
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for file in distribution.files or ():
            if file.name not in ("cuda_runtime.h", "host_config.h", "fatbinary_section.h", "target"):
                continue
            header = Path(distribution.locate_file(file)).resolve()
            include_dir = header.parent.parent if header.parent.name in ("crt", "nv") else header.parent
            if include_dir not in include_dirs:
                include_dirs.append(include_dir)
    if not include_dirs:
        raise RuntimeError("The CUDA 13 runtime headers are not installed")
    return tuple(include_dirs)


def _cuda_toolchain_root(build_dir: Path) -> Path:
    toolchain_root = build_dir / "cuda"
    for distribution_name in ("nvidia-cuda-nvcc", "nvidia-nvvm", "nvidia-cuda-runtime"):
        distribution = importlib.metadata.distribution(distribution_name)
        for file in distribution.files or ():
            package_path = Path(file)
            try:
                relative_path = package_path.relative_to(Path("nvidia/cu13"))
            except ValueError:
                continue
            if relative_path.parts[0] not in ("bin", "lib", "nvvm"):
                continue
            source = Path(distribution.locate_file(file)).resolve()
            destination = toolchain_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                continue
            if destination.name == "nvcc":
                shutil.copy2(source, destination)
            else:
                os.symlink(source, destination)
    cudart = toolchain_root / "lib" / "libcudart.so.13"
    cudart_link = toolchain_root / "lib" / "libcudart.so"
    if cudart.is_file() and not cudart_link.exists():
        os.symlink(cudart.name, cudart_link)
    nvcc = toolchain_root / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError("The CUDA 13 compiler is not installed")
    return toolchain_root


def _ffi_source() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "mok_forward_ffi.cu"


def _prepared_source_bytes(source_root: Path) -> tuple[bytes, bytes, bytes]:
    mok_lines = (source_root / "csrc" / "mok_megakernel.cuh").read_text().splitlines(keepends=True)
    first_host_wrapper = next(
        index for index, line in enumerate(mok_lines) if "dispatch_mlp_swiglu_combine_fwd_mxfp8(" in line
    )
    while first_host_wrapper > 0 and "static __host__" not in mok_lines[first_host_wrapper]:
        first_host_wrapper -= 1
    mok_text = "".join(mok_lines[:first_host_wrapper])
    mok_text = mok_text.replace('#include "pyutils/torchutils.cuh"\n', "")
    mok_text = mok_text.replace("#include <ATen/ops/empty.h>\n", "")
    mok_text = mok_text.replace("#include <ATen/ops/empty_like.h>\n", "")
    mok_text = mok_text.replace("#include <ATen/ops/zeros.h>\n", "")
    mok_text += "\n};  // struct dispatch_mlp_swiglu_combiner\n"

    mxfp8_lines = (source_root / "csrc" / "mxfp8.cuh").read_text().splitlines(keepends=True)
    first_mxfp8_host = next(index for index, line in enumerate(mxfp8_lines) if "static __host__" in line)
    mxfp8_text = "".join(mxfp8_lines[:first_mxfp8_host])
    mxfp8_text = mxfp8_text.replace('#include "pyutils/torchutils.cuh"\n', "")
    mxfp8_text = mxfp8_text.replace("#include <ATen/ops/empty.h>\n", "")
    mxfp8_text += "\n}  // namespace mxfp8_quantize\n"

    utils_text = """#pragma once
#include "kittens.cuh"
namespace utils {
enum class RoutedPrecision { BF16, MXFP8 };
}  // namespace utils
"""
    return mok_text.encode(), mxfp8_text.encode(), utils_text.encode()


def _build_path() -> tuple[Path, Path]:
    source_root = mok_source_root()
    prepared = _prepared_source_bytes(source_root)
    key = hashlib.sha256()
    key.update(_ffi_source().read_bytes())
    for data in prepared:
        key.update(data)
    key.update(_BUILD_SCHEMA.encode())
    key.update(mok_cuda_arch_flag().encode())
    for include_dir in _cuda_include_dirs():
        key.update(str(include_dir).encode())
    digest = key.hexdigest()[:16]
    build_dir = mok_cache_root("mok_forward_ffi") / digest
    return build_dir, build_dir / "libmok_forward_ffi.so"


def _write_prepared_sources(build_dir: Path) -> None:
    source_root = mok_source_root()
    mok_text, mxfp8_text, utils_text = _prepared_source_bytes(source_root)
    generated = build_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "mok_megakernel.cuh").write_bytes(mok_text)
    (generated / "mxfp8.cuh").write_bytes(mxfp8_text)
    (generated / "utils.cuh").write_bytes(utils_text)


def _build_library() -> Path:
    build_dir, library_path = _build_path()
    if library_path.is_file():
        return library_path
    build_dir.mkdir(parents=True, exist_ok=True)
    _write_prepared_sources(build_dir)
    source_root = mok_source_root()
    toolchain_root = _cuda_toolchain_root(build_dir)
    command = [
        str(toolchain_root / "bin" / "nvcc"),
        str(_ffi_source()),
        "-o",
        str(library_path),
        "-std=c++20",
        "-shared",
        "-Xcompiler=-fPIC",
        "--cudart=shared",
        "--expt-extended-lambda",
        "--expt-relaxed-constexpr",
        "-forward-unknown-to-host-compiler",
        "-ftemplate-backtrace-limit=0",
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-DNDEBUG",
        f"-DKITTENS_{mok_cuda_arch().replace('sm_', 'SM', 1).replace('a', '')}",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        mok_cuda_arch_flag(),
        "-I",
        str(build_dir / "generated"),
        "-I",
        str(source_root / "third_party" / "ThunderKittens" / "include"),
        "-I",
        str(_jaxlib_include_dir()),
        "-L",
        str(toolchain_root / "lib"),
    ]
    for include_dir in _cuda_include_dirs():
        command.extend(("-I", str(include_dir)))
    subprocess.run(command, check=True)
    return library_path


def _load_library() -> ctypes.CDLL:
    library = getattr(_load_library, "_library", None)
    if library is not None:
        return library
    global_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    _load_library._cuda_driver = ctypes.CDLL("libcuda.so.1", mode=global_mode)
    library = ctypes.CDLL(str(_build_library()), mode=global_mode)
    _load_library._library = library
    return library


def _last_error(default: str) -> str:
    function = getattr(_load_library(), _LAST_ERROR_SYMBOL)
    function.argtypes = []
    function.restype = ctypes.c_char_p
    message = function()
    return message.decode() if message else default


def _register_target() -> None:
    if getattr(_register_target, "_done", False):
        return
    handler = getattr(_load_library(), _TARGET)
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        _TARGET,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )
    jax.ffi.register_ffi_target_as_batch_partitionable(_TARGET)
    _register_target._done = True


def _runtime_signature(num_tokens: int, hidden_dim: int, top_k: int) -> tuple[int, int, int, int]:
    return _NUM_DEVICES, num_tokens, hidden_dim, top_k


def ensure_runtime(*, num_tokens: int, hidden_dim: int, top_k: int) -> None:
    """Build the FFI library and allocate its peer-visible workspaces."""
    signature = _runtime_signature(num_tokens, hidden_dim, top_k)
    if getattr(ensure_runtime, "_signature", None) == signature:
        return
    _register_target()
    function = getattr(_load_library(), _INIT_SYMBOL)
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    function.restype = ctypes.c_int
    if function(*signature) != 0:
        raise RuntimeError(_last_error("Mixture-of-Kittens runtime initialization failed"))
    ensure_runtime._signature = signature


def shutdown_runtime() -> None:
    """Free peer-visible workspaces when the process exits."""
    if getattr(ensure_runtime, "_signature", None) is None:
        return
    function = getattr(_load_library(), _SHUTDOWN_SYMBOL)
    function.argtypes = []
    function.restype = ctypes.c_int
    if function() != 0:
        raise RuntimeError(_last_error("Mixture-of-Kittens runtime shutdown failed"))
    ensure_runtime._signature = None


atexit.register(shutdown_runtime)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def schedule_capacity(
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    config: MoKForwardConfig,
) -> int:
    """Return the static padded route capacity for one expert rank."""
    if config.schedule_capacity_factor < 1.0:
        raise ValueError("schedule_capacity_factor must be at least one")
    if num_local_experts < 1:
        raise ValueError("num_local_experts must be at least one")
    assignments = math.ceil(num_tokens * top_k * config.schedule_capacity_factor)
    expert_padding = num_local_experts * (_TILE_ROWS - 1)
    return _round_up(assignments + expert_padding, config.minibatch_size)


def forward_bf16_local(
    x: jax.Array,
    router_weights: jax.Array,
    shared_gate: jax.Array,
    routed_gate: jax.Array,
    shared_up: jax.Array,
    routed_up: jax.Array,
    shared_down: jax.Array,
    routed_down: jax.Array,
    schedule_peer_rank: jax.Array,
    schedule_peer_token_idx: jax.Array,
    num_scheduled_tokens: jax.Array,
    tokens_per_expert: jax.Array,
    *,
    config: MoKForwardConfig,
) -> jax.Array:
    """Run fused dispatch, shared and routed SwiGLU, combine, and epilogue."""
    if x.ndim != 2 or router_weights.ndim != 2:
        raise ValueError("x and router_weights must be rank two")
    num_tokens, hidden_dim = x.shape
    top_k = router_weights.shape[1]
    if router_weights.shape[0] != num_tokens:
        raise ValueError("router_weights must have the same token count as x")
    if schedule_peer_rank.shape != schedule_peer_token_idx.shape:
        raise ValueError("schedule arrays must have the same shape")
    capacity = schedule_peer_rank.shape[0]
    if capacity % config.minibatch_size != 0:
        raise ValueError("schedule capacity must be divisible by minibatch_size")
    if num_tokens % _TILE_ROWS != 0 or capacity % _TILE_ROWS != 0:
        raise ValueError("token count and schedule capacity must be divisible by 256")

    ensure_runtime(num_tokens=num_tokens, hidden_dim=hidden_dim, top_k=top_k)
    intermediate_dim = shared_gate.shape[0]
    routed_rows = capacity
    global_minibatches = capacity // config.minibatch_size
    global_row_blocks = capacity // (_TILE_ROWS // _CLUSTER_SIZE)
    shared_row_blocks = num_tokens // _TILE_ROWS
    routed_row_blocks = capacity // _TILE_ROWS
    gate_ready = (shared_row_blocks + routed_row_blocks) * (intermediate_dim // _TILE_ROWS)
    hidden_ready = shared_row_blocks + routed_row_blocks

    result_shapes = (
        jax.ShapeDtypeStruct((num_tokens, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, intermediate_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((routed_rows, hidden_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((global_minibatches,), jnp.int32),
        jax.ShapeDtypeStruct((gate_ready,), jnp.int32),
        jax.ShapeDtypeStruct((hidden_ready,), jnp.int32),
        jax.ShapeDtypeStruct((global_minibatches,), jnp.int32),
        jax.ShapeDtypeStruct((global_row_blocks,), jnp.int32),
    )
    results = jax.ffi.ffi_call(
        _TARGET,
        result_shapes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        jnp.asarray(x, dtype=jnp.bfloat16),
        jnp.asarray(router_weights, dtype=jnp.float32),
        jnp.asarray(shared_gate, dtype=jnp.bfloat16),
        jnp.asarray(routed_gate, dtype=jnp.bfloat16),
        jnp.asarray(shared_up, dtype=jnp.bfloat16),
        jnp.asarray(routed_up, dtype=jnp.bfloat16),
        jnp.asarray(shared_down, dtype=jnp.bfloat16),
        jnp.asarray(routed_down, dtype=jnp.bfloat16),
        jnp.asarray(schedule_peer_rank, dtype=jnp.int32),
        jnp.asarray(schedule_peer_token_idx, dtype=jnp.int32),
        jnp.reshape(jnp.asarray(num_scheduled_tokens, dtype=jnp.int32), (1,)),
        jnp.asarray(tokens_per_expert, dtype=jnp.int32),
        top_k=np.int32(top_k),
        num_comm_sms=np.int32(config.num_comm_sms),
        macrobatch_size=np.int32(capacity),
        minibatch_size=np.int32(config.minibatch_size),
    )
    return results[0]
