# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX FFI wrappers for DeepEP transport kernels."""

from __future__ import annotations

import atexit
import base64
import ctypes
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Literal, NamedTuple, cast

import jax
import jax._src.distributed as jax_distributed
import jax.numpy as jnp
import jaxlib
import numpy as np

from levanter.kernels.deepep.availability import (
    BUILD_WITH_TORCH_EXTENSION_ENV,
    DISABLE_SM90_ENV,
    INTERNODE_TRANSPORT_REQUIRED_FILES,
    LOAD_AS_PYTHON_MODULE_ENV,
    TRANSPORT_REQUIRED_FILES,
    deepep_cache_root,
    deepep_cuda_arch,
    deepep_cuda_arch_flag,
    deepep_cuda_include_dirs,
    deepep_cuda_library_dirs,
    deepep_layout_source,
    deepep_nvcc_path,
    deepep_nvshmem_config,
    deepep_rdma_include_dirs,
    deepep_source_root,
    deepep_torch_cuda_arch_list,
    env_flag,
)

_DISPATCH_TARGET = "levanter_deepep_dispatch_intranode"
_DISPATCH_WITH_ASSIGNMENTS_TARGET = "levanter_deepep_dispatch_intranode_with_assignments"
_DISPATCH_CACHED_TARGET = "levanter_deepep_dispatch_intranode_cached"
_COMBINE_TARGET = "levanter_deepep_combine_intranode"
_DISPATCH_INTERNODE_TARGET = "levanter_deepep_dispatch_internode"
_DISPATCH_INTERNODE_CACHED_TARGET = "levanter_deepep_dispatch_internode_cached"
_COMBINE_INTERNODE_TARGET = "levanter_deepep_combine_internode"
_COMBINE_INTERNODE_X_ONLY_TARGET = "levanter_deepep_combine_internode_x_only"
_COMBINE_INTERNODE_WITH_LOCAL_COLLAPSE_TARGET = "levanter_deepep_combine_internode_with_local_collapse"
_COMBINE_INTERNODE_X_ONLY_WITH_LOCAL_COLLAPSE_TARGET = "levanter_deepep_combine_internode_x_only_with_local_collapse"
_COLLAPSE_LOCAL_ASSIGNMENTS_INTERNODE_TARGET = "levanter_deepep_collapse_local_assignments_internode"
_COLLAPSE_LOCAL_ASSIGNMENTS_INTERNODE_BWD_TARGET = "levanter_deepep_collapse_local_assignments_internode_bwd"
_DISPATCH_INTERNODE_BWD_FUSED_TARGET = "levanter_deepep_dispatch_internode_bwd_fused"
_PACK_LOCAL_ASSIGNMENTS_TARGET = "levanter_deepep_pack_local_assignments"
_PACK_LOCAL_ASSIGNMENTS_FROM_COUNTS_TARGET = "levanter_deepep_pack_local_assignments_from_counts"
_COLLAPSE_LOCAL_ASSIGNMENTS_TARGET = "levanter_deepep_collapse_local_assignments"
_ASSIGNMENT_GRADIENTS_TARGET = "levanter_deepep_assignment_gradients"
_INIT_SYMBOL = "levanter_deepep_init_intranode_runtime"
_SHUTDOWN_SYMBOL = "levanter_deepep_shutdown_intranode_runtime"
_INIT_INTERNODE_SYMBOL = "levanter_deepep_init_internode_runtime"
_SHUTDOWN_INTERNODE_SYMBOL = "levanter_deepep_shutdown_internode_runtime"
_INTERNODE_STATUS_SYMBOL = "levanter_deepep_internode_runtime_status"
_RUN_INTERNODE_MAPPED_COUNTER_SMOKE_SYMBOL = "levanter_deepep_run_internode_mapped_counter_smoke"
_LAST_ERROR_SYMBOL = "levanter_deepep_last_error"
_LOCAL_DEVICE_ID_SYMBOL = "levanter_deepep_get_local_device_id"
_LOCAL_NVSHMEM_UNIQUE_ID_SIZE_SYMBOL = "levanter_deepep_get_local_nvshmem_unique_id_size"
_LOCAL_NVSHMEM_UNIQUE_ID_SYMBOL = "levanter_deepep_get_local_nvshmem_unique_id"
_PREPARE_INTERNODE_PROCESS_RUNTIME_SYMBOL = "levanter_deepep_prepare_internode_process_runtime"
_LOCAL_INTERNODE_IPC_HANDLE_SIZE_SYMBOL = "levanter_deepep_get_local_internode_ipc_handle_size"
_LOCAL_INTERNODE_IPC_HANDLE_SYMBOL = "levanter_deepep_get_local_internode_ipc_handle"
_INIT_INTERNODE_PROCESS_RUNTIME_SYMBOL = "levanter_deepep_init_internode_process_runtime"
_PROBE_DISPATCH_SYMBOL = "levanter_deepep_probe_dispatch_kernel_attributes"
_RUN_HOST_DISPATCH_SYMBOL = "levanter_deepep_run_host_dispatch_round"
_RUN_HOST_INTERNODE_DISPATCH_SYMBOL = "levanter_deepep_run_host_internode_dispatch_round"
_EXTENDED_INTRNODE_DISPATCH_MACRO = "LEVANTER_DEEPEP_EXTENDED_INTRNODE_DISPATCH"
_PYEXT_MODULE_NAME_MACRO = "LEVANTER_DEEPEP_PYEXT_MODULE_NAME"
_DISPATCH_THREADS_ENV = "DEEPEP_DISPATCH_NUM_THREADS"
_DISPATCH_NUM_SMS_ENV = "DEEPEP_DISPATCH_NUM_SMS"
_DISPATCH_MAX_SEND_TOKENS_ENV = "DEEPEP_DISPATCH_MAX_SEND_TOKENS"
_DISPATCH_MAX_RECV_TOKENS_ENV = "DEEPEP_DISPATCH_MAX_RECV_TOKENS"
_COMBINE_NUM_SMS_ENV = "DEEPEP_COMBINE_NUM_SMS"
_COMBINE_MAX_SEND_TOKENS_ENV = "DEEPEP_COMBINE_MAX_SEND_TOKENS"
_COMBINE_MAX_RECV_TOKENS_ENV = "DEEPEP_COMBINE_MAX_RECV_TOKENS"
_INTERNODE_DISPATCH_NUM_SMS_ENV = "DEEPEP_INTERNODE_DISPATCH_NUM_SMS"
_INTERNODE_DISPATCH_MAX_NVL_SEND_TOKENS_ENV = "DEEPEP_INTERNODE_DISPATCH_MAX_NVL_SEND_TOKENS"
_INTERNODE_DISPATCH_MAX_NVL_RECV_TOKENS_ENV = "DEEPEP_INTERNODE_DISPATCH_MAX_NVL_RECV_TOKENS"
_INTERNODE_DISPATCH_MAX_RDMA_SEND_TOKENS_ENV = "DEEPEP_INTERNODE_DISPATCH_MAX_RDMA_SEND_TOKENS"
_INTERNODE_DISPATCH_MAX_RDMA_RECV_TOKENS_ENV = "DEEPEP_INTERNODE_DISPATCH_MAX_RDMA_RECV_TOKENS"
_INTERNODE_LOW_LATENCY_MODE_ENV = "DEEPEP_INTERNODE_LOW_LATENCY_MODE"
_INTERNODE_SOURCE_META_BYTES_ENV = "DEEPEP_INTERNODE_SOURCE_META_BYTES"
_INTERNODE_RANKS_PER_NODE_ENV = "DEEPEP_RANKS_PER_NODE"
_INTERNODE_ASSIGNMENT_GRADIENT_MODE_ENV = "LEVANTER_DEEPEP_INTERNODE_ASSIGNMENT_GRADIENT_MODE"
_INTERNODE_ASSIGNMENT_GRADIENT_JAX = "jax"
_INTERNODE_ASSIGNMENT_GRADIENT_FFI = "ffi"
_INTERNODE_ASSIGNMENT_GRADIENT_FUSED = "fused"
_DEFAULT_INTERNODE_QPS_PER_RANK = 24
_DEFAULT_INTERNODE_SOURCE_META_BYTES = 8
_BUILD_CACHE_SCHEMA_VERSION = "transport_ffi_raw_dlink_v43"
_LIBRARY_DLOPEN_MODE = getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_GLOBAL", 0)
_SM100_TMA_DISPATCH_THREADS = 512
_UPSTREAM_DISPATCH_THREADS = 768


@dataclass(frozen=True)
class IntranodeConfig:
    num_sms: int
    num_max_send_tokens: int
    num_max_recv_tokens: int


@dataclass(frozen=True)
class InternodeConfig:
    num_sms: int
    num_max_nvl_chunked_send_tokens: int
    num_max_nvl_chunked_recv_tokens: int
    num_max_rdma_chunked_send_tokens: int
    num_max_rdma_chunked_recv_tokens: int


_internode_exchange_counter = 0


@dataclass(frozen=True)
class InternodeProcessBootstrapMetadata:
    """Per-process metadata DeepEP needs before internode dispatch/combine can run."""

    process_index: int
    local_device_ids: tuple[int, ...]
    nvshmem_unique_id: bytes | None
    local_ipc_handles: tuple[bytes, ...] = ()
    node_rank: int | None = None
    local_rank: int | None = None
    ranks_per_node: int | None = None
    process_model: Literal["process_per_gpu", "process_per_node"] | None = None


def _internode_assignment_gradient_mode() -> str:
    mode = os.environ.get(_INTERNODE_ASSIGNMENT_GRADIENT_MODE_ENV, _INTERNODE_ASSIGNMENT_GRADIENT_JAX)
    if mode not in {
        _INTERNODE_ASSIGNMENT_GRADIENT_JAX,
        _INTERNODE_ASSIGNMENT_GRADIENT_FFI,
        _INTERNODE_ASSIGNMENT_GRADIENT_FUSED,
    }:
        raise RuntimeError(
            f"{_INTERNODE_ASSIGNMENT_GRADIENT_MODE_ENV} must be "
            f"{_INTERNODE_ASSIGNMENT_GRADIENT_JAX!r}, {_INTERNODE_ASSIGNMENT_GRADIENT_FFI!r}, "
            f"or {_INTERNODE_ASSIGNMENT_GRADIENT_FUSED!r}, got {mode!r}"
        )
    return mode


def _internode_low_latency_mode() -> bool:
    return env_flag(_INTERNODE_LOW_LATENCY_MODE_ENV)


@dataclass(frozen=True)
class InternodeRuntimeStatus:
    initialized: bool
    process_rank: int
    process_count: int
    num_local_ranks: int
    num_global_ranks: int
    num_nvl_bytes: int
    num_rdma_bytes: int


@dataclass(frozen=True)
class InternodeProcessTopology:
    """JAX process topology for DeepEP normal-mode internode transport."""

    process_index: int
    process_count: int
    process_model: Literal["process_per_gpu", "process_per_node"]
    node_rank: int
    node_count: int
    ranks_per_node: int
    visible_local_gpus: int
    local_rank: int | None = None


@dataclass(frozen=True)
class BuildArtifact:
    library_path: Path
    module_name: str | None


class TransportBuildMode(StrEnum):
    INTRANODE = "intranode"
    INTERNODE = "internode"


class DeepEPDispatch(NamedTuple):
    recv_x: jax.Array
    recv_topk_idx: jax.Array
    recv_topk_weights: jax.Array
    recv_src_idx: jax.Array
    rank_prefix_matrix: jax.Array
    channel_prefix_matrix: jax.Array
    recv_channel_prefix_matrix: jax.Array
    send_head: jax.Array
    local_expert_counts: jax.Array
    num_recv_tokens: jax.Array


class DeepEPDispatchWithAssignments(NamedTuple):
    recv_x: jax.Array
    recv_topk_weights: jax.Array
    recv_src_idx: jax.Array
    rank_prefix_matrix: jax.Array
    channel_prefix_matrix: jax.Array
    recv_channel_prefix_matrix: jax.Array
    send_head: jax.Array
    local_group_sizes: jax.Array
    num_recv_tokens: jax.Array
    x_dispatch: jax.Array
    assignment_weights: jax.Array
    recv_token_indices: jax.Array
    assignment_destinations: jax.Array


class DeepEPInternodeDispatch(NamedTuple):
    recv_x: jax.Array
    recv_topk_idx: jax.Array
    recv_topk_weights: jax.Array
    is_token_in_rank: jax.Array
    recv_src_meta: jax.Array
    rdma_channel_prefix_matrix: jax.Array
    recv_rdma_channel_prefix_matrix: jax.Array
    recv_rdma_rank_prefix_sum: jax.Array
    gbl_channel_prefix_matrix: jax.Array
    recv_gbl_channel_prefix_matrix: jax.Array
    recv_gbl_rank_prefix_sum: jax.Array
    send_rdma_head: jax.Array
    send_nvl_head: jax.Array
    local_expert_counts: jax.Array
    num_recv_tokens: jax.Array
    num_recv_rdma_tokens: jax.Array
    local_group_sizes: jax.Array
    x_dispatch: jax.Array
    assignment_weights: jax.Array
    recv_token_indices: jax.Array
    assignment_destinations: jax.Array


_DEFAULT_DISPATCH_CONFIGS = {
    2: IntranodeConfig(num_sms=20, num_max_send_tokens=24, num_max_recv_tokens=256),
    4: IntranodeConfig(num_sms=120, num_max_send_tokens=12, num_max_recv_tokens=256),
    8: IntranodeConfig(num_sms=20, num_max_send_tokens=6, num_max_recv_tokens=256),
}

_DEFAULT_COMBINE_CONFIGS = {
    2: IntranodeConfig(num_sms=20, num_max_send_tokens=10, num_max_recv_tokens=256),
    4: IntranodeConfig(num_sms=120, num_max_send_tokens=18, num_max_recv_tokens=256),
    8: IntranodeConfig(num_sms=20, num_max_send_tokens=4, num_max_recv_tokens=256),
}

_DEFAULT_INTERNODE_DISPATCH_CONFIG = InternodeConfig(
    num_sms=24,
    num_max_nvl_chunked_send_tokens=8,
    num_max_nvl_chunked_recv_tokens=512,
    num_max_rdma_chunked_send_tokens=16,
    num_max_rdma_chunked_recv_tokens=128,
)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def internode_dispatch_clean_buffer_size_hint(
    *,
    hidden: int,
    topk: int,
    num_rdma_ranks: int,
    config: InternodeConfig | None = None,
    num_scales: int = 0,
    num_nvl_ranks: int = 8,
) -> tuple[int, int]:
    """Return minimum runtime bytes for pinned DeepEP internode clean regions.

    This mirrors pinned DeepEP's `get_rdma_clean_meta` and `get_nvl_clean_meta`
    bounds. DeepEP asserts these ranges fit inside the runtime NVL/RDMA buffers
    before normal-mode dispatch. The returned values are lower bounds, not a
    full workspace budget.
    """
    if hidden <= 0 or topk <= 0 or num_rdma_ranks <= 0 or num_nvl_ranks <= 0:
        raise ValueError(
            "internode buffer hint inputs must be positive, "
            f"got {hidden=} {topk=} {num_rdma_ranks=} {num_nvl_ranks=}"
        )
    dispatch_config = config or _DEFAULT_INTERNODE_DISPATCH_CONFIG
    num_channels = dispatch_config.num_sms // 2
    if num_channels <= 0:
        raise ValueError(f"internode dispatch config must use a positive num_sms, got {dispatch_config.num_sms}")

    int32_bytes = 4
    float32_bytes = 4
    bf16_bytes = 2
    int4_bytes = 16
    source_meta_bytes = _DEFAULT_INTERNODE_SOURCE_META_BYTES
    hidden_int4 = hidden * bf16_bytes // int4_bytes
    if hidden * bf16_bytes % int4_bytes != 0:
        raise ValueError(f"DeepEP internode requires hidden bf16 bytes divisible by int4 bytes, got {hidden=}")
    bytes_per_token = _align_up(
        hidden_int4 * int4_bytes
        + source_meta_bytes
        + num_scales * float32_bytes
        + topk * int32_bytes
        + topk * float32_bytes,
        int4_bytes,
    )
    rdma_offset_ints = (
        bytes_per_token * dispatch_config.num_max_rdma_chunked_recv_tokens * num_rdma_ranks * 2 * num_channels
    ) // int32_bytes
    rdma_clean_ints = (num_nvl_ranks * 2 + 4) * num_rdma_ranks * 2 * num_channels
    nvl_offset_ints = (
        dispatch_config.num_max_nvl_chunked_recv_tokens * bytes_per_token * num_nvl_ranks * num_channels
    ) // int32_bytes
    nvl_clean_ints = num_nvl_ranks * (2 * num_rdma_ranks + 2) * num_channels
    return (
        _align_up((nvl_offset_ints + nvl_clean_ints) * int32_bytes, 128),
        _align_up((rdma_offset_ints + rdma_clean_ints) * int32_bytes, 128),
    )


def _jaxlib_include_dir() -> Path:
    return Path(jaxlib.__file__).resolve().parent / "include"


def _ffi_source() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "deepep_transport_ffi.cu"


def _python_extension_source() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "deepep_transport_pyext.cc"


def _launch_compat_header() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "deepep_launch_compat.cuh"


def _intranode_source(deepep_root: Path) -> Path:
    return deepep_root / "csrc" / "kernels" / "intranode.cu"


def _internode_sources(deepep_root: Path) -> tuple[Path, ...]:
    return (
        deepep_root / "csrc" / "kernels" / "internode.cu",
        deepep_root / "csrc" / "kernels" / "internode_ll.cu",
        deepep_root / "csrc" / "kernels" / "pcie.cu",
    )


def _cuda_sources(
    deepep_root: Path, build_mode: TransportBuildMode = TransportBuildMode.INTRANODE
) -> tuple[Path, ...]:
    sources = (
        _ffi_source(),
        deepep_root / "csrc" / "kernels" / "runtime.cu",
        deepep_layout_source(deepep_root),
        _intranode_source(deepep_root),
    )
    if build_mode is TransportBuildMode.INTERNODE:
        return (*sources, *_internode_sources(deepep_root))
    return sources


def _deepep_source_root(build_mode: TransportBuildMode = TransportBuildMode.INTRANODE) -> Path:
    required_files = (
        INTERNODE_TRANSPORT_REQUIRED_FILES if build_mode is TransportBuildMode.INTERNODE else TRANSPORT_REQUIRED_FILES
    )
    return deepep_source_root(
        required_files=required_files,
        purpose="the DeepEP JAX FFI transport kernels",
        requires_layout_source=True,
    )


def _cache_root() -> Path:
    return deepep_cache_root("deepep_transport_ffi")


def _python_extension_suffix() -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if suffix:
        return suffix
    if not importlib.machinery.EXTENSION_SUFFIXES:
        raise RuntimeError("Could not determine the Python extension suffix for DeepEP transport FFI")
    return importlib.machinery.EXTENSION_SUFFIXES[0]


def _python_include_dirs() -> tuple[Path, ...]:
    include_dirs: list[Path] = []
    for key in ("include", "platinclude"):
        raw = sysconfig.get_paths().get(key)
        if not raw:
            continue
        path = Path(raw)
        if path not in include_dirs:
            include_dirs.append(path)
    if not include_dirs:
        raise RuntimeError("Could not determine Python include directories for DeepEP transport FFI")
    return tuple(include_dirs)


def _cuda_arch_flag() -> list[str]:
    return deepep_cuda_arch_flag()


def _sm90_compile_flags(*, include_launch_compat: bool = True) -> list[str]:
    flags: list[str] = []
    if include_launch_compat:
        flags.extend(["-include", str(_launch_compat_header())])
    if env_flag(DISABLE_SM90_ENV):
        flags.append("-DDISABLE_SM90_FEATURES")
    return flags


def _use_torch_extension_build() -> bool:
    return env_flag(BUILD_WITH_TORCH_EXTENSION_ENV)


def _load_as_python_module() -> bool:
    return env_flag(LOAD_AS_PYTHON_MODULE_ENV)


def _torch_cuda_arch_list() -> str:
    return deepep_torch_cuda_arch_list()


def _dispatch_thread_override() -> int | None:
    raw = os.environ.get(_DISPATCH_THREADS_ENV)
    if raw is not None:
        try:
            threads = int(raw)
        except ValueError as exc:
            raise RuntimeError(f"{_DISPATCH_THREADS_ENV} must be an integer, got {raw!r}") from exc
        if threads < 256 or threads % 32 != 0:
            raise RuntimeError(f"{_DISPATCH_THREADS_ENV} must be a multiple of 32 and at least 256, got {threads}")
        return threads
    if deepep_cuda_arch() == "sm_100" and not env_flag(DISABLE_SM90_ENV):
        return _SM100_TMA_DISPATCH_THREADS
    return None


def _intranode_source_bytes(deepep_root: Path) -> bytes:
    source = _intranode_source(deepep_root)
    text = source.read_text()
    if "#include <cuda_bf16.h>" not in text:
        text = "#include <cuda_bf16.h>\n" + text
    dispatch_threads = _dispatch_thread_override()
    if dispatch_threads is not None and dispatch_threads != _UPSTREAM_DISPATCH_THREADS:
        dispatch_start = text.find("\nvoid dispatch(")
        combine_start = text.find("\nvoid combine(", dispatch_start)
        needle = f"    constexpr int kNumThreads = {_UPSTREAM_DISPATCH_THREADS};"
        threads_start = text.find(needle, dispatch_start)
        if dispatch_start < 0 or threads_start < 0 or (combine_start >= 0 and threads_start > combine_start):
            raise RuntimeError("Could not patch DeepEP intranode dispatch thread count for this source tree")
        text = (
            text[:threads_start]
            + f"    constexpr int kNumThreads = {dispatch_threads};"
            + text[threads_start + len(needle) :]
        )

    replacements = (
        (
            "    auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \\\n"
            "    SET_SHARED_MEMORY_FOR_TMA(kernel); \\",
            "    SET_SHARED_MEMORY_FOR_TMA((dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>)); \\\n"
            "    auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \\",
        ),
        (
            "    auto kernel = combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>; \\\n"
            "    SET_SHARED_MEMORY_FOR_TMA(kernel); \\",
            "    SET_SHARED_MEMORY_FOR_TMA((combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>)); \\\n"
            "    auto kernel = combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>; \\",
        ),
    )
    for old, new in replacements:
        if old not in text and not env_flag(DISABLE_SM90_ENV):
            raise RuntimeError("Could not patch DeepEP intranode TMA launch pattern for this source tree")
        text = text.replace(old, new, 1)

    assignment_dispatch_threads = dispatch_threads or _UPSTREAM_DISPATCH_THREADS
    text = _add_assignment_dispatch_source(text, dispatch_threads=assignment_dispatch_threads)
    return text.encode("utf-8")


def _add_assignment_dispatch_source(text: str, *, dispatch_threads: int = _UPSTREAM_DISPATCH_THREADS) -> str:
    kernel_anchor = "__global__ void __launch_bounds__(kNumThreads, 1)\ndispatch("
    kernel_anchor_start = text.find(kernel_anchor)
    kernel_start = text.rfind("\ntemplate", 0, kernel_anchor_start) + 1
    host_start = text.find("\nvoid dispatch(", kernel_start)
    if kernel_anchor_start < 0 or kernel_start <= 0 or host_start < 0:
        raise RuntimeError("Could not find DeepEP intranode dispatch kernel for assignment-native patch")

    assignment_kernel = text[kernel_start:host_start]
    assignment_kernel = assignment_kernel.replace(
        "dispatch(int4* recv_x, float* recv_x_scales, float* recv_x_sf_scale_for_nvfp4, int* recv_src_idx, "
        "int64_t* recv_topk_idx, float* recv_topk_weights, int* recv_channel_offset,",
        "dispatch_assignments(int4* recv_x, float* recv_x_scales, float* recv_x_sf_scale_for_nvfp4, "
        "int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights, int* recv_channel_offset,\n"
        "                     int4* x_dispatch, nv_bfloat16* assignment_weights, int* recv_token_indices,\n"
        "                     int* local_group_cursors, int* recv_assignment_indices, int* assignment_destinations,",
        1,
    )

    receiver_anchor = "        // Workers for receiving and copying into buffer\n"
    receiver_start = assignment_kernel.find(receiver_anchor)
    receive_start = assignment_kernel.find("            // Copy data\n", receiver_start)
    receive_end = assignment_kernel.find("            // Copy `x_scales`\n", receive_start)
    if receiver_start < 0 or receive_start < 0 or receive_end < 0:
        raise RuntimeError("Could not find DeepEP intranode receive loop for assignment-native patch")
    assignment_receive = """            // Copy queue payloads directly into local-expert assignment order.
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens; chunk_idx += num_recv_warps_per_rank) {
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                int recv_token_idx = total_offset + chunk_idx;

                if (lane_id == 0)
                    recv_src_idx[recv_token_idx] = ld_nc_global(channel_src_idx_buffers.buffer() + token_idx_in_buffer);

                #pragma unroll
                for (int token_topk_idx = 0; token_topk_idx < num_topk; ++ token_topk_idx) {
                    auto buffer_idx = token_idx_in_buffer * num_topk + token_topk_idx;
                    auto recv_idx = static_cast<int64_t>(recv_token_idx) * num_topk + token_topk_idx;
                    int local_expert = ld_nc_global(channel_topk_idx_buffers.buffer() + buffer_idx);
                    float weight = ld_nc_global(channel_topk_weights_buffers.buffer() + buffer_idx);
                    if (lane_id == token_topk_idx) {
                        recv_topk_idx[recv_idx] = local_expert;
                        recv_topk_weights[recv_idx] = weight;
                    }
                    if (local_expert < 0)
                        continue;

                    int destination = -1;
                    if (lane_id == 0) {
                        destination = atomicAdd(local_group_cursors + local_expert, 1);
                        int assignment_idx = recv_token_idx * num_topk + token_topk_idx;
                        recv_token_indices[destination] = recv_token_idx;
                        recv_assignment_indices[destination] = assignment_idx;
                        assignment_destinations[assignment_idx] = destination;
                        assignment_weights[destination] = __float2bfloat16(weight);
                    }
                    destination = __shfl_sync(0xffffffff, destination, 0);
                    auto shifted_buffer_x_int4 = channel_x_buffers.buffer() + token_idx_in_buffer * hidden_int4;
                    auto shifted_x_dispatch_int4 = x_dispatch + static_cast<int64_t>(destination) * hidden_int4;
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_x_dispatch_int4, shifted_buffer_x_int4,
                                       ld_nc_global, st_na_global);
                }
            }

"""
    assignment_kernel = assignment_kernel[:receive_start] + assignment_receive + assignment_kernel[receive_end:]

    combine_start = text.find(
        "\ntemplate<typename dtype_t, int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>", host_start
    )
    if combine_start < 0:
        raise RuntimeError("Could not find DeepEP intranode combine kernel insertion point")
    assignment_host = """
void dispatch_assignments(void* recv_x, float* recv_x_scales, float* recv_x_sf_scale_for_nvfp4,
                          int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights,
                          int* recv_channel_offset, void* x_dispatch, nv_bfloat16* assignment_weights,
                          int* recv_token_indices, int* local_group_cursors,
                          int* recv_assignment_indices, int* assignment_destinations, int* send_head,
                          const void* x, const float* x_scales, const float* sf_scale_for_nvfp4,
                          const int64_t* topk_idx, const float* topk_weights,
                          const bool* is_token_in_rank, const int* channel_prefix_matrix,
                          int num_tokens, int num_worst_tokens, int hidden_int4, int num_topk, int num_experts,
                          int num_scales, int num_sf_scales_for_nvfp4, int scale_token_stride,
                          int scale_hidden_stride, int sf_scale_for_nvfp4_token_stride,
                          int sf_scale_for_nvfp4_hidden_stride, void** buffer_ptrs, int rank, int num_ranks,
                          cudaStream_t stream, int num_sms, int num_max_send_tokens,
                          int num_recv_buffer_tokens) {
    constexpr int kNumThreads = __DISPATCH_THREADS__;
    constexpr int kNumTMABytesPerWarp = 8192;
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);
#endif

    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(static_cast<int64_t>(num_sf_scales_for_nvfp4) * sf_scale_for_nvfp4_hidden_stride < std::numeric_limits<int>::max());

#define DISPATCH_ASSIGNMENTS_LAUNCH_CASE(ranks) { \\
    SET_SHARED_MEMORY_FOR_TMA((dispatch_assignments<ranks, kNumThreads, kNumTMABytesPerWarp>)); \\
    auto kernel = dispatch_assignments<ranks, kNumThreads, kNumTMABytesPerWarp>; \\
    LAUNCH_KERNEL(&cfg, kernel, \\
        reinterpret_cast<int4*>(recv_x), recv_x_scales, recv_x_sf_scale_for_nvfp4, recv_src_idx, recv_topk_idx, recv_topk_weights, recv_channel_offset, \\
        reinterpret_cast<int4*>(x_dispatch), assignment_weights, recv_token_indices, local_group_cursors, recv_assignment_indices, assignment_destinations, \\
        send_head, reinterpret_cast<const int4*>(x), x_scales, sf_scale_for_nvfp4, topk_idx, topk_weights, \\
        is_token_in_rank, channel_prefix_matrix, \\
        num_tokens, num_worst_tokens, hidden_int4, num_topk, num_experts, num_scales, num_sf_scales_for_nvfp4, \\
        scale_token_stride, scale_hidden_stride, sf_scale_for_nvfp4_token_stride, sf_scale_for_nvfp4_hidden_stride, \\
        buffer_ptrs, rank, \\
        num_max_send_tokens, num_recv_buffer_tokens); \\
    } \\
    break

    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_RANKS(DISPATCH_ASSIGNMENTS_LAUNCH_CASE);
#undef DISPATCH_ASSIGNMENTS_LAUNCH_CASE
}

""".replace(
        "__DISPATCH_THREADS__", str(dispatch_threads)
    )
    return text[:combine_start] + "\n" + assignment_kernel + "\n" + assignment_host + text[combine_start:]


def _prepare_intranode_source(build_dir: Path, deepep_root: Path) -> Path:
    patched_source = build_dir / "generated" / "intranode.cu"
    patched_source.parent.mkdir(parents=True, exist_ok=True)
    patched_source.write_bytes(_intranode_source_bytes(deepep_root))
    return patched_source


def _prepared_cuda_sources(
    build_dir: Path,
    deepep_root: Path,
    build_mode: TransportBuildMode = TransportBuildMode.INTRANODE,
) -> tuple[Path, ...]:
    sources = (
        _ffi_source(),
        deepep_root / "csrc" / "kernels" / "runtime.cu",
        deepep_layout_source(deepep_root),
        _prepare_intranode_source(build_dir, deepep_root),
    )
    if build_mode is TransportBuildMode.INTERNODE:
        return (*sources, *_internode_sources(deepep_root))
    return sources


def _preload_torch_shared_libraries() -> None:
    import torch  # noqa: PLC0415  # optional dep: torch

    lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if not lib_dir.is_dir():
        raise RuntimeError(f"Could not find Torch shared libraries under {lib_dir}")
    torch_libraries = (
        "libc10.so",
        "libc10_cuda.so",
        "libtorch_cpu.so",
        "libtorch_cuda.so",
        "libtorch.so",
        "libtorch_python.so",
    )
    for library_name in torch_libraries:
        library_path = lib_dir / library_name
        if not library_path.is_file():
            continue
        ctypes.CDLL(str(library_path), mode=_LIBRARY_DLOPEN_MODE)


def _build_artifact(build_mode: TransportBuildMode = TransportBuildMode.INTRANODE) -> BuildArtifact:
    deepep_root = _deepep_source_root(build_mode)
    key = hashlib.sha256()
    sources = _cuda_sources(deepep_root, build_mode) + (
        deepep_root / "csrc" / "config.hpp",
        deepep_root / "csrc" / "kernels" / "api.cuh",
        deepep_root / "csrc" / "kernels" / "configs.cuh",
    )
    for path in sources:
        key.update(str(path).encode("utf-8"))
        if path == _intranode_source(deepep_root):
            key.update(_intranode_source_bytes(deepep_root))
        else:
            key.update(path.read_bytes())
    if _load_as_python_module():
        key.update(_python_extension_source().read_bytes())
    key.update(_launch_compat_header().read_bytes())
    key.update(Path(__file__).read_bytes())
    key.update(str(_jaxlib_include_dir()).encode("utf-8"))
    key.update(" ".join(str(path) for path in deepep_cuda_include_dirs()).encode("utf-8"))
    key.update(" ".join(str(path) for path in deepep_cuda_library_dirs()).encode("utf-8"))
    key.update(str(deepep_root).encode("utf-8"))
    key.update(_BUILD_CACHE_SCHEMA_VERSION.encode("utf-8"))
    key.update(build_mode.value.encode("utf-8"))
    key.update(" ".join(_cuda_arch_flag()).encode("utf-8"))
    key.update(" ".join(_sm90_compile_flags()).encode("utf-8"))
    key.update(str(_dispatch_thread_override()).encode("utf-8"))
    key.update(str(int(_has_extended_intranode_dispatch_signature(deepep_root))).encode("utf-8"))
    key.update(str(int(_use_torch_extension_build())).encode("utf-8"))
    key.update(str(int(_load_as_python_module())).encode("utf-8"))
    if build_mode is TransportBuildMode.INTERNODE:
        nvshmem_config = deepep_nvshmem_config()
        for path in (nvshmem_config.host_library_path, nvshmem_config.device_library_path):
            stat = path.stat()
            key.update(str(path).encode("utf-8"))
            key.update(str(stat.st_size).encode("utf-8"))
            key.update(str(stat.st_mtime_ns).encode("utf-8"))
    digest = key.hexdigest()[:16]
    out_dir = _cache_root() / digest
    out_dir.mkdir(parents=True, exist_ok=True)
    if _load_as_python_module():
        module_name = f"deepep_transport_ffi_{digest}"
        return BuildArtifact(
            library_path=out_dir / f"{module_name}{_python_extension_suffix()}",
            module_name=module_name,
        )
    return BuildArtifact(library_path=out_dir / "libdeepep_transport_ffi.so", module_name=None)


def _shared_library_path() -> Path:
    return _build_artifact().library_path


def _nvshmem_compile_flags(build_mode: TransportBuildMode) -> list[str]:
    if build_mode is TransportBuildMode.INTRANODE:
        return ["-DDISABLE_NVSHMEM"]
    config = deepep_nvshmem_config()
    flags: list[str] = []
    for include_dir in config.include_dirs:
        flags.extend(["-I", str(include_dir)])
    for include_dir in deepep_rdma_include_dirs():
        flags.extend(["-I", str(include_dir)])
    return flags


def _nvshmem_link_flags(build_mode: TransportBuildMode) -> list[str]:
    if build_mode is TransportBuildMode.INTRANODE:
        return []
    config = deepep_nvshmem_config()
    return [
        "-L",
        str(config.host_library_path.parent),
        f"-l:{config.host_library_name}",
        "-Xlinker",
        "-rpath",
        "-Xlinker",
        str(config.host_library_path.parent),
    ]


def _nvshmem_device_link_flags(build_mode: TransportBuildMode) -> list[str]:
    if build_mode is TransportBuildMode.INTRANODE:
        return []
    return [str(deepep_nvshmem_config().device_library_path)]


def _nvshmem_torch_link_flags(build_mode: TransportBuildMode) -> list[str]:
    if build_mode is TransportBuildMode.INTRANODE:
        return []
    config = deepep_nvshmem_config()
    return [
        "-L",
        str(config.host_library_path.parent),
        f"-l:{config.host_library_name}",
        f"-Wl,-rpath,{config.host_library_path.parent}",
    ]


def _nvcc_common_flags(
    deepep_root: Path,
    compatibility_flags: list[str],
    *,
    build_mode: TransportBuildMode = TransportBuildMode.INTRANODE,
    include_launch_compat: bool = True,
) -> list[str]:
    return [
        "-std=c++17",
        "-Xcompiler",
        "-fPIC",
        "--expt-relaxed-constexpr",
        "-O3",
        *_nvshmem_compile_flags(build_mode),
        "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
        *compatibility_flags,
        *_sm90_compile_flags(include_launch_compat=include_launch_compat),
        *_cuda_arch_flag(),
        "-I",
        str(_jaxlib_include_dir()),
        *[flag for include_dir in deepep_cuda_include_dirs() for flag in ("-I", str(include_dir))],
        "-I",
        str(deepep_root),
        "-I",
        str(deepep_root / "csrc"),
        "-I",
        str(deepep_root / "csrc" / "kernels"),
    ]


def _require_nvcc() -> str:
    nvcc = deepep_nvcc_path()
    if nvcc is None:
        raise RuntimeError(
            "DeepEP transport FFI build requires nvcc. Install nvidia-cuda-nvcc or use a CUDA devel image with nvcc on PATH."
        )
    return nvcc


def _build_object_files(
    *,
    build_dir: Path,
    deepep_root: Path,
    compatibility_flags: list[str],
    build_mode: TransportBuildMode,
) -> list[Path]:
    objects_dir = build_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    common_flags = _nvcc_common_flags(deepep_root, compatibility_flags, build_mode=build_mode)
    compile_flags = [
        _require_nvcc(),
        *common_flags,
        "-rdc=true",
        "--ptxas-options=--register-usage-level=10",
        "-c",
    ]

    object_paths: list[Path] = []
    for source in _prepared_cuda_sources(objects_dir, deepep_root, build_mode):
        object_path = objects_dir / f"{source.stem}.o"
        cmd = [
            *compile_flags,
            str(source),
            "-o",
            str(object_path),
        ]
        subprocess.run(cmd, check=True)
        object_paths.append(object_path)
    return object_paths


def _build_python_extension_shim_object(*, build_dir: Path, module_name: str) -> Path:
    object_path = build_dir / "deepep_transport_pyext.o"
    include_flags: list[str] = []
    for include_dir in _python_include_dirs():
        include_flags.extend(["-I", str(include_dir)])
    cmd = [
        "c++",
        "-std=c++17",
        "-O3",
        "-fPIC",
        f"-D{_PYEXT_MODULE_NAME_MACRO}={module_name}",
        *include_flags,
        "-c",
        str(_python_extension_source()),
        "-o",
        str(object_path),
    ]
    subprocess.run(cmd, check=True)
    return object_path


def _device_link_objects(
    *,
    build_dir: Path,
    deepep_root: Path,
    compatibility_flags: list[str],
    object_paths: list[Path],
    build_mode: TransportBuildMode,
) -> Path:
    dlink_object = build_dir / "deepep_transport_ffi.dlink.o"
    common_flags = _nvcc_common_flags(
        deepep_root,
        compatibility_flags,
        build_mode=build_mode,
        include_launch_compat=False,
    )
    cmd = [
        _require_nvcc(),
        *common_flags,
        "-dlink",
        *[str(path) for path in object_paths],
        *_nvshmem_device_link_flags(build_mode),
        "-o",
        str(dlink_object),
    ]
    subprocess.run(cmd, check=True)
    return dlink_object


def _link_shared_library(
    *,
    out_path: Path,
    object_paths: list[Path],
    dlink_object: Path,
    build_mode: TransportBuildMode,
    extra_object_paths: list[Path] | None = None,
) -> None:
    all_object_paths = [*object_paths, *(extra_object_paths or [])]
    cuda_library_flags = [flag for library_dir in deepep_cuda_library_dirs() for flag in ("-L", str(library_dir))]
    cuda_rpath_flags = [
        flag
        for library_dir in deepep_cuda_library_dirs()
        for flag in ("-Xlinker", "-rpath", "-Xlinker", str(library_dir))
    ]
    cmd = [
        _require_nvcc(),
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "--cudart=shared",
        *_cuda_arch_flag(),
        *cuda_library_flags,
        *[str(path) for path in all_object_paths],
        str(dlink_object),
        *_nvshmem_device_link_flags(build_mode),
        "-lcuda",
        *_nvshmem_link_flags(build_mode),
        *cuda_rpath_flags,
        "-o",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def _build_raw_shared_library(
    artifact: BuildArtifact,
    deepep_root: Path,
    compatibility_flags: list[str],
    build_mode: TransportBuildMode,
) -> None:
    if _use_torch_extension_build() and _load_as_python_module():
        raise RuntimeError(
            f"{BUILD_WITH_TORCH_EXTENSION_ENV}=1 is not yet supported with {LOAD_AS_PYTHON_MODULE_ENV}=1"
        )
    out_path = artifact.library_path
    build_dir = out_path.parent / "raw_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    object_paths = _build_object_files(
        build_dir=build_dir,
        deepep_root=deepep_root,
        compatibility_flags=compatibility_flags,
        build_mode=build_mode,
    )
    dlink_object = _device_link_objects(
        build_dir=build_dir,
        deepep_root=deepep_root,
        compatibility_flags=compatibility_flags,
        object_paths=object_paths,
        build_mode=build_mode,
    )
    extra_object_paths: list[Path] = []
    if artifact.module_name is not None:
        extra_object_paths.append(
            _build_python_extension_shim_object(build_dir=build_dir, module_name=artifact.module_name)
        )
    _link_shared_library(
        out_path=out_path,
        object_paths=object_paths,
        dlink_object=dlink_object,
        build_mode=build_mode,
        extra_object_paths=extra_object_paths,
    )


def _build_shared_library(
    artifact: BuildArtifact,
    build_mode: TransportBuildMode = TransportBuildMode.INTRANODE,
) -> None:
    deepep_root = _deepep_source_root(build_mode)
    compatibility_flags = _compatibility_compile_flags(deepep_root)
    out_path = artifact.library_path
    if _use_torch_extension_build() and _load_as_python_module():
        raise RuntimeError(
            f"{BUILD_WITH_TORCH_EXTENSION_ENV}=1 is not yet supported with {LOAD_AS_PYTHON_MODULE_ENV}=1"
        )
    if _use_torch_extension_build():
        _build_with_torch_extension(out_path, deepep_root, compatibility_flags, build_mode)
        return
    _build_raw_shared_library(artifact, deepep_root, compatibility_flags, build_mode)


def _build_with_torch_extension(
    out_path: Path,
    deepep_root: Path,
    compatibility_flags: list[str],
    build_mode: TransportBuildMode,
) -> None:
    from torch.utils import cpp_extension  # noqa: PLC0415  # optional dep: torch

    build_dir = out_path.parent
    name = f"deepep_transport_ffi_{build_dir.name}"
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = _torch_cuda_arch_list()
    try:
        cpp_extension.load(
            name=name,
            sources=[
                *[str(source) for source in _prepared_cuda_sources(build_dir, deepep_root, build_mode)],
            ],
            extra_cuda_cflags=[
                "-O3",
                *_nvshmem_compile_flags(build_mode),
                "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
                *compatibility_flags,
                "-rdc=true",
                "--ptxas-options=--register-usage-level=10",
                *_sm90_compile_flags(),
            ],
            extra_ldflags=[
                *_nvshmem_torch_link_flags(build_mode),
            ],
            extra_include_paths=[
                str(_jaxlib_include_dir()),
                *[str(path) for path in deepep_cuda_include_dirs()],
                str(deepep_root),
                str(deepep_root / "csrc"),
            ],
            build_directory=str(build_dir),
            verbose=True,
            with_cuda=True,
            is_python_module=False,
        )
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list

    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    candidates = []
    for suffix in suffixes:
        candidates.extend(build_dir.glob(f"{name}*{suffix}"))
    if not candidates:
        raise RuntimeError(f"torch cpp_extension did not produce a shared library in {build_dir}")
    built_path = max(candidates, key=lambda path: path.stat().st_mtime)
    if built_path != out_path:
        shutil.copy2(built_path, out_path)


def _load_torch_extension_python_module(artifact: BuildArtifact):
    if artifact.module_name is None:
        raise RuntimeError("Torch extension Python-module load requires a module-named build artifact")

    cached_module = getattr(_load_torch_extension_python_module, "_module", None)
    cached_path = getattr(_load_torch_extension_python_module, "_path", None)
    if cached_module is not None and cached_path == artifact.library_path:
        return cached_module

    build_mode = TransportBuildMode.INTRANODE
    deepep_root = _deepep_source_root(build_mode)
    compatibility_flags = _compatibility_compile_flags(deepep_root)
    build_dir = artifact.library_path.parent
    build_dir.mkdir(parents=True, exist_ok=True)
    setup_path = build_dir / "setup.py"
    setup_path.write_text(
        "\n".join(
            (
                "from setuptools import setup",
                "from torch.utils.cpp_extension import BuildExtension, CUDAExtension",
                "",
                f"MODULE_NAME = {artifact.module_name!r}",
                f"SOURCES = {([str(_python_extension_source()), str(_ffi_source()), *[str(path) for path in _prepared_cuda_sources(build_dir, deepep_root, build_mode)[1:]]])!r}",
                (
                    "INCLUDE_DIRS = "
                    + repr(
                        [
                            str(_jaxlib_include_dir()),
                            *[str(path) for path in deepep_cuda_include_dirs()],
                            str(deepep_root),
                            str(deepep_root / "csrc"),
                        ]
                    )
                ),
                f"CXX_FLAGS = {['-O3', f'-D{_PYEXT_MODULE_NAME_MACRO}={artifact.module_name}']!r}",
                (
                    "NVCC_FLAGS = "
                    + repr(
                        [
                            "-O3",
                            *_nvshmem_compile_flags(build_mode),
                            "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
                            *compatibility_flags,
                            "-rdc=true",
                            "--ptxas-options=--register-usage-level=10",
                            *_sm90_compile_flags(),
                        ]
                    )
                ),
                f"NVCC_DLINK_FLAGS = {['-dlink', *_nvshmem_device_link_flags(build_mode)]!r}",
                "",
                "setup(",
                "    name=MODULE_NAME,",
                "    ext_modules=[",
                "        CUDAExtension(",
                "            name=MODULE_NAME,",
                "            sources=SOURCES,",
                "            include_dirs=INCLUDE_DIRS,",
                "            extra_compile_args={",
                "                'cxx': CXX_FLAGS,",
                "                'nvcc': NVCC_FLAGS,",
                "                'nvcc_dlink': NVCC_DLINK_FLAGS,",
                "            },",
                (
                    "            extra_link_args="
                    + repr(
                        [
                            *[
                                flag
                                for library_dir in deepep_cuda_library_dirs()
                                for flag in ("-L", str(library_dir), f"-Wl,-rpath,{library_dir}")
                            ],
                            "-lcuda",
                            *_nvshmem_torch_link_flags(build_mode),
                        ]
                    )
                    + ","
                ),
                "            dlink=True,",
                "        )",
                "    ],",
                "    cmdclass={'build_ext': BuildExtension},",
                ")",
            )
        )
        + "\n"
    )
    build_env = os.environ.copy()
    build_env["TORCH_CUDA_ARCH_LIST"] = _torch_cuda_arch_list()
    subprocess.run(
        [sys.executable, str(setup_path), "build_ext", "--inplace"],
        cwd=build_dir,
        env=build_env,
        check=True,
    )
    if not artifact.library_path.exists():
        candidates = sorted(build_dir.glob(f"{artifact.module_name}*{_python_extension_suffix()}"))
        if not candidates:
            raise RuntimeError(f"CUDAExtension build did not produce {artifact.library_path.name} in {build_dir}")
        if candidates[-1] != artifact.library_path:
            shutil.copy2(candidates[-1], artifact.library_path)

    _preload_torch_shared_libraries()
    module = _load_python_module(artifact)
    module_path = Path(module.__file__).resolve()
    _load_torch_extension_python_module._module = module
    _load_torch_extension_python_module._path = module_path
    return module


def _has_extended_intranode_dispatch_signature(deepep_root: Path) -> bool:
    api_header = deepep_root / "csrc" / "kernels" / "api.cuh"
    return "recv_x_sf_scale_for_nvfp4" in api_header.read_text()


def _compatibility_compile_flags(deepep_root: Path) -> list[str]:
    if _has_extended_intranode_dispatch_signature(deepep_root):
        return [f"-D{_EXTENDED_INTRNODE_DISPATCH_MACRO}=1"]
    return []


def build_transport_library(
    build_mode: TransportBuildMode = TransportBuildMode.INTRANODE,
) -> BuildArtifact:
    """Compile the DeepEP transport FFI shared library and return its cache artifact."""
    artifact = _build_artifact(build_mode)
    if not artifact.library_path.exists():
        _build_shared_library(artifact, build_mode)
    return artifact


def _load_library(build_mode: TransportBuildMode = TransportBuildMode.INTRANODE) -> ctypes.CDLL:
    artifact = _build_artifact(build_mode)
    if build_mode is TransportBuildMode.INTERNODE and _use_torch_extension_build() and _load_as_python_module():
        raise RuntimeError(f"{LOAD_AS_PYTHON_MODULE_ENV}=1 is not supported for DeepEP internode transport builds yet")
    if _use_torch_extension_build() and _load_as_python_module():
        module = _load_torch_extension_python_module(artifact)
        return ctypes.CDLL(str(Path(module.__file__).resolve()), mode=_LIBRARY_DLOPEN_MODE)
    if not artifact.library_path.exists():
        _build_shared_library(artifact, build_mode)
    if artifact.module_name is not None:
        module = _load_python_module(artifact)
        return ctypes.CDLL(str(Path(module.__file__).resolve()), mode=_LIBRARY_DLOPEN_MODE)
    if build_mode is TransportBuildMode.INTERNODE:
        _preload_nvshmem_host_library()
    return ctypes.CDLL(str(artifact.library_path), mode=_LIBRARY_DLOPEN_MODE)


def _preload_nvshmem_host_library() -> None:
    """Force the DeepEP internode FFI to use the configured NVSHMEM host library."""
    config = deepep_nvshmem_config()
    cached_path = getattr(_preload_nvshmem_host_library, "_path", None)
    if cached_path == config.host_library_path:
        return
    ctypes.CDLL(str(config.host_library_path), mode=_LIBRARY_DLOPEN_MODE)
    _preload_nvshmem_host_library._path = config.host_library_path


def _load_python_module(artifact: BuildArtifact):
    if artifact.module_name is None:
        raise RuntimeError("Build artifact does not describe a Python extension module")
    cached_module = getattr(_load_python_module, "_module", None)
    cached_path = getattr(_load_python_module, "_path", None)
    if cached_module is not None and cached_path == artifact.library_path:
        return cached_module

    spec = importlib.util.spec_from_file_location(artifact.module_name, artifact.library_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create a Python extension spec for {artifact.library_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[artifact.module_name] = module
    spec.loader.exec_module(module)
    _load_python_module._module = module
    _load_python_module._path = artifact.library_path
    return module


def _register_targets() -> None:
    if getattr(_register_targets, "_done", False):
        return
    library = _load_library()
    for target in (
        _DISPATCH_TARGET,
        _DISPATCH_WITH_ASSIGNMENTS_TARGET,
        _DISPATCH_CACHED_TARGET,
        _COMBINE_TARGET,
        _PACK_LOCAL_ASSIGNMENTS_TARGET,
        _PACK_LOCAL_ASSIGNMENTS_FROM_COUNTS_TARGET,
        _COLLAPSE_LOCAL_ASSIGNMENTS_TARGET,
        _ASSIGNMENT_GRADIENTS_TARGET,
    ):
        handler = getattr(library, target)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(
            target,
            jax.ffi.pycapsule(handler),
            platform="CUDA",
            api_version=1,
        )
        jax.ffi.register_ffi_target_as_batch_partitionable(target)
    _register_targets._done = True


def _register_internode_targets() -> None:
    if getattr(_register_internode_targets, "_done", False):
        return
    library = _load_library(TransportBuildMode.INTERNODE)
    for target in (
        _DISPATCH_INTERNODE_TARGET,
        _DISPATCH_INTERNODE_CACHED_TARGET,
        _COMBINE_INTERNODE_TARGET,
        _COMBINE_INTERNODE_X_ONLY_TARGET,
        _COMBINE_INTERNODE_WITH_LOCAL_COLLAPSE_TARGET,
        _COMBINE_INTERNODE_X_ONLY_WITH_LOCAL_COLLAPSE_TARGET,
        _COLLAPSE_LOCAL_ASSIGNMENTS_INTERNODE_TARGET,
        _COLLAPSE_LOCAL_ASSIGNMENTS_INTERNODE_BWD_TARGET,
        _DISPATCH_INTERNODE_BWD_FUSED_TARGET,
    ):
        handler = getattr(library, target)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(
            target,
            jax.ffi.pycapsule(handler),
            platform="CUDA",
            api_version=1,
        )
        jax.ffi.register_ffi_target_as_batch_partitionable(target)
    _register_internode_targets._done = True


def _library_function(name: str, build_mode: TransportBuildMode = TransportBuildMode.INTRANODE):
    library = _load_library(build_mode)
    return getattr(library, name)


def _last_error(build_mode: TransportBuildMode = TransportBuildMode.INTRANODE) -> str:
    last_error = _library_function(_LAST_ERROR_SYMBOL, build_mode)
    last_error.argtypes = []
    last_error.restype = ctypes.c_char_p
    message = last_error()
    return message.decode("utf-8") if message else ""


def local_device_id(build_mode: TransportBuildMode = TransportBuildMode.INTERNODE) -> int:
    """Return the current CUDA device id through the DeepEP transport library."""
    get_device_id = _library_function(_LOCAL_DEVICE_ID_SYMBOL, build_mode)
    get_device_id.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_device_id.restype = ctypes.c_int
    device_id = ctypes.c_int()
    status = get_device_id(ctypes.byref(device_id))
    if status != 0:
        raise RuntimeError(f"Failed to get DeepEP local device id: {_last_error(build_mode) or 'unknown error'}")
    return int(device_id.value)


def local_nvshmem_unique_id() -> bytes:
    """Return this process's DeepEP NVSHMEM unique id from an internode-enabled build."""
    get_size = _library_function(_LOCAL_NVSHMEM_UNIQUE_ID_SIZE_SYMBOL, TransportBuildMode.INTERNODE)
    get_size.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    get_size.restype = ctypes.c_int
    unique_id_bytes = ctypes.c_size_t()
    status = get_size(ctypes.byref(unique_id_bytes))
    if status != 0:
        raise RuntimeError(f"Failed to get DeepEP NVSHMEM unique id size: {_last_error(TransportBuildMode.INTERNODE)}")

    buffer = (ctypes.c_uint8 * unique_id_bytes.value)()
    get_unique_id = _library_function(_LOCAL_NVSHMEM_UNIQUE_ID_SYMBOL, TransportBuildMode.INTERNODE)
    get_unique_id.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    get_unique_id.restype = ctypes.c_int
    written = ctypes.c_size_t()
    status = get_unique_id(buffer, unique_id_bytes.value, ctypes.byref(written))
    if status != 0:
        raise RuntimeError(f"Failed to get DeepEP NVSHMEM unique id: {_last_error(TransportBuildMode.INTERNODE)}")
    return bytes(buffer[: written.value])


def local_internode_bootstrap_metadata(
    *,
    include_nvshmem_unique_id: bool | None = None,
    num_nvl_bytes: int | None = None,
    ranks_per_node: int | None = None,
) -> InternodeProcessBootstrapMetadata:
    """Return host-local DeepEP internode bootstrap metadata for the current JAX process."""
    local_gpu_devices = tuple(device for device in jax.local_devices() if device.platform == "gpu")
    if not local_gpu_devices:
        raise RuntimeError("DeepEP internode bootstrap metadata requires at least one local GPU device")
    topology: InternodeProcessTopology | None = None
    local_ipc_handles: tuple[bytes, ...] = ()
    if ranks_per_node is not None or os.environ.get(_INTERNODE_RANKS_PER_NODE_ENV):
        topology = current_internode_process_topology(
            ranks_per_node=ranks_per_node,
            visible_local_gpus=len(local_gpu_devices),
        )
    if include_nvshmem_unique_id is None:
        include_nvshmem_unique_id = _should_publish_nvshmem_unique_id(topology)
    if topology is not None and topology.process_model == "process_per_gpu":
        if num_nvl_bytes is None:
            raise RuntimeError("DeepEP process-per-GPU bootstrap metadata requires num_nvl_bytes for IPC export")
        local_ipc_handles = (local_internode_ipc_handle(topology=topology, num_nvl_bytes=num_nvl_bytes),)
    unique_id = local_nvshmem_unique_id() if include_nvshmem_unique_id else None
    return InternodeProcessBootstrapMetadata(
        process_index=jax.process_index(),
        local_device_ids=tuple(int(device.id) for device in local_gpu_devices),
        nvshmem_unique_id=unique_id,
        local_ipc_handles=local_ipc_handles,
        node_rank=None if topology is None else topology.node_rank,
        local_rank=None if topology is None else topology.local_rank,
        ranks_per_node=None if topology is None else topology.ranks_per_node,
        process_model=None if topology is None else topology.process_model,
    )


def _should_publish_nvshmem_unique_id(topology: InternodeProcessTopology | None) -> bool:
    if topology is None or topology.process_model == "process_per_node":
        return jax.process_index() == 0
    return topology.node_rank == 0


def _prepare_internode_process_runtime(*, topology: InternodeProcessTopology, num_nvl_bytes: int) -> None:
    prepare = _library_function(_PREPARE_INTERNODE_PROCESS_RUNTIME_SYMBOL, TransportBuildMode.INTERNODE)
    prepare.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
    ]
    prepare.restype = ctypes.c_int
    if topology.local_rank is None:
        raise RuntimeError("DeepEP process-per-GPU runtime preparation requires a local rank")
    status = prepare(
        topology.process_index,
        topology.process_count,
        topology.local_rank,
        topology.ranks_per_node,
        num_nvl_bytes,
    )
    if status != 0:
        raise RuntimeError(
            f"Failed to prepare DeepEP internode process runtime: {_last_error(TransportBuildMode.INTERNODE)}"
        )


def local_internode_ipc_handle(*, topology: InternodeProcessTopology, num_nvl_bytes: int) -> bytes:
    """Prepare the local process-per-GPU runtime and return its CUDA IPC handle."""
    _prepare_internode_process_runtime(topology=topology, num_nvl_bytes=num_nvl_bytes)
    get_size = _library_function(_LOCAL_INTERNODE_IPC_HANDLE_SIZE_SYMBOL, TransportBuildMode.INTERNODE)
    get_size.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    get_size.restype = ctypes.c_int
    handle_bytes = ctypes.c_size_t()
    status = get_size(ctypes.byref(handle_bytes))
    if status != 0:
        raise RuntimeError(f"Failed to get DeepEP IPC handle size: {_last_error(TransportBuildMode.INTERNODE)}")

    buffer = (ctypes.c_uint8 * handle_bytes.value)()
    get_handle = _library_function(_LOCAL_INTERNODE_IPC_HANDLE_SYMBOL, TransportBuildMode.INTERNODE)
    get_handle.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    get_handle.restype = ctypes.c_int
    written = ctypes.c_size_t()
    status = get_handle(buffer, handle_bytes.value, ctypes.byref(written))
    if status != 0:
        raise RuntimeError(f"Failed to get DeepEP IPC handle: {_last_error(TransportBuildMode.INTERNODE)}")
    return bytes(buffer[: written.value])


def _encode_optional_bytes(value: bytes | None) -> str | None:
    if value is None:
        return None
    return base64.b64encode(value).decode("ascii")


def _decode_optional_bytes(value: str | None) -> bytes | None:
    if value is None:
        return None
    return base64.b64decode(value.encode("ascii"))


def _metadata_to_json(metadata: InternodeProcessBootstrapMetadata) -> str:
    return json.dumps(
        {
            "process_index": metadata.process_index,
            "local_device_ids": list(metadata.local_device_ids),
            "local_ipc_handles": [_encode_optional_bytes(handle) for handle in metadata.local_ipc_handles],
            "nvshmem_unique_id": _encode_optional_bytes(metadata.nvshmem_unique_id),
            "node_rank": metadata.node_rank,
            "local_rank": metadata.local_rank,
            "ranks_per_node": metadata.ranks_per_node,
            "process_model": metadata.process_model,
        },
        sort_keys=True,
    )


def _metadata_from_json(payload: str) -> InternodeProcessBootstrapMetadata:
    raw = json.loads(payload)
    return InternodeProcessBootstrapMetadata(
        process_index=int(raw["process_index"]),
        local_device_ids=tuple(int(device_id) for device_id in raw["local_device_ids"]),
        local_ipc_handles=tuple(base64.b64decode(handle.encode("ascii")) for handle in raw["local_ipc_handles"]),
        nvshmem_unique_id=_decode_optional_bytes(raw["nvshmem_unique_id"]),
        node_rank=None if raw.get("node_rank") is None else int(raw["node_rank"]),
        local_rank=None if raw.get("local_rank") is None else int(raw["local_rank"]),
        ranks_per_node=None if raw.get("ranks_per_node") is None else int(raw["ranks_per_node"]),
        process_model=cast(Literal["process_per_gpu", "process_per_node"] | None, raw.get("process_model")),
    )


def exchange_internode_bootstrap_metadata(
    metadata: InternodeProcessBootstrapMetadata,
    *,
    timeout: float = 200.0,
) -> tuple[InternodeProcessBootstrapMetadata, ...]:
    """All-gather DeepEP internode bootstrap metadata across JAX processes."""
    global _internode_exchange_counter
    if jax.process_count() == 1:
        return (metadata,)

    client = jax_distributed.global_state.client
    if client is None:
        raise RuntimeError("DeepEP internode metadata exchange requires jax.distributed to be initialized")

    exchange_id = _internode_exchange_counter
    key_prefix = f"levanter_deepep_internode_bootstrap_{exchange_id}"
    timeout_ms = int(timeout * 1000.0)
    client.key_value_set(f"{key_prefix}_{metadata.process_index}", _metadata_to_json(metadata))
    client.wait_at_barrier(f"{key_prefix}_barrier", timeout_in_ms=timeout_ms)
    gathered = tuple(
        _metadata_from_json(client.blocking_key_value_get(f"{key_prefix}_{process_index}", timeout_in_ms=timeout_ms))
        for process_index in range(jax.process_count())
    )
    _internode_exchange_counter += 1
    return gathered


def _root_nvshmem_unique_id(
    metadata: tuple[InternodeProcessBootstrapMetadata, ...],
    *,
    topology: InternodeProcessTopology | None = None,
) -> bytes:
    if topology is not None and topology.process_model == "process_per_gpu":
        if topology.local_rank is None:
            raise RuntimeError("DeepEP process-per-GPU root NVSHMEM id requires a local rank")
        roots = [
            entry.nvshmem_unique_id
            for entry in metadata
            if entry.node_rank == 0 and entry.local_rank == topology.local_rank and entry.nvshmem_unique_id is not None
        ]
        if len(roots) != 1:
            raise RuntimeError(
                "DeepEP process-per-GPU bootstrap expected exactly one node-0 NVSHMEM root id for "
                f"local_rank={topology.local_rank}, got {len(roots)}"
            )
        return roots[0]
    roots = [entry.nvshmem_unique_id for entry in metadata if entry.nvshmem_unique_id is not None]
    if len(roots) != 1:
        raise RuntimeError(f"DeepEP internode bootstrap expected exactly one root NVSHMEM unique id, got {len(roots)}")
    return roots[0]


def _internode_local_rank_count(metadata: tuple[InternodeProcessBootstrapMetadata, ...]) -> int:
    if not metadata:
        raise RuntimeError("DeepEP internode bootstrap metadata is empty")
    process_indices = tuple(entry.process_index for entry in metadata)
    expected_indices = tuple(range(len(metadata)))
    if process_indices != expected_indices:
        raise RuntimeError(
            "DeepEP internode bootstrap metadata must be ordered by process index; "
            f"got {process_indices}, expected {expected_indices}"
        )
    local_rank_counts = {len(entry.local_device_ids) for entry in metadata}
    if len(local_rank_counts) != 1:
        raise RuntimeError(f"DeepEP internode bootstrap needs equal local GPU counts, got {local_rank_counts}")
    num_local_ranks = next(iter(local_rank_counts))
    if num_local_ranks <= 0:
        raise RuntimeError("DeepEP internode bootstrap needs at least one local GPU per process")
    return num_local_ranks


def _internode_metadata_process_model(
    metadata: tuple[InternodeProcessBootstrapMetadata, ...],
) -> Literal["process_per_gpu", "process_per_node"] | None:
    """Return the explicit DeepEP process model encoded in exchanged metadata."""
    explicit_models = {entry.process_model for entry in metadata if entry.process_model is not None}
    if not explicit_models:
        return None
    if len(explicit_models) != 1:
        raise RuntimeError(f"DeepEP internode bootstrap metadata has mixed process models: {explicit_models}")
    process_model = next(iter(explicit_models))
    missing = [
        entry.process_index
        for entry in metadata
        if entry.process_model != process_model
        or entry.node_rank is None
        or entry.ranks_per_node is None
        or (process_model == "process_per_gpu" and entry.local_rank is None)
    ]
    if missing:
        raise RuntimeError(
            "DeepEP internode bootstrap metadata is missing explicit topology fields for processes "
            f"{missing}; process_model={process_model!r}"
        )
    return process_model


def _internode_ipc_handle_blob(metadata: tuple[InternodeProcessBootstrapMetadata, ...]) -> bytes:
    handles: list[bytes] = []
    handle_bytes: int | None = None
    for entry in metadata:
        if len(entry.local_ipc_handles) != 1:
            raise RuntimeError(
                "DeepEP process-per-GPU bootstrap metadata requires exactly one IPC handle per process; "
                f"process {entry.process_index} has {len(entry.local_ipc_handles)}"
            )
        handle = entry.local_ipc_handles[0]
        if handle_bytes is None:
            handle_bytes = len(handle)
        if len(handle) != handle_bytes:
            raise RuntimeError("DeepEP process-per-GPU IPC handles have inconsistent sizes")
        handles.append(handle)
    return b"".join(handles)


def _configure_internode_nvshmem_env(*, num_qps_per_rank: int = _DEFAULT_INTERNODE_QPS_PER_RANK) -> None:
    """Set DeepEP's required normal-mode NVSHMEM/IBGDA defaults before internode init."""
    if num_qps_per_rank <= 0:
        raise ValueError("num_qps_per_rank must be positive for DeepEP internode NVSHMEM")

    # CoreWeave's IB/IPoIB interfaces are not the right TCP bootstrap path.
    # Match the cluster-level NCCL OOB interface when Iris injects an exact
    # interface such as "=enp157s0np0"; otherwise fall back to the broad
    # CoreWeave exclusion list. IBGDA still owns the data path.
    bootstrap_ifname = os.environ.get("NCCL_SOCKET_IFNAME", "^ibs,ibp,lo,docker,veth,cilium,lxc")
    os.environ.setdefault("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", bootstrap_ifname)
    os.environ.setdefault("NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY", os.environ.get("NCCL_SOCKET_FAMILY", "AF_INET"))
    os.environ.setdefault("NVSHMEM_DISABLE_P2P", "1")
    os.environ.setdefault("NVSHMEM_IB_ENABLE_IBGDA", "1")
    hca_prefix = os.environ.get("LEVANTER_DEEPEP_NVSHMEM_HCA_PREFIX")
    hca_prefix = hca_prefix or os.environ.get("DEEPEP_NVSHMEM_HCA_PREFIX", "ibp")
    os.environ.setdefault("NVSHMEM_HCA_PREFIX", hca_prefix)
    os.environ.setdefault("NVSHMEM_IBGDA_NUM_RC_PER_PE", str(num_qps_per_rank))
    os.environ.setdefault("NVSHMEM_QP_DEPTH", "1024")
    os.environ.setdefault("NVSHMEM_MAX_TEAMS", "7")
    os.environ.setdefault("NVSHMEM_DISABLE_NVLS", "1")
    os.environ.setdefault("NVSHMEM_CUMEM_GRANULARITY", str(2**29))
    os.environ.setdefault("NVSHMEM_DISABLE_MNNVL", "1")


def shutdown_internode_runtime() -> None:
    if getattr(ensure_internode_runtime, "_signature", None) is None:
        return
    shutdown = _library_function(_SHUTDOWN_INTERNODE_SYMBOL, TransportBuildMode.INTERNODE)
    shutdown.argtypes = []
    shutdown.restype = None
    shutdown()
    setattr(ensure_internode_runtime, "_signature", None)


def internode_runtime_status() -> InternodeRuntimeStatus:
    """Return the current host-side DeepEP internode runtime status."""
    status_fn = _library_function(_INTERNODE_STATUS_SYMBOL, TransportBuildMode.INTERNODE)
    status_fn.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    status_fn.restype = ctypes.c_int
    initialized = ctypes.c_int()
    process_rank = ctypes.c_int()
    process_count = ctypes.c_int()
    num_local_ranks = ctypes.c_int()
    num_global_ranks = ctypes.c_int()
    num_nvl_bytes = ctypes.c_int64()
    num_rdma_bytes = ctypes.c_int64()
    status = status_fn(
        ctypes.byref(initialized),
        ctypes.byref(process_rank),
        ctypes.byref(process_count),
        ctypes.byref(num_local_ranks),
        ctypes.byref(num_global_ranks),
        ctypes.byref(num_nvl_bytes),
        ctypes.byref(num_rdma_bytes),
    )
    if status != 0:
        raise RuntimeError(
            f"Failed to query DeepEP internode runtime status: {_last_error(TransportBuildMode.INTERNODE)}"
        )
    return InternodeRuntimeStatus(
        initialized=bool(initialized.value),
        process_rank=int(process_rank.value),
        process_count=int(process_count.value),
        num_local_ranks=int(num_local_ranks.value),
        num_global_ranks=int(num_global_ranks.value),
        num_nvl_bytes=int(num_nvl_bytes.value),
        num_rdma_bytes=int(num_rdma_bytes.value),
    )


def run_internode_mapped_counter_smoke() -> dict[str, int | str]:
    """Verify that internode runtime device kernels can update host-mapped recv counters."""
    run = _library_function(_RUN_INTERNODE_MAPPED_COUNTER_SMOKE_SYMBOL, TransportBuildMode.INTERNODE)
    run.argtypes = [ctypes.POINTER(ctypes.c_int)]
    run.restype = ctypes.c_int
    num_checked = ctypes.c_int()
    status = run(ctypes.byref(num_checked))
    result: dict[str, int | str] = {
        "internode_mapped_counter_smoke_status_code": int(status),
        "num_checked": int(num_checked.value),
    }
    text = _last_error(TransportBuildMode.INTERNODE)
    result["last_error"] = text
    if status != 0:
        raise RuntimeError(f"Failed to run DeepEP internode mapped-counter smoke: {text}")
    return result


def ensure_internode_runtime(
    *,
    num_nvl_bytes: int,
    num_rdma_bytes: int,
    num_qps_per_rank: int = _DEFAULT_INTERNODE_QPS_PER_RANK,
    configure_nvshmem_env: bool = False,
    ranks_per_node: int | None = None,
    metadata: tuple[InternodeProcessBootstrapMetadata, ...] | None = None,
) -> None:
    """Initialize the DeepEP internode runtime after host metadata exchange."""
    if configure_nvshmem_env:
        _configure_internode_nvshmem_env(num_qps_per_rank=num_qps_per_rank)
    topology: InternodeProcessTopology | None = None
    if ranks_per_node is not None or os.environ.get(_INTERNODE_RANKS_PER_NODE_ENV):
        topology = preflight_internode_process_topology(ranks_per_node=ranks_per_node)
    if metadata is None:
        metadata = exchange_internode_bootstrap_metadata(
            local_internode_bootstrap_metadata(num_nvl_bytes=num_nvl_bytes, ranks_per_node=ranks_per_node)
        )
    process_rank = jax.process_index()
    process_count = jax.process_count()
    if len(metadata) != process_count:
        raise RuntimeError(
            f"DeepEP internode bootstrap metadata has {len(metadata)} entries for {process_count} JAX processes"
        )
    process_model = _internode_metadata_process_model(metadata)
    if topology is None and process_model == "process_per_gpu":
        topology = current_internode_process_topology(
            ranks_per_node=metadata[process_rank].ranks_per_node,
            visible_local_gpus=len(metadata[process_rank].local_device_ids),
        )
    num_local_ranks = _internode_local_rank_count(metadata)
    signature = (process_rank, process_count, num_local_ranks, topology, num_nvl_bytes, num_rdma_bytes, metadata)
    if getattr(ensure_internode_runtime, "_signature", None) == signature:
        return
    root_id = _root_nvshmem_unique_id(metadata, topology=topology)
    root_buffer = (ctypes.c_uint8 * len(root_id)).from_buffer_copy(root_id)
    if process_model == "process_per_gpu":
        if topology is None or topology.local_rank is None:
            raise RuntimeError("DeepEP process-per-GPU init requires explicit topology")
        ipc_blob = _internode_ipc_handle_blob(metadata)
        ipc_buffer = (ctypes.c_uint8 * len(ipc_blob)).from_buffer_copy(ipc_blob)
        init_process = _library_function(_INIT_INTERNODE_PROCESS_RUNTIME_SYMBOL, TransportBuildMode.INTERNODE)
        init_process.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_int64,
        ]
        init_process.restype = ctypes.c_int
        status = init_process(
            topology.process_index,
            topology.process_count,
            topology.node_rank,
            topology.node_count,
            topology.local_rank,
            topology.ranks_per_node,
            num_nvl_bytes,
            ipc_buffer,
            len(ipc_blob),
            root_buffer,
            len(root_id),
            num_rdma_bytes,
        )
        if status != 0:
            raise RuntimeError(
                f"Failed to initialize DeepEP process-per-GPU runtime: {_last_error(TransportBuildMode.INTERNODE)}"
            )
        ensure_internode_runtime._signature = signature
        return

    init = _library_function(_INIT_INTERNODE_SYMBOL, TransportBuildMode.INTERNODE)
    init.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_int64,
    ]
    init.restype = ctypes.c_int
    status = init(
        process_rank,
        process_count,
        num_local_ranks,
        num_nvl_bytes,
        root_buffer,
        len(root_id),
        num_rdma_bytes,
    )
    if status != 0:
        raise RuntimeError(
            f"Failed to initialize DeepEP internode runtime: {_last_error(TransportBuildMode.INTERNODE)}"
        )
    ensure_internode_runtime._signature = signature


def _materialize_cotangent(
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
    *,
    dtype: jnp.dtype,
    shape: tuple[int, ...] | None = None,
    reference: jax.Array | None = None,
) -> jax.Array:
    if reference is None and shape is None:
        raise ValueError("Either reference or shape must be provided when materializing a cotangent.")
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        if reference is not None:
            return jnp.zeros_like(reference, dtype=dtype)
        return jnp.zeros(shape, dtype=dtype)
    return jnp.asarray(cotangent, dtype=dtype)


def _default_dispatch_config(num_ranks: int) -> IntranodeConfig:
    if num_ranks not in _DEFAULT_DISPATCH_CONFIGS:
        raise ValueError(f"Unsupported DeepEP intranode dispatch rank count: {num_ranks}")
    return _config_with_env_overrides(
        _DEFAULT_DISPATCH_CONFIGS[num_ranks],
        num_sms_env=_DISPATCH_NUM_SMS_ENV,
        max_send_tokens_env=_DISPATCH_MAX_SEND_TOKENS_ENV,
        max_recv_tokens_env=_DISPATCH_MAX_RECV_TOKENS_ENV,
    )


def _default_combine_config(num_ranks: int) -> IntranodeConfig:
    if num_ranks not in _DEFAULT_COMBINE_CONFIGS:
        raise ValueError(f"Unsupported DeepEP intranode combine rank count: {num_ranks}")
    return _config_with_env_overrides(
        _DEFAULT_COMBINE_CONFIGS[num_ranks],
        num_sms_env=_COMBINE_NUM_SMS_ENV,
        max_send_tokens_env=_COMBINE_MAX_SEND_TOKENS_ENV,
        max_recv_tokens_env=_COMBINE_MAX_RECV_TOKENS_ENV,
    )


def _default_internode_dispatch_config() -> InternodeConfig:
    return _internode_config_with_env_overrides(_DEFAULT_INTERNODE_DISPATCH_CONFIG)


def _default_internode_source_meta_bytes() -> int:
    override = _env_int(_INTERNODE_SOURCE_META_BYTES_ENV)
    source_meta_bytes = override or _DEFAULT_INTERNODE_SOURCE_META_BYTES
    if source_meta_bytes <= 0:
        raise RuntimeError(f"{_INTERNODE_SOURCE_META_BYTES_ENV} must be positive, got {source_meta_bytes}")
    return source_meta_bytes


def _visible_local_gpu_count() -> int:
    return sum(1 for device in jax.local_devices() if device.platform == "gpu")


def _default_num_local_ranks() -> int:
    local_ranks = _visible_local_gpu_count()
    if local_ranks <= 0:
        raise RuntimeError("DeepEP internode dispatch requires at least one local GPU rank")
    return local_ranks


def current_internode_process_topology(
    *,
    ranks_per_node: int | None = None,
    process_index: int | None = None,
    process_count: int | None = None,
    visible_local_gpus: int | None = None,
) -> InternodeProcessTopology:
    """Return the current JAX process topology for DeepEP normal-mode transport.

    DeepEP normal mode distinguishes RDMA nodes from NVLink-local GPU ranks.
    Marin has two relevant process models:

    * ``process_per_node``: one JAX process owns all local GPUs on a node. This
      is what the current CUDA FFI runtime can emulate.
    * ``process_per_gpu``: Iris/JAX launches one process per GPU. This is the
      target CoreWeave layout for EP16, but it requires CUDA IPC/local peer
      handle exchange that the current runtime does not implement yet.
    """
    process_index = jax.process_index() if process_index is None else process_index
    process_count = jax.process_count() if process_count is None else process_count
    visible_local_gpus = _visible_local_gpu_count() if visible_local_gpus is None else visible_local_gpus
    ranks_per_node = _env_int(_INTERNODE_RANKS_PER_NODE_ENV) if ranks_per_node is None else ranks_per_node
    if ranks_per_node is None:
        ranks_per_node = visible_local_gpus

    if process_index < 0 or process_index >= process_count:
        raise RuntimeError(f"JAX process index {process_index} is out of range for process_count={process_count}")
    if process_count <= 0:
        raise RuntimeError(f"JAX process_count must be positive, got {process_count}")
    if ranks_per_node <= 0:
        raise RuntimeError(f"{_INTERNODE_RANKS_PER_NODE_ENV} must be positive, got {ranks_per_node}")
    if visible_local_gpus <= 0:
        raise RuntimeError("DeepEP internode topology requires at least one visible local GPU")

    if process_count % ranks_per_node == 0:
        node_count = process_count // ranks_per_node
        return InternodeProcessTopology(
            process_index=process_index,
            process_count=process_count,
            process_model="process_per_gpu",
            node_rank=process_index // ranks_per_node,
            node_count=node_count,
            ranks_per_node=ranks_per_node,
            visible_local_gpus=visible_local_gpus,
            local_rank=process_index % ranks_per_node,
        )

    if visible_local_gpus == ranks_per_node:
        return InternodeProcessTopology(
            process_index=process_index,
            process_count=process_count,
            process_model="process_per_node",
            node_rank=process_index,
            node_count=process_count,
            ranks_per_node=ranks_per_node,
            visible_local_gpus=visible_local_gpus,
            local_rank=None,
        )

    raise RuntimeError(
        "Cannot classify DeepEP internode process topology: "
        f"process_count={process_count}, visible_local_gpus={visible_local_gpus}, "
        f"{_INTERNODE_RANKS_PER_NODE_ENV}={ranks_per_node}. For Iris one-process-per-GPU EP16 on two "
        "H100x8 nodes set DEEPEP_RANKS_PER_NODE=8; the current runtime will then fail early until CUDA IPC "
        "local-peer exchange is implemented."
    )


def preflight_internode_process_topology(
    *,
    ranks_per_node: int | None = None,
    require_cross_node: bool = True,
) -> InternodeProcessTopology:
    """Validate that the current process model is supported by the DeepEP FFI runtime."""
    topology = current_internode_process_topology(ranks_per_node=ranks_per_node)
    if require_cross_node and topology.node_count <= 1:
        raise RuntimeError(
            "DeepEP internode transport requires at least two nodes; "
            f"resolved topology has node_count={topology.node_count}"
        )
    return topology


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def _config_with_env_overrides(
    config: IntranodeConfig,
    *,
    num_sms_env: str,
    max_send_tokens_env: str,
    max_recv_tokens_env: str,
) -> IntranodeConfig:
    num_sms = _env_int(num_sms_env) or config.num_sms
    max_send_tokens = _env_int(max_send_tokens_env) or config.num_max_send_tokens
    max_recv_tokens = _env_int(max_recv_tokens_env) or config.num_max_recv_tokens
    if num_sms <= 0 or num_sms % 2 != 0:
        raise RuntimeError(f"{num_sms_env} must be a positive even integer, got {num_sms}")
    if max_send_tokens <= 0:
        raise RuntimeError(f"{max_send_tokens_env} must be positive, got {max_send_tokens}")
    if max_recv_tokens <= 0:
        raise RuntimeError(f"{max_recv_tokens_env} must be positive, got {max_recv_tokens}")
    return IntranodeConfig(
        num_sms=num_sms,
        num_max_send_tokens=max_send_tokens,
        num_max_recv_tokens=max_recv_tokens,
    )


def _internode_config_with_env_overrides(config: InternodeConfig) -> InternodeConfig:
    num_sms = _env_int(_INTERNODE_DISPATCH_NUM_SMS_ENV) or config.num_sms
    nvl_send_tokens = _env_int(_INTERNODE_DISPATCH_MAX_NVL_SEND_TOKENS_ENV) or config.num_max_nvl_chunked_send_tokens
    nvl_recv_tokens = _env_int(_INTERNODE_DISPATCH_MAX_NVL_RECV_TOKENS_ENV) or config.num_max_nvl_chunked_recv_tokens
    rdma_send_tokens = (
        _env_int(_INTERNODE_DISPATCH_MAX_RDMA_SEND_TOKENS_ENV) or config.num_max_rdma_chunked_send_tokens
    )
    rdma_recv_tokens = (
        _env_int(_INTERNODE_DISPATCH_MAX_RDMA_RECV_TOKENS_ENV) or config.num_max_rdma_chunked_recv_tokens
    )
    if num_sms <= 0 or num_sms % 2 != 0:
        raise RuntimeError(f"{_INTERNODE_DISPATCH_NUM_SMS_ENV} must be a positive even integer, got {num_sms}")
    if nvl_send_tokens <= 0:
        raise RuntimeError(f"{_INTERNODE_DISPATCH_MAX_NVL_SEND_TOKENS_ENV} must be positive, got {nvl_send_tokens}")
    if nvl_recv_tokens <= 0:
        raise RuntimeError(f"{_INTERNODE_DISPATCH_MAX_NVL_RECV_TOKENS_ENV} must be positive, got {nvl_recv_tokens}")
    if rdma_send_tokens <= 0:
        raise RuntimeError(f"{_INTERNODE_DISPATCH_MAX_RDMA_SEND_TOKENS_ENV} must be positive, got {rdma_send_tokens}")
    if rdma_recv_tokens <= 0:
        raise RuntimeError(f"{_INTERNODE_DISPATCH_MAX_RDMA_RECV_TOKENS_ENV} must be positive, got {rdma_recv_tokens}")
    return InternodeConfig(
        num_sms=num_sms,
        num_max_nvl_chunked_send_tokens=nvl_send_tokens,
        num_max_nvl_chunked_recv_tokens=nvl_recv_tokens,
        num_max_rdma_chunked_send_tokens=rdma_send_tokens,
        num_max_rdma_chunked_recv_tokens=rdma_recv_tokens,
    )


def shutdown_intranode_runtime() -> None:
    if getattr(ensure_intranode_runtime, "_signature", None) is None:
        return
    shutdown = _library_function(_SHUTDOWN_SYMBOL)
    shutdown.argtypes = []
    shutdown.restype = None
    shutdown()
    setattr(ensure_intranode_runtime, "_signature", None)


atexit.register(shutdown_intranode_runtime)
atexit.register(shutdown_internode_runtime)


def ensure_intranode_runtime(
    *,
    num_ranks: int,
    hidden_bytes: int,
    dispatch_config: IntranodeConfig | None = None,
    combine_config: IntranodeConfig | None = None,
) -> None:
    _register_targets()
    dispatch = dispatch_config or _default_dispatch_config(num_ranks)
    combine = combine_config or _default_combine_config(num_ranks)
    signature = (num_ranks, hidden_bytes, dispatch, combine)
    if getattr(ensure_intranode_runtime, "_signature", None) == signature:
        return

    local_gpu_devices = [device for device in jax.local_devices() if device.platform == "gpu"]
    if len(local_gpu_devices) != num_ranks:
        raise RuntimeError(
            "DeepEP JAX training dispatch/combine is currently intranode-only and expects the expert group "
            "to span all visible local GPUs. Upstream DeepEP has an internode path, but this Marin backend "
            "does not yet wire it into the jitted MoE transport. "
            f"got num_ranks={num_ranks} and visible_gpus={len(local_gpu_devices)}."
        )

    init = _library_function(_INIT_SYMBOL)
    init.argtypes = [
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    init.restype = ctypes.c_int

    status = init(
        num_ranks,
        hidden_bytes,
        dispatch.num_sms,
        dispatch.num_max_send_tokens,
        dispatch.num_max_recv_tokens,
        combine.num_sms,
        combine.num_max_send_tokens,
        combine.num_max_recv_tokens,
    )
    if status != 0:
        last_error = _library_function(_LAST_ERROR_SYMBOL)
        last_error.argtypes = []
        last_error.restype = ctypes.c_char_p
        message = last_error()
        text = message.decode("utf-8") if message else "unknown error"
        raise RuntimeError(f"Failed to initialize DeepEP intranode JAX runtime: {text}")
    ensure_intranode_runtime._signature = signature


def probe_dispatch_kernel_attributes() -> dict[str, int | str]:
    """Call the DeepEP dispatch host wrapper outside XLA execution."""
    probe = _library_function(_PROBE_DISPATCH_SYMBOL)
    probe.argtypes = []
    probe.restype = ctypes.c_int

    status = probe()
    result: dict[str, int | str] = {"probe_status_code": int(status)}
    last_error = _library_function(_LAST_ERROR_SYMBOL)
    last_error.argtypes = []
    last_error.restype = ctypes.c_char_p
    message = last_error()
    text = message.decode("utf-8") if message else ""
    result["last_error"] = text
    if status != 0:
        raise RuntimeError(f"Failed to probe DeepEP dispatch kernel attributes: {text}")
    return result


def run_host_dispatch_round(
    *,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
) -> dict[str, int | str]:
    """Run a same-process all-ranks dispatch round outside XLA execution."""
    run = _library_function(_RUN_HOST_DISPATCH_SYMBOL)
    run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    run.restype = ctypes.c_int

    status = run(num_tokens, hidden, num_experts, num_topk)
    result: dict[str, int | str] = {
        "host_dispatch_status_code": int(status),
        "num_tokens": int(num_tokens),
        "hidden": int(hidden),
        "num_experts": int(num_experts),
        "num_topk": int(num_topk),
    }
    last_error = _library_function(_LAST_ERROR_SYMBOL)
    last_error.argtypes = []
    last_error.restype = ctypes.c_char_p
    message = last_error()
    text = message.decode("utf-8") if message else ""
    result["last_error"] = text
    if status != 0:
        raise RuntimeError(f"Failed to run DeepEP host dispatch round: {text}")
    return result


def run_host_internode_dispatch_round(
    *,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
) -> dict[str, int | str]:
    """Run a same-process local-rank dispatch round against the initialized internode runtime."""
    run = _library_function(_RUN_HOST_INTERNODE_DISPATCH_SYMBOL, TransportBuildMode.INTERNODE)
    run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    run.restype = ctypes.c_int

    status = run(num_tokens, hidden, num_experts, num_topk)
    result: dict[str, int | str] = {
        "host_internode_dispatch_status_code": int(status),
        "num_tokens": int(num_tokens),
        "hidden": int(hidden),
        "num_experts": int(num_experts),
        "num_topk": int(num_topk),
    }
    last_error = _library_function(_LAST_ERROR_SYMBOL, TransportBuildMode.INTERNODE)
    last_error.argtypes = []
    last_error.restype = ctypes.c_char_p
    message = last_error()
    text = message.decode("utf-8") if message else ""
    result["last_error"] = text
    if status != 0:
        raise RuntimeError(f"Failed to run DeepEP host internode dispatch round: {text}")
    return result


def _dispatch_internode_impl(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_rdma_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    *,
    num_experts: int,
    max_recv_tokens: int | None = None,
    max_rdma_recv_tokens: int | None = None,
    source_meta_bytes: int | None = None,
    num_local_ranks: int | None = None,
    assignment_capacity: int | None = None,
    dispatch_config: InternodeConfig | None = None,
) -> DeepEPInternodeDispatch:
    """Dispatch tokens with DeepEP normal-mode internode transport."""
    _register_internode_targets()
    config = dispatch_config or _default_internode_dispatch_config()
    if config.num_sms <= 0 or config.num_sms % 2 != 0:
        raise ValueError(f"internode dispatch num_sms must be a positive even integer, got {config.num_sms}")

    x_bf16 = jnp.asarray(x, dtype=jnp.bfloat16)
    topk_idx_i32 = jnp.asarray(topk_idx, dtype=jnp.int32)
    topk_weights_f32 = jnp.asarray(topk_weights, dtype=jnp.float32)
    num_tokens_per_rank_i32 = jnp.asarray(num_tokens_per_rank, dtype=jnp.int32)
    num_tokens_per_rdma_rank_i32 = jnp.asarray(num_tokens_per_rdma_rank, dtype=jnp.int32)
    num_tokens_per_expert_i32 = jnp.asarray(num_tokens_per_expert, dtype=jnp.int32)
    is_token_in_rank_bool = jnp.asarray(is_token_in_rank, dtype=jnp.bool_)

    num_ranks = int(num_tokens_per_rank_i32.shape[0])
    num_rdma_ranks = int(num_tokens_per_rdma_rank_i32.shape[0])
    if num_ranks <= 0 or num_rdma_ranks <= 1:
        raise ValueError(
            f"internode dispatch requires positive global ranks and >1 RDMA rank, got {num_ranks=} {num_rdma_ranks=}"
        )
    if num_experts % num_ranks != 0:
        raise ValueError(
            f"internode dispatch requires num_experts divisible by num_ranks: {num_experts=} {num_ranks=}"
        )
    if max_recv_tokens is None:
        max_recv_tokens = x_bf16.shape[0] * topk_idx_i32.shape[1] * num_rdma_ranks
    if max_rdma_recv_tokens is None:
        max_rdma_recv_tokens = max_recv_tokens
    if max_recv_tokens <= 0 or max_rdma_recv_tokens <= 0:
        raise ValueError(
            f"internode dispatch capacities must be positive, got {max_recv_tokens=} {max_rdma_recv_tokens=}"
        )
    if source_meta_bytes is None:
        source_meta_bytes = _default_internode_source_meta_bytes()
    if source_meta_bytes <= 0:
        raise ValueError(f"source_meta_bytes must be positive, got {source_meta_bytes}")
    if num_local_ranks is None:
        num_local_ranks = _default_num_local_ranks()
    if num_local_ranks <= 0:
        raise ValueError(f"num_local_ranks must be positive, got {num_local_ranks}")

    num_channels = config.num_sms // 2
    num_tokens = x_bf16.shape[0]
    hidden = x_bf16.shape[1]
    topk = topk_idx_i32.shape[1]
    local_experts = num_experts // num_ranks
    max_assignments = max_recv_tokens * topk
    output_assignments = max_assignments if assignment_capacity is None else assignment_capacity
    if output_assignments <= 0:
        raise ValueError(f"assignment_capacity must be positive, got {assignment_capacity}")
    if output_assignments > max_assignments:
        raise ValueError(
            f"assignment_capacity={output_assignments} cannot exceed max_recv_tokens * topk={max_assignments}"
        )
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((max_recv_tokens, hidden), x_bf16.dtype),
        jax.ShapeDtypeStruct((max_recv_tokens, topk), jnp.int32),
        jax.ShapeDtypeStruct((max_recv_tokens, topk), jnp.float32),
        jax.ShapeDtypeStruct((num_tokens, num_ranks), jnp.bool_),
        jax.ShapeDtypeStruct((max_recv_tokens, source_meta_bytes), jnp.uint8),
        jax.ShapeDtypeStruct((num_rdma_ranks, num_channels), jnp.int32),
        jax.ShapeDtypeStruct((num_rdma_ranks, num_channels), jnp.int32),
        jax.ShapeDtypeStruct((num_rdma_ranks,), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks,), jnp.int32),
        jax.ShapeDtypeStruct((num_tokens, num_rdma_ranks), jnp.int32),
        jax.ShapeDtypeStruct((max_rdma_recv_tokens, num_local_ranks), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((output_assignments, hidden), x_bf16.dtype),
        jax.ShapeDtypeStruct((output_assignments,), x_bf16.dtype),
        jax.ShapeDtypeStruct((output_assignments,), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((output_assignments,), jnp.int32),
        jax.ShapeDtypeStruct((max_assignments,), jnp.int32),
        jax.ShapeDtypeStruct((num_tokens, topk * 2), jnp.int32),
        jax.ShapeDtypeStruct((max_recv_tokens, topk * 2), jnp.int32),
    )
    results = jax.ffi.ffi_call(
        _DISPATCH_INTERNODE_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        x_bf16,
        topk_idx_i32,
        topk_weights_f32,
        num_tokens_per_rank_i32,
        num_tokens_per_rdma_rank_i32,
        num_tokens_per_expert_i32,
        is_token_in_rank_bool,
        num_experts=np.int32(num_experts),
        num_sms=np.int32(config.num_sms),
        num_max_nvl_chunked_send_tokens=np.int32(config.num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens=np.int32(config.num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens=np.int32(config.num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens=np.int32(config.num_max_rdma_chunked_recv_tokens),
        assignment_capacity=np.int32(output_assignments),
        low_latency_mode=_internode_low_latency_mode(),
    )
    return DeepEPInternodeDispatch(
        recv_x=results[0],
        recv_topk_idx=results[1],
        recv_topk_weights=results[2],
        is_token_in_rank=results[3],
        recv_src_meta=results[4],
        rdma_channel_prefix_matrix=results[5],
        recv_rdma_channel_prefix_matrix=results[6],
        recv_rdma_rank_prefix_sum=results[7],
        gbl_channel_prefix_matrix=results[8],
        recv_gbl_channel_prefix_matrix=results[9],
        recv_gbl_rank_prefix_sum=results[10],
        send_rdma_head=results[11],
        send_nvl_head=results[12],
        local_expert_counts=results[13],
        num_recv_tokens=results[14],
        num_recv_rdma_tokens=results[15],
        local_group_sizes=results[16],
        x_dispatch=results[17],
        assignment_weights=results[18],
        recv_token_indices=results[19],
        assignment_destinations=results[22],
    )


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13))
def _dispatch_internode_with_vjp(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_rdma_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    num_experts: int,
    max_recv_tokens: int | None,
    max_rdma_recv_tokens: int | None,
    source_meta_bytes: int | None,
    num_local_ranks: int | None,
    assignment_capacity: int | None,
    dispatch_config: InternodeConfig | None,
) -> tuple[jax.Array, ...]:
    return tuple(
        _dispatch_internode_impl(
            x,
            topk_idx,
            topk_weights,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            num_experts=num_experts,
            max_recv_tokens=max_recv_tokens,
            max_rdma_recv_tokens=max_rdma_recv_tokens,
            source_meta_bytes=source_meta_bytes,
            num_local_ranks=num_local_ranks,
            assignment_capacity=assignment_capacity,
            dispatch_config=dispatch_config,
        )
    )


def _dispatch_internode_with_vjp_fwd(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_rdma_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    num_experts: int,
    max_recv_tokens: int | None,
    max_rdma_recv_tokens: int | None,
    source_meta_bytes: int | None,
    num_local_ranks: int | None,
    assignment_capacity: int | None,
    dispatch_config: InternodeConfig | None,
):
    outputs = tuple(
        _dispatch_internode_impl(
            x,
            topk_idx,
            topk_weights,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            num_experts=num_experts,
            max_recv_tokens=max_recv_tokens,
            max_rdma_recv_tokens=max_rdma_recv_tokens,
            source_meta_bytes=source_meta_bytes,
            num_local_ranks=num_local_ranks,
            assignment_capacity=assignment_capacity,
            dispatch_config=dispatch_config,
        )
    )
    (
        recv_x,
        _recv_topk_idx,
        recv_topk_weights,
        is_token_in_rank_out,
        recv_src_meta,
        _rdma_channel_prefix_matrix,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        _gbl_channel_prefix_matrix,
        recv_gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        send_rdma_head,
        send_nvl_head,
        _local_expert_counts,
        num_recv_tokens,
        num_recv_rdma_tokens,
        _local_group_sizes,
        x_dispatch,
        assignment_weights,
        _recv_token_indices,
        assignment_destinations,
    ) = outputs
    residuals = (
        recv_x,
        recv_topk_weights,
        is_token_in_rank_out,
        recv_src_meta,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        recv_gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        x_dispatch.shape,
        assignment_weights.shape,
        assignment_destinations,
    )
    return outputs, residuals


def _dispatch_internode_with_vjp_bwd(
    num_experts: int,
    max_recv_tokens: int | None,
    max_rdma_recv_tokens: int | None,
    source_meta_bytes: int | None,
    num_local_ranks: int | None,
    assignment_capacity: int | None,
    dispatch_config: InternodeConfig | None,
    residuals,
    cotangents,
):
    del num_experts, max_recv_tokens, max_rdma_recv_tokens, source_meta_bytes, num_local_ranks, assignment_capacity
    (
        recv_x,
        recv_topk_weights,
        is_token_in_rank,
        recv_src_meta,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        recv_gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        x_dispatch_shape,
        assignment_weights_shape,
        assignment_destinations,
    ) = residuals
    grad_recv_x = _materialize_cotangent(cotangents[0], dtype=recv_x.dtype, reference=recv_x)
    grad_recv_topk_weights = _materialize_cotangent(
        cotangents[2],
        dtype=recv_topk_weights.dtype,
        reference=recv_topk_weights,
    )
    if not (
        isinstance(cotangents[17], jax.custom_derivatives.SymbolicZero)
        and isinstance(cotangents[18], jax.custom_derivatives.SymbolicZero)
    ):
        grad_x_dispatch = _materialize_cotangent(cotangents[17], dtype=recv_x.dtype, shape=x_dispatch_shape)
        grad_assignment_weights = _materialize_cotangent(
            cotangents[18],
            dtype=jnp.float32,
            shape=assignment_weights_shape,
        )
        assignment_gradient_mode = _internode_assignment_gradient_mode()
        if assignment_gradient_mode == _INTERNODE_ASSIGNMENT_GRADIENT_FUSED:
            grad_x, grad_topk_weights = _dispatch_internode_bwd_fused_impl(
                grad_recv_x_base=grad_recv_x,
                grad_recv_topk_weights_base=grad_recv_topk_weights,
                grad_x_dispatch=grad_x_dispatch,
                grad_assignment_weights=grad_assignment_weights,
                assignment_destinations=assignment_destinations,
                is_token_in_rank=is_token_in_rank,
                recv_src_meta=recv_src_meta,
                recv_rdma_channel_prefix_matrix=recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum=recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix=recv_gbl_channel_prefix_matrix,
                send_rdma_head=send_rdma_head,
                send_nvl_head=send_nvl_head,
                num_recv_tokens=num_recv_tokens,
                num_recv_rdma_tokens=num_recv_rdma_tokens,
                dispatch_config=dispatch_config,
            )
            return grad_x, None, grad_topk_weights, None, None, None, None
        if assignment_gradient_mode == _INTERNODE_ASSIGNMENT_GRADIENT_FFI:
            grad_recv_x_from_assignments, grad_recv_topk_weights_from_assignments = _assignment_gradients_impl(
                grad_x_dispatch=grad_x_dispatch,
                grad_assignment_weights=grad_assignment_weights,
                assignment_destinations=assignment_destinations,
                num_recv_tokens=num_recv_tokens,
                recv_x_shape=recv_x.shape,
                recv_topk_weights_shape=recv_topk_weights.shape,
            )
        else:
            grad_recv_x_from_assignments, grad_recv_topk_weights_from_assignments = _assignment_gradients_jax(
                grad_x_dispatch=grad_x_dispatch,
                grad_assignment_weights=grad_assignment_weights,
                assignment_destinations=assignment_destinations,
                num_recv_tokens=num_recv_tokens,
                recv_x_shape=recv_x.shape,
                recv_topk_weights_shape=recv_topk_weights.shape,
            )
        grad_recv_x += grad_recv_x_from_assignments
        grad_recv_topk_weights += grad_recv_topk_weights_from_assignments
    grad_x, grad_topk_weights = _combine_internode_impl(
        grad_recv_x,
        grad_recv_topk_weights,
        is_token_in_rank,
        recv_src_meta,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        recv_gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        dispatch_config=dispatch_config,
    )
    return grad_x, None, grad_topk_weights, None, None, None, None


_dispatch_internode_with_vjp.defvjp(
    _dispatch_internode_with_vjp_fwd,
    _dispatch_internode_with_vjp_bwd,
)


def deepep_dispatch_internode(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_rdma_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    *,
    num_experts: int,
    max_recv_tokens: int | None = None,
    max_rdma_recv_tokens: int | None = None,
    source_meta_bytes: int | None = None,
    num_local_ranks: int | None = None,
    assignment_capacity: int | None = None,
    dispatch_config: InternodeConfig | None = None,
) -> DeepEPInternodeDispatch:
    """Dispatch tokens with DeepEP normal-mode internode transport."""
    return DeepEPInternodeDispatch(
        *_dispatch_internode_with_vjp(
            x,
            topk_idx,
            topk_weights,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            num_experts,
            max_recv_tokens,
            max_rdma_recv_tokens,
            source_meta_bytes,
            num_local_ranks,
            assignment_capacity,
            dispatch_config,
        )
    )


def _dispatch_internode_cached_impl(
    x: jax.Array,
    is_token_in_rank: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    *,
    max_recv_tokens: int,
    num_topk: int,
    dispatch_config: InternodeConfig | None = None,
) -> jax.Array:
    """Cached internode dispatch used as the reverse of internode combine."""
    _register_internode_targets()
    config = dispatch_config or _default_internode_dispatch_config()
    if max_recv_tokens <= 0:
        raise ValueError(f"max_recv_tokens must be positive, got {max_recv_tokens}")
    if config.num_sms <= 0 or config.num_sms % 2 != 0:
        raise ValueError(f"internode cached dispatch num_sms must be a positive even integer, got {config.num_sms}")
    if num_topk < 0:
        raise ValueError(f"num_topk must be nonnegative, got {num_topk}")

    x_bf16 = jnp.asarray(x, dtype=jnp.bfloat16)
    is_token_in_rank_bool = jnp.asarray(is_token_in_rank, dtype=jnp.bool_)
    rdma_channel_prefix_matrix_i32 = jnp.asarray(rdma_channel_prefix_matrix, dtype=jnp.int32)
    recv_rdma_rank_prefix_sum_i32 = jnp.asarray(recv_rdma_rank_prefix_sum, dtype=jnp.int32)
    gbl_channel_prefix_matrix_i32 = jnp.asarray(gbl_channel_prefix_matrix, dtype=jnp.int32)
    recv_gbl_rank_prefix_sum_i32 = jnp.asarray(recv_gbl_rank_prefix_sum, dtype=jnp.int32)
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    num_recv_rdma_tokens_i32 = jnp.asarray(num_recv_rdma_tokens, dtype=jnp.int32)
    if num_recv_tokens_i32.ndim == 0:
        num_recv_tokens_i32 = jnp.reshape(num_recv_tokens_i32, (1,))
    if num_recv_rdma_tokens_i32.ndim == 0:
        num_recv_rdma_tokens_i32 = jnp.reshape(num_recv_rdma_tokens_i32, (1,))

    result_shape_dtypes = (jax.ShapeDtypeStruct((max_recv_tokens, x_bf16.shape[1]), x_bf16.dtype),)
    (recv_x,) = jax.ffi.ffi_call(
        _DISPATCH_INTERNODE_CACHED_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        x_bf16,
        is_token_in_rank_bool,
        rdma_channel_prefix_matrix_i32,
        recv_rdma_rank_prefix_sum_i32,
        gbl_channel_prefix_matrix_i32,
        recv_gbl_rank_prefix_sum_i32,
        num_recv_tokens_i32,
        num_recv_rdma_tokens_i32,
        num_topk=np.int32(num_topk),
        num_sms=np.int32(config.num_sms),
        num_max_nvl_chunked_send_tokens=np.int32(config.num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens=np.int32(config.num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens=np.int32(config.num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens=np.int32(config.num_max_rdma_chunked_recv_tokens),
        low_latency_mode=_internode_low_latency_mode(),
    )
    recv_token_limit = jnp.squeeze(num_recv_tokens_i32, axis=0)
    recv_valid = jnp.arange(max_recv_tokens, dtype=jnp.int32) < recv_token_limit
    return jnp.where(recv_valid[:, None], recv_x, 0)


def _combine_internode_impl(
    recv_x: jax.Array,
    recv_topk_weights: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    *,
    dispatch_config: InternodeConfig | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Combine tokens with DeepEP normal-mode internode transport."""
    _register_internode_targets()
    config = dispatch_config or _default_internode_dispatch_config()
    if config.num_sms <= 0 or config.num_sms % 2 != 0:
        raise ValueError(f"internode combine num_sms must be a positive even integer, got {config.num_sms}")
    recv_x_bf16 = jnp.asarray(recv_x, dtype=jnp.bfloat16)
    recv_topk_weights_f32 = jnp.asarray(recv_topk_weights, dtype=jnp.float32)
    is_token_in_rank_bool = jnp.asarray(is_token_in_rank, dtype=jnp.bool_)
    recv_src_meta_u8 = jnp.asarray(recv_src_meta, dtype=jnp.uint8)
    recv_rdma_channel_prefix_matrix_i32 = jnp.asarray(recv_rdma_channel_prefix_matrix, dtype=jnp.int32)
    recv_rdma_rank_prefix_sum_i32 = jnp.asarray(recv_rdma_rank_prefix_sum, dtype=jnp.int32)
    recv_gbl_channel_prefix_matrix_i32 = jnp.asarray(recv_gbl_channel_prefix_matrix, dtype=jnp.int32)
    send_rdma_head_i32 = jnp.asarray(send_rdma_head, dtype=jnp.int32)
    send_nvl_head_i32 = jnp.asarray(send_nvl_head, dtype=jnp.int32)
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    num_recv_rdma_tokens_i32 = jnp.asarray(num_recv_rdma_tokens, dtype=jnp.int32)

    if num_recv_tokens_i32.ndim == 0:
        num_recv_tokens_i32 = jnp.reshape(num_recv_tokens_i32, (1,))
    if num_recv_rdma_tokens_i32.ndim == 0:
        num_recv_rdma_tokens_i32 = jnp.reshape(num_recv_rdma_tokens_i32, (1,))

    combined_tokens = send_rdma_head_i32.shape[0]
    hidden = recv_x_bf16.shape[1]
    topk = recv_topk_weights_f32.shape[1]
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((combined_tokens, hidden), recv_x_bf16.dtype),
        jax.ShapeDtypeStruct((combined_tokens, topk), jnp.float32),
    )
    combined, combined_weights = jax.ffi.ffi_call(
        _COMBINE_INTERNODE_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        recv_x_bf16,
        recv_topk_weights_f32,
        is_token_in_rank_bool,
        recv_src_meta_u8,
        recv_rdma_channel_prefix_matrix_i32,
        recv_rdma_rank_prefix_sum_i32,
        recv_gbl_channel_prefix_matrix_i32,
        send_rdma_head_i32,
        send_nvl_head_i32,
        num_recv_tokens_i32,
        num_recv_rdma_tokens_i32,
        num_sms=np.int32(config.num_sms),
        num_max_nvl_chunked_send_tokens=np.int32(config.num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens=np.int32(config.num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens=np.int32(config.num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens=np.int32(config.num_max_rdma_chunked_recv_tokens),
        low_latency_mode=_internode_low_latency_mode(),
    )
    return combined, combined_weights


def _combine_internode_x_only_impl(
    recv_x: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    *,
    dispatch_config: InternodeConfig | None = None,
) -> jax.Array:
    """Combine token payloads without materializing unused combined top-k weights."""
    _register_internode_targets()
    config = dispatch_config or _default_internode_dispatch_config()
    if config.num_sms <= 0 or config.num_sms % 2 != 0:
        raise ValueError(f"internode x-only combine num_sms must be a positive even integer, got {config.num_sms}")
    recv_x_bf16 = jnp.asarray(recv_x, dtype=jnp.bfloat16)
    is_token_in_rank_bool = jnp.asarray(is_token_in_rank, dtype=jnp.bool_)
    recv_src_meta_u8 = jnp.asarray(recv_src_meta, dtype=jnp.uint8)
    recv_rdma_channel_prefix_matrix_i32 = jnp.asarray(recv_rdma_channel_prefix_matrix, dtype=jnp.int32)
    recv_rdma_rank_prefix_sum_i32 = jnp.asarray(recv_rdma_rank_prefix_sum, dtype=jnp.int32)
    recv_gbl_channel_prefix_matrix_i32 = jnp.asarray(recv_gbl_channel_prefix_matrix, dtype=jnp.int32)
    send_rdma_head_i32 = jnp.asarray(send_rdma_head, dtype=jnp.int32)
    send_nvl_head_i32 = jnp.asarray(send_nvl_head, dtype=jnp.int32)
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    num_recv_rdma_tokens_i32 = jnp.asarray(num_recv_rdma_tokens, dtype=jnp.int32)

    if num_recv_tokens_i32.ndim == 0:
        num_recv_tokens_i32 = jnp.reshape(num_recv_tokens_i32, (1,))
    if num_recv_rdma_tokens_i32.ndim == 0:
        num_recv_rdma_tokens_i32 = jnp.reshape(num_recv_rdma_tokens_i32, (1,))

    combined_tokens = send_rdma_head_i32.shape[0]
    hidden = recv_x_bf16.shape[1]
    (combined,) = jax.ffi.ffi_call(
        _COMBINE_INTERNODE_X_ONLY_TARGET,
        (jax.ShapeDtypeStruct((combined_tokens, hidden), recv_x_bf16.dtype),),
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        recv_x_bf16,
        is_token_in_rank_bool,
        recv_src_meta_u8,
        recv_rdma_channel_prefix_matrix_i32,
        recv_rdma_rank_prefix_sum_i32,
        recv_gbl_channel_prefix_matrix_i32,
        send_rdma_head_i32,
        send_nvl_head_i32,
        num_recv_tokens_i32,
        num_recv_rdma_tokens_i32,
        num_sms=np.int32(config.num_sms),
        num_max_nvl_chunked_send_tokens=np.int32(config.num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens=np.int32(config.num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens=np.int32(config.num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens=np.int32(config.num_max_rdma_chunked_recv_tokens),
        low_latency_mode=_internode_low_latency_mode(),
    )
    return combined


def _dispatch_internode_bwd_fused_impl(
    *,
    grad_recv_x_base: jax.Array,
    grad_recv_topk_weights_base: jax.Array,
    grad_x_dispatch: jax.Array,
    grad_assignment_weights: jax.Array,
    assignment_destinations: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    dispatch_config: InternodeConfig | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Fuse internode dispatch-backward assignment gradients with cached combine."""
    _register_internode_targets()
    config = dispatch_config or _default_internode_dispatch_config()
    if config.num_sms <= 0 or config.num_sms % 2 != 0:
        raise ValueError(
            f"internode fused dispatch backward num_sms must be a positive even integer, got {config.num_sms}"
        )

    grad_recv_x_base_bf16 = jnp.asarray(grad_recv_x_base, dtype=jnp.bfloat16)
    grad_recv_topk_weights_base_f32 = jnp.asarray(grad_recv_topk_weights_base, dtype=jnp.float32)
    grad_x_dispatch_bf16 = jnp.asarray(grad_x_dispatch, dtype=jnp.bfloat16)
    grad_assignment_weights_f32 = jnp.asarray(grad_assignment_weights, dtype=jnp.float32)
    assignment_destinations_i32 = jnp.asarray(assignment_destinations, dtype=jnp.int32)
    is_token_in_rank_bool = jnp.asarray(is_token_in_rank, dtype=jnp.bool_)
    recv_src_meta_u8 = jnp.asarray(recv_src_meta, dtype=jnp.uint8)
    recv_rdma_channel_prefix_matrix_i32 = jnp.asarray(recv_rdma_channel_prefix_matrix, dtype=jnp.int32)
    recv_rdma_rank_prefix_sum_i32 = jnp.asarray(recv_rdma_rank_prefix_sum, dtype=jnp.int32)
    recv_gbl_channel_prefix_matrix_i32 = jnp.asarray(recv_gbl_channel_prefix_matrix, dtype=jnp.int32)
    send_rdma_head_i32 = jnp.asarray(send_rdma_head, dtype=jnp.int32)
    send_nvl_head_i32 = jnp.asarray(send_nvl_head, dtype=jnp.int32)
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    num_recv_rdma_tokens_i32 = jnp.asarray(num_recv_rdma_tokens, dtype=jnp.int32)

    if num_recv_tokens_i32.ndim == 0:
        num_recv_tokens_i32 = jnp.reshape(num_recv_tokens_i32, (1,))
    if num_recv_rdma_tokens_i32.ndim == 0:
        num_recv_rdma_tokens_i32 = jnp.reshape(num_recv_rdma_tokens_i32, (1,))

    combined_tokens = send_rdma_head_i32.shape[0]
    hidden = grad_recv_x_base_bf16.shape[1]
    topk = grad_recv_topk_weights_base_f32.shape[1]
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((combined_tokens, hidden), grad_recv_x_base_bf16.dtype),
        jax.ShapeDtypeStruct((combined_tokens, topk), jnp.float32),
    )
    grad_x, grad_topk_weights = jax.ffi.ffi_call(
        _DISPATCH_INTERNODE_BWD_FUSED_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        grad_recv_x_base_bf16,
        grad_recv_topk_weights_base_f32,
        grad_x_dispatch_bf16,
        grad_assignment_weights_f32,
        assignment_destinations_i32,
        is_token_in_rank_bool,
        recv_src_meta_u8,
        recv_rdma_channel_prefix_matrix_i32,
        recv_rdma_rank_prefix_sum_i32,
        recv_gbl_channel_prefix_matrix_i32,
        send_rdma_head_i32,
        send_nvl_head_i32,
        num_recv_tokens_i32,
        num_recv_rdma_tokens_i32,
        num_sms=np.int32(config.num_sms),
        num_max_nvl_chunked_send_tokens=np.int32(config.num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens=np.int32(config.num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens=np.int32(config.num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens=np.int32(config.num_max_rdma_chunked_recv_tokens),
        low_latency_mode=_internode_low_latency_mode(),
    )
    return grad_x, grad_topk_weights


@partial(jax.custom_vjp, nondiff_argnums=(14,))
def _combine_internode_with_vjp(
    recv_x: jax.Array,
    recv_topk_weights: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    dispatch_config: InternodeConfig | None,
) -> tuple[jax.Array, jax.Array]:
    return _combine_internode_impl(
        recv_x,
        recv_topk_weights,
        is_token_in_rank,
        recv_src_meta,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        recv_gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        dispatch_config=dispatch_config,
    )


def _combine_internode_with_vjp_fwd(
    recv_x: jax.Array,
    recv_topk_weights: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    dispatch_config: InternodeConfig | None,
):
    outputs = _combine_internode_impl(
        recv_x,
        recv_topk_weights,
        is_token_in_rank,
        recv_src_meta,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        recv_gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        dispatch_config=dispatch_config,
    )
    residuals = (
        recv_x,
        recv_topk_weights,
        is_token_in_rank,
        rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        num_recv_tokens,
        num_recv_rdma_tokens,
    )
    return outputs, residuals


def _combine_internode_with_vjp_bwd(
    dispatch_config: InternodeConfig | None,
    residuals,
    cotangents,
):
    (
        recv_x,
        recv_topk_weights,
        is_token_in_rank,
        rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        num_recv_tokens,
        num_recv_rdma_tokens,
    ) = residuals
    grad_combined_x = _materialize_cotangent(
        cotangents[0],
        dtype=recv_x.dtype,
        shape=(is_token_in_rank.shape[0], recv_x.shape[1]),
    )
    grad_recv_x = _dispatch_internode_cached_impl(
        grad_combined_x,
        is_token_in_rank,
        rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        num_recv_tokens,
        num_recv_rdma_tokens,
        max_recv_tokens=recv_topk_weights.shape[0],
        num_topk=recv_topk_weights.shape[1],
        dispatch_config=dispatch_config,
    )
    grad_recv_topk_weights = jnp.zeros_like(recv_topk_weights)
    return (
        grad_recv_x,
        grad_recv_topk_weights,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_combine_internode_with_vjp.defvjp(
    _combine_internode_with_vjp_fwd,
    _combine_internode_with_vjp_bwd,
)


def deepep_combine_internode(
    recv_x: jax.Array,
    recv_topk_weights: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    *,
    dispatch_config: InternodeConfig | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Combine tokens with DeepEP normal-mode internode transport."""
    return _combine_internode_with_vjp(
        recv_x,
        recv_topk_weights,
        is_token_in_rank,
        recv_src_meta,
        rdma_channel_prefix_matrix,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        dispatch_config,
    )


@partial(jax.custom_vjp, nondiff_argnums=(13, 14))
def _combine_internode_x_only_with_vjp(
    recv_x: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    num_topk: int,
    dispatch_config: InternodeConfig | None,
) -> jax.Array:
    del num_topk
    del rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum
    return _combine_internode_x_only_impl(
        recv_x,
        is_token_in_rank,
        recv_src_meta,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        recv_gbl_channel_prefix_matrix,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        dispatch_config=dispatch_config,
    )


def _combine_internode_x_only_with_vjp_fwd(
    recv_x: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    num_topk: int,
    dispatch_config: InternodeConfig | None,
):
    del num_topk
    output = _combine_internode_x_only_impl(
        recv_x,
        is_token_in_rank,
        recv_src_meta,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        recv_gbl_channel_prefix_matrix,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        dispatch_config=dispatch_config,
    )
    residuals = (
        recv_x,
        is_token_in_rank,
        rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        num_recv_tokens,
        num_recv_rdma_tokens,
    )
    return output, residuals


def _combine_internode_x_only_with_vjp_bwd(
    num_topk: int,
    dispatch_config: InternodeConfig | None,
    residuals,
    cotangent,
):
    (
        recv_x,
        is_token_in_rank,
        rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        num_recv_tokens,
        num_recv_rdma_tokens,
    ) = residuals
    grad_combined_x = _materialize_cotangent(
        cotangent,
        dtype=recv_x.dtype,
        shape=(is_token_in_rank.shape[0], recv_x.shape[1]),
    )
    grad_recv_x = _dispatch_internode_cached_impl(
        grad_combined_x,
        is_token_in_rank,
        rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        num_recv_tokens,
        num_recv_rdma_tokens,
        max_recv_tokens=recv_x.shape[0],
        num_topk=num_topk,
        dispatch_config=dispatch_config,
    )
    return (
        grad_recv_x,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_combine_internode_x_only_with_vjp.defvjp(
    _combine_internode_x_only_with_vjp_fwd,
    _combine_internode_x_only_with_vjp_bwd,
)


def deepep_combine_internode_x_only(
    recv_x: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    *,
    num_topk: int,
    dispatch_config: InternodeConfig | None = None,
) -> jax.Array:
    """Combine token payloads when the caller does not need combined top-k weights."""
    if num_topk <= 0:
        raise ValueError(f"num_topk must be positive for x-only combine backward routing, got {num_topk}")
    return _combine_internode_x_only_with_vjp(
        recv_x,
        is_token_in_rank,
        recv_src_meta,
        rdma_channel_prefix_matrix,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        num_topk,
        dispatch_config,
    )


def _combine_internode_with_local_collapse_impl(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    recv_topk_weights: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    *,
    dispatch_config: InternodeConfig | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Collapse locally packed expert outputs inside the internode combine FFI."""
    _register_internode_targets()
    config = dispatch_config or _default_internode_dispatch_config()
    if config.num_sms <= 0 or config.num_sms % 2 != 0:
        raise ValueError(f"internode combine num_sms must be a positive even integer, got {config.num_sms}")

    out_dispatch_bf16 = jnp.asarray(out_dispatch, dtype=jnp.bfloat16)
    assignment_weights_bf16 = jnp.asarray(assignment_weights, dtype=jnp.bfloat16)
    assignment_destinations_i32 = jnp.asarray(assignment_destinations, dtype=jnp.int32)
    local_group_sizes_i32 = jnp.asarray(local_group_sizes, dtype=jnp.int32)
    accepted_total_i32 = jnp.reshape(jnp.sum(local_group_sizes_i32, dtype=jnp.int32), (1,))
    recv_topk_weights_f32 = jnp.asarray(recv_topk_weights, dtype=jnp.float32)
    is_token_in_rank_bool = jnp.asarray(is_token_in_rank, dtype=jnp.bool_)
    recv_src_meta_u8 = jnp.asarray(recv_src_meta, dtype=jnp.uint8)
    recv_rdma_channel_prefix_matrix_i32 = jnp.asarray(recv_rdma_channel_prefix_matrix, dtype=jnp.int32)
    recv_rdma_rank_prefix_sum_i32 = jnp.asarray(recv_rdma_rank_prefix_sum, dtype=jnp.int32)
    recv_gbl_channel_prefix_matrix_i32 = jnp.asarray(recv_gbl_channel_prefix_matrix, dtype=jnp.int32)
    send_rdma_head_i32 = jnp.asarray(send_rdma_head, dtype=jnp.int32)
    send_nvl_head_i32 = jnp.asarray(send_nvl_head, dtype=jnp.int32)
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    num_recv_rdma_tokens_i32 = jnp.asarray(num_recv_rdma_tokens, dtype=jnp.int32)

    if num_recv_tokens_i32.ndim == 0:
        num_recv_tokens_i32 = jnp.reshape(num_recv_tokens_i32, (1,))
    if num_recv_rdma_tokens_i32.ndim == 0:
        num_recv_rdma_tokens_i32 = jnp.reshape(num_recv_rdma_tokens_i32, (1,))

    combined_tokens = send_rdma_head_i32.shape[0]
    hidden = out_dispatch_bf16.shape[1]
    topk = recv_topk_weights_f32.shape[1]
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((combined_tokens, hidden), out_dispatch_bf16.dtype),
        jax.ShapeDtypeStruct((combined_tokens, topk), jnp.float32),
    )
    combined, combined_weights = jax.ffi.ffi_call(
        _COMBINE_INTERNODE_WITH_LOCAL_COLLAPSE_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        out_dispatch_bf16,
        assignment_weights_bf16,
        assignment_destinations_i32,
        accepted_total_i32,
        recv_topk_weights_f32,
        is_token_in_rank_bool,
        recv_src_meta_u8,
        recv_rdma_channel_prefix_matrix_i32,
        recv_rdma_rank_prefix_sum_i32,
        recv_gbl_channel_prefix_matrix_i32,
        send_rdma_head_i32,
        send_nvl_head_i32,
        num_recv_tokens_i32,
        num_recv_rdma_tokens_i32,
        num_sms=np.int32(config.num_sms),
        num_max_nvl_chunked_send_tokens=np.int32(config.num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens=np.int32(config.num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens=np.int32(config.num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens=np.int32(config.num_max_rdma_chunked_recv_tokens),
        low_latency_mode=_internode_low_latency_mode(),
    )
    return combined, combined_weights


@partial(jax.custom_vjp, nondiff_argnums=(18,))
def _combine_internode_with_local_collapse_vjp(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    recv_topk_weights: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    dispatch_config: InternodeConfig | None,
) -> tuple[jax.Array, jax.Array]:
    del recv_token_indices, rdma_channel_prefix_matrix, recv_gbl_rank_prefix_sum
    return _combine_internode_with_local_collapse_impl(
        out_dispatch,
        assignment_weights,
        assignment_destinations,
        local_group_sizes,
        recv_topk_weights,
        is_token_in_rank,
        recv_src_meta,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        recv_gbl_channel_prefix_matrix,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        dispatch_config=dispatch_config,
    )


def _combine_internode_with_local_collapse_vjp_fwd(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    recv_topk_weights: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    dispatch_config: InternodeConfig | None,
):
    outputs = _combine_internode_with_local_collapse_impl(
        out_dispatch,
        assignment_weights,
        assignment_destinations,
        local_group_sizes,
        recv_topk_weights,
        is_token_in_rank,
        recv_src_meta,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        recv_gbl_channel_prefix_matrix,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        dispatch_config=dispatch_config,
    )
    residuals = (
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        local_group_sizes,
        recv_topk_weights,
        is_token_in_rank,
        rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        num_recv_tokens,
        num_recv_rdma_tokens,
    )
    return outputs, residuals


def _combine_internode_with_local_collapse_vjp_bwd(
    dispatch_config: InternodeConfig | None,
    residuals,
    cotangents,
):
    (
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        local_group_sizes,
        recv_topk_weights,
        is_token_in_rank,
        rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        num_recv_tokens,
        num_recv_rdma_tokens,
    ) = residuals
    grad_combined_x = _materialize_cotangent(
        cotangents[0],
        dtype=out_dispatch.dtype,
        shape=(is_token_in_rank.shape[0], out_dispatch.shape[1]),
    )
    grad_recv_out = _dispatch_internode_cached_impl(
        grad_combined_x,
        is_token_in_rank,
        rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        num_recv_tokens,
        num_recv_rdma_tokens,
        max_recv_tokens=recv_topk_weights.shape[0],
        num_topk=recv_topk_weights.shape[1],
        dispatch_config=dispatch_config,
    )
    grad_out_dispatch, grad_assignment_weights = _collapse_local_assignments_internode_bwd_impl(
        grad_recv_out,
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        local_group_sizes,
    )
    return (
        grad_out_dispatch,
        grad_assignment_weights,
        None,
        None,
        None,
        jnp.zeros_like(recv_topk_weights),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_combine_internode_with_local_collapse_vjp.defvjp(
    _combine_internode_with_local_collapse_vjp_fwd,
    _combine_internode_with_local_collapse_vjp_bwd,
)


def _collapse_local_assignments_internode_bwd_impl(
    grad_recv_out: jax.Array,
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    local_group_sizes: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Backward pass for internode local assignment collapse using a CUDA FFI target."""
    _register_internode_targets()
    grad_recv_out_bf16 = jnp.asarray(grad_recv_out, dtype=jnp.bfloat16)
    out_dispatch_bf16 = jnp.asarray(out_dispatch, dtype=jnp.bfloat16)
    assignment_weights_bf16 = jnp.asarray(assignment_weights, dtype=jnp.bfloat16)
    recv_token_indices_i32 = jnp.asarray(recv_token_indices, dtype=jnp.int32)
    local_group_sizes_i32 = jnp.asarray(local_group_sizes, dtype=jnp.int32)
    accepted_total_i32 = jnp.reshape(jnp.sum(local_group_sizes_i32, dtype=jnp.int32), (1,))

    result_shape_dtypes = (
        jax.ShapeDtypeStruct(out_dispatch_bf16.shape, out_dispatch_bf16.dtype),
        jax.ShapeDtypeStruct(assignment_weights_bf16.shape, jnp.float32),
    )
    grad_out_dispatch, grad_assignment_weights = jax.ffi.ffi_call(
        _COLLAPSE_LOCAL_ASSIGNMENTS_INTERNODE_BWD_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        grad_recv_out_bf16,
        out_dispatch_bf16,
        assignment_weights_bf16,
        recv_token_indices_i32,
        accepted_total_i32,
    )
    return grad_out_dispatch, grad_assignment_weights


def deepep_combine_internode_with_local_collapse(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    recv_topk_weights: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    *,
    dispatch_config: InternodeConfig | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Collapse packed local expert outputs inside the DeepEP internode combine FFI."""
    return _combine_internode_with_local_collapse_vjp(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        assignment_destinations,
        local_group_sizes,
        recv_topk_weights,
        is_token_in_rank,
        recv_src_meta,
        rdma_channel_prefix_matrix,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        dispatch_config,
    )


def _combine_internode_x_only_with_local_collapse_impl(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    *,
    num_topk: int,
    dispatch_config: InternodeConfig | None = None,
) -> jax.Array:
    """Collapse locally packed expert outputs inside the x-only internode combine FFI."""
    _register_internode_targets()
    config = dispatch_config or _default_internode_dispatch_config()
    if config.num_sms <= 0 or config.num_sms % 2 != 0:
        raise ValueError(f"internode x-only combine num_sms must be a positive even integer, got {config.num_sms}")
    if num_topk <= 0:
        raise ValueError(f"num_topk must be positive for x-only fused local collapse, got {num_topk}")

    out_dispatch_bf16 = jnp.asarray(out_dispatch, dtype=jnp.bfloat16)
    assignment_weights_bf16 = jnp.asarray(assignment_weights, dtype=jnp.bfloat16)
    assignment_destinations_i32 = jnp.asarray(assignment_destinations, dtype=jnp.int32)
    local_group_sizes_i32 = jnp.asarray(local_group_sizes, dtype=jnp.int32)
    accepted_total_i32 = jnp.reshape(jnp.sum(local_group_sizes_i32, dtype=jnp.int32), (1,))
    is_token_in_rank_bool = jnp.asarray(is_token_in_rank, dtype=jnp.bool_)
    recv_src_meta_u8 = jnp.asarray(recv_src_meta, dtype=jnp.uint8)
    recv_rdma_channel_prefix_matrix_i32 = jnp.asarray(recv_rdma_channel_prefix_matrix, dtype=jnp.int32)
    recv_rdma_rank_prefix_sum_i32 = jnp.asarray(recv_rdma_rank_prefix_sum, dtype=jnp.int32)
    recv_gbl_channel_prefix_matrix_i32 = jnp.asarray(recv_gbl_channel_prefix_matrix, dtype=jnp.int32)
    send_rdma_head_i32 = jnp.asarray(send_rdma_head, dtype=jnp.int32)
    send_nvl_head_i32 = jnp.asarray(send_nvl_head, dtype=jnp.int32)
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    num_recv_rdma_tokens_i32 = jnp.asarray(num_recv_rdma_tokens, dtype=jnp.int32)

    if num_recv_tokens_i32.ndim == 0:
        num_recv_tokens_i32 = jnp.reshape(num_recv_tokens_i32, (1,))
    if num_recv_rdma_tokens_i32.ndim == 0:
        num_recv_rdma_tokens_i32 = jnp.reshape(num_recv_rdma_tokens_i32, (1,))

    combined_tokens = send_rdma_head_i32.shape[0]
    hidden = out_dispatch_bf16.shape[1]
    result_shape_dtype = jax.ShapeDtypeStruct((combined_tokens, hidden), out_dispatch_bf16.dtype)
    combined = jax.ffi.ffi_call(
        _COMBINE_INTERNODE_X_ONLY_WITH_LOCAL_COLLAPSE_TARGET,
        result_shape_dtype,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        out_dispatch_bf16,
        assignment_weights_bf16,
        assignment_destinations_i32,
        accepted_total_i32,
        is_token_in_rank_bool,
        recv_src_meta_u8,
        recv_rdma_channel_prefix_matrix_i32,
        recv_rdma_rank_prefix_sum_i32,
        recv_gbl_channel_prefix_matrix_i32,
        send_rdma_head_i32,
        send_nvl_head_i32,
        num_recv_tokens_i32,
        num_recv_rdma_tokens_i32,
        num_topk=np.int32(num_topk),
        num_sms=np.int32(config.num_sms),
        num_max_nvl_chunked_send_tokens=np.int32(config.num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens=np.int32(config.num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens=np.int32(config.num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens=np.int32(config.num_max_rdma_chunked_recv_tokens),
        low_latency_mode=_internode_low_latency_mode(),
    )
    return combined


@partial(jax.custom_vjp, nondiff_argnums=(17, 18))
def _combine_internode_x_only_with_local_collapse_vjp(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    num_topk: int,
    dispatch_config: InternodeConfig | None,
) -> jax.Array:
    del recv_token_indices, rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum
    return _combine_internode_x_only_with_local_collapse_impl(
        out_dispatch,
        assignment_weights,
        assignment_destinations,
        local_group_sizes,
        is_token_in_rank,
        recv_src_meta,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        recv_gbl_channel_prefix_matrix,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        num_topk=num_topk,
        dispatch_config=dispatch_config,
    )


def _combine_internode_x_only_with_local_collapse_vjp_fwd(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    num_topk: int,
    dispatch_config: InternodeConfig | None,
):
    output = _combine_internode_x_only_with_local_collapse_impl(
        out_dispatch,
        assignment_weights,
        assignment_destinations,
        local_group_sizes,
        is_token_in_rank,
        recv_src_meta,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        recv_gbl_channel_prefix_matrix,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        num_topk=num_topk,
        dispatch_config=dispatch_config,
    )
    residuals = (
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        assignment_destinations,
        local_group_sizes,
        is_token_in_rank,
        rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        num_recv_tokens,
        num_recv_rdma_tokens,
    )
    return output, residuals


def _combine_internode_x_only_with_local_collapse_vjp_bwd(
    num_topk: int,
    dispatch_config: InternodeConfig | None,
    residuals,
    cotangent,
):
    (
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        assignment_destinations,
        local_group_sizes,
        is_token_in_rank,
        rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        num_recv_tokens,
        num_recv_rdma_tokens,
    ) = residuals
    if assignment_destinations.shape[0] % num_topk != 0:
        raise ValueError(
            f"assignment_destinations length must be divisible by num_topk, "
            f"got {assignment_destinations.shape[0]=} {num_topk=}"
        )
    recv_capacity = assignment_destinations.shape[0] // num_topk
    grad_combined_x = _materialize_cotangent(
        cotangent,
        dtype=out_dispatch.dtype,
        shape=(is_token_in_rank.shape[0], out_dispatch.shape[1]),
    )
    grad_recv_out = _dispatch_internode_cached_impl(
        grad_combined_x,
        is_token_in_rank,
        rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        num_recv_tokens,
        num_recv_rdma_tokens,
        max_recv_tokens=recv_capacity,
        num_topk=num_topk,
        dispatch_config=dispatch_config,
    )
    grad_out_dispatch, grad_assignment_weights = _collapse_local_assignments_internode_bwd_impl(
        grad_recv_out,
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        local_group_sizes,
    )
    return (
        grad_out_dispatch,
        grad_assignment_weights,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_combine_internode_x_only_with_local_collapse_vjp.defvjp(
    _combine_internode_x_only_with_local_collapse_vjp_fwd,
    _combine_internode_x_only_with_local_collapse_vjp_bwd,
)


def deepep_combine_internode_x_only_with_local_collapse(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    is_token_in_rank: jax.Array,
    recv_src_meta: jax.Array,
    rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_channel_prefix_matrix: jax.Array,
    recv_rdma_rank_prefix_sum: jax.Array,
    gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_channel_prefix_matrix: jax.Array,
    recv_gbl_rank_prefix_sum: jax.Array,
    send_rdma_head: jax.Array,
    send_nvl_head: jax.Array,
    num_recv_tokens: jax.Array,
    num_recv_rdma_tokens: jax.Array,
    *,
    num_topk: int,
    dispatch_config: InternodeConfig | None = None,
) -> jax.Array:
    """Collapse packed local expert outputs inside x-only DeepEP internode combine."""
    return _combine_internode_x_only_with_local_collapse_vjp(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        assignment_destinations,
        local_group_sizes,
        is_token_in_rank,
        recv_src_meta,
        rdma_channel_prefix_matrix,
        recv_rdma_channel_prefix_matrix,
        recv_rdma_rank_prefix_sum,
        gbl_channel_prefix_matrix,
        recv_gbl_channel_prefix_matrix,
        recv_gbl_rank_prefix_sum,
        send_rdma_head,
        send_nvl_head,
        num_recv_tokens,
        num_recv_rdma_tokens,
        num_topk,
        dispatch_config,
    )


def _resolve_runtime(
    *,
    x: jax.Array,
    num_ranks: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
) -> IntranodeConfig:
    _register_targets()
    hidden_bytes = x.shape[1] * max(jnp.dtype(x.dtype).itemsize, 2)
    ensure_intranode_runtime(
        num_ranks=num_ranks,
        hidden_bytes=hidden_bytes,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
    )
    return dispatch_config or _default_dispatch_config(num_ranks)


def _dispatch_intranode_impl(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    *,
    num_experts: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int | None,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
    num_ranks = int(num_tokens_per_rank.shape[0])
    resolved_dispatch_config = _resolve_runtime(
        x=x,
        num_ranks=num_ranks,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
    )

    x_bf16 = jnp.asarray(x, dtype=jnp.bfloat16)
    topk_idx_i32 = jnp.asarray(topk_idx, dtype=jnp.int32)
    topk_weights_f32 = jnp.asarray(topk_weights, dtype=jnp.float32)
    num_tokens_per_rank_i32 = jnp.asarray(num_tokens_per_rank, dtype=jnp.int32)
    num_tokens_per_expert_i32 = jnp.asarray(num_tokens_per_expert, dtype=jnp.int32)
    local_experts = num_experts // num_ranks
    if max_recv_tokens is None:
        max_recv_tokens = x_bf16.shape[0] * num_ranks
    elif max_recv_tokens <= 0:
        raise ValueError(f"max_recv_tokens must be positive, got {max_recv_tokens}")
    num_channels = resolved_dispatch_config.num_sms // 2
    topk = topk_idx_i32.shape[1]
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((max_recv_tokens, x_bf16.shape[1]), x_bf16.dtype),
        jax.ShapeDtypeStruct((max_recv_tokens, topk), jnp.int32),
        jax.ShapeDtypeStruct((max_recv_tokens, topk), jnp.float32),
        jax.ShapeDtypeStruct((max_recv_tokens,), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_ranks), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),
        jax.ShapeDtypeStruct((x_bf16.shape[0], num_ranks), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((x_bf16.shape[0], topk * 2), jnp.int32),
        jax.ShapeDtypeStruct((max_recv_tokens, topk * 2), jnp.int32),
    )
    results = jax.ffi.ffi_call(
        _DISPATCH_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        x_bf16,
        topk_idx_i32,
        topk_weights_f32,
        num_tokens_per_rank_i32,
        num_tokens_per_expert_i32,
        is_token_in_rank,
        num_experts=np.int32(num_experts),
    )
    return (
        results[0],
        results[1],
        results[2],
        results[3],
        results[4],
        results[5],
        results[6],
        results[7],
        results[8],
        results[9],
    )


def _dispatch_intranode_with_assignments_impl(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    *,
    num_experts: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int | None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    _register_targets()
    num_ranks = int(num_tokens_per_rank.shape[0])
    resolved_dispatch_config = _resolve_runtime(
        x=x,
        num_ranks=num_ranks,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
    )

    x_bf16 = jnp.asarray(x, dtype=jnp.bfloat16)
    topk_idx_i32 = jnp.asarray(topk_idx, dtype=jnp.int32)
    topk_weights_f32 = jnp.asarray(topk_weights, dtype=jnp.float32)
    num_tokens_per_rank_i32 = jnp.asarray(num_tokens_per_rank, dtype=jnp.int32)
    num_tokens_per_expert_i32 = jnp.asarray(num_tokens_per_expert, dtype=jnp.int32)
    local_experts = num_experts // num_ranks
    if max_recv_tokens is None:
        max_recv_tokens = x_bf16.shape[0] * num_ranks
    elif max_recv_tokens <= 0:
        raise ValueError(f"max_recv_tokens must be positive, got {max_recv_tokens}")
    num_channels = resolved_dispatch_config.num_sms // 2
    topk = topk_idx_i32.shape[1]
    max_assignments = max_recv_tokens * topk
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((max_recv_tokens, x_bf16.shape[1]), x_bf16.dtype),
        jax.ShapeDtypeStruct((max_recv_tokens, topk), jnp.float32),
        jax.ShapeDtypeStruct((max_recv_tokens,), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_ranks), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),
        jax.ShapeDtypeStruct((x_bf16.shape[0], num_ranks), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((x_bf16.shape[0], topk * 2), jnp.int32),
        jax.ShapeDtypeStruct((max_recv_tokens, topk * 2), jnp.int32),
        jax.ShapeDtypeStruct((max_assignments, x_bf16.shape[1]), x_bf16.dtype),
        jax.ShapeDtypeStruct((max_assignments,), x_bf16.dtype),
        jax.ShapeDtypeStruct((max_assignments,), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((max_assignments,), jnp.int32),
        jax.ShapeDtypeStruct((max_assignments,), jnp.int32),
    )
    (
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        local_group_sizes,
        num_recv_tokens,
        _topk_idx_s64_scratch,
        _recv_topk_idx_s64_scratch,
        x_dispatch,
        assignment_weights,
        recv_token_indices,
        _local_group_cursors,
        recv_assignment_indices,
        assignment_destinations,
    ) = jax.ffi.ffi_call(
        _DISPATCH_WITH_ASSIGNMENTS_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        x_bf16,
        topk_idx_i32,
        topk_weights_f32,
        num_tokens_per_rank_i32,
        num_tokens_per_expert_i32,
        is_token_in_rank,
        num_experts=np.int32(num_experts),
    )
    return (
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        local_group_sizes,
        num_recv_tokens,
        x_dispatch,
        assignment_weights,
        recv_token_indices,
        recv_assignment_indices,
        assignment_destinations,
    )


def _dispatch_intranode_cached_impl(
    x: jax.Array,
    is_token_in_rank: jax.Array,
    rank_prefix_matrix: jax.Array,
    channel_prefix_matrix: jax.Array,
    num_recv_tokens: jax.Array,
    *,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int,
) -> jax.Array:
    if max_recv_tokens <= 0:
        raise ValueError(f"max_recv_tokens must be positive, got {max_recv_tokens}")
    num_ranks = int(rank_prefix_matrix.shape[0])
    resolved_dispatch_config = _resolve_runtime(
        x=x,
        num_ranks=num_ranks,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
    )

    x_bf16 = jnp.asarray(x, dtype=jnp.bfloat16)
    is_token_in_rank_bool = jnp.asarray(is_token_in_rank, dtype=jnp.bool_)
    rank_prefix_matrix_i32 = jnp.asarray(rank_prefix_matrix, dtype=jnp.int32)
    channel_prefix_matrix_i32 = jnp.asarray(channel_prefix_matrix, dtype=jnp.int32)
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    num_channels = resolved_dispatch_config.num_sms // 2
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((max_recv_tokens, x_bf16.shape[1]), x_bf16.dtype),
        jax.ShapeDtypeStruct((max_recv_tokens,), jnp.int32),
        jax.ShapeDtypeStruct((num_ranks, num_channels), jnp.int32),
        jax.ShapeDtypeStruct((x_bf16.shape[0], num_ranks), jnp.int32),
    )
    recv_x, _, _, _ = jax.ffi.ffi_call(
        _DISPATCH_CACHED_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        x_bf16,
        is_token_in_rank_bool,
        rank_prefix_matrix_i32,
        channel_prefix_matrix_i32,
        num_recv_tokens_i32,
    )
    recv_token_limit = jnp.squeeze(num_recv_tokens_i32, axis=0)
    recv_valid = jnp.arange(max_recv_tokens, dtype=jnp.int32) < recv_token_limit
    return jnp.where(recv_valid[:, None], recv_x, 0)


def _combine_intranode_impl(
    recv_x: jax.Array,
    recv_topk_weights: jax.Array,
    recv_src_idx: jax.Array,
    rank_prefix_matrix: jax.Array,
    channel_prefix_matrix: jax.Array,
    send_head: jax.Array,
    num_recv_tokens: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    _register_targets()
    recv_x_bf16 = jnp.asarray(recv_x, dtype=jnp.bfloat16)
    recv_topk_weights_f32 = jnp.asarray(recv_topk_weights, dtype=jnp.float32)
    recv_src_idx_i32 = jnp.asarray(recv_src_idx, dtype=jnp.int32)
    rank_prefix_matrix_i32 = jnp.asarray(rank_prefix_matrix, dtype=jnp.int32)
    channel_prefix_matrix_i32 = jnp.asarray(channel_prefix_matrix, dtype=jnp.int32)
    send_head_i32 = jnp.asarray(send_head, dtype=jnp.int32)
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    topk = recv_topk_weights_f32.shape[1]
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((send_head_i32.shape[0], recv_x_bf16.shape[1]), recv_x_bf16.dtype),
        jax.ShapeDtypeStruct((send_head_i32.shape[0], topk), jnp.float32),
        jax.ShapeDtypeStruct(send_head_i32.shape, send_head_i32.dtype),
    )
    combined_x, combined_topk_weights, _ = jax.ffi.ffi_call(
        _COMBINE_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
        input_output_aliases={5: 2},
    )(
        recv_x_bf16,
        recv_topk_weights_f32,
        recv_src_idx_i32,
        rank_prefix_matrix_i32,
        channel_prefix_matrix_i32,
        send_head_i32,
        num_recv_tokens_i32,
    )
    return combined_x, combined_topk_weights


def deepep_pack_local_assignments(
    recv_x: jax.Array,
    recv_topk_idx: jax.Array,
    recv_topk_weights: jax.Array,
    num_recv_tokens: jax.Array,
    *,
    local_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    x_dispatch, assignment_weights, recv_token_indices, local_group_sizes, _ = _pack_local_assignments_with_vjp(
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens,
        local_experts,
    )
    return x_dispatch, assignment_weights, recv_token_indices, local_group_sizes


def deepep_pack_local_assignments_from_counts(
    recv_x: jax.Array,
    recv_topk_idx: jax.Array,
    recv_topk_weights: jax.Array,
    num_recv_tokens: jax.Array,
    local_group_sizes: jax.Array,
    *,
    assignment_capacity: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    x_dispatch, assignment_weights, recv_token_indices, recv_assignment_indices, assignment_destinations = (
        _pack_local_assignments_from_counts_with_vjp(
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens,
            local_group_sizes,
            assignment_capacity,
        )
    )
    del recv_assignment_indices
    return x_dispatch, assignment_weights, recv_token_indices, local_group_sizes, assignment_destinations


def _pack_local_assignments_impl(
    recv_x: jax.Array,
    recv_topk_idx: jax.Array,
    recv_topk_weights: jax.Array,
    num_recv_tokens: jax.Array,
    *,
    local_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    _register_targets()
    if local_experts <= 0:
        raise ValueError(f"local_experts must be positive, got {local_experts}")
    recv_x_bf16 = jnp.asarray(recv_x, dtype=jnp.bfloat16)
    recv_topk_idx_i32 = jnp.asarray(recv_topk_idx, dtype=jnp.int32)
    recv_topk_weights_f32 = jnp.asarray(recv_topk_weights, dtype=jnp.float32)
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    if num_recv_tokens_i32.ndim == 0:
        num_recv_tokens_i32 = jnp.reshape(num_recv_tokens_i32, (1,))

    recv_capacity, hidden = recv_x_bf16.shape
    topk = recv_topk_idx_i32.shape[1]
    max_assignments = recv_capacity * topk
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((max_assignments, hidden), recv_x_bf16.dtype),
        jax.ShapeDtypeStruct((max_assignments,), recv_x_bf16.dtype),
        jax.ShapeDtypeStruct((max_assignments,), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((max_assignments,), jnp.int32),
        jax.ShapeDtypeStruct((max_assignments,), jnp.int32),
    )
    (
        x_dispatch,
        assignment_weights,
        recv_token_indices,
        local_group_sizes,
        _,
        recv_assignment_indices,
        _,
    ) = jax.ffi.ffi_call(
        _PACK_LOCAL_ASSIGNMENTS_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        recv_x_bf16,
        recv_topk_idx_i32,
        recv_topk_weights_f32,
        num_recv_tokens_i32,
        local_experts=np.int32(local_experts),
    )
    return x_dispatch, assignment_weights, recv_token_indices, local_group_sizes, recv_assignment_indices


def _pack_local_assignments_from_counts_impl(
    recv_x: jax.Array,
    recv_topk_idx: jax.Array,
    recv_topk_weights: jax.Array,
    num_recv_tokens: jax.Array,
    local_group_sizes: jax.Array,
    assignment_capacity: int | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    _register_targets()
    recv_x_bf16 = jnp.asarray(recv_x, dtype=jnp.bfloat16)
    recv_topk_idx_i32 = jnp.asarray(recv_topk_idx, dtype=jnp.int32)
    recv_topk_weights_f32 = jnp.asarray(recv_topk_weights, dtype=jnp.float32)
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    local_group_sizes_i32 = jnp.asarray(local_group_sizes, dtype=jnp.int32)
    if num_recv_tokens_i32.ndim == 0:
        num_recv_tokens_i32 = jnp.reshape(num_recv_tokens_i32, (1,))

    recv_capacity, hidden = recv_x_bf16.shape
    topk = recv_topk_idx_i32.shape[1]
    max_assignments = recv_capacity * topk
    output_assignments = max_assignments if assignment_capacity is None else assignment_capacity
    if output_assignments <= 0:
        raise ValueError(f"assignment_capacity must be positive, got {assignment_capacity}")
    if output_assignments > max_assignments:
        raise ValueError(
            f"assignment_capacity={output_assignments} cannot exceed recv_capacity * topk={max_assignments}"
        )
    result_shape_dtypes = (
        jax.ShapeDtypeStruct((output_assignments, hidden), recv_x_bf16.dtype),
        jax.ShapeDtypeStruct((output_assignments,), recv_x_bf16.dtype),
        jax.ShapeDtypeStruct((output_assignments,), jnp.int32),
        jax.ShapeDtypeStruct((local_group_sizes_i32.shape[0],), jnp.int32),
        jax.ShapeDtypeStruct((output_assignments,), jnp.int32),
        jax.ShapeDtypeStruct((max_assignments,), jnp.int32),
    )
    (
        x_dispatch,
        assignment_weights,
        recv_token_indices,
        _,
        recv_assignment_indices,
        assignment_destinations,
    ) = jax.ffi.ffi_call(
        _PACK_LOCAL_ASSIGNMENTS_FROM_COUNTS_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        recv_x_bf16,
        recv_topk_idx_i32,
        recv_topk_weights_f32,
        num_recv_tokens_i32,
        local_group_sizes_i32,
    )
    return x_dispatch, assignment_weights, recv_token_indices, recv_assignment_indices, assignment_destinations


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _pack_local_assignments_with_vjp(
    recv_x: jax.Array,
    recv_topk_idx: jax.Array,
    recv_topk_weights: jax.Array,
    num_recv_tokens: jax.Array,
    local_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return _pack_local_assignments_impl(
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens,
        local_experts=local_experts,
    )


def _pack_local_assignments_with_vjp_fwd(
    recv_x: jax.Array,
    recv_topk_idx: jax.Array,
    recv_topk_weights: jax.Array,
    num_recv_tokens: jax.Array,
    local_experts: int,
):
    outputs = _pack_local_assignments_impl(
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens,
        local_experts=local_experts,
    )
    residuals = (outputs[2], outputs[3], outputs[4], recv_x.shape, recv_topk_weights.shape)
    return outputs, residuals


def _pack_assignment_gradients(
    *,
    recv_token_indices: jax.Array,
    recv_assignment_indices: jax.Array,
    local_group_sizes: jax.Array,
    recv_x_shape: tuple[int, ...],
    recv_topk_weights_shape: tuple[int, ...],
    cotangents,
) -> tuple[jax.Array, jax.Array]:
    valid_assignments = jnp.arange(recv_assignment_indices.shape[0], dtype=jnp.int32) < jnp.sum(local_group_sizes)
    safe_recv_token_indices = jnp.where(valid_assignments, recv_token_indices, 0)
    safe_recv_assignment_indices = jnp.where(valid_assignments, recv_assignment_indices, 0)
    grad_x_dispatch = _materialize_cotangent(
        cotangents[0],
        dtype=jnp.bfloat16,
        shape=(recv_assignment_indices.shape[0], recv_x_shape[1]),
    )
    grad_x_dispatch = jnp.where(valid_assignments[:, None], grad_x_dispatch, 0)
    grad_assignment_weights = _materialize_cotangent(
        cotangents[1],
        dtype=jnp.float32,
        shape=(recv_assignment_indices.shape[0],),
    )
    grad_assignment_weights = jnp.where(valid_assignments, grad_assignment_weights, 0)
    grad_recv_x = jax.ops.segment_sum(
        grad_x_dispatch,
        safe_recv_token_indices,
        num_segments=recv_x_shape[0],
        indices_are_sorted=False,
    )
    grad_recv_topk_weights = jax.ops.segment_sum(
        grad_assignment_weights.astype(jnp.float32),
        safe_recv_assignment_indices,
        num_segments=recv_topk_weights_shape[0] * recv_topk_weights_shape[1],
        indices_are_sorted=False,
    ).reshape(recv_topk_weights_shape)
    return grad_recv_x, grad_recv_topk_weights


def _assignment_gradients_impl(
    *,
    grad_x_dispatch: jax.Array,
    grad_assignment_weights: jax.Array,
    assignment_destinations: jax.Array,
    num_recv_tokens: jax.Array,
    recv_x_shape: tuple[int, ...],
    recv_topk_weights_shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    _register_targets()
    grad_x_dispatch_bf16 = jnp.asarray(grad_x_dispatch, dtype=jnp.bfloat16)
    grad_assignment_weights_f32 = jnp.asarray(grad_assignment_weights, dtype=jnp.float32)
    assignment_destinations_i32 = jnp.asarray(assignment_destinations, dtype=jnp.int32)
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    if num_recv_tokens_i32.ndim == 0:
        num_recv_tokens_i32 = jnp.reshape(num_recv_tokens_i32, (1,))

    result_shape_dtypes = (
        jax.ShapeDtypeStruct(recv_x_shape, grad_x_dispatch_bf16.dtype),
        jax.ShapeDtypeStruct(recv_topk_weights_shape, grad_assignment_weights_f32.dtype),
    )
    grad_recv_x, grad_recv_topk_weights = jax.ffi.ffi_call(
        _ASSIGNMENT_GRADIENTS_TARGET,
        result_shape_dtypes,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        grad_x_dispatch_bf16,
        grad_assignment_weights_f32,
        assignment_destinations_i32,
        num_recv_tokens_i32,
    )
    return grad_recv_x, grad_recv_topk_weights


def _assignment_gradients_jax(
    *,
    grad_x_dispatch: jax.Array,
    grad_assignment_weights: jax.Array,
    assignment_destinations: jax.Array,
    num_recv_tokens: jax.Array,
    recv_x_shape: tuple[int, ...],
    recv_topk_weights_shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    grad_x_dispatch = jnp.asarray(grad_x_dispatch, dtype=jnp.bfloat16)
    grad_assignment_weights = jnp.asarray(grad_assignment_weights, dtype=jnp.float32)
    assignment_destinations = jnp.asarray(assignment_destinations, dtype=jnp.int32)
    num_recv_tokens = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    if num_recv_tokens.ndim == 0:
        num_recv_tokens = jnp.reshape(num_recv_tokens, (1,))

    topk = recv_topk_weights_shape[1]
    assignment_index = jnp.arange(assignment_destinations.shape[0], dtype=jnp.int32)
    token_index = assignment_index // topk
    active = assignment_index < num_recv_tokens[0] * topk
    valid = active & (assignment_destinations >= 0)
    safe_destination = jnp.where(valid, assignment_destinations, 0)
    safe_token_index = jnp.where(valid, token_index, 0)

    grad_recv_x_rows = jnp.where(valid[:, None], grad_x_dispatch[safe_destination], 0)
    grad_recv_x = jax.ops.segment_sum(
        grad_recv_x_rows,
        safe_token_index,
        num_segments=recv_x_shape[0],
        indices_are_sorted=False,
    )

    grad_topk_rows = jnp.where(valid, grad_assignment_weights[safe_destination], 0)
    grad_recv_topk_weights = jax.ops.segment_sum(
        grad_topk_rows,
        jnp.where(valid, assignment_index, 0),
        num_segments=recv_topk_weights_shape[0] * recv_topk_weights_shape[1],
        indices_are_sorted=False,
    ).reshape(recv_topk_weights_shape)
    return grad_recv_x, grad_recv_topk_weights


def _pack_local_assignments_with_vjp_bwd(local_experts: int, residuals, cotangents):
    del local_experts
    recv_token_indices, local_group_sizes, recv_assignment_indices, recv_x_shape, recv_topk_weights_shape = residuals
    grad_recv_x, grad_recv_topk_weights = _pack_assignment_gradients(
        recv_token_indices=recv_token_indices,
        recv_assignment_indices=recv_assignment_indices,
        local_group_sizes=local_group_sizes,
        recv_x_shape=recv_x_shape,
        recv_topk_weights_shape=recv_topk_weights_shape,
        cotangents=cotangents,
    )
    return grad_recv_x, None, grad_recv_topk_weights, None


_pack_local_assignments_with_vjp.defvjp(
    _pack_local_assignments_with_vjp_fwd,
    _pack_local_assignments_with_vjp_bwd,
)


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _pack_local_assignments_from_counts_with_vjp(
    recv_x: jax.Array,
    recv_topk_idx: jax.Array,
    recv_topk_weights: jax.Array,
    num_recv_tokens: jax.Array,
    local_group_sizes: jax.Array,
    assignment_capacity: int | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return _pack_local_assignments_from_counts_impl(
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens,
        local_group_sizes,
        assignment_capacity,
    )


def _pack_local_assignments_from_counts_with_vjp_fwd(
    recv_x: jax.Array,
    recv_topk_idx: jax.Array,
    recv_topk_weights: jax.Array,
    num_recv_tokens: jax.Array,
    local_group_sizes: jax.Array,
    assignment_capacity: int | None,
):
    outputs = _pack_local_assignments_from_counts_impl(
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens,
        local_group_sizes,
        assignment_capacity,
    )
    residuals = (
        outputs[4],
        num_recv_tokens,
        recv_x.shape,
        recv_topk_weights.shape,
        outputs[0].shape,
        outputs[1].shape,
    )
    return outputs, residuals


def _pack_local_assignments_from_counts_with_vjp_bwd(assignment_capacity: int | None, residuals, cotangents):
    del assignment_capacity
    assignment_destinations, num_recv_tokens, recv_x_shape, recv_topk_weights_shape, x_dispatch_shape, weight_shape = (
        residuals
    )
    grad_x_dispatch = _materialize_cotangent(cotangents[0], dtype=jnp.bfloat16, shape=x_dispatch_shape)
    grad_assignment_weights = _materialize_cotangent(cotangents[1], dtype=jnp.float32, shape=weight_shape)
    grad_recv_x, grad_recv_topk_weights = _assignment_gradients_impl(
        grad_x_dispatch=grad_x_dispatch,
        grad_assignment_weights=grad_assignment_weights,
        assignment_destinations=assignment_destinations,
        num_recv_tokens=num_recv_tokens,
        recv_x_shape=recv_x_shape,
        recv_topk_weights_shape=recv_topk_weights_shape,
    )
    return grad_recv_x, None, grad_recv_topk_weights, None, None


_pack_local_assignments_from_counts_with_vjp.defvjp(
    _pack_local_assignments_from_counts_with_vjp_fwd,
    _pack_local_assignments_from_counts_with_vjp_bwd,
)


def deepep_collapse_local_assignments(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    num_recv_tokens: jax.Array,
    *,
    recv_capacity: int,
    internode: bool = False,
) -> jax.Array:
    if internode:
        return _collapse_local_assignments_internode_with_vjp(
            out_dispatch,
            assignment_weights,
            recv_token_indices,
            assignment_destinations,
            local_group_sizes,
            num_recv_tokens,
            recv_capacity,
        )
    return _collapse_local_assignments_with_vjp(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        assignment_destinations,
        local_group_sizes,
        num_recv_tokens,
        recv_capacity,
    )


def _collapse_local_assignments_impl(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    num_recv_tokens: jax.Array,
    *,
    recv_capacity: int,
) -> jax.Array:
    _register_targets()
    if recv_capacity <= 0:
        raise ValueError(f"recv_capacity must be positive, got {recv_capacity}")
    out_dispatch_bf16 = jnp.asarray(out_dispatch, dtype=jnp.bfloat16)
    assignment_weights_bf16 = jnp.asarray(assignment_weights, dtype=jnp.bfloat16)
    del recv_token_indices
    assignment_destinations_i32 = jnp.asarray(assignment_destinations, dtype=jnp.int32)
    local_group_sizes_i32 = jnp.asarray(local_group_sizes, dtype=jnp.int32)
    accepted_total_i32 = jnp.reshape(jnp.sum(local_group_sizes_i32, dtype=jnp.int32), (1,))
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    if num_recv_tokens_i32.ndim == 0:
        num_recv_tokens_i32 = jnp.reshape(num_recv_tokens_i32, (1,))

    result_shape_dtype = jax.ShapeDtypeStruct((recv_capacity, out_dispatch_bf16.shape[1]), out_dispatch_bf16.dtype)
    recv_out = jax.ffi.ffi_call(
        _COLLAPSE_LOCAL_ASSIGNMENTS_TARGET,
        result_shape_dtype,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        out_dispatch_bf16,
        assignment_weights_bf16,
        assignment_destinations_i32,
        accepted_total_i32,
        num_recv_tokens_i32,
    )
    return recv_out


def _collapse_local_assignments_internode_impl(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    num_recv_tokens: jax.Array,
    *,
    recv_capacity: int,
) -> jax.Array:
    del recv_token_indices
    _register_internode_targets()
    if recv_capacity <= 0:
        raise ValueError(f"recv_capacity must be positive, got {recv_capacity}")
    out_dispatch_bf16 = jnp.asarray(out_dispatch, dtype=jnp.bfloat16)
    assignment_weights_bf16 = jnp.asarray(assignment_weights, dtype=jnp.bfloat16)
    assignment_destinations_i32 = jnp.asarray(assignment_destinations, dtype=jnp.int32)
    local_group_sizes_i32 = jnp.asarray(local_group_sizes, dtype=jnp.int32)
    accepted_total_i32 = jnp.reshape(jnp.sum(local_group_sizes_i32, dtype=jnp.int32), (1,))
    num_recv_tokens_i32 = jnp.asarray(num_recv_tokens, dtype=jnp.int32)
    if num_recv_tokens_i32.ndim == 0:
        num_recv_tokens_i32 = jnp.reshape(num_recv_tokens_i32, (1,))

    result_shape_dtype = jax.ShapeDtypeStruct((recv_capacity, out_dispatch_bf16.shape[1]), out_dispatch_bf16.dtype)
    recv_out = jax.ffi.ffi_call(
        _COLLAPSE_LOCAL_ASSIGNMENTS_INTERNODE_TARGET,
        result_shape_dtype,
        has_side_effect=True,
        vmap_method="broadcast_all",
    )(
        out_dispatch_bf16,
        assignment_weights_bf16,
        assignment_destinations_i32,
        accepted_total_i32,
        num_recv_tokens_i32,
    )
    return recv_out


@partial(jax.custom_vjp, nondiff_argnums=(6,))
def _collapse_local_assignments_with_vjp(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    num_recv_tokens: jax.Array,
    recv_capacity: int,
) -> jax.Array:
    return _collapse_local_assignments_impl(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        assignment_destinations,
        local_group_sizes,
        num_recv_tokens,
        recv_capacity=recv_capacity,
    )


def _collapse_local_assignments_with_vjp_fwd(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    num_recv_tokens: jax.Array,
    recv_capacity: int,
):
    output = _collapse_local_assignments_impl(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        assignment_destinations,
        local_group_sizes,
        num_recv_tokens,
        recv_capacity=recv_capacity,
    )
    return output, (out_dispatch, assignment_weights, recv_token_indices, local_group_sizes)


def _collapse_local_assignments_with_vjp_bwd(recv_capacity: int, residuals, cotangent):
    out_dispatch, assignment_weights, recv_token_indices, local_group_sizes = residuals
    valid_assignments = jnp.arange(assignment_weights.shape[0], dtype=jnp.int32) < jnp.sum(local_group_sizes)
    safe_recv_token_indices = jnp.where(valid_assignments, recv_token_indices, 0)
    grad_recv_out = _materialize_cotangent(
        cotangent,
        dtype=jnp.bfloat16,
        shape=(recv_capacity, out_dispatch.shape[1]),
    )
    gathered_grad = jnp.take(grad_recv_out, safe_recv_token_indices, axis=0)
    gathered_grad = jnp.where(valid_assignments[:, None], gathered_grad, 0)
    out_dispatch = jnp.where(valid_assignments[:, None], out_dispatch, 0)
    grad_out_dispatch = gathered_grad * assignment_weights[:, None].astype(gathered_grad.dtype)
    grad_assignment_weights = jnp.sum(gathered_grad.astype(jnp.float32) * out_dispatch.astype(jnp.float32), axis=1)
    return grad_out_dispatch, grad_assignment_weights, None, None, None, None


_collapse_local_assignments_with_vjp.defvjp(
    _collapse_local_assignments_with_vjp_fwd,
    _collapse_local_assignments_with_vjp_bwd,
)


@partial(jax.custom_vjp, nondiff_argnums=(6,))
def _collapse_local_assignments_internode_with_vjp(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    num_recv_tokens: jax.Array,
    recv_capacity: int,
) -> jax.Array:
    return _collapse_local_assignments_internode_impl(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        assignment_destinations,
        local_group_sizes,
        num_recv_tokens,
        recv_capacity=recv_capacity,
    )


def _collapse_local_assignments_internode_with_vjp_fwd(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    assignment_destinations: jax.Array,
    local_group_sizes: jax.Array,
    num_recv_tokens: jax.Array,
    recv_capacity: int,
):
    output = _collapse_local_assignments_internode_impl(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        assignment_destinations,
        local_group_sizes,
        num_recv_tokens,
        recv_capacity=recv_capacity,
    )
    return output, (out_dispatch, assignment_weights, recv_token_indices, local_group_sizes)


def _collapse_local_assignments_internode_with_vjp_bwd(recv_capacity: int, residuals, cotangent):
    out_dispatch, assignment_weights, recv_token_indices, local_group_sizes = residuals
    grad_recv_out = _materialize_cotangent(
        cotangent,
        dtype=jnp.bfloat16,
        shape=(recv_capacity, out_dispatch.shape[1]),
    )
    grad_out_dispatch, grad_assignment_weights = _collapse_local_assignments_internode_bwd_impl(
        grad_recv_out,
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        local_group_sizes,
    )
    return grad_out_dispatch, grad_assignment_weights, None, None, None, None


_collapse_local_assignments_internode_with_vjp.defvjp(
    _collapse_local_assignments_internode_with_vjp_fwd,
    _collapse_local_assignments_internode_with_vjp_bwd,
)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9))
def _dispatch_intranode_with_vjp(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    num_experts: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int | None,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
    return _dispatch_intranode_impl(
        x,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts=num_experts,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
        max_recv_tokens=max_recv_tokens,
    )


def _dispatch_intranode_with_vjp_fwd(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    num_experts: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int | None,
):
    outputs = _dispatch_intranode_impl(
        x,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts=num_experts,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
        max_recv_tokens=max_recv_tokens,
    )
    (
        recv_x,
        _recv_topk_idx,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        _,
        recv_channel_prefix_matrix,
        send_head,
        _,
        num_recv_tokens,
    ) = outputs
    residuals = (
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
    )
    return outputs, residuals


def _dispatch_intranode_with_vjp_bwd(
    num_experts: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int | None,
    residuals,
    cotangents,
):
    (
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
    ) = residuals
    grad_recv_x = _materialize_cotangent(cotangents[0], dtype=recv_x.dtype, reference=recv_x)
    grad_recv_topk_weights = _materialize_cotangent(
        cotangents[2],
        dtype=recv_topk_weights.dtype,
        reference=recv_topk_weights,
    )
    grad_x, grad_topk_weights = _combine_intranode_impl(
        grad_recv_x,
        grad_recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
    )
    return grad_x, None, grad_topk_weights, None, None, None


_dispatch_intranode_with_vjp.defvjp(
    _dispatch_intranode_with_vjp_fwd,
    _dispatch_intranode_with_vjp_bwd,
)


def _slice_assignment_dispatch_outputs(outputs, assignment_capacity: int | None):
    if assignment_capacity is None:
        return outputs
    (
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        local_group_sizes,
        num_recv_tokens,
        x_dispatch,
        assignment_weights,
        recv_token_indices,
        recv_assignment_indices,
        assignment_destinations,
    ) = outputs
    return (
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        local_group_sizes,
        num_recv_tokens,
        x_dispatch[:assignment_capacity],
        assignment_weights[:assignment_capacity],
        recv_token_indices[:assignment_capacity],
        recv_assignment_indices[:assignment_capacity],
        assignment_destinations,
    )


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _dispatch_intranode_with_assignments_with_vjp(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    num_experts: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int | None,
    assignment_capacity: int | None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    return _slice_assignment_dispatch_outputs(
        _dispatch_intranode_with_assignments_impl(
            x,
            topk_idx,
            topk_weights,
            num_tokens_per_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            num_experts=num_experts,
            dispatch_config=dispatch_config,
            combine_config=combine_config,
            max_recv_tokens=max_recv_tokens,
        ),
        assignment_capacity,
    )


def _dispatch_intranode_with_assignments_with_vjp_fwd(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    num_experts: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int | None,
    assignment_capacity: int | None,
):
    outputs = _slice_assignment_dispatch_outputs(
        _dispatch_intranode_with_assignments_impl(
            x,
            topk_idx,
            topk_weights,
            num_tokens_per_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            num_experts=num_experts,
            dispatch_config=dispatch_config,
            combine_config=combine_config,
            max_recv_tokens=max_recv_tokens,
        ),
        assignment_capacity,
    )
    (
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        _,
        recv_channel_prefix_matrix,
        send_head,
        _local_group_sizes,
        num_recv_tokens,
        _,
        _,
        _recv_token_indices,
        _recv_assignment_indices,
        assignment_destinations,
    ) = outputs
    residuals = (
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        assignment_destinations,
    )
    return outputs, residuals


def _dispatch_intranode_with_assignments_with_vjp_bwd(
    num_experts: int,
    dispatch_config: IntranodeConfig | None,
    combine_config: IntranodeConfig | None,
    max_recv_tokens: int | None,
    assignment_capacity: int | None,
    residuals,
    cotangents,
):
    del num_experts, dispatch_config, combine_config, max_recv_tokens
    (
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        assignment_destinations,
    ) = residuals
    grad_assignment_capacity = assignment_capacity or assignment_destinations.shape[0]

    grad_recv_x = _materialize_cotangent(cotangents[0], dtype=recv_x.dtype, reference=recv_x)
    grad_recv_topk_weights = _materialize_cotangent(
        cotangents[1],
        dtype=recv_topk_weights.dtype,
        reference=recv_topk_weights,
    )

    grad_x_dispatch = _materialize_cotangent(
        cotangents[9],
        dtype=recv_x.dtype,
        shape=(grad_assignment_capacity, recv_x.shape[1]),
    )
    grad_assignment_weights = _materialize_cotangent(
        cotangents[10],
        dtype=jnp.float32,
        shape=(grad_assignment_capacity,),
    )

    grad_recv_x_from_assignments, grad_recv_topk_weights_from_assignments = _assignment_gradients_impl(
        grad_x_dispatch=grad_x_dispatch,
        grad_assignment_weights=grad_assignment_weights,
        assignment_destinations=assignment_destinations,
        num_recv_tokens=num_recv_tokens,
        recv_x_shape=recv_x.shape,
        recv_topk_weights_shape=recv_topk_weights.shape,
    )
    grad_recv_x += grad_recv_x_from_assignments
    grad_recv_topk_weights += grad_recv_topk_weights_from_assignments

    grad_x, grad_topk_weights = _combine_intranode_impl(
        grad_recv_x,
        grad_recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
    )
    return grad_x, None, grad_topk_weights, None, None, None


_dispatch_intranode_with_assignments_with_vjp.defvjp(
    _dispatch_intranode_with_assignments_with_vjp_fwd,
    _dispatch_intranode_with_assignments_with_vjp_bwd,
)


@jax.custom_vjp
def _combine_intranode_with_vjp(
    recv_x: jax.Array,
    recv_topk_weights: jax.Array,
    recv_src_idx: jax.Array,
    rank_prefix_matrix: jax.Array,
    dispatch_channel_prefix_matrix: jax.Array,
    recv_channel_prefix_matrix: jax.Array,
    send_head: jax.Array,
    num_recv_tokens: jax.Array,
    is_token_in_rank: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _combine_intranode_impl(
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
    )


def _combine_intranode_with_vjp_fwd(
    recv_x: jax.Array,
    recv_topk_weights: jax.Array,
    recv_src_idx: jax.Array,
    rank_prefix_matrix: jax.Array,
    dispatch_channel_prefix_matrix: jax.Array,
    recv_channel_prefix_matrix: jax.Array,
    send_head: jax.Array,
    num_recv_tokens: jax.Array,
    is_token_in_rank: jax.Array,
):
    outputs = _combine_intranode_impl(
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
    )
    residuals = (
        recv_x,
        recv_topk_weights,
        rank_prefix_matrix,
        dispatch_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )
    return outputs, residuals


def _combine_intranode_with_vjp_bwd(residuals, cotangents):
    (
        recv_x,
        recv_topk_weights,
        rank_prefix_matrix,
        dispatch_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    ) = residuals
    grad_combined_x = _materialize_cotangent(
        cotangents[0],
        dtype=recv_x.dtype,
        shape=(send_head.shape[0], recv_x.shape[1]),
    )
    grad_recv_x = _dispatch_intranode_cached_impl(
        grad_combined_x,
        is_token_in_rank,
        rank_prefix_matrix,
        dispatch_channel_prefix_matrix,
        num_recv_tokens,
        dispatch_config=None,
        combine_config=None,
        max_recv_tokens=recv_x.shape[0],
    )
    return (
        grad_recv_x,
        jnp.zeros_like(recv_topk_weights),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_combine_intranode_with_vjp.defvjp(
    _combine_intranode_with_vjp_fwd,
    _combine_intranode_with_vjp_bwd,
)


def deepep_dispatch_intranode(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    *,
    num_experts: int,
    dispatch_config: IntranodeConfig | None = None,
    combine_config: IntranodeConfig | None = None,
    max_recv_tokens: int | None = None,
) -> DeepEPDispatch:
    return DeepEPDispatch(
        *_dispatch_intranode_with_vjp(
            x,
            topk_idx,
            topk_weights,
            num_tokens_per_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            num_experts,
            dispatch_config,
            combine_config,
            max_recv_tokens,
        )
    )


def deepep_dispatch_intranode_with_assignments(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    num_tokens_per_rank: jax.Array,
    num_tokens_per_expert: jax.Array,
    is_token_in_rank: jax.Array,
    *,
    num_experts: int,
    dispatch_config: IntranodeConfig | None = None,
    combine_config: IntranodeConfig | None = None,
    max_recv_tokens: int | None = None,
    assignment_capacity: int | None = None,
) -> DeepEPDispatchWithAssignments:
    (
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        local_group_sizes,
        num_recv_tokens,
        x_dispatch,
        assignment_weights,
        recv_token_indices,
        _recv_assignment_indices,
        assignment_destinations,
    ) = _dispatch_intranode_with_assignments_with_vjp(
        x,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts,
        dispatch_config,
        combine_config,
        max_recv_tokens,
        assignment_capacity,
    )
    return DeepEPDispatchWithAssignments(
        recv_x=recv_x,
        recv_topk_weights=recv_topk_weights,
        recv_src_idx=recv_src_idx,
        rank_prefix_matrix=rank_prefix_matrix,
        channel_prefix_matrix=channel_prefix_matrix,
        recv_channel_prefix_matrix=recv_channel_prefix_matrix,
        send_head=send_head,
        local_group_sizes=local_group_sizes,
        num_recv_tokens=num_recv_tokens,
        x_dispatch=x_dispatch,
        assignment_weights=assignment_weights,
        recv_token_indices=recv_token_indices,
        assignment_destinations=assignment_destinations,
    )


def deepep_combine_intranode(
    recv_x: jax.Array,
    recv_topk_weights: jax.Array,
    recv_src_idx: jax.Array,
    rank_prefix_matrix: jax.Array,
    dispatch_channel_prefix_matrix: jax.Array,
    recv_channel_prefix_matrix: jax.Array,
    send_head: jax.Array,
    num_recv_tokens: jax.Array,
    is_token_in_rank: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _combine_intranode_with_vjp(
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        dispatch_channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )
