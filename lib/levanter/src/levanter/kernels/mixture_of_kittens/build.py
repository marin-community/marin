# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Native build for Marin's pinned MoK-like CUDA adapter."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import jax
import jaxlib

from levanter.kernels.mixture_of_kittens.source import (
    MokLikeBuildConfig,
    MOK_KNOWN_GOOD_COMMIT,
    THUNDERKITTENS_KNOWN_GOOD_COMMIT,
    mok_cache_root,
    mok_cuda_arch_flag,
    mok_source_root,
)


_BUILD_SCHEMA = "mok_forward_backward_ffi_v12"
_CUDA_DISTRIBUTIONS = (
    "nvidia-cuda-runtime",
    "nvidia-cuda-nvcc",
    "nvidia-cuda-crt",
    "nvidia-cuda-cccl",
    "nvidia-cuda-runtime-cu13",
    "nvidia-cuda-nvcc-cu13",
)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _jaxlib_include_dir() -> Path:
    return Path(jaxlib.__file__).resolve().parent / "include"


def _cuda_include_dirs() -> tuple[Path, ...]:
    include_dirs: list[Path] = []
    for distribution_name in _CUDA_DISTRIBUTIONS:
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
    backward_device_kernel = next(
        index for index, line in enumerate(mok_lines) if "dispatch_mlp_swiglu_combine_bwd_kernel(" in line
    )
    while backward_device_kernel > 0 and "template <bool" not in mok_lines[backward_device_kernel]:
        backward_device_kernel -= 1
    backward_host_wrapper = next(
        index
        for index, line in enumerate(mok_lines[backward_device_kernel:], start=backward_device_kernel)
        if "static __host__" in line
    )
    mok_text = "".join(mok_lines[:first_host_wrapper] + mok_lines[backward_device_kernel:backward_host_wrapper])
    mok_text = mok_text.replace('#include "pyutils/torchutils.cuh"\n', "")
    mok_text = mok_text.replace("#include <ATen/ops/empty.h>\n", "")
    mok_text = mok_text.replace("#include <ATen/ops/empty_like.h>\n", "")
    mok_text = mok_text.replace("#include <ATen/ops/zeros.h>\n", "")
    source_edits = (
        (
            """static __device__ __forceinline__ void barrier_arrive(const index_gl &counter, int index, int increment = 1) {
    asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&counter[{index}]), "r"(increment) : "memory");
}""",
            """static __device__ __forceinline__ void barrier_arrive(const index_gl &counter, int index, int increment = 1) {
    asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&counter[{index}]), "r"(increment) : "memory");
}

static __device__ __forceinline__ unsigned long long system_generation_load(const uint64_t *counter) {
    unsigned long long value;
    asm volatile("{ld.acquire.sys.global.u64 %0, [%1];}" : "=l"(value) : "l"(counter) : "memory");
    return value;
}

static __device__ __forceinline__ unsigned long long system_generation_wait(
    const uint64_t *counter,
    unsigned long long target
) {
    unsigned long long observed = system_generation_load(counter);
    while (observed < target) {
        __nanosleep(64);
        observed = system_generation_load(counter);
    }
    return observed;
}""",
        ),
        (
            """    activation_bf16_pgl x_routed_send_buffer; // (num_local_tokens, H)
    activation_bf16_pgl y_routed_recv_buffer; // (num_local_tokens * topk, H)""",
            """    activation_bf16_pgl x_routed_send_buffer; // (num_local_tokens, H)
    activation_bf16_pgl y_routed_recv_buffer; // (num_local_tokens * topk, H)
    const uint64_t *peer_input_ready;
    const uint64_t *peer_destination_ready;
    const uint64_t *input_generation;
    uint64_t *peer_ready_wait_counter;
    uint64_t *generation_mismatch_counter;
    uint64_t *peer_wait_events;
    uint64_t *peer_wait_cycles;
    uint64_t *peer_wait_max_cycles;""",
        ),
        (
            """    activation_bf16_pgl d_x_routed_buffer;                // (num_local_tokens * topk, H)
    router_weight_pgl router_weight_buffer;               // (num_local_tokens, topk)
    router_weight_pgl d_router_weight_buffer;             // (num_local_tokens, topk)""",
            """    activation_bf16_pgl d_x_routed_buffer;                // (num_local_tokens * topk, H)
    router_weight_pgl router_weight_buffer;               // (num_local_tokens, topk)
    router_weight_pgl d_router_weight_buffer;             // (num_local_tokens, topk)
    const uint64_t *peer_input_ready;
    const uint64_t *peer_destination_ready;
    const uint64_t *input_generation;
    uint64_t *peer_ready_wait_counter;
    uint64_t *generation_mismatch_counter;
    uint64_t *peer_wait_events;
    uint64_t *peer_wait_cycles;
    uint64_t *peer_wait_max_cycles;""",
        ),
        (
            """    const int previous_macrobatch_idx,
    const int buffer_ready_required_count
) {""",
            """    const int previous_macrobatch_idx,
    const int buffer_ready_required_count,
    const uint64_t *peer_input_ready = nullptr,
    const uint64_t *input_generation = nullptr,
    uint64_t *peer_ready_wait_counter = nullptr,
    uint64_t *generation_mismatch_counter = nullptr,
    uint64_t *peer_wait_events = nullptr,
    uint64_t *peer_wait_cycles = nullptr,
    uint64_t *peer_wait_max_cycles = nullptr
) {""",
        ),
        (
            """    const int minibatch_size,
    const int macrobatch_idx,
    const int task_idx,
    const uint64_t smem_base_addr
) {
    auto &token_chunks = *reinterpret_cast<bf16 (*)[config::COMBINE_PIPE_DEPTH][config::COMBINE_Mb][config::COMBINE_Nb]>(smem_base_addr);""",
            """    const int minibatch_size,
    const int macrobatch_idx,
    const int task_idx,
    const uint64_t smem_base_addr,
    const uint64_t *peer_input_ready = nullptr,
    const uint64_t *input_generation = nullptr,
    uint64_t *peer_ready_wait_counter = nullptr,
    uint64_t *generation_mismatch_counter = nullptr,
    uint64_t *peer_wait_events = nullptr,
    uint64_t *peer_wait_cycles = nullptr,
    uint64_t *peer_wait_max_cycles = nullptr
) {
    auto &token_chunks = *reinterpret_cast<bf16 (*)[config::COMBINE_PIPE_DEPTH][config::COMBINE_Mb][config::COMBINE_Nb]>(smem_base_addr);""",
        ),
        (
            """    // Store each tile out as its loads arrive
    #pragma unroll
    for (int stage = 0; stage < config::COMBINE_PIPE_DEPTH; ++stage) {""",
            """    // Remote destinations must finish clearing their peer-written buffers before this CTA stores.
    __shared__ uint32_t destination_ready_mask;
    if (peer_input_ready != nullptr) {
        if (tid == 0) destination_ready_mask = 0;
        __syncthreads();
        #pragma unroll
        for (int stage = 0; stage < config::COMBINE_PIPE_DEPTH; ++stage)
            if (peer_rank[stage] >= 0) atomicOr(&destination_ready_mask, 1U << peer_rank[stage]);
        __syncthreads();
        if (tid < NUM_DEVICES && (destination_ready_mask & (1U << tid)) != 0) {
            const unsigned long long target = system_generation_load(input_generation);
            const unsigned long long initial = system_generation_load(peer_input_ready + tid);
            unsigned long long observed = initial;
            if (initial < target) {
                const unsigned long long wait_start = clock64();
                observed = system_generation_wait(peer_input_ready + tid, target);
                const unsigned long long wait_cycles = clock64() - wait_start;
                if (peer_ready_wait_counter != nullptr)
                    atomicAdd(reinterpret_cast<unsigned long long *>(peer_ready_wait_counter), 1ULL);
                if (peer_wait_events != nullptr)
                    atomicAdd(reinterpret_cast<unsigned long long *>(peer_wait_events + tid), 1ULL);
                if (peer_wait_cycles != nullptr)
                    atomicAdd(reinterpret_cast<unsigned long long *>(peer_wait_cycles + tid), wait_cycles);
                if (peer_wait_max_cycles != nullptr)
                    atomicMax(reinterpret_cast<unsigned long long *>(peer_wait_max_cycles + tid), wait_cycles);
            }
            if (observed > target && generation_mismatch_counter != nullptr)
                atomicAdd(reinterpret_cast<unsigned long long *>(generation_mismatch_counter), 1ULL);
        }
        __syncthreads();
    }

    // Store each tile out as its loads arrive
    #pragma unroll
    for (int stage = 0; stage < config::COMBINE_PIPE_DEPTH; ++stage) {""",
        ),
        (
            """        const int peer_rank = schedule_peer_rank[{macrobatch_offset + row}];
        const int peer_token_idx = schedule_peer_token_idx[{macrobatch_offset + row}];
        router_weights.raw_ptr[row] = peer_rank >= 0 ? peer_buf[peer_rank][peer_token_idx] : 0.0f;""",
            """        const int peer_rank = schedule_peer_rank[{macrobatch_offset + row}];
        const int peer_token_idx = schedule_peer_token_idx[{macrobatch_offset + row}];
        if (peer_rank >= 0 && peer_input_ready != nullptr) {
            const unsigned long long target = system_generation_load(input_generation);
            const unsigned long long initial = system_generation_load(peer_input_ready + peer_rank);
            unsigned long long observed = initial;
            if (initial < target) {
                const unsigned long long wait_start = clock64();
                observed = system_generation_wait(peer_input_ready + peer_rank, target);
                const unsigned long long wait_cycles = clock64() - wait_start;
                if (peer_ready_wait_counter != nullptr)
                    atomicAdd(reinterpret_cast<unsigned long long *>(peer_ready_wait_counter), 1ULL);
                if (peer_wait_events != nullptr)
                    atomicAdd(reinterpret_cast<unsigned long long *>(peer_wait_events + peer_rank), 1ULL);
                if (peer_wait_cycles != nullptr)
                    atomicAdd(reinterpret_cast<unsigned long long *>(peer_wait_cycles + peer_rank), wait_cycles);
                if (peer_wait_max_cycles != nullptr)
                    atomicMax(reinterpret_cast<unsigned long long *>(peer_wait_max_cycles + peer_rank), wait_cycles);
            }
            if (observed > target && generation_mismatch_counter != nullptr)
                atomicAdd(reinterpret_cast<unsigned long long *>(generation_mismatch_counter), 1ULL);
        }
        router_weights.raw_ptr[row] = peer_rank >= 0 ? peer_buf[peer_rank][peer_token_idx] : 0.0f;""",
        ),
        (
            """    const int previous_macrobatch_idx,
    const int buffer_ready_required_count,
    const uint64_t smem_base_addr
) {""",
            """    const int previous_macrobatch_idx,
    const int buffer_ready_required_count,
    const uint64_t smem_base_addr,
    const uint64_t *peer_input_ready = nullptr,
    const uint64_t *input_generation = nullptr,
    uint64_t *peer_ready_wait_counter = nullptr,
    uint64_t *generation_mismatch_counter = nullptr,
    uint64_t *peer_wait_events = nullptr,
    uint64_t *peer_wait_cycles = nullptr,
    uint64_t *peer_wait_max_cycles = nullptr
) {""",
        ),
        (
            """    const int peer_rank = is_worker ? schedule_peer_rank[{macrobatch_offset + row_idx + tid}] : -1;
    const int peer_token_idx = is_worker ? schedule_peer_token_idx[{macrobatch_offset + row_idx + tid}] : -1;
    const int num_valid = __syncthreads_count(peer_rank >= 0);

    if (tid == 0) {""",
            """    const int peer_rank = is_worker ? schedule_peer_rank[{macrobatch_offset + row_idx + tid}] : -1;
    const int peer_token_idx = is_worker ? schedule_peer_token_idx[{macrobatch_offset + row_idx + tid}] : -1;
    const int num_valid = __syncthreads_count(peer_rank >= 0);

    __shared__ uint32_t peer_ready_mask;
    if (peer_input_ready != nullptr) {
        if (tid == 0) peer_ready_mask = 0;
        __syncthreads();
        if (peer_rank >= 0) atomicOr(&peer_ready_mask, 1U << peer_rank);
        __syncthreads();
        if (tid < NUM_DEVICES && (peer_ready_mask & (1U << tid)) != 0) {
            const unsigned long long target = system_generation_load(input_generation);
            const unsigned long long initial = system_generation_load(peer_input_ready + tid);
            unsigned long long observed = initial;
            if (initial < target) {
                const unsigned long long wait_start = clock64();
                observed = system_generation_wait(peer_input_ready + tid, target);
                const unsigned long long wait_cycles = clock64() - wait_start;
                if (peer_ready_wait_counter != nullptr)
                    atomicAdd(reinterpret_cast<unsigned long long *>(peer_ready_wait_counter), 1ULL);
                if (peer_wait_events != nullptr)
                    atomicAdd(reinterpret_cast<unsigned long long *>(peer_wait_events + tid), 1ULL);
                if (peer_wait_cycles != nullptr)
                    atomicAdd(reinterpret_cast<unsigned long long *>(peer_wait_cycles + tid), wait_cycles);
                if (peer_wait_max_cycles != nullptr)
                    atomicMax(reinterpret_cast<unsigned long long *>(peer_wait_max_cycles + tid), wait_cycles);
            }
            if (observed > target && generation_mismatch_counter != nullptr)
                atomicAdd(reinterpret_cast<unsigned long long *>(generation_mismatch_counter), 1ULL);
        }
        __syncthreads();
    }

    if (tid == 0) {""",
        ),
        (
            """                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, g.topk,
                                     macrobatch_idx + 1, 0, smem_base_addr);""",
            """                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, g.topk,
                                     macrobatch_idx + 1, 0, smem_base_addr,
                                     g.peer_input_ready, g.input_generation, g.peer_ready_wait_counter,
                                     g.generation_mismatch_counter, g.peer_wait_events, g.peer_wait_cycles,
                                     g.peer_wait_max_cycles);""",
        ),
        (
            """                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, g.topk,
                                     -1, 0, smem_base_addr);
        };
        auto reverse_dispatch""",
            """                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, g.topk,
                                     -1, 0, smem_base_addr,
                                     g.peer_input_ready, g.input_generation, g.peer_ready_wait_counter,
                                     g.generation_mismatch_counter, g.peer_wait_events, g.peer_wait_cycles,
                                     g.peer_wait_max_cycles);
        };
        auto reverse_dispatch""",
        ),
        (
            """                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, g.topk,
                                     -1, 0, smem_base_addr);
        };
        preload_router_weights_kernel""",
            """                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, g.topk,
                                     -1, 0, smem_base_addr,
                                     g.peer_input_ready, g.input_generation, g.peer_ready_wait_counter,
                                     g.generation_mismatch_counter, g.peer_wait_events, g.peer_wait_cycles,
                                     g.peer_wait_max_cycles);
        };
        preload_router_weights_kernel""",
        ),
        (
            """                                      nullptr, g.router_weights_ready,
                                      num_tokens, macrobatch_size, 0, comm_cta_idx, g.num_comm_sms, -1, 0);""",
            """                                      nullptr, g.router_weights_ready,
                                      num_tokens, macrobatch_size, 0, comm_cta_idx, g.num_comm_sms, -1, 0,
                                      g.peer_input_ready, g.input_generation, g.peer_ready_wait_counter,
                                      g.generation_mismatch_counter, g.peer_wait_events, g.peer_wait_cycles,
                                      g.peer_wait_max_cycles);""",
        ),
        (
            """                                              num_tokens, macrobatch_size, macrobatch_idx + 1, comm_cta_idx, g.num_comm_sms,
                                              macrobatch_idx, routed_buffers_done_required_count_of(macrobatch_idx));""",
            """                                              num_tokens, macrobatch_size, macrobatch_idx + 1, comm_cta_idx, g.num_comm_sms,
                                              macrobatch_idx, routed_buffers_done_required_count_of(macrobatch_idx),
                                              g.peer_input_ready, g.input_generation, g.peer_ready_wait_counter,
                                              g.generation_mismatch_counter, g.peer_wait_events, g.peer_wait_cycles,
                                              g.peer_wait_max_cycles);""",
        ),
        (
            """                           combine_inputs_arrived, combine_bitfield,
                           num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, smem_base_addr);
        };
        auto replay_dispatch""",
            """                           combine_inputs_arrived, combine_bitfield,
                           num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, smem_base_addr,
                           g.peer_destination_ready, g.input_generation, g.peer_ready_wait_counter,
                           g.generation_mismatch_counter, g.peer_wait_events, g.peer_wait_cycles,
                           g.peer_wait_max_cycles);
        };
        auto replay_dispatch""",
        ),
        (
            """    // The next task on this CTA reuses token_chunks; make sure outgoing stores are done reading shared memory
    tma::store_async_read_wait();
    __syncthreads();""",
            """    // The destination completion generation is published after this kernel returns.
    // Wait for remote stores to complete, not only for their shared-memory reads.
    tma::store_async_wait();
    __syncthreads();""",
        ),
        (
            "using d_weight_bf16_gl = gl<bf16, 1, -1, -1, -1, mlp_bf16_d_tile>;",
            """using d_weight_bf16_gl = gl<bf16, 1, -1, -1, -1, mlp_bf16_d_tile>;
using mlp_f32_d_tile = st_fl<config::MLP_Mb / 2, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH>;
using d_routed_weight_f32_gl = gl<float, 1, -1, -1, -1, mlp_f32_d_tile>;""",
        ),
        ("d_weight_bf16_gl d_w_routed_gate;", "d_routed_weight_f32_gl d_w_routed_gate;"),
        ("d_weight_bf16_gl d_w_routed_up;", "d_routed_weight_f32_gl d_w_routed_up;"),
        ("d_weight_bf16_gl d_w_routed_down;", "d_routed_weight_f32_gl d_w_routed_down;"),
        (
            "const std::conditional_t<IS_WGRAD, d_weight_bf16_gl, epi_bf16_gl> &d_gmem,",
            """const std::conditional_t<IS_WGRAD,
        std::conditional_t<IS_SHARED, d_weight_bf16_gl, d_routed_weight_f32_gl>, epi_bf16_gl> &d_gmem,""",
        ),
        (
            """        auto store_bf16 = [&]() {
            rt_bf<config::MLP_Mb / 8, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH> d_reg[config::MLP_EPI_PIPE_DEPTH];""",
            """        auto store_output = [&]() {
            if constexpr (IS_WGRAD && !IS_SHARED) {
                auto &d_f32_smem = *reinterpret_cast<mlp_f32_d_tile *>(&d_bf16_smem[0]);
                #pragma unroll
                for (int i = 0; i < config::MLP_EPI_PIPE_DEPTH; ++i) {
                    rt_fl<config::MLP_Mb / 8, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH> d_reg;
                    warpgroup::load_async(d_reg, d_tt.template subtile<tt<float, config::MLP_Mb / 2, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH>>(0, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH * i));
                    tensor_load_wait();
                    warpgroup::sync(1);
                    warpgroup::store(d_f32_smem, d_reg);
                    warpgroup::sync(1);
                    const coord<mlp_f32_d_tile> output_coord = {tile_coord.z, 2 * tile_coord.x + cta_rank, config::MLP_EPI_PIPE_DEPTH * tile_coord.y + i};
                    if (is_first_wgrad_contribution)
                        warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_f32_smem, output_coord);
                    else
                        // Macrobatches are serialized by routed_buffers_done, so FP32 additions are deterministic.
                        warpgroup::tma::store_add_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_f32_smem, output_coord);
                    warpgroup::tma::store_async_read_wait();
                }
                warpgroup::tma::cluster::arrive(gemm_outputs_finished, 0);
            } else {
            rt_bf<config::MLP_Mb / 8, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH> d_reg[config::MLP_EPI_PIPE_DEPTH];""",
        ),
        (
            """            warpgroup::tma::store_async_read_wait();
        };
        if constexpr (USE_ROUTED_MXFP8) {""",
            """            warpgroup::tma::store_async_read_wait();
            }
        };
        if constexpr (USE_ROUTED_MXFP8) {""",
        ),
        ("store_bf16();", "store_output();"),
    )
    for old, new in source_edits:
        if old not in mok_text:
            raise RuntimeError("The pinned Mixture-of-Kittens source changed at a Marin kernel edit")
        mok_text = mok_text.replace(old, new)
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


def _build_path(build_config: MokLikeBuildConfig) -> tuple[Path, Path]:
    source_root = mok_source_root(build_config)
    prepared = _prepared_source_bytes(source_root)
    key = hashlib.sha256()
    key.update(_ffi_source().read_bytes())
    for data in prepared:
        key.update(data)
    toolchain_versions = tuple(
        (name, version) for name in _CUDA_DISTRIBUTIONS if (version := _distribution_version(name)) is not None
    )
    key.update(
        repr(
            (
                _BUILD_SCHEMA,
                MOK_KNOWN_GOOD_COMMIT,
                THUNDERKITTENS_KNOWN_GOOD_COMMIT,
                jax.__version__,
                jaxlib.__version__,
                toolchain_versions,
            )
        ).encode()
    )
    key.update(mok_cuda_arch_flag(build_config).encode())
    for include_dir in _cuda_include_dirs():
        key.update(str(include_dir).encode())
    digest = key.hexdigest()[:16]
    build_dir = mok_cache_root(build_config, "mok_forward_ffi") / digest
    return build_dir, build_dir / "libmok_forward_ffi.so"


def _write_prepared_sources(build_dir: Path, build_config: MokLikeBuildConfig) -> None:
    source_root = mok_source_root(build_config)
    mok_text, mxfp8_text, utils_text = _prepared_source_bytes(source_root)
    generated = build_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "mok_megakernel.cuh").write_bytes(mok_text)
    (generated / "mxfp8.cuh").write_bytes(mxfp8_text)
    (generated / "utils.cuh").write_bytes(utils_text)


def build_native_library(build_config: MokLikeBuildConfig) -> Path:
    """Build the pinned adapter and return its cached shared-library path."""

    build_dir, library_path = _build_path(build_config)
    build_dir.mkdir(parents=True, exist_ok=True)
    with (build_dir / ".build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if library_path.is_file():
            return library_path
        _write_prepared_sources(build_dir, build_config)
        source_root = mok_source_root(build_config)
        toolchain_root = _cuda_toolchain_root(build_dir)
        temporary_library = library_path.with_name(f"{library_path.name}.{uuid.uuid4().hex}.tmp")
        command = [
            str(toolchain_root / "bin" / "nvcc"),
            str(_ffi_source()),
            "-o",
            str(temporary_library),
            "-std=c++20",
            "-shared",
            "-Xcompiler=-fPIC",
            "-Xcompiler=-pthread",
            "--cudart=shared",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-forward-unknown-to-host-compiler",
            "-ftemplate-backtrace-limit=0",
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-DNDEBUG",
            f"-DKITTENS_{build_config.cuda_arch.replace('sm_', 'SM', 1).replace('a', '')}",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            mok_cuda_arch_flag(build_config),
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
        try:
            subprocess.run(command, check=True)
            os.replace(temporary_library, library_path)
        finally:
            temporary_library.unlink(missing_ok=True)
    return library_path
