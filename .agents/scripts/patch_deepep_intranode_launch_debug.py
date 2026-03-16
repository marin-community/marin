#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Patch DeepEP intranode dispatch to emit env-gated launch debug records."""

from __future__ import annotations

import argparse
from pathlib import Path

INCLUDE_NEEDLE = '#include "utils.cuh"\n'
INCLUDE_REPLACEMENT = '#include "utils.cuh"\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n'
HELPER_NEEDLE = "namespace intranode {\n"
HELPER_BLOCK = """
namespace intranode {

// LEVANTER_DEEPEP_LAUNCH_DEBUG_BEGIN
namespace {

bool levanter_launch_debug_enabled() {
    const char* raw = std::getenv("LEVANTER_DEEPEP_LAUNCH_DEBUG");
    return raw != nullptr && raw[0] != '\\0' && raw[0] != '0';
}

const char* levanter_launch_debug_label() {
    const char* raw = std::getenv("LEVANTER_DEEPEP_LAUNCH_DEBUG_LABEL");
    return (raw != nullptr && raw[0] != '\\0') ? raw : "unknown";
}

template <typename KernelT>
void levanter_log_dispatch_launch(KernelT kernel,
                                  int rank, int num_ranks, cudaStream_t stream,
                                  int num_sms, int num_tokens, int hidden_int4, int num_topk, int num_experts,
                                  int num_max_send_tokens, int num_recv_buffer_tokens,
                                  int num_threads, int num_tma_bytes_per_warp) {
    if (!levanter_launch_debug_enabled() || rank != 0)
        return;

    int device = -1;
    cudaError_t device_status = cudaGetDevice(&device);
    cudaFuncAttributes attrs{};
    cudaError_t attr_status = cudaFuncGetAttributes(&attrs, kernel);
#ifndef DISABLE_SM90_FEATURES
    const int requested_smem_bytes = num_tma_bytes_per_warp * (num_threads / 32);
    cudaError_t set_attr_status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, requested_smem_bytes);
    const int cooperative = 1;
    const int cluster_dim_x = (num_sms % 2 == 0 ? 2 : 1);
    const char* compile_mode = "sm90_tma";
#else
    const int requested_smem_bytes = 0;
    cudaError_t set_attr_status = cudaSuccess;
    const int cooperative = 0;
    const int cluster_dim_x = 1;
    const char* compile_mode = "fallback";
#endif

    fprintf(
        stderr,
        "DEEPEP_LAUNCH_DEBUG {"
        "\\"label\\":\\"%s\\","
        "\\"kernel\\":\\"dispatch\\","
        "\\"compile_mode\\":\\"%s\\","
        "\\"rank\\":%d,"
        "\\"num_ranks\\":%d,"
        "\\"device\\":%d,"
        "\\"device_status_code\\":%d,"
        "\\"device_status\\":\\"%s\\","
        "\\"num_sms\\":%d,"
        "\\"num_threads\\":%d,"
        "\\"num_tma_bytes_per_warp\\":%d,"
        "\\"cluster_dim_x\\":%d,"
        "\\"cooperative\\":%d,"
        "\\"num_tokens\\":%d,"
        "\\"hidden_int4\\":%d,"
        "\\"num_topk\\":%d,"
        "\\"num_experts\\":%d,"
        "\\"num_max_send_tokens\\":%d,"
        "\\"num_recv_buffer_tokens\\":%d,"
        "\\"requested_smem_bytes\\":%d,"
        "\\"stream_ptr\\":\\"0x%llx\\","
        "\\"attr_status_code\\":%d,"
        "\\"attr_status\\":\\"%s\\","
        "\\"set_attr_status_code\\":%d,"
        "\\"set_attr_status\\":\\"%s\\","
        "\\"binary_version\\":%d,"
        "\\"ptx_version\\":%d,"
        "\\"max_dynamic_smem_bytes\\":%d,"
        "\\"shared_size_bytes\\":%llu,"
        "\\"local_size_bytes\\":%llu,"
        "\\"const_size_bytes\\":%llu,"
        "\\"num_regs\\":%d,"
        "\\"max_threads_per_block\\":%d,"
        "\\"preferred_shmem_carveout\\":%d"
        "}\\n",
        levanter_launch_debug_label(),
        compile_mode,
        rank,
        num_ranks,
        device,
        static_cast<int>(device_status),
        cudaGetErrorString(device_status),
        num_sms,
        num_threads,
        num_tma_bytes_per_warp,
        cluster_dim_x,
        cooperative,
        num_tokens,
        hidden_int4,
        num_topk,
        num_experts,
        num_max_send_tokens,
        num_recv_buffer_tokens,
        requested_smem_bytes,
        static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(stream)),
        static_cast<int>(attr_status),
        cudaGetErrorString(attr_status),
        static_cast<int>(set_attr_status),
        cudaGetErrorString(set_attr_status),
        attrs.binaryVersion,
        attrs.ptxVersion,
        attrs.maxDynamicSharedSizeBytes,
        static_cast<unsigned long long>(attrs.sharedSizeBytes),
        static_cast<unsigned long long>(attrs.localSizeBytes),
        static_cast<unsigned long long>(attrs.constSizeBytes),
        attrs.numRegs,
        attrs.maxThreadsPerBlock,
        attrs.preferredShmemCarveout);
}

}  // namespace
// LEVANTER_DEEPEP_LAUNCH_DEBUG_END
"""
MACRO_NEEDLE = """    auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \\
    SET_SHARED_MEMORY_FOR_TMA(kernel); \\"""
MACRO_REPLACEMENT = """    auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \\
    levanter_log_dispatch_launch( \\
        kernel, rank, num_ranks, stream, num_sms, num_tokens, hidden_int4, num_topk, num_experts, \\
        num_max_send_tokens, num_recv_buffer_tokens, kNumThreads, kNumTMABytesPerWarp); \\
    SET_SHARED_MEMORY_FOR_TMA(kernel); \\"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Path to the unpacked DeepEP checkout")
    return parser.parse_args()


def _replace_once(text: str, needle: str, replacement: str, *, description: str) -> str:
    if needle not in text:
        raise RuntimeError(f"Could not find {description} in intranode.cu")
    return text.replace(needle, replacement, 1)


def main() -> int:
    args = _parse_args()
    intranode = args.root / "csrc" / "kernels" / "intranode.cu"
    source = intranode.read_text()
    if "LEVANTER_DEEPEP_LAUNCH_DEBUG_BEGIN" in source:
        print(f"UNCHANGED {intranode}")
        return 0

    source = _replace_once(source, INCLUDE_NEEDLE, INCLUDE_REPLACEMENT, description="include insertion point")
    source = _replace_once(source, HELPER_NEEDLE, HELPER_BLOCK, description="namespace insertion point")
    source = _replace_once(source, MACRO_NEEDLE, MACRO_REPLACEMENT, description="dispatch launch macro")
    intranode.write_text(source)
    print(f"PATCHED {intranode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
