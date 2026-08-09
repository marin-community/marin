// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Torch-free runtime wrapper around a generic SM100 segmented contraction.
// The included primitive still owns its internal mbarrier sites. Shuttle owns
// only the generated synchronization ABI and the outer same-stream readiness.

#include <atomic>
#include <cstdint>
#include <string>

#include "mok_megakernel.cuh"
#include "xla/ffi/api/ffi.h"

#include "generated_event_schedule.inc"

namespace ffi = xla::ffi;

namespace tile_lifetime::segmented_contract_ffi {

using primitive = dispatch_mlp_swiglu_combiner<4, utils::RoutedPrecision::BF16>;
using config = primitive::config;

static_assert(config::CLUSTER_SIZE == shuttle_grouped_contract_event::kClusterCtas);
static_assert(config::MLP_LOAD_PIPE_DEPTH == shuttle_grouped_contract_event::kLoadPipelineStages);
static_assert(
    config::CLUSTER_SIZE * config::NUM_PRODUCERS
    == shuttle_grouped_contract_event::kOperandReadyLogicalCount
);
static_assert(config::NUM_CONSUMERS == shuttle_grouped_contract_event::kOperandReleaseLogicalCount);
static_assert(config::NUM_CONSUMERS == shuttle_grouped_contract_event::kOutputReadyLogicalCount);
static_assert(
    config::CLUSTER_SIZE * config::NUM_CONSUMERS
    == shuttle_grouped_contract_event::kOutputReleaseLogicalCount
);
static_assert(
    config::CLUSTER_SIZE * 2 * sizeof(primitive::mlp_bf16_tile)
    == shuttle_grouped_contract_event::kOperandTransactionBytes
);

std::atomic<int> relation_plan_call_count{0};
std::atomic<int> grouped_contract_call_count{0};

__global__ void pack_relation_segments(
    const __nv_bfloat16* source,
    const int32_t* counts,
    const int32_t* offsets,
    const int32_t* edge_sources,
    __nv_bfloat16* packed,
    int32_t* padded_counts,
    int group_count,
    int capacity,
    int reduction
) {
    const int group = static_cast<int>(blockIdx.x);
    if (group >= group_count) return;
    const int count = counts[group];
    int compact_group = 0;
    for (int prior = 0; prior < group; ++prior) compact_group += counts[prior] > 0;
    if (threadIdx.x == 0) padded_counts[group] = count > 0 ? capacity : 0;
    if (count == 0) return;
    for (int index = static_cast<int>(threadIdx.x); index < capacity * reduction; index += blockDim.x) {
        const int row = index / reduction;
        const int column = index % reduction;
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < count) {
            const int source_row = edge_sources[offsets[group] + row];
            value = source[static_cast<int64_t>(source_row) * reduction + column];
        }
        packed[(static_cast<int64_t>(compact_group) * capacity + row) * reduction + column] = value;
    }
}

ffi::Error ShuttleRelationSegmentPack(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> source,
    ffi::Buffer<ffi::S32, 1> counts,
    ffi::Buffer<ffi::S32, 1> offsets,
    ffi::Buffer<ffi::S32, 1> edge_sources,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> packed,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> padded_counts,
    std::int64_t group_count,
    std::int64_t capacity,
    std::int64_t reduction
) {
    if (group_count <= 0 || capacity <= 0 || reduction <= 0) {
        return ffi::Error::InvalidArgument("group_count, capacity, and reduction must be positive");
    }
    pack_relation_segments<<<static_cast<int>(group_count), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(source.typed_data()),
        counts.typed_data(),
        offsets.typed_data(),
        edge_sources.typed_data(),
        reinterpret_cast<__nv_bfloat16*>(packed->typed_data()),
        padded_counts->typed_data(),
        static_cast<int>(group_count),
        static_cast<int>(capacity),
        static_cast<int>(reduction)
    );
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return ffi::Error::Internal(std::string("pack_relation_segments: ") + cudaGetErrorString(status));
    }
    relation_plan_call_count.fetch_add(1, std::memory_order_relaxed);
    return ffi::Error::Success();
}

auto ShuttleRelationSegmentPackBinding() {
    return ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16, 2>>()
        .Arg<ffi::Buffer<ffi::S32, 1>>()
        .Arg<ffi::Buffer<ffi::S32, 1>>()
        .Arg<ffi::Buffer<ffi::S32, 1>>()
        .Ret<ffi::Buffer<ffi::BF16, 2>>()
        .Ret<ffi::Buffer<ffi::S32, 1>>()
        .Attr<std::int64_t>("group_count")
        .Attr<std::int64_t>("capacity")
        .Attr<std::int64_t>("reduction");
}

struct contract_globals {
    primitive::routed_activation_gl activations;
    primitive::routed_weight_gl weights;
    primitive::epi_bf16_gl output;
    primitive::index_gl padded_counts;

    __host__ inline dim3 grid() const {
        const int row_blocks = activations.rows() / config::MLP_Mb;
        const int column_blocks = output.cols() / config::MLP_Nb;
        return dim3(config::CLUSTER_SIZE * row_blocks * column_blocks);
    }
};

__device__ __forceinline__ void grouped_contract_kernel(const contract_globals& g) {
    const int task_index = clusterIdx().x;
    const int cta_rank = cluster_ctarank();
    warpgroup::increase_registers<256>();

    extern __shared__ int shared_memory[];
    const uint64_t shared_memory_base =
        (reinterpret_cast<uint64_t>(&shared_memory[0]) + 1023) & ~uint64_t(1023);

    __shared__ semaphore inputs_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < config::MLP_LOAD_PIPE_DEPTH; ++stage) {
            init_semaphore(inputs_arrived[stage], 0, shuttle_grouped_contract_event::kTransactionCompletionEnabled);
            init_semaphore(scales_arrived[stage], 0, shuttle_grouped_contract_event::kTransactionCompletionEnabled);
            init_semaphore(inputs_finished[stage], 0, shuttle_grouped_contract_event::kOperandReleaseLogicalCount);
            init_semaphore(scales_finished[stage], 0, shuttle_grouped_contract_event::kOperandReleaseLogicalCount);
        }
        init_semaphore(outputs_arrived, 0, shuttle_grouped_contract_event::kTransactionCompletionEnabled);
        init_semaphore(outputs_finished, 0, shuttle_grouped_contract_event::kOutputReleaseLogicalCount);
    }

    uint32_t bitfield = 0xFFFF0000;
    tensor_allocator<1, config::CLUSTER_SIZE> tensor_memory{};
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> accumulator =
        tensor_memory.template allocate<tt<float, config::MLP_Mb / 2, config::MLP_Nb>>(0);
    full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH> activation_scales =
        tensor_memory.template allocate<full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH>>(256);
    full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH> weight_scales =
        tensor_memory.template allocate<full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH>>(384);
    everyone::tma::cluster::sync();

    primitive::expert_grouped_gemm_kernel<false>(
        g.activations, g.weights,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        g.output, nullptr, nullptr, g.padded_counts,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        accumulator, activation_scales, weight_scales,
        inputs_arrived, scales_arrived, inputs_finished, scales_finished,
        outputs_arrived, outputs_finished, bitfield,
        g.activations.rows(), g.activations.rows(), g.activations.rows(),
        0, 0, task_index, cta_rank,
        0, 0, 0, 0, 0, shared_memory_base
    );
    everyone::tma::cluster::sync();
}

ffi::Error ShuttleGroupedContract(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> activations,
    ffi::Buffer<ffi::BF16, 3> weights,
    ffi::Buffer<ffi::S32, 1> padded_counts,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> output,
    std::int64_t groups,
    std::int64_t rows,
    std::int64_t reduction,
    std::int64_t columns
) {
    if (groups <= 0 || rows <= 0 || rows % config::MLP_Mb != 0) {
        return ffi::Error::InvalidArgument("groups and 256-aligned rows must be positive");
    }
    if (reduction <= 0 || reduction % config::MLP_BF16_Kb != 0) {
        return ffi::Error::InvalidArgument("reduction must be positive and divisible by 64");
    }
    if (columns <= 0 || columns % config::MLP_Nb != 0) {
        return ffi::Error::InvalidArgument("columns must be positive and divisible by 256");
    }
    contract_globals g{
        .activations = kittens::make_gl<primitive::routed_activation_gl>(
            reinterpret_cast<uint64_t>(activations.typed_data()), 1, 1, static_cast<int>(rows), static_cast<int>(reduction)),
        .weights = kittens::make_gl<primitive::routed_weight_gl>(
            reinterpret_cast<uint64_t>(weights.typed_data()), 1, static_cast<int>(groups), static_cast<int>(columns), static_cast<int>(reduction)),
        .output = kittens::make_gl<primitive::epi_bf16_gl>(
            reinterpret_cast<uint64_t>(output->typed_data()), 1, 1, static_cast<int>(rows), static_cast<int>(columns)),
        .padded_counts = kittens::make_gl<primitive::index_gl>(
            reinterpret_cast<uint64_t>(padded_counts.typed_data()), 1, 1, 1, static_cast<int>(groups)),
    };

    const int dynamic_shared_memory = config::DYNAMIC_SHARED_MEMORY;
    cudaError_t status = cudaFuncSetAttribute(
        kittens::py::global_kernel<config, contract_globals, grouped_contract_kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory
    );
    if (status != cudaSuccess) {
        return ffi::Error::Internal(std::string("cudaFuncSetAttribute: ") + cudaGetErrorString(status));
    }
    kittens::LaunchConfig<true, false> launch(
        g.grid(), dim3(config::NUM_THREADS), dynamic_shared_memory, stream, config::CLUSTER_SIZE
    );
    status = cudaLaunchKernelEx(
        launch,
        kittens::py::global_kernel<config, contract_globals, grouped_contract_kernel>,
        g
    );
    if (status != cudaSuccess) {
        return ffi::Error::Internal(std::string("grouped Contract launch: ") + cudaGetErrorString(status));
    }
    grouped_contract_call_count.fetch_add(1, std::memory_order_relaxed);
    return ffi::Error::Success();
}

auto ShuttleGroupedContractBinding() {
    return ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16, 2>>()
        .Arg<ffi::Buffer<ffi::BF16, 3>>()
        .Arg<ffi::Buffer<ffi::S32, 1>>()
        .Ret<ffi::Buffer<ffi::BF16, 2>>()
        .Attr<std::int64_t>("groups")
        .Attr<std::int64_t>("rows")
        .Attr<std::int64_t>("reduction")
        .Attr<std::int64_t>("columns");
}

}  // namespace tile_lifetime::segmented_contract_ffi

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    shuttle_relation_segment_pack_ffi,
    tile_lifetime::segmented_contract_ffi::ShuttleRelationSegmentPack,
    tile_lifetime::segmented_contract_ffi::ShuttleRelationSegmentPackBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    shuttle_grouped_contract_ffi,
    tile_lifetime::segmented_contract_ffi::ShuttleGroupedContract,
    tile_lifetime::segmented_contract_ffi::ShuttleGroupedContractBinding());

extern "C" int shuttle_relation_segment_pack_ffi_call_count() {
    return tile_lifetime::segmented_contract_ffi::relation_plan_call_count.load(std::memory_order_relaxed);
}

extern "C" int shuttle_grouped_contract_ffi_call_count() {
    return tile_lifetime::segmented_contract_ffi::grouped_contract_call_count.load(std::memory_order_relaxed);
}

extern "C" const char* shuttle_grouped_contract_event_fingerprint() {
    return SHUTTLE_GROUPED_CONTRACT_EVENT_SHA256;
}
