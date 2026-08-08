#include "mok_megakernel.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/csrc/utils/pybind.h>

namespace tile_lifetime::mok_gmm_probe {

using combiner = dispatch_mlp_swiglu_combiner<4, utils::RoutedPrecision::BF16>;
using config = combiner::config;

struct globals {
    combiner::routed_activation_gl activations;
    combiner::routed_weight_gl weights;
    combiner::epi_bf16_gl output;
    combiner::index_gl padded_counts;

    __host__ inline dim3 grid() const {
        const int row_blocks = activations.rows() / config::MLP_Mb;
        const int column_blocks = output.cols() / config::MLP_Nb;
        return dim3(config::CLUSTER_SIZE * row_blocks * column_blocks);
    }
};

static __device__ __forceinline__ void grouped_gemm_kernel(const globals &g) {
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
            init_semaphore(inputs_arrived[stage], 0, 1);
            init_semaphore(scales_arrived[stage], 0, 1);
            init_semaphore(inputs_finished[stage], 0, 1);
            init_semaphore(scales_finished[stage], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, config::CLUSTER_SIZE);
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

    combiner::expert_grouped_gemm_kernel<false>(
        g.activations,
        g.weights,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        g.output,
        nullptr,
        nullptr,
        g.padded_counts,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        accumulator,
        activation_scales,
        weight_scales,
        inputs_arrived,
        scales_arrived,
        inputs_finished,
        scales_finished,
        outputs_arrived,
        outputs_finished,
        bitfield,
        g.activations.rows(),
        g.activations.rows(),
        g.activations.rows(),
        0,
        0,
        task_index,
        cta_rank,
        0,
        0,
        0,
        0,
        0,
        shared_memory_base
    );

    everyone::tma::cluster::sync();
}

static void grouped_gemm_out(
    const at::Tensor &activations,
    const at::Tensor &weights,
    const at::Tensor &padded_counts,
    const at::Tensor &output
) {
    TORCH_CHECK(activations.is_cuda() && weights.is_cuda() && padded_counts.is_cuda() && output.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(activations.is_contiguous() && weights.is_contiguous() && padded_counts.is_contiguous()
                    && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(activations.scalar_type() == at::kBFloat16 && weights.scalar_type() == at::kBFloat16
                    && output.scalar_type() == at::kBFloat16,
                "activations, weights, and output must be BF16");
    TORCH_CHECK(padded_counts.scalar_type() == at::kInt, "padded_counts must be int32");
    TORCH_CHECK(activations.device() == weights.device() && activations.device() == padded_counts.device()
                    && activations.device() == output.device(),
                "all tensors must be on the same device");
    TORCH_CHECK(activations.dim() == 2, "activations must have shape [rows, K]");
    TORCH_CHECK(weights.dim() == 3, "weights must have shape [experts, N, K]");
    TORCH_CHECK(output.dim() == 2, "output must have shape [rows, N]");
    TORCH_CHECK(padded_counts.dim() == 1, "padded_counts must have shape [experts]");
    TORCH_CHECK(weights.size(0) == padded_counts.size(0), "weights and padded_counts disagree on experts");
    TORCH_CHECK(weights.size(2) == activations.size(1), "weights K must match activations K");
    TORCH_CHECK(output.size(0) == activations.size(0) && output.size(1) == weights.size(1),
                "output must have shape [activations.rows, weights.N]");
    TORCH_CHECK(activations.size(0) > 0 && activations.size(0) % config::MLP_Mb == 0,
                "activation rows must be positive and divisible by 256");
    TORCH_CHECK(activations.size(1) > 0 && activations.size(1) % config::MLP_BF16_Kb == 0,
                "K must be positive and divisible by 64");
    TORCH_CHECK(output.size(1) > 0 && output.size(1) % config::MLP_Nb == 0,
                "N must be positive and divisible by 256");

    globals g{
        .activations = kittens::py::tensor_to_gl<combiner::routed_activation_gl>(activations),
        .weights = kittens::py::tensor_to_gl<combiner::routed_weight_gl>(weights),
        .output = kittens::py::tensor_to_gl<combiner::epi_bf16_gl>(output),
        .padded_counts = kittens::py::tensor_to_gl<combiner::index_gl>(padded_counts),
    };
    kittens::py::launch_kernel<config, globals, grouped_gemm_kernel>(g);
}

static __global__ void swiglu_bf16x2_kernel(
    const __nv_bfloat162 *gate,
    const __nv_bfloat162 *up,
    __nv_bfloat162 *output,
    int64_t pairs
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pairs) {
        return;
    }
    const float2 gate_values = __bfloat1622float2(gate[index]);
    const float2 up_values = __bfloat1622float2(up[index]);
    const float output_0 = gate_values.x / (1.0f + expf(-gate_values.x)) * up_values.x;
    const float output_1 = gate_values.y / (1.0f + expf(-gate_values.y)) * up_values.y;
    output[index] = __floats2bfloat162_rn(output_0, output_1);
}

static void swiglu_bf16_out(const at::Tensor &gate, const at::Tensor &up, const at::Tensor &output) {
    TORCH_CHECK(gate.is_cuda() && up.is_cuda() && output.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(gate.is_contiguous() && up.is_contiguous() && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(gate.scalar_type() == at::kBFloat16 && up.scalar_type() == at::kBFloat16
                    && output.scalar_type() == at::kBFloat16,
                "gate, up, and output must be BF16");
    TORCH_CHECK(gate.sizes() == up.sizes() && gate.sizes() == output.sizes(),
                "gate, up, and output must have identical shapes");
    TORCH_CHECK(gate.device() == up.device() && gate.device() == output.device(),
                "all tensors must be on the same device");
    TORCH_CHECK(gate.numel() % 2 == 0, "SwiGLU tensors must contain an even number of elements");

    constexpr int threads = 256;
    const int64_t pairs = gate.numel() / 2;
    const int blocks = static_cast<int>((pairs + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    swiglu_bf16x2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162 *>(gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat162 *>(up.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
        pairs
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __global__ void swiglu_row_halves_bf16x2_kernel(
    const __nv_bfloat162 *gate_up,
    __nv_bfloat162 *output,
    int rows,
    int intermediate_pairs
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(rows) * intermediate_pairs;
    if (index >= elements) {
        return;
    }
    const int row = static_cast<int>(index / intermediate_pairs);
    const int pair = static_cast<int>(index % intermediate_pairs);
    const int64_t gate_index = static_cast<int64_t>(row) * 2 * intermediate_pairs + pair;
    const int64_t up_index = gate_index + intermediate_pairs;
    const float2 gate_values = __bfloat1622float2(gate_up[gate_index]);
    const float2 up_values = __bfloat1622float2(gate_up[up_index]);
    const float output_0 = gate_values.x / (1.0f + expf(-gate_values.x)) * up_values.x;
    const float output_1 = gate_values.y / (1.0f + expf(-gate_values.y)) * up_values.y;
    output[index] = __floats2bfloat162_rn(output_0, output_1);
}

static void swiglu_row_halves_bf16_out(const at::Tensor &gate_up, const at::Tensor &output) {
    TORCH_CHECK(gate_up.is_cuda() && output.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(gate_up.is_contiguous() && output.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(gate_up.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
                "gate_up and output must be BF16");
    TORCH_CHECK(gate_up.dim() == 2 && output.dim() == 2, "gate_up and output must be rank 2");
    TORCH_CHECK(gate_up.size(0) == output.size(0), "gate_up and output row counts must match");
    TORCH_CHECK(gate_up.size(1) == 2 * output.size(1), "gate_up width must be twice the output width");
    TORCH_CHECK(gate_up.device() == output.device(), "gate_up and output must be on the same device");
    TORCH_CHECK(output.size(1) % 2 == 0, "output width must be even");

    constexpr int threads = 256;
    const int rows = static_cast<int>(output.size(0));
    const int intermediate_pairs = static_cast<int>(output.size(1) / 2);
    const int64_t elements = static_cast<int64_t>(rows) * intermediate_pairs;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    swiglu_row_halves_bf16x2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162 *>(gate_up.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
        rows,
        intermediate_pairs
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __global__ void padded_pack_bf16x2_kernel(
    const __nv_bfloat162 *received,
    const int64_t *padded_receiver_rows,
    __nv_bfloat162 *output,
    int padded_rows,
    int hidden_pairs
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(padded_rows) * hidden_pairs;
    if (index >= elements) {
        return;
    }
    const int padded_row = static_cast<int>(index / hidden_pairs);
    const int hidden_pair = static_cast<int>(index % hidden_pairs);
    const int64_t receiver_row = padded_receiver_rows[padded_row];
    output[index] = receiver_row >= 0 ? received[receiver_row * hidden_pairs + hidden_pair]
                                      : __floats2bfloat162_rn(0.0f, 0.0f);
}

static void padded_pack_bf16_out(
    const at::Tensor &received,
    const at::Tensor &padded_receiver_rows,
    const at::Tensor &output
) {
    TORCH_CHECK(received.is_cuda() && padded_receiver_rows.is_cuda() && output.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(received.is_contiguous() && padded_receiver_rows.is_contiguous() && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(received.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
                "received and output must be BF16");
    TORCH_CHECK(padded_receiver_rows.scalar_type() == at::kLong, "padded_receiver_rows must be int64");
    TORCH_CHECK(received.dim() == 2 && padded_receiver_rows.dim() == 1 && output.dim() == 2,
                "received/output must be rank 2 and padded_receiver_rows rank 1");
    TORCH_CHECK(output.size(0) == padded_receiver_rows.size(0),
                "output rows must match padded_receiver_rows");
    TORCH_CHECK(output.size(1) == received.size(1), "received and output hidden dimensions must match");
    TORCH_CHECK(received.device() == padded_receiver_rows.device() && received.device() == output.device(),
                "all tensors must be on the same device");
    TORCH_CHECK(received.size(1) % 2 == 0, "hidden dimension must be even");

    constexpr int threads = 256;
    const int padded_rows = static_cast<int>(output.size(0));
    const int hidden_pairs = static_cast<int>(output.size(1) / 2);
    const int64_t elements = static_cast<int64_t>(padded_rows) * hidden_pairs;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    padded_pack_bf16x2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162 *>(received.data_ptr<at::BFloat16>()),
        padded_receiver_rows.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
        padded_rows,
        hidden_pairs
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Build the receiver-local physical index plane for a bounded segmented
// contraction.  One CTA owns one destination group and walks the received
// edge table in source/slot order.  This deliberately favors a deterministic
// generic construction over atomics: changing the group relation or edge
// weights does not require changing the expert contraction body.
static __global__ void fixed_capacity_relation_plan_kernel(
    const int64_t *destination_group,
    const float *edge_weight,
    int64_t *padded_source_rows,
    int64_t *edge_destination_rows,
    float *ordered_edge_weight,
    int32_t *group_counts,
    int32_t *group_overflow,
    int source_rows,
    int route_slots,
    int group_count,
    int group_capacity
) {
    const int group = static_cast<int>(blockIdx.x);
    if (group >= group_count) {
        return;
    }
    __shared__ int warp_counts[8];
    __shared__ int warp_offsets[8];
    __shared__ int tile_count;
    for (int row = static_cast<int>(threadIdx.x); row < group_capacity; row += blockDim.x) {
        padded_source_rows[static_cast<int64_t>(group) * group_capacity + row] = -1;
    }
    __syncthreads();

    int running_count = 0;
    const int64_t edge_count = static_cast<int64_t>(source_rows) * route_slots;
    for (int64_t edge_base = 0; edge_base < edge_count; edge_base += blockDim.x) {
        const int64_t edge = edge_base + threadIdx.x;
        const bool matches = edge < edge_count && destination_group[edge] == group;
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const unsigned int match_mask = __ballot_sync(0xffffffffu, matches);
        if (lane == 0) {
            warp_counts[warp] = __popc(match_mask);
        }
        __syncthreads();
        if (threadIdx.x < 8) {
            int offset = 0;
            for (int prior_warp = 0; prior_warp < static_cast<int>(threadIdx.x); ++prior_warp) {
                offset += warp_counts[prior_warp];
            }
            warp_offsets[threadIdx.x] = offset;
            if (threadIdx.x == 7) {
                tile_count = offset + warp_counts[7];
            }
        }
        __syncthreads();
        if (matches) {
            const unsigned int lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
            const int stable_offset = warp_offsets[warp] + __popc(match_mask & lower_lanes);
            const int destination_offset = running_count + stable_offset;
            if (destination_offset < group_capacity) {
                const int64_t destination_row =
                    static_cast<int64_t>(group) * group_capacity + destination_offset;
                padded_source_rows[destination_row] = edge / route_slots;
                edge_destination_rows[edge] = destination_row;
                ordered_edge_weight[edge] = edge_weight[edge];
            }
        }
        __syncthreads();
        running_count += tile_count;
    }
    if (threadIdx.x == 0) {
        group_counts[group] = running_count;
        const int excess = running_count - group_capacity;
        group_overflow[group] = excess > 0 ? excess : 0;
    }
}

static void fixed_capacity_relation_plan_out(
    const at::Tensor &destination_group,
    const at::Tensor &edge_weight,
    const at::Tensor &padded_source_rows,
    const at::Tensor &edge_destination_rows,
    const at::Tensor &ordered_edge_weight,
    const at::Tensor &group_counts,
    const at::Tensor &overflow
) {
    TORCH_CHECK(destination_group.is_cuda() && edge_weight.is_cuda() && padded_source_rows.is_cuda()
                    && edge_destination_rows.is_cuda() && ordered_edge_weight.is_cuda()
                    && group_counts.is_cuda() && overflow.is_cuda(),
                "all relation-plan tensors must be CUDA tensors");
    TORCH_CHECK(destination_group.is_contiguous() && edge_weight.is_contiguous()
                    && padded_source_rows.is_contiguous() && edge_destination_rows.is_contiguous()
                    && ordered_edge_weight.is_contiguous() && group_counts.is_contiguous()
                    && overflow.is_contiguous(),
                "all relation-plan tensors must be contiguous");
    TORCH_CHECK(destination_group.scalar_type() == at::kLong, "destination_group must be int64");
    TORCH_CHECK(edge_weight.scalar_type() == at::kFloat && ordered_edge_weight.scalar_type() == at::kFloat,
                "edge weights must be FP32");
    TORCH_CHECK(padded_source_rows.scalar_type() == at::kLong
                    && edge_destination_rows.scalar_type() == at::kLong,
                "relation row maps must be int64");
    TORCH_CHECK(group_counts.scalar_type() == at::kInt && overflow.scalar_type() == at::kInt,
                "group_counts and overflow must be int32");
    TORCH_CHECK(destination_group.dim() == 2 && edge_weight.sizes() == destination_group.sizes(),
                "destination groups and weights must have identical [source, slot] shape");
    TORCH_CHECK(edge_destination_rows.sizes() == destination_group.sizes()
                    && ordered_edge_weight.sizes() == destination_group.sizes(),
                "edge outputs must match the input relation shape");
    TORCH_CHECK(padded_source_rows.dim() == 1 && group_counts.dim() == 1
                    && overflow.sizes() == group_counts.sizes(),
                "invalid relation-plan output ranks");
    TORCH_CHECK(group_counts.size(0) > 0
                    && padded_source_rows.size(0) % group_counts.size(0) == 0,
                "padded rows must define one fixed positive capacity per group");
    TORCH_CHECK(destination_group.device() == edge_weight.device()
                    && destination_group.device() == padded_source_rows.device()
                    && destination_group.device() == edge_destination_rows.device()
                    && destination_group.device() == ordered_edge_weight.device()
                    && destination_group.device() == group_counts.device()
                    && destination_group.device() == overflow.device(),
                "all relation-plan tensors must be on the same device");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        edge_destination_rows.data_ptr<int64_t>(),
        0xff,
        edge_destination_rows.numel() * sizeof(int64_t),
        stream
    ));
    C10_CUDA_CHECK(cudaMemsetAsync(group_counts.data_ptr<int32_t>(), 0, group_counts.numel() * sizeof(int32_t), stream));
    constexpr int threads = 256;
    const int groups = static_cast<int>(group_counts.size(0));
    const int capacity = static_cast<int>(padded_source_rows.size(0) / group_counts.size(0));
    fixed_capacity_relation_plan_kernel<<<groups, threads, 0, stream>>>(
        destination_group.data_ptr<int64_t>(),
        edge_weight.data_ptr<float>(),
        padded_source_rows.data_ptr<int64_t>(),
        edge_destination_rows.data_ptr<int64_t>(),
        ordered_edge_weight.data_ptr<float>(),
        group_counts.data_ptr<int32_t>(),
        overflow.data_ptr<int32_t>(),
        static_cast<int>(destination_group.size(0)),
        static_cast<int>(destination_group.size(1)),
        groups,
        capacity
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool explicit_rounding>
static __global__ void fixed_route_merge_kernel(
    const __nv_bfloat162 *expert_output,
    const int64_t *route_rows,
    const float *route_weights,
    __nv_bfloat162 *output,
    int tokens,
    int topk,
    int hidden_pairs
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(tokens) * hidden_pairs;
    if (index >= elements) {
        return;
    }
    const int token = static_cast<int>(index / hidden_pairs);
    const int feature_pair = static_cast<int>(index % hidden_pairs);
    float2 value = make_float2(0.0f, 0.0f);
    #pragma unroll 1
    for (int route_slot = 0; route_slot < topk; ++route_slot) {
        const int64_t route_index = static_cast<int64_t>(token) * topk + route_slot;
        const int64_t expert_row = route_rows[route_index];
        if (expert_row >= 0) {
            const float2 route_value = __bfloat1622float2(
                expert_output[expert_row * hidden_pairs + feature_pair]
            );
            const float weight = route_weights[route_index];
            if constexpr (explicit_rounding) {
                value.x = __fadd_rn(value.x, __fmul_rn(route_value.x, weight));
                value.y = __fadd_rn(value.y, __fmul_rn(route_value.y, weight));
            } else {
                value.x += route_value.x * weight;
                value.y += route_value.y * weight;
            }
        }
    }
    output[index] = __floats2bfloat162_rn(value.x, value.y);
}

static void fixed_route_merge_out_impl(
    const at::Tensor &expert_output,
    const at::Tensor &route_rows,
    const at::Tensor &route_weights,
    const at::Tensor &output,
    bool explicit_rounding
) {
    TORCH_CHECK(expert_output.is_cuda() && route_rows.is_cuda() && route_weights.is_cuda() && output.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(expert_output.is_contiguous() && route_rows.is_contiguous() && route_weights.is_contiguous()
                    && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(expert_output.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
                "expert_output and output must be BF16");
    TORCH_CHECK(route_rows.scalar_type() == at::kLong, "route_rows must be int64");
    TORCH_CHECK(route_weights.scalar_type() == at::kFloat, "route_weights must be FP32");
    TORCH_CHECK(expert_output.dim() == 2 && route_rows.dim() == 2 && route_weights.dim() == 2
                    && output.dim() == 2,
                "invalid tensor ranks for fixed route merge");
    TORCH_CHECK(route_rows.sizes() == route_weights.sizes(), "route rows and weights must have identical shapes");
    TORCH_CHECK(output.size(0) == route_rows.size(0), "output token count must match route metadata");
    TORCH_CHECK(output.size(1) == expert_output.size(1), "hidden dimensions must match");
    TORCH_CHECK(output.size(1) % 2 == 0, "hidden dimension must be even for BF16x2 merge");
    TORCH_CHECK(expert_output.device() == route_rows.device() && expert_output.device() == route_weights.device()
                    && expert_output.device() == output.device(),
                "all tensors must be on the same device");

    constexpr int threads = 256;
    const int tokens = static_cast<int>(output.size(0));
    const int topk = static_cast<int>(route_rows.size(1));
    const int hidden_pairs = static_cast<int>(output.size(1) / 2);
    const int64_t elements = static_cast<int64_t>(tokens) * hidden_pairs;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (explicit_rounding) {
        fixed_route_merge_kernel<true><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162 *>(expert_output.data_ptr<at::BFloat16>()),
            route_rows.data_ptr<int64_t>(),
            route_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
            tokens,
            topk,
            hidden_pairs
        );
    } else {
        fixed_route_merge_kernel<false><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162 *>(expert_output.data_ptr<at::BFloat16>()),
            route_rows.data_ptr<int64_t>(),
            route_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
            tokens,
            topk,
            hidden_pairs
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void fixed_route_merge_out(
    const at::Tensor &expert_output,
    const at::Tensor &route_rows,
    const at::Tensor &route_weights,
    const at::Tensor &output
) {
    fixed_route_merge_out_impl(expert_output, route_rows, route_weights, output, true);
}

static void fixed_route_merge_fma_out(
    const at::Tensor &expert_output,
    const at::Tensor &route_rows,
    const at::Tensor &route_weights,
    const at::Tensor &output
) {
    fixed_route_merge_out_impl(expert_output, route_rows, route_weights, output, false);
}

static __global__ void fixed_route_merge_shared_kernel(
    const __nv_bfloat16 *expert_output,
    const int64_t *route_rows,
    const float *route_weights,
    const int64_t *source_receiver_rows,
    const __nv_bfloat16 *shared_output,
    __nv_bfloat16 *output,
    int tokens,
    int topk,
    int hidden
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(tokens) * hidden;
    if (index >= elements) {
        return;
    }
    const int token = static_cast<int>(index / hidden);
    const int feature = static_cast<int>(index % hidden);
    const int64_t receiver_row = source_receiver_rows[token];
    float routed_value = 0.0f;
    if (receiver_row >= 0) {
        #pragma unroll 1
        for (int route_slot = 0; route_slot < topk; ++route_slot) {
            const int64_t route_index = receiver_row * topk + route_slot;
            const int64_t expert_row = route_rows[route_index];
            if (expert_row >= 0) {
                const float route_value = __bfloat162float(expert_output[expert_row * hidden + feature]);
                routed_value = __fadd_rn(routed_value, __fmul_rn(route_value, route_weights[route_index]));
            }
        }
    }
    const __nv_bfloat16 routed_bf16 = __float2bfloat16_rn(routed_value);
    const float combined_value = __bfloat162float(routed_bf16) + __bfloat162float(shared_output[index]);
    output[index] = __float2bfloat16_rn(combined_value);
}

static void fixed_route_merge_shared_out(
    const at::Tensor &expert_output,
    const at::Tensor &route_rows,
    const at::Tensor &route_weights,
    const at::Tensor &source_receiver_rows,
    const at::Tensor &shared_output,
    const at::Tensor &output
) {
    TORCH_CHECK(expert_output.is_cuda() && route_rows.is_cuda() && route_weights.is_cuda()
                    && source_receiver_rows.is_cuda() && shared_output.is_cuda() && output.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(expert_output.is_contiguous() && route_rows.is_contiguous() && route_weights.is_contiguous()
                    && source_receiver_rows.is_contiguous() && shared_output.is_contiguous() && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(expert_output.scalar_type() == at::kBFloat16 && shared_output.scalar_type() == at::kBFloat16
                    && output.scalar_type() == at::kBFloat16,
                "expert_output, shared_output, and output must be BF16");
    TORCH_CHECK(route_rows.scalar_type() == at::kLong && source_receiver_rows.scalar_type() == at::kLong,
                "route_rows and source_receiver_rows must be int64");
    TORCH_CHECK(route_weights.scalar_type() == at::kFloat, "route_weights must be FP32");
    TORCH_CHECK(expert_output.dim() == 2 && route_rows.dim() == 2 && route_weights.dim() == 2
                    && source_receiver_rows.dim() == 1 && shared_output.dim() == 2 && output.dim() == 2,
                "invalid tensor ranks for fixed route merge and shared add");
    TORCH_CHECK(route_rows.sizes() == route_weights.sizes(), "route rows and weights must have identical shapes");
    TORCH_CHECK(output.sizes() == shared_output.sizes(), "output and shared_output must have identical shapes");
    TORCH_CHECK(output.size(0) == source_receiver_rows.size(0),
                "output token count must match source_receiver_rows");
    TORCH_CHECK(output.size(1) == expert_output.size(1), "hidden dimensions must match");
    TORCH_CHECK(expert_output.device() == route_rows.device() && expert_output.device() == route_weights.device()
                    && expert_output.device() == source_receiver_rows.device()
                    && expert_output.device() == shared_output.device() && expert_output.device() == output.device(),
                "all tensors must be on the same device");

    constexpr int threads = 256;
    const int tokens = static_cast<int>(output.size(0));
    const int topk = static_cast<int>(route_rows.size(1));
    const int hidden = static_cast<int>(output.size(1));
    const int64_t elements = static_cast<int64_t>(tokens) * hidden;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fixed_route_merge_shared_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(expert_output.data_ptr<at::BFloat16>()),
        route_rows.data_ptr<int64_t>(),
        route_weights.data_ptr<float>(),
        source_receiver_rows.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16 *>(shared_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>()),
        tokens,
        topk,
        hidden
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __global__ void fixed_rank_merge_shared_kernel(
    const __nv_bfloat162 *rank_partials,
    const int64_t *rank_token_rows,
    const __nv_bfloat162 *shared_output,
    __nv_bfloat162 *output,
    int ranks,
    int tokens,
    int hidden_pairs
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(tokens) * hidden_pairs;
    if (index >= elements) {
        return;
    }
    const int token = static_cast<int>(index / hidden_pairs);
    const int feature_pair = static_cast<int>(index % hidden_pairs);
    float2 value = __bfloat1622float2(shared_output[index]);
    #pragma unroll 1
    for (int rank = 0; rank < ranks; ++rank) {
        const int64_t row = rank_token_rows[static_cast<int64_t>(rank) * tokens + token];
        if (row >= 0) {
            const float2 partial = __bfloat1622float2(
                rank_partials[row * hidden_pairs + feature_pair]
            );
            value.x = __fadd_rn(value.x, partial.x);
            value.y = __fadd_rn(value.y, partial.y);
        }
    }
    output[index] = __floats2bfloat162_rn(value.x, value.y);
}

static void fixed_rank_merge_shared_out(
    const at::Tensor &rank_partials,
    const at::Tensor &rank_token_rows,
    const at::Tensor &shared_output,
    const at::Tensor &output
) {
    TORCH_CHECK(rank_partials.is_cuda() && rank_token_rows.is_cuda() && shared_output.is_cuda()
                    && output.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(rank_partials.is_contiguous() && rank_token_rows.is_contiguous() && shared_output.is_contiguous()
                    && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(rank_partials.scalar_type() == at::kBFloat16 && shared_output.scalar_type() == at::kBFloat16
                    && output.scalar_type() == at::kBFloat16,
                "rank_partials, shared_output, and output must be BF16");
    TORCH_CHECK(rank_token_rows.scalar_type() == at::kLong, "rank_token_rows must be int64");
    TORCH_CHECK(rank_partials.dim() == 2 && rank_token_rows.dim() == 2 && shared_output.dim() == 2
                    && output.dim() == 2,
                "invalid tensor ranks for fixed rank merge and shared add");
    TORCH_CHECK(output.sizes() == shared_output.sizes(), "output and shared_output must have identical shapes");
    TORCH_CHECK(rank_token_rows.size(1) == output.size(0), "rank_token_rows token count must match output");
    TORCH_CHECK(rank_partials.size(1) == output.size(1), "hidden dimensions must match");
    TORCH_CHECK(output.size(1) % 2 == 0, "hidden dimension must be even for BF16x2 merge");
    TORCH_CHECK(rank_partials.device() == rank_token_rows.device() && rank_partials.device() == shared_output.device()
                    && rank_partials.device() == output.device(),
                "all tensors must be on the same device");

    constexpr int threads = 256;
    const int ranks = static_cast<int>(rank_token_rows.size(0));
    const int tokens = static_cast<int>(output.size(0));
    const int hidden_pairs = static_cast<int>(output.size(1) / 2);
    const int64_t elements = static_cast<int64_t>(tokens) * hidden_pairs;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fixed_rank_merge_shared_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162 *>(rank_partials.data_ptr<at::BFloat16>()),
        rank_token_rows.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat162 *>(shared_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
        ranks,
        tokens,
        hidden_pairs
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace tile_lifetime::mok_gmm_probe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "grouped_gemm_out",
        &tile_lifetime::mok_gmm_probe::grouped_gemm_out,
        pybind11::arg("activations"),
        pybind11::arg("weights"),
        pybind11::arg("padded_counts"),
        pybind11::arg("output")
    );
    module.def(
        "swiglu_bf16_out",
        &tile_lifetime::mok_gmm_probe::swiglu_bf16_out,
        pybind11::arg("gate"),
        pybind11::arg("up"),
        pybind11::arg("output")
    );
    module.def(
        "swiglu_row_halves_bf16_out",
        &tile_lifetime::mok_gmm_probe::swiglu_row_halves_bf16_out,
        pybind11::arg("gate_up"),
        pybind11::arg("output")
    );
    module.def(
        "padded_pack_bf16_out",
        &tile_lifetime::mok_gmm_probe::padded_pack_bf16_out,
        pybind11::arg("received"),
        pybind11::arg("padded_receiver_rows"),
        pybind11::arg("output")
    );
    module.def(
        "fixed_capacity_relation_plan_out",
        &tile_lifetime::mok_gmm_probe::fixed_capacity_relation_plan_out,
        pybind11::arg("destination_group"),
        pybind11::arg("edge_weight"),
        pybind11::arg("padded_source_rows"),
        pybind11::arg("edge_destination_rows"),
        pybind11::arg("ordered_edge_weight"),
        pybind11::arg("group_counts"),
        pybind11::arg("overflow")
    );
    module.def(
        "fixed_route_merge_out",
        &tile_lifetime::mok_gmm_probe::fixed_route_merge_out,
        pybind11::arg("expert_output"),
        pybind11::arg("route_rows"),
        pybind11::arg("route_weights"),
        pybind11::arg("output")
    );
    module.def(
        "fixed_route_merge_fma_out",
        &tile_lifetime::mok_gmm_probe::fixed_route_merge_fma_out,
        pybind11::arg("expert_output"),
        pybind11::arg("route_rows"),
        pybind11::arg("route_weights"),
        pybind11::arg("output")
    );
    module.def(
        "fixed_route_merge_shared_out",
        &tile_lifetime::mok_gmm_probe::fixed_route_merge_shared_out,
        pybind11::arg("expert_output"),
        pybind11::arg("route_rows"),
        pybind11::arg("route_weights"),
        pybind11::arg("source_receiver_rows"),
        pybind11::arg("shared_output"),
        pybind11::arg("output")
    );
    module.def(
        "fixed_rank_merge_shared_out",
        &tile_lifetime::mok_gmm_probe::fixed_rank_merge_shared_out,
        pybind11::arg("rank_partials"),
        pybind11::arg("rank_token_rows"),
        pybind11::arg("shared_output"),
        pybind11::arg("output")
    );
}
