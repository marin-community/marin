#include "mok_megakernel.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/csrc/utils/pybind.h>

namespace tile_lifetime::mok_gmm_probe {

using combiner = dispatch_mlp_swiglu_combiner<4, utils::RoutedPrecision::BF16>;
using config = combiner::config;

#include "generated_map_fold.inc"
#include "generated_training.inc"
#include "generated_event_schedule.inc"

static_assert(config::CLUSTER_SIZE == shuttle_grouped_contract_event::kClusterCtas);
static_assert(config::MLP_LOAD_PIPE_DEPTH == shuttle_grouped_contract_event::kLoadPipelineStages);
static_assert(
    shuttle_grouped_contract_event::kOperandReadyLogicalCount
    == config::CLUSTER_SIZE * config::NUM_PRODUCERS
);
static_assert(shuttle_grouped_contract_event::kOperandReleaseLogicalCount == config::NUM_CONSUMERS);
static_assert(shuttle_grouped_contract_event::kOutputReadyLogicalCount == config::NUM_CONSUMERS);
static_assert(
    shuttle_grouped_contract_event::kOutputReleaseLogicalCount
    == config::CLUSTER_SIZE * config::NUM_CONSUMERS
);
static_assert(
    shuttle_grouped_contract_event::kOperandTransactionBytes
    == config::CLUSTER_SIZE * 2 * sizeof(combiner::mlp_bf16_tile)
);

static const char *generated_map_fold_program_sha256() {
    return SHUTTLE_MAP_FOLD_PROGRAM_SHA256;
}

static const char *generated_expert_training_program_sha256() {
    return SHUTTLE_EXPERT_TRAINING_PROGRAM_SHA256;
}

static const char *generated_grouped_contract_event_sha256() {
    return SHUTTLE_GROUPED_CONTRACT_EVENT_SHA256;
}

static std::vector<int64_t> grouped_contract_event_attributes() {
    return {
        shuttle_grouped_contract_event::kClusterCtas,
        shuttle_grouped_contract_event::kLoadPipelineStages,
        shuttle_grouped_contract_event::kOperandReadyLogicalCount,
        shuttle_grouped_contract_event::kOperandReleaseLogicalCount,
        shuttle_grouped_contract_event::kOutputReadyLogicalCount,
        shuttle_grouped_contract_event::kOutputReleaseLogicalCount,
        shuttle_grouped_contract_event::kOperandTransactionBytes,
        shuttle_grouped_contract_event::kTransactionCompletionEnabled,
    };
}

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
            // Transaction completion uses a byte expectation and is distinct
            // from the two-owner logical operand-ready Event Tensor count.
            init_semaphore(
                inputs_arrived[stage],
                0,
                shuttle_grouped_contract_event::kTransactionCompletionEnabled
            );
            init_semaphore(
                scales_arrived[stage],
                0,
                shuttle_grouped_contract_event::kTransactionCompletionEnabled
            );
            init_semaphore(
                inputs_finished[stage],
                0,
                shuttle_grouped_contract_event::kOperandReleaseLogicalCount
            );
            init_semaphore(
                scales_finished[stage],
                0,
                shuttle_grouped_contract_event::kOperandReleaseLogicalCount
            );
        }
        init_semaphore(
            outputs_arrived,
            0,
            shuttle_grouped_contract_event::kTransactionCompletionEnabled
        );
        init_semaphore(
            outputs_finished,
            0,
            shuttle_grouped_contract_event::kOutputReleaseLogicalCount
        );
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

static __global__ void adjacent_pair_map_bf16x2_kernel(
    const __nv_bfloat162 *left,
    const __nv_bfloat162 *right,
    __nv_bfloat162 *output,
    int64_t pairs
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pairs) {
        return;
    }
    const float2 left_values = __bfloat1622float2(left[index]);
    const float2 right_values = __bfloat1622float2(right[index]);
    const float output_0 = generated_pair_map(left_values.x, right_values.x);
    const float output_1 = generated_pair_map(left_values.y, right_values.y);
    output[index] = __floats2bfloat162_rn(output_0, output_1);
}

static void adjacent_pair_map_bf16_out(const at::Tensor &left, const at::Tensor &right, const at::Tensor &output) {
    TORCH_CHECK(left.is_cuda() && right.is_cuda() && output.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(left.is_contiguous() && right.is_contiguous() && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(left.scalar_type() == at::kBFloat16 && right.scalar_type() == at::kBFloat16
                    && output.scalar_type() == at::kBFloat16,
                "left, right, and output must be BF16");
    TORCH_CHECK(left.sizes() == right.sizes() && left.sizes() == output.sizes(),
                "left, right, and output must have identical shapes");
    TORCH_CHECK(left.device() == right.device() && left.device() == output.device(),
                "all tensors must be on the same device");
    TORCH_CHECK(left.numel() % 2 == 0, "pair-map tensors must contain an even number of elements");

    constexpr int threads = 256;
    const int64_t pairs = left.numel() / 2;
    const int blocks = static_cast<int>((pairs + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    adjacent_pair_map_bf16x2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162 *>(left.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat162 *>(right.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
        pairs
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __global__ void row_halves_pair_map_bf16x2_kernel(
    const __nv_bfloat162 *pairs,
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
    const int64_t left_index = static_cast<int64_t>(row) * 2 * intermediate_pairs + pair;
    const int64_t right_index = left_index + intermediate_pairs;
    const float2 left_values = __bfloat1622float2(pairs[left_index]);
    const float2 right_values = __bfloat1622float2(pairs[right_index]);
    const float output_0 = generated_pair_map(left_values.x, right_values.x);
    const float output_1 = generated_pair_map(left_values.y, right_values.y);
    output[index] = __floats2bfloat162_rn(output_0, output_1);
}

static void row_halves_pair_map_bf16_out(const at::Tensor &pairs, const at::Tensor &output) {
    TORCH_CHECK(pairs.is_cuda() && output.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(pairs.is_contiguous() && output.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(pairs.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
                "pairs and output must be BF16");
    TORCH_CHECK(pairs.dim() == 2 && output.dim() == 2, "pairs and output must be rank 2");
    TORCH_CHECK(pairs.size(0) == output.size(0), "pairs and output row counts must match");
    TORCH_CHECK(pairs.size(1) == 2 * output.size(1), "pairs width must be twice the output width");
    TORCH_CHECK(pairs.device() == output.device(), "pairs and output must be on the same device");
    TORCH_CHECK(output.size(1) % 2 == 0, "output width must be even");

    constexpr int threads = 256;
    const int rows = static_cast<int>(output.size(0));
    const int intermediate_pairs = static_cast<int>(output.size(1) / 2);
    const int64_t elements = static_cast<int64_t>(rows) * intermediate_pairs;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    row_halves_pair_map_bf16x2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162 *>(pairs.data_ptr<at::BFloat16>()),
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

static __global__ void route_weighted_padded_pack_bf16x2_kernel(
    const __nv_bfloat162 *received,
    const int64_t *route_padded_rows,
    const float *route_weights,
    __nv_bfloat162 *output,
    int received_rows,
    int route_slots,
    int hidden_pairs
) {
    const int64_t edge_feature = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(received_rows) * route_slots * hidden_pairs;
    if (edge_feature >= elements) {
        return;
    }
    const int feature_pair = static_cast<int>(edge_feature % hidden_pairs);
    const int64_t edge = edge_feature / hidden_pairs;
    const int received_row = static_cast<int>(edge / route_slots);
    const int64_t destination_row = route_padded_rows[edge];
    if (destination_row < 0) {
        return;
    }
    const float2 value = __bfloat1622float2(received[static_cast<int64_t>(received_row) * hidden_pairs + feature_pair]);
    const float weight = route_weights[edge];
    output[destination_row * hidden_pairs + feature_pair] = __floats2bfloat162_rn(
        generated_edge_cotangent_map(value.x, weight),
        generated_edge_cotangent_map(value.y, weight)
    );
}

static void route_weighted_padded_pack_bf16_out(
    const at::Tensor &received,
    const at::Tensor &route_padded_rows,
    const at::Tensor &route_weights,
    const at::Tensor &output
) {
    TORCH_CHECK(received.is_cuda() && route_padded_rows.is_cuda() && route_weights.is_cuda() && output.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(received.is_contiguous() && route_padded_rows.is_contiguous()
                    && route_weights.is_contiguous() && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(received.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
                "received and output must be BF16");
    TORCH_CHECK(route_padded_rows.scalar_type() == at::kLong && route_weights.scalar_type() == at::kFloat,
                "route rows must be int64 and route weights must be FP32");
    TORCH_CHECK(received.dim() == 2 && route_padded_rows.dim() == 2 && route_weights.dim() == 2
                    && output.dim() == 2,
                "invalid tensor ranks for weighted padded pack");
    TORCH_CHECK(route_padded_rows.sizes() == route_weights.sizes(), "route rows and weights must match");
    TORCH_CHECK(route_padded_rows.size(0) == received.size(0), "one route row is required per received row");
    TORCH_CHECK(received.size(1) == output.size(1) && output.size(1) % 2 == 0,
                "received and output hidden dimensions must match and be even");
    TORCH_CHECK(received.device() == route_padded_rows.device() && received.device() == route_weights.device()
                    && received.device() == output.device(),
                "all tensors must be on the same device");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        output.data_ptr<at::BFloat16>(),
        0,
        output.numel() * output.element_size(),
        stream
    ));
    constexpr int threads = 256;
    const int received_rows = static_cast<int>(received.size(0));
    const int route_slots = static_cast<int>(route_padded_rows.size(1));
    const int hidden_pairs = static_cast<int>(output.size(1) / 2);
    const int64_t elements = static_cast<int64_t>(received_rows) * route_slots * hidden_pairs;
    if (elements == 0) {
        return;
    }
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    route_weighted_padded_pack_bf16x2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162 *>(received.data_ptr<at::BFloat16>()),
        route_padded_rows.data_ptr<int64_t>(),
        route_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
        received_rows,
        route_slots,
        hidden_pairs
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __global__ void row_halves_pair_map_vjp_bf16x2_kernel(
    const __nv_bfloat162 *pairs,
    const __nv_bfloat162 *cotangent,
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
    const float2 left = __bfloat1622float2(pairs[static_cast<int64_t>(row) * 2 * intermediate_pairs + pair]);
    const float2 right = __bfloat1622float2(
        pairs[static_cast<int64_t>(row) * 2 * intermediate_pairs + intermediate_pairs + pair]
    );
    const float2 dy = __bfloat1622float2(cotangent[index]);
    output[static_cast<int64_t>(row) * 2 * intermediate_pairs + pair] = __floats2bfloat162_rn(
        generated_pair_left_vjp_uniform(left.x, right.x, dy.x),
        generated_pair_left_vjp_uniform(left.y, right.y, dy.y)
    );
    output[static_cast<int64_t>(row) * 2 * intermediate_pairs + intermediate_pairs + pair] =
        __floats2bfloat162_rn(
            generated_pair_right_vjp_uniform(left.x, right.x, dy.x),
            generated_pair_right_vjp_uniform(left.y, right.y, dy.y)
        );
}

static void row_halves_pair_map_vjp_bf16_out(
    const at::Tensor &pairs,
    const at::Tensor &cotangent,
    const at::Tensor &output
) {
    TORCH_CHECK(pairs.is_cuda() && cotangent.is_cuda() && output.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(pairs.is_contiguous() && cotangent.is_contiguous() && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(pairs.scalar_type() == at::kBFloat16 && cotangent.scalar_type() == at::kBFloat16
                    && output.scalar_type() == at::kBFloat16,
                "all tensors must be BF16");
    TORCH_CHECK(pairs.dim() == 2 && cotangent.dim() == 2 && output.dim() == 2,
                "pair VJP tensors must have rank two");
    TORCH_CHECK(output.sizes() == pairs.sizes() && pairs.size(0) == cotangent.size(0)
                    && pairs.size(1) == 2 * cotangent.size(1) && cotangent.size(1) % 2 == 0,
                "pair VJP requires [rows,2I], [rows,I], [rows,2I] with even I");
    TORCH_CHECK(pairs.device() == cotangent.device() && pairs.device() == output.device(),
                "all tensors must be on the same device");

    constexpr int threads = 256;
    const int rows = static_cast<int>(pairs.size(0));
    const int intermediate_pairs = static_cast<int>(cotangent.size(1) / 2);
    const int64_t elements = static_cast<int64_t>(rows) * intermediate_pairs;
    if (elements == 0) {
        return;
    }
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    row_halves_pair_map_vjp_bf16x2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162 *>(pairs.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat162 *>(cotangent.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
        rows,
        intermediate_pairs
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __global__ void route_weight_feature_fold_kernel(
    const __nv_bfloat162 *edge_output,
    const __nv_bfloat162 *received_cotangent,
    const int64_t *route_padded_rows,
    float *output,
    int received_rows,
    int route_slots,
    int hidden_pairs
) {
    const int64_t edge = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (edge >= static_cast<int64_t>(received_rows) * route_slots) {
        return;
    }
    const int received_row = static_cast<int>(edge / route_slots);
    const int64_t padded_row = route_padded_rows[edge];
    float state = 0.0f;
    if (padded_row >= 0) {
        for (int feature_pair = 0; feature_pair < hidden_pairs; ++feature_pair) {
            const float2 value = __bfloat1622float2(edge_output[padded_row * hidden_pairs + feature_pair]);
            const float2 cotangent = __bfloat1622float2(
                received_cotangent[static_cast<int64_t>(received_row) * hidden_pairs + feature_pair]
            );
            state = generated_route_weight_fold_update(
                state,
                generated_route_weight_fold_contribution(value.x, cotangent.x)
            );
            state = generated_route_weight_fold_update(
                state,
                generated_route_weight_fold_contribution(value.y, cotangent.y)
            );
        }
    }
    output[edge] = state;
}

static void route_weight_feature_fold_out(
    const at::Tensor &edge_output,
    const at::Tensor &received_cotangent,
    const at::Tensor &route_padded_rows,
    const at::Tensor &output
) {
    TORCH_CHECK(edge_output.is_cuda() && received_cotangent.is_cuda() && route_padded_rows.is_cuda()
                    && output.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(edge_output.is_contiguous() && received_cotangent.is_contiguous()
                    && route_padded_rows.is_contiguous() && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(edge_output.scalar_type() == at::kBFloat16 && received_cotangent.scalar_type() == at::kBFloat16,
                "feature Fold payloads must be BF16");
    TORCH_CHECK(route_padded_rows.scalar_type() == at::kLong && output.scalar_type() == at::kFloat,
                "route rows must be int64 and Fold output must be FP32");
    TORCH_CHECK(edge_output.dim() == 2 && received_cotangent.dim() == 2
                    && route_padded_rows.dim() == 2 && output.dim() == 2,
                "invalid tensor ranks for route-weight feature Fold");
    TORCH_CHECK(received_cotangent.size(0) == route_padded_rows.size(0)
                    && output.sizes() == route_padded_rows.sizes(),
                "route-weight Fold metadata and output shapes must match");
    TORCH_CHECK(edge_output.size(1) == received_cotangent.size(1) && edge_output.size(1) % 2 == 0,
                "feature dimensions must match and be even");
    TORCH_CHECK(edge_output.device() == received_cotangent.device()
                    && edge_output.device() == route_padded_rows.device() && edge_output.device() == output.device(),
                "all tensors must be on the same device");

    constexpr int threads = 256;
    const int received_rows = static_cast<int>(received_cotangent.size(0));
    const int route_slots = static_cast<int>(route_padded_rows.size(1));
    const int64_t edges = static_cast<int64_t>(received_rows) * route_slots;
    if (edges == 0) {
        return;
    }
    const int blocks = static_cast<int>((edges + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    route_weight_feature_fold_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162 *>(edge_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat162 *>(received_cotangent.data_ptr<at::BFloat16>()),
        route_padded_rows.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        received_rows,
        route_slots,
        static_cast<int>(edge_output.size(1) / 2)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __global__ void indexed_ordered_source_fold_bf16x2_kernel(
    const __nv_bfloat162 *values,
    const int64_t *row_indices,
    __nv_bfloat162 *output,
    int source_rows,
    int route_slots,
    int hidden_pairs
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(source_rows) * hidden_pairs) {
        return;
    }
    const int source_row = static_cast<int>(index / hidden_pairs);
    const int feature_pair = static_cast<int>(index % hidden_pairs);
    float2 state = make_float2(0.0f, 0.0f);
    for (int route_slot = 0; route_slot < route_slots; ++route_slot) {
        const int64_t value_row = row_indices[static_cast<int64_t>(source_row) * route_slots + route_slot];
        if (value_row >= 0) {
            const float2 contribution = __bfloat1622float2(values[value_row * hidden_pairs + feature_pair]);
            state.x = generated_source_input_fold_update(state.x, contribution.x);
            state.y = generated_source_input_fold_update(state.y, contribution.y);
        }
    }
    output[index] = __floats2bfloat162_rn(state.x, state.y);
}

static void indexed_ordered_source_fold_bf16_out(
    const at::Tensor &values,
    const at::Tensor &row_indices,
    const at::Tensor &output
) {
    TORCH_CHECK(values.is_cuda() && row_indices.is_cuda() && output.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(values.is_contiguous() && row_indices.is_contiguous() && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(values.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
                "values and output must be BF16");
    TORCH_CHECK(row_indices.scalar_type() == at::kLong, "row indices must be int64");
    TORCH_CHECK(values.dim() == 2 && row_indices.dim() == 2 && output.dim() == 2,
                "invalid tensor ranks for source Fold");
    TORCH_CHECK(output.size(0) == row_indices.size(0) && output.size(1) == values.size(1)
                    && output.size(1) % 2 == 0,
                "source Fold output and metadata dimensions disagree");
    TORCH_CHECK(values.device() == row_indices.device() && values.device() == output.device(),
                "all tensors must be on the same device");

    constexpr int threads = 256;
    const int source_rows = static_cast<int>(output.size(0));
    const int route_slots = static_cast<int>(row_indices.size(1));
    const int hidden_pairs = static_cast<int>(output.size(1) / 2);
    const int64_t elements = static_cast<int64_t>(source_rows) * hidden_pairs;
    if (elements == 0) {
        return;
    }
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    indexed_ordered_source_fold_bf16x2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162 *>(values.data_ptr<at::BFloat16>()),
        row_indices.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
        source_rows,
        route_slots,
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
static __global__ void indexed_weighted_ordered_fold_bf16x2_kernel(
    const __nv_bfloat162 *values,
    const int64_t *row_indices,
    const float *weights,
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
        const int64_t fold_index = static_cast<int64_t>(token) * topk + route_slot;
        const int64_t value_row = row_indices[fold_index];
        if (value_row >= 0) {
            const float2 item = __bfloat1622float2(
                values[value_row * hidden_pairs + feature_pair]
            );
            const float weight = weights[fold_index];
            if constexpr (explicit_rounding) {
                value.x = generated_fold_update(value.x, generated_fold_contribution(item.x, weight));
                value.y = generated_fold_update(value.y, generated_fold_contribution(item.y, weight));
            } else {
                value.x = generated_fold_update_relaxed(
                    value.x,
                    generated_fold_contribution_relaxed(item.x, weight)
                );
                value.y = generated_fold_update_relaxed(
                    value.y,
                    generated_fold_contribution_relaxed(item.y, weight)
                );
            }
        }
    }
    output[index] = __floats2bfloat162_rn(value.x, value.y);
}

static void indexed_weighted_ordered_fold_bf16_out_impl(
    const at::Tensor &values,
    const at::Tensor &row_indices,
    const at::Tensor &weights,
    const at::Tensor &output,
    bool explicit_rounding
) {
    TORCH_CHECK(values.is_cuda() && row_indices.is_cuda() && weights.is_cuda() && output.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(values.is_contiguous() && row_indices.is_contiguous() && weights.is_contiguous()
                    && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(values.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
                "values and output must be BF16");
    TORCH_CHECK(row_indices.scalar_type() == at::kLong, "row_indices must be int64");
    TORCH_CHECK(weights.scalar_type() == at::kFloat, "weights must be FP32");
    TORCH_CHECK(values.dim() == 2 && row_indices.dim() == 2 && weights.dim() == 2
                    && output.dim() == 2,
                "invalid tensor ranks for indexed ordered Fold");
    TORCH_CHECK(row_indices.sizes() == weights.sizes(), "row indices and weights must have identical shapes");
    TORCH_CHECK(output.size(0) == row_indices.size(0), "output rows must match Fold metadata");
    TORCH_CHECK(output.size(1) == values.size(1), "feature dimensions must match");
    TORCH_CHECK(output.size(1) % 2 == 0, "hidden dimension must be even for BF16x2 merge");
    TORCH_CHECK(values.device() == row_indices.device() && values.device() == weights.device()
                    && values.device() == output.device(),
                "all tensors must be on the same device");

    constexpr int threads = 256;
    const int tokens = static_cast<int>(output.size(0));
    const int topk = static_cast<int>(row_indices.size(1));
    const int hidden_pairs = static_cast<int>(output.size(1) / 2);
    const int64_t elements = static_cast<int64_t>(tokens) * hidden_pairs;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (explicit_rounding) {
        indexed_weighted_ordered_fold_bf16x2_kernel<true><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162 *>(values.data_ptr<at::BFloat16>()),
            row_indices.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
            tokens,
            topk,
            hidden_pairs
        );
    } else {
        indexed_weighted_ordered_fold_bf16x2_kernel<false><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162 *>(values.data_ptr<at::BFloat16>()),
            row_indices.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
            tokens,
            topk,
            hidden_pairs
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void indexed_weighted_ordered_fold_bf16_out(
    const at::Tensor &values,
    const at::Tensor &row_indices,
    const at::Tensor &weights,
    const at::Tensor &output
) {
    indexed_weighted_ordered_fold_bf16_out_impl(values, row_indices, weights, output, true);
}

static void indexed_weighted_ordered_fold_relaxed_bf16_out(
    const at::Tensor &values,
    const at::Tensor &row_indices,
    const at::Tensor &weights,
    const at::Tensor &output
) {
    indexed_weighted_ordered_fold_bf16_out_impl(values, row_indices, weights, output, false);
}

static __global__ void indirect_weighted_fold_base_map_kernel(
    const __nv_bfloat16 *values,
    const int64_t *row_indices,
    const float *weights,
    const int64_t *source_rows,
    const __nv_bfloat16 *base,
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
    const int64_t source_row = source_rows[token];
    float folded = 0.0f;
    if (source_row >= 0) {
        #pragma unroll 1
        for (int route_slot = 0; route_slot < topk; ++route_slot) {
            const int64_t fold_index = source_row * topk + route_slot;
            const int64_t value_row = row_indices[fold_index];
            if (value_row >= 0) {
                const float item = __bfloat162float(values[value_row * hidden + feature]);
                folded = generated_fold_update(
                    folded,
                    generated_fold_contribution(item, weights[fold_index])
                );
            }
        }
    }
    const __nv_bfloat16 folded_bf16 = __float2bfloat16_rn(folded);
    const float combined_value = generated_post_fold_map(
        __bfloat162float(folded_bf16),
        __bfloat162float(base[index])
    );
    output[index] = __float2bfloat16_rn(combined_value);
}

static void indirect_weighted_fold_base_map_out(
    const at::Tensor &values,
    const at::Tensor &row_indices,
    const at::Tensor &weights,
    const at::Tensor &source_rows,
    const at::Tensor &base,
    const at::Tensor &output
) {
    TORCH_CHECK(values.is_cuda() && row_indices.is_cuda() && weights.is_cuda()
                    && source_rows.is_cuda() && base.is_cuda() && output.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(values.is_contiguous() && row_indices.is_contiguous() && weights.is_contiguous()
                    && source_rows.is_contiguous() && base.is_contiguous() && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(values.scalar_type() == at::kBFloat16 && base.scalar_type() == at::kBFloat16
                    && output.scalar_type() == at::kBFloat16,
                "values, base, and output must be BF16");
    TORCH_CHECK(row_indices.scalar_type() == at::kLong && source_rows.scalar_type() == at::kLong,
                "row_indices and source_rows must be int64");
    TORCH_CHECK(weights.scalar_type() == at::kFloat, "weights must be FP32");
    TORCH_CHECK(values.dim() == 2 && row_indices.dim() == 2 && weights.dim() == 2
                    && source_rows.dim() == 1 && base.dim() == 2 && output.dim() == 2,
                "invalid tensor ranks for indirect ordered Fold and base Map");
    TORCH_CHECK(row_indices.sizes() == weights.sizes(), "row indices and weights must have identical shapes");
    TORCH_CHECK(output.sizes() == base.sizes(), "output and base must have identical shapes");
    TORCH_CHECK(output.size(0) == source_rows.size(0), "output rows must match source_rows");
    TORCH_CHECK(output.size(1) == values.size(1), "feature dimensions must match");
    TORCH_CHECK(values.device() == row_indices.device() && values.device() == weights.device()
                    && values.device() == source_rows.device()
                    && values.device() == base.device() && values.device() == output.device(),
                "all tensors must be on the same device");

    constexpr int threads = 256;
    const int tokens = static_cast<int>(output.size(0));
    const int topk = static_cast<int>(row_indices.size(1));
    const int hidden = static_cast<int>(output.size(1));
    const int64_t elements = static_cast<int64_t>(tokens) * hidden;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    indirect_weighted_fold_base_map_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(values.data_ptr<at::BFloat16>()),
        row_indices.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        source_rows.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16 *>(base.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>()),
        tokens,
        topk,
        hidden
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __global__ void partitioned_ordered_fold_base_map_bf16x2_kernel(
    const __nv_bfloat162 *partials,
    const int64_t *partition_rows,
    const __nv_bfloat162 *base,
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
    float2 value = make_float2(0.0f, 0.0f);
    #pragma unroll 1
    for (int rank = 0; rank < ranks; ++rank) {
        const int64_t row = partition_rows[static_cast<int64_t>(rank) * tokens + token];
        if (row >= 0) {
            const float2 partial = __bfloat1622float2(
                partials[row * hidden_pairs + feature_pair]
            );
            value.x = generated_fold_update(value.x, partial.x);
            value.y = generated_fold_update(value.y, partial.y);
        }
    }
    const float2 base_value = __bfloat1622float2(base[index]);
    output[index] = __floats2bfloat162_rn(
        generated_post_fold_map(value.x, base_value.x),
        generated_post_fold_map(value.y, base_value.y)
    );
}

static void partitioned_ordered_fold_base_map_bf16_out(
    const at::Tensor &partials,
    const at::Tensor &partition_rows,
    const at::Tensor &base,
    const at::Tensor &output
) {
    TORCH_CHECK(partials.is_cuda() && partition_rows.is_cuda() && base.is_cuda()
                    && output.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(partials.is_contiguous() && partition_rows.is_contiguous() && base.is_contiguous()
                    && output.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(partials.scalar_type() == at::kBFloat16 && base.scalar_type() == at::kBFloat16
                    && output.scalar_type() == at::kBFloat16,
                "partials, base, and output must be BF16");
    TORCH_CHECK(partition_rows.scalar_type() == at::kLong, "partition_rows must be int64");
    TORCH_CHECK(partials.dim() == 2 && partition_rows.dim() == 2 && base.dim() == 2
                    && output.dim() == 2,
                "invalid tensor ranks for partitioned ordered Fold and base Map");
    TORCH_CHECK(output.sizes() == base.sizes(), "output and base must have identical shapes");
    TORCH_CHECK(partition_rows.size(1) == output.size(0), "partition_rows item count must match output");
    TORCH_CHECK(partials.size(1) == output.size(1), "feature dimensions must match");
    TORCH_CHECK(output.size(1) % 2 == 0, "hidden dimension must be even for BF16x2 merge");
    TORCH_CHECK(partials.device() == partition_rows.device() && partials.device() == base.device()
                    && partials.device() == output.device(),
                "all tensors must be on the same device");

    constexpr int threads = 256;
    const int ranks = static_cast<int>(partition_rows.size(0));
    const int tokens = static_cast<int>(output.size(0));
    const int hidden_pairs = static_cast<int>(output.size(1) / 2);
    const int64_t elements = static_cast<int64_t>(tokens) * hidden_pairs;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    partitioned_ordered_fold_base_map_bf16x2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162 *>(partials.data_ptr<at::BFloat16>()),
        partition_rows.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat162 *>(base.data_ptr<at::BFloat16>()),
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
        "generated_map_fold_program_sha256",
        &tile_lifetime::mok_gmm_probe::generated_map_fold_program_sha256
    );
    module.def(
        "generated_expert_training_program_sha256",
        &tile_lifetime::mok_gmm_probe::generated_expert_training_program_sha256
    );
    module.def(
        "generated_grouped_contract_event_sha256",
        &tile_lifetime::mok_gmm_probe::generated_grouped_contract_event_sha256
    );
    module.def(
        "grouped_contract_event_attributes",
        &tile_lifetime::mok_gmm_probe::grouped_contract_event_attributes
    );
    module.def(
        "grouped_gemm_out",
        &tile_lifetime::mok_gmm_probe::grouped_gemm_out,
        pybind11::arg("activations"),
        pybind11::arg("weights"),
        pybind11::arg("padded_counts"),
        pybind11::arg("output")
    );
    module.def(
        "adjacent_pair_map_bf16_out",
        &tile_lifetime::mok_gmm_probe::adjacent_pair_map_bf16_out,
        pybind11::arg("left"),
        pybind11::arg("right"),
        pybind11::arg("output")
    );
    module.def(
        "row_halves_pair_map_bf16_out",
        &tile_lifetime::mok_gmm_probe::row_halves_pair_map_bf16_out,
        pybind11::arg("pairs"),
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
        "route_weighted_padded_pack_bf16_out",
        &tile_lifetime::mok_gmm_probe::route_weighted_padded_pack_bf16_out,
        pybind11::arg("received"),
        pybind11::arg("route_padded_rows"),
        pybind11::arg("route_weights"),
        pybind11::arg("output")
    );
    module.def(
        "row_halves_pair_map_vjp_bf16_out",
        &tile_lifetime::mok_gmm_probe::row_halves_pair_map_vjp_bf16_out,
        pybind11::arg("pairs"),
        pybind11::arg("cotangent"),
        pybind11::arg("output")
    );
    module.def(
        "route_weight_feature_fold_out",
        &tile_lifetime::mok_gmm_probe::route_weight_feature_fold_out,
        pybind11::arg("edge_output"),
        pybind11::arg("received_cotangent"),
        pybind11::arg("route_padded_rows"),
        pybind11::arg("output")
    );
    module.def(
        "indexed_ordered_source_fold_bf16_out",
        &tile_lifetime::mok_gmm_probe::indexed_ordered_source_fold_bf16_out,
        pybind11::arg("values"),
        pybind11::arg("row_indices"),
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
        "indexed_weighted_ordered_fold_bf16_out",
        &tile_lifetime::mok_gmm_probe::indexed_weighted_ordered_fold_bf16_out,
        pybind11::arg("values"),
        pybind11::arg("row_indices"),
        pybind11::arg("weights"),
        pybind11::arg("output")
    );
    module.def(
        "indexed_weighted_ordered_fold_relaxed_bf16_out",
        &tile_lifetime::mok_gmm_probe::indexed_weighted_ordered_fold_relaxed_bf16_out,
        pybind11::arg("values"),
        pybind11::arg("row_indices"),
        pybind11::arg("weights"),
        pybind11::arg("output")
    );
    module.def(
        "indirect_weighted_fold_base_map_out",
        &tile_lifetime::mok_gmm_probe::indirect_weighted_fold_base_map_out,
        pybind11::arg("values"),
        pybind11::arg("row_indices"),
        pybind11::arg("weights"),
        pybind11::arg("source_rows"),
        pybind11::arg("base"),
        pybind11::arg("output")
    );
    module.def(
        "partitioned_ordered_fold_base_map_bf16_out",
        &tile_lifetime::mok_gmm_probe::partitioned_ordered_fold_base_map_bf16_out,
        pybind11::arg("partials"),
        pybind11::arg("partition_rows"),
        pybind11::arg("base"),
        pybind11::arg("output")
    );
}
