// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from a generic phased Contract/Fold/Contract task graph; do not edit.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int kMaximumPipelineDepth = 32;

__device__ __forceinline__ void wait_for_generation(int* address, int generation) {
  while (atomicAdd(address, 0) != generation) __nanosleep(64);
}

__global__ void shuttle_phased_contract_fold_pipeline(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int generation_count,
    int pipeline_depth,
    int dimension) {
  __shared__ int first_ready[kMaximumPipelineDepth];
  __shared__ int state_ready[kMaximumPipelineDepth];
  __shared__ int slot_reusable[kMaximumPipelineDepth];
  __shared__ float score[kMaximumPipelineDepth];
  __shared__ float state_max[kMaximumPipelineDepth];
  __shared__ float state_sum[kMaximumPipelineDepth];
  __shared__ float state_weighted[kMaximumPipelineDepth];

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (threadIdx.x < pipeline_depth) {
    first_ready[threadIdx.x] = -1;
    state_ready[threadIdx.x] = -1;
    slot_reusable[threadIdx.x] = 0;
  }
  __syncthreads();

  if (warp == 0 && lane < pipeline_depth) {
    const int slot = lane;
    for (int generation = 0; generation < generation_count; ++generation) {
      wait_for_generation(&slot_reusable[slot], generation);
      float accumulator = 0.0f;
      for (int index = 0; index < dimension; ++index) {
        accumulator = fmaf(
            query[generation * dimension + index],
            key[(generation * pipeline_depth + slot) * dimension + index],
            accumulator);
      }
      score[slot] = accumulator;
      __threadfence_block();
      atomicExch(&first_ready[slot], generation);
    }
  } else if (warp == 1 && lane < pipeline_depth) {
    const int slot = lane;
    for (int generation = 0; generation < generation_count; ++generation) {
      wait_for_generation(&first_ready[slot], generation);
      const float local_score = score[slot];
      state_max[slot] = local_score;
      state_sum[slot] = 1.0f;
      state_weighted[slot] = value[generation * pipeline_depth + slot];
      __threadfence_block();
      atomicExch(&state_ready[slot], generation);
    }
  } else if (warp == 2 && lane == 0) {
    for (int generation = 0; generation < generation_count; ++generation) {
      for (int slot = 0; slot < pipeline_depth; ++slot) {
        wait_for_generation(&state_ready[slot], generation);
      }
      float running_max = -INFINITY;
      float running_sum = 0.0f;
      float running_weighted = 0.0f;
      for (int slot = 0; slot < pipeline_depth; ++slot) {
        const float next_max = fmaxf(running_max, state_max[slot]);
        const float prior_scale = expf(running_max - next_max);
        const float next_scale = expf(state_max[slot] - next_max);
        running_sum = prior_scale * running_sum + next_scale * state_sum[slot];
        running_weighted = prior_scale * running_weighted + next_scale * state_weighted[slot];
        running_max = next_max;
      }
      output[generation] = running_weighted / running_sum;
      __threadfence_block();
      for (int slot = 0; slot < pipeline_depth; ++slot) {
        atomicExch(&slot_reusable[slot], generation + 1);
      }
    }
  }
}

}  // namespace

void run_phased_contract_fold_pipeline_out(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda() && output.is_cuda(),
              "pipeline tensors must be CUDA tensors");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous() && output.is_contiguous(),
              "pipeline tensors must be contiguous");
  TORCH_CHECK(query.scalar_type() == torch::kFloat32 && key.scalar_type() == torch::kFloat32 &&
              value.scalar_type() == torch::kFloat32 && output.scalar_type() == torch::kFloat32,
              "pipeline tensors must be FP32");
  TORCH_CHECK(query.dim() == 2 && key.dim() == 3 && value.dim() == 2 && output.dim() == 1,
              "pipeline tensor ranks are invalid");
  const int generations = query.size(0);
  const int dimension = query.size(1);
  const int depth = key.size(1);
  TORCH_CHECK(depth > 0 && depth <= kMaximumPipelineDepth, "pipeline depth must be in [1, 32]");
  TORCH_CHECK(key.size(0) == generations && key.size(2) == dimension, "key extent mismatch");
  TORCH_CHECK(value.size(0) == generations && value.size(1) == depth, "value extent mismatch");
  TORCH_CHECK(output.numel() == generations, "output extent mismatch");
  const c10::cuda::CUDAGuard device_guard(query.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shuttle_phased_contract_fold_pipeline<<<1, 96, 0, stream>>>(
      query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(), output.data_ptr<float>(),
      generations, depth, dimension);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run_phased_contract_fold_pipeline_out", &run_phased_contract_fold_pipeline_out);
}
