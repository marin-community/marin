// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from a generic phased Contract/Fold/Contract task graph; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <cmath>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {
std::atomic<int> call_count{0};
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

ffi::Error ShuttlePhasedPipelineRegion(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32, 2> query,
    ffi::Buffer<ffi::F32, 3> key,
    ffi::Buffer<ffi::F32, 2> value,
    ffi::Result<ffi::Buffer<ffi::F32, 1>> output,
    std::int64_t generation_count,
    std::int64_t pipeline_depth,
    std::int64_t dimension) {
  if (generation_count <= 0 || dimension <= 0) {
    return ffi::Error::InvalidArgument("generation_count and dimension must be positive");
  }
  if (pipeline_depth <= 0 || pipeline_depth > kMaximumPipelineDepth) {
    return ffi::Error::InvalidArgument("pipeline_depth must be in [1, 32]");
  }
  shuttle_phased_contract_fold_pipeline<<<1, 96, 0, stream>>>(
      query.typed_data(),
      key.typed_data(),
      value.typed_data(),
      output->typed_data(),
      static_cast<int>(generation_count),
      static_cast<int>(pipeline_depth),
      static_cast<int>(dimension));
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("shuttle_phased_contract_fold_pipeline: ") + cudaGetErrorString(status));
  }
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}

auto ShuttlePhasedPipelineRegionBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 3>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Ret<ffi::Buffer<ffi::F32, 1>>()
      .Attr<std::int64_t>("generation_count")
      .Attr<std::int64_t>("pipeline_depth")
      .Attr<std::int64_t>("dimension");
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    shuttle_event_tensor_phased_gpu_replay_v1,
    ShuttlePhasedPipelineRegion,
    ShuttlePhasedPipelineRegionBinding());

extern "C" int shuttle_event_tensor_phased_gpu_replay_v1_call_count() {
  return call_count.load(std::memory_order_relaxed);
}
