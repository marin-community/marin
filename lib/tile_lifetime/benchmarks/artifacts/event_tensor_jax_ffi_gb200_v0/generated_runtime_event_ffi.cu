// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from generic EventTensorPlan runtime tables; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {
std::atomic<int> call_count{0};
__global__ void shuttle_runtime_counted_event(
    const float* input,
    float* partials,
    float* output,
    const int* event_counts,
    const int* event_source_offsets,
    const int* event_sources,
    int event_count) {
  __shared__ int remaining;
  const int event_index = blockIdx.x;
  if (event_index >= event_count) return;
  const int producer_count = event_counts[event_index];
  const int source_begin = event_source_offsets[event_index];
  const int source_end = event_source_offsets[event_index + 1];
  if (producer_count < 0 || producer_count != source_end - source_begin || producer_count > blockDim.x) {
    if (threadIdx.x == 0) output[event_index] = NAN;
    return;
  }
  if (producer_count == 0) {
    if (threadIdx.x == 0) output[event_index] = 0.0f;
    return;
  }
  if (threadIdx.x == 0) remaining = producer_count;
  __syncthreads();
  if (threadIdx.x >= producer_count) return;

  const int source = event_sources[source_begin + threadIdx.x];
  partials[source] = input[source];
  __threadfence_block();
  const int prior_remaining = atomicSub(&remaining, 1);
  if (prior_remaining == 1) {
    float accumulator = 0.0f;
    for (int index = source_begin; index < source_end; ++index) {
      accumulator = __fadd_rn(accumulator, partials[event_sources[index]]);
    }
    output[event_index] = accumulator;
  }
}

ffi::Error ShuttleRuntimeEventRegion(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32, 1> input,
    ffi::Buffer<ffi::S32, 1> event_counts,
    ffi::Buffer<ffi::S32, 1> event_source_offsets,
    ffi::Buffer<ffi::S32, 1> event_sources,
    ffi::Result<ffi::Buffer<ffi::F32, 1>> partials,
    ffi::Result<ffi::Buffer<ffi::F32, 1>> output,
    std::int64_t maximum_count,
    std::int64_t event_count) {
  if (maximum_count <= 0 || maximum_count > 1024) {
    return ffi::Error::InvalidArgument("maximum_count must be in [1, 1024]");
  }
  if (event_count <= 0) {
    return ffi::Error::InvalidArgument("event_count must be positive");
  }
  int threads = 32;
  while (threads < maximum_count) threads *= 2;
  shuttle_runtime_counted_event<<<event_count, threads, 0, stream>>>(
      input.typed_data(),
      partials->typed_data(),
      output->typed_data(),
      event_counts.typed_data(),
      event_source_offsets.typed_data(),
      event_sources.typed_data(),
      static_cast<int>(event_count));
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("shuttle_runtime_counted_event: ") + cudaGetErrorString(status));
  }
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}

auto ShuttleRuntimeEventRegionBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::F32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::F32, 1>>()
      .Ret<ffi::Buffer<ffi::F32, 1>>()
      .Attr<std::int64_t>("maximum_count")
      .Attr<std::int64_t>("event_count");
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    shuttle_event_tensor_runtime_gpu_replay_v1,
    ShuttleRuntimeEventRegion,
    ShuttleRuntimeEventRegionBinding());

extern "C" int shuttle_event_tensor_runtime_gpu_replay_v1_call_count() {
  return call_count.load(std::memory_order_relaxed);
}
