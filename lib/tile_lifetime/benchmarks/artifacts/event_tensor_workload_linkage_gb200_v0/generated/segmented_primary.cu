// Generated from RelationPlan + SegmentedContract + EventTensorPlan; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace {
constexpr int kSourceItemCount = 64;
constexpr int kSourceCount = 128;
constexpr int kSegmentCount = 8;
constexpr int kReductionDimension = 32;
constexpr int kOutputDimension = 16;
std::atomic<int> call_count{0};

__global__ void shuttle_segmented_contract(
    const float* source,
    const float* weight,
    const int* event_counts,
    const int* event_offsets,
    const int* edge_sources,
    float* output) {
  const int segment = blockIdx.x;
  const int count = event_counts[segment];
  const int begin = event_offsets[segment];
  const int end = event_offsets[segment + 1];
  if (count != end - begin) return;
  for (int item = threadIdx.x; item < count * kOutputDimension; item += blockDim.x) {
    const int local_edge = item / kOutputDimension;
    const int feature = item - local_edge * kOutputDimension;
    const int edge = begin + local_edge;
    const int source_row = edge_sources[edge];
    float accumulator = 0.0f;
    for (int reduction = 0; reduction < kReductionDimension; ++reduction) {
      accumulator = fmaf(
          source[source_row * kReductionDimension + reduction],
          weight[(segment * kReductionDimension + reduction) * kOutputDimension + feature],
          accumulator);
    }
    output[edge * kOutputDimension + feature] = accumulator;
  }
}

ffi::Error ShuttleSegmentedContract(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32, 2> source,
    ffi::Buffer<ffi::F32, 3> weight,
    ffi::Buffer<ffi::S32, 1> event_counts,
    ffi::Buffer<ffi::S32, 1> event_offsets,
    ffi::Buffer<ffi::S32, 1> edge_sources,
    ffi::Result<ffi::Buffer<ffi::F32, 2>> output) {
  shuttle_segmented_contract<<<kSegmentCount, 256, 0, stream>>>(
      source.typed_data(), weight.typed_data(), event_counts.typed_data(),
      event_offsets.typed_data(), edge_sources.typed_data(), output->typed_data());
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return ffi::Error::Internal(std::string("shuttle_segmented_contract: ") + cudaGetErrorString(status));
  }
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}

auto ShuttleSegmentedContractBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 3>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::F32, 2>>();
}
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    shuttle_event_segmented_primary,
    ShuttleSegmentedContract,
    ShuttleSegmentedContractBinding());

extern "C" int shuttle_event_segmented_primary_call_count() {
  return call_count.load(std::memory_order_relaxed);
}
