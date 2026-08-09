// Generated from Contract + Fold + DomainRestriction + EventTensorPlan; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <cmath>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace {
constexpr int kRowTileCount = 2;
constexpr int kPartitionCount = 4;
constexpr int kPipelineDepth = 2;
constexpr int kQueryTile = 4;
constexpr int kKeyValueTile = 4;
constexpr int kReductionDimension = 16;
constexpr int kValueDimension = 8;
constexpr float kScoreScale = 0.25f;
constexpr int kSharedBytes = 768;
std::atomic<int> call_count{0};

__global__ void shuttle_streaming_contract_fold(
    const float* query,
    const float* key,
    const float* value,
    const int* domain_valid,
    float* output) {
  extern __shared__ float staged[];
  float* staged_key = staged;
  float* staged_value = staged + kPipelineDepth * kKeyValueTile * kReductionDimension;
  __shared__ int slot_generation[kPipelineDepth];
  __shared__ int generation_valid;
  const int row_tile = blockIdx.x;
  if (threadIdx.x < kPipelineDepth) slot_generation[threadIdx.x] = 0;
  __syncthreads();

  float row_max = -INFINITY;
  float row_sum = 0.0f;
  float weighted[kValueDimension];
  if (threadIdx.x < kQueryTile) {
    for (int feature = 0; feature < kValueDimension; ++feature) weighted[feature] = 0.0f;
  }

  for (int partition = 0; partition < kPartitionCount; ++partition) {
    const int slot = partition % kPipelineDepth;
    const int generation = partition / kPipelineDepth;
    if (threadIdx.x == 0) generation_valid = slot_generation[slot] == generation;
    __syncthreads();
    if (!generation_valid) return;
    const int key_items = kKeyValueTile * kReductionDimension;
    for (int index = threadIdx.x; index < key_items; index += blockDim.x) {
      staged_key[slot * key_items + index] =
          key[((row_tile * kPartitionCount + partition) * kKeyValueTile * kReductionDimension) + index];
    }
    const int value_items = kKeyValueTile * kValueDimension;
    for (int index = threadIdx.x; index < value_items; index += blockDim.x) {
      staged_value[slot * value_items + index] =
          value[((row_tile * kPartitionCount + partition) * kKeyValueTile * kValueDimension) + index];
    }
    // Physical realization of key_value_stage -> QK/PV acquire readiness.
    __syncthreads();

    if (threadIdx.x < kQueryTile) {
      const int query_row = threadIdx.x;
      float score[kKeyValueTile];
      float tile_max = -INFINITY;
      for (int key_row = 0; key_row < kKeyValueTile; ++key_row) {
        const int valid_index =
            ((row_tile * kQueryTile + query_row) * kPartitionCount + partition) * kKeyValueTile + key_row;
        float accumulator = 0.0f;
        if (domain_valid[valid_index] != 0) {
          for (int reduction = 0; reduction < kReductionDimension; ++reduction) {
            accumulator = fmaf(
                query[(row_tile * kQueryTile + query_row) * kReductionDimension + reduction],
                staged_key[(slot * kKeyValueTile + key_row) * kReductionDimension + reduction],
                accumulator);
          }
          accumulator *= kScoreScale;
        } else {
          accumulator = -INFINITY;
        }
        score[key_row] = accumulator;
        tile_max = fmaxf(tile_max, accumulator);
      }
      const float next_max = fmaxf(row_max, tile_max);
      const float prior_scale = row_sum > 0.0f ? expf(row_max - next_max) : 0.0f;
      float tile_sum = 0.0f;
      for (int feature = 0; feature < kValueDimension; ++feature) weighted[feature] *= prior_scale;
      for (int key_row = 0; key_row < kKeyValueTile; ++key_row) {
        const float probability = isfinite(score[key_row]) ? expf(score[key_row] - next_max) : 0.0f;
        tile_sum += probability;
        for (int feature = 0; feature < kValueDimension; ++feature) {
          weighted[feature] = fmaf(
              probability,
              staged_value[(slot * kKeyValueTile + key_row) * kValueDimension + feature],
              weighted[feature]);
        }
      }
      row_sum = prior_scale * row_sum + tile_sum;
      row_max = next_max;
    }

    // Physical last-consumer release before the circular slot is reused.
    __syncthreads();
    if (threadIdx.x == 0) slot_generation[slot] = generation + 1;
    __syncthreads();
  }
  if (threadIdx.x < kQueryTile) {
    const int query_row = threadIdx.x;
    for (int feature = 0; feature < kValueDimension; ++feature) {
      output[(row_tile * kQueryTile + query_row) * kValueDimension + feature] = weighted[feature] / row_sum;
    }
  }
}

ffi::Error ShuttleStreamingContractFold(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32, 3> query,
    ffi::Buffer<ffi::F32, 4> key,
    ffi::Buffer<ffi::F32, 4> value,
    ffi::Buffer<ffi::S32, 4> domain_valid,
    ffi::Result<ffi::Buffer<ffi::F32, 3>> output) {
  shuttle_streaming_contract_fold<<<kRowTileCount, 128, kSharedBytes, stream>>>(
      query.typed_data(), key.typed_data(), value.typed_data(),
      domain_valid.typed_data(), output->typed_data());
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return ffi::Error::Internal(std::string("shuttle_streaming_contract_fold: ") + cudaGetErrorString(status));
  }
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}

auto ShuttleStreamingContractFoldBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::F32, 3>>()
      .Arg<ffi::Buffer<ffi::F32, 4>>()
      .Arg<ffi::Buffer<ffi::F32, 4>>()
      .Arg<ffi::Buffer<ffi::S32, 4>>()
      .Ret<ffi::Buffer<ffi::F32, 3>>();
}
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    shuttle_event_streaming_primary,
    ShuttleStreamingContractFold,
    ShuttleStreamingContractFoldBinding());

extern "C" int shuttle_event_streaming_primary_call_count() {
  return call_count.load(std::memory_order_relaxed);
}
