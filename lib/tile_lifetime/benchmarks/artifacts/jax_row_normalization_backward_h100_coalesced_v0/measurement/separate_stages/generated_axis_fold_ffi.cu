// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from generic rank-two Map/Fold semantics; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {
std::atomic<int> call_count{0};

constexpr int kProgram0Rows = 2048;
constexpr int kProgram0Columns = 4096;
constexpr int kProgram0Threads = 256;

__global__ __launch_bounds__(kProgram0Threads) void ShuttleAxisFoldKernel0(const __nv_bfloat16* primal, float* output) {
  __shared__ float shared_sum_square[kProgram0Threads];
  const int group = blockIdx.x;
  if (group >= kProgram0Rows) return;
  float local_sum_square = 0.0f;
  for (int reduction_index = threadIdx.x; reduction_index < kProgram0Columns;
       reduction_index += kProgram0Threads) {
    const int row = group;
    const int column = reduction_index;
    local_sum_square = __fadd_rn(local_sum_square, __fmul_rn(__bfloat162float(primal[row * kProgram0Columns + column]), __bfloat162float(primal[row * kProgram0Columns + column])));
  }
  shared_sum_square[threadIdx.x] = local_sum_square;
  __syncthreads();
  for (int stride = kProgram0Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sum_square[threadIdx.x] = __fadd_rn(shared_sum_square[threadIdx.x], shared_sum_square[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[group] = rsqrtf(__fadd_rn((shared_sum_square[0] / 4096.0f), 9.99999974e-06f));
  }
}

constexpr int kProgram1Rows = 2048;
constexpr int kProgram1Columns = 4096;
constexpr int kProgram1Threads = 256;

__global__ __launch_bounds__(kProgram1Threads) void ShuttleAxisFoldKernel1(const __nv_bfloat16* primal, const __nv_bfloat16* feature_scale, const __nv_bfloat16* output_cotangent, const float* inverse_scale, __nv_bfloat16* output) {
  __shared__ float shared_correlation[kProgram1Threads];
  const int group = blockIdx.x;
  if (group >= kProgram1Rows) return;
  float local_correlation = 0.0f;
  for (int reduction_index = threadIdx.x; reduction_index < kProgram1Columns;
       reduction_index += kProgram1Threads) {
    const int row = group;
    const int column = reduction_index;
    local_correlation = __fadd_rn(local_correlation, __fmul_rn(__fmul_rn(__bfloat162float(output_cotangent[row * kProgram1Columns + column]), __bfloat162float(feature_scale[column])), __bfloat162float(primal[row * kProgram1Columns + column])));
  }
  shared_correlation[threadIdx.x] = local_correlation;
  __syncthreads();
  for (int stride = kProgram1Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_correlation[threadIdx.x] = __fadd_rn(shared_correlation[threadIdx.x], shared_correlation[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  for (int element = threadIdx.x; element < kProgram1Columns; element += blockDim.x) {
    const int row = group;
    const int column = element;
    output[row * kProgram1Columns + column] = __float2bfloat16_rn(__fmul_rn(inverse_scale[group], __fsub_rn(__fmul_rn(__bfloat162float(output_cotangent[group * kProgram1Columns + element]), __bfloat162float(feature_scale[element])), __fmul_rn(__bfloat162float(primal[group * kProgram1Columns + element]), __fmul_rn(__fmul_rn(inverse_scale[group], inverse_scale[group]), (shared_correlation[0] / 4096.0f))))));
  }
}

constexpr int kProgram2Rows = 2048;
constexpr int kProgram2Columns = 4096;
constexpr int kProgram2Threads = 256;
constexpr int kProgram2GroupsPerBlock = 32;
constexpr int kProgram2ReductionLanes = kProgram2Threads / kProgram2GroupsPerBlock;

__global__ __launch_bounds__(kProgram2Threads) void ShuttleAxisFoldKernel2(const __nv_bfloat16* primal, const __nv_bfloat16* output_cotangent, const float* inverse_scale, __nv_bfloat16* output) {
  __shared__ float shared_scale_cotangent_sum[kProgram2Threads];
  const int group_lane = threadIdx.x % kProgram2GroupsPerBlock;
  const int reduction_lane = threadIdx.x / kProgram2GroupsPerBlock;
  const int group = blockIdx.x * kProgram2GroupsPerBlock + group_lane;
  float local_scale_cotangent_sum = 0.0f;
  if (group < kProgram2Columns) {
    for (int row = reduction_lane; row < kProgram2Rows; row += kProgram2ReductionLanes) {
      const int column = group;
      local_scale_cotangent_sum = __fadd_rn(local_scale_cotangent_sum, __fmul_rn(__fmul_rn(__bfloat162float(output_cotangent[row * kProgram2Columns + column]), __bfloat162float(primal[row * kProgram2Columns + column])), inverse_scale[row]));
    }
  }
  shared_scale_cotangent_sum[threadIdx.x] = local_scale_cotangent_sum;
  __syncthreads();
  for (int stride = kProgram2ReductionLanes / 2; stride > 0; stride /= 2) {
    if (reduction_lane < stride) {
      shared_scale_cotangent_sum[threadIdx.x] = __fadd_rn(shared_scale_cotangent_sum[threadIdx.x], shared_scale_cotangent_sum[threadIdx.x + stride * kProgram2GroupsPerBlock]);
    }
    __syncthreads();
  }
  if (reduction_lane == 0 && group < kProgram2Columns) {
    output[group] = __float2bfloat16_rn(shared_scale_cotangent_sum[threadIdx.x]);
  }
}

ffi::Error ShuttleAxisFoldRegion(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::Buffer<ffi::BF16, 2> primal_buffer,
    ffi::Buffer<ffi::BF16, 1> feature_scale_buffer,
    ffi::Buffer<ffi::BF16, 2> output_cotangent_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> input_cotangent,
    ffi::Result<ffi::Buffer<ffi::BF16, 1>> feature_scale_cotangent) {
  const auto* primal = reinterpret_cast<const __nv_bfloat16*>(primal_buffer.typed_data());
  const auto* feature_scale = reinterpret_cast<const __nv_bfloat16*>(feature_scale_buffer.typed_data());
  const auto* output_cotangent = reinterpret_cast<const __nv_bfloat16*>(output_cotangent_buffer.typed_data());
  auto* input_cotangent_data = reinterpret_cast<__nv_bfloat16*>(input_cotangent->typed_data());
  auto* feature_scale_cotangent_data = reinterpret_cast<__nv_bfloat16*>(feature_scale_cotangent->typed_data());

  auto inverse_scale_storage = scratch.Allocate(
      sizeof(float) * 2048, alignof(float));
  if (!inverse_scale_storage) {
    return ffi::Error::Internal("failed to allocate axis-Fold pipeline value inverse_scale");
  }
  auto* inverse_scale = static_cast<float*>(*inverse_scale_storage);

  ShuttleAxisFoldKernel0<<<kProgram0Rows, kProgram0Threads, 0, stream>>>(
      primal,
      inverse_scale);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("ShuttleAxisFoldKernel0: ") + cudaGetErrorString(status));
  }

  ShuttleAxisFoldKernel1<<<kProgram1Rows, kProgram1Threads, 0, stream>>>(
      primal,
      feature_scale,
      output_cotangent,
      inverse_scale,
      input_cotangent_data);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("ShuttleAxisFoldKernel1: ") + cudaGetErrorString(status));
  }

  ShuttleAxisFoldKernel2<<<(kProgram2Columns + kProgram2GroupsPerBlock - 1) / kProgram2GroupsPerBlock, kProgram2Threads, 0, stream>>>(
      primal,
      output_cotangent,
      inverse_scale,
      feature_scale_cotangent_data);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("ShuttleAxisFoldKernel2: ") + cudaGetErrorString(status));
  }
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}

auto ShuttleAxisFoldRegionBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::ScratchAllocator>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 1>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 1>>();
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    shuttle_axis_fold_reverse_v1_separate_stages,
    ShuttleAxisFoldRegion,
    ShuttleAxisFoldRegionBinding());

extern "C" int shuttle_axis_fold_ffi_call_count() {
  return call_count.load(std::memory_order_relaxed);
}
