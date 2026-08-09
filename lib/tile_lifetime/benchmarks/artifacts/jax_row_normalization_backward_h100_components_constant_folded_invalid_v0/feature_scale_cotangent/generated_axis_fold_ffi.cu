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
constexpr int kProgram0GroupsPerBlock = 32;
constexpr int kProgram0ReductionLanes = kProgram0Threads / kProgram0GroupsPerBlock;

__global__ __launch_bounds__(kProgram0Threads) void ShuttleAxisFoldKernel0(const float* projected, const __nv_bfloat16* standardized, float* output) {
  __shared__ float shared_feature_scale_sum[kProgram0Threads];
  const int group_lane = threadIdx.x % kProgram0GroupsPerBlock;
  const int reduction_lane = threadIdx.x / kProgram0GroupsPerBlock;
  const int group = blockIdx.x * kProgram0GroupsPerBlock + group_lane;
  float local_feature_scale_sum = 0.0f;
  if (group < kProgram0Columns) {
    for (int row = reduction_lane; row < kProgram0Rows; row += kProgram0ReductionLanes) {
      const int column = group;
      local_feature_scale_sum = __fadd_rn(local_feature_scale_sum, __fmul_rn(projected[row * kProgram0Columns + column], __bfloat162float(standardized[row * kProgram0Columns + column])));
    }
  }
  shared_feature_scale_sum[threadIdx.x] = local_feature_scale_sum;
  __syncthreads();
  for (int stride = kProgram0ReductionLanes / 2; stride > 0; stride /= 2) {
    if (reduction_lane < stride) {
      shared_feature_scale_sum[threadIdx.x] = __fadd_rn(shared_feature_scale_sum[threadIdx.x], shared_feature_scale_sum[threadIdx.x + stride * kProgram0GroupsPerBlock]);
    }
    __syncthreads();
  }
  if (reduction_lane == 0 && group < kProgram0Columns) {
    output[group] = shared_feature_scale_sum[threadIdx.x];
  }
}

ffi::Error ShuttleAxisFoldRegion(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32, 2> projected_buffer,
    ffi::Buffer<ffi::BF16, 2> standardized_buffer,
    ffi::Result<ffi::Buffer<ffi::F32, 1>> output0) {
  const auto* projected = projected_buffer.typed_data();
  const auto* standardized = reinterpret_cast<const __nv_bfloat16*>(standardized_buffer.typed_data());
  auto* output0_data = output0->typed_data();

  ShuttleAxisFoldKernel0<<<(kProgram0Columns + kProgram0GroupsPerBlock - 1) / kProgram0GroupsPerBlock, kProgram0Threads, 0, stream>>>(
      projected,
      standardized,
      output0_data);
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("ShuttleAxisFoldKernel0: ") + cudaGetErrorString(status));
  }
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}

auto ShuttleAxisFoldRegionBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::F32, 1>>();
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    shuttle_axis_fold_reverse_feature_v1,
    ShuttleAxisFoldRegion,
    ShuttleAxisFoldRegionBinding());

extern "C" int shuttle_axis_fold_ffi_call_count() {
  return call_count.load(std::memory_order_relaxed);
}
