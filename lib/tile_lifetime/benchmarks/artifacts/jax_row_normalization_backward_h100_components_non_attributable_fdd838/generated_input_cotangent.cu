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

__global__ __launch_bounds__(kProgram0Threads) void ShuttleAxisFoldKernel0(const float* projected, const __nv_bfloat16* feature_scale, const __nv_bfloat16* standardized, const float* inverse_scale, float* output) {
  __shared__ float shared_correlation_sum[kProgram0Threads];
  const int group = blockIdx.x;
  if (group >= kProgram0Rows) return;
  float local_correlation_sum = 0.0f;
  for (int reduction_index = threadIdx.x; reduction_index < kProgram0Columns;
       reduction_index += kProgram0Threads) {
    const int row = group;
    const int column = reduction_index;
    local_correlation_sum = __fadd_rn(local_correlation_sum, __fmul_rn(__fmul_rn(projected[row * kProgram0Columns + column], __bfloat162float(feature_scale[column])), __bfloat162float(standardized[row * kProgram0Columns + column])));
  }
  shared_correlation_sum[threadIdx.x] = local_correlation_sum;
  __syncthreads();
  for (int stride = kProgram0Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_correlation_sum[threadIdx.x] = __fadd_rn(shared_correlation_sum[threadIdx.x], shared_correlation_sum[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  for (int element = threadIdx.x; element < kProgram0Columns; element += blockDim.x) {
    const int row = group;
    const int column = element;
    output[row * kProgram0Columns + column] = __fmul_rn(inverse_scale[group], __fsub_rn(__fmul_rn(projected[group * kProgram0Columns + element], __bfloat162float(feature_scale[element])), __fmul_rn(__bfloat162float(standardized[group * kProgram0Columns + element]), (shared_correlation_sum[0] / 4096.0f))));
  }
}

ffi::Error ShuttleAxisFoldRegion(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::Buffer<ffi::F32, 2> projected_buffer,
    ffi::Buffer<ffi::BF16, 1> feature_scale_buffer,
    ffi::Buffer<ffi::BF16, 2> standardized_buffer,
    ffi::Buffer<ffi::F32, 1> inverse_scale_buffer,
    ffi::Result<ffi::Buffer<ffi::F32, 2>> output0) {
  const auto* projected = projected_buffer.typed_data();
  const auto* feature_scale = reinterpret_cast<const __nv_bfloat16*>(feature_scale_buffer.typed_data());
  const auto* standardized = reinterpret_cast<const __nv_bfloat16*>(standardized_buffer.typed_data());
  const auto* inverse_scale = inverse_scale_buffer.typed_data();
  auto* output0_data = output0->typed_data();


  ShuttleAxisFoldKernel0<<<kProgram0Rows, kProgram0Threads, 0, stream>>>(
      projected,
      feature_scale,
      standardized,
      inverse_scale,
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
      .Ctx<ffi::ScratchAllocator>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 1>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 1>>()
      .Ret<ffi::Buffer<ffi::F32, 2>>();
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    shuttle_axis_fold_reverse_input_v1,
    ShuttleAxisFoldRegion,
    ShuttleAxisFoldRegionBinding());

extern "C" int shuttle_axis_fold_ffi_call_count() {
  return call_count.load(std::memory_order_relaxed);
}
