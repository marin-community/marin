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

constexpr int kProgram0Rows = 8;
constexpr int kProgram0Columns = 32;
constexpr int kProgram0Threads = 256;

__global__ __launch_bounds__(kProgram0Threads) void ShuttleAxisFoldKernel0(const float* input0, const float* input1, const float* input2, const float* input3, const float* input4, __nv_bfloat16* output) {
  __shared__ float shared_fold_sum[kProgram0Threads];
  const int group = blockIdx.x;
  if (group >= kProgram0Rows) return;
  float local_fold_sum = 0.0f;
  for (int reduction_index = threadIdx.x; reduction_index < kProgram0Columns;
       reduction_index += kProgram0Threads) {
    const int row = group;
    const int column = reduction_index;
    local_fold_sum = __fadd_rn(local_fold_sum, input3[row * kProgram0Columns + column]);
  }
  shared_fold_sum[threadIdx.x] = local_fold_sum;
  __syncthreads();
  for (int stride = kProgram0Threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_fold_sum[threadIdx.x] = __fadd_rn(shared_fold_sum[threadIdx.x], shared_fold_sum[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  for (int element = threadIdx.x; element < kProgram0Columns; element += blockDim.x) {
    const int row = group;
    const int column = element;
    output[row * kProgram0Columns + column] = __float2bfloat16_rn(__fadd_rn(input2[group * kProgram0Columns + element], __fmul_rn(input1[group * kProgram0Columns + element], __fmul_rn(__fmul_rn(shared_fold_sum[0], input4[group]), input0[group]))));
  }
}

ffi::Error ShuttleAxisFoldRegion(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::Buffer<ffi::F32, 1> input0_buffer,
    ffi::Buffer<ffi::F32, 2> input1_buffer,
    ffi::Buffer<ffi::F32, 2> input2_buffer,
    ffi::Buffer<ffi::F32, 2> input3_buffer,
    ffi::Buffer<ffi::F32, 1> input4_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> output0) {
  const auto* input0 = input0_buffer.typed_data();
  const auto* input1 = input1_buffer.typed_data();
  const auto* input2 = input2_buffer.typed_data();
  const auto* input3 = input3_buffer.typed_data();
  const auto* input4 = input4_buffer.typed_data();
  auto* output0_data = reinterpret_cast<__nv_bfloat16*>(output0->typed_data());


  ShuttleAxisFoldKernel0<<<kProgram0Rows, kProgram0Threads, 0, stream>>>(
      input0,
      input1,
      input2,
      input3,
      input4,
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
      .Arg<ffi::Buffer<ffi::F32, 1>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 1>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>();
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    shuttle_routed_training_axis_fold_0_v1,
    ShuttleAxisFoldRegion,
    ShuttleAxisFoldRegionBinding());

extern "C" int shuttle_axis_fold_ffi_call_count() {
  return call_count.load(std::memory_order_relaxed);
}