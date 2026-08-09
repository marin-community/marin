// Generated from generic Contract/Fold/DomainRestriction reverse semantics; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

extern "C" CUresult shuttle_streaming_forward_de92f901_0d1d2d3d4d5d6d7d8d91011121314151617181920212223242526272829303132333435363738394041424344454647(CUstream stream, CUdeviceptr pointer_0, CUdeviceptr pointer_1, CUdeviceptr pointer_2, CUdeviceptr pointer_3, CUdeviceptr pointer_4, CUdeviceptr pointer_5, CUdeviceptr pointer_6, CUdeviceptr pointer_7, CUdeviceptr pointer_8);
extern "C" CUresult shuttle_streaming_dq_f0d1870d_0d1d2d3d4d5d6d7d891011121314151617181920212223242526272829303132333435363738394041424344(CUstream stream, CUdeviceptr pointer_0, CUdeviceptr pointer_1, CUdeviceptr pointer_2, CUdeviceptr pointer_3, CUdeviceptr pointer_4, CUdeviceptr pointer_5, CUdeviceptr pointer_6, CUdeviceptr pointer_7);
extern "C" CUresult shuttle_streaming_dkdv_c8bb3723_0d1d2d3d4d5d6d7d891011121314151617181920212223242526272829303132333435363738394041424344(CUstream stream, CUdeviceptr pointer_0, CUdeviceptr pointer_1, CUdeviceptr pointer_2, CUdeviceptr pointer_3, CUdeviceptr pointer_4, CUdeviceptr pointer_5, CUdeviceptr pointer_6, CUdeviceptr pointer_7);

namespace ffi = xla::ffi;

namespace {
std::atomic<int> call_count{0};

__global__ void ShuttleIotaKernel(int32_t* output, int extent) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < extent) output[index] = index;
}

ffi::Error DriverError(const char* stage, CUresult result) {
  const char* name = nullptr;
  cuGetErrorName(result, &name);
  return ffi::Error::Internal(std::string(stage) + ": " + (name == nullptr ? "unknown CUDA driver error" : name));
}

ffi::Error ShuttleStreamingAttentionBackward(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::Buffer<ffi::BF16, 4> query_buffer,
    ffi::Buffer<ffi::BF16, 4> key_buffer,
    ffi::Buffer<ffi::BF16, 4> value_buffer,
    ffi::Buffer<ffi::BF16, 4> output_cotangent_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 4>> query_cotangent,
    ffi::Result<ffi::Buffer<ffi::BF16, 4>> key_cotangent,
    ffi::Result<ffi::Buffer<ffi::BF16, 4>> value_cotangent) {
  auto* query = reinterpret_cast<__nv_bfloat16*>(query_buffer.typed_data());
  auto* key = reinterpret_cast<__nv_bfloat16*>(key_buffer.typed_data());
  auto* value = reinterpret_cast<__nv_bfloat16*>(value_buffer.typed_data());
  auto* output_cotangent = reinterpret_cast<__nv_bfloat16*>(output_cotangent_buffer.typed_data());
  auto* query_cotangent_pointer = reinterpret_cast<__nv_bfloat16*>(query_cotangent->typed_data());
  auto* key_cotangent_pointer = reinterpret_cast<__nv_bfloat16*>(key_cotangent->typed_data());
  auto* value_cotangent_pointer = reinterpret_cast<__nv_bfloat16*>(value_cotangent->typed_data());
  auto dot_storage = scratch.Allocate(sizeof(float) * 65536, alignof(float));
  if (!dot_storage) return ffi::Error::Internal("failed to allocate streaming reverse output-dot state");
  auto* output_dot = static_cast<float*>(*dot_storage);

  auto output_storage = scratch.Allocate(sizeof(__nv_bfloat16) * 8388608, alignof(__nv_bfloat16));
  auto lse_storage = scratch.Allocate(sizeof(float) * 65536, alignof(float));
  auto position_storage = scratch.Allocate(sizeof(int32_t) * 2048, alignof(int32_t));
  if (!output_storage || !lse_storage || !position_storage) {
    return ffi::Error::Internal("failed to allocate streaming reverse recompute state");
  }
  auto* output = static_cast<__nv_bfloat16*>(*output_storage);
  auto* log_sum_exp = static_cast<float*>(*lse_storage);
  auto* positions = static_cast<int32_t*>(*position_storage);
  constexpr int kPositionThreads = 256;
  ShuttleIotaKernel<<<(2048 + kPositionThreads - 1) / kPositionThreads, kPositionThreads, 0, stream>>>(
      positions, 2048);
  cudaError_t runtime_status = cudaGetLastError();
  if (runtime_status != cudaSuccess) {
    return ffi::Error::Internal(std::string("ShuttleIotaKernel: ") + cudaGetErrorString(runtime_status));
  }
  CUresult driver_status = shuttle_streaming_forward_de92f901_0d1d2d3d4d5d6d7d8d91011121314151617181920212223242526272829303132333435363738394041424344454647(
      reinterpret_cast<CUstream>(stream),
      reinterpret_cast<CUdeviceptr>(query),
      reinterpret_cast<CUdeviceptr>(key),
      reinterpret_cast<CUdeviceptr>(value),
      reinterpret_cast<CUdeviceptr>(output),
      reinterpret_cast<CUdeviceptr>(log_sum_exp),
      reinterpret_cast<CUdeviceptr>(query),
      reinterpret_cast<CUdeviceptr>(query),
      reinterpret_cast<CUdeviceptr>(positions),
      reinterpret_cast<CUdeviceptr>(positions));
  if (driver_status != CUDA_SUCCESS) return DriverError("streaming forward", driver_status);

  driver_status = shuttle_streaming_dq_f0d1870d_0d1d2d3d4d5d6d7d891011121314151617181920212223242526272829303132333435363738394041424344(
      reinterpret_cast<CUstream>(stream),
      reinterpret_cast<CUdeviceptr>(query),
      reinterpret_cast<CUdeviceptr>(key),
      reinterpret_cast<CUdeviceptr>(value),
      reinterpret_cast<CUdeviceptr>(output),
      reinterpret_cast<CUdeviceptr>(output_cotangent),
      reinterpret_cast<CUdeviceptr>(log_sum_exp),
      reinterpret_cast<CUdeviceptr>(output_dot),
      reinterpret_cast<CUdeviceptr>(query_cotangent_pointer));
  if (driver_status != CUDA_SUCCESS) return DriverError("query cotangent", driver_status);
  driver_status = shuttle_streaming_dkdv_c8bb3723_0d1d2d3d4d5d6d7d891011121314151617181920212223242526272829303132333435363738394041424344(
      reinterpret_cast<CUstream>(stream),
      reinterpret_cast<CUdeviceptr>(query),
      reinterpret_cast<CUdeviceptr>(key),
      reinterpret_cast<CUdeviceptr>(value),
      reinterpret_cast<CUdeviceptr>(output_cotangent),
      reinterpret_cast<CUdeviceptr>(log_sum_exp),
      reinterpret_cast<CUdeviceptr>(output_dot),
      reinterpret_cast<CUdeviceptr>(key_cotangent_pointer),
      reinterpret_cast<CUdeviceptr>(value_cotangent_pointer));
  if (driver_status != CUDA_SUCCESS) return DriverError("key/value cotangents", driver_status);
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}

auto ShuttleStreamingAttentionBackwardBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::ScratchAllocator>()
      .Arg<ffi::Buffer<ffi::BF16, 4>>()
      .Arg<ffi::Buffer<ffi::BF16, 4>>()
      .Arg<ffi::Buffer<ffi::BF16, 4>>()
      .Arg<ffi::Buffer<ffi::BF16, 4>>()
      .Ret<ffi::Buffer<ffi::BF16, 4>>()
      .Ret<ffi::Buffer<ffi::BF16, 4>>()
      .Ret<ffi::Buffer<ffi::BF16, 4>>();
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    shuttle_streaming_reverse_recompute_s2048_d128_bm32_bn32_v1,
    ShuttleStreamingAttentionBackward,
    ShuttleStreamingAttentionBackwardBinding());

extern "C" int shuttle_streaming_attention_backward_ffi_call_count() {
  return call_count.load(std::memory_order_relaxed);
}

