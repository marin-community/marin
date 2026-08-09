// Generated generic group-batched Contract; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {
constexpr int kGroups = 4;
constexpr int kReduction = 512;
constexpr int kLhsFeatures = 32;
constexpr int kRhsFeatures = 64;
std::atomic<int> call_count{0};
thread_local cublasHandle_t contract_handle = nullptr;

ffi::Error ShuttleGroupBatchedContract(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 3> lhs_buffer,
    ffi::Buffer<ffi::BF16, 3> rhs_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 3>> output_buffer) {
  if (contract_handle == nullptr) {
    const cublasStatus_t create_status = cublasCreate(&contract_handle);
    if (create_status != CUBLAS_STATUS_SUCCESS) {
      return ffi::Error::Internal(
          "cublasCreate failed with status " + std::to_string(static_cast<int>(create_status)));
    }
  }
  cublasStatus_t status = cublasSetStream(contract_handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return ffi::Error::Internal(
        "cublasSetStream failed with status " + std::to_string(static_cast<int>(status)));
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const auto* lhs = reinterpret_cast<const std::uint16_t*>(lhs_buffer.typed_data());
  const auto* rhs = reinterpret_cast<const std::uint16_t*>(rhs_buffer.typed_data());
  auto* output = reinterpret_cast<std::uint16_t*>(output_buffer->typed_data());
  status = cublasGemmStridedBatchedEx(
      contract_handle,
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      kRhsFeatures,
      kLhsFeatures,
      kReduction,
      &alpha,
      rhs,
      CUDA_R_16BF,
      kRhsFeatures,
      static_cast<long long>(kReduction) * kRhsFeatures,
      lhs,
      CUDA_R_16BF,
      kLhsFeatures,
      static_cast<long long>(kReduction) * kLhsFeatures,
      &beta,
      output,
      CUDA_R_16BF,
      kRhsFeatures,
      static_cast<long long>(kLhsFeatures) * kRhsFeatures,
      kGroups,
      CUBLAS_COMPUTE_32F_PEDANTIC,
      CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return ffi::Error::Internal(
        "cublasGemmStridedBatchedEx failed with status " +
        std::to_string(static_cast<int>(status)));
  }
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}

auto ShuttleGroupBatchedContractBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Ret<ffi::Buffer<ffi::BF16, 3>>();
}
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    shuttle_routed_training_weight_gradient_0_v1,
    ShuttleGroupBatchedContract,
    ShuttleGroupBatchedContractBinding());

extern "C" int shuttle_routed_weight_gradient_call_count() {
  return call_count.load(std::memory_order_relaxed);
}
