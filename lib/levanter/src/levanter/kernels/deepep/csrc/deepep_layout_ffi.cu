// Copyright The Levanter Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>
#include <string>

#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

#include "kernels/api.cuh"

namespace ffi = xla::ffi;

namespace {

__global__ void CastInt32ToInt64Kernel(const int* src, int64_t* dst, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = static_cast<int64_t>(src[idx]);
  }
}

ffi::Error CudaError(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return ffi::Error::Success();
  }
  return ffi::Error::Internal(std::string(context) + ": " + cudaGetErrorString(status));
}

ffi::Error DeepepGetDispatchLayout(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32, 2> topk_idx,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> num_tokens_per_rank,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> num_tokens_per_expert,
    ffi::Result<ffi::Buffer<ffi::PRED, 2>> is_token_in_rank,
    ffi::Result<ffi::Buffer<ffi::S32, 2>> topk_idx_i64_scratch) {
  const auto topk_dims = topk_idx.dimensions();
  const auto rank_dims = num_tokens_per_rank->dimensions();
  const auto expert_dims = num_tokens_per_expert->dimensions();
  const auto token_rank_dims = is_token_in_rank->dimensions();
  const auto scratch_dims = topk_idx_i64_scratch->dimensions();

  if (topk_dims.size() != 2) {
    return ffi::Error::InvalidArgument("topk_idx must be rank-2");
  }
  if (rank_dims.size() != 1 || expert_dims.size() != 1) {
    return ffi::Error::InvalidArgument("DeepEP layout outputs have unexpected ranks");
  }
  if (token_rank_dims.size() != 2) {
    return ffi::Error::InvalidArgument("is_token_in_rank must be rank-2");
  }
  if (scratch_dims.size() != 2) {
    return ffi::Error::InvalidArgument("DeepEP layout scratch must be rank-2");
  }

  const int64_t num_tokens = topk_dims[0];
  const int64_t num_topk = topk_dims[1];
  const int64_t num_ranks = rank_dims[0];
  const int64_t num_experts = expert_dims[0];

  if (token_rank_dims[0] != num_tokens || token_rank_dims[1] != num_ranks) {
    return ffi::Error::InvalidArgument("is_token_in_rank shape must be [tokens, num_ranks]");
  }
  if (scratch_dims[0] != num_tokens || scratch_dims[1] != num_topk * 2) {
    return ffi::Error::InvalidArgument("DeepEP layout scratch shape must be [tokens, topk * 2]");
  }
  if (num_tokens > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
      num_topk > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
      num_ranks > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
      num_experts > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    return ffi::Error::InvalidArgument("DeepEP layout dimensions exceed int32 kernel limits");
  }

  const int topk_count = static_cast<int>(num_tokens * num_topk);
  auto* topk_idx_i64 = reinterpret_cast<int64_t*>(topk_idx_i64_scratch->typed_data());
  const int threads = 256;
  const int blocks = (topk_count + threads - 1) / threads;
  CastInt32ToInt64Kernel<<<blocks, threads, 0, stream>>>(topk_idx.typed_data(), topk_idx_i64, topk_count);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return CudaError(status, "CastInt32ToInt64Kernel(layout)");
  }

  deep_ep::layout::get_dispatch_layout(
      topk_idx_i64,
      num_tokens_per_rank->typed_data(),
      nullptr,
      num_tokens_per_expert->typed_data(),
      is_token_in_rank->typed_data(),
      static_cast<int>(num_tokens),
      static_cast<int>(num_topk),
      static_cast<int>(num_ranks),
      static_cast<int>(num_experts),
      stream);
  return ffi::Error::Success();
}

auto DeepepGetDispatchLayoutBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::PRED, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 2>>();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_deepep_get_dispatch_layout,
    DeepepGetDispatchLayout,
    DeepepGetDispatchLayoutBinding(),
    {ffi::Traits::kCmdBufferCompatible});
