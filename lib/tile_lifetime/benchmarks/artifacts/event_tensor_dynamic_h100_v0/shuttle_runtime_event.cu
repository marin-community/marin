// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from generic EventTensorPlan runtime tables; do not edit.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

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

void check_int32(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), name, " must be a contiguous CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must be int32");
}

}  // namespace

void run_runtime_counted_event_out(
    torch::Tensor input,
    torch::Tensor partials,
    torch::Tensor output,
    torch::Tensor event_counts,
    torch::Tensor event_source_offsets,
    torch::Tensor event_sources,
    int maximum_count) {
  TORCH_CHECK(input.is_cuda() && partials.is_cuda() && output.is_cuda(), "payload tensors must be CUDA tensors");
  TORCH_CHECK(input.is_contiguous() && partials.is_contiguous() && output.is_contiguous(),
              "payload tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 && partials.scalar_type() == torch::kFloat32 &&
              output.scalar_type() == torch::kFloat32, "payload tensors must be FP32");
  check_int32(event_counts, "event counts");
  check_int32(event_source_offsets, "event source offsets");
  check_int32(event_sources, "event sources");
  TORCH_CHECK(maximum_count > 0 && maximum_count <= 1024, "maximum count must be in [1, 1024]");
  TORCH_CHECK(event_source_offsets.numel() == event_counts.numel() + 1, "event offset extent mismatch");
  TORCH_CHECK(output.numel() == event_counts.numel(), "event output extent mismatch");
  TORCH_CHECK(input.numel() == partials.numel() && input.numel() == event_sources.numel(),
              "event source extent mismatch");
  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int threads = 32;
  while (threads < maximum_count) threads *= 2;
  shuttle_runtime_counted_event<<<event_counts.numel(), threads, 0, stream>>>(
      input.data_ptr<float>(), partials.data_ptr<float>(), output.data_ptr<float>(),
      event_counts.data_ptr<int>(), event_source_offsets.data_ptr<int>(), event_sources.data_ptr<int>(),
      event_counts.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run_runtime_counted_event_out", &run_runtime_counted_event_out);
}
