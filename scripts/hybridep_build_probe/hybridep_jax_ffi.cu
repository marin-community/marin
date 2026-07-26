// Copyright The Levanter Authors
// SPDX-License-Identifier: Apache-2.0

#include <Python.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "hybrid_ep.cuh"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace py = pybind11;

namespace {

std::mutex& RuntimeMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unique_ptr<HybridEPBuffer>& Runtime() {
  static std::unique_ptr<HybridEPBuffer> runtime;
  return runtime;
}

HybridEpConfigInstance& RuntimeConfig() {
  static HybridEpConfigInstance config{};
  return config;
}

std::unordered_map<int32_t, HandleImpl>& Handles() {
  static std::unordered_map<int32_t, HandleImpl> handles;
  return handles;
}

int32_t& NextHandleId() {
  static int32_t handle_id = 1;
  return handle_id;
}

size_t& MaxActiveHandles() {
  static size_t max_active_handles = 0;
  return max_active_handles;
}

bool TraceHandles() {
  static const bool enabled = std::getenv("HYBRID_EP_TRACE_HANDLES") != nullptr;
  return enabled;
}

void TraceHandleEvent(const char* operation, int32_t handle_id) {
  if (!TraceHandles()) {
    return;
  }
  std::fprintf(
      stderr,
      "HybridEP JAX handle event: operation=%s handle=%d active=%zu\n",
      operation,
      handle_id,
      Handles().size());
  std::fflush(stderr);
}

std::string& LastError() {
  static std::string error;
  return error;
}

ffi::Error CudaError(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return ffi::Error::Success();
  }
  return ffi::Error::Internal(std::string(context) + ": " + cudaGetErrorString(status));
}

ffi::Error ReadHandle(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32, 0> handle_token,
    const char* consumer,
    HandleImpl* handle) {
  float encoded_handle = 0;
  cudaError_t status = cudaMemcpyAsync(
      &encoded_handle,
      handle_token.untyped_data(),
      sizeof(encoded_handle),
      cudaMemcpyDeviceToHost,
      stream);
  if (status != cudaSuccess) {
    return CudaError(status, "cudaMemcpyAsync(HybridEP handle token)");
  }
  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    return CudaError(status, "cudaStreamSynchronize(HybridEP handle token)");
  }
  const auto handle_id = static_cast<int32_t>(std::lrint(encoded_handle));
  auto iterator = Handles().find(handle_id);
  if (handle_id <= 0 ||
      encoded_handle != static_cast<float>(handle_id) ||
      iterator == Handles().end()) {
    return ffi::Error(
        ffi::ErrorCode::kFailedPrecondition,
        "HybridEP combine received an unknown dispatch handle");
  }
  *handle = std::move(iterator->second);
  Handles().erase(iterator);
  TraceHandleEvent(consumer, handle_id);
  return ffi::Error::Success();
}

at::Tensor TensorFromBF16(void* data, int64_t rows, int64_t columns) {
  return torch::from_blob(
      data,
      {rows, columns},
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
}

at::Tensor TensorFromBool(void* data, int64_t rows, int64_t columns) {
  return torch::from_blob(
      data,
      {rows, columns},
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
}

at::Tensor TensorFromFloat(void* data, int64_t rows, int64_t columns) {
  return torch::from_blob(
      data,
      {rows, columns},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
}

void ReserveMetadataAllocatorHeadroom(
    const HybridEpConfigInstance& config,
    int64_t tokens_per_rank) {
  constexpr int kReservationSlots = 3;
  const int64_t total_ranks =
      static_cast<int64_t>(config.num_of_ranks_per_node) *
      config.num_of_nodes;
  const auto device = torch::TensorOptions().device(torch::kCUDA);

  // DeepEP constructs these fixed-shape metadata tensors inside the FFI call.
  // Reserve matching PyTorch caching-allocator blocks before JAX initializes so
  // XLA cannot claim the memory and leave the first dispatch unable to allocate.
  // The tensors themselves die here; their blocks remain reusable by PyTorch.
  std::vector<at::Tensor> metadata_reservations;
  metadata_reservations.reserve(3 * kReservationSlots);
  for (int slot = 0; slot < kReservationSlots; ++slot) {
    metadata_reservations.push_back(torch::empty(
        {tokens_per_rank * config.num_of_nodes, config.num_of_ranks_per_node},
        device.dtype(torch::kInt32)));
    metadata_reservations.push_back(torch::empty(
        {tokens_per_rank * total_ranks, config.num_of_experts_per_rank},
        device.dtype(torch::kBool)));
    metadata_reservations.push_back(torch::empty(
        {tokens_per_rank * total_ranks, config.num_of_experts_per_rank},
        device.dtype(torch::kInt32)));
  }
}

ffi::Error HybridEPDispatch(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> hidden,
    ffi::Buffer<ffi::PRED, 2> routing_map,
    ffi::Buffer<ffi::F32, 2> probabilities,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> dispatched_hidden,
    ffi::Result<ffi::Buffer<ffi::F32, 1>> dispatched_probabilities,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> tokens_per_expert,
    ffi::Result<ffi::Buffer<ffi::F32, 0>> handle_token) {
  try {
    std::lock_guard<std::mutex> lock(RuntimeMutex());
    if (Runtime() == nullptr) {
      return ffi::Error(
          ffi::ErrorCode::kFailedPrecondition,
          "HybridEP JAX runtime is not initialized");
    }
    const auto hidden_dims = hidden.dimensions();
    const auto routing_dims = routing_map.dimensions();
    const auto probability_dims = probabilities.dimensions();
    const auto output_dims = dispatched_hidden->dimensions();
    if (hidden_dims[0] != routing_dims[0] ||
        hidden_dims[0] != probability_dims[0] ||
        routing_dims[1] != probability_dims[1] ||
        hidden_dims[1] != output_dims[1] ||
        dispatched_probabilities->dimensions()[0] != output_dims[0] ||
        tokens_per_expert->dimensions()[0] != RuntimeConfig().num_of_experts_per_rank) {
      return ffi::Error::InvalidArgument("HybridEP dispatch received incompatible shapes");
    }

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
      return CudaError(status, "cudaGetDevice(HybridEP dispatch)");
    }
    c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromExternal(stream, device));
    py::gil_scoped_acquire acquire;

    auto hidden_tensor = TensorFromBF16(
        hidden.untyped_data(),
        hidden_dims[0],
        hidden_dims[1]);
    auto routing_tensor = TensorFromBool(
        routing_map.untyped_data(),
        routing_dims[0],
        routing_dims[1]);
    auto probability_tensor = TensorFromFloat(
        probabilities.untyped_data(),
        probability_dims[0],
        probability_dims[1]);
    auto dispatched_hidden_tensor = TensorFromBF16(
        dispatched_hidden->untyped_data(),
        output_dims[0],
        output_dims[1]);
    auto dispatched_probability_tensor = TensorFromFloat(
        dispatched_probabilities->untyped_data(),
        output_dims[0],
        1).reshape({output_dims[0]});

    HandleImpl handle = Runtime()->metadata_preprocessing(
        RuntimeConfig(),
        routing_tensor,
        hidden_dims[0],
        output_dims[0],
        1,
        true,
        true,
        true);
    auto [output_tensor, output_probability, output_scale] =
        Runtime()->dispatch_with_permute(
            hidden_tensor,
            probability_tensor,
            std::nullopt,
            handle,
            1,
            true,
            true,
            true,
            dispatched_hidden_tensor,
            dispatched_probability_tensor);
    (void)output_scale;
    if (!output_probability.has_value()) {
      return ffi::Error::Internal("HybridEP dispatch did not return probabilities");
    }
    if (output_tensor.numel() != output_dims[0] * output_dims[1] ||
        output_probability->numel() != output_dims[0]) {
      return ffi::Error::Internal("HybridEP dispatch returned an unexpected output size");
    }
    if (output_tensor.data_ptr() != dispatched_hidden->untyped_data() ||
        output_probability->data_ptr() != dispatched_probabilities->untyped_data()) {
      return ffi::Error::Internal("HybridEP dispatch did not use the JAX-owned output buffers");
    }
    status = cudaMemcpyAsync(
        tokens_per_expert->untyped_data(),
        handle.tokens_per_expert.data_ptr(),
        handle.tokens_per_expert.numel() * handle.tokens_per_expert.element_size(),
        cudaMemcpyDeviceToDevice,
        stream);
    if (status != cudaSuccess) {
      return CudaError(status, "cudaMemcpyAsync(HybridEP tokens per expert)");
    }
    const int32_t handle_id = NextHandleId()++;
    const float encoded_handle = static_cast<float>(handle_id);
    status = cudaMemcpyAsync(
        handle_token->untyped_data(),
        &encoded_handle,
        sizeof(encoded_handle),
        cudaMemcpyHostToDevice,
        stream);
    if (status != cudaSuccess) {
      return CudaError(status, "cudaMemcpyAsync(HybridEP handle token)");
    }
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
      return CudaError(status, "cudaStreamSynchronize(HybridEP dispatch)");
    }

    Handles().emplace(handle_id, std::move(handle));
    TraceHandleEvent("dispatch", handle_id);
    if (Handles().size() > MaxActiveHandles()) {
      MaxActiveHandles() = Handles().size();
      std::fprintf(
          stderr,
          "HybridEP JAX peak active dispatch handles: %zu\n",
          MaxActiveHandles());
      std::fflush(stderr);
    }
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::Internal(std::string("HybridEP dispatch: ") + error.what());
  }
}

ffi::Error HybridEPCombine(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> expert_hidden,
    ffi::Buffer<ffi::F32, 0> handle_token,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> combined_hidden) {
  try {
    std::lock_guard<std::mutex> lock(RuntimeMutex());
    if (Runtime() == nullptr) {
      return ffi::Error(
          ffi::ErrorCode::kFailedPrecondition,
          "HybridEP JAX runtime is not initialized");
    }
    HandleImpl handle;
    ffi::Error handle_error = ReadHandle(stream, handle_token, "combine", &handle);
    if (handle_error.failure()) {
      return handle_error;
    }
    const auto input_dims = expert_hidden.dimensions();
    const auto output_dims = combined_hidden->dimensions();
    if (input_dims[1] != output_dims[1] ||
        input_dims[0] != handle.num_permuted_tokens ||
        output_dims[0] != handle.num_of_tokens_per_rank) {
      return ffi::Error::InvalidArgument("HybridEP combine received incompatible shapes");
    }

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
      return CudaError(status, "cudaGetDevice(HybridEP combine)");
    }
    c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromExternal(stream, device));
    py::gil_scoped_acquire acquire;

    auto expert_tensor = TensorFromBF16(
        expert_hidden.untyped_data(),
        input_dims[0],
        input_dims[1]);
    auto combined_hidden_tensor = TensorFromBF16(
        combined_hidden->untyped_data(),
        output_dims[0],
        output_dims[1]);
    auto [output_tensor, output_probabilities] =
        Runtime()->combine_with_unpermute(
            expert_tensor,
            std::nullopt,
            handle,
            1,
            true,
            false,
            combined_hidden_tensor,
            std::nullopt);
    (void)output_probabilities;
    if (output_tensor.numel() != output_dims[0] * output_dims[1]) {
      return ffi::Error::Internal("HybridEP combine returned an unexpected output size");
    }
    if (output_tensor.data_ptr() != combined_hidden->untyped_data()) {
      return ffi::Error::Internal("HybridEP combine did not use the JAX-owned output buffer");
    }
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
      return CudaError(status, "cudaStreamSynchronize(HybridEP combine)");
    }
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::Internal(std::string("HybridEP combine: ") + error.what());
  }
}

ffi::Error HybridEPCombineWithProbabilities(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> expert_hidden,
    ffi::Buffer<ffi::F32, 1> expert_probabilities,
    ffi::Buffer<ffi::F32, 0> handle_token,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> combined_hidden,
    ffi::Result<ffi::Buffer<ffi::F32, 2>> combined_probabilities) {
  try {
    std::lock_guard<std::mutex> lock(RuntimeMutex());
    if (Runtime() == nullptr) {
      return ffi::Error(
          ffi::ErrorCode::kFailedPrecondition,
          "HybridEP JAX runtime is not initialized");
    }
    HandleImpl handle;
    ffi::Error handle_error =
        ReadHandle(stream, handle_token, "combine_with_probabilities", &handle);
    if (handle_error.failure()) {
      return handle_error;
    }
    const auto input_dims = expert_hidden.dimensions();
    const auto output_dims = combined_hidden->dimensions();
    const auto probability_dims = combined_probabilities->dimensions();
    if (input_dims[1] != output_dims[1] ||
        input_dims[0] != handle.num_permuted_tokens ||
        expert_probabilities.dimensions()[0] != input_dims[0] ||
        output_dims[0] != handle.num_of_tokens_per_rank ||
        probability_dims[0] != output_dims[0] ||
        probability_dims[1] !=
            RuntimeConfig().num_of_experts_per_rank *
                RuntimeConfig().num_of_ranks_per_node *
                RuntimeConfig().num_of_nodes) {
      return ffi::Error::InvalidArgument(
          "HybridEP combine-with-probabilities received incompatible shapes");
    }

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
      return CudaError(status, "cudaGetDevice(HybridEP combine-with-probabilities)");
    }
    c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromExternal(stream, device));
    py::gil_scoped_acquire acquire;

    auto expert_tensor = TensorFromBF16(
        expert_hidden.untyped_data(),
        input_dims[0],
        input_dims[1]);
    auto probability_tensor = TensorFromFloat(
        expert_probabilities.untyped_data(),
        input_dims[0],
        1).reshape({input_dims[0]});
    auto combined_hidden_tensor = TensorFromBF16(
        combined_hidden->untyped_data(),
        output_dims[0],
        output_dims[1]);
    auto combined_probability_tensor = TensorFromFloat(
        combined_probabilities->untyped_data(),
        probability_dims[0],
        probability_dims[1]);
    auto [output_tensor, output_probability] =
        Runtime()->combine_with_unpermute(
            expert_tensor,
            probability_tensor,
            handle,
            1,
            true,
            true,
            combined_hidden_tensor,
            combined_probability_tensor);
    if (!output_probability.defined()) {
      return ffi::Error::Internal(
          "HybridEP combine-with-probabilities did not return probabilities");
    }
    if (output_tensor.numel() != output_dims[0] * output_dims[1] ||
        output_probability.numel() != probability_dims[0] * probability_dims[1]) {
      return ffi::Error::Internal(
          "HybridEP combine-with-probabilities returned an unexpected output size");
    }
    if (output_tensor.data_ptr() != combined_hidden->untyped_data() ||
        output_probability.data_ptr() != combined_probabilities->untyped_data()) {
      return ffi::Error::Internal(
          "HybridEP combine-with-probabilities did not use the JAX-owned output buffers");
    }
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
      return CudaError(
          status,
          "cudaStreamSynchronize(HybridEP combine-with-probabilities)");
    }
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::Internal(
        std::string("HybridEP combine-with-probabilities: ") + error.what());
  }
}

auto HybridEPDispatchBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::PRED, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::F32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::F32, 0>>();
}

auto HybridEPCombineBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 0>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>();
}

auto HybridEPCombineWithProbabilitiesBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 1>>()
      .Arg<ffi::Buffer<ffi::F32, 0>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::F32, 2>>();
}

}  // namespace

extern "C" const char* levanter_hybridep_last_error() {
  return LastError().c_str();
}

extern "C" int levanter_hybridep_init(
    PyObject* process_group,
    int rank,
    int world_size,
    int hidden,
    int tokens,
    int local_experts,
    int dispatch_sms,
    int combine_sms,
    const char* base_path) {
  try {
    std::lock_guard<std::mutex> lock(RuntimeMutex());
    py::gil_scoped_acquire acquire;
    auto group = py::reinterpret_borrow<py::object>(process_group);
    Configurer configurer(
        hidden,
        tokens,
        local_experts,
        world_size,
        1,
        false,
        dispatch_sms,
        combine_sms);
    HybridEpConfigInstance config = configurer.get_default_config(true);
    config.pad_multiple = 1;
    configurer.adjust_template(config, true);
    Runtime() = std::make_unique<HybridEPBuffer>(
        group,
        configurer.buffer_config,
        rank,
        0,
        world_size,
        base_path,
        false,
        true,
        true);
    Runtime()->update_buffer(config);
    RuntimeConfig() = config;
    Handles().clear();
    NextHandleId() = 1;
    MaxActiveHandles() = 0;
    ReserveMetadataAllocatorHeadroom(config, tokens);
    LastError().clear();
    return 0;
  } catch (const std::exception& error) {
    Runtime().reset();
    LastError() = error.what();
    return 1;
  }
}

extern "C" void levanter_hybridep_shutdown() {
  std::lock_guard<std::mutex> lock(RuntimeMutex());
  py::gil_scoped_acquire acquire;
  Runtime().reset();
  Handles().clear();
  NextHandleId() = 1;
  MaxActiveHandles() = 0;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_hybridep_dispatch,
    HybridEPDispatch,
    HybridEPDispatchBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_hybridep_combine,
    HybridEPCombine,
    HybridEPCombineBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_hybridep_combine_with_probabilities,
    HybridEPCombineWithProbabilities,
    HybridEPCombineWithProbabilitiesBinding());
