// Copyright The Levanter Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "mok_megakernel.cuh"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kNumDevices = 4;

using MoK = dispatch_mlp_swiglu_combiner<kNumDevices, utils::RoutedPrecision::BF16>;

std::string& LastErrorStorage() {
  static std::string error;
  return error;
}

void SetLastError(std::string message) { LastErrorStorage() = std::move(message); }

void ThrowOnCuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

void EnablePeerAccess(int peer_device) {
  cudaError_t status = cudaDeviceEnablePeerAccess(peer_device, 0);
  if (status == cudaSuccess) {
    return;
  }
  if (status == cudaErrorPeerAccessAlreadyEnabled) {
    (void)cudaGetLastError();
    return;
  }
  ThrowOnCuda(status, "cudaDeviceEnablePeerAccess");
}

struct DeviceRuntime {
  int device = -1;
  int rank = -1;
  int num_devices = 0;
  void* x = nullptr;
  void* combine = nullptr;
  void* d_y = nullptr;
  void* d_x_routed = nullptr;
  void* router_weights = nullptr;
  void* d_router_weights = nullptr;
  uint32_t* signals = nullptr;
  uint32_t* epoch = nullptr;
  std::array<void*, kNumDevices> x_ptrs{};
  std::array<void*, kNumDevices> combine_ptrs{};
  std::array<void*, kNumDevices> d_y_ptrs{};
  std::array<void*, kNumDevices> d_x_routed_ptrs{};
  std::array<void*, kNumDevices> router_weight_ptrs{};
  std::array<void*, kNumDevices> d_router_weight_ptrs{};
  std::array<uint32_t*, kNumDevices> signal_ptrs{};
};

class RuntimeManager {
 public:
  static RuntimeManager& Instance() {
    static RuntimeManager manager;
    return manager;
  }

  void Init(int num_devices, int num_tokens, int hidden_dim, int top_k) {
    std::lock_guard<std::mutex> lock(mu_);
    if (initialized_ && num_devices_ == num_devices && num_tokens_ == num_tokens &&
        hidden_dim_ == hidden_dim && top_k_ == top_k) {
      return;
    }
    DestroyLocked();
    if (num_devices != kNumDevices) {
      throw std::runtime_error("Mixture-of-Kittens forward requires exactly four visible GPUs");
    }
    int visible_devices = 0;
    ThrowOnCuda(cudaGetDeviceCount(&visible_devices), "cudaGetDeviceCount");
    if (visible_devices != num_devices) {
      throw std::runtime_error("Mixture-of-Kittens forward requires the expert group to span all visible GPUs");
    }
    if (num_tokens <= 0 || hidden_dim <= 0 || top_k <= 0) {
      throw std::runtime_error("Mixture-of-Kittens runtime dimensions must be positive");
    }

    num_devices_ = num_devices;
    num_tokens_ = num_tokens;
    hidden_dim_ = hidden_dim;
    top_k_ = top_k;
    runtimes_.resize(num_devices);
    int original_device = 0;
    ThrowOnCuda(cudaGetDevice(&original_device), "cudaGetDevice(init)");
    try {
      for (int rank = 0; rank < num_devices; ++rank) {
        ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(allocate)");
        auto runtime = std::make_unique<DeviceRuntime>();
        runtime->device = rank;
        runtime->rank = rank;
        runtime->num_devices = num_devices;
        const size_t x_bytes = static_cast<size_t>(num_tokens) * hidden_dim * sizeof(uint16_t);
        const size_t combine_bytes = static_cast<size_t>(num_tokens) * top_k * hidden_dim * sizeof(uint16_t);
        const size_t router_bytes = static_cast<size_t>(num_tokens) * top_k * sizeof(float);
        ThrowOnCuda(cudaMalloc(&runtime->x, x_bytes), "cudaMalloc(x)");
        ThrowOnCuda(cudaMalloc(&runtime->combine, combine_bytes), "cudaMalloc(combine)");
        ThrowOnCuda(cudaMalloc(&runtime->d_y, x_bytes), "cudaMalloc(d_y)");
        ThrowOnCuda(cudaMalloc(&runtime->d_x_routed, combine_bytes), "cudaMalloc(d_x_routed)");
        ThrowOnCuda(cudaMalloc(&runtime->router_weights, router_bytes), "cudaMalloc(router_weights)");
        ThrowOnCuda(cudaMalloc(&runtime->d_router_weights, router_bytes), "cudaMalloc(d_router_weights)");
        ThrowOnCuda(cudaMalloc(&runtime->signals, num_devices * sizeof(uint32_t)), "cudaMalloc(signals)");
        ThrowOnCuda(cudaMalloc(&runtime->epoch, sizeof(uint32_t)), "cudaMalloc(epoch)");
        ThrowOnCuda(cudaMemset(runtime->signals, 0, num_devices * sizeof(uint32_t)), "cudaMemset(signals)");
        ThrowOnCuda(cudaMemset(runtime->epoch, 0, sizeof(uint32_t)), "cudaMemset(epoch)");
        runtimes_[rank] = std::move(runtime);
      }
      for (int rank = 0; rank < num_devices; ++rank) {
        ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(peer)");
        auto& runtime = *runtimes_[rank];
        for (int peer = 0; peer < num_devices; ++peer) {
          runtime.x_ptrs[peer] = runtimes_[peer]->x;
          runtime.combine_ptrs[peer] = runtimes_[peer]->combine;
          runtime.d_y_ptrs[peer] = runtimes_[peer]->d_y;
          runtime.d_x_routed_ptrs[peer] = runtimes_[peer]->d_x_routed;
          runtime.router_weight_ptrs[peer] = runtimes_[peer]->router_weights;
          runtime.d_router_weight_ptrs[peer] = runtimes_[peer]->d_router_weights;
          runtime.signal_ptrs[peer] = runtimes_[peer]->signals;
          if (peer == rank) {
            continue;
          }
          int can_access = 0;
          ThrowOnCuda(cudaDeviceCanAccessPeer(&can_access, rank, peer), "cudaDeviceCanAccessPeer");
          if (can_access == 0) {
            throw std::runtime_error("Mixture-of-Kittens requires peer access between all GPUs");
          }
          int native_atomics = 0;
          ThrowOnCuda(
              cudaDeviceGetP2PAttribute(&native_atomics, cudaDevP2PAttrNativeAtomicSupported, rank, peer),
              "cudaDeviceGetP2PAttribute(native atomics)");
          if (native_atomics == 0) {
            throw std::runtime_error("Mixture-of-Kittens requires native peer atomics between all GPUs");
          }
          EnablePeerAccess(peer);
        }
        ThrowOnCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");
      }
    } catch (...) {
      (void)cudaSetDevice(original_device);
      DestroyLocked();
      throw;
    }
    ThrowOnCuda(cudaSetDevice(original_device), "cudaSetDevice(restore)");
    initialized_ = true;
  }

  void Shutdown() {
    std::lock_guard<std::mutex> lock(mu_);
    DestroyLocked();
  }

  DeviceRuntime& Current() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_) {
      throw std::runtime_error("Mixture-of-Kittens runtime is not initialized");
    }
    int device = -1;
    ThrowOnCuda(cudaGetDevice(&device), "cudaGetDevice(current)");
    if (device < 0 || device >= static_cast<int>(runtimes_.size()) || runtimes_[device] == nullptr) {
      throw std::runtime_error("No Mixture-of-Kittens runtime exists for the current GPU");
    }
    return *runtimes_[device];
  }

 private:
  void DestroyLocked() {
    if (runtimes_.empty()) {
      initialized_ = false;
      return;
    }
    int original_device = 0;
    (void)cudaGetDevice(&original_device);
    for (auto& runtime : runtimes_) {
      if (runtime == nullptr) {
        continue;
      }
      (void)cudaSetDevice(runtime->device);
      (void)cudaFree(runtime->epoch);
      (void)cudaFree(runtime->signals);
      (void)cudaFree(runtime->d_router_weights);
      (void)cudaFree(runtime->router_weights);
      (void)cudaFree(runtime->d_x_routed);
      (void)cudaFree(runtime->d_y);
      (void)cudaFree(runtime->combine);
      (void)cudaFree(runtime->x);
    }
    (void)cudaSetDevice(original_device);
    runtimes_.clear();
    initialized_ = false;
    num_devices_ = 0;
    num_tokens_ = 0;
    hidden_dim_ = 0;
    top_k_ = 0;
  }

  std::mutex mu_;
  bool initialized_ = false;
  int num_devices_ = 0;
  int num_tokens_ = 0;
  int hidden_dim_ = 0;
  int top_k_ = 0;
  std::vector<std::unique_ptr<DeviceRuntime>> runtimes_;
};

struct PeerBarrierArgs {
  std::array<uint32_t*, kNumDevices> signal_ptrs;
  uint32_t* local_signals;
  uint32_t* epoch;
  int rank;
};

__global__ void PeerBarrierKernel(PeerBarrierArgs args) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  const uint32_t target = atomicAdd_system(args.epoch, 1U) + 1U;
  __threadfence_system();
  for (int peer = 0; peer < kNumDevices; ++peer) {
    atomicExch_system(args.signal_ptrs[peer] + args.rank, target);
  }
  for (int peer = 0; peer < kNumDevices; ++peer) {
    while (atomicAdd_system(args.local_signals + peer, 0U) < target) {
      __nanosleep(64);
    }
  }
  __threadfence_system();
}

void LaunchPeerBarrier(DeviceRuntime& runtime, cudaStream_t stream) {
  PeerBarrierArgs args{
      .signal_ptrs = runtime.signal_ptrs,
      .local_signals = runtime.signals,
      .epoch = runtime.epoch,
      .rank = runtime.rank,
  };
  PeerBarrierKernel<<<1, 1, 0, stream>>>(args);
  ThrowOnCuda(cudaGetLastError(), "PeerBarrierKernel");
}

template <typename Config, typename Globals, auto Kernel>
__global__ __launch_bounds__(Config::NUM_THREADS, 1) void GlobalKernel(const __grid_constant__ Globals globals) {
  Kernel(globals);
}

template <typename Config, typename Globals, auto Kernel>
void LaunchKernel(const Globals& globals, cudaStream_t stream) {
  const dim3 grid = globals.grid();
  const dim3 block{Config::NUM_THREADS, 1, 1};
  const int shared_memory = Config::DYNAMIC_SHARED_MEMORY;
  ThrowOnCuda(
      cudaFuncSetAttribute(
          GlobalKernel<Config, Globals, Kernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          shared_memory),
      "cudaFuncSetAttribute(shared memory)");
  kittens::LaunchConfig<true, false> launch_config(
      grid,
      block,
      shared_memory,
      stream,
      Config::CLUSTER_SIZE);
  ThrowOnCuda(
      cudaLaunchKernelEx(launch_config, GlobalKernel<Config, Globals, Kernel>, globals),
      "cudaLaunchKernelEx(MoK forward)");
}

__global__ void ForwardEpilogueKernel(
    const __nv_bfloat16* y_shared,
    const __nv_bfloat16* combine,
    const float* router_weights,
    __nv_bfloat16* output,
    int num_tokens,
    int hidden_dim,
    int top_k) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t num_elements = static_cast<size_t>(num_tokens) * hidden_dim;
  if (index >= num_elements) {
    return;
  }
  const int token = static_cast<int>(index / hidden_dim);
  const int hidden = static_cast<int>(index % hidden_dim);
  float value = __bfloat162float(y_shared[static_cast<size_t>(token) * hidden_dim + hidden]);
  for (int k = 0; k < top_k; ++k) {
    const size_t route = static_cast<size_t>(token) * top_k + k;
    value += router_weights[route] * __bfloat162float(combine[route * hidden_dim + hidden]);
  }
  output[static_cast<size_t>(token) * hidden_dim + hidden] = __float2bfloat16_rn(value);
}

__global__ void BackwardEpilogueKernel(
    __nv_bfloat16* d_x,
    const __nv_bfloat16* d_x_routed,
    int num_tokens,
    int hidden_dim,
    int top_k) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t num_elements = static_cast<size_t>(num_tokens) * hidden_dim;
  if (index >= num_elements) {
    return;
  }
  const int token = static_cast<int>(index / hidden_dim);
  const int hidden = static_cast<int>(index % hidden_dim);
  float value = __bfloat162float(d_x[static_cast<size_t>(token) * hidden_dim + hidden]);
  for (int k = 0; k < top_k; ++k) {
    const size_t route = static_cast<size_t>(token) * top_k + k;
    value += __bfloat162float(d_x_routed[route * hidden_dim + hidden]);
  }
  d_x[static_cast<size_t>(token) * hidden_dim + hidden] = __float2bfloat16_rn(value);
}

template <typename GL, typename T>
GL MakeGl(T* pointer, int batch, int depth, int rows, int columns) {
  return kittens::make_gl<GL>(
      reinterpret_cast<uint64_t>(pointer), batch, depth, rows, columns);
}

ffi::Error ForwardBf16(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> x,
    ffi::Buffer<ffi::F32, 2> router_weights,
    ffi::Buffer<ffi::BF16, 2> shared_gate,
    ffi::Buffer<ffi::BF16, 3> routed_gate,
    ffi::Buffer<ffi::BF16, 2> shared_up,
    ffi::Buffer<ffi::BF16, 3> routed_up,
    ffi::Buffer<ffi::BF16, 2> shared_down,
    ffi::Buffer<ffi::BF16, 3> routed_down,
    ffi::Buffer<ffi::S32, 1> schedule_peer_rank,
    ffi::Buffer<ffi::S32, 1> schedule_peer_token_idx,
    ffi::Buffer<ffi::S32, 1> num_tokens,
    ffi::Buffer<ffi::S32, 1> tokens_per_expert,
    int32_t top_k,
    int32_t num_comm_sms,
    int32_t macrobatch_size,
    int32_t minibatch_size,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> output,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> x_routed,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> gate_shared,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> gate_routed,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> up_shared,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> up_routed,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> hidden_shared,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> hidden_routed,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> y_shared,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> y_routed,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> x_routed_ready,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> gate_up_tile_ready,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> hidden_row_block_ready,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> y_routed_ready,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> y_routed_done) {
  try {
    DeviceRuntime& runtime = RuntimeManager::Instance().Current();
    const int local_tokens = static_cast<int>(x.dimensions()[0]);
    const int hidden_dim = static_cast<int>(x.dimensions()[1]);
    const int intermediate_dim = static_cast<int>(shared_gate.dimensions()[0]);
    const int local_experts = static_cast<int>(routed_gate.dimensions()[0]);
    const int schedule_capacity = static_cast<int>(schedule_peer_rank.dimensions()[0]);
    if (router_weights.dimensions()[0] != local_tokens || router_weights.dimensions()[1] != top_k) {
      return ffi::Error::InvalidArgument("router_weights shape does not match x and top_k");
    }
    if (schedule_peer_token_idx.dimensions()[0] != schedule_capacity) {
      return ffi::Error::InvalidArgument("schedule arrays have different sizes");
    }
    if (num_tokens.dimensions()[0] != 1 || tokens_per_expert.dimensions()[0] != local_experts) {
      return ffi::Error::InvalidArgument("schedule count tensor shape mismatch");
    }
    if (shared_gate.dimensions()[1] != hidden_dim || shared_up.dimensions()[0] != intermediate_dim ||
        shared_up.dimensions()[1] != hidden_dim || shared_down.dimensions()[0] != hidden_dim ||
        shared_down.dimensions()[1] != intermediate_dim) {
      return ffi::Error::InvalidArgument("shared weight shape mismatch");
    }
    if (routed_gate.dimensions()[1] != intermediate_dim || routed_gate.dimensions()[2] != hidden_dim ||
        routed_up.dimensions() != routed_gate.dimensions() || routed_down.dimensions()[0] != local_experts ||
        routed_down.dimensions()[1] != hidden_dim || routed_down.dimensions()[2] != intermediate_dim) {
      return ffi::Error::InvalidArgument("routed weight shape mismatch");
    }
    if (local_tokens % MoK::config::MLP_Mb != 0 || schedule_capacity % MoK::config::MLP_Mb != 0 ||
        hidden_dim % MoK::config::MLP_Nb != 0 || intermediate_dim % MoK::config::MLP_Nb != 0 ||
        macrobatch_size <= 0 || minibatch_size <= 0 || macrobatch_size % minibatch_size != 0 ||
        schedule_capacity % minibatch_size != 0) {
      return ffi::Error::InvalidArgument("MoK dimensions do not meet the 256-row and 256-column tile rules");
    }

    const size_t x_bytes = static_cast<size_t>(local_tokens) * hidden_dim * sizeof(uint16_t);
    ThrowOnCuda(
        cudaMemcpyAsync(runtime.x, x.typed_data(), x_bytes, cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpyAsync(x workspace)");
    ThrowOnCuda(cudaMemsetAsync(x_routed_ready->typed_data(), 0, x_routed_ready->size_bytes(), stream), "memset(x ready)");
    ThrowOnCuda(cudaMemsetAsync(gate_up_tile_ready->typed_data(), 0, gate_up_tile_ready->size_bytes(), stream), "memset(gate ready)");
    ThrowOnCuda(
        cudaMemsetAsync(hidden_row_block_ready->typed_data(), 0, hidden_row_block_ready->size_bytes(), stream),
        "memset(hidden ready)");
    ThrowOnCuda(cudaMemsetAsync(y_routed_ready->typed_data(), 0, y_routed_ready->size_bytes(), stream), "memset(y ready)");
    ThrowOnCuda(cudaMemsetAsync(y_routed_done->typed_data(), 0, y_routed_done->size_bytes(), stream), "memset(y done)");
    LaunchPeerBarrier(runtime, stream);

    MoK::activation_bf16_pgl x_pointer_data;
    MoK::activation_bf16_pgl combine_pointer_data;
    for (int peer = 0; peer < kNumDevices; ++peer) {
      x_pointer_data[peer] = reinterpret_cast<kittens::bf16*>(runtime.x_ptrs[peer]);
      combine_pointer_data[peer] = reinterpret_cast<kittens::bf16*>(runtime.combine_ptrs[peer]);
    }

    MoK::globals_fwd globals{
        .x_shared = MakeGl<MoK::mlp_bf16_gl>(reinterpret_cast<kittens::bf16*>(runtime.x), 1, 1, local_tokens, hidden_dim),
        .x_fp8_routed = MakeGl<MoK::routed_activation_gl>(reinterpret_cast<kittens::bf16*>(x_routed->typed_data()), 1, 1, macrobatch_size, hidden_dim),
        .x_sc_routed = {},
        .x_fp8_t_routed = MakeGl<MoK::routed_transposed_gl>(reinterpret_cast<kittens::bf16*>(x_routed->typed_data()), 1, 1, macrobatch_size, hidden_dim),
        .x_sc_t_routed = {},
        .gate_shared = MakeGl<MoK::epi_bf16_gl>(reinterpret_cast<kittens::bf16*>(gate_shared->typed_data()), 1, 1, local_tokens, intermediate_dim),
        .gate_routed = MakeGl<MoK::epi_bf16_gl>(reinterpret_cast<kittens::bf16*>(gate_routed->typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .gate_fp8_routed = MakeGl<MoK::routed_gate_up_gl>(reinterpret_cast<kittens::bf16*>(gate_routed->typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .gate_sc_routed = {},
        .up_shared = MakeGl<MoK::epi_bf16_gl>(reinterpret_cast<kittens::bf16*>(up_shared->typed_data()), 1, 1, local_tokens, intermediate_dim),
        .up_routed = MakeGl<MoK::epi_bf16_gl>(reinterpret_cast<kittens::bf16*>(up_routed->typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .up_fp8_routed = MakeGl<MoK::routed_gate_up_gl>(reinterpret_cast<kittens::bf16*>(up_routed->typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .up_sc_routed = {},
        .hidden_shared = MakeGl<MoK::mlp_bf16_gl>(reinterpret_cast<kittens::bf16*>(hidden_shared->typed_data()), 1, 1, local_tokens, intermediate_dim),
        .hidden_fp8_routed = MakeGl<MoK::routed_activation_gl>(reinterpret_cast<kittens::bf16*>(hidden_routed->typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .hidden_sc_routed = {},
        .hidden_fp8_t_routed = MakeGl<MoK::routed_transposed_gl>(reinterpret_cast<kittens::bf16*>(hidden_routed->typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .hidden_sc_t_routed = {},
        .y_shared = MakeGl<MoK::epi_bf16_gl>(reinterpret_cast<kittens::bf16*>(y_shared->typed_data()), 1, 1, local_tokens, hidden_dim),
        .y_routed = MakeGl<MoK::epi_bf16_gl>(reinterpret_cast<kittens::bf16*>(y_routed->typed_data()), 1, 1, macrobatch_size, hidden_dim),
        .x_routed_send_buffer = x_pointer_data,
        .y_routed_recv_buffer = combine_pointer_data,
        .w_shared_gate = MakeGl<MoK::weight_bf16_gl>(reinterpret_cast<kittens::bf16*>(shared_gate.typed_data()), 1, 1, intermediate_dim, hidden_dim),
        .w_routed_gate = MakeGl<MoK::routed_weight_gl>(reinterpret_cast<kittens::bf16*>(routed_gate.typed_data()), 1, local_experts, intermediate_dim, hidden_dim),
        .w_routed_gate_sc = {},
        .w_shared_up = MakeGl<MoK::weight_bf16_gl>(reinterpret_cast<kittens::bf16*>(shared_up.typed_data()), 1, 1, intermediate_dim, hidden_dim),
        .w_routed_up = MakeGl<MoK::routed_weight_gl>(reinterpret_cast<kittens::bf16*>(routed_up.typed_data()), 1, local_experts, intermediate_dim, hidden_dim),
        .w_routed_up_sc = {},
        .w_shared_down = MakeGl<MoK::weight_bf16_gl>(reinterpret_cast<kittens::bf16*>(shared_down.typed_data()), 1, 1, hidden_dim, intermediate_dim),
        .w_routed_down = MakeGl<MoK::routed_weight_gl>(reinterpret_cast<kittens::bf16*>(routed_down.typed_data()), 1, local_experts, hidden_dim, intermediate_dim),
        .w_routed_down_sc = {},
        .schedule_peer_rank = MakeGl<MoK::index_gl>(schedule_peer_rank.typed_data(), 1, 1, 1, schedule_capacity),
        .schedule_peer_token_idx = MakeGl<MoK::index_gl>(schedule_peer_token_idx.typed_data(), 1, 1, 1, schedule_capacity),
        .num_tokens = MakeGl<MoK::index_gl>(num_tokens.typed_data(), 1, 1, 1, 1),
        .tokens_per_expert = MakeGl<MoK::index_gl>(tokens_per_expert.typed_data(), 1, 1, 1, local_experts),
        .gate_up_tile_ready = MakeGl<MoK::index_gl>(gate_up_tile_ready->typed_data(), 1, 1, 1, gate_up_tile_ready->dimensions()[0]),
        .hidden_row_block_ready = MakeGl<MoK::index_gl>(hidden_row_block_ready->typed_data(), 1, 1, 1, hidden_row_block_ready->dimensions()[0]),
        .x_routed_ready = MakeGl<MoK::index_gl>(x_routed_ready->typed_data(), 1, 1, 1, x_routed_ready->dimensions()[0]),
        .y_routed_ready = MakeGl<MoK::index_gl>(y_routed_ready->typed_data(), 1, 1, 1, y_routed_ready->dimensions()[0]),
        .y_routed_done = MakeGl<MoK::index_gl>(y_routed_done->typed_data(), 1, 1, 1, y_routed_done->dimensions()[0]),
        .topk = top_k,
        .swiglu_limit = 0.0F,
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size,
    };
    LaunchKernel<MoK::config, MoK::globals_fwd, MoK::dispatch_mlp_swiglu_combine_fwd_kernel<false>>(
        globals,
        stream);
    LaunchPeerBarrier(runtime, stream);

    constexpr int kThreads = 256;
    const size_t output_elements = static_cast<size_t>(local_tokens) * hidden_dim;
    const dim3 grid((output_elements + kThreads - 1) / kThreads, 1, 1);
    ForwardEpilogueKernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(y_shared->typed_data()),
        reinterpret_cast<const __nv_bfloat16*>(runtime.combine),
        router_weights.typed_data(),
        reinterpret_cast<__nv_bfloat16*>(output->typed_data()),
        local_tokens,
        hidden_dim,
        top_k);
    ThrowOnCuda(cudaGetLastError(), "ForwardEpilogueKernel");
    return ffi::Error::Success();
  } catch (const std::exception& exc) {
    return ffi::Error::Internal(exc.what());
  }
}

ffi::Error BackwardBf16(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> grad_output,
    ffi::Buffer<ffi::BF16, 2> x,
    ffi::Buffer<ffi::F32, 2> router_weights,
    ffi::Buffer<ffi::BF16, 2> shared_gate,
    ffi::Buffer<ffi::BF16, 3> routed_gate,
    ffi::Buffer<ffi::BF16, 2> shared_up,
    ffi::Buffer<ffi::BF16, 3> routed_up,
    ffi::Buffer<ffi::BF16, 2> shared_down,
    ffi::Buffer<ffi::BF16, 3> routed_down,
    ffi::Buffer<ffi::BF16, 2> x_routed,
    ffi::Buffer<ffi::BF16, 2> gate_shared,
    ffi::Buffer<ffi::BF16, 2> gate_routed,
    ffi::Buffer<ffi::BF16, 2> up_shared,
    ffi::Buffer<ffi::BF16, 2> up_routed,
    ffi::Buffer<ffi::BF16, 2> hidden_shared,
    ffi::Buffer<ffi::BF16, 2> hidden_routed,
    ffi::Buffer<ffi::S32, 1> schedule_peer_rank,
    ffi::Buffer<ffi::S32, 1> schedule_peer_token_idx,
    ffi::Buffer<ffi::S32, 1> num_tokens,
    ffi::Buffer<ffi::S32, 1> tokens_per_expert,
    int32_t top_k,
    int32_t num_comm_sms,
    int32_t macrobatch_size,
    int32_t minibatch_size,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> d_x,
    ffi::Result<ffi::Buffer<ffi::F32, 2>> d_router_weights,
    ffi::Result<ffi::Buffer<ffi::F32, 4>> d_w_routed_gate,
    ffi::Result<ffi::Buffer<ffi::F32, 4>> d_w_routed_up,
    ffi::Result<ffi::Buffer<ffi::F32, 4>> d_w_routed_down,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> d_w_shared_gate,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> d_w_shared_up,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> d_w_shared_down,
    ffi::Result<ffi::Buffer<ffi::F32, 1>> router_weights_staged,
    ffi::Result<ffi::Buffer<ffi::F32, 2>> d_router_weight_partials,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> d_y_routed,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> d_hidden_shared,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> d_hidden_routed,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> d_gate_shared,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> d_gate_routed,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> d_up_shared,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> d_up_routed,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> d_x_routed,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> router_weights_ready,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> d_y_routed_ready,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> d_hidden_ready,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> d_gate_up_ready,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> d_x_routed_ready,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> replayed_x_routed_ready,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> replayed_gate_up_ready,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> replayed_hidden_ready,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> routed_buffers_done) {
  try {
    DeviceRuntime& runtime = RuntimeManager::Instance().Current();
    const int local_tokens = static_cast<int>(x.dimensions()[0]);
    const int hidden_dim = static_cast<int>(x.dimensions()[1]);
    const int intermediate_dim = static_cast<int>(shared_gate.dimensions()[0]);
    const int local_experts = static_cast<int>(routed_gate.dimensions()[0]);
    const int schedule_capacity = static_cast<int>(schedule_peer_rank.dimensions()[0]);
    if (grad_output.dimensions() != x.dimensions() || router_weights.dimensions()[0] != local_tokens ||
        router_weights.dimensions()[1] != top_k) {
      return ffi::Error::InvalidArgument("backward token input shapes do not match");
    }
    if (schedule_peer_token_idx.dimensions()[0] != schedule_capacity || num_tokens.dimensions()[0] != 1 ||
        tokens_per_expert.dimensions()[0] != local_experts) {
      return ffi::Error::InvalidArgument("backward schedule tensor shape mismatch");
    }
    if (macrobatch_size <= 0 || minibatch_size <= 0 || macrobatch_size % minibatch_size != 0 ||
        schedule_capacity % minibatch_size != 0) {
      return ffi::Error::InvalidArgument("backward batch sizes do not meet the kernel rules");
    }
    const int num_macrobatches = (schedule_capacity + macrobatch_size - 1) / macrobatch_size;
    if (x_routed.dimensions()[0] != macrobatch_size || x_routed.dimensions()[1] != hidden_dim ||
        gate_shared.dimensions()[0] != local_tokens || gate_shared.dimensions()[1] != intermediate_dim ||
        gate_routed.dimensions()[0] != macrobatch_size || gate_routed.dimensions()[1] != intermediate_dim ||
        up_shared.dimensions() != gate_shared.dimensions() || up_routed.dimensions() != gate_routed.dimensions() ||
        hidden_shared.dimensions() != gate_shared.dimensions() ||
        hidden_routed.dimensions() != gate_routed.dimensions()) {
      return ffi::Error::InvalidArgument("backward forward-context shape mismatch");
    }
    if (d_w_routed_gate->dimensions()[0] != num_macrobatches ||
        d_w_routed_gate->dimensions()[1] != local_experts ||
        d_w_routed_gate->dimensions()[2] != intermediate_dim ||
        d_w_routed_gate->dimensions()[3] != hidden_dim ||
        d_w_routed_up->dimensions() != d_w_routed_gate->dimensions() ||
        d_w_routed_down->dimensions()[0] != num_macrobatches ||
        d_w_routed_down->dimensions()[1] != local_experts ||
        d_w_routed_down->dimensions()[2] != hidden_dim ||
        d_w_routed_down->dimensions()[3] != intermediate_dim) {
      return ffi::Error::InvalidArgument("backward routed weight-gradient shape mismatch");
    }
    if (d_router_weight_partials->dimensions()[0] != macrobatch_size ||
        d_router_weight_partials->dimensions()[1] != intermediate_dim / MoK::config::SWIGLU_Nb) {
      return ffi::Error::InvalidArgument("backward router-gradient partial shape mismatch");
    }

    const size_t x_bytes = static_cast<size_t>(local_tokens) * hidden_dim * sizeof(uint16_t);
    const size_t routed_x_bytes = static_cast<size_t>(local_tokens) * top_k * hidden_dim * sizeof(uint16_t);
    const size_t router_bytes = static_cast<size_t>(local_tokens) * top_k * sizeof(float);
    ThrowOnCuda(cudaMemcpyAsync(runtime.d_y, grad_output.typed_data(), x_bytes, cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpyAsync(d_y workspace)");
    ThrowOnCuda(cudaMemcpyAsync(runtime.x, x.typed_data(), x_bytes, cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpyAsync(x backward workspace)");
    ThrowOnCuda(cudaMemcpyAsync(runtime.router_weights, router_weights.typed_data(), router_bytes,
                                cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpyAsync(router workspace)");
    ThrowOnCuda(cudaMemsetAsync(runtime.d_x_routed, 0, routed_x_bytes, stream), "memset(d_x routed workspace)");
    ThrowOnCuda(cudaMemsetAsync(runtime.d_router_weights, 0, router_bytes, stream),
                "memset(d_router workspace)");

    auto clear = [&](auto& result, const char* context) {
      ThrowOnCuda(cudaMemsetAsync(result->typed_data(), 0, result->size_bytes(), stream), context);
    };
    clear(router_weights_ready, "memset(router ready)");
    clear(d_y_routed_ready, "memset(d_y ready)");
    clear(d_hidden_ready, "memset(d_hidden ready)");
    clear(d_gate_up_ready, "memset(d_gate_up ready)");
    clear(d_x_routed_ready, "memset(d_x ready)");
    clear(replayed_x_routed_ready, "memset(replayed x ready)");
    clear(replayed_gate_up_ready, "memset(replayed gate/up ready)");
    clear(replayed_hidden_ready, "memset(replayed hidden ready)");
    clear(routed_buffers_done, "memset(routed buffers done)");
    clear(d_w_routed_gate, "memset(routed gate gradient partials)");
    clear(d_w_routed_up, "memset(routed up gradient partials)");
    clear(d_w_routed_down, "memset(routed down gradient partials)");
    LaunchPeerBarrier(runtime, stream);

    MoK::activation_bf16_pgl x_pointer_data;
    MoK::activation_bf16_pgl d_y_pointer_data;
    MoK::activation_bf16_pgl d_x_pointer_data;
    MoK::router_weight_pgl router_pointer_data;
    MoK::router_weight_pgl d_router_pointer_data;
    for (int peer = 0; peer < kNumDevices; ++peer) {
      x_pointer_data[peer] = reinterpret_cast<kittens::bf16*>(runtime.x_ptrs[peer]);
      d_y_pointer_data[peer] = reinterpret_cast<kittens::bf16*>(runtime.d_y_ptrs[peer]);
      d_x_pointer_data[peer] = reinterpret_cast<kittens::bf16*>(runtime.d_x_routed_ptrs[peer]);
      router_pointer_data[peer] = reinterpret_cast<float*>(runtime.router_weight_ptrs[peer]);
      d_router_pointer_data[peer] = reinterpret_cast<float*>(runtime.d_router_weight_ptrs[peer]);
    }

    MoK::globals_bwd globals{
        .x_shared = MakeGl<MoK::wgrad_bf16_gl>(reinterpret_cast<kittens::bf16*>(runtime.x), 1, 1, local_tokens, hidden_dim),
        .x_fp8_routed = MakeGl<MoK::routed_activation_gl>(reinterpret_cast<kittens::bf16*>(x_routed.typed_data()), 1, 1, macrobatch_size, hidden_dim),
        .x_sc_routed = {},
        .x_fp8_t_routed = MakeGl<MoK::routed_transposed_gl>(reinterpret_cast<kittens::bf16*>(x_routed.typed_data()), 1, 1, macrobatch_size, hidden_dim),
        .x_sc_t_routed = {},
        .gate_shared = MakeGl<MoK::swiglu_bf16_gl>(reinterpret_cast<kittens::bf16*>(gate_shared.typed_data()), 1, 1, local_tokens, intermediate_dim),
        .gate_routed = MakeGl<MoK::epi_bf16_gl>(reinterpret_cast<kittens::bf16*>(gate_routed.typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .gate_fp8_routed = MakeGl<MoK::routed_gate_up_gl>(reinterpret_cast<kittens::bf16*>(gate_routed.typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .gate_sc_routed = {},
        .up_shared = MakeGl<MoK::swiglu_bf16_gl>(reinterpret_cast<kittens::bf16*>(up_shared.typed_data()), 1, 1, local_tokens, intermediate_dim),
        .up_routed = MakeGl<MoK::epi_bf16_gl>(reinterpret_cast<kittens::bf16*>(up_routed.typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .up_fp8_routed = MakeGl<MoK::routed_gate_up_gl>(reinterpret_cast<kittens::bf16*>(up_routed.typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .up_sc_routed = {},
        .hidden_shared = MakeGl<MoK::wgrad_bf16_gl>(reinterpret_cast<kittens::bf16*>(hidden_shared.typed_data()), 1, 1, local_tokens, intermediate_dim),
        .hidden_fp8_routed = MakeGl<MoK::routed_activation_gl>(reinterpret_cast<kittens::bf16*>(hidden_routed.typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .hidden_sc_routed = {},
        .hidden_fp8_t_routed = MakeGl<MoK::routed_transposed_gl>(reinterpret_cast<kittens::bf16*>(hidden_routed.typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .hidden_sc_t_routed = {},
        .d_y_shared = MakeGl<MoK::mlp_bf16_gl>(reinterpret_cast<kittens::bf16*>(runtime.d_y), 1, 1, local_tokens, hidden_dim),
        .d_y_fp8_routed = MakeGl<MoK::routed_activation_gl>(reinterpret_cast<kittens::bf16*>(d_y_routed->typed_data()), 1, 1, macrobatch_size, hidden_dim),
        .d_y_sc_routed = {},
        .d_y_fp8_t_routed = MakeGl<MoK::routed_transposed_gl>(reinterpret_cast<kittens::bf16*>(d_y_routed->typed_data()), 1, 1, macrobatch_size, hidden_dim),
        .d_y_sc_t_routed = {},
        .d_hidden_shared = MakeGl<MoK::epi_bf16_gl>(reinterpret_cast<kittens::bf16*>(d_hidden_shared->typed_data()), 1, 1, local_tokens, intermediate_dim),
        .d_hidden_routed = MakeGl<MoK::epi_bf16_gl>(reinterpret_cast<kittens::bf16*>(d_hidden_routed->typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .d_gate_shared = MakeGl<MoK::mlp_bf16_gl>(reinterpret_cast<kittens::bf16*>(d_gate_shared->typed_data()), 1, 1, local_tokens, intermediate_dim),
        .d_gate_fp8_routed = MakeGl<MoK::routed_activation_gl>(reinterpret_cast<kittens::bf16*>(d_gate_routed->typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .d_gate_sc_routed = {},
        .d_gate_fp8_t_routed = MakeGl<MoK::routed_transposed_gl>(reinterpret_cast<kittens::bf16*>(d_gate_routed->typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .d_gate_sc_t_routed = {},
        .d_up_shared = MakeGl<MoK::mlp_bf16_gl>(reinterpret_cast<kittens::bf16*>(d_up_shared->typed_data()), 1, 1, local_tokens, intermediate_dim),
        .d_up_fp8_routed = MakeGl<MoK::routed_activation_gl>(reinterpret_cast<kittens::bf16*>(d_up_routed->typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .d_up_sc_routed = {},
        .d_up_fp8_t_routed = MakeGl<MoK::routed_transposed_gl>(reinterpret_cast<kittens::bf16*>(d_up_routed->typed_data()), 1, 1, macrobatch_size, intermediate_dim),
        .d_up_sc_t_routed = {},
        .d_x_shared = MakeGl<MoK::epi_bf16_gl>(reinterpret_cast<kittens::bf16*>(d_x->typed_data()), 1, 1, local_tokens, hidden_dim),
        .d_x_routed = MakeGl<MoK::epi_bf16_gl>(reinterpret_cast<kittens::bf16*>(d_x_routed->typed_data()), 1, 1, macrobatch_size, hidden_dim),
        .x_routed_send_buffer = x_pointer_data,
        .d_y_buffer = d_y_pointer_data,
        .d_x_routed_buffer = d_x_pointer_data,
        .router_weight_buffer = router_pointer_data,
        .d_router_weight_buffer = d_router_pointer_data,
        .router_weights = MakeGl<MoK::router_weight_gl>(router_weights_staged->typed_data(), 1, 1, 1, macrobatch_size),
        .d_router_weight_partials = MakeGl<MoK::d_router_weight_partials_gl>(d_router_weight_partials->typed_data(), 1, 1, macrobatch_size, intermediate_dim / MoK::config::SWIGLU_Nb),
        .w_routed_gate = MakeGl<MoK::routed_weight_gl>(reinterpret_cast<kittens::bf16*>(routed_gate.typed_data()), 1, local_experts, intermediate_dim, hidden_dim),
        .w_routed_gate_sc = {},
        .w_routed_up = MakeGl<MoK::routed_weight_gl>(reinterpret_cast<kittens::bf16*>(routed_up.typed_data()), 1, local_experts, intermediate_dim, hidden_dim),
        .w_routed_up_sc = {},
        .w_shared_gate = MakeGl<MoK::weight_bf16_gl>(reinterpret_cast<kittens::bf16*>(shared_gate.typed_data()), 1, 1, intermediate_dim, hidden_dim),
        .w_routed_gate_T = MakeGl<MoK::routed_weight_gl>(reinterpret_cast<kittens::bf16*>(routed_gate.typed_data()), 1, local_experts, intermediate_dim, hidden_dim),
        .w_routed_gate_T_sc = {},
        .w_shared_up = MakeGl<MoK::weight_bf16_gl>(reinterpret_cast<kittens::bf16*>(shared_up.typed_data()), 1, 1, intermediate_dim, hidden_dim),
        .w_routed_up_T = MakeGl<MoK::routed_weight_gl>(reinterpret_cast<kittens::bf16*>(routed_up.typed_data()), 1, local_experts, intermediate_dim, hidden_dim),
        .w_routed_up_T_sc = {},
        .w_shared_down = MakeGl<MoK::weight_bf16_gl>(reinterpret_cast<kittens::bf16*>(shared_down.typed_data()), 1, 1, hidden_dim, intermediate_dim),
        .w_routed_down_T = MakeGl<MoK::routed_weight_gl>(reinterpret_cast<kittens::bf16*>(routed_down.typed_data()), 1, local_experts, hidden_dim, intermediate_dim),
        .w_routed_down_T_sc = {},
        .d_w_shared_gate = MakeGl<MoK::d_weight_bf16_gl>(reinterpret_cast<kittens::bf16*>(d_w_shared_gate->typed_data()), 1, 1, intermediate_dim, hidden_dim),
        .d_w_routed_gate = MakeGl<MoK::d_routed_weight_f32_gl>(d_w_routed_gate->typed_data(), 1, num_macrobatches * local_experts, intermediate_dim, hidden_dim),
        .d_w_shared_up = MakeGl<MoK::d_weight_bf16_gl>(reinterpret_cast<kittens::bf16*>(d_w_shared_up->typed_data()), 1, 1, intermediate_dim, hidden_dim),
        .d_w_routed_up = MakeGl<MoK::d_routed_weight_f32_gl>(d_w_routed_up->typed_data(), 1, num_macrobatches * local_experts, intermediate_dim, hidden_dim),
        .d_w_shared_down = MakeGl<MoK::d_weight_bf16_gl>(reinterpret_cast<kittens::bf16*>(d_w_shared_down->typed_data()), 1, 1, hidden_dim, intermediate_dim),
        .d_w_routed_down = MakeGl<MoK::d_routed_weight_f32_gl>(d_w_routed_down->typed_data(), 1, num_macrobatches * local_experts, hidden_dim, intermediate_dim),
        .schedule_peer_rank = MakeGl<MoK::index_gl>(schedule_peer_rank.typed_data(), 1, 1, 1, schedule_capacity),
        .schedule_peer_token_idx = MakeGl<MoK::index_gl>(schedule_peer_token_idx.typed_data(), 1, 1, 1, schedule_capacity),
        .num_tokens = MakeGl<MoK::index_gl>(num_tokens.typed_data(), 1, 1, 1, 1),
        .tokens_per_expert = MakeGl<MoK::index_gl>(tokens_per_expert.typed_data(), 1, 1, 1, local_experts),
        .router_weights_ready = MakeGl<MoK::index_gl>(router_weights_ready->typed_data(), 1, 1, 1, router_weights_ready->dimensions()[0]),
        .d_y_routed_ready = MakeGl<MoK::index_gl>(d_y_routed_ready->typed_data(), 1, 1, 1, d_y_routed_ready->dimensions()[0]),
        .d_hidden_ready = MakeGl<MoK::index_gl>(d_hidden_ready->typed_data(), 1, 1, 1, d_hidden_ready->dimensions()[0]),
        .d_gate_up_ready = MakeGl<MoK::index_gl>(d_gate_up_ready->typed_data(), 1, 1, 1, d_gate_up_ready->dimensions()[0]),
        .d_x_routed_ready = MakeGl<MoK::index_gl>(d_x_routed_ready->typed_data(), 1, 1, 1, d_x_routed_ready->dimensions()[0]),
        .replayed_x_routed_ready = MakeGl<MoK::index_gl>(replayed_x_routed_ready->typed_data(), 1, 1, 1, replayed_x_routed_ready->dimensions()[0]),
        .replayed_gate_up_ready = MakeGl<MoK::index_gl>(replayed_gate_up_ready->typed_data(), 1, 1, 1, replayed_gate_up_ready->dimensions()[0]),
        .replayed_hidden_ready = MakeGl<MoK::index_gl>(replayed_hidden_ready->typed_data(), 1, 1, 1, replayed_hidden_ready->dimensions()[0]),
        .routed_buffers_done = MakeGl<MoK::index_gl>(routed_buffers_done->typed_data(), 1, 1, 1, routed_buffers_done->dimensions()[0]),
        .topk = top_k,
        .swiglu_limit = 0.0F,
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size,
    };
    LaunchKernel<MoK::config, MoK::globals_bwd, MoK::dispatch_mlp_swiglu_combine_bwd_kernel<false>>(
        globals,
        stream);
    LaunchPeerBarrier(runtime, stream);

    constexpr int kThreads = 256;
    const size_t output_elements = static_cast<size_t>(local_tokens) * hidden_dim;
    const dim3 epilogue_grid((output_elements + kThreads - 1) / kThreads, 1, 1);
    BackwardEpilogueKernel<<<epilogue_grid, kThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(d_x->typed_data()),
        reinterpret_cast<const __nv_bfloat16*>(runtime.d_x_routed),
        local_tokens,
        hidden_dim,
        top_k);
    ThrowOnCuda(cudaGetLastError(), "BackwardEpilogueKernel");
    ThrowOnCuda(cudaMemcpyAsync(d_router_weights->typed_data(), runtime.d_router_weights, router_bytes,
                                cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpyAsync(d_router output)");
    return ffi::Error::Success();
  } catch (const std::exception& exc) {
    return ffi::Error::Internal(exc.what());
  }
}

auto ForwardBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Attr<int32_t>("top_k")
      .Attr<int32_t>("num_comm_sms")
      .Attr<int32_t>("macrobatch_size")
      .Attr<int32_t>("minibatch_size")
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>();
}

auto BackwardBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 3>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Attr<int32_t>("top_k")
      .Attr<int32_t>("num_comm_sms")
      .Attr<int32_t>("macrobatch_size")
      .Attr<int32_t>("minibatch_size")
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::F32, 2>>()
      .Ret<ffi::Buffer<ffi::F32, 4>>()
      .Ret<ffi::Buffer<ffi::F32, 4>>()
      .Ret<ffi::Buffer<ffi::F32, 4>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::F32, 1>>()
      .Ret<ffi::Buffer<ffi::F32, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>();
}

}  // namespace

extern "C" int levanter_mok_init_runtime(int num_devices, int num_tokens, int hidden_dim, int top_k) {
  try {
    RuntimeManager::Instance().Init(num_devices, num_tokens, hidden_dim, top_k);
    SetLastError("");
    return 0;
  } catch (const std::exception& exc) {
    SetLastError(exc.what());
    return 1;
  }
}

extern "C" int levanter_mok_shutdown_runtime() {
  try {
    RuntimeManager::Instance().Shutdown();
    SetLastError("");
    return 0;
  } catch (const std::exception& exc) {
    SetLastError(exc.what());
    return 1;
  }
}

extern "C" const char* levanter_mok_last_error() { return LastErrorStorage().c_str(); }

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_mok_forward_bf16_4,
    ForwardBf16,
    ForwardBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_mok_backward_bf16_4,
    BackwardBf16,
    BackwardBinding());
