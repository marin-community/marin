// Copyright The Levanter Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "config.hpp"
#include "kernels/api.cuh"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kMaxPeers = NUM_MAX_NVL_PEERS;
constexpr int kCounterTimeoutSeconds = NUM_CPU_TIMEOUT_SECS;
constexpr int kProbeDispatchTokens = 4096;
constexpr int kProbeDispatchHidden = 2048;
constexpr int kProbeDispatchTopK = 2;
constexpr int kProbeDispatchExperts = 128;

struct DispatchConfig {
  int num_sms;
  int num_max_send_tokens;
  int num_max_recv_tokens;

  bool operator==(const DispatchConfig& other) const {
    return num_sms == other.num_sms && num_max_send_tokens == other.num_max_send_tokens &&
           num_max_recv_tokens == other.num_max_recv_tokens;
  }
};

struct DeviceRuntime {
  int rank = -1;
  int num_ranks = 0;
  int device_id = -1;
  int64_t num_nvl_bytes = 0;
  bool peer_synced = false;
  DispatchConfig dispatch_config{};
  DispatchConfig combine_config{};

  void* local_buffer = nullptr;
  void* buffer_ptrs[kMaxPeers] = {nullptr};
  void** buffer_ptrs_gpu = nullptr;
  int* barrier_signal_ptrs[kMaxPeers] = {nullptr};
  int** barrier_signal_ptrs_gpu = nullptr;
  int* recv_channel_offset_scratch = nullptr;

  volatile int* moe_recv_counter = nullptr;
  int* moe_recv_counter_mapped = nullptr;
  volatile int* moe_recv_expert_counter = nullptr;
  int* moe_recv_expert_counter_mapped = nullptr;
  int last_num_recv_tokens = -1;

  cudaStream_t aux_stream = nullptr;

  int dispatch_num_channels() const { return dispatch_config.num_sms / 2; }
  int combine_num_channels() const { return combine_config.num_sms / 2; }
};

class ThreadBarrier {
 public:
  explicit ThreadBarrier(int count) : count_(count), remaining_(count), generation_(0) {}

  void Wait() {
    std::unique_lock<std::mutex> lock(mu_);
    const int generation = generation_;
    remaining_ -= 1;
    if (remaining_ == 0) {
      generation_ += 1;
      remaining_ = count_;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [&] { return generation_ != generation; });
  }

 private:
  const int count_;
  int remaining_;
  int generation_;
  std::mutex mu_;
  std::condition_variable cv_;
};

std::string& LastErrorStorage() {
  static std::string error;
  return error;
}

void SetLastError(std::string message) { LastErrorStorage() = std::move(message); }

bool HostDispatchDebugEnabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("LEVANTER_DEEPEP_HOST_DISPATCH_DEBUG");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

bool TrustRuntimeRecvCountEnabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("LEVANTER_DEEPEP_TRUST_RUNTIME_RECV_COUNT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

int64_t HostDispatchTimestampMicros() {
  static const auto start = std::chrono::steady_clock::now();
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
}

void LogHostDispatchStage(
    int rank,
    const char* stage,
    int num_tokens,
    int hidden,
    int num_experts,
    int num_topk,
    int num_recv_tokens = -1) {
  if (!HostDispatchDebugEnabled()) {
    return;
  }
  fprintf(
      stderr,
      "HOST_DISPATCH_STAGE {\"ts_us\":%lld,\"rank\":%d,\"stage\":\"%s\",\"num_tokens\":%d,\"hidden\":%d,"
      "\"num_experts\":%d,\"num_topk\":%d,\"num_recv_tokens\":%d}\n",
      static_cast<long long>(HostDispatchTimestampMicros()),
      rank,
      stage,
      num_tokens,
      hidden,
      num_experts,
      num_topk,
      num_recv_tokens);
}

ffi::Error CudaError(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return ffi::Error::Success();
  }
  return ffi::Error::Internal(std::string(context) + ": " + cudaGetErrorString(status));
}

void ThrowOnCuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

void EnablePeerAccess(int peer_device_id) {
  cudaError_t status = cudaDeviceEnablePeerAccess(peer_device_id, 0);
  if (status == cudaSuccess) {
    return;
  }
  if (status == cudaErrorPeerAccessAlreadyEnabled) {
    // cudaDeviceEnablePeerAccess leaves the runtime last-error state set even when
    // peer access is already enabled. Clear it so later JAX/XLA CUDA calls do not
    // fail before their own work starts.
    (void)cudaGetLastError();
    return;
  }
  throw std::runtime_error(std::string("cudaDeviceEnablePeerAccess: ") + cudaGetErrorString(status));
}

size_t Align128(size_t value) {
  return ((value + NUM_BUFFER_ALIGNMENT_BYTES - 1) / NUM_BUFFER_ALIGNMENT_BYTES) * NUM_BUFFER_ALIGNMENT_BYTES;
}

size_t GetNvlBufferSizeHint(const DispatchConfig& config, int64_t hidden_bytes, int num_ranks) {
  constexpr int kNumMaxTopK = 128;
  constexpr int kNumMaxScales = 128;
  const int num_rdma_ranks = std::max(num_ranks / NUM_MAX_NVL_PEERS, 1);
  const int num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
  const int num_channels = config.num_sms / 2;

  size_t num_bytes = 0;
  num_bytes += static_cast<size_t>(num_channels) * num_nvl_ranks * (2 * num_rdma_ranks + 3) * sizeof(int);
  num_bytes += static_cast<size_t>(num_channels) * num_nvl_ranks * config.num_max_recv_tokens * hidden_bytes;
  num_bytes += static_cast<size_t>(num_channels) * num_nvl_ranks * config.num_max_recv_tokens * kNumMaxTopK * sizeof(int64_t);
  num_bytes += static_cast<size_t>(num_channels) * num_nvl_ranks * config.num_max_recv_tokens * kNumMaxTopK * sizeof(float);
  num_bytes += static_cast<size_t>(num_channels) * num_nvl_ranks * config.num_max_recv_tokens * kNumMaxScales * sizeof(float);
  return Align128(num_bytes);
}

class RuntimeManager {
 public:
  static RuntimeManager& Instance() {
    static RuntimeManager manager;
    return manager;
  }

  void Init(
      int num_ranks,
      int64_t hidden_bytes,
      DispatchConfig dispatch_config,
      DispatchConfig combine_config) {
    std::lock_guard<std::mutex> lock(mu_);
    if (initialized_ && num_ranks_ == num_ranks && hidden_bytes_ == hidden_bytes &&
        dispatch_config_ == dispatch_config && combine_config_ == combine_config) {
      return;
    }
    DestroyLocked();

    if (num_ranks <= 0 || num_ranks > kMaxPeers) {
      throw std::runtime_error("DeepEP intranode JAX runtime only supports 1..8 ranks");
    }
    if (dispatch_config.num_sms % 2 != 0 || combine_config.num_sms % 2 != 0) {
      throw std::runtime_error("DeepEP intranode configs must use an even SM count");
    }
    int device_count = 0;
    ThrowOnCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count != num_ranks) {
      throw std::runtime_error(
          "DeepEP intranode JAX runtime currently requires the expert group to span all visible local GPUs");
    }

    num_ranks_ = num_ranks;
    hidden_bytes_ = hidden_bytes;
    dispatch_config_ = dispatch_config;
    combine_config_ = combine_config;
    num_nvl_bytes_ = static_cast<int64_t>(std::max(
        GetNvlBufferSizeHint(dispatch_config, hidden_bytes, num_ranks),
        GetNvlBufferSizeHint(combine_config, hidden_bytes, num_ranks)));
    runtimes_.resize(num_ranks);

    int original_device = 0;
    ThrowOnCuda(cudaGetDevice(&original_device), "cudaGetDevice");
    try {
      for (int rank = 0; rank < num_ranks; ++rank) {
        ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(init)");
        runtimes_[rank] = std::make_unique<DeviceRuntime>();
        InitLocalRuntime(*runtimes_[rank], rank);
      }
      for (int rank = 0; rank < num_ranks; ++rank) {
        ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(sync)");
        SyncLocalRuntime(*runtimes_[rank]);
      }
    } catch (...) {
      ThrowOnCuda(cudaSetDevice(original_device), "cudaSetDevice(restore-after-init-failure)");
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

  DeviceRuntime& RuntimeForCurrentDevice() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_) {
      throw std::runtime_error("DeepEP intranode JAX runtime is not initialized");
    }
    int device_id = 0;
    ThrowOnCuda(cudaGetDevice(&device_id), "cudaGetDevice(current)");
    for (auto& runtime : runtimes_) {
      if (runtime != nullptr && runtime->device_id == device_id) {
        return *runtime;
      }
    }
    throw std::runtime_error("No DeepEP intranode JAX runtime found for the current CUDA device");
  }

 private:
  RuntimeManager() = default;

  void InitLocalRuntime(DeviceRuntime& runtime, int rank) {
    runtime.rank = rank;
    runtime.num_ranks = num_ranks_;
    runtime.device_id = rank;
    runtime.num_nvl_bytes = num_nvl_bytes_;
    runtime.dispatch_config = dispatch_config_;
    runtime.combine_config = combine_config_;

    const size_t barrier_signal_bytes = kMaxPeers * sizeof(int);
    const size_t buffer_ptr_bytes = kMaxPeers * sizeof(void*);
    const size_t barrier_signal_ptr_bytes = kMaxPeers * sizeof(int*);
    ThrowOnCuda(
        cudaMalloc(
            &runtime.local_buffer,
            runtime.num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes),
        "cudaMalloc(local_buffer)");
    runtime.buffer_ptrs[rank] = runtime.local_buffer;
    runtime.buffer_ptrs_gpu = reinterpret_cast<void**>(
        static_cast<uint8_t*>(runtime.local_buffer) + runtime.num_nvl_bytes + barrier_signal_bytes);
    runtime.barrier_signal_ptrs[rank] =
        reinterpret_cast<int*>(static_cast<uint8_t*>(runtime.local_buffer) + runtime.num_nvl_bytes);
    runtime.barrier_signal_ptrs_gpu = reinterpret_cast<int**>(
        static_cast<uint8_t*>(runtime.local_buffer) + runtime.num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);
    ThrowOnCuda(cudaMemset(runtime.barrier_signal_ptrs[rank], 0, barrier_signal_bytes), "cudaMemset(barrier_signal)");

    ThrowOnCuda(
        cudaMalloc(
            &runtime.recv_channel_offset_scratch,
            runtime.dispatch_num_channels() * runtime.num_ranks * sizeof(int)),
        "cudaMalloc(recv_channel_offset_scratch)");

    void* recv_counter = nullptr;
    ThrowOnCuda(cudaMallocHost(&recv_counter, sizeof(int), cudaHostAllocMapped), "cudaMallocHost(moe_recv_counter)");
    runtime.moe_recv_counter = reinterpret_cast<volatile int*>(recv_counter);
    int* recv_counter_host = const_cast<int*>(runtime.moe_recv_counter);
    ThrowOnCuda(
        cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&runtime.moe_recv_counter_mapped),
            recv_counter_host,
            0),
        "cudaHostGetDevicePointer(moe_recv_counter)");
    *runtime.moe_recv_counter = -1;

    void* recv_expert_counter = nullptr;
    ThrowOnCuda(
        cudaMallocHost(&recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped),
        "cudaMallocHost(moe_recv_expert_counter)");
    runtime.moe_recv_expert_counter = reinterpret_cast<volatile int*>(recv_expert_counter);
    int* recv_expert_counter_host = const_cast<int*>(runtime.moe_recv_expert_counter);
    ThrowOnCuda(
        cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&runtime.moe_recv_expert_counter_mapped),
            recv_expert_counter_host,
            0),
        "cudaHostGetDevicePointer(moe_recv_expert_counter)");
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i) {
      runtime.moe_recv_expert_counter[i] = -1;
    }

    ThrowOnCuda(cudaStreamCreateWithFlags(&runtime.aux_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  }

  void SyncLocalRuntime(DeviceRuntime& runtime) {
    for (int peer = 0; peer < runtime.num_ranks; ++peer) {
      if (peer == runtime.rank) {
        continue;
      }
      int can_access_peer = 0;
      ThrowOnCuda(
          cudaDeviceCanAccessPeer(&can_access_peer, runtime.device_id, runtimes_[peer]->device_id),
          "cudaDeviceCanAccessPeer");
      if (can_access_peer == 0) {
        throw std::runtime_error("DeepEP intranode JAX runtime requires peer access between all local GPUs");
      }
      EnablePeerAccess(runtimes_[peer]->device_id);
      runtime.buffer_ptrs[peer] = runtimes_[peer]->local_buffer;
      runtime.barrier_signal_ptrs[peer] =
          reinterpret_cast<int*>(static_cast<uint8_t*>(runtime.buffer_ptrs[peer]) + runtime.num_nvl_bytes);
    }
    ThrowOnCuda(
        cudaMemcpy(
            runtime.buffer_ptrs_gpu,
            runtime.buffer_ptrs,
            sizeof(void*) * kMaxPeers,
            cudaMemcpyHostToDevice),
        "cudaMemcpy(buffer_ptrs_gpu)");
    ThrowOnCuda(
        cudaMemcpy(
            runtime.barrier_signal_ptrs_gpu,
            runtime.barrier_signal_ptrs,
            sizeof(int*) * kMaxPeers,
            cudaMemcpyHostToDevice),
        "cudaMemcpy(barrier_signal_ptrs_gpu)");
    ThrowOnCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(sync)");
    runtime.peer_synced = true;
  }

  void DestroyLocked() {
    if (runtimes_.empty()) {
      initialized_ = false;
      num_ranks_ = 0;
      hidden_bytes_ = 0;
      num_nvl_bytes_ = 0;
      return;
    }

    int original_device = 0;
    cudaGetDevice(&original_device);
    for (auto& runtime_ptr : runtimes_) {
      if (runtime_ptr == nullptr) {
        continue;
      }
      DeviceRuntime& runtime = *runtime_ptr;
      cudaSetDevice(runtime.device_id);
      if (runtime.peer_synced && runtime.barrier_signal_ptrs_gpu != nullptr && runtime.aux_stream != nullptr) {
        deep_ep::intranode::barrier(runtime.barrier_signal_ptrs_gpu, runtime.rank, runtime.num_ranks, runtime.aux_stream);
        cudaStreamSynchronize(runtime.aux_stream);
      }
      if (runtime.aux_stream != nullptr) {
        cudaStreamDestroy(runtime.aux_stream);
      }
      if (runtime.recv_channel_offset_scratch != nullptr) {
        cudaFree(runtime.recv_channel_offset_scratch);
      }
      if (runtime.local_buffer != nullptr) {
        cudaFree(runtime.local_buffer);
      }
      if (runtime.moe_recv_counter != nullptr) {
        cudaFreeHost(const_cast<int*>(runtime.moe_recv_counter));
      }
      if (runtime.moe_recv_expert_counter != nullptr) {
        cudaFreeHost(const_cast<int*>(runtime.moe_recv_expert_counter));
      }
    }
    cudaSetDevice(original_device);
    runtimes_.clear();
    initialized_ = false;
    num_ranks_ = 0;
    hidden_bytes_ = 0;
    num_nvl_bytes_ = 0;
  }

  std::mutex mu_;
  bool initialized_ = false;
  int num_ranks_ = 0;
  int64_t hidden_bytes_ = 0;
  int64_t num_nvl_bytes_ = 0;
  DispatchConfig dispatch_config_{};
  DispatchConfig combine_config_{};
  std::vector<std::unique_ptr<DeviceRuntime>> runtimes_;
};

void ResetRecvCounters(DeviceRuntime& runtime, int num_local_experts) {
  *runtime.moe_recv_counter = -1;
  for (int i = 0; i < num_local_experts; ++i) {
    runtime.moe_recv_expert_counter[i] = -1;
  }
}

void WaitForRecvCounts(DeviceRuntime& runtime, int num_local_experts, int* num_recv_tokens_out) {
  auto start_time = std::chrono::high_resolution_clock::now();
  while (true) {
    int num_recv_tokens = static_cast<int>(*runtime.moe_recv_counter);
    bool ready = (num_recv_tokens >= 0);
    for (int i = 0; i < num_local_experts && ready; ++i) {
      ready &= runtime.moe_recv_expert_counter[i] >= 0;
    }
    if (ready) {
      *num_recv_tokens_out = num_recv_tokens;
      return;
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    if (elapsed.count() > kCounterTimeoutSeconds) {
      throw std::runtime_error("DeepEP intranode JAX dispatch timed out waiting for recv counters");
    }
  }
}

void DispatchOnCurrentDevice(
    DeviceRuntime& runtime,
    cudaStream_t stream,
    const nv_bfloat16* x,
    const int64_t* topk_idx,
    const float* topk_weights,
    const int* num_tokens_per_rank,
    const int* num_tokens_per_expert,
    const bool* is_token_in_rank,
    int num_tokens,
    int hidden,
    int num_topk,
    int num_experts,
    nv_bfloat16* recv_x,
    int64_t* recv_topk_idx,
    float* recv_topk_weights,
    int* recv_src_idx,
    int* rank_prefix_matrix,
    int* channel_prefix_matrix,
    int* recv_channel_prefix_matrix,
    int* send_head,
    int* local_expert_counts,
    int* num_recv_tokens_out,
    int max_recv_tokens,
    bool synchronize_after_launch) {
  if (hidden <= 0 || (hidden * static_cast<int>(sizeof(nv_bfloat16))) % sizeof(int4) != 0) {
    throw std::runtime_error("DeepEP intranode dispatch requires hidden*element_size divisible by int4");
  }
  if (num_experts % runtime.num_ranks != 0) {
    throw std::runtime_error("DeepEP intranode dispatch requires num_experts divisible by num_ranks");
  }
  const int num_local_experts = num_experts / runtime.num_ranks;

  ResetRecvCounters(runtime, num_local_experts);
  LogHostDispatchStage(runtime.rank, "before_notify_dispatch", num_tokens, hidden, num_experts, num_topk);
  const int num_memset_int = runtime.dispatch_num_channels() * runtime.num_ranks * 4;
  deep_ep::intranode::notify_dispatch(
      num_tokens_per_rank,
      runtime.moe_recv_counter_mapped,
      runtime.num_ranks,
      num_tokens_per_expert,
      runtime.moe_recv_expert_counter_mapped,
      num_experts,
      num_tokens,
      is_token_in_rank,
      channel_prefix_matrix,
      rank_prefix_matrix,
      num_memset_int,
      1,
      runtime.buffer_ptrs_gpu,
      runtime.barrier_signal_ptrs_gpu,
      runtime.rank,
      stream,
      runtime.dispatch_num_channels());
  LogHostDispatchStage(runtime.rank, "after_notify_dispatch", num_tokens, hidden, num_experts, num_topk);

  int num_recv_tokens = -1;
  WaitForRecvCounts(runtime, num_local_experts, &num_recv_tokens);
  LogHostDispatchStage(
      runtime.rank,
      "after_wait_for_recv_counts",
      num_tokens,
      hidden,
      num_experts,
      num_topk,
      num_recv_tokens);
  if (num_recv_tokens > max_recv_tokens) {
    throw std::runtime_error("DeepEP intranode dispatch recv buffer is smaller than actual recv tokens");
  }
  ThrowOnCuda(
      cudaMemcpyAsync(
          local_expert_counts,
          const_cast<int*>(runtime.moe_recv_expert_counter),
          sizeof(int) * num_local_experts,
          cudaMemcpyHostToDevice,
          stream),
      "cudaMemcpyAsync(local_expert_counts)");
  *num_recv_tokens_out = num_recv_tokens;

#if defined(LEVANTER_DEEPEP_EXTENDED_INTRNODE_DISPATCH)
  deep_ep::intranode::dispatch(
      recv_x,
      nullptr,
      nullptr,
      recv_src_idx,
      recv_topk_idx,
      recv_topk_weights,
      recv_channel_prefix_matrix,
      send_head,
      x,
      nullptr,
      nullptr,
      topk_idx,
      topk_weights,
      is_token_in_rank,
      channel_prefix_matrix,
      num_tokens,
      0,
      hidden * static_cast<int>(sizeof(nv_bfloat16)) / sizeof(int4),
      num_topk,
      num_experts,
      0,
      0,
      0,
      0,
      0,
      0,
      runtime.buffer_ptrs_gpu,
      runtime.rank,
      runtime.num_ranks,
      stream,
      runtime.dispatch_config.num_sms,
      runtime.dispatch_config.num_max_send_tokens,
      runtime.dispatch_config.num_max_recv_tokens);
#else
  deep_ep::intranode::dispatch(
      recv_x,
      nullptr,
      recv_src_idx,
      recv_topk_idx,
      recv_topk_weights,
      recv_channel_prefix_matrix,
      send_head,
      x,
      nullptr,
      topk_idx,
      topk_weights,
      is_token_in_rank,
      channel_prefix_matrix,
      num_tokens,
      0,
      hidden * static_cast<int>(sizeof(nv_bfloat16)) / sizeof(int4),
      num_topk,
      num_experts,
      0,
      0,
      0,
      runtime.buffer_ptrs_gpu,
      runtime.rank,
      runtime.num_ranks,
      stream,
      runtime.dispatch_config.num_sms,
      runtime.dispatch_config.num_max_send_tokens,
      runtime.dispatch_config.num_max_recv_tokens);
#endif
  LogHostDispatchStage(
      runtime.rank,
      "after_dispatch_launch",
      num_tokens,
      hidden,
      num_experts,
      num_topk,
      num_recv_tokens);
  if (!synchronize_after_launch) {
    return;
  }

  ThrowOnCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(dispatch)");
  LogHostDispatchStage(
      runtime.rank,
      "after_dispatch_sync",
      num_tokens,
      hidden,
      num_experts,
      num_topk,
      num_recv_tokens);
}

ffi::Error DispatchIntranode(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> x,
    ffi::Buffer<ffi::S64, 2> topk_idx,
    ffi::Buffer<ffi::F32, 2> topk_weights,
    ffi::Buffer<ffi::S32, 1> num_tokens_per_rank,
    ffi::Buffer<ffi::S32, 1> num_tokens_per_expert,
    ffi::Buffer<ffi::PRED, 2> is_token_in_rank,
    int32_t num_experts,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> recv_x,
    ffi::Result<ffi::Buffer<ffi::S64, 2>> recv_topk_idx,
    ffi::Result<ffi::Buffer<ffi::F32, 2>> recv_topk_weights,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> recv_src_idx,
    ffi::Result<ffi::Buffer<ffi::S32, 2>> rank_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32, 2>> channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32, 2>> recv_channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32, 2>> send_head,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> local_expert_counts,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> num_recv_tokens_buffer) {
  try {
    DeviceRuntime& runtime = RuntimeManager::Instance().RuntimeForCurrentDevice();
    const auto x_dims = x.dimensions();
    const auto topk_dims = topk_idx.dimensions();
    const auto rank_dims = num_tokens_per_rank.dimensions();
    const auto expert_dims = num_tokens_per_expert.dimensions();
    const auto token_rank_dims = is_token_in_rank.dimensions();
    if (x_dims.size() != 2 || topk_dims.size() != 2) {
      return ffi::Error::InvalidArgument("DeepEP intranode dispatch expects rank-2 x and topk_idx");
    }
    if (rank_dims.size() != 1 || expert_dims.size() != 1 || token_rank_dims.size() != 2) {
      return ffi::Error::InvalidArgument("DeepEP intranode dispatch metadata ranks are invalid");
    }
    const int num_tokens = static_cast<int>(x_dims[0]);
    const int hidden = static_cast<int>(x_dims[1]);
    const int num_topk = static_cast<int>(topk_dims[1]);
    if (topk_dims[0] != num_tokens || topk_weights.dimensions()[0] != num_tokens ||
        topk_weights.dimensions()[1] != num_topk) {
      return ffi::Error::InvalidArgument("DeepEP intranode dispatch top-k tensors must match x");
    }
    if (rank_dims[0] != runtime.num_ranks || token_rank_dims[0] != num_tokens ||
        token_rank_dims[1] != runtime.num_ranks) {
      return ffi::Error::InvalidArgument("DeepEP intranode dispatch rank metadata shape mismatch");
    }
    if (expert_dims[0] != num_experts) {
      return ffi::Error::InvalidArgument("DeepEP intranode dispatch expert metadata shape mismatch");
    }
    if (hidden <= 0 || (hidden * static_cast<int>(ffi::ByteWidth(ffi::BF16))) % sizeof(int4) != 0) {
      return ffi::Error::InvalidArgument("DeepEP intranode dispatch requires hidden*element_size divisible by int4");
    }
    if (num_experts % runtime.num_ranks != 0) {
      return ffi::Error::InvalidArgument("DeepEP intranode dispatch requires num_experts divisible by num_ranks");
    }
    const int num_local_experts = num_experts / runtime.num_ranks;
    if (local_expert_counts->dimensions().size() != 1 || local_expert_counts->dimensions()[0] != num_local_experts) {
      return ffi::Error::InvalidArgument("DeepEP intranode dispatch local_expert_counts shape mismatch");
    }
    if (num_recv_tokens_buffer->dimensions().size() != 1 || num_recv_tokens_buffer->dimensions()[0] != 1) {
      return ffi::Error::InvalidArgument("DeepEP intranode dispatch num_recv_tokens buffer must have shape [1]");
    }
    const int num_channels = runtime.dispatch_num_channels();
    if (rank_prefix_matrix->dimensions().size() != 2 ||
        rank_prefix_matrix->dimensions()[0] != runtime.num_ranks ||
        rank_prefix_matrix->dimensions()[1] != runtime.num_ranks ||
        channel_prefix_matrix->dimensions().size() != 2 ||
        channel_prefix_matrix->dimensions()[0] != runtime.num_ranks ||
        channel_prefix_matrix->dimensions()[1] != num_channels ||
        recv_channel_prefix_matrix->dimensions().size() != 2 ||
        recv_channel_prefix_matrix->dimensions()[0] != runtime.num_ranks ||
        recv_channel_prefix_matrix->dimensions()[1] != num_channels ||
        send_head->dimensions().size() != 2 ||
        send_head->dimensions()[0] != num_tokens ||
        send_head->dimensions()[1] != runtime.num_ranks) {
      return ffi::Error::InvalidArgument("DeepEP intranode dispatch handle tensor shapes are invalid");
    }
    if (recv_x->dimensions().size() != 2 || recv_x->dimensions()[1] != hidden ||
        recv_src_idx->dimensions().size() != 1 || recv_src_idx->dimensions()[0] != recv_x->dimensions()[0] ||
        recv_topk_idx->dimensions().size() != 2 || recv_topk_idx->dimensions()[0] != recv_x->dimensions()[0] ||
        recv_topk_idx->dimensions()[1] != num_topk ||
        recv_topk_weights->dimensions().size() != 2 || recv_topk_weights->dimensions()[0] != recv_x->dimensions()[0] ||
        recv_topk_weights->dimensions()[1] != num_topk) {
      return ffi::Error::InvalidArgument("DeepEP intranode dispatch recv tensor shapes are invalid");
    }

    int num_recv_tokens = -1;
    DispatchOnCurrentDevice(
        runtime,
        stream,
        reinterpret_cast<const nv_bfloat16*>(x.typed_data()),
        topk_idx.typed_data(),
        topk_weights.typed_data(),
        num_tokens_per_rank.typed_data(),
        num_tokens_per_expert.typed_data(),
        is_token_in_rank.typed_data(),
        num_tokens,
        hidden,
        num_topk,
        num_experts,
        reinterpret_cast<nv_bfloat16*>(recv_x->typed_data()),
        recv_topk_idx->typed_data(),
        recv_topk_weights->typed_data(),
        recv_src_idx->typed_data(),
        rank_prefix_matrix->typed_data(),
        channel_prefix_matrix->typed_data(),
        recv_channel_prefix_matrix->typed_data(),
        send_head->typed_data(),
        local_expert_counts->typed_data(),
        &num_recv_tokens,
        static_cast<int>(recv_x->dimensions()[0]),
        false);

    cudaError_t status = cudaSuccess;
    runtime.last_num_recv_tokens = num_recv_tokens;
    status = cudaMemcpyAsync(
        num_recv_tokens_buffer->typed_data(),
        &runtime.last_num_recv_tokens,
        sizeof(int),
        cudaMemcpyHostToDevice,
        stream);
    if (status != cudaSuccess) {
      return CudaError(status, "cudaMemcpyAsync(num_recv_tokens)");
    }
    return ffi::Error::Success();
  } catch (const std::exception& exc) {
    return ffi::Error::Internal(exc.what());
  }
}

ffi::Error DispatchIntranodeCached(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> x,
    ffi::Buffer<ffi::PRED, 2> is_token_in_rank,
    ffi::Buffer<ffi::S32, 2> rank_prefix_matrix,
    ffi::Buffer<ffi::S32, 2> channel_prefix_matrix,
    ffi::Buffer<ffi::S32, 1> num_recv_tokens_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> recv_x,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> recv_src_idx,
    ffi::Result<ffi::Buffer<ffi::S32, 2>> recv_channel_prefix_matrix,
    ffi::Result<ffi::Buffer<ffi::S32, 2>> send_head) {
  try {
    DeviceRuntime& runtime = RuntimeManager::Instance().RuntimeForCurrentDevice();
    if (num_recv_tokens_buffer.dimensions().size() != 1 || num_recv_tokens_buffer.dimensions()[0] != 1) {
      return ffi::Error::InvalidArgument(
          "DeepEP intranode cached dispatch expects num_recv_tokens shape [1]");
    }
    int num_recv_tokens = -1;
    cudaError_t status = cudaSuccess;
    if (TrustRuntimeRecvCountEnabled() && runtime.last_num_recv_tokens >= 0) {
      num_recv_tokens = runtime.last_num_recv_tokens;
    } else {
      status = cudaMemcpyAsync(
          &num_recv_tokens,
          num_recv_tokens_buffer.typed_data(),
          sizeof(int),
          cudaMemcpyDeviceToHost,
          stream);
      if (status != cudaSuccess) {
        return CudaError(status, "cudaMemcpyAsync(read num_recv_tokens)");
      }
      status = cudaStreamSynchronize(stream);
      if (status != cudaSuccess) {
        return CudaError(status, "cudaStreamSynchronize(read num_recv_tokens)");
      }
      runtime.last_num_recv_tokens = num_recv_tokens;
    }

    const auto x_dims = x.dimensions();
    const auto token_rank_dims = is_token_in_rank.dimensions();
    const auto rank_dims = rank_prefix_matrix.dimensions();
    const auto channel_dims = channel_prefix_matrix.dimensions();
    const auto recv_x_dims = recv_x->dimensions();
    const auto recv_src_dims = recv_src_idx->dimensions();
    const auto recv_channel_dims = recv_channel_prefix_matrix->dimensions();
    const auto send_head_dims = send_head->dimensions();
    if (x_dims.size() != 2 || token_rank_dims.size() != 2 || rank_dims.size() != 2 || channel_dims.size() != 2 ||
        recv_x_dims.size() != 2 || recv_src_dims.size() != 1 || recv_channel_dims.size() != 2 ||
        send_head_dims.size() != 2) {
      return ffi::Error::InvalidArgument("DeepEP intranode cached dispatch expects rank-1/2 tensors");
    }
    const int num_tokens = static_cast<int>(x_dims[0]);
    const int hidden = static_cast<int>(x_dims[1]);
    const int num_channels = runtime.dispatch_num_channels();
    if (hidden <= 0 || (hidden * static_cast<int>(ffi::ByteWidth(ffi::BF16))) % sizeof(int4) != 0) {
      return ffi::Error::InvalidArgument(
          "DeepEP intranode cached dispatch requires hidden*element_size divisible by int4");
    }
    if (token_rank_dims[0] != num_tokens || token_rank_dims[1] != runtime.num_ranks) {
      return ffi::Error::InvalidArgument(
          "DeepEP intranode cached dispatch token/rank metadata shape mismatch");
    }
    if (rank_dims[0] != runtime.num_ranks || rank_dims[1] != runtime.num_ranks ||
        channel_dims[0] != runtime.num_ranks || channel_dims[1] != num_channels) {
      return ffi::Error::InvalidArgument(
          "DeepEP intranode cached dispatch handle tensor shapes are invalid");
    }
    if (recv_x_dims[1] != hidden || recv_src_dims[0] != recv_x_dims[0] || recv_x_dims[0] < num_recv_tokens ||
        recv_channel_dims[0] != runtime.num_ranks || recv_channel_dims[1] != num_channels ||
        send_head_dims[0] != num_tokens || send_head_dims[1] != runtime.num_ranks) {
      return ffi::Error::InvalidArgument(
          "DeepEP intranode cached dispatch output tensor shapes are invalid");
    }
    if (num_recv_tokens < 0) {
      return ffi::Error::InvalidArgument("DeepEP intranode cached dispatch num_recv_tokens is out of range");
    }

    const int num_memset_int = runtime.dispatch_num_channels() * runtime.num_ranks * 4;
    deep_ep::intranode::cached_notify_dispatch(
        rank_prefix_matrix.typed_data(),
        num_memset_int,
        runtime.buffer_ptrs_gpu,
        runtime.barrier_signal_ptrs_gpu,
        runtime.rank,
        runtime.num_ranks,
        stream);

#if defined(LEVANTER_DEEPEP_EXTENDED_INTRNODE_DISPATCH)
    deep_ep::intranode::dispatch(
        recv_x->typed_data(),
        nullptr,
        nullptr,
        recv_src_idx->typed_data(),
        nullptr,
        nullptr,
        recv_channel_prefix_matrix->typed_data(),
        send_head->typed_data(),
        x.typed_data(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        is_token_in_rank.typed_data(),
        channel_prefix_matrix.typed_data(),
        num_tokens,
        num_recv_tokens,
        hidden * static_cast<int>(ffi::ByteWidth(ffi::BF16)) / sizeof(int4),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        runtime.buffer_ptrs_gpu,
        runtime.rank,
        runtime.num_ranks,
        stream,
        runtime.dispatch_config.num_sms,
        runtime.dispatch_config.num_max_send_tokens,
        runtime.dispatch_config.num_max_recv_tokens);
#else
    deep_ep::intranode::dispatch(
        recv_x->typed_data(),
        nullptr,
        recv_src_idx->typed_data(),
        nullptr,
        nullptr,
        recv_channel_prefix_matrix->typed_data(),
        send_head->typed_data(),
        x.typed_data(),
        nullptr,
        nullptr,
        nullptr,
        is_token_in_rank.typed_data(),
        channel_prefix_matrix.typed_data(),
        num_tokens,
        num_recv_tokens,
        hidden * static_cast<int>(ffi::ByteWidth(ffi::BF16)) / sizeof(int4),
        0,
        0,
        0,
        0,
        0,
        runtime.buffer_ptrs_gpu,
        runtime.rank,
        runtime.num_ranks,
        stream,
        runtime.dispatch_config.num_sms,
        runtime.dispatch_config.num_max_send_tokens,
        runtime.dispatch_config.num_max_recv_tokens);
#endif
    return ffi::Error::Success();
  } catch (const std::exception& exc) {
    return ffi::Error::Internal(exc.what());
  }
}

ffi::Error CombineIntranode(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> recv_x,
    ffi::Buffer<ffi::F32, 2> recv_topk_weights,
    ffi::Buffer<ffi::S32, 1> recv_src_idx,
    ffi::Buffer<ffi::S32, 2> rank_prefix_matrix,
    ffi::Buffer<ffi::S32, 2> channel_prefix_matrix,
    ffi::Buffer<ffi::S32, 2> send_head,
    ffi::Buffer<ffi::S32, 1> num_recv_tokens_buffer,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> combined_x,
    ffi::Result<ffi::Buffer<ffi::F32, 2>> combined_topk_weights,
    ffi::Result<ffi::Buffer<ffi::S32, 2>> send_head_out) {
  try {
    DeviceRuntime& runtime = RuntimeManager::Instance().RuntimeForCurrentDevice();
    if (num_recv_tokens_buffer.dimensions().size() != 1 || num_recv_tokens_buffer.dimensions()[0] != 1) {
      return ffi::Error::InvalidArgument("DeepEP intranode combine expects num_recv_tokens shape [1]");
    }
    int num_recv_tokens = -1;
    cudaError_t status = cudaSuccess;
    if (TrustRuntimeRecvCountEnabled() && runtime.last_num_recv_tokens >= 0) {
      num_recv_tokens = runtime.last_num_recv_tokens;
    } else {
      status = cudaMemcpyAsync(
          &num_recv_tokens,
          num_recv_tokens_buffer.typed_data(),
          sizeof(int),
          cudaMemcpyDeviceToHost,
          stream);
      if (status != cudaSuccess) {
        return CudaError(status, "cudaMemcpyAsync(read num_recv_tokens)");
      }
      status = cudaStreamSynchronize(stream);
      if (status != cudaSuccess) {
        return CudaError(status, "cudaStreamSynchronize(read num_recv_tokens)");
      }
      runtime.last_num_recv_tokens = num_recv_tokens;
    }

    const auto recv_x_dims = recv_x.dimensions();
    const auto recv_topk_dims = recv_topk_weights.dimensions();
    const auto src_dims = recv_src_idx.dimensions();
    const auto rank_dims = rank_prefix_matrix.dimensions();
    const auto channel_dims = channel_prefix_matrix.dimensions();
    const auto send_head_dims = send_head.dimensions();
    if (recv_x_dims.size() != 2 || recv_topk_dims.size() != 2 || src_dims.size() != 1 ||
        rank_dims.size() != 2 || channel_dims.size() != 2 || send_head_dims.size() != 2) {
      return ffi::Error::InvalidArgument("DeepEP intranode combine expects rank-1/2 tensors");
    }
    const int hidden = static_cast<int>(recv_x_dims[1]);
    const int num_topk = static_cast<int>(recv_topk_dims[1]);
    const int combined_tokens = static_cast<int>(send_head_dims[0]);
    if (recv_topk_dims[0] != recv_x_dims[0] || src_dims[0] != recv_x_dims[0]) {
      return ffi::Error::InvalidArgument("DeepEP intranode combine recv tensors must share the same leading dim");
    }
    if (num_recv_tokens < 0 || num_recv_tokens > recv_x_dims[0]) {
      return ffi::Error::InvalidArgument("DeepEP intranode combine num_recv_tokens is out of range");
    }
    if (rank_dims[0] != runtime.num_ranks || rank_dims[1] != runtime.num_ranks ||
        channel_dims[0] != runtime.num_ranks || channel_dims[1] != runtime.combine_num_channels() ||
        send_head_dims[1] != runtime.num_ranks) {
      return ffi::Error::InvalidArgument("DeepEP intranode combine handle tensor shapes are invalid");
    }
    if (combined_x->dimensions().size() != 2 || combined_x->dimensions()[0] != combined_tokens ||
        combined_x->dimensions()[1] != hidden ||
        combined_topk_weights->dimensions().size() != 2 ||
        combined_topk_weights->dimensions()[0] != combined_tokens ||
        combined_topk_weights->dimensions()[1] != num_topk ||
        send_head_out->dimensions().size() != 2 ||
        send_head_out->dimensions()[0] != combined_tokens ||
        send_head_out->dimensions()[1] != runtime.num_ranks) {
      return ffi::Error::InvalidArgument("DeepEP intranode combine output shapes are invalid");
    }

    if (send_head_out->typed_data() != send_head.typed_data()) {
      status = cudaMemcpyAsync(
          send_head_out->typed_data(),
          send_head.typed_data(),
          sizeof(int) * combined_tokens * runtime.num_ranks,
          cudaMemcpyDeviceToDevice,
          stream);
      if (status != cudaSuccess) {
        return CudaError(status, "cudaMemcpyAsync(copy send_head)");
      }
    }

    deep_ep::intranode::cached_notify_combine(
        runtime.buffer_ptrs_gpu,
        send_head_out->typed_data(),
        runtime.combine_num_channels(),
        combined_tokens,
        runtime.combine_num_channels() * runtime.num_ranks * 2,
        runtime.barrier_signal_ptrs_gpu,
        runtime.rank,
        runtime.num_ranks,
        stream);

    deep_ep::intranode::combine(
        CUDA_R_16BF,
        combined_x->typed_data(),
        combined_topk_weights->typed_data(),
        recv_x.typed_data(),
        recv_topk_weights.typed_data(),
        nullptr,
        nullptr,
        recv_src_idx.typed_data(),
        rank_prefix_matrix.typed_data(),
        channel_prefix_matrix.typed_data(),
        send_head_out->typed_data(),
        num_recv_tokens,
        combined_tokens,
        hidden,
        num_topk,
        runtime.buffer_ptrs_gpu,
        runtime.rank,
        runtime.num_ranks,
        stream,
        runtime.combine_config.num_sms,
        runtime.combine_config.num_max_send_tokens,
        runtime.combine_config.num_max_recv_tokens);
    return ffi::Error::Success();
  } catch (const std::exception& exc) {
    return ffi::Error::Internal(exc.what());
  }
}

auto DispatchBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::S64, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::PRED, 2>>()
      .Attr<int32_t>("num_experts")
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::S64, 2>>()
      .Ret<ffi::Buffer<ffi::F32, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>();
}

auto DispatchCachedBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::PRED, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 2>>();
}

auto CombineBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::F32, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 2>>();
}

}  // namespace

extern "C" int levanter_deepep_init_intranode_runtime(
    int num_ranks,
    int64_t hidden_bytes,
    int dispatch_num_sms,
    int dispatch_num_max_send_tokens,
    int dispatch_num_max_recv_tokens,
    int combine_num_sms,
    int combine_num_max_send_tokens,
    int combine_num_max_recv_tokens) {
  try {
    RuntimeManager::Instance().Init(
        num_ranks,
        hidden_bytes,
        DispatchConfig{dispatch_num_sms, dispatch_num_max_send_tokens, dispatch_num_max_recv_tokens},
        DispatchConfig{combine_num_sms, combine_num_max_send_tokens, combine_num_max_recv_tokens});
    SetLastError("");
    return 0;
  } catch (const std::exception& exc) {
    SetLastError(exc.what());
    return 1;
  }
}

extern "C" void levanter_deepep_shutdown_intranode_runtime() {
  try {
    RuntimeManager::Instance().Shutdown();
    SetLastError("");
  } catch (const std::exception& exc) {
    SetLastError(exc.what());
  }
}

extern "C" const char* levanter_deepep_last_error() { return LastErrorStorage().c_str(); }

extern "C" int levanter_deepep_run_host_dispatch_round(
    int num_tokens,
    int hidden,
    int num_experts,
    int num_topk) {
  try {
    if (num_tokens <= 0 || hidden <= 0 || num_experts <= 0 || num_topk <= 0) {
      throw std::runtime_error("host dispatch round requires positive num_tokens/hidden/experts/topk");
    }

    RuntimeManager& manager = RuntimeManager::Instance();
    DeviceRuntime& runtime0 = manager.RuntimeForCurrentDevice();
    const int num_ranks = runtime0.num_ranks;
    const int local_experts = num_experts / num_ranks;
    if (num_experts % num_ranks != 0) {
      throw std::runtime_error("host dispatch round requires num_experts divisible by num_ranks");
    }
    if (num_topk > num_ranks) {
      throw std::runtime_error("host dispatch round currently requires num_topk <= num_ranks");
    }

    ThreadBarrier barrier(num_ranks);
    std::mutex error_mu;
    std::optional<std::string> first_error;
    std::vector<std::thread> threads;
    threads.reserve(num_ranks);

    for (int rank = 0; rank < num_ranks; ++rank) {
      threads.emplace_back([&, rank]() {
        try {
          ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(host dispatch round)");
          DeviceRuntime& runtime = manager.RuntimeForCurrentDevice();
          LogHostDispatchStage(rank, "thread_start", num_tokens, hidden, num_experts, num_topk);

          std::vector<void*> allocations;
          auto allocate_zeroed = [&](size_t num_bytes, const char* context) -> void* {
            void* ptr = nullptr;
            ThrowOnCuda(cudaMalloc(&ptr, num_bytes), context);
            allocations.push_back(ptr);
            ThrowOnCuda(cudaMemset(ptr, 0, num_bytes), context);
            return ptr;
          };

          const int recv_capacity = num_tokens * num_ranks;
          const int num_channels = runtime.dispatch_num_channels();
          const size_t x_bytes = static_cast<size_t>(num_tokens) * hidden * sizeof(nv_bfloat16);
          const size_t topk_idx_bytes = static_cast<size_t>(num_tokens) * num_topk * sizeof(int64_t);
          const size_t topk_weights_bytes = static_cast<size_t>(num_tokens) * num_topk * sizeof(float);
          const size_t rank_counts_bytes = static_cast<size_t>(num_ranks) * sizeof(int);
          const size_t expert_counts_bytes = static_cast<size_t>(num_experts) * sizeof(int);
          const size_t token_rank_bytes = static_cast<size_t>(num_tokens) * num_ranks * sizeof(bool);
          const size_t recv_x_bytes = static_cast<size_t>(recv_capacity) * hidden * sizeof(nv_bfloat16);
          const size_t recv_topk_idx_bytes = static_cast<size_t>(recv_capacity) * num_topk * sizeof(int64_t);
          const size_t recv_topk_weights_bytes = static_cast<size_t>(recv_capacity) * num_topk * sizeof(float);
          const size_t recv_src_idx_bytes = static_cast<size_t>(recv_capacity) * sizeof(int);
          const size_t rank_prefix_bytes = static_cast<size_t>(num_ranks) * num_ranks * sizeof(int);
          const size_t channel_prefix_bytes = static_cast<size_t>(num_ranks) * num_channels * sizeof(int);
          const size_t send_head_bytes = static_cast<size_t>(num_tokens) * num_ranks * sizeof(int);
          const size_t local_expert_counts_bytes = static_cast<size_t>(local_experts) * sizeof(int);

          auto* x = reinterpret_cast<nv_bfloat16*>(allocate_zeroed(x_bytes, "cudaMalloc(host round x)"));
          auto* topk_idx = reinterpret_cast<int64_t*>(allocate_zeroed(topk_idx_bytes, "cudaMalloc(host round topk_idx)"));
          auto* topk_weights =
              reinterpret_cast<float*>(allocate_zeroed(topk_weights_bytes, "cudaMalloc(host round topk_weights)"));
          auto* num_tokens_per_rank =
              reinterpret_cast<int*>(allocate_zeroed(rank_counts_bytes, "cudaMalloc(host round num_tokens_per_rank)"));
          auto* num_tokens_per_expert = reinterpret_cast<int*>(
              allocate_zeroed(expert_counts_bytes, "cudaMalloc(host round num_tokens_per_expert)"));
          auto* is_token_in_rank =
              reinterpret_cast<bool*>(allocate_zeroed(token_rank_bytes, "cudaMalloc(host round is_token_in_rank)"));
          auto* recv_x = reinterpret_cast<nv_bfloat16*>(allocate_zeroed(recv_x_bytes, "cudaMalloc(host round recv_x)"));
          auto* recv_topk_idx = reinterpret_cast<int64_t*>(
              allocate_zeroed(recv_topk_idx_bytes, "cudaMalloc(host round recv_topk_idx)"));
          auto* recv_topk_weights = reinterpret_cast<float*>(
              allocate_zeroed(recv_topk_weights_bytes, "cudaMalloc(host round recv_topk_weights)"));
          auto* recv_src_idx =
              reinterpret_cast<int*>(allocate_zeroed(recv_src_idx_bytes, "cudaMalloc(host round recv_src_idx)"));
          auto* rank_prefix_matrix =
              reinterpret_cast<int*>(allocate_zeroed(rank_prefix_bytes, "cudaMalloc(host round rank_prefix)"));
          auto* channel_prefix_matrix =
              reinterpret_cast<int*>(allocate_zeroed(channel_prefix_bytes, "cudaMalloc(host round channel_prefix)"));
          auto* recv_channel_prefix_matrix = reinterpret_cast<int*>(
              allocate_zeroed(channel_prefix_bytes, "cudaMalloc(host round recv_channel_prefix)"));
          auto* send_head =
              reinterpret_cast<int*>(allocate_zeroed(send_head_bytes, "cudaMalloc(host round send_head)"));
          auto* local_expert_counts = reinterpret_cast<int*>(
              allocate_zeroed(local_expert_counts_bytes, "cudaMalloc(host round local_expert_counts)"));

          std::vector<int64_t> host_topk_idx(static_cast<size_t>(num_tokens) * num_topk);
          std::vector<float> host_topk_weights(static_cast<size_t>(num_tokens) * num_topk, 1.0f / num_topk);
          std::vector<int> host_num_tokens_per_rank(num_ranks, 0);
          std::vector<int> host_num_tokens_per_expert(num_experts, 0);
          std::vector<uint8_t> host_is_token_in_rank(static_cast<size_t>(num_tokens) * num_ranks, 0);

          for (int slot = 0; slot < num_topk; ++slot) {
            const int dest_rank = (rank + 1 + slot) % num_ranks;
            const int expert = dest_rank * local_experts + (slot % local_experts);
            host_num_tokens_per_rank[dest_rank] = num_tokens;
            host_num_tokens_per_expert[expert] = num_tokens;
            for (int token = 0; token < num_tokens; ++token) {
              host_topk_idx[token * num_topk + slot] = expert;
              host_is_token_in_rank[token * num_ranks + dest_rank] = 1;
            }
          }

          ThrowOnCuda(
              cudaMemcpy(topk_idx, host_topk_idx.data(), topk_idx_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy(host round topk_idx)");
          ThrowOnCuda(
              cudaMemcpy(topk_weights, host_topk_weights.data(), topk_weights_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy(host round topk_weights)");
          ThrowOnCuda(
              cudaMemcpy(
                  num_tokens_per_rank, host_num_tokens_per_rank.data(), rank_counts_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy(host round num_tokens_per_rank)");
          ThrowOnCuda(
              cudaMemcpy(
                  num_tokens_per_expert,
                  host_num_tokens_per_expert.data(),
                  expert_counts_bytes,
                  cudaMemcpyHostToDevice),
              "cudaMemcpy(host round num_tokens_per_expert)");
          ThrowOnCuda(
              cudaMemcpy(is_token_in_rank, host_is_token_in_rank.data(), token_rank_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy(host round is_token_in_rank)");

          LogHostDispatchStage(rank, "before_barrier", num_tokens, hidden, num_experts, num_topk);
          barrier.Wait();
          LogHostDispatchStage(rank, "after_barrier", num_tokens, hidden, num_experts, num_topk);

          int num_recv_tokens = -1;
          DispatchOnCurrentDevice(
              runtime,
              runtime.aux_stream,
              x,
              topk_idx,
              topk_weights,
              num_tokens_per_rank,
              num_tokens_per_expert,
              is_token_in_rank,
              num_tokens,
              hidden,
              num_topk,
              num_experts,
              recv_x,
              recv_topk_idx,
              recv_topk_weights,
              recv_src_idx,
              rank_prefix_matrix,
              channel_prefix_matrix,
              recv_channel_prefix_matrix,
              send_head,
              local_expert_counts,
              &num_recv_tokens,
              recv_capacity,
              false);
          LogHostDispatchStage(
              rank,
              "before_launch_sync_barrier",
              num_tokens,
              hidden,
              num_experts,
              num_topk,
              num_recv_tokens);
          barrier.Wait();
          LogHostDispatchStage(
              rank,
              "after_launch_sync_barrier",
              num_tokens,
              hidden,
              num_experts,
              num_topk,
              num_recv_tokens);
          ThrowOnCuda(cudaStreamSynchronize(runtime.aux_stream), "cudaStreamSynchronize(dispatch)");
          LogHostDispatchStage(
              rank,
              "after_dispatch_sync",
              num_tokens,
              hidden,
              num_experts,
              num_topk,
              num_recv_tokens);
          LogHostDispatchStage(
              rank,
              "thread_done",
              num_tokens,
              hidden,
              num_experts,
              num_topk,
              num_recv_tokens);

          for (void* ptr : allocations) {
            cudaFree(ptr);
          }
        } catch (const std::exception& exc) {
          fprintf(stderr, "HOST_DISPATCH_ERROR {\"rank\":%d,\"error\":\"%s\"}\n", rank, exc.what());
          std::lock_guard<std::mutex> lock(error_mu);
          if (!first_error.has_value()) {
            first_error = "rank " + std::to_string(rank) + ": " + exc.what();
          }
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
    if (first_error.has_value()) {
      throw std::runtime_error(*first_error);
    }

    SetLastError("");
    return 0;
  } catch (const std::exception& exc) {
    SetLastError(exc.what());
    return 1;
  }
}

extern "C" int levanter_deepep_probe_dispatch_kernel_attributes() {
  try {
    DeviceRuntime& runtime = RuntimeManager::Instance().RuntimeForCurrentDevice();
    std::vector<void*> allocations;
    auto allocate_zeroed = [&](size_t num_bytes, const char* context) -> void* {
      void* ptr = nullptr;
      ThrowOnCuda(cudaMalloc(&ptr, num_bytes), context);
      allocations.push_back(ptr);
      ThrowOnCuda(cudaMemset(ptr, 0, num_bytes), context);
      return ptr;
    };

    const int hidden_int4 = kProbeDispatchHidden * static_cast<int>(sizeof(nv_bfloat16)) / sizeof(int4);
    const int max_recv_tokens = kProbeDispatchTokens * runtime.num_ranks;
    const size_t x_bytes = static_cast<size_t>(kProbeDispatchTokens) * kProbeDispatchHidden * sizeof(nv_bfloat16);
    const size_t recv_x_bytes = static_cast<size_t>(max_recv_tokens) * kProbeDispatchHidden * sizeof(nv_bfloat16);
    const size_t topk_idx_bytes = static_cast<size_t>(kProbeDispatchTokens) * kProbeDispatchTopK * sizeof(int64_t);
    const size_t recv_topk_idx_bytes = static_cast<size_t>(max_recv_tokens) * kProbeDispatchTopK * sizeof(int64_t);
    const size_t topk_weights_bytes = static_cast<size_t>(kProbeDispatchTokens) * kProbeDispatchTopK * sizeof(float);
    const size_t recv_topk_weights_bytes = static_cast<size_t>(max_recv_tokens) * kProbeDispatchTopK * sizeof(float);
    const size_t recv_src_idx_bytes = static_cast<size_t>(max_recv_tokens) * sizeof(int);
    const size_t send_head_bytes = static_cast<size_t>(kProbeDispatchTokens) * runtime.num_ranks * sizeof(int);
    const size_t token_rank_bytes = static_cast<size_t>(kProbeDispatchTokens) * runtime.num_ranks * sizeof(bool);
    const size_t channel_prefix_bytes =
        static_cast<size_t>(runtime.num_ranks) * runtime.dispatch_num_channels() * sizeof(int);

    void* recv_x = allocate_zeroed(recv_x_bytes, "cudaMalloc(probe recv_x)");
    void* recv_src_idx = allocate_zeroed(recv_src_idx_bytes, "cudaMalloc(probe recv_src_idx)");
    void* recv_topk_idx = allocate_zeroed(recv_topk_idx_bytes, "cudaMalloc(probe recv_topk_idx)");
    void* recv_topk_weights = allocate_zeroed(recv_topk_weights_bytes, "cudaMalloc(probe recv_topk_weights)");
    void* send_head = allocate_zeroed(send_head_bytes, "cudaMalloc(probe send_head)");
    void* x = allocate_zeroed(x_bytes, "cudaMalloc(probe x)");
    void* topk_idx = allocate_zeroed(topk_idx_bytes, "cudaMalloc(probe topk_idx)");
    void* topk_weights = allocate_zeroed(topk_weights_bytes, "cudaMalloc(probe topk_weights)");
    void* is_token_in_rank = allocate_zeroed(token_rank_bytes, "cudaMalloc(probe is_token_in_rank)");
    void* channel_prefix_matrix = allocate_zeroed(channel_prefix_bytes, "cudaMalloc(probe channel_prefix_matrix)");

#if defined(LEVANTER_DEEPEP_EXTENDED_INTRNODE_DISPATCH)
    deep_ep::intranode::dispatch(
        recv_x,
        nullptr,
        nullptr,
        reinterpret_cast<int*>(recv_src_idx),
        reinterpret_cast<int64_t*>(recv_topk_idx),
        reinterpret_cast<float*>(recv_topk_weights),
        runtime.recv_channel_offset_scratch,
        reinterpret_cast<int*>(send_head),
        x,
        nullptr,
        nullptr,
        reinterpret_cast<const int64_t*>(topk_idx),
        reinterpret_cast<const float*>(topk_weights),
        reinterpret_cast<const bool*>(is_token_in_rank),
        reinterpret_cast<const int*>(channel_prefix_matrix),
        kProbeDispatchTokens,
        0,
        hidden_int4,
        kProbeDispatchTopK,
        kProbeDispatchExperts,
        0,
        0,
        0,
        0,
        0,
        0,
        runtime.buffer_ptrs_gpu,
        runtime.rank,
        runtime.num_ranks,
        runtime.aux_stream,
        runtime.dispatch_config.num_sms,
        runtime.dispatch_config.num_max_send_tokens,
        runtime.dispatch_config.num_max_recv_tokens);
#else
    deep_ep::intranode::dispatch(
        recv_x,
        nullptr,
        reinterpret_cast<int*>(recv_src_idx),
        reinterpret_cast<int64_t*>(recv_topk_idx),
        reinterpret_cast<float*>(recv_topk_weights),
        runtime.recv_channel_offset_scratch,
        reinterpret_cast<int*>(send_head),
        x,
        nullptr,
        reinterpret_cast<const int64_t*>(topk_idx),
        reinterpret_cast<const float*>(topk_weights),
        reinterpret_cast<const bool*>(is_token_in_rank),
        reinterpret_cast<const int*>(channel_prefix_matrix),
        kProbeDispatchTokens,
        0,
        hidden_int4,
        kProbeDispatchTopK,
        kProbeDispatchExperts,
        0,
        0,
        0,
        runtime.buffer_ptrs_gpu,
        runtime.rank,
        runtime.num_ranks,
        runtime.aux_stream,
        runtime.dispatch_config.num_sms,
        runtime.dispatch_config.num_max_send_tokens,
        runtime.dispatch_config.num_max_recv_tokens);
#endif
    ThrowOnCuda(cudaStreamSynchronize(runtime.aux_stream), "cudaStreamSynchronize(probe dispatch)");

    for (void* ptr : allocations) {
      cudaFree(ptr);
    }

    SetLastError("");
    return 0;
  } catch (const std::exception& exc) {
    SetLastError(exc.what());
    return 1;
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_deepep_dispatch_intranode,
    DispatchIntranode,
    DispatchBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_deepep_dispatch_intranode_cached,
    DispatchIntranodeCached,
    DispatchCachedBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_deepep_combine_intranode,
    CombineIntranode,
    CombineBinding());
