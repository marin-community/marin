// Copyright The Levanter Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "mok_megakernel.cuh"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kNumDevices = 4;
constexpr int kMaxWorkspaceSlots = 2;
constexpr uint8_t kAllRanksMask = static_cast<uint8_t>((1U << kNumDevices) - 1U);
static_assert((kMaxWorkspaceSlots & (kMaxWorkspaceSlots - 1)) == 0);
constexpr int kPeerWaitPhaseCount = 4;
constexpr int kPeerWaitCellCount = kPeerWaitPhaseCount * kNumDevices;
constexpr auto kWorkspaceAcquireTimeout = std::chrono::minutes(5);
constexpr int kMemoryPoolStatsPerRank = 10;
constexpr int kMemoryPoolTrimOutputCount = kNumDevices * kMemoryPoolStatsPerRank + 3;

enum class PeerWaitPhase : int {
  kForwardPre = 0,
  kForwardPost = 1,
  kBackwardPre = 2,
  kBackwardPost = 3,
};

enum DebugCounterOffset : int {
  kPeerReadyWaits = 0,
  kCompletionWaits,
  kGenerationMismatches,
  kSlotReuseFailures,
  kSlotZeroAcquisitions,
  kSlotOneAcquisitions,
  kMaxActiveSlots,
  kPeerWaitEvents,
  kPeerWaitCycles = kPeerWaitEvents + kPeerWaitCellCount,
  kPeerWaitMaxCycles = kPeerWaitCycles + kPeerWaitCellCount,
  kForwardStagingCopyCalls = kPeerWaitMaxCycles + kPeerWaitCellCount,
  kForwardStagingCopyBytes,
  kBackwardStagingCopyCalls,
  kBackwardStagingCopyBytes,
  kDebugCounterCount,
};
static_assert(kPeerWaitEvents == 7);
static_assert(kPeerWaitCycles == 23);
static_assert(kPeerWaitMaxCycles == 39);
static_assert(kForwardStagingCopyCalls == 55);
static_assert(kDebugCounterCount == 59);

struct DebugCounters {
  uint64_t values[kDebugCounterCount];
};

std::atomic<int64_t> g_forward_calls{0};
std::atomic<int64_t> g_backward_calls{0};

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
  int slot = -1;
  int num_devices = 0;
  int workspace_slots = 0;
  void* x = nullptr;
  void* combine = nullptr;
  void* d_y = nullptr;
  void* d_x_routed = nullptr;
  void* router_weights = nullptr;
  void* d_router_weights = nullptr;
  uint64_t* generation = nullptr;
  uint64_t* forward_input_ready = nullptr;
  uint64_t* backward_input_ready = nullptr;
  uint64_t* forward_completions = nullptr;
  uint64_t* backward_completions = nullptr;
  uint64_t* last_forward_completion = nullptr;
  uint64_t* cancellation = nullptr;
  DebugCounters* debug_counters = nullptr;
  cudaStream_t cancellation_stream = nullptr;
  std::array<void*, kNumDevices> x_ptrs{};
  std::array<void*, kNumDevices> combine_ptrs{};
  std::array<void*, kNumDevices> d_y_ptrs{};
  std::array<void*, kNumDevices> d_x_routed_ptrs{};
  std::array<void*, kNumDevices> router_weight_ptrs{};
  std::array<void*, kNumDevices> d_router_weight_ptrs{};
  std::array<uint64_t*, kNumDevices> forward_input_ready_ptrs{};
  std::array<uint64_t*, kNumDevices> backward_input_ready_ptrs{};
  std::array<uint64_t*, kNumDevices> forward_completion_ptrs{};
  std::array<uint64_t*, kNumDevices> backward_completion_ptrs{};
  std::array<uint64_t*, kNumDevices> cancellation_ptrs{};
  std::array<uint64_t*, kMaxWorkspaceSlots> local_slot_forward_completion_ptrs{};

  DeviceRuntime() = default;
  DeviceRuntime(const DeviceRuntime&) = delete;
  DeviceRuntime& operator=(const DeviceRuntime&) = delete;

  ~DeviceRuntime() {
    if (device < 0) {
      return;
    }
    int original_device = device;
    (void)cudaGetDevice(&original_device);
    (void)cudaSetDevice(device);
    (void)cudaStreamDestroy(cancellation_stream);
    (void)cudaFree(debug_counters);
    (void)cudaFree(cancellation);
    (void)cudaFree(last_forward_completion);
    (void)cudaFree(backward_completions);
    (void)cudaFree(forward_completions);
    (void)cudaFree(backward_input_ready);
    (void)cudaFree(forward_input_ready);
    (void)cudaFree(generation);
    (void)cudaFree(d_router_weights);
    (void)cudaFree(router_weights);
    (void)cudaFree(d_x_routed);
    (void)cudaFree(d_y);
    (void)cudaFree(combine);
    (void)cudaFree(x);
    (void)cudaSetDevice(original_device);
  }
};

enum class InvocationPhase : uint8_t {
  kForward = 0,
  kBackward = 1,
};

enum class TestFailurePoint : int {
  kBeforeInputReady = 0,
  kBeforeCompletion = 1,
};

enum class ForwardXStorage : int32_t {
  kRuntimeStaged = 0,
  kXlaPeerExperimental = 1,
};

enum class BackwardPeerStorage : int32_t {
  kRuntimeStaged = 0,
  kXlaPeerExperimental = 1,
  kXlaPeerInputsExperimental = 2,
};

struct ForwardXRegistration {
  const void* pointer;
  size_t size_bytes;
  ForwardXStorage storage;
};

struct BackwardPeerRegistration {
  const void* d_y_pointer;
  const void* x_pointer;
  const void* router_weight_pointer;
  void* d_router_weight_pointer;
  size_t activation_size_bytes;
  size_t router_size_bytes;
  BackwardPeerStorage storage;
};

struct InvocationKey {
  int64_t run_id;
  int64_t collective_id;
  uint64_t ordinal;
  InvocationPhase phase;

  bool operator==(const InvocationKey& other) const {
    return run_id == other.run_id && collective_id == other.collective_id && ordinal == other.ordinal &&
           phase == other.phase;
  }
};

struct InvocationKeyHash {
  size_t operator()(const InvocationKey& key) const {
    size_t value = std::hash<int64_t>{}(key.run_id);
    value ^= std::hash<int64_t>{}(key.collective_id) + 0x9e3779b9 + (value << 6) + (value >> 2);
    value ^= std::hash<uint64_t>{}(key.ordinal) + 0x9e3779b9 + (value << 6) + (value >> 2);
    value ^= std::hash<uint8_t>{}(static_cast<uint8_t>(key.phase)) + 0x9e3779b9 + (value << 6) + (value >> 2);
    return value;
  }
};

struct RunPhaseKey {
  int64_t run_id;
  int64_t collective_id;
  InvocationPhase phase;

  bool operator==(const RunPhaseKey& other) const {
    return run_id == other.run_id && collective_id == other.collective_id && phase == other.phase;
  }
};

struct RunPhaseKeyHash {
  size_t operator()(const RunPhaseKey& key) const {
    size_t value = std::hash<int64_t>{}(key.run_id);
    value ^= std::hash<int64_t>{}(key.collective_id) + 0x9e3779b9 + (value << 6) + (value >> 2);
    value ^= std::hash<uint8_t>{}(static_cast<uint8_t>(key.phase)) + 0x9e3779b9 + (value << 6) + (value >> 2);
    return value;
  }
};

struct InvocationState {
  int slot = -1;
  uint64_t generation = 0;
  uint8_t arrival_mask = 0;
  uint8_t leased_mask = 0;
  uint8_t completion_mask = 0;
  bool cancelled = false;
  bool slot_released = false;
  std::array<const void*, kNumDevices> forward_x_ptrs{};
  size_t forward_x_size_bytes = 0;
  ForwardXStorage forward_x_storage = ForwardXStorage::kRuntimeStaged;
  uint8_t forward_x_mask = 0;
  std::array<const void*, kNumDevices> backward_d_y_ptrs{};
  std::array<const void*, kNumDevices> backward_x_ptrs{};
  std::array<const void*, kNumDevices> backward_router_weight_ptrs{};
  std::array<void*, kNumDevices> backward_d_router_weight_ptrs{};
  size_t backward_activation_size_bytes = 0;
  size_t backward_router_size_bytes = 0;
  BackwardPeerStorage backward_peer_storage = BackwardPeerStorage::kRuntimeStaged;
  uint8_t backward_peer_mask = 0;
  std::string error;
};

struct RuntimeLease {
  InvocationKey key;
  int rank;
  int slot;
  uint64_t generation;
  std::array<const void*, kNumDevices> forward_x_ptrs;
  size_t forward_x_size_bytes;
  ForwardXStorage forward_x_storage;
  std::array<const void*, kNumDevices> backward_d_y_ptrs;
  std::array<const void*, kNumDevices> backward_x_ptrs;
  std::array<const void*, kNumDevices> backward_router_weight_ptrs;
  std::array<void*, kNumDevices> backward_d_router_weight_ptrs;
  size_t backward_activation_size_bytes;
  size_t backward_router_size_bytes;
  BackwardPeerStorage backward_peer_storage;
};

class RuntimeManager {
 public:
  static RuntimeManager& Instance() {
    static RuntimeManager manager;
    return manager;
  }

  void Init(int num_devices, int num_tokens, int hidden_dim, int top_k, int workspace_slots) {
    std::unique_lock<std::mutex> lock(mu_);
    const bool maintenance_finished = cv_.wait_for(lock, kWorkspaceAcquireTimeout, [&] { return !maintenance_; });
    if (!maintenance_finished) {
      throw std::runtime_error("Mixture-of-Kittens runtime maintenance did not finish within five minutes");
    }
    if (initialized_ && num_devices_ == num_devices && num_tokens_ == num_tokens &&
        hidden_dim_ == hidden_dim && top_k_ == top_k && workspace_slots_ == workspace_slots) {
      if (!failure_message_.empty()) {
        throw std::runtime_error(failure_message_);
      }
      return;
    }
    if (initialized_) {
      throw std::runtime_error(
          "Mixture-of-Kittens runtime must be shut down before initializing a different signature");
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
    if (workspace_slots <= 0 || workspace_slots > kMaxWorkspaceSlots) {
      throw std::runtime_error("Mixture-of-Kittens workspace slots must be one or two");
    }

    num_devices_ = num_devices;
    num_tokens_ = num_tokens;
    hidden_dim_ = hidden_dim;
    top_k_ = top_k;
    workspace_slots_ = workspace_slots;
    runtimes_.resize(num_devices);
    int original_device = 0;
    ThrowOnCuda(cudaGetDevice(&original_device), "cudaGetDevice(init)");
    try {
      for (int rank = 0; rank < num_devices; ++rank) {
        ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(allocate)");
        for (int slot = 0; slot < workspace_slots_; ++slot) {
          auto runtime = std::make_unique<DeviceRuntime>();
          runtime->device = rank;
          runtime->rank = rank;
          runtime->slot = slot;
          runtime->num_devices = num_devices;
          runtime->workspace_slots = workspace_slots_;
          const size_t x_bytes = static_cast<size_t>(num_tokens) * hidden_dim * sizeof(uint16_t);
          const size_t combine_bytes = static_cast<size_t>(num_tokens) * top_k * hidden_dim * sizeof(uint16_t);
          const size_t router_bytes = static_cast<size_t>(num_tokens) * top_k * sizeof(float);
          ThrowOnCuda(cudaMalloc(&runtime->x, x_bytes), "cudaMalloc(x)");
          ThrowOnCuda(cudaMalloc(&runtime->combine, combine_bytes), "cudaMalloc(combine)");
          ThrowOnCuda(cudaMalloc(&runtime->d_y, x_bytes), "cudaMalloc(d_y)");
          ThrowOnCuda(cudaMalloc(&runtime->d_x_routed, combine_bytes), "cudaMalloc(d_x_routed)");
          ThrowOnCuda(cudaMalloc(&runtime->router_weights, router_bytes), "cudaMalloc(router_weights)");
          ThrowOnCuda(cudaMalloc(&runtime->d_router_weights, router_bytes), "cudaMalloc(d_router_weights)");
          ThrowOnCuda(cudaMalloc(&runtime->generation, sizeof(uint64_t)), "cudaMalloc(generation)");
          ThrowOnCuda(
              cudaMalloc(&runtime->forward_input_ready, num_devices * sizeof(uint64_t)),
              "cudaMalloc(forward input ready)");
          ThrowOnCuda(
              cudaMalloc(&runtime->backward_input_ready, num_devices * sizeof(uint64_t)),
              "cudaMalloc(backward input ready)");
          ThrowOnCuda(
              cudaMalloc(&runtime->forward_completions, num_devices * sizeof(uint64_t)),
              "cudaMalloc(forward completions)");
          ThrowOnCuda(
              cudaMalloc(&runtime->backward_completions, num_devices * sizeof(uint64_t)),
              "cudaMalloc(backward completions)");
          ThrowOnCuda(
              cudaMalloc(&runtime->last_forward_completion, sizeof(uint64_t)),
              "cudaMalloc(last forward completion)");
          ThrowOnCuda(cudaMalloc(&runtime->cancellation, sizeof(uint64_t)), "cudaMalloc(cancellation)");
          ThrowOnCuda(cudaMalloc(&runtime->debug_counters, sizeof(DebugCounters)), "cudaMalloc(debug counters)");
          ThrowOnCuda(
              cudaStreamCreateWithFlags(&runtime->cancellation_stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags(cancellation)");
          ThrowOnCuda(cudaMemset(runtime->generation, 0, sizeof(uint64_t)), "cudaMemset(generation)");
          ThrowOnCuda(
              cudaMemset(runtime->forward_input_ready, 0, num_devices * sizeof(uint64_t)),
              "cudaMemset(forward input ready)");
          ThrowOnCuda(
              cudaMemset(runtime->backward_input_ready, 0, num_devices * sizeof(uint64_t)),
              "cudaMemset(backward input ready)");
          ThrowOnCuda(
              cudaMemset(runtime->forward_completions, 0, num_devices * sizeof(uint64_t)),
              "cudaMemset(forward completions)");
          ThrowOnCuda(
              cudaMemset(runtime->backward_completions, 0, num_devices * sizeof(uint64_t)),
              "cudaMemset(backward completions)");
          ThrowOnCuda(
              cudaMemset(runtime->last_forward_completion, 0, sizeof(uint64_t)),
              "cudaMemset(last forward completion)");
          ThrowOnCuda(cudaMemset(runtime->cancellation, 0, sizeof(uint64_t)), "cudaMemset(cancellation)");
          ThrowOnCuda(cudaMemset(runtime->debug_counters, 0, sizeof(DebugCounters)), "cudaMemset(debug counters)");
          runtimes_[rank][slot] = std::move(runtime);
        }
      }
      for (int rank = 0; rank < num_devices; ++rank) {
        ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(peer)");
        for (int slot = 0; slot < workspace_slots_; ++slot) {
          auto& runtime = *runtimes_[rank][slot];
          for (int peer = 0; peer < num_devices; ++peer) {
            runtime.x_ptrs[peer] = runtimes_[peer][slot]->x;
            runtime.combine_ptrs[peer] = runtimes_[peer][slot]->combine;
            runtime.d_y_ptrs[peer] = runtimes_[peer][slot]->d_y;
            runtime.d_x_routed_ptrs[peer] = runtimes_[peer][slot]->d_x_routed;
            runtime.router_weight_ptrs[peer] = runtimes_[peer][slot]->router_weights;
            runtime.d_router_weight_ptrs[peer] = runtimes_[peer][slot]->d_router_weights;
            runtime.forward_input_ready_ptrs[peer] = runtimes_[peer][slot]->forward_input_ready;
            runtime.backward_input_ready_ptrs[peer] = runtimes_[peer][slot]->backward_input_ready;
            runtime.forward_completion_ptrs[peer] = runtimes_[peer][slot]->forward_completions;
            runtime.backward_completion_ptrs[peer] = runtimes_[peer][slot]->backward_completions;
            runtime.cancellation_ptrs[peer] = runtimes_[peer][slot]->cancellation;
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
          for (int local_slot = 0; local_slot < workspace_slots_; ++local_slot) {
            runtime.local_slot_forward_completion_ptrs[local_slot] =
                runtimes_[rank][local_slot]->last_forward_completion;
          }
        }
        ThrowOnCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");
      }
    } catch (...) {
      (void)cudaSetDevice(original_device);
      DestroyLocked();
      throw;
    }
    ThrowOnCuda(cudaSetDevice(original_device), "cudaSetDevice(restore)");
    if (++runtime_epoch_ == 0) {
      ++runtime_epoch_;
    }
    initialized_ = true;
  }

  void Shutdown() {
    std::unique_lock<std::mutex> lock(mu_);
    const bool maintenance_finished = cv_.wait_for(lock, kWorkspaceAcquireTimeout, [&] { return !maintenance_; });
    if (!maintenance_finished) {
      throw std::runtime_error("Mixture-of-Kittens runtime maintenance did not finish within five minutes");
    }
    if (!invocations_.empty()) {
      throw std::runtime_error("Mixture-of-Kittens runtime shutdown has active workspace reservations");
    }
    maintenance_ = true;
    DestroyLocked();
    maintenance_ = false;
    cv_.notify_all();
  }

  void ArmTestFailure(
      int rank,
      InvocationPhase phase,
      TestFailurePoint point,
      bool require_two_active_slots) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_) {
      throw std::runtime_error("Mixture-of-Kittens runtime is not initialized");
    }
    if (!invocations_.empty()) {
      throw std::runtime_error("Mixture-of-Kittens test failure injection requires a quiescent runtime");
    }
    if (rank < 0 || rank >= kNumDevices) {
      throw std::runtime_error("Mixture-of-Kittens test failure rank is out of range");
    }
    if (require_two_active_slots && workspace_slots_ != 2) {
      throw std::runtime_error("Mixture-of-Kittens concurrent failure gate requires two workspace slots");
    }
    test_failure_rank_ = rank;
    test_failure_phase_ = phase;
    test_failure_point_ = point;
    test_failure_require_two_active_slots_ = require_two_active_slots;
    test_failure_armed_ = true;
  }

  bool ConsumeTestFailure(int rank, InvocationPhase phase, TestFailurePoint point) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!test_failure_armed_ || test_failure_rank_ != rank || test_failure_phase_ != phase ||
        test_failure_point_ != point) {
      return false;
    }
    if (test_failure_require_two_active_slots_) {
      const bool both_slots_active = cv_.wait_for(lock, kWorkspaceAcquireTimeout, [&] {
        const int fully_leased_invocations = static_cast<int>(std::count_if(
            invocations_.begin(), invocations_.end(), [](const auto& item) {
              return !item.second->cancelled && item.second->leased_mask == kAllRanksMask;
            }));
        return !test_failure_armed_ || fully_leased_invocations >= 2;
      });
      if (!both_slots_active) {
        test_failure_armed_ = false;
        throw std::runtime_error("Mixture-of-Kittens concurrent failure gate did not occupy both workspace slots");
      }
      if (!test_failure_armed_) {
        return false;
      }
    }
    test_failure_armed_ = false;
    return true;
  }

  void ResetDebugCounters() {
    std::unique_lock<std::mutex> lock(mu_);
    const bool maintenance_finished = cv_.wait_for(lock, kWorkspaceAcquireTimeout, [&] { return !maintenance_; });
    if (!maintenance_finished) {
      throw std::runtime_error("Mixture-of-Kittens runtime maintenance did not finish within five minutes");
    }
    if (!initialized_) {
      throw std::runtime_error("Mixture-of-Kittens runtime is not initialized");
    }
    if (!invocations_.empty()) {
      throw std::runtime_error("Mixture-of-Kittens debug counters require a quiescent runtime");
    }
    maintenance_ = true;
    host_slot_reuse_failures_.fill(0);
    for (auto& acquisitions : host_slot_acquisitions_) {
      acquisitions.fill(0);
    }
    host_max_active_slots_.fill(0);
    for (auto& calls : host_staging_copy_calls_) {
      calls.fill(0);
    }
    for (auto& bytes : host_staging_copy_bytes_) {
      bytes.fill(0);
    }
    lock.unlock();
    try {
      int original_device = 0;
      ThrowOnCuda(cudaGetDevice(&original_device), "cudaGetDevice(reset debug counters)");
      for (const auto& rank_runtimes : runtimes_) {
        for (const auto& runtime : rank_runtimes) {
          if (runtime == nullptr) {
            continue;
          }
          ThrowOnCuda(cudaSetDevice(runtime->device), "cudaSetDevice(reset debug counters)");
          ThrowOnCuda(cudaMemset(runtime->debug_counters, 0, sizeof(DebugCounters)), "cudaMemset(debug counters)");
        }
      }
      ThrowOnCuda(cudaSetDevice(original_device), "cudaSetDevice(restore debug counters)");
    } catch (...) {
      EndMaintenance();
      throw;
    }
    EndMaintenance();
  }

  void ReadDebugCounters(uint64_t* output, int64_t count) {
    std::unique_lock<std::mutex> lock(mu_);
    const bool maintenance_finished = cv_.wait_for(lock, kWorkspaceAcquireTimeout, [&] { return !maintenance_; });
    if (!maintenance_finished) {
      throw std::runtime_error("Mixture-of-Kittens runtime maintenance did not finish within five minutes");
    }
    if (!initialized_) {
      throw std::runtime_error("Mixture-of-Kittens runtime is not initialized");
    }
    const int64_t expected = static_cast<int64_t>(kNumDevices) * kDebugCounterCount;
    if (output == nullptr || count != expected) {
      throw std::runtime_error("Mixture-of-Kittens debug counter output has the wrong size");
    }
    if (!invocations_.empty()) {
      throw std::runtime_error("Mixture-of-Kittens debug counters require a quiescent runtime");
    }
    maintenance_ = true;
    const auto host_slot_reuse_failures = host_slot_reuse_failures_;
    const auto host_slot_acquisitions = host_slot_acquisitions_;
    const auto host_max_active_slots = host_max_active_slots_;
    const auto host_staging_copy_calls = host_staging_copy_calls_;
    const auto host_staging_copy_bytes = host_staging_copy_bytes_;
    lock.unlock();
    try {
      int original_device = 0;
      ThrowOnCuda(cudaGetDevice(&original_device), "cudaGetDevice(read debug counters)");
      for (int rank = 0; rank < kNumDevices; ++rank) {
        const auto& rank_runtimes = runtimes_[rank];
        ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(read debug counters)");
        ThrowOnCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(read debug counters)");
        auto* rank_output = output + static_cast<int64_t>(rank) * kDebugCounterCount;
        std::fill(rank_output, rank_output + kDebugCounterCount, 0ULL);
        for (const auto& runtime : rank_runtimes) {
          if (runtime == nullptr) {
            continue;
          }
          DebugCounters slot_counters{};
          ThrowOnCuda(
              cudaMemcpy(
                  &slot_counters,
                  runtime->debug_counters,
                  sizeof(DebugCounters),
                  cudaMemcpyDeviceToHost),
              "cudaMemcpy(debug counters)");
          for (int counter = 0; counter < kDebugCounterCount; ++counter) {
            rank_output[counter] += slot_counters.values[counter];
          }
        }
        rank_output[kSlotReuseFailures] += host_slot_reuse_failures[rank];
        rank_output[kSlotZeroAcquisitions] += host_slot_acquisitions[rank][0];
        rank_output[kSlotOneAcquisitions] += host_slot_acquisitions[rank][1];
        rank_output[kMaxActiveSlots] += host_max_active_slots[rank];
        rank_output[kForwardStagingCopyCalls] += host_staging_copy_calls[rank][0];
        rank_output[kForwardStagingCopyBytes] += host_staging_copy_bytes[rank][0];
        rank_output[kBackwardStagingCopyCalls] += host_staging_copy_calls[rank][1];
        rank_output[kBackwardStagingCopyBytes] += host_staging_copy_bytes[rank][1];
      }
      ThrowOnCuda(cudaSetDevice(original_device), "cudaSetDevice(restore debug counters)");
    } catch (...) {
      EndMaintenance();
      throw;
    }
    EndMaintenance();
  }

  void TrimDefaultMemoryPools(uint64_t* output, int64_t count) {
    std::unique_lock<std::mutex> lock(mu_);
    const bool maintenance_finished = cv_.wait_for(lock, kWorkspaceAcquireTimeout, [&] { return !maintenance_; });
    if (!maintenance_finished) {
      throw std::runtime_error("Mixture-of-Kittens runtime maintenance did not finish within five minutes");
    }
    if (!initialized_) {
      throw std::runtime_error("Mixture-of-Kittens runtime is not initialized");
    }
    if (output == nullptr || count != kMemoryPoolTrimOutputCount) {
      throw std::runtime_error("Mixture-of-Kittens memory-pool trim output has the wrong size");
    }

    const auto start = std::chrono::steady_clock::now();
    // Maintenance prevents a new FFI invocation from acquiring a workspace while the
    // process-local device workers drain and trim independent CUDA pools.
    maintenance_ = true;
    lock.unlock();
    try {
      int original_device = 0;
      ThrowOnCuda(cudaGetDevice(&original_device), "cudaGetDevice(trim default memory pools)");

      std::array<std::exception_ptr, kNumDevices> synchronize_errors{};
      std::array<std::jthread, kNumDevices> synchronize_workers;
      for (int rank = 0; rank < kNumDevices; ++rank) {
        synchronize_workers[rank] = std::jthread([rank, &synchronize_errors] {
          try {
            ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(synchronize before default memory-pool trim)");
            ThrowOnCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(default memory-pool trim)");
          } catch (...) {
            synchronize_errors[rank] = std::current_exception();
          }
        });
      }
      for (auto& worker : synchronize_workers) {
        worker.join();
      }
      for (int rank = 0; rank < kNumDevices; ++rank) {
        if (synchronize_errors[rank] != nullptr) {
          ThrowOnCuda(
              cudaSetDevice(original_device),
              "cudaSetDevice(restore after failed default memory-pool synchronization)");
          std::rethrow_exception(synchronize_errors[rank]);
        }
      }

      lock.lock();
      const uint64_t active_reservations = invocations_.size();
      const uint64_t active_workspace_slots = static_cast<uint64_t>(
          std::count(slot_active_.begin(), slot_active_.begin() + workspace_slots_, true));
      if (active_reservations != 0 || active_workspace_slots != 0) {
        lock.unlock();
        ThrowOnCuda(
            cudaSetDevice(original_device),
            "cudaSetDevice(restore after non-quiescent default memory-pool trim)");
        throw std::runtime_error(
            "Mixture-of-Kittens default memory-pool trim requires zero active workspace reservations");
      }
      lock.unlock();

      std::array<std::exception_ptr, kNumDevices> trim_errors{};
      std::array<std::jthread, kNumDevices> trim_workers;
      for (int rank = 0; rank < kNumDevices; ++rank) {
        trim_workers[rank] = std::jthread([rank, output, &trim_errors] {
          try {
            ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(default memory-pool trim)");
            cudaMemPool_t pool = nullptr;
            ThrowOnCuda(cudaDeviceGetDefaultMemPool(&pool, rank), "cudaDeviceGetDefaultMemPool");
            auto* rank_output = output + rank * kMemoryPoolStatsPerRank;
            ThrowOnCuda(
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, rank_output + 0),
                "cudaMemPoolGetAttribute(reserved before trim)");
            ThrowOnCuda(
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, rank_output + 1),
                "cudaMemPoolGetAttribute(used before trim)");
            size_t device_free_bytes_before = 0;
            size_t device_total_bytes_before = 0;
            ThrowOnCuda(
                cudaMemGetInfo(&device_free_bytes_before, &device_total_bytes_before),
                "cudaMemGetInfo(before default memory-pool trim)");
            ThrowOnCuda(cudaMemPoolTrimTo(pool, 0), "cudaMemPoolTrimTo");
            ThrowOnCuda(
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, rank_output + 2),
                "cudaMemPoolGetAttribute(reserved after trim)");
            ThrowOnCuda(
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, rank_output + 3),
                "cudaMemPoolGetAttribute(used after trim)");
            size_t device_free_bytes_after = 0;
            size_t device_total_bytes_after = 0;
            ThrowOnCuda(
                cudaMemGetInfo(&device_free_bytes_after, &device_total_bytes_after),
                "cudaMemGetInfo(after default memory-pool trim)");
            rank_output[4] = static_cast<uint64_t>(device_free_bytes_before);
            rank_output[5] = static_cast<uint64_t>(device_total_bytes_before);
            rank_output[6] = static_cast<uint64_t>(device_free_bytes_after);
            rank_output[7] = static_cast<uint64_t>(device_total_bytes_after);
            ThrowOnCuda(
                cudaDeviceGetGraphMemAttribute(rank, cudaGraphMemAttrReservedMemCurrent, rank_output + 8),
                "cudaDeviceGetGraphMemAttribute(reserved after trim)");
            ThrowOnCuda(
                cudaDeviceGetGraphMemAttribute(rank, cudaGraphMemAttrUsedMemCurrent, rank_output + 9),
                "cudaDeviceGetGraphMemAttribute(used after trim)");
          } catch (...) {
            trim_errors[rank] = std::current_exception();
          }
        });
      }
      for (auto& worker : trim_workers) {
        worker.join();
      }
      for (int rank = 0; rank < kNumDevices; ++rank) {
        if (trim_errors[rank] != nullptr) {
          ThrowOnCuda(cudaSetDevice(original_device), "cudaSetDevice(restore after failed default memory-pool trim)");
          std::rethrow_exception(trim_errors[rank]);
        }
      }
      ThrowOnCuda(cudaSetDevice(original_device), "cudaSetDevice(restore after default memory-pool trim)");
      output[kNumDevices * kMemoryPoolStatsPerRank] = active_reservations;
      output[kNumDevices * kMemoryPoolStatsPerRank + 1] = active_workspace_slots;
    } catch (...) {
      EndMaintenance();
      throw;
    }
    EndMaintenance();
    output[kNumDevices * kMemoryPoolStatsPerRank + 2] = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
  }

  RuntimeLease Acquire(
      ffi::RunId run_id,
      int64_t collective_id,
      InvocationPhase phase,
      std::optional<ForwardXRegistration> forward_x = std::nullopt,
      std::optional<BackwardPeerRegistration> backward_peer = std::nullopt) {
    int rank = -1;
    ThrowOnCuda(cudaGetDevice(&rank), "cudaGetDevice(acquire workspace)");
    if (rank < 0 || rank >= kNumDevices) {
      throw std::runtime_error("No Mixture-of-Kittens runtime exists for the current GPU");
    }

    std::unique_lock<std::mutex> lock(mu_);
    const bool maintenance_finished = cv_.wait_for(lock, kWorkspaceAcquireTimeout, [&] { return !maintenance_; });
    if (!maintenance_finished) {
      throw std::runtime_error("Mixture-of-Kittens runtime maintenance did not finish within five minutes");
    }
    if (!initialized_) {
      throw std::runtime_error("Mixture-of-Kittens runtime is not initialized");
    }
    if (!failure_message_.empty()) {
      throw std::runtime_error(failure_message_);
    }
    RunPhaseKey run_phase{.run_id = run_id.run_id, .collective_id = collective_id, .phase = phase};
    auto& ordinals = run_ordinals_[run_phase];
    const uint64_t ordinal = ordinals[rank]++;
    InvocationKey key{
        .run_id = run_id.run_id,
        .collective_id = collective_id,
        .ordinal = ordinal,
        .phase = phase,
    };
    auto [invocation, inserted] = invocations_.try_emplace(key, std::make_shared<InvocationState>());
    const std::shared_ptr<InvocationState> state = invocation->second;
    if (inserted) {
      const auto active_slots_end = slot_active_.begin() + workspace_slots_;
      const bool slot_available = cv_.wait_for(lock, kWorkspaceAcquireTimeout, [&] {
        return state->cancelled || !failure_message_.empty() ||
               std::find(slot_active_.begin(), active_slots_end, false) != active_slots_end;
      });
      if (!slot_available) {
        ++host_slot_reuse_failures_[rank];
        CancelReservationLocked(key, state, "Mixture-of-Kittens workspace slots remained occupied for five minutes");
      }
      if (state->cancelled || !failure_message_.empty()) {
        throw std::runtime_error(state->error.empty() ? failure_message_ : state->error);
      }
      state->slot = static_cast<int>(
          std::find(slot_active_.begin(), active_slots_end, false) - slot_active_.begin());
      slot_active_[state->slot] = true;
      state->generation = (++next_generation_ << 1) | static_cast<uint64_t>(state->slot);
      cv_.notify_all();
    }
    const uint8_t rank_bit = static_cast<uint8_t>(1U << rank);
    if ((state->arrival_mask & rank_bit) != 0) {
      ++host_slot_reuse_failures_[rank];
      CancelReservationLocked(key, state, "Mixture-of-Kittens workspace reservation received a duplicate rank");
      throw std::runtime_error(state->error);
    }
    if (phase == InvocationPhase::kForward) {
      if (!forward_x.has_value() || forward_x->pointer == nullptr || forward_x->size_bytes == 0) {
        state->error = "Mixture-of-Kittens forward received an invalid XLA x buffer registration";
      } else {
        if (state->forward_x_mask == 0) {
          state->forward_x_size_bytes = forward_x->size_bytes;
          state->forward_x_storage = forward_x->storage;
        } else if (state->forward_x_size_bytes != forward_x->size_bytes ||
                   state->forward_x_storage != forward_x->storage) {
          state->error = "Mixture-of-Kittens forward ranks disagreed on x buffer size or storage mode";
        }
        state->forward_x_ptrs[rank] = forward_x->pointer;
        state->forward_x_mask |= rank_bit;
      }
    } else {
      if (forward_x.has_value()) {
        state->error = "Mixture-of-Kittens backward unexpectedly received a forward x registration";
      }
      if (!backward_peer.has_value() || backward_peer->d_y_pointer == nullptr ||
          backward_peer->x_pointer == nullptr || backward_peer->router_weight_pointer == nullptr ||
          backward_peer->d_router_weight_pointer == nullptr || backward_peer->activation_size_bytes == 0 ||
          backward_peer->router_size_bytes == 0) {
        state->error = "Mixture-of-Kittens backward received an invalid XLA peer-buffer registration";
      } else {
        if (state->backward_peer_mask == 0) {
          state->backward_activation_size_bytes = backward_peer->activation_size_bytes;
          state->backward_router_size_bytes = backward_peer->router_size_bytes;
          state->backward_peer_storage = backward_peer->storage;
        } else if (state->backward_activation_size_bytes != backward_peer->activation_size_bytes ||
                   state->backward_router_size_bytes != backward_peer->router_size_bytes ||
                   state->backward_peer_storage != backward_peer->storage) {
          state->error = "Mixture-of-Kittens backward ranks disagreed on peer-buffer sizes or storage mode";
        }
        state->backward_d_y_ptrs[rank] = backward_peer->d_y_pointer;
        state->backward_x_ptrs[rank] = backward_peer->x_pointer;
        state->backward_router_weight_ptrs[rank] = backward_peer->router_weight_pointer;
        state->backward_d_router_weight_ptrs[rank] = backward_peer->d_router_weight_pointer;
        state->backward_peer_mask |= rank_bit;
      }
    }
    state->arrival_mask |= rank_bit;
    if (state->arrival_mask == kAllRanksMask &&
        (phase == InvocationPhase::kForward && state->forward_x_mask != kAllRanksMask)) {
      state->error = "Mixture-of-Kittens forward did not register one XLA x buffer per rank";
    }
    if (state->arrival_mask == kAllRanksMask &&
        (phase == InvocationPhase::kBackward && state->backward_peer_mask != kAllRanksMask)) {
      state->error = "Mixture-of-Kittens backward did not register all four XLA peer buffers per rank";
    }
    if (state->arrival_mask == kAllRanksMask && !state->error.empty()) {
      state->cancelled = true;
      if (failure_message_.empty()) {
        failure_message_ = state->error;
      }
    }
    cv_.notify_all();
    const bool all_ranks_arrived = cv_.wait_for(lock, kWorkspaceAcquireTimeout, [&] {
      return state->cancelled || (state->slot >= 0 && state->arrival_mask == kAllRanksMask);
    });
    if (!all_ranks_arrived) {
      ++host_slot_reuse_failures_[rank];
      CancelReservationLocked(
          key,
          state,
          "Mixture-of-Kittens workspace reservation did not rendezvous all four ranks");
    }
    if (state->cancelled) {
      if (state->arrival_mask == kAllRanksMask && state->slot >= 0) {
        FinishRankLocked(key, state, rank);
      }
      throw std::runtime_error(state->error);
    }
    state->leased_mask |= rank_bit;
    cv_.notify_all();
    ++host_slot_acquisitions_[rank][state->slot];
    host_max_active_slots_[rank] = std::max(
        host_max_active_slots_[rank],
        static_cast<uint64_t>(
            std::count(slot_active_.begin(), slot_active_.begin() + workspace_slots_, true)));
    if (phase == InvocationPhase::kForward && state->forward_x_storage == ForwardXStorage::kRuntimeStaged) {
      host_staging_copy_calls_[rank][0] += 1;
      host_staging_copy_bytes_[rank][0] += state->forward_x_size_bytes;
    } else if (phase == InvocationPhase::kBackward) {
      if (state->backward_peer_storage == BackwardPeerStorage::kRuntimeStaged) {
        host_staging_copy_calls_[rank][1] += 4;
        host_staging_copy_bytes_[rank][1] +=
            2 * state->backward_activation_size_bytes + 2 * state->backward_router_size_bytes;
      } else if (state->backward_peer_storage == BackwardPeerStorage::kXlaPeerInputsExperimental) {
        host_staging_copy_calls_[rank][1] += 1;
        host_staging_copy_bytes_[rank][1] += state->backward_router_size_bytes;
      }
    }
    return RuntimeLease{
        .key = key,
        .rank = rank,
        .slot = state->slot,
        .generation = state->generation,
        .forward_x_ptrs = state->forward_x_ptrs,
        .forward_x_size_bytes = state->forward_x_size_bytes,
        .forward_x_storage = state->forward_x_storage,
        .backward_d_y_ptrs = state->backward_d_y_ptrs,
        .backward_x_ptrs = state->backward_x_ptrs,
        .backward_router_weight_ptrs = state->backward_router_weight_ptrs,
        .backward_d_router_weight_ptrs = state->backward_d_router_weight_ptrs,
        .backward_activation_size_bytes = state->backward_activation_size_bytes,
        .backward_router_size_bytes = state->backward_router_size_bytes,
        .backward_peer_storage = state->backward_peer_storage,
    };
  }

  void ReleaseAfterStream(const RuntimeLease& lease, cudaStream_t stream) {
    auto* callback = new CompletionCallback{.manager = this, .key = lease.key, .rank = lease.rank};
    cudaError_t status = cudaLaunchHostFunc(stream, &RuntimeManager::CompleteCallback, callback);
    if (status != cudaSuccess) {
      delete callback;
      ThrowOnCuda(status, "cudaLaunchHostFunc(release workspace)");
    }
  }

  void MarkFailure(const RuntimeLease& lease, const std::string& rank_error) {
    std::lock_guard<std::mutex> lock(mu_);
    auto invocation = invocations_.find(lease.key);
    if (invocation == invocations_.end()) {
      ++host_slot_reuse_failures_[lease.rank];
      return;
    }
    const std::shared_ptr<InvocationState> state = invocation->second;
    if (!state->cancelled) {
      state->cancelled = true;
      state->error = rank_error;
    }
    cv_.notify_all();
  }

  uint32_t RuntimeEpoch() {
    std::lock_guard<std::mutex> lock(mu_);
    return runtime_epoch_;
  }

  DeviceRuntime& Current(int slot) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_) {
      throw std::runtime_error("Mixture-of-Kittens runtime is not initialized");
    }
    int device = -1;
    ThrowOnCuda(cudaGetDevice(&device), "cudaGetDevice(current)");
    if (device < 0 || device >= static_cast<int>(runtimes_.size()) || slot < 0 || slot >= workspace_slots_ ||
        runtimes_[device][slot] == nullptr) {
      throw std::runtime_error("No Mixture-of-Kittens runtime exists for the current GPU");
    }
    return *runtimes_[device][slot];
  }

 private:
  struct CompletionCallback {
    RuntimeManager* manager;
    InvocationKey key;
    int rank;
  };

  static void CompleteCallback(void* opaque) {
    std::unique_ptr<CompletionCallback> callback(static_cast<CompletionCallback*>(opaque));
    callback->manager->Complete(callback->key, callback->rank);
  }

  static RunPhaseKey RunPhase(const InvocationKey& key) {
    return RunPhaseKey{
        .run_id = key.run_id,
        .collective_id = key.collective_id,
        .phase = key.phase,
    };
  }

  void EndMaintenance() {
    std::lock_guard<std::mutex> lock(mu_);
    maintenance_ = false;
    cv_.notify_all();
  }

  void ReleaseSlotLocked(const std::shared_ptr<InvocationState>& state) {
    if (state->slot < 0 || state->slot_released) {
      return;
    }
    slot_active_[state->slot] = false;
    state->slot_released = true;
  }

  void MaybeReclaimRunPhaseLocked(const RunPhaseKey& run_phase) {
    const bool still_active = std::any_of(invocations_.begin(), invocations_.end(), [&](const auto& item) {
      return RunPhase(item.first) == run_phase;
    });
    if (!still_active) {
      run_ordinals_.erase(run_phase);
    }
  }

  void CancelReservationLocked(
      const InvocationKey& key,
      const std::shared_ptr<InvocationState>& state,
      const std::string& error) {
    if (!state->cancelled) {
      state->cancelled = true;
      state->error = error;
      if (failure_message_.empty()) {
        failure_message_ = error;
      }
      ReleaseSlotLocked(state);
      auto invocation = invocations_.find(key);
      if (invocation != invocations_.end() && invocation->second == state) {
        invocations_.erase(invocation);
      }
      MaybeReclaimRunPhaseLocked(RunPhase(key));
    }
    cv_.notify_all();
  }

  void FinishRankLocked(const InvocationKey& key, const std::shared_ptr<InvocationState>& state, int rank) {
    const uint8_t rank_bit = static_cast<uint8_t>(1U << rank);
    if ((state->completion_mask & rank_bit) != 0) {
      ++host_slot_reuse_failures_[rank];
      return;
    }
    state->completion_mask |= rank_bit;
    if (state->completion_mask != kAllRanksMask) {
      return;
    }
    ReleaseSlotLocked(state);
    auto invocation = invocations_.find(key);
    if (invocation != invocations_.end() && invocation->second == state) {
      invocations_.erase(invocation);
    }
    MaybeReclaimRunPhaseLocked(RunPhase(key));
    cv_.notify_all();
  }

  void Complete(const InvocationKey& key, int rank) {
    std::lock_guard<std::mutex> lock(mu_);
    auto invocation = invocations_.find(key);
    if (invocation == invocations_.end()) {
      ++host_slot_reuse_failures_[rank];
      return;
    }
    const std::shared_ptr<InvocationState> state = invocation->second;
    const uint8_t rank_bit = static_cast<uint8_t>(1U << rank);
    if ((state->leased_mask & rank_bit) == 0 || (state->completion_mask & rank_bit) != 0) {
      ++host_slot_reuse_failures_[rank];
      if (!state->cancelled) {
        state->cancelled = true;
        state->error = "Mixture-of-Kittens workspace invocation received an invalid rank completion";
        if (failure_message_.empty()) {
          failure_message_ = state->error;
        }
      }
      cv_.notify_all();
      return;
    }
    FinishRankLocked(key, state, rank);
  }

  void DestroyLocked() {
    runtimes_.clear();
    initialized_ = false;
    num_devices_ = 0;
    num_tokens_ = 0;
    hidden_dim_ = 0;
    top_k_ = 0;
    workspace_slots_ = 0;
    slot_active_.fill(false);
    next_generation_ = 0;
    host_slot_reuse_failures_.fill(0);
    for (auto& acquisitions : host_slot_acquisitions_) {
      acquisitions.fill(0);
    }
    host_max_active_slots_.fill(0);
    for (auto& calls : host_staging_copy_calls_) {
      calls.fill(0);
    }
    for (auto& bytes : host_staging_copy_bytes_) {
      bytes.fill(0);
    }
    invocations_.clear();
    run_ordinals_.clear();
    failure_message_.clear();
    test_failure_armed_ = false;
    test_failure_require_two_active_slots_ = false;
  }

  std::mutex mu_;
  std::condition_variable cv_;
  bool initialized_ = false;
  bool maintenance_ = false;
  int num_devices_ = 0;
  int num_tokens_ = 0;
  int hidden_dim_ = 0;
  int top_k_ = 0;
  int workspace_slots_ = 0;
  std::vector<std::array<std::unique_ptr<DeviceRuntime>, kMaxWorkspaceSlots>> runtimes_;
  std::array<bool, kMaxWorkspaceSlots> slot_active_{};
  uint64_t next_generation_ = 0;
  uint32_t runtime_epoch_ = 0;
  std::array<uint64_t, kNumDevices> host_slot_reuse_failures_{};
  std::array<std::array<uint64_t, kMaxWorkspaceSlots>, kNumDevices> host_slot_acquisitions_{};
  std::array<uint64_t, kNumDevices> host_max_active_slots_{};
  std::array<std::array<uint64_t, 2>, kNumDevices> host_staging_copy_calls_{};
  std::array<std::array<uint64_t, 2>, kNumDevices> host_staging_copy_bytes_{};
  std::string failure_message_;
  bool test_failure_armed_ = false;
  int test_failure_rank_ = -1;
  InvocationPhase test_failure_phase_ = InvocationPhase::kForward;
  TestFailurePoint test_failure_point_ = TestFailurePoint::kBeforeInputReady;
  bool test_failure_require_two_active_slots_ = false;
  std::unordered_map<InvocationKey, std::shared_ptr<InvocationState>, InvocationKeyHash> invocations_;
  std::unordered_map<RunPhaseKey, std::array<uint64_t, kNumDevices>, RunPhaseKeyHash> run_ordinals_;
};

struct GenerationArgs {
  std::array<uint64_t*, kNumDevices> input_ready_ptrs;
  std::array<uint64_t*, kNumDevices> completion_ptrs;
  uint64_t* local_completions;
  uint64_t* generation;
  uint64_t* cancellation;
  DebugCounters* debug_counters;
  uint64_t target;
  int rank;
  PeerWaitPhase wait_phase;
};

struct CancellationArgs {
  std::array<uint64_t*, kNumDevices> cancellation_ptrs;
  std::array<uint64_t*, kNumDevices> input_ready_ptrs;
  std::array<uint64_t*, kNumDevices> completion_ptrs;
  uint64_t target;
  int rank;
};

__global__ void PublishCancellationKernel(CancellationArgs args) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  __threadfence_system();
  for (int peer = 0; peer < kNumDevices; ++peer) {
    auto* cancellation = reinterpret_cast<unsigned long long*>(args.cancellation_ptrs[peer]);
    auto* ready = reinterpret_cast<unsigned long long*>(args.input_ready_ptrs[peer] + args.rank);
    auto* completion = reinterpret_cast<unsigned long long*>(args.completion_ptrs[peer] + args.rank);
    atomicExch_system(cancellation, static_cast<unsigned long long>(args.target));
    // A failed rank's buffers remain allocated for the operation lifetime. Mark
    // its signals ready so healthy peers can finish disposable work and reach
    // the JAX status fence instead of spinning in a native wait.
    atomicExch_system(ready, static_cast<unsigned long long>(args.target));
    atomicExch_system(completion, static_cast<unsigned long long>(args.target));
  }
}

void PublishFailureSignals(DeviceRuntime& runtime, InvocationPhase phase, uint64_t generation) {
  // This closes synchronous handler failures only while a separate CUDA stream
  // remains usable. An asynchronous device trap can poison the CUDA context;
  // detecting that case would require synchronizing the possibly blocked FFI
  // stream and therefore cannot provide a bounded in-process failure protocol.
  CancellationArgs args{
      .cancellation_ptrs = runtime.cancellation_ptrs,
      .input_ready_ptrs = phase == InvocationPhase::kForward
                              ? runtime.forward_input_ready_ptrs
                              : runtime.backward_input_ready_ptrs,
      .completion_ptrs = phase == InvocationPhase::kForward
                             ? runtime.forward_completion_ptrs
                             : runtime.backward_completion_ptrs,
      .target = generation,
      .rank = runtime.rank,
  };
  PublishCancellationKernel<<<1, 1, 0, runtime.cancellation_stream>>>(args);
  ThrowOnCuda(cudaGetLastError(), "PublishCancellationKernel");
  ThrowOnCuda(cudaStreamSynchronize(runtime.cancellation_stream), "cudaStreamSynchronize(failure signals)");
}

[[noreturn]] void TerminalCudaFailure(const std::string& error) {
  std::fprintf(stderr,
               "Mixture-of-Kittens terminal CUDA failure: the CUDA context may be poisoned and "
               "cannot be closed in process: %s\n",
               error.c_str());
  std::fflush(stderr);
  std::abort();
}

__global__ void PublishForwardInputReadyKernel(GenerationArgs args) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  auto* generation = reinterpret_cast<unsigned long long*>(args.generation);
  const unsigned long long target = args.target;
  atomicExch_system(generation, target);
  __threadfence_system();
  for (int peer = 0; peer < kNumDevices; ++peer) {
    auto* ready = reinterpret_cast<unsigned long long*>(args.input_ready_ptrs[peer] + args.rank);
    atomicExch_system(ready, target);
  }
}

__global__ void PublishBackwardInputReadyKernel(GenerationArgs args) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  auto* generation = reinterpret_cast<unsigned long long*>(args.generation);
  const unsigned long long target = args.target;
  atomicExch_system(generation, target);
  __threadfence_system();
  for (int peer = 0; peer < kNumDevices; ++peer) {
    auto* ready = reinterpret_cast<unsigned long long*>(args.input_ready_ptrs[peer] + args.rank);
    atomicExch_system(ready, target);
  }
}

__global__ void PublishCompletionKernel(GenerationArgs args) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  const unsigned long long target = args.target;
  __threadfence_system();
  for (int peer = 0; peer < kNumDevices; ++peer) {
    auto* completion = reinterpret_cast<unsigned long long*>(args.completion_ptrs[peer] + args.rank);
    atomicExch_system(completion, target);
  }
}

__global__ void WaitCompletionKernel(GenerationArgs args) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  const unsigned long long target = args.target;
  auto* cancellation = reinterpret_cast<unsigned long long*>(args.cancellation);
  if (atomicAdd_system(cancellation, 0ULL) >= target) {
    __threadfence_system();
    return;
  }
  unsigned long long wait_events = 0;
  unsigned long long future_generations = 0;
  bool cancelled = false;
  for (int peer = 0; peer < kNumDevices; ++peer) {
    auto* completion = reinterpret_cast<unsigned long long*>(args.local_completions + peer);
    unsigned long long observed = atomicAdd_system(completion, 0ULL);
    if (observed < target) {
      ++wait_events;
      const unsigned long long wait_start = clock64();
      while (observed < target) {
        if (atomicAdd_system(cancellation, 0ULL) >= target) {
          cancelled = true;
          break;
        }
        __nanosleep(64);
        observed = atomicAdd_system(completion, 0ULL);
      }
      const unsigned long long wait_cycles = clock64() - wait_start;
      const int cell = static_cast<int>(args.wait_phase) * kNumDevices + peer;
      atomicAdd(
          reinterpret_cast<unsigned long long*>(args.debug_counters->values + kPeerWaitEvents + cell),
          1ULL);
      atomicAdd(
          reinterpret_cast<unsigned long long*>(args.debug_counters->values + kPeerWaitCycles + cell),
          wait_cycles);
      atomicMax(
          reinterpret_cast<unsigned long long*>(args.debug_counters->values + kPeerWaitMaxCycles + cell),
          wait_cycles);
    }
    if (observed > target) {
      ++future_generations;
    }
    if (cancelled) {
      break;
    }
  }
  atomicAdd(
      reinterpret_cast<unsigned long long*>(args.debug_counters->values + kCompletionWaits),
      wait_events);
  if (future_generations != 0) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(args.debug_counters->values + kGenerationMismatches),
        future_generations);
  }
  __threadfence_system();
}

__global__ void WriteFailureStatusKernel(
    const uint64_t* cancellation,
    uint64_t generation,
    int32_t* failure_status) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  auto* value = reinterpret_cast<unsigned long long*>(const_cast<uint64_t*>(cancellation));
  failure_status[0] = atomicAdd_system(value, 0ULL) >= generation ? 1 : 0;
}

void LaunchFailureStatus(
    DeviceRuntime& runtime,
    cudaStream_t stream,
    uint64_t generation,
    int32_t* failure_status) {
  WriteFailureStatusKernel<<<1, 1, 0, stream>>>(runtime.cancellation, generation, failure_status);
  ThrowOnCuda(cudaGetLastError(), "WriteFailureStatusKernel");
}

void CloseSynchronousFailure(
    const RuntimeLease& lease,
    DeviceRuntime& runtime,
    cudaStream_t stream,
    int32_t* failure_status,
    const std::string& error) {
  PublishFailureSignals(runtime, lease.key.phase, lease.generation);
  RuntimeManager::Instance().MarkFailure(lease, error);
  LaunchFailureStatus(runtime, stream, lease.generation, failure_status);
  RuntimeManager::Instance().ReleaseAfterStream(lease, stream);
}

void LaunchForwardInputReady(DeviceRuntime& runtime, cudaStream_t stream, uint64_t generation) {
  GenerationArgs args{
      .input_ready_ptrs = runtime.forward_input_ready_ptrs,
      .completion_ptrs = runtime.forward_completion_ptrs,
      .local_completions = runtime.forward_completions,
      .generation = runtime.generation,
      .cancellation = runtime.cancellation,
      .debug_counters = runtime.debug_counters,
      .target = generation,
      .rank = runtime.rank,
      .wait_phase = PeerWaitPhase::kForwardPre,
  };
  PublishForwardInputReadyKernel<<<1, 1, 0, stream>>>(args);
  ThrowOnCuda(cudaGetLastError(), "PublishForwardInputReadyKernel");
}

void LaunchBackwardInputReady(DeviceRuntime& runtime, cudaStream_t stream, uint64_t generation) {
  GenerationArgs args{
      .input_ready_ptrs = runtime.backward_input_ready_ptrs,
      .completion_ptrs = runtime.backward_completion_ptrs,
      .local_completions = runtime.backward_completions,
      .generation = runtime.generation,
      .cancellation = runtime.cancellation,
      .debug_counters = runtime.debug_counters,
      .target = generation,
      .rank = runtime.rank,
      .wait_phase = PeerWaitPhase::kBackwardPre,
  };
  PublishBackwardInputReadyKernel<<<1, 1, 0, stream>>>(args);
  ThrowOnCuda(cudaGetLastError(), "PublishBackwardInputReadyKernel");
}

void LaunchCompletion(
    DeviceRuntime& runtime,
    cudaStream_t stream,
    const std::array<uint64_t*, kNumDevices>& completion_ptrs,
    uint64_t* local_completions,
    uint64_t generation,
    PeerWaitPhase wait_phase) {
  GenerationArgs args{
      .input_ready_ptrs = runtime.forward_input_ready_ptrs,
      .completion_ptrs = completion_ptrs,
      .local_completions = local_completions,
      .generation = runtime.generation,
      .cancellation = runtime.cancellation,
      .debug_counters = runtime.debug_counters,
      .target = generation,
      .rank = runtime.rank,
      .wait_phase = wait_phase,
  };
  PublishCompletionKernel<<<1, 1, 0, stream>>>(args);
  ThrowOnCuda(cudaGetLastError(), "PublishCompletionKernel");
  WaitCompletionKernel<<<1, 1, 0, stream>>>(args);
  ThrowOnCuda(cudaGetLastError(), "WaitCompletionKernel");
}

struct ForwardStampArgs {
  uint64_t* last_forward_completion;
  int32_t* slot_output;
  int32_t* generation_high_output;
  int32_t* generation_low_output;
  int32_t* runtime_epoch_output;
  uint64_t generation;
  uint32_t runtime_epoch;
  int slot;
};

__global__ void PublishForwardStampKernel(ForwardStampArgs args) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  atomicExch_system(
      reinterpret_cast<unsigned long long*>(args.last_forward_completion),
      static_cast<unsigned long long>(args.generation));
  args.slot_output[0] = args.slot;
  args.generation_high_output[0] = static_cast<int32_t>(args.generation >> 32);
  args.generation_low_output[0] = static_cast<int32_t>(args.generation & 0xffffffffULL);
  args.runtime_epoch_output[0] = static_cast<int32_t>(args.runtime_epoch);
}

void LaunchForwardStamp(
    DeviceRuntime& runtime,
    cudaStream_t stream,
    const RuntimeLease& lease,
    uint32_t runtime_epoch,
    int32_t* slot_output,
    int32_t* generation_high_output,
    int32_t* generation_low_output,
    int32_t* runtime_epoch_output) {
  ForwardStampArgs args{
      .last_forward_completion = runtime.last_forward_completion,
      .slot_output = slot_output,
      .generation_high_output = generation_high_output,
      .generation_low_output = generation_low_output,
      .runtime_epoch_output = runtime_epoch_output,
      .generation = lease.generation,
      .runtime_epoch = runtime_epoch,
      .slot = lease.slot,
  };
  PublishForwardStampKernel<<<1, 1, 0, stream>>>(args);
  ThrowOnCuda(cudaGetLastError(), "PublishForwardStampKernel");
}

struct ValidateForwardStampArgs {
  std::array<uint64_t*, kMaxWorkspaceSlots> last_forward_completions;
  const int32_t* slot;
  const int32_t* generation_high;
  const int32_t* generation_low;
  const int32_t* runtime_epoch;
  DebugCounters* debug_counters;
  uint32_t expected_runtime_epoch;
  int workspace_slots;
};

__global__ void ValidateForwardStampKernel(ValidateForwardStampArgs args) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  const int slot = args.slot[0];
  const uint64_t generation =
      (static_cast<uint64_t>(static_cast<uint32_t>(args.generation_high[0])) << 32) |
      static_cast<uint32_t>(args.generation_low[0]);
  const uint32_t runtime_epoch = static_cast<uint32_t>(args.runtime_epoch[0]);
  bool mismatch = slot < 0 || slot >= args.workspace_slots || runtime_epoch != args.expected_runtime_epoch;
  if (!mismatch) {
    mismatch = static_cast<int>(generation & static_cast<uint64_t>(kMaxWorkspaceSlots - 1)) != slot;
  }
  if (!mismatch) {
    const auto* completed = reinterpret_cast<const unsigned long long*>(args.last_forward_completions[slot]);
    mismatch = atomicAdd_system(const_cast<unsigned long long*>(completed), 0ULL) < generation;
  }
  if (mismatch) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(args.debug_counters->values + kGenerationMismatches),
        1ULL);
    // Continuing would consume activations from an unproven runtime generation.
    // A device trap turns stale/corrupt context into a visible execution failure
    // without synchronizing the host in every healthy backward call.
    asm volatile("trap;");
  }
}

void LaunchForwardStampValidation(
    DeviceRuntime& runtime,
    cudaStream_t stream,
    uint32_t runtime_epoch,
    const int32_t* slot,
    const int32_t* generation_high,
    const int32_t* generation_low,
    const int32_t* saved_runtime_epoch) {
  ValidateForwardStampArgs args{
      .last_forward_completions = runtime.local_slot_forward_completion_ptrs,
      .slot = slot,
      .generation_high = generation_high,
      .generation_low = generation_low,
      .runtime_epoch = saved_runtime_epoch,
      .debug_counters = runtime.debug_counters,
      .expected_runtime_epoch = runtime_epoch,
      .workspace_slots = runtime.workspace_slots,
  };
  ValidateForwardStampKernel<<<1, 1, 0, stream>>>(args);
  ThrowOnCuda(cudaGetLastError(), "ValidateForwardStampKernel");
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
    ffi::RunId run_id,
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
    int32_t forward_x_storage,
    int64_t collective_id,
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
    ffi::Result<ffi::Buffer<ffi::S32, 1>> y_routed_done,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> stamp_slot,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> stamp_generation_high,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> stamp_generation_low,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> stamp_runtime_epoch,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> failure_status) {
  g_forward_calls.fetch_add(1, std::memory_order_relaxed);
  std::optional<RuntimeLease> lease;
  try {
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
    if (forward_x_storage != static_cast<int32_t>(ForwardXStorage::kRuntimeStaged) &&
        forward_x_storage != static_cast<int32_t>(ForwardXStorage::kXlaPeerExperimental)) {
      return ffi::Error::InvalidArgument("unsupported Mixture-of-Kittens forward x storage mode");
    }
    const auto x_storage = static_cast<ForwardXStorage>(forward_x_storage);

    const size_t x_bytes = static_cast<size_t>(local_tokens) * hidden_dim * sizeof(uint16_t);
    lease = RuntimeManager::Instance().Acquire(
        run_id,
        collective_id,
        InvocationPhase::kForward,
        ForwardXRegistration{
            .pointer = x.typed_data(),
            .size_bytes = x_bytes,
            .storage = x_storage,
        });
    DeviceRuntime& runtime = RuntimeManager::Instance().Current(lease->slot);
    const ForwardXStorage active_x_storage = lease->forward_x_storage;
    if (RuntimeManager::Instance().ConsumeTestFailure(
            lease->rank, InvocationPhase::kForward, TestFailurePoint::kBeforeInputReady)) {
      throw std::runtime_error("injected forward handler failure before input readiness");
    }

    if (active_x_storage == ForwardXStorage::kRuntimeStaged) {
      ThrowOnCuda(
          cudaMemcpyAsync(runtime.x, x.typed_data(), x_bytes, cudaMemcpyDeviceToDevice, stream),
          "cudaMemcpyAsync(x workspace)");
    }
    ThrowOnCuda(cudaMemsetAsync(x_routed_ready->typed_data(), 0, x_routed_ready->size_bytes(), stream), "memset(x ready)");
    ThrowOnCuda(cudaMemsetAsync(gate_up_tile_ready->typed_data(), 0, gate_up_tile_ready->size_bytes(), stream), "memset(gate ready)");
    ThrowOnCuda(
        cudaMemsetAsync(hidden_row_block_ready->typed_data(), 0, hidden_row_block_ready->size_bytes(), stream),
        "memset(hidden ready)");
    ThrowOnCuda(cudaMemsetAsync(y_routed_ready->typed_data(), 0, y_routed_ready->size_bytes(), stream), "memset(y ready)");
    ThrowOnCuda(cudaMemsetAsync(y_routed_done->typed_data(), 0, y_routed_done->size_bytes(), stream), "memset(y done)");
    LaunchForwardInputReady(runtime, stream, lease->generation);

    MoK::activation_bf16_pgl x_pointer_data;
    MoK::activation_bf16_pgl combine_pointer_data;
    for (int peer = 0; peer < kNumDevices; ++peer) {
      const void* x_pointer = active_x_storage == ForwardXStorage::kXlaPeerExperimental
                                  ? lease->forward_x_ptrs[peer]
                                  : runtime.x_ptrs[peer];
      x_pointer_data[peer] = reinterpret_cast<kittens::bf16*>(const_cast<void*>(x_pointer));
      combine_pointer_data[peer] = reinterpret_cast<kittens::bf16*>(runtime.combine_ptrs[peer]);
    }

    const void* local_x_pointer = active_x_storage == ForwardXStorage::kXlaPeerExperimental
                                      ? lease->forward_x_ptrs[lease->rank]
                                      : runtime.x;

    MoK::globals_fwd globals{
        .x_shared = MakeGl<MoK::mlp_bf16_gl>(
            reinterpret_cast<kittens::bf16*>(const_cast<void*>(local_x_pointer)),
            1,
            1,
            local_tokens,
            hidden_dim),
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
        .peer_input_ready = runtime.forward_input_ready,
        .peer_destination_ready = nullptr,
        .input_generation = runtime.generation,
        .cancellation = runtime.cancellation,
        .peer_ready_wait_counter = runtime.debug_counters->values + kPeerReadyWaits,
        .generation_mismatch_counter = runtime.debug_counters->values + kGenerationMismatches,
        .peer_wait_events = runtime.debug_counters->values + kPeerWaitEvents,
        .peer_wait_cycles = runtime.debug_counters->values + kPeerWaitCycles,
        .peer_wait_max_cycles = runtime.debug_counters->values + kPeerWaitMaxCycles,
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
    if (RuntimeManager::Instance().ConsumeTestFailure(
            lease->rank, InvocationPhase::kForward, TestFailurePoint::kBeforeCompletion)) {
      throw std::runtime_error("injected forward handler failure before completion publication");
    }
    LaunchCompletion(
        runtime,
        stream,
        runtime.forward_completion_ptrs,
        runtime.forward_completions,
        lease->generation,
        PeerWaitPhase::kForwardPost);

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
    LaunchForwardStamp(
        runtime,
        stream,
        *lease,
        RuntimeManager::Instance().RuntimeEpoch(),
        stamp_slot->typed_data(),
        stamp_generation_high->typed_data(),
        stamp_generation_low->typed_data(),
        stamp_runtime_epoch->typed_data());
    LaunchFailureStatus(runtime, stream, lease->generation, failure_status->typed_data());
    RuntimeManager::Instance().ReleaseAfterStream(*lease, stream);
    lease.reset();
    return ffi::Error::Success();
  } catch (const std::exception& exc) {
    if (lease.has_value()) {
      DeviceRuntime& runtime = RuntimeManager::Instance().Current(lease->slot);
      try {
        CloseSynchronousFailure(*lease, runtime, stream, failure_status->typed_data(), exc.what());
        lease.reset();
        return ffi::Error::Success();
      } catch (const std::exception& closure_error) {
        TerminalCudaFailure(
            std::string("could not close a rank-local forward failure: ") + closure_error.what());
      }
    }
    return ffi::Error::Internal(exc.what());
  }
}

ffi::Error BackwardBf16(
    cudaStream_t stream,
    ffi::RunId run_id,
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
    ffi::Buffer<ffi::S32, 1> stamp_slot,
    ffi::Buffer<ffi::S32, 1> stamp_generation_high,
    ffi::Buffer<ffi::S32, 1> stamp_generation_low,
    ffi::Buffer<ffi::S32, 1> stamp_runtime_epoch,
    ffi::Buffer<ffi::S32, 1> schedule_peer_rank,
    ffi::Buffer<ffi::S32, 1> schedule_peer_token_idx,
    ffi::Buffer<ffi::S32, 1> num_tokens,
    ffi::Buffer<ffi::S32, 1> tokens_per_expert,
    int32_t top_k,
    int32_t num_comm_sms,
    int32_t macrobatch_size,
    int32_t minibatch_size,
    int32_t backward_peer_storage,
    int64_t collective_id,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> d_x,
    ffi::Result<ffi::Buffer<ffi::F32, 2>> d_router_weights,
    ffi::Result<ffi::Buffer<ffi::F32, 3>> d_w_routed_gate,
    ffi::Result<ffi::Buffer<ffi::F32, 3>> d_w_routed_up,
    ffi::Result<ffi::Buffer<ffi::F32, 3>> d_w_routed_down,
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
    ffi::Result<ffi::Buffer<ffi::S32, 1>> routed_buffers_done,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> failure_status) {
  g_backward_calls.fetch_add(1, std::memory_order_relaxed);
  std::optional<RuntimeLease> lease;
  try {
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
    if (x_routed.dimensions()[0] != macrobatch_size || x_routed.dimensions()[1] != hidden_dim ||
        gate_shared.dimensions()[0] != local_tokens || gate_shared.dimensions()[1] != intermediate_dim ||
        gate_routed.dimensions()[0] != macrobatch_size || gate_routed.dimensions()[1] != intermediate_dim ||
        up_shared.dimensions() != gate_shared.dimensions() || up_routed.dimensions() != gate_routed.dimensions() ||
        hidden_shared.dimensions() != gate_shared.dimensions() ||
        hidden_routed.dimensions() != gate_routed.dimensions()) {
      return ffi::Error::InvalidArgument("backward forward-context shape mismatch");
    }
    if (d_w_routed_gate->dimensions()[0] != local_experts ||
        d_w_routed_gate->dimensions()[1] != intermediate_dim ||
        d_w_routed_gate->dimensions()[2] != hidden_dim ||
        d_w_routed_up->dimensions() != d_w_routed_gate->dimensions() ||
        d_w_routed_down->dimensions()[0] != local_experts ||
        d_w_routed_down->dimensions()[1] != hidden_dim ||
        d_w_routed_down->dimensions()[2] != intermediate_dim) {
      return ffi::Error::InvalidArgument("backward routed weight-gradient shape mismatch");
    }
    if (d_router_weight_partials->dimensions()[0] != macrobatch_size ||
        d_router_weight_partials->dimensions()[1] != intermediate_dim / MoK::config::SWIGLU_Nb) {
      return ffi::Error::InvalidArgument("backward router-gradient partial shape mismatch");
    }
    if (backward_peer_storage != static_cast<int32_t>(BackwardPeerStorage::kRuntimeStaged) &&
        backward_peer_storage != static_cast<int32_t>(BackwardPeerStorage::kXlaPeerExperimental) &&
        backward_peer_storage != static_cast<int32_t>(BackwardPeerStorage::kXlaPeerInputsExperimental)) {
      return ffi::Error::InvalidArgument("unsupported Mixture-of-Kittens backward peer storage mode");
    }
    const auto peer_storage = static_cast<BackwardPeerStorage>(backward_peer_storage);

    const size_t x_bytes = static_cast<size_t>(local_tokens) * hidden_dim * sizeof(uint16_t);
    const size_t routed_x_bytes = static_cast<size_t>(local_tokens) * top_k * hidden_dim * sizeof(uint16_t);
    const size_t router_bytes = static_cast<size_t>(local_tokens) * top_k * sizeof(float);
    lease = RuntimeManager::Instance().Acquire(
        run_id,
        collective_id,
        InvocationPhase::kBackward,
        std::nullopt,
        BackwardPeerRegistration{
            .d_y_pointer = grad_output.typed_data(),
            .x_pointer = x.typed_data(),
            .router_weight_pointer = router_weights.typed_data(),
            .d_router_weight_pointer = d_router_weights->typed_data(),
            .activation_size_bytes = x_bytes,
            .router_size_bytes = router_bytes,
            .storage = peer_storage,
        });
    DeviceRuntime& runtime = RuntimeManager::Instance().Current(lease->slot);
    const BackwardPeerStorage active_peer_storage = lease->backward_peer_storage;
    const bool direct_inputs = active_peer_storage != BackwardPeerStorage::kRuntimeStaged;
    const bool direct_router_output = active_peer_storage == BackwardPeerStorage::kXlaPeerExperimental;
    if (RuntimeManager::Instance().ConsumeTestFailure(
            lease->rank, InvocationPhase::kBackward, TestFailurePoint::kBeforeInputReady)) {
      throw std::runtime_error("injected backward handler failure before input readiness");
    }
    LaunchForwardStampValidation(
        runtime,
        stream,
        RuntimeManager::Instance().RuntimeEpoch(),
        stamp_slot.typed_data(),
        stamp_generation_high.typed_data(),
        stamp_generation_low.typed_data(),
        stamp_runtime_epoch.typed_data());

    if (!direct_inputs) {
      ThrowOnCuda(cudaMemcpyAsync(runtime.d_y, grad_output.typed_data(), x_bytes, cudaMemcpyDeviceToDevice, stream),
                  "cudaMemcpyAsync(d_y workspace)");
      ThrowOnCuda(cudaMemcpyAsync(runtime.x, x.typed_data(), x_bytes, cudaMemcpyDeviceToDevice, stream),
                  "cudaMemcpyAsync(x backward workspace)");
      ThrowOnCuda(cudaMemcpyAsync(runtime.router_weights, router_weights.typed_data(), router_bytes,
                                  cudaMemcpyDeviceToDevice, stream),
                  "cudaMemcpyAsync(router workspace)");
    }
    if (!direct_router_output) {
      ThrowOnCuda(cudaMemsetAsync(runtime.d_router_weights, 0, router_bytes, stream),
                  "memset(d_router workspace)");
    } else {
      ThrowOnCuda(cudaMemsetAsync(d_router_weights->typed_data(), 0, router_bytes, stream),
                  "memset(d_router XLA output)");
    }
    ThrowOnCuda(cudaMemsetAsync(runtime.d_x_routed, 0, routed_x_bytes, stream), "memset(d_x routed workspace)");
    // Publish as soon as every peer-read input and peer-written destination is ready. The
    // remaining clears are local-only outputs and must not delay remote dispatch.
    LaunchBackwardInputReady(runtime, stream, lease->generation);

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
    clear(d_w_routed_gate, "memset(routed gate gradient)");
    clear(d_w_routed_up, "memset(routed up gradient)");
    clear(d_w_routed_down, "memset(routed down gradient)");

    MoK::activation_bf16_pgl x_pointer_data;
    MoK::activation_bf16_pgl d_y_pointer_data;
    MoK::activation_bf16_pgl d_x_pointer_data;
    MoK::router_weight_pgl router_pointer_data;
    MoK::router_weight_pgl d_router_pointer_data;
    for (int peer = 0; peer < kNumDevices; ++peer) {
      x_pointer_data[peer] = reinterpret_cast<kittens::bf16*>(const_cast<void*>(
          direct_inputs ? lease->backward_x_ptrs[peer] : runtime.x_ptrs[peer]));
      d_y_pointer_data[peer] = reinterpret_cast<kittens::bf16*>(const_cast<void*>(
          direct_inputs ? lease->backward_d_y_ptrs[peer] : runtime.d_y_ptrs[peer]));
      d_x_pointer_data[peer] = reinterpret_cast<kittens::bf16*>(runtime.d_x_routed_ptrs[peer]);
      router_pointer_data[peer] = reinterpret_cast<float*>(const_cast<void*>(
          direct_inputs ? lease->backward_router_weight_ptrs[peer] : runtime.router_weight_ptrs[peer]));
      d_router_pointer_data[peer] = reinterpret_cast<float*>(
          direct_router_output ? lease->backward_d_router_weight_ptrs[peer] : runtime.d_router_weight_ptrs[peer]);
    }

    auto* local_x = reinterpret_cast<kittens::bf16*>(
        direct_inputs ? x.typed_data() : runtime.x);
    auto* local_d_y = reinterpret_cast<kittens::bf16*>(
        direct_inputs ? grad_output.typed_data() : runtime.d_y);

    MoK::globals_bwd globals{
        .x_shared = MakeGl<MoK::wgrad_bf16_gl>(local_x, 1, 1, local_tokens, hidden_dim),
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
        .d_y_shared = MakeGl<MoK::mlp_bf16_gl>(local_d_y, 1, 1, local_tokens, hidden_dim),
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
        .peer_input_ready = runtime.backward_input_ready,
        .peer_destination_ready = direct_router_output ? runtime.backward_input_ready : nullptr,
        .input_generation = runtime.generation,
        .cancellation = runtime.cancellation,
        .peer_ready_wait_counter = runtime.debug_counters->values + kPeerReadyWaits,
        .generation_mismatch_counter = runtime.debug_counters->values + kGenerationMismatches,
        .peer_wait_events = runtime.debug_counters->values + kPeerWaitEvents +
                            static_cast<int>(PeerWaitPhase::kBackwardPre) * kNumDevices,
        .peer_wait_cycles = runtime.debug_counters->values + kPeerWaitCycles +
                            static_cast<int>(PeerWaitPhase::kBackwardPre) * kNumDevices,
        .peer_wait_max_cycles = runtime.debug_counters->values + kPeerWaitMaxCycles +
                                static_cast<int>(PeerWaitPhase::kBackwardPre) * kNumDevices,
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
        .d_w_routed_gate = MakeGl<MoK::d_routed_weight_f32_gl>(d_w_routed_gate->typed_data(), 1, local_experts, intermediate_dim, hidden_dim),
        .d_w_shared_up = MakeGl<MoK::d_weight_bf16_gl>(reinterpret_cast<kittens::bf16*>(d_w_shared_up->typed_data()), 1, 1, intermediate_dim, hidden_dim),
        .d_w_routed_up = MakeGl<MoK::d_routed_weight_f32_gl>(d_w_routed_up->typed_data(), 1, local_experts, intermediate_dim, hidden_dim),
        .d_w_shared_down = MakeGl<MoK::d_weight_bf16_gl>(reinterpret_cast<kittens::bf16*>(d_w_shared_down->typed_data()), 1, 1, hidden_dim, intermediate_dim),
        .d_w_routed_down = MakeGl<MoK::d_routed_weight_f32_gl>(d_w_routed_down->typed_data(), 1, local_experts, hidden_dim, intermediate_dim),
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
    if (RuntimeManager::Instance().ConsumeTestFailure(
            lease->rank, InvocationPhase::kBackward, TestFailurePoint::kBeforeCompletion)) {
      throw std::runtime_error("injected backward handler failure before completion publication");
    }
    LaunchCompletion(
        runtime,
        stream,
        runtime.backward_completion_ptrs,
        runtime.backward_completions,
        lease->generation,
        PeerWaitPhase::kBackwardPost);

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
    if (!direct_router_output) {
      ThrowOnCuda(cudaMemcpyAsync(d_router_weights->typed_data(), runtime.d_router_weights, router_bytes,
                                  cudaMemcpyDeviceToDevice, stream),
                  "cudaMemcpyAsync(d_router output)");
    }
    LaunchFailureStatus(runtime, stream, lease->generation, failure_status->typed_data());
    RuntimeManager::Instance().ReleaseAfterStream(*lease, stream);
    lease.reset();
    return ffi::Error::Success();
  } catch (const std::exception& exc) {
    if (lease.has_value()) {
      DeviceRuntime& runtime = RuntimeManager::Instance().Current(lease->slot);
      try {
        CloseSynchronousFailure(*lease, runtime, stream, failure_status->typed_data(), exc.what());
        lease.reset();
        return ffi::Error::Success();
      } catch (const std::exception& closure_error) {
        TerminalCudaFailure(
            std::string("could not close a rank-local backward failure: ") + closure_error.what());
      }
    }
    return ffi::Error::Internal(exc.what());
  }
}

auto ForwardBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::RunId>()
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
      .Attr<int32_t>("forward_x_storage")
      .Attr<int64_t>("collective_id")
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
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>();
}

auto BackwardBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::RunId>()
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
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Attr<int32_t>("top_k")
      .Attr<int32_t>("num_comm_sms")
      .Attr<int32_t>("macrobatch_size")
      .Attr<int32_t>("minibatch_size")
      .Attr<int32_t>("backward_peer_storage")
      .Attr<int64_t>("collective_id")
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::F32, 2>>()
      .Ret<ffi::Buffer<ffi::F32, 3>>()
      .Ret<ffi::Buffer<ffi::F32, 3>>()
      .Ret<ffi::Buffer<ffi::F32, 3>>()
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
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>();
}

ffi::Error FailureFence() {
  return ffi::Error::Internal("Mixture-of-Kittens failed on at least one mesh rank");
}

auto FailureFenceBinding() { return ffi::Ffi::Bind(); }

}  // namespace

extern "C" int levanter_mok_init_runtime(
    int num_devices,
    int num_tokens,
    int hidden_dim,
    int top_k,
    int workspace_slots) {
  try {
    RuntimeManager::Instance().Init(num_devices, num_tokens, hidden_dim, top_k, workspace_slots);
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

extern "C" int levanter_mok_arm_test_failure(int rank, int phase, int point, int require_two_active_slots) {
  try {
    if (phase < static_cast<int>(InvocationPhase::kForward) ||
        phase > static_cast<int>(InvocationPhase::kBackward)) {
      throw std::runtime_error("Mixture-of-Kittens test failure phase is out of range");
    }
    if (point < static_cast<int>(TestFailurePoint::kBeforeInputReady) ||
        point > static_cast<int>(TestFailurePoint::kBeforeCompletion)) {
      throw std::runtime_error("Mixture-of-Kittens test failure point is out of range");
    }
    RuntimeManager::Instance().ArmTestFailure(
        rank,
        static_cast<InvocationPhase>(phase),
        static_cast<TestFailurePoint>(point),
        require_two_active_slots != 0);
    SetLastError("");
    return 0;
  } catch (const std::exception& exc) {
    SetLastError(exc.what());
    return 1;
  }
}

extern "C" void levanter_mok_reset_call_counts() {
  g_forward_calls.store(0, std::memory_order_relaxed);
  g_backward_calls.store(0, std::memory_order_relaxed);
}

extern "C" int64_t levanter_mok_forward_call_count() {
  return g_forward_calls.load(std::memory_order_relaxed);
}

extern "C" int64_t levanter_mok_backward_call_count() {
  return g_backward_calls.load(std::memory_order_relaxed);
}

extern "C" int64_t levanter_mok_debug_counter_count() {
  return static_cast<int64_t>(kNumDevices) * kDebugCounterCount;
}

extern "C" int levanter_mok_reset_debug_counters() {
  try {
    RuntimeManager::Instance().ResetDebugCounters();
    SetLastError("");
    return 0;
  } catch (const std::exception& exc) {
    SetLastError(exc.what());
    return 1;
  }
}

extern "C" int levanter_mok_read_debug_counters(uint64_t* output, int64_t count) {
  try {
    RuntimeManager::Instance().ReadDebugCounters(output, count);
    SetLastError("");
    return 0;
  } catch (const std::exception& exc) {
    SetLastError(exc.what());
    return 1;
  }
}

extern "C" int levanter_mok_trim_default_memory_pools(uint64_t* output, int64_t count) {
  try {
    RuntimeManager::Instance().TrimDefaultMemoryPools(output, count);
    SetLastError("");
    return 0;
  } catch (const std::exception& exc) {
    SetLastError(exc.what());
    return 1;
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_mok_forward_bf16_4,
    ForwardBf16,
    ForwardBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_mok_backward_bf16_4,
    BackwardBf16,
    BackwardBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_mok_failure_fence,
    FailureFence,
    FailureFenceBinding());
