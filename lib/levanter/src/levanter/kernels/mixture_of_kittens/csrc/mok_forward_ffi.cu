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

#include "fabric_workspace.cuh"
#include "mok_megakernel.cuh"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

// Rank count of one expert-parallel group. Upstream instantiates the megakernel
// for 4, 8, 16, 32 and 64 devices, so this is a build-time knob rather than a
// property of the kernel. Override with -DMOK_NUM_DEVICES=<n> at compile time.
#ifndef MOK_NUM_DEVICES
#define MOK_NUM_DEVICES 4
#endif
constexpr int kNumDevices = MOK_NUM_DEVICES;
static_assert(kNumDevices == 4 || kNumDevices == 8 || kNumDevices == 16 || kNumDevices == 32 ||
                  kNumDevices == 64,
              "MOK_NUM_DEVICES must match an upstream dispatch_mlp_swiglu_combiner instantiation");
constexpr int kMaxWorkspaceSlots = 2;
// Rank bitmask. Must hold one bit per rank, so it is 64-bit wide to cover EP64;
// it was uint8_t while kNumDevices was pinned to 4.
using RankMask = uint64_t;
constexpr RankMask kAllRanksMask =
    kNumDevices == 64 ? ~static_cast<RankMask>(0)
                      : static_cast<RankMask>((static_cast<RankMask>(1) << kNumDevices) - 1);
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
// Offsets are relative to kNumDevices; the literals below hold at kNumDevices == 4
// (23/39/55/59) and scale with the rank count.
static_assert(kPeerWaitEvents == 7);
static_assert(kPeerWaitCycles == kPeerWaitEvents + kPeerWaitCellCount);
static_assert(kPeerWaitMaxCycles == kPeerWaitEvents + 2 * kPeerWaitCellCount);
static_assert(kForwardStagingCopyCalls == kPeerWaitEvents + 3 * kPeerWaitCellCount);
static_assert(kDebugCounterCount == kForwardStagingCopyCalls + 4);

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
  // True when the peer-visible buffers are offsets into a symmetric arena rather
  // than individual cudaMalloc allocations. The arena owns that memory, so the
  // destructor must not free the sub-buffers.
  bool arena_backed = false;

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
    // Debug counters and the local-only completion cursor are always private
    // allocations, arena or not.
    (void)cudaFree(debug_counters);
    (void)cudaFree(last_forward_completion);
    if (!arena_backed) {
      (void)cudaFree(cancellation);
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
    }
    (void)cudaSetDevice(original_device);
  }
};

// Point one DeviceRuntime at a symmetric arena instead of per-buffer cudaMalloc.
//
// This is the cross-process replacement for the in-process pointer table. In the
// sealed v15 path a single process allocates every rank's buffers and fills the
// peer arrays with pointers it already holds. Here each process allocates only
// its own rank's arena and maps its peers' arenas by fabric handle, so a peer
// pointer is that peer's mapped base plus the offset both ranks computed from the
// same shape parameters.
//
// `workspace` must already have imported every peer.
void BindRuntimeToArena(DeviceRuntime* runtime,
                        const mok_fabric::SymmetricWorkspace& workspace,
                        const mok_fabric::ArenaLayout& layout) {
  if (!workspace.ready()) {
    throw std::runtime_error("symmetric workspace must import peers before binding a runtime");
  }
  if (workspace.num_ranks() != runtime->num_devices) {
    throw std::runtime_error("symmetric workspace rank count does not match the runtime");
  }

  auto at = [](CUdeviceptr base, size_t offset) -> void* {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(base) + offset);
  };

  const CUdeviceptr local = workspace.peer_base(workspace.rank());
  runtime->x = at(local, layout.x);
  runtime->combine = at(local, layout.combine);
  runtime->d_y = at(local, layout.d_y);
  runtime->d_x_routed = at(local, layout.d_x_routed);
  runtime->router_weights = at(local, layout.router_weights);
  runtime->d_router_weights = at(local, layout.d_router_weights);
  runtime->generation = static_cast<uint64_t*>(at(local, layout.generation));
  runtime->forward_input_ready = static_cast<uint64_t*>(at(local, layout.forward_input_ready));
  runtime->backward_input_ready = static_cast<uint64_t*>(at(local, layout.backward_input_ready));
  runtime->forward_completions = static_cast<uint64_t*>(at(local, layout.forward_completions));
  runtime->backward_completions = static_cast<uint64_t*>(at(local, layout.backward_completions));
  runtime->cancellation = static_cast<uint64_t*>(at(local, layout.cancellation));

  // Diagnostic bisection: point every peer at this rank's own arena. Numerically wrong by
  // construction, but it separates a fault in the local layout from one in peer addressing --
  // if an illegal access survives this, the offsets are wrong, not the imported bases.
  const bool self_peers = std::getenv("MOK_ARENA_SELF_PEERS") != nullptr;
  for (int peer = 0; peer < runtime->num_devices; ++peer) {
    const CUdeviceptr base = self_peers ? local : workspace.peer_base(peer);
    runtime->x_ptrs[peer] = at(base, layout.x);
    runtime->combine_ptrs[peer] = at(base, layout.combine);
    runtime->d_y_ptrs[peer] = at(base, layout.d_y);
    runtime->d_x_routed_ptrs[peer] = at(base, layout.d_x_routed);
    runtime->router_weight_ptrs[peer] = at(base, layout.router_weights);
    runtime->d_router_weight_ptrs[peer] = at(base, layout.d_router_weights);
    runtime->forward_input_ready_ptrs[peer] =
        static_cast<uint64_t*>(at(base, layout.forward_input_ready));
    runtime->backward_input_ready_ptrs[peer] =
        static_cast<uint64_t*>(at(base, layout.backward_input_ready));
    runtime->forward_completion_ptrs[peer] =
        static_cast<uint64_t*>(at(base, layout.forward_completions));
    runtime->backward_completion_ptrs[peer] =
        static_cast<uint64_t*>(at(base, layout.backward_completions));
    runtime->cancellation_ptrs[peer] = static_cast<uint64_t*>(at(base, layout.cancellation));
  }
  runtime->arena_backed = true;
}

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
  RankMask arrival_mask = 0;
  RankMask leased_mask = 0;
  RankMask completion_mask = 0;
  bool cancelled = false;
  bool slot_released = false;
  std::array<const void*, kNumDevices> forward_x_ptrs{};
  size_t forward_x_size_bytes = 0;
  ForwardXStorage forward_x_storage = ForwardXStorage::kRuntimeStaged;
  RankMask forward_x_mask = 0;
  std::array<const void*, kNumDevices> backward_d_y_ptrs{};
  std::array<const void*, kNumDevices> backward_x_ptrs{};
  std::array<const void*, kNumDevices> backward_router_weight_ptrs{};
  std::array<void*, kNumDevices> backward_d_router_weight_ptrs{};
  size_t backward_activation_size_bytes = 0;
  size_t backward_router_size_bytes = 0;
  BackwardPeerStorage backward_peer_storage = BackwardPeerStorage::kRuntimeStaged;
  RankMask backward_peer_mask = 0;
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

  // Phase one of the fabric transport: allocate this process's own rank and
  // export one handle per workspace slot.
  //
  // Unlike `Init`, this allocates a single rank rather than every rank, because
  // under fabric transport the other ranks belong to other processes -- possibly
  // on other hosts. `out_handles` receives `workspace_slots` blobs of
  // `kFabricHandleBytes`, which the caller gathers across the expert axis.
  void InitLocalArena(int rank, int num_devices, int num_tokens, int hidden_dim, int top_k,
                      int workspace_slots, int device_ordinal, unsigned char* out_handles) {
    std::unique_lock<std::mutex> lock(mu_);
    const bool maintenance_finished =
        cv_.wait_for(lock, kWorkspaceAcquireTimeout, [&] { return !maintenance_; });
    if (!maintenance_finished) {
      throw std::runtime_error("Mixture-of-Kittens runtime maintenance did not finish within five minutes");
    }
    if (initialized_) {
      throw std::runtime_error(
          "Mixture-of-Kittens runtime must be shut down before initializing a different signature");
    }
    if (num_devices != kNumDevices) {
      throw std::runtime_error(
          "Mixture-of-Kittens expert group size does not match the compiled MOK_NUM_DEVICES");
    }
    if (rank < 0 || rank >= num_devices) {
      throw std::runtime_error("Mixture-of-Kittens local rank must be in [0, num_devices)");
    }
    if (num_tokens <= 0 || hidden_dim <= 0 || top_k <= 0) {
      throw std::runtime_error("Mixture-of-Kittens runtime dimensions must be positive");
    }
    if (workspace_slots <= 0 || workspace_slots > kMaxWorkspaceSlots) {
      throw std::runtime_error("Mixture-of-Kittens workspace slots must be one or two");
    }
    // Bind to the device the caller owns. cudaGetDeviceCount cannot stand in for this:
    // JAX restricts a process to its slice through jax_cuda_visible_devices, which does not
    // touch CUDA_VISIBLE_DEVICES, so the CUDA runtime still enumerates every GPU on the node.
    int device_count = 0;
    ThrowOnCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_ordinal < 0 || device_ordinal >= device_count) {
      throw std::runtime_error("Mixture-of-Kittens device ordinal is outside the visible range");
    }
    ThrowOnCuda(cudaSetDevice(device_ordinal), "cudaSetDevice");
    int device = 0;
    ThrowOnCuda(cudaGetDevice(&device), "cudaGetDevice(arena)");
    if (!mok_fabric::FabricHandlesSupported(device)) {
      throw std::runtime_error(
          "Mixture-of-Kittens fabric transport is unavailable on this device; the driver "
          "advertises fabric handles but exporting one failed, which usually means no IMEX "
          "channel is configured");
    }

    DestroyLocked();
    arena_mode_ = true;
    local_rank_ = rank;
    num_devices_ = num_devices;
    num_tokens_ = num_tokens;
    hidden_dim_ = hidden_dim;
    top_k_ = top_k;
    workspace_slots_ = workspace_slots;
    arena_layout_ = mok_fabric::ComputeArenaLayout(num_tokens, hidden_dim, top_k, num_devices);

    for (int slot = 0; slot < workspace_slots; ++slot) {
      arenas_[slot].CreateLocal(rank, num_devices, device, arena_layout_.total,
                                out_handles + static_cast<size_t>(slot) * mok_fabric::kFabricHandleBytes);
    }
  }

  // Phase two: import the gathered handles and bind this rank's runtimes.
  //
  // `handles` is `workspace_slots * num_devices` blobs ordered slot-major then by
  // rank, matching what phase one produced once gathered over the expert axis.
  void ImportArenaPeers(const unsigned char* handles) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!arena_mode_ || local_rank_ < 0) {
      throw std::runtime_error("ImportArenaPeers requires a prior InitLocalArena");
    }
    if (initialized_) {
      throw std::runtime_error("Mixture-of-Kittens runtime is already initialized");
    }

    int device = 0;
    ThrowOnCuda(cudaGetDevice(&device), "cudaGetDevice(import)");
    runtimes_.resize(1);

    for (int slot = 0; slot < workspace_slots_; ++slot) {
      const unsigned char* slot_handles =
          handles + static_cast<size_t>(slot) * num_devices_ * mok_fabric::kFabricHandleBytes;
      arenas_[slot].ImportPeers(slot_handles);

      auto runtime = std::make_unique<DeviceRuntime>();
      runtime->device = device;
      runtime->rank = local_rank_;
      runtime->slot = slot;
      runtime->num_devices = num_devices_;
      runtime->workspace_slots = workspace_slots_;
      BindRuntimeToArena(runtime.get(), arenas_[slot], arena_layout_);
      // Debug counters stay private per rank; they are never read by a peer.
      ThrowOnCuda(cudaMalloc(&runtime->debug_counters, sizeof(DebugCounters)),
                  "cudaMalloc(debug counters)");
      ThrowOnCuda(cudaMemset(runtime->debug_counters, 0, sizeof(DebugCounters)),
                  "cudaMemset(debug counters)");
      ThrowOnCuda(cudaMalloc(&runtime->last_forward_completion, sizeof(uint64_t)),
                  "cudaMalloc(last forward completion)");
      ThrowOnCuda(cudaMemset(runtime->last_forward_completion, 0, sizeof(uint64_t)),
                  "cudaMemset(last forward completion)");
      ThrowOnCuda(cudaStreamCreateWithFlags(&runtime->cancellation_stream, cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags(cancellation)");
      runtimes_[0][slot] = std::move(runtime);
    }

    // Each slot's runtime carries a view of every slot's completion cursor, so this can only be
    // filled once all slots exist -- the in-process path does the same in a second pass. The
    // kernel dereferences this array on device inside the forward stamp validator, so leaving it
    // null faults the first launch with an illegal address instead of failing here.
    for (int slot = 0; slot < workspace_slots_; ++slot) {
      for (int other = 0; other < workspace_slots_; ++other) {
        runtimes_[0][slot]->local_slot_forward_completion_ptrs[other] =
            runtimes_[0][other]->last_forward_completion;
      }
    }

    for (int slot = 0; slot < workspace_slots_; ++slot) {
      ValidateRuntimePointers(*runtimes_[0][slot]);
      ProbePeerArenas(arenas_[slot], slot);
    }
    initialized_ = true;
  }

  // Touch every imported peer arena before the kernel does.
  //
  // An import can succeed and still yield a mapping this device cannot reach, and the first
  // evidence would otherwise be CUDA_ERROR_ILLEGAL_ADDRESS inside the megakernel -- which
  // destroys the context and names neither the peer nor the region. A byte written and read back
  // at each peer's base, and again at its last byte, localizes that to a rank while the failure
  // is still attributable.
  void ProbePeerArenas(const mok_fabric::SymmetricWorkspace& workspace, int slot) {
    const size_t bytes = workspace.bytes();
    for (int peer = 0; peer < workspace.num_ranks(); ++peer) {
      const auto base = static_cast<uintptr_t>(workspace.peer_base(peer));
      for (const size_t offset : {static_cast<size_t>(0), bytes - sizeof(uint32_t)}) {
        void* target = reinterpret_cast<void*>(base + offset);
        const cudaError_t status = cudaMemset(target, 0, sizeof(uint32_t));
        if (status != cudaSuccess) {
          throw std::runtime_error(
              "Mixture-of-Kittens peer arena for rank " + std::to_string(peer) + " slot " +
              std::to_string(slot) + " is not reachable at offset " + std::to_string(offset) +
              " of " + std::to_string(bytes) + " bytes: " + cudaGetErrorString(status));
        }
      }
    }
    ThrowOnCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(peer arena probe)");
  }

  // Fail on the host for any pointer the kernel will dereference on device.
  //
  // A null here surfaces as CUDA_ERROR_ILLEGAL_ADDRESS on the first launch, which poisons the
  // context: the process dies without unwinding, so no Python handler runs and no traceback is
  // recorded. That is expensive to diagnose and easy to prevent, since every table is fully
  // determined once peers are imported.
  static void ValidateRuntimePointers(const DeviceRuntime& runtime) {
    auto require = [](const void* pointer, const char* what) {
      if (pointer == nullptr) {
        throw std::runtime_error(std::string("Mixture-of-Kittens arena binding left ") + what +
                                 " null; the kernel would fault on its first launch");
      }
    };
    require(runtime.x, "x");
    require(runtime.combine, "combine");
    require(runtime.d_y, "d_y");
    require(runtime.d_x_routed, "d_x_routed");
    require(runtime.router_weights, "router_weights");
    require(runtime.d_router_weights, "d_router_weights");
    require(runtime.generation, "generation");
    require(runtime.forward_input_ready, "forward_input_ready");
    require(runtime.backward_input_ready, "backward_input_ready");
    require(runtime.forward_completions, "forward_completions");
    require(runtime.backward_completions, "backward_completions");
    require(runtime.cancellation, "cancellation");
    require(runtime.debug_counters, "debug_counters");
    require(runtime.last_forward_completion, "last_forward_completion");
    for (int peer = 0; peer < runtime.num_devices; ++peer) {
      require(runtime.x_ptrs[peer], "a peer x pointer");
      require(runtime.combine_ptrs[peer], "a peer combine pointer");
      require(runtime.d_y_ptrs[peer], "a peer d_y pointer");
      require(runtime.d_x_routed_ptrs[peer], "a peer d_x_routed pointer");
      require(runtime.router_weight_ptrs[peer], "a peer router_weights pointer");
      require(runtime.d_router_weight_ptrs[peer], "a peer d_router_weights pointer");
      require(runtime.forward_input_ready_ptrs[peer], "a peer forward_input_ready pointer");
      require(runtime.backward_input_ready_ptrs[peer], "a peer backward_input_ready pointer");
      require(runtime.forward_completion_ptrs[peer], "a peer forward_completions pointer");
      require(runtime.backward_completion_ptrs[peer], "a peer backward_completions pointer");
      require(runtime.cancellation_ptrs[peer], "a peer cancellation pointer");
    }
    for (int slot = 0; slot < runtime.workspace_slots; ++slot) {
      require(runtime.local_slot_forward_completion_ptrs[slot], "a local slot completion cursor");
    }
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
    test_failure_first_key_.reset();
    test_failure_armed_ = true;
  }

  bool ConsumeTestFailure(
      const InvocationKey& key,
      int rank,
      InvocationPhase phase,
      TestFailurePoint point) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!test_failure_armed_ || test_failure_rank_ != rank || test_failure_phase_ != phase ||
        test_failure_point_ != point) {
      return false;
    }
    if (test_failure_require_two_active_slots_) {
      if (!test_failure_first_key_.has_value()) {
        test_failure_first_key_ = key;
        return false;
      }
      if (*test_failure_first_key_ == key) {
        return false;
      }
    }
    test_failure_armed_ = false;
    test_failure_first_key_.reset();
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
    const bool arena_mode = arena_mode_;
    const int local_rank = local_rank_;
    lock.unlock();
    try {
      int original_device = 0;
      ThrowOnCuda(cudaGetDevice(&original_device), "cudaGetDevice(read debug counters)");
      std::fill(output, output + static_cast<int64_t>(kNumDevices) * kDebugCounterCount, 0ULL);
      for (int rank = 0; rank < kNumDevices; ++rank) {
        // Under the fabric transport this process owns exactly one rank, its runtime sits at
        // storage index 0, and the process has a single visible GPU whose ordinal is not the rank.
        // The peers' counters live in other processes, which report their own rows. Indexing
        // `runtimes_` by rank here walks off a one-element vector and selects an absent device.
        if (arena_mode && rank != local_rank) {
          continue;
        }
        const auto& rank_runtimes = arena_mode ? runtimes_[0] : runtimes_[rank];
        if (!arena_mode) {
          ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(read debug counters)");
        }
        ThrowOnCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(read debug counters)");
        auto* rank_output = output + static_cast<int64_t>(rank) * kDebugCounterCount;
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
    // A rank is an index on the expert axis, not a device ordinal. In-process the two coincide
    // because one process owns ranks 0..kNumDevices-1 on the matching CUDA devices. Under the
    // fabric transport they cannot: at EP64 the axis runs 0..63 while every node's ordinals run
    // 0..3, so the rank has to come from the identity this process was initialized with.
    int rank = -1;
    if (arena_mode_) {
      rank = local_rank_;
    } else {
      ThrowOnCuda(cudaGetDevice(&rank), "cudaGetDevice(acquire workspace)");
    }
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
      // Under the fabric transport this counter is per process, so agreement across ranks rests
      // on every process reserving the same invocations in the same order -- which SPMD execution
      // guarantees, since the ordinal is per (run, collective, phase) and slot availability
      // evolves identically. A rank that ever diverges would stamp a generation its peers reject,
      // which the kernel reports as a generation mismatch rather than reading stale memory.
      state->generation = (++next_generation_ << 1) | static_cast<uint64_t>(state->slot);
      cv_.notify_all();
    }
    const RankMask rank_bit = static_cast<RankMask>(1) << rank;
    // Which ranks this host-side reservation must see before the invocation may proceed.
    // In-process, one manager owns every rank and collects each rank's XLA buffer pointers, so it
    // waits for all of them. Under the fabric transport each process owns exactly one rank and no
    // pointer exchange is needed -- peers are reached through the arena at a fixed offset -- so
    // waiting for absent ranks would deadlock. Cross-rank ordering is carried by the arena's
    // generation and readiness flags on device, not by this mutex.
    const RankMask rendezvous_mask = arena_mode_ ? rank_bit : kAllRanksMask;
    if ((state->arrival_mask & rank_bit) != 0) {
      ++host_slot_reuse_failures_[rank];
      CancelReservationLocked(key, state, "Mixture-of-Kittens workspace reservation received a duplicate rank");
      throw std::runtime_error(state->error);
    }
    if (phase == InvocationPhase::kForward) {
      // The peer-visible x region was sized from the num_tokens this runtime was initialized
      // with. Nothing downstream re-checks that the invocation's token count still fits, so a
      // larger one runs off the end of the arena and faults with an illegal address, destroying
      // the context. Compare the two while the sizes are still on the host.
      const size_t x_capacity_bytes =
          static_cast<size_t>(num_tokens_) * static_cast<size_t>(hidden_dim_) * sizeof(uint16_t);
      if (forward_x.has_value() && forward_x->size_bytes > x_capacity_bytes) {
        state->error = "Mixture-of-Kittens forward x buffer (" + std::to_string(forward_x->size_bytes) +
                       " bytes) exceeds the workspace sized for " + std::to_string(num_tokens_) +
                       " tokens x " + std::to_string(hidden_dim_) + " hidden (" +
                       std::to_string(x_capacity_bytes) + " bytes)";
      } else if (!forward_x.has_value() || forward_x->pointer == nullptr || forward_x->size_bytes == 0) {
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
    if (state->arrival_mask == rendezvous_mask &&
        (phase == InvocationPhase::kForward && state->forward_x_mask != rendezvous_mask)) {
      state->error = "Mixture-of-Kittens forward did not register one XLA x buffer per rank";
    }
    if (state->arrival_mask == rendezvous_mask &&
        (phase == InvocationPhase::kBackward && state->backward_peer_mask != rendezvous_mask)) {
      state->error = "Mixture-of-Kittens backward did not register all four XLA peer buffers per rank";
    }
    if (state->arrival_mask == rendezvous_mask && !state->error.empty()) {
      state->cancelled = true;
      if (failure_message_.empty()) {
        failure_message_ = state->error;
      }
    }
    cv_.notify_all();
    const bool all_ranks_arrived = cv_.wait_for(lock, kWorkspaceAcquireTimeout, [&] {
      return state->cancelled || (state->slot >= 0 && state->arrival_mask == rendezvous_mask);
    });
    if (!all_ranks_arrived) {
      ++host_slot_reuse_failures_[rank];
      CancelReservationLocked(
          key,
          state,
          "Mixture-of-Kittens workspace reservation did not rendezvous all four ranks");
    }
    if (state->cancelled) {
      if (state->arrival_mask == rendezvous_mask && state->slot >= 0) {
        FinishRankLocked(key, state, rank);
      }
      throw std::runtime_error(state->error);
    }
    state->leased_mask |= rank_bit;
    cv_.notify_all();
    if (trace_invocations_) {
      // Under the fabric transport the ordinal counter is per process, so cross-rank agreement is
      // an assumption rather than something the host can check. Printing the resolved reservation
      // makes a divergence visible by diffing the ranks' streams, which is otherwise only
      // observable as a downstream fault.
      std::fprintf(
          stderr,
          "[mok-trace] rank=%d run=%lld collective=%lld phase=%s ordinal=%llu slot=%d generation=%llu\n",
          rank,
          static_cast<long long>(key.run_id),
          static_cast<long long>(key.collective_id),
          phase == InvocationPhase::kForward ? "forward" : "backward",
          static_cast<unsigned long long>(key.ordinal),
          state->slot,
          static_cast<unsigned long long>(state->generation));
      std::fflush(stderr);
    }
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
    // Storage index, which is not the rank. In-process, runtimes_ is indexed by CUDA device and
    // the two coincide. Under the fabric transport the process holds exactly one rank's runtime in
    // a one-element vector, so the entry is at 0 while the rank itself may be anywhere on the
    // expert axis -- indexing by local_rank_ here would run off the end for every rank but 0.
    int index = -1;
    if (arena_mode_) {
      index = 0;
    } else {
      ThrowOnCuda(cudaGetDevice(&index), "cudaGetDevice(current)");
    }
    if (index < 0 || index >= static_cast<int>(runtimes_.size()) || slot < 0 || slot >= workspace_slots_ ||
        runtimes_[index][slot] == nullptr) {
      throw std::runtime_error("No Mixture-of-Kittens runtime exists for the current GPU");
    }
    return *runtimes_[index][slot];
  }

  bool TraceInvocations() const { return trace_invocations_; }

  // The shape the arenas were sized for, which the per-invocation shapes must fit inside.
  void ConfiguredShape(int& num_tokens, int& hidden_dim, int& top_k, int& workspace_slots) {
    std::lock_guard<std::mutex> lock(mu_);
    num_tokens = num_tokens_;
    hidden_dim = hidden_dim_;
    top_k = top_k_;
    workspace_slots = workspace_slots_;
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
    const RankMask rank_bit = static_cast<RankMask>(1) << rank;
    if ((state->completion_mask & rank_bit) != 0) {
      ++host_slot_reuse_failures_[rank];
      return;
    }
    state->completion_mask |= rank_bit;
    // In-process, one manager owns every rank and each rank reports separately, so the slot may
    // only be recycled once all of them have. Under the fabric transport this process owns exactly
    // one rank and the peers are unreachable from here -- but waiting for them is also unnecessary.
    // `Complete` arrives via cudaLaunchHostFunc on the invocation's stream, behind the
    // WaitCompletionKernel that spins until every peer has stamped this generation. Reaching this
    // point therefore already proves, on device, that no peer is still reading the slot.
    if (state->completion_mask != (arena_mode_ ? rank_bit : kAllRanksMask)) {
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
    const RankMask rank_bit = static_cast<RankMask>(1) << rank;
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
    // Runtimes must go first: under fabric transport their buffers are offsets
    // into the arenas, so releasing the arenas first would leave dangling peer
    // pointers in a DeviceRuntime that has not yet been torn down.
    runtimes_.clear();
    for (auto& arena : arenas_) {
      arena.Destroy();
    }
    arena_mode_ = false;
    local_rank_ = -1;
    arena_layout_ = mok_fabric::ArenaLayout{};
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
    test_failure_first_key_.reset();
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
  std::optional<InvocationKey> test_failure_first_key_;
  std::unordered_map<InvocationKey, std::shared_ptr<InvocationState>, InvocationKeyHash> invocations_;
  std::unordered_map<RunPhaseKey, std::array<uint64_t, kNumDevices>, RunPhaseKeyHash> run_ordinals_;

  // Cross-process arena state, used only by the fabric-symmetric transport.
  //
  // In the in-process path this manager owns every rank's DeviceRuntime and
  // publishes peer pointers it already holds. Under fabric transport each process
  // owns exactly one rank: `local_rank_` is that rank, `arenas_[slot]` holds this
  // rank's symmetric segment plus mapped views of its peers, and the peer tables
  // are derived from those bases. The in-process path leaves all of this unset.
  bool arena_mode_ = false;
  int local_rank_ = -1;
  mok_fabric::ArenaLayout arena_layout_{};
  std::array<mok_fabric::SymmetricWorkspace, kMaxWorkspaceSlots> arenas_{};
  const bool trace_invocations_ = std::getenv("MOK_TRACE_INVOCATIONS") != nullptr;
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

    if (RuntimeManager::Instance().TraceInvocations()) {
      // The megakernel indexes the arenas from these shapes. Only x is checked against the arena
      // capacity, so printing the rest next to the configured shape shows which one runs off the
      // end when the kernel faults on an address the host never validated.
      int arena_tokens = 0;
      int arena_hidden = 0;
      int arena_top_k = 0;
      int arena_slots = 0;
      RuntimeManager::Instance().ConfiguredShape(arena_tokens, arena_hidden, arena_top_k, arena_slots);
      std::fprintf(
          stderr,
          "[mok-trace] forward shapes local_tokens=%d hidden=%d intermediate=%d local_experts=%d "
          "top_k=%d schedule_capacity=%d macrobatch=%d minibatch=%d storage=%d | arena tokens=%d "
          "hidden=%d top_k=%d slots=%d\n",
          local_tokens,
          hidden_dim,
          intermediate_dim,
          local_experts,
          top_k,
          schedule_capacity,
          macrobatch_size,
          minibatch_size,
          forward_x_storage,
          arena_tokens,
          arena_hidden,
          arena_top_k,
          arena_slots);
      std::fflush(stderr);
    }

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
            lease->key, lease->rank, InvocationPhase::kForward, TestFailurePoint::kBeforeInputReady)) {
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
            lease->key, lease->rank, InvocationPhase::kForward, TestFailurePoint::kBeforeCompletion)) {
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
        // Report the original failure, not just the closure's. Once a device trap poisons the
        // context every later CUDA call returns the same sticky error, so the closure error is
        // usually a restatement of the damage rather than its cause -- and reporting it alone
        // discards the only description of what actually went wrong.
        TerminalCudaFailure(
            std::string("could not close a rank-local forward failure: ") + exc.what() +
            " (closure failed with: " + closure_error.what() + ")");
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
            lease->key, lease->rank, InvocationPhase::kBackward, TestFailurePoint::kBeforeInputReady)) {
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
            lease->key, lease->rank, InvocationPhase::kBackward, TestFailurePoint::kBeforeCompletion)) {
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
            std::string("could not close a rank-local backward failure: ") + exc.what() +
            " (closure failed with: " + closure_error.what() + ")");
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

ffi::Error FailureFence(
    ffi::Buffer<ffi::S32, 0> marker,
    ffi::Result<ffi::Buffer<ffi::S32, 0>> returned_marker) {
  (void)marker;
  (void)returned_marker;
  // XLA discards results when a typed FFI handler returns an error. The result
  // exists only to keep this call data-dependent and out of the runtime token.
  return ffi::Error::Internal("Mixture-of-Kittens failed on at least one mesh rank");
}

auto FailureFenceBinding() {
  return ffi::Ffi::Bind()
      .Arg<ffi::Buffer<ffi::S32, 0>>()
      .Ret<ffi::Buffer<ffi::S32, 0>>();
}

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

// Phase one of the fabric transport. `out_handles` must have room for
// `workspace_slots * levanter_mok_fabric_handle_bytes()` bytes; the caller
// gathers them across the expert axis and passes the result to
// `levanter_mok_import_arena_peers`.
extern "C" int levanter_mok_init_local_arena(
    int rank,
    int num_devices,
    int num_tokens,
    int hidden_dim,
    int top_k,
    int workspace_slots,
    int device_ordinal,
    unsigned char* out_handles) {
  try {
    if (out_handles == nullptr) {
      throw std::runtime_error("out_handles must not be null");
    }
    RuntimeManager::Instance().InitLocalArena(rank, num_devices, num_tokens, hidden_dim, top_k,
                                              workspace_slots, device_ordinal, out_handles);
    SetLastError("");
    return 0;
  } catch (const std::exception& exc) {
    SetLastError(exc.what());
    return 1;
  }
}

// Phase two. `handles` holds `workspace_slots * num_devices` blobs ordered
// slot-major then by rank.
extern "C" int levanter_mok_import_arena_peers(const unsigned char* handles) {
  try {
    if (handles == nullptr) {
      throw std::runtime_error("handles must not be null");
    }
    RuntimeManager::Instance().ImportArenaPeers(handles);
    SetLastError("");
    return 0;
  } catch (const std::exception& exc) {
    SetLastError(exc.what());
    return 1;
  }
}

// Size of one exported handle, so the Python side never hardcodes it.
extern "C" int levanter_mok_fabric_handle_bytes() { return mok_fabric::kFabricHandleBytes; }

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

// Handler symbols carry the compiled rank count so a mismatched object cannot be
// loaded silently: MOK_NUM_DEVICES=64 exports `levanter_mok_forward_bf16_64`.
// At the default of 4 these expand to the original names.
#define MOK_CONCAT_INNER(a, b) a##b
#define MOK_CONCAT(a, b) MOK_CONCAT_INNER(a, b)
#define MOK_FORWARD_SYMBOL MOK_CONCAT(levanter_mok_forward_bf16_, MOK_NUM_DEVICES)
#define MOK_BACKWARD_SYMBOL MOK_CONCAT(levanter_mok_backward_bf16_, MOK_NUM_DEVICES)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MOK_FORWARD_SYMBOL,
    ForwardBf16,
    ForwardBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MOK_BACKWARD_SYMBOL,
    BackwardBf16,
    BackwardBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_mok_failure_fence,
    FailureFence,
    FailureFenceBinding());
