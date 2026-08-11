// Copyright The Levanter Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kNumRanks = 4;
constexpr uint8_t kAllRanksMask = (1U << kNumRanks) - 1U;
constexpr int kThreads = 256;
constexpr auto kRendezvousTimeout = std::chrono::minutes(5);

struct RankBuffers {
  const uint32_t *input = nullptr;
  uint32_t *local_read = nullptr;
  uint32_t *remote_written = nullptr;
  int64_t size = 0;
};

struct RendezvousState {
  std::array<RankBuffers, kNumRanks> ranks{};
  uint8_t arrival_mask = 0;
  uint8_t setup_mask = 0;
  uint8_t completion_mask = 0;
  int participants = 0;
  bool abort_arrival = false;
  std::string error;
};

void SetError(std::string *destination, std::string error) {
  if (destination->empty()) {
    *destination = std::move(error);
  }
}

std::string CudaError(cudaError_t status, const char *operation) {
  if (status == cudaSuccess) {
    return {};
  }
  return std::string(operation) + ": " + cudaGetErrorString(status);
}

class RendezvousRegistry {
public:
  static RendezvousRegistry &Instance() {
    static RendezvousRegistry registry;
    return registry;
  }

  std::shared_ptr<RendezvousState> Join(int64_t run_id, int rank,
                                        const RankBuffers &buffers,
                                        std::string validation_error,
                                        bool *all_arrived) {
    std::unique_lock<std::mutex> lock(mu_);
    auto [entry, inserted] =
        states_.try_emplace(run_id, std::make_shared<RendezvousState>());
    (void)inserted;
    const std::shared_ptr<RendezvousState> state = entry->second;
    ++state->participants;

    const uint8_t rank_bit = static_cast<uint8_t>(1U << rank);
    if ((state->arrival_mask & rank_bit) != 0) {
      std::ostringstream message;
      message << "collective-memory probe RunId " << run_id
              << " received duplicate rank " << rank;
      SetError(&state->error, message.str());
      state->abort_arrival = true;
    } else {
      state->arrival_mask |= rank_bit;
      state->ranks[rank] = buffers;
      SetError(&state->error, std::move(validation_error));
    }

    if (state->arrival_mask == kAllRanksMask) {
      const int64_t expected_size = state->ranks[0].size;
      for (int peer = 1; peer < kNumRanks; ++peer) {
        if (state->ranks[peer].size != expected_size) {
          SetError(&state->error, "collective-memory probe buffers must have "
                                  "the same size on all ranks");
        }
      }
    }
    cv_.notify_all();

    const bool finished = cv_.wait_for(lock, kRendezvousTimeout, [&] {
      return state->abort_arrival || state->arrival_mask == kAllRanksMask;
    });
    if (!finished) {
      std::ostringstream message;
      message << "collective-memory probe RunId " << run_id
              << " did not rendezvous all four ranks within five minutes; "
                 "arrival mask=0x"
              << std::hex << static_cast<int>(state->arrival_mask);
      SetError(&state->error, message.str());
      state->abort_arrival = true;
      cv_.notify_all();
    }
    *all_arrived =
        !state->abort_arrival && state->arrival_mask == kAllRanksMask;
    return state;
  }

  std::string CompleteSetup(const std::shared_ptr<RendezvousState> &state,
                            int rank, std::string error) {
    std::unique_lock<std::mutex> lock(mu_);
    const uint8_t rank_bit = static_cast<uint8_t>(1U << rank);
    if ((state->setup_mask & rank_bit) != 0) {
      SetError(&state->error,
               "collective-memory probe received duplicate setup completion");
    }
    state->setup_mask |= rank_bit;
    SetError(&state->error, std::move(error));
    cv_.notify_all();
    const bool finished = cv_.wait_for(lock, kRendezvousTimeout, [&] {
      return state->setup_mask == kAllRanksMask;
    });
    if (!finished) {
      SetError(&state->error, "collective-memory probe setup did not complete "
                              "on all ranks within five minutes");
      cv_.notify_all();
    }
    return state->error;
  }

  std::string CompleteWork(const std::shared_ptr<RendezvousState> &state,
                           int rank, std::string error) {
    std::unique_lock<std::mutex> lock(mu_);
    const uint8_t rank_bit = static_cast<uint8_t>(1U << rank);
    if ((state->completion_mask & rank_bit) != 0) {
      SetError(&state->error,
               "collective-memory probe received duplicate work completion");
    }
    state->completion_mask |= rank_bit;
    SetError(&state->error, std::move(error));
    cv_.notify_all();
    const bool finished = cv_.wait_for(lock, kRendezvousTimeout, [&] {
      return state->completion_mask == kAllRanksMask;
    });
    if (!finished) {
      SetError(&state->error, "collective-memory probe work did not complete "
                              "on all ranks within five minutes");
      cv_.notify_all();
    }
    return state->error;
  }

  std::string Error(const std::shared_ptr<RendezvousState> &state) {
    std::lock_guard<std::mutex> lock(mu_);
    return state->error;
  }

  void Depart(int64_t run_id, const std::shared_ptr<RendezvousState> &state) {
    std::lock_guard<std::mutex> lock(mu_);
    --state->participants;
    if (state->participants != 0) {
      return;
    }
    auto entry = states_.find(run_id);
    if (entry != states_.end() && entry->second == state) {
      states_.erase(entry);
    }
  }

private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::unordered_map<int64_t, std::shared_ptr<RendezvousState>> states_;
};

class DepartureGuard {
public:
  DepartureGuard(int64_t run_id, std::shared_ptr<RendezvousState> state)
      : run_id_(run_id), state_(std::move(state)) {}

  DepartureGuard(const DepartureGuard &) = delete;
  DepartureGuard &operator=(const DepartureGuard &) = delete;

  ~DepartureGuard() { RendezvousRegistry::Instance().Depart(run_id_, state_); }

private:
  int64_t run_id_;
  std::shared_ptr<RendezvousState> state_;
};

std::string EnablePeerAccess(int rank) {
  for (int peer = 0; peer < kNumRanks; ++peer) {
    if (peer == rank) {
      continue;
    }
    int can_access = 0;
    cudaError_t status = cudaDeviceCanAccessPeer(&can_access, rank, peer);
    if (status != cudaSuccess) {
      return CudaError(status, "cudaDeviceCanAccessPeer");
    }
    if (can_access == 0) {
      std::ostringstream message;
      message << "GPU " << rank << " cannot access peer GPU " << peer;
      return message.str();
    }
    status = cudaDeviceEnablePeerAccess(peer, 0);
    if (status == cudaErrorPeerAccessAlreadyEnabled) {
      (void)cudaGetLastError();
      continue;
    }
    if (status != cudaSuccess) {
      return CudaError(status, "cudaDeviceEnablePeerAccess");
    }
  }
  return {};
}

__global__ void CollectiveMemoryRingU32Kernel(const uint32_t *peer_input,
                                              uint32_t *local_read,
                                              uint32_t *peer_remote_written,
                                              int64_t size,
                                              uint32_t sentinel_prefix) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  local_read[index] = peer_input[index];
  peer_remote_written[index] = sentinel_prefix | static_cast<uint32_t>(index);
}

ffi::Error
CollectiveMemoryRingU32(cudaStream_t stream, ffi::RunId run_id,
                        ffi::Buffer<ffi::U32, 1> input,
                        ffi::Result<ffi::Buffer<ffi::U32, 1>> local_read,
                        ffi::Result<ffi::Buffer<ffi::U32, 1>> remote_written) {
  int rank = -1;
  cudaError_t status = cudaGetDevice(&rank);
  if (status != cudaSuccess) {
    return ffi::Error::Internal(CudaError(status, "cudaGetDevice"));
  }
  if (rank < 0 || rank >= kNumRanks) {
    return ffi::Error::InvalidArgument(
        "collective-memory probe requires exactly four visible local GPUs");
  }

  const int64_t size = input.dimensions()[0];
  std::string validation_error;
  if (local_read->dimensions()[0] != size ||
      remote_written->dimensions()[0] != size) {
    validation_error =
        "collective-memory probe input and outputs must have identical shapes";
  }
  const RankBuffers buffers{
      .input = input.typed_data(),
      .local_read = local_read->typed_data(),
      .remote_written = remote_written->typed_data(),
      .size = size,
  };

  bool all_arrived = false;
  const std::shared_ptr<RendezvousState> state =
      RendezvousRegistry::Instance().Join(run_id.run_id, rank, buffers,
                                          std::move(validation_error),
                                          &all_arrived);
  DepartureGuard departure(run_id.run_id, state);
  if (!all_arrived) {
    return ffi::Error::Internal(RendezvousRegistry::Instance().Error(state));
  }

  std::string setup_error;
  if (RendezvousRegistry::Instance().Error(state).empty()) {
    setup_error = EnablePeerAccess(rank);
  }
  SetError(&setup_error, CudaError(cudaStreamSynchronize(stream),
                                   "cudaStreamSynchronize(input readiness)"));
  const std::string shared_setup_error =
      RendezvousRegistry::Instance().CompleteSetup(state, rank,
                                                   std::move(setup_error));

  std::string work_error;
  if (shared_setup_error.empty() && size > 0) {
    const int peer = (rank + 1) % kNumRanks;
    const int64_t block_count = (size + kThreads - 1) / kThreads;
    CollectiveMemoryRingU32Kernel<<<static_cast<unsigned int>(block_count),
                                    kThreads, 0, stream>>>(
        state->ranks[peer].input, buffers.local_read,
        state->ranks[peer].remote_written, size,
        0xA5000000U | (static_cast<uint32_t>(rank) << 20));
    work_error = CudaError(cudaGetLastError(), "CollectiveMemoryRingU32Kernel");
  }
  const std::string synchronize_error = CudaError(
      cudaStreamSynchronize(stream), "cudaStreamSynchronize(work completion)");
  SetError(&work_error, synchronize_error);

  const std::string shared_work_error =
      RendezvousRegistry::Instance().CompleteWork(state, rank,
                                                  std::move(work_error));
  if (!shared_work_error.empty()) {
    return ffi::Error::Internal(shared_work_error);
  }
  return ffi::Error::Success();
}

auto CollectiveMemoryRingU32Binding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::RunId>()
      .Arg<ffi::Buffer<ffi::U32, 1>>()
      .Ret<ffi::Buffer<ffi::U32, 1>>()
      .Ret<ffi::Buffer<ffi::U32, 1>>();
}

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(levanter_collective_memory_ring_u32,
                              CollectiveMemoryRingU32,
                              CollectiveMemoryRingU32Binding());
