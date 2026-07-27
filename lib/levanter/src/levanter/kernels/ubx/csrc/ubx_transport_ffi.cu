// Copyright The Levanter Authors
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <dlfcn.h>
#include <unistd.h>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <nccl_device.h>
#include <ubx/ubx.h>
#include <xla/ffi/api/ffi.h>

namespace ffi = xla::ffi;

namespace {

constexpr int kSupportedRanks = 8;
constexpr size_t kReg0Bytes = 4096;
constexpr size_t kPayloadAlignment = 256;
constexpr size_t kNcclAllocationAlignment = 2 * 1024 * 1024;

std::string& LastErrorStorage() {
  static std::string error;
  return error;
}

void SetLastError(std::string error) { LastErrorStorage() = std::move(error); }

void ThrowOnCuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

void ThrowOnNccl(ncclResult_t status, const char* context) {
  if (status != ncclSuccess) {
    throw std::runtime_error(std::string(context) + ": " + ncclGetErrorString(status));
  }
}

size_t CheckedMultiply(size_t left, size_t right, const char* context) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::runtime_error(std::string(context) + " overflows size_t");
  }
  return left * right;
}

size_t Align(size_t value, size_t alignment) {
  if (value > std::numeric_limits<size_t>::max() - alignment + 1) {
    throw std::runtime_error("UB-X pool layout alignment overflows size_t");
  }
  return (value + alignment - 1) / alignment * alignment;
}

std::string CanonicalPath(const char* path) {
  if (path == nullptr || path[0] == '\0') {
    return "";
  }
  char* resolved = realpath(path, nullptr);
  if (resolved == nullptr) {
    throw std::runtime_error(std::string("realpath failed for ") + path);
  }
  std::string result(resolved);
  std::free(resolved);
  return result;
}

std::string LoadedNcclPath() {
  Dl_info info{};
  auto address = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(&ncclGetVersion));
  if (dladdr(address, &info) == 0 || info.dli_fname == nullptr) {
    throw std::runtime_error("dladdr could not resolve ncclGetVersion");
  }
  return CanonicalPath(info.dli_fname);
}

void VerifyNcclRuntime(const char* expected_library, int expected_version) {
  int version = 0;
  ThrowOnNccl(ncclGetVersion(&version), "ncclGetVersion");
  if (version != expected_version) {
    throw std::runtime_error(
        "Loaded NCCL version " + std::to_string(version) + " does not match pinned version " +
        std::to_string(expected_version) + " from " + LoadedNcclPath());
  }
  const std::string expected = CanonicalPath(expected_library);
  const std::string loaded = LoadedNcclPath();
  if (loaded != expected) {
    throw std::runtime_error("Loaded NCCL library " + loaded + " does not match pinned library " + expected);
  }
}

struct RuntimeConfig {
  int num_ranks = 0;
  int max_tokens_per_rank = 0;
  int max_local_tokens = 0;
  int hidden_size = 0;
  int top_k = 0;
  int experts_per_rank = 0;
  int default_sms = 0;
  int sm_limit = 0;
  unsigned long long timeout_clocks = 0;

  bool operator==(const RuntimeConfig& other) const {
    return num_ranks == other.num_ranks && max_tokens_per_rank == other.max_tokens_per_rank &&
           max_local_tokens == other.max_local_tokens && hidden_size == other.hidden_size &&
           top_k == other.top_k && experts_per_rank == other.experts_per_rank &&
           default_sms == other.default_sms && sm_limit == other.sm_limit &&
           timeout_clocks == other.timeout_clocks;
  }
};

struct PoolLayout {
  size_t dispatch_offsets[2] = {0, 0};
  size_t dispatch_bytes = 0;
  size_t combine_offsets[2] = {0, 0};
  size_t combine_bytes = 0;
  size_t pool_bytes = 0;
};

PoolLayout MakePoolLayout(const RuntimeConfig& config) {
  const size_t hidden_bytes = CheckedMultiply(config.hidden_size, sizeof(__nv_bfloat16), "hidden bytes");
  PoolLayout layout;
  layout.dispatch_bytes =
      CheckedMultiply(config.max_tokens_per_rank, hidden_bytes, "UB-X dispatch region");
  layout.combine_bytes = CheckedMultiply(
      CheckedMultiply(config.max_local_tokens, config.top_k, "UB-X combine rows"),
      hidden_bytes,
      "UB-X combine region");
  layout.dispatch_offsets[0] = Align(kReg0Bytes, kPayloadAlignment);
  layout.dispatch_offsets[1] =
      Align(layout.dispatch_offsets[0] + layout.dispatch_bytes, kPayloadAlignment);
  layout.combine_offsets[0] =
      Align(layout.dispatch_offsets[1] + layout.dispatch_bytes, kPayloadAlignment);
  layout.combine_offsets[1] = Align(layout.combine_offsets[0] + layout.combine_bytes, kPayloadAlignment);
  layout.pool_bytes = Align(layout.combine_offsets[1] + layout.combine_bytes, kNcclAllocationAlignment);
  return layout;
}

void ValidateConfig(const RuntimeConfig& config) {
  if (config.num_ranks != kSupportedRanks) {
    throw std::runtime_error("UB-X FFI currently supports exactly 8 local ranks");
  }
  if (config.max_tokens_per_rank <= 0 || config.max_local_tokens <= 0 || config.hidden_size <= 0 ||
      config.top_k <= 0 || config.experts_per_rank <= 0) {
    throw std::runtime_error("UB-X FFI shape parameters must be positive");
  }
  if (config.hidden_size % 32 != 0) {
    throw std::runtime_error("UB-X FFI hidden size must be divisible by 32");
  }
  if (config.default_sms < 0 || config.sm_limit < 0 || config.timeout_clocks == 0) {
    throw std::runtime_error("UB-X FFI launch parameters are invalid");
  }
}

struct DeviceRuntime {
  int rank = -1;
  int device_id = -1;
  ncclComm_t comm = nullptr;
  void* pool = nullptr;
  ncclWindow_t window = nullptr;
  ncclDevComm_t dev_comm{};
  bool dev_comm_created = false;
  std::atomic<uint64_t> dispatch_calls{0};
  std::atomic<uint64_t> combine_calls{0};
};

template <typename Function>
void RunOnAllRanks(int num_ranks, Function function) {
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(num_ranks);
  threads.reserve(num_ranks);
  for (int rank = 0; rank < num_ranks; ++rank) {
    threads.emplace_back([&, rank]() {
      try {
        function(rank);
      } catch (...) {
        errors[rank] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    if (error != nullptr) {
      std::rethrow_exception(error);
    }
  }
}

class RuntimeManager {
 public:
  static RuntimeManager& Instance() {
    static RuntimeManager manager;
    return manager;
  }

  void Init(RuntimeConfig config, const char* expected_nccl_library, int expected_nccl_version) {
    std::lock_guard<std::mutex> lock(mu_);
    ValidateConfig(config);
    VerifyNcclRuntime(expected_nccl_library, expected_nccl_version);
    if (initialized_ && config_ == config) {
      return;
    }
    DestroyLocked(true);

    int device_count = 0;
    ThrowOnCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count != config.num_ranks) {
      throw std::runtime_error(
          "UB-X FFI requires its expert group to span every visible local GPU; visible=" +
          std::to_string(device_count));
    }

    config_ = config;
    layout_ = MakePoolLayout(config_);
    runtimes_.resize(config_.num_ranks);
    std::vector<int> devices(config_.num_ranks);
    std::vector<ncclComm_t> comms(config_.num_ranks, nullptr);
    for (int rank = 0; rank < config_.num_ranks; ++rank) {
      devices[rank] = rank;
      runtimes_[rank] = std::make_unique<DeviceRuntime>();
      runtimes_[rank]->rank = rank;
      runtimes_[rank]->device_id = rank;
    }

    int original_device = 0;
    ThrowOnCuda(cudaGetDevice(&original_device), "cudaGetDevice(init)");
    try {
      ThrowOnNccl(
          ncclCommInitAll(comms.data(), config_.num_ranks, devices.data()),
          "ncclCommInitAll");
      for (int rank = 0; rank < config_.num_ranks; ++rank) {
        runtimes_[rank]->comm = comms[rank];
      }
      for (int rank = 0; rank < config_.num_ranks; ++rank) {
        DeviceRuntime& runtime = *runtimes_[rank];
        ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(communicator validation)");
        int comm_rank = -1;
        int comm_count = 0;
        int comm_device = -1;
        ThrowOnNccl(ncclCommUserRank(runtime.comm, &comm_rank), "ncclCommUserRank");
        ThrowOnNccl(ncclCommCount(runtime.comm, &comm_count), "ncclCommCount");
        ThrowOnNccl(ncclCommCuDevice(runtime.comm, &comm_device), "ncclCommCuDevice");
        if (comm_rank != rank || comm_count != config_.num_ranks || comm_device != rank) {
          throw std::runtime_error("ncclCommInitAll returned an unexpected local rank mapping");
        }
        ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
        ThrowOnNccl(ncclCommQueryProperties(runtime.comm, &properties), "ncclCommQueryProperties");
        if (!properties.deviceApiSupport) {
          throw std::runtime_error("Pinned NCCL reports no device API support");
        }
      }

      for (int rank = 0; rank < config_.num_ranks; ++rank) {
        DeviceRuntime& runtime = *runtimes_[rank];
        ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(pool allocation)");
        ThrowOnNccl(ncclMemAlloc(&runtime.pool, layout_.pool_bytes), "ncclMemAlloc");
        if (runtime.pool == nullptr) {
          throw std::runtime_error("ncclMemAlloc returned null");
        }
        ThrowOnCuda(cudaMemset(runtime.pool, 0, layout_.pool_bytes), "cudaMemset(UB-X pool)");
        ThrowOnCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(UB-X pool)");
      }

      RunOnAllRanks(config_.num_ranks, [&](int rank) {
        DeviceRuntime& runtime = *runtimes_[rank];
        ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(window init)");
        ThrowOnNccl(
            ncclCommWindowRegister(
                runtime.comm,
                runtime.pool,
                layout_.pool_bytes,
                &runtime.window,
                NCCL_WIN_COLL_SYMMETRIC),
            "ncclCommWindowRegister");
        ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        requirements.lsaMultimem = false;
        ThrowOnNccl(
            ncclDevCommCreate(runtime.comm, &requirements, &runtime.dev_comm),
            "ncclDevCommCreate");
        runtime.dev_comm_created = true;
        for (int peer = 0; peer < config_.num_ranks; ++peer) {
          void* peer_pointer = nullptr;
          ThrowOnNccl(
              ncclGetLsaDevicePointer(runtime.window, 0, peer, &peer_pointer),
              "ncclGetLsaDevicePointer");
          if (peer_pointer == nullptr) {
            throw std::runtime_error("NCCL returned a null LSA peer pointer");
          }
        }
        ubx_set_timeout(config_.timeout_clocks);
        ThrowOnCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(window init)");
      });
    } catch (...) {
      cudaSetDevice(original_device);
      DestroyLocked(false);
      throw;
    }
    ThrowOnCuda(cudaSetDevice(original_device), "cudaSetDevice(restore after init)");
    initialized_ = true;
  }

  void Shutdown() {
    std::lock_guard<std::mutex> lock(mu_);
    DestroyLocked(true);
  }

  DeviceRuntime& RuntimeForCurrentDevice() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_) {
      throw std::runtime_error("UB-X local runtime is not initialized");
    }
    // cudaGetDevice returns the process-local CUDA ordinal selected by XLA's
    // FFI execution context. Global JAX Device.id values never enter this map.
    int device = -1;
    ThrowOnCuda(cudaGetDevice(&device), "cudaGetDevice(FFI)");
    if (device < 0 || device >= static_cast<int>(runtimes_.size()) || runtimes_[device] == nullptr) {
      throw std::runtime_error("No UB-X runtime exists for the current CUDA device");
    }
    return *runtimes_[device];
  }

  const RuntimeConfig& config() const { return config_; }
  const PoolLayout& layout() const { return layout_; }

 private:
  RuntimeManager() = default;

  void DestroyLocked(bool throw_errors) {
    if (runtimes_.empty()) {
      initialized_ = false;
      config_ = RuntimeConfig{};
      layout_ = PoolLayout{};
      return;
    }

    int original_device = 0;
    cudaGetDevice(&original_device);
    std::string first_error;
    auto record_cuda = [&](cudaError_t status, const char* context) {
      if (status != cudaSuccess && first_error.empty()) {
        first_error = std::string(context) + ": " + cudaGetErrorString(status);
      }
    };
    auto record_nccl = [&](ncclResult_t status, const char* context) {
      if (status != ncclSuccess && first_error.empty()) {
        first_error = std::string(context) + ": " + ncclGetErrorString(status);
      }
    };

    for (auto& runtime : runtimes_) {
      if (runtime == nullptr) {
        continue;
      }
      record_cuda(cudaSetDevice(runtime->device_id), "cudaSetDevice(shutdown sync)");
      record_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(shutdown)");
    }

    try {
      RunOnAllRanks(static_cast<int>(runtimes_.size()), [&](int rank) {
        DeviceRuntime& runtime = *runtimes_[rank];
        ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(devcomm destroy)");
        if (runtime.dev_comm_created) {
          ThrowOnNccl(
              ncclDevCommDestroy(runtime.comm, &runtime.dev_comm),
              "ncclDevCommDestroy");
          runtime.dev_comm_created = false;
        }
      });
    } catch (const std::exception& error) {
      if (first_error.empty()) {
        first_error = error.what();
      }
    }

    try {
      RunOnAllRanks(static_cast<int>(runtimes_.size()), [&](int rank) {
        DeviceRuntime& runtime = *runtimes_[rank];
        ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(window destroy)");
        if (runtime.window != nullptr) {
          ThrowOnNccl(
              ncclCommWindowDeregister(runtime.comm, runtime.window),
              "ncclCommWindowDeregister");
          runtime.window = nullptr;
        }
      });
    } catch (const std::exception& error) {
      if (first_error.empty()) {
        first_error = error.what();
      }
    }

    for (auto& runtime : runtimes_) {
      if (runtime == nullptr) {
        continue;
      }
      record_cuda(cudaSetDevice(runtime->device_id), "cudaSetDevice(pool free)");
      if (runtime->pool != nullptr) {
        record_nccl(ncclMemFree(runtime->pool), "ncclMemFree");
        runtime->pool = nullptr;
      }
    }

    try {
      RunOnAllRanks(static_cast<int>(runtimes_.size()), [&](int rank) {
        DeviceRuntime& runtime = *runtimes_[rank];
        ThrowOnCuda(cudaSetDevice(rank), "cudaSetDevice(communicator destroy)");
        if (runtime.comm != nullptr) {
          ThrowOnNccl(ncclCommFinalize(runtime.comm), "ncclCommFinalize");
          ThrowOnNccl(ncclCommDestroy(runtime.comm), "ncclCommDestroy");
          runtime.comm = nullptr;
        }
      });
    } catch (const std::exception& error) {
      if (first_error.empty()) {
        first_error = error.what();
      }
    }

    cudaSetDevice(original_device);
    runtimes_.clear();
    initialized_ = false;
    config_ = RuntimeConfig{};
    layout_ = PoolLayout{};
    if (throw_errors && !first_error.empty()) {
      throw std::runtime_error(first_error);
    }
  }

  std::mutex mu_;
  bool initialized_ = false;
  RuntimeConfig config_{};
  PoolLayout layout_{};
  std::vector<std::unique_ptr<DeviceRuntime>> runtimes_;
};

__global__ void CopyAndMaskDispatchKernel(
    const __nv_bfloat16* source,
    const bool* valid,
    __nv_bfloat16* destination,
    int rows,
    int hidden) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(rows) * hidden;
  if (index >= count) {
    return;
  }
  const int row = static_cast<int>(index / hidden);
  destination[index] = valid[row] ? source[index] : __float2bfloat16(0.0f);
}

ffi::Error DispatchTopkBf16(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> x,
    ffi::Buffer<ffi::S32, 2> dispatch_topk_expert,
    ffi::Buffer<ffi::S32, 2> dispatch_topk_slot,
    ffi::Buffer<ffi::PRED, 1> dispatch_valid,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> dispatch_output) {
  try {
    RuntimeManager& manager = RuntimeManager::Instance();
    DeviceRuntime& runtime = manager.RuntimeForCurrentDevice();
    const RuntimeConfig& config = manager.config();
    const PoolLayout& layout = manager.layout();
    const auto x_dims = x.dimensions();
    const auto expert_dims = dispatch_topk_expert.dimensions();
    const auto slot_dims = dispatch_topk_slot.dimensions();
    const auto valid_dims = dispatch_valid.dimensions();
    const auto output_dims = dispatch_output->dimensions();
    if (x_dims[0] != config.max_local_tokens || x_dims[1] != config.hidden_size ||
        expert_dims[0] != config.max_local_tokens || expert_dims[1] != config.top_k ||
        slot_dims[0] != config.max_local_tokens || slot_dims[1] != config.top_k ||
        valid_dims[0] != config.max_tokens_per_rank ||
        output_dims[0] != config.max_tokens_per_rank || output_dims[1] != config.hidden_size) {
      return ffi::Error::InvalidArgument("UB-X dispatch tensor shapes do not match the initialized runtime");
    }

    const uint64_t call = runtime.dispatch_calls.fetch_add(1, std::memory_order_relaxed);
    const size_t dispatch_offset = layout.dispatch_offsets[call % 2];
    ubx_a2av_token_bf16_bf16_topk(
        config.num_ranks,
        runtime.rank,
        config.max_local_tokens,
        config.hidden_size / 32,
        config.experts_per_rank,
        config.top_k,
        reinterpret_cast<uintptr_t>(dispatch_topk_expert.typed_data()),
        reinterpret_cast<uintptr_t>(dispatch_topk_slot.typed_data()),
        static_cast<int64_t>(dispatch_offset / sizeof(uint4)),
        &runtime.dev_comm,
        runtime.window,
        reinterpret_cast<uintptr_t>(runtime.pool),
        reinterpret_cast<uintptr_t>(x.typed_data()),
        config.default_sms,
        config.sm_limit,
        1,
        stream);

    constexpr int kThreads = 256;
    const size_t count = static_cast<size_t>(config.max_tokens_per_rank) * config.hidden_size;
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    const auto* dispatch_source = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<const uint8_t*>(runtime.pool) + dispatch_offset);
    CopyAndMaskDispatchKernel<<<blocks, kThreads, 0, stream>>>(
        dispatch_source,
        dispatch_valid.typed_data(),
        reinterpret_cast<__nv_bfloat16*>(dispatch_output->typed_data()),
        config.max_tokens_per_rank,
        config.hidden_size);
    ThrowOnCuda(cudaGetLastError(), "CopyAndMaskDispatchKernel");
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
}

ffi::Error CombinePush3Bf16(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> expert_outputs,
    ffi::Buffer<ffi::S32, 2> inverse_map,
    ffi::Buffer<ffi::S32, 2> topk_idx,
    ffi::Buffer<ffi::F32, 2> gate_weights,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> combined_output) {
  try {
    RuntimeManager& manager = RuntimeManager::Instance();
    DeviceRuntime& runtime = manager.RuntimeForCurrentDevice();
    const RuntimeConfig& config = manager.config();
    const PoolLayout& layout = manager.layout();
    const auto expert_dims = expert_outputs.dimensions();
    const auto inverse_dims = inverse_map.dimensions();
    const auto topk_dims = topk_idx.dimensions();
    const auto gate_dims = gate_weights.dimensions();
    const auto output_dims = combined_output->dimensions();
    const int total_experts = config.num_ranks * config.experts_per_rank;
    if (expert_dims[0] != config.max_tokens_per_rank || expert_dims[1] != config.hidden_size ||
        inverse_dims[0] != config.max_tokens_per_rank || inverse_dims[1] != 4 ||
        topk_dims[0] != config.max_local_tokens || topk_dims[1] != config.top_k ||
        gate_dims[0] != config.max_local_tokens || gate_dims[1] != total_experts ||
        output_dims[0] != config.max_local_tokens || output_dims[1] != config.hidden_size) {
      return ffi::Error::InvalidArgument("UB-X combine tensor shapes do not match the initialized runtime");
    }

    const uint64_t call = runtime.combine_calls.fetch_add(1, std::memory_order_relaxed);
    const size_t destination_offset = layout.combine_offsets[call % 2];
    const int64_t lineoffset_destination =
        static_cast<int64_t>(destination_offset / sizeof(uint4));
    const int blocks_per_token = config.hidden_size / 32;

    ubx_combine_push3_phase1_write(
        config.num_ranks,
        config.max_tokens_per_rank,
        blocks_per_token,
        config.top_k,
        reinterpret_cast<uintptr_t>(inverse_map.typed_data()),
        config.max_tokens_per_rank,
        lineoffset_destination,
        &runtime.dev_comm,
        runtime.window,
        reinterpret_cast<uintptr_t>(expert_outputs.typed_data()),
        config.default_sms,
        config.sm_limit,
        stream);
    ubx_combine_push3_phase2_signal(
        config.num_ranks,
        &runtime.dev_comm,
        runtime.window,
        reinterpret_cast<uintptr_t>(runtime.pool),
        stream);
    ubx_combine_push3_phase3_sum(
        config.max_local_tokens,
        blocks_per_token,
        config.top_k,
        total_experts,
        reinterpret_cast<uintptr_t>(topk_idx.typed_data()),
        reinterpret_cast<uintptr_t>(gate_weights.typed_data()),
        lineoffset_destination,
        reinterpret_cast<uintptr_t>(runtime.pool),
        reinterpret_cast<uintptr_t>(combined_output->typed_data()),
        config.default_sms,
        config.sm_limit,
        stream);
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
}

auto DispatchBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Arg<ffi::Buffer<ffi::PRED, 1>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>();
}

auto CombineBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Arg<ffi::Buffer<ffi::F32, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>();
}

}  // namespace

extern "C" int levanter_ubx_init_local_runtime(
    int num_ranks,
    int max_tokens_per_rank,
    int max_local_tokens,
    int hidden_size,
    int top_k,
    int experts_per_rank,
    int default_sms,
    int sm_limit,
    unsigned long long timeout_clocks,
    const char* expected_nccl_library,
    int expected_nccl_version) {
  try {
    RuntimeManager::Instance().Init(
        RuntimeConfig{
            num_ranks,
            max_tokens_per_rank,
            max_local_tokens,
            hidden_size,
            top_k,
            experts_per_rank,
            default_sms,
            sm_limit,
            timeout_clocks},
        expected_nccl_library,
        expected_nccl_version);
    SetLastError("");
    return 0;
  } catch (const std::exception& error) {
    SetLastError(error.what());
    return 1;
  }
}

extern "C" void levanter_ubx_shutdown_local_runtime() {
  try {
    RuntimeManager::Instance().Shutdown();
    SetLastError("");
  } catch (const std::exception& error) {
    SetLastError(error.what());
  }
}

extern "C" const char* levanter_ubx_last_error() { return LastErrorStorage().c_str(); }

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_ubx_dispatch_topk_bf16,
    DispatchTopkBf16,
    DispatchBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_ubx_combine_push3_bf16,
    CombinePush3Bf16,
    CombineBinding());
