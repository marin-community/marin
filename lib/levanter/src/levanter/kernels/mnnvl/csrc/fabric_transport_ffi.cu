// Copyright The Levanter Authors
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kMaxRanks = 128;
constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / 32;
constexpr int kMaxExchangeBlocks = 192;

size_t AlignUp(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

void ThrowOnDriver(CUresult status, const char* context) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* description = nullptr;
  cuGetErrorString(status, &description);
  throw std::runtime_error(
      std::string(context) + ": " + (description == nullptr ? "unknown CUDA driver error" : description));
}

void ThrowOnCuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

ffi::Error CudaError(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return ffi::Error::Success();
  }
  return ffi::Error::Internal(std::string(context) + ": " + cudaGetErrorString(status));
}

std::string& LastErrorStorage() {
  static std::string error;
  return error;
}

__global__ void ProbeWriteKernel(uint8_t** peer_bases, int rank, int world_size) {
  const int destination = threadIdx.x;
  if (destination < world_size) {
    reinterpret_cast<int32_t*>(peer_bases[destination])[rank] = rank * 1000 + destination;
  }
  __threadfence_system();
}

__global__ void SendRowsKernel(
    const uint4* input,
    const int32_t* source_rows,
    const int32_t* destination_ranks,
    const int32_t* destination_slots,
    uint8_t** peer_bases,
    size_t data_offset,
    size_t slot_epoch_offset,
    size_t source_rank_offset,
    size_t source_slot_offset,
    int rank,
    int world_size,
    int input_value_rows,
    int send_rows,
    int output_rows,
    int vectors_per_row,
    int32_t epoch) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int first_row = blockIdx.x * kWarpsPerBlock + warp;
  const int row_stride = gridDim.x * kWarpsPerBlock;
  for (int row = first_row; row < send_rows; row += row_stride) {
    const int source_row = source_rows == nullptr ? row : source_rows[row];
    const int destination = destination_ranks[row];
    const int slot = destination_slots[row];
    if (source_row < 0 || source_row >= input_value_rows ||
        destination < 0 || destination >= world_size || slot < 0 || slot >= output_rows) {
      continue;
    }

    uint8_t* peer = peer_bases[destination];
    auto* peer_data = reinterpret_cast<uint4*>(peer + data_offset);
    for (int vector = lane; vector < vectors_per_row; vector += 32) {
      peer_data[static_cast<size_t>(slot) * vectors_per_row + vector] =
          input[static_cast<size_t>(source_row) * vectors_per_row + vector];
    }
    if (lane == 0) {
      reinterpret_cast<int32_t*>(peer + source_rank_offset)[slot] = rank;
      reinterpret_cast<int32_t*>(peer + source_slot_offset)[slot] = row;
      reinterpret_cast<int32_t*>(peer + slot_epoch_offset)[slot] = epoch;
    }
  }
  // Every lane owns part of each row. Publish all lanes' remote stores before
  // the subsequent signal kernel tells receivers that this sender is complete.
  __threadfence_system();
}

__global__ void SignalPeersKernel(
    uint8_t** peer_bases,
    size_t signal_offset,
    int rank,
    int world_size,
    int32_t epoch) {
  const int destination = threadIdx.x;
  if (destination < world_size) {
    __threadfence_system();
    reinterpret_cast<int32_t*>(peer_bases[destination] + signal_offset)[rank] = epoch;
  }
}

__global__ void WaitSignalsKernel(const int32_t* signals, int world_size, int32_t epoch) {
  const int peer = threadIdx.x;
  if (peer < world_size) {
    while (reinterpret_cast<const volatile int32_t*>(signals)[peer] != epoch) {
      __nanosleep(64);
    }
  }
  __syncthreads();
}

__global__ void CopyRowsKernel(
    const uint4* local_data,
    const int32_t* slot_epochs,
    const int32_t* local_source_ranks,
    const int32_t* local_source_slots,
    uint4* output,
    int32_t* output_source_ranks,
    int32_t* output_source_slots,
    int world_size,
    int input_rows,
    int output_rows,
    int vectors_per_row,
    int32_t epoch) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int first_row = blockIdx.x * kWarpsPerBlock + warp;
  const int row_stride = gridDim.x * kWarpsPerBlock;
  for (int row = first_row; row < output_rows; row += row_stride) {
    const bool valid = slot_epochs[row] == epoch;
    const uint4 zero{};
    for (int vector = lane; vector < vectors_per_row; vector += 32) {
      output[static_cast<size_t>(row) * vectors_per_row + vector] =
          valid ? local_data[static_cast<size_t>(row) * vectors_per_row + vector] : zero;
    }
    if (lane == 0) {
      output_source_ranks[row] = valid ? local_source_ranks[row] : world_size;
      output_source_slots[row] = valid ? local_source_slots[row] : input_rows;
    }
  }
}

class FabricRuntime {
 public:
  static FabricRuntime& Instance() {
    static FabricRuntime runtime;
    return runtime;
  }

  void InitializeLocal(
      int rank,
      int world_size,
      int64_t buffer_rows,
      int64_t row_bytes,
      uint8_t* exported_handle,
      int handle_size) {
    Shutdown();
    if (rank < 0 || rank >= world_size || world_size <= 0 || world_size > kMaxRanks) {
      throw std::runtime_error("invalid MNNVL rank or world size");
    }
    if (buffer_rows <= 0 || buffer_rows > std::numeric_limits<int>::max()) {
      throw std::runtime_error("invalid MNNVL buffer row count");
    }
    if (row_bytes <= 0 || row_bytes % sizeof(uint4) != 0) {
      throw std::runtime_error("MNNVL row bytes must be a positive multiple of 16");
    }
    if (handle_size != static_cast<int>(sizeof(CUmemFabricHandle))) {
      throw std::runtime_error("unexpected CUmemFabricHandle size");
    }

    ThrowOnDriver(cuInit(0), "cuInit");
    ThrowOnDriver(cuCtxGetDevice(&device_), "cuCtxGetDevice");
    int fabric_supported = 0;
    ThrowOnDriver(
        cuDeviceGetAttribute(&fabric_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device_),
        "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED)");
    if (fabric_supported == 0) {
      throw std::runtime_error("current GPU does not support CUDA fabric handles");
    }

    allocation_properties_ = {};
    allocation_properties_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocation_properties_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocation_properties_.location.id = device_;
    allocation_properties_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    ThrowOnDriver(
        cuMemGetAllocationGranularity(
            &allocation_granularity_, &allocation_properties_, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity");

    rank_ = rank;
    world_size_ = world_size;
    buffer_rows_ = static_cast<int>(buffer_rows);
    row_bytes_ = static_cast<size_t>(row_bytes);
    probe_offset_ = 0;
    done_offset_ = AlignUp(static_cast<size_t>(world_size_) * sizeof(int32_t), 128);
    finished_offset_ = done_offset_ + AlignUp(static_cast<size_t>(world_size_) * sizeof(int32_t), 128);
    slot_epoch_offset_ = finished_offset_ + AlignUp(static_cast<size_t>(world_size_) * sizeof(int32_t), 128);
    source_rank_offset_ =
        slot_epoch_offset_ + AlignUp(static_cast<size_t>(buffer_rows_) * sizeof(int32_t), 128);
    source_slot_offset_ =
        source_rank_offset_ + AlignUp(static_cast<size_t>(buffer_rows_) * sizeof(int32_t), 128);
    data_offset_ = source_slot_offset_ + AlignUp(static_cast<size_t>(buffer_rows_) * sizeof(int32_t), 128);
    allocation_bytes_ =
        AlignUp(data_offset_ + static_cast<size_t>(buffer_rows_) * row_bytes_, allocation_granularity_);

    ThrowOnDriver(
        cuMemCreate(&local_allocation_, allocation_bytes_, &allocation_properties_, 0),
        "cuMemCreate");
    ThrowOnDriver(
        cuMemAddressReserve(&local_address_, allocation_bytes_, allocation_granularity_, 0, 0),
        "cuMemAddressReserve");
    ThrowOnDriver(
        cuMemMap(local_address_, allocation_bytes_, 0, local_allocation_, 0),
        "cuMemMap");
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    ThrowOnDriver(cuMemSetAccess(local_address_, allocation_bytes_, &access, 1), "cuMemSetAccess");

    CUmemFabricHandle handle{};
    ThrowOnDriver(
        cuMemExportToShareableHandle(&handle, local_allocation_, CU_MEM_HANDLE_TYPE_FABRIC, 0),
        "cuMemExportToShareableHandle");
    std::memcpy(exported_handle, &handle, sizeof(handle));
    ThrowOnCuda(
        cudaMemset(reinterpret_cast<void*>(local_address_), 0, data_offset_),
        "cudaMemset(MNNVL control)");
  }

  void SynchronizeHandles(const uint8_t* handles, int count, int handle_size) {
    if (local_address_ == 0 || count != world_size_ ||
        handle_size != static_cast<int>(sizeof(CUmemFabricHandle))) {
      throw std::runtime_error("MNNVL handle synchronization does not match the local allocation");
    }
    remote_addresses_.assign(world_size_, 0);
    imported_allocations_.assign(world_size_, 0);
    for (int peer = 0; peer < world_size_; ++peer) {
      if (peer == rank_) {
        remote_addresses_[peer] = local_address_;
        continue;
      }
      CUmemFabricHandle handle{};
      std::memcpy(&handle, handles + static_cast<size_t>(peer) * handle_size, sizeof(handle));
      CUmemGenericAllocationHandle imported = 0;
      CUdeviceptr address = 0;
      ThrowOnDriver(
          cuMemImportFromShareableHandle(&imported, &handle, CU_MEM_HANDLE_TYPE_FABRIC),
          "cuMemImportFromShareableHandle");
      ThrowOnDriver(cuMemAddressReserve(&address, allocation_bytes_, 0, 0, 0), "cuMemAddressReserve(peer)");
      ThrowOnDriver(cuMemMap(address, allocation_bytes_, 0, imported, 0), "cuMemMap(peer)");
      CUmemAccessDesc access{};
      access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      access.location.id = device_;
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      ThrowOnDriver(cuMemSetAccess(address, allocation_bytes_, &access, 1), "cuMemSetAccess(peer)");
      imported_allocations_[peer] = imported;
      remote_addresses_[peer] = address;
    }

    std::vector<uint8_t*> host_bases(world_size_);
    for (int peer = 0; peer < world_size_; ++peer) {
      host_bases[peer] = reinterpret_cast<uint8_t*>(remote_addresses_[peer]);
    }
    ThrowOnCuda(cudaMalloc(&remote_bases_, world_size_ * sizeof(uint8_t*)), "cudaMalloc(remote bases)");
    ThrowOnCuda(
        cudaMemcpy(
            remote_bases_,
            host_bases.data(),
            world_size_ * sizeof(uint8_t*),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(remote bases)");
  }

  int32_t NextEpoch() {
    const uint32_t epoch = next_epoch_.fetch_add(1, std::memory_order_relaxed);
    if (epoch == 0 || epoch > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error("MNNVL exchange epoch exhausted");
    }
    return static_cast<int32_t>(epoch);
  }

  ffi::Error Exchange(
      cudaStream_t stream,
      const uint16_t* input,
      const int32_t* destination_ranks,
      const int32_t* destination_slots,
      uint16_t* output,
      int32_t* output_source_ranks,
      int32_t* output_source_slots,
      int input_value_rows,
      int send_rows,
      int output_rows,
      int hidden,
      const int32_t* source_rows = nullptr) {
    if (remote_bases_ == nullptr) {
      return ffi::Error(
          ffi::ErrorCode::kFailedPrecondition,
          "MNNVL peer handles are not synchronized");
    }
    if (input_value_rows <= 0 || send_rows <= 0 || output_rows <= 0 ||
        send_rows > buffer_rows_ || output_rows > buffer_rows_) {
      return ffi::Error::InvalidArgument("MNNVL exchange rows exceed the configured fabric buffer");
    }
    const size_t requested_row_bytes = static_cast<size_t>(hidden) * sizeof(uint16_t);
    if (requested_row_bytes != row_bytes_ || requested_row_bytes % sizeof(uint4) != 0) {
      return ffi::Error::InvalidArgument("MNNVL exchange hidden size does not match the configured row bytes");
    }

    const int32_t epoch = NextEpoch();
    const int vectors_per_row = static_cast<int>(row_bytes_ / sizeof(uint4));
    const int required_send_blocks = (send_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int send_blocks =
        required_send_blocks < kMaxExchangeBlocks ? required_send_blocks : kMaxExchangeBlocks;
    SendRowsKernel<<<send_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(input),
        source_rows,
        destination_ranks,
        destination_slots,
        remote_bases_,
        data_offset_,
        slot_epoch_offset_,
        source_rank_offset_,
        source_slot_offset_,
        rank_,
        world_size_,
        input_value_rows,
        send_rows,
        output_rows,
        vectors_per_row,
        epoch);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
      return CudaError(status, "SendRowsKernel");
    }

    SignalPeersKernel<<<1, kMaxRanks, 0, stream>>>(
        remote_bases_, done_offset_, rank_, world_size_, epoch);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return CudaError(status, "SignalPeersKernel(done)");
    }
    WaitSignalsKernel<<<1, kMaxRanks, 0, stream>>>(
        reinterpret_cast<const int32_t*>(local_address_ + done_offset_), world_size_, epoch);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return CudaError(status, "WaitSignalsKernel(done)");
    }

    const int required_copy_blocks = (output_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int copy_blocks =
        required_copy_blocks < kMaxExchangeBlocks ? required_copy_blocks : kMaxExchangeBlocks;
    CopyRowsKernel<<<copy_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(local_address_ + data_offset_),
        reinterpret_cast<const int32_t*>(local_address_ + slot_epoch_offset_),
        reinterpret_cast<const int32_t*>(local_address_ + source_rank_offset_),
        reinterpret_cast<const int32_t*>(local_address_ + source_slot_offset_),
        reinterpret_cast<uint4*>(output),
        output_source_ranks,
        output_source_slots,
        world_size_,
        send_rows,
        output_rows,
        vectors_per_row,
        epoch);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return CudaError(status, "CopyRowsKernel");
    }

    SignalPeersKernel<<<1, kMaxRanks, 0, stream>>>(
        remote_bases_, finished_offset_, rank_, world_size_, epoch);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return CudaError(status, "SignalPeersKernel(finished)");
    }
    WaitSignalsKernel<<<1, kMaxRanks, 0, stream>>>(
        reinterpret_cast<const int32_t*>(local_address_ + finished_offset_), world_size_, epoch);
    return CudaError(cudaGetLastError(), "WaitSignalsKernel(finished)");
  }

  void ProbeWrite() {
    if (remote_bases_ == nullptr) {
      throw std::runtime_error("MNNVL peer handles are not synchronized");
    }
    ProbeWriteKernel<<<1, world_size_>>>(remote_bases_, rank_, world_size_);
    ThrowOnCuda(cudaGetLastError(), "ProbeWriteKernel");
    ThrowOnCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(probe write)");
  }

  void ProbeRead(int32_t* output, int count) {
    if (count != world_size_) {
      throw std::runtime_error("MNNVL probe output size does not match the world size");
    }
    ThrowOnCuda(
        cudaMemcpy(
            output,
            reinterpret_cast<void*>(local_address_ + probe_offset_),
            static_cast<size_t>(world_size_) * sizeof(int32_t),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(probe read)");
  }

  void Shutdown() {
    if (remote_bases_ != nullptr) {
      cudaFree(remote_bases_);
      remote_bases_ = nullptr;
    }
    for (int peer = 0; peer < static_cast<int>(remote_addresses_.size()); ++peer) {
      if (peer == rank_ || remote_addresses_[peer] == 0) {
        continue;
      }
      cuMemUnmap(remote_addresses_[peer], allocation_bytes_);
      cuMemAddressFree(remote_addresses_[peer], allocation_bytes_);
      if (imported_allocations_[peer] != 0) {
        cuMemRelease(imported_allocations_[peer]);
      }
    }
    remote_addresses_.clear();
    imported_allocations_.clear();
    if (local_address_ != 0) {
      cuMemUnmap(local_address_, allocation_bytes_);
      cuMemAddressFree(local_address_, allocation_bytes_);
      local_address_ = 0;
    }
    if (local_allocation_ != 0) {
      cuMemRelease(local_allocation_);
      local_allocation_ = 0;
    }
    rank_ = -1;
    world_size_ = 0;
    buffer_rows_ = 0;
    row_bytes_ = 0;
    allocation_bytes_ = 0;
    next_epoch_.store(1, std::memory_order_relaxed);
  }

 private:
  int rank_ = -1;
  int world_size_ = 0;
  int buffer_rows_ = 0;
  CUdevice device_ = 0;
  CUmemAllocationProp allocation_properties_{};
  size_t allocation_granularity_ = 0;
  size_t row_bytes_ = 0;
  size_t allocation_bytes_ = 0;
  size_t probe_offset_ = 0;
  size_t done_offset_ = 0;
  size_t finished_offset_ = 0;
  size_t slot_epoch_offset_ = 0;
  size_t source_rank_offset_ = 0;
  size_t source_slot_offset_ = 0;
  size_t data_offset_ = 0;
  CUmemGenericAllocationHandle local_allocation_ = 0;
  CUdeviceptr local_address_ = 0;
  std::vector<CUmemGenericAllocationHandle> imported_allocations_;
  std::vector<CUdeviceptr> remote_addresses_;
  uint8_t** remote_bases_ = nullptr;
  std::atomic<uint32_t> next_epoch_{1};
};

ffi::Error MnnvlExchange(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> input,
    ffi::Buffer<ffi::S32, 1> destination_ranks,
    ffi::Buffer<ffi::S32, 1> destination_slots,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> output,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> output_source_ranks,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> output_source_slots) {
  const auto input_dims = input.dimensions();
  const auto destination_rank_dims = destination_ranks.dimensions();
  const auto destination_slot_dims = destination_slots.dimensions();
  const auto output_dims = output->dimensions();
  if (input_dims.size() != 2 || output_dims.size() != 2 ||
      destination_rank_dims.size() != 1 || destination_slot_dims.size() != 1) {
    return ffi::Error::InvalidArgument("MNNVL exchange received an invalid buffer rank");
  }
  if (destination_rank_dims[0] != input_dims[0] || destination_slot_dims[0] != input_dims[0]) {
    return ffi::Error::InvalidArgument("MNNVL destination metadata must match the input row count");
  }
  if (input_dims[1] != output_dims[1]) {
    return ffi::Error::InvalidArgument("MNNVL input and output hidden dimensions must match");
  }
  if (output_source_ranks->dimensions()[0] != output_dims[0] ||
      output_source_slots->dimensions()[0] != output_dims[0]) {
    return ffi::Error::InvalidArgument("MNNVL output source metadata must match the output row count");
  }
  if (input_dims[0] > std::numeric_limits<int>::max() ||
      input_dims[1] > std::numeric_limits<int>::max() ||
      output_dims[0] > std::numeric_limits<int>::max()) {
    return ffi::Error::InvalidArgument("MNNVL exchange dimensions exceed int32 kernel limits");
  }
  return FabricRuntime::Instance().Exchange(
      stream,
      reinterpret_cast<const uint16_t*>(input.typed_data()),
      destination_ranks.typed_data(),
      destination_slots.typed_data(),
      reinterpret_cast<uint16_t*>(output->typed_data()),
      output_source_ranks->typed_data(),
      output_source_slots->typed_data(),
      static_cast<int>(input_dims[0]),
      static_cast<int>(input_dims[0]),
      static_cast<int>(output_dims[0]),
      static_cast<int>(input_dims[1]));
}

ffi::Error MnnvlGatherExchange(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16, 2> input,
    ffi::Buffer<ffi::S32, 1> source_rows,
    ffi::Buffer<ffi::S32, 1> destination_ranks,
    ffi::Buffer<ffi::S32, 1> destination_slots,
    ffi::Result<ffi::Buffer<ffi::BF16, 2>> output,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> output_source_ranks,
    ffi::Result<ffi::Buffer<ffi::S32, 1>> output_source_slots) {
  const auto input_dims = input.dimensions();
  const auto source_row_dims = source_rows.dimensions();
  const auto destination_rank_dims = destination_ranks.dimensions();
  const auto destination_slot_dims = destination_slots.dimensions();
  const auto output_dims = output->dimensions();
  if (input_dims.size() != 2 || output_dims.size() != 2 ||
      source_row_dims.size() != 1 || destination_rank_dims.size() != 1 ||
      destination_slot_dims.size() != 1) {
    return ffi::Error::InvalidArgument("MNNVL gather exchange received an invalid buffer rank");
  }
  const int64_t send_rows = source_row_dims[0];
  if (destination_rank_dims[0] != send_rows || destination_slot_dims[0] != send_rows) {
    return ffi::Error::InvalidArgument("MNNVL gather destination metadata must match the source row count");
  }
  if (input_dims[1] != output_dims[1]) {
    return ffi::Error::InvalidArgument("MNNVL gather input and output hidden dimensions must match");
  }
  if (output_source_ranks->dimensions()[0] != output_dims[0] ||
      output_source_slots->dimensions()[0] != output_dims[0]) {
    return ffi::Error::InvalidArgument("MNNVL gather output source metadata must match the output row count");
  }
  if (input_dims[0] > std::numeric_limits<int>::max() ||
      input_dims[1] > std::numeric_limits<int>::max() ||
      send_rows > std::numeric_limits<int>::max() ||
      output_dims[0] > std::numeric_limits<int>::max()) {
    return ffi::Error::InvalidArgument("MNNVL gather exchange dimensions exceed int32 kernel limits");
  }
  return FabricRuntime::Instance().Exchange(
      stream,
      reinterpret_cast<const uint16_t*>(input.typed_data()),
      destination_ranks.typed_data(),
      destination_slots.typed_data(),
      reinterpret_cast<uint16_t*>(output->typed_data()),
      output_source_ranks->typed_data(),
      output_source_slots->typed_data(),
      static_cast<int>(input_dims[0]),
      static_cast<int>(send_rows),
      static_cast<int>(output_dims[0]),
      static_cast<int>(input_dims[1]),
      source_rows.typed_data());
}

auto MnnvlExchangeBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>();
}

auto MnnvlGatherExchangeBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::BF16, 2>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Arg<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::BF16, 2>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>()
      .Ret<ffi::Buffer<ffi::S32, 1>>();
}

}  // namespace

extern "C" int levanter_mnnvl_fabric_handle_size() {
  return static_cast<int>(sizeof(CUmemFabricHandle));
}

extern "C" const char* levanter_mnnvl_last_error() {
  return LastErrorStorage().c_str();
}

extern "C" int levanter_mnnvl_init_local(
    int rank,
    int world_size,
    int64_t buffer_rows,
    int64_t row_bytes,
    uint8_t* exported_handle,
    int handle_size) {
  try {
    FabricRuntime::Instance().InitializeLocal(
        rank, world_size, buffer_rows, row_bytes, exported_handle, handle_size);
    LastErrorStorage().clear();
    return 0;
  } catch (const std::exception& error) {
    FabricRuntime::Instance().Shutdown();
    LastErrorStorage() = error.what();
    return 1;
  }
}

extern "C" int levanter_mnnvl_sync_handles(const uint8_t* handles, int count, int handle_size) {
  try {
    FabricRuntime::Instance().SynchronizeHandles(handles, count, handle_size);
    LastErrorStorage().clear();
    return 0;
  } catch (const std::exception& error) {
    LastErrorStorage() = error.what();
    return 1;
  }
}

extern "C" int levanter_mnnvl_probe_write() {
  try {
    FabricRuntime::Instance().ProbeWrite();
    LastErrorStorage().clear();
    return 0;
  } catch (const std::exception& error) {
    LastErrorStorage() = error.what();
    return 1;
  }
}

extern "C" int levanter_mnnvl_probe_read(int32_t* output, int count) {
  try {
    FabricRuntime::Instance().ProbeRead(output, count);
    LastErrorStorage().clear();
    return 0;
  } catch (const std::exception& error) {
    LastErrorStorage() = error.what();
    return 1;
  }
}

extern "C" void levanter_mnnvl_shutdown() {
  FabricRuntime::Instance().Shutdown();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_mnnvl_exchange,
    MnnvlExchange,
    MnnvlExchangeBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    levanter_mnnvl_gather_exchange,
    MnnvlGatherExchange,
    MnnvlGatherExchangeBinding());
