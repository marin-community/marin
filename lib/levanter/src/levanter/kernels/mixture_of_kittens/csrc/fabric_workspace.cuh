// Copyright The Levanter Authors
// SPDX-License-Identifier: Apache-2.0
//
// Symmetric workspace backed by CUDA VMM fabric handles.
//
// The sealed v15 workspace is plain device memory whose peer pointers are
// published through a process-local registry, so it can only ever address ranks
// the calling process owns. That caps an expert group at one node's four GPUs.
//
// This allocator instead reserves a virtual range per rank and backs it with a
// physical allocation exported as a `CUmemFabricHandle`. Within one NVLink
// domain -- a GB200 NVL72 rack is 16 nodes x 4 GPUs = 64 GPUs -- an imported
// handle is mappable and directly addressable from a peer, including a peer in
// another process on another host. That is what an EP64 group needs.
//
// The handle exchange itself is deliberately NOT done here. Handles are opaque
// 64-byte blobs; the caller gathers them across the expert axis with an ordinary
// JAX collective and hands the gathered bytes back in. Keeping the rendezvous in
// the framework avoids depending on XLA's private collective-context headers,
// which are not shipped in jaxlib's public includes.

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

namespace mok_fabric {

// Opaque exported handle. `CUmemFabricHandle` is 64 bytes; the size is asserted
// below so a driver change cannot silently corrupt the exchange.
constexpr int kFabricHandleBytes = 64;
static_assert(sizeof(CUmemFabricHandle) == kFabricHandleBytes,
              "CUmemFabricHandle is expected to be 64 bytes");

inline void ThrowOnDriver(CUresult status, const char *what) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char *name = nullptr;
  const char *message = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &message);
  std::ostringstream stream;
  stream << what << " failed: " << (name ? name : "unknown") << " ("
         << (message ? message : "no description") << ")";
  throw std::runtime_error(stream.str());
}

// Round `value` up to the next multiple of `multiple`.
inline size_t RoundUp(size_t value, size_t multiple) {
  if (multiple == 0) {
    throw std::runtime_error("allocation granularity must be non-zero");
  }
  return ((value + multiple - 1) / multiple) * multiple;
}

// Allocation property for a fabric-exportable physical allocation on `device`.
inline CUmemAllocationProp FabricAllocationProp(int device) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  return prop;
}

// Minimum granularity a fabric-exportable allocation on `device` must respect.
inline size_t FabricGranularity(int device) {
  const CUmemAllocationProp prop = FabricAllocationProp(device);
  size_t granularity = 0;
  ThrowOnDriver(
      cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
      "cuMemGetAllocationGranularity");
  return granularity;
}

// Report whether `device` can actually export fabric handles.
//
// CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED is NOT sufficient on its own:
// it advertises driver support, not usability. Measured on an RTX 3090 with
// driver 595.71.05, the attribute reports 1 while `cuMemCreate` with
// CU_MEM_HANDLE_TYPE_FABRIC fails CUDA_ERROR_NOT_PERMITTED because no IMEX
// channel is configured. Trusting the attribute would push that failure into the
// first training step instead of surfacing it at setup.
//
// So the attribute is used only as a cheap negative filter, and the positive
// answer comes from a trial allocation of one granularity unit.
inline bool FabricHandlesSupported(int device) {
  int advertised = 0;
  const CUresult status = cuDeviceGetAttribute(
      &advertised, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
  if (status != CUDA_SUCCESS || advertised == 0) {
    // Older drivers do not know the attribute at all.
    return false;
  }

  size_t granularity = 0;
  const CUmemAllocationProp prop = FabricAllocationProp(device);
  if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) !=
          CUDA_SUCCESS ||
      granularity == 0) {
    return false;
  }

  CUmemGenericAllocationHandle probe{};
  if (cuMemCreate(&probe, granularity, &prop, 0) != CUDA_SUCCESS) {
    return false;
  }
  // Exporting is the operation that actually needs the IMEX channel, so probe it
  // too rather than inferring from a successful create.
  CUmemFabricHandle handle{};
  const CUresult exported =
      cuMemExportToShareableHandle(&handle, probe, CU_MEM_HANDLE_TYPE_FABRIC, 0);
  cuMemRelease(probe);
  return exported == CUDA_SUCCESS;
}

// One rank's symmetric segment: a physical allocation plus the virtual range it
// is mapped into. Peers map their own view of the same physical memory.
struct LocalSegment {
  CUmemGenericAllocationHandle allocation{};
  CUdeviceptr base = 0;
  size_t bytes = 0;
  int device = -1;
  bool valid = false;
};

// A peer's imported segment.
struct PeerSegment {
  CUmemGenericAllocationHandle allocation{};
  CUdeviceptr base = 0;
  bool imported = false;
};

// Allocate `bytes` of fabric-exportable memory on `device`, map it, and grant
// this device read/write access to it.
inline LocalSegment CreateLocalSegment(int device, size_t bytes) {
  if (bytes == 0) {
    throw std::runtime_error("symmetric segment size must be positive");
  }
  const size_t granularity = FabricGranularity(device);
  const size_t padded = RoundUp(bytes, granularity);

  LocalSegment segment;
  segment.device = device;
  segment.bytes = padded;

  const CUmemAllocationProp prop = FabricAllocationProp(device);
  ThrowOnDriver(cuMemCreate(&segment.allocation, padded, &prop, 0), "cuMemCreate");

  // Reserve at the same granularity so every rank's range is laid out
  // identically; the kernel indexes peers by rank and offset, never by an
  // address it computed locally.
  ThrowOnDriver(cuMemAddressReserve(&segment.base, padded, granularity, 0, 0),
                "cuMemAddressReserve");
  ThrowOnDriver(cuMemMap(segment.base, padded, 0, segment.allocation, 0), "cuMemMap");

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  ThrowOnDriver(cuMemSetAccess(segment.base, padded, &access, 1), "cuMemSetAccess");

  segment.valid = true;
  return segment;
}

// Export `segment` as an opaque blob for transport through the framework.
inline void ExportSegment(const LocalSegment &segment, unsigned char *out_handle) {
  if (!segment.valid) {
    throw std::runtime_error("cannot export an uninitialized symmetric segment");
  }
  CUmemFabricHandle handle{};
  ThrowOnDriver(
      cuMemExportToShareableHandle(&handle, segment.allocation, CU_MEM_HANDLE_TYPE_FABRIC, 0),
      "cuMemExportToShareableHandle");
  std::memcpy(out_handle, &handle, kFabricHandleBytes);
}

// Import a peer's exported blob and map it into this process's address space,
// granting `device` read/write access. `bytes` must be the padded size the
// exporter used; every rank allocates the same size, so the caller passes its
// own local padded size.
inline PeerSegment ImportPeerSegment(const unsigned char *handle_bytes, int device, size_t bytes) {
  CUmemFabricHandle handle{};
  std::memcpy(&handle, handle_bytes, kFabricHandleBytes);

  PeerSegment peer;
  ThrowOnDriver(cuMemImportFromShareableHandle(&peer.allocation, &handle, CU_MEM_HANDLE_TYPE_FABRIC),
                "cuMemImportFromShareableHandle");

  const size_t granularity = FabricGranularity(device);
  ThrowOnDriver(cuMemAddressReserve(&peer.base, bytes, granularity, 0, 0),
                "cuMemAddressReserve(peer)");
  ThrowOnDriver(cuMemMap(peer.base, bytes, 0, peer.allocation, 0), "cuMemMap(peer)");

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  ThrowOnDriver(cuMemSetAccess(peer.base, bytes, &access, 1), "cuMemSetAccess(peer)");

  peer.imported = true;
  return peer;
}

inline void DestroyPeerSegment(PeerSegment &peer, size_t bytes) {
  if (!peer.imported) {
    return;
  }
  cuMemUnmap(peer.base, bytes);
  cuMemAddressFree(peer.base, bytes);
  cuMemRelease(peer.allocation);
  peer.imported = false;
  peer.base = 0;
}

inline void DestroyLocalSegment(LocalSegment &segment) {
  if (!segment.valid) {
    return;
  }
  cuMemUnmap(segment.base, segment.bytes);
  cuMemAddressFree(segment.base, segment.bytes);
  cuMemRelease(segment.allocation);
  segment.valid = false;
  segment.base = 0;
}

// Byte offsets of the peer-visible buffers inside one rank's symmetric segment.
//
// The v15 runtime allocates these six buffers with separate `cudaMalloc` calls
// and publishes six pointers per rank. Exchanging six fabric handles per rank
// per slot would be needlessly expensive, so instead one arena is allocated per
// (rank, slot) and the buffers live at fixed offsets inside it. Exactly one
// handle crosses the wire per rank, and a peer pointer is `peer_base + offset`.
//
// Every rank computes the identical layout from the identical shape parameters,
// so an offset computed locally is valid against any peer's base.
struct ArenaLayout {
  size_t x = 0;
  size_t combine = 0;
  size_t d_y = 0;
  size_t d_x_routed = 0;
  size_t router_weights = 0;
  size_t d_router_weights = 0;
  size_t generation = 0;
  size_t forward_input_ready = 0;
  size_t backward_input_ready = 0;
  size_t forward_completions = 0;
  size_t backward_completions = 0;
  size_t cancellation = 0;
  size_t total = 0;
};

// 256-byte alignment keeps every sub-buffer TMA-friendly; MoK's dispatch tile is
// 256 rows and the kernel issues aligned bulk copies out of these buffers.
constexpr size_t kArenaAlignment = 256;

inline size_t AlignUp(size_t offset) { return RoundUp(offset, kArenaAlignment); }

inline ArenaLayout ComputeArenaLayout(int num_tokens, int hidden_dim, int top_k, int num_ranks) {
  if (num_tokens <= 0 || hidden_dim <= 0 || top_k <= 0 || num_ranks <= 0) {
    throw std::runtime_error("arena layout requires positive shape parameters");
  }
  const size_t tokens = static_cast<size_t>(num_tokens);
  const size_t hidden = static_cast<size_t>(hidden_dim);
  const size_t k = static_cast<size_t>(top_k);
  const size_t ranks = static_cast<size_t>(num_ranks);

  const size_t x_bytes = tokens * hidden * sizeof(uint16_t);
  const size_t combine_bytes = tokens * k * hidden * sizeof(uint16_t);
  const size_t router_bytes = tokens * k * sizeof(float);
  const size_t flag_bytes = ranks * sizeof(uint64_t);

  ArenaLayout layout;
  size_t offset = 0;
  auto place = [&](size_t &field, size_t bytes) {
    offset = AlignUp(offset);
    field = offset;
    offset += bytes;
  };

  place(layout.x, x_bytes);
  place(layout.combine, combine_bytes);
  place(layout.d_y, x_bytes);
  place(layout.d_x_routed, combine_bytes);
  place(layout.router_weights, router_bytes);
  place(layout.d_router_weights, router_bytes);
  place(layout.generation, sizeof(uint64_t));
  place(layout.forward_input_ready, flag_bytes);
  place(layout.backward_input_ready, flag_bytes);
  place(layout.forward_completions, flag_bytes);
  place(layout.backward_completions, flag_bytes);
  place(layout.cancellation, sizeof(uint64_t));
  layout.total = AlignUp(offset);
  return layout;
}

// The full symmetric workspace for one expert group: this rank's segment plus a
// mapped view of every peer's. `peer_base(rank)` is what the megakernel needs to
// build its `std::array<bf16*, NUM_DEVICES>` peer pointer list.
class SymmetricWorkspace {
 public:
  SymmetricWorkspace() = default;
  SymmetricWorkspace(const SymmetricWorkspace &) = delete;
  SymmetricWorkspace &operator=(const SymmetricWorkspace &) = delete;

  ~SymmetricWorkspace() { Destroy(); }

  // Phase one: allocate and export. Returns this rank's handle bytes.
  void CreateLocal(int rank, int num_ranks, int device, size_t bytes, unsigned char *out_handle) {
    if (rank < 0 || num_ranks <= 0 || rank >= num_ranks) {
      throw std::runtime_error("rank must be in [0, num_ranks)");
    }
    Destroy();
    rank_ = rank;
    num_ranks_ = num_ranks;
    device_ = device;
    local_ = CreateLocalSegment(device, bytes);
    peers_.assign(static_cast<size_t>(num_ranks), PeerSegment{});
    ExportSegment(local_, out_handle);
  }

  // Phase two: import every peer's handle. `handles` is `num_ranks` blobs of
  // `kFabricHandleBytes`, ordered by rank, as gathered by the framework. The
  // local rank's slot is filled from the existing mapping rather than
  // re-imported, since a process cannot import its own fabric handle.
  void ImportPeers(const unsigned char *handles) {
    if (!local_.valid) {
      throw std::runtime_error("ImportPeers requires a created local segment");
    }
    for (int peer = 0; peer < num_ranks_; ++peer) {
      if (peer == rank_) {
        continue;
      }
      const unsigned char *blob = handles + static_cast<size_t>(peer) * kFabricHandleBytes;
      peers_[static_cast<size_t>(peer)] = ImportPeerSegment(blob, device_, local_.bytes);
    }
    ready_ = true;
  }

  // Base address of `peer`'s segment as seen from this device.
  CUdeviceptr peer_base(int peer) const {
    if (!ready_) {
      throw std::runtime_error("symmetric workspace is not ready");
    }
    if (peer < 0 || peer >= num_ranks_) {
      throw std::runtime_error("peer rank out of range");
    }
    if (peer == rank_) {
      return local_.base;
    }
    return peers_[static_cast<size_t>(peer)].base;
  }

  bool ready() const { return ready_; }
  int rank() const { return rank_; }
  int num_ranks() const { return num_ranks_; }
  size_t bytes() const { return local_.bytes; }

  void Destroy() {
    for (auto &peer : peers_) {
      DestroyPeerSegment(peer, local_.bytes);
    }
    peers_.clear();
    DestroyLocalSegment(local_);
    ready_ = false;
    rank_ = -1;
    num_ranks_ = 0;
  }

 private:
  LocalSegment local_{};
  std::vector<PeerSegment> peers_{};
  int rank_ = -1;
  int num_ranks_ = 0;
  int device_ = -1;
  bool ready_ = false;
};

}  // namespace mok_fabric
