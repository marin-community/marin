// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
//
// Does a TMA bulk copy work when its source is an imported CUDA VMM fabric mapping?
//
// The mok_like fabric transport reaches EP64 by exporting each rank's arena as a
// CU_MEM_HANDLE_TYPE_FABRIC handle and importing its peers'. Ordinary loads and stores reach
// those mappings (the runtime probes them at import), but the megakernel's dispatch reads peers
// through `tma::load_async`, i.e. `cp.async.bulk` with a raw global address. Whether that
// instruction accepts an imported fabric mapping decides whether the PyTorch-free transport can
// carry the kernel at all, so it is worth answering on its own rather than inside a training run.
//
// Two devices in one process: device 0 owns and fills a fabric allocation, device 1 imports it and
// copies from it two ways. The plain-load copy is the control -- if it fails, the mapping is bad
// and the TMA result says nothing.

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK_CUDA(expr)                                                                      \
  do {                                                                                        \
    const cudaError_t status = (expr);                                                        \
    if (status != cudaSuccess) {                                                              \
      std::printf("FAIL %s: %s\n", #expr, cudaGetErrorString(status));                        \
      return 2;                                                                               \
    }                                                                                         \
  } while (0)

#define CHECK_DRV(expr)                                                                       \
  do {                                                                                        \
    const CUresult status = (expr);                                                           \
    if (status != CUDA_SUCCESS) {                                                             \
      const char* name = nullptr;                                                             \
      cuGetErrorName(status, &name);                                                          \
      std::printf("FAIL %s: %s\n", #expr, name ? name : "unknown");                           \
      return 2;                                                                               \
    }                                                                                         \
  } while (0)

constexpr size_t kCopyBytes = 4096;

// Device-side atomics on the peer mapping. The megakernel signals readiness and completion with
// atomics on peer flags, and the in-process path validates cudaDevP2PAttrNativeAtomicSupported
// between every device pair before its first launch. Nothing establishes the equivalent for an
// imported fabric mapping, and atomics are a distinct capability from the loads and bulk copies
// tested above.
__global__ void AtomicOnPeer(unsigned long long* target) {
  atomicAdd(target, 1ull);
}

// Release/acquire across the mapping, which is how the readiness flags are actually consumed.
__global__ void ReleaseAcquireOnPeer(unsigned long long* flag, unsigned long long* observed) {
  if (threadIdx.x == 0) {
    asm volatile("st.release.sys.global.u64 [%0], %1;\n" ::"l"(flag), "l"(7ull) : "memory");
    unsigned long long value = 0;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];\n" : "=l"(value) : "l"(flag) : "memory");
    *observed = value;
  }
}

// Plain vectorized loads from the peer mapping: the control path.
__global__ void CopyWithLoads(const uint4* __restrict__ src, uint4* __restrict__ dst, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    dst[index] = src[index];
  }
}

// The path under test: one `cp.async.bulk` from global into shared, then out to `dst`.
__global__ void CopyWithTma(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int bytes) {
  extern __shared__ __align__(128) uint8_t staging[];
  __shared__ __align__(8) uint64_t barrier;

  const uint32_t staging_addr = static_cast<uint32_t>(__cvta_generic_to_shared(staging));
  const uint32_t barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));

  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(barrier_addr) : "memory");
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(barrier_addr),
                 "r"(bytes)
                 : "memory");
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
            staging_addr),
        "l"(src), "r"(bytes), "r"(barrier_addr)
        : "memory");
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    uint32_t ready = 0;
    while (ready == 0) {
      asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n"
                   " selp.b32 %0, 1, 0, p;\n}\n"
                   : "=r"(ready)
                   : "r"(barrier_addr)
                   : "memory");
    }
  }
  __syncthreads();

  for (int offset = threadIdx.x; offset < bytes; offset += blockDim.x) {
    dst[offset] = staging[offset];
  }
}

int main() {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    std::printf("SKIP: need two devices, found %d\n", device_count);
    return 1;
  }
  CHECK_DRV(cuInit(0));

  // Device 0 allocates a fabric-exportable segment and fills it with a known pattern.
  CHECK_CUDA(cudaSetDevice(0));
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t granularity = 0;
  CHECK_DRV(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  const size_t bytes = ((kCopyBytes + granularity - 1) / granularity) * granularity;

  CUmemGenericAllocationHandle allocation{};
  CHECK_DRV(cuMemCreate(&allocation, bytes, &prop, 0));

  CUdeviceptr owner_base = 0;
  CHECK_DRV(cuMemAddressReserve(&owner_base, bytes, granularity, 0, 0));
  CHECK_DRV(cuMemMap(owner_base, bytes, 0, allocation, 0));
  CUmemAccessDesc owner_access{};
  owner_access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  owner_access.location.id = 0;
  owner_access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK_DRV(cuMemSetAccess(owner_base, bytes, &owner_access, 1));

  std::vector<uint8_t> pattern(kCopyBytes);
  for (size_t i = 0; i < kCopyBytes; ++i) {
    pattern[i] = static_cast<uint8_t>(i * 7 + 11);
  }
  CHECK_CUDA(cudaMemcpy(reinterpret_cast<void*>(owner_base), pattern.data(), kCopyBytes,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaDeviceSynchronize());

  CUmemFabricHandle fabric{};
  CHECK_DRV(cuMemExportToShareableHandle(&fabric, allocation, CU_MEM_HANDLE_TYPE_FABRIC, 0));
  std::printf("exported a fabric handle over %zu bytes (granularity %zu)\n", bytes, granularity);

  // Device 1 imports it, exactly as a peer rank would.
  CHECK_CUDA(cudaSetDevice(1));
  CUmemGenericAllocationHandle imported{};
  CHECK_DRV(cuMemImportFromShareableHandle(&imported, &fabric, CU_MEM_HANDLE_TYPE_FABRIC));

  CUdeviceptr peer_base = 0;
  CHECK_DRV(cuMemAddressReserve(&peer_base, bytes, granularity, 0, 0));
  CHECK_DRV(cuMemMap(peer_base, bytes, 0, imported, 0));
  CUmemAccessDesc peer_access{};
  peer_access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  peer_access.location.id = 1;
  peer_access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK_DRV(cuMemSetAccess(peer_base, bytes, &peer_access, 1));
  std::printf("imported the handle on device 1\n");

  uint8_t* destination = nullptr;
  CHECK_CUDA(cudaMalloc(&destination, kCopyBytes));
  std::vector<uint8_t> readback(kCopyBytes);

  // Control: ordinary loads from the imported mapping.
  CHECK_CUDA(cudaMemset(destination, 0, kCopyBytes));
  CopyWithLoads<<<(kCopyBytes / sizeof(uint4) + 127) / 128, 128>>>(
      reinterpret_cast<const uint4*>(peer_base), reinterpret_cast<uint4*>(destination),
      static_cast<int>(kCopyBytes / sizeof(uint4)));
  const cudaError_t loads_status = cudaDeviceSynchronize();
  if (loads_status != cudaSuccess) {
    std::printf("RESULT loads=FAIL(%s) -- the mapping itself is unusable; TMA result is moot\n",
                cudaGetErrorString(loads_status));
    return 3;
  }
  CHECK_CUDA(cudaMemcpy(readback.data(), destination, kCopyBytes, cudaMemcpyDeviceToHost));
  const bool loads_match = std::memcmp(readback.data(), pattern.data(), kCopyBytes) == 0;
  std::printf("RESULT loads=%s\n", loads_match ? "PASS" : "MISMATCH");

  // Under test: a TMA bulk copy whose source is the imported mapping.
  CHECK_CUDA(cudaMemset(destination, 0, kCopyBytes));
  CopyWithTma<<<1, 256, kCopyBytes>>>(reinterpret_cast<const uint8_t*>(peer_base), destination,
                                      static_cast<int>(kCopyBytes));
  const cudaError_t tma_status = cudaDeviceSynchronize();
  if (tma_status != cudaSuccess) {
    std::printf("RESULT tma=FAIL(%s)\n", cudaGetErrorString(tma_status));
    std::printf("VERDICT: cp.async.bulk cannot source from an imported fabric mapping\n");
    return 4;
  }
  CHECK_CUDA(cudaMemcpy(readback.data(), destination, kCopyBytes, cudaMemcpyDeviceToHost));
  const bool tma_match = std::memcmp(readback.data(), pattern.data(), kCopyBytes) == 0;
  std::printf("RESULT tma=%s\n", tma_match ? "PASS" : "MISMATCH");

  // Atomics on the peer mapping.
  auto* peer_counter = reinterpret_cast<unsigned long long*>(peer_base);
  CHECK_CUDA(cudaMemset(peer_counter, 0, sizeof(unsigned long long)));
  CHECK_CUDA(cudaDeviceSynchronize());
  AtomicOnPeer<<<1, 128>>>(peer_counter);
  const cudaError_t atomic_status = cudaDeviceSynchronize();
  if (atomic_status != cudaSuccess) {
    std::printf("RESULT atomics=FAIL(%s)\n", cudaGetErrorString(atomic_status));
    std::printf("VERDICT: imported fabric mappings do not support device-side atomics\n");
    return 6;
  }
  unsigned long long counter = 0;
  CHECK_CUDA(cudaMemcpy(&counter, peer_counter, sizeof(counter), cudaMemcpyDeviceToHost));
  const bool atomics_match = counter == 128ull;
  std::printf("RESULT atomics=%s (counter=%llu, expected 128)\n",
              atomics_match ? "PASS" : "MISMATCH", counter);

  // Release/acquire, the ordering the readiness protocol depends on.
  unsigned long long* observed = nullptr;
  CHECK_CUDA(cudaMalloc(&observed, sizeof(unsigned long long)));
  CHECK_CUDA(cudaMemset(observed, 0, sizeof(unsigned long long)));
  ReleaseAcquireOnPeer<<<1, 32>>>(peer_counter, observed);
  const cudaError_t ordering_status = cudaDeviceSynchronize();
  if (ordering_status != cudaSuccess) {
    std::printf("RESULT release_acquire=FAIL(%s)\n", cudaGetErrorString(ordering_status));
    std::printf("VERDICT: imported fabric mappings do not support system-scope ordering\n");
    return 7;
  }
  unsigned long long seen = 0;
  CHECK_CUDA(cudaMemcpy(&seen, observed, sizeof(seen), cudaMemcpyDeviceToHost));
  std::printf("RESULT release_acquire=%s (observed=%llu, expected 7)\n",
              seen == 7ull ? "PASS" : "MISMATCH", seen);
  std::printf("VERDICT: %s\n", (loads_match && tma_match)
                                   ? "fabric mappings carry both plain loads and TMA"
                                   : "fabric mappings do not carry TMA correctly");
  return (loads_match && tma_match) ? 0 : 5;
}
