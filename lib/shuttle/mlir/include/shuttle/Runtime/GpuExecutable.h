// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_RUNTIME_GPUEXECUTABLE_H_
#define SHUTTLE_RUNTIME_GPUEXECUTABLE_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::shuttle {

inline constexpr char kGpuExecutableBundleFfiTarget[] =
    "shuttle.gpu.executable_bundle.v1";
inline constexpr uint64_t kMaximumGpuTransportBytes = 16 * 1024 * 1024;
inline constexpr uint64_t kMaximumGpuExecutableRecords = 256;
inline constexpr uint64_t kMaximumGpuCodeBytes = 8 * 1024 * 1024;
inline constexpr uint64_t kMaximumGpuEntryCodeBytes = 512 * 1024;
inline constexpr uint64_t kMaximumGpuSlotBytes = 32 * 1024 * 1024;
inline constexpr uint64_t kMaximumGpuTemporaryBytes = 256 * 1024 * 1024;
inline constexpr uint64_t kMaximumGpuAggregateTaskPositions = 67129347;
inline constexpr uint64_t kMaximumGpuDynamicSharedMemoryBytes = 16 * 1024;

struct GpuLaunch {
  int64_t taskOrdinal;
  int64_t codeOffset;
  int64_t codeLength;
  std::string codeDigest;
  std::array<uint64_t, 3> grid;
  std::array<uint64_t, 3> block;
  uint64_t dynamicSharedMemoryBytes;
  std::vector<int64_t> inputSlots;
  std::vector<int64_t> outputSlots;
  std::vector<int64_t> dependencies;
};

struct GpuSlot {
  int64_t ordinal;
  Type tensorType;
  int64_t requiredBytes;
  llvm::SmallVector<int64_t> byteStrides;
  int64_t byteOffset;
  int64_t alignment;
  ExecutableAddressSpace addressSpace;
  ExecutableAccess access;
  MaterializationStorage storage;
  int64_t aliasGroup;
  int64_t reuseGroup;
  ExecutableBindingKind bindingKind;
  int64_t bindingIndex;
};

struct GpuExternalBinding {
  ExecutableBindingKind kind;
  int64_t index;
  int64_t slotOrdinal;
  Type tensorType;
  int64_t requiredBytes;
  int64_t alignment;
};

class GpuExecutable {
public:
  static absl::StatusOr<std::shared_ptr<const GpuExecutable>>
  Load(llvm::ArrayRef<uint8_t> bytes);

  llvm::ArrayRef<uint8_t> codeBytes() const;
  llvm::ArrayRef<GpuLaunch> launches() const;
  llvm::ArrayRef<GpuSlot> slots() const;
  llvm::ArrayRef<GpuExternalBinding> externalBindings() const;

private:
  class Impl;
  explicit GpuExecutable(std::shared_ptr<const Impl> implementation);
  std::shared_ptr<const Impl> implementation;
};

FailureOr<llvm::SmallVector<uint8_t>>
serializeGpuExecutableBundle(ModuleOp module);
std::string gpuExecutableBundleDigest(llvm::ArrayRef<uint8_t> bytes);

} // namespace mlir::shuttle

#endif // SHUTTLE_RUNTIME_GPUEXECUTABLE_H_
