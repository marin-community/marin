// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_RUNTIME_CPUBYTECODE_H_
#define SHUTTLE_RUNTIME_CPUBYTECODE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::shuttle {

struct CpuExternalBuffer {
  int64_t ordinal;
  llvm::MutableArrayRef<uint8_t> bytes;
};

inline constexpr char kCpuExecutableBundleFfiTarget[] =
    "shuttle.cpu.executable_bundle.v1";
// Canonical transport v1 is deliberately bounded before immutable state or
// per-invocation temporary storage is allocated.
inline constexpr uint64_t kMaximumCpuExecutableRecords = 256;
inline constexpr uint64_t kMaximumCpuTaskElements = 256;
inline constexpr uint64_t kMaximumCpuSlotBytes = 1024 * 1024;
inline constexpr uint64_t kMaximumCpuTemporaryBytes = 16 * 1024 * 1024;

struct CpuExternalBinding {
  ExecutableBindingKind kind;
  int64_t index;
  int64_t slotOrdinal;
  Type tensorType;
  int64_t requiredBytes;
  int64_t alignment;
};

class CpuExecutable {
public:
  static absl::StatusOr<std::shared_ptr<const CpuExecutable>>
  Load(llvm::ArrayRef<uint8_t> bytes);

  absl::Status Execute(llvm::ArrayRef<CpuExternalBuffer> externalBuffers) const;
  llvm::ArrayRef<CpuExternalBinding> externalBindings() const;

private:
  class Impl;
  explicit CpuExecutable(std::shared_ptr<const Impl> implementation);
  std::shared_ptr<const Impl> implementation;
};

FailureOr<llvm::SmallVector<uint8_t>>
serializeCpuExecutableBundle(ModuleOp module);
std::string cpuExecutableBundleDigest(llvm::ArrayRef<uint8_t> bytes);

// Converts one IEEE-754 binary32 bit pattern to BF16 with round-to-nearest,
// ties-to-even semantics while preserving NaNs as NaNs.
uint16_t roundF32ToBf16Rne(uint32_t bits);

// Executes inline CPU bytecode synchronously. The call retains all temporary
// storage and code views until every entry completes and returns only after
// output writes are visible to the caller.
LogicalResult
executeCpuExecutableBundle(ModuleOp module,
                           llvm::ArrayRef<CpuExternalBuffer> externalBuffers);

} // namespace mlir::shuttle

#endif // SHUTTLE_RUNTIME_CPUBYTECODE_H_
