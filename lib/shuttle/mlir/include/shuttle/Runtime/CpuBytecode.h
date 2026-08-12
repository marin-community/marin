// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_RUNTIME_CPUBYTECODE_H_
#define SHUTTLE_RUNTIME_CPUBYTECODE_H_

#include <cstdint>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::shuttle {

struct CpuExternalBuffer {
  int64_t ordinal;
  llvm::MutableArrayRef<uint8_t> bytes;
};

// Executes inline CPU bytecode synchronously. The call retains all temporary
// storage and code views until every entry completes and returns only after
// output writes are visible to the caller.
LogicalResult
executeCpuExecutableBundle(ModuleOp module,
                           llvm::ArrayRef<CpuExternalBuffer> externalBuffers);

} // namespace mlir::shuttle

#endif // SHUTTLE_RUNTIME_CPUBYTECODE_H_
