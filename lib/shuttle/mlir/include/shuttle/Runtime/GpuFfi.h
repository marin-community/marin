// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_RUNTIME_GPUFFI_H_
#define SHUTTLE_RUNTIME_GPUFFI_H_

#include "xla/ffi/api/c_api.h"

namespace mlir::shuttle {

XLA_FFI_Handler_Bundle gpuExecutableBundleFfiHandlerBundle();

} // namespace mlir::shuttle

#endif // SHUTTLE_RUNTIME_GPUFFI_H_
