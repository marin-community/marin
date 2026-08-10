// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_TRANSFORMS_XLAREGISTRATION_H_
#define SHUTTLE_TRANSFORMS_XLAREGISTRATION_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "shuttle/Transforms/Observer.h"

namespace mlir::shuttle {

// Composite transform callback for the XLA registry translation unit. The XLA
// registry owns transactional cloning and canonical option parsing; this
// callback runs the exact shared Shuttle pipeline and native observer path.
LogicalResult runShuttleXlaTransform(ModuleOp module,
                                     const ShuttlePipelineOptions &options);

} // namespace mlir::shuttle

#endif // SHUTTLE_TRANSFORMS_XLAREGISTRATION_H_
