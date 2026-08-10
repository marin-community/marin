// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Transforms/XlaRegistration.h"

#include "mlir/Pass/PassManager.h"
#include "shuttle/Transforms/Passes.h"

namespace mlir::shuttle {

LogicalResult runShuttleXlaTransform(ModuleOp module,
                                     const ShuttlePipelineOptions &options) {
  PassManager manager(module.getContext(), ModuleOp::getOperationName());
  buildShuttleStablehloPipeline(manager, options);
  return manager.run(module);
}

} // namespace mlir::shuttle
