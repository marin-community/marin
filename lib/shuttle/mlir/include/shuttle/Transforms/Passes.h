// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_TRANSFORMS_PASSES_H_
#define SHUTTLE_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"
#include "shuttle/IR/ShuttleDialect.h"

namespace mlir::shuttle {

#define GEN_PASS_DECL
#include "shuttle/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createAnnotateSourcePass();
std::unique_ptr<Pass> createFormStructuralRegionsPass();
std::unique_ptr<Pass> createConvertStablehloToAlgebraPass();
std::unique_ptr<Pass> createVerifySourceCoveragePass();
std::unique_ptr<Pass> createVerifySemanticErasurePass();
std::unique_ptr<Pass> createShuttleCanonicalizePass();
std::unique_ptr<Pass> createLowerAlgebraToStablehloPass();
std::unique_ptr<Pass> createVerifyNoShuttleOpsPass();

#define GEN_PASS_REGISTRATION
#include "shuttle/Transforms/Passes.h.inc"

} // namespace mlir::shuttle

#endif // SHUTTLE_TRANSFORMS_PASSES_H_
