// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_TRANSFORMS_PASSES_H_
#define SHUTTLE_TRANSFORMS_PASSES_H_

#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/Transforms/Observer.h"

namespace mlir::shuttle {

#define GEN_PASS_DECL
#include "shuttle/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createAnnotateSourcePass();
std::unique_ptr<Pass> createFormStructuralRegionsPass();
std::unique_ptr<Pass>
createFormStructuralRegionsPass(NumericalPolicy numerics,
                                std::string canonicalOptions,
                                std::string canonicalTuning);
std::unique_ptr<Pass> createConvertStablehloToAlgebraPass();
std::unique_ptr<Pass> createVerifySourceCoveragePass();
std::unique_ptr<Pass> createVerifySemanticErasurePass();
std::unique_ptr<Pass> createShuttleCanonicalizePass();
std::unique_ptr<Pass> createLowerAlgebraToStablehloPass();
std::unique_ptr<Pass> createStripSourceProvenancePass();
std::unique_ptr<Pass> createVerifyNoShuttleOpsPass();
std::unique_ptr<Pass> createPlanRowFoldMaterializationPass();
std::unique_ptr<Pass> createVerifyMaterializationPlanPass();
std::unique_ptr<Pass> createPlanSimt32RowFoldSchedulePass();
std::unique_ptr<Pass> createVerifySimt32RowFoldSchedulePass();
std::unique_ptr<Pass> createBuildCpuExecutableBundlePass();
std::unique_ptr<Pass> createVerifyCpuExecutableBundlePass();

std::string normalizedStablehloFingerprint(ModuleOp module);

void buildShuttleStablehloPipeline(
    OpPassManager &manager, const ShuttlePipelineOptions &options,
    std::shared_ptr<const ShuttlePipelineObserver> observer = {});
void registerShuttleStablehloPipelines();

#define GEN_PASS_REGISTRATION
#include "shuttle/Transforms/Passes.h.inc"

} // namespace mlir::shuttle

#endif // SHUTTLE_TRANSFORMS_PASSES_H_
