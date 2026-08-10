// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef SHUTTLE_TRANSFORMS_PASSES_H_
#define SHUTTLE_TRANSFORMS_PASSES_H_

#include <cstdint>
#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/IR/ShuttleDialect.h"

namespace mlir::shuttle {

enum class ShuttlePipelinePhase {
  AlgebraCoverage,
  LoweredCoverage,
  FinalErasure,
  Failure,
};

struct ShuttlePipelineEvent {
  uint64_t invocationId;
  ShuttlePipelinePhase phase;
  std::string policyDigest;
  std::string tuningDigest;
  std::string regionMembership;
  std::string coverageManifest;
  std::string unsupportedFingerprint;
  std::string normalizedModuleFingerprint;
  bool noShuttleSemantics;
};

class ShuttlePipelineObserver {
public:
  virtual ~ShuttlePipelineObserver() = default;
  virtual void observe(const ShuttlePipelineEvent &event) const = 0;
};

#define GEN_PASS_DECL
#include "shuttle/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createAnnotateSourcePass();
std::unique_ptr<Pass> createFormStructuralRegionsPass();
std::unique_ptr<Pass> createFormStructuralRegionsPass(NumericalPolicy numerics);
std::unique_ptr<Pass>
createFormStructuralRegionsPass(NumericalPolicy numerics,
                                std::string canonicalTuning);
std::unique_ptr<Pass> createConvertStablehloToAlgebraPass();
std::unique_ptr<Pass> createVerifySourceCoveragePass();
std::unique_ptr<Pass> createVerifySemanticErasurePass();
std::unique_ptr<Pass> createShuttleCanonicalizePass();
std::unique_ptr<Pass> createLowerAlgebraToStablehloPass();
std::unique_ptr<Pass> createStripSourceProvenancePass();
std::unique_ptr<Pass> createVerifyNoShuttleOpsPass();

std::string normalizedStablehloFingerprint(ModuleOp module);

struct ShuttlePipelineOptions {
  NumericalPolicy numerics = NumericalPolicy::SourceOrdered;
  std::string canonicalTuning = "{}";
};

void buildShuttleStablehloPipeline(
    OpPassManager &manager, const ShuttlePipelineOptions &options,
    std::shared_ptr<const ShuttlePipelineObserver> observer = {});
void registerShuttleStablehloPipelines();

#define GEN_PASS_REGISTRATION
#include "shuttle/Transforms/Passes.h.inc"

} // namespace mlir::shuttle

#endif // SHUTTLE_TRANSFORMS_PASSES_H_
