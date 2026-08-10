// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "llvm/Support/raw_ostream.h"

namespace {

class RecordingObserver final : public mlir::shuttle::ShuttlePipelineObserver {
public:
  void observe(const mlir::shuttle::ShuttlePipelineEvent &event) const final {
    events.push_back(event);
  }

  mutable std::vector<mlir::shuttle::ShuttlePipelineEvent> events;
};

std::vector<mlir::shuttle::ShuttlePipelineEvent>
runPipeline(llvm::StringRef source,
            const mlir::shuttle::ShuttlePipelineOptions &options) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::shuttle::ShuttleDialect>();
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "failed to parse observer fixture\n";
    return {};
  }
  auto observer = std::make_shared<RecordingObserver>();
  mlir::PassManager manager(&context);
  mlir::shuttle::buildShuttleStablehloPipeline(manager, options, observer);
  if (mlir::failed(manager.run(*module))) {
    llvm::errs() << "observer fixture pipeline failed\n";
    return {};
  }
  return observer->events;
}

bool checkEvents(
    const std::vector<mlir::shuttle::ShuttlePipelineEvent> &events) {
  using mlir::shuttle::ShuttlePipelinePhase;
  if (events.size() != 3 ||
      events[0].phase != ShuttlePipelinePhase::AlgebraCoverage ||
      events[1].phase != ShuttlePipelinePhase::LoweredCoverage ||
      events[2].phase != ShuttlePipelinePhase::FinalErasure) {
    llvm::errs() << "observer phases are incomplete or out of order\n";
    return false;
  }
  if (events[0].invocationId != events[1].invocationId ||
      events[1].invocationId != events[2].invocationId ||
      events[0].coverageManifest.empty() ||
      events[1].coverageManifest.empty() ||
      !events[2].coverageManifest.empty() || !events[2].noShuttleSemantics ||
      events[2].normalizedModuleFingerprint.empty()) {
    llvm::errs() << "observer snapshots violate provenance lifetime\n";
    return false;
  }
  return true;
}

} // namespace

int main() {
  constexpr llvm::StringLiteral kLeft = R"mlir(
module @left {
  func.func @first(%arg0: tensor<7xf32>) -> tensor<7xf32> {
    %0 = stablehlo.tanh %arg0 : tensor<7xf32>
    %1 = stablehlo.negate %0 : tensor<7xf32>
    return %1 : tensor<7xf32>
  }
}
)mlir";
  constexpr llvm::StringLiteral kRenamed = R"mlir(
module @renamed {
  func.func @second(%input: tensor<7xf32>) -> tensor<7xf32> {
    %mapped = stablehlo.tanh %input : tensor<7xf32>
    %result = stablehlo.negate %mapped : tensor<7xf32>
    return %result : tensor<7xf32>
  }
}
)mlir";

  mlir::shuttle::ShuttlePipelineOptions sourceOrdered;
  sourceOrdered.numerics = mlir::shuttle::NumericalPolicy::SourceOrdered;
  auto sourceEvents = runPipeline(kLeft, sourceOrdered);
  auto renamedEvents = runPipeline(kRenamed, sourceOrdered);
  if (!checkEvents(sourceEvents) || !checkEvents(renamedEvents)) {
    return 1;
  }
  if (sourceEvents[0].regionMembership != renamedEvents[0].regionMembership ||
      sourceEvents[0].unsupportedFingerprint !=
          renamedEvents[0].unsupportedFingerprint ||
      sourceEvents[2].normalizedModuleFingerprint !=
          renamedEvents[2].normalizedModuleFingerprint) {
    llvm::errs() << "normalized observer snapshots depend on symbol spelling\n";
    return 1;
  }

  mlir::shuttle::ShuttlePipelineOptions fast;
  fast.numerics = mlir::shuttle::NumericalPolicy::Fast;
  auto fastEvents = runPipeline(kLeft, fast);
  if (!checkEvents(fastEvents) ||
      sourceEvents[0].policyDigest == fastEvents[0].policyDigest ||
      sourceEvents[0].tuningDigest != fastEvents[0].tuningDigest ||
      sourceEvents[2].normalizedModuleFingerprint !=
          fastEvents[2].normalizedModuleFingerprint) {
    llvm::errs() << "FAST and SOURCE_ORDERED policy identity is invalid\n";
    return 1;
  }

  fast.canonicalTuning = R"json({"tile":2})json";
  auto tunedEvents = runPipeline(kLeft, fast);
  if (!checkEvents(tunedEvents) ||
      fastEvents[0].tuningDigest == tunedEvents[0].tuningDigest ||
      fastEvents[0].policyDigest == tunedEvents[0].policyDigest) {
    llvm::errs()
        << "canonical tuning does not affect semantic policy identity\n";
    return 1;
  }
  return 0;
}
