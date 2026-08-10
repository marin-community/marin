// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "TestPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/IR/ShuttleOps.h"
#include "shuttle/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::shuttle::test {
namespace {

constexpr llvm::StringLiteral kManifest = "shuttle.coverage_manifest";
constexpr llvm::StringLiteral kSourceRefs = "shuttle.source_refs";

llvm::SmallDenseSet<Attribute> excludedSources(ModuleOp module) {
  llvm::SmallDenseSet<Attribute> sources;
  auto manifest = module->getAttrOfType<DictionaryAttr>(kManifest);
  auto excluded =
      manifest ? manifest.getAs<ArrayAttr>("excluded") : ArrayAttr{};
  if (!excluded) {
    return sources;
  }
  for (Attribute attribute : excluded) {
    if (auto record = dyn_cast<DictionaryAttr>(attribute)) {
      if (Attribute source = record.get("source")) {
        sources.insert(source);
      }
    }
  }
  return sources;
}

bool isExcluded(Operation *operation,
                const llvm::SmallDenseSet<Attribute> &excluded) {
  auto refs = operation->getAttrOfType<ArrayAttr>(kSourceRefs);
  return refs && llvm::any_of(refs, [&](Attribute ref) {
           return excluded.contains(ref);
         });
}

template <typename Derived>
class MutationPass : public PassWrapper<Derived, OperationPass<ModuleOp>> {};

class ExcludedKindPass : public MutationPass<ExcludedKindPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExcludedKindPass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-excluded-kind";
  }
  StringRef getDescription() const final {
    return "Mutate one excluded binary operation kind";
  }
  void runOnOperation() override {
    auto excluded = excludedSources(getOperation());
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      for (Operation &operation :
           llvm::make_early_inc_range(function.getBody().front())) {
        if (!isExcluded(&operation, excluded) ||
            operation.getNumOperands() != 2 || operation.getNumResults() != 1 ||
            operation.getOperand(0).getType() !=
                operation.getOperand(1).getType()) {
          continue;
        }
        OpBuilder builder(&operation);
        OperationState state(operation.getLoc(),
                             stablehlo::AddOp::getOperationName());
        state.addOperands(operation.getOperands());
        state.addTypes(operation.getResultTypes());
        state.addAttribute(kSourceRefs, operation.getAttr(kSourceRefs));
        Operation *replacement = builder.create(state);
        operation.replaceAllUsesWith(replacement);
        operation.erase();
        return;
      }
    }
    getOperation().emitError("test fixture has no excluded binary operation");
    this->signalPassFailure();
  }
};

class ExcludedAttributePass : public MutationPass<ExcludedAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExcludedAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-excluded-attribute";
  }
  StringRef getDescription() const final {
    return "Mutate one excluded operation attribute";
  }
  void runOnOperation() override {
    auto excluded = excludedSources(getOperation());
    WalkResult result = getOperation().walk([&](Operation *operation) {
      if (!isExcluded(operation, excluded)) {
        return WalkResult::advance();
      }
      operation->setAttr("stablehlo.mutated_test",
                         UnitAttr::get(operation->getContext()));
      return WalkResult::interrupt();
    });
    if (!result.wasInterrupted()) {
      getOperation().emitError("test fixture has no excluded operation");
      this->signalPassFailure();
    }
  }
};

class ExcludedOperandPass : public MutationPass<ExcludedOperandPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExcludedOperandPass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-excluded-operand";
  }
  StringRef getDescription() const final {
    return "Rewire one excluded operation operand";
  }
  void runOnOperation() override {
    auto excluded = excludedSources(getOperation());
    WalkResult result = getOperation().walk([&](Operation *operation) {
      if (!isExcluded(operation, excluded) || operation->getNumOperands() < 2 ||
          operation->getOperand(0).getType() !=
              operation->getOperand(1).getType()) {
        return WalkResult::advance();
      }
      Value first = operation->getOperand(0);
      operation->setOperand(0, operation->getOperand(1));
      operation->setOperand(1, first);
      return WalkResult::interrupt();
    });
    if (!result.wasInterrupted()) {
      getOperation().emitError(
          "test fixture has no excluded same-type binary operation");
      this->signalPassFailure();
    }
  }
};

class UnsupportedAbsorptionPass
    : public MutationPass<UnsupportedAbsorptionPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnsupportedAbsorptionPass)
  StringRef getArgument() const final {
    return "shuttle-test-absorb-excluded-source";
  }
  StringRef getDescription() const final {
    return "Add one excluded source to a selected region";
  }
  void runOnOperation() override {
    auto excluded = excludedSources(getOperation());
    RegionOp region;
    getOperation().walk([&](RegionOp candidate) {
      if (!region) {
        region = candidate;
      }
    });
    if (!region || excluded.empty()) {
      getOperation().emitError("test fixture has no region or excluded source");
      this->signalPassFailure();
      return;
    }
    SmallVector<Attribute> refs(region.getSourceRefs().begin(),
                                region.getSourceRefs().end());
    refs.push_back(*excluded.begin());
    region->setAttr("source_refs",
                    ArrayAttr::get(getOperation().getContext(), refs));
  }
};

class ReturnRewirePass : public MutationPass<ReturnRewirePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReturnRewirePass)
  StringRef getArgument() const final { return "shuttle-test-rewire-return"; }
  StringRef getDescription() const final {
    return "Rewire one return to a same-type function argument";
  }
  void runOnOperation() override {
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      auto returnOp =
          dyn_cast<func::ReturnOp>(function.getBody().front().back());
      if (!returnOp) {
        continue;
      }
      for (OpOperand &returned : returnOp->getOpOperands()) {
        for (BlockArgument argument : function.getArguments()) {
          if (returned.get().getType() == argument.getType() &&
              returned.get() != argument) {
            returned.set(argument);
            return;
          }
        }
      }
    }
    getOperation().emitError(
        "test fixture has no same-type return alternative");
    this->signalPassFailure();
  }
};

class ManifestDigestPass : public MutationPass<ManifestDigestPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ManifestDigestPass)
  StringRef getArgument() const final {
    return "shuttle-test-corrupt-policy-digest";
  }
  StringRef getDescription() const final {
    return "Corrupt the policy digest in a coverage manifest";
  }
  void runOnOperation() override {
    auto manifest = getOperation()->getAttrOfType<DictionaryAttr>(kManifest);
    if (!manifest) {
      getOperation().emitError("test fixture has no coverage manifest");
      this->signalPassFailure();
      return;
    }
    NamedAttrList fields(manifest);
    fields.set("policy_digest",
               StringAttr::get(getOperation().getContext(), "corrupted"));
    getOperation()->setAttr(
        kManifest, DictionaryAttr::get(getOperation().getContext(), fields));
  }
};

class ReportNormalizedFingerprintPass
    : public MutationPass<ReportNormalizedFingerprintPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReportNormalizedFingerprintPass)
  StringRef getArgument() const final {
    return "shuttle-test-report-normalized-fingerprint";
  }
  StringRef getDescription() const final {
    return "Report the normalized structural StableHLO fingerprint";
  }
  void runOnOperation() override {
    llvm::outs() << normalizedStablehloFingerprint(getOperation()) << '\n';
  }
};

} // namespace

void registerMutationPasses() {
  PassRegistration<ExcludedKindPass>();
  PassRegistration<ExcludedAttributePass>();
  PassRegistration<ExcludedOperandPass>();
  PassRegistration<UnsupportedAbsorptionPass>();
  PassRegistration<ReturnRewirePass>();
  PassRegistration<ManifestDigestPass>();
  PassRegistration<ReportNormalizedFingerprintPass>();
}

} // namespace mlir::shuttle::test
