// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "TestPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
constexpr llvm::StringLiteral kOperationRef = "shuttle.operation_ref";

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

class NestedExcludedAttributePass
    : public MutationPass<NestedExcludedAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NestedExcludedAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-nested-excluded-attribute";
  }
  StringRef getDescription() const final {
    return "Mutate one excluded operation nested in a source region";
  }
  void runOnOperation() override {
    auto excluded = excludedSources(getOperation());
    WalkResult result = getOperation().walk([&](Operation *operation) {
      if (!isExcluded(operation, excluded) ||
          isa<func::FuncOp>(operation->getParentOp())) {
        return WalkResult::advance();
      }
      operation->setAttr("stablehlo.mutated_nested_test",
                         UnitAttr::get(operation->getContext()));
      return WalkResult::interrupt();
    });
    if (!result.wasInterrupted()) {
      getOperation().emitError("test fixture has no nested excluded operation");
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

template <typename Derived>
class FoldMutationPass : public MutationPass<Derived> {
protected:
  FoldOp fold() {
    FoldOp result;
    this->getOperation().walk([&](FoldOp candidate) {
      if (!result) {
        result = candidate;
      }
    });
    if (!result) {
      this->getOperation().emitError("test fixture has no shuttle.fold");
      this->signalPassFailure();
    }
    return result;
  }
};

class FoldOwnerPass : public FoldMutationPass<FoldOwnerPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldOwnerPass)
  StringRef getArgument() const final {
    return "shuttle-test-remove-fold-owner-ref";
  }
  void runOnOperation() override {
    if (FoldOp operation = fold()) {
      operation->removeAttr(kOperationRef);
    }
  }
};

class FoldOwnerMismatchPass : public FoldMutationPass<FoldOwnerMismatchPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldOwnerMismatchPass)
  StringRef getArgument() const final {
    return "shuttle-test-mismatch-fold-owner-ref";
  }
  void runOnOperation() override {
    if (FoldOp operation = fold()) {
      operation->setAttr(kOperationRef, DenseI64ArrayAttr::get(
                                            operation.getContext(), {9, 9, 9}));
    }
  }
};

class FoldAddSourcePass : public FoldMutationPass<FoldAddSourcePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAddSourcePass)
  StringRef getArgument() const final {
    return "shuttle-test-remove-fold-add-source";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (operation) {
      operation.getCombiner().walk(
          [&](arith::AddFOp add) { add->removeAttr(kSourceRefs); });
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

class FoldAddOwnerMismatchPass
    : public FoldMutationPass<FoldAddOwnerMismatchPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAddOwnerMismatchPass)
  StringRef getArgument() const final {
    return "shuttle-test-mismatch-fold-add-owner-ref";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (operation) {
      operation.getCombiner().walk([&](arith::AddFOp add) {
        add->setAttr(kOperationRef,
                     DenseI64ArrayAttr::get(add.getContext(), {8, 8, 8}));
      });
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

class FoldAddOwnerMissingPass
    : public FoldMutationPass<FoldAddOwnerMissingPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAddOwnerMissingPass)
  StringRef getArgument() const final {
    return "shuttle-test-remove-fold-add-owner-ref";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (operation) {
      operation.getCombiner().walk(
          [&](arith::AddFOp add) { add->removeAttr(kOperationRef); });
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

class FoldAddDuplicateSourcePass
    : public FoldMutationPass<FoldAddDuplicateSourcePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAddDuplicateSourcePass)
  StringRef getArgument() const final {
    return "shuttle-test-duplicate-fold-add-source";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (operation) {
      operation.getCombiner().walk([&](arith::AddFOp add) {
        add->setAttr(kSourceRefs,
                     ArrayAttr::get(add.getContext(), {operation.getSource()}));
      });
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

class FoldYieldOwnerPass : public FoldMutationPass<FoldYieldOwnerPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldYieldOwnerPass)
  StringRef getArgument() const final {
    return "shuttle-test-remove-fold-yield-ref";
  }
  void runOnOperation() override {
    if (FoldOp operation = fold()) {
      operation.getCombiner().front().getTerminator()->removeAttr(
          kOperationRef);
    }
  }
};

class FoldOwnerDuplicatePass : public FoldMutationPass<FoldOwnerDuplicatePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldOwnerDuplicatePass)
  StringRef getArgument() const final {
    return "shuttle-test-duplicate-fold-owner-ref";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (!operation) {
      return;
    }
    Attribute owner = operation->getAttr(kOperationRef);
    operation.getCombiner().walk(
        [&](arith::AddFOp add) { add->setAttr(kOperationRef, owner); });
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

class FoldYieldRewirePass : public FoldMutationPass<FoldYieldRewirePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldYieldRewirePass)
  StringRef getArgument() const final {
    return "shuttle-test-rewire-fold-yield";
  }
  void runOnOperation() override {
    if (FoldOp operation = fold()) {
      auto yield = cast<YieldOp>(operation.getCombiner().front().back());
      yield.getValuesMutable().assign(
          operation.getCombiner().front().getArgument(0));
    }
  }
};

class FoldAddFastMathPass : public FoldMutationPass<FoldAddFastMathPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAddFastMathPass)
  StringRef getArgument() const final {
    return "shuttle-test-add-fold-fastmath";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (operation) {
      operation.getCombiner().walk([&](arith::AddFOp add) {
        add.setFastmathAttr(arith::FastMathFlagsAttr::get(
            add.getContext(), arith::FastMathFlags::fast));
      });
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};

class FoldYieldAttributePass : public FoldMutationPass<FoldYieldAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldYieldAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-fold-yield-attribute";
  }
  void runOnOperation() override {
    FoldOp operation = fold();
    if (operation) {
      operation.getCombiner().front().getTerminator()->setAttr(
          "shuttle.test_semantic",
          IntegerAttr::get(IntegerType::get(operation.getContext(), 64), 7));
    }
  }
};

class FoldAttributePass : public FoldMutationPass<FoldAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-fold-attribute";
  }
  void runOnOperation() override {
    if (FoldOp operation = fold()) {
      operation->setAttr(
          "shuttle.test_semantic",
          IntegerAttr::get(IntegerType::get(operation.getContext(), 64), 7));
    }
  }
};

class ManifestVersionPass : public MutationPass<ManifestVersionPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ManifestVersionPass)
  StringRef getArgument() const final {
    return "shuttle-test-downgrade-manifest-version";
  }
  void runOnOperation() override {
    auto manifest = getOperation()->getAttrOfType<DictionaryAttr>(kManifest);
    if (!manifest) {
      getOperation().emitError("test fixture has no coverage manifest");
      signalPassFailure();
      return;
    }
    NamedAttrList fields(manifest);
    fields.set(
        "version",
        IntegerAttr::get(IntegerType::get(getOperation().getContext(), 64), 1));
    getOperation()->setAttr(
        kManifest, DictionaryAttr::get(getOperation().getContext(), fields));
  }
};

class ManifestVersionMissingPass
    : public MutationPass<ManifestVersionMissingPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ManifestVersionMissingPass)
  StringRef getArgument() const final {
    return "shuttle-test-remove-manifest-version";
  }
  void runOnOperation() override {
    auto manifest = getOperation()->getAttrOfType<DictionaryAttr>(kManifest);
    if (!manifest) {
      getOperation().emitError("test fixture has no coverage manifest");
      signalPassFailure();
      return;
    }
    NamedAttrList fields(manifest);
    fields.erase("version");
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
  PassRegistration<NestedExcludedAttributePass>();
  PassRegistration<ExcludedOperandPass>();
  PassRegistration<UnsupportedAbsorptionPass>();
  PassRegistration<ReturnRewirePass>();
  PassRegistration<ManifestDigestPass>();
  PassRegistration<FoldOwnerPass>();
  PassRegistration<FoldOwnerMismatchPass>();
  PassRegistration<FoldAddSourcePass>();
  PassRegistration<FoldAddOwnerMismatchPass>();
  PassRegistration<FoldAddOwnerMissingPass>();
  PassRegistration<FoldAddDuplicateSourcePass>();
  PassRegistration<FoldYieldOwnerPass>();
  PassRegistration<FoldOwnerDuplicatePass>();
  PassRegistration<FoldYieldRewirePass>();
  PassRegistration<FoldAddFastMathPass>();
  PassRegistration<FoldYieldAttributePass>();
  PassRegistration<FoldAttributePass>();
  PassRegistration<ManifestVersionPass>();
  PassRegistration<ManifestVersionMissingPass>();
  PassRegistration<ReportNormalizedFingerprintPass>();
}

} // namespace mlir::shuttle::test
