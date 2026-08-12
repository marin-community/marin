// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "TestPasses.h"

#include <type_traits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
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

template <typename Derived>
class MapMutationPass : public MutationPass<Derived> {
protected:
  MapOp map(MapSemantics semantics) {
    MapOp result;
    this->getOperation().walk([&](MapOp candidate) {
      if (!result && candidate.getSemantics() == semantics) {
        result = candidate;
      }
    });
    if (!result) {
      this->getOperation().emitError(
          "test fixture has no matching shuttle.map");
      this->signalPassFailure();
    }
    return result;
  }

  void replaceInputMap(MapOp operation, AffineMap inputMap) {
    SmallVector<Attribute> maps(operation.getIndexingMaps().begin(),
                                operation.getIndexingMaps().end());
    maps[0] = AffineMapAttr::get(inputMap);
    operation->setAttr("indexing_maps",
                       ArrayAttr::get(operation.getContext(), maps));
  }

  MapOp rowSingletonBroadcast() {
    MapOp result;
    this->getOperation().walk([&](MapOp candidate) {
      if (result || candidate.getSemantics() != MapSemantics::BroadcastInDim ||
          candidate.getInputs().size() != 1) {
        return;
      }
      auto input =
          dyn_cast<RankedTensorType>(candidate.getInputs()[0].getType());
      auto output =
          dyn_cast<RankedTensorType>(candidate.getResult(0).getType());
      if (input && output && input.getRank() == 2 && output.getRank() == 2 &&
          input.getDimSize(0) == 7 && input.getDimSize(1) == 1 &&
          output.getDimSize(0) == 7 && output.getDimSize(1) == 13) {
        result = candidate;
      }
    });
    if (!result) {
      this->getOperation().emitError(
          "test fixture has no 7x1 to 7x13 broadcast Map");
      this->signalPassFailure();
    }
    return result;
  }
};

class BroadcastMappedAxesSwapPass
    : public MapMutationPass<BroadcastMappedAxesSwapPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastMappedAxesSwapPass)
  StringRef getArgument() const final {
    return "shuttle-test-swap-mapped-broadcast-axes";
  }
  void runOnOperation() override {
    MapOp operation = rowSingletonBroadcast();
    if (!operation) {
      return;
    }
    MLIRContext *context = operation.getContext();
    replaceInputMap(operation,
                    AffineMap::get(2, 0,
                                   {getAffineDimExpr(1, context),
                                    getAffineDimExpr(0, context).floorDiv(7)},
                                   context));
  }
};

class BroadcastMappedAxisDuplicatePass
    : public MapMutationPass<BroadcastMappedAxisDuplicatePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastMappedAxisDuplicatePass)
  StringRef getArgument() const final {
    return "shuttle-test-duplicate-mapped-broadcast-axis";
  }
  void runOnOperation() override {
    MapOp operation = rowSingletonBroadcast();
    if (!operation) {
      return;
    }
    MLIRContext *context = operation.getContext();
    replaceInputMap(operation,
                    AffineMap::get(2, 0,
                                   {getAffineDimExpr(0, context),
                                    getAffineDimExpr(0, context).floorDiv(7)},
                                   context));
  }
};

class BroadcastWrongDivisorPass
    : public MapMutationPass<BroadcastWrongDivisorPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastWrongDivisorPass)
  StringRef getArgument() const final {
    return "shuttle-test-set-wrong-broadcast-divisor";
  }
  void runOnOperation() override {
    MapOp operation = rowSingletonBroadcast();
    if (!operation) {
      return;
    }
    MLIRContext *context = operation.getContext();
    replaceInputMap(operation,
                    AffineMap::get(2, 0,
                                   {getAffineDimExpr(0, context),
                                    getAffineDimExpr(1, context).floorDiv(7)},
                                   context));
  }
};

class BroadcastLiteralZeroPass
    : public MapMutationPass<BroadcastLiteralZeroPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastLiteralZeroPass)
  StringRef getArgument() const final {
    return "shuttle-test-set-broadcast-literal-zero";
  }
  void runOnOperation() override {
    MapOp operation = rowSingletonBroadcast();
    if (!operation) {
      return;
    }
    MLIRContext *context = operation.getContext();
    replaceInputMap(operation,
                    AffineMap::get(2, 0,
                                   {getAffineDimExpr(0, context),
                                    getAffineConstantExpr(0, context)},
                                   context));
  }
};

class BroadcastCompositeDividendPass
    : public MapMutationPass<BroadcastCompositeDividendPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastCompositeDividendPass)
  StringRef getArgument() const final {
    return "shuttle-test-set-broadcast-composite-dividend";
  }
  void runOnOperation() override {
    MapOp operation = rowSingletonBroadcast();
    if (!operation) {
      return;
    }
    MLIRContext *context = operation.getContext();
    AffineExpr shiftedDimension =
        getAffineDimExpr(1, context) + getAffineConstantExpr(1, context);
    replaceInputMap(operation, AffineMap::get(2, 0,
                                              {getAffineDimExpr(0, context),
                                               shiftedDimension.floorDiv(13)},
                                              context));
  }
};

class BroadcastWrongResultExtentPass
    : public MapMutationPass<BroadcastWrongResultExtentPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastWrongResultExtentPass)
  StringRef getArgument() const final {
    return "shuttle-test-set-wrong-broadcast-result-extent";
  }
  void runOnOperation() override {
    if (MapOp operation = rowSingletonBroadcast()) {
      operation.getResult(0).setType(RankedTensorType::get(
          {7, 12}, Float32Type::get(operation.getContext())));
    }
  }
};

class BroadcastDirectSingletonPass
    : public MapMutationPass<BroadcastDirectSingletonPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastDirectSingletonPass)
  StringRef getArgument() const final {
    return "shuttle-test-direct-expanded-singleton";
  }
  void runOnOperation() override {
    MapOp operation = rowSingletonBroadcast();
    if (!operation) {
      return;
    }
    MLIRContext *context = operation.getContext();
    replaceInputMap(operation, AffineMap::getMultiDimIdentityMap(2, context));
  }
};

class BroadcastExpandNonSingletonPass
    : public MapMutationPass<BroadcastExpandNonSingletonPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastExpandNonSingletonPass)
  StringRef getArgument() const final {
    return "shuttle-test-expand-nonsingleton-axis";
  }
  void runOnOperation() override {
    MapOp operation = rowSingletonBroadcast();
    if (!operation) {
      return;
    }
    MLIRContext *context = operation.getContext();
    replaceInputMap(operation,
                    AffineMap::get(2, 0,
                                   {getAffineDimExpr(0, context).floorDiv(7),
                                    getAffineDimExpr(1, context).floorDiv(13)},
                                   context));
  }
};

class MapConstantZeroPointwisePass
    : public MapMutationPass<MapConstantZeroPointwisePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapConstantZeroPointwisePass)
  StringRef getArgument() const final {
    return "shuttle-test-set-pointwise-constant-zero";
  }
  void runOnOperation() override {
    MapOp operation = map(MapSemantics::Pointwise);
    if (!operation) {
      return;
    }
    auto input = cast<RankedTensorType>(operation.getInputs()[0].getType());
    MLIRContext *context = operation.getContext();
    SmallVector<AffineExpr> expressions;
    expressions.push_back(getAffineConstantExpr(0, context));
    for (unsigned dimension = 1; dimension < input.getRank(); ++dimension) {
      expressions.push_back(getAffineDimExpr(dimension, context));
    }
    replaceInputMap(operation,
                    AffineMap::get(input.getRank(), 0, expressions, context));
  }
};

class BroadcastReplayAsReshapePass
    : public MapMutationPass<BroadcastReplayAsReshapePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastReplayAsReshapePass)
  StringRef getArgument() const final {
    return "shuttle-test-replay-broadcast-as-reshape";
  }
  void runOnOperation() override {
    if (MapOp operation = rowSingletonBroadcast()) {
      operation->setAttr(
          "semantics",
          MapSemanticsAttr::get(operation.getContext(), MapSemantics::Reshape));
    }
  }
};

class ReshapeReplayAsBroadcastPass
    : public MapMutationPass<ReshapeReplayAsBroadcastPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReshapeReplayAsBroadcastPass)
  StringRef getArgument() const final {
    return "shuttle-test-replay-reshape-as-broadcast";
  }
  void runOnOperation() override {
    MapOp operation;
    getOperation().walk([&](MapOp candidate) {
      if (operation || candidate.getSemantics() != MapSemantics::Reshape) {
        return;
      }
      auto input =
          dyn_cast<RankedTensorType>(candidate.getInputs()[0].getType());
      auto result =
          dyn_cast<RankedTensorType>(candidate.getResult(0).getType());
      if (input && result && input.getRank() > result.getRank()) {
        operation = candidate;
      }
    });
    if (!operation) {
      getOperation().emitError("test fixture has no rank-reducing reshape Map");
      signalPassFailure();
      return;
    }
    operation->setAttr("semantics",
                       MapSemanticsAttr::get(operation.getContext(),
                                             MapSemantics::BroadcastInDim));
  }
};

class BroadcastDuplicatePass : public MapMutationPass<BroadcastDuplicatePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastDuplicatePass)
  StringRef getArgument() const final {
    return "shuttle-test-duplicate-broadcast-dimension";
  }
  void runOnOperation() override {
    MapOp operation = map(MapSemantics::BroadcastInDim);
    if (!operation) {
      return;
    }
    MLIRContext *context = operation.getContext();
    replaceInputMap(operation, AffineMap::get(3, 0,
                                              {getAffineDimExpr(0, context),
                                               getAffineDimExpr(0, context)},
                                              context));
  }
};

class ReshapeAmbiguousPass : public MapMutationPass<ReshapeAmbiguousPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReshapeAmbiguousPass)
  StringRef getArgument() const final {
    return "shuttle-test-ambiguate-reshape-map";
  }
  void runOnOperation() override {
    MapOp operation;
    getOperation().walk([&](MapOp candidate) {
      auto input =
          dyn_cast<RankedTensorType>(candidate.getInputs()[0].getType());
      auto result =
          dyn_cast<RankedTensorType>(candidate.getResult(0).getType());
      if (!operation && candidate.getSemantics() == MapSemantics::Reshape &&
          input && result && input.getRank() == 2 && result.getRank() == 3) {
        operation = candidate;
      }
    });
    if (!operation) {
      getOperation().emitError("test fixture has no rank-two reshape Map");
      signalPassFailure();
      return;
    }
    MLIRContext *context = operation.getContext();
    replaceInputMap(operation, AffineMap::get(3, 0,
                                              {getAffineDimExpr(2, context),
                                               getAffineDimExpr(0, context)},
                                              context));
  }
};

class MapAttributePass : public MapMutationPass<MapAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-map-attribute";
  }
  void runOnOperation() override {
    if (MapOp operation = map(MapSemantics::BroadcastInDim)) {
      operation->setAttr(
          "shuttle.test_semantic",
          IntegerAttr::get(IntegerType::get(operation.getContext(), 64), 7));
    }
  }
};

class MapYieldAttributePass : public MapMutationPass<MapYieldAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapYieldAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-map-yield-attribute";
  }
  void runOnOperation() override {
    if (MapOp operation = map(MapSemantics::BroadcastInDim)) {
      operation.getBody().front().getTerminator()->setAttr(
          "shuttle.test_semantic",
          IntegerAttr::get(IntegerType::get(operation.getContext(), 64), 7));
    }
  }
};

class MapStructuralSemanticPass
    : public MapMutationPass<MapStructuralSemanticPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapStructuralSemanticPass)
  StringRef getArgument() const final {
    return "shuttle-test-set-structural-map-semantic";
  }
  void runOnOperation() override {
    if (MapOp operation = map(MapSemantics::Pointwise)) {
      operation->setAttr(
          "semantics",
          MapSemanticsAttr::get(operation.getContext(), MapSemantics::Reshape));
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

MaterializationPlanOp materializationPlan(ModuleOp module) {
  auto plans = module.getOps<MaterializationPlanOp>();
  return plans.empty() ? MaterializationPlanOp{} : *plans.begin();
}

void refreshMaterializationFingerprint(MaterializationPlanOp plan) {
  plan.setFingerprint(materializationPlanFingerprint(plan));
}

SchedulePlanOp schedulePlan(ModuleOp module) {
  auto plans = module.getOps<SchedulePlanOp>();
  return plans.empty() ? SchedulePlanOp{} : *plans.begin();
}

void refreshScheduleFingerprint(SchedulePlanOp plan) {
  plan.setFingerprint(schedulePlanFingerprint(plan));
}

void refreshMaterializationEdges(MaterializationPlanOp plan) {
  SmallVector<MaterializationBufferOp> buffers;
  SmallVector<MaterializationTaskOp> tasks;
  llvm::append_range(buffers,
                     plan.getBody().front().getOps<MaterializationBufferOp>());
  llvm::append_range(tasks,
                     plan.getBody().front().getOps<MaterializationTaskOp>());
  SmallVector<SmallVector<int64_t>> consumers(buffers.size());
  for (auto [taskOrdinal, task] : llvm::enumerate(tasks)) {
    SmallVector<int64_t> dependencies;
    for (int64_t bufferOrdinal : task.getInputBuffers()) {
      if (consumers[bufferOrdinal].empty() ||
          consumers[bufferOrdinal].back() !=
              static_cast<int64_t>(taskOrdinal)) {
        consumers[bufferOrdinal].push_back(taskOrdinal);
      }
      if (auto producer =
              buffers[bufferOrdinal]->getAttrOfType<IntegerAttr>("producer")) {
        if (!llvm::is_contained(dependencies, producer.getInt())) {
          dependencies.push_back(producer.getInt());
        }
      }
    }
    llvm::sort(dependencies);
    task->setAttr("dependencies",
                  DenseI64ArrayAttr::get(task.getContext(), dependencies));
  }
  for (auto [ordinal, buffer] : llvm::enumerate(buffers)) {
    buffer->setAttr("consumers", DenseI64ArrayAttr::get(buffer.getContext(),
                                                        consumers[ordinal]));
    int64_t lifetimeEnd =
        buffer.getLiveOut()
            ? static_cast<int64_t>(tasks.size())
            : (consumers[ordinal].empty() ? buffer.getLifetimeStart()
                                          : consumers[ordinal].back());
    buffer.setLifetimeEnd(lifetimeEnd);
  }
  refreshMaterializationFingerprint(plan);
}

class ReportMaterializationFingerprintPass
    : public MutationPass<ReportMaterializationFingerprintPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ReportMaterializationFingerprintPass)
  StringRef getArgument() const final {
    return "shuttle-test-report-materialization-fingerprint";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no materialization plan");
      return signalPassFailure();
    }
    llvm::outs() << plan.getFingerprint() << '\n';
  }
};

class MaterializationDeleteTaskPass
    : public MutationPass<MaterializationDeleteTaskPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializationDeleteTaskPass)
  StringRef getArgument() const final {
    return "shuttle-test-delete-materialization-task";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    MaterializationTaskOp task;
    if (plan) {
      for (MaterializationTaskOp candidate :
           plan.getBody().front().getOps<MaterializationTaskOp>()) {
        task = candidate;
        break;
      }
    }
    if (!task) {
      getOperation().emitError("test fixture has no materialization task");
      return signalPassFailure();
    }
    task.erase();
    refreshMaterializationFingerprint(plan);
  }
};

class MaterializationReorderTasksPass
    : public MutationPass<MaterializationReorderTasksPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializationReorderTasksPass)
  StringRef getArgument() const final {
    return "shuttle-test-reorder-materialization-tasks";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    SmallVector<MaterializationTaskOp> tasks;
    if (plan) {
      llvm::append_range(
          tasks, plan.getBody().front().getOps<MaterializationTaskOp>());
    }
    if (tasks.size() < 2) {
      getOperation().emitError("test fixture has fewer than two tasks");
      return signalPassFailure();
    }
    tasks[1]->moveBefore(tasks[0]);
    refreshMaterializationFingerprint(plan);
  }
};

class MaterializationSelfConsistentReorderPass
    : public MutationPass<MaterializationSelfConsistentReorderPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializationSelfConsistentReorderPass)
  StringRef getArgument() const final {
    return "shuttle-test-reorder-materialization-tasks-consistently";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    SmallVector<MaterializationTaskOp> tasks;
    SmallVector<MaterializationBufferOp> buffers;
    if (plan) {
      llvm::append_range(
          tasks, plan.getBody().front().getOps<MaterializationTaskOp>());
      llvm::append_range(
          buffers, plan.getBody().front().getOps<MaterializationBufferOp>());
    }
    size_t left = 0;
    while (left < tasks.size() && !tasks[left].getDependencies().empty()) {
      ++left;
    }
    size_t right = left + 1;
    while (right < tasks.size() && !tasks[right].getDependencies().empty()) {
      ++right;
    }
    if (right >= tasks.size()) {
      getOperation().emitError("test fixture has no independent task pair");
      return signalPassFailure();
    }
    SmallVector<int64_t> oldToNew(tasks.size());
    for (auto [ordinal, ignored] : llvm::enumerate(tasks)) {
      (void)ignored;
      oldToNew[ordinal] = ordinal;
    }
    oldToNew[right] = left;
    for (size_t ordinal = left; ordinal < right; ++ordinal) {
      oldToNew[ordinal] = ordinal + 1;
    }
    tasks[right]->moveBefore(tasks[left]);
    tasks.clear();
    llvm::append_range(tasks,
                       plan.getBody().front().getOps<MaterializationTaskOp>());
    for (auto [ordinal, task] : llvm::enumerate(tasks)) {
      task.setOrdinal(ordinal);
      SmallVector<int64_t> dependencies;
      for (int64_t dependency : task.getDependencies()) {
        dependencies.push_back(oldToNew[dependency]);
      }
      llvm::sort(dependencies);
      task->setAttr("dependencies",
                    DenseI64ArrayAttr::get(task.getContext(), dependencies));
    }
    for (MaterializationBufferOp buffer : buffers) {
      if (auto producer = buffer->getAttrOfType<IntegerAttr>("producer")) {
        buffer->setAttr(
            "producer",
            IntegerAttr::get(producer.getType(), oldToNew[producer.getInt()]));
        buffer.setLifetimeStart(oldToNew[producer.getInt()]);
      }
      SmallVector<int64_t> consumers;
      for (int64_t consumer : buffer.getConsumers()) {
        consumers.push_back(oldToNew[consumer]);
      }
      llvm::sort(consumers);
      buffer->setAttr("consumers",
                      DenseI64ArrayAttr::get(buffer.getContext(), consumers));
      if (!buffer.getLiveOut() && !consumers.empty()) {
        buffer.setLifetimeEnd(consumers.back());
      }
    }
    refreshMaterializationFingerprint(plan);
  }
};

class MaterializationReplaySourcePass
    : public MutationPass<MaterializationReplaySourcePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializationReplaySourcePass)
  StringRef getArgument() const final {
    return "shuttle-test-replay-materialization-source";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    SmallVector<MaterializationTaskOp> tasks;
    if (plan) {
      llvm::append_range(
          tasks, plan.getBody().front().getOps<MaterializationTaskOp>());
    }
    if (tasks.size() < 2) {
      getOperation().emitError("test fixture has fewer than two tasks");
      return signalPassFailure();
    }
    tasks[1]->setAttr("source", tasks[0].getSource());
    refreshMaterializationFingerprint(plan);
  }
};

class MaterializationDuplicateAlgebraSourcePass
    : public MutationPass<MaterializationDuplicateAlgebraSourcePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializationDuplicateAlgebraSourcePass)
  StringRef getArgument() const final {
    return "shuttle-test-duplicate-materialization-algebra-source";
  }
  void runOnOperation() override {
    SmallVector<MapOp> maps;
    getOperation().walk([&](MapOp map) { maps.push_back(map); });
    if (maps.size() < 2) {
      getOperation().emitError("test fixture has fewer than two Maps");
      return signalPassFailure();
    }
    maps[1]->setAttr("source", maps[0].getSource());
  }
};

class MaterializationSwapSameTypeEdgesPass
    : public MutationPass<MaterializationSwapSameTypeEdgesPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializationSwapSameTypeEdgesPass)
  StringRef getArgument() const final {
    return "shuttle-test-swap-materialization-edges";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    SmallVector<MaterializationBufferOp> buffers;
    if (plan) {
      llvm::append_range(
          buffers, plan.getBody().front().getOps<MaterializationBufferOp>());
    }
    if (plan) {
      for (MaterializationTaskOp task :
           plan.getBody().front().getOps<MaterializationTaskOp>()) {
        SmallVector<int64_t> inputs(task.getInputBuffers().begin(),
                                    task.getInputBuffers().end());
        for (size_t left = 0; left < inputs.size(); ++left) {
          for (size_t right = left + 1; right < inputs.size(); ++right) {
            if (inputs[left] == inputs[right] ||
                buffers[inputs[left]].getTensorType() !=
                    buffers[inputs[right]].getTensorType()) {
              continue;
            }
            std::swap(inputs[left], inputs[right]);
            task->setAttr("input_buffers",
                          DenseI64ArrayAttr::get(task.getContext(), inputs));
            refreshMaterializationEdges(plan);
            return;
          }
        }
      }
    }
    getOperation().emitError("test fixture has no same-type task edge pair");
    signalPassFailure();
  }
};

template <bool EmptyToScalar>
class MaterializationDomainPass
    : public MutationPass<MaterializationDomainPass<EmptyToScalar>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializationDomainPass<EmptyToScalar>)
  StringRef getArgument() const final {
    return EmptyToScalar ? "shuttle-test-add-scalar-materialization-domain"
                         : "shuttle-test-empty-tensor-materialization-domain";
  }
  void runOnOperation() override {
    ModuleOp module = this->getOperation();
    MaterializationPlanOp plan = materializationPlan(module);
    if (plan) {
      for (MaterializationTaskOp task :
           plan.getBody().front().getOps<MaterializationTaskOp>()) {
        if (task.getKind() != MaterializationTaskKind::Map ||
            task.getDomainShape().empty() != EmptyToScalar) {
          continue;
        }
        task->setAttr("domain_shape", DenseI64ArrayAttr::get(
                                          task.getContext(),
                                          EmptyToScalar ? ArrayRef<int64_t>{1}
                                                        : ArrayRef<int64_t>{}));
        refreshMaterializationFingerprint(plan);
        return;
      }
    }
    module.emitError("test fixture has no matching Map domain");
    this->signalPassFailure();
  }
};

class MaterializationUnknownAttributePass
    : public MutationPass<MaterializationUnknownAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializationUnknownAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-materialization-attribute";
  }
  void runOnOperation() override {
    MaterializationPlanOp plan = materializationPlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no materialization plan");
      return signalPassFailure();
    }
    plan->setAttr("shuttle.test_semantic", UnitAttr::get(plan.getContext()));
  }
};

class ReportScheduleFingerprintPass
    : public MutationPass<ReportScheduleFingerprintPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReportScheduleFingerprintPass)
  StringRef getArgument() const final {
    return "shuttle-test-report-simt32-schedule-fingerprint";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no schedule plan");
      return signalPassFailure();
    }
    llvm::outs() << plan.getFingerprint() << '\n';
  }
};

class ScheduleIndexingPass : public MutationPass<ScheduleIndexingPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleIndexingPass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-indexing";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleBufferOp buffer :
           plan.getBody().front().getOps<ScheduleBufferOp>()) {
        auto type = cast<RankedTensorType>(buffer.getTensorType());
        if (type.getRank() == 2) {
          buffer->setAttr("iteration_order",
                          DenseI64ArrayAttr::get(buffer.getContext(), {1, 0}));
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no rank-two schedule buffer");
    signalPassFailure();
  }
};

class ScheduleAxisPass : public MutationPass<ScheduleAxisPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleAxisPass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-axis";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleTaskOp task :
           plan.getBody().front().getOps<ScheduleTaskOp>()) {
        if (task.getKind() == ScheduleTaskKind::RowFold) {
          task->setAttr(
              "reduction_axis",
              IntegerAttr::get(task.getReductionAxisAttr().getType(), 0));
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no row Fold schedule");
    signalPassFailure();
  }
};

class ScheduleTilePass : public MutationPass<ScheduleTilePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleTilePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-tile";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleTaskOp task :
           plan.getBody().front().getOps<ScheduleTaskOp>()) {
        if (task.getKind() == ScheduleTaskKind::RowFold) {
          task->setAttr("tile_shape",
                        DenseI64ArrayAttr::get(task.getContext(), {1, 12}));
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no row Fold schedule");
    signalPassFailure();
  }
};

class ScheduleResourcePass : public MutationPass<ScheduleResourcePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleResourcePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-resource";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleTaskOp task :
           plan.getBody().front().getOps<ScheduleTaskOp>()) {
        if (task.getKind() == ScheduleTaskKind::RowFold) {
          task.setScratchBytes(task.getScratchBytes() + 4);
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no row Fold schedule");
    signalPassFailure();
  }
};

class ScheduleDependencyPass : public MutationPass<ScheduleDependencyPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleDependencyPass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-dependency";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleTaskOp task :
           plan.getBody().front().getOps<ScheduleTaskOp>()) {
        if (!task.getDependencies().empty()) {
          SmallVector<int64_t> dependencies(task.getDependencies().begin(),
                                            task.getDependencies().end());
          dependencies.pop_back();
          task->setAttr("dependencies", DenseI64ArrayAttr::get(
                                            task.getContext(), dependencies));
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no dependent schedule task");
    signalPassFailure();
  }
};

class ScheduleReplaySourceTaskPass
    : public MutationPass<ScheduleReplaySourceTaskPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleReplaySourceTaskPass)
  StringRef getArgument() const final {
    return "shuttle-test-replay-schedule-source-task";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    SmallVector<ScheduleTaskOp> tasks;
    if (plan) {
      llvm::append_range(tasks,
                         plan.getBody().front().getOps<ScheduleTaskOp>());
    }
    if (tasks.size() < 2) {
      getOperation().emitError(
          "test fixture has fewer than two schedule tasks");
      return signalPassFailure();
    }
    tasks[1].setSourceTask(tasks[0].getSourceTask());
    refreshScheduleFingerprint(plan);
  }
};

class ScheduleTypePass : public MutationPass<ScheduleTypePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleTypePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-type";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleBufferOp buffer :
           plan.getBody().front().getOps<ScheduleBufferOp>()) {
        auto type = cast<RankedTensorType>(buffer.getTensorType());
        if (type.getElementType().isBF16()) {
          buffer->setAttr(
              "tensor_type",
              TypeAttr::get(RankedTensorType::get(
                  type.getShape(), Float32Type::get(buffer.getContext()))));
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no bf16 schedule buffer");
    signalPassFailure();
  }
};

template <typename PlanOp>
class ClonePlanPass : public MutationPass<ClonePlanPass<PlanOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ClonePlanPass<PlanOp>)
  StringRef getArgument() const final {
    return std::is_same_v<PlanOp, MaterializationPlanOp>
               ? "shuttle-test-clone-materialization-plan"
               : "shuttle-test-clone-schedule-plan";
  }
  void runOnOperation() override {
    ModuleOp module = this->getOperation();
    auto plans = module.getOps<PlanOp>();
    if (plans.empty()) {
      module.emitError("test fixture has no plan to clone");
      return this->signalPassFailure();
    }
    PlanOp plan = *plans.begin();
    Operation *clone = plan->clone();
    module.getBody()->getOperations().push_back(clone);
  }
};

class ScheduleUnknownAttributePass
    : public MutationPass<ScheduleUnknownAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleUnknownAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-schedule-attribute";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no schedule plan");
      return signalPassFailure();
    }
    plan->setAttr("shuttle.test_semantic", UnitAttr::get(plan.getContext()));
  }
};

class ScheduleTaskUnknownAttributePass
    : public MutationPass<ScheduleTaskUnknownAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleTaskUnknownAttributePass)
  StringRef getArgument() const final {
    return "shuttle-test-add-schedule-task-attribute";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (!plan) {
      getOperation().emitError("test fixture has no schedule plan");
      return signalPassFailure();
    }
    auto tasks = plan.getBody().front().getOps<ScheduleTaskOp>();
    if (tasks.empty()) {
      getOperation().emitError("test fixture has no schedule task");
      return signalPassFailure();
    }
    (*tasks.begin())
        ->setAttr("shuttle.test_semantic", UnitAttr::get(plan.getContext()));
  }
};

class ScheduleTargetProfilePass
    : public MutationPass<ScheduleTargetProfilePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleTargetProfilePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-target";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleTaskOp task :
           plan.getBody().front().getOps<ScheduleTaskOp>()) {
        if (task.getKind() == ScheduleTaskKind::RowFold) {
          task.setSubgroupSize(64);
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no row Fold schedule");
    signalPassFailure();
  }
};

class ScheduleReductionOrderPass
    : public MutationPass<ScheduleReductionOrderPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleReductionOrderPass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-order";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleTaskOp task :
           plan.getBody().front().getOps<ScheduleTaskOp>()) {
        if (task.getKind() == ScheduleTaskKind::RowFold) {
          task->setAttr(
              "reduction_order",
              ScheduleReductionOrderAttr::get(
                  task.getContext(), ScheduleReductionOrder::LeafOrderFree));
          refreshScheduleFingerprint(plan);
          return;
        }
      }
    }
    getOperation().emitError("test fixture has no row Fold schedule");
    signalPassFailure();
  }
};

class ScheduleLifetimePass : public MutationPass<ScheduleLifetimePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleLifetimePass)
  StringRef getArgument() const final {
    return "shuttle-test-mutate-schedule-lifetime";
  }
  void runOnOperation() override {
    SchedulePlanOp plan = schedulePlan(getOperation());
    if (plan) {
      for (ScheduleBufferOp buffer :
           plan.getBody().front().getOps<ScheduleBufferOp>()) {
        buffer.setLifetimeEnd(buffer.getLifetimeEnd() + 1);
        refreshScheduleFingerprint(plan);
        return;
      }
    }
    getOperation().emitError("test fixture has no mutable schedule lifetime");
    signalPassFailure();
  }
};

class RenameSymbolsPass : public MutationPass<RenameSymbolsPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RenameSymbolsPass)
  StringRef getArgument() const final { return "shuttle-test-rename-symbols"; }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.setSymName("renamed_module");
    int64_t ordinal = 0;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      function.setSymName(("renamed_function_" + Twine(ordinal++)).str());
    }
  }
};

class SetFastRegionPolicyPass : public MutationPass<SetFastRegionPolicyPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetFastRegionPolicyPass)
  StringRef getArgument() const final {
    return "shuttle-test-set-fast-region-policy";
  }
  void runOnOperation() override {
    bool changed = false;
    getOperation().walk([&](RegionOp region) {
      region.setPolicy(NumericalPolicy::Fast);
      changed = true;
    });
    if (!changed) {
      getOperation().emitError("test fixture has no Shuttle region");
      signalPassFailure();
    }
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
  PassRegistration<BroadcastDuplicatePass>();
  PassRegistration<BroadcastMappedAxesSwapPass>();
  PassRegistration<BroadcastMappedAxisDuplicatePass>();
  PassRegistration<BroadcastWrongDivisorPass>();
  PassRegistration<BroadcastLiteralZeroPass>();
  PassRegistration<BroadcastCompositeDividendPass>();
  PassRegistration<BroadcastWrongResultExtentPass>();
  PassRegistration<BroadcastDirectSingletonPass>();
  PassRegistration<BroadcastExpandNonSingletonPass>();
  PassRegistration<MapConstantZeroPointwisePass>();
  PassRegistration<BroadcastReplayAsReshapePass>();
  PassRegistration<ReshapeReplayAsBroadcastPass>();
  PassRegistration<ReshapeAmbiguousPass>();
  PassRegistration<MapAttributePass>();
  PassRegistration<MapYieldAttributePass>();
  PassRegistration<MapStructuralSemanticPass>();
  PassRegistration<ManifestVersionPass>();
  PassRegistration<ManifestVersionMissingPass>();
  PassRegistration<ReportNormalizedFingerprintPass>();
  PassRegistration<ReportMaterializationFingerprintPass>();
  PassRegistration<MaterializationDeleteTaskPass>();
  PassRegistration<MaterializationReorderTasksPass>();
  PassRegistration<MaterializationSelfConsistentReorderPass>();
  PassRegistration<MaterializationReplaySourcePass>();
  PassRegistration<MaterializationDuplicateAlgebraSourcePass>();
  PassRegistration<MaterializationSwapSameTypeEdgesPass>();
  PassRegistration<MaterializationDomainPass<true>>();
  PassRegistration<MaterializationDomainPass<false>>();
  PassRegistration<MaterializationUnknownAttributePass>();
  PassRegistration<ReportScheduleFingerprintPass>();
  PassRegistration<ScheduleIndexingPass>();
  PassRegistration<ScheduleAxisPass>();
  PassRegistration<ScheduleTilePass>();
  PassRegistration<ScheduleResourcePass>();
  PassRegistration<ScheduleDependencyPass>();
  PassRegistration<ScheduleReplaySourceTaskPass>();
  PassRegistration<ScheduleTypePass>();
  PassRegistration<ClonePlanPass<MaterializationPlanOp>>();
  PassRegistration<ClonePlanPass<SchedulePlanOp>>();
  PassRegistration<ScheduleUnknownAttributePass>();
  PassRegistration<ScheduleTaskUnknownAttributePass>();
  PassRegistration<ScheduleTargetProfilePass>();
  PassRegistration<ScheduleReductionOrderPass>();
  PassRegistration<ScheduleLifetimePass>();
  PassRegistration<RenameSymbolsPass>();
  PassRegistration<SetFastRegionPolicyPass>();
}

} // namespace mlir::shuttle::test
