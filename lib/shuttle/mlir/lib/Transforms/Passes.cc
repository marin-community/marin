#include "shuttle/Transforms/Passes.h"

#include <cstdint>
#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/IR/ShuttleOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallDenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace mlir::shuttle {

#define GEN_PASS_DEF_SHUTTLEANNOTATESOURCEPASS
#define GEN_PASS_DEF_SHUTTLEFORMSTRUCTURALREGIONSPASS
#define GEN_PASS_DEF_SHUTTLECONVERTSTABLEHLOTOALGEBRAPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYSOURCECOVERAGEPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYSEMANTICERASUREPASS
#define GEN_PASS_DEF_SHUTTLECANONICALIZEPASS
#define GEN_PASS_DEF_SHUTTLELOWERALGEBRATOSTABLEHLOPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYNOSHUTTLEOPSPASS
#include "shuttle/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kSourceRefsAttribute = "shuttle.source_refs";
constexpr llvm::StringLiteral kSelectedAttribute = "shuttle.selected";

bool containsShuttleAttribute(Attribute attribute) {
  if (attribute.getDialect().getNamespace() ==
      ShuttleDialect::getDialectNamespace()) {
    return true;
  }
  if (auto array = dyn_cast<ArrayAttr>(attribute)) {
    return llvm::any_of(array, containsShuttleAttribute);
  }
  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute)) {
    return llvm::any_of(dictionary, [](NamedAttribute namedAttribute) {
      return containsShuttleAttribute(namedAttribute.getValue());
    });
  }
  return false;
}

void annotateRegion(Region &region, uint64_t functionOrdinal,
                    uint64_t &nextBlockOrdinal) {
  for (Block &block : region) {
    const uint64_t blockOrdinal = nextBlockOrdinal++;
    uint64_t operationOrdinal = 0;
    for (Operation &operation : block) {
      Dialect *dialect = operation.getDialect();
      if (dialect == nullptr ||
          dialect->getNamespace() != ShuttleDialect::getDialectNamespace()) {
        SmallVector<Attribute> sourceRefs;
        sourceRefs.reserve(operation.getNumResults());
        for (uint64_t resultOrdinal = 0;
             resultOrdinal < operation.getNumResults(); ++resultOrdinal) {
          sourceRefs.push_back(SourceRefAttr::get(
              operation.getContext(), functionOrdinal, blockOrdinal,
              operationOrdinal, resultOrdinal));
        }
        if (!sourceRefs.empty()) {
          operation.setAttr(kSourceRefsAttribute,
                            ArrayAttr::get(operation.getContext(), sourceRefs));
        }
      }
      for (Region &nested : operation.getRegions()) {
        annotateRegion(nested, functionOrdinal, nextBlockOrdinal);
      }
      ++operationOrdinal;
    }
  }
}

LogicalResult emitUnimplementedTransform(ModuleOp module,
                                         StringRef capability) {
  module.emitError() << capability
                     << " is declared but not implemented in the first native "
                        "Shuttle dialect slice";
  return failure();
}

struct AnnotateSourcePass
    : impl::ShuttleAnnotateSourcePassBase<AnnotateSourcePass> {
  void runOnOperation() override {
    uint64_t functionOrdinal = 0;
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      uint64_t nextBlockOrdinal = 0;
      annotateRegion(function.getBody(), functionOrdinal++, nextBlockOrdinal);
    }
  }
};

struct FormStructuralRegionsPass
    : impl::ShuttleFormStructuralRegionsPassBase<FormStructuralRegionsPass> {
  void runOnOperation() override {
    if (failed(emitUnimplementedTransform(
            getOperation(), "structural StableHLO region formation"))) {
      signalPassFailure();
    }
  }
};

struct ConvertStablehloToAlgebraPass
    : impl::ShuttleConvertStablehloToAlgebraPassBase<
          ConvertStablehloToAlgebraPass> {
  void runOnOperation() override {
    if (failed(emitUnimplementedTransform(getOperation(),
                                          "StableHLO-to-Shuttle conversion"))) {
      signalPassFailure();
    }
  }
};

struct VerifySourceCoveragePass
    : impl::ShuttleVerifySourceCoveragePassBase<VerifySourceCoveragePass> {
  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](RegionOp region) {
      llvm::SmallDenseSet<Attribute> declaredSources;
      llvm::SmallDenseSet<Attribute> representedSources;
      for (Attribute source : region.getSourceRefs()) {
        auto sourceRef = dyn_cast<SourceRefAttr>(source);
        if (!sourceRef || !declaredSources.insert(sourceRef).second) {
          region.emitOpError("requires unique #shuttle.source_ref entries");
          return WalkResult::interrupt();
        }
      }
      WalkResult nestedResult = region.getBody().walk([&](Operation
                                                              *operation) {
        if (isa<RegionOp>(operation)) {
          return WalkResult::skip();
        }
        Attribute source;
        if (auto map = dyn_cast<MapOp>(operation)) {
          source = map.getSource();
        } else if (auto contract = dyn_cast<ContractOp>(operation)) {
          source = contract.getSource();
        } else if (auto fold = dyn_cast<FoldOp>(operation)) {
          source = fold.getSource();
        } else {
          return WalkResult::advance();
        }
        if (!declaredSources.contains(source)) {
          operation->emitOpError(
              "source reference is absent from the enclosing shuttle.region");
          return WalkResult::interrupt();
        }
        representedSources.insert(source);
        return WalkResult::advance();
      });
      if (nestedResult.wasInterrupted()) {
        return WalkResult::interrupt();
      }
      for (Attribute declaredSource : declaredSources) {
        if (!representedSources.contains(declaredSource)) {
          region.emitOpError(
              "declared source reference is not represented by algebra");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

struct VerifySemanticErasurePass
    : impl::ShuttleVerifySemanticErasurePassBase<VerifySemanticErasurePass> {
  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](Operation *operation) {
      if (!operation->hasAttr(kSelectedAttribute)) {
        return WalkResult::advance();
      }
      operation->emitOpError(
          "selected source operation survived Shuttle conversion");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

struct ShuttleCanonicalizePass
    : impl::ShuttleCanonicalizePassBase<ShuttleCanonicalizePass> {
  void runOnOperation() override {
    if (failed(emitUnimplementedTransform(
            getOperation(), "Shuttle algebra canonicalization"))) {
      signalPassFailure();
    }
  }
};

struct LowerAlgebraToStablehloPass
    : impl::ShuttleLowerAlgebraToStablehloPassBase<
          LowerAlgebraToStablehloPass> {
  void runOnOperation() override {
    if (failed(emitUnimplementedTransform(getOperation(),
                                          "Shuttle-to-StableHLO lowering"))) {
      signalPassFailure();
    }
  }
};

struct VerifyNoShuttleOpsPass
    : impl::ShuttleVerifyNoShuttleOpsPassBase<VerifyNoShuttleOpsPass> {
  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](Operation *operation) {
      Dialect *dialect = operation->getDialect();
      if (dialect != nullptr &&
          dialect->getNamespace() == ShuttleDialect::getDialectNamespace()) {
        operation->emitOpError("Shuttle operation remains before HLO export");
        return WalkResult::interrupt();
      }
      for (NamedAttribute namedAttribute : operation->getAttrs()) {
        if (namedAttribute.getName().strref().starts_with("shuttle.") ||
            containsShuttleAttribute(namedAttribute.getValue())) {
          operation->emitOpError()
              << "Shuttle attribute remains before HLO export: "
              << namedAttribute.getName();
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createAnnotateSourcePass() {
  return std::make_unique<AnnotateSourcePass>();
}

std::unique_ptr<Pass> createFormStructuralRegionsPass() {
  return std::make_unique<FormStructuralRegionsPass>();
}

std::unique_ptr<Pass> createConvertStablehloToAlgebraPass() {
  return std::make_unique<ConvertStablehloToAlgebraPass>();
}

std::unique_ptr<Pass> createVerifySourceCoveragePass() {
  return std::make_unique<VerifySourceCoveragePass>();
}

std::unique_ptr<Pass> createVerifySemanticErasurePass() {
  return std::make_unique<VerifySemanticErasurePass>();
}

std::unique_ptr<Pass> createShuttleCanonicalizePass() {
  return std::make_unique<ShuttleCanonicalizePass>();
}

std::unique_ptr<Pass> createLowerAlgebraToStablehloPass() {
  return std::make_unique<LowerAlgebraToStablehloPass>();
}

std::unique_ptr<Pass> createVerifyNoShuttleOpsPass() {
  return std::make_unique<VerifyNoShuttleOpsPass>();
}

} // namespace mlir::shuttle
