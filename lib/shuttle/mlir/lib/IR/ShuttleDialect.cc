#include "shuttle/IR/ShuttleDialect.h"

#include <cstddef>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/IR/ShuttleOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallDenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include "shuttle/IR/ShuttleEnums.cc.inc"

#define GET_ATTRDEF_CLASSES
#include "shuttle/IR/ShuttleAttrs.cc.inc"

#define GET_OP_CLASSES
#include "shuttle/IR/ShuttleOps.cc.inc"

namespace mlir::shuttle {
namespace {

LogicalResult verifySingleBlockRegion(Operation *owner, Region &region) {
  if (region.empty()) {
    return owner->emitOpError("requires a non-empty body");
  }
  if (!llvm::hasSingleElement(region)) {
    return owner->emitOpError("requires exactly one body block");
  }
  if (!isa<YieldOp>(region.front().getTerminator())) {
    return owner->emitOpError("body must terminate with shuttle.yield");
  }
  return success();
}

LogicalResult verifyStringArray(Operation *owner, ArrayAttr values,
                                StringRef name,
                                ArrayRef<StringRef> allowedValues = {}) {
  for (Attribute value : values) {
    auto stringValue = dyn_cast<StringAttr>(value);
    if (!stringValue) {
      return owner->emitOpError() << name << " must contain only strings";
    }
    if (!allowedValues.empty() &&
        !llvm::is_contained(allowedValues, stringValue.getValue())) {
      return owner->emitOpError() << name << " contains unsupported value '"
                                  << stringValue.getValue() << "'";
    }
  }
  return success();
}

Type scalarType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    return shaped.getElementType();
  }
  return type;
}

LogicalResult verifyScalarBody(Operation *owner, Region &region,
                               ValueRange inputs, TypeRange results) {
  if (failed(verifySingleBlockRegion(owner, region))) {
    return failure();
  }
  Block &block = region.front();
  if (block.getNumArguments() != inputs.size()) {
    return owner->emitOpError(
        "scalar body argument count must equal input count");
  }
  for (auto [argument, input] : llvm::zip_equal(block.getArguments(), inputs)) {
    if (argument.getType() != scalarType(input.getType())) {
      return owner->emitOpError(
          "scalar body argument types must equal input element types");
    }
  }
  auto yield = cast<YieldOp>(block.getTerminator());
  if (yield.getValues().size() != results.size()) {
    return owner->emitOpError(
        "scalar body yield count must equal result count");
  }
  for (auto [yielded, resultType] :
       llvm::zip_equal(yield.getValues(), results)) {
    if (yielded.getType() != scalarType(resultType)) {
      return owner->emitOpError(
          "scalar body yield types must equal result element types");
    }
  }
  return success();
}

LogicalResult verifyIndexingMaps(Operation *owner, ArrayAttr indexingMaps,
                                 TypeRange indexedTypes) {
  if (indexingMaps.size() != indexedTypes.size()) {
    return owner->emitOpError() << "requires " << indexedTypes.size()
                                << " indexing maps, one per input and result";
  }
  if (indexingMaps.empty()) {
    return success();
  }
  AffineMap first = cast<AffineMapAttr>(indexingMaps.front()).getValue();
  for (auto [mapAttribute, indexedType] :
       llvm::zip_equal(indexingMaps, indexedTypes)) {
    AffineMap map = cast<AffineMapAttr>(mapAttribute).getValue();
    if (map.getNumDims() != first.getNumDims() ||
        map.getNumSymbols() != first.getNumSymbols()) {
      return owner->emitOpError(
          "indexing maps must share one domain and symbol space");
    }
    if (auto shaped = dyn_cast<ShapedType>(indexedType);
        shaped && shaped.hasRank() && map.getNumResults() != shaped.getRank()) {
      return owner->emitOpError(
          "indexing map result count must equal the indexed value rank");
    }
  }
  return success();
}

} // namespace

void ShuttleDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "shuttle/IR/ShuttleAttrs.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "shuttle/IR/ShuttleOps.cc.inc"
      >();
}

LogicalResult RegionOp::verifyRegions() {
  if (failed(verifySingleBlockRegion(*this, getBody()))) {
    return failure();
  }
  Block &block = getBody().front();
  if (block.getNumArguments() != getInputs().size()) {
    return emitOpError("body argument count must equal input count");
  }
  for (auto [argument, input] :
       llvm::zip_equal(block.getArguments(), getInputs())) {
    if (argument.getType() != input.getType()) {
      return emitOpError("body argument types must equal input types");
    }
  }
  auto yield = cast<YieldOp>(block.getTerminator());
  if (!llvm::equal(yield.getValues().getTypes(), getResults().getTypes())) {
    return emitOpError("yielded types must equal result types");
  }
  for (Attribute source : getSourceRefs()) {
    if (!isa<SourceRefAttr>(source)) {
      return emitOpError(
          "source_refs must contain only #shuttle.source_ref attributes");
    }
  }
  if (getSourceRefs().empty()) {
    return emitOpError("requires at least one declared source reference");
  }
  return success();
}

LogicalResult MapOp::verifyRegions() {
  if (failed(verifyScalarBody(*this, getBody(), getInputs(),
                              getResults().getTypes()))) {
    return failure();
  }
  SmallVector<Type> indexedTypes;
  llvm::append_range(indexedTypes, getInputs().getTypes());
  llvm::append_range(indexedTypes, getResults().getTypes());
  return verifyIndexingMaps(*this, getIndexingMaps(), indexedTypes);
}

LogicalResult ContractOp::verify() {
  SmallVector<Type> indexedTypes;
  llvm::append_range(indexedTypes, getInputs().getTypes());
  llvm::append_range(indexedTypes, getResults().getTypes());
  if (failed(verifyIndexingMaps(*this, getIndexingMaps(), indexedTypes))) {
    return failure();
  }
  if (getAccumulatorTypes().size() != getResults().size()) {
    return emitOpError("requires one accumulator type per result");
  }
  constexpr StringRef kIteratorKinds[] = {"parallel", "reduction"};
  constexpr StringRef kPrecisionValues[] = {"DEFAULT", "HIGH", "HIGHEST"};
  if (failed(verifyStringArray(*this, getIteratorKinds(), "iterator_kinds",
                               kIteratorKinds)) ||
      failed(verifyStringArray(*this, getPrecision(), "precision",
                               kPrecisionValues))) {
    return failure();
  }
  if (!getIndexingMaps().empty()) {
    AffineMap domain =
        cast<AffineMapAttr>(getIndexingMaps().front()).getValue();
    if (getIteratorKinds().size() != domain.getNumDims()) {
      return emitOpError(
          "requires one iterator kind per indexing-map dimension");
    }
  }
  if (getPrecision().size() != getInputs().size()) {
    return emitOpError("requires one precision entry per input");
  }
  if (getAlgorithm().empty()) {
    return emitOpError("requires a non-empty algorithm identifier");
  }
  return success();
}

LogicalResult FoldOp::verifyRegions() {
  if (failed(verifySingleBlockRegion(*this, getCombiner()))) {
    return failure();
  }
  if (getAccumulatorTypes().size() != getResults().size()) {
    return emitOpError("requires one accumulator type per result");
  }
  if (getInputs().size() != getResults().size()) {
    return emitOpError("requires one input per result");
  }
  if (getInitializers().size() != getResults().size()) {
    return emitOpError("requires one explicit initializer per result");
  }
  llvm::SmallDenseSet<int64_t> dimensions;
  for (int64_t dimension : getReductionDimensions()) {
    if (dimension < 0 || !dimensions.insert(dimension).second) {
      return emitOpError(
          "reduction dimensions must be non-negative and unique");
    }
  }
  for (auto [input, result] : llvm::zip_equal(getInputs(), getResults())) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!inputType || !resultType) {
      continue;
    }
    for (int64_t dimension : dimensions) {
      if (dimension >= inputType.getRank()) {
        return emitOpError("reduction dimension is outside an input rank");
      }
    }
    SmallVector<int64_t> expectedShape;
    for (int64_t dimension = 0; dimension < inputType.getRank(); ++dimension) {
      if (!dimensions.contains(dimension)) {
        expectedShape.push_back(inputType.getDimSize(dimension));
      }
    }
    if (resultType.getShape() != ArrayRef<int64_t>(expectedShape)) {
      return emitOpError("result shape must equal the input shape with reduced "
                         "dimensions removed");
    }
  }
  Block &combiner = getCombiner().front();
  const size_t resultCount = getResults().size();
  if (combiner.getNumArguments() != 2 * resultCount) {
    return emitOpError("combiner requires two block arguments per result");
  }
  auto yield = cast<YieldOp>(combiner.getTerminator());
  if (yield.getValues().size() != resultCount) {
    return emitOpError("combiner requires one yielded value per result");
  }
  for (size_t index = 0; index < resultCount; ++index) {
    Type accumulatorType =
        cast<TypeAttr>(getAccumulatorTypes()[index]).getValue();
    Type inputElementType = scalarType(getInputs()[index].getType());
    if (getInitializers()[index].getType() != accumulatorType ||
        combiner.getArgument(index).getType() != inputElementType ||
        combiner.getArgument(index + resultCount).getType() !=
            accumulatorType ||
        yield.getValues()[index].getType() != accumulatorType ||
        scalarType(getResults()[index].getType()) != accumulatorType) {
      return emitOpError(
          "combiner input arguments must use input element types, while "
          "initializers, accumulator arguments, combiner yields, and result "
          "elements must use accumulator types");
    }
  }
  return success();
}

} // namespace mlir::shuttle
