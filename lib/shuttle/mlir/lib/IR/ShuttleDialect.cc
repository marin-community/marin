// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/IR/ShuttleDialect.h"

#include <cstddef>

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/IR/ShuttleOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include "shuttle/IR/ShuttleEnums.cc.inc"

#define GET_ATTRDEF_CLASSES
#include "shuttle/IR/ShuttleAttrs.cc.inc"

#define GET_OP_CLASSES
#include "shuttle/IR/ShuttleOps.cc.inc"

#include "shuttle/IR/ShuttleDialect.cc.inc"

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

bool isScalarNumericType(Type type) {
  return isa<IntegerType, FloatType, ComplexType>(type);
}

bool haveCompatibleShapes(RankedTensorType lhs, RankedTensorType rhs) {
  if (lhs.getRank() != rhs.getRank()) {
    return false;
  }
  for (auto [lhsSize, rhsSize] :
       llvm::zip_equal(lhs.getShape(), rhs.getShape())) {
    if (!ShapedType::isDynamic(lhsSize) && !ShapedType::isDynamic(rhsSize) &&
        lhsSize != rhsSize) {
      return false;
    }
  }
  return true;
}

LogicalResult verifyPureScalarComputation(Operation *owner, Region &region) {
  for (Operation &operation : region.front()) {
    if (operation.getNumRegions() != 0) {
      return owner->emitOpError(
          "scalar body operations must not contain nested regions");
    }
    if (llvm::any_of(operation.getOperandTypes(),
                     [](Type type) { return isa<ShapedType>(type); }) ||
        llvm::any_of(operation.getResultTypes(),
                     [](Type type) { return isa<ShapedType>(type); })) {
      return owner->emitOpError(
          "scalar body operations must not use shaped values");
    }
    if (!isMemoryEffectFree(&operation)) {
      return owner->emitOpError(
          "scalar body operations must have proven no memory effects");
    }
  }
  return success();
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
  return verifyPureScalarComputation(owner, region);
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
  AffineMap first = cast<AffineMapAttr>(indexingMaps[0]).getValue();
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

LogicalResult verifyMapIndexingMaps(MapOp map, TypeRange indexedTypes) {
  ArrayAttr indexingMaps = map.getIndexingMaps();
  if (failed(verifyIndexingMaps(map, indexingMaps, indexedTypes))) {
    return failure();
  }
  if (indexingMaps.empty()) {
    return map.emitOpError("requires at least one indexing map");
  }

  AffineMap domain = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  SmallVector<int64_t> domainExtents(domain.getNumDims(), ShapedType::kDynamic);
  SmallVector<char> boundDimensions(domain.getNumDims(), 0);
  bool hasRankedTensor = false;
  bool resultMapProjectsDomain = false;
  size_t mapPosition = 0;
  for (auto [mapAttribute, indexedType] :
       llvm::zip_equal(indexingMaps, indexedTypes)) {
    AffineMap indexingMap = cast<AffineMapAttr>(mapAttribute).getValue();
    const bool isResultMap = mapPosition++ >= map.getInputs().size();
    if (indexingMap.getNumSymbols() != 0) {
      return map.emitOpError(
          "map indexing maps must not contain affine symbols");
    }
    llvm::SmallDenseSet<unsigned> seenDimensions;
    if (isResultMap &&
        indexingMap.getNumResults() != indexingMap.getNumDims()) {
      resultMapProjectsDomain = true;
    }
    auto tensorType = dyn_cast<RankedTensorType>(indexedType);
    if (!tensorType) {
      if (isa<ShapedType>(indexedType)) {
        return map.emitOpError(
            "map inputs and results must be scalars or ranked tensors");
      }
      if (indexingMap.getNumResults() != 0) {
        return map.emitOpError(
            "scalar inputs and results require zero-result indexing maps");
      }
      continue;
    }

    hasRankedTensor = true;
    for (auto [resultPosition, expression] :
         llvm::enumerate(indexingMap.getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension) {
        auto constant = dyn_cast<AffineConstantExpr>(expression);
        if (isResultMap || !constant || constant.getValue() != 0 ||
            tensorType.getDimSize(resultPosition) != 1) {
          return map.emitOpError(
              "map input indexing maps may use constant zero only for "
              "static singleton tensor dimensions");
        }
        continue;
      }
      const unsigned domainPosition = dimension.getPosition();
      if (!seenDimensions.insert(domainPosition).second) {
        return map.emitOpError(
            "map indexing maps must use each direct domain dimension at "
            "most once");
      }
      boundDimensions[domainPosition] = 1;
      const int64_t extent = tensorType.getDimSize(resultPosition);
      if (ShapedType::isDynamic(extent)) {
        continue;
      }
      int64_t &knownExtent = domainExtents[domainPosition];
      if (!ShapedType::isDynamic(knownExtent) && knownExtent != extent) {
        return map.emitOpError(
            "map indexing maps bind one domain dimension to inconsistent "
            "static extents");
      }
      knownExtent = extent;
    }
  }
  if (!hasRankedTensor && domain.getNumDims() != 0) {
    return map.emitOpError(
        "a scalar-only map requires a zero-dimensional indexing domain");
  }
  if (hasRankedTensor && llvm::is_contained(boundDimensions, 0)) {
    return map.emitOpError(
        "every map domain dimension must be bound by a ranked tensor "
        "dimension");
  }
  if (resultMapProjectsDomain) {
    return map.emitOpError(
        "map result indexing maps must cover every domain dimension "
        "exactly once");
  }
  return success();
}

LogicalResult verifyContractMaps(ContractOp contract, TypeRange indexedTypes) {
  ArrayAttr indexingMaps = contract.getIndexingMaps();
  if (failed(verifyIndexingMaps(contract, indexingMaps, indexedTypes))) {
    return failure();
  }
  if (indexingMaps.empty()) {
    return contract.emitOpError("requires non-empty indexing maps");
  }

  AffineMap domain = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  SmallVector<int64_t> domainExtents(domain.getNumDims(), ShapedType::kDynamic);
  SmallVector<char> boundDimensions(domain.getNumDims(), 0);
  for (auto [mapAttribute, indexedTypeValue] :
       llvm::zip_equal(indexingMaps, indexedTypes)) {
    auto indexedType = cast<RankedTensorType>(indexedTypeValue);
    AffineMap map = cast<AffineMapAttr>(mapAttribute).getValue();
    if (map.getNumSymbols() != 0) {
      return contract.emitOpError(
          "contraction indexing maps must not contain affine symbols");
    }
    if (!map.isProjectedPermutation()) {
      return contract.emitOpError(
          "contraction indexing maps must be projected permutations of "
          "direct domain dimensions");
    }
    for (auto [resultPosition, expression] :
         llvm::enumerate(map.getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension) {
        return contract.emitOpError(
            "contraction indexing maps must contain only direct dimensions");
      }
      const unsigned domainPosition = dimension.getPosition();
      boundDimensions[domainPosition] = 1;
      const int64_t extent = indexedType.getDimSize(resultPosition);
      if (ShapedType::isDynamic(extent)) {
        continue;
      }
      int64_t &knownExtent = domainExtents[domainPosition];
      if (!ShapedType::isDynamic(knownExtent) && knownExtent != extent) {
        return contract.emitOpError(
            "contraction indexing maps bind one domain dimension to "
            "inconsistent static extents");
      }
      knownExtent = extent;
    }
  }
  if (llvm::is_contained(boundDimensions, 0)) {
    return contract.emitOpError(
        "every contraction domain dimension must be bound by an indexing map");
  }
  return success();
}

bool mapContainsDimension(AffineMap map, unsigned dimension) {
  return llvm::any_of(map.getResults(), [dimension](AffineExpr expression) {
    auto dimExpression = dyn_cast<AffineDimExpr>(expression);
    return dimExpression && dimExpression.getPosition() == dimension;
  });
}

LogicalResult verifyDotGeneralIterators(ContractOp contract) {
  ArrayAttr indexingMaps = contract.getIndexingMaps();
  AffineMap lhsMap = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  AffineMap rhsMap = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  AffineMap resultMap = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  size_t reductionCount = 0;
  for (auto [dimension, iteratorAttribute] :
       llvm::enumerate(contract.getIteratorKinds())) {
    StringRef iterator = cast<StringAttr>(iteratorAttribute).getValue();
    const bool inLhs = mapContainsDimension(lhsMap, dimension);
    const bool inRhs = mapContainsDimension(rhsMap, dimension);
    const bool inResult = mapContainsDimension(resultMap, dimension);
    if (iterator == "reduction") {
      ++reductionCount;
      if (!inLhs || !inRhs || inResult) {
        return contract.emitOpError(
            "each reduction dimension must appear in both input maps and not "
            "in the result map");
      }
      continue;
    }
    if ((!inLhs && !inRhs) || !inResult) {
      return contract.emitOpError(
          "each parallel dimension must appear in an input map and exactly "
          "once in the result map");
    }
  }
  if (reductionCount == 0) {
    return contract.emitOpError(
        "the 'dot_general' algorithm requires at least one reduction iterator");
  }
  return success();
}

LogicalResult verifyDotGeneralElementTypes(ContractOp contract) {
  Type lhsElement = cast<RankedTensorType>(contract.getInputs()[0].getType())
                        .getElementType();
  Type rhsElement = cast<RankedTensorType>(contract.getInputs()[1].getType())
                        .getElementType();
  Type resultElement =
      cast<RankedTensorType>(contract.getResults()[0].getType())
          .getElementType();
  Type accumulator =
      cast<TypeAttr>(contract.getAccumulatorTypes()[0]).getValue();

  if (lhsElement != rhsElement) {
    return contract.emitOpError(
        "dot_general lhs and rhs element types must match");
  }
  if (!lhsElement.isF32() && !lhsElement.isBF16()) {
    return contract.emitOpError(
        "dot_general supports only bf16 or f32 operand elements");
  }
  if (!accumulator.isF32()) {
    return contract.emitOpError("dot_general requires an f32 accumulator");
  }
  if (!resultElement.isF32()) {
    return contract.emitOpError("dot_general requires f32 result elements");
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
  if (getSemantics() != MapSemantics::Pointwise) {
    Block &body = getBody().front();
    auto yield = cast<YieldOp>(body.getTerminator());
    if (getInputs().size() != 1 || getResults().size() != 1 ||
        !body.without_terminator().empty() || yield.getValues().size() != 1 ||
        yield.getValues()[0] != body.getArgument(0)) {
      return emitOpError(
          "structural semantics require one input, one result, and a direct "
          "scalar identity body");
    }
  }
  SmallVector<Type> indexedTypes;
  llvm::append_range(indexedTypes, getInputs().getTypes());
  llvm::append_range(indexedTypes, getResults().getTypes());
  return verifyMapIndexingMaps(*this, indexedTypes);
}

LogicalResult ContractOp::verify() {
  if (getInputs().empty() || getResults().empty()) {
    return emitOpError("requires at least one input and one result");
  }
  SmallVector<Type> indexedTypes;
  llvm::append_range(indexedTypes, getInputs().getTypes());
  llvm::append_range(indexedTypes, getResults().getTypes());
  for (Type type : indexedTypes) {
    if (!isa<RankedTensorType>(type)) {
      return emitOpError("requires ranked tensor inputs and results");
    }
  }
  if (failed(verifyContractMaps(*this, indexedTypes))) {
    return failure();
  }
  if (getAccumulatorTypes().size() != getResults().size()) {
    return emitOpError("requires one accumulator type per result");
  }
  for (Attribute accumulator : getAccumulatorTypes()) {
    Type accumulatorType = cast<TypeAttr>(accumulator).getValue();
    if (!isScalarNumericType(accumulatorType)) {
      return emitOpError("accumulator types must be scalar numeric types");
    }
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
    AffineMap domain = cast<AffineMapAttr>(getIndexingMaps()[0]).getValue();
    if (getIteratorKinds().size() != domain.getNumDims()) {
      return emitOpError(
          "requires one iterator kind per indexing-map dimension");
    }
  }
  if (getPrecision().size() != getInputs().size()) {
    return emitOpError("requires one precision entry per input");
  }
  if (getAlgorithm() != "dot_general") {
    return emitOpError("supports only the 'dot_general' algorithm");
  }
  if (getInputs().size() != 2 || getResults().size() != 1) {
    return emitOpError(
        "the 'dot_general' algorithm requires two inputs and one result");
  }
  if (failed(verifyDotGeneralElementTypes(*this))) {
    return failure();
  }
  return verifyDotGeneralIterators(*this);
}

LogicalResult ScalarConvertOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (isa<ShapedType>(inputType) || isa<ShapedType>(resultType)) {
    return emitOpError("requires scalar input and result types");
  }
  if ((!inputType.isBF16() && !inputType.isF32()) ||
      (!resultType.isBF16() && !resultType.isF32())) {
    return emitOpError("supports only bf16 and f32 scalar types");
  }
  if (inputType == resultType) {
    if (!inputType.isF32() || getSemantics() != ScalarConvertSemantics::Exact) {
      return emitOpError(
          "same-type conversion supports only f32 exact semantics");
    }
    return success();
  }
  if (inputType.isBF16()) {
    if (getSemantics() != ScalarConvertSemantics::Exact) {
      return emitOpError("bf16 to f32 requires exact semantics");
    }
    return success();
  }
  if (getSemantics() != ScalarConvertSemantics::RoundNearestEven) {
    return emitOpError("f32 to bf16 requires round_nearest_even semantics");
  }
  return success();
}

LogicalResult FoldOp::verifyRegions() {
  if (failed(verifySingleBlockRegion(*this, getCombiner()))) {
    return failure();
  }
  if (getInputs().empty() || getResults().empty()) {
    return emitOpError("requires at least one input and one result");
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
  if (getReductionDimensions().empty()) {
    return emitOpError("requires at least one reduction dimension");
  }
  llvm::SmallDenseSet<int64_t> dimensions;
  for (int64_t dimension : getReductionDimensions()) {
    if (dimension < 0 || !dimensions.insert(dimension).second) {
      return emitOpError(
          "reduction dimensions must be non-negative and unique");
    }
  }
  SmallVector<RankedTensorType> inputTypes;
  SmallVector<RankedTensorType> resultTypes;
  for (auto [input, result] : llvm::zip_equal(getInputs(), getResults())) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!inputType || inputType.getRank() == 0 || !resultType) {
      return emitOpError(
          "requires positive-rank tensor inputs and ranked tensor results");
    }
    inputTypes.push_back(inputType);
    resultTypes.push_back(resultType);
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
  for (size_t index = 1; index < inputTypes.size(); ++index) {
    if (!haveCompatibleShapes(inputTypes.front(), inputTypes[index]) ||
        !haveCompatibleShapes(resultTypes.front(), resultTypes[index])) {
      return emitOpError(
          "multi-input folds require compatible input and result shapes");
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
    auto initializerType =
        dyn_cast<RankedTensorType>(getInitializers()[index].getType());
    if (!isScalarNumericType(accumulatorType)) {
      return emitOpError("accumulator types must be scalar numeric types");
    }
    if (inputElementType != accumulatorType || !initializerType ||
        initializerType.getRank() != 0 ||
        initializerType.getElementType() != accumulatorType ||
        combiner.getArgument(index).getType() != inputElementType ||
        combiner.getArgument(index + resultCount).getType() !=
            accumulatorType ||
        yield.getValues()[index].getType() != accumulatorType ||
        scalarType(getResults()[index].getType()) != accumulatorType) {
      return emitOpError("input elements, rank-zero initializers, combiner "
                         "arguments and yields, and "
                         "result elements must use accumulator types");
    }
  }
  return verifyPureScalarComputation(*this, getCombiner());
}

} // namespace mlir::shuttle
