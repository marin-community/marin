// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/IR/ShuttleDialect.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>

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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

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

bool isLowerHexDigest(StringRef value) {
  return value.size() == 64 && llvm::all_of(value, [](char character) {
           return llvm::isDigit(character) ||
                  (character >= 'a' && character <= 'f');
         });
}

std::optional<int64_t> optionalIntegerAttribute(Operation *operation,
                                                StringRef name) {
  if (auto attribute = operation->getAttrOfType<IntegerAttr>(name)) {
    return attribute.getInt();
  }
  return std::nullopt;
}

SmallVector<int64_t> integerArray(DenseI64ArrayAttr attribute) {
  return SmallVector<int64_t>(attribute.asArrayRef());
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
  SmallVector<std::pair<unsigned, int64_t>> boundedZeroDivisors;
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
      bool boundedZero = false;
      if (!dimension) {
        auto constant = dyn_cast<AffineConstantExpr>(expression);
        if (constant) {
          if (isResultMap || map.getSemantics() != MapSemantics::Reshape ||
              constant.getValue() != 0 ||
              tensorType.getDimSize(resultPosition) != 1) {
            return map.emitOpError(
                "constant-zero input indexing is reserved for typed static "
                "singleton reshapes");
          }
          continue;
        }
        auto floorDiv = dyn_cast<AffineBinaryOpExpr>(expression);
        AffineExpr dividendExpression =
            floorDiv && floorDiv.getKind() == AffineExprKind::FloorDiv
                ? floorDiv.getLHS()
                : AffineExpr{};
        AffineExpr divisorExpression =
            floorDiv && floorDiv.getKind() == AffineExprKind::FloorDiv
                ? floorDiv.getRHS()
                : AffineExpr{};
        auto dividend = dyn_cast_if_present<AffineDimExpr>(dividendExpression);
        auto divisor =
            dyn_cast_if_present<AffineConstantExpr>(divisorExpression);
        if (isResultMap || map.getSemantics() != MapSemantics::BroadcastInDim ||
            !dividend || !divisor || divisor.getValue() <= 1 ||
            tensorType.getDimSize(resultPosition) != 1) {
          return map.emitOpError(
              "bounded-zero floordiv input indexing is reserved for typed "
              "static singleton broadcasts");
        }
        dimension = dividend;
        boundedZero = true;
        boundedZeroDivisors.emplace_back(dimension.getPosition(),
                                         divisor.getValue());
      }
      const unsigned domainPosition = dimension.getPosition();
      if (!seenDimensions.insert(domainPosition).second) {
        return map.emitOpError(
            "map indexing maps must use each direct domain dimension at "
            "most once");
      }
      boundDimensions[domainPosition] = 1;
      if (boundedZero) {
        continue;
      }
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
  for (auto [domainPosition, divisor] : boundedZeroDivisors) {
    if (ShapedType::isDynamic(domainExtents[domainPosition]) ||
        domainExtents[domainPosition] != divisor) {
      return map.emitOpError(
          "bounded-zero broadcast divisor must equal the static domain "
          "extent");
    }
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

std::string materializationPlanFingerprint(MaterializationPlanOp plan) {
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  stream << "schema=" << plan.getSchemaVersion() << ';';
  Attribute(plan.getPolicyAttr()).print(stream);
  stream << '\n';
  for (Operation &operation : plan.getBody().front()) {
    if (isa<MaterializationPlanYieldOp>(operation)) {
      continue;
    }
    stream << operation.getName() << '{';
    for (NamedAttribute attribute : operation.getAttrs()) {
      stream << attribute.getName() << '=';
      attribute.getValue().print(stream);
      stream << ',';
    }
    stream << "}\n";
  }
  stream.flush();
  llvm::SHA256 digest;
  digest.update(normalized);
  return llvm::toHex(digest.final(), true);
}

LogicalResult MaterializationPlanOp::verifyRegions() {
  if (!getOperation()->getDiscardableAttrs().empty()) {
    return emitOpError("does not permit discardable attributes");
  }
  Region &body = getBody();
  if (body.empty() || !llvm::hasSingleElement(body)) {
    return emitOpError("requires exactly one non-empty body block");
  }
  Block &block = body.front();
  if (!isa<MaterializationPlanYieldOp>(block.getTerminator())) {
    return emitOpError(
        "body must terminate with shuttle.materialization_plan_yield");
  }
  if (getSchemaVersion() != 1) {
    return emitOpError("requires materialization plan schema version 1");
  }
  if (!isLowerHexDigest(getFingerprint())) {
    return emitOpError("fingerprint must be a lowercase SHA-256 digest");
  }

  SmallVector<MaterializationBufferOp> buffers;
  SmallVector<MaterializationTaskOp> tasks;
  bool sawTask = false;
  for (Operation &operation : block.without_terminator()) {
    if (auto buffer = dyn_cast<MaterializationBufferOp>(operation)) {
      if (!buffer->getDiscardableAttrs().empty()) {
        return buffer.emitOpError("does not permit discardable attributes");
      }
      if (sawTask) {
        return emitOpError(
            "materialization buffers must precede materialization tasks");
      }
      buffers.push_back(buffer);
      continue;
    }
    if (auto task = dyn_cast<MaterializationTaskOp>(operation)) {
      if (!task->getDiscardableAttrs().empty()) {
        return task.emitOpError("does not permit discardable attributes");
      }
      sawTask = true;
      tasks.push_back(task);
      continue;
    }
    return emitOpError(
        "body may contain only materialization buffers and tasks");
  }
  if (buffers.empty() || tasks.empty()) {
    return emitOpError("requires at least one buffer and one task");
  }
  if (!block.getTerminator()->getDiscardableAttrs().empty()) {
    return block.getTerminator()->emitOpError(
        "does not permit discardable attributes");
  }

  unsigned foldCount = 0;
  SmallVector<SmallVector<int64_t>> actualConsumers(buffers.size());
  for (auto [position, buffer] : llvm::enumerate(buffers)) {
    if (buffer.getOrdinal() != static_cast<int64_t>(position)) {
      return buffer.emitOpError("buffer ordinals must be contiguous");
    }
    auto tensorType = dyn_cast<RankedTensorType>(buffer.getTensorType());
    if (!tensorType || !tensorType.hasStaticShape() ||
        llvm::any_of(tensorType.getShape(),
                     [](int64_t extent) { return extent <= 0; }) ||
        (!tensorType.getElementType().isF32() &&
         !tensorType.getElementType().isBF16())) {
      return buffer.emitOpError(
          "requires a positive static bf16 or f32 ranked tensor type");
    }
    std::optional<int64_t> producer =
        optionalIntegerAttribute(buffer, "producer");
    if (buffer.getLiveIn() != !producer.has_value()) {
      return buffer.emitOpError(
          "live-in status must exactly identify producer-free buffers");
    }
    const bool external =
        buffer.getStorage() == MaterializationStorage::External;
    if (external != (buffer.getLiveIn() || buffer.getLiveOut())) {
      return buffer.emitOpError(
          "external storage must exactly identify live-in or live-out buffers");
    }
    if (producer && (*producer < 0 || *producer >= tasks.size())) {
      return buffer.emitOpError("producer is outside the task range");
    }
    if (buffer.getLifetimeStart() != producer.value_or(0)) {
      return buffer.emitOpError("lifetime start must equal producer or zero");
    }
    if (buffer.getLifetimeEnd() < buffer.getLifetimeStart() ||
        buffer.getLifetimeEnd() > static_cast<int64_t>(tasks.size())) {
      return buffer.emitOpError("has an invalid task lifetime interval");
    }
  }

  for (auto [position, task] : llvm::enumerate(tasks)) {
    const int64_t ordinal = static_cast<int64_t>(position);
    if (task.getOrdinal() != ordinal) {
      return task.emitOpError("task ordinals must be contiguous");
    }
    if (!isLowerHexDigest(task.getSemanticFingerprint())) {
      return task.emitOpError(
          "semantic fingerprint must be a lowercase SHA-256 digest");
    }
    if (llvm::any_of(task.getDomainShape(),
                     [](int64_t extent) { return extent <= 0; })) {
      return task.emitOpError("requires a positive static task domain");
    }
    llvm::SmallDenseSet<int64_t> outputSet;
    llvm::SmallDenseSet<int64_t> dependencySet;
    SmallVector<int64_t> expectedDependencies;
    for (int64_t bufferOrdinal : task.getInputBuffers()) {
      if (bufferOrdinal < 0 || bufferOrdinal >= buffers.size()) {
        return task.emitOpError("input buffers must be in range");
      }
      if (actualConsumers[bufferOrdinal].empty() ||
          actualConsumers[bufferOrdinal].back() != ordinal) {
        actualConsumers[bufferOrdinal].push_back(ordinal);
      }
      std::optional<int64_t> producer =
          optionalIntegerAttribute(buffers[bufferOrdinal], "producer");
      if (producer && !llvm::is_contained(expectedDependencies, *producer)) {
        expectedDependencies.push_back(*producer);
      }
    }
    llvm::sort(expectedDependencies);
    for (int64_t bufferOrdinal : task.getOutputBuffers()) {
      if (bufferOrdinal < 0 || bufferOrdinal >= buffers.size() ||
          !outputSet.insert(bufferOrdinal).second ||
          optionalIntegerAttribute(buffers[bufferOrdinal], "producer") !=
              ordinal) {
        return task.emitOpError(
            "output buffers must be unique, in range, and owned by the task");
      }
    }
    SmallVector<int64_t> dependencies =
        integerArray(task.getDependenciesAttr());
    for (int64_t dependency : dependencies) {
      if (dependency < 0 || dependency >= ordinal ||
          !dependencySet.insert(dependency).second) {
        return task.emitOpError(
            "dependencies must be unique earlier task ordinals");
      }
    }
    if (dependencies != expectedDependencies) {
      return task.emitOpError(
          "dependencies must equal input-buffer producer tasks");
    }
    if (task.getKind() == MaterializationTaskKind::Fold) {
      ++foldCount;
      if (task.getReductionDimensions().size() != 1 ||
          task.getReductionDimensions().front() != 1 ||
          !task.getOrderFree().value_or(false) ||
          task.getDomainShape().size() != 2 ||
          task.getInputBuffers().size() != 2 ||
          task.getOutputBuffers().size() != 1) {
        return task.emitOpError(
            "row Fold tasks require rank-two domain, dimension 1, and "
            "order_free=true");
      }
      auto inputType = cast<RankedTensorType>(
          buffers[task.getInputBuffers()[0]].getTensorType());
      auto initializerType = cast<RankedTensorType>(
          buffers[task.getInputBuffers()[1]].getTensorType());
      auto resultType = cast<RankedTensorType>(
          buffers[task.getOutputBuffers()[0]].getTensorType());
      if (inputType.getShape() != task.getDomainShape() ||
          inputType.getRank() != 2 || !inputType.getElementType().isF32() ||
          initializerType.getRank() != 0 ||
          !initializerType.getElementType().isF32() ||
          resultType.getShape() !=
              ArrayRef<int64_t>{task.getDomainShape().front()} ||
          !resultType.getElementType().isF32()) {
        return task.emitOpError(
            "row Fold buffer types must match its f32 domain and accumulator");
      }
    } else {
      if (!task.getReductionDimensions().empty() || task.getOrderFree()) {
        return task.emitOpError(
            "Map tasks must not carry reduction or ordering metadata");
      }
      bool hasTensorResult = false;
      for (int64_t bufferOrdinal : task.getOutputBuffers()) {
        auto resultType =
            cast<RankedTensorType>(buffers[bufferOrdinal].getTensorType());
        hasTensorResult |= resultType.getRank() != 0;
      }
      if (task.getDomainShape().empty() == hasTensorResult) {
        return task.emitOpError(
            "empty Map domains must exactly identify scalar result tasks");
      }
    }
  }
  if (foldCount != 1) {
    return emitOpError("requires exactly one Fold task");
  }

  for (auto [position, buffer] : llvm::enumerate(buffers)) {
    SmallVector<int64_t> consumers = integerArray(buffer.getConsumersAttr());
    if (consumers != actualConsumers[position]) {
      return buffer.emitOpError(
          "consumer list must equal the tasks that read the buffer");
    }
    int64_t expectedEnd = buffer.getLifetimeStart();
    if (!consumers.empty()) {
      expectedEnd = consumers.back();
    }
    if (buffer.getLiveOut()) {
      expectedEnd = tasks.size();
    } else if (consumers.empty() && !buffer.getLiveIn()) {
      return buffer.emitOpError(
          "produced buffers require a consumer or live-out use");
    }
    if (buffer.getLifetimeEnd() != expectedEnd) {
      return buffer.emitOpError(
          "lifetime end must equal the final consumer or plan exit");
    }
  }

  if (getFingerprint() != materializationPlanFingerprint(*this)) {
    return emitOpError(
        "fingerprint does not match the closed materialization plan");
  }
  return success();
}

std::string schedulePlanFingerprint(SchedulePlanOp plan) {
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  stream << "schema=" << plan.getSchemaVersion() << ';';
  Attribute(plan.getTargetAttr()).print(stream);
  stream << ';';
  Attribute(plan.getPolicyAttr()).print(stream);
  stream << ";source=" << plan.getSourcePlanFingerprint() << '\n';
  for (Operation &operation : plan.getBody().front()) {
    if (isa<SchedulePlanYieldOp>(operation)) {
      continue;
    }
    stream << operation.getName() << '{';
    for (NamedAttribute attribute : operation.getAttrs()) {
      stream << attribute.getName() << '=';
      attribute.getValue().print(stream);
      stream << ',';
    }
    stream << "}\n";
  }
  stream.flush();
  llvm::SHA256 digest;
  digest.update(normalized);
  return llvm::toHex(digest.final(), true);
}

namespace {

FailureOr<int64_t> staticElementCount(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t extent : shape) {
    if (extent <= 0 || count > std::numeric_limits<int64_t>::max() / extent) {
      return failure();
    }
    count *= extent;
  }
  return count;
}

int64_t roundUpToSubgroup(int64_t value) {
  constexpr int64_t kSubgroup = 32;
  return ((value + kSubgroup - 1) / kSubgroup) * kSubgroup;
}

struct Simt32Geometry {
  SmallVector<int64_t> grid;
  SmallVector<int64_t> tile;
  int64_t serialTiles;
  int64_t workgroupThreads;
  int64_t subgroupSize;
  int64_t scratchBytes;
  std::optional<int64_t> reductionAxis;
};

FailureOr<Simt32Geometry> simt32Geometry(ScheduleTaskKind kind,
                                         ArrayRef<int64_t> domain) {
  constexpr int64_t kSubgroup = 32;
  constexpr int64_t kMaxThreads = 256;
  if (kind == ScheduleTaskKind::Scalar) {
    if (!domain.empty()) {
      return failure();
    }
    return Simt32Geometry{{1}, {}, 1, 1, kSubgroup, 0, std::nullopt};
  }
  if (kind == ScheduleTaskKind::Elementwise) {
    FailureOr<int64_t> elements = staticElementCount(domain);
    if (failed(elements)) {
      return failure();
    }
    int64_t tile = std::min(*elements, kMaxThreads);
    return Simt32Geometry{{(*elements + tile - 1) / tile},
                          {tile},
                          1,
                          roundUpToSubgroup(tile),
                          kSubgroup,
                          0,
                          std::nullopt};
  }
  if (domain.size() != 2 || domain[0] <= 0 || domain[1] <= 0) {
    return failure();
  }
  int64_t tile = std::min(domain[1], kMaxThreads);
  int64_t threads = roundUpToSubgroup(tile);
  return Simt32Geometry{{domain[0]},
                        {1, tile},
                        (domain[1] + tile - 1) / tile,
                        threads,
                        kSubgroup,
                        threads * static_cast<int64_t>(sizeof(float)),
                        1};
}

} // namespace

LogicalResult SchedulePlanOp::verifyRegions() {
  if (!getOperation()->getDiscardableAttrs().empty()) {
    return emitOpError("does not permit discardable attributes");
  }
  if (getSchemaVersion() != 1 || getTarget() != ScheduleTarget::Simt32) {
    return emitOpError("requires schedule schema 1 and target simt32");
  }
  if (!isLowerHexDigest(getSourcePlanFingerprint()) ||
      !isLowerHexDigest(getFingerprint())) {
    return emitOpError("fingerprints must be lowercase SHA-256 digests");
  }
  Region &body = getBody();
  if (body.empty() || !llvm::hasSingleElement(body) ||
      !isa<SchedulePlanYieldOp>(body.front().getTerminator())) {
    return emitOpError(
        "requires one block terminated by shuttle.schedule_plan_yield");
  }

  SmallVector<ScheduleBufferOp> buffers;
  SmallVector<ScheduleTaskOp> tasks;
  bool sawTask = false;
  for (Operation &operation : body.front().without_terminator()) {
    if (auto buffer = dyn_cast<ScheduleBufferOp>(operation)) {
      if (sawTask) {
        return buffer.emitOpError("schedule buffers must precede tasks");
      }
      if (!buffer->getDiscardableAttrs().empty()) {
        return buffer.emitOpError("does not permit discardable attributes");
      }
      buffers.push_back(buffer);
      continue;
    }
    if (auto task = dyn_cast<ScheduleTaskOp>(operation)) {
      sawTask = true;
      if (!task->getDiscardableAttrs().empty()) {
        return task.emitOpError("does not permit discardable attributes");
      }
      tasks.push_back(task);
      continue;
    }
    return emitOpError("body may contain only schedule buffers and tasks");
  }
  if (buffers.empty() || tasks.empty()) {
    return emitOpError("requires at least one schedule buffer and task");
  }
  if (!body.front().getTerminator()->getDiscardableAttrs().empty()) {
    return body.front().getTerminator()->emitOpError(
        "does not permit discardable attributes");
  }

  for (auto [ordinal, buffer] : llvm::enumerate(buffers)) {
    if (buffer.getOrdinal() != static_cast<int64_t>(ordinal) ||
        buffer.getSourceBuffer() != static_cast<int64_t>(ordinal)) {
      return buffer.emitOpError(
          "schedule buffer and source ordinals must be contiguous");
    }
    auto type = dyn_cast<RankedTensorType>(buffer.getTensorType());
    if (!type || !type.hasStaticShape() ||
        failed(staticElementCount(type.getShape())) ||
        (!type.getElementType().isBF16() && !type.getElementType().isF32())) {
      return buffer.emitOpError(
          "requires a positive static bf16 or f32 tensor type");
    }
    ScheduleBufferIndexing expectedIndexing =
        type.getRank() == 0 ? ScheduleBufferIndexing::Scalar
                            : ScheduleBufferIndexing::Lexicographic;
    SmallVector<int64_t> expectedOrder;
    for (int64_t axis = 0; axis < type.getRank(); ++axis) {
      expectedOrder.push_back(axis);
    }
    if (buffer.getIndexing() != expectedIndexing ||
        buffer.getIterationOrder() != ArrayRef<int64_t>(expectedOrder)) {
      return buffer.emitOpError(
          "buffer indexing must equal logical tensor rank");
    }
    if (buffer.getLifetimeStart() < 0 ||
        buffer.getLifetimeEnd() < buffer.getLifetimeStart() ||
        buffer.getLifetimeEnd() > static_cast<int64_t>(tasks.size())) {
      return buffer.emitOpError("has an invalid schedule lifetime interval");
    }
  }

  unsigned rowFoldCount = 0;
  for (auto [ordinal, task] : llvm::enumerate(tasks)) {
    if (task.getOrdinal() != static_cast<int64_t>(ordinal) ||
        task.getSourceTask() != static_cast<int64_t>(ordinal)) {
      return task.emitOpError(
          "schedule source tasks must be unique structural ordinals");
    }
    if (!isLowerHexDigest(task.getSemanticFingerprint())) {
      return task.emitOpError(
          "semantic fingerprint must be a lowercase SHA-256 digest");
    }
    for (ArrayRef<int64_t> references :
         {task.getInputBuffers(), task.getOutputBuffers()}) {
      for (int64_t buffer : references) {
        if (buffer < 0 || buffer >= buffers.size()) {
          return task.emitOpError("schedule buffer reference is out of range");
        }
      }
    }
    llvm::SmallDenseSet<int64_t> dependencies;
    for (int64_t dependency : task.getDependencies()) {
      if (dependency < 0 || dependency >= static_cast<int64_t>(ordinal) ||
          !dependencies.insert(dependency).second) {
        return task.emitOpError(
            "dependencies must be unique earlier schedule tasks");
      }
    }
    if (task.getKind() == ScheduleTaskKind::RowFold &&
        task.getReductionAxis() != 1) {
      return task.emitOpError("row Fold schedule requires reduction axis one");
    }
    std::optional<ScheduleReductionOrder> expectedOrder;
    if (task.getKind() == ScheduleTaskKind::RowFold) {
      expectedOrder = ScheduleReductionOrder::TreeAssociationFreeLeafOrderFixed;
    }
    if (task.getReductionOrder() != expectedOrder) {
      return task.emitOpError(
          "Fold reduction order must equal bound Fold semantics");
    }
    FailureOr<Simt32Geometry> expected =
        simt32Geometry(task.getKind(), task.getDomainShape());
    if (failed(expected) ||
        task.getGridShape() != ArrayRef<int64_t>(expected->grid) ||
        task.getTileShape() != ArrayRef<int64_t>(expected->tile) ||
        task.getSerialTiles() != expected->serialTiles ||
        task.getWorkgroupThreads() != expected->workgroupThreads ||
        task.getSubgroupSize() != expected->subgroupSize ||
        task.getScratchBytes() != expected->scratchBytes ||
        task.getReductionAxis() != expected->reductionAxis) {
      return task.emitOpError(
          "schedule geometry must equal the SIMT32 derivation");
    }
    if (task.getKind() == ScheduleTaskKind::RowFold) {
      ++rowFoldCount;
    }
  }
  if (rowFoldCount != 1) {
    return emitOpError("requires exactly one row Fold schedule task");
  }
  if (getFingerprint() != schedulePlanFingerprint(*this)) {
    return emitOpError("fingerprint does not match the closed schedule plan");
  }
  return success();
}

} // namespace mlir::shuttle
