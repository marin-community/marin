// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Transforms/Passes.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>

#include "ObserverInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "shuttle/IR/ShuttleAttrs.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/IR/ShuttleOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::shuttle {

#define GEN_PASS_DEF_SHUTTLEANNOTATESOURCEPASS
#define GEN_PASS_DEF_SHUTTLEFORMSTRUCTURALREGIONSPASS
#define GEN_PASS_DEF_SHUTTLECONVERTSTABLEHLOTOALGEBRAPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYSOURCECOVERAGEPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYSEMANTICERASUREPASS
#define GEN_PASS_DEF_SHUTTLECANONICALIZEPASS
#define GEN_PASS_DEF_SHUTTLELOWERALGEBRATOSTABLEHLOPASS
#define GEN_PASS_DEF_SHUTTLESTRIPSOURCEPROVENANCEPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYNOSHUTTLEOPSPASS
#define GEN_PASS_DEF_SHUTTLEPLANROWFOLDMATERIALIZATIONPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYMATERIALIZATIONPLANPASS
#define GEN_PASS_DEF_SHUTTLEPLANSIMT32ROWFOLDSCHEDULEPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYSIMT32ROWFOLDSCHEDULEPASS
#define GEN_PASS_DEF_SHUTTLEBUILDCPUEXECUTABLEBUNDLEPASS
#define GEN_PASS_DEF_SHUTTLEVERIFYCPUEXECUTABLEBUNDLEPASS
#include "shuttle/Transforms/Passes.h.inc"

namespace {

ShuttlePipelineOptions commandLinePipelineOptions(NumericalPolicy numerics) {
  ShuttlePipelineOptions options;
  options.numerics = numerics;
  if (numerics == NumericalPolicy::Fast) {
    options.canonicalOptions =
        R"json({"numerics":"fast","pipeline_abi_version":5,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
  }
  return options;
}

constexpr llvm::StringLiteral kSourceRefsAttribute = "shuttle.source_refs";
constexpr llvm::StringLiteral kSelectedAttribute = "shuttle.selected";
constexpr llvm::StringLiteral kFunctionOrdinalAttribute =
    "shuttle.function_ordinal";
constexpr llvm::StringLiteral kOperationRefAttribute = "shuttle.operation_ref";
constexpr llvm::StringLiteral kCoverageManifestAttribute =
    "shuttle.coverage_manifest";
constexpr llvm::StringLiteral kRegionResultSourcesAttribute =
    "shuttle.result_source_refs";

constexpr llvm::StringLiteral kManifestComplete = "complete";
constexpr llvm::StringLiteral kManifestSelectedRegions = "selected_regions";
constexpr llvm::StringLiteral kManifestExcluded = "excluded";
constexpr llvm::StringLiteral kManifestZeroResultOperations =
    "zero_result_operations";
constexpr llvm::StringLiteral kManifestFunctionResults = "function_results";

std::string sha256(llvm::StringRef value) {
  llvm::SHA256 digest;
  digest.update(value);
  return llvm::toHex(digest.final(), true);
}

StringRef policyName(NumericalPolicy numerics) {
  return numerics == NumericalPolicy::SourceOrdered ? "source_ordered" : "fast";
}

struct CandidateComponent {
  SmallVector<Operation *> operations;
};

LogicalResult
walkSourcePreorder(Region &region,
                   llvm::function_ref<LogicalResult(Operation *)> visitor) {
  for (Block &block : region) {
    for (Operation &operation : block) {
      if (failed(visitor(&operation))) {
        return failure();
      }
      for (Region &nested : operation.getRegions()) {
        if (failed(walkSourcePreorder(nested, visitor))) {
          return failure();
        }
      }
    }
  }
  return success();
}

LogicalResult
walkSourcePreorder(Operation *operation,
                   llvm::function_ref<LogicalResult(Operation *)> visitor) {
  if (failed(visitor(operation))) {
    return failure();
  }
  for (Region &nested : operation->getRegions()) {
    if (failed(walkSourcePreorder(nested, visitor))) {
      return failure();
    }
  }
  return success();
}

bool containsShuttleAttribute(Attribute attribute) {
  bool found = false;
  attribute.walk([&](Attribute nested) {
    if (nested.getDialect().getNamespace() ==
        ShuttleDialect::getDialectNamespace()) {
      found = true;
      return;
    }
    if (auto opaque = dyn_cast<OpaqueAttr>(nested);
        opaque && opaque.getDialectNamespace().getValue() ==
                      ShuttleDialect::getDialectNamespace()) {
      found = true;
    }
  });
  return found;
}

ArrayAttr sourceRefs(Operation *operation) {
  return operation->getAttrOfType<ArrayAttr>(kSourceRefsAttribute);
}

SourceRefAttr singleSourceRef(Operation *operation) {
  ArrayAttr refs = sourceRefs(operation);
  if (!refs || refs.size() != 1) {
    return {};
  }
  return dyn_cast<SourceRefAttr>(refs[0]);
}

DenseI64ArrayAttr operationRefForSource(SourceRefAttr source) {
  return DenseI64ArrayAttr::get(
      source.getContext(),
      {static_cast<int64_t>(source.getFunctionOrdinal()),
       static_cast<int64_t>(source.getBlockOrdinal()),
       static_cast<int64_t>(source.getOperationOrdinal())});
}

DictionaryAttr sourceAttributes(Operation *operation) {
  SmallVector<NamedAttribute> attributes;
  for (NamedAttribute attribute : operation->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name == kSourceRefsAttribute || name == kSelectedAttribute ||
        name == kFunctionOrdinalAttribute || name == kOperationRefAttribute ||
        name == kCoverageManifestAttribute ||
        name == kRegionResultSourcesAttribute) {
      continue;
    }
    attributes.push_back(attribute);
  }
  return DictionaryAttr::get(operation->getContext(), attributes);
}

DictionaryAttr normalizedOperationFingerprint(Operation *operation) {
  MLIRContext *context = operation->getContext();
  SmallVector<Attribute> resultTypes;
  for (Type type : operation->getResultTypes()) {
    resultTypes.push_back(TypeAttr::get(type));
  }
  NamedAttribute fields[] = {
      NamedAttribute(StringAttr::get(context, "attributes"),
                     sourceAttributes(operation)),
      NamedAttribute(
          StringAttr::get(context, "name"),
          StringAttr::get(context, operation->getName().getStringRef())),
      NamedAttribute(StringAttr::get(context, "result_types"),
                     ArrayAttr::get(context, resultTypes)),
  };
  return DictionaryAttr::get(context, fields);
}

Attribute valueAnchor(Value value) {
  MLIRContext *context = value.getContext();
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *parent = argument.getOwner()->getParentOp();
    auto function = dyn_cast<func::FuncOp>(parent);
    auto functionOrdinal =
        function
            ? function->getAttrOfType<IntegerAttr>(kFunctionOrdinalAttribute)
            : IntegerAttr{};
    if (!functionOrdinal ||
        &function.getBody().front() != argument.getOwner()) {
      auto ownerRef =
          parent
              ? parent->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute)
              : DenseI64ArrayAttr{};
      if (!ownerRef) {
        return {};
      }
      unsigned regionOrdinal = 0;
      unsigned blockOrdinal = 0;
      bool found = false;
      for (auto [regionIndex, region] : llvm::enumerate(parent->getRegions())) {
        for (auto [blockIndex, block] : llvm::enumerate(region)) {
          if (&block == argument.getOwner()) {
            regionOrdinal = regionIndex;
            blockOrdinal = blockIndex;
            found = true;
          }
        }
      }
      if (!found) {
        return {};
      }
      NamedAttribute nestedFields[] = {
          NamedAttribute(StringAttr::get(context, "owner"), ownerRef),
          NamedAttribute(
              StringAttr::get(context, "region"),
              IntegerAttr::get(IntegerType::get(context, 64), regionOrdinal)),
          NamedAttribute(
              StringAttr::get(context, "block"),
              IntegerAttr::get(IntegerType::get(context, 64), blockOrdinal)),
          NamedAttribute(StringAttr::get(context, "argument"),
                         IntegerAttr::get(IntegerType::get(context, 64),
                                          argument.getArgNumber())),
      };
      return DictionaryAttr::get(context, nestedFields);
    }
    NamedAttribute fields[] = {
        NamedAttribute(StringAttr::get(context, "argument"),
                       UnitAttr::get(context)),
        NamedAttribute(StringAttr::get(context, "function"), functionOrdinal),
        NamedAttribute(StringAttr::get(context, "ordinal"),
                       IntegerAttr::get(IntegerType::get(context, 64),
                                        argument.getArgNumber())),
    };
    return DictionaryAttr::get(context, fields);
  }

  auto result = dyn_cast<OpResult>(value);
  if (!result) {
    return {};
  }
  Operation *owner = result.getOwner();
  ArrayAttr refs = sourceRefs(owner);
  if (!refs) {
    refs = owner->getAttrOfType<ArrayAttr>(kRegionResultSourcesAttribute);
  }
  if (!refs || result.getResultNumber() >= refs.size()) {
    return {};
  }
  return refs[result.getResultNumber()];
}

bool hasOnlyRankedSupportedFloats(ValueRange values) {
  return llvm::all_of(values, [](Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type &&
           (type.getElementType().isF32() || type.getElementType().isBF16());
  });
}

Type elementType(Value value) {
  return cast<RankedTensorType>(value.getType()).getElementType();
}

bool hasEqualShapes(Operation *operation) {
  auto first = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = cast<RankedTensorType>(operation->getResult(0).getType());
  return first.getShape() == result.getShape();
}

bool hasValidBroadcastDimensions(Operation *operation,
                                 DenseI64ArrayAttr dimensions) {
  auto result = cast<RankedTensorType>(operation->getResult(0).getType());
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t dimension : dimensions.asArrayRef()) {
    if (dimension < 0 || dimension >= result.getRank() ||
        !seen.insert(dimension).second) {
      return false;
    }
  }
  return true;
}

bool isSupportedAffineBroadcast(Operation *operation) {
  if (operation->getNumOperands() != 1 || operation->getNumResults() != 1) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  auto dimensions =
      operation->getAttrOfType<DenseI64ArrayAttr>("broadcast_dimensions");
  DictionaryAttr attributes = sourceAttributes(operation);
  if (!input || !result || !input.getElementType().isF32() ||
      !result.getElementType().isF32() || !input.hasStaticShape() ||
      !result.hasStaticShape() || !dimensions ||
      dimensions.size() != static_cast<size_t>(input.getRank()) ||
      attributes.size() != 1 || !attributes.get("broadcast_dimensions") ||
      input == result || !hasValidBroadcastDimensions(operation, dimensions)) {
    return false;
  }
  unsigned expandingSingletons = 0;
  for (auto [inputDimension, resultDimension] :
       llvm::enumerate(dimensions.asArrayRef())) {
    const int64_t inputExtent = input.getDimSize(inputDimension);
    const int64_t resultExtent = result.getDimSize(resultDimension);
    if (inputExtent == resultExtent) {
      continue;
    }
    if (inputExtent != 1 || resultExtent <= 1) {
      return false;
    }
    ++expandingSingletons;
  }
  if (expandingSingletons == 0) {
    return true;
  }
  return input.getRank() == 2 && result.getRank() == 2 &&
         expandingSingletons == 1;
}

bool isCopyFreeSingletonReshape(Operation *operation) {
  if (operation->getNumOperands() != 1 || operation->getNumResults() != 1 ||
      !sourceAttributes(operation).empty()) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!input || !result || !input.getElementType().isF32() ||
      !result.getElementType().isF32() || !input.hasStaticShape() ||
      !result.hasStaticShape()) {
    return false;
  }
  SmallVector<int64_t> inputNonSingleton;
  SmallVector<int64_t> resultNonSingleton;
  llvm::copy_if(input.getShape(), std::back_inserter(inputNonSingleton),
                [](int64_t extent) { return extent != 1; });
  llvm::copy_if(result.getShape(), std::back_inserter(resultNonSingleton),
                [](int64_t extent) { return extent != 1; });
  return inputNonSingleton == resultNonSingleton && input != result;
}

bool hasDefaultDotPrecision(Operation *operation) {
  auto precision = operation->getAttrOfType<ArrayAttr>("precision_config");
  if (!precision) {
    return true;
  }
  return precision.size() == operation->getNumOperands() &&
         llvm::all_of(precision, [](Attribute attribute) {
           auto value = dyn_cast<stablehlo::PrecisionAttr>(attribute);
           return value && value.getValue() == stablehlo::Precision::DEFAULT;
         });
}

bool isSupportedStablehloReduce(Operation *operation) {
  if (operation->getName().getStringRef() !=
          stablehlo::ReduceOp::getOperationName() ||
      operation->getNumOperands() != 2 || operation->getNumResults() != 1 ||
      operation->getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation->getRegion(0))) {
    return false;
  }
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto init = dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  auto dimensions = operation->getAttrOfType<DenseI64ArrayAttr>("dimensions");
  DictionaryAttr reduceAttributes = sourceAttributes(operation);
  if (!input || input.getRank() == 0 || !input.getElementType().isF32() ||
      !init || init.getRank() != 0 || !init.getElementType().isF32() ||
      !result || !result.getElementType().isF32() || !dimensions ||
      dimensions.empty() || reduceAttributes.size() != 1 ||
      !reduceAttributes.get("dimensions")) {
    return false;
  }
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t dimension : dimensions.asArrayRef()) {
    if (dimension < 0 || dimension >= input.getRank() ||
        !seen.insert(dimension).second) {
      return false;
    }
  }
  SmallVector<int64_t> expected;
  for (int64_t dimension = 0; dimension < input.getRank(); ++dimension) {
    if (!seen.contains(dimension)) {
      expected.push_back(input.getDimSize(dimension));
    }
  }
  Block &body = operation->getRegion(0).front();
  if (result.getShape() != ArrayRef<int64_t>(expected) ||
      body.getNumArguments() != 2 ||
      !llvm::all_of(body.getArguments(),
                    [](BlockArgument argument) {
                      auto type =
                          dyn_cast<RankedTensorType>(argument.getType());
                      return type && type.getRank() == 0 &&
                             type.getElementType().isF32();
                    }) ||
      body.getOperations().size() != 2) {
    return false;
  }
  Operation &add = body.front();
  Operation &terminator = body.back();
  return add.getName().getStringRef() == stablehlo::AddOp::getOperationName() &&
         sourceAttributes(&add).empty() && add.getNumOperands() == 2 &&
         add.getNumResults() == 1 && add.getOperand(0) == body.getArgument(0) &&
         add.getOperand(1) == body.getArgument(1) &&
         add.getResult(0).getType() == body.getArgument(0).getType() &&
         terminator.getName().getStringRef() ==
             stablehlo::ReturnOp::getOperationName() &&
         sourceAttributes(&terminator).empty() &&
         terminator.getNumOperands() == 1 &&
         terminator.getOperand(0) == add.getResult(0);
}

bool isSupportedStablehlo(Operation *operation) {
  if (isSupportedStablehloReduce(operation)) {
    return true;
  }
  if (operation->getNumRegions() != 0 || !isMemoryEffectFree(operation) ||
      !hasOnlyRankedSupportedFloats(operation->getOperands()) ||
      !hasOnlyRankedSupportedFloats(operation->getResults())) {
    return false;
  }

  StringRef name = operation->getName().getStringRef();
  if (name == stablehlo::DotGeneralOp::getOperationName()) {
    if (operation->getNumOperands() != 2 || operation->getNumResults() != 1 ||
        operation->hasAttr("algorithm") || !hasDefaultDotPrecision(operation)) {
      return false;
    }
    return elementType(operation->getOperand(0)) ==
               elementType(operation->getOperand(1)) &&
           (elementType(operation->getOperand(0)).isF32() ||
            elementType(operation->getOperand(0)).isBF16()) &&
           elementType(operation->getResult(0)).isF32() &&
           static_cast<bool>(
               operation->getAttrOfType<stablehlo::DotDimensionNumbersAttr>(
                   "dot_dimension_numbers"));
  }
  if (name == stablehlo::ConvertOp::getOperationName()) {
    if (operation->getNumOperands() != 1 || operation->getNumResults() != 1 ||
        !hasEqualShapes(operation)) {
      return false;
    }
    Type input = elementType(operation->getOperand(0));
    Type result = elementType(operation->getResult(0));
    return (input.isF32() && result.isF32()) ||
           (input.isF32() && result.isBF16()) ||
           (input.isBF16() && result.isF32());
  }
  if (name == stablehlo::TanhOp::getOperationName() ||
      name == stablehlo::ExpOp::getOperationName() ||
      name == stablehlo::RsqrtOp::getOperationName() ||
      name == stablehlo::NegOp::getOperationName()) {
    return operation->getNumOperands() == 1 &&
           operation->getNumResults() == 1 &&
           !operation->hasAttr("result_accuracy") &&
           sourceAttributes(operation).empty() &&
           elementType(operation->getOperand(0)).isF32() &&
           operation->getOperand(0).getType() ==
               operation->getResult(0).getType();
  }
  if (name == stablehlo::AddOp::getOperationName() ||
      name == stablehlo::MulOp::getOperationName() ||
      name == stablehlo::SubtractOp::getOperationName() ||
      name == stablehlo::DivOp::getOperationName()) {
    return operation->getNumOperands() == 2 &&
           operation->getNumResults() == 1 &&
           elementType(operation->getOperand(0)).isF32() &&
           operation->getOperand(0).getType() ==
               operation->getOperand(1).getType() &&
           operation->getOperand(0).getType() ==
               operation->getResult(0).getType();
  }
  if (name == stablehlo::TransposeOp::getOperationName()) {
    auto permutation =
        operation->getAttrOfType<DenseI64ArrayAttr>("permutation");
    auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
    return operation->getNumOperands() == 1 &&
           operation->getNumResults() == 1 && permutation &&
           elementType(operation->getOperand(0)).isF32() &&
           permutation.size() == static_cast<size_t>(input.getRank());
  }
  if (name == stablehlo::ConstantOp::getOperationName()) {
    auto value = operation->getAttrOfType<DenseElementsAttr>("value");
    auto result = cast<RankedTensorType>(operation->getResult(0).getType());
    return operation->getNumOperands() == 0 &&
           operation->getNumResults() == 1 && result.getRank() == 0 &&
           result.getElementType().isF32() && value && value.isSplat();
  }
  if (name == stablehlo::BroadcastInDimOp::getOperationName()) {
    return isSupportedAffineBroadcast(operation);
  }
  if (name == stablehlo::ReshapeOp::getOperationName()) {
    return isCopyFreeSingletonReshape(operation);
  }
  return false;
}

SmallVector<CandidateComponent>
partitionSupportedInterval(ArrayRef<Operation *> interval) {
  llvm::SmallPtrSet<Operation *, 16> members(interval.begin(), interval.end());
  llvm::SmallPtrSet<Operation *, 16> visited;
  SmallVector<CandidateComponent> components;
  for (Operation *seed : interval) {
    if (!visited.insert(seed).second) {
      continue;
    }
    CandidateComponent component;
    SmallVector<Operation *> worklist{seed};
    while (!worklist.empty()) {
      Operation *operation = worklist.pop_back_val();
      component.operations.push_back(operation);
      for (Value operand : operation->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (producer && members.contains(producer) &&
            visited.insert(producer).second) {
          worklist.push_back(producer);
        }
      }
      for (Value result : operation->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (members.contains(user) && visited.insert(user).second) {
            worklist.push_back(user);
          }
        }
      }
    }
    llvm::sort(component.operations,
               [interval](Operation *left, Operation *right) {
                 auto position = [interval](Operation *operation) {
                   return llvm::find(interval, operation) - interval.begin();
                 };
                 return position(left) < position(right);
               });
    components.push_back(std::move(component));
  }
  llvm::DenseMap<Operation *, unsigned> componentFor;
  for (auto [componentOrdinal, component] : llvm::enumerate(components)) {
    for (Operation *operation : component.operations) {
      componentFor[operation] = componentOrdinal;
    }
  }
  SmallVector<CandidateComponent> sourceContiguousComponents;
  unsigned previousComponent = componentFor[interval.front()];
  sourceContiguousComponents.emplace_back();
  for (Operation *operation : interval) {
    unsigned component = componentFor[operation];
    if (component != previousComponent) {
      sourceContiguousComponents.emplace_back();
      previousComponent = component;
    }
    sourceContiguousComponents.back().operations.push_back(operation);
  }
  return sourceContiguousComponents;
}

SmallVector<CandidateComponent> candidateComponents(ModuleOp module) {
  SmallVector<CandidateComponent> components;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    for (Block &block : function.getBody()) {
      SmallVector<Operation *> interval;
      auto flushInterval = [&]() {
        if (interval.empty()) {
          return;
        }
        llvm::append_range(components, partitionSupportedInterval(interval));
        interval.clear();
      };
      for (Operation &operation : block) {
        if (isSupportedStablehlo(&operation)) {
          interval.push_back(&operation);
        } else {
          flushInterval();
        }
      }
      flushInterval();
    }
  }
  return components;
}

ArrayAttr operationSourceRefs(Operation *operation) {
  return sourceRefs(operation);
}

DictionaryAttr zeroResultRecord(Operation *operation) {
  MLIRContext *context = operation->getContext();
  SmallVector<Attribute> operands;
  operands.reserve(operation->getNumOperands());
  for (Value operand : operation->getOperands()) {
    Attribute anchor = valueAnchor(operand);
    if (!anchor) {
      return {};
    }
    operands.push_back(anchor);
  }
  NamedAttribute fields[] = {
      NamedAttribute(
          StringAttr::get(context, "classification"),
          StringAttr::get(context, operation->hasTrait<OpTrait::IsTerminator>()
                                       ? "terminator"
                                       : "zero_result_operation")),
      NamedAttribute(StringAttr::get(context, "fingerprint"),
                     normalizedOperationFingerprint(operation)),
      NamedAttribute(StringAttr::get(context, "operation_ref"),
                     operation->getAttr(kOperationRefAttribute)),
      NamedAttribute(StringAttr::get(context, "operands"),
                     ArrayAttr::get(context, operands)),
  };
  return DictionaryAttr::get(context, fields);
}

FailureOr<DictionaryAttr>
buildCoverageManifest(ModuleOp module, ArrayRef<CandidateComponent> components,
                      NumericalPolicy numerics, StringRef canonicalOptions,
                      StringRef canonicalTuning) {
  MLIRContext *context = module.getContext();
  llvm::SmallDenseSet<Attribute> selected;
  SmallVector<Attribute> selectedRegions;
  for (const CandidateComponent &component : components) {
    SmallVector<Attribute> refs;
    for (Operation *operation : component.operations) {
      LogicalResult nested =
          walkSourcePreorder(operation, [&](Operation *sourceOperation) {
            ArrayAttr operationRefs = operationSourceRefs(sourceOperation);
            if (sourceOperation->getNumResults() != 0 && !operationRefs) {
              sourceOperation->emitOpError(
                  "is missing structural source references");
              return failure();
            }
            if (operationRefs) {
              for (Attribute ref : operationRefs) {
                if (!selected.insert(ref).second) {
                  module.emitError("a source result belongs to two regions");
                  return failure();
                }
                refs.push_back(ref);
              }
            }
            return success();
          });
      if (failed(nested)) {
        return failure();
      }
    }
    selectedRegions.push_back(ArrayAttr::get(context, refs));
  }

  SmallVector<Attribute> complete;
  SmallVector<Attribute> excluded;
  SmallVector<Attribute> zeroResultOperations;
  SmallVector<Attribute> functionResults;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto functionOrdinal =
        function->getAttrOfType<IntegerAttr>(kFunctionOrdinalAttribute);
    LogicalResult sourceWalk =
        walkSourcePreorder(function.getBody(), [&](Operation *operation) {
          if (operation->getNumResults() == 0) {
            DictionaryAttr record = zeroResultRecord(operation);
            if (!record) {
              operation->emitOpError(
                  "has an operand without a structural source anchor");
              return failure();
            }
            zeroResultOperations.push_back(record);
            return success();
          }
          ArrayAttr operationRefs = operationSourceRefs(operation);
          if (!operationRefs) {
            operation->emitOpError("is missing structural source references");
            return failure();
          }
          SmallVector<Attribute> operandAnchors;
          for (Value operand : operation->getOperands()) {
            Attribute anchor = valueAnchor(operand);
            if (!anchor) {
              operation->emitOpError(
                  "has an operand without a structural source anchor");
              return failure();
            }
            operandAnchors.push_back(anchor);
          }
          for (Attribute ref : operationRefs) {
            complete.push_back(ref);
            if (selected.contains(ref)) {
              continue;
            }
            NamedAttribute fields[] = {
                NamedAttribute(StringAttr::get(context, "fingerprint"),
                               normalizedOperationFingerprint(operation)),
                NamedAttribute(StringAttr::get(context, "operands"),
                               ArrayAttr::get(context, operandAnchors)),
                NamedAttribute(StringAttr::get(context, "source"), ref),
                NamedAttribute(
                    StringAttr::get(context, "reason"),
                    StringAttr::get(
                        context,
                        operation->getParentOp() &&
                                !isa<func::FuncOp>(operation->getParentOp()) &&
                                operation->getParentOp()->getNumRegions() != 0
                            ? "enclosing_region_excluded"
                            : "unsupported_operation")),
            };
            excluded.push_back(DictionaryAttr::get(context, fields));
          }
          return success();
        });
    if (failed(sourceWalk)) {
      return failure();
    }
    auto returnOp = dyn_cast<func::ReturnOp>(function.getBody().front().back());
    if (!returnOp) {
      function.emitOpError(
          "the first offline slice requires a direct func.return");
      return failure();
    }
    for (auto [ordinal, operand] : llvm::enumerate(returnOp.getOperands())) {
      Attribute anchor = valueAnchor(operand);
      if (!anchor) {
        returnOp.emitOpError("has a result without a structural source anchor");
        return failure();
      }
      NamedAttribute fields[] = {
          NamedAttribute(StringAttr::get(context, "function"), functionOrdinal),
          NamedAttribute(
              StringAttr::get(context, "result"),
              IntegerAttr::get(IntegerType::get(context, 64), ordinal)),
          NamedAttribute(StringAttr::get(context, "anchor"), anchor),
      };
      functionResults.push_back(DictionaryAttr::get(context, fields));
    }
  }

  std::string tuningDigest = sha256(canonicalTuning);
  NamedAttribute fields[] = {
      NamedAttribute(StringAttr::get(context, "version"),
                     IntegerAttr::get(IntegerType::get(context, 64), 2)),
      NamedAttribute(StringAttr::get(context, "policy"),
                     StringAttr::get(context, policyName(numerics))),
      NamedAttribute(StringAttr::get(context, "policy_digest"),
                     StringAttr::get(context, sha256(canonicalOptions))),
      NamedAttribute(StringAttr::get(context, "canonical_options"),
                     StringAttr::get(context, canonicalOptions)),
      NamedAttribute(StringAttr::get(context, "canonical_tuning"),
                     StringAttr::get(context, canonicalTuning)),
      NamedAttribute(StringAttr::get(context, kManifestComplete),
                     ArrayAttr::get(context, complete)),
      NamedAttribute(StringAttr::get(context, kManifestSelectedRegions),
                     ArrayAttr::get(context, selectedRegions)),
      NamedAttribute(StringAttr::get(context, kManifestExcluded),
                     ArrayAttr::get(context, excluded)),
      NamedAttribute(StringAttr::get(context, kManifestZeroResultOperations),
                     ArrayAttr::get(context, zeroResultOperations)),
      NamedAttribute(StringAttr::get(context, kManifestFunctionResults),
                     ArrayAttr::get(context, functionResults)),
      NamedAttribute(StringAttr::get(context, "tuning_digest"),
                     StringAttr::get(context, tuningDigest)),
  };
  return DictionaryAttr::get(context, fields);
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
        operation.setAttr(
            kOperationRefAttribute,
            DenseI64ArrayAttr::get(operation.getContext(),
                                   {static_cast<int64_t>(functionOrdinal),
                                    static_cast<int64_t>(blockOrdinal),
                                    static_cast<int64_t>(operationOrdinal)}));
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

struct AnnotateSourcePass
    : impl::ShuttleAnnotateSourcePassBase<AnnotateSourcePass> {
  void runOnOperation() override {
    uint64_t functionOrdinal = 0;
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      function->setAttr(
          kFunctionOrdinalAttribute,
          IntegerAttr::get(IntegerType::get(function.getContext(), 64),
                           functionOrdinal));
      uint64_t nextBlockOrdinal = 0;
      annotateRegion(function.getBody(), functionOrdinal++, nextBlockOrdinal);
    }
  }
};

struct FormStructuralRegionsPass
    : impl::ShuttleFormStructuralRegionsPassBase<FormStructuralRegionsPass> {
  FormStructuralRegionsPass() = default;
  FormStructuralRegionsPass(NumericalPolicy numerics,
                            std::string canonicalOptions,
                            std::string canonicalTuning)
      : numerics(numerics), canonicalOptions(std::move(canonicalOptions)),
        canonicalTuning(std::move(canonicalTuning)) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr(kCoverageManifestAttribute)) {
      module.emitError("already has a Shuttle coverage manifest");
      signalPassFailure();
      return;
    }
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (!function->hasAttr(kFunctionOrdinalAttribute)) {
        function.emitOpError(
            "requires shuttle-annotate-source before region formation");
        signalPassFailure();
        return;
      }
      if (!llvm::hasSingleElement(function.getBody())) {
        function.emitOpError(
            "the first offline slice requires one function block");
        signalPassFailure();
        return;
      }
    }

    SmallVector<CandidateComponent> components = candidateComponents(module);
    FailureOr<DictionaryAttr> manifest = buildCoverageManifest(
        module, components, numerics, canonicalOptions, canonicalTuning);
    if (failed(manifest)) {
      signalPassFailure();
      return;
    }
    module->setAttr(kCoverageManifestAttribute, *manifest);

    for (const CandidateComponent &component : components) {
      if (failed(materializeRegion(component))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult materializeRegion(const CandidateComponent &component) {
    assert(!component.operations.empty());
    MLIRContext *context = component.operations.front()->getContext();
    llvm::SmallPtrSet<Operation *, 16> members(component.operations.begin(),
                                               component.operations.end());
    llvm::SmallDenseSet<Value> seenInputs;
    SmallVector<Value> inputs;
    for (Operation *operation : component.operations) {
      for (Value operand : operation->getOperands()) {
        if (!members.contains(operand.getDefiningOp()) &&
            seenInputs.insert(operand).second) {
          inputs.push_back(operand);
        }
      }
    }

    SmallVector<Value> outputs;
    for (Operation *operation : component.operations) {
      for (Value result : operation->getResults()) {
        if (llvm::any_of(result.getUsers(), [&](Operation *user) {
              return !members.contains(user);
            })) {
          outputs.push_back(result);
        }
      }
    }

    SmallVector<Attribute> declaredSources;
    for (Operation *operation : component.operations) {
      LogicalResult nested =
          walkSourcePreorder(operation, [&](Operation *sourceOperation) {
            ArrayAttr refs = operationSourceRefs(sourceOperation);
            if (sourceOperation->getNumResults() != 0 && !refs) {
              sourceOperation->emitOpError(
                  "is missing structural source references");
              return failure();
            }
            if (refs) {
              llvm::append_range(declaredSources, refs);
              sourceOperation->setAttr(kSelectedAttribute,
                                       UnitAttr::get(context));
            }
            return success();
          });
      if (failed(nested)) {
        return failure();
      }
    }
    SmallVector<Attribute> outputSources;
    for (Value output : outputs) {
      Attribute anchor = valueAnchor(output);
      if (!anchor || !isa<SourceRefAttr>(anchor)) {
        return component.operations.front()->emitOpError(
            "has a live-out without a source-result anchor");
      }
      outputSources.push_back(anchor);
    }

    OpBuilder builder(component.operations.front());
    OperationState state(component.operations.front()->getLoc(),
                         RegionOp::getOperationName());
    state.addOperands(inputs);
    state.addTypes(TypeRange(outputs));
    state.addAttribute("policy", NumericalPolicyAttr::get(context, numerics));
    state.addAttribute("source_refs", ArrayAttr::get(context, declaredSources));
    state.addAttribute(kRegionResultSourcesAttribute,
                       ArrayAttr::get(context, outputSources));
    state.addRegion();
    auto region = cast<RegionOp>(builder.create(state));
    Block *body = new Block();
    region.getBody().push_back(body);
    for (Value input : inputs) {
      body->addArgument(input.getType(), input.getLoc());
    }

    for (auto [input, argument] :
         llvm::zip_equal(inputs, body->getArguments())) {
      for (Operation *operation : component.operations) {
        for (OpOperand &operand : operation->getOpOperands()) {
          if (operand.get() == input) {
            operand.set(argument);
          }
        }
      }
    }
    for (Operation *operation : component.operations) {
      operation->moveBefore(body, body->end());
    }

    builder.setInsertionPointToEnd(body);
    OperationState yieldState(region.getLoc(), YieldOp::getOperationName());
    yieldState.addOperands(outputs);
    builder.create(yieldState);

    for (auto [output, replacement] :
         llvm::zip_equal(outputs, region.getResults())) {
      output.replaceUsesWithIf(replacement, [&](OpOperand &use) {
        return !region->isAncestor(use.getOwner());
      });
    }
    return success();
  }

  NumericalPolicy numerics = NumericalPolicy::SourceOrdered;
  std::string canonicalOptions = ShuttlePipelineOptions{}.canonicalOptions;
  std::string canonicalTuning = ShuttlePipelineOptions{}.canonicalTuning;
};

ArrayAttr affineMapsAttr(MLIRContext *context, ArrayRef<AffineMap> maps) {
  SmallVector<Attribute> attributes;
  attributes.reserve(maps.size());
  for (AffineMap map : maps) {
    attributes.push_back(AffineMapAttr::get(map));
  }
  return ArrayAttr::get(context, attributes);
}

FailureOr<SmallVector<AffineMap>> dotIndexingMaps(Operation *operation) {
  MLIRContext *context = operation->getContext();
  auto dimensions =
      operation->getAttrOfType<stablehlo::DotDimensionNumbersAttr>(
          "dot_dimension_numbers");
  auto lhsType = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto rhsType = dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
  auto resultType =
      dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!dimensions || !lhsType || !rhsType || !resultType ||
      dimensions.getLhsBatchingDimensions().size() !=
          dimensions.getRhsBatchingDimensions().size() ||
      dimensions.getLhsContractingDimensions().size() !=
          dimensions.getRhsContractingDimensions().size()) {
    operation->emitOpError("has unsupported dot dimension metadata");
    return failure();
  }

  SmallVector<int64_t> lhsDomain(lhsType.getRank(), -1);
  SmallVector<int64_t> rhsDomain(rhsType.getRank(), -1);
  int64_t nextDomain = 0;
  for (auto [lhs, rhs] :
       llvm::zip_equal(dimensions.getLhsBatchingDimensions(),
                       dimensions.getRhsBatchingDimensions())) {
    lhsDomain[lhs] = nextDomain;
    rhsDomain[rhs] = nextDomain++;
  }
  for (int64_t dimension = 0; dimension < lhsType.getRank(); ++dimension) {
    if (lhsDomain[dimension] < 0 &&
        !llvm::is_contained(dimensions.getLhsContractingDimensions(),
                            dimension)) {
      lhsDomain[dimension] = nextDomain++;
    }
  }
  for (int64_t dimension = 0; dimension < rhsType.getRank(); ++dimension) {
    if (rhsDomain[dimension] < 0 &&
        !llvm::is_contained(dimensions.getRhsContractingDimensions(),
                            dimension)) {
      rhsDomain[dimension] = nextDomain++;
    }
  }
  const int64_t parallelDimensions = nextDomain;
  for (auto [lhs, rhs] :
       llvm::zip_equal(dimensions.getLhsContractingDimensions(),
                       dimensions.getRhsContractingDimensions())) {
    lhsDomain[lhs] = nextDomain;
    rhsDomain[rhs] = nextDomain++;
  }
  if (llvm::is_contained(lhsDomain, -1) || llvm::is_contained(rhsDomain, -1) ||
      resultType.getRank() != parallelDimensions) {
    operation->emitOpError("has an incomplete normalized dot domain");
    return failure();
  }

  auto mapFor = [&](ArrayRef<int64_t> domainPositions) {
    SmallVector<AffineExpr> expressions;
    expressions.reserve(domainPositions.size());
    for (int64_t position : domainPositions) {
      expressions.push_back(getAffineDimExpr(position, context));
    }
    return AffineMap::get(nextDomain, 0, expressions, context);
  };
  SmallVector<int64_t> resultDomain(parallelDimensions);
  for (int64_t position = 0; position < parallelDimensions; ++position) {
    resultDomain[position] = position;
  }
  return SmallVector<AffineMap>{mapFor(lhsDomain), mapFor(rhsDomain),
                                mapFor(resultDomain)};
}

FailureOr<SmallVector<AffineMap>> mapIndexingMaps(Operation *operation) {
  MLIRContext *context = operation->getContext();
  auto resultType =
      dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!resultType) {
    operation->emitOpError("requires a ranked map result");
    return failure();
  }
  int64_t rank = resultType.getRank();
  AffineMap identity = AffineMap::getMultiDimIdentityMap(rank, context);
  if (operation->getName().getStringRef() ==
      stablehlo::ReshapeOp::getOperationName()) {
    auto inputType = cast<RankedTensorType>(operation->getOperand(0).getType());
    SmallVector<unsigned> resultNonSingleton;
    for (auto [position, extent] : llvm::enumerate(resultType.getShape())) {
      if (extent != 1) {
        resultNonSingleton.push_back(position);
      }
    }
    SmallVector<AffineExpr> inputExpressions;
    unsigned nextNonSingleton = 0;
    for (int64_t extent : inputType.getShape()) {
      if (extent == 1) {
        inputExpressions.push_back(getAffineConstantExpr(0, context));
        continue;
      }
      if (nextNonSingleton >= resultNonSingleton.size()) {
        operation->emitOpError("has incompatible singleton reshape shapes");
        return failure();
      }
      inputExpressions.push_back(
          getAffineDimExpr(resultNonSingleton[nextNonSingleton++], context));
    }
    if (nextNonSingleton != resultNonSingleton.size()) {
      operation->emitOpError("has incompatible singleton reshape shapes");
      return failure();
    }
    return SmallVector<AffineMap>{
        AffineMap::get(rank, 0, inputExpressions, context), identity};
  }
  if (operation->getName().getStringRef() ==
      stablehlo::BroadcastInDimOp::getOperationName()) {
    auto dimensions =
        operation->getAttrOfType<DenseI64ArrayAttr>("broadcast_dimensions");
    if (!dimensions) {
      operation->emitOpError("has no broadcast dimension metadata");
      return failure();
    }
    SmallVector<AffineExpr> inputExpressions;
    if (!hasValidBroadcastDimensions(operation, dimensions)) {
      operation->emitOpError("requires unique in-range broadcast dimensions");
      return failure();
    }
    for (int64_t dimension : dimensions.asArrayRef()) {
      if (dimension < 0 || dimension >= rank) {
        operation->emitOpError("has an out-of-range broadcast dimension");
        return failure();
      }
      AffineExpr expression = getAffineDimExpr(dimension, context);
      auto inputType =
          cast<RankedTensorType>(operation->getOperand(0).getType());
      if (inputType.getDimSize(inputExpressions.size()) == 1 &&
          resultType.getDimSize(dimension) > 1) {
        expression = expression.floorDiv(resultType.getDimSize(dimension));
      }
      inputExpressions.push_back(expression);
    }
    return SmallVector<AffineMap>{
        AffineMap::get(rank, 0, inputExpressions, context), identity};
  }
  if (operation->getName().getStringRef() !=
      stablehlo::TransposeOp::getOperationName()) {
    SmallVector<AffineMap> maps(operation->getNumOperands() + 1, identity);
    return maps;
  }

  auto permutation = operation->getAttrOfType<DenseI64ArrayAttr>("permutation");
  auto inputType =
      dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  if (!permutation || !inputType || inputType.getRank() != rank) {
    operation->emitOpError("has an invalid transpose permutation");
    return failure();
  }
  SmallVector<int64_t> inverse(rank, -1);
  for (auto [outputDimension, inputDimension] :
       llvm::enumerate(permutation.asArrayRef())) {
    if (inputDimension < 0 || inputDimension >= rank ||
        inverse[inputDimension] >= 0) {
      operation->emitOpError("has a non-permutation transpose map");
      return failure();
    }
    inverse[inputDimension] = outputDimension;
  }
  SmallVector<AffineExpr> inputExpressions;
  for (int64_t outputDimension : inverse) {
    inputExpressions.push_back(getAffineDimExpr(outputDimension, context));
  }
  return SmallVector<AffineMap>{
      AffineMap::get(rank, 0, inputExpressions, context), identity};
}

Operation *createScalarOperation(OpBuilder &builder, Operation *operation,
                                 ValueRange arguments) {
  Location location = operation->getLoc();
  StringRef stablehloName = operation->getName().getStringRef();
  StringRef scalarName;
  if (stablehloName == stablehlo::TanhOp::getOperationName()) {
    scalarName = math::TanhOp::getOperationName();
  } else if (stablehloName == stablehlo::ExpOp::getOperationName()) {
    scalarName = math::ExpOp::getOperationName();
  } else if (stablehloName == stablehlo::RsqrtOp::getOperationName()) {
    scalarName = math::RsqrtOp::getOperationName();
  } else if (stablehloName == stablehlo::NegOp::getOperationName()) {
    scalarName = arith::NegFOp::getOperationName();
  } else if (stablehloName == stablehlo::MulOp::getOperationName()) {
    scalarName = arith::MulFOp::getOperationName();
  } else if (stablehloName == stablehlo::AddOp::getOperationName()) {
    scalarName = arith::AddFOp::getOperationName();
  } else if (stablehloName == stablehlo::SubtractOp::getOperationName()) {
    scalarName = arith::SubFOp::getOperationName();
  } else if (stablehloName == stablehlo::DivOp::getOperationName()) {
    scalarName = arith::DivFOp::getOperationName();
  } else if (stablehloName == stablehlo::ConvertOp::getOperationName()) {
    OperationState state(location, ScalarConvertOp::getOperationName());
    state.addOperands(arguments);
    Type input = cast<RankedTensorType>(operation->getOperand(0).getType())
                     .getElementType();
    Type result = cast<RankedTensorType>(operation->getResult(0).getType())
                      .getElementType();
    ScalarConvertSemantics semantics =
        input.isF32() && result.isBF16()
            ? ScalarConvertSemantics::RoundNearestEven
            : ScalarConvertSemantics::Exact;
    state.addTypes(result);
    state.addAttribute("semantics", ScalarConvertSemanticsAttr::get(
                                        operation->getContext(), semantics));
    return builder.create(state);
  } else if (stablehloName == stablehlo::ConstantOp::getOperationName()) {
    auto value = operation->getAttrOfType<DenseElementsAttr>("value");
    if (!value || !value.isSplat()) {
      return nullptr;
    }
    OperationState state(location, arith::ConstantOp::getOperationName());
    Type result = cast<RankedTensorType>(operation->getResult(0).getType())
                      .getElementType();
    state.addTypes(result);
    state.addAttribute("value", value.getSplatValue<Attribute>());
    return builder.create(state);
  } else {
    return nullptr;
  }
  OperationState state(location, scalarName);
  state.addOperands(arguments);
  state.addTypes(cast<RankedTensorType>(operation->getResult(0).getType())
                     .getElementType());
  return builder.create(state);
}

LogicalResult convertMapOperation(Operation *operation) {
  FailureOr<SmallVector<AffineMap>> maps = mapIndexingMaps(operation);
  if (failed(maps)) {
    return failure();
  }
  SourceRefAttr source = singleSourceRef(operation);
  if (!source) {
    return operation->emitOpError("requires exactly one source result");
  }

  StringRef stablehloName = operation->getName().getStringRef();
  OpBuilder builder(operation);
  OperationState state(operation->getLoc(), MapOp::getOperationName());
  state.addOperands(operation->getOperands());
  state.addTypes(operation->getResultTypes());
  state.addAttribute("indexing_maps",
                     affineMapsAttr(operation->getContext(), *maps));
  MapSemantics semantics =
      llvm::StringSwitch<MapSemantics>(stablehloName)
          .Case(stablehlo::BroadcastInDimOp::getOperationName(),
                MapSemantics::BroadcastInDim)
          .Case(stablehlo::ReshapeOp::getOperationName(), MapSemantics::Reshape)
          .Case(stablehlo::TransposeOp::getOperationName(),
                MapSemantics::Transpose)
          .Default(MapSemantics::Pointwise);
  state.addAttribute("semantics",
                     MapSemanticsAttr::get(operation->getContext(), semantics));
  state.addAttribute("source", source);
  state.addRegion();
  auto map = cast<MapOp>(builder.create(state));

  Block *body = new Block();
  map.getBody().push_back(body);
  for (Value input : operation->getOperands()) {
    auto inputType = cast<RankedTensorType>(input.getType());
    body->addArgument(inputType.getElementType(), input.getLoc());
  }
  builder.setInsertionPointToEnd(body);
  Value scalarResult;
  if (stablehloName == stablehlo::TransposeOp::getOperationName() ||
      stablehloName == stablehlo::BroadcastInDimOp::getOperationName() ||
      stablehloName == stablehlo::ReshapeOp::getOperationName()) {
    scalarResult = body->getArgument(0);
  } else {
    Operation *scalar =
        createScalarOperation(builder, operation, body->getArguments());
    if (!scalar) {
      return operation->emitOpError("has no scalar Map lowering");
    }
    scalarResult = scalar->getResult(0);
  }
  OperationState yieldState(operation->getLoc(), YieldOp::getOperationName());
  yieldState.addOperands(scalarResult);
  builder.create(yieldState);

  operation->getResult(0).replaceAllUsesWith(map.getResult(0));
  operation->erase();
  return success();
}

LogicalResult convertContractOperation(Operation *operation) {
  FailureOr<SmallVector<AffineMap>> maps = dotIndexingMaps(operation);
  if (failed(maps)) {
    return failure();
  }
  SourceRefAttr source = singleSourceRef(operation);
  if (!source) {
    return operation->emitOpError("requires exactly one source result");
  }
  const unsigned domainRank = maps->front().getNumDims();
  const unsigned parallelRank = maps->back().getNumResults();
  OpBuilder builder(operation);
  SmallVector<Attribute> iteratorKinds;
  for (unsigned dimension = 0; dimension < domainRank; ++dimension) {
    iteratorKinds.push_back(builder.getStringAttr(
        dimension < parallelRank ? "parallel" : "reduction"));
  }
  SmallVector<Attribute> precision(operation->getNumOperands(),
                                   builder.getStringAttr("DEFAULT"));
  auto resultType = cast<RankedTensorType>(operation->getResult(0).getType());

  OperationState state(operation->getLoc(), ContractOp::getOperationName());
  state.addOperands(operation->getOperands());
  state.addTypes(operation->getResultTypes());
  state.addAttribute("indexing_maps",
                     affineMapsAttr(operation->getContext(), *maps));
  state.addAttribute("iterator_kinds",
                     ArrayAttr::get(operation->getContext(), iteratorKinds));
  state.addAttribute(
      "accumulator_types",
      ArrayAttr::get(operation->getContext(),
                     {TypeAttr::get(resultType.getElementType())}));
  state.addAttribute("precision",
                     ArrayAttr::get(operation->getContext(), precision));
  state.addAttribute("algorithm", builder.getStringAttr("dot_general"));
  state.addAttribute("source", source);
  Operation *contract = builder.create(state);
  operation->getResult(0).replaceAllUsesWith(contract->getResult(0));
  operation->erase();
  return success();
}

LogicalResult convertReduceOperation(Operation *operation) {
  if (!isSupportedStablehloReduce(operation)) {
    return operation->emitOpError(
        "is outside the first Fold conversion contract");
  }
  SourceRefAttr source = singleSourceRef(operation);
  auto ownerRef =
      operation->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  if (!source || !ownerRef) {
    return operation->emitOpError(
        "requires one result source and one owner operation reference");
  }
  Block &sourceBody = operation->getRegion(0).front();
  Operation &sourceAdd = sourceBody.front();
  Operation &sourceReturn = sourceBody.back();
  ArrayAttr addRefs = sourceRefs(&sourceAdd);
  auto addOwner =
      sourceAdd.getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  auto returnOwner =
      sourceReturn.getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  if (!addRefs || addRefs.size() != 1 || !addOwner || !returnOwner) {
    return operation->emitOpError("has incomplete nested reducer provenance");
  }

  OpBuilder builder(operation);
  OperationState state(operation->getLoc(), FoldOp::getOperationName());
  state.addOperands(operation->getOperands());
  state.addTypes(operation->getResultTypes());
  state.addAttribute("operandSegmentSizes",
                     DenseI32ArrayAttr::get(operation->getContext(), {1, 1}));
  state.addAttribute("reduction_dimensions", operation->getAttr("dimensions"));
  state.addAttribute("accumulator_types",
                     ArrayAttr::get(operation->getContext(),
                                    {TypeAttr::get(builder.getF32Type())}));
  state.addAttribute("order_free", builder.getBoolAttr(true));
  state.addAttribute("source", source);
  state.addAttribute(kOperationRefAttribute, ownerRef);
  state.addRegion();
  auto fold = cast<FoldOp>(builder.create(state));

  Block *body = new Block();
  fold.getCombiner().push_back(body);
  body->addArgument(builder.getF32Type(), sourceBody.getArgument(0).getLoc());
  body->addArgument(builder.getF32Type(), sourceBody.getArgument(1).getLoc());
  builder.setInsertionPointToEnd(body);
  OperationState addState(sourceAdd.getLoc(),
                          arith::AddFOp::getOperationName());
  addState.addOperands(body->getArguments());
  addState.addTypes(builder.getF32Type());
  addState.addAttribute(kSourceRefsAttribute, addRefs);
  addState.addAttribute(kOperationRefAttribute, addOwner);
  Operation *add = builder.create(addState);
  OperationState yieldState(sourceReturn.getLoc(), YieldOp::getOperationName());
  yieldState.addOperands(add->getResult(0));
  yieldState.addAttribute(kOperationRefAttribute, returnOwner);
  builder.create(yieldState);

  operation->getResult(0).replaceAllUsesWith(fold.getResult(0));
  operation->erase();
  return success();
}

struct ConvertStablehloToAlgebraPass
    : impl::ShuttleConvertStablehloToAlgebraPassBase<
          ConvertStablehloToAlgebraPass> {
  void runOnOperation() override {
    if (!getOperation()->hasAttr(kCoverageManifestAttribute)) {
      getOperation().emitError(
          "requires shuttle-form-structural-regions before conversion");
      signalPassFailure();
      return;
    }
    SmallVector<RegionOp> regions;
    getOperation().walk([&](RegionOp region) { regions.push_back(region); });
    for (RegionOp region : regions) {
      SmallVector<Operation *> sourceOperations;
      for (Operation &operation : region.getBody().front()) {
        if (!isa<YieldOp>(operation)) {
          sourceOperations.push_back(&operation);
        }
      }
      for (Operation *operation : sourceOperations) {
        StringRef name = operation->getName().getStringRef();
        LogicalResult result =
            name == stablehlo::ReduceOp::getOperationName()
                ? convertReduceOperation(operation)
            : name == stablehlo::DotGeneralOp::getOperationName()
                ? convertContractOperation(operation)
                : convertMapOperation(operation);
        if (failed(result)) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};

LogicalResult verifyRegionLocalCoverage(ModuleOp module) {
  WalkResult result = module.walk([&](RegionOp region) {
    llvm::SmallDenseSet<Attribute> declaredSources;
    llvm::SmallDenseSet<Attribute> representedSources;
    for (Attribute source : region.getSourceRefs()) {
      auto sourceRef = dyn_cast<SourceRefAttr>(source);
      if (!sourceRef || !declaredSources.insert(sourceRef).second) {
        region.emitOpError("requires unique #shuttle.source_ref entries");
        return WalkResult::interrupt();
      }
    }
    WalkResult nestedResult = region.getBody().walk([&](Operation *operation) {
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
      } else if (operation->getParentOfType<FoldOp>()) {
        ArrayAttr refs = sourceRefs(operation);
        if (!refs) {
          return WalkResult::advance();
        }
        for (Attribute nestedSource : refs) {
          if (!declaredSources.contains(nestedSource) ||
              !representedSources.insert(nestedSource).second) {
            operation->emitOpError("has invalid nested Fold source coverage");
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      } else {
        return WalkResult::advance();
      }
      if (!declaredSources.contains(source)) {
        operation->emitOpError(
            "source reference is absent from the enclosing shuttle.region");
        return WalkResult::interrupt();
      }
      if (!representedSources.insert(source).second) {
        operation->emitOpError("duplicates region-local source coverage");
        return WalkResult::interrupt();
      }
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
  return success(!result.wasInterrupted());
}

llvm::SmallDenseSet<Attribute> attributeSet(ArrayAttr attributes) {
  llvm::SmallDenseSet<Attribute> result;
  if (attributes) {
    result.insert(attributes.begin(), attributes.end());
  }
  return result;
}

bool sameAttributeSet(const llvm::SmallDenseSet<Attribute> &left,
                      const llvm::SmallDenseSet<Attribute> &right) {
  return left.size() == right.size() &&
         llvm::all_of(left, [&](Attribute attribute) {
           return right.contains(attribute);
         });
}

FailureOr<llvm::SmallDenseSet<Attribute>>
selectedManifestSources(DictionaryAttr manifest) {
  auto groups = manifest.getAs<ArrayAttr>(kManifestSelectedRegions);
  if (!groups) {
    return failure();
  }
  llvm::SmallDenseSet<Attribute> selected;
  for (Attribute groupAttribute : groups) {
    auto group = dyn_cast<ArrayAttr>(groupAttribute);
    if (!group) {
      return failure();
    }
    for (Attribute source : group) {
      if (!isa<SourceRefAttr>(source) || !selected.insert(source).second) {
        return failure();
      }
    }
  }
  return selected;
}

FailureOr<llvm::SmallDenseSet<Attribute>>
excludedManifestSources(DictionaryAttr manifest) {
  auto records = manifest.getAs<ArrayAttr>(kManifestExcluded);
  if (!records) {
    return failure();
  }
  llvm::SmallDenseSet<Attribute> excluded;
  for (Attribute recordAttribute : records) {
    auto record = dyn_cast<DictionaryAttr>(recordAttribute);
    auto source =
        record ? record.getAs<SourceRefAttr>("source") : SourceRefAttr{};
    auto reason = record ? record.getAs<StringAttr>("reason") : StringAttr{};
    if (!source || !reason ||
        (reason.getValue() != "unsupported_operation" &&
         reason.getValue() != "enclosing_region_excluded") ||
        !excluded.insert(source).second) {
      return failure();
    }
  }
  return excluded;
}

FailureOr<ArrayAttr> currentExcludedRecords(ModuleOp module,
                                            DictionaryAttr manifest) {
  auto manifestRecords = manifest.getAs<ArrayAttr>(kManifestExcluded);
  if (!manifestRecords) {
    return failure();
  }
  llvm::SmallDenseMap<Attribute, StringAttr> reasons;
  for (Attribute recordAttribute : manifestRecords) {
    auto record = dyn_cast<DictionaryAttr>(recordAttribute);
    auto source =
        record ? record.getAs<SourceRefAttr>("source") : SourceRefAttr{};
    auto reason = record ? record.getAs<StringAttr>("reason") : StringAttr{};
    if (!source || !reason || !reasons.try_emplace(source, reason).second) {
      return failure();
    }
  }

  SmallVector<Attribute> records;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    LogicalResult walk =
        walkSourcePreorder(function.getBody(), [&](Operation *operation) {
          ArrayAttr refs = sourceRefs(operation);
          if (!refs) {
            return success();
          }
          SmallVector<Attribute> operandAnchors;
          for (Value operand : operation->getOperands()) {
            Attribute anchor = valueAnchor(operand);
            if (!anchor) {
              return failure();
            }
            operandAnchors.push_back(anchor);
          }
          for (Attribute source : refs) {
            auto reason = reasons.find(source);
            if (reason == reasons.end()) {
              continue;
            }
            NamedAttribute fields[] = {
                NamedAttribute(
                    StringAttr::get(module.getContext(), "fingerprint"),
                    normalizedOperationFingerprint(operation)),
                NamedAttribute(
                    StringAttr::get(module.getContext(), "operands"),
                    ArrayAttr::get(module.getContext(), operandAnchors)),
                NamedAttribute(StringAttr::get(module.getContext(), "source"),
                               source),
                NamedAttribute(StringAttr::get(module.getContext(), "reason"),
                               reason->second),
            };
            records.push_back(DictionaryAttr::get(module.getContext(), fields));
          }
          return success();
        });
    if (failed(walk)) {
      return failure();
    }
  }
  return ArrayAttr::get(module.getContext(), records);
}

FailureOr<ArrayAttr> currentZeroResultRecords(ModuleOp module) {
  SmallVector<Attribute> records;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    LogicalResult walk =
        walkSourcePreorder(function.getBody(), [&](Operation *operation) {
          if (operation->getNumResults() != 0 ||
              !operation->hasAttr(kOperationRefAttribute)) {
            return success();
          }
          DictionaryAttr record = zeroResultRecord(operation);
          if (!record) {
            return failure();
          }
          records.push_back(record);
          return success();
        });
    if (failed(walk)) {
      return failure();
    }
  }
  return ArrayAttr::get(module.getContext(), records);
}

FailureOr<ArrayAttr> currentFunctionResultRecords(ModuleOp module) {
  SmallVector<Attribute> records;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto functionOrdinal =
        function->getAttrOfType<IntegerAttr>(kFunctionOrdinalAttribute);
    if (!functionOrdinal || !llvm::hasSingleElement(function.getBody())) {
      return failure();
    }
    auto returnOp = dyn_cast<func::ReturnOp>(function.getBody().front().back());
    if (!returnOp) {
      return failure();
    }
    for (auto [ordinal, operand] : llvm::enumerate(returnOp.getOperands())) {
      Attribute anchor = valueAnchor(operand);
      if (!anchor) {
        return failure();
      }
      NamedAttribute fields[] = {
          NamedAttribute(StringAttr::get(module.getContext(), "function"),
                         functionOrdinal),
          NamedAttribute(
              StringAttr::get(module.getContext(), "result"),
              IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                               ordinal)),
          NamedAttribute(StringAttr::get(module.getContext(), "anchor"),
                         anchor),
      };
      records.push_back(DictionaryAttr::get(module.getContext(), fields));
    }
  }
  return ArrayAttr::get(module.getContext(), records);
}

bool zeroResultRecordsMatch(ArrayAttr current, ArrayAttr expected) {
  if (!current || !expected || current.size() != expected.size()) {
    return false;
  }
  llvm::SmallDenseSet<Attribute> matchedOperationRefs;
  for (Attribute currentAttribute : current) {
    auto currentRecord = dyn_cast<DictionaryAttr>(currentAttribute);
    Attribute operationRef =
        currentRecord ? currentRecord.get("operation_ref") : Attribute{};
    auto expectedPosition = llvm::find_if(expected, [&](Attribute attribute) {
      auto record = dyn_cast<DictionaryAttr>(attribute);
      return record && record.get("operation_ref") == operationRef;
    });
    auto expectedRecord = expectedPosition == expected.end()
                              ? DictionaryAttr{}
                              : dyn_cast<DictionaryAttr>(*expectedPosition);
    if (!currentRecord || !operationRef || !expectedRecord ||
        !matchedOperationRefs.insert(operationRef).second ||
        currentRecord.get("classification") !=
            expectedRecord.get("classification") ||
        currentRecord.get("operands") != expectedRecord.get("operands")) {
      return false;
    }
    if (currentRecord.get("fingerprint") == expectedRecord.get("fingerprint")) {
      continue;
    }
    auto currentFingerprint =
        currentRecord.getAs<DictionaryAttr>("fingerprint");
    auto expectedFingerprint =
        expectedRecord.getAs<DictionaryAttr>("fingerprint");
    auto currentName = currentFingerprint
                           ? currentFingerprint.getAs<StringAttr>("name")
                           : StringAttr{};
    auto expectedName = expectedFingerprint
                            ? expectedFingerprint.getAs<StringAttr>("name")
                            : StringAttr{};
    if (!currentName || !expectedName ||
        currentName.getValue() != YieldOp::getOperationName() ||
        expectedName.getValue() != stablehlo::ReturnOp::getOperationName() ||
        currentFingerprint.size() != expectedFingerprint.size() ||
        currentFingerprint.get("attributes") !=
            expectedFingerprint.get("attributes") ||
        currentFingerprint.get("result_types") !=
            expectedFingerprint.get("result_types")) {
      return false;
    }
  }
  return true;
}

LogicalResult verifyManifestCoverage(ModuleOp module, DictionaryAttr manifest) {
  auto version = manifest.getAs<IntegerAttr>("version");
  auto completeArray = manifest.getAs<ArrayAttr>(kManifestComplete);
  auto selectedGroups = manifest.getAs<ArrayAttr>(kManifestSelectedRegions);
  auto policy = manifest.getAs<StringAttr>("policy");
  auto policyDigest = manifest.getAs<StringAttr>("policy_digest");
  auto canonicalOptions = manifest.getAs<StringAttr>("canonical_options");
  auto canonicalTuning = manifest.getAs<StringAttr>("canonical_tuning");
  auto tuningDigest = manifest.getAs<StringAttr>("tuning_digest");
  FailureOr<llvm::SmallDenseSet<Attribute>> selected =
      selectedManifestSources(manifest);
  FailureOr<llvm::SmallDenseSet<Attribute>> excluded =
      excludedManifestSources(manifest);
  if (!version || version.getInt() != 2 || !completeArray || !selectedGroups ||
      !policy || !policyDigest || !canonicalOptions || !canonicalTuning ||
      !tuningDigest || failed(selected) || failed(excluded)) {
    return module.emitError("has a malformed Shuttle coverage manifest");
  }
  std::string policyPrefix = "{\"numerics\":\"";
  policyPrefix.append(policy.getValue().data(), policy.getValue().size());
  policyPrefix.push_back('"');
  if ((policy.getValue() != "source_ordered" && policy.getValue() != "fast") ||
      tuningDigest.getValue().size() != 64 ||
      !llvm::all_of(tuningDigest.getValue(), llvm::isHexDigit) ||
      tuningDigest.getValue() != sha256(canonicalTuning.getValue()) ||
      policyDigest.getValue() != sha256(canonicalOptions.getValue()) ||
      !canonicalOptions.getValue().starts_with(policyPrefix)) {
    return module.emitError("has inconsistent Shuttle policy digests");
  }
  llvm::SmallDenseSet<Attribute> complete = attributeSet(completeArray);
  if (complete.size() != completeArray.size()) {
    return module.emitError(
        "coverage manifest contains duplicate source results");
  }
  for (Attribute source : *selected) {
    if (excluded->contains(source)) {
      return module.emitError(
          "coverage manifest selects and excludes one source result");
    }
  }
  llvm::SmallDenseSet<Attribute> partition = *selected;
  partition.insert(excluded->begin(), excluded->end());
  if (!sameAttributeSet(partition, complete)) {
    return module.emitError(
        "coverage manifest is not a total source-result partition");
  }

  llvm::SmallDenseSet<Attribute> operationRefs;
  WalkResult operationRefWalk = module.walk([&](Operation *operation) {
    Attribute operationRef = operation->getAttr(kOperationRefAttribute);
    ArrayAttr refs = sourceRefs(operation);
    if (refs && operation->getParentOfType<FoldOp>() && !operationRef) {
      operation->emitOpError(
          "has nested Fold sources without an operation reference");
      return WalkResult::interrupt();
    }
    if (!operationRef) {
      return WalkResult::advance();
    }
    auto denseOperationRef = dyn_cast<DenseI64ArrayAttr>(operationRef);
    if (!denseOperationRef || denseOperationRef.size() != 3 ||
        !operationRefs.insert(operationRef).second) {
      operation->emitOpError(
          "has a missing-format or duplicate operation reference");
      return WalkResult::interrupt();
    }
    if (refs && llvm::any_of(refs, [&](Attribute ref) {
          auto source = dyn_cast<SourceRefAttr>(ref);
          return !source || operationRefForSource(source) != denseOperationRef;
        })) {
      operation->emitOpError(
          "has an operation reference that differs from its source results");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (operationRefWalk.wasInterrupted()) {
    return failure();
  }

  const bool algebraStage = !module.getOps<func::FuncOp>().empty() && [&]() {
    bool found = false;
    module.walk([&](RegionOp) { found = true; });
    return found;
  }();
  if (algebraStage) {
    SmallVector<Attribute> actualGroups;
    bool policyMismatch = false;
    module.walk([&](RegionOp region) {
      actualGroups.push_back(region.getSourceRefs());
      if (policyName(region.getPolicy()) != policy.getValue()) {
        policyMismatch = true;
      }
    });
    if (policyMismatch) {
      return module.emitError(
          "structural region policy does not equal manifest policy");
    }
    if (ArrayAttr::get(module.getContext(), actualGroups) != selectedGroups) {
      return module.emitError(
          "structural regions do not equal manifest region groups");
    }
  }
  llvm::SmallDenseSet<Attribute> representedSelected;
  llvm::SmallDenseSet<Attribute> representedExcluded;
  auto recordSelected = [&](Operation *operation,
                            Attribute source) -> WalkResult {
    if (!selected->contains(source)) {
      operation->emitOpError(
          "represents a source absent from selected manifest coverage");
      return WalkResult::interrupt();
    }
    if (!representedSelected.insert(source).second) {
      operation->emitOpError("duplicates selected source coverage");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  WalkResult walk = module.walk([&](Operation *operation) {
    if (auto region = dyn_cast<RegionOp>(operation)) {
      for (Attribute source : region.getSourceRefs()) {
        if (!selected->contains(source)) {
          region.emitOpError(
              "declares a source absent from selected manifest coverage");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    }
    if (auto map = dyn_cast<MapOp>(operation)) {
      return recordSelected(operation, map.getSource());
    }
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      return recordSelected(operation, contract.getSource());
    }
    if (auto fold = dyn_cast<FoldOp>(operation)) {
      auto owner =
          fold->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
      if (!owner) {
        fold.emitOpError("requires Reduce owner operation provenance");
        return WalkResult::interrupt();
      }
      if (owner != operationRefForSource(fold.getSource())) {
        fold.emitOpError(
            "has Reduce owner provenance that differs from its source");
        return WalkResult::interrupt();
      }
      for (BlockArgument argument : fold.getCombiner().front().getArguments()) {
        auto anchor = dyn_cast_or_null<DictionaryAttr>(valueAnchor(argument));
        if (!anchor || anchor.get("owner") != owner) {
          fold.emitOpError(
              "has a combiner argument without its Reduce owner anchor");
          return WalkResult::interrupt();
        }
      }
      return recordSelected(operation, fold.getSource());
    }
    ArrayAttr refs = sourceRefs(operation);
    if (!refs) {
      return WalkResult::advance();
    }
    for (Attribute source : refs) {
      if (selected->contains(source)) {
        if (algebraStage) {
          if (!operation->getParentOfType<FoldOp>()) {
            operation->emitOpError(
                "selected source operation survived algebra conversion");
            return WalkResult::interrupt();
          }
          if (!representedSelected.insert(source).second) {
            operation->emitOpError("duplicates selected source coverage");
            return WalkResult::interrupt();
          }
          continue;
        }
        if (!representedSelected.insert(source).second) {
          operation->emitOpError("duplicates selected source coverage");
          return WalkResult::interrupt();
        }
      } else if (excluded->contains(source)) {
        if (!representedExcluded.insert(source).second) {
          operation->emitOpError("duplicates excluded source coverage");
          return WalkResult::interrupt();
        }
      } else {
        operation->emitOpError(
            "carries a source absent from the coverage manifest");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (walk.wasInterrupted()) {
    return failure();
  }
  if (!sameAttributeSet(representedSelected, *selected) ||
      !sameAttributeSet(representedExcluded, *excluded)) {
    return module.emitError(
        "represented source results do not equal manifest coverage");
  }

  FailureOr<ArrayAttr> excludedRecords =
      currentExcludedRecords(module, manifest);
  if (failed(excludedRecords) ||
      *excludedRecords != manifest.getAs<ArrayAttr>(kManifestExcluded)) {
    return module.emitError(
        "excluded operation fingerprint or operand anchors changed");
  }
  FailureOr<ArrayAttr> zeroResultRecords = currentZeroResultRecords(module);
  FailureOr<ArrayAttr> functionResultRecords =
      currentFunctionResultRecords(module);
  if (failed(zeroResultRecords) || failed(functionResultRecords) ||
      !zeroResultRecordsMatch(
          *zeroResultRecords,
          manifest.getAs<ArrayAttr>(kManifestZeroResultOperations)) ||
      *functionResultRecords !=
          manifest.getAs<ArrayAttr>(kManifestFunctionResults)) {
    return module.emitError(
        "zero-result or function-result source anchors changed");
  }
  return success();
}

struct VerifySourceCoveragePass
    : impl::ShuttleVerifySourceCoveragePassBase<VerifySourceCoveragePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto manifest =
        module->getAttrOfType<DictionaryAttr>(kCoverageManifestAttribute);
    LogicalResult result = manifest ? verifyManifestCoverage(module, manifest)
                                    : verifyRegionLocalCoverage(module);
    if (failed(result)) {
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
  void runOnOperation() override {}
};

FailureOr<unsigned> mapResultPosition(AffineMap map, unsigned domainPosition) {
  for (auto [resultPosition, expression] : llvm::enumerate(map.getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (dimension && dimension.getPosition() == domainPosition) {
      return resultPosition;
    }
  }
  return failure();
}

bool isIdentityMap(AffineMap map) {
  return map ==
         AffineMap::getMultiDimIdentityMap(map.getNumDims(), map.getContext());
}

LogicalResult verifyDefaultScalarSemantics(Operation *operation) {
  if (operation->getNumRegions() != 0 || !isMemoryEffectFree(operation)) {
    return operation->emitOpError(
        "requires one pure region-free scalar operation");
  }
  auto fastMath =
      operation->getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
  if (fastMath && fastMath.getValue() != arith::FastMathFlags::none) {
    return operation->emitOpError(
        "has fast-math semantics with no StableHLO representation");
  }
  for (NamedAttribute attribute : operation->getAttrs()) {
    if (attribute.getName().strref() != "fastmath") {
      return operation->emitOpError(
          "has scalar attributes with no StableHLO representation");
    }
  }
  return success();
}

FailureOr<OperationState> lowerMapState(MapOp map, ValueRange operands) {
  if (map.getInputs().size() != operands.size() ||
      map.getResults().size() != 1 || map.getBody().empty()) {
    map.emitOpError("is outside the first Map lowering contract");
    return failure();
  }
  Block &body = map.getBody().front();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1) {
    map.emitOpError("requires one scalar yield for StableHLO lowering");
    return failure();
  }
  if (!map->getDiscardableAttrs().empty()) {
    map.emitOpError("has attributes with no StableHLO Map representation");
    return failure();
  }
  if (!yield->getDiscardableAttrs().empty()) {
    yield.emitOpError("has attributes with no stablehlo.return representation");
    return failure();
  }

  SmallVector<Operation *> scalarOperations;
  for (Operation &operation : body.without_terminator()) {
    scalarOperations.push_back(&operation);
  }
  ArrayAttr indexingMaps = map.getIndexingMaps();
  if (indexingMaps.size() != operands.size() + 1) {
    map.emitOpError("has incomplete indexing maps");
    return failure();
  }

  StringRef stablehloName;
  NamedAttrList attributes;
  if (scalarOperations.empty()) {
    if (operands.size() != 1 || yield.getValues()[0] != body.getArgument(0)) {
      map.emitOpError("has no authoritative scalar operation");
      return failure();
    }
    AffineMap inputMap = cast<AffineMapAttr>(indexingMaps[0]).getValue();
    AffineMap resultMap = cast<AffineMapAttr>(indexingMaps[1]).getValue();
    if (inputMap.getNumDims() != resultMap.getNumDims() ||
        !isIdentityMap(resultMap)) {
      map.emitOpError("has an incompatible identity result map");
      return failure();
    }
    if (map.getSemantics() == MapSemantics::BroadcastInDim) {
      auto inputType = dyn_cast<RankedTensorType>(operands[0].getType());
      auto resultType = dyn_cast<RankedTensorType>(map.getResult(0).getType());
      if (!inputType || !resultType ||
          inputMap.getNumResults() != inputType.getRank() ||
          resultMap.getNumResults() != resultType.getRank() ||
          inputMap.getNumResults() > resultMap.getNumResults()) {
        map.emitOpError("has incompatible broadcast indexing maps");
        return failure();
      }
      SmallVector<int64_t> dimensions;
      llvm::SmallDenseSet<int64_t> seenDimensions;
      unsigned expandingSingletons = 0;
      for (auto [inputDimension, expression] :
           llvm::enumerate(inputMap.getResults())) {
        auto dimension = dyn_cast<AffineDimExpr>(expression);
        int64_t position =
            dimension ? static_cast<int64_t>(dimension.getPosition()) : -1;
        bool expandingSingleton = false;
        bool boundedZero = false;
        if (!dimension) {
          auto floorDiv = dyn_cast<AffineBinaryOpExpr>(expression);
          AffineExpr dividendExpression =
              floorDiv && floorDiv.getKind() == AffineExprKind::FloorDiv
                  ? floorDiv.getLHS()
                  : AffineExpr{};
          AffineExpr divisorExpression =
              floorDiv && floorDiv.getKind() == AffineExprKind::FloorDiv
                  ? floorDiv.getRHS()
                  : AffineExpr{};
          auto dividend =
              dyn_cast_if_present<AffineDimExpr>(dividendExpression);
          auto divisor =
              dyn_cast_if_present<AffineConstantExpr>(divisorExpression);
          if (!dividend || !divisor || divisor.getValue() <= 1) {
            map.emitOpError("expanded singleton broadcast dimensions require a "
                            "bounded-zero floordiv expression");
            return failure();
          }
          boundedZero = true;
          position = dividend.getPosition();
          expandingSingleton =
              inputType.getDimSize(inputDimension) == 1 &&
              position < resultType.getRank() &&
              resultType.getDimSize(position) == divisor.getValue();
        }
        if (position < 0 || position >= resultType.getRank() ||
            !seenDimensions.insert(position).second ||
            (boundedZero && !expandingSingleton) ||
            (!expandingSingleton && inputType.getDimSize(inputDimension) !=
                                        resultType.getDimSize(position))) {
          map.emitOpError(
              "broadcast input map dimensions must be unique, in range, and "
              "match direct or bounded singleton extents");
          return failure();
        }
        expandingSingletons += expandingSingleton;
        dimensions.push_back(position);
      }
      if (expandingSingletons != 0 &&
          (inputType.getRank() != 2 || resultType.getRank() != 2 ||
           dimensions.size() != 2 || expandingSingletons != 1)) {
        map.emitOpError(
            "mapped singleton broadcast requires one expanded axis in a "
            "rank-two dimension permutation");
        return failure();
      }
      stablehloName = stablehlo::BroadcastInDimOp::getOperationName();
      attributes.append("broadcast_dimensions",
                        DenseI64ArrayAttr::get(map.getContext(), dimensions));
    } else if (map.getSemantics() == MapSemantics::Reshape) {
      auto inputType = dyn_cast<RankedTensorType>(operands[0].getType());
      auto resultType = dyn_cast<RankedTensorType>(map.getResult(0).getType());
      if (!inputType || !resultType || !inputType.hasStaticShape() ||
          !resultType.hasStaticShape()) {
        map.emitOpError("requires static singleton reshape types");
        return failure();
      }
      SmallVector<int64_t> inputNonSingleton;
      SmallVector<int64_t> resultNonSingleton;
      llvm::copy_if(inputType.getShape(), std::back_inserter(inputNonSingleton),
                    [](int64_t extent) { return extent != 1; });
      llvm::copy_if(resultType.getShape(),
                    std::back_inserter(resultNonSingleton),
                    [](int64_t extent) { return extent != 1; });
      SmallVector<unsigned> resultNonSingletonPositions;
      for (auto [position, extent] : llvm::enumerate(resultType.getShape())) {
        if (extent != 1) {
          resultNonSingletonPositions.push_back(position);
        }
      }
      if (inputNonSingleton != resultNonSingleton || inputType == resultType ||
          inputMap.getNumResults() != inputType.getRank()) {
        map.emitOpError("has incompatible singleton reshape types");
        return failure();
      }
      unsigned nextNonSingleton = 0;
      for (auto [inputDimension, expression] :
           llvm::enumerate(inputMap.getResults())) {
        if (inputType.getDimSize(inputDimension) == 1) {
          auto constant = dyn_cast<AffineConstantExpr>(expression);
          if (!constant || constant.getValue() != 0) {
            map.emitOpError(
                "singleton reshape input dimensions must use constant zero");
            return failure();
          }
          continue;
        }
        auto dimension = dyn_cast<AffineDimExpr>(expression);
        if (!dimension ||
            nextNonSingleton >= resultNonSingletonPositions.size() ||
            dimension.getPosition() !=
                resultNonSingletonPositions[nextNonSingleton++]) {
          map.emitOpError("has an ambiguous singleton reshape indexing map");
          return failure();
        }
      }
      if (nextNonSingleton != resultNonSingletonPositions.size()) {
        map.emitOpError("has an incomplete singleton reshape indexing map");
        return failure();
      }
      stablehloName = stablehlo::ReshapeOp::getOperationName();
    } else if (map.getSemantics() == MapSemantics::Transpose) {
      if (inputMap.getNumResults() != resultMap.getNumResults()) {
        map.emitOpError("has incompatible transpose indexing maps");
        return failure();
      }
      SmallVector<int64_t> permutation;
      permutation.reserve(resultMap.getNumResults());
      for (AffineExpr resultExpression : resultMap.getResults()) {
        auto resultDimension = dyn_cast<AffineDimExpr>(resultExpression);
        if (!resultDimension) {
          map.emitOpError("has a non-structural transpose result map");
          return failure();
        }
        FailureOr<unsigned> inputPosition =
            mapResultPosition(inputMap, resultDimension.getPosition());
        if (failed(inputPosition)) {
          map.emitOpError("has an incomplete transpose input map");
          return failure();
        }
        permutation.push_back(*inputPosition);
      }
      stablehloName = stablehlo::TransposeOp::getOperationName();
      attributes.append("permutation",
                        DenseI64ArrayAttr::get(map.getContext(), permutation));
    } else {
      map.emitOpError("has pointwise semantics but no scalar operation");
      return failure();
    }
  } else {
    if (map.getSemantics() != MapSemantics::Pointwise) {
      map.emitOpError("structural Map semantics require an empty scalar body");
      return failure();
    }
    if (scalarOperations.size() != 1) {
      map.emitOpError("requires exactly one representable scalar operation");
      return failure();
    }
    Operation *scalar = scalarOperations.front();
    if (yield.getValues()[0] != scalar->getResult(0) ||
        scalar->getOperands() != body.getArguments()) {
      map.emitOpError(
          "scalar operands and yield must directly match the Map boundary");
      return failure();
    }
    if (!llvm::all_of(indexingMaps, [](Attribute attribute) {
          return isIdentityMap(cast<AffineMapAttr>(attribute).getValue());
        })) {
      map.emitOpError(
          "pointwise StableHLO lowering requires identity indexing maps");
      return failure();
    }
    StringRef scalarName = scalar->getName().getStringRef();
    if (scalarName == math::TanhOp::getOperationName() &&
        operands.size() == 1) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::TanhOp::getOperationName();
    } else if (scalarName == math::ExpOp::getOperationName() &&
               operands.size() == 1) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::ExpOp::getOperationName();
    } else if (scalarName == math::RsqrtOp::getOperationName() &&
               operands.size() == 1) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::RsqrtOp::getOperationName();
    } else if (scalarName == arith::NegFOp::getOperationName() &&
               operands.size() == 1) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::NegOp::getOperationName();
    } else if (scalarName == arith::MulFOp::getOperationName() &&
               operands.size() == 2) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::MulOp::getOperationName();
    } else if (scalarName == arith::AddFOp::getOperationName() &&
               operands.size() == 2) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::AddOp::getOperationName();
    } else if (scalarName == arith::SubFOp::getOperationName() &&
               operands.size() == 2) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::SubtractOp::getOperationName();
    } else if (scalarName == arith::DivFOp::getOperationName() &&
               operands.size() == 2) {
      if (failed(verifyDefaultScalarSemantics(scalar))) {
        return failure();
      }
      stablehloName = stablehlo::DivOp::getOperationName();
    } else if (auto convert = dyn_cast<ScalarConvertOp>(scalar);
               convert && operands.size() == 1) {
      Type inputType = convert.getInput().getType();
      Type resultType = convert.getResult().getType();
      const bool validSemantics =
          (inputType.isF32() && resultType.isF32() &&
           convert.getSemantics() == ScalarConvertSemantics::Exact) ||
          (inputType.isBF16() && resultType.isF32() &&
           convert.getSemantics() == ScalarConvertSemantics::Exact) ||
          (inputType.isF32() && resultType.isBF16() &&
           convert.getSemantics() == ScalarConvertSemantics::RoundNearestEven);
      if (!validSemantics) {
        convert.emitOpError("has semantics with no StableHLO conversion");
        return failure();
      }
      stablehloName = stablehlo::ConvertOp::getOperationName();
    } else if (auto constant = dyn_cast<arith::ConstantOp>(scalar);
               constant && operands.empty()) {
      auto value = dyn_cast<FloatAttr>(constant.getValue());
      auto resultType = dyn_cast<RankedTensorType>(map.getResult(0).getType());
      if (!value || !resultType || resultType.getRank() != 0 ||
          value.getType() != resultType.getElementType()) {
        constant.emitOpError("has no rank-zero StableHLO constant lowering");
        return failure();
      }
      stablehloName = stablehlo::ConstantOp::getOperationName();
      SmallVector<Attribute> values{value};
      attributes.append("value", DenseElementsAttr::get(resultType, values));
    } else {
      scalar->emitOpError("has no authoritative StableHLO Map lowering");
      return failure();
    }
  }

  OperationState state(map.getLoc(), stablehloName);
  state.addOperands(operands);
  state.addTypes(map.getResultTypes());
  state.addAttributes(attributes);
  return state;
}

FailureOr<stablehlo::Precision> stablehloPrecision(Attribute attribute) {
  auto value = dyn_cast<StringAttr>(attribute);
  if (!value) {
    return failure();
  }
  return llvm::StringSwitch<FailureOr<stablehlo::Precision>>(value.getValue())
      .Case("DEFAULT", stablehlo::Precision::DEFAULT)
      .Case("HIGH", stablehlo::Precision::HIGH)
      .Case("HIGHEST", stablehlo::Precision::HIGHEST)
      .Default(failure());
}

FailureOr<OperationState> lowerContractState(ContractOp contract,
                                             ValueRange operands) {
  if (operands.size() != 2 || contract.getResults().size() != 1 ||
      contract.getAlgorithm() != "dot_general" ||
      contract.getIndexingMaps().size() != 3) {
    contract.emitOpError("is outside the first Contract lowering contract");
    return failure();
  }
  AffineMap lhsMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[0]).getValue();
  AffineMap rhsMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[1]).getValue();
  AffineMap resultMap =
      cast<AffineMapAttr>(contract.getIndexingMaps()[2]).getValue();
  if (contract.getIteratorKinds().size() != lhsMap.getNumDims()) {
    contract.emitOpError("has incomplete iterator metadata");
    return failure();
  }

  SmallVector<int64_t> lhsBatching;
  SmallVector<int64_t> rhsBatching;
  SmallVector<int64_t> lhsContracting;
  SmallVector<int64_t> rhsContracting;
  SmallVector<unsigned> batchingDomains;
  for (unsigned domain = 0; domain < lhsMap.getNumDims(); ++domain) {
    FailureOr<unsigned> lhsPosition = mapResultPosition(lhsMap, domain);
    FailureOr<unsigned> rhsPosition = mapResultPosition(rhsMap, domain);
    FailureOr<unsigned> resultPosition = mapResultPosition(resultMap, domain);
    auto iterator = dyn_cast<StringAttr>(contract.getIteratorKinds()[domain]);
    if (!iterator) {
      contract.emitOpError("has a non-string iterator kind");
      return failure();
    }
    if (iterator.getValue() == "reduction") {
      if (failed(lhsPosition) || failed(rhsPosition) ||
          succeeded(resultPosition)) {
        contract.emitOpError("has a non-dot reduction domain");
        return failure();
      }
      lhsContracting.push_back(*lhsPosition);
      rhsContracting.push_back(*rhsPosition);
      continue;
    }
    if (iterator.getValue() != "parallel") {
      contract.emitOpError("has an unsupported iterator kind");
      return failure();
    }
    if (succeeded(lhsPosition) && succeeded(rhsPosition) &&
        succeeded(resultPosition)) {
      lhsBatching.push_back(*lhsPosition);
      rhsBatching.push_back(*rhsPosition);
      batchingDomains.push_back(domain);
    }
  }

  SmallVector<unsigned> expectedResultDomains(batchingDomains);
  auto appendFreeDomains = [&](AffineMap operandMap, AffineMap otherMap) {
    for (AffineExpr expression : operandMap.getResults()) {
      unsigned domain = cast<AffineDimExpr>(expression).getPosition();
      if (failed(mapResultPosition(otherMap, domain)) &&
          succeeded(mapResultPosition(resultMap, domain))) {
        expectedResultDomains.push_back(domain);
      }
    }
  };
  appendFreeDomains(lhsMap, rhsMap);
  appendFreeDomains(rhsMap, lhsMap);
  if (expectedResultDomains.size() != resultMap.getNumResults()) {
    contract.emitOpError("has a non-normalized dot result map");
    return failure();
  }
  for (auto [resultPosition, expression] :
       llvm::enumerate(resultMap.getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension ||
        dimension.getPosition() != expectedResultDomains[resultPosition]) {
      contract.emitOpError("has a non-normalized dot result order");
      return failure();
    }
  }

  SmallVector<Attribute> precision;
  for (Attribute attribute : contract.getPrecision()) {
    FailureOr<stablehlo::Precision> value = stablehloPrecision(attribute);
    if (failed(value)) {
      contract.emitOpError("has unsupported precision metadata");
      return failure();
    }
    precision.push_back(
        stablehlo::PrecisionAttr::get(contract.getContext(), *value));
  }
  NamedAttrList attributes;
  attributes.append("dot_dimension_numbers",
                    stablehlo::DotDimensionNumbersAttr::get(
                        contract.getContext(), lhsBatching, rhsBatching,
                        lhsContracting, rhsContracting));
  attributes.append("precision_config",
                    ArrayAttr::get(contract.getContext(), precision));
  OperationState state(contract.getLoc(),
                       stablehlo::DotGeneralOp::getOperationName());
  state.addOperands(operands);
  state.addTypes(contract.getResultTypes());
  state.addAttributes(attributes);
  return state;
}

FailureOr<Operation *> lowerFold(OpBuilder &builder, FoldOp fold,
                                 ValueRange operands) {
  if (!fold.getOrderFree()) {
    fold.emitOpError(
        "order_free=false has no lossless StableHLO Reduce lowering");
    return failure();
  }
  if (llvm::any_of(fold->getDiscardableAttrs(), [](NamedAttribute attribute) {
        return attribute.getName().strref() != kOperationRefAttribute;
      })) {
    fold.emitOpError("has an unsupported discardable Fold attribute");
    return failure();
  }
  if (operands.size() != 2 || fold.getInputs().size() != 1 ||
      fold.getInitializers().size() != 1 || fold.getNumResults() != 1 ||
      fold.getCombiner().empty()) {
    fold.emitOpError("is outside the first Fold lowering contract");
    return failure();
  }
  Block &body = fold.getCombiner().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2) {
    fold.emitOpError("requires exactly one scalar add and one yield");
    return failure();
  }
  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.back());
  auto fastMath = add ? add->getAttrOfType<arith::FastMathFlagsAttr>("fastmath")
                      : arith::FastMathFlagsAttr{};
  if (!add || !yield || add.getLhs() != body.getArgument(0) ||
      add.getRhs() != body.getArgument(1) || yield.getValues().size() != 1 ||
      yield.getValues()[0] != add.getResult() ||
      !body.getArgument(0).getType().isF32() ||
      !body.getArgument(1).getType().isF32() ||
      (fastMath && fastMath.getValue() != arith::FastMathFlags::none) ||
      llvm::any_of(add->getAttrs(),
                   [](NamedAttribute attribute) {
                     StringRef name = attribute.getName().strref();
                     return name != "fastmath" &&
                            name != kSourceRefsAttribute &&
                            name != kOperationRefAttribute;
                   }) ||
      llvm::any_of(yield->getAttrs(), [](NamedAttribute attribute) {
        return attribute.getName().strref() != kOperationRefAttribute;
      })) {
    fold.emitOpError("requires the closed ordered scalar f32 add combiner");
    return failure();
  }
  ArrayAttr addRefs = sourceRefs(add);
  auto foldOwner =
      fold->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  auto addOwner = add->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  auto yieldOwner =
      yield->getAttrOfType<DenseI64ArrayAttr>(kOperationRefAttribute);
  if (!addRefs || addRefs.size() != 1 || !foldOwner || !addOwner ||
      !yieldOwner) {
    fold.emitOpError("has incomplete Fold source provenance");
    return failure();
  }

  OperationState state(fold.getLoc(), stablehlo::ReduceOp::getOperationName());
  state.addOperands(operands);
  state.addTypes(fold.getResultTypes());
  state.addAttribute("dimensions", fold.getReductionDimensionsAttr());
  state.addAttribute(kSourceRefsAttribute,
                     ArrayAttr::get(fold.getContext(), {fold.getSource()}));
  state.addAttribute(kOperationRefAttribute, foldOwner);
  state.addRegion();
  Operation *reduce = builder.create(state);
  Block *reducer = new Block();
  reduce->getRegion(0).push_back(reducer);
  auto scalarTensor = RankedTensorType::get({}, builder.getF32Type());
  reducer->addArgument(scalarTensor, body.getArgument(0).getLoc());
  reducer->addArgument(scalarTensor, body.getArgument(1).getLoc());
  OpBuilder reducerBuilder = OpBuilder::atBlockEnd(reducer);
  OperationState addState(add.getLoc(), stablehlo::AddOp::getOperationName());
  addState.addOperands(reducer->getArguments());
  addState.addTypes(scalarTensor);
  addState.addAttribute(kSourceRefsAttribute, addRefs);
  addState.addAttribute(kOperationRefAttribute, addOwner);
  Operation *loweredAdd = reducerBuilder.create(addState);
  OperationState returnState(yield.getLoc(),
                             stablehlo::ReturnOp::getOperationName());
  returnState.addOperands(loweredAdd->getResult(0));
  returnState.addAttribute(kOperationRefAttribute, yieldOwner);
  reducerBuilder.create(returnState);
  return reduce;
}

LogicalResult lowerRegion(RegionOp region) {
  OpBuilder builder(region);
  IRMapping mapping;
  for (auto [argument, input] : llvm::zip_equal(
           region.getBody().front().getArguments(), region.getInputs())) {
    mapping.map(argument, input);
  }

  auto yield = cast<YieldOp>(region.getBody().front().getTerminator());
  for (Operation &operation :
       llvm::make_early_inc_range(region.getBody().front())) {
    if (isa<YieldOp>(operation)) {
      continue;
    }
    SmallVector<Value> operands;
    operands.reserve(operation.getNumOperands());
    for (Value operand : operation.getOperands()) {
      Value mapped = mapping.lookupOrNull(operand);
      if (!mapped) {
        return operation.emitOpError("has an unmapped algebra operand");
      }
      operands.push_back(mapped);
    }
    Attribute source;
    FailureOr<OperationState> state = failure();
    if (auto fold = dyn_cast<FoldOp>(operation)) {
      FailureOr<Operation *> lowered = lowerFold(builder, fold, operands);
      if (failed(lowered)) {
        return failure();
      }
      for (auto [result, loweredResult] :
           llvm::zip_equal(operation.getResults(), (*lowered)->getResults())) {
        mapping.map(result, loweredResult);
      }
      continue;
    }
    if (auto map = dyn_cast<MapOp>(operation)) {
      source = map.getSource();
      state = lowerMapState(map, operands);
    } else if (auto contract = dyn_cast<ContractOp>(operation)) {
      source = contract.getSource();
      state = lowerContractState(contract, operands);
    } else {
      return operation.emitOpError("has no source-ordered StableHLO lowering");
    }
    if (failed(state)) {
      return failure();
    }
    state->addAttribute(kSourceRefsAttribute,
                        ArrayAttr::get(region.getContext(), {source}));
    Operation *lowered = builder.create(*state);
    for (auto [result, loweredResult] :
         llvm::zip_equal(operation.getResults(), lowered->getResults())) {
      mapping.map(result, loweredResult);
    }
  }

  if (yield.getValues().size() != region.getNumResults()) {
    return region.emitOpError("yield/result arity changed before lowering");
  }
  for (auto [result, yielded] :
       llvm::zip_equal(region.getResults(), yield.getValues())) {
    Value replacement = mapping.lookupOrNull(yielded);
    if (!replacement) {
      return region.emitOpError("has an unmapped yielded value");
    }
    result.replaceAllUsesWith(replacement);
  }
  region.erase();
  return success();
}

struct LowerAlgebraToStablehloPass
    : impl::ShuttleLowerAlgebraToStablehloPassBase<
          LowerAlgebraToStablehloPass> {
  void runOnOperation() override {
    if (!getOperation()->hasAttr(kCoverageManifestAttribute)) {
      getOperation().emitError(
          "requires structural coverage before StableHLO lowering");
      signalPassFailure();
      return;
    }
    SmallVector<RegionOp> regions;
    getOperation().walk([&](RegionOp region) { regions.push_back(region); });
    for (RegionOp region : regions) {
      if (failed(lowerRegion(region))) {
        signalPassFailure();
        return;
      }
    }
  }
};

struct StripSourceProvenancePass
    : impl::ShuttleStripSourceProvenancePassBase<StripSourceProvenancePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module->removeAttr(kCoverageManifestAttribute);
    module.walk([&](Operation *operation) {
      operation->removeAttr(kSourceRefsAttribute);
      operation->removeAttr(kSelectedAttribute);
      operation->removeAttr(kFunctionOrdinalAttribute);
      operation->removeAttr(kOperationRefAttribute);
      operation->removeAttr(kRegionResultSourcesAttribute);
      if (containsShuttleAttribute(operation->getLoc())) {
        operation->setLoc(UnknownLoc::get(operation->getContext()));
      }
    });
  }
};

struct VerifyNoShuttleOpsPass
    : impl::ShuttleVerifyNoShuttleOpsPassBase<VerifyNoShuttleOpsPass> {
  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](Operation *operation) {
      if (operation->getName().getDialectNamespace() ==
          ShuttleDialect::getDialectNamespace()) {
        operation->emitOpError("Shuttle operation remains before HLO export");
        return WalkResult::interrupt();
      }
      if (containsShuttleAttribute(operation->getLoc())) {
        operation->emitOpError(
            "Shuttle attribute remains in a location before HLO export");
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

std::string attributeText(Attribute attribute) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return text;
}

std::string normalizedModuleFingerprintImpl(ModuleOp module) {
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  for (NamedAttribute attribute : module->getAttrs()) {
    if (attribute.getName().strref() == SymbolTable::getSymbolAttrName()) {
      continue;
    }
    stream << "module_attr:" << attribute.getName() << '=';
    attribute.getValue().print(stream);
    stream << '\n';
  }
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    stream << "function:";
    function.getFunctionType().print(stream);
    stream << '\n';
    llvm::SmallDenseMap<Value, uint64_t> values;
    uint64_t nextValue = 0;
    for (BlockArgument argument : function.getArguments()) {
      values.try_emplace(argument, nextValue++);
    }
    std::function<void(Operation &)> printOperation =
        [&](Operation &operation) {
          stream << operation.getName() << '(';
          for (Value operand : operation.getOperands()) {
            auto position = values.find(operand);
            if (position == values.end()) {
              stream << "missing";
            } else {
              stream << position->second;
            }
            stream << ',';
          }
          stream << ")->(";
          for (Type type : operation.getResultTypes()) {
            type.print(stream);
            stream << ',';
          }
          stream << "){";
          for (NamedAttribute attribute : operation.getAttrs()) {
            StringRef name = attribute.getName().strref();
            if (name == SymbolTable::getSymbolAttrName() ||
                name == kSourceRefsAttribute || name == kSelectedAttribute ||
                name == kFunctionOrdinalAttribute ||
                name == kOperationRefAttribute ||
                name == kCoverageManifestAttribute ||
                name == kRegionResultSourcesAttribute) {
              continue;
            }
            stream << name << '=';
            attribute.getValue().print(stream);
            stream << ',';
          }
          stream << "}\n";
          for (Value result : operation.getResults()) {
            values.try_emplace(result, nextValue++);
          }
          for (auto [regionOrdinal, region] :
               llvm::enumerate(operation.getRegions())) {
            stream << "region:" << regionOrdinal << '\n';
            for (auto [blockOrdinal, block] : llvm::enumerate(region)) {
              stream << "block:" << blockOrdinal << '(';
              for (BlockArgument argument : block.getArguments()) {
                argument.getType().print(stream);
                stream << ':' << nextValue << ',';
                values.try_emplace(argument, nextValue++);
              }
              stream << ")\n";
              for (Operation &nested : block) {
                printOperation(nested);
              }
            }
          }
        };
    for (Operation &operation : function.getBody().front()) {
      printOperation(operation);
    }
  }
  stream.flush();
  return sha256(normalized);
}

bool isMaterializationFingerprintProvenance(StringRef name) {
  return name == kSourceRefsAttribute || name == kSelectedAttribute ||
         name == kFunctionOrdinalAttribute || name == kOperationRefAttribute ||
         name == kCoverageManifestAttribute ||
         name == kRegionResultSourcesAttribute || name == "source";
}

std::string semanticTaskFingerprint(Operation *root) {
  std::string normalized;
  llvm::raw_string_ostream stream(normalized);
  llvm::SmallDenseMap<Value, uint64_t> values;
  uint64_t nextValue = 0;
  for (Value operand : root->getOperands()) {
    values.try_emplace(operand, nextValue++);
  }
  std::function<void(Operation &)> printOperation = [&](Operation &operation) {
    stream << operation.getName() << '(';
    for (Value operand : operation.getOperands()) {
      auto position = values.find(operand);
      stream << (position == values.end() ? UINT64_MAX : position->second)
             << ',';
    }
    stream << ")->(";
    for (Type type : operation.getResultTypes()) {
      type.print(stream);
      stream << ',';
    }
    stream << "){";
    for (NamedAttribute attribute : operation.getAttrs()) {
      if (isMaterializationFingerprintProvenance(
              attribute.getName().strref())) {
        continue;
      }
      stream << attribute.getName() << '=';
      attribute.getValue().print(stream);
      stream << ',';
    }
    stream << "}\n";
    for (Value result : operation.getResults()) {
      values.try_emplace(result, nextValue++);
    }
    for (Region &region : operation.getRegions()) {
      for (Block &block : region) {
        stream << "block(";
        for (BlockArgument argument : block.getArguments()) {
          argument.getType().print(stream);
          stream << ':' << nextValue << ',';
          values.try_emplace(argument, nextValue++);
        }
        stream << ")\n";
        for (Operation &nested : block) {
          printOperation(nested);
        }
      }
    }
  };
  printOperation(*root);
  stream.flush();
  return sha256(normalized);
}

FailureOr<SmallVector<int64_t>> mapDomainShape(MapOp map) {
  if (map.getNumResults() != 1) {
    return failure();
  }
  auto resultType = dyn_cast<RankedTensorType>(map.getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape() ||
      llvm::any_of(resultType.getShape(),
                   [](int64_t extent) { return extent <= 0; })) {
    return failure();
  }
  AffineMap resultMap =
      cast<AffineMapAttr>(
          map.getIndexingMaps()[map.getIndexingMaps().size() - 1])
          .getValue();
  SmallVector<int64_t> domain(resultMap.getNumDims(), ShapedType::kDynamic);
  for (auto [resultPosition, expression] :
       llvm::enumerate(resultMap.getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension) {
      return failure();
    }
    domain[dimension.getPosition()] = resultType.getDimSize(resultPosition);
  }
  if (llvm::is_contained(domain, ShapedType::kDynamic)) {
    return failure();
  }
  return domain;
}

bool isClosedRowFold(FoldOp fold) {
  auto inputType =
      fold.getInputs().size() == 1
          ? dyn_cast<RankedTensorType>(fold.getInputs()[0].getType())
          : RankedTensorType{};
  auto initType =
      fold.getInitializers().size() == 1
          ? dyn_cast<RankedTensorType>(fold.getInitializers()[0].getType())
          : RankedTensorType{};
  auto resultType =
      fold.getNumResults() == 1
          ? dyn_cast<RankedTensorType>(fold.getResult(0).getType())
          : RankedTensorType{};
  if (!inputType || !initType || !resultType || !inputType.hasStaticShape() ||
      inputType.getRank() != 2 || inputType.getDimSize(0) <= 0 ||
      inputType.getDimSize(1) <= 0 || !inputType.getElementType().isF32() ||
      initType.getRank() != 0 || !initType.getElementType().isF32() ||
      resultType.getShape() != ArrayRef<int64_t>{inputType.getDimSize(0)} ||
      !resultType.getElementType().isF32() ||
      fold.getReductionDimensions().size() != 1 ||
      fold.getReductionDimensions().front() != 1 || !fold.getOrderFree() ||
      fold.getAccumulatorTypes().size() != 1 ||
      cast<TypeAttr>(fold.getAccumulatorTypes()[0]).getValue() !=
          Float32Type::get(fold.getContext())) {
    return false;
  }
  Block &body = fold.getCombiner().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2) {
    return false;
  }
  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<YieldOp>(body.back());
  auto fastMath = add ? add->getAttrOfType<arith::FastMathFlagsAttr>("fastmath")
                      : arith::FastMathFlagsAttr{};
  if (!add || !yield || add.getLhs() != body.getArgument(0) ||
      add.getRhs() != body.getArgument(1) || yield.getValues().size() != 1 ||
      yield.getValues()[0] != add.getResult() ||
      (fastMath && fastMath.getValue() != arith::FastMathFlags::none)) {
    return false;
  }
  return true;
}

struct PlannedTask {
  Operation *operation;
  MaterializationTaskKind kind;
  SmallVector<int64_t> domain;
  SmallVector<int64_t> inputs;
  SmallVector<int64_t> outputs;
  SmallVector<int64_t> dependencies;
};

struct PlannedBuffer {
  Value value;
  std::optional<int64_t> producer;
  SmallVector<int64_t> consumers;
  bool liveOut = false;
};

struct DerivedPlan {
  SmallVector<PlannedTask, 0> tasks;
  SmallVector<PlannedBuffer, 0> buffers;
};

FailureOr<DerivedPlan> deriveRowFoldPlan(RegionOp region) {
  SmallVector<Operation *> algebra;
  for (Operation &operation : region.getBody().front().without_terminator()) {
    if (!isa<MapOp, FoldOp>(operation)) {
      return failure();
    }
    algebra.push_back(&operation);
  }
  if (algebra.empty()) {
    return failure();
  }
  SmallVector<PlannedBuffer, 0> buffers;
  llvm::SmallDenseMap<Value, int64_t> bufferForValue;
  for (BlockArgument input : region.getBody().front().getArguments()) {
    auto type = dyn_cast<RankedTensorType>(input.getType());
    if (!type || !type.hasStaticShape() ||
        llvm::any_of(type.getShape(),
                     [](int64_t extent) { return extent <= 0; })) {
      return failure();
    }
    bufferForValue.try_emplace(input, buffers.size());
    buffers.push_back({input, std::nullopt, {}, false});
  }

  SmallVector<PlannedTask, 0> tasks;
  unsigned foldCount = 0;
  for (Operation *operation : algebra) {
    PlannedTask task;
    task.operation = operation;
    task.kind = isa<FoldOp>(operation) ? MaterializationTaskKind::Fold
                                       : MaterializationTaskKind::Map;
    if (auto fold = dyn_cast<FoldOp>(operation)) {
      if (!isClosedRowFold(fold)) {
        return failure();
      }
      ++foldCount;
      task.domain.assign(cast<RankedTensorType>(fold.getInputs()[0].getType())
                             .getShape()
                             .begin(),
                         cast<RankedTensorType>(fold.getInputs()[0].getType())
                             .getShape()
                             .end());
    } else {
      FailureOr<SmallVector<int64_t>> domain =
          mapDomainShape(cast<MapOp>(operation));
      if (failed(domain)) {
        return failure();
      }
      task.domain = std::move(*domain);
    }
    for (Value operand : operation->getOperands()) {
      auto found = bufferForValue.find(operand);
      if (found == bufferForValue.end()) {
        return failure();
      }
      task.inputs.push_back(found->second);
      if (buffers[found->second].consumers.empty() ||
          buffers[found->second].consumers.back() !=
              static_cast<int64_t>(tasks.size())) {
        buffers[found->second].consumers.push_back(tasks.size());
      }
      if (buffers[found->second].producer &&
          !llvm::is_contained(task.dependencies,
                              *buffers[found->second].producer)) {
        task.dependencies.push_back(*buffers[found->second].producer);
      }
    }
    llvm::sort(task.dependencies);
    for (Value result : operation->getResults()) {
      auto type = dyn_cast<RankedTensorType>(result.getType());
      if (!type || !type.hasStaticShape() ||
          llvm::any_of(type.getShape(),
                       [](int64_t extent) { return extent <= 0; })) {
        return failure();
      }
      int64_t ordinal = buffers.size();
      bufferForValue.try_emplace(result, ordinal);
      buffers.push_back(
          {result, static_cast<int64_t>(tasks.size()), {}, false});
      task.outputs.push_back(ordinal);
    }
    tasks.push_back(std::move(task));
  }
  if (foldCount != 1) {
    return failure();
  }
  auto regionYield = cast<YieldOp>(region.getBody().front().getTerminator());
  for (Value value : regionYield.getValues()) {
    auto found = bufferForValue.find(value);
    if (found == bufferForValue.end()) {
      return failure();
    }
    buffers[found->second].liveOut = true;
  }
  for (PlannedBuffer &buffer : buffers) {
    if (!buffer.liveOut && buffer.consumers.empty()) {
      return failure();
    }
  }

  llvm::SmallDenseSet<Operation *> connected;
  SmallVector<Operation *> worklist;
  for (Operation *operation : algebra) {
    if (isa<FoldOp>(operation)) {
      connected.insert(operation);
      worklist.push_back(operation);
      break;
    }
  }
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    for (Operation *candidate : algebra) {
      bool adjacent = llvm::any_of(candidate->getOperands(),
                                   [&](Value value) {
                                     return value.getDefiningOp() == operation;
                                   }) ||
                      llvm::any_of(operation->getOperands(), [&](Value value) {
                        return value.getDefiningOp() == candidate;
                      });
      if (adjacent && connected.insert(candidate).second) {
        worklist.push_back(candidate);
      }
    }
  }
  if (connected.size() != algebra.size()) {
    return failure();
  }
  return DerivedPlan{std::move(tasks), std::move(buffers)};
}

Operation *createMaterializationRecord(OpBuilder &builder, Location location,
                                       StringRef name,
                                       ArrayRef<NamedAttribute> attrs) {
  OperationState state(location, name);
  state.addAttributes(attrs);
  return builder.create(state);
}

class PlanRowFoldMaterializationPass
    : public impl::ShuttlePlanRowFoldMaterializationPassBase<
          PlanRowFoldMaterializationPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module.getOps<MaterializationPlanOp>().empty()) {
      module.emitError("already contains a materialization plan");
      return signalPassFailure();
    }
    RegionOp selected;
    DerivedPlan derived;
    module.walk([&](RegionOp region) {
      FailureOr<DerivedPlan> candidate = deriveRowFoldPlan(region);
      if (succeeded(candidate)) {
        if (selected) {
          selected = RegionOp{};
          return WalkResult::interrupt();
        }
        selected = region;
        derived = std::move(*candidate);
      }
      return WalkResult::advance();
    });
    if (!selected) {
      module.emitError(
          "requires exactly one connected static row Fold and Map region");
      return signalPassFailure();
    }

    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    OperationState planState(module.getLoc(),
                             MaterializationPlanOp::getOperationName());
    planState.addAttribute("schema_version", builder.getI64IntegerAttr(1));
    planState.addAttribute("policy", selected.getPolicyAttr());
    planState.addAttribute("fingerprint",
                           builder.getStringAttr(std::string(64, '0')));
    planState.addRegion();
    auto plan = cast<MaterializationPlanOp>(builder.create(planState));
    Block *body = new Block();
    plan.getBody().push_back(body);
    builder.setInsertionPointToEnd(body);

    auto &tasks = derived.tasks;
    auto &buffers = derived.buffers;
    for (auto [ordinal, buffer] : llvm::enumerate(buffers)) {
      int64_t end =
          buffer.liveOut
              ? tasks.size()
              : (buffer.consumers.empty() ? 0 : buffer.consumers.back());
      SmallVector<NamedAttribute> attrs{
          builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
          builder.getNamedAttr("tensor_type",
                               TypeAttr::get(buffer.value.getType())),
          builder.getNamedAttr("storage",
                               MaterializationStorageAttr::get(
                                   module.getContext(),
                                   (!buffer.producer || buffer.liveOut)
                                       ? MaterializationStorage::External
                                       : MaterializationStorage::Temporary)),
          builder.getNamedAttr("live_in",
                               builder.getBoolAttr(!buffer.producer)),
          builder.getNamedAttr("live_out", builder.getBoolAttr(buffer.liveOut)),
          builder.getNamedAttr(
              "consumers",
              DenseI64ArrayAttr::get(module.getContext(), buffer.consumers)),
          builder.getNamedAttr(
              "lifetime_start",
              builder.getI64IntegerAttr(buffer.producer.value_or(0))),
          builder.getNamedAttr("lifetime_end", builder.getI64IntegerAttr(end))};
      if (buffer.producer) {
        attrs.push_back(builder.getNamedAttr(
            "producer", builder.getI64IntegerAttr(*buffer.producer)));
      }
      createMaterializationRecord(builder, module.getLoc(),
                                  MaterializationBufferOp::getOperationName(),
                                  attrs);
    }
    for (auto [ordinal, task] : llvm::enumerate(tasks)) {
      auto source = isa<MapOp>(task.operation)
                        ? cast<MapOp>(task.operation).getSource()
                        : cast<FoldOp>(task.operation).getSource();
      SmallVector<NamedAttribute> attrs{
          builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
          builder.getNamedAttr("kind", MaterializationTaskKindAttr::get(
                                           module.getContext(), task.kind)),
          builder.getNamedAttr(
              "domain_shape",
              DenseI64ArrayAttr::get(module.getContext(), task.domain)),
          builder.getNamedAttr(
              "reduction_dimensions",
              DenseI64ArrayAttr::get(module.getContext(),
                                     task.kind == MaterializationTaskKind::Fold
                                         ? ArrayRef<int64_t>{1}
                                         : ArrayRef<int64_t>{})),
          builder.getNamedAttr(
              "input_buffers",
              DenseI64ArrayAttr::get(module.getContext(), task.inputs)),
          builder.getNamedAttr(
              "output_buffers",
              DenseI64ArrayAttr::get(module.getContext(), task.outputs)),
          builder.getNamedAttr(
              "dependencies",
              DenseI64ArrayAttr::get(module.getContext(), task.dependencies)),
          builder.getNamedAttr(
              "semantic_fingerprint",
              builder.getStringAttr(semanticTaskFingerprint(task.operation))),
          builder.getNamedAttr("source", source)};
      if (task.kind == MaterializationTaskKind::Fold) {
        attrs.push_back(
            builder.getNamedAttr("order_free", builder.getBoolAttr(true)));
      }
      createMaterializationRecord(builder, task.operation->getLoc(),
                                  MaterializationTaskOp::getOperationName(),
                                  attrs);
    }
    builder.create<MaterializationPlanYieldOp>(module.getLoc());
    plan.setFingerprint(materializationPlanFingerprint(plan));
    if (failed(plan.verifyRegions())) {
      signalPassFailure();
    }
  }
};

LogicalResult
verifyMaterializationPlanAgainstSource(ModuleOp module,
                                       MaterializationPlanOp plan) {
  if (failed(plan.verifyRegions())) {
    return failure();
  }
  llvm::SmallDenseMap<Attribute, Operation *> sourceOperations;
  llvm::SmallDenseSet<Attribute> duplicateSources;
  module.walk([&](Operation *operation) {
    if (auto map = dyn_cast<MapOp>(operation)) {
      if (!sourceOperations.try_emplace(map.getSource(), operation).second) {
        duplicateSources.insert(map.getSource());
      }
    } else if (auto fold = dyn_cast<FoldOp>(operation)) {
      if (!sourceOperations.try_emplace(fold.getSource(), operation).second) {
        duplicateSources.insert(fold.getSource());
      }
    }
  });
  SmallVector<MaterializationBufferOp> buffers;
  SmallVector<MaterializationTaskOp> tasks;
  for (Operation &operation : plan.getBody().front()) {
    if (auto buffer = dyn_cast<MaterializationBufferOp>(operation)) {
      buffers.push_back(buffer);
    } else if (auto task = dyn_cast<MaterializationTaskOp>(operation)) {
      tasks.push_back(task);
    }
  }
  llvm::SmallDenseSet<Operation *> bound;
  for (MaterializationTaskOp task : tasks) {
    auto found = sourceOperations.find(task.getSource());
    if (found == sourceOperations.end() ||
        duplicateSources.contains(task.getSource()) ||
        !bound.insert(found->second).second) {
      task.emitOpError("source must uniquely bind one surviving algebra task");
      return failure();
    }
    Operation *source = found->second;
    const bool kindMatches = (task.getKind() == MaterializationTaskKind::Map &&
                              isa<MapOp>(source)) ||
                             (task.getKind() == MaterializationTaskKind::Fold &&
                              isa<FoldOp>(source));
    if (!kindMatches ||
        task.getSemanticFingerprint() != semanticTaskFingerprint(source) ||
        task.getInputBuffers().size() != source->getNumOperands() ||
        task.getOutputBuffers().size() != source->getNumResults()) {
      task.emitOpError("does not match its bound algebra semantics");
      return failure();
    }
    for (auto [bufferOrdinal, value] :
         llvm::zip_equal(task.getInputBuffers(), source->getOperands())) {
      if (buffers[bufferOrdinal].getTensorType() != value.getType()) {
        task.emitOpError("input buffer type does not match bound algebra");
        return failure();
      }
    }
    for (auto [bufferOrdinal, value] :
         llvm::zip_equal(task.getOutputBuffers(), source->getResults())) {
      if (buffers[bufferOrdinal].getTensorType() != value.getType()) {
        task.emitOpError("output buffer type does not match bound algebra");
        return failure();
      }
    }
    SmallVector<int64_t> expectedDomain;
    if (auto map = dyn_cast<MapOp>(source)) {
      FailureOr<SmallVector<int64_t>> domain = mapDomainShape(map);
      if (failed(domain)) {
        task.emitOpError("bound Map has no closed static domain");
        return failure();
      }
      expectedDomain = std::move(*domain);
    } else {
      auto fold = cast<FoldOp>(source);
      expectedDomain.assign(
          cast<RankedTensorType>(fold.getInputs()[0].getType())
              .getShape()
              .begin(),
          cast<RankedTensorType>(fold.getInputs()[0].getType())
              .getShape()
              .end());
    }
    if (task.getDomainShape() != ArrayRef<int64_t>(expectedDomain)) {
      task.emitOpError("domain does not match bound algebra indexing");
      return failure();
    }
  }
  RegionOp owner;
  if (!bound.empty()) {
    owner = (*bound.begin())->getParentOfType<RegionOp>();
  }
  if (!owner || owner.getPolicy() != plan.getPolicy() ||
      bound.size() != owner.getBody().front().getOperations().size() - 1) {
    plan.emitOpError("must cover exactly one connected algebra region");
    return failure();
  }
  for (Operation *operation : bound) {
    if (operation->getParentOfType<RegionOp>() != owner) {
      plan.emitOpError("tasks must bind one algebra region");
      return failure();
    }
  }
  FailureOr<DerivedPlan> expected = deriveRowFoldPlan(owner);
  if (failed(expected) || expected->tasks.size() != tasks.size() ||
      expected->buffers.size() != buffers.size()) {
    plan.emitOpError("does not equal the plan derived from bound algebra");
    return failure();
  }
  for (auto [ordinal, expectedTask] : llvm::enumerate(expected->tasks)) {
    MaterializationTaskOp actual = tasks[ordinal];
    if (expectedTask.operation != sourceOperations.lookup(actual.getSource()) ||
        actual.getInputBuffers() != ArrayRef<int64_t>(expectedTask.inputs) ||
        actual.getOutputBuffers() != ArrayRef<int64_t>(expectedTask.outputs) ||
        actual.getDependencies() !=
            ArrayRef<int64_t>(expectedTask.dependencies)) {
      actual.emitOpError(
          "task order and edges must equal bound algebra SSA dependencies");
      return failure();
    }
  }
  for (auto [ordinal, expectedBuffer] : llvm::enumerate(expected->buffers)) {
    MaterializationBufferOp actual = buffers[ordinal];
    auto actualProducer = actual->getAttrOfType<IntegerAttr>("producer");
    std::optional<int64_t> producer =
        actualProducer ? std::optional<int64_t>(actualProducer.getInt())
                       : std::nullopt;
    if (actual.getTensorType() != expectedBuffer.value.getType() ||
        producer != expectedBuffer.producer ||
        actual.getConsumers() != ArrayRef<int64_t>(expectedBuffer.consumers) ||
        actual.getLiveOut() != expectedBuffer.liveOut) {
      actual.emitOpError(
          "buffer ownership and uses must equal bound algebra SSA edges");
      return failure();
    }
  }
  return success();
}

class VerifyMaterializationPlanPass
    : public impl::ShuttleVerifyMaterializationPlanPassBase<
          VerifyMaterializationPlanPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<MaterializationPlanOp> plans(
        module.getOps<MaterializationPlanOp>());
    if (plans.size() != 1 ||
        failed(verifyMaterializationPlanAgainstSource(module, plans[0]))) {
      module.emitError("requires one source-bound materialization plan");
      signalPassFailure();
    }
  }
};

struct ScheduledGeometry {
  SmallVector<int64_t> grid;
  SmallVector<int64_t> tile;
  int64_t serialTiles;
  int64_t workgroupThreads;
  int64_t subgroupSize;
  int64_t scratchBytes;
  std::optional<int64_t> reductionAxis;
};

FailureOr<int64_t> scheduleElementCount(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t extent : shape) {
    if (extent <= 0 || count > std::numeric_limits<int64_t>::max() / extent) {
      return failure();
    }
    count *= extent;
  }
  return count;
}

int64_t scheduleThreads(int64_t tile) {
  constexpr int64_t kSubgroup = 32;
  return ceilDivPositive(tile, kSubgroup) * kSubgroup;
}

FailureOr<ScheduledGeometry> deriveSimt32Geometry(ScheduleTaskKind kind,
                                                  ArrayRef<int64_t> domain) {
  constexpr int64_t kSubgroup = 32;
  constexpr int64_t kMaxThreads = 256;
  if (kind == ScheduleTaskKind::Scalar) {
    if (!domain.empty()) {
      return failure();
    }
    return ScheduledGeometry{{1}, {}, 1, 1, kSubgroup, 0, std::nullopt};
  }
  if (kind == ScheduleTaskKind::Elementwise) {
    FailureOr<int64_t> elements = scheduleElementCount(domain);
    if (failed(elements)) {
      return failure();
    }
    int64_t tile = std::min(*elements, kMaxThreads);
    return ScheduledGeometry{{ceilDivPositive(*elements, tile)},
                             {tile},
                             1,
                             scheduleThreads(tile),
                             kSubgroup,
                             0,
                             std::nullopt};
  }
  if (domain.size() != 2 || domain[0] <= 0 || domain[1] <= 0) {
    return failure();
  }
  int64_t tile = std::min(domain[1], kMaxThreads);
  int64_t threads = scheduleThreads(tile);
  return ScheduledGeometry{{domain[0]},
                           {1, tile},
                           ceilDivPositive(domain[1], tile),
                           threads,
                           kSubgroup,
                           threads * static_cast<int64_t>(sizeof(float)),
                           1};
}

void createScheduleRecord(OpBuilder &builder, Location location,
                          StringRef operationName,
                          ArrayRef<NamedAttribute> attributes) {
  OperationState state(location, operationName);
  state.addAttributes(attributes);
  builder.create(state);
}

class PlanSimt32RowFoldSchedulePass
    : public impl::ShuttlePlanSimt32RowFoldSchedulePassBase<
          PlanSimt32RowFoldSchedulePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<MaterializationPlanOp> materializationPlans(
        module.getOps<MaterializationPlanOp>());
    if (materializationPlans.size() != 1 ||
        failed(verifyMaterializationPlanAgainstSource(
            module, materializationPlans.front()))) {
      module.emitError(
          "requires exactly one source-bound materialization plan");
      return signalPassFailure();
    }
    if (!module.getOps<SchedulePlanOp>().empty()) {
      module.emitError("already contains a schedule plan");
      return signalPassFailure();
    }
    MaterializationPlanOp source = materializationPlans.front();
    SmallVector<MaterializationBufferOp> sourceBuffers(
        source.getBody().front().getOps<MaterializationBufferOp>());
    SmallVector<MaterializationTaskOp> sourceTasks(
        source.getBody().front().getOps<MaterializationTaskOp>());

    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    OperationState planState(module.getLoc(),
                             SchedulePlanOp::getOperationName());
    planState.addAttribute("schema_version", builder.getI64IntegerAttr(1));
    planState.addAttribute(
        "target",
        ScheduleTargetAttr::get(module.getContext(), ScheduleTarget::Simt32));
    planState.addAttribute("policy", source.getPolicyAttr());
    planState.addAttribute("source_plan_fingerprint",
                           builder.getStringAttr(source.getFingerprint()));
    planState.addAttribute("fingerprint",
                           builder.getStringAttr(std::string(64, '0')));
    planState.addRegion();
    auto schedule = cast<SchedulePlanOp>(builder.create(planState));
    Block *body = new Block();
    schedule.getBody().push_back(body);
    builder.setInsertionPointToEnd(body);

    for (auto [ordinal, buffer] : llvm::enumerate(sourceBuffers)) {
      auto type = cast<RankedTensorType>(buffer.getTensorType());
      SmallVector<int64_t> order;
      for (int64_t axis = 0; axis < type.getRank(); ++axis) {
        order.push_back(axis);
      }
      createScheduleRecord(
          builder, buffer.getLoc(), ScheduleBufferOp::getOperationName(),
          {builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
           builder.getNamedAttr("source_buffer",
                                builder.getI64IntegerAttr(ordinal)),
           builder.getNamedAttr("tensor_type", TypeAttr::get(type)),
           builder.getNamedAttr(
               "indexing", ScheduleBufferIndexingAttr::get(
                               module.getContext(),
                               type.getRank() == 0
                                   ? ScheduleBufferIndexing::Scalar
                                   : ScheduleBufferIndexing::Lexicographic)),
           builder.getNamedAttr(
               "iteration_order",
               DenseI64ArrayAttr::get(module.getContext(), order)),
           builder.getNamedAttr(
               "lifetime_start",
               builder.getI64IntegerAttr(buffer.getLifetimeStart())),
           builder.getNamedAttr("lifetime_end", builder.getI64IntegerAttr(
                                                    buffer.getLifetimeEnd()))});
    }
    for (auto [ordinal, task] : llvm::enumerate(sourceTasks)) {
      ScheduleTaskKind kind =
          task.getKind() == MaterializationTaskKind::Fold
              ? ScheduleTaskKind::RowFold
              : (task.getDomainShape().empty() ? ScheduleTaskKind::Scalar
                                               : ScheduleTaskKind::Elementwise);
      FailureOr<ScheduledGeometry> geometry =
          deriveSimt32Geometry(kind, task.getDomainShape());
      if (failed(geometry)) {
        task.emitOpError("cannot derive a bounded SIMT32 schedule candidate");
        return signalPassFailure();
      }
      SmallVector<NamedAttribute> attrs{
          builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
          builder.getNamedAttr("source_task",
                               builder.getI64IntegerAttr(ordinal)),
          builder.getNamedAttr(
              "kind", ScheduleTaskKindAttr::get(module.getContext(), kind)),
          builder.getNamedAttr("domain_shape", task.getDomainShapeAttr()),
          builder.getNamedAttr(
              "grid_shape",
              DenseI64ArrayAttr::get(module.getContext(), geometry->grid)),
          builder.getNamedAttr(
              "tile_shape",
              DenseI64ArrayAttr::get(module.getContext(), geometry->tile)),
          builder.getNamedAttr(
              "serial_tiles", builder.getI64IntegerAttr(geometry->serialTiles)),
          builder.getNamedAttr(
              "workgroup_threads",
              builder.getI64IntegerAttr(geometry->workgroupThreads)),
          builder.getNamedAttr("subgroup_size", builder.getI64IntegerAttr(
                                                    geometry->subgroupSize)),
          builder.getNamedAttr("scratch_bytes", builder.getI64IntegerAttr(
                                                    geometry->scratchBytes)),
          builder.getNamedAttr("input_buffers", task.getInputBuffersAttr()),
          builder.getNamedAttr("output_buffers", task.getOutputBuffersAttr()),
          builder.getNamedAttr("dependencies", task.getDependenciesAttr()),
          builder.getNamedAttr("semantic_fingerprint",
                               task.getSemanticFingerprintAttr())};
      if (geometry->reductionAxis) {
        attrs.push_back(builder.getNamedAttr(
            "reduction_axis",
            builder.getI64IntegerAttr(*geometry->reductionAxis)));
        attrs.push_back(builder.getNamedAttr(
            "reduction_order",
            ScheduleReductionOrderAttr::get(
                module.getContext(),
                ScheduleReductionOrder::TreeAssociationFreeLeafOrderFixed)));
      }
      createScheduleRecord(builder, task.getLoc(),
                           ScheduleTaskOp::getOperationName(), attrs);
    }
    builder.create<SchedulePlanYieldOp>(module.getLoc());
    schedule.setFingerprint(schedulePlanFingerprint(schedule));
    if (failed(schedule.verifyRegions())) {
      signalPassFailure();
    }
  }
};

class VerifySimt32RowFoldSchedulePass
    : public impl::ShuttleVerifySimt32RowFoldSchedulePassBase<
          VerifySimt32RowFoldSchedulePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<MaterializationPlanOp> materializationPlans(
        module.getOps<MaterializationPlanOp>());
    SmallVector<SchedulePlanOp> schedules(module.getOps<SchedulePlanOp>());
    if (materializationPlans.size() != 1 || schedules.size() != 1 ||
        failed(verifyMaterializationPlanAgainstSource(
            module, materializationPlans.front())) ||
        failed(schedules.front().verifyRegions())) {
      module.emitError(
          "requires exactly one valid materialization and schedule plan");
      return signalPassFailure();
    }
    MaterializationPlanOp source = materializationPlans.front();
    SchedulePlanOp schedule = schedules.front();
    if (schedule.getTarget() != ScheduleTarget::Simt32 ||
        schedule.getPolicy() != source.getPolicy() ||
        schedule.getSourcePlanFingerprint() != source.getFingerprint()) {
      schedule.emitOpError(
          "target, policy, and source fingerprint must bind materialization");
      return signalPassFailure();
    }
    SmallVector<MaterializationBufferOp> sourceBuffers(
        source.getBody().front().getOps<MaterializationBufferOp>());
    SmallVector<MaterializationTaskOp> sourceTasks(
        source.getBody().front().getOps<MaterializationTaskOp>());
    SmallVector<ScheduleBufferOp> buffers(
        schedule.getBody().front().getOps<ScheduleBufferOp>());
    SmallVector<ScheduleTaskOp> tasks(
        schedule.getBody().front().getOps<ScheduleTaskOp>());
    if (buffers.size() != sourceBuffers.size() ||
        tasks.size() != sourceTasks.size()) {
      schedule.emitOpError(
          "must cover every materialization buffer and task exactly once");
      return signalPassFailure();
    }
    for (auto [ordinal, actual, expected] :
         llvm::enumerate(buffers, sourceBuffers)) {
      if (actual.getSourceBuffer() != static_cast<int64_t>(ordinal) ||
          actual.getTensorType() != expected.getTensorType() ||
          actual.getLifetimeStart() != expected.getLifetimeStart() ||
          actual.getLifetimeEnd() != expected.getLifetimeEnd()) {
        actual.emitOpError(
            "schedule buffer type and lifetime must equal materialization");
        return signalPassFailure();
      }
    }
    for (auto [ordinal, actual, expected] :
         llvm::enumerate(tasks, sourceTasks)) {
      ScheduleTaskKind expectedKind =
          expected.getKind() == MaterializationTaskKind::Fold
              ? ScheduleTaskKind::RowFold
              : (expected.getDomainShape().empty()
                     ? ScheduleTaskKind::Scalar
                     : ScheduleTaskKind::Elementwise);
      if (actual.getSourceTask() != static_cast<int64_t>(ordinal) ||
          actual.getKind() != expectedKind ||
          actual.getDomainShape() != expected.getDomainShape() ||
          actual.getInputBuffers() != expected.getInputBuffers() ||
          actual.getOutputBuffers() != expected.getOutputBuffers() ||
          actual.getDependencies() != expected.getDependencies() ||
          actual.getSemanticFingerprint() !=
              expected.getSemanticFingerprint()) {
        actual.emitOpError(
            "schedule dependencies must equal the materialization task");
        return signalPassFailure();
      }
    }
  }
};

enum class CpuOpcode : uint8_t {
  ConstantF32 = 0,
  AddF32 = 1,
  MultiplyF32 = 2,
  DivideF32 = 3,
  RsqrtF32 = 4,
  Bf16ToF32 = 5,
  F32ToBf16Rne = 6,
};

enum class CpuTaskKind : uint8_t { Map = 0, RowFold = 1 };
enum class CpuElementType : uint8_t { Bf16 = 0, F32 = 1 };

void appendByte(SmallVectorImpl<int8_t> &bytes, uint8_t value) {
  bytes.push_back(static_cast<int8_t>(value));
}

void appendU32(SmallVectorImpl<int8_t> &bytes, uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    appendByte(bytes, static_cast<uint8_t>(value >> shift));
  }
}

FailureOr<CpuElementType> cpuElementType(Type type) {
  if (type.isBF16()) {
    return CpuElementType::Bf16;
  }
  if (type.isF32()) {
    return CpuElementType::F32;
  }
  return failure();
}

FailureOr<uint8_t> byteRegister(llvm::DenseMap<Value, uint8_t> &registers,
                                Value value) {
  auto position = registers.find(value);
  return position == registers.end() ? FailureOr<uint8_t>(failure())
                                     : FailureOr<uint8_t>(position->second);
}

LogicalResult encodeScalarBody(Operation *owner, Block &block,
                               SmallVectorImpl<int8_t> &bytes) {
  if (block.getNumArguments() > UINT8_MAX ||
      block.getOperations().size() - 1 > UINT8_MAX) {
    return owner->emitOpError("CPU bytecode scalar body is too large");
  }
  llvm::DenseMap<Value, uint8_t> registers;
  uint16_t nextRegister = 0;
  appendByte(bytes, block.getNumArguments());
  for (BlockArgument argument : block.getArguments()) {
    FailureOr<CpuElementType> type = cpuElementType(argument.getType());
    if (failed(type) || nextRegister > UINT8_MAX) {
      return owner->emitOpError("CPU bytecode requires bf16 or f32 scalars");
    }
    appendByte(bytes, static_cast<uint8_t>(*type));
    registers.try_emplace(argument, static_cast<uint8_t>(nextRegister++));
  }
  appendByte(bytes, block.getOperations().size() - 1);
  for (Operation &operation : block.without_terminator()) {
    if (nextRegister > UINT8_MAX || operation.getNumResults() != 1) {
      return owner->emitOpError("CPU bytecode requires one-result scalar ops");
    }
    CpuOpcode opcode;
    SmallVector<uint8_t> operands;
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      auto value = dyn_cast<FloatAttr>(constant.getValue());
      if (!value || !constant.getType().isF32()) {
        return owner->emitOpError("CPU bytecode supports only f32 constants");
      }
      opcode = CpuOpcode::ConstantF32;
      appendByte(bytes, static_cast<uint8_t>(opcode));
      float scalar = value.getValueAsDouble();
      uint32_t bits;
      std::memcpy(&bits, &scalar, sizeof(bits));
      appendU32(bytes, bits);
    } else {
      if (isa<arith::AddFOp>(operation)) {
        opcode = CpuOpcode::AddF32;
      } else if (isa<arith::MulFOp>(operation)) {
        opcode = CpuOpcode::MultiplyF32;
      } else if (isa<arith::DivFOp>(operation)) {
        opcode = CpuOpcode::DivideF32;
      } else if (isa<math::RsqrtOp>(operation)) {
        opcode = CpuOpcode::RsqrtF32;
      } else if (auto convert = dyn_cast<ScalarConvertOp>(operation)) {
        opcode = convert.getSemantics() == ScalarConvertSemantics::Exact
                     ? CpuOpcode::Bf16ToF32
                     : CpuOpcode::F32ToBf16Rne;
      } else {
        return owner->emitOpError("contains an unsupported CPU bytecode op");
      }
      appendByte(bytes, static_cast<uint8_t>(opcode));
      for (Value operand : operation.getOperands()) {
        FailureOr<uint8_t> position = byteRegister(registers, operand);
        if (failed(position)) {
          return owner->emitOpError(
              "CPU bytecode operand is not locally bound");
        }
        appendByte(bytes, *position);
      }
    }
    FailureOr<CpuElementType> resultType =
        cpuElementType(operation.getResult(0).getType());
    if (failed(resultType)) {
      return owner->emitOpError("CPU bytecode result must be bf16 or f32");
    }
    appendByte(bytes, static_cast<uint8_t>(*resultType));
    registers.try_emplace(operation.getResult(0),
                          static_cast<uint8_t>(nextRegister++));
  }
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().size() != 1) {
    return owner->emitOpError("CPU bytecode requires one yielded scalar");
  }
  FailureOr<uint8_t> yielded = byteRegister(registers, yield.getValues()[0]);
  if (failed(yielded)) {
    return owner->emitOpError("CPU bytecode yield is not locally bound");
  }
  appendByte(bytes, *yielded);
  return success();
}

LogicalResult encodeIndexMap(Operation *owner, AffineMap map,
                             SmallVectorImpl<int8_t> &bytes) {
  if (map.getNumResults() > UINT8_MAX) {
    return owner->emitOpError("CPU bytecode indexing rank is too large");
  }
  appendByte(bytes, map.getNumResults());
  for (AffineExpr expression : map.getResults()) {
    uint64_t divisor = 1;
    std::optional<AffineDimExpr> dimension;
    if (auto direct = dyn_cast<AffineDimExpr>(expression)) {
      dimension = direct;
    } else if (auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
               binary && binary.getKind() == AffineExprKind::FloorDiv) {
      if (auto dividedDimension = dyn_cast<AffineDimExpr>(binary.getLHS())) {
        dimension = dividedDimension;
      }
      auto constant = dyn_cast<AffineConstantExpr>(binary.getRHS());
      if (!dimension || !constant || constant.getValue() <= 1) {
        return owner->emitOpError("CPU bytecode indexing map is not bounded");
      }
      divisor = constant.getValue();
    } else {
      return owner->emitOpError("CPU bytecode indexing map is unsupported");
    }
    if (dimension->getPosition() >= UINT8_MAX || divisor > UINT32_MAX) {
      return owner->emitOpError("CPU bytecode indexing value is too large");
    }
    appendByte(bytes, dimension->getPosition());
    appendU32(bytes, divisor);
  }
  return success();
}

struct CpuEntrySpec {
  int64_t ordinal;
  int64_t codeOffset;
  int64_t codeLength;
  SmallVector<int64_t> inputs;
  SmallVector<int64_t> outputs;
  SmallVector<int64_t> dependencies;
  ExecutablePredication predication;
  std::optional<ScheduleReductionOrder> reductionOrder;
};

struct CpuSlotSpec {
  Type tensorType;
  int64_t requiredBytes;
  SmallVector<int64_t> strides;
  int64_t alignment;
  ExecutableAccess access;
  MaterializationStorage storage;
};

struct CpuExecutableSpec {
  SmallVector<int8_t> code;
  SmallVector<CpuEntrySpec> entries;
  SmallVector<CpuSlotSpec> slots;
};

LogicalResult appendTaskHeader(Operation *owner, CpuTaskKind kind,
                               ArrayRef<int64_t> domain,
                               SmallVectorImpl<int8_t> &bytes) {
  for (uint8_t byte : ArrayRef<uint8_t>{'S', 'B', 'C', 1}) {
    appendByte(bytes, byte);
  }
  appendByte(bytes, static_cast<uint8_t>(kind));
  if (domain.size() > UINT8_MAX) {
    return owner->emitOpError("CPU bytecode domain rank is too large");
  }
  appendByte(bytes, domain.size());
  for (int64_t extent : domain) {
    if (extent <= 0 || extent > UINT32_MAX) {
      return owner->emitOpError("CPU bytecode domain extent is unsupported");
    }
    appendU32(bytes, extent);
  }
  return success();
}

FailureOr<SmallVector<int8_t>> encodeCpuTask(Operation *operation,
                                             ScheduleTaskOp scheduleTask) {
  SmallVector<int8_t> bytes;
  if (auto map = dyn_cast<MapOp>(operation)) {
    if (failed(appendTaskHeader(map, CpuTaskKind::Map,
                                scheduleTask.getDomainShape(), bytes)) ||
        map.getInputs().size() > UINT8_MAX || map.getNumResults() != 1) {
      return failure();
    }
    appendByte(bytes, map.getInputs().size());
    for (auto [input, attribute] : llvm::zip_equal(
             map.getInputs(), map.getIndexingMaps().getValue().drop_back())) {
      FailureOr<CpuElementType> type = cpuElementType(
          cast<RankedTensorType>(input.getType()).getElementType());
      if (failed(type)) {
        return failure();
      }
      appendByte(bytes, static_cast<uint8_t>(*type));
      if (failed(encodeIndexMap(map, cast<AffineMapAttr>(attribute).getValue(),
                                bytes))) {
        return failure();
      }
    }
    FailureOr<CpuElementType> outputType = cpuElementType(
        cast<RankedTensorType>(map.getResult(0).getType()).getElementType());
    if (failed(outputType)) {
      return failure();
    }
    appendByte(bytes, static_cast<uint8_t>(*outputType));
    if (failed(encodeScalarBody(map, map.getBody().front(), bytes))) {
      return failure();
    }
    return bytes;
  }
  auto fold = dyn_cast<FoldOp>(operation);
  if (!fold || failed(appendTaskHeader(fold, CpuTaskKind::RowFold,
                                       scheduleTask.getDomainShape(), bytes))) {
    return failure();
  }
  appendByte(bytes, 2);
  appendByte(bytes, static_cast<uint8_t>(CpuElementType::F32));
  appendByte(bytes, 2);
  appendByte(bytes, 0);
  appendU32(bytes, 1);
  appendByte(bytes, 1);
  appendU32(bytes, 1);
  appendByte(bytes, static_cast<uint8_t>(CpuElementType::F32));
  appendByte(bytes, 0);
  appendByte(bytes, static_cast<uint8_t>(CpuElementType::F32));
  appendByte(bytes, 1);
  appendByte(bytes,
             static_cast<uint8_t>(
                 ScheduleReductionOrder::TreeAssociationFreeLeafOrderFixed));
  if (failed(encodeScalarBody(fold, fold.getCombiner().front(), bytes))) {
    return failure();
  }
  return bytes;
}

FailureOr<CpuExecutableSpec>
deriveCpuExecutable(ModuleOp module, MaterializationPlanOp materialization,
                    SchedulePlanOp schedule) {
  SmallVector<RegionOp> regions;
  module.walk([&](RegionOp region) { regions.push_back(region); });
  if (regions.size() != 1 ||
      schedule.getPolicy() != regions.front().getPolicy()) {
    return failure();
  }
  SmallVector<Operation *> algebra;
  for (Operation &operation :
       regions.front().getBody().front().without_terminator()) {
    if (!isa<MapOp, FoldOp>(operation)) {
      return failure();
    }
    algebra.push_back(&operation);
  }
  SmallVector<MaterializationBufferOp> materializationBuffers(
      materialization.getBody().front().getOps<MaterializationBufferOp>());
  SmallVector<ScheduleTaskOp> scheduleTasks(
      schedule.getBody().front().getOps<ScheduleTaskOp>());
  if (algebra.size() != scheduleTasks.size()) {
    return failure();
  }
  for (ScheduleTaskOp task : scheduleTasks) {
    FailureOr<int64_t> elements = scheduleElementCount(task.getDomainShape());
    if ((!task.getDomainShape().empty() &&
         (failed(elements) || *elements > 256)) ||
        task.getSerialTiles() != 1) {
      return failure();
    }
  }

  CpuExecutableSpec spec;
  for (auto [ordinal, operation, scheduleTask] :
       llvm::enumerate(algebra, scheduleTasks)) {
    FailureOr<SmallVector<int8_t>> encoded =
        encodeCpuTask(operation, scheduleTask);
    if (failed(encoded)) {
      return failure();
    }
    const int64_t offset = spec.code.size();
    spec.code.append(*encoded);
    spec.entries.push_back(
        CpuEntrySpec{static_cast<int64_t>(ordinal), offset,
                     static_cast<int64_t>(encoded->size()),
                     SmallVector<int64_t>(scheduleTask.getInputBuffers()),
                     SmallVector<int64_t>(scheduleTask.getOutputBuffers()),
                     SmallVector<int64_t>(scheduleTask.getDependencies()),
                     scheduleTask.getKind() == ScheduleTaskKind::Scalar
                         ? ExecutablePredication::None
                         : ExecutablePredication::DomainBounds,
                     scheduleTask.getReductionOrder()});
  }
  for (MaterializationBufferOp buffer : materializationBuffers) {
    auto type = cast<RankedTensorType>(buffer.getTensorType());
    const int64_t elementBytes = type.getElementType().isBF16() ? 2 : 4;
    FailureOr<int64_t> elements = scheduleElementCount(type.getShape());
    if (failed(elements) ||
        *elements > std::numeric_limits<int64_t>::max() / elementBytes) {
      return failure();
    }
    SmallVector<int64_t> strides(type.getRank());
    int64_t stride = elementBytes;
    for (int64_t axis = type.getRank(); axis > 0; --axis) {
      strides[axis - 1] = stride;
      stride *= type.getDimSize(axis - 1);
    }
    ExecutableAccess access = ExecutableAccess::ReadWrite;
    if (buffer.getLiveIn()) {
      access = ExecutableAccess::Read;
    } else if (buffer.getLiveOut()) {
      access = ExecutableAccess::Write;
    }
    spec.slots.push_back(CpuSlotSpec{type, *elements * elementBytes,
                                     std::move(strides), elementBytes, access,
                                     buffer.getStorage()});
  }
  return spec;
}

ArrayAttr accessArray(MLIRContext *context, ExecutableAccess access,
                      size_t count) {
  SmallVector<Attribute> values(count,
                                ExecutableAccessAttr::get(context, access));
  return ArrayAttr::get(context, values);
}

class BuildCpuExecutableBundlePass
    : public impl::ShuttleBuildCpuExecutableBundlePassBase<
          BuildCpuExecutableBundlePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<MaterializationPlanOp> materializations(
        module.getOps<MaterializationPlanOp>());
    SmallVector<SchedulePlanOp> schedules(module.getOps<SchedulePlanOp>());
    if (materializations.size() != 1 || schedules.size() != 1 ||
        !module.getOps<DeviceModuleOp>().empty() ||
        !module.getOps<InvocationAbiOp>().empty() ||
        !module.getOps<ExecutableBundleOp>().empty() ||
        failed(verifyMaterializationPlanAgainstSource(
            module, materializations.front())) ||
        failed(schedules.front().verifyRegions())) {
      module.emitError(
          "requires one source-bound plan and no executable bundle");
      return signalPassFailure();
    }
    FailureOr<CpuExecutableSpec> spec = deriveCpuExecutable(
        module, materializations.front(), schedules.front());
    if (failed(spec)) {
      module.emitError(
          "requires a bounded generated Map/Fold CPU bytecode subset");
      return signalPassFailure();
    }
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    const std::string codeDigest = executableCodeDigest(spec->code);

    OperationState deviceState(module.getLoc(),
                               DeviceModuleOp::getOperationName());
    deviceState.addAttribute("schema_version", builder.getI64IntegerAttr(1));
    deviceState.addAttribute(
        "code_format",
        ExecutableCodeFormatAttr::get(module.getContext(),
                                      ExecutableCodeFormat::CpuBytecodeV1));
    deviceState.addAttribute("policy", schedules.front().getPolicyAttr());
    deviceState.addAttribute("source_schedule_fingerprint",
                             schedules.front().getFingerprintAttr());
    deviceState.addAttribute(
        "code", DenseI8ArrayAttr::get(module.getContext(), spec->code));
    deviceState.addAttribute("code_digest", builder.getStringAttr(codeDigest));
    deviceState.addAttribute("fingerprint",
                             builder.getStringAttr(std::string(64, '0')));
    deviceState.addRegion();
    auto device = cast<DeviceModuleOp>(builder.create(deviceState));
    Block *deviceBody = new Block();
    device.getBody().push_back(deviceBody);
    builder.setInsertionPointToEnd(deviceBody);
    for (const CpuEntrySpec &entry : spec->entries) {
      SmallVector<NamedAttribute> attributes{
          builder.getNamedAttr("ordinal",
                               builder.getI64IntegerAttr(entry.ordinal)),
          builder.getNamedAttr("source_task",
                               builder.getI64IntegerAttr(entry.ordinal)),
          builder.getNamedAttr("code_offset",
                               builder.getI64IntegerAttr(entry.codeOffset)),
          builder.getNamedAttr("code_length",
                               builder.getI64IntegerAttr(entry.codeLength)),
          builder.getNamedAttr(
              "input_buffers",
              DenseI64ArrayAttr::get(module.getContext(), entry.inputs)),
          builder.getNamedAttr(
              "output_buffers",
              DenseI64ArrayAttr::get(module.getContext(), entry.outputs)),
          builder.getNamedAttr("input_accesses",
                               accessArray(module.getContext(),
                                           ExecutableAccess::Read,
                                           entry.inputs.size())),
          builder.getNamedAttr("output_accesses",
                               accessArray(module.getContext(),
                                           ExecutableAccess::Write,
                                           entry.outputs.size())),
          builder.getNamedAttr(
              "dependencies",
              DenseI64ArrayAttr::get(module.getContext(), entry.dependencies)),
          builder.getNamedAttr(
              "predication", ExecutablePredicationAttr::get(module.getContext(),
                                                            entry.predication)),
          builder.getNamedAttr("code_digest",
                               builder.getStringAttr(codeDigest))};
      if (entry.reductionOrder) {
        attributes.push_back(builder.getNamedAttr(
            "reduction_order",
            ScheduleReductionOrderAttr::get(module.getContext(),
                                            *entry.reductionOrder)));
      }
      createScheduleRecord(builder, module.getLoc(),
                           DeviceEntryOp::getOperationName(), attributes);
    }
    builder.create<DeviceModuleYieldOp>(module.getLoc());
    device.setFingerprint(deviceModuleFingerprint(device));

    builder.setInsertionPointToEnd(module.getBody());
    OperationState abiState(module.getLoc(),
                            InvocationAbiOp::getOperationName());
    abiState.addAttribute("schema_version", builder.getI64IntegerAttr(1));
    abiState.addAttribute("source_plan_fingerprint",
                          materializations.front().getFingerprintAttr());
    abiState.addAttribute("source_schedule_fingerprint",
                          schedules.front().getFingerprintAttr());
    abiState.addAttribute("fingerprint",
                          builder.getStringAttr(std::string(64, '0')));
    abiState.addRegion();
    auto abi = cast<InvocationAbiOp>(builder.create(abiState));
    Block *abiBody = new Block();
    abi.getBody().push_back(abiBody);
    builder.setInsertionPointToEnd(abiBody);
    for (auto [ordinal, slot] : llvm::enumerate(spec->slots)) {
      createScheduleRecord(
          builder, module.getLoc(), InvocationSlotOp::getOperationName(),
          {builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
           builder.getNamedAttr("source_buffer",
                                builder.getI64IntegerAttr(ordinal)),
           builder.getNamedAttr("tensor_type", TypeAttr::get(slot.tensorType)),
           builder.getNamedAttr("required_bytes",
                                builder.getI64IntegerAttr(slot.requiredBytes)),
           builder.getNamedAttr(
               "strides",
               DenseI64ArrayAttr::get(module.getContext(), slot.strides)),
           builder.getNamedAttr("offset", builder.getI64IntegerAttr(0)),
           builder.getNamedAttr("alignment",
                                builder.getI64IntegerAttr(slot.alignment)),
           builder.getNamedAttr(
               "address_space",
               ExecutableAddressSpaceAttr::get(module.getContext(),
                                               ExecutableAddressSpace::Host)),
           builder.getNamedAttr(
               "access",
               ExecutableAccessAttr::get(module.getContext(), slot.access)),
           builder.getNamedAttr(
               "storage", MaterializationStorageAttr::get(module.getContext(),
                                                          slot.storage)),
           builder.getNamedAttr("alias_group",
                                builder.getI64IntegerAttr(ordinal)),
           builder.getNamedAttr("reuse_group",
                                builder.getI64IntegerAttr(ordinal))});
    }
    builder.create<InvocationAbiYieldOp>(module.getLoc());
    abi.setFingerprint(invocationAbiFingerprint(abi));

    builder.setInsertionPointToEnd(module.getBody());
    OperationState bundleState(module.getLoc(),
                               ExecutableBundleOp::getOperationName());
    bundleState.addAttribute("schema_version", builder.getI64IntegerAttr(1));
    bundleState.addAttribute("source_schedule_fingerprint",
                             schedules.front().getFingerprintAttr());
    bundleState.addAttribute("device_module_fingerprint",
                             device.getFingerprintAttr());
    bundleState.addAttribute("invocation_abi_fingerprint",
                             abi.getFingerprintAttr());
    bundleState.addAttribute(
        "completion",
        ExecutableCompletionAttr::get(module.getContext(),
                                      ExecutableCompletion::Synchronous));
    bundleState.addAttribute("fingerprint",
                             builder.getStringAttr(std::string(64, '0')));
    auto bundle = cast<ExecutableBundleOp>(builder.create(bundleState));
    bundle.setFingerprint(executableBundleFingerprint(bundle));
  }
};

LogicalResult verifyCpuExecutableAgainstSource(ModuleOp module) {
  SmallVector<MaterializationPlanOp> materializations(
      module.getOps<MaterializationPlanOp>());
  SmallVector<SchedulePlanOp> schedules(module.getOps<SchedulePlanOp>());
  SmallVector<DeviceModuleOp> devices(module.getOps<DeviceModuleOp>());
  SmallVector<InvocationAbiOp> abis(module.getOps<InvocationAbiOp>());
  SmallVector<ExecutableBundleOp> bundles(module.getOps<ExecutableBundleOp>());
  if (materializations.size() != 1 || schedules.size() != 1 ||
      devices.size() != 1 || abis.size() != 1 || bundles.size() != 1 ||
      failed(verifyMaterializationPlanAgainstSource(
          module, materializations.front())) ||
      failed(schedules.front().verifyRegions()) ||
      failed(devices.front().verifyRegions()) ||
      failed(abis.front().verifyRegions()) ||
      failed(bundles.front().verify())) {
    return failure();
  }
  FailureOr<CpuExecutableSpec> expected =
      deriveCpuExecutable(module, materializations.front(), schedules.front());
  if (failed(expected) ||
      devices.front().getPolicy() != schedules.front().getPolicy() ||
      devices.front().getSourceScheduleFingerprint() !=
          schedules.front().getFingerprint() ||
      abis.front().getSourcePlanFingerprint() !=
          materializations.front().getFingerprint() ||
      devices.front().getCode() != ArrayRef<int8_t>(expected->code)) {
    return failure();
  }
  SmallVector<DeviceEntryOp> entries(
      devices.front().getBody().front().getOps<DeviceEntryOp>());
  SmallVector<InvocationSlotOp> slots(
      abis.front().getBody().front().getOps<InvocationSlotOp>());
  if (entries.size() != expected->entries.size() ||
      slots.size() != expected->slots.size()) {
    return failure();
  }
  for (auto [actual, spec] : llvm::zip_equal(entries, expected->entries)) {
    if (actual.getCodeOffset() != spec.codeOffset ||
        actual.getCodeLength() != spec.codeLength ||
        actual.getInputBuffers() != ArrayRef<int64_t>(spec.inputs) ||
        actual.getOutputBuffers() != ArrayRef<int64_t>(spec.outputs) ||
        actual.getDependencies() != ArrayRef<int64_t>(spec.dependencies) ||
        actual.getPredication() != spec.predication ||
        actual.getReductionOrder() != spec.reductionOrder) {
      return failure();
    }
  }
  for (auto [actual, spec] : llvm::zip_equal(slots, expected->slots)) {
    if (actual.getTensorType() != spec.tensorType ||
        actual.getRequiredBytes() != spec.requiredBytes ||
        actual.getStrides() != ArrayRef<int64_t>(spec.strides) ||
        actual.getOffset() != 0 || actual.getAlignment() != spec.alignment ||
        actual.getAddressSpace() != ExecutableAddressSpace::Host ||
        actual.getAccess() != spec.access ||
        actual.getStorage() != spec.storage ||
        actual.getAliasGroup() != actual.getOrdinal() ||
        actual.getReuseGroup() != actual.getOrdinal()) {
      return failure();
    }
  }
  return success();
}

class VerifyCpuExecutableBundlePass
    : public impl::ShuttleVerifyCpuExecutableBundlePassBase<
          VerifyCpuExecutableBundlePass> {
public:
  void runOnOperation() override {
    if (failed(verifyCpuExecutableAgainstSource(getOperation()))) {
      getOperation().emitError(
          "executable bundle no longer matches Shuttle algebra and schedule");
      signalPassFailure();
    }
  }
};

void emitObserverSnapshot(
    const std::shared_ptr<detail::ShuttleObserverInvocation> &invocation,
    ModuleOp module, ShuttlePipelinePhase phase,
    llvm::StringRef failurePass = {}) {
  std::string regionMembership;
  std::string coverageManifest;
  std::string unsupportedFingerprint;
  if (auto manifest =
          module->getAttrOfType<DictionaryAttr>(kCoverageManifestAttribute)) {
    coverageManifest = attributeText(manifest);
    if (ArrayAttr regions =
            manifest.getAs<ArrayAttr>(kManifestSelectedRegions)) {
      regionMembership = attributeText(regions);
    }
    if (ArrayAttr excluded = manifest.getAs<ArrayAttr>(kManifestExcluded)) {
      unsupportedFingerprint = sha256(attributeText(excluded));
    }
  }
  invocation->emit(
      phase, std::move(regionMembership), std::move(coverageManifest),
      std::move(unsupportedFingerprint),
      phase == ShuttlePipelinePhase::FinalErasure
          ? normalizedModuleFingerprintImpl(module)
          : std::string{},
      phase == ShuttlePipelinePhase::FinalErasure, failurePass.str());
}

class EmitObserverPass
    : public PassWrapper<EmitObserverPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitObserverPass)

  EmitObserverPass(
      ShuttlePipelinePhase phase,
      std::shared_ptr<detail::ShuttleObserverInvocation> invocation)
      : phase(phase), invocation(std::move(invocation)) {}

  StringRef getArgument() const final { return "shuttle-observer-event"; }
  StringRef getDescription() const final {
    return "Emit one immutable Shuttle pipeline observer snapshot";
  }

  void runOnOperation() override {
    emitObserverSnapshot(invocation, getOperation(), phase);
  }

private:
  ShuttlePipelinePhase phase;
  std::shared_ptr<detail::ShuttleObserverInvocation> invocation;
};

struct FailureCaptureState {
  std::string pass;
};

class FailureCaptureInstrumentation final : public PassInstrumentation {
public:
  explicit FailureCaptureInstrumentation(
      std::shared_ptr<FailureCaptureState> state)
      : state(std::move(state)) {}

  void runAfterPassFailed(Pass *pass, Operation *) final {
    llvm::StringRef argument = pass->getArgument();
    state->pass = argument.empty() ? pass->getName().str() : argument.str();
  }

private:
  std::shared_ptr<FailureCaptureState> state;
};

void buildShuttleStablehloCorePipeline(
    OpPassManager &manager, const ShuttlePipelineOptions &options,
    const std::shared_ptr<detail::ShuttleObserverInvocation> &invocation) {
  manager.addPass(createAnnotateSourcePass());
  manager.addPass(createFormStructuralRegionsPass(
      options.numerics, options.canonicalOptions, options.canonicalTuning));
  manager.addPass(createConvertStablehloToAlgebraPass());
  manager.addPass(createVerifySourceCoveragePass());
  manager.addPass(std::make_unique<EmitObserverPass>(
      ShuttlePipelinePhase::AlgebraCoverage, invocation));
  manager.addPass(createVerifySemanticErasurePass());
  manager.addPass(createShuttleCanonicalizePass());
  manager.addPass(createLowerAlgebraToStablehloPass());
  manager.addPass(createVerifySourceCoveragePass());
  manager.addPass(std::make_unique<EmitObserverPass>(
      ShuttlePipelinePhase::LoweredCoverage, invocation));
  manager.addPass(createStripSourceProvenancePass());
  manager.addPass(createVerifyNoShuttleOpsPass());
  manager.addPass(std::make_unique<EmitObserverPass>(
      ShuttlePipelinePhase::FinalErasure, invocation));
}

class RunShuttleStablehloPipelinePass
    : public PassWrapper<RunShuttleStablehloPipelinePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RunShuttleStablehloPipelinePass)
  using Base =
      PassWrapper<RunShuttleStablehloPipelinePass, OperationPass<ModuleOp>>;

  RunShuttleStablehloPipelinePass(
      ShuttlePipelineOptions options,
      std::shared_ptr<const ShuttlePipelineObserver> observer)
      : options(std::move(options)), observer(std::move(observer)) {}
  RunShuttleStablehloPipelinePass(const RunShuttleStablehloPipelinePass &other)
      : Base(other), options(other.options), observer(other.observer) {}

  StringRef getArgument() const final { return "shuttle-stablehlo-pipeline"; }
  StringRef getDescription() const final {
    return "Run one observed fail-closed Shuttle StableHLO invocation";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, ShuttleDialect,
                    stablehlo::StablehloDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto invocation = detail::beginShuttleObserverInvocation(
        shuttlePipelineIdentity(options), observer);
    auto failure = std::make_shared<FailureCaptureState>();
    PassManager manager(module.getContext(), ModuleOp::getOperationName());
    manager.addInstrumentation(
        std::make_unique<FailureCaptureInstrumentation>(failure));
    buildShuttleStablehloCorePipeline(manager, options, invocation);
    if (failed(manager.run(module))) {
      emitObserverSnapshot(invocation, module, ShuttlePipelinePhase::Failure,
                           failure->pass);
      signalPassFailure();
    }
  }

private:
  ShuttlePipelineOptions options;
  std::shared_ptr<const ShuttlePipelineObserver> observer;
};

} // namespace

std::string normalizedStablehloFingerprint(ModuleOp module) {
  return normalizedModuleFingerprintImpl(module);
}

std::unique_ptr<Pass> createAnnotateSourcePass() {
  return std::make_unique<AnnotateSourcePass>();
}

std::unique_ptr<Pass> createFormStructuralRegionsPass() {
  return std::make_unique<FormStructuralRegionsPass>();
}

std::unique_ptr<Pass>
createFormStructuralRegionsPass(NumericalPolicy numerics,
                                std::string canonicalOptions,
                                std::string canonicalTuning) {
  return std::make_unique<FormStructuralRegionsPass>(
      numerics, std::move(canonicalOptions), std::move(canonicalTuning));
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

std::unique_ptr<Pass> createStripSourceProvenancePass() {
  return std::make_unique<StripSourceProvenancePass>();
}

std::unique_ptr<Pass> createVerifyNoShuttleOpsPass() {
  return std::make_unique<VerifyNoShuttleOpsPass>();
}

std::unique_ptr<Pass> createPlanRowFoldMaterializationPass() {
  return std::make_unique<PlanRowFoldMaterializationPass>();
}

std::unique_ptr<Pass> createVerifyMaterializationPlanPass() {
  return std::make_unique<VerifyMaterializationPlanPass>();
}

std::unique_ptr<Pass> createPlanSimt32RowFoldSchedulePass() {
  return std::make_unique<PlanSimt32RowFoldSchedulePass>();
}

std::unique_ptr<Pass> createVerifySimt32RowFoldSchedulePass() {
  return std::make_unique<VerifySimt32RowFoldSchedulePass>();
}

std::unique_ptr<Pass> createBuildCpuExecutableBundlePass() {
  return std::make_unique<BuildCpuExecutableBundlePass>();
}

std::unique_ptr<Pass> createVerifyCpuExecutableBundlePass() {
  return std::make_unique<VerifyCpuExecutableBundlePass>();
}

void buildShuttleStablehloPipeline(
    OpPassManager &manager, const ShuttlePipelineOptions &options,
    std::shared_ptr<const ShuttlePipelineObserver> observer) {
  manager.addPass(std::make_unique<RunShuttleStablehloPipelinePass>(
      options, std::move(observer)));
}

void registerShuttleStablehloPipelines() {
  PassPipelineRegistration<>(
      "shuttle-stablehlo-source-ordered-pipeline",
      "Run the complete source-ordered Shuttle StableHLO pipeline",
      [](OpPassManager &manager) {
        buildShuttleStablehloPipeline(
            manager,
            commandLinePipelineOptions(NumericalPolicy::SourceOrdered));
      });
  PassPipelineRegistration<>(
      "shuttle-stablehlo-fast-pipeline",
      "Run the complete fast-policy Shuttle StableHLO pipeline",
      [](OpPassManager &manager) {
        buildShuttleStablehloPipeline(
            manager, commandLinePipelineOptions(NumericalPolicy::Fast));
      });
}

} // namespace mlir::shuttle
