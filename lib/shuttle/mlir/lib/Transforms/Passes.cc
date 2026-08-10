// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Transforms/Passes.h"

#include <cstdint>
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
#include "shuttle/Transforms/Passes.h.inc"

namespace {

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
    auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
    auto functionOrdinal =
        function
            ? function->getAttrOfType<IntegerAttr>(kFunctionOrdinalAttribute)
            : IntegerAttr{};
    if (!functionOrdinal ||
        &function.getBody().front() != argument.getOwner()) {
      return {};
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

bool hasOnlyRankedF32(ValueRange values) {
  return llvm::all_of(values, [](Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type && type.getElementType().isF32();
  });
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

bool isSupportedStablehlo(Operation *operation) {
  if (operation->getNumRegions() != 0 || !isMemoryEffectFree(operation) ||
      !hasOnlyRankedF32(operation->getOperands()) ||
      !hasOnlyRankedF32(operation->getResults())) {
    return false;
  }

  StringRef name = operation->getName().getStringRef();
  if (name == stablehlo::DotGeneralOp::getOperationName()) {
    if (operation->getNumOperands() != 2 || operation->getNumResults() != 1 ||
        operation->hasAttr("algorithm") || !hasDefaultDotPrecision(operation)) {
      return false;
    }
    return static_cast<bool>(
        operation->getAttrOfType<stablehlo::DotDimensionNumbersAttr>(
            "dot_dimension_numbers"));
  }
  if (name == stablehlo::TanhOp::getOperationName()) {
    return operation->getNumOperands() == 1 && operation->getNumResults() == 1;
  }
  if (name == stablehlo::AddOp::getOperationName() ||
      name == stablehlo::MulOp::getOperationName()) {
    return operation->getNumOperands() == 2 &&
           operation->getNumResults() == 1 &&
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
           permutation.size() == static_cast<size_t>(input.getRank());
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
  return components;
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
      NamedAttribute(StringAttr::get(context, "classification"),
                     StringAttr::get(context, isa<func::ReturnOp>(operation)
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
                      NumericalPolicy numerics, StringRef canonicalTuning) {
  MLIRContext *context = module.getContext();
  llvm::SmallDenseSet<Attribute> selected;
  SmallVector<Attribute> selectedRegions;
  for (const CandidateComponent &component : components) {
    SmallVector<Attribute> refs;
    for (Operation *operation : component.operations) {
      ArrayAttr operationRefs = operationSourceRefs(operation);
      if (!operationRefs) {
        operation->emitOpError("is missing structural source references");
        return failure();
      }
      for (Attribute ref : operationRefs) {
        if (!selected.insert(ref).second) {
          module.emitError("a source result belongs to two regions");
          return failure();
        }
        refs.push_back(ref);
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
    for (Block &block : function.getBody()) {
      for (Operation &operation : block) {
        if (operation.getNumResults() == 0) {
          DictionaryAttr record = zeroResultRecord(&operation);
          if (!record) {
            operation.emitOpError(
                "has an operand without a structural source anchor");
            return failure();
          }
          zeroResultOperations.push_back(record);
          continue;
        }
        ArrayAttr operationRefs = operationSourceRefs(&operation);
        if (!operationRefs) {
          operation.emitOpError("is missing structural source references");
          return failure();
        }
        SmallVector<Attribute> operandAnchors;
        for (Value operand : operation.getOperands()) {
          Attribute anchor = valueAnchor(operand);
          if (!anchor) {
            operation.emitOpError(
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
                             normalizedOperationFingerprint(&operation)),
              NamedAttribute(StringAttr::get(context, "operands"),
                             ArrayAttr::get(context, operandAnchors)),
              NamedAttribute(StringAttr::get(context, "source"), ref),
              NamedAttribute(StringAttr::get(context, "reason"),
                             StringAttr::get(context, "unsupported_operation")),
          };
          excluded.push_back(DictionaryAttr::get(context, fields));
        }
      }
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
  std::string semanticPolicy =
      (policyName(numerics) + "\n" + tuningDigest).str();
  NamedAttribute fields[] = {
      NamedAttribute(StringAttr::get(context, "version"),
                     IntegerAttr::get(IntegerType::get(context, 64), 1)),
      NamedAttribute(StringAttr::get(context, "policy"),
                     StringAttr::get(context, policyName(numerics))),
      NamedAttribute(StringAttr::get(context, "policy_digest"),
                     StringAttr::get(context, sha256(semanticPolicy))),
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
        } else {
          operation.setAttr(
              kOperationRefAttribute,
              DenseI64ArrayAttr::get(operation.getContext(),
                                     {static_cast<int64_t>(functionOrdinal),
                                      static_cast<int64_t>(blockOrdinal),
                                      static_cast<int64_t>(operationOrdinal)}));
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
  explicit FormStructuralRegionsPass(NumericalPolicy numerics)
      : numerics(numerics) {}
  FormStructuralRegionsPass(NumericalPolicy numerics,
                            std::string canonicalTuning)
      : numerics(numerics), canonicalTuning(std::move(canonicalTuning)) {}

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
      for (Operation &operation : function.getBody().front()) {
        if (operation.getNumRegions() != 0) {
          operation.emitOpError(
              "the first offline slice requires region-free source operations");
          signalPassFailure();
          return;
        }
      }
    }

    SmallVector<CandidateComponent> components = candidateComponents(module);
    FailureOr<DictionaryAttr> manifest =
        buildCoverageManifest(module, components, numerics, canonicalTuning);
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
      ArrayAttr refs = operationSourceRefs(operation);
      if (!refs) {
        return operation->emitOpError(
            "is missing structural source references");
      }
      llvm::append_range(declaredSources, refs);
      operation->setAttr(kSelectedAttribute, UnitAttr::get(context));
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
  std::string canonicalTuning = "{}";
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

Operation *createScalarOperation(OpBuilder &builder, Location location,
                                 StringRef stablehloName,
                                 ValueRange arguments) {
  StringRef scalarName;
  if (stablehloName == stablehlo::TanhOp::getOperationName()) {
    scalarName = math::TanhOp::getOperationName();
  } else if (stablehloName == stablehlo::MulOp::getOperationName()) {
    scalarName = arith::MulFOp::getOperationName();
  } else if (stablehloName == stablehlo::AddOp::getOperationName()) {
    scalarName = arith::AddFOp::getOperationName();
  } else {
    return nullptr;
  }
  OperationState state(location, scalarName);
  state.addOperands(arguments);
  state.addTypes(builder.getF32Type());
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
  if (stablehloName == stablehlo::TransposeOp::getOperationName()) {
    scalarResult = body->getArgument(0);
  } else {
    Operation *scalar = createScalarOperation(
        builder, operation->getLoc(), stablehloName, body->getArguments());
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
            name == stablehlo::DotGeneralOp::getOperationName()
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
    if (!source || !reason || reason.getValue() != "unsupported_operation" ||
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
    for (Operation &operation : function.getBody().front()) {
      ArrayAttr refs = sourceRefs(&operation);
      if (!refs) {
        continue;
      }
      SmallVector<Attribute> operandAnchors;
      for (Value operand : operation.getOperands()) {
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
            NamedAttribute(StringAttr::get(module.getContext(), "fingerprint"),
                           normalizedOperationFingerprint(&operation)),
            NamedAttribute(StringAttr::get(module.getContext(), "operands"),
                           ArrayAttr::get(module.getContext(), operandAnchors)),
            NamedAttribute(StringAttr::get(module.getContext(), "source"),
                           source),
            NamedAttribute(StringAttr::get(module.getContext(), "reason"),
                           reason->second),
        };
        records.push_back(DictionaryAttr::get(module.getContext(), fields));
      }
    }
  }
  return ArrayAttr::get(module.getContext(), records);
}

FailureOr<ArrayAttr> currentZeroResultRecords(ModuleOp module) {
  SmallVector<Attribute> records;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    for (Block &block : function.getBody()) {
      for (Operation &operation : block) {
        if (operation.getNumResults() != 0) {
          continue;
        }
        DictionaryAttr record = zeroResultRecord(&operation);
        if (!record) {
          return failure();
        }
        records.push_back(record);
      }
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

LogicalResult verifyManifestCoverage(ModuleOp module, DictionaryAttr manifest) {
  auto completeArray = manifest.getAs<ArrayAttr>(kManifestComplete);
  auto selectedGroups = manifest.getAs<ArrayAttr>(kManifestSelectedRegions);
  auto policy = manifest.getAs<StringAttr>("policy");
  auto policyDigest = manifest.getAs<StringAttr>("policy_digest");
  auto tuningDigest = manifest.getAs<StringAttr>("tuning_digest");
  FailureOr<llvm::SmallDenseSet<Attribute>> selected =
      selectedManifestSources(manifest);
  FailureOr<llvm::SmallDenseSet<Attribute>> excluded =
      excludedManifestSources(manifest);
  if (!completeArray || !selectedGroups || !policy || !policyDigest ||
      !tuningDigest || failed(selected) || failed(excluded)) {
    return module.emitError("has a malformed Shuttle coverage manifest");
  }
  if ((policy.getValue() != "source_ordered" && policy.getValue() != "fast") ||
      tuningDigest.getValue().size() != 64 ||
      !llvm::all_of(tuningDigest.getValue(), llvm::isHexDigit) ||
      policyDigest.getValue() !=
          sha256((policy.getValue() + "\n" + tuningDigest.getValue()).str())) {
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
      return recordSelected(operation, fold.getSource());
    }
    ArrayAttr refs = sourceRefs(operation);
    if (!refs) {
      return WalkResult::advance();
    }
    for (Attribute source : refs) {
      if (selected->contains(source)) {
        if (algebraStage) {
          operation->emitOpError(
              "selected source operation survived algebra conversion");
          return WalkResult::interrupt();
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
      *zeroResultRecords !=
          manifest.getAs<ArrayAttr>(kManifestZeroResultOperations) ||
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
        inputMap.getNumResults() != resultMap.getNumResults()) {
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
    if (scalarOperations.size() != 1 ||
        failed(verifyDefaultScalarSemantics(scalarOperations.front()))) {
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
      stablehloName = stablehlo::TanhOp::getOperationName();
    } else if (scalarName == arith::MulFOp::getOperationName() &&
               operands.size() == 2) {
      stablehloName = stablehlo::MulOp::getOperationName();
    } else if (scalarName == arith::AddFOp::getOperationName() &&
               operands.size() == 2) {
      stablehloName = stablehlo::AddOp::getOperationName();
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
    for (Operation &operation : function.getBody().front()) {
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
    }
  }
  stream.flush();
  return sha256(normalized);
}

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
  manager.addPass(createFormStructuralRegionsPass(options.numerics,
                                                  options.canonicalTuning));
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
createFormStructuralRegionsPass(NumericalPolicy numerics) {
  return std::make_unique<FormStructuralRegionsPass>(numerics);
}

std::unique_ptr<Pass>
createFormStructuralRegionsPass(NumericalPolicy numerics,
                                std::string canonicalTuning) {
  return std::make_unique<FormStructuralRegionsPass>(
      numerics, std::move(canonicalTuning));
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
            manager, ShuttlePipelineOptions{NumericalPolicy::SourceOrdered});
      });
  PassPipelineRegistration<>(
      "shuttle-stablehlo-fast-pipeline",
      "Run the complete fast-policy Shuttle StableHLO pipeline",
      [](OpPassManager &manager) {
        buildShuttleStablehloPipeline(
            manager, ShuttlePipelineOptions{NumericalPolicy::Fast});
      });
}

} // namespace mlir::shuttle
