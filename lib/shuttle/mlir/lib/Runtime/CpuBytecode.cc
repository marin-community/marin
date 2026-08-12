// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Runtime/CpuBytecode.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinTypes.h"
#include "shuttle/IR/ShuttleOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace mlir::shuttle {
namespace {

enum class Opcode : uint8_t {
  ConstantF32 = 0,
  AddF32 = 1,
  MultiplyF32 = 2,
  DivideF32 = 3,
  RsqrtF32 = 4,
  Bf16ToF32 = 5,
  F32ToBf16Rne = 6,
};

enum class TaskKind : uint8_t { Map = 0, RowFold = 1 };
enum class ElementType : uint8_t { Bf16 = 0, F32 = 1 };

struct IndexAxis {
  uint8_t domainAxis;
  uint32_t divisor;
};

struct InputMap {
  ElementType elementType;
  SmallVector<IndexAxis> axes;
};

struct Instruction {
  Opcode opcode;
  SmallVector<uint8_t, 2> operands;
  uint32_t immediate = 0;
  ElementType resultType;
};

struct ParsedTask {
  TaskKind kind;
  SmallVector<int64_t> domain;
  SmallVector<InputMap> inputs;
  ElementType outputType;
  std::optional<uint8_t> reductionAxis;
  std::optional<ScheduleReductionOrder> reductionOrder;
  SmallVector<ElementType> argumentTypes;
  SmallVector<Instruction> instructions;
  uint8_t yieldRegister;
};

class Reader {
public:
  explicit Reader(ArrayRef<int8_t> bytes) : bytes(bytes) {}

  std::optional<uint8_t> byte() {
    if (position == bytes.size()) {
      return std::nullopt;
    }
    return static_cast<uint8_t>(bytes[position++]);
  }

  std::optional<uint32_t> u32() {
    uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
      std::optional<uint8_t> next = byte();
      if (!next) {
        return std::nullopt;
      }
      value |= static_cast<uint32_t>(*next) << shift;
    }
    return value;
  }

  bool done() const { return position == bytes.size(); }

private:
  ArrayRef<int8_t> bytes;
  size_t position = 0;
};

template <typename Enum>
std::optional<Enum> checkedEnum(std::optional<uint8_t> value, uint8_t maximum) {
  if (!value || *value > maximum) {
    return std::nullopt;
  }
  return static_cast<Enum>(*value);
}

FailureOr<ParsedTask> parseTask(ArrayRef<int8_t> bytes) {
  Reader reader(bytes);
  for (uint8_t expected : ArrayRef<uint8_t>{'S', 'B', 'C', 1}) {
    if (reader.byte() != expected) {
      return failure();
    }
  }
  std::optional<TaskKind> kind = checkedEnum<TaskKind>(reader.byte(), 1);
  std::optional<uint8_t> rank = reader.byte();
  if (!kind || !rank) {
    return failure();
  }
  ParsedTask task;
  task.kind = *kind;
  for (uint8_t axis = 0; axis < *rank; ++axis) {
    std::optional<uint32_t> extent = reader.u32();
    if (!extent || *extent == 0) {
      return failure();
    }
    task.domain.push_back(*extent);
  }
  std::optional<uint8_t> inputCount = reader.byte();
  if (!inputCount) {
    return failure();
  }
  for (uint8_t input = 0; input < *inputCount; ++input) {
    std::optional<ElementType> type =
        checkedEnum<ElementType>(reader.byte(), 1);
    std::optional<uint8_t> inputRank = reader.byte();
    if (!type || !inputRank) {
      return failure();
    }
    InputMap map{*type, {}};
    for (uint8_t axis = 0; axis < *inputRank; ++axis) {
      std::optional<uint8_t> domainAxis = reader.byte();
      std::optional<uint32_t> divisor = reader.u32();
      if (!domainAxis || *domainAxis >= task.domain.size() || !divisor ||
          *divisor == 0 ||
          (*divisor != 1 && task.domain[*domainAxis] > *divisor)) {
        return failure();
      }
      map.axes.push_back(IndexAxis{*domainAxis, *divisor});
    }
    task.inputs.push_back(std::move(map));
  }
  std::optional<ElementType> outputType =
      checkedEnum<ElementType>(reader.byte(), 1);
  if (!outputType) {
    return failure();
  }
  task.outputType = *outputType;
  if (task.kind == TaskKind::RowFold) {
    std::optional<uint8_t> reductionAxis = reader.byte();
    std::optional<ScheduleReductionOrder> order =
        checkedEnum<ScheduleReductionOrder>(reader.byte(), 1);
    if (!reductionAxis || *reductionAxis >= task.domain.size() || !order) {
      return failure();
    }
    task.reductionAxis = reductionAxis;
    task.reductionOrder = order;
  }
  std::optional<uint8_t> argumentCount = reader.byte();
  if (!argumentCount || *argumentCount != task.inputs.size()) {
    return failure();
  }
  for (uint8_t argument = 0; argument < *argumentCount; ++argument) {
    std::optional<ElementType> type =
        checkedEnum<ElementType>(reader.byte(), 1);
    if (!type || *type != task.inputs[argument].elementType) {
      return failure();
    }
    task.argumentTypes.push_back(*type);
  }
  std::optional<uint8_t> instructionCount = reader.byte();
  if (!instructionCount) {
    return failure();
  }
  SmallVector<ElementType> registerTypes(task.argumentTypes);
  for (uint8_t index = 0; index < *instructionCount; ++index) {
    std::optional<Opcode> opcode = checkedEnum<Opcode>(reader.byte(), 6);
    if (!opcode) {
      return failure();
    }
    Instruction instruction;
    instruction.opcode = *opcode;
    unsigned operandCount = 0;
    if (*opcode == Opcode::AddF32 || *opcode == Opcode::MultiplyF32 ||
        *opcode == Opcode::DivideF32) {
      operandCount = 2;
    } else if (*opcode != Opcode::ConstantF32) {
      operandCount = 1;
    }
    if (*opcode == Opcode::ConstantF32) {
      std::optional<uint32_t> immediate = reader.u32();
      if (!immediate) {
        return failure();
      }
      instruction.immediate = *immediate;
    }
    for (unsigned operand = 0; operand < operandCount; ++operand) {
      std::optional<uint8_t> position = reader.byte();
      if (!position || *position >= registerTypes.size()) {
        return failure();
      }
      instruction.operands.push_back(*position);
    }
    std::optional<ElementType> resultType =
        checkedEnum<ElementType>(reader.byte(), 1);
    if (!resultType) {
      return failure();
    }
    instruction.resultType = *resultType;
    auto isF32 = [&](unsigned operand) {
      return registerTypes[instruction.operands[operand]] == ElementType::F32;
    };
    const bool valid =
        (*opcode == Opcode::ConstantF32 && *resultType == ElementType::F32) ||
        ((*opcode == Opcode::AddF32 || *opcode == Opcode::MultiplyF32 ||
          *opcode == Opcode::DivideF32) &&
         isF32(0) && isF32(1) && *resultType == ElementType::F32) ||
        (*opcode == Opcode::RsqrtF32 && isF32(0) &&
         *resultType == ElementType::F32) ||
        (*opcode == Opcode::Bf16ToF32 &&
         registerTypes[instruction.operands[0]] == ElementType::Bf16 &&
         *resultType == ElementType::F32) ||
        (*opcode == Opcode::F32ToBf16Rne && isF32(0) &&
         *resultType == ElementType::Bf16);
    if (!valid || registerTypes.size() == UINT8_MAX) {
      return failure();
    }
    registerTypes.push_back(*resultType);
    task.instructions.push_back(std::move(instruction));
  }
  std::optional<uint8_t> yielded = reader.byte();
  if (!yielded || *yielded >= registerTypes.size() ||
      registerTypes[*yielded] != task.outputType || !reader.done()) {
    return failure();
  }
  task.yieldRegister = *yielded;
  return task;
}

struct Scalar {
  ElementType type;
  uint32_t bits;
};

float scalarF32(Scalar value) {
  float result;
  std::memcpy(&result, &value.bits, sizeof(result));
  return result;
}

Scalar f32Scalar(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return Scalar{ElementType::F32, bits};
}

Scalar evaluate(const ParsedTask &task, ArrayRef<Scalar> arguments) {
  SmallVector<Scalar> registers(arguments);
  for (const Instruction &instruction : task.instructions) {
    switch (instruction.opcode) {
    case Opcode::ConstantF32: {
      registers.push_back(Scalar{ElementType::F32, instruction.immediate});
      break;
    }
    case Opcode::AddF32: {
      const float lhs = scalarF32(registers[instruction.operands[0]]);
      const float rhs = scalarF32(registers[instruction.operands[1]]);
      registers.push_back(f32Scalar(static_cast<float>(lhs + rhs)));
      break;
    }
    case Opcode::MultiplyF32: {
      const float lhs = scalarF32(registers[instruction.operands[0]]);
      const float rhs = scalarF32(registers[instruction.operands[1]]);
      registers.push_back(f32Scalar(static_cast<float>(lhs * rhs)));
      break;
    }
    case Opcode::DivideF32: {
      const float lhs = scalarF32(registers[instruction.operands[0]]);
      const float rhs = scalarF32(registers[instruction.operands[1]]);
      registers.push_back(f32Scalar(static_cast<float>(lhs / rhs)));
      break;
    }
    case Opcode::RsqrtF32: {
      const float value = scalarF32(registers[instruction.operands[0]]);
      const float root = std::sqrt(value);
      registers.push_back(f32Scalar(static_cast<float>(1.0f / root)));
      break;
    }
    case Opcode::Bf16ToF32: {
      const uint32_t bits = registers[instruction.operands[0]].bits << 16;
      registers.push_back(Scalar{ElementType::F32, bits});
      break;
    }
    case Opcode::F32ToBf16Rne: {
      const uint32_t bits = registers[instruction.operands[0]].bits;
      registers.push_back(Scalar{ElementType::Bf16, roundF32ToBf16Rne(bits)});
      break;
    }
    }
  }
  return registers[task.yieldRegister];
}

struct RuntimeSlot {
  InvocationSlotOp descriptor;
  MutableArrayRef<uint8_t> bytes;
};

FailureOr<int64_t> byteOffset(InvocationSlotOp slot,
                              ArrayRef<int64_t> coordinates) {
  auto type = cast<RankedTensorType>(slot.getTensorType());
  if (coordinates.size() != static_cast<size_t>(type.getRank())) {
    return failure();
  }
  int64_t offset = slot.getOffset();
  for (auto [axis, coordinate] : llvm::enumerate(coordinates)) {
    if (coordinate < 0 || coordinate >= type.getDimSize(axis) ||
        coordinate > (std::numeric_limits<int64_t>::max() - offset) /
                         slot.getStrides()[axis]) {
      return failure();
    }
    offset += coordinate * slot.getStrides()[axis];
  }
  return offset;
}

FailureOr<Scalar> loadScalar(const RuntimeSlot &slot,
                             ArrayRef<int64_t> coordinates) {
  FailureOr<int64_t> offset = byteOffset(slot.descriptor, coordinates);
  InvocationSlotOp descriptor = slot.descriptor;
  auto type = cast<RankedTensorType>(descriptor.getTensorType());
  const bool bf16 = type.getElementType().isBF16();
  const size_t width = bf16 ? 2 : 4;
  if (failed(offset) || *offset < 0 || slot.bytes.size() < width ||
      static_cast<size_t>(*offset) > slot.bytes.size() - width) {
    return failure();
  }
  uint32_t bits = 0;
  std::memcpy(&bits, slot.bytes.data() + *offset, width);
  return Scalar{bf16 ? ElementType::Bf16 : ElementType::F32, bits};
}

LogicalResult storeScalar(const RuntimeSlot &slot,
                          ArrayRef<int64_t> coordinates, Scalar value) {
  FailureOr<int64_t> offset = byteOffset(slot.descriptor, coordinates);
  InvocationSlotOp descriptor = slot.descriptor;
  auto type = cast<RankedTensorType>(descriptor.getTensorType());
  const bool bf16 = type.getElementType().isBF16();
  const size_t width = bf16 ? 2 : 4;
  if (failed(offset) ||
      value.type != (bf16 ? ElementType::Bf16 : ElementType::F32) ||
      *offset < 0 || slot.bytes.size() < width ||
      static_cast<size_t>(*offset) > slot.bytes.size() - width) {
    return failure();
  }
  std::memcpy(slot.bytes.data() + *offset, &value.bits, width);
  return success();
}

SmallVector<int64_t> coordinatesForLinear(ArrayRef<int64_t> shape,
                                          int64_t linear) {
  SmallVector<int64_t> coordinates(shape.size());
  for (int64_t axis = shape.size(); axis > 0; --axis) {
    coordinates[axis - 1] = linear % shape[axis - 1];
    linear /= shape[axis - 1];
  }
  return coordinates;
}

LogicalResult executeMap(const ParsedTask &task, DeviceEntryOp entry,
                         ArrayRef<RuntimeSlot> slots) {
  if ((task.domain.empty() &&
       entry.getPredication() != ExecutablePredication::None) ||
      (!task.domain.empty() &&
       entry.getPredication() != ExecutablePredication::DomainBounds) ||
      entry.getInputBuffers().size() != task.inputs.size() ||
      entry.getOutputBuffers().size() != 1) {
    return failure();
  }
  int64_t elements = 1;
  for (int64_t extent : task.domain) {
    elements *= extent;
  }
  RuntimeSlot output = slots[entry.getOutputBuffers().front()];
  auto outputType = cast<RankedTensorType>(output.descriptor.getTensorType());
  if (outputType.getShape() != ArrayRef<int64_t>(task.domain)) {
    return failure();
  }
  for (int64_t linear = 0; linear < elements; ++linear) {
    SmallVector<int64_t> domainCoordinates =
        coordinatesForLinear(task.domain, linear);
    SmallVector<Scalar> arguments;
    for (auto [ordinal, inputMap] : llvm::enumerate(task.inputs)) {
      SmallVector<int64_t> inputCoordinates;
      for (IndexAxis axis : inputMap.axes) {
        inputCoordinates.push_back(domainCoordinates[axis.domainAxis] /
                                   axis.divisor);
      }
      FailureOr<Scalar> value =
          loadScalar(slots[entry.getInputBuffers()[ordinal]], inputCoordinates);
      if (failed(value) || value->type != inputMap.elementType) {
        return failure();
      }
      arguments.push_back(*value);
    }
    Scalar result = evaluate(task, arguments);
    if (failed(storeScalar(output, domainCoordinates, result))) {
      return failure();
    }
  }
  return success();
}

LogicalResult executeFold(const ParsedTask &task, DeviceEntryOp entry,
                          ArrayRef<RuntimeSlot> slots) {
  if (task.domain.size() != 2 || task.reductionAxis != 1 ||
      task.reductionOrder !=
          ScheduleReductionOrder::TreeAssociationFreeLeafOrderFixed ||
      entry.getReductionOrder() != task.reductionOrder ||
      entry.getPredication() != ExecutablePredication::DomainBounds ||
      task.inputs.size() != 2 ||
      task.inputs[0].elementType != ElementType::F32 ||
      task.inputs[0].axes.size() != 2 ||
      task.inputs[0].axes[0].domainAxis != 0 ||
      task.inputs[0].axes[0].divisor != 1 ||
      task.inputs[0].axes[1].domainAxis != 1 ||
      task.inputs[0].axes[1].divisor != 1 ||
      task.inputs[1].elementType != ElementType::F32 ||
      !task.inputs[1].axes.empty() || task.outputType != ElementType::F32 ||
      entry.getInputBuffers().size() != 2 ||
      entry.getOutputBuffers().size() != 1) {
    return failure();
  }
  RuntimeSlot input = slots[entry.getInputBuffers()[0]];
  RuntimeSlot initializer = slots[entry.getInputBuffers()[1]];
  RuntimeSlot output = slots[entry.getOutputBuffers()[0]];
  auto inputType = cast<RankedTensorType>(input.descriptor.getTensorType());
  auto initializerType =
      cast<RankedTensorType>(initializer.descriptor.getTensorType());
  auto outputType = cast<RankedTensorType>(output.descriptor.getTensorType());
  if (!inputType.getElementType().isF32() ||
      inputType.getShape() != ArrayRef<int64_t>(task.domain) ||
      !initializerType.getElementType().isF32() ||
      initializerType.getRank() != 0 || !outputType.getElementType().isF32() ||
      outputType.getShape() != ArrayRef<int64_t>(task.domain).take_front()) {
    return failure();
  }
  for (int64_t row = 0; row < task.domain[0]; ++row) {
    FailureOr<Scalar> accumulator = loadScalar(initializer, {});
    if (failed(accumulator)) {
      return failure();
    }
    for (int64_t feature = 0; feature < task.domain[1]; ++feature) {
      FailureOr<Scalar> leaf = loadScalar(input, {row, feature});
      if (failed(leaf)) {
        return failure();
      }
      std::array<Scalar, 2> arguments{*leaf, *accumulator};
      accumulator = evaluate(task, arguments);
    }
    if (failed(storeScalar(output, {row}, *accumulator))) {
      return failure();
    }
  }
  return success();
}

LogicalResult runtimeError(ModuleOp module, StringRef message) {
  module.emitError(message);
  return failure();
}

} // namespace

uint16_t roundF32ToBf16Rne(uint32_t bits) {
  constexpr uint32_t kMagnitudeMask = 0x7fffffffu;
  constexpr uint32_t kInfinity = 0x7f800000u;
  constexpr uint16_t kBf16QuietNaN = 0x0040u;
  if ((bits & kMagnitudeMask) > kInfinity) {
    return static_cast<uint16_t>(bits >> 16) | kBf16QuietNaN;
  }
  const uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(rounded >> 16);
}

LogicalResult
executeCpuExecutableBundle(ModuleOp module,
                           ArrayRef<CpuExternalBuffer> externalBuffers) {
  SmallVector<DeviceModuleOp> devices(module.getOps<DeviceModuleOp>());
  SmallVector<InvocationAbiOp> abis(module.getOps<InvocationAbiOp>());
  SmallVector<ExecutableBundleOp> bundles(module.getOps<ExecutableBundleOp>());
  if (devices.size() != 1 || abis.size() != 1 || bundles.size() != 1 ||
      failed(devices.front().verifyRegions()) ||
      failed(abis.front().verifyRegions()) ||
      failed(bundles.front().verify())) {
    return runtimeError(module, "invalid CPU executable bundle");
  }
  DeviceModuleOp device = devices.front();
  InvocationAbiOp abi = abis.front();
  SmallVector<InvocationSlotOp> descriptors(
      abi.getBody().front().getOps<InvocationSlotOp>());
  SmallVector<DeviceEntryOp> entries(
      device.getBody().front().getOps<DeviceEntryOp>());

  llvm::DenseMap<int64_t, MutableArrayRef<uint8_t>> supplied;
  for (const CpuExternalBuffer &buffer : externalBuffers) {
    if (buffer.ordinal < 0 || buffer.ordinal >= descriptors.size() ||
        !supplied.try_emplace(buffer.ordinal, buffer.bytes).second) {
      return runtimeError(module, "external buffer binding is invalid");
    }
  }
  std::vector<std::vector<uint8_t>> owned(descriptors.size());
  SmallVector<RuntimeSlot> slots;
  slots.reserve(descriptors.size());
  SmallVector<std::pair<uintptr_t, uintptr_t>> externalRanges;
  for (InvocationSlotOp descriptor : descriptors) {
    MutableArrayRef<uint8_t> bytes;
    if (descriptor.getStorage() == MaterializationStorage::External) {
      auto found = supplied.find(descriptor.getOrdinal());
      if (found == supplied.end()) {
        return runtimeError(module, "missing external buffer binding");
      }
      bytes = found->second;
      uintptr_t begin = reinterpret_cast<uintptr_t>(bytes.data());
      if (bytes.size() < static_cast<size_t>(descriptor.getRequiredBytes()) ||
          begin % descriptor.getAlignment() != 0) {
        return runtimeError(module,
                            "external buffer span or alignment mismatch");
      }
      externalRanges.push_back({begin, begin + descriptor.getRequiredBytes()});
    } else {
      const size_t alignment = descriptor.getAlignment();
      owned[descriptor.getOrdinal()].resize(descriptor.getRequiredBytes() +
                                            alignment - 1);
      uintptr_t begin =
          reinterpret_cast<uintptr_t>(owned[descriptor.getOrdinal()].data());
      begin = (begin + alignment - 1) / alignment * alignment;
      bytes = MutableArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(begin),
                                       descriptor.getRequiredBytes());
    }
    if (reinterpret_cast<uintptr_t>(bytes.data()) % descriptor.getAlignment() !=
        0) {
      return runtimeError(module, "buffer alignment mismatch");
    }
    slots.push_back(RuntimeSlot{descriptor, bytes});
  }
  if (supplied.size() != externalRanges.size()) {
    return runtimeError(module, "unexpected external buffer binding");
  }
  for (size_t lhs = 0; lhs < externalRanges.size(); ++lhs) {
    for (size_t rhs = lhs + 1; rhs < externalRanges.size(); ++rhs) {
      if (externalRanges[lhs].first < externalRanges[rhs].second &&
          externalRanges[rhs].first < externalRanges[lhs].second) {
        return runtimeError(module, "external buffers violate no-alias ABI");
      }
    }
  }

  SmallVector<bool> completed(entries.size(), false);
  for (DeviceEntryOp entry : entries) {
    for (int64_t dependency : entry.getDependencies()) {
      if (!completed[dependency]) {
        return runtimeError(module, "entry dependency has not completed");
      }
    }
    for (int64_t input : entry.getInputBuffers()) {
      if (input < 0 || input >= slots.size() ||
          slots[input].descriptor.getAccess() == ExecutableAccess::Write) {
        return runtimeError(module, "entry input violates ABI access mode");
      }
    }
    for (int64_t output : entry.getOutputBuffers()) {
      if (output < 0 || output >= slots.size() ||
          slots[output].descriptor.getAccess() == ExecutableAccess::Read) {
        return runtimeError(module, "entry output violates ABI access mode");
      }
    }
    ArrayRef<int8_t> entryBytes =
        device.getCode().slice(entry.getCodeOffset(), entry.getCodeLength());
    FailureOr<ParsedTask> task = parseTask(entryBytes);
    if (failed(task)) {
      return runtimeError(module, "CPU bytecode entry is invalid");
    }
    LogicalResult executed = task->kind == TaskKind::Map
                                 ? executeMap(*task, entry, slots)
                                 : executeFold(*task, entry, slots);
    if (failed(executed)) {
      return runtimeError(module, "CPU bytecode execution contract failed");
    }
    completed[entry.getOrdinal()] = true;
  }
  return success();
}

} // namespace mlir::shuttle
