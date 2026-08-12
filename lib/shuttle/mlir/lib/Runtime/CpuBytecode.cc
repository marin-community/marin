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
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "shuttle/IR/ShuttleOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SHA256.h"

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

enum class TaskKind : uint8_t { Map = 0, Fold = 1 };
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
  if (task.kind == TaskKind::Fold) {
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
  if (task.domain.size() != 2 || !task.reductionAxis ||
      *task.reductionAxis >= task.domain.size() ||
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
  const int64_t reductionAxis = *task.reductionAxis;
  const int64_t outputAxis = 1 - reductionAxis;
  if (!inputType.getElementType().isF32() ||
      inputType.getShape() != ArrayRef<int64_t>(task.domain) ||
      !initializerType.getElementType().isF32() ||
      initializerType.getRank() != 0 || !outputType.getElementType().isF32() ||
      outputType.getShape() != ArrayRef<int64_t>{task.domain[outputAxis]}) {
    return failure();
  }
  for (int64_t outputIndex = 0; outputIndex < task.domain[outputAxis];
       ++outputIndex) {
    FailureOr<Scalar> accumulator = loadScalar(initializer, {});
    if (failed(accumulator)) {
      return failure();
    }
    for (int64_t leafIndex = 0; leafIndex < task.domain[reductionAxis];
         ++leafIndex) {
      std::array<int64_t, 2> coordinates{};
      coordinates[outputAxis] = outputIndex;
      coordinates[reductionAxis] = leafIndex;
      FailureOr<Scalar> leaf = loadScalar(input, coordinates);
      if (failed(leaf)) {
        return failure();
      }
      std::array<Scalar, 2> arguments{*leaf, *accumulator};
      accumulator = evaluate(task, arguments);
      if (failed(accumulator)) {
        return failure();
      }
    }
    if (failed(storeScalar(output, {outputIndex}, *accumulator))) {
      return failure();
    }
  }
  return success();
}

LogicalResult runtimeError(ModuleOp module, StringRef message) {
  module.emitError(message);
  return failure();
}

constexpr uint8_t kTransportMagic[] = {'S', 'H', 'U', 'T', 'C', 'P', 'U', 0};
constexpr uint32_t kTransportVersion = 1;
constexpr uint64_t kMaximumTransportBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaximumTransportRecords = 4096;

class BundleWriter {
public:
  void u8(uint8_t value) { bytes.push_back(value); }
  void u32(uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      u8(static_cast<uint8_t>(value >> shift));
    }
  }
  void u64(uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
      u8(static_cast<uint8_t>(value >> shift));
    }
  }
  void i64(int64_t value) { u64(static_cast<uint64_t>(value)); }
  void string(StringRef value) {
    u64(value.size());
    bytes.append(value.bytes_begin(), value.bytes_end());
  }
  template <typename T> void i64Array(T values) {
    u64(values.size());
    for (int64_t value : values) {
      i64(value);
    }
  }
  void byteArray(ArrayRef<int8_t> values) {
    u64(values.size());
    bytes.append(reinterpret_cast<const uint8_t *>(values.data()),
                 reinterpret_cast<const uint8_t *>(values.data()) +
                     values.size());
  }
  void tensorType(Type value) {
    auto type = cast<RankedTensorType>(value);
    u8(type.getElementType().isBF16() ? 0 : 1);
    i64Array(type.getShape());
  }
  SmallVector<uint8_t> take() { return std::move(bytes); }

private:
  SmallVector<uint8_t> bytes;
};

class BundleReader {
public:
  explicit BundleReader(ArrayRef<uint8_t> bytes) : bytes(bytes) {}

  std::optional<uint8_t> u8() {
    if (position == bytes.size()) {
      return std::nullopt;
    }
    return bytes[position++];
  }
  std::optional<uint32_t> u32() {
    uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
      std::optional<uint8_t> next = u8();
      if (!next) {
        return std::nullopt;
      }
      value |= static_cast<uint32_t>(*next) << shift;
    }
    return value;
  }
  std::optional<uint64_t> u64() {
    uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
      std::optional<uint8_t> next = u8();
      if (!next) {
        return std::nullopt;
      }
      value |= static_cast<uint64_t>(*next) << shift;
    }
    return value;
  }
  std::optional<int64_t> i64() {
    std::optional<uint64_t> value = u64();
    if (!value) {
      return std::nullopt;
    }
    return static_cast<int64_t>(*value);
  }
  std::optional<StringRef> string() {
    std::optional<uint64_t> size = u64();
    if (!size || !bounded(*size)) {
      return std::nullopt;
    }
    StringRef value(reinterpret_cast<const char *>(bytes.data() + position),
                    *size);
    position += *size;
    return value;
  }
  std::optional<ArrayRef<uint8_t>> byteArray() {
    std::optional<uint64_t> size = u64();
    if (!size || !bounded(*size)) {
      return std::nullopt;
    }
    ArrayRef<uint8_t> value = bytes.slice(position, *size);
    position += *size;
    return value;
  }
  FailureOr<SmallVector<int64_t>> i64Array() {
    std::optional<uint64_t> size = u64();
    if (!size || *size > kMaximumTransportRecords ||
        *size > (bytes.size() - position) / sizeof(int64_t)) {
      return failure();
    }
    SmallVector<int64_t> values;
    values.reserve(*size);
    for (uint64_t index = 0; index < *size; ++index) {
      std::optional<int64_t> value = i64();
      if (!value) {
        return failure();
      }
      values.push_back(*value);
    }
    return values;
  }
  FailureOr<RankedTensorType> tensorType(MLIRContext *context) {
    std::optional<uint8_t> element = u8();
    FailureOr<SmallVector<int64_t>> shape = i64Array();
    if (!element || *element > 1 || failed(shape) || shape->size() > 8 ||
        llvm::any_of(*shape, [](int64_t extent) { return extent <= 0; })) {
      return failure();
    }
    Type elementType = *element == 0 ? Type(BFloat16Type::get(context))
                                     : Type(Float32Type::get(context));
    return RankedTensorType::get(*shape, elementType);
  }
  bool done() const { return position == bytes.size(); }

private:
  bool bounded(uint64_t size) const {
    return size <= kMaximumTransportBytes && size <= bytes.size() - position;
  }
  ArrayRef<uint8_t> bytes;
  size_t position = 0;
};

template <typename Enum>
FailureOr<Enum> transportEnum(std::optional<uint8_t> value, uint8_t maximum) {
  if (!value || *value > maximum) {
    return failure();
  }
  return static_cast<Enum>(*value);
}

void writeEntry(BundleWriter &writer, DeviceEntryOp entry) {
  writer.i64(entry.getOrdinal());
  writer.i64(entry.getSourceTask());
  writer.i64(entry.getCodeOffset());
  writer.i64(entry.getCodeLength());
  writer.i64Array(entry.getInputBuffers());
  writer.i64Array(entry.getOutputBuffers());
  writer.u64(entry.getInputAccesses().size());
  for (Attribute access : entry.getInputAccesses()) {
    writer.u8(
        static_cast<uint8_t>(cast<ExecutableAccessAttr>(access).getValue()));
  }
  writer.u64(entry.getOutputAccesses().size());
  for (Attribute access : entry.getOutputAccesses()) {
    writer.u8(
        static_cast<uint8_t>(cast<ExecutableAccessAttr>(access).getValue()));
  }
  writer.i64Array(entry.getDependencies());
  writer.u8(static_cast<uint8_t>(entry.getPredication()));
  writer.u8(entry.getReductionOrder().has_value());
  if (entry.getReductionOrder()) {
    writer.u8(static_cast<uint8_t>(*entry.getReductionOrder()));
  }
  writer.string(entry.getCodeDigest());
}

FailureOr<ArrayAttr> readAccessArray(BundleReader &reader,
                                     MLIRContext *context) {
  std::optional<uint64_t> size = reader.u64();
  if (!size || *size > kMaximumTransportRecords) {
    return failure();
  }
  SmallVector<Attribute> accesses;
  accesses.reserve(*size);
  for (uint64_t index = 0; index < *size; ++index) {
    FailureOr<ExecutableAccess> access =
        transportEnum<ExecutableAccess>(reader.u8(), 2);
    if (failed(access)) {
      return failure();
    }
    accesses.push_back(ExecutableAccessAttr::get(context, *access));
  }
  return ArrayAttr::get(context, accesses);
}

struct DecodedModule {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
};

FailureOr<DecodedModule> decodeCpuExecutableBundle(ArrayRef<uint8_t> bytes) {
  if (bytes.size() > kMaximumTransportBytes) {
    return failure();
  }
  BundleReader reader(bytes);
  for (uint8_t expected : kTransportMagic) {
    if (reader.u8() != expected) {
      return failure();
    }
  }
  if (reader.u32() != kTransportVersion) {
    return failure();
  }
  auto context = std::make_unique<MLIRContext>();
  context->getOrLoadDialect<ShuttleDialect>();
  OpBuilder builder(context.get());
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module->getBody());

  std::optional<int64_t> deviceSchema = reader.i64();
  FailureOr<ExecutableCodeFormat> codeFormat =
      transportEnum<ExecutableCodeFormat>(reader.u8(), 0);
  FailureOr<NumericalPolicy> policy =
      transportEnum<NumericalPolicy>(reader.u8(), 1);
  std::optional<StringRef> scheduleFingerprint = reader.string();
  std::optional<ArrayRef<uint8_t>> codeBytes = reader.byteArray();
  std::optional<StringRef> codeDigest = reader.string();
  std::optional<StringRef> deviceFingerprint = reader.string();
  std::optional<uint64_t> entryCount = reader.u64();
  if (!deviceSchema || failed(codeFormat) || failed(policy) ||
      !scheduleFingerprint || !codeBytes || !codeDigest || !deviceFingerprint ||
      !entryCount || *entryCount > kMaximumTransportRecords) {
    return failure();
  }
  OperationState deviceState(builder.getUnknownLoc(),
                             DeviceModuleOp::getOperationName());
  deviceState.addAttribute("schema_version",
                           builder.getI64IntegerAttr(*deviceSchema));
  deviceState.addAttribute(
      "code_format", ExecutableCodeFormatAttr::get(context.get(), *codeFormat));
  deviceState.addAttribute("policy",
                           NumericalPolicyAttr::get(context.get(), *policy));
  deviceState.addAttribute("source_schedule_fingerprint",
                           builder.getStringAttr(*scheduleFingerprint));
  SmallVector<int8_t> signedCode;
  signedCode.reserve(codeBytes->size());
  for (uint8_t byte : *codeBytes) {
    signedCode.push_back(static_cast<int8_t>(byte));
  }
  deviceState.addAttribute("code",
                           DenseI8ArrayAttr::get(context.get(), signedCode));
  deviceState.addAttribute("code_digest", builder.getStringAttr(*codeDigest));
  deviceState.addAttribute("fingerprint",
                           builder.getStringAttr(*deviceFingerprint));
  deviceState.addRegion();
  auto device = cast<DeviceModuleOp>(builder.create(deviceState));
  Block *deviceBody = new Block();
  device.getBody().push_back(deviceBody);
  builder.setInsertionPointToEnd(deviceBody);
  for (uint64_t ordinal = 0; ordinal < *entryCount; ++ordinal) {
    std::optional<int64_t> entryOrdinal = reader.i64();
    std::optional<int64_t> sourceTask = reader.i64();
    std::optional<int64_t> codeOffset = reader.i64();
    std::optional<int64_t> codeLength = reader.i64();
    FailureOr<SmallVector<int64_t>> inputs = reader.i64Array();
    FailureOr<SmallVector<int64_t>> outputs = reader.i64Array();
    FailureOr<ArrayAttr> inputAccesses = readAccessArray(reader, context.get());
    FailureOr<ArrayAttr> outputAccesses =
        readAccessArray(reader, context.get());
    FailureOr<SmallVector<int64_t>> dependencies = reader.i64Array();
    FailureOr<ExecutablePredication> predication =
        transportEnum<ExecutablePredication>(reader.u8(), 1);
    std::optional<uint8_t> hasReductionOrder = reader.u8();
    std::optional<ScheduleReductionOrder> reductionOrder;
    if (hasReductionOrder && *hasReductionOrder == 1) {
      FailureOr<ScheduleReductionOrder> decoded =
          transportEnum<ScheduleReductionOrder>(reader.u8(), 1);
      if (failed(decoded)) {
        return failure();
      }
      reductionOrder = *decoded;
    } else if (!hasReductionOrder || *hasReductionOrder != 0) {
      return failure();
    }
    std::optional<StringRef> entryCodeDigest = reader.string();
    if (!entryOrdinal || !sourceTask || !codeOffset || !codeLength ||
        failed(inputs) || failed(outputs) || failed(inputAccesses) ||
        failed(outputAccesses) || failed(dependencies) || failed(predication) ||
        !entryCodeDigest) {
      return failure();
    }
    OperationState state(builder.getUnknownLoc(),
                         DeviceEntryOp::getOperationName());
    state.addAttribute("ordinal", builder.getI64IntegerAttr(*entryOrdinal));
    state.addAttribute("source_task", builder.getI64IntegerAttr(*sourceTask));
    state.addAttribute("code_offset", builder.getI64IntegerAttr(*codeOffset));
    state.addAttribute("code_length", builder.getI64IntegerAttr(*codeLength));
    state.addAttribute("input_buffers",
                       DenseI64ArrayAttr::get(context.get(), *inputs));
    state.addAttribute("output_buffers",
                       DenseI64ArrayAttr::get(context.get(), *outputs));
    state.addAttribute("input_accesses", *inputAccesses);
    state.addAttribute("output_accesses", *outputAccesses);
    state.addAttribute("dependencies",
                       DenseI64ArrayAttr::get(context.get(), *dependencies));
    state.addAttribute("predication", ExecutablePredicationAttr::get(
                                          context.get(), *predication));
    if (reductionOrder) {
      state.addAttribute(
          "reduction_order",
          ScheduleReductionOrderAttr::get(context.get(), *reductionOrder));
    }
    state.addAttribute("code_digest", builder.getStringAttr(*entryCodeDigest));
    builder.create(state);
  }
  builder.create<DeviceModuleYieldOp>(builder.getUnknownLoc());

  builder.setInsertionPointToEnd(module->getBody());
  std::optional<int64_t> abiSchema = reader.i64();
  std::optional<StringRef> planFingerprint = reader.string();
  std::optional<StringRef> abiScheduleFingerprint = reader.string();
  std::optional<StringRef> abiFingerprint = reader.string();
  std::optional<uint64_t> slotCount = reader.u64();
  if (!abiSchema || !planFingerprint || !abiScheduleFingerprint ||
      !abiFingerprint || !slotCount || *slotCount > kMaximumTransportRecords) {
    return failure();
  }
  OperationState abiState(builder.getUnknownLoc(),
                          InvocationAbiOp::getOperationName());
  abiState.addAttribute("schema_version",
                        builder.getI64IntegerAttr(*abiSchema));
  abiState.addAttribute("source_plan_fingerprint",
                        builder.getStringAttr(*planFingerprint));
  abiState.addAttribute("source_schedule_fingerprint",
                        builder.getStringAttr(*abiScheduleFingerprint));
  abiState.addAttribute("fingerprint", builder.getStringAttr(*abiFingerprint));
  abiState.addRegion();
  auto abi = cast<InvocationAbiOp>(builder.create(abiState));
  Block *abiBody = new Block();
  abi.getBody().push_back(abiBody);
  builder.setInsertionPointToEnd(abiBody);
  for (uint64_t ordinal = 0; ordinal < *slotCount; ++ordinal) {
    std::optional<int64_t> slotOrdinal = reader.i64();
    std::optional<int64_t> sourceBuffer = reader.i64();
    FailureOr<RankedTensorType> tensorType = reader.tensorType(context.get());
    std::optional<int64_t> requiredBytes = reader.i64();
    FailureOr<SmallVector<int64_t>> strides = reader.i64Array();
    std::optional<int64_t> offset = reader.i64();
    std::optional<int64_t> alignment = reader.i64();
    FailureOr<ExecutableAddressSpace> addressSpace =
        transportEnum<ExecutableAddressSpace>(reader.u8(), 1);
    FailureOr<ExecutableAccess> access =
        transportEnum<ExecutableAccess>(reader.u8(), 2);
    FailureOr<MaterializationStorage> storage =
        transportEnum<MaterializationStorage>(reader.u8(), 1);
    std::optional<int64_t> aliasGroup = reader.i64();
    std::optional<int64_t> reuseGroup = reader.i64();
    FailureOr<ExecutableBindingKind> binding =
        transportEnum<ExecutableBindingKind>(reader.u8(), 2);
    std::optional<uint8_t> hasBindingIndex = reader.u8();
    std::optional<int64_t> bindingIndex;
    if (hasBindingIndex && *hasBindingIndex == 1) {
      bindingIndex = reader.i64();
      if (!bindingIndex) {
        return failure();
      }
    } else if (!hasBindingIndex || *hasBindingIndex != 0) {
      return failure();
    }
    if (!slotOrdinal || !sourceBuffer || failed(tensorType) || !requiredBytes ||
        failed(strides) || !offset || !alignment || failed(addressSpace) ||
        failed(access) || failed(storage) || !aliasGroup || !reuseGroup ||
        failed(binding)) {
      return failure();
    }
    OperationState state(builder.getUnknownLoc(),
                         InvocationSlotOp::getOperationName());
    state.addAttribute("ordinal", builder.getI64IntegerAttr(*slotOrdinal));
    state.addAttribute("source_buffer",
                       builder.getI64IntegerAttr(*sourceBuffer));
    state.addAttribute("tensor_type", TypeAttr::get(*tensorType));
    state.addAttribute("required_bytes",
                       builder.getI64IntegerAttr(*requiredBytes));
    state.addAttribute("strides",
                       DenseI64ArrayAttr::get(context.get(), *strides));
    state.addAttribute("offset", builder.getI64IntegerAttr(*offset));
    state.addAttribute("alignment", builder.getI64IntegerAttr(*alignment));
    state.addAttribute("address_space", ExecutableAddressSpaceAttr::get(
                                            context.get(), *addressSpace));
    state.addAttribute("access",
                       ExecutableAccessAttr::get(context.get(), *access));
    state.addAttribute(
        "storage", MaterializationStorageAttr::get(context.get(), *storage));
    state.addAttribute("alias_group", builder.getI64IntegerAttr(*aliasGroup));
    state.addAttribute("reuse_group", builder.getI64IntegerAttr(*reuseGroup));
    state.addAttribute("binding",
                       ExecutableBindingKindAttr::get(context.get(), *binding));
    if (bindingIndex) {
      state.addAttribute("binding_index",
                         builder.getI64IntegerAttr(*bindingIndex));
    }
    builder.create(state);
  }
  builder.create<InvocationAbiYieldOp>(builder.getUnknownLoc());

  builder.setInsertionPointToEnd(module->getBody());
  std::optional<int64_t> bundleSchema = reader.i64();
  std::optional<StringRef> bundleScheduleFingerprint = reader.string();
  std::optional<StringRef> bundleDeviceFingerprint = reader.string();
  std::optional<StringRef> bundleAbiFingerprint = reader.string();
  FailureOr<ExecutableCompletion> completion =
      transportEnum<ExecutableCompletion>(reader.u8(), 0);
  std::optional<StringRef> bundleFingerprint = reader.string();
  if (!bundleSchema || !bundleScheduleFingerprint || !bundleDeviceFingerprint ||
      !bundleAbiFingerprint || failed(completion) || !bundleFingerprint ||
      !reader.done()) {
    return failure();
  }
  OperationState bundleState(builder.getUnknownLoc(),
                             ExecutableBundleOp::getOperationName());
  bundleState.addAttribute("schema_version",
                           builder.getI64IntegerAttr(*bundleSchema));
  bundleState.addAttribute("source_schedule_fingerprint",
                           builder.getStringAttr(*bundleScheduleFingerprint));
  bundleState.addAttribute("device_module_fingerprint",
                           builder.getStringAttr(*bundleDeviceFingerprint));
  bundleState.addAttribute("invocation_abi_fingerprint",
                           builder.getStringAttr(*bundleAbiFingerprint));
  bundleState.addAttribute(
      "completion", ExecutableCompletionAttr::get(context.get(), *completion));
  bundleState.addAttribute("fingerprint",
                           builder.getStringAttr(*bundleFingerprint));
  builder.create(bundleState);
  if (failed(verify(module.get()))) {
    return failure();
  }
  return DecodedModule{std::move(context), std::move(module)};
}

LogicalResult
executeCpuExecutableBundleImpl(ModuleOp module,
                               ArrayRef<CpuExternalBuffer> externalBuffers,
                               ArrayRef<ParsedTask> preParsedTasks);

} // namespace

class CpuExecutable::Impl {
public:
  Impl(DecodedModule decoded, SmallVector<ParsedTask, 0> tasks)
      : decoded(std::move(decoded)), tasks(std::move(tasks)) {
    InvocationAbiOp abi =
        *this->decoded.module->getOps<InvocationAbiOp>().begin();
    for (InvocationSlotOp slot :
         abi.getBody().front().getOps<InvocationSlotOp>()) {
      if (slot.getBinding() == ExecutableBindingKind::None) {
        continue;
      }
      bindings.push_back(CpuExternalBinding{
          slot.getBinding(), static_cast<int64_t>(*slot.getBindingIndex()),
          static_cast<int64_t>(slot.getOrdinal()), slot.getTensorType(),
          static_cast<int64_t>(slot.getRequiredBytes()),
          static_cast<int64_t>(slot.getAlignment())});
    }
  }

  DecodedModule decoded;
  SmallVector<ParsedTask, 0> tasks;
  SmallVector<CpuExternalBinding> bindings;
};

CpuExecutable::CpuExecutable(std::shared_ptr<const Impl> implementation)
    : implementation(std::move(implementation)) {}

absl::StatusOr<std::shared_ptr<const CpuExecutable>>
CpuExecutable::Load(ArrayRef<uint8_t> bytes) {
  FailureOr<DecodedModule> decoded = decodeCpuExecutableBundle(bytes);
  if (failed(decoded)) {
    return absl::InvalidArgumentError(
        "invalid canonical Shuttle CPU executable transport");
  }
  FailureOr<SmallVector<uint8_t>> canonical =
      serializeCpuExecutableBundle(*decoded->module);
  if (failed(canonical) || ArrayRef<uint8_t>(*canonical) != bytes) {
    return absl::InvalidArgumentError(
        "non-canonical Shuttle CPU executable transport");
  }
  DeviceModuleOp device = *decoded->module->getOps<DeviceModuleOp>().begin();
  SmallVector<ParsedTask, 0> tasks;
  for (DeviceEntryOp entry : device.getBody().front().getOps<DeviceEntryOp>()) {
    FailureOr<ParsedTask> task = parseTask(
        device.getCode().slice(entry.getCodeOffset(), entry.getCodeLength()));
    if (failed(task)) {
      return absl::InvalidArgumentError(
          "invalid Shuttle CPU bytecode body in executable transport");
    }
    tasks.push_back(std::move(*task));
  }
  std::shared_ptr<const Impl> implementation =
      std::make_shared<Impl>(std::move(*decoded), std::move(tasks));
  return std::shared_ptr<const CpuExecutable>(
      new CpuExecutable(std::move(implementation)));
}

ArrayRef<CpuExternalBinding> CpuExecutable::externalBindings() const {
  return implementation->bindings;
}

absl::Status
CpuExecutable::Execute(ArrayRef<CpuExternalBuffer> externalBuffers) const {
  if (failed(executeCpuExecutableBundleImpl(*implementation->decoded.module,
                                            externalBuffers,
                                            implementation->tasks))) {
    return absl::InvalidArgumentError("Shuttle CPU executable failed");
  }
  return absl::OkStatus();
}

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

std::string cpuExecutableBundleDigest(ArrayRef<uint8_t> bytes) {
  llvm::SHA256 digest;
  digest.update(bytes);
  return llvm::toHex(digest.final(), true);
}

FailureOr<SmallVector<uint8_t>> serializeCpuExecutableBundle(ModuleOp module) {
  for (Operation &operation : module.getBody()->without_terminator()) {
    if (!isa<DeviceModuleOp, InvocationAbiOp, ExecutableBundleOp>(operation)) {
      return failure();
    }
  }
  SmallVector<DeviceModuleOp> devices(module.getOps<DeviceModuleOp>());
  SmallVector<InvocationAbiOp> abis(module.getOps<InvocationAbiOp>());
  SmallVector<ExecutableBundleOp> bundles(module.getOps<ExecutableBundleOp>());
  if (devices.size() != 1 || abis.size() != 1 || bundles.size() != 1 ||
      failed(devices.front().verifyRegions()) ||
      failed(abis.front().verifyRegions()) ||
      failed(bundles.front().verify())) {
    return failure();
  }
  DeviceModuleOp device = devices.front();
  InvocationAbiOp abi = abis.front();
  ExecutableBundleOp bundle = bundles.front();
  BundleWriter writer;
  for (uint8_t byte : kTransportMagic) {
    writer.u8(byte);
  }
  writer.u32(kTransportVersion);
  writer.i64(device.getSchemaVersion());
  writer.u8(static_cast<uint8_t>(device.getCodeFormat()));
  writer.u8(static_cast<uint8_t>(device.getPolicy()));
  writer.string(device.getSourceScheduleFingerprint());
  writer.byteArray(device.getCode());
  writer.string(device.getCodeDigest());
  writer.string(device.getFingerprint());
  SmallVector<DeviceEntryOp> entries(
      device.getBody().front().getOps<DeviceEntryOp>());
  writer.u64(entries.size());
  for (DeviceEntryOp entry : entries) {
    writeEntry(writer, entry);
  }
  writer.i64(abi.getSchemaVersion());
  writer.string(abi.getSourcePlanFingerprint());
  writer.string(abi.getSourceScheduleFingerprint());
  writer.string(abi.getFingerprint());
  SmallVector<InvocationSlotOp> slots(
      abi.getBody().front().getOps<InvocationSlotOp>());
  writer.u64(slots.size());
  for (InvocationSlotOp slot : slots) {
    writer.i64(slot.getOrdinal());
    writer.i64(slot.getSourceBuffer());
    writer.tensorType(slot.getTensorType());
    writer.i64(slot.getRequiredBytes());
    writer.i64Array(slot.getStrides());
    writer.i64(slot.getOffset());
    writer.i64(slot.getAlignment());
    writer.u8(static_cast<uint8_t>(slot.getAddressSpace()));
    writer.u8(static_cast<uint8_t>(slot.getAccess()));
    writer.u8(static_cast<uint8_t>(slot.getStorage()));
    writer.i64(slot.getAliasGroup());
    writer.i64(slot.getReuseGroup());
    writer.u8(static_cast<uint8_t>(slot.getBinding()));
    writer.u8(slot.getBindingIndex().has_value());
    if (slot.getBindingIndex()) {
      writer.i64(*slot.getBindingIndex());
    }
  }
  writer.i64(bundle.getSchemaVersion());
  writer.string(bundle.getSourceScheduleFingerprint());
  writer.string(bundle.getDeviceModuleFingerprint());
  writer.string(bundle.getInvocationAbiFingerprint());
  writer.u8(static_cast<uint8_t>(bundle.getCompletion()));
  writer.string(bundle.getFingerprint());
  SmallVector<uint8_t> bytes = writer.take();
  if (bytes.size() > kMaximumTransportBytes) {
    return failure();
  }
  return bytes;
}

LogicalResult
executeCpuExecutableBundle(ModuleOp module,
                           ArrayRef<CpuExternalBuffer> externalBuffers) {
  return executeCpuExecutableBundleImpl(module, externalBuffers, {});
}

namespace {

LogicalResult
executeCpuExecutableBundleImpl(ModuleOp module,
                               ArrayRef<CpuExternalBuffer> externalBuffers,
                               ArrayRef<ParsedTask> preParsedTasks) {
  for (Operation &operation : module.getBody()->without_terminator()) {
    if (!isa<DeviceModuleOp, InvocationAbiOp, ExecutableBundleOp>(operation)) {
      return runtimeError(
          module, "CPU executable module contains an unstripped sidecar");
    }
  }
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
  if (!preParsedTasks.empty() && preParsedTasks.size() != entries.size()) {
    return runtimeError(module, "pre-parsed bytecode does not match entries");
  }
  for (auto [entryOrdinal, entry] : llvm::enumerate(entries)) {
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
    FailureOr<ParsedTask> decodedTask =
        preParsedTasks.empty()
            ? parseTask(entryBytes)
            : FailureOr<ParsedTask>(preParsedTasks[entryOrdinal]);
    if (failed(decodedTask)) {
      return runtimeError(module, "CPU bytecode entry is invalid");
    }
    LogicalResult executed = decodedTask->kind == TaskKind::Map
                                 ? executeMap(*decodedTask, entry, slots)
                                 : executeFold(*decodedTask, entry, slots);
    if (failed(executed)) {
      return runtimeError(module, "CPU bytecode execution contract failed");
    }
    completed[entry.getOrdinal()] = true;
  }
  return success();
}

} // namespace

} // namespace mlir::shuttle
