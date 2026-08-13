// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "shuttle/Runtime/GpuExecutable.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "shuttle/IR/ShuttleOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

namespace mlir::shuttle {
namespace {

constexpr std::array<uint8_t, 8> kMagic{'S', 'H', 'U', 'T', 'G', 'P', 'U', 0};

std::string digest(ArrayRef<uint8_t> bytes) {
  llvm::SHA256 sha;
  sha.update(bytes);
  return llvm::toHex(sha.final(), true);
}

std::string digest(ArrayRef<uint8_t> prefix, ArrayRef<uint8_t> suffix) {
  llvm::SHA256 sha;
  sha.update(prefix);
  sha.update(suffix);
  return llvm::toHex(sha.final(), true);
}

bool isDigest(StringRef value) {
  return value.size() == 64 && llvm::all_of(value, [](char character) {
           return llvm::isDigit(character) ||
                  (character >= 'a' && character <= 'f');
         });
}

class Writer {
public:
  void u8(uint8_t value) { bytes.push_back(value); }
  void u16(uint16_t value) { integer(value); }
  void u32(uint32_t value) { integer(value); }
  void u64(uint64_t value) { integer(value); }
  void i64(int64_t value) { u64(static_cast<uint64_t>(value)); }
  void text(StringRef value) {
    u64(value.size());
    bytes.append(value.bytes_begin(), value.bytes_end());
  }
  void blob(ArrayRef<int8_t> value) {
    u64(value.size());
    for (int8_t byte : value)
      bytes.push_back(static_cast<uint8_t>(byte));
  }
  void integers(ArrayRef<int64_t> values) {
    u64(values.size());
    for (int64_t value : values)
      i64(value);
  }
  void append(ArrayRef<uint8_t> values) {
    bytes.append(values.begin(), values.end());
  }
  ArrayRef<uint8_t> view() const { return bytes; }
  SmallVector<uint8_t> take() { return std::move(bytes); }

private:
  template <typename T> void integer(T value) {
    for (unsigned byte = 0; byte < sizeof(T); ++byte)
      bytes.push_back(static_cast<uint8_t>(value >> (byte * 8)));
  }
  SmallVector<uint8_t> bytes;
};

class Reader {
public:
  explicit Reader(ArrayRef<uint8_t> bytes) : bytes(bytes) {}
  std::optional<uint8_t> u8() { return integer<uint8_t>(); }
  std::optional<uint16_t> u16() { return integer<uint16_t>(); }
  std::optional<uint32_t> u32() { return integer<uint32_t>(); }
  std::optional<uint64_t> u64() { return integer<uint64_t>(); }
  std::optional<int64_t> i64() {
    auto value = u64();
    return value ? std::optional<int64_t>(static_cast<int64_t>(*value))
                 : std::nullopt;
  }
  std::optional<StringRef> text() {
    auto size = u64();
    if (!size || !bounded(*size))
      return std::nullopt;
    StringRef value(reinterpret_cast<const char *>(bytes.data() + offset),
                    *size);
    offset += *size;
    return value;
  }
  std::optional<ArrayRef<uint8_t>> blob() {
    auto size = u64();
    if (!size || !bounded(*size))
      return std::nullopt;
    ArrayRef<uint8_t> value = bytes.slice(offset, *size);
    offset += *size;
    return value;
  }
  std::optional<SmallVector<int64_t>> integers() {
    auto size = u64();
    if (!size || *size > kMaximumGpuExecutableRecords ||
        *size > (bytes.size() - offset) / sizeof(int64_t))
      return std::nullopt;
    SmallVector<int64_t> values;
    values.reserve(*size);
    for (uint64_t index = 0; index < *size; ++index) {
      auto value = i64();
      if (!value)
        return std::nullopt;
      values.push_back(*value);
    }
    return values;
  }
  size_t position() const { return offset; }
  bool done() const { return offset == bytes.size(); }

private:
  template <typename T> std::optional<T> integer() {
    if (!bounded(sizeof(T)))
      return std::nullopt;
    T value = 0;
    for (unsigned byte = 0; byte < sizeof(T); ++byte)
      value |= static_cast<T>(bytes[offset++]) << (byte * 8);
    return value;
  }
  bool bounded(uint64_t size) const { return size <= bytes.size() - offset; }
  ArrayRef<uint8_t> bytes;
  size_t offset = 0;
};

void writeAccesses(Writer &writer, ArrayAttr accesses) {
  writer.u64(accesses.size());
  for (Attribute attribute : accesses)
    writer.u8(
        static_cast<uint8_t>(cast<ExecutableAccessAttr>(attribute).getValue()));
}

void writeTensor(Writer &writer, Type type) {
  auto tensor = cast<RankedTensorType>(type);
  writer.u8(tensor.getElementType().isBF16() ? 0 : 1);
  writer.u64(tensor.getRank());
  for (int64_t extent : tensor.getShape())
    writer.i64(extent);
}

void writeEntry(Writer &writer, DeviceEntryOp entry) {
  writer.i64(entry.getOrdinal());
  writer.i64(entry.getSourceTask());
  writer.i64(entry.getCodeOffset());
  writer.i64(entry.getCodeLength());
  writer.integers(entry.getInputBuffers());
  writer.integers(entry.getOutputBuffers());
  writeAccesses(writer, entry.getInputAccesses());
  writeAccesses(writer, entry.getOutputAccesses());
  writer.integers(entry.getDependencies());
  writer.u8(static_cast<uint8_t>(entry.getPredication()));
  writer.u8(entry.getReductionOrder().has_value());
  if (entry.getReductionOrder())
    writer.u8(static_cast<uint8_t>(*entry.getReductionOrder()));
  writer.text(entry.getCodeDigest());
  for (int64_t extent : *entry.getGrid())
    writer.u32(extent);
  for (int64_t extent : *entry.getBlock())
    writer.u16(extent);
  writer.u32(*entry.getDynamicSharedBytes());
  writer.u16(*entry.getKernelArity());
}

void writeSlot(Writer &writer, InvocationSlotOp slot) {
  writer.i64(slot.getOrdinal());
  writer.i64(slot.getSourceBuffer());
  writeTensor(writer, slot.getTensorType());
  writer.i64(slot.getRequiredBytes());
  writer.integers(slot.getStrides());
  writer.i64(slot.getOffset());
  writer.i64(slot.getAlignment());
  writer.u8(static_cast<uint8_t>(slot.getAddressSpace()));
  writer.u8(static_cast<uint8_t>(slot.getAccess()));
  writer.u8(static_cast<uint8_t>(slot.getStorage()));
  writer.i64(slot.getAliasGroup());
  writer.i64(slot.getReuseGroup());
  writer.u8(static_cast<uint8_t>(slot.getBinding()));
  writer.u8(slot.getBindingIndex().has_value());
  if (slot.getBindingIndex())
    writer.i64(*slot.getBindingIndex());
}

absl::Status invalidTransport() {
  return absl::InvalidArgumentError("invalid Shuttle GPU executable transport");
}

struct Parsed {
  std::unique_ptr<MLIRContext> context;
  SmallVector<uint8_t> code;
  std::vector<GpuLaunch> launches;
  std::vector<GpuSlot> slots;
  std::vector<GpuExternalBinding> bindings;
};

std::optional<uint64_t> checkedProduct(ArrayRef<uint64_t> values,
                                       uint64_t maximum) {
  uint64_t product = 1;
  for (uint64_t value : values) {
    if (!value || value > maximum / product)
      return std::nullopt;
    product *= value;
  }
  return product;
}

absl::StatusOr<Parsed> parse(ArrayRef<uint8_t> bytes) {
  if (bytes.size() > kMaximumGpuTransportBytes)
    return invalidTransport();
  Reader reader(bytes);
  for (uint8_t expected : kMagic)
    if (reader.u8() != expected)
      return invalidTransport();
  if (reader.u32() != 2)
    return invalidTransport();
  constexpr size_t deviceStart = 12;
  auto deviceSchema = reader.i64();
  auto format = reader.u8();
  auto policy = reader.u8();
  auto schedule = reader.text();
  auto code = reader.blob();
  auto codeDigest = reader.text();
  size_t deviceRootOffset = reader.position();
  auto deviceRoot = reader.text();
  size_t deviceSuffixStart = reader.position();
  auto entryCount = reader.u64();
  if (deviceSchema != 3 || format != 2 || policy != 0 || !schedule ||
      !isDigest(*schedule) || !code || code->empty() ||
      code->size() > kMaximumGpuCodeBytes || !codeDigest ||
      *codeDigest != digest(*code) || !deviceRoot || !isDigest(*deviceRoot) ||
      entryCount != 19)
    return invalidTransport();

  Parsed parsed;
  parsed.context = std::make_unique<MLIRContext>();
  parsed.code.assign(code->begin(), code->end());
  int64_t expectedOffset = 0;
  uint64_t aggregatePositions = 0;
  DenseMap<int64_t, int64_t> producers;
  for (int64_t ordinal = 0; ordinal < 19; ++ordinal) {
    auto entryOrdinal = reader.i64();
    auto sourceTask = reader.i64();
    auto codeOffset = reader.i64();
    auto codeLength = reader.i64();
    auto inputs = reader.integers();
    auto outputs = reader.integers();
    auto inputCount = reader.u64();
    if (entryOrdinal != ordinal || sourceTask != ordinal ||
        codeOffset != expectedOffset || !codeLength || *codeLength <= 0 ||
        *codeLength > static_cast<int64_t>(kMaximumGpuEntryCodeBytes) ||
        !inputs || !outputs || !inputCount || *inputCount != inputs->size())
      return invalidTransport();
    for (uint64_t index = 0; index < *inputCount; ++index)
      if (reader.u8() != 0)
        return invalidTransport();
    auto outputCount = reader.u64();
    if (!outputCount || *outputCount != outputs->size())
      return invalidTransport();
    for (uint64_t index = 0; index < *outputCount; ++index)
      if (reader.u8() != 1)
        return invalidTransport();
    auto dependencies = reader.integers();
    auto predication = reader.u8();
    auto hasReduction = reader.u8();
    std::optional<uint8_t> reduction;
    if (hasReduction == 1)
      reduction = reader.u8();
    else if (hasReduction != 0)
      return invalidTransport();
    auto entryDigest = reader.text();
    if (!dependencies || !predication || *predication > 1 ||
        (reduction && *reduction != 0) || !entryDigest ||
        expectedOffset > static_cast<int64_t>(code->size()) - *codeLength)
      return invalidTransport();
    ArrayRef<uint8_t> slice = code->slice(expectedOffset, *codeLength);
    if (*entryDigest != digest(slice))
      return invalidTransport();
    std::array<uint64_t, 3> grid;
    std::array<uint64_t, 3> block;
    for (uint64_t &extent : grid) {
      auto value = reader.u32();
      if (!value)
        return invalidTransport();
      extent = *value;
    }
    for (uint64_t &extent : block) {
      auto value = reader.u16();
      if (!value)
        return invalidTransport();
      extent = *value;
    }
    auto shared = reader.u32();
    auto arity = reader.u16();
    SmallVector<uint64_t> geometry(grid.begin(), grid.end());
    geometry.append(block.begin(), block.end());
    auto blockThreads = checkedProduct(block, 1024);
    auto positions =
        checkedProduct(geometry, kMaximumGpuAggregateTaskPositions);
    if (!blockThreads || !positions ||
        *positions > kMaximumGpuAggregateTaskPositions - aggregatePositions ||
        !shared || *shared > kMaximumGpuDynamicSharedMemoryBytes || !arity ||
        *arity != inputs->size() + outputs->size())
      return invalidTransport();
    aggregatePositions += *positions;
    if (llvm::any_of(*inputs,
                     [](int64_t slot) { return slot < 0 || slot >= 21; }) ||
        llvm::any_of(*outputs,
                     [](int64_t slot) { return slot < 0 || slot >= 21; }))
      return invalidTransport();
    SmallVector<int64_t> expectedDependencies;
    for (int64_t input : *inputs) {
      auto producer = producers.find(input);
      if (producer != producers.end() &&
          !llvm::is_contained(expectedDependencies, producer->second))
        expectedDependencies.push_back(producer->second);
    }
    llvm::sort(expectedDependencies);
    llvm::SmallDenseSet<int64_t> uniqueDependencies;
    for (int64_t dependency : *dependencies)
      if (dependency < 0 || dependency >= ordinal ||
          !uniqueDependencies.insert(dependency).second)
        return invalidTransport();
    if (*dependencies != expectedDependencies)
      return invalidTransport();
    for (int64_t output : *outputs)
      if (!producers.try_emplace(output, ordinal).second)
        return invalidTransport();
    StringRef ptx(reinterpret_cast<const char *>(slice.data()), slice.size());
    if (!ptx.ends_with("\n") ||
        ptx.find(".visible .entry shuttle_entry(") == StringRef::npos)
      return invalidTransport();
    parsed.launches.push_back(
        {ordinal, expectedOffset, *codeLength, entryDigest->str(), grid, block,
         *shared, std::vector<int64_t>(inputs->begin(), inputs->end()),
         std::vector<int64_t>(outputs->begin(), outputs->end()),
         std::vector<int64_t>(dependencies->begin(), dependencies->end())});
    expectedOffset += *codeLength;
  }
  size_t invocationStart = reader.position();
  if (expectedOffset != code->size() || producers.size() != 19 ||
      *deviceRoot !=
          digest(bytes.slice(deviceStart, deviceRootOffset - deviceStart),
                 bytes.slice(deviceSuffixStart,
                             invocationStart - deviceSuffixStart)))
    return invalidTransport();

  auto abiSchema = reader.i64();
  auto plan = reader.text();
  auto abiSchedule = reader.text();
  size_t abiRootOffset = reader.position();
  auto abiRoot = reader.text();
  size_t abiSuffixStart = reader.position();
  auto slotCount = reader.u64();
  if (abiSchema != 3 || !plan || !isDigest(*plan) || !abiSchedule ||
      *abiSchedule != *schedule || !abiRoot || !isDigest(*abiRoot) ||
      slotCount != 21)
    return invalidTransport();
  uint64_t temporaryBytes = 0;
  for (int64_t ordinal = 0; ordinal < 21; ++ordinal) {
    auto slotOrdinal = reader.i64();
    auto sourceBuffer = reader.i64();
    auto element = reader.u8();
    auto rank = reader.u64();
    if (slotOrdinal != ordinal || sourceBuffer != ordinal || !element ||
        *element > 1 || !rank || *rank > 8)
      return invalidTransport();
    SmallVector<int64_t> shape;
    int64_t required = *element == 0 ? 2 : 4;
    for (uint64_t axis = 0; axis < *rank; ++axis) {
      auto extent = reader.i64();
      if (!extent || *extent <= 0 ||
          required > std::numeric_limits<int64_t>::max() / *extent)
        return invalidTransport();
      shape.push_back(*extent);
      required *= *extent;
    }
    auto encodedRequired = reader.i64();
    auto strides = reader.integers();
    auto offset = reader.i64();
    auto alignment = reader.i64();
    auto addressSpace = reader.u8();
    auto access = reader.u8();
    auto storage = reader.u8();
    auto alias = reader.i64();
    auto reuse = reader.i64();
    auto binding = reader.u8();
    auto hasIndex = reader.u8();
    std::optional<int64_t> index;
    if (hasIndex == 1)
      index = reader.i64();
    else if (hasIndex != 0)
      return invalidTransport();
    if (encodedRequired != required || required > kMaximumGpuSlotBytes ||
        !strides || strides->size() != *rank || offset != 0 ||
        alignment != (*element == 0 ? 2 : 4) || addressSpace != 1 || !access ||
        *access > 2 || !storage || *storage > 1 || alias != ordinal ||
        reuse != ordinal || !binding || *binding > 2)
      return invalidTransport();
    SmallVector<int64_t> expectedStrides(*rank);
    int64_t stride = *element == 0 ? 2 : 4;
    for (int64_t axis = *rank; axis > 0; --axis) {
      expectedStrides[axis - 1] = stride;
      stride *= shape[axis - 1];
    }
    if (*strides != expectedStrides)
      return invalidTransport();
    auto tensor = RankedTensorType::get(
        shape, *element == 0 ? Type(BFloat16Type::get(parsed.context.get()))
                             : Type(Float32Type::get(parsed.context.get())));
    GpuSlot slot{ordinal,
                 tensor,
                 required,
                 std::move(*strides),
                 0,
                 *alignment,
                 ExecutableAddressSpace::Device,
                 static_cast<ExecutableAccess>(*access),
                 static_cast<MaterializationStorage>(*storage),
                 ordinal,
                 ordinal,
                 static_cast<ExecutableBindingKind>(*binding),
                 index.value_or(-1)};
    if (*storage == 1) {
      if (*binding != 0 || index || *access != 2 ||
          required > kMaximumGpuTemporaryBytes - temporaryBytes)
        return invalidTransport();
      temporaryBytes += required;
    } else {
      if (!index || *binding == 0 || (*binding == 1 && *access != 0) ||
          (*binding == 2 && *access != 1))
        return invalidTransport();
      parsed.bindings.push_back({static_cast<ExecutableBindingKind>(*binding),
                                 *index, ordinal, tensor, required,
                                 *alignment});
    }
    parsed.slots.push_back(std::move(slot));
  }
  size_t bundleStart = reader.position();
  if (*abiRoot !=
      digest(bytes.slice(invocationStart, abiRootOffset - invocationStart),
             bytes.slice(abiSuffixStart, bundleStart - abiSuffixStart)))
    return invalidTransport();
  if (parsed.bindings.size() != 3)
    return invalidTransport();
  auto bindingMatches = [&](size_t position, ExecutableBindingKind kind,
                            int64_t index, int64_t slot,
                            ArrayRef<int64_t> shape) {
    const GpuExternalBinding &binding = parsed.bindings[position];
    auto tensor = dyn_cast<RankedTensorType>(binding.tensorType);
    return binding.kind == kind && binding.index == index &&
           binding.slotOrdinal == slot && tensor &&
           tensor.getElementType().isBF16() && tensor.getShape() == shape &&
           binding.requiredBytes ==
               (shape.size() == 1 ? 4096 * 2 : 2048 * 4096 * 2) &&
           binding.alignment == 2;
  };
  if (!bindingMatches(0, ExecutableBindingKind::Operand, 0, 0, {2048, 4096}) ||
      !bindingMatches(1, ExecutableBindingKind::Operand, 1, 1, {4096}) ||
      !bindingMatches(2, ExecutableBindingKind::Result, 0, 20, {2048, 4096}))
    return invalidTransport();
  auto bundleSchema = reader.i64();
  auto bundleSchedule = reader.text();
  auto bundleDevice = reader.text();
  auto bundleAbi = reader.text();
  auto completion = reader.u8();
  size_t bundleRootOffset = reader.position();
  auto bundleRoot = reader.text();
  if (temporaryBytes != 201416716 || bundleSchema != 2 || !bundleSchedule ||
      *bundleSchedule != *schedule || !bundleDevice ||
      *bundleDevice != *deviceRoot || !bundleAbi || *bundleAbi != *abiRoot ||
      completion != 1 || !bundleRoot ||
      *bundleRoot != digest(bytes.take_front(bundleRootOffset)) ||
      !reader.done())
    return invalidTransport();
  return parsed;
}

} // namespace

class GpuExecutable::Impl {
public:
  explicit Impl(Parsed parsed) : parsed(std::move(parsed)) {}
  Parsed parsed;
};

GpuExecutable::GpuExecutable(std::shared_ptr<const Impl> implementation)
    : implementation(std::move(implementation)) {}

absl::StatusOr<std::shared_ptr<const GpuExecutable>>
GpuExecutable::Load(ArrayRef<uint8_t> bytes) {
  try {
    absl::StatusOr<Parsed> parsed = parse(bytes);
    if (!parsed.ok())
      return parsed.status();
    auto implementation = std::make_shared<const Impl>(std::move(*parsed));
    return std::shared_ptr<const GpuExecutable>(
        new GpuExecutable(std::move(implementation)));
  } catch (const std::exception &) {
    return invalidTransport();
  } catch (...) {
    return invalidTransport();
  }
}

ArrayRef<uint8_t> GpuExecutable::codeBytes() const {
  return implementation->parsed.code;
}
ArrayRef<GpuLaunch> GpuExecutable::launches() const {
  return implementation->parsed.launches;
}
ArrayRef<GpuSlot> GpuExecutable::slots() const {
  return implementation->parsed.slots;
}
ArrayRef<GpuExternalBinding> GpuExecutable::externalBindings() const {
  return implementation->parsed.bindings;
}

std::string gpuExecutableBundleDigest(ArrayRef<uint8_t> bytes) {
  return digest(bytes);
}

FailureOr<SmallVector<uint8_t>> serializeGpuExecutableBundle(ModuleOp module) {
  SmallVector<DeviceModuleOp> devices(module.getOps<DeviceModuleOp>());
  SmallVector<InvocationAbiOp> abis(module.getOps<InvocationAbiOp>());
  SmallVector<ExecutableBundleOp> bundles(module.getOps<ExecutableBundleOp>());
  if (devices.size() != 1 || abis.size() != 1 || bundles.size() != 1 ||
      failed(devices.front().verifyRegions()) ||
      failed(abis.front().verifyRegions()) || failed(bundles.front().verify()))
    return failure();
  DeviceModuleOp device = devices.front();
  InvocationAbiOp abi = abis.front();
  ExecutableBundleOp bundle = bundles.front();
  if (device.getSchemaVersion() != 3 ||
      device.getCodeFormat() != ExecutableCodeFormat::CudaPtxSm90V1 ||
      abi.getSchemaVersion() != 3 || bundle.getSchemaVersion() != 2)
    return failure();

  Writer devicePrefix;
  devicePrefix.i64(3);
  devicePrefix.u8(2);
  devicePrefix.u8(0);
  devicePrefix.text(device.getSourceScheduleFingerprint());
  devicePrefix.blob(device.getCode());
  devicePrefix.text(device.getCodeDigest());
  Writer deviceSuffix;
  SmallVector<DeviceEntryOp> entries(
      device.getBody().front().getOps<DeviceEntryOp>());
  deviceSuffix.u64(entries.size());
  for (DeviceEntryOp entry : entries)
    writeEntry(deviceSuffix, entry);
  std::string deviceRoot = digest(devicePrefix.view(), deviceSuffix.view());

  Writer abiPrefix;
  abiPrefix.i64(3);
  abiPrefix.text(abi.getSourcePlanFingerprint());
  abiPrefix.text(abi.getSourceScheduleFingerprint());
  Writer abiSuffix;
  SmallVector<InvocationSlotOp> slots(
      abi.getBody().front().getOps<InvocationSlotOp>());
  abiSuffix.u64(slots.size());
  for (InvocationSlotOp slot : slots)
    writeSlot(abiSuffix, slot);
  std::string abiRoot = digest(abiPrefix.view(), abiSuffix.view());

  Writer writer;
  writer.append(kMagic);
  writer.u32(2);
  writer.append(devicePrefix.view());
  writer.text(deviceRoot);
  writer.append(deviceSuffix.view());
  writer.append(abiPrefix.view());
  writer.text(abiRoot);
  writer.append(abiSuffix.view());
  writer.i64(2);
  writer.text(bundle.getSourceScheduleFingerprint());
  writer.text(deviceRoot);
  writer.text(abiRoot);
  writer.u8(1);
  writer.text(digest(writer.view()));
  SmallVector<uint8_t> bytes = writer.take();
  if (bytes.size() > kMaximumGpuTransportBytes)
    return failure();
  if (!GpuExecutable::Load(bytes).ok())
    return failure();
  return bytes;
}

} // namespace mlir::shuttle
