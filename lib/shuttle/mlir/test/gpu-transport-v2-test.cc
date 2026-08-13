// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "shuttle/Runtime/GpuFfi.h"
#include "shuttle/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "tools/cpp/runfiles/runfiles.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/invoke.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "gtest/gtest.h"

namespace {

class Reader {
public:
  explicit Reader(llvm::ArrayRef<uint8_t> bytes) : bytes(bytes) {}

  uint8_t u8() { return integer<uint8_t>(); }
  uint16_t u16() { return integer<uint16_t>(); }
  uint32_t u32() { return integer<uint32_t>(); }
  uint64_t u64() { return integer<uint64_t>(); }
  int64_t i64() { return static_cast<int64_t>(u64()); }
  llvm::StringRef text(uint64_t *payloadOffset = nullptr) {
    uint64_t size = u64();
    if (payloadOffset)
      *payloadOffset = position();
    auto value = take(size);
    return {reinterpret_cast<const char *>(value.data()), value.size()};
  }
  llvm::ArrayRef<uint8_t> blob(uint64_t *payloadOffset = nullptr) {
    uint64_t size = u64();
    if (payloadOffset)
      *payloadOffset = position();
    return take(size);
  }
  llvm::SmallVector<int64_t> integers() {
    uint64_t count = u64();
    if (count > 256) {
      failed = true;
      return {};
    }
    llvm::SmallVector<int64_t> values;
    for (uint64_t index = 0; index < count; ++index)
      values.push_back(i64());
    return values;
  }
  bool ok() const { return !failed; }
  bool done() const { return ok() && offset == bytes.size(); }
  uint64_t position() const { return offset; }

private:
  llvm::ArrayRef<uint8_t> take(uint64_t count) {
    if (failed || count > bytes.size() - offset) {
      failed = true;
      return {};
    }
    auto value = bytes.slice(offset, count);
    offset += count;
    return value;
  }
  template <typename T> T integer() {
    auto encoded = take(sizeof(T));
    if (encoded.size() != sizeof(T))
      return 0;
    T value = 0;
    for (unsigned index = 0; index < sizeof(T); ++index)
      value |= static_cast<T>(encoded[index]) << (index * 8);
    return value;
  }

  llvm::ArrayRef<uint8_t> bytes;
  uint64_t offset = 0;
  bool failed = false;
};

std::string sha256(llvm::ArrayRef<uint8_t> bytes) {
  llvm::SHA256 sha;
  sha.update(bytes);
  return llvm::toHex(sha.final(), true);
}

std::string sha256(llvm::ArrayRef<uint8_t> prefix,
                   llvm::ArrayRef<uint8_t> suffix) {
  llvm::SHA256 sha;
  sha.update(prefix);
  sha.update(suffix);
  return llvm::toHex(sha.final(), true);
}

bool isDigest(llvm::StringRef value) {
  return value.size() == 64 && llvm::all_of(value, [](char character) {
           return llvm::isDigit(character) ||
                  (character >= 'a' && character <= 'f');
         });
}

struct Decoded {
  llvm::SmallVector<uint8_t> code;
  llvm::SmallVector<mlir::shuttle::GpuLaunch> launches;
  int64_t slots = 0;
  uint64_t deviceRootOffset = 0;
  uint64_t deviceRootPayloadOffset = 0;
  uint64_t deviceSuffixStart = 0;
  uint64_t invocationStart = 0;
  uint64_t bundleDeviceRootPayloadOffset = 0;
  uint64_t bundleRootOffset = 0;
  uint64_t bundleRootPayloadOffset = 0;
  llvm::SmallVector<uint64_t> launchGridOffsets;
};

std::optional<Decoded> decodeInlineSchema2(llvm::ArrayRef<uint8_t> bytes) {
  if (bytes.size() > 16 * 1024 * 1024)
    return std::nullopt;
  Reader reader(bytes);
  for (char expected : llvm::StringRef("SHUTGPU\0", 8))
    if (reader.u8() != static_cast<uint8_t>(expected))
      return std::nullopt;
  if (reader.u32() != 2 || reader.i64() != 3 || reader.u8() != 2 ||
      reader.u8() != 0)
    return std::nullopt;
  constexpr uint64_t deviceStart = 12;
  llvm::StringRef schedule = reader.text();
  if (!isDigest(schedule))
    return std::nullopt;
  auto code = reader.blob();
  if (code.size() > 8 * 1024 * 1024 || reader.text() != sha256(code))
    return std::nullopt;
  uint64_t deviceRootOffset = reader.position();
  uint64_t deviceRootPayloadOffset = 0;
  llvm::StringRef deviceRoot = reader.text(&deviceRootPayloadOffset);
  uint64_t deviceSuffixStart = reader.position();
  if (!isDigest(deviceRoot) || reader.u64() != 19)
    return std::nullopt;

  llvm::SmallVector<mlir::shuttle::GpuLaunch> launches;
  llvm::SmallVector<uint64_t> launchGridOffsets;
  int64_t nextOffset = 0;
  int64_t taskPositions = 0;
  for (int64_t ordinal = 0; ordinal < 19; ++ordinal) {
    if (reader.i64() != ordinal || reader.i64() != ordinal ||
        reader.i64() != nextOffset)
      return std::nullopt;
    int64_t length = reader.i64();
    if (length <= 0 || length > 512 * 1024 ||
        nextOffset > static_cast<int64_t>(code.size()) - length)
      return std::nullopt;
    auto inputs = reader.integers();
    auto outputs = reader.integers();
    if (llvm::any_of(inputs,
                     [](int64_t slot) { return slot < 0 || slot >= 21; }) ||
        llvm::any_of(outputs,
                     [](int64_t slot) { return slot < 0 || slot >= 21; }))
      return std::nullopt;
    if (reader.u64() != inputs.size())
      return std::nullopt;
    for (size_t index = 0; index < inputs.size(); ++index)
      if (reader.u8() != 0)
        return std::nullopt;
    if (reader.u64() != outputs.size())
      return std::nullopt;
    for (size_t index = 0; index < outputs.size(); ++index)
      if (reader.u8() != 1)
        return std::nullopt;
    auto dependencies = reader.integers();
    if (llvm::any_of(dependencies,
                     [ordinal](int64_t dependency) {
                       return dependency < 0 || dependency >= ordinal;
                     }) ||
        reader.u8() > 1)
      return std::nullopt;
    uint8_t hasReduction = reader.u8();
    if (hasReduction > 1 || (hasReduction && reader.u8() > 0))
      return std::nullopt;
    llvm::StringRef codeDigest = reader.text();
    auto slice = code.slice(nextOffset, length);
    if (codeDigest != sha256(slice))
      return std::nullopt;
    nextOffset += length;
    uint64_t launchGridOffset = reader.position();
    std::array<uint64_t, 3> grid{reader.u32(), reader.u32(), reader.u32()};
    std::array<uint64_t, 3> block{reader.u16(), reader.u16(), reader.u16()};
    uint32_t shared = reader.u32();
    if (llvm::is_contained(grid, 0) || llvm::is_contained(block, 0) ||
        block[0] * block[1] * block[2] > 1024 || shared > 16 * 1024 ||
        reader.u16() != inputs.size() + outputs.size())
      return std::nullopt;
    constexpr uint64_t maximumTaskPositions = 67129347;
    uint64_t positions = 1;
    for (uint64_t extent : llvm::concat<uint64_t>(grid, block)) {
      if (extent > maximumTaskPositions / positions)
        return std::nullopt;
      positions *= extent;
    }
    if (positions > maximumTaskPositions - taskPositions)
      return std::nullopt;
    taskPositions += positions;
    mlir::shuttle::GpuLaunch launch;
    launch.taskOrdinal = ordinal;
    launch.codeOffset = nextOffset - length;
    launch.codeLength = length;
    launch.codeDigest = codeDigest.str();
    launch.grid = grid;
    launch.block = block;
    launch.dynamicSharedMemoryBytes = shared;
    launch.inputSlots.assign(inputs.begin(), inputs.end());
    launch.outputSlots.assign(outputs.begin(), outputs.end());
    launch.dependencies.assign(dependencies.begin(), dependencies.end());
    launches.push_back(std::move(launch));
    launchGridOffsets.push_back(launchGridOffset);
  }
  uint64_t invocationStart = reader.position();
  if (deviceRoot !=
      sha256(
          bytes.slice(deviceStart, deviceRootOffset - deviceStart),
          bytes.slice(deviceSuffixStart, invocationStart - deviceSuffixStart)))
    return std::nullopt;
  if (!reader.ok() || nextOffset != code.size() || reader.i64() != 3 ||
      !isDigest(reader.text()) || reader.text() != schedule)
    return std::nullopt;
  uint64_t invocationRootOffset = reader.position();
  llvm::StringRef invocationRoot = reader.text();
  uint64_t invocationSuffixStart = reader.position();
  if (!isDigest(invocationRoot) || reader.u64() != 21)
    return std::nullopt;
  int64_t temporaryBytes = 0;
  int64_t scalarSlots = 0;
  int64_t externalSlots = 0;
  for (int64_t ordinal = 0; ordinal < 21; ++ordinal) {
    if (reader.i64() != ordinal || reader.i64() != ordinal)
      return std::nullopt;
    uint8_t elementType = reader.u8();
    if (elementType > 1)
      return std::nullopt;
    uint64_t rank = reader.u64();
    if (rank > 8)
      return std::nullopt;
    llvm::SmallVector<int64_t> dimensions;
    int64_t expectedBytes = elementType == 0 ? 2 : 4;
    for (uint64_t axis = 0; axis < rank; ++axis) {
      int64_t extent = reader.i64();
      if (extent <= 0 ||
          expectedBytes > std::numeric_limits<int64_t>::max() / extent)
        return std::nullopt;
      dimensions.push_back(extent);
      expectedBytes *= extent;
    }
    int64_t required = reader.i64();
    auto strides = reader.integers();
    if (required != expectedBytes || required > 32 * 1024 * 1024 ||
        strides.size() != rank || reader.i64() != 0 ||
        reader.i64() != (elementType == 0 ? 2 : 4) || reader.u8() != 1 ||
        reader.u8() > 2)
      return std::nullopt;
    uint8_t storage = reader.u8();
    if (storage > 1 || reader.i64() != ordinal || reader.i64() != ordinal)
      return std::nullopt;
    uint8_t binding = reader.u8();
    uint8_t hasBindingIndex = reader.u8();
    if (binding > 2 || hasBindingIndex > 1)
      return std::nullopt;
    int64_t bindingIndex = hasBindingIndex ? reader.i64() : -1;
    if (rank == 0)
      ++scalarSlots;
    if (storage == 1) {
      if (binding != 0 || hasBindingIndex)
        return std::nullopt;
      temporaryBytes += required;
    } else {
      ++externalSlots;
      if (!((ordinal == 0 && binding == 1 && bindingIndex == 0) ||
            (ordinal == 1 && binding == 1 && bindingIndex == 1) ||
            (ordinal == 20 && binding == 2 && bindingIndex == 0)))
        return std::nullopt;
    }
  }
  uint64_t bundleStart = reader.position();
  if (invocationRoot !=
      sha256(
          bytes.slice(invocationStart, invocationRootOffset - invocationStart),
          bytes.slice(invocationSuffixStart,
                      bundleStart - invocationSuffixStart)))
    return std::nullopt;
  if (temporaryBytes != 201416716 || scalarSlots != 3 || externalSlots != 3 ||
      reader.i64() != 2 || reader.text() != schedule)
    return std::nullopt;
  uint64_t bundleDeviceRootPayloadOffset = 0;
  if (reader.text(&bundleDeviceRootPayloadOffset) != deviceRoot ||
      reader.text() != invocationRoot || reader.u8() != 1)
    return std::nullopt;
  uint64_t rootOffset = reader.position();
  uint64_t bundleRootPayloadOffset = 0;
  llvm::StringRef bundleRoot = reader.text(&bundleRootPayloadOffset);
  if (!reader.done() || bundleRoot != sha256(bytes.take_front(rootOffset)))
    return std::nullopt;
  return Decoded{llvm::SmallVector<uint8_t>(code),
                 std::move(launches),
                 21,
                 deviceRootOffset,
                 deviceRootPayloadOffset,
                 deviceSuffixStart,
                 invocationStart,
                 bundleDeviceRootPayloadOffset,
                 rootOffset,
                 bundleRootPayloadOffset,
                 std::move(launchGridOffsets)};
}

void writeU32(llvm::MutableArrayRef<uint8_t> bytes, uint64_t offset,
              uint32_t value) {
  for (unsigned byte = 0; byte < sizeof(value); ++byte)
    bytes[offset + byte] = static_cast<uint8_t>(value >> (byte * 8));
}

void writeDigest(llvm::MutableArrayRef<uint8_t> bytes, uint64_t offset,
                 llvm::StringRef digest) {
  assert(digest.size() == 64);
  llvm::copy(digest, bytes.begin() + offset);
}

void rehashDeviceAndBundle(llvm::MutableArrayRef<uint8_t> bytes,
                           const Decoded &decoded) {
  std::string deviceRoot =
      sha256(bytes.slice(/*deviceStart=*/12, decoded.deviceRootOffset - 12),
             bytes.slice(decoded.deviceSuffixStart,
                         decoded.invocationStart - decoded.deviceSuffixStart));
  writeDigest(bytes, decoded.deviceRootPayloadOffset, deviceRoot);
  writeDigest(bytes, decoded.bundleDeviceRootPayloadOffset, deviceRoot);
  writeDigest(bytes, decoded.bundleRootPayloadOffset,
              sha256(bytes.take_front(decoded.bundleRootOffset)));
}

llvm::SmallVector<uint8_t> oversizedLaunchGrid(llvm::ArrayRef<uint8_t> bytes,
                                               const Decoded &decoded) {
  llvm::SmallVector<uint8_t> result(bytes);
  uint64_t gridOffset = decoded.launchGridOffsets.front();
  writeU32(result, gridOffset, uint32_t{1} << 31);
  writeU32(result, gridOffset + 4, uint32_t{1} << 31);
  writeU32(result, gridOffset + 8, 4);
  rehashDeviceAndBundle(result, decoded);
  return result;
}

struct BuiltBundle {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

std::optional<std::string> fixtureSource() {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  if (!runfiles)
    return std::nullopt;
  std::ifstream input(runfiles->Rlocation(
      "shuttle_mlir/test/Inputs/"
      "jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir"));
  std::ostringstream source;
  source << input.rdbuf();
  if (!input)
    return std::nullopt;
  return source.str();
}

std::unique_ptr<BuiltBundle> buildFixture(llvm::StringRef source) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::shuttle::ShuttleDialect>();
  auto result = std::make_unique<BuiltBundle>();
  result->context = std::make_unique<mlir::MLIRContext>(registry);
  result->module =
      mlir::parseSourceString<mlir::ModuleOp>(source, result->context.get());
  if (!result->module)
    return {};
  mlir::PassManager manager(result->context.get());
  manager.addPass(mlir::shuttle::createAnnotateSourcePass());
  manager.addPass(mlir::shuttle::createFormStructuralRegionsPass());
  manager.addPass(mlir::shuttle::createConvertStablehloToAlgebraPass());
  manager.addPass(mlir::shuttle::createPlanRowFoldMaterializationPass());
  manager.addPass(mlir::shuttle::createPlanSimt32RowFoldSchedulePass());
  manager.addPass(mlir::shuttle::createBuildGpuExecutableBundlePass());
  manager.addPass(mlir::shuttle::createVerifyGpuExecutableBundlePass());
  if (mlir::failed(manager.run(*result->module)))
    return {};
  return result;
}

std::unique_ptr<BuiltBundle> buildFixture() {
  auto source = fixtureSource();
  if (!source)
    return {};
  return buildFixture(*source);
}

std::unique_ptr<BuiltBundle> buildAlternateFixture() {
  auto source = fixtureSource();
  if (!source)
    return {};
  constexpr llvm::StringLiteral original = "9.99999974E-6";
  constexpr llvm::StringLiteral replacement = "2.49999994E-6";
  size_t position = source->find(original.str());
  if (position == std::string::npos)
    return {};
  source->replace(position, original.size(), replacement.data(),
                  replacement.size());
  return buildFixture(*source);
}

TEST(GpuTransportV2Test, IndependentInlineDecoderMatchesPublicDescriptor) {
  auto built = buildFixture();
  ASSERT_TRUE(built);
  auto bytes = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(bytes));
  auto independent = decodeInlineSchema2(*bytes);
  ASSERT_TRUE(independent);
  auto publicDescriptor = mlir::shuttle::GpuExecutable::Load(*bytes);
  ASSERT_TRUE(publicDescriptor.ok()) << publicDescriptor.status();
  ASSERT_EQ(independent->launches.size(), 19);
  ASSERT_EQ((*publicDescriptor)->launches().size(), 19);
  EXPECT_EQ(independent->slots, 21);
  EXPECT_EQ(llvm::ArrayRef<uint8_t>(independent->code),
            (*publicDescriptor)->codeBytes());
  for (auto [decoded, loaded] : llvm::zip_equal(
           independent->launches, (*publicDescriptor)->launches())) {
    EXPECT_EQ(decoded.taskOrdinal, loaded.taskOrdinal);
    EXPECT_EQ(decoded.codeOffset, loaded.codeOffset);
    EXPECT_EQ(decoded.codeLength, loaded.codeLength);
    EXPECT_EQ(decoded.codeDigest, loaded.codeDigest);
    EXPECT_EQ(decoded.grid, loaded.grid);
    EXPECT_EQ(decoded.block, loaded.block);
    EXPECT_EQ(decoded.dynamicSharedMemoryBytes,
              loaded.dynamicSharedMemoryBytes);
    EXPECT_EQ(decoded.inputSlots, loaded.inputSlots);
    EXPECT_EQ(decoded.outputSlots, loaded.outputSlots);
    EXPECT_EQ(decoded.dependencies, loaded.dependencies);
  }
  llvm::SmallVector<uint8_t> corruptRoot(*bytes);
  corruptRoot.back() ^= 1;
  EXPECT_FALSE(decodeInlineSchema2(corruptRoot));
  EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(corruptRoot).ok());

  auto oversized = oversizedLaunchGrid(*bytes, *independent);
  EXPECT_FALSE(decodeInlineSchema2(oversized));
  EXPECT_FALSE(mlir::shuttle::GpuExecutable::Load(oversized).ok());
}

class SyntheticAllocator final
    : public stream_executor::DeviceAddressAllocator {
public:
  explicit SyntheticAllocator(uintptr_t base = 0x10000000)
      : DeviceAddressAllocator(nullptr), base(base) {}

  absl::StatusOr<stream_executor::ScopedDeviceAddress<uint8_t>>
  Allocate(int deviceOrdinal, uint64_t size, bool, int64_t) final {
    if (failAt && sizes.size() == *failAt)
      return absl::ResourceExhaustedError("synthetic allocation failure");
    sizes.push_back(size);
    uintptr_t address = base;
    for (uint64_t previous : sizes)
      address += previous + 256;
    return stream_executor::ScopedDeviceAddress<uint8_t>(
        stream_executor::DeviceAddressBase(reinterpret_cast<void *>(address),
                                           size),
        deviceOrdinal, this);
  }
  absl::Status Deallocate(int, stream_executor::DeviceAddressBase) final {
    ++deallocations;
    return absl::OkStatus();
  }
  absl::StatusOr<stream_executor::Stream *> GetStream(int) final {
    return absl::UnimplementedError("test allocator has no private stream");
  }

  llvm::SmallVector<uint64_t> sizes;
  int64_t deallocations = 0;
  std::optional<size_t> failAt;

private:
  uintptr_t base;
};

struct LoadedKernel {
  std::string ptx;
  std::string symbol;
  size_t arity = 0;
};

struct KernelLaunch {
  size_t ordinal = 0;
  std::array<uint64_t, 3> grid;
  std::array<uint64_t, 3> block;
  uint64_t sharedMemoryBytes = 0;
  llvm::SmallVector<uintptr_t> arguments;
};

struct ExecutionTrace {
  std::mutex mutex;
  llvm::SmallVector<LoadedKernel> loads;
  llvm::SmallVector<KernelLaunch> launches;
  std::atomic<int64_t> launchAttempts{0};
  std::atomic<int64_t> failAt{-1};
};

class RecordingKernel final : public stream_executor::Kernel {
public:
  RecordingKernel(unsigned arity, size_t ordinal,
                  std::shared_ptr<ExecutionTrace> trace)
      : arity(arity), ordinal(ordinal), trace(std::move(trace)) {}
  unsigned Arity() const final { return arity; }
  absl::StatusOr<int32_t>
  GetMaxOccupiedBlocksPerCore(stream_executor::ThreadDim, size_t) const final {
    return 1;
  }
  absl::Status Launch(const stream_executor::ThreadDim &thread,
                      const stream_executor::BlockDim &block,
                      const std::optional<stream_executor::ClusterDim> &,
                      stream_executor::Stream *,
                      const stream_executor::KernelArgs &args) final {
    int64_t attempt = trace->launchAttempts.fetch_add(1);
    if (trace->failAt.load() == attempt)
      return absl::InternalError("synthetic enqueue failure");
    auto *packed =
        stream_executor::DynCast<stream_executor::KernelArgsPackedArrayBase>(
            &args);
    KernelLaunch launch{ordinal,
                        {block.x, block.y, block.z},
                        {thread.x, thread.y, thread.z},
                        args.number_of_shared_bytes(),
                        {}};
    for (const void *argument : packed->argument_addresses()) {
      void *address = nullptr;
      std::memcpy(&address, argument, sizeof(address));
      launch.arguments.push_back(reinterpret_cast<uintptr_t>(address));
    }
    std::lock_guard lock(trace->mutex);
    trace->launches.push_back(std::move(launch));
    return absl::OkStatus();
  }

private:
  unsigned arity;
  size_t ordinal;
  std::shared_ptr<ExecutionTrace> trace;
};

xla::ffi::CallFrame callFrame(llvm::ArrayRef<uint8_t> transport,
                              llvm::StringRef transportDigest) {
  xla::ffi::CallFrameBuilder builder(/*num_args=*/2, /*num_rets=*/1);
  constexpr std::array<int64_t, 2> matrix{2048, 4096};
  constexpr std::array<int64_t, 1> vector{4096};
  builder.AddBufferArg(
      stream_executor::DeviceAddressBase(reinterpret_cast<void *>(0x40000000),
                                         2048 * 4096 * 2),
      xla::PrimitiveType::BF16, matrix);
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           reinterpret_cast<void *>(0x50000000), 4096 * 2),
                       xla::PrimitiveType::BF16, vector);
  builder.AddBufferRet(
      stream_executor::DeviceAddressBase(reinterpret_cast<void *>(0x60000000),
                                         2048 * 4096 * 2),
      xla::PrimitiveType::BF16, matrix);
  xla::ffi::CallFrameBuilder::AttributesBuilder attributes;
  attributes.Insert("bundle_bytes", std::string(reinterpret_cast<const char *>(
                                                    transport.data()),
                                                transport.size()));
  attributes.Insert("bundle_sha256", transportDigest.str());
  attributes.Insert("bundle_size", static_cast<int64_t>(transport.size()));
  attributes.Insert("transport_schema_version", int64_t{2});
  attributes.Insert("device_schema_version", int64_t{3});
  attributes.Insert("invocation_abi_schema_version", int64_t{3});
  attributes.Insert("bundle_schema_version", int64_t{2});
  attributes.Insert("completion", std::string("stream_ordered"));
  builder.AddAttributes(attributes.Build());
  return builder.Build();
}

std::unique_ptr<stream_executor::Kernel>
recordLoadedKernel(const stream_executor::KernelLoaderSpec &spec,
                   const std::shared_ptr<ExecutionTrace> &trace) {
  auto ptx = spec.cuda_ptx_in_memory();
  if (!ptx)
    return nullptr;
  size_t ordinal = 0;
  {
    std::lock_guard lock(trace->mutex);
    ordinal = trace->loads.size();
    trace->loads.push_back(
        {std::string(ptx->ptx), spec.kernel_name(), spec.arity()});
  }
  return std::make_unique<RecordingKernel>(spec.arity(), ordinal, trace);
}

void expectLoadedKernels(const Decoded &decoded, const ExecutionTrace &trace,
                         size_t start) {
  ASSERT_GE(trace.loads.size(), start + decoded.launches.size());
  for (size_t ordinal = 0; ordinal < decoded.launches.size(); ++ordinal) {
    const auto &expected = decoded.launches[ordinal];
    const auto &actual = trace.loads[start + ordinal];
    EXPECT_EQ(llvm::StringRef(actual.ptx),
              llvm::StringRef(reinterpret_cast<const char *>(
                                  decoded.code.data() + expected.codeOffset),
                              expected.codeLength));
    EXPECT_EQ(actual.symbol, "shuttle_entry");
    EXPECT_EQ(actual.arity,
              expected.inputSlots.size() + expected.outputSlots.size());
  }
}

void expectKernelLaunches(const Decoded &decoded, const ExecutionTrace &trace,
                          size_t start,
                          const llvm::DenseMap<int64_t, uintptr_t> &slots) {
  ASSERT_GE(trace.launches.size(), start + decoded.launches.size());
  for (size_t ordinal = 0; ordinal < decoded.launches.size(); ++ordinal) {
    const auto &expected = decoded.launches[ordinal];
    const auto &actual = trace.launches[start + ordinal];
    EXPECT_EQ(actual.ordinal, ordinal);
    EXPECT_EQ(actual.grid, expected.grid);
    EXPECT_EQ(actual.block, expected.block);
    EXPECT_EQ(actual.sharedMemoryBytes, expected.dynamicSharedMemoryBytes);
    llvm::SmallVector<uintptr_t> expectedArguments;
    for (int64_t slot : expected.inputSlots)
      expectedArguments.push_back(slots.lookup(slot));
    for (int64_t slot : expected.outputSlots)
      expectedArguments.push_back(slots.lookup(slot));
    EXPECT_EQ(actual.arguments, expectedArguments);
  }
}

TEST(GpuTransportV2Test, ExportedFfiBundleOwnsEveryLifecycleStage) {
  auto built = buildFixture();
  ASSERT_TRUE(built);
  auto transport = mlir::shuttle::serializeGpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(transport));
  std::string transportDigest =
      mlir::shuttle::gpuExecutableBundleDigest(*transport);
  auto frame = callFrame(*transport, transportDigest);
  auto decoded = decodeInlineSchema2(*transport);
  ASSERT_TRUE(decoded);
  auto alternateBuilt = buildAlternateFixture();
  ASSERT_TRUE(alternateBuilt);
  auto alternateTransport =
      mlir::shuttle::serializeGpuExecutableBundle(*alternateBuilt->module);
  ASSERT_TRUE(mlir::succeeded(alternateTransport));
  auto alternateDecoded = decodeInlineSchema2(*alternateTransport);
  ASSERT_TRUE(alternateDecoded);
  ASSERT_NE(decoded->code, alternateDecoded->code);
  XLA_FFI_Handler_Bundle handlers =
      mlir::shuttle::gpuExecutableBundleFfiHandlerBundle();
  ASSERT_NE(handlers.instantiate, nullptr);
  ASSERT_NE(handlers.prepare, nullptr);
  ASSERT_NE(handlers.initialize, nullptr);
  ASSERT_NE(handlers.execute, nullptr);

  SyntheticAllocator allocator;
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  stream_executor::GpuComputeCapability capability{
      stream_executor::CudaComputeCapability{9, 0}};
  auto trace = std::make_shared<ExecutionTrace>();
  EXPECT_CALL(stream, parent()).WillRepeatedly(testing::Return(&executor));
  EXPECT_CALL(executor, LoadKernel(testing::_))
      .Times(38)
      .WillRepeatedly([trace](const stream_executor::KernelLoaderSpec &spec) {
        return recordLoadedKernel(spec, trace);
      });

  xla::ffi::ExecutionState instantiateState;
  xla::ffi::ExecutionState prepareState;
  xla::ffi::ExecutionState initializeState;
  xla::ffi::InvokeContext context;
  context.device_ordinal = 0;
  context.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &allocator, nullptr, nullptr,
      nullptr, nullptr,    nullptr, &capability};
  context.state_context = {&instantiateState, &prepareState, &initializeState};

  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                               frame, context,
                               XLA_FFI_ExecutionStage_INSTANTIATE)
                  .ok());
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                               frame, context, XLA_FFI_ExecutionStage_PREPARE)
                  .ok());
  ASSERT_EQ(allocator.sizes.size(), 18);
  EXPECT_EQ(llvm::accumulate(allocator.sizes, uint64_t{0}), 201416716);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.initialize,
                               frame, context,
                               XLA_FFI_ExecutionStage_INITIALIZE)
                  .ok());
  expectLoadedKernels(*decoded, *trace, 0);
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                               frame, context, XLA_FFI_ExecutionStage_EXECUTE)
                  .ok());
  EXPECT_EQ(trace->launchAttempts.load(), 19);
  llvm::DenseMap<int64_t, uintptr_t> firstSlots{
      {0, 0x40000000}, {1, 0x50000000}, {20, 0x60000000}};
  ASSERT_EQ(allocator.sizes.size(), 18);
  uintptr_t temporaryAddress = 0x10000000;
  for (int64_t slot = 2; slot < 20; ++slot) {
    temporaryAddress += allocator.sizes[slot - 2] + 256;
    firstSlots[slot] = temporaryAddress;
  }
  expectKernelLaunches(*decoded, *trace, 0, firstSlots);

  trace->launchAttempts.store(0);
  trace->failAt.store(7);
  EXPECT_FALSE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                                frame, context, XLA_FFI_ExecutionStage_EXECUTE)
                   .ok());
  EXPECT_EQ(trace->launchAttempts.load(), 8);
  trace->failAt.store(-1);

  trace->launchAttempts.store(0);
  SyntheticAllocator secondAllocator(0x70000000);
  xla::ffi::ExecutionState secondPrepareState;
  xla::ffi::InvokeContext secondContext = context;
  secondContext.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &secondAllocator, nullptr, nullptr,
      nullptr, nullptr,          nullptr, &capability};
  secondContext.state_context.prepare = &secondPrepareState;
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                               frame, secondContext,
                               XLA_FFI_ExecutionStage_PREPARE)
                  .ok());
  ASSERT_EQ(secondAllocator.sizes, allocator.sizes);
  size_t concurrentStart = 0;
  {
    std::lock_guard lock(trace->mutex);
    concurrentStart = trace->launches.size();
  }
  auto firstFrame = frame.Copy();
  auto secondFrame = frame.Copy();
  auto first = std::async(std::launch::async, [&] {
    return xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                            firstFrame, context,
                            XLA_FFI_ExecutionStage_EXECUTE);
  });
  auto second = std::async(std::launch::async, [&] {
    return xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.execute,
                            secondFrame, secondContext,
                            XLA_FFI_ExecutionStage_EXECUTE);
  });
  EXPECT_TRUE(first.get().ok());
  EXPECT_TRUE(second.get().ok());
  EXPECT_EQ(trace->launchAttempts.load(), 38);
  llvm::DenseMap<int64_t, uintptr_t> secondSlots{
      {0, 0x40000000}, {1, 0x50000000}, {20, 0x60000000}};
  temporaryAddress = 0x70000000;
  for (int64_t slot = 2; slot < 20; ++slot) {
    temporaryAddress += secondAllocator.sizes[slot - 2] + 256;
    secondSlots[slot] = temporaryAddress;
  }
  // Independent allocators must yield distinct invocation-local addresses.
  EXPECT_NE(firstSlots.lookup(2), secondSlots.lookup(2));
  llvm::SmallVector<int64_t> firstLaunchesPerOrdinal(19);
  llvm::SmallVector<int64_t> secondLaunchesPerOrdinal(19);
  {
    std::lock_guard lock(trace->mutex);
    ASSERT_EQ(trace->launches.size() - concurrentStart, 38);
    for (const auto &actual :
         llvm::drop_begin(trace->launches, concurrentStart)) {
      ASSERT_LT(actual.ordinal, decoded->launches.size());
      const auto &expected = decoded->launches[actual.ordinal];
      EXPECT_EQ(actual.grid, expected.grid);
      EXPECT_EQ(actual.block, expected.block);
      EXPECT_EQ(actual.sharedMemoryBytes, expected.dynamicSharedMemoryBytes);
      auto expectedArguments =
          [&](const llvm::DenseMap<int64_t, uintptr_t> &slots) {
            llvm::SmallVector<uintptr_t> arguments;
            for (int64_t slot : expected.inputSlots)
              arguments.push_back(slots.lookup(slot));
            for (int64_t slot : expected.outputSlots)
              arguments.push_back(slots.lookup(slot));
            return arguments;
          };
      auto firstArguments = expectedArguments(firstSlots);
      auto secondArguments = expectedArguments(secondSlots);
      ASSERT_NE(firstArguments, secondArguments);
      bool usesFirstState = actual.arguments == firstArguments;
      bool usesSecondState = actual.arguments == secondArguments;
      EXPECT_NE(usesFirstState, usesSecondState);
      if (usesFirstState)
        ++firstLaunchesPerOrdinal[actual.ordinal];
      if (usesSecondState)
        ++secondLaunchesPerOrdinal[actual.ordinal];
    }
  }
  EXPECT_TRUE(llvm::all_of(firstLaunchesPerOrdinal,
                           [](int64_t count) { return count == 1; }));
  EXPECT_TRUE(llvm::all_of(secondLaunchesPerOrdinal,
                           [](int64_t count) { return count == 1; }));

  auto alternateFrame =
      callFrame(*alternateTransport,
                mlir::shuttle::gpuExecutableBundleDigest(*alternateTransport));
  xla::ffi::ExecutionState alternateInstantiateState;
  xla::ffi::ExecutionState alternatePrepareState;
  xla::ffi::ExecutionState alternateInitializeState;
  SyntheticAllocator alternateAllocator(0x80000000);
  xla::ffi::InvokeContext alternateContext = context;
  alternateContext.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &alternateAllocator, nullptr, nullptr, nullptr, nullptr,
      nullptr, &capability};
  alternateContext.state_context = {&alternateInstantiateState,
                                    &alternatePrepareState,
                                    &alternateInitializeState};
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                               alternateFrame, alternateContext,
                               XLA_FFI_ExecutionStage_INSTANTIATE)
                  .ok());
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                               alternateFrame, alternateContext,
                               XLA_FFI_ExecutionStage_PREPARE)
                  .ok());
  ASSERT_TRUE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.initialize,
                               alternateFrame, alternateContext,
                               XLA_FFI_ExecutionStage_INITIALIZE)
                  .ok());
  expectLoadedKernels(*alternateDecoded, *trace, decoded->launches.size());

  auto corrupt = callFrame(*transport, std::string(64, '0'));
  EXPECT_FALSE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.instantiate,
                                corrupt, context,
                                XLA_FFI_ExecutionStage_INSTANTIATE)
                   .ok());

  SyntheticAllocator failingAllocator;
  failingAllocator.failAt = 5;
  xla::ffi::ExecutionState failedPrepare;
  context.backend_context = xla::ffi::InvokeContext::GpuContext{
      &stream, &failingAllocator, nullptr, nullptr,
      nullptr, nullptr,           nullptr, &capability};
  context.state_context.prepare = &failedPrepare;
  EXPECT_FALSE(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(), handlers.prepare,
                                frame, context, XLA_FFI_ExecutionStage_PREPARE)
                   .ok());
  EXPECT_EQ(failingAllocator.sizes.size(), 5);
  EXPECT_EQ(failingAllocator.deallocations, 5);
}

} // namespace
