// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/IR/ShuttleOps.h"
#include "shuttle/Runtime/CpuBytecode.h"
#include "shuttle/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "tools/cpp/runfiles/runfiles.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

namespace {

struct BuiltBundle {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

std::string runfile(llvm::StringRef name) {
  std::string error;
  std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> runfiles(
      bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error));
  if (!runfiles) {
    return {};
  }
  return runfiles->Rlocation(("shuttle_mlir/test/Inputs/" + name).str());
}

std::string readText(llvm::StringRef name) {
  std::ifstream input(runfile(name));
  std::ostringstream contents;
  contents << input.rdbuf();
  return input && !contents.str().empty() ? contents.str() : std::string();
}

std::unique_ptr<BuiltBundle> buildBundle(llvm::StringRef program) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::shuttle::ShuttleDialect>();
  auto built = std::make_unique<BuiltBundle>();
  built->context = std::make_unique<mlir::MLIRContext>(registry);
  built->module =
      mlir::parseSourceString<mlir::ModuleOp>(program, built->context.get());
  if (!built->module) {
    return {};
  }
  mlir::PassManager manager(built->context.get());
  manager.addPass(mlir::shuttle::createAnnotateSourcePass());
  manager.addPass(mlir::shuttle::createFormStructuralRegionsPass());
  manager.addPass(mlir::shuttle::createConvertStablehloToAlgebraPass());
  manager.addPass(mlir::shuttle::createPlanRowFoldMaterializationPass());
  manager.addPass(mlir::shuttle::createPlanSimt32RowFoldSchedulePass());
  manager.addPass(mlir::shuttle::createBuildCpuExecutableBundlePass());
  manager.addPass(mlir::shuttle::createVerifyCpuExecutableBundlePass());
  if (mlir::failed(manager.run(*built->module))) {
    return {};
  }
  for (mlir::Operation &operation :
       llvm::make_early_inc_range(built->module->getBody()->getOperations())) {
    if (!mlir::isa<mlir::shuttle::DeviceModuleOp,
                   mlir::shuttle::InvocationAbiOp,
                   mlir::shuttle::ExecutableBundleOp>(operation)) {
      operation.erase();
    }
  }
  return built;
}

llvm::SmallVector<uint8_t> goldenBytes(llvm::StringRef name) {
  std::string hex = readText(name);
  std::string compact;
  compact.reserve(hex.size());
  for (char value : hex) {
    if (value == '\n') {
      continue;
    }
    if (llvm::hexDigitValue(value) == -1U) {
      return {};
    }
    compact.push_back(value);
  }
  if (compact.size() % 2 != 0) {
    return {};
  }
  llvm::SmallVector<uint8_t> bytes;
  bytes.reserve(compact.size() / 2);
  for (size_t index = 0; index < compact.size(); index += 2) {
    unsigned high = llvm::hexDigitValue(compact[index]);
    unsigned low = llvm::hexDigitValue(compact[index + 1]);
    if (high == -1U || low == -1U) {
      return {};
    }
    bytes.push_back(static_cast<uint8_t>((high << 4) | low));
  }
  return bytes;
}

uint16_t toBf16(float value) {
  llvm::APFloat converted(value);
  bool losesInfo = false;
  converted.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                    &losesInfo);
  return converted.bitcastToAPInt().getZExtValue();
}

float fromBf16(uint16_t bits) {
  llvm::APFloat converted(llvm::APFloat::BFloat(), llvm::APInt(16, bits));
  bool losesInfo = false;
  converted.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return converted.convertToFloat();
}

struct Buffers {
  alignas(64) std::array<uint16_t, 7 * 13> input{};
  alignas(64) std::array<uint16_t, 13> scale{};
  alignas(64) std::array<uint16_t, 7 * 13> output{};

  std::array<mlir::shuttle::CpuExternalBuffer, 3> views() {
    return {{{0, llvm::MutableArrayRef<uint8_t>(
                     reinterpret_cast<uint8_t *>(input.data()), sizeof(input))},
             {1, llvm::MutableArrayRef<uint8_t>(
                     reinterpret_cast<uint8_t *>(scale.data()), sizeof(scale))},
             {20, llvm::MutableArrayRef<uint8_t>(
                      reinterpret_cast<uint8_t *>(output.data()),
                      sizeof(output))}}};
  }
};

void populate(Buffers &buffers, float shift) {
  for (size_t index = 0; index < buffers.input.size(); ++index) {
    buffers.input[index] =
        toBf16(shift + static_cast<float>((index * 7) % 29) / 16.0f);
  }
  for (size_t index = 0; index < buffers.scale.size(); ++index) {
    buffers.scale[index] = toBf16(0.75f + static_cast<float>(index) / 32.0f);
  }
}

std::array<uint16_t, 7 * 13> reference(const Buffers &buffers) {
  std::array<uint16_t, 7 * 13> expected{};
  for (int64_t row = 0; row < 7; ++row) {
    float sum = 0.0f;
    for (int64_t feature = 0; feature < 13; ++feature) {
      float value = fromBf16(buffers.input[row * 13 + feature]);
      sum = static_cast<float>(sum + static_cast<float>(value * value));
    }
    float inverse =
        static_cast<float>(1.0f / std::sqrt(sum / 13.0f + 9.99999974E-6f));
    for (int64_t feature = 0; feature < 13; ++feature) {
      const int64_t index = row * 13 + feature;
      expected[index] = toBf16(static_cast<float>(
          static_cast<float>(fromBf16(buffers.input[index]) * inverse) *
          fromBf16(buffers.scale[feature])));
    }
  }
  return expected;
}

void refreshAbiRoots(mlir::ModuleOp module) {
  auto abi = *module.getOps<mlir::shuttle::InvocationAbiOp>().begin();
  abi.setFingerprint(mlir::shuttle::invocationAbiFingerprint(abi));
  auto bundle = *module.getOps<mlir::shuttle::ExecutableBundleOp>().begin();
  bundle.setInvocationAbiFingerprint(abi.getFingerprint());
  bundle.setFingerprint(mlir::shuttle::executableBundleFingerprint(bundle));
}

TEST(CpuTransportContractTest, SerializerMatchesIndependentCanonicalGolden) {
  std::string program =
      readText("jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir");
  auto bundle = buildBundle(program);
  ASSERT_TRUE(bundle);
  auto serialized =
      mlir::shuttle::serializeCpuExecutableBundle(*bundle->module);
  ASSERT_TRUE(mlir::succeeded(serialized));
  llvm::SmallVector<uint8_t> golden =
      goldenBytes("cpu-forward-7x13-transport.hex");
  ASSERT_FALSE(golden.empty());
  EXPECT_EQ(*serialized, golden);
  EXPECT_EQ(mlir::shuttle::cpuExecutableBundleDigest(*serialized),
            "99e63ac5a004f5abce7b88fc12bd0fbf9d8fc14785fc9ae87ca32781165d0c31");
}

TEST(CpuTransportContractTest, DistinctProgramHasDistinctCanonicalTransport) {
  auto bundle = buildBundle(readText("cpu-forward-7x13-epsilon-quarter.mlir"));
  ASSERT_TRUE(bundle);
  auto serialized =
      mlir::shuttle::serializeCpuExecutableBundle(*bundle->module);
  ASSERT_TRUE(mlir::succeeded(serialized));
  llvm::SmallVector<uint8_t> golden =
      goldenBytes("cpu-forward-7x13-epsilon-quarter-transport.hex");
  ASSERT_FALSE(golden.empty());
  EXPECT_EQ(*serialized, golden);
  EXPECT_EQ(mlir::shuttle::cpuExecutableBundleDigest(*serialized),
            "8613f1c0fef79d343ec5dc161ec3a1ee458342b27dc381b7b371adbad5c9c15d");
  EXPECT_NE(golden, goldenBytes("cpu-forward-7x13-transport.hex"));
}

TEST(CpuTransportContractTest,
     VjpBundlesHaveFrozenCanonicalTransportAndExternalOrder) {
  struct ExpectedBundle {
    llvm::StringLiteral boundary;
    int64_t entries;
    int64_t slots;
    std::string digest;
    size_t bytes;
    llvm::SmallVector<int64_t> resultSlots;
  };
  const std::array<ExpectedBundle, 2> expected{{
      {"backward", 48, 51,
       "f7354ae3435e05d287306b55ff7931c86103bf041a4f1440b5970303e8c55af5",
       16103, {32, 50}},
      {"composed", 51, 54,
       "4088b329ab172b3cf98ebcfd12a066e2f5f3a25f9b70d9a6949e094c2f7a8b19",
       17090, {25, 35, 53}},
  }};

  for (const ExpectedBundle &item : expected) {
    auto built = buildBundle(readText(
        ("jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-" +
         item.boundary + ".mlir")
            .str()));
    ASSERT_TRUE(built) << item.boundary.str();
    auto device =
        *built->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
    auto abi =
        *built->module->getOps<mlir::shuttle::InvocationAbiOp>().begin();
    EXPECT_EQ(std::distance(device.getBody()
                                .front()
                                .getOps<mlir::shuttle::DeviceEntryOp>()
                                .begin(),
                            device.getBody()
                                .front()
                                .getOps<mlir::shuttle::DeviceEntryOp>()
                                .end()),
              item.entries);
    EXPECT_EQ(std::distance(abi.getBody()
                                .front()
                                .getOps<mlir::shuttle::InvocationSlotOp>()
                                .begin(),
                            abi.getBody()
                                .front()
                                .getOps<mlir::shuttle::InvocationSlotOp>()
                                .end()),
              item.slots);

    auto serialized =
        mlir::shuttle::serializeCpuExecutableBundle(*built->module);
    ASSERT_TRUE(mlir::succeeded(serialized));
    EXPECT_EQ(serialized->size(), item.bytes);
    EXPECT_EQ(mlir::shuttle::cpuExecutableBundleDigest(*serialized),
              item.digest);
    auto loaded = mlir::shuttle::CpuExecutable::Load(*serialized);
    ASSERT_TRUE(loaded.ok()) << loaded.status();
    auto bindings = (*loaded)->externalBindings();
    ASSERT_EQ(bindings.size(), 3 + item.resultSlots.size());
    for (int64_t index = 0; index < 3; ++index) {
      EXPECT_EQ(bindings[index].kind,
                mlir::shuttle::ExecutableBindingKind::Operand);
      EXPECT_EQ(bindings[index].index, index);
      EXPECT_EQ(bindings[index].slotOrdinal, index);
    }
    for (auto [index, slot] : llvm::enumerate(item.resultSlots)) {
      const auto &binding = bindings[3 + index];
      EXPECT_EQ(binding.kind,
                mlir::shuttle::ExecutableBindingKind::Result);
      EXPECT_EQ(binding.index, index);
      EXPECT_EQ(binding.slotOrdinal, slot);
    }
  }
}

TEST(CpuTransportContractTest, PublicLoaderRejectsStructuralBindingMutations) {
  std::string program =
      readText("jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir");
  auto baseline = buildBundle(program);
  ASSERT_TRUE(baseline);
  using Mutation = std::function<void(mlir::shuttle::InvocationSlotOp)>;
  std::array<Mutation, 4> mutations{
      [](mlir::shuttle::InvocationSlotOp slot) {
        slot.setBinding(mlir::shuttle::ExecutableBindingKind::None);
        slot.removeBindingIndexAttr();
      },
      [](mlir::shuttle::InvocationSlotOp slot) { slot.setBindingIndex(0); },
      [](mlir::shuttle::InvocationSlotOp slot) {
        slot.setBinding(mlir::shuttle::ExecutableBindingKind::Result);
        slot.setBindingIndex(0);
      },
      [](mlir::shuttle::InvocationSlotOp slot) {
        slot.setBinding(mlir::shuttle::ExecutableBindingKind::Operand);
        slot.setBindingIndex(2);
      }};
  std::array<int64_t, 4> ordinals{0, 1, 0, 2};

  for (auto [mutation, ordinal] : llvm::zip_equal(mutations, ordinals)) {
    auto cloned = mlir::cast<mlir::ModuleOp>(baseline->module->clone());
    auto abi = *cloned.getOps<mlir::shuttle::InvocationAbiOp>().begin();
    auto slots =
        abi.getBody().front().getOps<mlir::shuttle::InvocationSlotOp>();
    mutation(*std::next(slots.begin(), ordinal));
    refreshAbiRoots(cloned);
    EXPECT_TRUE(
        mlir::failed(mlir::shuttle::serializeCpuExecutableBundle(cloned)));
    cloned.erase();
  }
}

TEST(CpuTransportContractTest,
     WrongTypedProjectionIsCanonicalButNotTheFfiContract) {
  auto built = buildBundle(
      readText("jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir"));
  ASSERT_TRUE(built);
  auto abi = *built->module->getOps<mlir::shuttle::InvocationAbiOp>().begin();
  auto slots = abi.getBody().front().getOps<mlir::shuttle::InvocationSlotOp>();
  auto first = slots.begin();
  mlir::Builder builder(built->context.get());
  (*first).setTensorType(
      mlir::RankedTensorType::get({7, 13}, builder.getF32Type()));
  (*first).setRequiredBytes(7 * 13 * sizeof(float));
  (*first).setStridesAttr(builder.getDenseI64ArrayAttr({13 * 4, 4}));
  (*first).setAlignment(sizeof(float));
  refreshAbiRoots(*built->module);
  auto serialized = mlir::shuttle::serializeCpuExecutableBundle(*built->module);
  ASSERT_TRUE(mlir::succeeded(serialized));
  ASSERT_TRUE(mlir::shuttle::CpuExecutable::Load(*serialized).ok());
  llvm::SmallVector<uint8_t> golden =
      goldenBytes("cpu-forward-7x13-wrong-projection-transport.hex");
  EXPECT_EQ(*serialized, golden);
  EXPECT_EQ(mlir::shuttle::cpuExecutableBundleDigest(golden),
            "8c51fd1aedeb8b37958926b909071e607caa51a2f9157d4b37ce3f1f484ec570");
}

TEST(CpuTransportContractTest,
     LoadedExecutableIsImmutableAcrossConcurrentCalls) {
  auto executable = mlir::shuttle::CpuExecutable::Load(
      goldenBytes("cpu-forward-7x13-transport.hex"));
  ASSERT_TRUE(executable.ok()) << executable.status();
  Buffers first;
  Buffers second;
  populate(first, 0.25f);
  populate(second, 1.25f);
  auto firstExpected = reference(first);
  auto secondExpected = reference(second);

  auto firstCall = std::async(std::launch::async, [&] {
    auto views = first.views();
    return (*executable)->Execute(views);
  });
  auto secondCall = std::async(std::launch::async, [&] {
    auto views = second.views();
    return (*executable)->Execute(views);
  });
  EXPECT_TRUE(firstCall.get().ok());
  EXPECT_TRUE(secondCall.get().ok());
  EXPECT_EQ(first.output, firstExpected);
  EXPECT_EQ(second.output, secondExpected);
  EXPECT_NE(first.output, second.output);
}

} // namespace
