// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
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

constexpr llvm::StringLiteral kProgram = R"mlir(
module @jit_forward {
  func.func public @main(%arg0: tensor<7x13xbf16>, %arg1: tensor<13xbf16>) -> tensor<7x13xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<7x13xbf16>) -> tensor<7x13xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<7x13xf32>
    %c0 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %c0) applies stablehlo.add across dimensions = [1] : (tensor<7x13xf32>, tensor<f32>) -> tensor<7xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<7xf32>) -> tensor<7x1xf32>
    %c13 = stablehlo.constant dense<13.0> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %c13, dims = [] : (tensor<f32>) -> tensor<7x1xf32>
    %5 = stablehlo.divide %3, %4 : tensor<7x1xf32>
    %ce = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %ce, dims = [] : (tensor<f32>) -> tensor<7x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<7x1xf32>
    %8 = stablehlo.rsqrt %7 : tensor<7x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<7x1xf32>) -> tensor<7x13xf32>
    %10 = stablehlo.multiply %0, %9 : tensor<7x13xf32>
    %11 = stablehlo.convert %arg1 : (tensor<13xbf16>) -> tensor<13xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<13xf32>) -> tensor<1x13xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x13xf32>) -> tensor<7x13xf32>
    %14 = stablehlo.multiply %10, %13 : tensor<7x13xf32>
    %15 = stablehlo.convert %14 : (tensor<7x13xf32>) -> tensor<7x13xbf16>
    return %15 : tensor<7x13xbf16>
  }
}
)mlir";

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

uint16_t apFloatToBf16(uint32_t bits) {
  llvm::APFloat converted(llvm::APFloat::IEEEsingle(), llvm::APInt(32, bits));
  bool losesInfo = false;
  converted.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                    &losesInfo);
  return converted.bitcastToAPInt().getZExtValue();
}

struct BuiltBundle {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

std::unique_ptr<BuiltBundle> compileBundle(llvm::StringRef program, bool fast) {
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
  mlir::PassManager algebra(built->context.get());
  algebra.addPass(mlir::shuttle::createAnnotateSourcePass());
  algebra.addPass(mlir::shuttle::createFormStructuralRegionsPass());
  algebra.addPass(mlir::shuttle::createConvertStablehloToAlgebraPass());
  if (mlir::failed(algebra.run(*built->module))) {
    return {};
  }
  if (fast) {
    built->module->walk([](mlir::shuttle::RegionOp region) {
      region.setPolicy(mlir::shuttle::NumericalPolicy::Fast);
    });
  }
  mlir::PassManager manager(built->context.get());
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

std::unique_ptr<BuiltBundle> buildBundle() {
  return compileBundle(kProgram, false);
}

std::unique_ptr<BuiltBundle> buildFixtureBundle(llvm::StringRef boundary,
                                                bool fast) {
  std::string error;
  std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> runfiles(
      bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error));
  if (!runfiles) {
    return {};
  }
  std::string path =
      runfiles->Rlocation(("shuttle_mlir/test/Inputs/"
                           "jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-" +
                           boundary + ".mlir")
                              .str());
  std::ifstream input(path);
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input || contents.str().empty()) {
    return {};
  }
  return compileBundle(contents.str(), fast);
}

struct ExternalBuffers {
  alignas(64) std::array<uint16_t, 7 * 13> input{};
  alignas(64) std::array<uint16_t, 13> scale{};
  alignas(64) std::array<uint16_t, 7 * 13> output{};

  std::array<mlir::shuttle::CpuExternalBuffer, 3> views() {
    return {
        mlir::shuttle::CpuExternalBuffer{
            0, llvm::MutableArrayRef<uint8_t>(
                   reinterpret_cast<uint8_t *>(input.data()), sizeof(input))},
        mlir::shuttle::CpuExternalBuffer{
            1, llvm::MutableArrayRef<uint8_t>(
                   reinterpret_cast<uint8_t *>(scale.data()), sizeof(scale))},
        mlir::shuttle::CpuExternalBuffer{
            20,
            llvm::MutableArrayRef<uint8_t>(
                reinterpret_cast<uint8_t *>(output.data()), sizeof(output))}};
  }
};

void populateInputs(ExternalBuffers &buffers) {
  for (size_t index = 0; index < buffers.input.size(); ++index) {
    buffers.input[index] =
        toBf16(0.25f + static_cast<float>((index * 7) % 29) / 16.0f);
  }
  for (size_t index = 0; index < buffers.scale.size(); ++index) {
    buffers.scale[index] = toBf16(0.75f + static_cast<float>(index) / 32.0f);
  }
}

std::array<uint16_t, 7 * 13>
independentReference(const ExternalBuffers &buffers) {
  std::array<uint16_t, 7 * 13> expected{};
  for (int64_t row = 0; row < 7; ++row) {
    float sum = 0.0f;
    for (int64_t feature = 0; feature < 13; ++feature) {
      const float value = fromBf16(buffers.input[row * 13 + feature]);
      sum = static_cast<float>(sum + static_cast<float>(value * value));
    }
    const float mean = static_cast<float>(sum / 13.0f);
    const float root = std::sqrt(static_cast<float>(mean + 9.99999974E-6f));
    const float inverse = static_cast<float>(1.0f / root);
    for (int64_t feature = 0; feature < 13; ++feature) {
      const float value = fromBf16(buffers.input[row * 13 + feature]);
      const float scale = fromBf16(buffers.scale[feature]);
      expected[row * 13 + feature] = toBf16(
          static_cast<float>(static_cast<float>(value * inverse) * scale));
    }
  }
  return expected;
}

uint16_t doubleToBf16(double value) {
  llvm::APFloat converted(value);
  bool losesInfo = false;
  converted.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                    &losesInfo);
  return converted.bitcastToAPInt().getZExtValue();
}

template <size_t Size>
void populateLinspace(std::array<uint16_t, Size> &values, float start,
                      float stop) {
  for (size_t index = 0; index < Size; ++index) {
    const float step =
        static_cast<float>(stop - start) / static_cast<float>(Size - 1);
    values[index] =
        toBf16(static_cast<float>(start + static_cast<float>(index) * step));
  }
}

struct VjpBuffers {
  alignas(64) std::array<uint16_t, 7 * 13> x{};
  alignas(64) std::array<uint16_t, 13> gamma{};
  alignas(64) std::array<uint16_t, 7 * 13> dy{};
  alignas(64) std::array<uint16_t, 7 * 13> y{};
  alignas(64) std::array<uint16_t, 7 * 13> dx{};
  alignas(64) std::array<uint16_t, 13> dgamma{};

  llvm::SmallVector<mlir::shuttle::CpuExternalBuffer, 6>
  views(llvm::StringRef boundary) {
    llvm::SmallVector<mlir::shuttle::CpuExternalBuffer, 6> result{
        {0, llvm::MutableArrayRef<uint8_t>(
                reinterpret_cast<uint8_t *>(x.data()), sizeof(x))},
        {1, llvm::MutableArrayRef<uint8_t>(
                reinterpret_cast<uint8_t *>(gamma.data()), sizeof(gamma))},
        {2, llvm::MutableArrayRef<uint8_t>(
                reinterpret_cast<uint8_t *>(dy.data()), sizeof(dy))}};
    if (boundary == "backward") {
      result.push_back({32, llvm::MutableArrayRef<uint8_t>(
                                reinterpret_cast<uint8_t *>(dgamma.data()),
                                sizeof(dgamma))});
      result.push_back(
          {50, llvm::MutableArrayRef<uint8_t>(
                   reinterpret_cast<uint8_t *>(dx.data()), sizeof(dx))});
    } else {
      result.push_back(
          {25, llvm::MutableArrayRef<uint8_t>(
                   reinterpret_cast<uint8_t *>(y.data()), sizeof(y))});
      result.push_back({35, llvm::MutableArrayRef<uint8_t>(
                                reinterpret_cast<uint8_t *>(dgamma.data()),
                                sizeof(dgamma))});
      result.push_back(
          {53, llvm::MutableArrayRef<uint8_t>(
                   reinterpret_cast<uint8_t *>(dx.data()), sizeof(dx))});
    }
    return result;
  }
};

void populateVjpInputs(VjpBuffers &buffers) {
  populateLinspace(buffers.x, -0.75f, 0.875f);
  populateLinspace(buffers.gamma, -0.625f, 1.0f);
  populateLinspace(buffers.dy, -0.5f, 1.125f);
}

struct VjpReference {
  std::array<uint16_t, 7 * 13> y{};
  std::array<uint16_t, 7 * 13> dx{};
  std::array<uint16_t, 13> dgamma{};
};

VjpReference independentVjpReference(const VjpBuffers &buffers) {
  VjpReference result;
  std::array<double, 7> inverse{};
  for (int64_t row = 0; row < 7; ++row) {
    double sumSquares = 0.0;
    for (int64_t feature = 0; feature < 13; ++feature) {
      const double x = fromBf16(buffers.x[row * 13 + feature]);
      sumSquares += x * x;
    }
    inverse[row] = 1.0 / std::sqrt(sumSquares / 13.0 + 1.0e-5);
    double rowCotangent = 0.0;
    for (int64_t feature = 0; feature < 13; ++feature) {
      const int64_t index = row * 13 + feature;
      const double x = fromBf16(buffers.x[index]);
      const double gamma = fromBf16(buffers.gamma[feature]);
      const double dy = fromBf16(buffers.dy[index]);
      rowCotangent += dy * x * gamma;
      result.y[index] = doubleToBf16(x * inverse[row] * gamma);
    }
    for (int64_t feature = 0; feature < 13; ++feature) {
      const int64_t index = row * 13 + feature;
      const double x = fromBf16(buffers.x[index]);
      const double gamma = fromBf16(buffers.gamma[feature]);
      const double dy = fromBf16(buffers.dy[index]);
      result.dx[index] = doubleToBf16(dy * gamma * inverse[row] -
                                      x * inverse[row] * inverse[row] *
                                          inverse[row] * rowCotangent / 13.0);
    }
  }
  for (int64_t feature = 0; feature < 13; ++feature) {
    double sum = 0.0;
    for (int64_t row = 0; row < 7; ++row) {
      const int64_t index = row * 13 + feature;
      sum += fromBf16(buffers.dy[index]) * fromBf16(buffers.x[index]) *
             inverse[row];
    }
    result.dgamma[feature] = doubleToBf16(sum);
  }
  return result;
}

void refreshClosedFingerprints(mlir::ModuleOp module, bool codeChanged) {
  auto device = *module.getOps<mlir::shuttle::DeviceModuleOp>().begin();
  if (codeChanged) {
    std::string digest = mlir::shuttle::executableCodeDigest(device.getCode());
    device.setCodeDigest(digest);
    for (auto entry :
         device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>()) {
      entry.setCodeDigest(digest);
    }
  }
  device.setFingerprint(mlir::shuttle::deviceModuleFingerprint(device));
  auto abi = *module.getOps<mlir::shuttle::InvocationAbiOp>().begin();
  abi.setFingerprint(mlir::shuttle::invocationAbiFingerprint(abi));
  auto bundle = *module.getOps<mlir::shuttle::ExecutableBundleOp>().begin();
  bundle.setDeviceModuleFingerprint(device.getFingerprint());
  bundle.setInvocationAbiFingerprint(abi.getFingerprint());
  bundle.setFingerprint(mlir::shuttle::executableBundleFingerprint(bundle));
}

TEST(CpuBytecodeRuntimeTest, ExecutesGeneratedBodyWithRawAbiBuffers) {
  auto module = buildBundle();
  ASSERT_TRUE(module);
  EXPECT_TRUE(module->module->getOps<mlir::func::FuncOp>().empty());
  EXPECT_TRUE(
      module->module->getOps<mlir::shuttle::MaterializationPlanOp>().empty());
  EXPECT_TRUE(module->module->getOps<mlir::shuttle::SchedulePlanOp>().empty());

  ExternalBuffers buffers;
  populateInputs(buffers);
  auto expected = independentReference(buffers);
  auto views = buffers.views();
  ASSERT_TRUE(mlir::succeeded(
      mlir::shuttle::executeCpuExecutableBundle(*module->module, views)));
  EXPECT_EQ(buffers.output, expected);
}

TEST(CpuBytecodeRuntimeTest, CanonicalTransportLoadsAndExecutesImmutableBody) {
  auto module = buildBundle();
  ASSERT_TRUE(module);
  auto bytes = mlir::shuttle::serializeCpuExecutableBundle(*module->module);
  ASSERT_TRUE(mlir::succeeded(bytes));
  EXPECT_EQ(mlir::shuttle::cpuExecutableBundleDigest(*bytes).size(), 64);
  auto executable = mlir::shuttle::CpuExecutable::Load(*bytes);
  ASSERT_TRUE(executable.ok()) << executable.status();
  ASSERT_EQ((*executable)->externalBindings().size(), 3);
  EXPECT_EQ((*executable)->externalBindings()[0].kind,
            mlir::shuttle::ExecutableBindingKind::Operand);
  EXPECT_EQ((*executable)->externalBindings()[0].index, 0);
  EXPECT_EQ((*executable)->externalBindings()[0].slotOrdinal, 0);
  EXPECT_EQ((*executable)->externalBindings()[1].kind,
            mlir::shuttle::ExecutableBindingKind::Operand);
  EXPECT_EQ((*executable)->externalBindings()[1].index, 1);
  EXPECT_EQ((*executable)->externalBindings()[1].slotOrdinal, 1);
  EXPECT_EQ((*executable)->externalBindings()[2].kind,
            mlir::shuttle::ExecutableBindingKind::Result);
  EXPECT_EQ((*executable)->externalBindings()[2].index, 0);
  EXPECT_EQ((*executable)->externalBindings()[2].slotOrdinal, 20);

  ExternalBuffers buffers;
  populateInputs(buffers);
  auto expected = independentReference(buffers);
  auto views = buffers.views();
  ASSERT_TRUE((*executable)->Execute(views).ok());
  EXPECT_EQ(buffers.output, expected);

  llvm::SmallVector<uint8_t> corrupted = *bytes;
  corrupted[corrupted.size() / 2] ^= 1;
  EXPECT_FALSE(mlir::shuttle::CpuExecutable::Load(corrupted).ok());
  corrupted = *bytes;
  corrupted.push_back(0);
  EXPECT_FALSE(mlir::shuttle::CpuExecutable::Load(corrupted).ok());

  auto selfConsistentInvalid = buildBundle();
  ASSERT_TRUE(selfConsistentInvalid);
  auto device =
      *selfConsistentInvalid->module->getOps<mlir::shuttle::DeviceModuleOp>()
           .begin();
  llvm::SmallVector<int8_t> code(device.getCode());
  code.front() ^= 1;
  device.setCodeAttr(
      mlir::DenseI8ArrayAttr::get(selfConsistentInvalid->context.get(), code));
  refreshClosedFingerprints(*selfConsistentInvalid->module, true);
  auto invalidBytes = mlir::shuttle::serializeCpuExecutableBundle(
      *selfConsistentInvalid->module);
  ASSERT_TRUE(mlir::succeeded(invalidBytes));
  EXPECT_FALSE(mlir::shuttle::CpuExecutable::Load(*invalidBytes).ok());
}

TEST(CpuBytecodeRuntimeTest, ExecutesGeneratedVjpBodiesWithRawAbiBuffers) {
  for (llvm::StringRef boundary :
       {llvm::StringRef("backward"), llvm::StringRef("composed")}) {
    for (bool fast : {false, true}) {
      auto bundle = buildFixtureBundle(boundary, fast);
      ASSERT_TRUE(bundle) << boundary.str() << " fast=" << fast;
      ASSERT_TRUE(bundle->module->getOps<mlir::func::FuncOp>().empty());
      ASSERT_TRUE(bundle->module->getOps<mlir::shuttle::MaterializationPlanOp>()
                      .empty());
      ASSERT_TRUE(
          bundle->module->getOps<mlir::shuttle::SchedulePlanOp>().empty());
      auto device =
          *bundle->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
      EXPECT_EQ(device.getPolicy(),
                fast ? mlir::shuttle::NumericalPolicy::Fast
                     : mlir::shuttle::NumericalPolicy::SourceOrdered);
      EXPECT_EQ(std::distance(device.getBody()
                                  .front()
                                  .getOps<mlir::shuttle::DeviceEntryOp>()
                                  .begin(),
                              device.getBody()
                                  .front()
                                  .getOps<mlir::shuttle::DeviceEntryOp>()
                                  .end()),
                boundary == "backward" ? 48 : 51);

      VjpBuffers buffers;
      populateVjpInputs(buffers);
      VjpReference expected = independentVjpReference(buffers);
      auto views = buffers.views(boundary);
      ASSERT_TRUE(mlir::succeeded(
          mlir::shuttle::executeCpuExecutableBundle(*bundle->module, views)))
          << boundary.str() << " fast=" << fast;
      EXPECT_EQ(buffers.dx, expected.dx);
      EXPECT_EQ(buffers.dgamma, expected.dgamma);
      if (boundary == "composed") {
        EXPECT_EQ(buffers.y, expected.y);
      }
    }
  }
}

TEST(CpuBytecodeRuntimeTest, ObservesMutatedVjpOutputBinding) {
  auto bundle = buildFixtureBundle("composed", false);
  ASSERT_TRUE(bundle);
  auto device =
      *bundle->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
  mlir::shuttle::DeviceEntryOp yProducer;
  mlir::shuttle::DeviceEntryOp dxProducer;
  for (auto entry :
       device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>()) {
    if (entry.getOutputBuffers() == llvm::ArrayRef<int64_t>{25}) {
      yProducer = entry;
    } else if (entry.getOutputBuffers() == llvm::ArrayRef<int64_t>{53}) {
      dxProducer = entry;
    }
  }
  ASSERT_TRUE(yProducer);
  ASSERT_TRUE(dxProducer);
  yProducer.setOutputBuffersAttr(
      mlir::DenseI64ArrayAttr::get(bundle->context.get(), {53}));
  dxProducer.setOutputBuffersAttr(
      mlir::DenseI64ArrayAttr::get(bundle->context.get(), {25}));
  refreshClosedFingerprints(*bundle->module, false);

  VjpBuffers buffers;
  populateVjpInputs(buffers);
  VjpReference expected = independentVjpReference(buffers);
  auto views = buffers.views("composed");
  ASSERT_TRUE(mlir::succeeded(
      mlir::shuttle::executeCpuExecutableBundle(*bundle->module, views)));
  EXPECT_EQ(buffers.y, expected.dx);
  EXPECT_EQ(buffers.dx, expected.y);
  EXPECT_EQ(buffers.dgamma, expected.dgamma);
}

TEST(CpuBytecodeRuntimeTest, Bf16RoundingMatchesApFloatEdgeCases) {
  constexpr std::array<uint32_t, 14> cases{
      0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7f800001u,
      0xff800001u, 0x7fc00000u, 0xffc00000u, 0x3f808000u, 0x3f818000u,
      0xbf808000u, 0xbf818000u, 0x00008000u, 0x80008000u};
  for (uint32_t bits : cases) {
    EXPECT_EQ(mlir::shuttle::roundF32ToBf16Rne(bits), apFloatToBf16(bits))
        << "f32 bits: " << bits;
  }
}

TEST(CpuBytecodeRuntimeTest, RejectsCorruptedClosedContracts) {
  auto baseline = buildBundle();
  ASSERT_TRUE(baseline);
  using Mutation = std::function<void(mlir::ModuleOp)>;
  std::array<Mutation, 15> mutations{
      [](mlir::ModuleOp module) {
        auto abi = *module.getOps<mlir::shuttle::InvocationAbiOp>().begin();
        (*abi.getBody()
              .front()
              .getOps<mlir::shuttle::InvocationSlotOp>()
              .begin())
            .setSourceBuffer(1);
        refreshClosedFingerprints(module, false);
      },
      [](mlir::ModuleOp module) {
        auto abi = *module.getOps<mlir::shuttle::InvocationAbiOp>().begin();
        auto slot = *abi.getBody()
                         .front()
                         .getOps<mlir::shuttle::InvocationSlotOp>()
                         .begin();
        slot.setStridesAttr(
            mlir::DenseI64ArrayAttr::get(module.getContext(), {2, 26}));
        refreshClosedFingerprints(module, false);
      },
      [](mlir::ModuleOp module) {
        auto abi = *module.getOps<mlir::shuttle::InvocationAbiOp>().begin();
        (*abi.getBody()
              .front()
              .getOps<mlir::shuttle::InvocationSlotOp>()
              .begin())
            .setOffset(2);
        refreshClosedFingerprints(module, false);
      },
      [](mlir::ModuleOp module) {
        auto abi = *module.getOps<mlir::shuttle::InvocationAbiOp>().begin();
        (*abi.getBody()
              .front()
              .getOps<mlir::shuttle::InvocationSlotOp>()
              .begin())
            .setAlignment(1);
        refreshClosedFingerprints(module, false);
      },
      [](mlir::ModuleOp module) {
        auto abi = *module.getOps<mlir::shuttle::InvocationAbiOp>().begin();
        (*abi.getBody()
              .front()
              .getOps<mlir::shuttle::InvocationSlotOp>()
              .begin())
            .setAddressSpace(mlir::shuttle::ExecutableAddressSpace::Device);
        refreshClosedFingerprints(module, false);
      },
      [](mlir::ModuleOp module) {
        auto abi = *module.getOps<mlir::shuttle::InvocationAbiOp>().begin();
        (*abi.getBody()
              .front()
              .getOps<mlir::shuttle::InvocationSlotOp>()
              .begin())
            .setAccess(mlir::shuttle::ExecutableAccess::Write);
        refreshClosedFingerprints(module, false);
      },
      [](mlir::ModuleOp module) {
        auto abi = *module.getOps<mlir::shuttle::InvocationAbiOp>().begin();
        auto slots =
            abi.getBody().front().getOps<mlir::shuttle::InvocationSlotOp>();
        (*std::next(slots.begin())).setAliasGroup(0);
        refreshClosedFingerprints(module, false);
      },
      [](mlir::ModuleOp module) {
        auto device = *module.getOps<mlir::shuttle::DeviceModuleOp>().begin();
        auto entries =
            device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>();
        (*std::next(entries.begin(), 4))
            .setDependenciesAttr(
                mlir::DenseI64ArrayAttr::get(module.getContext(), {}));
        refreshClosedFingerprints(module, false);
      },
      [](mlir::ModuleOp module) {
        auto device = *module.getOps<mlir::shuttle::DeviceModuleOp>().begin();
        (*device.getBody()
              .front()
              .getOps<mlir::shuttle::DeviceEntryOp>()
              .begin())
            .setPredication(mlir::shuttle::ExecutablePredication::None);
        refreshClosedFingerprints(module, false);
      },
      [](mlir::ModuleOp module) {
        auto device = *module.getOps<mlir::shuttle::DeviceModuleOp>().begin();
        llvm::SmallVector<int8_t> code(device.getCode());
        code.back() ^= 1;
        device.setCodeAttr(
            mlir::DenseI8ArrayAttr::get(module.getContext(), code));
      },
      [](mlir::ModuleOp module) {
        auto device = *module.getOps<mlir::shuttle::DeviceModuleOp>().begin();
        device.setCodeDigest(std::string(64, '0'));
      },
      [](mlir::ModuleOp module) {
        auto device = *module.getOps<mlir::shuttle::DeviceModuleOp>().begin();
        llvm::SmallVector<int8_t> code(device.getCode());
        auto entries =
            device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>();
        auto entry = *std::next(entries.begin(), 1);
        code[entry.getCodeOffset() + entry.getCodeLength() - 1] ^= 1;
        device.setCodeAttr(
            mlir::DenseI8ArrayAttr::get(module.getContext(), code));
        refreshClosedFingerprints(module, true);
      },
      [](mlir::ModuleOp module) {
        auto device = *module.getOps<mlir::shuttle::DeviceModuleOp>().begin();
        llvm::SmallVector<int8_t> code(device.getCode());
        auto entries =
            device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>();
        auto fold = *std::next(entries.begin(), 3);
        code[fold.getCodeOffset() + 17] = 1;
        device.setCodeAttr(
            mlir::DenseI8ArrayAttr::get(module.getContext(), code));
        refreshClosedFingerprints(module, true);
      },
      [](mlir::ModuleOp module) {
        auto device = *module.getOps<mlir::shuttle::DeviceModuleOp>().begin();
        (*device.getBody()
              .front()
              .getOps<mlir::shuttle::DeviceEntryOp>()
              .begin())
            .setCodeDigest(std::string(64, '0'));
      },
      [](mlir::ModuleOp module) {
        auto bundle =
            *module.getOps<mlir::shuttle::ExecutableBundleOp>().begin();
        bundle.setDeviceModuleFingerprint(std::string(64, '0'));
        bundle.setFingerprint(
            mlir::shuttle::executableBundleFingerprint(bundle));
      }};

  for (const Mutation &mutation : mutations) {
    auto cloned = mlir::cast<mlir::ModuleOp>(baseline->module->clone());
    mutation(cloned);
    ExternalBuffers buffers;
    populateInputs(buffers);
    auto views = buffers.views();
    EXPECT_TRUE(
        mlir::failed(mlir::shuttle::executeCpuExecutableBundle(cloned, views)));
    cloned.erase();
  }
}

TEST(CpuBytecodeRuntimeTest, RejectsInvalidDynamicBufferBindings) {
  auto bundle = buildBundle();
  ASSERT_TRUE(bundle);
  ExternalBuffers buffers;
  populateInputs(buffers);

  auto shortSpan = buffers.views();
  shortSpan[0].bytes = shortSpan[0].bytes.drop_back();
  EXPECT_TRUE(mlir::failed(
      mlir::shuttle::executeCpuExecutableBundle(*bundle->module, shortSpan)));

  alignas(64) std::array<uint8_t, sizeof(buffers.input) + 1> misaligned{};
  auto misalignedSpan = buffers.views();
  misalignedSpan[0].bytes = llvm::MutableArrayRef<uint8_t>(
      misaligned.data() + 1, sizeof(buffers.input));
  EXPECT_TRUE(mlir::failed(mlir::shuttle::executeCpuExecutableBundle(
      *bundle->module, misalignedSpan)));

  auto aliased = buffers.views();
  aliased[2].bytes = aliased[0].bytes;
  EXPECT_TRUE(mlir::failed(
      mlir::shuttle::executeCpuExecutableBundle(*bundle->module, aliased)));
}

TEST(CpuBytecodeRuntimeTest, RejectsUnstrippedModuleSidecars) {
  auto bundle = buildBundle();
  ASSERT_TRUE(bundle);
  mlir::OpBuilder builder(bundle->context.get());
  builder.setInsertionPointToStart(bundle->module->getBody());
  builder.create<mlir::func::FuncOp>(bundle->module->getLoc(), "source_sidecar",
                                     builder.getFunctionType({}, {}));
  ExternalBuffers buffers;
  populateInputs(buffers);
  auto views = buffers.views();
  EXPECT_TRUE(mlir::failed(
      mlir::shuttle::executeCpuExecutableBundle(*bundle->module, views)));
}

} // namespace
