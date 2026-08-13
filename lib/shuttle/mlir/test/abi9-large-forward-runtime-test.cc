// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <vector>

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

constexpr int64_t kRows = 2048;
constexpr int64_t kFeatures = 4096;
constexpr int64_t kTaskCount = 19;
constexpr int64_t kSlotCount = 21;
constexpr int64_t kTemporaryBytes = 201416716;
constexpr int64_t kAggregateTaskPositions = 67129347;

struct BuiltBundle {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

std::unique_ptr<BuiltBundle> buildLargeForward() {
  std::string error;
  std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> runfiles(
      bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error));
  if (!runfiles) {
    return {};
  }
  std::ifstream input(runfiles->Rlocation(
      "shuttle_mlir/test/Inputs/"
      "jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir"));
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input || contents.str().empty()) {
    return {};
  }

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::shuttle::ShuttleDialect>();
  auto built = std::make_unique<BuiltBundle>();
  built->context = std::make_unique<mlir::MLIRContext>(registry);
  built->module = mlir::parseSourceString<mlir::ModuleOp>(
      contents.str(), built->context.get());
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

float balancedAdjacentSum(std::vector<float> leaves) {
  while (leaves.size() > 1) {
    size_t output = 0;
    for (size_t input = 0; input < leaves.size(); input += 2) {
      leaves[output++] = input + 1 == leaves.size()
                             ? leaves[input]
                             : static_cast<float>(leaves[input] +
                                                  leaves[input + 1]);
    }
    leaves.resize(output);
  }
  // cpu_bytecode_v2 merges the one StableHLO initializer after the data tree.
  return static_cast<float>(leaves.front() + 0.0f);
}

struct AlignedDelete {
  void operator()(uint16_t *value) const {
    ::operator delete[](value, std::align_val_t(64));
  }
};

using AlignedBf16 = std::unique_ptr<uint16_t[], AlignedDelete>;

AlignedBf16 allocateBf16(size_t elements) {
  return AlignedBf16(static_cast<uint16_t *>(::operator new[](
      elements * sizeof(uint16_t), std::align_val_t(64))));
}

void refreshFingerprints(mlir::ModuleOp module, bool codeChanged) {
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
  auto root = *module.getOps<mlir::shuttle::ExecutableBundleOp>().begin();
  root.setDeviceModuleFingerprint(device.getFingerprint());
  root.setInvocationAbiFingerprint(abi.getFingerprint());
  root.setFingerprint(mlir::shuttle::executableBundleFingerprint(root));
}

TEST(Abi9LargeForwardRuntimeTest,
     BuildsClosedV2TransportWithExactScheduleAndNoReuseCensus) {
  auto bundle = buildLargeForward();
  ASSERT_TRUE(bundle);
  auto device = *bundle->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
  auto abi = *bundle->module->getOps<mlir::shuttle::InvocationAbiOp>().begin();
  auto root = *bundle->module->getOps<mlir::shuttle::ExecutableBundleOp>().begin();

  EXPECT_EQ(device.getSchemaVersion(), 2);
  EXPECT_EQ(abi.getSchemaVersion(), 2);
  EXPECT_EQ(root.getSchemaVersion(), 1);
  EXPECT_EQ(
      mlir::shuttle::stringifyExecutableCodeFormat(device.getCodeFormat()),
      "cpu_bytecode_v2");
  ASSERT_GE(device.getCode().size(), 4U);
  constexpr std::array<int8_t, 4> kVersionTwoMagic{'S', 'B', 'C', 2};
  EXPECT_EQ(llvm::ArrayRef<int8_t>(device.getCode()).take_front(4),
            llvm::ArrayRef<int8_t>(kVersionTwoMagic));
  EXPECT_EQ(device.getPolicy(), mlir::shuttle::NumericalPolicy::SourceOrdered);
  EXPECT_EQ(std::distance(device.getBody().front()
                              .getOps<mlir::shuttle::DeviceEntryOp>()
                              .begin(),
                          device.getBody().front()
                              .getOps<mlir::shuttle::DeviceEntryOp>()
                              .end()),
            kTaskCount);
  EXPECT_EQ(std::distance(abi.getBody().front()
                              .getOps<mlir::shuttle::InvocationSlotOp>()
                              .begin(),
                          abi.getBody().front()
                              .getOps<mlir::shuttle::InvocationSlotOp>()
                              .end()),
            kSlotCount);

  int64_t temporaryBytes = 0;
  for (auto slot :
       abi.getBody().front().getOps<mlir::shuttle::InvocationSlotOp>()) {
    if (slot.getStorage() == mlir::shuttle::MaterializationStorage::Temporary) {
      temporaryBytes += slot.getRequiredBytes();
    }
  }
  EXPECT_EQ(temporaryBytes, kTemporaryBytes);
  EXPECT_EQ(mlir::shuttle::kMaximumCpuTaskElements, kRows * kFeatures);
  EXPECT_EQ(mlir::shuttle::kMaximumCpuSlotBytes, 32 * 1024 * 1024);
  EXPECT_EQ(mlir::shuttle::kMaximumCpuTemporaryBytes, 256 * 1024 * 1024);
  EXPECT_EQ(mlir::shuttle::kMaximumCpuAggregateTaskElements,
            kAggregateTaskPositions);
  EXPECT_EQ(mlir::shuttle::kMaximumCpuFoldScratchBytes, 16 * 1024);

  auto bytes = mlir::shuttle::serializeCpuExecutableBundle(*bundle->module);
  ASSERT_TRUE(mlir::succeeded(bytes));
  ASSERT_GE(bytes->size(), 12U);
  EXPECT_EQ((*bytes)[8], 1);
  EXPECT_EQ((*bytes)[9], 0);
  EXPECT_EQ((*bytes)[10], 0);
  EXPECT_EQ((*bytes)[11], 0);
  EXPECT_TRUE(mlir::shuttle::CpuExecutable::Load(*bytes).ok());
}

TEST(Abi9LargeForwardRuntimeTest,
     RawBuffersExecuteBalancedAdjacentFoldWithInitializerMergedLast) {
  auto bundle = buildLargeForward();
  ASSERT_TRUE(bundle);
  const size_t matrixElements = kRows * kFeatures;
  AlignedBf16 input = allocateBf16(matrixElements);
  AlignedBf16 scale = allocateBf16(kFeatures);
  AlignedBf16 output = allocateBf16(matrixElements);
  for (size_t index = 0; index < matrixElements; ++index) {
    input[index] = toBf16(-0.75f + static_cast<float>((index * 17) % 257) /
                                      128.0f);
  }
  for (size_t feature = 0; feature < kFeatures; ++feature) {
    scale[feature] =
        toBf16(0.5f + static_cast<float>((feature * 5) % 193) / 128.0f);
  }
  std::array<mlir::shuttle::CpuExternalBuffer, 3> buffers{{
      {0, llvm::MutableArrayRef<uint8_t>(
              reinterpret_cast<uint8_t *>(input.get()),
              matrixElements * sizeof(uint16_t))},
      {1, llvm::MutableArrayRef<uint8_t>(
              reinterpret_cast<uint8_t *>(scale.get()),
              kFeatures * sizeof(uint16_t))},
      {20, llvm::MutableArrayRef<uint8_t>(
               reinterpret_cast<uint8_t *>(output.get()),
               matrixElements * sizeof(uint16_t))},
  }};
  ASSERT_TRUE(mlir::succeeded(
      mlir::shuttle::executeCpuExecutableBundle(*bundle->module, buffers)));

  std::vector<float> leaves(kFeatures);
  for (int64_t row = 0; row < kRows; ++row) {
    for (int64_t feature = 0; feature < kFeatures; ++feature) {
      const float value = fromBf16(input[row * kFeatures + feature]);
      leaves[feature] = static_cast<float>(value * value);
    }
    const float sum = balancedAdjacentSum(leaves);
    const float inverse = 1.0f / std::sqrt(
                                     static_cast<float>(sum / kFeatures) +
                                     9.99999974E-6f);
    for (int64_t feature = 0; feature < kFeatures; ++feature) {
      const float value = fromBf16(input[row * kFeatures + feature]);
      const float gamma = fromBf16(scale[feature]);
      EXPECT_EQ(output[row * kFeatures + feature],
                toBf16(static_cast<float>(
                    static_cast<float>(value * inverse) * gamma)));
    }
  }
}

TEST(Abi9LargeForwardRuntimeTest,
     RejectsCrossedDeviceFormatAndUnknownFoldRealization) {
  auto schemaOneV2 = buildLargeForward();
  ASSERT_TRUE(schemaOneV2);
  auto device =
      *schemaOneV2->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
  device.setSchemaVersion(1);
  refreshFingerprints(*schemaOneV2->module, false);
  EXPECT_TRUE(mlir::failed(
      mlir::shuttle::serializeCpuExecutableBundle(*schemaOneV2->module)));

  auto schemaTwoV1 = buildLargeForward();
  ASSERT_TRUE(schemaTwoV1);
  device =
      *schemaTwoV1->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
  device.setCodeFormat(mlir::shuttle::ExecutableCodeFormat::CpuBytecodeV1);
  refreshFingerprints(*schemaTwoV1->module, false);
  EXPECT_TRUE(mlir::failed(
      mlir::shuttle::serializeCpuExecutableBundle(*schemaTwoV1->module)));

  auto unknownRealization = buildLargeForward();
  ASSERT_TRUE(unknownRealization);
  device = *unknownRealization->module
                ->getOps<mlir::shuttle::DeviceModuleOp>()
                .begin();
  llvm::SmallVector<int8_t> code(device.getCode());
  bool mutated = false;
  for (auto entry :
       device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>()) {
    size_t position = entry.getCodeOffset();
    if (static_cast<uint8_t>(code[position + 4]) != 1) {
      continue;
    }
    const uint8_t rank = code[position + 5];
    position += 6 + rank * 4;
    const uint8_t inputCount = code[position++];
    for (uint8_t input = 0; input < inputCount; ++input) {
      ++position;
      const uint8_t inputRank = code[position++];
      position += inputRank * 5;
    }
    position += 3; // output type, reduction axis, ScheduleReductionOrder.
    ASSERT_EQ(static_cast<uint8_t>(code[position]), 0);
    code[position] = static_cast<int8_t>(0xff);
    mutated = true;
    break;
  }
  ASSERT_TRUE(mutated);
  device.setCodeAttr(mlir::DenseI8ArrayAttr::get(
      unknownRealization->context.get(), code));
  refreshFingerprints(*unknownRealization->module, true);
  EXPECT_TRUE(mlir::failed(mlir::shuttle::serializeCpuExecutableBundle(
      *unknownRealization->module)));
}

} // namespace
