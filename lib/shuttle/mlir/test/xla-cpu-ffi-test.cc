// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <future>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
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
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "gtest/gtest.h"

namespace {

constexpr llvm::StringLiteral kCpuExecutableBundleFfiTargetV2 =
    "shuttle.cpu.executable_bundle.v2";

std::string runfile(llvm::StringRef name) {
  std::string error;
  std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> runfiles(
      bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error));
  return runfiles
             ? runfiles->Rlocation(("shuttle_mlir/test/Inputs/" + name).str())
             : std::string();
}

llvm::SmallVector<uint8_t> goldenBytes(llvm::StringRef name) {
  std::ifstream input(runfile(name));
  std::string compact;
  char value;
  while (input.get(value)) {
    if (value != '\n') {
      compact.push_back(value);
    }
  }
  if (!input.eof() || compact.size() % 2 != 0) {
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

std::string readText(llvm::StringRef name) {
  std::ifstream input(runfile(name));
  std::ostringstream contents;
  contents << input.rdbuf();
  return input && !contents.str().empty() ? contents.str() : std::string();
}

llvm::SmallVector<uint8_t> fixtureBundle(llvm::StringRef boundary,
                                         bool fast = false) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::shuttle::ShuttleDialect>();
  mlir::MLIRContext context(registry);
  std::string name =
      ("jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-" + boundary + ".mlir")
          .str();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(readText(name), &context);
  if (!module) {
    return {};
  }
  mlir::PassManager manager(&context);
  manager.addPass(mlir::shuttle::createAnnotateSourcePass());
  manager.addPass(mlir::shuttle::createFormStructuralRegionsPass());
  manager.addPass(mlir::shuttle::createConvertStablehloToAlgebraPass());
  if (mlir::failed(manager.run(*module))) {
    return {};
  }
  if (fast) {
    module->walk([](mlir::shuttle::RegionOp region) {
      region.setPolicy(mlir::shuttle::NumericalPolicy::Fast);
    });
  }
  mlir::PassManager physical(&context);
  physical.addPass(mlir::shuttle::createPlanRowFoldMaterializationPass());
  physical.addPass(mlir::shuttle::createPlanSimt32RowFoldSchedulePass());
  physical.addPass(mlir::shuttle::createBuildCpuExecutableBundlePass());
  physical.addPass(mlir::shuttle::createVerifyCpuExecutableBundlePass());
  if (mlir::failed(physical.run(*module))) {
    return {};
  }
  for (mlir::Operation &operation :
       llvm::make_early_inc_range(module->getBody()->getOperations())) {
    if (!mlir::isa<mlir::shuttle::DeviceModuleOp,
                   mlir::shuttle::InvocationAbiOp,
                   mlir::shuttle::ExecutableBundleOp>(operation)) {
      operation.erase();
    }
  }
  auto serialized = mlir::shuttle::serializeCpuExecutableBundle(*module);
  return mlir::succeeded(serialized) ? std::move(*serialized)
                                     : llvm::SmallVector<uint8_t>{};
}

std::string escapedMlirString(llvm::ArrayRef<uint8_t> bytes) {
  static constexpr char digits[] = "0123456789ABCDEF";
  std::string escaped;
  escaped.reserve(bytes.size() * 3);
  for (uint8_t byte : bytes) {
    escaped.push_back('\\');
    escaped.push_back(digits[byte >> 4]);
    escaped.push_back(digits[byte & 15]);
  }
  return escaped;
}

absl::StatusOr<xla::LocalClient *> hostClient() {
  TF_ASSIGN_OR_RETURN(stream_executor::Platform * platform,
                      xla::PlatformUtil::GetPlatform("Host"));
  xla::LocalClientOptions options(platform, 1, 1, std::nullopt);
  return xla::ClientLibrary::GetOrCreateLocalClient(options);
}

uint16_t toBf16(float value) {
  llvm::APFloat converted(value);
  bool losesInfo = false;
  converted.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                    &losesInfo);
  return converted.bitcastToAPInt().getZExtValue();
}

struct CompiledCall {
  std::unique_ptr<xla::LocalExecutable> executable;
};

absl::StatusOr<CompiledCall>
compileCall(xla::LocalClient *client, llvm::ArrayRef<uint8_t> bundle,
            int64_t declaredSize, llvm::StringRef declaredDigest,
            int64_t transportSchema = 1,
            llvm::StringRef additionalAttribute = {}, bool aliasInput = false) {
  xla::Shape inputShape =
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::BF16, {7, 13}, {1, 0});
  xla::Shape scaleShape =
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::BF16, {13}, {0});
  xla::XlaBuilder builder("shuttle_cpu_ffi_forward_7x13");
  xla::XlaOp input = xla::Parameter(&builder, 0, inputShape, "input");
  xla::XlaOp scale = xla::Parameter(&builder, 1, scaleShape, "scale");
  std::string backendConfig =
      "{bundle_bytes = \"" + escapedMlirString(bundle) +
      "\", bundle_sha256 = \"" + declaredDigest.str() +
      "\", bundle_size = " + std::to_string(declaredSize) +
      " : i64, transport_schema_version = " + std::to_string(transportSchema) +
      " : i64" + additionalAttribute.str() + "}";
  std::vector<std::pair<xla::ShapeIndex, std::pair<int64_t, xla::ShapeIndex>>>
      aliases;
  if (aliasInput) {
    aliases.push_back({{}, {0, {}}});
  }
  xla::CustomCall(&builder, mlir::shuttle::kCpuExecutableBundleFfiTarget,
                  {input, scale}, inputShape, backendConfig,
                  /*has_side_effect=*/false,
                  /*output_operand_aliasing=*/aliases, /*literal=*/nullptr,
                  xla::CustomCallSchedule::SCHEDULE_NONE,
                  xla::CustomCallApiVersion::API_VERSION_TYPED_FFI);
  TF_ASSIGN_OR_RETURN(auto computation, builder.Build());
  xla::ExecutableBuildOptions buildOptions;
  buildOptions.set_device_ordinal(0);
  const xla::Shape *argumentLayouts[] = {&inputShape, &scaleShape};
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<xla::LocalExecutable>> executables,
      client->Compile(computation, argumentLayouts, buildOptions));
  if (executables.size() != 1) {
    return absl::InternalError("expected one Host executable");
  }
  return CompiledCall{std::move(executables.front())};
}

absl::StatusOr<CompiledCall> compileVjpCall(xla::LocalClient *client,
                                            llvm::ArrayRef<uint8_t> bundle,
                                            llvm::StringRef boundary,
                                            bool wrongResultOrder = false) {
  xla::Shape matrix =
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::BF16, {7, 13}, {1, 0});
  xla::Shape vector =
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::BF16, {13}, {0});
  std::vector<xla::Shape> results =
      boundary == "backward" ? std::vector<xla::Shape>{vector, matrix}
                             : std::vector<xla::Shape>{matrix, vector, matrix};
  if (wrongResultOrder) {
    std::swap(results[0], results[1]);
  }
  xla::Shape tuple = xla::ShapeUtil::MakeTupleShape(results);
  xla::XlaBuilder builder(("shuttle_cpu_ffi_" + boundary).str());
  xla::XlaOp input = xla::Parameter(&builder, 0, matrix, "x");
  xla::XlaOp scale = xla::Parameter(&builder, 1, vector, "gamma");
  xla::XlaOp cotangent = xla::Parameter(&builder, 2, matrix, "dy");
  std::string backendConfig =
      "{bundle_bytes = \"" + escapedMlirString(bundle) +
      "\", bundle_sha256 = \"" +
      mlir::shuttle::cpuExecutableBundleDigest(bundle) +
      "\", bundle_size = " + std::to_string(bundle.size()) +
      " : i64, transport_schema_version = 1 : i64}";
  xla::CustomCall(&builder, kCpuExecutableBundleFfiTargetV2.str(),
                  {input, scale, cotangent}, tuple, backendConfig,
                  /*has_side_effect=*/false,
                  /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
                  xla::CustomCallSchedule::SCHEDULE_NONE,
                  xla::CustomCallApiVersion::API_VERSION_TYPED_FFI);
  TF_ASSIGN_OR_RETURN(auto computation, builder.Build());
  xla::ExecutableBuildOptions buildOptions;
  buildOptions.set_device_ordinal(0);
  const xla::Shape *argumentLayouts[] = {&matrix, &vector, &matrix};
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<xla::LocalExecutable>> executables,
      client->Compile(computation, argumentLayouts, buildOptions));
  if (executables.size() != 1) {
    return absl::InternalError("expected one Host executable");
  }
  return CompiledCall{std::move(executables.front())};
}

xla::Literal inputLiteral(float shift = 0.25f) {
  xla::Literal literal = xla::Literal::CreateFromShape(
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::BF16, {7, 13}, {1, 0}));
  auto values = literal.data<uint16_t>();
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] =
        toBf16(shift + static_cast<float>((index * 7) % 29) / 16.0f);
  }
  return literal;
}

xla::Literal scaleLiteral() {
  xla::Literal literal = xla::Literal::CreateFromShape(
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::BF16, {13}, {0}));
  auto values = literal.data<uint16_t>();
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] = toBf16(0.75f + static_cast<float>(index) / 32.0f);
  }
  return literal;
}

xla::Literal linspaceMatrixLiteral(float start, float stop) {
  xla::Literal literal = xla::Literal::CreateFromShape(
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::BF16, {7, 13}, {1, 0}));
  auto values = literal.data<uint16_t>();
  float step = (stop - start) / static_cast<float>(values.size() - 1);
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] = toBf16(start + static_cast<float>(index) * step);
  }
  return literal;
}

xla::Literal linspaceVectorLiteral(float start, float stop) {
  xla::Literal literal = xla::Literal::CreateFromShape(
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::BF16, {13}, {0}));
  auto values = literal.data<uint16_t>();
  float step = (stop - start) / static_cast<float>(values.size() - 1);
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] = toBf16(start + static_cast<float>(index) * step);
  }
  return literal;
}

float fromBf16(uint16_t bits) {
  llvm::APFloat converted(llvm::APFloat::BFloat(), llvm::APInt(16, bits));
  bool losesInfo = false;
  converted.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return converted.convertToFloat();
}

uint16_t doubleToBf16(double value) {
  llvm::APFloat converted(value);
  bool losesInfo = false;
  converted.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                    &losesInfo);
  return converted.bitcastToAPInt().getZExtValue();
}

struct VjpReference {
  std::array<uint16_t, 7 * 13> y{};
  std::array<uint16_t, 7 * 13> dx{};
  std::array<uint16_t, 13> dgamma{};
};

VjpReference referenceVjp(const xla::Literal &x, const xla::Literal &gamma,
                          const xla::Literal &dy) {
  auto xValues = x.data<uint16_t>();
  auto gammaValues = gamma.data<uint16_t>();
  auto dyValues = dy.data<uint16_t>();
  VjpReference result;
  std::array<double, 7> inverse{};
  for (int64_t row = 0; row < 7; ++row) {
    double sumSquares = 0.0;
    for (int64_t feature = 0; feature < 13; ++feature) {
      double value = fromBf16(xValues[row * 13 + feature]);
      sumSquares += value * value;
    }
    inverse[row] = 1.0 / std::sqrt(sumSquares / 13.0 + 1.0e-5);
    double rowCotangent = 0.0;
    for (int64_t feature = 0; feature < 13; ++feature) {
      int64_t index = row * 13 + feature;
      double input = fromBf16(xValues[index]);
      double scale = fromBf16(gammaValues[feature]);
      double cotangent = fromBf16(dyValues[index]);
      rowCotangent += cotangent * input * scale;
      result.y[index] = doubleToBf16(input * inverse[row] * scale);
    }
    for (int64_t feature = 0; feature < 13; ++feature) {
      int64_t index = row * 13 + feature;
      double input = fromBf16(xValues[index]);
      double scale = fromBf16(gammaValues[feature]);
      double cotangent = fromBf16(dyValues[index]);
      result.dx[index] = doubleToBf16(cotangent * scale * inverse[row] -
                                      input * inverse[row] * inverse[row] *
                                          inverse[row] * rowCotangent / 13.0);
    }
  }
  for (int64_t feature = 0; feature < 13; ++feature) {
    double sum = 0.0;
    for (int64_t row = 0; row < 7; ++row) {
      int64_t index = row * 13 + feature;
      sum +=
          fromBf16(dyValues[index]) * fromBf16(xValues[index]) * inverse[row];
    }
    result.dgamma[feature] = doubleToBf16(sum);
  }
  return result;
}

absl::StatusOr<std::vector<std::vector<uint16_t>>>
runVjpCompiled(xla::LocalClient *client, xla::LocalExecutable *executable,
               xla::Literal &x, xla::Literal &gamma, xla::Literal &dy) {
  xla::ExecutableRunOptions runOptions;
  runOptions.set_allocator(client->backend().memory_allocator());
  TF_ASSIGN_OR_RETURN(auto xBuffer, client->LiteralToShapedBuffer(x, 0));
  TF_ASSIGN_OR_RETURN(auto gammaBuffer,
                      client->LiteralToShapedBuffer(gamma, 0));
  TF_ASSIGN_OR_RETURN(auto dyBuffer, client->LiteralToShapedBuffer(dy, 0));
  std::vector<const xla::ShapedBuffer *> arguments{&xBuffer, &gammaBuffer,
                                                   &dyBuffer};
  TF_ASSIGN_OR_RETURN(auto result, executable->Run(arguments, runOptions));
  TF_ASSIGN_OR_RETURN(xla::Literal actual,
                      client->ShapedBufferToLiteral(result));
  std::vector<xla::Literal> leaves = actual.DecomposeTuple();
  std::vector<std::vector<uint16_t>> values;
  for (xla::Literal &leaf : leaves) {
    auto data = leaf.data<uint16_t>();
    values.emplace_back(data.begin(), data.end());
  }
  return values;
}

absl::StatusOr<std::vector<uint16_t>>
runCompiled(xla::LocalClient *client, xla::LocalExecutable *executable,
            xla::Literal &input, xla::Literal &scale) {
  xla::ExecutableRunOptions runOptions;
  runOptions.set_allocator(client->backend().memory_allocator());
  TF_ASSIGN_OR_RETURN(auto inputBuffer,
                      client->LiteralToShapedBuffer(input, 0));
  TF_ASSIGN_OR_RETURN(auto scaleBuffer,
                      client->LiteralToShapedBuffer(scale, 0));
  std::vector<const xla::ShapedBuffer *> arguments{&inputBuffer, &scaleBuffer};
  TF_ASSIGN_OR_RETURN(auto result, executable->Run(arguments, runOptions));
  TF_ASSIGN_OR_RETURN(xla::Literal actual,
                      client->ShapedBufferToLiteral(result));
  auto values = actual.data<uint16_t>();
  return std::vector<uint16_t>(values.begin(), values.end());
}

absl::StatusOr<std::vector<uint16_t>>
runTypedCall(xla::LocalClient *client, llvm::ArrayRef<uint8_t> bytes,
             xla::Literal &input, xla::Literal &scale) {
  TF_ASSIGN_OR_RETURN(
      CompiledCall call,
      compileCall(client, bytes, bytes.size(),
                  mlir::shuttle::cpuExecutableBundleDigest(bytes)));
  return runCompiled(client, call.executable.get(), input, scale);
}

std::vector<uint16_t> runDirect(llvm::ArrayRef<uint8_t> bytes,
                                xla::Literal &input, xla::Literal &scale) {
  auto loaded = mlir::shuttle::CpuExecutable::Load(bytes);
  EXPECT_TRUE(loaded.ok()) << loaded.status();
  if (!loaded.ok()) {
    return {};
  }

  alignas(64) std::array<uint16_t, 7 * 13> expected{};
  std::array<mlir::shuttle::CpuExternalBuffer, 3> direct{
      {{0, llvm::MutableArrayRef<uint8_t>(
               reinterpret_cast<uint8_t *>(input.data<uint16_t>().data()),
               input.size_bytes())},
       {1, llvm::MutableArrayRef<uint8_t>(
               reinterpret_cast<uint8_t *>(scale.data<uint16_t>().data()),
               scale.size_bytes())},
       {20,
        llvm::MutableArrayRef<uint8_t>(
            reinterpret_cast<uint8_t *>(expected.data()), sizeof(expected))}}};
  EXPECT_TRUE((*loaded)->Execute(direct).ok());
  return std::vector<uint16_t>(expected.begin(), expected.end());
}

TEST(XlaCpuFfiTest, SameTargetExecutesTwoDistinctCanonicalBundlesOnHost) {
  llvm::SmallVector<uint8_t> baseline =
      goldenBytes("cpu-forward-7x13-transport.hex");
  llvm::SmallVector<uint8_t> epsilonQuarter =
      goldenBytes("cpu-forward-7x13-epsilon-quarter-transport.hex");
  ASSERT_FALSE(baseline.empty());
  ASSERT_FALSE(epsilonQuarter.empty());
  ASSERT_NE(baseline, epsilonQuarter);
  ASSERT_NE(mlir::shuttle::cpuExecutableBundleDigest(baseline),
            mlir::shuttle::cpuExecutableBundleDigest(epsilonQuarter));
  TF_ASSERT_OK_AND_ASSIGN(xla::LocalClient * client, hostClient());
  xla::Literal input = inputLiteral();
  xla::Literal scale = scaleLiteral();

  TF_ASSERT_OK_AND_ASSIGN(auto baselineActual,
                          runTypedCall(client, baseline, input, scale));
  TF_ASSERT_OK_AND_ASSIGN(auto epsilonQuarterActual,
                          runTypedCall(client, epsilonQuarter, input, scale));

  EXPECT_EQ(baselineActual, runDirect(baseline, input, scale));
  EXPECT_EQ(epsilonQuarterActual, runDirect(epsilonQuarter, input, scale));
  EXPECT_NE(baselineActual, epsilonQuarterActual);
}

TEST(XlaCpuFfiTest, InstantiateRejectsMismatchedBundleMetadata) {
  llvm::SmallVector<uint8_t> bytes =
      goldenBytes("cpu-forward-7x13-transport.hex");
  ASSERT_FALSE(bytes.empty());
  ASSERT_TRUE(xla::ffi::FindHandler(
                  mlir::shuttle::kCpuExecutableBundleFfiTarget, "Host")
                  .ok());
  TF_ASSERT_OK_AND_ASSIGN(xla::LocalClient * client, hostClient());
  EXPECT_FALSE(compileCall(client, bytes, bytes.size() + 1,
                           mlir::shuttle::cpuExecutableBundleDigest(bytes))
                   .ok());
  EXPECT_FALSE(
      compileCall(client, bytes, bytes.size(), std::string(64, '0')).ok());
  EXPECT_FALSE(compileCall(client, bytes, bytes.size(),
                           mlir::shuttle::cpuExecutableBundleDigest(bytes), 2)
                   .ok());
  EXPECT_FALSE(compileCall(client, bytes, bytes.size(),
                           mlir::shuttle::cpuExecutableBundleDigest(bytes), 1,
                           R"(, workload = "named")")
                   .ok());
}

TEST(XlaCpuFfiTest, ExecuteRejectsAliasedExternalBuffers) {
  llvm::SmallVector<uint8_t> bytes =
      goldenBytes("cpu-forward-7x13-transport.hex");
  ASSERT_FALSE(bytes.empty());
  TF_ASSERT_OK_AND_ASSIGN(xla::LocalClient * client, hostClient());
  TF_ASSERT_OK_AND_ASSIGN(
      CompiledCall call,
      compileCall(client, bytes, bytes.size(),
                  mlir::shuttle::cpuExecutableBundleDigest(bytes),
                  /*transportSchema=*/1, /*additionalAttribute=*/{},
                  /*aliasInput=*/true));
  xla::Literal input = inputLiteral();
  xla::Literal scale = scaleLiteral();
  auto executed = runCompiled(client, call.executable.get(), input, scale);
  ASSERT_FALSE(executed.ok());
  EXPECT_NE(executed.status().message().find(
                "typed FFI external buffers must not alias"),
            std::string::npos);
}

TEST(XlaCpuFfiTest, InstantiateRejectsWrongTypedCanonicalProjection) {
  llvm::SmallVector<uint8_t> bytes =
      goldenBytes("cpu-forward-7x13-wrong-projection-transport.hex");
  ASSERT_FALSE(bytes.empty());
  ASSERT_TRUE(mlir::shuttle::CpuExecutable::Load(bytes).ok());
  TF_ASSERT_OK_AND_ASSIGN(xla::LocalClient * client, hostClient());
  EXPECT_FALSE(compileCall(client, bytes, bytes.size(),
                           mlir::shuttle::cpuExecutableBundleDigest(bytes))
                   .ok());
}

TEST(XlaCpuFfiTest,
     V2TargetExecutesBackwardAndComposedTupleLeavesInBundleOrder) {
  EXPECT_EQ(llvm::StringRef(mlir::shuttle::kCpuExecutableBundleFfiTarget),
            kCpuExecutableBundleFfiTargetV2);
  TF_ASSERT_OK_AND_ASSIGN(xla::LocalClient * client, hostClient());
  for (llvm::StringRef boundary :
       {llvm::StringRef("backward"), llvm::StringRef("composed")}) {
    llvm::SmallVector<uint8_t> bytes = fixtureBundle(boundary);
    ASSERT_FALSE(bytes.empty()) << boundary.str();
    TF_ASSERT_OK_AND_ASSIGN(CompiledCall call,
                            compileVjpCall(client, bytes, boundary));
    xla::Literal x = linspaceMatrixLiteral(-0.75f, 0.875f);
    xla::Literal gamma = linspaceVectorLiteral(-0.625f, 1.0f);
    xla::Literal dy = linspaceMatrixLiteral(-0.5f, 1.125f);
    VjpReference expected = referenceVjp(x, gamma, dy);
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::vector<uint16_t>> actual,
        runVjpCompiled(client, call.executable.get(), x, gamma, dy));
    std::vector<std::vector<uint16_t>> expectedLeaves{
        {expected.dgamma.begin(), expected.dgamma.end()},
        {expected.dx.begin(), expected.dx.end()}};
    if (boundary == "composed") {
      expectedLeaves.insert(
          expectedLeaves.begin(),
          std::vector<uint16_t>(expected.y.begin(), expected.y.end()));
    }
    EXPECT_EQ(actual, expectedLeaves) << boundary.str();
  }
}

TEST(XlaCpuFfiTest, V2TargetExecutesIdentityFastPayloadsForAllBoundaries) {
  EXPECT_EQ(llvm::StringRef(mlir::shuttle::kCpuExecutableBundleFfiTarget),
            kCpuExecutableBundleFfiTargetV2);
  ASSERT_TRUE(xla::ffi::FindHandler(
                  mlir::shuttle::kCpuExecutableBundleFfiTarget, "Host")
                  .ok());
  TF_ASSERT_OK_AND_ASSIGN(xla::LocalClient * client, hostClient());
  for (llvm::StringRef boundary :
       {llvm::StringRef("forward"), llvm::StringRef("backward"),
        llvm::StringRef("composed")}) {
    llvm::SmallVector<uint8_t> sourceOrdered = fixtureBundle(boundary, false);
    llvm::SmallVector<uint8_t> fast = fixtureBundle(boundary, true);
    ASSERT_FALSE(sourceOrdered.empty()) << boundary.str();
    ASSERT_FALSE(fast.empty()) << boundary.str();
    ASSERT_NE(sourceOrdered, fast) << boundary.str();
    ASSERT_NE(mlir::shuttle::cpuExecutableBundleDigest(sourceOrdered),
              mlir::shuttle::cpuExecutableBundleDigest(fast))
        << boundary.str();
    ASSERT_TRUE(mlir::shuttle::CpuExecutable::Load(fast).ok())
        << boundary.str();

    if (boundary == "forward") {
      xla::Literal input = inputLiteral();
      xla::Literal scale = scaleLiteral();
      TF_ASSERT_OK_AND_ASSIGN(auto actual,
                              runTypedCall(client, fast, input, scale));
      EXPECT_EQ(actual, runDirect(fast, input, scale));
      continue;
    }

    TF_ASSERT_OK_AND_ASSIGN(CompiledCall call,
                            compileVjpCall(client, fast, boundary));
    xla::Literal x = linspaceMatrixLiteral(-0.75f, 0.875f);
    xla::Literal gamma = linspaceVectorLiteral(-0.625f, 1.0f);
    xla::Literal dy = linspaceMatrixLiteral(-0.5f, 1.125f);
    VjpReference expected = referenceVjp(x, gamma, dy);
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::vector<uint16_t>> actual,
        runVjpCompiled(client, call.executable.get(), x, gamma, dy));
    std::vector<std::vector<uint16_t>> expectedLeaves{
        {expected.dgamma.begin(), expected.dgamma.end()},
        {expected.dx.begin(), expected.dx.end()}};
    if (boundary == "composed") {
      expectedLeaves.insert(
          expectedLeaves.begin(),
          std::vector<uint16_t>(expected.y.begin(), expected.y.end()));
    }
    EXPECT_EQ(actual, expectedLeaves) << boundary.str();
  }
}

TEST(XlaCpuFfiTest, V2ExecuteRejectsSelfConsistentWrongResultTupleOrder) {
  TF_ASSERT_OK_AND_ASSIGN(xla::LocalClient * client, hostClient());
  llvm::SmallVector<uint8_t> bytes = fixtureBundle("backward");
  ASSERT_FALSE(bytes.empty());
  TF_ASSERT_OK_AND_ASSIGN(
      CompiledCall call,
      compileVjpCall(client, bytes, "backward", /*wrongResultOrder=*/true));
  xla::Literal x = linspaceMatrixLiteral(-0.75f, 0.875f);
  xla::Literal gamma = linspaceVectorLiteral(-0.625f, 1.0f);
  xla::Literal dy = linspaceMatrixLiteral(-0.5f, 1.125f);
  auto executed = runVjpCompiled(client, call.executable.get(), x, gamma, dy);
  ASSERT_FALSE(executed.ok());
  EXPECT_NE(executed.status().message().find(
                "external bindings do not match the typed FFI contract"),
            std::string::npos);
}

TEST(XlaCpuFfiTest, SharedV2ComposedExecutableRunsConcurrently) {
  TF_ASSERT_OK_AND_ASSIGN(xla::LocalClient * client, hostClient());
  llvm::SmallVector<uint8_t> bytes = fixtureBundle("composed", true);
  ASSERT_FALSE(bytes.empty());
  TF_ASSERT_OK_AND_ASSIGN(CompiledCall call,
                          compileVjpCall(client, bytes, "composed"));
  xla::Literal firstX = linspaceMatrixLiteral(-0.75f, 0.875f);
  xla::Literal firstGamma = linspaceVectorLiteral(-0.625f, 1.0f);
  xla::Literal firstDy = linspaceMatrixLiteral(-0.5f, 1.125f);
  xla::Literal secondX = linspaceMatrixLiteral(0.25f, 1.875f);
  xla::Literal secondGamma = linspaceVectorLiteral(-1.0f, 0.625f);
  xla::Literal secondDy = linspaceMatrixLiteral(-1.125f, 0.5f);

  auto first = std::async(std::launch::async, [&] {
    return runVjpCompiled(client, call.executable.get(), firstX, firstGamma,
                          firstDy);
  });
  auto second = std::async(std::launch::async, [&] {
    return runVjpCompiled(client, call.executable.get(), secondX, secondGamma,
                          secondDy);
  });
  auto firstResult = first.get();
  auto secondResult = second.get();
  ASSERT_TRUE(firstResult.ok()) << firstResult.status();
  ASSERT_TRUE(secondResult.ok()) << secondResult.status();
  EXPECT_NE(*firstResult, *secondResult);
}

TEST(XlaCpuFfiTest, SharedCompiledExecutableRunsConcurrently) {
  llvm::SmallVector<uint8_t> bytes =
      goldenBytes("cpu-forward-7x13-transport.hex");
  ASSERT_FALSE(bytes.empty());
  TF_ASSERT_OK_AND_ASSIGN(xla::LocalClient * client, hostClient());
  TF_ASSERT_OK_AND_ASSIGN(
      CompiledCall call,
      compileCall(client, bytes, bytes.size(),
                  mlir::shuttle::cpuExecutableBundleDigest(bytes)));
  xla::Literal firstInput = inputLiteral();
  xla::Literal secondInput = inputLiteral(1.25f);
  xla::Literal firstScale = scaleLiteral();
  xla::Literal secondScale = scaleLiteral();

  auto first = std::async(std::launch::async, [&] {
    return runCompiled(client, call.executable.get(), firstInput, firstScale);
  });
  auto second = std::async(std::launch::async, [&] {
    return runCompiled(client, call.executable.get(), secondInput, secondScale);
  });
  auto firstResult = first.get();
  auto secondResult = second.get();
  ASSERT_TRUE(firstResult.ok()) << firstResult.status();
  ASSERT_TRUE(secondResult.ok()) << secondResult.status();
  EXPECT_NE(*firstResult, *secondResult);
  EXPECT_EQ(*firstResult, runDirect(bytes, firstInput, firstScale));
  EXPECT_EQ(*secondResult, runDirect(bytes, secondInput, secondScale));
}

} // namespace
