// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <fstream>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "shuttle/Runtime/CpuBytecode.h"
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
#include "llvm/ADT/StringExtras.h"
#include "gtest/gtest.h"

namespace {

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
            llvm::StringRef additionalAttribute = {}) {
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
  xla::CustomCall(&builder, mlir::shuttle::kCpuExecutableBundleFfiTarget,
                  {input, scale}, inputShape, backendConfig,
                  /*has_side_effect=*/false,
                  /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
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
