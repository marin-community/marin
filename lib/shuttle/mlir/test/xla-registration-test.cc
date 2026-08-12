// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "shuttle/Transforms/Observer.h"
#include "shuttle/Transforms/XlaRegistration.h"
#include "stablehlo/dialect/Register.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace {

constexpr char kSourceOrderedOptions[] =
    R"json({"numerics":"source_ordered","pipeline_abi_version":3,"schema_version":1,"tuning":{"cluster_shape":[2,1,1],"materialization":"prefer_fusion","maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]}})json";
constexpr char kFastOptions[] =
    R"json({"numerics":"fast","pipeline_abi_version":3,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr char kProgram[] = R"mlir(
module {
  func.func @main(%arg0: tensor<7xf32>) -> tensor<7xf32> {
    %0 = stablehlo.tanh %arg0 : tensor<7xf32>
    return %0 : tensor<7xf32>
  }
}
)mlir";

mlir::OwningOpRef<mlir::ModuleOp> parseModule(mlir::MLIRContext &context) {
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::func::FuncDialect>();
  mlir::stablehlo::registerAllDialects(dialects);
  context.appendDialectRegistry(dialects);
  return mlir::parseSourceString<mlir::ModuleOp>(kProgram, &context);
}

TEST(ShuttleXlaOptionsTest, ParsesCanonicalPythonWireFormat) {
  absl::StatusOr<mlir::shuttle::ShuttlePipelineOptions> source =
      mlir::shuttle::parseShuttleXlaOptions(kSourceOrderedOptions);
  ASSERT_TRUE(source.ok()) << source.status();
  EXPECT_EQ(source->numerics, mlir::shuttle::NumericalPolicy::SourceOrdered);
  EXPECT_EQ(source->canonicalOptions, kSourceOrderedOptions);
  EXPECT_EQ(
      source->canonicalTuning,
      R"json({"cluster_shape":[2,1,1],"materialization":"prefer_fusion","maximum_candidates":16,"pipeline_stages":3,"tile_sizes":[64,128]})json");
  mlir::shuttle::ShuttlePipelineIdentity sourceIdentity =
      mlir::shuttle::shuttlePipelineIdentity(*source);
  EXPECT_EQ(sourceIdentity.policyDigest,
            "903f6abe997e8f055181375474580bba191298e8acf6d8dc42b8024d7b63675d");
  EXPECT_EQ(sourceIdentity.tuningDigest,
            "ae69cb474b1ddc91067687e7351ee27afe4e3b0814ae59e310a42bec5911326f");

  absl::StatusOr<mlir::shuttle::ShuttlePipelineOptions> fast =
      mlir::shuttle::parseShuttleXlaOptions(kFastOptions);
  ASSERT_TRUE(fast.ok()) << fast.status();
  EXPECT_EQ(fast->numerics, mlir::shuttle::NumericalPolicy::Fast);
  EXPECT_NE(mlir::shuttle::shuttlePipelineIdentity(*source).policyDigest,
            mlir::shuttle::shuttlePipelineIdentity(*fast).policyDigest);
}

TEST(ShuttleXlaOptionsTest, RejectsInvalidOrNoncanonicalWireFormats) {
  constexpr const char *invalidOptions[] = {
      R"json({"numerics":"source_ordered","numerics":"fast","pipeline_abi_version":3,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"numerics":"source_ordered","pipeline_abi_version":3,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]},"workload":"named"})json",
      R"json({"numerics":"source_ordered","pipeline_abi_version":2,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"numerics":"source_ordered","pipeline_abi_version":3,"schema_version":2,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"numerics":"source_ordered","pipeline_abi_version":3,"schema_version":1})json",
      R"json({"numerics":"source_ordered","pipeline_abi_version":3,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1}})json",
      R"json({"numerics":"source_ordered","pipeline_abi_version":3,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[],"warps":4}})json",
      R"json({"numerics":"source_ordered","pipeline_abi_version":3,"schema_version":1,"tuning":{"cluster_shape":[1,1,1,1],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"numerics":"source_ordered","pipeline_abi_version":3,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[1,1,1,1,1,1,1,1,1]}})json",
      R"json({"numerics":"source_ordered","pipeline_abi_version":3,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":0,"tile_sizes":[]}})json",
      R"json({"numerics":"source_ordered","pipeline_abi_version":3,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":true,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"numerics":"source_ordered","pipeline_abi_version":3,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":2147483648,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"numerics":"source_ordered","pipeline_abi_version":3,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"always_fuse","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({ "numerics":"source_ordered","pipeline_abi_version":3,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
      R"json({"schema_version":1,"numerics":"source_ordered","pipeline_abi_version":3,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json",
  };
  for (const char *serialized : invalidOptions) {
    absl::StatusOr<mlir::shuttle::ShuttlePipelineOptions> parsed =
        mlir::shuttle::parseShuttleXlaOptions(serialized);
    EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument)
        << serialized << "\n"
        << parsed.status();
  }
}

TEST(ShuttleXlaOptionsTest, RunsTheSharedPipelineFromSerializedOptions) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(
      mlir::shuttle::runShuttleXlaTransform(*module, kSourceOrderedOptions)
          .ok());

  std::string transformed;
  llvm::raw_string_ostream output(transformed);
  module->print(output);
  output.flush();
  EXPECT_EQ(transformed.find("shuttle."), std::string::npos);
  EXPECT_NE(transformed.find("stablehlo.tanh"), std::string::npos);
}

} // namespace
