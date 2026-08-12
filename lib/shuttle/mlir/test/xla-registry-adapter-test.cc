// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "shuttle/Transforms/Observer.h"
#include "stablehlo/dialect/Register.h"
#include "xla/pjrt/stablehlo_module_transform.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace {

constexpr char kOptions[] =
    R"json({"numerics":"source_ordered","pipeline_abi_version":5,"schema_version":1,"tuning":{"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]}})json";
constexpr char kProgram[] = R"mlir(
module {
  func.func @main(%arg0: tensor<7xf32>) -> tensor<7xf32> {
    %0 = stablehlo.tanh %arg0 : tensor<7xf32>
    return %0 : tensor<7xf32>
  }
}
)mlir";
constexpr char kFailingProgram[] = R"mlir(
module {
  func.func @main(%arg0: tensor<7xf32>) -> tensor<7xf32> {
    return %arg0 : tensor<7xf32>
  ^bb1(%value: tensor<7xf32>):
    return %value : tensor<7xf32>
  }
}
)mlir";

class RecordingObserver final : public mlir::shuttle::ShuttlePipelineObserver {
public:
  struct Snapshot {
    std::vector<mlir::shuttle::ShuttlePipelinePhase> phases;
    std::vector<std::string> policyDigests;
  };

  void observe(const mlir::shuttle::ShuttlePipelineEvent &event) const final {
    std::lock_guard<std::mutex> lock(mutex);
    phases.push_back(event.phase());
    policyDigests.push_back(event.identity().policyDigest);
  }

  Snapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    return Snapshot{phases, policyDigests};
  }

private:
  mutable std::mutex mutex;
  mutable std::vector<mlir::shuttle::ShuttlePipelinePhase> phases;
  mutable std::vector<std::string> policyDigests;
};

mlir::OwningOpRef<mlir::ModuleOp> parseModule(mlir::MLIRContext &context,
                                              llvm::StringRef source) {
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  mlir::stablehlo::registerAllDialects(dialects);
  context.appendDialectRegistry(dialects);
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream output(result);
  module.print(output);
  output.flush();
  return result;
}

TEST(ShuttleXlaRegistryAdapterTest,
     RegistersAndRunsTheExactObservedPipelineTransactionally) {
  auto observer = std::make_shared<RecordingObserver>();
  auto subscription = mlir::shuttle::subscribeShuttlePipelineObserver(observer);
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, kProgram);
  ASSERT_TRUE(module);

  absl::Status status = xla::StablehloModuleTransformRegistry::Global().Run(
      *module, kOptions, "shuttle");
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(moduleText(*module).find("shuttle."), std::string::npos);
  RecordingObserver::Snapshot observed = observer->snapshot();
  ASSERT_EQ(observed.phases.size(), 3);
  EXPECT_EQ(observed.phases[0],
            mlir::shuttle::ShuttlePipelinePhase::AlgebraCoverage);
  EXPECT_EQ(observed.phases[1],
            mlir::shuttle::ShuttlePipelinePhase::LoweredCoverage);
  EXPECT_EQ(observed.phases[2],
            mlir::shuttle::ShuttlePipelinePhase::FinalErasure);
  EXPECT_EQ(observed.policyDigests[0], observed.policyDigests[2]);

  mlir::MLIRContext invalidContext;
  mlir::OwningOpRef<mlir::ModuleOp> invalid =
      parseModule(invalidContext, kProgram);
  ASSERT_TRUE(invalid);
  std::string invalidBefore = moduleText(*invalid);
  status = xla::StablehloModuleTransformRegistry::Global().Run(*invalid, "{}",
                                                               "shuttle");
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(moduleText(*invalid), invalidBefore);
  EXPECT_EQ(observer->snapshot().phases.size(), 3);

  mlir::MLIRContext failingContext;
  mlir::OwningOpRef<mlir::ModuleOp> failing =
      parseModule(failingContext, kFailingProgram);
  ASSERT_TRUE(failing);
  std::string failingBefore = moduleText(*failing);
  status = xla::StablehloModuleTransformRegistry::Global().Run(
      *failing, kOptions, "shuttle");
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(moduleText(*failing), failingBefore);
  observed = observer->snapshot();
  ASSERT_EQ(observed.phases.size(), 4);
  EXPECT_EQ(observed.phases.back(),
            mlir::shuttle::ShuttlePipelinePhase::Failure);
}

} // namespace
