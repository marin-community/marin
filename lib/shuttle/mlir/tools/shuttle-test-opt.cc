// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "test/TestPasses.h"

int main(int argc, char **argv) {
  mlir::shuttle::registerShuttlePasses();
  mlir::shuttle::registerShuttleStablehloPipelines();
  mlir::shuttle::test::registerMutationPasses();

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::scf::SCFDialect,
                  mlir::shuttle::ShuttleDialect, mlir::tensor::TensorDialect>();

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "Shuttle MLIR test pass driver\n", registry));
}
