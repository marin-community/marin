// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::shuttle::registerShuttlePasses();
  mlir::stablehlo::registerPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::shuttle::ShuttleDialect>();

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "Shuttle MLIR pass driver\n", registry));
}
