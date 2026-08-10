// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic %s | FileCheck %s --check-prefix=ALGEBRA
// RUN: shuttle-opt --shuttle-stablehlo-source-ordered-pipeline %s | FileCheck %s --check-prefix=LOWER

module @unrelated_symbol_and_dimensions {
  func.func public @generic_graph(%arg0: tensor<7x13xf32>,
      %arg1: tensor<13x3xf32>, %arg2: tensor<3x11xf32>)
      -> tensor<7x11xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1,
        contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
        : (tensor<7x13xf32>, tensor<13x3xf32>) -> tensor<7x3xf32>
    %1 = stablehlo.tanh %0 : tensor<7x3xf32>
    %2 = stablehlo.dot_general %1, %arg2,
        contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
        : (tensor<7x3xf32>, tensor<3x11xf32>) -> tensor<7x11xf32>
    return %2 : tensor<7x11xf32>
  }
}

// ALGEBRA: "shuttle.region"
// ALGEBRA: "shuttle.contract"
// ALGEBRA: "shuttle.map"
// ALGEBRA: "shuttle.contract"
// ALGEBRA-NOT: "stablehlo.dot_general"(
// ALGEBRA-NOT: "stablehlo.tanh"(

// LOWER-NOT: shuttle.
// LOWER: stablehlo.dot_general
// LOWER: stablehlo.tanh
// LOWER: stablehlo.dot_general
