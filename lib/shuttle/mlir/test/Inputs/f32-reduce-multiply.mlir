// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module {
  func.func @reduce_multiply(%input: tensor<2x4xf32>, %init: tensor<f32>) -> tensor<2xf32> {
    %result = "stablehlo.reduce"(%input, %init) ({
    ^bb0(%element: tensor<f32>, %accumulator: tensor<f32>):
      %product = stablehlo.multiply %element, %accumulator : tensor<f32>
      stablehlo.return %product : tensor<f32>
    }) {dimensions = array<i64: 1>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}
