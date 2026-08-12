// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module @two_row_folds {
  func.func @first(%input: tensor<2x4xf32>, %init: tensor<f32>)
      -> tensor<2xf32> {
    %result = "stablehlo.reduce"(%input, %init) ({
    ^bb0(%element: tensor<f32>, %accumulator: tensor<f32>):
      %sum = stablehlo.add %element, %accumulator : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {dimensions = array<i64: 1>} :
        (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }

  func.func @second(%input: tensor<7x13xf32>, %init: tensor<f32>)
      -> tensor<7xf32> {
    %result = "stablehlo.reduce"(%input, %init) ({
    ^bb0(%element: tensor<f32>, %accumulator: tensor<f32>):
      %sum = stablehlo.add %element, %accumulator : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {dimensions = array<i64: 1>} :
        (tensor<7x13xf32>, tensor<f32>) -> tensor<7xf32>
    return %result : tensor<7xf32>
  }
}
