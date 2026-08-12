// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module @exact_tile {
  func.func @main(%input: tensor<1x256xf32>, %init: tensor<f32>) -> tensor<1xf32> {
    %mapped = stablehlo.multiply %input, %input : tensor<1x256xf32>
    %result = "stablehlo.reduce"(%mapped, %init) ({
    ^bb0(%element: tensor<f32>, %accumulator: tensor<f32>):
      %sum = stablehlo.add %element, %accumulator : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {dimensions = array<i64: 1>} : (tensor<1x256xf32>, tensor<f32>) -> tensor<1xf32>
    return %result : tensor<1xf32>
  }
}

// -----

module @max_extent {
  func.func @main(%input: tensor<1x9223372036854775807xf32>, %init: tensor<f32>) -> tensor<1xf32> {
    %mapped = stablehlo.multiply %input, %input : tensor<1x9223372036854775807xf32>
    %result = "stablehlo.reduce"(%mapped, %init) ({
    ^bb0(%element: tensor<f32>, %accumulator: tensor<f32>):
      %sum = stablehlo.add %element, %accumulator : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {dimensions = array<i64: 1>} : (tensor<1x9223372036854775807xf32>, tensor<f32>) -> tensor<1xf32>
    return %result : tensor<1xf32>
  }
}
