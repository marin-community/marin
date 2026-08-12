// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module {
  func.func @coverage_mutation(%lhs: tensor<2x3xf32>,
      %rhs: tensor<3x4xf32>, %alternate: tensor<2x4xf32>)
      -> tensor<2x4xf32> {
    %0 = stablehlo.dot_general %lhs, %rhs,
        contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
        : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %1 = stablehlo.tanh %0 : tensor<2x4xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<2x4xf32>
    return %2 : tensor<2x4xf32>
  }
}
