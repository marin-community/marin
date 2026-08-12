// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module @map_shape {
  func.func @main(%input: tensor<3xf32>, %matrix: tensor<2x2xf32>)
      -> (tensor<2x4x2xf32>, tensor<3xf32>, tensor<2x1x2xf32>) {
    %expanded = stablehlo.reshape %input : (tensor<3xf32>) -> tensor<1x3xf32>
    %restored = stablehlo.reshape %expanded : (tensor<1x3xf32>) -> tensor<3xf32>
    %reshaped = stablehlo.reshape %matrix : (tensor<2x2xf32>) -> tensor<2x1x2xf32>
    %broadcast = stablehlo.broadcast_in_dim %matrix, dims = [0, 2] : (tensor<2x2xf32>) -> tensor<2x4x2xf32>
    %inverse_root = stablehlo.rsqrt %broadcast : tensor<2x4x2xf32>
    return %inverse_root, %restored, %reshaped : tensor<2x4x2xf32>, tensor<3xf32>, tensor<2x1x2xf32>
  }
}
