// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module {
  func.func @main(%arg0: tensor<1x3xf32>) -> tensor<2x4x3xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2]
        : (tensor<1x3xf32>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
}
