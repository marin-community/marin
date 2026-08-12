// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module @mapped_singleton_broadcast {
  func.func @main(%rows: tensor<7x1xf32>, %columns: tensor<1x13xf32>,
      %permuted: tensor<1x7xf32>)
      -> (tensor<7x13xf32>, tensor<7x13xf32>, tensor<7x7xf32>) {
    %row_broadcast = stablehlo.broadcast_in_dim %rows, dims = [0, 1] :
        (tensor<7x1xf32>) -> tensor<7x13xf32>
    %column_broadcast = stablehlo.broadcast_in_dim %columns, dims = [0, 1] :
        (tensor<1x13xf32>) -> tensor<7x13xf32>
    %permuted_broadcast = stablehlo.broadcast_in_dim %permuted, dims = [1, 0] :
        (tensor<1x7xf32>) -> tensor<7x7xf32>
    return %row_broadcast, %column_broadcast, %permuted_broadcast :
        tensor<7x13xf32>, tensor<7x13xf32>, tensor<7x7xf32>
  }
}
