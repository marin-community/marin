// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Test-only structural variant of the frozen 7x13 forward fixture. It changes
// only epsilon so the same typed-FFI target must instantiate distinct payloads.
// This file is not an ordinary-JAX fixture or acceptance evidence.
module @jit_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<7x13xbf16>, %arg1: tensor<13xbf16>) -> (tensor<7x13xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<7x13xbf16>) -> tensor<7x13xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<7x13xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<7x13xf32>, tensor<f32>) -> tensor<7xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<7xf32>) -> tensor<7x1xf32>
    %cst_0 = stablehlo.constant dense<1.300000e+01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<7x1xf32>
    %5 = stablehlo.divide %3, %4 : tensor<7x1xf32>
    %cst_1 = stablehlo.constant dense<2.500000e-1> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<7x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<7x1xf32>
    %8 = stablehlo.rsqrt %7 : tensor<7x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<7x1xf32>) -> tensor<7x13xf32>
    %10 = stablehlo.multiply %0, %9 : tensor<7x13xf32>
    %11 = stablehlo.convert %arg1 : (tensor<13xbf16>) -> tensor<13xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<13xf32>) -> tensor<1x13xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x13xf32>) -> tensor<7x13xf32>
    %14 = stablehlo.multiply %10, %13 : tensor<7x13xf32>
    %15 = stablehlo.convert %14 : (tensor<7x13xf32>) -> tensor<7x13xbf16>
    return %15 : tensor<7x13xbf16>
  }
}
