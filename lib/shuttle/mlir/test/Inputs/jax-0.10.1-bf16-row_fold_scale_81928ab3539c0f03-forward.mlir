// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-row-fold-scale-fixtures.py
// Generator source SHA-256: 17B39945C103E03749B8BFD53C5BE61899ADD22C318098633E579D2C5AC4F6A0
// Source: jax.jit(ordinary JAX row Fold plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)
// Case ID: row_fold_scale_81928ab3539c0f03
// Structural fields: {"boundary":"forward","epsilon":1e-05,"features":13,"rows":7,"shape_role":"structural_shape_mutation"}
// Inputs: x=(7, 13):bfloat16, gamma=(13,):bfloat16
// Outputs: y=(7, 13):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: 5366D8FB253E770D4B84514C7951E6C7463EEBC8E71BA6353BFC485B6F5486C4
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 50386174720C4BB5B28E127E8CD5D6863F14CEC2C990072247DAF8B4EA1D56A2
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
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
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
