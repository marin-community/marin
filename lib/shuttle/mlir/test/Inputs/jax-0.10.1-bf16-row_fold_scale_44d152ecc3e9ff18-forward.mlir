// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-row-fold-scale-fixtures.py
// Generator source SHA-256: 17B39945C103E03749B8BFD53C5BE61899ADD22C318098633E579D2C5AC4F6A0
// Source: jax.jit(ordinary JAX row Fold plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)
// Case ID: row_fold_scale_44d152ecc3e9ff18
// Structural fields: {"boundary":"forward","epsilon":1e-05,"features":4096,"rows":2048,"shape_role":"primary_shape_candidate"}
// Inputs: x=(2048, 4096):bfloat16, gamma=(4096,):bfloat16
// Outputs: y=(2048, 4096):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: 3D62EDA2A5BA167842045F702178762E7CAA75C0CAD3C6D43EAE37D42D47B908
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 037F050F3DDBA7CEDEE2DA2AECA35613CA412913AFA7C3703BDC8C47C8317244
module @jit_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2048x4096xbf16>, %arg1: tensor<4096xbf16>) -> (tensor<2048x4096xbf16> {jax.result_info = "result"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2048x4096xbf16>) -> tensor<2048x4096xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<2048x4096xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2048x4096xf32>, tensor<f32>) -> tensor<2048xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2048xf32>) -> tensor<2048x1xf32>
    %cst_0 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2048x1xf32>
    %5 = stablehlo.divide %3, %4 : tensor<2048x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2048x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<2048x1xf32>
    %8 = stablehlo.rsqrt %7 : tensor<2048x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<2048x1xf32>) -> tensor<2048x4096xf32>
    %10 = stablehlo.multiply %0, %9 : tensor<2048x4096xf32>
    %11 = stablehlo.convert %arg1 : (tensor<4096xbf16>) -> tensor<4096xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x4096xf32>) -> tensor<2048x4096xf32>
    %14 = stablehlo.multiply %10, %13 : tensor<2048x4096xf32>
    %15 = stablehlo.convert %14 : (tensor<2048x4096xf32>) -> tensor<2048x4096xbf16>
    return %15 : tensor<2048x4096xbf16>
  }
}
