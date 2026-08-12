// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Artifact classification: ordinary_jax_fixture_only
// Evaluation oracle status: not_pinned
// Hardware evidence: none
// Generator: regenerate-jax-bf16-row-fold-scale-fixtures.py
// Generator source SHA-256: 17B39945C103E03749B8BFD53C5BE61899ADD22C318098633E579D2C5AC4F6A0
// Source: jax.jit(ordinary JAX row Fold plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)
// Case ID: row_fold_scale_81928ab3539c0f03
// Structural fields: {"boundary":"backward","epsilon":1e-05,"features":13,"rows":7,"shape_role":"structural_shape_mutation"}
// Inputs: x=(7, 13):bfloat16, gamma=(13,):bfloat16, dy=(7, 13):bfloat16
// Outputs: dx=(7, 13):bfloat16, dgamma=(13,):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; JAX revision: 619764c15117fbefc4ba13ab941871cb514c23f6
// XLA revision: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69; StableHLO current version: 1.17.0
// Raw StableHLO SHA-256: 3F4A271E961B55D9B5923C57BE0EBF56C01CA7C7EAA32F9E9B7B7B8554190F67
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 8416A9EEB38CD7EDD09CC470E50325080D730FAF85FC78CAB1ADC66873934F7C
module @jit_backward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<7x13xbf16>, %arg1: tensor<13xbf16>, %arg2: tensor<7x13xbf16>) -> (tensor<7x13xbf16> {jax.result_info = "result[0]"}, tensor<13xbf16> {jax.result_info = "result[1]"}) {
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
    %9 = stablehlo.divide %8, %7 : tensor<7x1xf32>
    %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<7x1xf32>
    %11 = stablehlo.multiply %10, %9 : tensor<7x1xf32>
    %12 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<7x1xf32>) -> tensor<7x13xf32>
    %13 = stablehlo.multiply %0, %12 : tensor<7x13xf32>
    %14 = stablehlo.convert %arg1 : (tensor<13xbf16>) -> tensor<13xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1] : (tensor<13xf32>) -> tensor<1x13xf32>
    %16 = stablehlo.convert %arg2 : (tensor<7x13xbf16>) -> tensor<7x13xf32>
    %17 = stablehlo.multiply %13, %16 : tensor<7x13xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.reduce(%17 init: %cst_3) applies stablehlo.add across dimensions = [0] : (tensor<7x13xf32>, tensor<f32>) -> tensor<13xf32>
    %19 = stablehlo.reshape %18 : (tensor<13xf32>) -> tensor<1x13xf32>
    %20 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<1x13xf32>) -> tensor<7x13xf32>
    %21 = stablehlo.multiply %16, %20 : tensor<7x13xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.reduce(%19 init: %cst_4) applies stablehlo.add across dimensions = [0] : (tensor<1x13xf32>, tensor<f32>) -> tensor<13xf32>
    %23 = stablehlo.convert %22 : (tensor<13xf32>) -> tensor<13xbf16>
    %24 = stablehlo.multiply %0, %21 : tensor<7x13xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.reduce(%24 init: %cst_5) applies stablehlo.add across dimensions = [1] : (tensor<7x13xf32>, tensor<f32>) -> tensor<7xf32>
    %26 = stablehlo.reshape %25 : (tensor<7xf32>) -> tensor<7x1xf32>
    %27 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<7x1xf32>) -> tensor<7x13xf32>
    %28 = stablehlo.multiply %21, %27 : tensor<7x13xf32>
    %29 = stablehlo.multiply %26, %11 : tensor<7x1xf32>
    %cst_6 = stablehlo.constant dense<1.300000e+01> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<7x1xf32>
    %31 = stablehlo.divide %29, %30 : tensor<7x1xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %32 = stablehlo.reduce(%31 init: %cst_7) applies stablehlo.add across dimensions = [1] : (tensor<7x1xf32>, tensor<f32>) -> tensor<7xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0] : (tensor<7xf32>) -> tensor<7x13xf32>
    %34 = stablehlo.multiply %0, %33 : tensor<7x13xf32>
    %35 = stablehlo.add %28, %34 : tensor<7x13xf32>
    %36 = stablehlo.multiply %33, %0 : tensor<7x13xf32>
    %37 = stablehlo.add %35, %36 : tensor<7x13xf32>
    %38 = stablehlo.convert %37 : (tensor<7x13xf32>) -> tensor<7x13xbf16>
    return %38, %23 : tensor<7x13xbf16>, tensor<13xbf16>
  }
}
