// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX composed BF16 fixture inventory; not native acceptance evidence
// Generator: regenerate-jax-bf16-composed-fixtures.py
// Generator source SHA-256: 98A579CBDA8B09ABB72F186B19E833CF44209C4C0EB6B653DD50086B16FDBBEA
// Source: jax.jit(natural JAX primal plus jax.vjp).lower(...).compiler_ir(StableHLO)
// Case ID: contract_map_9836cdbed389db24
// Structural fields: {"features":72,"reduction":104,"rows":43,"scalar_map":"sigmoid_product"}
// Inputs: (43, 104):bfloat16, (104, 72):bfloat16, (72, 104):bfloat16, (43, 104):bfloat16
// Outputs: forward=(43, 104):bfloat16, dx=(43, 104):bfloat16, dw0=(104, 72):bfloat16, dw1=(72, 104):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw StableHLO SHA-256: 8F9D9A5A41346971E0683D7AFDE80FB7A197A489F30A300D2950B1665C25AC96
// Raw normalized StableHLO SHA-256: E0DDB89478782C6A93FE75140605B9ABBE039CF305E71068DF067F3D576A5D34
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 07CC868CE58CB85F8D9F579B2B527CCD1DEBA6EB314A4FE9801F29A6B5CE304E
// XLA hook-boundary normalized StableHLO SHA-256: A221DCA8D47FA204F073123924B8194CE30DD354234D91F58E8055CFF9D74858
module @jit_composed_primal_and_vjp attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<43x104xbf16>, %arg1: tensor<104x72xbf16>, %arg2: tensor<72x104xbf16>, %arg3: tensor<43x104xbf16>) -> (tensor<43x104xbf16> {jax.result_info = "result[0]"}, tensor<43x104xbf16> {jax.result_info = "result[1]"}, tensor<104x72xbf16> {jax.result_info = "result[2]"}, tensor<72x104xbf16> {jax.result_info = "result[3]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<43x104xbf16>, tensor<104x72xbf16>) -> tensor<43x72xf32>
    %1 = stablehlo.convert %0 : (tensor<43x72xf32>) -> tensor<43x72xbf16>
    %2 = stablehlo.convert %1 : (tensor<43x72xbf16>) -> tensor<43x72xf32>
    %3 = stablehlo.negate %2 : tensor<43x72xf32>
    %4 = stablehlo.exponential %3 : tensor<43x72xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<43x72xf32>
    %6 = stablehlo.add %5, %4 : tensor<43x72xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<43x72xf32>
    %8 = stablehlo.divide %7, %6 : tensor<43x72xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<43x72xf32>
    %10 = stablehlo.subtract %9, %8 : tensor<43x72xf32>
    %11 = stablehlo.multiply %8, %10 : tensor<43x72xf32>
    %12 = stablehlo.multiply %2, %8 : tensor<43x72xf32>
    %13 = stablehlo.convert %12 : (tensor<43x72xf32>) -> tensor<43x72xbf16>
    %14 = stablehlo.dot_general %13, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<43x72xbf16>, tensor<72x104xbf16>) -> tensor<43x104xf32>
    %15 = stablehlo.convert %14 : (tensor<43x104xf32>) -> tensor<43x104xbf16>
    %16 = stablehlo.convert %arg3 : (tensor<43x104xbf16>) -> tensor<43x104xf32>
    %17 = stablehlo.convert %16 : tensor<43x104xf32>
    %18 = stablehlo.convert %13 : (tensor<43x72xbf16>) -> tensor<43x72xf32>
    %19 = stablehlo.dot_general %17, %18, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<43x104xf32>, tensor<43x72xf32>) -> tensor<104x72xf32>
    %20 = stablehlo.transpose %19, dims = [1, 0] : (tensor<104x72xf32>) -> tensor<72x104xf32>
    %21 = stablehlo.convert %20 : (tensor<72x104xf32>) -> tensor<72x104xbf16>
    %22 = stablehlo.convert %16 : tensor<43x104xf32>
    %23 = stablehlo.convert %arg2 : (tensor<72x104xbf16>) -> tensor<72x104xf32>
    %24 = stablehlo.dot_general %22, %23, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<43x104xf32>, tensor<72x104xf32>) -> tensor<43x72xf32>
    %25 = stablehlo.convert %24 : (tensor<43x72xf32>) -> tensor<43x72xbf16>
    %26 = stablehlo.convert %25 : (tensor<43x72xbf16>) -> tensor<43x72xf32>
    %27 = stablehlo.multiply %2, %26 : tensor<43x72xf32>
    %28 = stablehlo.multiply %26, %8 : tensor<43x72xf32>
    %29 = stablehlo.multiply %27, %11 : tensor<43x72xf32>
    %30 = stablehlo.add %28, %29 : tensor<43x72xf32>
    %31 = stablehlo.convert %30 : (tensor<43x72xf32>) -> tensor<43x72xbf16>
    %32 = stablehlo.convert %31 : (tensor<43x72xbf16>) -> tensor<43x72xf32>
    %33 = stablehlo.convert %32 : tensor<43x72xf32>
    %34 = stablehlo.convert %arg0 : (tensor<43x104xbf16>) -> tensor<43x104xf32>
    %35 = stablehlo.dot_general %33, %34, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<43x72xf32>, tensor<43x104xf32>) -> tensor<72x104xf32>
    %36 = stablehlo.transpose %35, dims = [1, 0] : (tensor<72x104xf32>) -> tensor<104x72xf32>
    %37 = stablehlo.convert %36 : (tensor<104x72xf32>) -> tensor<104x72xbf16>
    %38 = stablehlo.convert %32 : tensor<43x72xf32>
    %39 = stablehlo.convert %arg1 : (tensor<104x72xbf16>) -> tensor<104x72xf32>
    %40 = stablehlo.dot_general %38, %39, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<43x72xf32>, tensor<104x72xf32>) -> tensor<43x104xf32>
    %41 = stablehlo.convert %40 : (tensor<43x104xf32>) -> tensor<43x104xbf16>
    return %15, %41, %37, %21 : tensor<43x104xbf16>, tensor<43x104xbf16>, tensor<104x72xbf16>, tensor<72x104xbf16>
  }
}
