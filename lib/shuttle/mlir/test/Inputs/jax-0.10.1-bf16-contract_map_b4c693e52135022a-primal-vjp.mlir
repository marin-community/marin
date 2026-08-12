// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX composed BF16 fixture inventory; not native acceptance evidence
// Generator: regenerate-jax-bf16-composed-fixtures.py
// Generator source SHA-256: 98A579CBDA8B09ABB72F186B19E833CF44209C4C0EB6B653DD50086B16FDBBEA
// Source: jax.jit(natural JAX primal plus jax.vjp).lower(...).compiler_ir(StableHLO)
// Case ID: contract_map_b4c693e52135022a
// Structural fields: {"features":136,"reduction":232,"rows":269,"scalar_map":"cubic_mix"}
// Inputs: (269, 232):bfloat16, (232, 136):bfloat16, (136, 232):bfloat16, (269, 232):bfloat16
// Outputs: forward=(269, 232):bfloat16, dx=(269, 232):bfloat16, dw0=(232, 136):bfloat16, dw1=(136, 232):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw StableHLO SHA-256: 3A1B7CE3763604ECF1404E0548BB524D2015A1E17FCF28E5DF9A9163A0999D4F
// Raw normalized StableHLO SHA-256: F8F4D759F6C73B8E19EA42D0DBC790C1B3E521F1EF2BDAA8978F7572BAD06617
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 3A1B7CE3763604ECF1404E0548BB524D2015A1E17FCF28E5DF9A9163A0999D4F
// XLA hook-boundary normalized StableHLO SHA-256: F8F4D759F6C73B8E19EA42D0DBC790C1B3E521F1EF2BDAA8978F7572BAD06617
module @jit_composed_primal_and_vjp attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<269x232xbf16>, %arg1: tensor<232x136xbf16>, %arg2: tensor<136x232xbf16>, %arg3: tensor<269x232xbf16>) -> (tensor<269x232xbf16> {jax.result_info = "result[0]"}, tensor<269x232xbf16> {jax.result_info = "result[1]"}, tensor<232x136xbf16> {jax.result_info = "result[2]"}, tensor<136x232xbf16> {jax.result_info = "result[3]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<269x232xbf16>, tensor<232x136xbf16>) -> tensor<269x136xf32>
    %1 = stablehlo.convert %0 : (tensor<269x136xf32>) -> tensor<269x136xbf16>
    %2 = stablehlo.convert %1 : (tensor<269x136xbf16>) -> tensor<269x136xf32>
    %3 = stablehlo.multiply %2, %2 : tensor<269x136xf32>
    %4 = stablehlo.multiply %3, %2 : tensor<269x136xf32>
    %5 = stablehlo.add %2, %4 : tensor<269x136xf32>
    %6 = stablehlo.convert %5 : (tensor<269x136xf32>) -> tensor<269x136xbf16>
    %7 = stablehlo.dot_general %6, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<269x136xbf16>, tensor<136x232xbf16>) -> tensor<269x232xf32>
    %8 = stablehlo.convert %7 : (tensor<269x232xf32>) -> tensor<269x232xbf16>
    %9 = stablehlo.convert %arg3 : (tensor<269x232xbf16>) -> tensor<269x232xf32>
    %10 = stablehlo.convert %9 : tensor<269x232xf32>
    %11 = stablehlo.convert %6 : (tensor<269x136xbf16>) -> tensor<269x136xf32>
    %12 = stablehlo.dot_general %10, %11, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<269x232xf32>, tensor<269x136xf32>) -> tensor<232x136xf32>
    %13 = stablehlo.transpose %12, dims = [1, 0] : (tensor<232x136xf32>) -> tensor<136x232xf32>
    %14 = stablehlo.convert %13 : (tensor<136x232xf32>) -> tensor<136x232xbf16>
    %15 = stablehlo.convert %9 : tensor<269x232xf32>
    %16 = stablehlo.convert %arg2 : (tensor<136x232xbf16>) -> tensor<136x232xf32>
    %17 = stablehlo.dot_general %15, %16, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<269x232xf32>, tensor<136x232xf32>) -> tensor<269x136xf32>
    %18 = stablehlo.convert %17 : (tensor<269x136xf32>) -> tensor<269x136xbf16>
    %19 = stablehlo.convert %18 : (tensor<269x136xbf16>) -> tensor<269x136xf32>
    %20 = stablehlo.multiply %3, %19 : tensor<269x136xf32>
    %21 = stablehlo.add %19, %20 : tensor<269x136xf32>
    %22 = stablehlo.multiply %19, %2 : tensor<269x136xf32>
    %23 = stablehlo.multiply %2, %22 : tensor<269x136xf32>
    %24 = stablehlo.add %21, %23 : tensor<269x136xf32>
    %25 = stablehlo.multiply %22, %2 : tensor<269x136xf32>
    %26 = stablehlo.add %24, %25 : tensor<269x136xf32>
    %27 = stablehlo.convert %26 : (tensor<269x136xf32>) -> tensor<269x136xbf16>
    %28 = stablehlo.convert %27 : (tensor<269x136xbf16>) -> tensor<269x136xf32>
    %29 = stablehlo.convert %28 : tensor<269x136xf32>
    %30 = stablehlo.convert %arg0 : (tensor<269x232xbf16>) -> tensor<269x232xf32>
    %31 = stablehlo.dot_general %29, %30, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<269x136xf32>, tensor<269x232xf32>) -> tensor<136x232xf32>
    %32 = stablehlo.transpose %31, dims = [1, 0] : (tensor<136x232xf32>) -> tensor<232x136xf32>
    %33 = stablehlo.convert %32 : (tensor<232x136xf32>) -> tensor<232x136xbf16>
    %34 = stablehlo.convert %28 : tensor<269x136xf32>
    %35 = stablehlo.convert %arg1 : (tensor<232x136xbf16>) -> tensor<232x136xf32>
    %36 = stablehlo.dot_general %34, %35, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<269x136xf32>, tensor<232x136xf32>) -> tensor<269x232xf32>
    %37 = stablehlo.convert %36 : (tensor<269x232xf32>) -> tensor<269x232xbf16>
    return %8, %37, %33, %14 : tensor<269x232xbf16>, tensor<269x232xbf16>, tensor<232x136xbf16>, tensor<136x232xbf16>
  }
}
