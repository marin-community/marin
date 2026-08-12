// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX composed BF16 fixture inventory; not native acceptance evidence
// Generator: regenerate-jax-bf16-composed-fixtures.py
// Generator source SHA-256: 98A579CBDA8B09ABB72F186B19E833CF44209C4C0EB6B653DD50086B16FDBBEA
// Source: jax.jit(natural JAX primal plus jax.vjp).lower(...).compiler_ir(StableHLO)
// Case ID: contract_map_79045ff9bdc7c783
// Structural fields: {"features":104,"reduction":168,"rows":131,"scalar_map":"tanh_product"}
// Inputs: (131, 168):bfloat16, (168, 104):bfloat16, (104, 168):bfloat16, (131, 168):bfloat16
// Outputs: forward=(131, 168):bfloat16, dx=(131, 168):bfloat16, dw0=(168, 104):bfloat16, dw1=(104, 168):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw StableHLO SHA-256: 55F00B4FF6A4168237863DE05B0ED45C7D81E54AD3E4B2917F2F42D591A5F6BC
// Raw normalized StableHLO SHA-256: B70E564634B02D69456293F093A9DFE0BA098545F5CD37A35EB1BF949C1E0C4D
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 5F92DC7FE79158D1E50440649FDD4AF3078FB460AA95D263F8EF5F51BF080650
// XLA hook-boundary normalized StableHLO SHA-256: 5F72B105E8DDD6CB87362A694F4ADE2A300AECCA692564A146D2B22ABF92EB80
module @jit_composed_primal_and_vjp attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<131x168xbf16>, %arg1: tensor<168x104xbf16>, %arg2: tensor<104x168xbf16>, %arg3: tensor<131x168xbf16>) -> (tensor<131x168xbf16> {jax.result_info = "result[0]"}, tensor<131x168xbf16> {jax.result_info = "result[1]"}, tensor<168x104xbf16> {jax.result_info = "result[2]"}, tensor<104x168xbf16> {jax.result_info = "result[3]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<131x168xbf16>, tensor<168x104xbf16>) -> tensor<131x104xf32>
    %1 = stablehlo.convert %0 : (tensor<131x104xf32>) -> tensor<131x104xbf16>
    %2 = stablehlo.convert %1 : (tensor<131x104xbf16>) -> tensor<131x104xf32>
    %3 = stablehlo.tanh %2 : tensor<131x104xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<131x104xf32>
    %5 = stablehlo.subtract %4, %3 : tensor<131x104xf32>
    %6 = stablehlo.multiply %2, %3 : tensor<131x104xf32>
    %7 = stablehlo.convert %6 : (tensor<131x104xf32>) -> tensor<131x104xbf16>
    %8 = stablehlo.dot_general %7, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<131x104xbf16>, tensor<104x168xbf16>) -> tensor<131x168xf32>
    %9 = stablehlo.convert %8 : (tensor<131x168xf32>) -> tensor<131x168xbf16>
    %10 = stablehlo.convert %arg3 : (tensor<131x168xbf16>) -> tensor<131x168xf32>
    %11 = stablehlo.convert %10 : tensor<131x168xf32>
    %12 = stablehlo.convert %7 : (tensor<131x104xbf16>) -> tensor<131x104xf32>
    %13 = stablehlo.dot_general %11, %12, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<131x168xf32>, tensor<131x104xf32>) -> tensor<168x104xf32>
    %14 = stablehlo.transpose %13, dims = [1, 0] : (tensor<168x104xf32>) -> tensor<104x168xf32>
    %15 = stablehlo.convert %14 : (tensor<104x168xf32>) -> tensor<104x168xbf16>
    %16 = stablehlo.convert %10 : tensor<131x168xf32>
    %17 = stablehlo.convert %arg2 : (tensor<104x168xbf16>) -> tensor<104x168xf32>
    %18 = stablehlo.dot_general %16, %17, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131x168xf32>, tensor<104x168xf32>) -> tensor<131x104xf32>
    %19 = stablehlo.convert %18 : (tensor<131x104xf32>) -> tensor<131x104xbf16>
    %20 = stablehlo.convert %19 : (tensor<131x104xbf16>) -> tensor<131x104xf32>
    %21 = stablehlo.multiply %2, %20 : tensor<131x104xf32>
    %22 = stablehlo.multiply %20, %3 : tensor<131x104xf32>
    %23 = stablehlo.multiply %21, %5 : tensor<131x104xf32>
    %24 = stablehlo.add %22, %23 : tensor<131x104xf32>
    %25 = stablehlo.multiply %23, %3 : tensor<131x104xf32>
    %26 = stablehlo.add %24, %25 : tensor<131x104xf32>
    %27 = stablehlo.convert %26 : (tensor<131x104xf32>) -> tensor<131x104xbf16>
    %28 = stablehlo.convert %27 : (tensor<131x104xbf16>) -> tensor<131x104xf32>
    %29 = stablehlo.convert %28 : tensor<131x104xf32>
    %30 = stablehlo.convert %arg0 : (tensor<131x168xbf16>) -> tensor<131x168xf32>
    %31 = stablehlo.dot_general %29, %30, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<131x104xf32>, tensor<131x168xf32>) -> tensor<104x168xf32>
    %32 = stablehlo.transpose %31, dims = [1, 0] : (tensor<104x168xf32>) -> tensor<168x104xf32>
    %33 = stablehlo.convert %32 : (tensor<168x104xf32>) -> tensor<168x104xbf16>
    %34 = stablehlo.convert %28 : tensor<131x104xf32>
    %35 = stablehlo.convert %arg1 : (tensor<168x104xbf16>) -> tensor<168x104xf32>
    %36 = stablehlo.dot_general %34, %35, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<131x104xf32>, tensor<168x104xf32>) -> tensor<131x168xf32>
    %37 = stablehlo.convert %36 : (tensor<131x168xf32>) -> tensor<131x168xbf16>
    return %9, %37, %33, %15 : tensor<131x168xbf16>, tensor<131x168xbf16>, tensor<168x104xbf16>, tensor<104x168xbf16>
  }
}
