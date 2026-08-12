// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX composed BF16 fixture inventory; not native acceptance evidence
// Generator: regenerate-jax-bf16-composed-fixtures.py
// Generator source SHA-256: 98A579CBDA8B09ABB72F186B19E833CF44209C4C0EB6B653DD50086B16FDBBEA
// Source: jax.jit(natural JAX primal plus jax.vjp).lower(...).compiler_ir(StableHLO)
// Case ID: contract_map_eb4a28b4408cfb90
// Structural fields: {"features":184,"reduction":328,"rows":521,"scalar_map":"sigmoid_product"}
// Inputs: (521, 328):bfloat16, (328, 184):bfloat16, (184, 328):bfloat16, (521, 328):bfloat16
// Outputs: forward=(521, 328):bfloat16, dx=(521, 328):bfloat16, dw0=(328, 184):bfloat16, dw1=(184, 328):bfloat16
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw StableHLO SHA-256: 9EA30F51CA19BDE8C06AF0E06273BB5C49D3F4BC9B520B19F66A61C78F370D5A
// Raw normalized StableHLO SHA-256: A1B6D2E277D12E0BA66D8AA06C266C99B8976A60E29F672AB6BC4F985EE1A916
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: A1C6F4302CD35E0F551D17CD060AA90850E0B0B4FB4165D713E3A22E98A07699
// XLA hook-boundary normalized StableHLO SHA-256: 617A3CC4E636241E4459B255AFFAE229B0A0D1B3C6974D858578CD655F4A387E
module @jit_composed_primal_and_vjp attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<521x328xbf16>, %arg1: tensor<328x184xbf16>, %arg2: tensor<184x328xbf16>, %arg3: tensor<521x328xbf16>) -> (tensor<521x328xbf16> {jax.result_info = "result[0]"}, tensor<521x328xbf16> {jax.result_info = "result[1]"}, tensor<328x184xbf16> {jax.result_info = "result[2]"}, tensor<184x328xbf16> {jax.result_info = "result[3]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<521x328xbf16>, tensor<328x184xbf16>) -> tensor<521x184xf32>
    %1 = stablehlo.convert %0 : (tensor<521x184xf32>) -> tensor<521x184xbf16>
    %2 = stablehlo.convert %1 : (tensor<521x184xbf16>) -> tensor<521x184xf32>
    %3 = stablehlo.negate %2 : tensor<521x184xf32>
    %4 = stablehlo.exponential %3 : tensor<521x184xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<521x184xf32>
    %6 = stablehlo.add %5, %4 : tensor<521x184xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<521x184xf32>
    %8 = stablehlo.divide %7, %6 : tensor<521x184xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<521x184xf32>
    %10 = stablehlo.subtract %9, %8 : tensor<521x184xf32>
    %11 = stablehlo.multiply %8, %10 : tensor<521x184xf32>
    %12 = stablehlo.multiply %2, %8 : tensor<521x184xf32>
    %13 = stablehlo.convert %12 : (tensor<521x184xf32>) -> tensor<521x184xbf16>
    %14 = stablehlo.dot_general %13, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<521x184xbf16>, tensor<184x328xbf16>) -> tensor<521x328xf32>
    %15 = stablehlo.convert %14 : (tensor<521x328xf32>) -> tensor<521x328xbf16>
    %16 = stablehlo.convert %arg3 : (tensor<521x328xbf16>) -> tensor<521x328xf32>
    %17 = stablehlo.convert %16 : tensor<521x328xf32>
    %18 = stablehlo.convert %13 : (tensor<521x184xbf16>) -> tensor<521x184xf32>
    %19 = stablehlo.dot_general %17, %18, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<521x328xf32>, tensor<521x184xf32>) -> tensor<328x184xf32>
    %20 = stablehlo.transpose %19, dims = [1, 0] : (tensor<328x184xf32>) -> tensor<184x328xf32>
    %21 = stablehlo.convert %20 : (tensor<184x328xf32>) -> tensor<184x328xbf16>
    %22 = stablehlo.convert %16 : tensor<521x328xf32>
    %23 = stablehlo.convert %arg2 : (tensor<184x328xbf16>) -> tensor<184x328xf32>
    %24 = stablehlo.dot_general %22, %23, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<521x328xf32>, tensor<184x328xf32>) -> tensor<521x184xf32>
    %25 = stablehlo.convert %24 : (tensor<521x184xf32>) -> tensor<521x184xbf16>
    %26 = stablehlo.convert %25 : (tensor<521x184xbf16>) -> tensor<521x184xf32>
    %27 = stablehlo.multiply %2, %26 : tensor<521x184xf32>
    %28 = stablehlo.multiply %26, %8 : tensor<521x184xf32>
    %29 = stablehlo.multiply %27, %11 : tensor<521x184xf32>
    %30 = stablehlo.add %28, %29 : tensor<521x184xf32>
    %31 = stablehlo.convert %30 : (tensor<521x184xf32>) -> tensor<521x184xbf16>
    %32 = stablehlo.convert %31 : (tensor<521x184xbf16>) -> tensor<521x184xf32>
    %33 = stablehlo.convert %32 : tensor<521x184xf32>
    %34 = stablehlo.convert %arg0 : (tensor<521x328xbf16>) -> tensor<521x328xf32>
    %35 = stablehlo.dot_general %33, %34, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<521x184xf32>, tensor<521x328xf32>) -> tensor<184x328xf32>
    %36 = stablehlo.transpose %35, dims = [1, 0] : (tensor<184x328xf32>) -> tensor<328x184xf32>
    %37 = stablehlo.convert %36 : (tensor<328x184xf32>) -> tensor<328x184xbf16>
    %38 = stablehlo.convert %32 : tensor<521x184xf32>
    %39 = stablehlo.convert %arg1 : (tensor<328x184xbf16>) -> tensor<328x184xf32>
    %40 = stablehlo.dot_general %38, %39, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<521x184xf32>, tensor<328x184xf32>) -> tensor<521x328xf32>
    %41 = stablehlo.convert %40 : (tensor<521x328xf32>) -> tensor<521x328xbf16>
    return %15, %41, %37, %21 : tensor<521x328xbf16>, tensor<521x328xbf16>, tensor<328x184xbf16>, tensor<184x328xbf16>
  }
}
