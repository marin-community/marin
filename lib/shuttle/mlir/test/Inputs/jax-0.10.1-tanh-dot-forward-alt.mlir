// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-fixtures.py (jax-0.10.1-tanh-dot-forward-alt.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: reference_function(x, w0, w1) = tanh(x @ w0) @ w1
// Inputs: (3, 2):f32, (2, 6):f32, (6, 4):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw StableHLO SHA-256: B0901591C2991D0F11ABC947893EE0163E8FFF86A5290B010CF1173ED942166E
// Normalized StableHLO SHA-256: D50BC960BC8BE1CF54EC9D0B77A11BB0EC9D0A007495D3D315AB188B3C4D0D4E

module @jit_reference_function attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3x2xf32>, %arg1: tensor<2x6xf32>, %arg2: tensor<6x4xf32>) -> (tensor<3x4xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x2xf32>, tensor<2x6xf32>) -> tensor<3x6xf32>
    %1 = stablehlo.tanh %0 : tensor<3x6xf32>
    %2 = stablehlo.dot_general %1, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x6xf32>, tensor<6x4xf32>) -> tensor<3x4xf32>
    return %2 : tensor<3x4xf32>
  }
}
