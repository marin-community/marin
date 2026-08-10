// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-fixtures.py (jax-0.10.1-tanh-dot-forward.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: reference_function(x, w0, w1) = tanh(x @ w0) @ w1
// Inputs: (2, 3):f32, (3, 4):f32, (4, 5):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw StableHLO SHA-256: 1CE4E09F216055D2CA682379A21984E2D2EF34560C6DDFD700A8306ABCE45F6A
// Normalized StableHLO SHA-256: 01539D7D3FEBF0814CCF67320863712FA19E0425BDDA9A716B4716FBE2EFC944
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: 1CE4E09F216055D2CA682379A21984E2D2EF34560C6DDFD700A8306ABCE45F6A
// XLA hook-boundary normalized StableHLO SHA-256: 01539D7D3FEBF0814CCF67320863712FA19E0425BDDA9A716B4716FBE2EFC944

module @jit_reference_function attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4x5xf32>) -> (tensor<2x5xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %1 = stablehlo.tanh %0 : tensor<2x4xf32>
    %2 = stablehlo.dot_general %1, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<4x5xf32>) -> tensor<2x5xf32>
    return %2 : tensor<2x5xf32>
  }
}
