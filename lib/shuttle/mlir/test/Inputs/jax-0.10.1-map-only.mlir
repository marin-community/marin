// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-fixtures.py (jax-0.10.1-map-only.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: transpose((a * b) + c)
// Inputs: (2, 3):f32, (2, 3):f32, (2, 3):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw StableHLO SHA-256: E371E5046686E5CF101C7851CF4798F9FF9667CDF8188306EE5B2BA654FDD320
// Normalized StableHLO SHA-256: 91F0D3806A5281F8B0E9CD242B9D035F42F393E7867CD91ACC3DD655FA02E796

module @jit_map_only attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>) -> (tensor<3x2xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x3xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<2x3xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %2 : tensor<3x2xf32>
  }
}
