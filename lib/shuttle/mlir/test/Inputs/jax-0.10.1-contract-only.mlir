// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-fixtures.py (jax-0.10.1-contract-only.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: a @ b
// Inputs: (3, 2):f32, (2, 4):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw StableHLO SHA-256: 883C8DB134527AE8052919D4DDDD387E6B693C79BC48FDF02A6E4F1931E74B02
// Normalized StableHLO SHA-256: 388748A381AD69EF162F92A647173E9EBF3285C29878B32CD4D6472245E69ABE

module @jit_contract attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3x2xf32>, %arg1: tensor<2x4xf32>) -> (tensor<3x4xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x2xf32>, tensor<2x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}
