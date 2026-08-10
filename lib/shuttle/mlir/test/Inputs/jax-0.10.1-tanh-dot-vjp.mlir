// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-fixtures.py (jax-0.10.1-tanh-dot-vjp.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: jax.vjp(reference_function, x, w0, w1)[1](output_cotangent)
// Inputs: (2, 3):f32, (3, 4):f32, (4, 5):f32, (2, 5):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw StableHLO SHA-256: 9A079C16D0BADFEF97282BB89FDB022B64DA52AC83F9C6CA4F171B479D3B5E1B
// Normalized StableHLO SHA-256: 2D557BD5D2F259A053335A6E004F9C5290D19713961E2C41787ED197ED042891
// XLA hook-boundary preprocessing: stablehlo-complex-math-expander
// XLA hook-boundary StableHLO SHA-256: B73249E4F90133826C587798D8DFB424A1756BAE6691F5AD82390DFE3094236A
// XLA hook-boundary normalized StableHLO SHA-256: D4DAD86C0C4ABF2F4A98BDD19879CBFB789C8D6CBA8B18FA56DECC4589A8DDB5

module @jit_reference_vjp attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4x5xf32>, %arg3: tensor<2x5xf32>) -> (tensor<2x3xf32> {jax.result_info = "result[0]"}, tensor<3x4xf32> {jax.result_info = "result[1]"}, tensor<4x5xf32> {jax.result_info = "result[2]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %1 = stablehlo.tanh %0 : tensor<2x4xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %3 = stablehlo.subtract %2, %1 : tensor<2x4xf32>
    %4 = stablehlo.dot_general %arg3, %1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x5xf32>, tensor<2x4xf32>) -> tensor<5x4xf32>
    %5 = stablehlo.transpose %4, dims = [1, 0] : (tensor<5x4xf32>) -> tensor<4x5xf32>
    %6 = stablehlo.dot_general %arg3, %arg2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x5xf32>, tensor<4x5xf32>) -> tensor<2x4xf32>
    %7 = stablehlo.multiply %6, %3 : tensor<2x4xf32>
    %8 = stablehlo.multiply %7, %1 : tensor<2x4xf32>
    %9 = stablehlo.add %7, %8 : tensor<2x4xf32>
    %10 = stablehlo.dot_general %9, %arg0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>
    %11 = stablehlo.transpose %10, dims = [1, 0] : (tensor<4x3xf32>) -> tensor<3x4xf32>
    %12 = stablehlo.dot_general %9, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<3x4xf32>) -> tensor<2x3xf32>
    return %12, %11, %5 : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4x5xf32>
  }
}
