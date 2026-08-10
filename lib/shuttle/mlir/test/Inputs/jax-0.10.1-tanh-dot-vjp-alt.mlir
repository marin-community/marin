// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-JAX export audit
// Generator: regenerate-jax-fixtures.py (jax-0.10.1-tanh-dot-vjp-alt.mlir)
// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")
// Expression: jax.vjp(reference_function, x, w0, w1)[1](output_cotangent)
// Inputs: (3, 2):f32, (2, 6):f32, (6, 4):f32, (3, 4):f32
// JAX: 0.10.1; jaxlib: 0.10.1; XLA: 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
// Raw StableHLO SHA-256: C60D86FDC5B0C692D75B80605BE32C9117E1DE37D4EDDC55D5BCDB30553C027F
// Normalized StableHLO SHA-256: EBB877A05E45B3A60A224ACE758BF841F7387AFDE5C64C2B9DD50DB5C48C9E54

module @jit_reference_vjp attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3x2xf32>, %arg1: tensor<2x6xf32>, %arg2: tensor<6x4xf32>, %arg3: tensor<3x4xf32>) -> (tensor<3x2xf32> {jax.result_info = "result[0]"}, tensor<2x6xf32> {jax.result_info = "result[1]"}, tensor<6x4xf32> {jax.result_info = "result[2]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x2xf32>, tensor<2x6xf32>) -> tensor<3x6xf32>
    %1 = stablehlo.tanh %0 : tensor<3x6xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x6xf32>
    %3 = stablehlo.subtract %2, %1 : tensor<3x6xf32>
    %4 = stablehlo.dot_general %arg3, %1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    %5 = stablehlo.transpose %4, dims = [1, 0] : (tensor<4x6xf32>) -> tensor<6x4xf32>
    %6 = stablehlo.dot_general %arg3, %arg2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<6x4xf32>) -> tensor<3x6xf32>
    %7 = stablehlo.multiply %6, %3 : tensor<3x6xf32>
    %8 = stablehlo.multiply %7, %1 : tensor<3x6xf32>
    %9 = stablehlo.add %7, %8 : tensor<3x6xf32>
    %10 = stablehlo.dot_general %9, %arg0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x6xf32>, tensor<3x2xf32>) -> tensor<6x2xf32>
    %11 = stablehlo.transpose %10, dims = [1, 0] : (tensor<6x2xf32>) -> tensor<2x6xf32>
    %12 = stablehlo.dot_general %9, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x6xf32>, tensor<2x6xf32>) -> tensor<3x2xf32>
    return %12, %11, %5 : tensor<3x2xf32>, tensor<2x6xf32>, tensor<6x4xf32>
  }
}
