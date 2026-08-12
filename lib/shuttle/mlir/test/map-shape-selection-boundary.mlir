// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra %s | FileCheck %s
// RUN: shuttle-test-opt --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra %s | not grep shuttle.map

module {
  func.func @rank_three_singleton_expansion(%input: tensor<7x1x13xf32>)
      -> tensor<7x5x13xf32> {
    %result = stablehlo.broadcast_in_dim %input, dims = [0, 1, 2] :
        (tensor<7x1x13xf32>) -> tensor<7x5x13xf32>
    return %result : tensor<7x5x13xf32>
  }
}

// -----

module {
  func.func @two_mapped_singleton_expansions(%input: tensor<1x1xf32>)
      -> tensor<7x13xf32> {
    %result = stablehlo.broadcast_in_dim %input, dims = [0, 1] :
        (tensor<1x1xf32>) -> tensor<7x13xf32>
    return %result : tensor<7x13xf32>
  }
}

// -----

module {
  func.func @dynamic_mapped_singleton_expansion(
      %input: tensor<?x1xf32, #stablehlo.bounds<7, ?>>)
      -> tensor<?x13xf32, #stablehlo.bounds<7, ?>> {
    %result = stablehlo.broadcast_in_dim %input, dims = [0, 1] :
        (tensor<?x1xf32, #stablehlo.bounds<7, ?>>) ->
        tensor<?x13xf32, #stablehlo.bounds<7, ?>>
    return %result : tensor<?x13xf32, #stablehlo.bounds<7, ?>>
  }
}

// -----

module {
  func.func @zero_extent_mapped_singleton_expansion(
      %input: tensor<7x1xf32>) -> tensor<7x0xf32> {
    %result = stablehlo.broadcast_in_dim %input, dims = [0, 1] :
        (tensor<7x1xf32>) -> tensor<7x0xf32>
    return %result : tensor<7x0xf32>
  }
}

// -----

module {
  func.func @flattening_reshape(%input: tensor<2x3xf32>) -> tensor<6xf32> {
    %result = stablehlo.reshape %input : (tensor<2x3xf32>) -> tensor<6xf32>
    return %result : tensor<6xf32>
  }
}

// -----

module {
  func.func @rsqrt_accuracy(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %result = "stablehlo.rsqrt"(%input) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00, rtol = 1.000000e+00, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }
}

// -----

module {
  func.func @broadcast_unknown_attribute(%input: tensor<3xf32>)
      -> tensor<2x3xf32> {
    %result = "stablehlo.broadcast_in_dim"(%input) {broadcast_dimensions = array<i64: 1>, shuttle.test_semantic = 7 : i64} : (tensor<3xf32>) -> tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }
}

// -----

module {
  func.func @reshape_unknown_attribute(%input: tensor<3xf32>)
      -> tensor<1x3xf32> {
    %result = "stablehlo.reshape"(%input) {shuttle.test_semantic = 7 : i64} : (tensor<3xf32>) -> tensor<1x3xf32>
    return %result : tensor<1x3xf32>
  }
}

// -----

module {
  func.func @rsqrt_unknown_attribute(%input: tensor<3xf32>)
      -> tensor<3xf32> {
    %result = "stablehlo.rsqrt"(%input) {shuttle.test_semantic = 7 : i64} : (tensor<3xf32>) -> tensor<3xf32>
    return %result : tensor<3xf32>
  }
}

// CHECK-COUNT-9: shuttle.coverage_manifest = {{.*}}reason = "unsupported_operation"
