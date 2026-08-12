// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra %s | FileCheck %s
// RUN: shuttle-test-opt --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra %s | not grep shuttle.fold

module {
  func.func @non_add(%input: tensor<2x4xf32>, %init: tensor<f32>) -> tensor<2xf32> {
    %result = "stablehlo.reduce"(%input, %init) ({
    ^bb0(%element: tensor<f32>, %accumulator: tensor<f32>):
      %product = stablehlo.multiply %element, %accumulator : tensor<f32>
      stablehlo.return %product : tensor<f32>
    }) {dimensions = array<i64: 1>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// -----

module {
  func.func @extra_combiner_op(%input: tensor<2x4xf32>, %init: tensor<f32>) -> tensor<2xf32> {
    %result = "stablehlo.reduce"(%input, %init) ({
    ^bb0(%element: tensor<f32>, %accumulator: tensor<f32>):
      %negated = stablehlo.negate %element : tensor<f32>
      %sum = stablehlo.add %negated, %accumulator : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {dimensions = array<i64: 1>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// -----

module {
  func.func @bf16_accumulator(%input: tensor<2x4xbf16>, %init: tensor<bf16>) -> tensor<2xbf16> {
    %result = "stablehlo.reduce"(%input, %init) ({
    ^bb0(%element: tensor<bf16>, %accumulator: tensor<bf16>):
      %sum = stablehlo.add %element, %accumulator : tensor<bf16>
      stablehlo.return %sum : tensor<bf16>
    }) {dimensions = array<i64: 1>} : (tensor<2x4xbf16>, tensor<bf16>) -> tensor<2xbf16>
    return %result : tensor<2xbf16>
  }
}

// -----

module {
  func.func @multi_result(%lhs: tensor<2x4xf32>, %rhs: tensor<2x4xf32>,
      %lhs_init: tensor<f32>, %rhs_init: tensor<f32>)
      -> (tensor<2xf32>, tensor<2xf32>) {
    %result:2 = "stablehlo.reduce"(%lhs, %rhs, %lhs_init, %rhs_init) ({
    ^bb0(%lhs_element: tensor<f32>, %rhs_element: tensor<f32>,
        %lhs_accumulator: tensor<f32>, %rhs_accumulator: tensor<f32>):
      %lhs_sum = stablehlo.add %lhs_element, %lhs_accumulator : tensor<f32>
      %rhs_sum = stablehlo.add %rhs_element, %rhs_accumulator : tensor<f32>
      stablehlo.return %lhs_sum, %rhs_sum : tensor<f32>, tensor<f32>
    }) {dimensions = array<i64: 1>} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<f32>, tensor<f32>) -> (tensor<2xf32>, tensor<2xf32>)
    return %result#0, %result#1 : tensor<2xf32>, tensor<2xf32>
  }
}

// -----

module {
  func.func @promoted_result(%input: tensor<4x4xf32>, %init: tensor<f32>)
      -> tensor<4xf64> {
    %result = "stablehlo.reduce"(%input, %init) ({
    ^bb0(%element: tensor<f64>, %accumulator: tensor<f64>):
      %sum = stablehlo.add %element, %accumulator : tensor<f64>
      stablehlo.return %sum : tensor<f64>
    }) {dimensions = array<i64: 0>} : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf64>
    return %result : tensor<4xf64>
  }
}

// CHECK-DAG: shuttle.coverage_manifest = {{.*}}reason = "unsupported_operation"{{.*}}reason = "enclosing_region_excluded"
// CHECK-DAG: shuttle.coverage_manifest = {{.*}}reason = "unsupported_operation"{{.*}}reason = "enclosing_region_excluded"
// CHECK-DAG: shuttle.coverage_manifest = {{.*}}reason = "unsupported_operation"{{.*}}reason = "enclosing_region_excluded"
// CHECK-DAG: shuttle.coverage_manifest = {{.*}}reason = "unsupported_operation"{{.*}}reason = "enclosing_region_excluded"
// CHECK-DAG: shuttle.coverage_manifest = {{.*}}reason = "unsupported_operation"{{.*}}reason = "enclosing_region_excluded"
// CHECK-DAG: func.func @non_add
// CHECK-DAG: func.func @extra_combiner_op
// CHECK-DAG: func.func @bf16_accumulator
// CHECK-DAG: func.func @multi_result
// CHECK-DAG: func.func @promoted_result
