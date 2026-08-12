// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --split-input-file --verify-diagnostics %s

module {
  func.func @zero_dimensions(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: tensor<2x4xf32>):
      %zero = arith.constant dense<0.0> : tensor<f32>
      // expected-error @+1 {{requires at least one reduction dimension}}
      %folded = "shuttle.fold"(%region_arg, %zero) ({
      ^bb0(%left: f32, %right: f32):
        %sum = arith.addf %left, %right : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>,
        order_free = false,
        reduction_dimensions = array<i64>,
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
      "shuttle.yield"(%folded) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @scalar_initializer(%arg: tensor<2x4xf32>) -> tensor<2xf32> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%input: tensor<2x4xf32>):
      %zero = arith.constant 0.0 : f32
      // expected-error @+1 {{input elements, rank-zero initializers, combiner arguments and yields}}
      %folded = "shuttle.fold"(%input, %zero) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<2x4xf32>, f32) -> tensor<2xf32>
      "shuttle.yield"(%folded) : (tensor<2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>]} : (tensor<2x4xf32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// -----

module {
  func.func @scalar_input(%arg: f32) -> f32 {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: f32):
      %zero = arith.constant dense<0.0> : tensor<f32>
      // expected-error @+1 {{requires positive-rank tensor inputs and ranked tensor results}}
      %folded = "shuttle.fold"(%region_arg, %zero) ({
      ^bb0(%left: f32, %right: f32):
        %sum = arith.addf %left, %right : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>,
        order_free = false,
        reduction_dimensions = array<i64: 0>,
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (f32, tensor<f32>) -> f32
      "shuttle.yield"(%folded) : (f32) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (f32) -> f32
    return %result : f32
  }
}

// -----

module {
  func.func @initializer_accumulator_dtype(%arg: tensor<2x4xf32>)
      -> tensor<2xf32> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%input: tensor<2x4xf32>):
      %zero = arith.constant dense<0.0> : tensor<f16>
      // expected-error @+1 {{input elements, rank-zero initializers, combiner arguments and yields}}
      %folded = "shuttle.fold"(%input, %zero) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<2x4xf32>, tensor<f16>) -> tensor<2xf32>
      "shuttle.yield"(%folded) : (tensor<2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>]} : (tensor<2x4xf32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// -----

module {
  func.func @input_accumulator_dtype(%arg: tensor<2x4xf16>)
      -> tensor<2xf32> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: tensor<2x4xf16>):
      %zero = arith.constant dense<0.0> : tensor<f32>
      // expected-error @+1 {{input elements, rank-zero initializers, combiner arguments and yields}}
      %folded = "shuttle.fold"(%region_arg, %zero) ({
      ^bb0(%element: f16, %accumulator: f32):
        "shuttle.yield"(%accumulator) : (f32) -> ()
      }) {
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>,
        order_free = false,
        reduction_dimensions = array<i64: 1>,
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x4xf16>, tensor<f32>) -> tensor<2xf32>
      "shuttle.yield"(%folded) : (tensor<2xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x4xf16>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// -----

module {
  func.func @unranked_input(%arg: tensor<*xf32>) -> tensor<f32> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: tensor<*xf32>):
      %zero = arith.constant dense<0.0> : tensor<f32>
      // expected-error @+1 {{requires positive-rank tensor inputs and ranked tensor results}}
      %folded = "shuttle.fold"(%region_arg, %zero) ({
      ^bb0(%left: f32, %right: f32):
        %sum = arith.addf %left, %right : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>,
        order_free = false,
        reduction_dimensions = array<i64: 0>,
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<*xf32>, tensor<f32>) -> tensor<f32>
      "shuttle.yield"(%folded) : (tensor<f32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<*xf32>) -> tensor<f32>
    return %result : tensor<f32>
  }
}

// -----

module {
  func.func @incompatible_shapes(%lhs: tensor<2x4xf32>,
      %rhs: tensor<3x4xf32>) -> (tensor<2xf32>, tensor<3xf32>) {
    %result:2 = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x4xf32>, %region_rhs: tensor<3x4xf32>):
      %zero_lhs = arith.constant dense<0.0> : tensor<f32>
      %zero_rhs = arith.constant dense<0.0> : tensor<f32>
      // expected-error @+1 {{multi-input folds require compatible input and result shapes}}
      %folded:2 = "shuttle.fold"(%region_lhs, %region_rhs, %zero_lhs, %zero_rhs) ({
      ^bb0(%lhs_element: f32, %rhs_element: f32,
          %lhs_accumulator: f32, %rhs_accumulator: f32):
        "shuttle.yield"(%lhs_accumulator, %rhs_accumulator) : (f32, f32) -> ()
      }) {
        accumulator_types = [f32, f32],
        operandSegmentSizes = array<i32: 2, 2>,
        order_free = false,
        reduction_dimensions = array<i64: 1>,
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x4xf32>, tensor<3x4xf32>, tensor<f32>, tensor<f32>)
          -> (tensor<2xf32>, tensor<3xf32>)
      "shuttle.yield"(%folded#0, %folded#1) : (tensor<2xf32>, tensor<3xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x4xf32>, tensor<3x4xf32>)
        -> (tensor<2xf32>, tensor<3xf32>)
    return %result#0, %result#1 : tensor<2xf32>, tensor<3xf32>
  }
}
