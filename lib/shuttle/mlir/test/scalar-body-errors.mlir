// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --allow-unregistered-dialect --split-input-file --verify-diagnostics %s

module {
  func.func @map_effect(%arg: f32) -> f32 {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: f32):
      // expected-error @+1 {{scalar body operations must have proven no memory effects}}
      %mapped = "shuttle.map"(%region_arg) ({
      ^bb0(%element: f32):
        "test.effect"() : () -> ()
        "shuttle.yield"(%element) : (f32) -> ()
      }) {
        indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (f32) -> f32
      "shuttle.yield"(%mapped) : (f32) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (f32) -> f32
    return %result : f32
  }
}

// -----

module {
  func.func @map_nested(%arg: f32) -> f32 {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: f32):
      // expected-error @+1 {{scalar body operations must not contain nested regions}}
      %mapped = "shuttle.map"(%region_arg) ({
      ^bb0(%element: f32):
        %nested = scf.execute_region -> f32 {
          scf.yield %element : f32
        }
        "shuttle.yield"(%nested) : (f32) -> ()
      }) {
        indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (f32) -> f32
      "shuttle.yield"(%mapped) : (f32) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (f32) -> f32
    return %result : f32
  }
}

// -----

module {
  func.func @fold_shaped(%arg: tensor<2x4xf32>) -> tensor<2xf32> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: tensor<2x4xf32>):
      %zero = arith.constant 0.0 : f32
      // expected-error @+1 {{scalar body operations must not use shaped values}}
      %folded = "shuttle.fold"(%region_arg, %zero) ({
      ^bb0(%left: f32, %right: f32):
        %shaped = tensor.from_elements %left : tensor<1xf32>
        %sum = arith.addf %left, %right : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>,
        order_free = false,
        reduction_dimensions = array<i64: 1>,
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x4xf32>, f32) -> tensor<2xf32>
      "shuttle.yield"(%folded) : (tensor<2xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x4xf32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}
