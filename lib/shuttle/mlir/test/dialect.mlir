// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --shuttle-verify-source-coverage %s | FileCheck %s

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>
#identity = affine_map<(m, n) -> (m, n)>
#row = affine_map<(m, n) -> (m)>

module {
  func.func @algebra(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>)
      -> tensor<2xf32> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xf32>, %region_rhs: tensor<3x4xf32>):
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [f32],
        algorithm = "dot_general",
        indexing_maps = [#lhs, #rhs, #out],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
      %mapped = "shuttle.map"(%contract) ({
      ^bb0(%element: f32):
        %negated = arith.negf %element : f32
        "shuttle.yield"(%negated) : (f32) -> ()
      }) {
        indexing_maps = [#identity, #identity],
        source = #shuttle.source_ref<0, 0, 1, 0>
      } : (tensor<2x4xf32>) -> tensor<2x4xf32>
      %zero = arith.constant 0.0 : f32
      %folded = "shuttle.fold"(%mapped, %zero) ({
      ^bb0(%left: f32, %right: f32):
        %sum = arith.addf %left, %right : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {
        accumulator_types = [f32],
        operandSegmentSizes = array<i32: 1, 1>,
        order_free = false,
        reduction_dimensions = array<i64: 1>,
        source = #shuttle.source_ref<0, 0, 2, 0>
      } : (tensor<2x4xf32>, f32) -> tensor<2xf32>
      "shuttle.yield"(%folded) : (tensor<2xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [
        #shuttle.source_ref<0, 0, 0, 0>,
        #shuttle.source_ref<0, 0, 1, 0>,
        #shuttle.source_ref<0, 0, 2, 0>
      ]
    } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// CHECK: #shuttle.policy<source_ordered>
// CHECK: "shuttle.contract"
// CHECK: "shuttle.map"
// CHECK: "shuttle.fold"
// CHECK: "shuttle.yield"
