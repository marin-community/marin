// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --split-input-file --shuttle-lower-algebra-to-stablehlo %s | FileCheck %s

#identity = affine_map<(d0, d1) -> (d0, d1)>

module attributes {shuttle.coverage_manifest = {}} {
  func.func @tanh_from_scalar_body(%arg0: tensor<2x3xf32>)
      -> tensor<2x3xf32> {
    %0 = "shuttle.region"(%arg0) ({
    ^bb0(%arg1: tensor<2x3xf32>):
      %1 = "shuttle.map"(%arg1) ({
      ^bb0(%element: f32):
        %2 = math.tanh %element : f32
        "shuttle.yield"(%2) : (f32) -> ()
      }) {indexing_maps = [#identity, #identity], source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
      "shuttle.yield"(%1) : (tensor<2x3xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>]} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// CHECK-LABEL: func.func @tanh_from_scalar_body
// CHECK: stablehlo.tanh

// -----

#identity = affine_map<(d0, d1) -> (d0, d1)>

module attributes {shuttle.coverage_manifest = {}} {
  func.func @add_from_mutated_scalar_body(%arg0: tensor<2x3xf32>,
      %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = "shuttle.region"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<2x3xf32>, %arg3: tensor<2x3xf32>):
      %1 = "shuttle.map"(%arg2, %arg3) ({
      ^bb0(%left: f32, %right: f32):
        %2 = arith.addf %left, %right : f32
        "shuttle.yield"(%2) : (f32) -> ()
      }) {indexing_maps = [#identity, #identity, #identity], source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
      "shuttle.yield"(%1) : (tensor<2x3xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>]} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// CHECK-LABEL: func.func @add_from_mutated_scalar_body
// CHECK: stablehlo.add

// -----

#lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>

module attributes {shuttle.coverage_manifest = {}} {
  func.func @precision_from_contract(%arg0: tensor<3x2xf32>,
      %arg1: tensor<2x4xf32>) -> tensor<3x4xf32> {
    %0 = "shuttle.region"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<3x2xf32>, %arg3: tensor<2x4xf32>):
      %1 = "shuttle.contract"(%arg2, %arg3) {accumulator_types = [f32], algorithm = "dot_general", indexing_maps = [#lhs, #rhs, #out], iterator_kinds = ["parallel", "parallel", "reduction"], precision = ["HIGH", "HIGH"], source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<3x2xf32>, tensor<2x4xf32>) -> tensor<3x4xf32>
      "shuttle.yield"(%1) : (tensor<3x4xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>]} : (tensor<3x2xf32>, tensor<2x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}

// CHECK-LABEL: func.func @precision_from_contract
// CHECK: stablehlo.dot_general
// CHECK-SAME: precision = [HIGH, HIGH]
