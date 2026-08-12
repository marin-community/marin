// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --split-input-file --shuttle-lower-algebra-to-stablehlo --verify-diagnostics %s

#identity = affine_map<(d0, d1) -> (d0, d1)>

module attributes {shuttle.coverage_manifest = {}} {
  func.func @unsupported_scalar_mutation(%arg0: tensor<2x3xf32>)
      -> tensor<2x3xf32> {
    %0 = "shuttle.region"(%arg0) ({
    ^bb0(%arg1: tensor<2x3xf32>):
      %1 = "shuttle.map"(%arg1) ({
      ^bb0(%element: f32):
        // expected-error @+1 {{has no authoritative StableHLO Map lowering}}
        %2 = math.sqrt %element : f32
        "shuttle.yield"(%2) : (f32) -> ()
      }) {indexing_maps = [#identity, #identity], source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
      "shuttle.yield"(%1) : (tensor<2x3xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>]} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----

module attributes {shuttle.coverage_manifest = {}} {
  func.func @ordered_fold(%arg0: tensor<2x4xf32>, %init: tensor<f32>)
      -> tensor<2xf32> {
    %0 = "shuttle.region"(%arg0, %init) ({
    ^bb0(%input: tensor<2x4xf32>, %initial: tensor<f32>):
      // expected-error @+1 {{order_free=false has no lossless StableHLO Reduce lowering}}
      %1 = "shuttle.fold"(%input, %initial) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) {shuttle.operation_ref = array<i64: 0, 1, 1>} : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = false, reduction_dimensions = array<i64: 1>, shuttle.operation_ref = array<i64: 0, 0, 0>, source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      "shuttle.yield"(%1) : (tensor<2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>]} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}

// -----

#input = affine_map<(d0, d1, d2) -> (d1, d1)>
#result = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module attributes {shuttle.coverage_manifest = {}} {
  func.func @duplicate_broadcast(%arg0: tensor<3x2xf32>)
      -> tensor<2x3x5xf32> {
    %0 = "shuttle.region"(%arg0) ({
    ^bb0(%arg1: tensor<3x2xf32>):
      // expected-error @+1 {{map indexing maps must use each direct domain dimension at most once}}
      %1 = "shuttle.map"(%arg1) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {indexing_maps = [#input, #result], semantics = #shuttle.map_semantics<broadcast_in_dim>, source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<3x2xf32>) -> tensor<2x3x5xf32>
      "shuttle.yield"(%1) : (tensor<2x3x5xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>]} : (tensor<3x2xf32>) -> tensor<2x3x5xf32>
    return %0 : tensor<2x3x5xf32>
  }
}
