// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --shuttle-lower-algebra-to-stablehlo --verify-diagnostics %s

#identity = affine_map<(d0, d1) -> (d0, d1)>

module attributes {shuttle.coverage_manifest = {}} {
  func.func @unsupported_scalar_mutation(%arg0: tensor<2x3xf32>)
      -> tensor<2x3xf32> {
    %0 = "shuttle.region"(%arg0) ({
    ^bb0(%arg1: tensor<2x3xf32>):
      %1 = "shuttle.map"(%arg1) ({
      ^bb0(%element: f32):
        // expected-error @+1 {{has no authoritative StableHLO Map lowering}}
        %2 = arith.negf %element : f32
        "shuttle.yield"(%2) : (f32) -> ()
      }) {indexing_maps = [#identity, #identity], source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
      "shuttle.yield"(%1) : (tensor<2x3xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>]} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
