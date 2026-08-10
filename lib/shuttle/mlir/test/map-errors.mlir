// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --split-input-file --verify-diagnostics %s

module {
  func.func @symbols(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: tensor<2x4xf32>):
      // expected-error @+1 {{map indexing maps must not contain affine symbols}}
      %mapped = "shuttle.map"(%region_arg) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {
        indexing_maps = [
          affine_map<(m, n)[s] -> (m, n)>,
          affine_map<(m, n)[s] -> (m, n)>
        ],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%mapped) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @non_projected(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: tensor<2x4xf32>):
      // expected-error @+1 {{map indexing maps must be projected permutations}}
      %mapped = "shuttle.map"(%region_arg) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {
        indexing_maps = [
          affine_map<(m, n) -> (m + n, n)>,
          affine_map<(m, n) -> (m, n)>
        ],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%mapped) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @unbound_domain(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: tensor<2x4xf32>):
      // expected-error @+1 {{every map domain dimension must be bound by a ranked tensor dimension}}
      %mapped = "shuttle.map"(%region_arg) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {
        indexing_maps = [
          affine_map<(m, n, unused) -> (m, n)>,
          affine_map<(m, n, unused) -> (m, n)>
        ],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%mapped) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @extent_mismatch(%arg: tensor<2x4xf32>) -> tensor<3x4xf32> {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: tensor<2x4xf32>):
      // expected-error @+1 {{map indexing maps bind one domain dimension to inconsistent static extents}}
      %mapped = "shuttle.map"(%region_arg) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {
        indexing_maps = [
          affine_map<(m, n) -> (m, n)>,
          affine_map<(m, n) -> (m, n)>
        ],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x4xf32>) -> tensor<3x4xf32>
      "shuttle.yield"(%mapped) : (tensor<3x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x4xf32>) -> tensor<3x4xf32>
    return %result : tensor<3x4xf32>
  }
}

// -----

module {
  func.func @scalar_domain(%arg: f32) -> f32 {
    %result = "shuttle.region"(%arg) ({
    ^bb0(%region_arg: f32):
      // expected-error @+1 {{a scalar-only map requires a zero-dimensional indexing domain}}
      %mapped = "shuttle.map"(%region_arg) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {
        indexing_maps = [
          affine_map<(unused) -> ()>,
          affine_map<(unused) -> ()>
        ],
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
