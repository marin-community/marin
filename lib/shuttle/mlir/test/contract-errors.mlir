// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --split-input-file --verify-diagnostics %s

module {
  func.func @symbols(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>)
      -> tensor<2x4xf32> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xf32>, %region_rhs: tensor<3x4xf32>):
      // expected-error @+1 {{contraction indexing maps must not contain affine symbols}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [f32],
        algorithm = "dot_general",
        indexing_maps = [
          affine_map<(m, n, k)[s] -> (m, k)>,
          affine_map<(m, n, k)[s] -> (k, n)>,
          affine_map<(m, n, k)[s] -> (m, n)>
        ],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%contract) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @non_projected(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>)
      -> tensor<2x4xf32> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xf32>, %region_rhs: tensor<3x4xf32>):
      // expected-error @+1 {{contraction indexing maps must be projected permutations}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [f32],
        algorithm = "dot_general",
        indexing_maps = [
          affine_map<(m, n, k) -> (m + k, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%contract) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @extent_mismatch(%lhs: tensor<2x3xf32>, %rhs: tensor<5x4xf32>)
      -> tensor<2x4xf32> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xf32>, %region_rhs: tensor<5x4xf32>):
      // expected-error @+1 {{one domain dimension to inconsistent static extents}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [f32],
        algorithm = "dot_general",
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xf32>, tensor<5x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%contract) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x3xf32>, tensor<5x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @algorithm(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>)
      -> tensor<2x4xf32> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xf32>, %region_rhs: tensor<3x4xf32>):
      // expected-error @+1 {{supports only the 'dot_general' algorithm}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [f32],
        algorithm = "gemm",
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%contract) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @no_reduction(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>)
      -> tensor<2x4x3xf32> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xf32>, %region_rhs: tensor<3x4xf32>):
      // expected-error @+1 {{requires at least one reduction iterator}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [f32],
        algorithm = "dot_general",
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n, k)>
        ],
        iterator_kinds = ["parallel", "parallel", "parallel"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4x3xf32>
      "shuttle.yield"(%contract) : (tensor<2x4x3xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4x3xf32>
    return %result : tensor<2x4x3xf32>
  }
}

// -----

module {
  func.func @reduction_in_result(%lhs: tensor<2x3xf32>,
      %rhs: tensor<3x4xf32>) -> tensor<2x4x3xf32> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xf32>, %region_rhs: tensor<3x4xf32>):
      // expected-error @+1 {{each reduction dimension must appear in both input maps and not in the result map}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [f32],
        algorithm = "dot_general",
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n, k)>
        ],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4x3xf32>
      "shuttle.yield"(%contract) : (tensor<2x4x3xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4x3xf32>
    return %result : tensor<2x4x3xf32>
  }
}

// -----

module {
  func.func @reduction_missing_input(%lhs: tensor<2x3xf32>,
      %rhs: tensor<4xf32>) -> tensor<2x4xf32> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xf32>, %region_rhs: tensor<4xf32>):
      // expected-error @+1 {{each reduction dimension must appear in both input maps and not in the result map}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [f32],
        algorithm = "dot_general",
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (n)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xf32>, tensor<4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%contract) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x3xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @parallel_missing_result(%lhs: tensor<2x3xf32>,
      %rhs: tensor<3x4xf32>) -> tensor<2xf32> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xf32>, %region_rhs: tensor<3x4xf32>):
      // expected-error @+1 {{each parallel dimension must appear in an input map and exactly once in the result map}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [f32],
        algorithm = "dot_general",
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m)>
        ],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2xf32>
      "shuttle.yield"(%contract) : (tensor<2xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// -----

module {
  func.func @accumulator(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>)
      -> tensor<2x4xf32> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xf32>, %region_rhs: tensor<3x4xf32>):
      // expected-error @+1 {{accumulator types must be scalar numeric types}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [tensor<f32>],
        algorithm = "dot_general",
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%contract) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @scalar_input(%lhs: f32, %rhs: tensor<3x4xf32>)
      -> tensor<2x4xf32> {
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: f32, %region_rhs: tensor<3x4xf32>):
      // expected-error @+1 {{requires ranked tensor inputs and results}}
      %contract = "shuttle.contract"(%region_lhs, %region_rhs) {
        accumulator_types = [f32],
        algorithm = "dot_general",
        indexing_maps = [
          affine_map<(m, n, k) -> ()>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_kinds = ["parallel", "parallel", "reduction"],
        precision = ["DEFAULT", "DEFAULT"],
        source = #shuttle.source_ref<0, 0, 0, 0>
      } : (f32, tensor<3x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%contract) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
    } : (f32, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  "shuttle.region"() ({
    // expected-error @+1 {{requires at least one input and one result}}
    "shuttle.contract"() {
      accumulator_types = [],
      algorithm = "dot_general",
      indexing_maps = [],
      iterator_kinds = [],
      precision = [],
      source = #shuttle.source_ref<0, 0, 0, 0>
    } : () -> ()
    "shuttle.yield"() : () -> ()
  }) {
    policy = #shuttle.policy<source_ordered>,
    source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
  } : () -> ()
}
