// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --shuttle-verify-source-coverage --verify-diagnostics %s

module {
  func.func @coverage(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>)
      -> tensor<2x4xf32> {
    // expected-error @+1 {{declared source reference is not represented by algebra}}
    %result = "shuttle.region"(%lhs, %rhs) ({
    ^bb0(%region_lhs: tensor<2x3xf32>, %region_rhs: tensor<3x4xf32>):
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
      } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
      "shuttle.yield"(%contract) : (tensor<2x4xf32>) -> ()
    }) {
      policy = #shuttle.policy<source_ordered>,
      source_refs = [
        #shuttle.source_ref<0, 0, 0, 0>,
        #shuttle.source_ref<0, 0, 1, 0>
      ]
    } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}
