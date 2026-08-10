// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --shuttle-verify-semantic-erasure --verify-diagnostics %s

module {
  func.func @selected(%arg: tensor<2xf32>) -> tensor<2xf32> {
    // expected-error @+1 {{selected source operation survived Shuttle conversion}}
    %selected = "stablehlo.negate"(%arg) {shuttle.selected} : (tensor<2xf32>) -> tensor<2xf32>
    return %selected : tensor<2xf32>
  }
}
