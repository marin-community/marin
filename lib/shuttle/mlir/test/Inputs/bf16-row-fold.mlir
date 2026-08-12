// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module {
  func.func @main(%input: tensor<7x13xbf16>) -> tensor<7xbf16> {
    %init = stablehlo.constant dense<0.0> : tensor<bf16>
    %result = stablehlo.reduce(%input init: %init) applies stablehlo.add across dimensions = [1] : (tensor<7x13xbf16>, tensor<bf16>) -> tensor<7xbf16>
    return %result : tensor<7xbf16>
  }
}
