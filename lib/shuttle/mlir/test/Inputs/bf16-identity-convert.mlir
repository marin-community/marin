// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module {
  func.func @main(%arg0: tensor<2xbf16>) -> tensor<2xbf16> {
    %0 = stablehlo.convert %arg0 : tensor<2xbf16>
    return %0 : tensor<2xbf16>
  }
}
