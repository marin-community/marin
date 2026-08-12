// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

module {
  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "stablehlo.exponential"(%arg0) {
      result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00,
          rtol = 1.000000e+00, ulps = 5,
          mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
    } : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
