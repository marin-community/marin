// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --shuttle-annotate-source --mlir-print-op-generic %s | FileCheck %s

module {
  func.func @name_does_not_enter_the_reference(%arg: tensor<2xf32>)
      -> tensor<2xf32> {
    %0 = stablehlo.negate %arg : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}

// CHECK: "stablehlo.negate"
// CHECK-SAME: shuttle.source_refs = [#shuttle.source_ref<0, 0, 0, 0>]
