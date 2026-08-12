// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-opt --shuttle-form-structural-regions %s 2>&1 | FileCheck %s --check-prefix=FORM
// RUN: not shuttle-opt --shuttle-convert-stablehlo-to-algebra %s 2>&1 | FileCheck %s --check-prefix=CONVERT
// RUN: not shuttle-opt --shuttle-lower-algebra-to-stablehlo %s 2>&1 | FileCheck %s --check-prefix=LOWER
// RUN: shuttle-opt --shuttle-stablehlo-source-ordered-pipeline %S/Inputs/result-accuracy.mlir | FileCheck %s --check-prefix=ACCURACY
// RUN: shuttle-opt --shuttle-stablehlo-source-ordered-pipeline %S/Inputs/non-scalar-broadcast.mlir | FileCheck %s --check-prefix=BROADCAST
// RUN: shuttle-opt --shuttle-stablehlo-source-ordered-pipeline %S/Inputs/bf16-identity-convert.mlir | FileCheck %s --check-prefix=BF16IDENTITY

module {
  func.func public @main() {
    return
  }
}

// FORM: requires shuttle-annotate-source before region formation
// CONVERT: requires shuttle-form-structural-regions before conversion
// LOWER: requires structural coverage before StableHLO lowering

// Explicit result-accuracy contracts stay outside the bounded scalar algebra.
// ACCURACY: stablehlo.exponential
// ACCURACY-SAME: result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00

// Singleton expansion needs richer Map semantics than the scalar broadcasts
// in the bounded corpus and therefore remains an unsupported island.
// BROADCAST: stablehlo.broadcast_in_dim
// BROADCAST-SAME: dims = [0, 2]

// The corpus contains exact f32 identity converts, not bf16 identities.
// BF16IDENTITY: stablehlo.convert
// BF16IDENTITY-SAME: tensor<2xbf16>
