// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-opt --shuttle-form-structural-regions %s 2>&1 | FileCheck %s --check-prefix=FORM
// RUN: not shuttle-opt --shuttle-convert-stablehlo-to-algebra %s 2>&1 | FileCheck %s --check-prefix=CONVERT
// RUN: not shuttle-opt --shuttle-lower-algebra-to-stablehlo %s 2>&1 | FileCheck %s --check-prefix=LOWER

module {
  func.func public @main() {
    return
  }
}

// FORM: requires shuttle-annotate-source before region formation
// CONVERT: requires shuttle-form-structural-regions before conversion
// LOWER: requires structural coverage before StableHLO lowering
