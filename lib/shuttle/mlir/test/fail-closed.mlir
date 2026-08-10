// RUN: not shuttle-opt --shuttle-form-structural-regions %s 2>&1 | FileCheck %s --check-prefix=FORM
// RUN: not shuttle-opt --shuttle-convert-stablehlo-to-algebra %s 2>&1 | FileCheck %s --check-prefix=CONVERT
// RUN: not shuttle-opt --shuttle-canonicalize %s 2>&1 | FileCheck %s --check-prefix=CANONICALIZE
// RUN: not shuttle-opt --shuttle-lower-algebra-to-stablehlo %s 2>&1 | FileCheck %s --check-prefix=LOWER

module {}

// FORM: structural StableHLO region formation is declared but not implemented
// CONVERT: StableHLO-to-Shuttle conversion is declared but not implemented
// CANONICALIZE: Shuttle algebra canonicalization is declared but not implemented
// LOWER: Shuttle-to-StableHLO lowering is declared but not implemented
