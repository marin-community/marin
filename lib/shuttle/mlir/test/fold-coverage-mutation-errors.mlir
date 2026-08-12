// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-remove-fold-owner-ref --shuttle-verify-source-coverage %S/Inputs/f32-reduce-add.mlir 2>&1 | FileCheck %s --check-prefix=OWNER
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-remove-fold-add-source --shuttle-verify-source-coverage %S/Inputs/f32-reduce-add.mlir 2>&1 | FileCheck %s --check-prefix=ADD
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-duplicate-fold-add-source --shuttle-verify-source-coverage %S/Inputs/f32-reduce-add.mlir 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-remove-fold-yield-ref --shuttle-verify-source-coverage %S/Inputs/f32-reduce-add.mlir 2>&1 | FileCheck %s --check-prefix=YIELD
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-duplicate-fold-owner-ref --shuttle-verify-source-coverage %S/Inputs/f32-reduce-add.mlir 2>&1 | FileCheck %s --check-prefix=OWNER-DUPLICATE
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-test-downgrade-manifest-version --shuttle-verify-source-coverage %S/Inputs/f32-reduce-add.mlir 2>&1 | FileCheck %s --check-prefix=VERSION
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-test-remove-manifest-version --shuttle-verify-source-coverage %S/Inputs/f32-reduce-add.mlir 2>&1 | FileCheck %s --check-prefix=VERSION
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-rewire-fold-yield --shuttle-verify-source-coverage %S/Inputs/f32-reduce-add.mlir 2>&1 | FileCheck %s --check-prefix=REWIRE
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-add-fold-fastmath --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-reduce-add.mlir 2>&1 | FileCheck %s --check-prefix=FASTMATH
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-add-fold-yield-attribute --shuttle-verify-source-coverage %S/Inputs/f32-reduce-add.mlir 2>&1 | FileCheck %s --check-prefix=YIELD-ATTRIBUTE

// OWNER: requires Reduce owner operation provenance
// ADD: represented source results do not equal manifest coverage
// DUPLICATE: duplicates selected source coverage
// YIELD: zero-result or function-result source anchors changed
// OWNER-DUPLICATE: has a missing-format or duplicate operation reference
// VERSION: malformed Shuttle coverage manifest
// REWIRE: zero-result or function-result source anchors changed
// FASTMATH: requires the closed ordered scalar f32 add combiner
// YIELD-ATTRIBUTE: zero-result or function-result source anchors changed
