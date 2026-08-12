// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-semantic-erasure --shuttle-test-mutate-excluded-kind --shuttle-verify-source-coverage %S/Inputs/coverage-mutation.mlir 2>&1 | FileCheck %s --check-prefix=KIND
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-semantic-erasure --shuttle-test-mutate-excluded-attribute --shuttle-verify-source-coverage %S/Inputs/coverage-mutation.mlir 2>&1 | FileCheck %s --check-prefix=ATTRIBUTE
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-semantic-erasure --shuttle-test-mutate-excluded-operand --shuttle-verify-source-coverage %S/Inputs/coverage-mutation.mlir 2>&1 | FileCheck %s --check-prefix=OPERAND
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-semantic-erasure --shuttle-test-absorb-excluded-source --shuttle-verify-source-coverage %S/Inputs/coverage-mutation.mlir 2>&1 | FileCheck %s --check-prefix=ABSORB
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-semantic-erasure --shuttle-test-rewire-return --shuttle-verify-source-coverage %S/Inputs/coverage-mutation.mlir 2>&1 | FileCheck %s --check-prefix=RETURN
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-semantic-erasure --shuttle-test-corrupt-policy-digest --shuttle-verify-source-coverage %S/Inputs/coverage-mutation.mlir 2>&1 | FileCheck %s --check-prefix=MANIFEST

// KIND: excluded operation fingerprint or operand anchors changed
// ATTRIBUTE: excluded operation fingerprint or operand anchors changed
// OPERAND: excluded operation fingerprint or operand anchors changed
// ABSORB: structural regions do not equal manifest region groups
// RETURN: zero-result or function-result source anchors changed
// MANIFEST: has inconsistent Shuttle policy digests
