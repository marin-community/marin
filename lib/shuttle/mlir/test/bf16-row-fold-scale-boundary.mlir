// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=PROVENANCE
// RUN: not shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --mlir-print-ir-after-failure %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir 2>&1 | FileCheck %s --check-prefix=FAIL
// RUN: not shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --mlir-print-ir-after-failure %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-backward.mlir 2>&1 | FileCheck %s --check-prefix=FAIL
// RUN: not shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --mlir-print-ir-after-failure %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-composed.mlir 2>&1 | FileCheck %s --check-prefix=FAIL
// RUN: not shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --mlir-print-ir-after-failure %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir 2>&1 | FileCheck %s --check-prefix=FAIL
// RUN: not shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --mlir-print-ir-after-failure %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-backward.mlir 2>&1 | FileCheck %s --check-prefix=FAIL
// RUN: not shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --mlir-print-ir-after-failure %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-composed.mlir 2>&1 | FileCheck %s --check-prefix=FAIL

// Source annotation owns nested reducer provenance even though selection does
// not yet traverse a region-bearing operation.
// PROVENANCE: "stablehlo.reduce"
// PROVENANCE: "stablehlo.add"
// PROVENANCE-SAME: shuttle.source_refs = [#shuttle.source_ref<0, 1, 0, 0>]
// PROVENANCE: "stablehlo.return"
// PROVENANCE-SAME: shuttle.operation_ref = array<i64: 0, 1, 1>
// PROVENANCE: }) {shuttle.source_refs = [#shuttle.source_ref<0, 0, 5, 0>]}

// Step 1 freezes the fail-closed boundary. Region formation rejects the first
// reduce before it can publish a coverage manifest.
// FAIL: error: 'stablehlo.reduce' op the first offline slice requires region-free source operations
// FAIL-NOT: shuttle.coverage_manifest
