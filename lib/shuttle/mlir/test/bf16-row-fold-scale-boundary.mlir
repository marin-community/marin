// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=PROVENANCE
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=BOUNDARY
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-backward.mlir | FileCheck %s --check-prefix=BOUNDARY
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-composed.mlir | FileCheck %s --check-prefix=BOUNDARY
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir | FileCheck %s --check-prefix=BOUNDARY
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-backward.mlir | FileCheck %s --check-prefix=BOUNDARY
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-composed.mlir | FileCheck %s --check-prefix=BOUNDARY

// Source annotation owns the reducer, nested add, and terminator provenance.
// PROVENANCE: "stablehlo.reduce"
// PROVENANCE: "stablehlo.add"
// PROVENANCE-SAME: shuttle.source_refs = [#shuttle.source_ref<0, 1, 0, 0>]
// PROVENANCE: "stablehlo.return"
// PROVENANCE-SAME: shuttle.operation_ref = array<i64: 0, 1, 1>
// PROVENANCE: }) {shuttle.operation_ref = array<i64: 0, 0, 5>, shuttle.source_refs = [#shuttle.source_ref<0, 0, 5, 0>]}

// The complete Target 1 Reduce and Map source language is selected.
// BOUNDARY: shuttle.coverage_manifest
// BOUNDARY-SAME: excluded = []
// BOUNDARY-SAME: version = 2
