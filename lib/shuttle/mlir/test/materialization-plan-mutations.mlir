// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-test-delete-materialization-task --shuttle-verify-materialization-plan %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir 2>&1 | FileCheck %s --check-prefix=DELETE
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-test-reorder-materialization-tasks --shuttle-verify-materialization-plan %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir 2>&1 | FileCheck %s --check-prefix=ORDER
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-test-reorder-materialization-tasks-consistently --shuttle-verify-materialization-plan %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir 2>&1 | FileCheck %s --check-prefix=SSA-ORDER
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-test-replay-materialization-source --shuttle-verify-materialization-plan %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir 2>&1 | FileCheck %s --check-prefix=REPLAY
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-test-duplicate-materialization-algebra-source --shuttle-verify-materialization-plan %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir 2>&1 | FileCheck %s --check-prefix=REPLAY
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-test-swap-materialization-edges --shuttle-verify-materialization-plan %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir 2>&1 | FileCheck %s --check-prefix=EDGE
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-test-add-scalar-materialization-domain --shuttle-verify-materialization-plan %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir 2>&1 | FileCheck %s --check-prefix=DOMAIN
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-test-empty-tensor-materialization-domain --shuttle-verify-materialization-plan %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir 2>&1 | FileCheck %s --check-prefix=DOMAIN
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-test-add-materialization-attribute --shuttle-verify-materialization-plan %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir 2>&1 | FileCheck %s --check-prefix=ATTRIBUTE
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-test-add-fold-fastmath --shuttle-verify-materialization-plan %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir 2>&1 | FileCheck %s --check-prefix=SEMANTICS

// DELETE: producer is outside the task range
// ORDER: task ordinals must be contiguous
// SSA-ORDER: task order and edges must equal bound algebra SSA dependencies
// REPLAY: source must uniquely bind one surviving algebra task
// EDGE: task order and edges must equal bound algebra SSA dependencies
// DOMAIN: Map domains must exactly identify scalar result tasks
// ATTRIBUTE: does not permit discardable attributes
// SEMANTICS: does not match its bound algebra semantics
