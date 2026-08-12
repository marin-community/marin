// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=PLAN
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=LARGE-COUNT
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir | FileCheck %s --check-prefix=SMALL-COUNT
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-test-report-materialization-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir -o /dev/null > %t.original
// RUN: shuttle-test-opt --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-test-report-materialization-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir -o /dev/null > %t.renamed
// RUN: diff %t.original %t.renamed
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-test-report-materialization-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir -o /dev/null > %t.fast
// RUN: not diff %t.original %t.fast
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir | FileCheck %s --check-prefix=FAST
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --mlir-print-op-generic %S/Inputs/f32-reduce-add-axis0.mlir | FileCheck %s --check-prefix=AXIS0
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization %S/Inputs/two-row-folds.mlir 2>&1 | FileCheck %s --check-prefix=BOUNDARY

// PLAN: "shuttle.materialization_plan"
// PLAN-SAME: fingerprint = "e6d3838bc6ae01c514adf9fb71236406da01b93ee866f411ffb4af4d0161384d"
// PLAN-SAME: policy = #shuttle.policy<source_ordered>
// PLAN-SAME: schema_version = 1
// PLAN: "shuttle.materialization_buffer"
// PLAN-SAME: consumers = array<i64: 0>
// PLAN-SAME: live_in = true
// PLAN-SAME: storage = #shuttle.materialization_storage<external>
// PLAN-SAME: tensor_type = tensor<2048x4096xbf16>
// PLAN: "shuttle.materialization_buffer"() <{{.*}}storage = #shuttle.materialization_storage<temporary>
// PLAN: "shuttle.materialization_task"() <{{.*}}domain_shape = array<i64>{{.*}}kind = #shuttle.materialization_task_kind<map>
// PLAN: "shuttle.materialization_task"() <{{.*}}dependencies = array<i64:{{.*}}domain_shape = array<i64: 2048, 4096>{{.*}}kind = #shuttle.materialization_task_kind<fold>{{.*}}order_free = true{{.*}}reduction_dimensions = array<i64: 1>
// PLAN: "shuttle.materialization_plan_yield"

// LARGE-COUNT-COUNT-21: "shuttle.materialization_buffer"
// LARGE-COUNT-COUNT-19: "shuttle.materialization_task"
// SMALL-COUNT-COUNT-21: "shuttle.materialization_buffer"
// SMALL-COUNT-COUNT-19: "shuttle.materialization_task"

// FAST: "shuttle.materialization_plan"
// FAST-SAME: policy = #shuttle.policy<fast>

// AXIS0: "shuttle.materialization_task"() <{{.*}}domain_shape = array<i64: 3, 2>{{.*}}kind = #shuttle.materialization_task_kind<fold>{{.*}}reduction_dimensions = array<i64: 0>
// BOUNDARY: requires exactly one connected static Fold and Map region
