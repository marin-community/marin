// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=PLAN
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=LARGE
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=LARGE-GEOMETRY
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir | FileCheck %s --check-prefix=SMALL
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-verify-materialization-plan --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir | FileCheck %s --check-prefix=SMALL-GEOMETRY
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-test-report-simt32-schedule-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir -o /dev/null > %t.original
// RUN: shuttle-test-opt --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-test-report-simt32-schedule-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir -o /dev/null > %t.renamed
// RUN: diff %t.original %t.renamed
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --shuttle-test-report-simt32-schedule-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir -o /dev/null > %t.fast
// RUN: not diff %t.original %t.fast
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir | FileCheck %s --check-prefix=FAST

// PLAN: "shuttle.schedule_plan"
// PLAN-SAME: policy = #shuttle.policy<source_ordered>
// PLAN-SAME: schema_version = 1
// PLAN-SAME: target = #shuttle.schedule_target<simt32>
// PLAN: "shuttle.schedule_buffer"() <{{.*}}indexing = #shuttle.schedule_buffer_indexing<lexicographic>{{.*}}iteration_order = array<i64: 0, 1>{{.*}}lifetime_end = 0{{.*}}lifetime_start = 0{{.*}}source_buffer = 0{{.*}}tensor_type = tensor<{{.*}}xbf16>
// PLAN: "shuttle.schedule_buffer"() <{{.*}}indexing = #shuttle.schedule_buffer_indexing<scalar>{{.*}}iteration_order = array<i64>{{.*}}tensor_type = tensor<f32>
// PLAN: "shuttle.schedule_task"() <{{.*}}kind = #shuttle.schedule_task_kind<scalar>{{.*}}subgroup_size = 32{{.*}}tile_shape = array<i64>
// PLAN: "shuttle.schedule_task"() <{{.*}}kind = #shuttle.schedule_task_kind<row_fold>{{.*}}reduction_axis = 1{{.*}}reduction_order = #shuttle.schedule_reduction_order<tree_association_free_leaf_order_fixed>{{.*}}scratch_bytes = 1024{{.*}}subgroup_size = 32{{.*}}workgroup_threads = 256
// PLAN: "shuttle.schedule_plan_yield"

// LARGE-COUNT-21: "shuttle.schedule_buffer"
// LARGE-COUNT-19: "shuttle.schedule_task"
// LARGE-GEOMETRY: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 2048, 4096>{{.*}}grid_shape = array<i64: 2048>{{.*}}kind = #shuttle.schedule_task_kind<row_fold>{{.*}}serial_tiles = 16{{.*}}tile_shape = array<i64: 1, 256>
// SMALL-COUNT-21: "shuttle.schedule_buffer"
// SMALL-COUNT-19: "shuttle.schedule_task"
// SMALL-GEOMETRY: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 7, 13>{{.*}}grid_shape = array<i64: 7>{{.*}}kind = #shuttle.schedule_task_kind<row_fold>{{.*}}serial_tiles = 1{{.*}}tile_shape = array<i64: 1, 13>{{.*}}workgroup_threads = 32

// FAST: "shuttle.schedule_plan"
// FAST-SAME: policy = #shuttle.policy<fast>
// FAST: reduction_order = #shuttle.schedule_reduction_order<tree_association_free_leaf_order_fixed>
