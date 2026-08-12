// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --split-input-file --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-verify-simt32-row-fold-schedule --mlir-print-op-generic %S/Inputs/f32-row-fold-ceil-div.mlir | FileCheck %s

// CHECK-LABEL: "builtin.module"() <{sym_name = "exact_tile"}>
// CHECK: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 1, 256>{{.*}}grid_shape = array<i64: 1>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}serial_tiles = 1{{.*}}tile_shape = array<i64: 256>
// CHECK: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 1, 256>{{.*}}grid_shape = array<i64: 1>{{.*}}kind = #shuttle.schedule_task_kind<row_fold>{{.*}}serial_tiles = 1{{.*}}tile_shape = array<i64: 1, 256>
// CHECK-LABEL: "builtin.module"() <{sym_name = "max_extent"}>
// CHECK: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 1, 9223372036854775807>{{.*}}grid_shape = array<i64: 36028797018963968>{{.*}}kind = #shuttle.schedule_task_kind<elementwise>{{.*}}serial_tiles = 1{{.*}}tile_shape = array<i64: 256>
// CHECK: "shuttle.schedule_task"() <{{.*}}domain_shape = array<i64: 1, 9223372036854775807>{{.*}}grid_shape = array<i64: 1>{{.*}}kind = #shuttle.schedule_task_kind<row_fold>{{.*}}serial_tiles = 36028797018963968{{.*}}tile_shape = array<i64: 1, 256>
