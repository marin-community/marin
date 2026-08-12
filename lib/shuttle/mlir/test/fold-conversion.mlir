// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-semantic-erasure --shuttle-verify-source-coverage --mlir-print-op-generic %S/Inputs/f32-reduce-add.mlir | FileCheck %s --check-prefix=ALGEBRA
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline %S/Inputs/f32-reduce-add.mlir | FileCheck %s --check-prefix=LOWERED
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline %S/Inputs/f32-reduce-add.mlir | FileCheck %s --check-prefix=LOWERED
// RUN: shuttle-test-opt --shuttle-test-report-normalized-fingerprint %S/Inputs/f32-reduce-add.mlir | FileCheck %s --check-prefix=HASH
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/f32-reduce-add.mlir | FileCheck %s --check-prefix=HASH
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/f32-reduce-add.mlir | FileCheck %s --check-prefix=HASH
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra %S/Inputs/f32-reduce-add-axis0.mlir | FileCheck %s --check-prefix=AXIS0
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline %S/Inputs/f32-reduce-add-axis0.mlir | FileCheck %s --check-prefix=AXIS0-LOWERED
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline %S/Inputs/f32-reduce-add-axis0.mlir | FileCheck %s --check-prefix=AXIS0-LOWERED
// RUN: shuttle-test-opt --shuttle-test-report-normalized-fingerprint %S/Inputs/f32-reduce-add-axis0.mlir | FileCheck %s --check-prefix=AXIS0-HASH
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/f32-reduce-add-axis0.mlir | FileCheck %s --check-prefix=AXIS0-HASH
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/f32-reduce-add-axis0.mlir | FileCheck %s --check-prefix=AXIS0-HASH

// ALGEBRA: "shuttle.fold"
// ALGEBRA-SAME: accumulator_types = [f32]
// ALGEBRA-SAME: operandSegmentSizes = array<i32: 1, 1>
// ALGEBRA-SAME: order_free = true
// ALGEBRA-SAME: reduction_dimensions = array<i64: 1>
// ALGEBRA: "arith.addf"
// ALGEBRA-SAME: shuttle.source_refs
// ALGEBRA: "shuttle.yield"
// ALGEBRA-SAME: shuttle.operation_ref
// ALGEBRA: }) {shuttle.operation_ref
// ALGEBRA: shuttle.coverage_manifest
// ALGEBRA-SAME: excluded = []
// ALGEBRA-SAME: version = 2

// LOWERED-NOT: shuttle.
// LOWERED: stablehlo.reduce
// LOWERED-SAME: applies stablehlo.add across dimensions = [1]

// HASH: 27fd1b08e74cfd2cd29510789f01f109f8fb3507fc2a943c4665a797c604bfcd

// AXIS0: shuttle.fold
// AXIS0-SAME: reduction_dimensions = array<i64: 0>
// AXIS0: : (tensor<3x2xf32>, tensor<f32>) -> tensor<2xf32>

// AXIS0-LOWERED-NOT: shuttle.
// AXIS0-LOWERED: stablehlo.reduce
// AXIS0-LOWERED-SAME: applies stablehlo.add across dimensions = [0]

// AXIS0-HASH: a3c45e994f6792ec8004237675ae858b9799f4b4193360f141575deda020d93d
