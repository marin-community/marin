// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --mlir-print-op-generic %S/Inputs/f32-mapped-singleton-broadcast.mlir | FileCheck %s --check-prefix=ALGEBRA
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline %S/Inputs/f32-mapped-singleton-broadcast.mlir | FileCheck %s --check-prefix=LOWERED
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline %S/Inputs/f32-mapped-singleton-broadcast.mlir | FileCheck %s --check-prefix=LOWERED
// RUN: shuttle-test-opt --shuttle-test-report-normalized-fingerprint %S/Inputs/f32-mapped-singleton-broadcast.mlir | FileCheck %s --check-prefix=HASH
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/f32-mapped-singleton-broadcast.mlir | FileCheck %s --check-prefix=HASH
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/f32-mapped-singleton-broadcast.mlir | FileCheck %s --check-prefix=HASH

// ALGEBRA: affine_map<(d0, d1) -> (d0, d1 floordiv 13)>
// ALGEBRA: affine_map<(d0, d1) -> (d0 floordiv 7, d1)>
// ALGEBRA: affine_map<(d0, d1) -> (d1 floordiv 7, d0)>
// ALGEBRA-COUNT-3: #shuttle.map_semantics<broadcast_in_dim>
// ALGEBRA: excluded = []

// LOWERED-NOT: shuttle.
// LOWERED-COUNT-2: stablehlo.broadcast_in_dim {{.*}} dims = [0, 1]
// LOWERED: stablehlo.broadcast_in_dim {{.*}} dims = [1, 0]

// HASH: 30b2a5c62f107f1a582179b2422230b4ae248d745a49b2e028dfa29bf3c3ec7c
