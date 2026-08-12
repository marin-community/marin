// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --mlir-print-op-generic %S/Inputs/f32-map-shape-ops.mlir | FileCheck %s --check-prefix=ALGEBRA
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline %S/Inputs/f32-map-shape-ops.mlir | FileCheck %s --check-prefix=LOWERED
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline %S/Inputs/f32-map-shape-ops.mlir | FileCheck %s --check-prefix=LOWERED
// RUN: shuttle-test-opt --shuttle-test-report-normalized-fingerprint %S/Inputs/f32-map-shape-ops.mlir | FileCheck %s --check-prefix=HASH
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/f32-map-shape-ops.mlir | FileCheck %s --check-prefix=HASH
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/f32-map-shape-ops.mlir | FileCheck %s --check-prefix=HASH

// ALGEBRA: affine_map<(d0, d1) -> (d1)>
// ALGEBRA: affine_map<(d0) -> (0, d0)>
// ALGEBRA: affine_map<(d0, d1, d2) -> (d0, d2)>
// ALGEBRA: #shuttle.map_semantics<reshape>
// ALGEBRA: #shuttle.map_semantics<reshape>
// ALGEBRA: #shuttle.map_semantics<reshape>
// ALGEBRA: #shuttle.map_semantics<broadcast_in_dim>
// ALGEBRA: #shuttle.map_semantics<pointwise>
// ALGEBRA: "math.rsqrt"
// ALGEBRA: excluded = []

// LOWERED-NOT: shuttle.
// LOWERED: stablehlo.reshape
// LOWERED: stablehlo.reshape
// LOWERED: stablehlo.reshape
// LOWERED: stablehlo.broadcast_in_dim
// LOWERED-SAME: dims = [0, 2]
// LOWERED: stablehlo.rsqrt

// HASH: e9ec737a02db9357539569fade4a686473c6066f756cff8468d197f98c9ab977
