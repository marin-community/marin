// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic %S/Inputs/jax-0.10.1-tanh-dot-forward.mlir | FileCheck %s --check-prefix=FORWARD
// RUN: shuttle-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic %S/Inputs/jax-0.10.1-tanh-dot-vjp.mlir | FileCheck %s --check-prefix=VJP

// The forward graph is one connected supported interval.
// FORWARD: "shuttle.region"
// FORWARD: "shuttle.contract"
// FORWARD: "shuttle.map"
// FORWARD: "math.tanh"
// FORWARD: "shuttle.contract"
// FORWARD: "shuttle.yield"
// FORWARD-NOT: "stablehlo.dot_general"
// FORWARD-NOT: "stablehlo.tanh"
// FORWARD-NOT: shuttle.lowering_
// FORWARD: shuttle.coverage_manifest
// FORWARD-SAME: complete
// FORWARD-SAME: function_results
// FORWARD-SAME: selected_regions
// FORWARD-SAME: zero_result_operations

// The complete VJP graph is represented by generic Contract and Map algebra.
// VJP: "shuttle.region"
// VJP: "shuttle.contract"
// VJP: "shuttle.map"
// VJP: "math.tanh"
// VJP: "shuttle.map"
// VJP: "arith.constant"
// VJP: "shuttle.map"
// VJP: "shuttle.map"
// VJP: "arith.subf"
// VJP: "shuttle.contract"
// VJP: "shuttle.map"
// VJP: "shuttle.contract"
// VJP: "arith.mulf"
// VJP: "shuttle.map"
// VJP: "arith.mulf"
// VJP: "shuttle.map"
// VJP: "arith.addf"
// VJP: "shuttle.contract"
// VJP: "shuttle.map"
// VJP: "shuttle.contract"
// VJP-NOT: "stablehlo.multiply"
// VJP-NOT: "stablehlo.add"
// VJP-NOT: "stablehlo.constant"
// VJP-NOT: "stablehlo.broadcast_in_dim"
// VJP-NOT: "stablehlo.subtract"
// VJP: shuttle.coverage_manifest
// VJP-SAME: excluded = []
