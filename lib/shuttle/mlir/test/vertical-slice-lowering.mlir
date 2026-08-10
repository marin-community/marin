// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-opt --shuttle-stablehlo-source-ordered-pipeline %S/Inputs/jax-0.10.1-tanh-dot-forward.mlir | FileCheck %s --check-prefix=FORWARD
// RUN: shuttle-opt --shuttle-stablehlo-source-ordered-pipeline %S/Inputs/jax-0.10.1-tanh-dot-vjp.mlir | FileCheck %s --check-prefix=VJP
// RUN: shuttle-opt --shuttle-stablehlo-source-ordered-pipeline %S/Inputs/jax-0.10.1-tanh-dot-forward-alt.mlir | FileCheck %s --check-prefix=FORWARD
// RUN: shuttle-opt --shuttle-stablehlo-source-ordered-pipeline %S/Inputs/jax-0.10.1-tanh-dot-vjp-alt.mlir | FileCheck %s --check-prefix=VJP
// RUN: shuttle-opt --shuttle-stablehlo-fast-pipeline %S/Inputs/jax-0.10.1-tanh-dot-forward.mlir | FileCheck %s --check-prefix=FORWARD
// RUN: shuttle-opt --shuttle-stablehlo-fast-pipeline %S/Inputs/jax-0.10.1-tanh-dot-vjp.mlir | FileCheck %s --check-prefix=VJP
// RUN: shuttle-opt --shuttle-stablehlo-source-ordered-pipeline %S/Inputs/jax-0.10.1-map-only.mlir | FileCheck %s --check-prefix=MAP
// RUN: shuttle-opt --shuttle-stablehlo-fast-pipeline %S/Inputs/jax-0.10.1-map-only.mlir | FileCheck %s --check-prefix=MAP
// RUN: shuttle-opt --shuttle-stablehlo-source-ordered-pipeline %S/Inputs/jax-0.10.1-contract-only.mlir | FileCheck %s --check-prefix=CONTRACT
// RUN: shuttle-opt --shuttle-stablehlo-fast-pipeline %S/Inputs/jax-0.10.1-contract-only.mlir | FileCheck %s --check-prefix=CONTRACT

// FORWARD-NOT: shuttle.
// FORWARD: %[[DOT:.*]] = stablehlo.dot_general
// FORWARD: %[[TANH:.*]] = stablehlo.tanh %[[DOT]]
// FORWARD: %[[OUT:.*]] = stablehlo.dot_general %[[TANH]],
// FORWARD: return %[[OUT]]

// Source-ordered lowering reconstructs the selected operations in their input
// order and leaves the unsupported island unchanged.
// VJP-NOT: shuttle.
// VJP: %[[DOT:.*]] = stablehlo.dot_general
// VJP: %[[TANH:.*]] = stablehlo.tanh %[[DOT]]
// VJP: %[[CST:.*]] = stablehlo.constant
// VJP: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[CST]]
// VJP: %[[SUB:.*]] = stablehlo.subtract %[[BROADCAST]], %[[TANH]]
// VJP: %[[DW1T:.*]] = stablehlo.dot_general
// VJP: %[[DW1:.*]] = stablehlo.transpose %[[DW1T]]
// VJP: %[[DOUT:.*]] = stablehlo.dot_general
// VJP: %[[MUL0:.*]] = stablehlo.multiply %[[DOUT]], %[[SUB]]
// VJP: %[[MUL1:.*]] = stablehlo.multiply %[[MUL0]], %[[TANH]]
// VJP: %[[ADD:.*]] = stablehlo.add %[[MUL0]], %[[MUL1]]
// VJP: %[[DW0T:.*]] = stablehlo.dot_general %[[ADD]],
// VJP: %[[DW0:.*]] = stablehlo.transpose %[[DW0T]]
// VJP: %[[DX:.*]] = stablehlo.dot_general %[[ADD]],
// VJP: return %[[DX]], %[[DW0]], %[[DW1]]

// MAP-NOT: shuttle.
// MAP: %[[MUL:.*]] = stablehlo.multiply
// MAP: %[[ADD:.*]] = stablehlo.add %[[MUL]],
// MAP: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[ADD]]
// MAP: return %[[TRANSPOSE]]

// CONTRACT-NOT: shuttle.
// CONTRACT: %[[DOT:.*]] = stablehlo.dot_general
// CONTRACT: return %[[DOT]]
