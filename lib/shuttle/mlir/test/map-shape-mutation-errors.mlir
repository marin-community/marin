// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-unorder-broadcast-map --verify-each=false --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-map-shape-ops.mlir 2>&1 | FileCheck %s --check-prefix=UNORDERED
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-duplicate-broadcast-dimension --verify-each=false --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-map-shape-ops.mlir 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-ambiguate-reshape-map --verify-each=false --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-map-shape-ops.mlir 2>&1 | FileCheck %s --check-prefix=RESHAPE
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-add-map-attribute --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-map-shape-ops.mlir 2>&1 | FileCheck %s --check-prefix=MAP-ATTR
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-add-map-yield-attribute --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-map-shape-ops.mlir 2>&1 | FileCheck %s --check-prefix=YIELD-ATTR
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-set-structural-map-semantic --verify-each=false --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-map-shape-ops.mlir 2>&1 | FileCheck %s --check-prefix=SEMANTICS

// UNORDERED: broadcast input map dimensions must be ordered, unique, and non-expanding
// DUPLICATE: broadcast input map dimensions must be ordered, unique, and non-expanding
// RESHAPE: has an ambiguous singleton reshape indexing map
// MAP-ATTR: has attributes with no StableHLO Map representation
// YIELD-ATTR: has attributes with no stablehlo.return representation
// SEMANTICS: structural Map semantics require an empty scalar body
