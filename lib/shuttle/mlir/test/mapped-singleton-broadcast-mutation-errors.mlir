// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-swap-mapped-broadcast-axes --verify-each=false --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-mapped-singleton-broadcast.mlir 2>&1 | FileCheck %s --check-prefix=SWAP
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-duplicate-mapped-broadcast-axis --verify-each=false --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-mapped-singleton-broadcast.mlir 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-set-wrong-broadcast-divisor --verify-each=false --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-mapped-singleton-broadcast.mlir 2>&1 | FileCheck %s --check-prefix=DIVISOR
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-set-broadcast-literal-zero --verify-each=false --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-mapped-singleton-broadcast.mlir 2>&1 | FileCheck %s --check-prefix=LITERAL-ZERO
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-set-broadcast-composite-dividend --verify-each=false --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-mapped-singleton-broadcast.mlir 2>&1 | FileCheck %s --check-prefix=COMPOSITE
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-set-wrong-broadcast-result-extent --verify-each=false --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-mapped-singleton-broadcast.mlir 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-direct-expanded-singleton --verify-each=false --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-mapped-singleton-broadcast.mlir 2>&1 | FileCheck %s --check-prefix=SINGLETON
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-expand-nonsingleton-axis --verify-each=false --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-mapped-singleton-broadcast.mlir 2>&1 | FileCheck %s --check-prefix=NON-SINGLETON
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-add-map-attribute --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-mapped-singleton-broadcast.mlir 2>&1 | FileCheck %s --check-prefix=MAP-ATTR
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-add-map-yield-attribute --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-mapped-singleton-broadcast.mlir 2>&1 | FileCheck %s --check-prefix=YIELD-ATTR
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-replay-broadcast-as-reshape --verify-each=false --shuttle-lower-algebra-to-stablehlo --shuttle-verify-source-coverage %S/Inputs/f32-mapped-singleton-broadcast.mlir 2>&1 | FileCheck %s --check-prefix=BROADCAST-REPLAY
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-set-pointwise-constant-zero --shuttle-lower-algebra-to-stablehlo %S/Inputs/f32-map-shape-ops.mlir 2>&1 | FileCheck %s --check-prefix=CONSTANT-ZERO
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-test-replay-reshape-as-broadcast --verify-each=false --shuttle-lower-algebra-to-stablehlo --shuttle-verify-source-coverage %S/Inputs/f32-map-shape-ops.mlir 2>&1 | FileCheck %s --check-prefix=RESHAPE-REPLAY

// SWAP: broadcast input map dimensions must be unique, in range, and match direct or bounded singleton extents
// DUPLICATE: broadcast input map dimensions must be unique, in range, and match direct or bounded singleton extents
// DIVISOR: broadcast input map dimensions must be unique, in range, and match direct or bounded singleton extents
// LITERAL-ZERO: expanded singleton broadcast dimensions require a bounded-zero floordiv expression
// COMPOSITE: expanded singleton broadcast dimensions require a bounded-zero floordiv expression
// EXTENT: broadcast input map dimensions must be unique, in range, and match direct or bounded singleton extents
// SINGLETON: broadcast input map dimensions must be unique, in range, and match direct or bounded singleton extents
// NON-SINGLETON: broadcast input map dimensions must be unique, in range, and match direct or bounded singleton extents
// MAP-ATTR: has attributes with no StableHLO Map representation
// YIELD-ATTR: has attributes with no stablehlo.return representation
// BROADCAST-REPLAY: has incompatible singleton reshape types
// CONSTANT-ZERO: constant-zero input indexing is reserved for typed static singleton reshapes
// RESHAPE-REPLAY: has incompatible broadcast indexing maps
