// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-contract_map_79045ff9bdc7c783-primal-vjp.mlir | FileCheck %s --check-prefix=CASE0
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-contract_map_9836cdbed389db24-primal-vjp.mlir | FileCheck %s --check-prefix=CASE1
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-contract_map_b4c693e52135022a-primal-vjp.mlir | FileCheck %s --check-prefix=CASE2
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-contract_map_eb4a28b4408cfb90-primal-vjp.mlir | FileCheck %s --check-prefix=CASE3
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-contract_map_79045ff9bdc7c783-primal-vjp.mlir | FileCheck %s --check-prefix=HOOK0
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-contract_map_9836cdbed389db24-primal-vjp.mlir | FileCheck %s --check-prefix=HOOK1
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-contract_map_b4c693e52135022a-primal-vjp.mlir | FileCheck %s --check-prefix=HOOK2
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-contract_map_eb4a28b4408cfb90-primal-vjp.mlir | FileCheck %s --check-prefix=HOOK3
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-contract_map_79045ff9bdc7c783-primal-vjp.mlir | FileCheck %s --check-prefix=COVERAGE0
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-contract_map_9836cdbed389db24-primal-vjp.mlir | FileCheck %s --check-prefix=COVERAGE1
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-contract_map_b4c693e52135022a-primal-vjp.mlir | FileCheck %s --check-prefix=COVERAGE2
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --shuttle-verify-semantic-erasure --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-contract_map_eb4a28b4408cfb90-primal-vjp.mlir | FileCheck %s --check-prefix=COVERAGE3
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra %S/Inputs/jax-0.10.1-bf16-contract_map_9836cdbed389db24-primal-vjp.mlir | FileCheck %s --check-prefix=ALGEBRA

// These are the frozen raw normalized hashes from the independent ordinary-JAX
// fixture oracle. Equality proves that source-ordered algebra round-trips the
// complete primal+VJP graphs without depending on case names or dimensions.
// CASE0: b70e564634b02d69456293f093a9dfe0ba098545f5cd37a35eb1bf949c1e0c4d
// CASE0-NOT: shuttle.
// CASE1: e0ddb89478782c6a93fe75140605b9abbe039cf305e71068df067f3d576a5d34
// CASE1-NOT: shuttle.
// CASE2: f8f4d759f6c73b8e19ea42d0dbc790c1b3e521f1ef2bdaa8978f7572bad06617
// CASE2-NOT: shuttle.
// CASE3: a1b6d2e277d12e0ba66d8aa06c266c99b8976a60e29f672ab6bc4f985ee1a916
// CASE3-NOT: shuttle.

// The same exact round trip is required at the XLA hook boundary after the
// independently pinned preprocessing pass.
// HOOK0: 5f72b105e8ddd6cb87362a694f4ade2a300aecca692564a146d2b22abf92eb80
// HOOK0-NOT: shuttle.
// HOOK1: a221dca8d47fa204f073123924b8194ce30dd354234d91f58e8055cff9d74858
// HOOK1-NOT: shuttle.
// HOOK2: f8f4d759f6c73b8e19ea42d0dbc790c1b3e521f1ef2bdaa8978f7572bad06617
// HOOK2-NOT: shuttle.
// HOOK3: 617a3cc4e636241e4459b255affae229b0a0d1b3c6974d858578cd655f4a387e
// HOOK3-NOT: shuttle.

// Hash equality alone could also describe a deliberately excluded no-op. Each
// hook-boundary graph must therefore have an empty exclusion inventory.
// COVERAGE0: shuttle.coverage_manifest
// COVERAGE0-SAME: excluded = []
// COVERAGE1: shuttle.coverage_manifest
// COVERAGE1-SAME: excluded = []
// COVERAGE2: shuttle.coverage_manifest
// COVERAGE2-SAME: excluded = []
// COVERAGE3: shuttle.coverage_manifest
// COVERAGE3-SAME: excluded = []

// A representative sigmoid graph must be entirely represented by generic
// algebra, including typed conversion provenance and FP32 Contract results.
// ALGEBRA: "shuttle.contract"
// ALGEBRA-SAME: ({{.*}}tensor<43x104xbf16>, tensor<104x72xbf16>) -> tensor<43x72xf32>
// ALGEBRA: "shuttle.scalar_convert"
// ALGEBRA-SAME: semantics = #shuttle.scalar_convert_semantics<round_nearest_even>
// ALGEBRA: math.exp
// ALGEBRA: arith.divf
// ALGEBRA: "shuttle.scalar_convert"({{.*}}) <{semantics = #shuttle.scalar_convert_semantics<exact>}> : (f32) -> f32
