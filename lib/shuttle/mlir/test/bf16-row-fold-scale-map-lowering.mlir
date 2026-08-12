// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=RAW0
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=RAW0
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-backward.mlir | FileCheck %s --check-prefix=RAW1
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-backward.mlir | FileCheck %s --check-prefix=RAW1
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-composed.mlir | FileCheck %s --check-prefix=RAW2
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-composed.mlir | FileCheck %s --check-prefix=RAW2
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir | FileCheck %s --check-prefix=RAW3
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir | FileCheck %s --check-prefix=RAW3
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-backward.mlir | FileCheck %s --check-prefix=RAW4
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-backward.mlir | FileCheck %s --check-prefix=RAW4
// RUN: shuttle-test-opt --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-composed.mlir | FileCheck %s --check-prefix=RAW5
// RUN: shuttle-test-opt --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-composed.mlir | FileCheck %s --check-prefix=RAW5

// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=HOOK0
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=HOOK0
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-backward.mlir | FileCheck %s --check-prefix=HOOK1
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-backward.mlir | FileCheck %s --check-prefix=HOOK1
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-composed.mlir | FileCheck %s --check-prefix=HOOK2
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-composed.mlir | FileCheck %s --check-prefix=HOOK2
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir | FileCheck %s --check-prefix=HOOK3
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir | FileCheck %s --check-prefix=HOOK3
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-backward.mlir | FileCheck %s --check-prefix=HOOK4
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-backward.mlir | FileCheck %s --check-prefix=HOOK4
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-source-ordered-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-composed.mlir | FileCheck %s --check-prefix=HOOK5
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-stablehlo-fast-pipeline --shuttle-test-report-normalized-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-composed.mlir | FileCheck %s --check-prefix=HOOK5

// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=FWD
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-backward.mlir | FileCheck %s --check-prefix=BWD
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-verify-source-coverage --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-composed.mlir | FileCheck %s --check-prefix=COMPOSED
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir | FileCheck %s --check-prefix=RSQRT
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-backward.mlir | FileCheck %s --check-prefix=RESHAPE
// RUN: shuttle-test-opt --stablehlo-complex-math-expander --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-composed.mlir | FileCheck %s --check-prefix=RESHAPE

// RAW0: fe44a46cf9498564c07cde73bc401a90358a2a5d0e2e41a562af08179821673c
// RAW1: e0d4e2f4d9c009477b50fbf58ee41ad10fb9c13e5d456218652786c2ae014e66
// RAW2: 46819f276e1e28df892fd9fdb05e66fb9a622184562d5013dcfd7e751b39dfb3
// RAW3: 1455d28a7365a7cd6998109e9fd119a206a0d9d19bd6f3a70c678bd520159ebf
// RAW4: 46d595da395434d1192e2f056e8f9ae33d686e8bf6018df282b6e014909ece64
// RAW5: 63cd04b3ec537acf6765405936edc018c834ee65accb5e0ef1ba0082dc0ae58f
// HOOK0: b0a69e21331c5ebc86681c44e8a2ae00ff7d7f21bc1b95539ba0e2496678af7e
// HOOK1: f6f78a1bc29a210e6c0a4a4b632307c93a6410508bc24be76a99cad7205ada92
// HOOK2: 8c0e9200787e5fc2b91f36ac83b62ab8d30979287edb4fd2b6dfcb12fd73ecbe
// HOOK3: 541d20db0599d64d0d7e853b8a3a8ed00004a19b5a67aea01105609d9bc83616
// HOOK4: 048bb8757432cb7b2ed65666189861ce92f339d19981f93aa71d084ec2d5f7b1
// HOOK5: b23d862fc56af03ec3538f32c857b84c19982b19fd3bca27111d5ff43976fe98

// FWD-COUNT-4: #shuttle.map_semantics<broadcast_in_dim>
// FWD-COUNT-2: name = "stablehlo.broadcast_in_dim"
// BWD-COUNT-7: #shuttle.map_semantics<broadcast_in_dim>
// BWD-COUNT-3: name = "stablehlo.broadcast_in_dim"
// COMPOSED-COUNT-7: #shuttle.map_semantics<broadcast_in_dim>
// COMPOSED-COUNT-4: name = "stablehlo.broadcast_in_dim"
// RSQRT: #shuttle.map_semantics<pointwise>
// RSQRT: "math.rsqrt"
// RESHAPE-COUNT-2: #shuttle.map_semantics<reshape>
