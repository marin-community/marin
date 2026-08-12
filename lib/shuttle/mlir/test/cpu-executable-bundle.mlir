// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle --shuttle-verify-cpu-executable-bundle --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir | FileCheck %s --check-prefix=BUNDLE
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle --shuttle-verify-cpu-executable-bundle --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir | FileCheck %s --check-prefix=COUNT
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle --shuttle-verify-cpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir -o /dev/null > %t.original
// RUN: shuttle-test-opt --shuttle-test-rename-symbols --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle --shuttle-verify-cpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir -o /dev/null > %t.renamed
// RUN: diff %t.original %t.renamed
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle --shuttle-verify-cpu-executable-bundle --shuttle-test-report-executable-bundle-fingerprint %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir -o /dev/null > %t.fast
// RUN: not diff %t.original %t.fast
// RUN: not shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir 2>&1 | FileCheck %s --check-prefix=BOUNDS
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle --shuttle-verify-cpu-executable-bundle --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-backward.mlir | FileCheck %s --check-prefix=BWD-SO
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle --shuttle-verify-cpu-executable-bundle --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-backward.mlir | FileCheck %s --check-prefix=BWD-FAST
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle --shuttle-verify-cpu-executable-bundle --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-backward.mlir | FileCheck %s --check-prefix=BWDC
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle --shuttle-verify-cpu-executable-bundle --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-backward.mlir | FileCheck %s --check-prefix=BWD-FOLDS
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle --shuttle-verify-cpu-executable-bundle --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-composed.mlir | FileCheck %s --check-prefix=COMPOSED-SO
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-test-set-fast-region-policy --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle --shuttle-verify-cpu-executable-bundle --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-composed.mlir | FileCheck %s --check-prefix=COMPOSED-FAST
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle --shuttle-verify-cpu-executable-bundle --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-composed.mlir | FileCheck %s --check-prefix=COMPOSEDC
// RUN: shuttle-test-opt --shuttle-annotate-source --shuttle-form-structural-regions --shuttle-convert-stablehlo-to-algebra --shuttle-plan-row-fold-materialization --shuttle-plan-simt32-row-fold-schedule --shuttle-build-cpu-executable-bundle --shuttle-verify-cpu-executable-bundle --mlir-print-op-generic %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-composed.mlir | FileCheck %s --check-prefix=COMPOSED-FOLDS

// BUNDLE: "shuttle.device_module"
// BUNDLE-SAME: code_format = #shuttle.executable_code_format<cpu_bytecode_v1>
// BUNDLE-SAME: policy = #shuttle.policy<source_ordered>
// BUNDLE-SAME: schema_version = 1

// BWD-SO: "shuttle.device_module"() <{{.*}}policy = #shuttle.policy<source_ordered>
// BWD-SO: "shuttle.invocation_slot"() <{{.*}}access = #shuttle.executable_access<write>{{.*}}ordinal = 32
// BWD-SO: "shuttle.invocation_slot"() <{{.*}}access = #shuttle.executable_access<write>{{.*}}ordinal = 50
// BWD-FAST: "shuttle.device_module"() <{{.*}}policy = #shuttle.policy<fast>
// BWD-FAST: "shuttle.invocation_slot"() <{{.*}}access = #shuttle.executable_access<write>{{.*}}ordinal = 32
// BWD-FAST: "shuttle.invocation_slot"() <{{.*}}access = #shuttle.executable_access<write>{{.*}}ordinal = 50
// BWDC-COUNT-48: "shuttle.device_entry"
// BWDC-COUNT-51: "shuttle.invocation_slot"
// BWD-FOLDS-COUNT-10: reduction_order = #shuttle.schedule_reduction_order<tree_association_free_leaf_order_fixed>

// COMPOSED-SO: "shuttle.device_module"() <{{.*}}policy = #shuttle.policy<source_ordered>
// COMPOSED-SO: "shuttle.invocation_slot"() <{{.*}}access = #shuttle.executable_access<write>{{.*}}ordinal = 25
// COMPOSED-SO: "shuttle.invocation_slot"() <{{.*}}access = #shuttle.executable_access<write>{{.*}}ordinal = 35
// COMPOSED-SO: "shuttle.invocation_slot"() <{{.*}}access = #shuttle.executable_access<write>{{.*}}ordinal = 53
// COMPOSED-FAST: "shuttle.device_module"() <{{.*}}policy = #shuttle.policy<fast>
// COMPOSED-FAST: "shuttle.invocation_slot"() <{{.*}}access = #shuttle.executable_access<write>{{.*}}ordinal = 25
// COMPOSED-FAST: "shuttle.invocation_slot"() <{{.*}}access = #shuttle.executable_access<write>{{.*}}ordinal = 35
// COMPOSED-FAST: "shuttle.invocation_slot"() <{{.*}}access = #shuttle.executable_access<write>{{.*}}ordinal = 53
// COMPOSEDC-COUNT-51: "shuttle.device_entry"
// COMPOSEDC-COUNT-54: "shuttle.invocation_slot"
// COMPOSED-FOLDS-COUNT-10: reduction_order = #shuttle.schedule_reduction_order<tree_association_free_leaf_order_fixed>
// BOUNDS: requires a bounded generated Map/Fold CPU bytecode subset

// COUNT-COUNT-19: "shuttle.device_entry"
// COUNT-COUNT-21: "shuttle.invocation_slot"
// BUNDLE: "shuttle.device_entry"() <{{.*}}ordinal = 3{{.*}}predication = #shuttle.executable_predication<domain_bounds>{{.*}}reduction_order = #shuttle.schedule_reduction_order<tree_association_free_leaf_order_fixed>
// BUNDLE: "shuttle.invocation_abi"
// BUNDLE-SAME: schema_version = 1
// BUNDLE: "shuttle.invocation_slot"() <{{.*}}access = #shuttle.executable_access<read>{{.*}}address_space = #shuttle.executable_address_space<host>{{.*}}offset = 0{{.*}}ordinal = 0{{.*}}strides = array<i64: 26, 2>
// BUNDLE: "shuttle.invocation_slot"() <{{.*}}access = #shuttle.executable_access<write>{{.*}}ordinal = 20
// BUNDLE: "shuttle.executable_bundle"
// BUNDLE-SAME: completion = #shuttle.executable_completion<synchronous>
// BUNDLE-SAME: schema_version = 1
