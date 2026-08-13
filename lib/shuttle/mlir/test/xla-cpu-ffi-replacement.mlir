// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir --shuttle-cpu-executable-bundle-pipeline | FileCheck %s --check-prefix=CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-backward.mlir --shuttle-cpu-executable-bundle-pipeline | FileCheck %s --check-prefix=BWD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-composed.mlir --shuttle-cpu-executable-bundle-pipeline | FileCheck %s --check-prefix=COMPOSED
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-forward.mlir --shuttle-cpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=FAST-CALL
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-backward.mlir --shuttle-cpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=FAST-BWD
// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_81928ab3539c0f03-composed.mlir --shuttle-cpu-executable-bundle-fast-pipeline | FileCheck %s --check-prefix=FAST-COMPOSED
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir --shuttle-cpu-executable-bundle-pipeline 2>&1 | FileCheck %s --check-prefix=SHAPE

// This opt-in replacement test proves exact source erasure and transport
// binding. Runtime execution is covered by the native typed-FFI test.

// CALL-LABEL: module @jit_forward
// CALL-NOT: shuttle.
// CALL-COUNT-1: stablehlo.custom_call @shuttle.cpu.executable_bundle.v2
// CALL-SAME: api_version = 4 : i32
// CALL-SAME: backend_config = {bundle_bytes = "SHUTCPU\00
// CALL-SAME: bundle_sha256 = "99e63ac5a004f5abce7b88fc12bd0fbf9d8fc14785fc9ae87ca32781165d0c31"
// CALL-SAME: bundle_size = 6899 : i64
// CALL-SAME: transport_schema_version = 1 : i64
// CALL-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]
// CALL-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>]
// CALL: return
// CALL-NOT: shuttle.

// BWD-LABEL: module @jit_backward
// BWD-NOT: shuttle.
// BWD-COUNT-1: %[[BWD_CALL:[A-Za-z0-9_]+]]:2 = stablehlo.custom_call @shuttle.cpu.executable_bundle.v2(%arg0, %arg1, %arg2)
// BWD-SAME: api_version = 4 : i32
// BWD-SAME: backend_config = {bundle_bytes = "SHUTCPU\00
// BWD-SAME: transport_schema_version = 1 : i64
// BWD-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>]
// BWD-SAME: result_layouts = [dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>]
// BWD: return %[[BWD_CALL]]#1, %[[BWD_CALL]]#0 : tensor<7x13xbf16>, tensor<13xbf16>
// BWD-NOT: shuttle.

// COMPOSED-LABEL: module @jit_composed
// COMPOSED-NOT: shuttle.

// FAST-CALL-LABEL: module @jit_forward
// FAST-CALL-NOT: shuttle.
// FAST-CALL-COUNT-1: stablehlo.custom_call @shuttle.cpu.executable_bundle.v2
// FAST-CALL-SAME: api_version = 4 : i32
// FAST-CALL-SAME: backend_config = {bundle_bytes = "SHUTCPU\00
// FAST-CALL-SAME: bundle_size = 6899 : i64
// FAST-CALL-SAME: transport_schema_version = 1 : i64
// FAST-CALL-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]
// FAST-CALL-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>]
// FAST-CALL: return
// FAST-CALL-NOT: shuttle.

// FAST-BWD-LABEL: module @jit_backward
// FAST-BWD-NOT: shuttle.
// FAST-BWD-COUNT-1: %[[FAST_BWD_CALL:[A-Za-z0-9_]+]]:2 = stablehlo.custom_call @shuttle.cpu.executable_bundle.v2(%arg0, %arg1, %arg2)
// FAST-BWD-SAME: api_version = 4 : i32
// FAST-BWD-SAME: backend_config = {bundle_bytes = "SHUTCPU\00
// FAST-BWD-SAME: bundle_size = 16103 : i64
// FAST-BWD-SAME: transport_schema_version = 1 : i64
// FAST-BWD-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>]
// FAST-BWD-SAME: result_layouts = [dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>]
// FAST-BWD: return %[[FAST_BWD_CALL]]#1, %[[FAST_BWD_CALL]]#0 : tensor<7x13xbf16>, tensor<13xbf16>
// FAST-BWD-NOT: shuttle.

// FAST-COMPOSED-LABEL: module @jit_composed
// FAST-COMPOSED-NOT: shuttle.
// FAST-COMPOSED-COUNT-1: %[[FAST_COMPOSED_CALL:[A-Za-z0-9_]+]]:3 = stablehlo.custom_call @shuttle.cpu.executable_bundle.v2(%arg0, %arg1, %arg2)
// FAST-COMPOSED-SAME: api_version = 4 : i32
// FAST-COMPOSED-SAME: backend_config = {bundle_bytes = "SHUTCPU\00
// FAST-COMPOSED-SAME: bundle_size = 17090 : i64
// FAST-COMPOSED-SAME: transport_schema_version = 1 : i64
// FAST-COMPOSED-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>]
// FAST-COMPOSED-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>]
// FAST-COMPOSED: return %[[FAST_COMPOSED_CALL]]#0, %[[FAST_COMPOSED_CALL]]#2, %[[FAST_COMPOSED_CALL]]#1 : tensor<7x13xbf16>, tensor<7x13xbf16>, tensor<13xbf16>
// FAST-COMPOSED-NOT: shuttle.
// COMPOSED-COUNT-1: %[[COMPOSED_CALL:[A-Za-z0-9_]+]]:3 = stablehlo.custom_call @shuttle.cpu.executable_bundle.v2(%arg0, %arg1, %arg2)
// COMPOSED-SAME: api_version = 4 : i32
// COMPOSED-SAME: backend_config = {bundle_bytes = "SHUTCPU\00
// COMPOSED-SAME: transport_schema_version = 1 : i64
// COMPOSED-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>]
// COMPOSED-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>]
// COMPOSED: return %[[COMPOSED_CALL]]#0, %[[COMPOSED_CALL]]#2, %[[COMPOSED_CALL]]#1 : tensor<7x13xbf16>, tensor<7x13xbf16>, tensor<13xbf16>
// COMPOSED-NOT: shuttle.

// SHAPE: requires a bounded generated Map/Fold CPU bytecode subset
