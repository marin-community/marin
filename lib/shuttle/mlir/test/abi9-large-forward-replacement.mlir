// RUN: shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir --shuttle-cpu-executable-bundle-pipeline | FileCheck %s --check-prefix=LARGE
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-backward.mlir --shuttle-cpu-executable-bundle-pipeline 2>&1 | FileCheck %s --check-prefix=VJP
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-composed.mlir --shuttle-cpu-executable-bundle-pipeline 2>&1 | FileCheck %s --check-prefix=COMPOSED
// RUN: not shuttle-test-opt %S/Inputs/jax-0.10.1-bf16-row_fold_scale_44d152ecc3e9ff18-forward.mlir --shuttle-cpu-executable-bundle-fast-pipeline 2>&1 | FileCheck %s --check-prefix=FAST

// ABI 9 admits only the representative-shape SOURCE_ORDERED forward graph.
// Device schema 2 and cpu_bytecode_v2 define a balanced, adjacent,
// leaf-order-preserving Fold and merge the single initializer after reducing
// the 4096 data leaves. Transport schema 1 already carries both identities.

// LARGE-LABEL: module @jit_forward
// LARGE-NOT: shuttle.
// LARGE-COUNT-1: stablehlo.custom_call @shuttle.cpu.executable_bundle.v3
// LARGE-SAME: api_version = 4 : i32
// LARGE-SAME: backend_config = {bundle_bytes = "SHUTCPU\00
// LARGE-SAME: transport_schema_version = 1 : i64
// LARGE-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]
// LARGE-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>]
// LARGE: return
// LARGE-NOT: shuttle.

// VJP: requires the bounded cpu_bytecode_v2 forward subset
// COMPOSED: requires the bounded cpu_bytecode_v2 forward subset
// FAST: cpu_bytecode_v2 representative-shape execution requires source_ordered
