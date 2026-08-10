# Debugging log for Shuttle native MLIR

## Goal

Restore staged native compilation for the Shuttle dialect against XLA commit
`9b635916ecc6df6efee62d8e4b0c7ef87ef84d69` while keeping each checkpoint and
its evidence explicit.

## Initial status

The native build reached Bazel analysis for `@shuttle_mlir//:shuttle-opt` and
failed in both generated-operation actions:

```text
external/shuttle_mlir/include/shuttle/IR/ShuttleOps.td:69:34:
Variable not defined: 'ReturnLike'
```

The failing commands were `mlir-tblgen -gen-op-decls` and
`mlir-tblgen -gen-op-defs`. No C++ compilation or lit tests ran.

## Hypothesis 1

XLA pins LLVM to `9a4faee1068c09efbf837cfb7b0f5693b24635f4`. At that exact
revision, `Terminator` is defined in `mlir/IR/OpBase.td`, which Shuttle already
includes. `ReturnLike` is defined in
`mlir/Interfaces/ControlFlowInterfaces.td`; it also declares
`RegionBranchTerminatorOpInterface` methods. `ShuttleOps.td` did not include
that file, and Shuttle's region operations do not implement the corresponding
region-branch control-flow interface.

The `ReturnLike` trait was therefore both unavailable and stronger than the
current dialect contract. Adding the missing include and dependency would make
an unsupported control-flow claim.

## Fix

Remove `ReturnLike` from `shuttle.yield` and retain `Pure` and `Terminator`.
Yield structure remains explicit in three independent contracts:

- `Shuttle_YieldOp` has the core `Terminator` ODS trait.
- `shuttle.region`, `shuttle.map`, and `shuttle.fold` use
  `SingleBlockImplicitTerminator<YieldOp>`.
- Their C++ region verifier requires the block terminator to be
  `shuttle.yield`.

The narrow generated-operations target is now documented as the first native
preflight:

```bash
bazel build @shuttle_mlir//:shuttle_ops_inc_gen
```

This invokes both TableGen actions and catches unavailable ODS traits before
the larger `shuttle-opt` build. A host-only textual assertion was not added:
it would duplicate TableGen name resolution and could pass while the pinned
toolchain still fails.

## Results

- Exact pinned LLVM sources inspected for `Terminator`, `ReturnLike`, and the
  region-branch interface contract.
- Repository formatting and lint gates run on the changed files.
- Native `@shuttle_mlir//:shuttle_ops_inc_gen` passed against the exact XLA pin
  after removing `ReturnLike`.
- The subsequent `@shuttle_mlir//:shuttle-opt` build reached C++ compilation
  and failed first in generated `ShuttleOps.h.inc`: `BytecodeOpInterface` was
  not declared in namespace `mlir`. Missing `DialectBytecodeReader` and
  `DialectBytecodeWriter` declarations and operation-base errors followed.

## Hypothesis 2

The generated operation header requires MLIR's bytecode operation interface,
but Shuttle's public operation header does not include it and its Bazel library
does not declare the corresponding dependency.

At exact LLVM revision `9a4faee1068c09efbf837cfb7b0f5693b24635f4`,
`OpDefinitionsGen.cpp` adds `mlir::BytecodeOpInterface::Trait` to every
operation with a non-empty generated properties struct. It also emits property
serialization methods using `mlir::DialectBytecodeReader` and
`mlir::DialectBytecodeWriter`. The pinned
`mlir/Bytecode/BytecodeOpInterface.h` header provides the interface and includes
`BytecodeImplementation.h`, which declares both reader and writer types.

Representative dialect headers at the same pin, including Arith and EmitC,
include `mlir/Bytecode/BytecodeOpInterface.h` before their generated operation
declarations. The pinned Bazel overlay exports that header and its generated
interface through `@llvm-project//mlir:BytecodeOpInterface`.

## Changes to make

- Include `mlir/Bytecode/BytecodeOpInterface.h` from Shuttle's public operation
  header.
- Add the exact pinned `BytecodeOpInterface` Bazel dependency to
  `ShuttleDialect`.
- Document `@shuttle_mlir//:ShuttleDialect` as the compile-only preflight
  between TableGen generation and the full driver build.

## Results 2

- Source inspection confirms the include supplies all three initially missing
  declarations and the Bazel target supplies the public header plus generated
  interface implementation.
- Repository formatting and lint gates run on the changed files.
- Native dialect compilation and lit execution remain pending on the exact-pin
  build host. This debugging task does not claim those gates passed.

## Follow-up

- [x] Run `@shuttle_mlir//:shuttle_ops_inc_gen` against the exact XLA pin.
- [ ] Build the narrower `@shuttle_mlir//:ShuttleDialect` target.
- [ ] If the dialect compiles, build `@shuttle_mlir//:shuttle-opt`.
- [ ] If compilation succeeds, run `@shuttle_mlir//:mlir_tests`.
