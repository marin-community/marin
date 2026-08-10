# Debugging log for Shuttle native MLIR

## Goal

Restore TableGen generation for the native Shuttle dialect against XLA commit
`9b635916ecc6df6efee62d8e4b0c7ef87ef84d69` without broadening its control-flow
semantics.

## Initial status

The native build reached Bazel analysis for `@shuttle_mlir//:shuttle-opt` and
failed in both generated-operation actions:

```text
external/shuttle_mlir/include/shuttle/IR/ShuttleOps.td:69:34:
Variable not defined: 'ReturnLike'
```

The failing commands were `mlir-tblgen -gen-op-decls` and
`mlir-tblgen -gen-op-defs`. No C++ compilation or lit tests ran.

## Investigation

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

## Validation

- Exact pinned LLVM sources inspected for `Terminator`, `ReturnLike`, and the
  region-branch interface contract.
- Repository formatting and lint gates run on the changed files.
- Native TableGen, C++ compilation, and lit execution remain pending on a host
  with the exact pinned XLA/LLVM source graph. This debugging task does not
  claim those gates passed.

## Follow-up

- [ ] Run `@shuttle_mlir//:shuttle_ops_inc_gen` against the exact XLA pin.
- [ ] If generation succeeds, build `@shuttle_mlir//:shuttle-opt`.
- [ ] If compilation succeeds, run `@shuttle_mlir//:mlir_tests`.
