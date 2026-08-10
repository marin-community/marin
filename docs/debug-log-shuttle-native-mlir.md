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
- Native `@shuttle_mlir//:ShuttleDialect` preprocessing reached the next
  handwritten include failure at `llvm/ADT/SmallDenseSet.h`. Compilation
  stopped before semantic parsing, so bytecode-interface compile validation
  remains pending.

## Hypothesis 3

The handwritten sources use a header name that is absent from the pinned LLVM
revision. Exact LLVM commit `9a4faee1068c09efbf837cfb7b0f5693b24635f4`
defines `llvm::SmallDenseSet` in `llvm/ADT/DenseSet.h`; it has no
`llvm/ADT/SmallDenseSet.h`.

Both `ShuttleDialect.cc` and `Passes.cc` include the absent header and use
`llvm::SmallDenseSet`. Replace both includes with `llvm/ADT/DenseSet.h` so the
dialect library and the subsequent pass library use the pinned API.

## Results 3

- Every handwritten `llvm/...` and `mlir/...` include under
  `lib/shuttle/mlir` was checked against exact LLVM commit `9a4faee1068`. The
  two `SmallDenseSet.h` occurrences were the only missing paths.
- `llvm/ADT/DenseSet.h` exists at the pin and contains the
  `llvm::SmallDenseSet` template used by both sources.
- Repository formatting and lint gates run on the changed files.
- Native dialect compilation, the full driver build, and lit execution remain
  pending. This debugging task does not claim those gates passed.

## Hypothesis 4

Generated attribute definitions and operation declarations require complete
MLIR builder types, but Shuttle's owning public headers provide only the
forward declarations available through other IR headers.

The exact-pin dialect build reached semantic C++ parsing and failed first in
`ShuttleAttrs.cc.inc` because `mlir::Builder` was incomplete. Generated
attribute and operation code also uses `mlir::OpBuilder` and
`mlir::ImplicitLocOpBuilder`. At pinned LLVM commit `9a4faee1068`,
`mlir/IR/Builders.h` defines these builder types. Representative generated
dialect headers at the same pin include it before generated declarations.

`ShuttleAttrs.h` owns the generated attribute declarations and precedes
`ShuttleAttrs.cc.inc` in the dialect translation unit. `ShuttleOps.h` owns the
generated operation declarations. Include `mlir/IR/Builders.h` directly from
both headers. The existing `@llvm-project//mlir:IR` dependency already owns
`Builders.h`, so the Bazel graph needs no additional target.

## Results 4

- Exact pinned header and Bazel ownership confirm `Builders.h` supplies the
  incomplete types and is exported by `@llvm-project//mlir:IR`.
- Generated Enums, Attrs, and Ops include sites were audited against exact-pin
  dialect header patterns. No additional missing public prerequisite was
  identified from the retained diagnostics or pinned declarations.
- Repository formatting and lint gates run on the changed files.
- Native compilation remains pending. This debugging task does not claim the
  generated attribute or operation code compiles after this source fix.

## Hypothesis 5

The handwritten verifier uses an `ArrayAttr` convenience method that is absent
from the pinned MLIR API. Exact LLVM commit `9a4faee1068` gives `ArrayAttr`
iterators, `size`, `empty`, and bounds-checked `operator[]`, but no `front`.

Four verifier sites call `front()` on an `ArrayAttr`. Each site already proves
the attribute is nonempty: the shared indexing-map verifier returns before the
first access, the Map and Contract helpers reject empty maps, and the Contract
verifier guards its access with `empty()`. Replace only those four calls with
indexed access at zero.

## Results 5

- The exact-pin `ArrayAttr` declaration confirms `operator[]` is supported and
  asserts that the index is in bounds.
- All handwritten `front`, `back`, indexed-access, `empty`, and `size` calls in
  the native slice were classified by container type. The four failing calls
  were the only `ArrayAttr` front accesses; remaining front calls operate on
  MLIR Region or LLVM SmallVector containers supported at the pin.
- Repository formatting and lint gates run on the changed files.
- Native compilation remains pending. This debugging task does not claim the
  verifier compiles after this source fix.

## Hypothesis 6

The dialect translation unit declares generated dialect methods but does not
compile the generated dialect definitions. This can leave the explicit
`ShuttleDialect` library compile gate green while the first executable link
reports undefined constructor and type-ID symbols.

The exact native run compiled operation generation, `ShuttleDialect`, and
`ShuttlePasses`. Linking `shuttle-opt` then reported undefined
`ShuttleDialect::ShuttleDialect(MLIRContext *)` and
`TypeIDResolver<ShuttleDialect>::id`, with references from the driver and pass
registration code. The generated dialect-definition include contract should be
checked against the pinned MLIR examples before another run.

Include `ShuttleDialect.cc.inc` after the generated attribute and operation
definitions and before the `mlir::shuttle` namespace containing handwritten
methods.

## Results 6

- The generated-definition audit found one enum definition include, one
  attribute class-definition include, and one operation class-definition
  include. Attribute and operation files are additionally included under
  `GET_ATTRDEF_LIST` and `GET_OP_LIST`; those expansions register types and do
  not duplicate definitions.
- `ShuttleDialect.h.inc` is included once by the public header. The matching
  `ShuttleDialect.cc.inc` had no include and is now included once by the dialect
  implementation, following the exact-pin MLIR pattern.
- `@shuttle_mlir//:shuttle_ops_inc_gen`,
  `@shuttle_mlir//:ShuttleDialect`, and
  `@shuttle_mlir//:ShuttlePasses` passed against the exact XLA and LLVM pins.
- `@shuttle_mlir//:shuttle-opt` failed only at link time on the missing dialect
  constructor and type ID. The MLIR lit suite and all four patched XLA tests did
  not run.
- The native artifact is retained under
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-link/`.
- Repository formatting and lint gates run on the changed files.
- Native linking remains pending. This debugging task does not claim the
  unresolved dialect symbols are fixed until the exact-pin binary links.

## Hypothesis 7

The lit suite's tool labels are interpreted relative to the external Shuttle
repository when XLA loads it through `local_repository`. A main-workspace label
written as `//xla/...` therefore resolves as `@@shuttle_mlir//xla/...` during
analysis.

The exact native run built and linked `shuttle-opt`, then failed before any lit
test ran. Bazel reported that `@@shuttle_mlir//xla` has no BUILD file and traced
the dependency from `test/verifier-errors.mlir.test` to Shuttle's BUILD file.

Patch the unconditional OSS CPU runner datum to construct
`Label("//xla:sh_test_with_runfiles.py")` inside `lit.bzl`. A Label constructed
by the defining `.bzl` file is anchored to XLA instead of the external caller.
Keep the macro implementation and Shuttle's fixture wiring unchanged.

## Results 7

- The one-line patch applies cleanly to exact XLA commit `9b635916ecc6` and
  changes only the unconditional OSS CPU runner label in `xla/lit.bzl`.
- Shuttle still uses XLA's pinned `lit_test_suite`, `FileCheck`, `shuttle-opt`,
  and the complete fixture glob.
- `bazel build @shuttle_mlir//:mlir_tests` is documented as the analysis and
  executable preflight before running the suite.
- Operation generation, `ShuttleDialect`, `ShuttlePasses`, and `shuttle-opt`
  all passed against the exact pins.
- `@shuttle_mlir//:mlir_tests` failed during target analysis; no lit case ran,
  and the four patched XLA tests did not run.
- The retained evidence is under
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-lit-label/`.
- Repository formatting and lint gates run on the changed files.
- Native lit analysis and fixture execution remain pending. This debugging task
  does not claim any lit test passed.

## Hypothesis 8

The first label patch fixed the unconditional runner datum, but the pinned
macro evaluates CUDA-configured `_tools_on_path` dependencies even for this lit
suite. Its raw `//xla/tsl/cuda` strings again resolve inside `@shuttle_mlir`.

The exact build advanced past the runner label and failed at
`@@shuttle_mlir//xla/tsl/cuda`. Audit every raw runtime label in `xla/lit.bzl`
and wrap each XLA-owned label with `Label(...)`: default config, GPU specs,
runner data, CUDA runtime and NVSHMEM, NCCL, and Google config. Do not change
load statements or labels supplied by a macro caller.

## Results 8

- The expanded patch applies cleanly after patch 0001 at exact XLA commit
  `9b635916ecc6`.
- All XLA-owned raw runtime `//` labels found in the pinned macro are anchored
  with `Label(...)`. Load labels and caller inputs are unchanged.
- The internal `//third_party/py/lit:lit` string is intentionally unchanged:
  exact OSS execution overwrites it with the local `lit_custom_*` target before
  constructing a rule.
- Repository formatting and lint gates run on the changed files.
- Native lit analysis and fixture execution remain pending. This debugging task
  does not claim any lit test passed.

## Hypothesis 8

Anchoring only the unconditional lit runner datum is insufficient for
Shuttle's OSS CPU/custom-config invocation. The generated target still reaches
another string label in XLA's lit macro, and Bazel resolves that label against
the external Shuttle repository.

The exact native run applied patches `0001` and `0002` in order, reverse-checked
both, and verified `Label("//xla:sh_test_with_runfiles.py")` before Bazel. It
then passed operation generation, both library builds, and the complete
`shuttle-opt` build and link. The separate `mlir_tests` build gate failed on
`@@shuttle_mlir//xla/tsl/cuda`, referenced by the generated
`semantic-erasure-errors.mlir.test_tools_on_path` target.

Audit every string label evaluated by the exact OSS CPU/custom-config macro
path and construct each repository-owned dependency with `Label(...)`. Do not
limit the audit to the first failure exposed by Bazel.

## Results 8

- Remote proof confirms both reviewed XLA patches applied at exact XLA commit
  `9b635916ecc6`, both reverse-application checks passed, the combined diff was
  clean, and the anchored runner label was present.
- `@shuttle_mlir//:shuttle_ops_inc_gen`,
  `@shuttle_mlir//:ShuttleDialect`, `@shuttle_mlir//:ShuttlePasses`, and
  `@shuttle_mlir//:shuttle-opt` passed against the exact pins.
- `bazel build @shuttle_mlir//:mlir_tests` failed during analysis on
  `@@shuttle_mlir//xla/tsl/cuda`; lit execution and all four patched XLA tests
  did not run.
- The retained evidence is under
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-lit-cuda-label/`.
- This run used one submission with zero retries. No relaunch occurred.

## Hypothesis 9

The repository-relative lit labels are fixed: the exact native run built all
four compiler gates and `@shuttle_mlir//:mlir_tests`, then executed all 11 lit
fixtures. Eight passed. The remaining failures are local fixture or verifier
issues rather than another external-repository analysis failure.

`fail-closed.mlir` invokes LLVM's `not` executable, but the suite stages only
`shuttle-opt` and `FileCheck`. Exact LLVM commit `9a4faee1068` exports that tool
as `@llvm-project//llvm:not`; add it to the suite's tools.

The Map verifier reports a projected result map inside its per-map loop. That
preempts the more fundamental scalar-domain and unbound-domain diagnostics in
two existing negative cases. Record result projection while inspecting maps,
then diagnose it only after the global domain checks. This preserves all three
semantic rejections and makes each intended verifier branch reachable.

`no-shuttle-errors.mlir` uses line-based diagnostic annotations for a recursive
walk. The pass correctly stops at the nested `shuttle.yield`, prints a fused
location as `<unknown>`, and attaches a nested attribute error to its owning
module; none of those locations is part of the pass contract. Check the three
stable rejection messages through `not` and `FileCheck` without weakening the
pass.

## Results 9

- The exact native run passed operation generation, `ShuttleDialect`,
  `ShuttlePasses`, `shuttle-opt`, and the separate `mlir_tests` build gate.
- Lit executed all 11 fixtures: eight passed and `fail-closed.mlir`,
  `map-errors.mlir`, and `no-shuttle-errors.mlir` failed as described above.
- All 14 RUN lines across 11 fixtures were audited. Their executable tool set
  is exactly `shuttle-opt`, `FileCheck`, and `not`; the suite now stages all
  three.
- Source inspection confirms the deferred Map diagnostic still rejects
  projected result maps after scalar-only and unbound-domain validation.
- Native lit validation of these changes remains pending. No remote build was
  launched for this source-only fix.

## Follow-up

- [x] Run `@shuttle_mlir//:shuttle_ops_inc_gen` against the exact XLA pin.
- [x] Build the narrower `@shuttle_mlir//:ShuttleDialect` target.
- [x] Build `@shuttle_mlir//:ShuttlePasses`.
- [x] Link `@shuttle_mlir//:shuttle-opt`.
- [x] Make `bazel build @shuttle_mlir//:mlir_tests` pass analysis.
- [x] Run `@shuttle_mlir//:mlir_tests`.
- [ ] Run the four patched XLA tests.

## Hypothesis 9

The comprehensive lit-label patch should make the external Shuttle suite fully
analyzable, but the fixtures have not run against the pinned native lit
environment. Source-only parsing cannot validate lit tool runfiles, native
diagnostic locations, or verifier diagnostic precedence.

Apply both XLA patches in order, prove all seven anchored runtime labels, build
the suite separately, then execute it before the four XLA tests. Stop at the
first failed gate.

## Results 9

- Remote proof confirms both reviewed XLA patches applied at exact XLA commit
  `9b635916ecc6`, both reverse-application checks passed, the combined diff was
  clean, every expected anchored label was present, and the exact anchored
  XLA-owned runtime-label count was seven.
- `@shuttle_mlir//:shuttle_ops_inc_gen`,
  `@shuttle_mlir//:ShuttleDialect`, `@shuttle_mlir//:ShuttlePasses`,
  `@shuttle_mlir//:shuttle-opt`, and the separate
  `bazel build @shuttle_mlir//:mlir_tests` gate passed against the exact pins.
- Lit executed all 11 fixtures: eight passed. `fail-closed.mlir` lacked the
  LLVM `not` tool at runtime, `map-errors.mlir` reached the newer result-map
  verifier before two expected diagnostics, and `no-shuttle-errors.mlir`
  emitted matching error text at different MLIR source locations.
- The four patched XLA tests did not run because the runner stopped at the lit
  failure.
- The retained evidence is under
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-lit-execution/`.
- This run used one submission with zero retries. No relaunch occurred.

## Hypothesis 10

The lit runtime fixes should close the three fixture failures and expose the
first native result from the four patched XLA tests. Apply both XLA patches,
retain the seven-label proof, rerun all six Shuttle gates, and stop at the
first failure in the XLA test gate.

## Results 10

- Remote proof confirms both reviewed XLA patches applied at exact XLA commit
  `9b635916ecc6`, both reverse-application checks passed, the combined diff was
  clean, every expected anchored label was present, and the exact anchored
  XLA-owned runtime-label count was seven.
- `@shuttle_mlir//:shuttle_ops_inc_gen`,
  `@shuttle_mlir//:ShuttleDialect`, `@shuttle_mlir//:ShuttlePasses`,
  `@shuttle_mlir//:shuttle-opt`, and the separate
  `bazel build @shuttle_mlir//:mlir_tests` gate passed against the exact pins.
- Lit executed all 11 fixtures and all 11 passed, including the three failures
  retained by Results 9.
- The four-test XLA gate analyzed all four targets, then failed compiling
  `xla/pjrt/stablehlo_module_transform.cc`. The reviewed patch calls
  `getAttrs()` on `mlir::ModuleOp`, which has no such member at the pinned MLIR
  revision.
- Zero of four XLA tests executed: one failed to build and three were skipped.
- The retained evidence is under
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-xla-compile/`.
- This run used one submission with zero retries. No relaunch occurred.

## Hypothesis 11

The exact native run closed the Shuttle slice but exposed an exact-pin API
mismatch in the XLA hook. Clang emitted one error:

```text
xla/pjrt/stablehlo_module_transform.cc:98:33: error:
no member named 'getAttrs' in 'mlir::ModuleOp'
```

At exact LLVM commit `9a4faee1068`,
`OwningOpRef<ModuleOp>::operator->` returns `ModuleOp *`. The complete
attribute dictionary API belongs to the underlying `Operation`, available
through `OpState::getOperation()`. Make the commit path explicit instead of
chaining through the owning reference.

The callback receives a mutable clone and can also change the module's
location and ODS properties. `ModuleOp` stores `sym_name` and
`sym_visibility` as properties at this pin. A transactional success must
therefore copy properties, the complete attribute dictionary, the location,
and the body. Copy all four only after the callback succeeds and the clone
verifies. Exercise property removal and a location change in the success test.

The same owning-reference audit found four generic attribute calls in the new
test code. Those test translation units had not completed compilation. Route
them through `getOperation()` as well; callback-local `ModuleOp` arrows and
generated region helpers remain valid.

## Results 11

- The retained compiler log contains one diagnostic and no error-cap cascade.
  Three XLA tests were skipped and one failed to build; zero executed.
- Exact pinned headers confirm `Operation::getAttrDictionary`, `setAttrs`,
  `copyProperties`, `getPropertiesStorage`, `getLoc`, and `setLoc`, plus
  `OpState::getOperation`. `getAttrDictionary` materializes inherent
  properties, while `copyProperties` also preserves removal of an optional
  property.
- The remaining custom hook calls were checked against exact pinned XLA/MLIR
  declarations. The failing library translation unit reported no other API
  error; the uncompiled owning-reference attribute calls in both test files
  were corrected proactively.
- Patch `0001` was regenerated from exact XLA commit `9b635916ecc6`. Its applied
  files byte-match the audited source tree. Patches `0001` and `0002` apply in
  order, pass `diff --check`, reverse in order, and leave a clean exact-pin
  tree.
- Native compilation and execution of the four XLA tests remain pending. No
  remote build was launched for this source-only fix.

## Hypothesis 12

Copying the transformed module's properties, complete attribute dictionary,
location, and body through the pinned `Operation` APIs should close the prior
hook-library compile failure and expose the XLA test executables. Apply both
reviewed patches, rerun all six Shuttle gates, and stop at the first failure in
the four-test XLA gate.

## Results 12

- Remote proof confirms both reviewed XLA patches applied at exact XLA commit
  `9b635916ecc6`, both reverse-application checks passed, the combined diff was
  clean, every expected anchored label was present, and the exact anchored
  XLA-owned runtime-label count was seven.
- `@shuttle_mlir//:shuttle_ops_inc_gen`,
  `@shuttle_mlir//:ShuttleDialect`, `@shuttle_mlir//:ShuttlePasses`,
  `@shuttle_mlir//:shuttle-opt`, and the separate
  `bazel build @shuttle_mlir//:mlir_tests` gate passed against the exact pins.
- Lit executed all 11 fixtures and all 11 passed.
- The XLA hook implementation advanced past the prior `ModuleOp::getAttrs`
  failure. The four-test gate then failed compiling
  `stablehlo_module_transform_test.cc` because `TF_ASSERT_OK` and
  `TF_EXPECT_OK` were undeclared. Clang emitted 13 instances across that test
  file.
- Zero of four XLA tests executed: one failed to build and three had no status.
- The retained evidence is under
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-moduleattrs/`.
- This run used one submission with zero retries. No relaunch occurred.

## Hypothesis 13

The exact native run passed all six Shuttle gates, including 11/11 lit tests,
and compiled the updated transform library. The four-test XLA gate then failed
while compiling `stablehlo_module_transform_test.cc`. Clang reported 13 uses
of undeclared `TF_ASSERT_OK` or `TF_EXPECT_OK`; zero XLA tests executed.

At the pinned XLA revision, `xla/tsl/lib/core/status_test_util.h` defines those
two macros and `//xla/tsl/lib/core:status_test_util` exports the header. The
registry test has neither direct dependency. `mlir_to_hlo_test.cc` adds one
`TF_ASSERT_OK` and two `TF_EXPECT_OK` uses for the hook but also relies on
transitive exposure, so fix both targets before the next compile.

`TF_ASSERT_OK_AND_ASSIGN` is a separate macro from
`xla/tsl/platform/statusor.h`. The unregistered-transform test uses only that
macro and already declares the corresponding include and dependency.
`pjrt_executable_test.cc` already declares both status headers and targets.

## Results 13

- The retained compiler log contains exactly ten `TF_ASSERT_OK` and three
  `TF_EXPECT_OK` failures in `stablehlo_module_transform_test.cc`. The other
  three XLA test targets have no compile status in this run.
- Both `stablehlo_module_transform_test.cc` and `mlir_to_hlo_test.cc` now
  include `xla/tsl/lib/core/status_test_util.h` directly, and both Bazel targets
  depend directly on `//xla/tsl/lib/core:status_test_util`.
- All four patched XLA test sources were audited by macro family. The
  unregistered-transform and executable tests require no status-macro wiring
  change.
- Patch `0001` was regenerated from exact XLA commit `9b635916ecc6`. Its applied
  files byte-match the audited source. Patches `0001` and `0002` apply in order,
  pass `diff --check`, reverse in order, and leave a clean exact-pin tree.
- Native compilation and execution of the four XLA tests remain pending. No
  remote build was launched for this source-only fix.

## Hypothesis 14

Direct status-macro includes and Bazel dependencies should let all four patched
XLA test targets compile and execute after the six native Shuttle gates. Apply
both reviewed patches, retain the seven-label proof, and stop at the first
failed XLA test gate.

## Results 14

- Remote proof confirms both reviewed XLA patches applied at exact XLA commit
  `9b635916ecc6`, both reverse-application checks passed, the combined diff was
  clean, every expected anchored label was present, and the exact anchored
  XLA-owned runtime-label count was seven.
- `@shuttle_mlir//:shuttle_ops_inc_gen`,
  `@shuttle_mlir//:ShuttleDialect`, `@shuttle_mlir//:ShuttlePasses`,
  `@shuttle_mlir//:shuttle-opt`, and the separate
  `bazel build @shuttle_mlir//:mlir_tests` gate passed against the exact pins.
- Lit executed all 11 fixtures and all 11 passed.
- All four patched XLA targets compiled and executed. The registry,
  unregistered-transform, and PJRT executable targets passed.
  `mlir_to_hlo_test` passed eight of nine internal tests.
- `MlirToHloTest.EnabledModuleTransformReceivesOptions` failed because its
  fixture defines `@negate`; conversion requires a module with `@main`. The
  later `forwarded_options` false assertion is secondary to the conversion
  failure.
- The retained evidence is under
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-statusmacros/`.
- This run used one submission with zero retries. No relaunch occurred.

## Hypothesis 15

The status-macro fix should let all four XLA tests compile and execute. The
remaining conversion test must provide the `@main` entry function required by
the pinned StableHLO-to-HLO importer; naming its only function `@negate` tests a
malformed fixture instead of options forwarding.

Rename that function to `@main`. Keep the existing success assertion around
`MlirToXlaComputationWithPjRtOptions` and the exact
`{"numerics":"source_ordered"}` module-attribute assertion so the test still
checks conversion and serialized-options forwarding through the public path.

## Results 15

- The exact native run passed all six Shuttle gates, including 11/11 lit tests.
  All four XLA targets then compiled and executed: three targets passed, while
  `//xla/pjrt:mlir_to_hlo_test` passed eight of nine internal tests.
- `MlirToHloTest.EnabledModuleTransformReceivesOptions` was the only failure.
  The pinned importer reported `conversion requires module with main function`;
  its dumped fixture had `sym_name = "negate"`. The missing forwarded-options
  attribute was a secondary assertion failure after conversion failed.
- The fixture now names its only function `@main`. Its existing assertions
  still require successful conversion, the exact forwarded JSON string, and
  the expected compile-option override behavior.
- The two patched tests that call
  `MlirToXlaComputationWithPjRtOptions` were audited. The fail-closed
  unregistered-transform fixture already uses `@main`; no other patched
  conversion fixture has an entry-name mismatch.
- Patch `0001` was regenerated from exact XLA commit `9b635916ecc6`. Its applied
  files byte-match the audited source. Patches `0001` and `0002` apply in order,
  pass `diff --check`, reverse in order, and leave a clean exact-pin tree.
- Native execution of this fixture correction remains pending. No remote build
  was launched for this source-only fix.

## Results 16

- The exact native run passed all six Shuttle gates, including 11/11 lit tests.
  All four XLA targets compiled and executed: three targets passed, while
  `//xla/pjrt:mlir_to_hlo_test` passed eight of nine internal tests.
- `MlirToHloTest.EnabledModuleTransformReceivesOptions` was the only failure.
  The preceding conversion assertion passed after the fixture rename to
  `@main`. The post-call lookup of the `test.forwarded_options` module
  attribute at `mlir_to_hlo_test.cc:110` returned false.
- This result closes the missing-entry-point failure. It does not establish
  that options forwarded through the public conversion call are observable on
  the caller's original module.
- The retained evidence is under
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-mainfixture/`.
- This run used one submission with zero retries. No relaunch occurred.

## Hypothesis 17

The transform callback writes `test.forwarded_options` as an unknown
discardable module attribute. `StablehloModuleTransformRegistry::Run` commits
that attribute to the caller's module, but the successful conversion continues
through `ConvertStablehloToHloWithOptions`. At the pinned revision, that path
runs `StablehloSanitizeDiscardableAttributesPass`, whose module rewrite removes
every discardable attribute outside XLA's 14-name allowlist. The test attribute
is not in that allowlist, so its absence after conversion is expected and does
not show that option forwarding failed.

Observe the callback outside the mutable MLIR module. A process-lifetime test
observer should record every options string under a mutex, clear its state
before conversion, and require the observation vector to contain exactly the
expected JSON string. This distinguishes no callback, duplicate callbacks, and
incorrect data without a summary boolean or a transient module attribute.

## Results 17

- The test transform now records received options in a mutex-protected vector.
  `ElementsAre("{\"numerics\":\"source_ordered\"}")` requires exactly one
  callback with the exact serialized options.
- The observer uses `absl::NoDestructor`, matching the global registry's process
  lifetime. The test resets the vector before conversion, so repeated test
  execution cannot reuse a prior observation.
- `mlir_to_hlo_test` contains the only enabled global-transform conversion in
  its test executable. Registration still occurs before the registry's first
  `Run`; the registry's cached registration status remains valid after sealing
  for repeated execution of the same test.
- The conversion-success assertion and the fast-math override deferral checks
  are unchanged. The callback no longer mutates the MLIR module for test
  observation.
- Patch `0001` was regenerated from exact XLA commit `9b635916ecc6`. Its applied
  files byte-match the audited source. Patches `0001` and `0002` apply in order,
  pass `diff --check`, reverse in order, and leave a clean exact-pin tree.
- Native execution of the observer-based test remains pending. No remote build
  was launched for this source-only fix.
