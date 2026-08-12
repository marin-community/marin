# Shuttle XLA CPU typed-FFI consumer design

Status: implemented as an opt-in local Host CPU proof for the exact `7x13`
forward `SOURCE_ORDERED` slice. This remains outside the default pipeline and
is not GPU, representative-shape, performance, or scorecard acceptance.

## Decision

The first XLA consumer should replace the verified local `7x13` forward
`SOURCE_ORDERED` Shuttle region with one synchronous Host typed-FFI custom
call. The call receives XLA-owned external buffers and carries a canonical,
closed serialization of the stripped CPU executable bundle. One fixed generic
handler decodes and verifies that serialization during FFI instantiation, then
executes the immutable decoded state for each invocation.

The implementation binds XLA operands and results to the closed local CPU
bundle, transports and verifies the stripped bundle, instantiates immutable
handler state, replaces the selected graph without fallback, and separates the
new execution mode in pipeline ABI 6. The remaining exclusions in this design
still apply.

## Pinned upstream contract

The first implementation must compile against these exact sources rather than
an API-compatible substitute:

| Component | Version and revision | Contract source |
| --- | --- | --- |
| JAX and `jaxlib` | `0.10.1`, `619764c15117fbefc4ba13ab941871cb514c23f6` | [`jax/_src/ffi.py`](https://github.com/jax-ml/jax/blob/619764c15117fbefc4ba13ab941871cb514c23f6/jax/_src/ffi.py), [`jaxlib/ffi.cc`](https://github.com/jax-ml/jax/blob/619764c15117fbefc4ba13ab941871cb514c23f6/jaxlib/ffi.cc), and [`jaxlib/xla_client.py`](https://github.com/jax-ml/jax/blob/619764c15117fbefc4ba13ab941871cb514c23f6/jaxlib/xla_client.py) |
| XLA | `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69` | [`custom_call_thunk.cc`](https://github.com/openxla/xla/blob/9b635916ecc6df6efee62d8e4b0c7ef87ef84d69/xla/backends/cpu/runtime/custom_call_thunk.cc), [`call_frame.h`](https://github.com/openxla/xla/blob/9b635916ecc6df6efee62d8e4b0c7ef87ef84d69/xla/ffi/call_frame.h), [`execution_state.h`](https://github.com/openxla/xla/blob/9b635916ecc6df6efee62d8e4b0c7ef87ef84d69/xla/ffi/execution_state.h), and [`alignment.h`](https://github.com/openxla/xla/blob/9b635916ecc6df6efee62d8e4b0c7ef87ef84d69/xla/backends/cpu/alignment.h) |
| StableHLO | `1.17.0` | [`custom_call`](https://github.com/openxla/stablehlo/blob/806a6844dfd92cca1ce5391c86dca0ef9e952550/docs/spec.md#custom_call) |

`jax.ffi.register_ffi_target(..., platform="cpu", api_version=1)` registers a
typed-FFI handler; `jaxlib` maps the JAX `cpu` platform to XLA `Host`. The
StableHLO operation is a different versioned boundary: it uses
`api_version = 4` (`API_VERSION_TYPED_FFI`) and a dictionary
`backend_config`. Both values are exact requirements.

On the pinned XLA CPU runtime, the custom-call thunk resolves a handler on the
`Host` platform, parses the backend configuration, invokes the optional
instantiate stage once per compiled call site, and builds the argument and
result call-frame prototype from the HLO types and shapes. Each execution
updates that frame with addresses from XLA buffer-allocation slices. Arguments
are declared reads and results writes.

An `XLA_FFI_Buffer` exposes dtype, data pointer, rank, and dimensions. It does
not expose allocation extent, byte offset, strides, alignment, or alias
metadata. `ffi::Buffer::size_bytes()` is shape-derived rather than an observed
allocation span. The replacement pass must establish those facts from exact
static shapes, layouts, and XLA alias declarations; the handler must not infer
them from the FFI view. XLA CPU uses `EIGEN_MAX_ALIGN_BYTES` as `MinAlign()` for
assigned buffers, while the handler still checks each pointer against the
bundle slot's required natural alignment.

## First supported slice

The first gate is exactly one ordinary-JAX forward graph with `R=7` and
`H=13`, BF16 input/output, F32 reduction arithmetic, `SOURCE_ORDERED` numerics,
the current static `cpu_bytecode_v1` device module, and synchronous Host
execution. It retains the current CPU proof's restrictions:

- static Map/Fold domains no larger than 256 elements;
- one single-tile row Fold with fixed leaf order;
- BF16 and F32 values, row-major contiguous storage, zero byte offsets, and
  natural alignment;
- external read-only inputs, external write-only results, distinct temporary
  storage, and no aliasing, reuse, donation, or tuple values;
- completion only after every output write is visible.

The first slice does not include the `2048x4096` representative shape,
backward, composed, FAST rewrites, dynamic shapes, multi-tile Fold, input-output
aliasing, tuples, asynchronous FFI, GPU, or accelerator execution. A
post-conversion FAST-policy identity round trip is not execution evidence and
is not part of this gate.

## Exact operand and result binding

`shuttle.invocation_slot` schema 1 lacks an external FFI position. Inferring
that position from slot order would make a self-consistent reorder executable.
The next invocation-ABI schema must add a closed binding kind and index:

| Slot storage and access | Binding |
| --- | --- |
| external, read | exactly one contiguous FFI operand index |
| external, write | exactly one contiguous FFI result index |
| temporary | no external index |

For the frozen forward graph, the exact external projection is:

| Invocation slot | Tensor and required bytes | FFI binding |
| --- | --- | --- |
| `0` | input `tensor<7x13xbf16>`, 182 bytes | operand `0`, read |
| `1` | scale `tensor<13xbf16>`, 26 bytes | operand `1`, read |
| `20` | output `tensor<7x13xbf16>`, 182 bytes | result `0`, write |

Slots `2` through `19` are invocation-owned temporaries and have no FFI
index. The replacement pass derives this projection from region live-ins and
results; the literal ordinals document this fixture but do not select behavior.

The verifier rejects missing or duplicate indices, gaps, an index on a
temporary, an external slot without an index, read-write external slots,
results bound as operands, operands bound as results, tuple leaves, and any
unknown binding kind. It rederives the mapping from the selected region's
ordered live-ins and results and compares tensor type, required byte span,
strides, offset, alignment, address space, storage, access, alias group, and
reuse group. Changing this schema changes the invocation fingerprint and the
closed executable-bundle root.

At invocation, the handler validates each FFI dtype, rank, dimension, and
pointer alignment against its indexed slot. It computes each required range
with checked arithmetic and rejects overlap between every pair of external
operand and result ranges. XLA construction supplies the proof that each
shape-derived range fits its allocation. The custom call requests no
`output_operand_aliases`, so XLA must not intentionally reuse an input for the
result in this slice.

## Canonical transport and immutable state

The custom call's dictionary has exactly four attributes:

| Attribute | Exact type and value |
| --- | --- |
| `transport_schema_version` | signed 64-bit integer `1` |
| `bundle_bytes` | MLIR string attribute containing the opaque canonical bytes |
| `bundle_size` | signed 64-bit integer equal to the byte-string length |
| `bundle_sha256` | string containing 64 lowercase hexadecimal digits |

`bundle_bytes` contains only the stripped `shuttle.device_module`,
`shuttle.invocation_abi`, and `shuttle.executable_bundle`. A new
`shuttle.cpu.ffi.bundle.v1` op-specific serializer must enumerate every
semantic field covered by the three fingerprints. It writes a fixed magic and
version, then the three records in that order. Integers are fixed-width
little-endian; enums use versioned integer codes; strings and opaque bytecode
are unsigned-64 length-prefixed bytes; repeated children retain verified
ordinal order. There are no optional or extension records in version 1. The
decoder rejects unknown codes or trailing bytes, duplicate or missing roots,
noncanonical child order, and lengths above the checked implementation limit.
Generic MLIR text or bytecode is not the canonical wire format.

The transport length and SHA-256 cover the full canonical byte string. The
digest verifies transport integrity and cache identity. It never selects a
handler or code path. Construction serializes twice and requires byte equality
before emitting the custom call.

The FFI instantiate stage performs the equivalent of
`CpuExecutable::Load(exact_bytes)`: parse the closed format, recompute the
device-module, invocation-ABI, and bundle fingerprints, verify the stored
external projection, and construct immutable decoded execution state. Source
and materialization binding are rederived and checked by the compiler before
serialization; they are not reconstructed from a stripped runtime payload.
The CPU thunk owns the decoded state for the compiled executable's lifetime
and shares it across concurrent calls. Per-invocation temporary buffers,
external buffer views, status, and diagnostics remain invocation-owned.
`Execute` is a const operation over the immutable state and returns an
`absl::Status`, not an FFI `Future`, so completion is synchronous.

The current MLIR-module runtime entry point and diagnostic engine do not
establish immutable concurrent access. The implementation should split loading
and execution rather than place a mutex around mutable MLIR state. Concurrent
calls must not share temporary storage or mutate cached decoder/interpreter
state.

## Replacement and registration ownership

The native module transform, not a Python `jax.ffi.ffi_call` wrapper, owns the
ordinary-JAX replacement:

```text
ordinary JAX StableHLO
  -> source annotation, region formation, algebra and coverage
  -> materialization, schedule, CPU bundle construction and source rederivation
  -> canonical stripped-bundle serialization
  -> replace the one selected shuttle.region with stablehlo.custom_call
  -> erase source, algebra, materialization, schedule, and executable sidecars
  -> require StableHLO with no remaining Shuttle operations
```

The call target is one fixed schema name,
`shuttle.cpu.executable_bundle.v1`. Its operands are the region's ordered
external live-ins and its results replace the region's ordered SSA results. It
uses explicit row-major operand and result layouts, `api_version = 4`, the
typed dictionary transport above, `has_side_effect = false`, and no output
aliases. Unsupported structure fails the opt-in compilation; it does not fall
back to the StableHLO round trip after selecting this execution mode.

The handler must be linked into the rebuilt `jaxlib` for the pinned XLA tree.
Extend the existing always-linked Shuttle XLA registry adapter or add one
always-linked companion that calls XLA's static FFI registration for `Host`.
There is one process-wide registration owner and one fixed target. Workload,
module, function, fixture, source-operation, bundle digest, and shape names
must not appear in target selection. Two distinct valid bundle payloads under
the same target must execute their embedded semantics.

## Pipeline ABI and cache identity

Add a closed execution-mode field to `xla_shuttle_options`, with at least the
existing StableHLO round trip and the new CPU executable-bundle consumer as
distinct values, and bump `pipeline_abi_version` from 5 to 6. The wire schema
version changes only if the JSON field shape requires it. The consumer mode,
pipeline ABI, numerical policy, target profile, canonical transport schema,
invocation-ABI schema, and code-generator/runtime ABI must all contribute to
the exact options or bundle fingerprint as appropriate.

JAX's persistent compilation cache hashes the pre-transform canonical MLIR and
serialized compile options. The generated custom-call payload is created after
that input hash. The new execution mode and ABI must therefore be present in
the serialized compile options; embedding a different bundle digest after the
lookup cannot repair a stale cache hit. Any later semantic runtime or codegen
change requires another ABI bump.

Serialized executable reuse must reconstruct instantiate state only from the
embedded canonical attributes. It must not consult mutable files, an external
bundle registry, workload names, or a previous process's state.

## Behavior-first implementation gate

The implementation starts with failing behavior tests at the real XLA
boundary:

1. A native CPU PJRT test compiles a typed-FFI custom call containing the valid
   stripped `7x13` bundle, lets XLA own all external buffers, and compares the
   BF16 result bitwise with the independent forward reference.
2. A rebuilt-`jaxlib` test runs the ordinary-JAX `7x13` fixture without a
   Python FFI wrapper. It observes one fixed custom-call target, complete
   source-to-bundle coverage, no surviving Shuttle operation, and a bitwise
   match with a disabled-Shuttle process. A test linked only against stubs or
   repository headers is not this gate.
3. The same compiled executable runs concurrently with distinct inputs. Calls
   share immutable instantiate state, allocate distinct temporaries, and return
   the corresponding bitwise outputs without cross-call contamination.
4. Separate processes populate and reuse the persistent compilation cache.
   Exact options and ABI produce a hit; changing consumer mode or ABI produces
   a miss. Deserializing an executable reconstructs state from its embedded
   attributes and produces the same result.

Mutation tests reject an unknown backend attribute, wrong schema or ABI,
truncated or oversized bytes, byte-length or SHA mismatch, substituted child
or root fingerprints, missing/duplicate/reordered external bindings, dtype,
rank, shape, or layout changes, insufficient alignment, overlapping external
ranges, sidecars surviving replacement, a missing handler, an unsupported
shape or boundary, an asynchronous completion path, tuple/donation/reuse
metadata, and a serialized executable whose embedded payload is corrupt.

Renaming module, function, and source symbols must preserve the generic target
and semantic bundle fingerprint. Two valid bundles with different embedded
programs must execute differently through the same registered target. These
tests rule out name or digest dispatch while proving that the bundle remains
the executed semantics.

## Evidence and acceptance boundary

This document records an architecture decision derived from the pinned local
JAX/XLA sources and the checked-in CPU consumer proof. It is not source,
numerical, performance, hardware, or acceptance evidence. It changes no
scorecard cell. Target 1 remains unaccepted, and the representative
`2048x4096` cells receive no evidence from a `7x13` Host test.
