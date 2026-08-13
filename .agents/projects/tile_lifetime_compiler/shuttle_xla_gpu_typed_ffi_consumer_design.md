# Shuttle XLA GPU typed-FFI consumer design

Status: design only for the exact BF16 `2048x4096` forward
`SOURCE_ORDERED` boundary at Marin commit
`146862564aec6e93ff55de92b8a84727f55a9a04`. No GPU executable, CUDA plugin,
H100 run, numerical result, performance result, or scorecard evidence exists.

## Decision

The first GPU consumer should preserve the ABI 9 materialization and schedule:
19 generated Map/Fold entries, 21 invocation slots, and 201,416,716 bytes of
distinct temporary storage. One API-version-4 StableHLO custom call carries a
canonical GPU executable bundle to one fixed CUDA typed-FFI target. The target
loads the generated PTX, allocates invocation-local device temporaries, and
enqueues the 19 entries on XLA's stream in dependency order.

The first GPU slice does not fuse the rowwise program into a named kernel.
Fusion would replace the source-bound 19-task schedule with a different
materialization and schedule. It needs its own source-rederivation and
numerical proof. Preserving the existing plan is the smallest execution
boundary that does not hide a new optimization inside the runtime handler.

The handler has no workload, module, function, fixture, shape-name, code-name,
or digest dispatch. The canonical bundle is the program. A digest verifies
bytes; it never selects bytes or behavior.

## Pinned upstream boundary

The design targets these exact sources:

| Component | Version and revision | Relevant contract |
| --- | --- | --- |
| JAX and `jaxlib` | `0.10.1`, `619764c15117fbefc4ba13ab941871cb514c23f6` | CUDA plugin startup in `jax_plugins/cuda/__init__.py` and handler capsules in `jaxlib/cuda/cuda_plugin_extension.cc` |
| XLA | `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69` | GPU `custom_call_thunk`, typed-FFI state, `KernelLoaderSpec`, `KernelThunk`, and CUDA PJRT custom-call registration |
| StableHLO | `1.17.0` | API-version-4 typed `custom_call` with dictionary `backend_config` |

The pinned GPU custom-call thunk invokes instantiate once per compiled call
site with null data pointers and the target GPU capability. It creates
execution-scoped Prepare and Initialize state before execution and destroys
that state after XLA execution completes. Initialize and Execute receive the
real XLA stream. This lifetime permits Prepare state to own device temporaries
across asynchronous kernel execution; the public FFI scratch allocator does
not, because it releases its allocations when the handler call returns.

`XLA_FFI_Buffer` data pointers are device addresses on this path. The handler
may compare pointer values for alignment and overlap, wrap them in
`stream_executor::DeviceAddressBase`, and pass them to a launch. It must never
dereference them on the host.

## Closed first slice

The accepted compiler input is exactly the ordinary-JAX forward signature:

```text
operand 0: tensor<2048x4096xbf16>, row-major, read-only
operand 1: tensor<4096xbf16>, row-major, read-only
result 0:  tensor<2048x4096xbf16>, row-major, write-only
```

The policy is `SOURCE_ORDERED`. The source has one FP32 row sum-square Fold,
FP32 scalar and elementwise Maps, `epsilon = 9.99999974e-6`, and one final
round-to-nearest-even BF16 conversion. The existing ABI 9 plan supplies 19
task ordinals, 21 buffer ordinals, one row Fold, 18 temporary slots, and the
`adjacent_balanced_initializer_last` Fold realization. The GPU source verifier
must rederive all of those facts from the source algebra, materialization plan,
and SIMT32 schedule before it emits PTX.

The first slice rejects `7x13`, backward, composed, FAST, dynamic shapes,
other dtypes, other CUDA capabilities, noncontiguous layouts, nonzero offsets,
aliasing, donation, reuse, tuples, collectives, multiple streams, command
buffers, external code lookup, and fallback to StableHLO after GPU mode has
been selected.

The code format targets CUDA compute capability 9.0. Capability 9.0 includes
more than one product configuration and does not identify an H100 by itself.
H100 remains an external acceptance coordinate, not a runtime selector or a
claim made by this design.

## Pipeline and replacement boundary

Add `gpu_executable_bundle` as a third closed `execution_mode` and advance
`pipeline_abi_version` from 9 to 10. ABI 10 admits GPU mode only with
`numerics = source_ordered`, the exact forward boundary above, and the exact
GPU schema versions below. ABI 9, `cpu_executable_bundle`, FAST, and
`stablehlo_round_trip` do not enter this path.

The module transform owns the replacement:

```text
ordinary JAX StableHLO
  -> source annotation, region formation, algebra, and coverage
  -> ABI 9 materialization and SIMT32 schedule
  -> GPU PTX generation and source rederivation
  -> canonical stripped GPU-bundle serialization
  -> one stablehlo.custom_call @shuttle.gpu.executable_bundle.v1
  -> erase source, algebra, materialization, schedule, device, ABI, and root
  -> require StableHLO with no Shuttle operations or sidecars
```

The custom call has `api_version = 4`, `has_side_effect = false`, explicit
row-major operand and result layouts, no called computations, and no output
aliases. Its ordered operands and results come from the verified external-slot
projection. Unsupported structure fails compilation. No fallback runs after
the transform has selected `gpu_executable_bundle`.

## Static schemas

The GPU path adds versions instead of reinterpreting the CPU schemas.

### Device module schema 3

`shuttle.device_module` schema 3 has code format `cuda_ptx_sm90_v1`, policy
`SOURCE_ORDERED`, the source-schedule fingerprint, one inline code byte array,
its lowercase SHA-256 digest, and the device fingerprint. The code array is a
concatenation of 19 complete PTX modules. Every module:

- is printable ASCII plus tab and line feed, contains no NUL or carriage
  return, and ends with one line feed;
- declares the fixed entry symbol `shuttle_entry` with no mangled aliases;
- declares exactly the pointer parameters implied by that entry's ordered
  input buffers followed by ordered output buffers;
- uses `.version 8.0`, `.target sm_90`, and `.address_size 64`;
- contains no external function, texture, surface, global-data, dynamic
  parallelism, cooperative launch, cluster launch, or device-side allocation.

Each schema-3 `shuttle.device_entry` retains the existing ordinal,
`source_task`, code offset and length, ordered input and output buffers, access
modes, dependencies, predication, reduction order, and code-object digest. It
adds these launch fields:

| Field | Type | First-slice rule |
| --- | --- | --- |
| `grid` | three unsigned 32-bit integers | exact SIMT32 grid with unused axes equal to 1 |
| `block` | three unsigned 16-bit integers | exact workgroup shape with unused axes equal to 1 and product at most 1,024 |
| `dynamic_shared_bytes` | unsigned 32-bit integer | exact scheduled scratch, at most 16 KiB |
| `kernel_arity` | unsigned 16-bit integer | exactly `inputs + outputs` |

The entry symbol is not serialized because every PTX module has the same
symbol. The code offset and positive length partition the device code bytes in
entry order. The runtime constructs one `KernelLoaderSpec` from each verified
slice and always requests `shuttle_entry`. An ordinal indexes the corresponding
decoded entry; it does not select a workload implementation.

The device verifier compares every entry with the source schedule: task
ordinal, buffer lists, access modes, dependency list, predication, reduction
constraint, grid, block, scratch, and generated PTX certificate must match. It
regenerates the PTX from the typed Map/Fold body and requires byte equality.
Rehashing substituted PTX cannot satisfy this check.

### Invocation ABI schema 3

`shuttle.invocation_abi` schema 3 retains the source-plan and source-schedule
fingerprints. It has exactly 21 contiguous slot ordinals. Each slot records
tensor type, checked required bytes, contiguous row-major byte strides, zero
offset, natural element alignment, Device address space, read/write access,
external or temporary storage, unique alias and reuse groups, and optional
operand/result binding.

External read slots map contiguously to operands 0 and 1. The external write
slot maps to result 0. Temporary slots have no binding index. All 21 ranges are
distinct; schema 3 allows no alias, reuse, donation, or read-write external
slot. The compiler compares every field with the materialization plan and the
selected region's ordered live-ins and result before erasing either source.

XLA owns the three external device allocations. The invocation ABI owns only
their static projection. FFI supplies shape-derived buffer views, not observed
allocation extents or strides. The replacement's exact shapes, layouts, and
empty alias list establish those properties at construction; Execute validates
dtype, rank, dimensions, pointer alignment, and non-overlapping pointer ranges
without treating the derived byte count as an observed allocation size.

### Executable bundle schema 2

`shuttle.executable_bundle` schema 2 binds the exact device-module and
invocation-ABI fingerprints to their common source-schedule fingerprint. Its
completion value is `stream_ordered`. That value means Execute returns after
all launches have been accepted on the supplied stream. XLA owns stream
completion and propagation of asynchronous device failure.

Schema 2 does not permit `synchronous`, an FFI Future, a second stream, host
callbacks, or command-buffer compatibility. The handler reports decode,
allocation, module-load, kernel-resolution, and launch-enqueue failures as
status. It does not synchronize the stream or poll CUDA launch status after
enqueue.

## Canonical transport schema 2

The custom-call dictionary has exactly four attributes:

| Attribute | Exact value |
| --- | --- |
| `transport_schema_version` | signed 64-bit integer `2` |
| `bundle_bytes` | opaque MLIR string containing the canonical bytes |
| `bundle_size` | signed 64-bit integer equal to the byte-string length |
| `bundle_sha256` | 64 lowercase hexadecimal SHA-256 digits over `bundle_bytes` |

Transport schema 2 starts with the eight bytes `SHUTGPU\0`, followed by the
unsigned little-endian 32-bit value `2`. It then serializes device module
schema 3, its entries, invocation ABI schema 3, its slots, and executable
bundle schema 2 in that order. Signed integers use fixed-width little-endian
two's-complement encoding. Enums use versioned one-byte codes. Strings and byte
arrays use an unsigned little-endian 64-bit length followed by exact bytes.
Repeated records remain in verified ordinal order. There are no optional
extensions, padding bytes, maps, or unknown records.

The record order is fixed:

```text
header:
  magic[8], transport_version:u32
device:
  schema:i64, code_format:u8, policy:u8,
  source_schedule_sha256:string, code:bytes, code_sha256:string,
  device_sha256:string, entry_count:u64
entry[entry_count]:
  ordinal:i64, source_task:i64, code_offset:i64, code_length:i64,
  input_buffers:i64_array, output_buffers:i64_array,
  input_accesses:u8_array, output_accesses:u8_array,
  dependencies:i64_array, predication:u8,
  has_reduction_order:u8, reduction_order:u8 if present,
  code_sha256:string,
  grid_x:u32, grid_y:u32, grid_z:u32,
  block_x:u16, block_y:u16, block_z:u16,
  dynamic_shared_bytes:u32, kernel_arity:u16
invocation:
  schema:i64, source_plan_sha256:string,
  source_schedule_sha256:string, invocation_sha256:string, slot_count:u64
slot[slot_count]:
  ordinal:i64, source_buffer:i64,
  element_type:u8, rank:u64, dimensions:i64[rank],
  required_bytes:i64, strides:i64_array, offset:i64, alignment:i64,
  address_space:u8, access:u8, storage:u8,
  alias_group:i64, reuse_group:i64, binding:u8,
  has_binding_index:u8, binding_index:i64 if present
bundle:
  schema:i64, source_schedule_sha256:string,
  device_sha256:string, invocation_sha256:string,
  completion:u8, bundle_sha256:string
```

Schema 2 assigns code-format value 2 to `cuda_ptx_sm90_v1` and completion
value 1 to `stream_ordered`. It retains the existing codes used by this slice:
`source_ordered = 0`, `read = 0`, `write = 1`, `device = 1`,
`external = 0`, `temporary = 1`, `none = 0`, `operand = 1`, `result = 2`,
`predication.none = 0`, `domain_bounds = 1`, and
`tree_association_free_leaf_order_fixed = 0`. The decoder rejects every other
value before constructing a record.

The schema-2 decoder checks every count, size, sum, and product before reserve,
copy, or allocation. It rejects truncation, trailing bytes, noncanonical
ordering, unknown enum values, invalid UTF-8 in fingerprints, non-lowercase
digests, duplicate or missing roots, and a deserialize-reserialize byte
mismatch. Allocation failures and C++ exceptions become `absl::Status` before
crossing the FFI boundary.

The closed resource policy is:

| Resource | Limit |
| --- | ---: |
| Complete transport | 16 MiB |
| Device entries or invocation slots | 256 each |
| Elements in any repeated integer/access record | 256 |
| Tensor rank | 8 |
| One PTX slice | 512 KiB |
| Aggregate PTX bytes | 8 MiB |
| One invocation slot | 32 MiB |
| Aggregate temporary bytes | 256 MiB |
| Aggregate task positions | 67,129,347 |
| Dynamic shared memory per entry | 16 KiB |

The first-slice projection further requires exactly 19 entries, 21 slots, 18
temporary allocations, and 201,416,716 temporary bytes. These checks are
structural limits. No workload string or digest chooses them.

## PTX generation and reduction order

The compiler emits PTX from each verified typed Map/Fold body. The runtime does
not contain a static row-normalization kernel or interpret scalar bytecode.
Map kernels flatten the scheduled domain, reconstruct affine input coordinates
with checked unsigned quotient and remainder operations, apply bounds
predication, evaluate the scalar instruction DAG, and write the bound output.
The permitted scalar operations are exact f32 constants, `add.rn.f32`,
`mul.rn.f32`, `div.rn.f32`, `sqrt.rn.f32` followed by division for rsqrt,
BF16-to-f32 bit extension, and integer round-to-nearest-even f32-to-BF16.
Generation disables contraction and flush-to-zero and emits no approximate
math opcode.

The `2048x4096` row Fold launches 2,048 blocks of 256 threads with 1,024 bytes
of shared memory. Thread `t` owns the contiguous 16-leaf interval
`[16t, 16t + 16)`. It forms an adjacent balanced tree over those leaves. The
256 partials are then combined as an adjacent balanced shared-memory tree in
increasing thread order. Thread 0 adds the rank-zero initializer once after the
root and writes the row result. This is the same 12-level adjacent tree over
4,096 ordered leaves as ABI 9's `adjacent_balanced_initializer_last`
realization. A strided `t + 256*k` lane assignment is forbidden because it
would change the leaf tree even if it used the same launch dimensions.

The PTX byte-equality certificate proves the selected instruction sequence and
lane mapping against the source task. Numerical acceptance still requires a
bitwise comparison with the pinned ordinary-JAX GPU execution on the same
architecture. Static reasoning and CPU parity do not establish H100 parity.

## Runtime owners and lifetimes

The CUDA handler bundle has instantiate, Prepare, Initialize, and Execute
stages.

Instantiate copies the four backend attributes, validates size and SHA-256,
decodes and canonically reserializes the bundle, validates the external
projection against the null-data FFI prototype, and checks that the target
capability is CUDA 9.0. Its immutable state owns the transport bytes, decoded
schemas, PTX slices, launch records, and binding projection for the compiled
call site's lifetime. The only mutable member is a mutex-protected vector of
loaded kernels keyed by `stream_executor::StreamExecutor*` and entry ordinal.
The runtime decoder produces plain immutable C++ records and does not retain an
MLIR context, compiler operation, diagnostic engine, or source-side plan.

Prepare obtains XLA's device allocator and device ordinal from the execution
context. It allocates one distinct device buffer for each of the 18 temporary
slots and returns execution-scoped state owning the resulting
`ScopedDeviceAddress` objects. Checked allocation stops at the exact
201,416,716-byte census and 256 MiB cap. Concurrent executions receive
different Prepare state and cannot share a temporary address. Destruction
occurs only after XLA reports that execution is complete.

Initialize receives the real stream. For each entry not loaded for that
stream's executor, it constructs an owning CUDA PTX `KernelLoaderSpec` with the
fixed symbol and exact arity, calls `StreamExecutor::LoadKernel`, and stores the
kernel under the mutex. The cache is safe for concurrent initialization on
multiple executors. Loaded kernels live with instantiate state, not with an
invocation.

Execute rechecks exact backend attributes against copied state, validates the
three external device views, and constructs a 21-slot address table from those
views and Prepare state. It walks entries in ordinal order, confirms each
dependency is earlier and enqueued, constructs kernel arguments from the
entry's ordered input buffers followed by output buffers, and calls XLA's
`ExecuteKernelOnStream` on the supplied stream. It performs no host
dereference, allocation, file access, module load, device synchronization,
stream creation, CUDA event creation, fallback, or semantic state mutation.

## CUDA plugin registration

The existing `ShuttleXlaRegistryAdapter` is linked into `_jax`. That linkage
owns the StableHLO module transform and Host handler, but it does not register a
typed handler inside the separate CUDA PJRT plugin.

Add a CUDA-only `ShuttleXlaCudaFfiHandlers` library to the pinned JAX build and
link it directly into `jaxlib/cuda:cuda_plugin_extension`. Its exported
function returns the four handler pointers for
`shuttle.gpu.executable_bundle.v1`. `cuda_plugin_extension.ffi_handlers()` adds
one dictionary entry containing the instantiate, Prepare, Initialize, and
Execute capsules. JAX's existing CUDA plugin startup registers that bundle
through `PJRT_Gpu_Register_Custom_Call` with platform `CUDA` and typed API
version 1.

The first slice registers no FFI traits. The pinned PJRT extension rejects
nonzero traits, so `kCmdBufferCompatible` would be both unsupported and false:
Prepare allocates invocation storage and Execute enqueues 19 kernels. CUDA-only
dependencies remain out of `_jax` and the CPU wheel.

## Serialization and cache identity

The GPU custom-call thunk proto serializes the fixed target, operand/result
slices, and exact backend attributes. Loaded kernels and device temporary
allocations are runtime state and are never serialized. If instantiate state
has no serializer, XLA omits it; executable deserialization invokes Instantiate
again from the embedded canonical attributes. Each execution creates fresh
Prepare state and Initialize reloads kernels for an executor when needed.

JAX's persistent compilation-cache lookup precedes the Shuttle module
transform. `execution_mode = gpu_executable_bundle` and
`pipeline_abi_version = 10` must therefore be in canonical compile options.
The GPU code-generator ABI, PTX ISA target, device/invocation/bundle schemas,
transport schema, runtime ABI, and Fold realization are fixed components of
ABI 10. Any semantic change to one of them requires a pipeline ABI bump; a new
post-lookup transport digest alone cannot invalidate an old cache entry.

An exact cache hit may still instantiate the handler and load or driver-JIT PTX
for the local executor. Evidence may claim unchanged persistent-cache bytes and
no Shuttle compilation observer events. It must not claim that CUDA module
loading or driver JIT was skipped unless separately observed.

## Red-first gates

The implementation starts with failing tests at each owner boundary.

Compiler and transport gates:

1. GPU mode produces one API-version-4 custom call with the fixed target,
   exact layouts and projection, schema-2 transport, 19 generated PTX entries,
   Device slots, stream-ordered completion, and no Shuttle operation or
   sidecar. CPU mode, ABI 9, FAST, other boundaries, and other shapes fail the
   GPU predicate.
2. Renaming modules, functions, source operations, and fixtures preserves the
   target and generated semantics. Two self-consistent test-only transports
   with different PTX bytes pass those exact bytes to `LoadKernel` through the
   same handler target. No target, entry symbol, workload, or digest switch
   exists.
3. Regeneration is deterministic. Each accepted source or schedule mutation
   either changes canonical PTX, transport, and fingerprints or fails source
   verification. Rehashed substituted PTX fails byte rederivation.
4. Pinned `ptxas -arch=sm_90` accepts every emitted slice. Mutations to opcode,
   parameter count, address size, target, symbol, barrier, predication, or
   terminal newline make the static gate fail or change canonical bytes.

Decoder and runtime mutations reject:

- wrong magic or schema, truncation, trailing bytes, noncanonical encodings,
  oversized lengths or counts, checked sum/product overflow, unknown enums or
  attributes, and allocation exceptions;
- changed source-plan or schedule binding, child or root fingerprint, code
  bytes or SHA, entry range, missing or duplicate entry, input/output buffer,
  access, dependency, predication, reduction realization, grid, block, shared
  memory, arity, or fixed symbol declaration;
- changed slot ordinal, tensor type, byte span, stride, offset, alignment,
  address space, access, storage, alias/reuse group, binding kind or index,
  aggregate temporary census, or external projection;
- Host buffers, null Execute pointers, misalignment, pointer-range overflow or
  overlap, an SM capability other than CUDA 9.0, module-load failure, missing
  entry symbol, and launch-enqueue failure;
- a Fold lane mapping that uses strided leaves, changes one tree edge, omits or
  duplicates one feature, adds the initializer early, skips a barrier, or
  allows a nonzero lane to write the row result.

Pinned-XLA gates compile the handler bundle through the CUDA PJRT registration
path, inspect `cuda_plugin_extension.ffi_handlers()`, instantiate from a valid
and corrupt golden transport, round-trip the custom-call thunk proto without
serializing loaded state, and run concurrent fake-executor lifecycle tests that
prove per-executor kernel caching and per-execution temporary ownership.
The rebuilt artifacts record SHA-256 for `_jax`, `cuda_plugin_extension`, the
CUDA PJRT plugin, the CUDA plugin wheel, and the canonical transport.

Persistent-cache gates use separate processes. Populate and reuse under exact
ABI 10 must produce one hit, no observer events on reuse, and unchanged cache
bytes. A pipeline/code-generator ABI mutation must miss. A bit flip in the
persisted transport must fail deserialization or instantiation; changing only a
self-consistent post-lookup payload is not accepted as a cache-key test.

Real device gates remain red until separately authorized and run. On the
declared H100 environment they must cover direct XLA execution, ordinary
`jax.jit`, repeated and concurrent invocations, allocation/load/launch error
paths, exact output bytes, and fresh-process cache reuse. `SOURCE_ORDERED` must
match the pinned ordinary-JAX H100 output bitwise and pass the frozen
independent-reference contract. Failure stops the slice. It does not authorize
post-observation reduction or PTX tuning under the same subject identity.

## Acceptance boundary

This design assigns owners and version boundaries using the checked-in ABI 9
CPU bundle and pinned JAX/XLA APIs. It changes no Target 1 status, cell, gate,
or evidence record. Static PTX tests, fake executors, rebuilt CUDA plugin
registration, and persistent-cache tests remain implementation gates. Only a
separately reviewed and authorized device run can supply H100 evidence.
