# Target 1 executable-device boundary

Status: bounded opt-in CPU consumer proof. No GPU, XLA runtime, or production
execution path is added by this slice.

## Decision

The compiler now has a closed local boundary below `shuttle.schedule_plan` for
one static `7x13` row-Fold graph. It generates typed CPU bytecode from the
actual Shuttle Map/Fold bodies, binds that code to a static byte-buffer ABI,
and executes the stripped bundle synchronously in a local interpreter. The
consumer does not inspect StableHLO, Shuttle algebra, task semantic digests, or
source names after bundle construction.

This is a CPU consumer proof, not accelerator code generation. Adding NVVM or a
custom call would still create an orphan backend: the XLA module-transform hook
owns StableHLO-to-StableHLO rewriting only and has no buffer-assignment result,
executable build callback, runtime registration, or invocation-time buffer
table. The core pipeline still removes all Shuttle operations before returning
StableHLO. The new build, source-verification, and runtime entry points are
opt-in and remain outside that pipeline, so pipeline ABI 5 and its cache
identity are unchanged.

An accelerator implementation must still begin with an explicit XLA/runtime
owner and a target code-object owner. Those choices cannot be inferred from
tensor types, affine maps, reduction dimensions, numerical policy, the CPU
proof, or the abstract `simt32` profile.

## Implemented CPU boundary

The static boundary has three closed operations:

- `shuttle.device_module` contains inline `cpu_bytecode_v1` bytes, their
  SHA-256 digest, one structural entry per task, exact byte ranges, ordered
  buffer references, access declarations, dependencies, predication, reduction
  order, policy, and a fingerprint over every semantic attribute and child;
- `shuttle.invocation_abi` contains one slot per materialization buffer with
  static tensor type, required span, contiguous byte strides, zero offset,
  natural alignment, host address space, read/write access, external or
  temporary ownership, and unique alias and reuse groups;
- `shuttle.executable_bundle` is the closed root binding the exact device-module
  and invocation-ABI fingerprints to their shared schedule and synchronous
  completion policy.

The generator consumes the complete scalar regions and affine indexing maps of
each Map and the exact Fold combiner. Its instruction set is limited to f32
constants, add, multiply, divide, rsqrt, exact BF16-to-f32 conversion, and
round-to-nearest-even f32-to-BF16 conversion. The row Fold is an explicit
left-to-right leaf traversal initialized from its rank-zero buffer. This is one
legal realization of `tree_association_free_leaf_order_fixed`; the test-only
post-conversion FAST policy does not grant another reduction order or
instruction set.

The local consumer receives raw mutable byte spans for external buffers,
allocates distinct host byte spans for temporaries, validates size, alignment,
access, no-alias, bytecode digest, root binding, and dependency completion, and
uses only ABI offsets and strides for loads and stores. It executes exact
domain loops with bounds predication and returns only after every entry and
output write completes. Code and temporary storage therefore remain owned for
the whole synchronous call.

The slice deliberately accepts only static Map/Fold task domains no larger than
256 elements, one single-tile row Fold, BF16/f32 values, host buffers, zero
offsets, natural alignment, unique alias/reuse groups, and one output per task.
This includes the frozen `7x13` forward graph and excludes the `2048x4096`
graph, multi-tile Fold realization, backward/composed graphs, dynamic shapes,
other types, reuse, donation, and asynchronous execution. These restrictions
are structural; module, function, workload, fixture, and source-operation names
never select code or runtime behavior.

## Missing contracts

### Buffer binding

Ranked tensor types supply shape and element type. They do not supply a base
pointer, byte strides, byte offset, required alignment, address space, alias
set, or allocation owner. `shuttle.materialization_buffer` supplies logical
producer, consumers, lifetime, and external-versus-temporary storage. It does
not authorize temporary reuse or prove that two external values do not alias.

An executable boundary needs an invocation-time buffer table with, for every
buffer ordinal:

- the exact tensor type and byte span, with checked multiplication;
- an argument or allocation slot and the component that owns that slot;
- address space, byte offset, strides, and minimum alignment;
- alias and donation constraints;
- tuple-leaf/result ownership and destination-passing relationships;
- allocation lifetime and reuse group;
- read, write, or read-write access.

The CPU proof forbids reuse, donation, nonzero offsets, and overlapping alias
groups. That is an explicit local policy, not a fact derivable from StableHLO.
A typed XLA FFI handler can observe each argument's dtype, data
pointer, rank, and dimensions. It can recompute a shape-derived required byte
count and check pointer alignment, but that count is not an observable
allocation extent and cannot prove the allocation is large enough. Contiguous,
zero-offset storage and its layout, alias, and donation guarantees must
therefore be established when constructing the custom call and participating in
XLA buffer assignment. If that proof is unavailable, the boundary requires an
explicit XLA ABI extension or backend metadata that carries the missing facts;
the handler must not infer them from the FFI buffer view.

Rank-zero tensors remain real one-element buffers. Scalar Map tasks are not
compile-time constants unless a separate verified constant-folding pass removes
them before materialization.

### Code and launch binding

`shuttle.schedule_task` records grid shape, tile shape, thread count, serial
tile count, scratch bytes, and dependencies. Its `semantic_fingerprint` is an
integrity digest, not executable semantics or a dispatch key. A code generator
must consume the bound `shuttle.map` and `shuttle.fold` bodies and produce a
typed code object for each launch or fused launch group.

An entrypoint cannot bind code by looking up a semantic fingerprint, source
reference, or structural task ordinal. The static device module must carry the
instruction/body IR used to generate the code object, or a lossless typed
lowering certificate that links each generated instruction region to the full
Map/Fold body. The verifier recomputes that linkage from operations, operands,
results, regions, attributes, indexing maps, and numerical constraints before
accepting the code-object digest.

When the device module stores an external code-object digest instead of inline
bytes, the runtime must resolve that digest to immutable bytes and recompute
SHA-256 over the exact bytes returned. A mismatch fails before module load,
entrypoint lookup, or enqueue. The runtime loads only those verified bytes and
resolves each entrypoint from that loaded object; an entrypoint from another
object cannot satisfy the binding. The exact bytes, loaded module, and
executable remain retained until the invocation completion token resolves.

An executable launch record needs:

- a code-object digest and entrypoint owned by the selected code generator;
- exact buffer-table operands and access modes;
- grid and workgroup dimensions with a target-defined dimensional mapping;
- dynamic scratch ownership and alignment;
- bounds predication for partial tiles;
- dependency events and their release/acquire visibility scope;
- failure and completion semantics visible to the runtime.

The current schedule's task dependencies can order accelerator launches. They
do not define intra-workgroup barriers or lane visibility. A device realization
of the `7x13` partial elementwise and Fold tiles requires explicit bounds
predication. The `2048x4096` Fold requires a defined merge of 16 serial feature
chunks.

### Fold algorithm and numerical policy

The Fold constraint is
`tree_association_free_leaf_order_fixed`. Tree association and initializer
placement may vary, but the input leaf order may not. This constraint is the
same for source-ordered policy and for the existing post-conversion FAST policy
mutation. FAST alone does not permit leaf reordering.

The schedule's `scratch_bytes`, threads, and serial tile count are a resource
envelope. They do not define lane loads, partial accumulators, the reduction
tree, serial-chunk merge order, barriers, or final write ownership. A code
generator must choose an algorithm and the verifier must prove that it preserves
leaf order while using only permitted tree association. Until then, an
accelerator row-Fold launch is not executable; the CPU proof uses a sequential
left fold instead.

### Runtime and XLA ownership

The current XLA registration in
`lib/shuttle/mlir/lib/Transforms/XlaRegistration.cc` registers a StableHLO
module transform. `buildShuttleStablehloPipeline` returns StableHLO after source
coverage, algebra lowering, provenance stripping, and rejection of remaining
Shuttle IR. It cannot obtain XLA buffer assignments or replace the selected
region with an executable call.

One of these ownership models must be selected before executable IR is useful:

1. an XLA backend integration that lowers the selected region and participates
   in buffer assignment and executable construction; or
2. a typed custom-call boundary with a registered runtime handler, explicit
   operand/result ABI, code-object lifecycle, stream ownership, and error
   propagation.

The bounded Host form of the second model is specified in the
[`shuttle_xla_cpu_typed_ffi_consumer_design.md`](shuttle_xla_cpu_typed_ffi_consumer_design.md).
That document is a design, not an implemented XLA path or acceptance claim.

The second model also needs a proof that replacing the StableHLO region with the
call preserves source coverage and that compilation cache identity includes the
device ABI, code generator, target profile, and code-object digest. Either model
changes observable compilation and requires a pipeline ABI/cache review. An
opt-in analysis that does not replace StableHLO leaves ABI 5 unchanged.

Under the custom-call model, XLA owns external operand and result allocations.
Typed FFI supplies their dtype, data pointer, rank, dimensions, and execution
stream, but not the backing allocation's available byte extent. Shuttle's
compiler output supplies the shape-derived required byte count and minimum
alignment. The handler can validate dtype, rank, dimensions, and pointer
alignment before enqueue; construction-time buffer assignment or explicit
backend metadata must establish the storage span, offset, layout, alias, and
donation contract.

Temporary buffers are compiler-planned but runtime-allocated. The ordinary FFI
scratch allocator may release an allocation when the handler returns; that is
insufficient for asynchronous launches unless its implementation is proven
stream-ordered or deferred through device completion. The runtime must instead
retain temporary allocations, loaded code, executable state, and asynchronous
error state through an explicit FFI `Future` or equivalent completion token,
and resolve that token with any asynchronous failure. A synchronizing handler
may release them before return only after synchronization succeeds. The handler
reports synchronous validation and enqueue failures directly to XLA. Input
donation and output aliasing follow XLA's buffer assignment, including
tuple-leaf destinations; Shuttle may only request relationships that the
assignment confirms. The XLA runtime owns the supplied stream, and no owner may
release temporary storage, code objects, or error state, until the completion
token resolves. The backend-integration model must name equivalent owners
explicitly.

## Prior artifacts

`ffi_command_buffer_boundary.md` describes two fixed generated CUDA handlers,
evidence from a prior sealed H100 profile, and an unexecuted one-H100 measurement
plan. Its handler names, fixed-shape code, CUDA launch policy, and command-buffer
eligibility audit are experiment evidence. They are not a generic Target 1
runtime contract and must not become dispatch keys for this boundary.

The EventTensor documents define derived dependency factorization and
release/acquire visibility. They can inform a later synchronization lowering,
but the current row-Fold schedule does not contain an EventTensor plan. An
executable verifier cannot claim event counts, barrier placement, or worker
progress from those prototype records.

`spec.md` permits a standalone compiler before XLA integration. It does not
assign buffer, stream, executable, or error ownership to the native MLIR path.
The CPU proof supplies synchronous local owners for its inline code, external
spans, and temporary spans. Accelerator ownership remains unassigned.

## Typed split and accelerator extension

The CPU proof keeps static code and ABI descriptors separate from dynamic
pointers and runtime state. Accelerator lowering should preserve that split.

`shuttle.device_module` is static compiler output. The CPU form binds exactly
one verified schedule and contains:

- schema, numerical policy, and schedule-plan fingerprint;
- inline code-object bytes, their immutable digest, and format;
- entrypoints with structural task ordinals, buffer access modes, byte ranges,
  dependencies, predication, and reduction constraints;
- a deterministic fingerprint covering every semantic field.

`shuttle.invocation_abi` is a static descriptor consumed by the runtime. The CPU
form contains buffer slots, types, required byte spans, strides, offsets,
alignments, host address space, ownership, access, and no-alias/no-reuse groups.
Donation, tuple-leaf destinations, device address spaces, and reuse are not
represented by schema 1 and therefore cannot be claimed. Actual base pointers
are dynamic invocation values and are not serialized into MLIR.

Both schemas must reject unknown attributes and recompute fingerprints. A
static `shuttle.executable_bundle` root binds the canonical device-module digest
and invocation-ABI digest and has a fingerprint covering that ordered pair and
the common source-plan identity. Before code lookup or enqueue, the runtime
recomputes both child digests and the root fingerprint and verifies all three.
Matching source-plan fingerprints alone are insufficient: a device module and
invocation ABI cannot be exchanged independently, even with counterparts built
from the same plan.

A source-binding verifier must rederive exact task and buffer ordinals from the
algebra, materialization plan, and schedule plan. It must compare each launch's
ordered buffer ordinals and access modes with the invocation ABI, not only their
types or counts. Code-object verification must bind generated entrypoints to the
complete typed Map/Fold bodies; workload, module, function, and source-operation
names are excluded.

The CPU proof uses one interpreter entry per schedule task, host buffers,
natural element alignment, zero offsets, unique alias groups, no reuse, and
sequential dependency completion. These are independently checked schema-1
policies, not facts inferred from RankedTensorType. A device extension must add
target launch shape, scratch, visibility, address space, stream, donation, and
tuple-result contracts rather than reinterpret these CPU fields.

## Behavior-first implementation gates

The checked CPU tests observe the runtime boundary, not only printed IR.

Positive gates:

- `7x13` source-ordered execution matches an independent finite CPU reference
  exactly under the selected Fold algorithm after all source algebra and plans
  are erased from the runtime input;
- the post-conversion FAST-policy plan retains
  `tree_association_free_leaf_order_fixed` and the same closed bytecode
  instruction subset; only its policy-bound fingerprints differ;
- `7x13`, scalar Maps, and rank-zero buffers produce complete, distinct slot
  and entry bindings, while `2048x4096` fails the bounded consumer predicate;
- symbol renaming preserves device-module and invocation-ABI fingerprints;
- partial `13` is executed; exact `256`, multi-chunk `4096`, and checked
  byte-span overflow remain accelerator-extension gates.

The checked CPU mutations reject changed slot/source ordinals, strides, offsets,
alignments, host address space, access, aliasing, dependencies, predication,
instruction bytes, code digest, cross-object entry binding, closed root binding,
and unknown attributes. The runtime test applies those corruptions after bundle
construction and independently checks that the stripped positive bundle
executes the generated instructions.

Future accelerator mutation gates must additionally reject:

- missing, duplicated, reordered, or replayed task and buffer bindings;
- changed base-pointer slot, byte span, stride, offset, alignment, address
  space, access mode, alias group, donation relation, tuple-leaf destination,
  reuse group, or allocation owner;
- changed launch grid, workgroup shape, scratch span, predication, dependency,
  visibility, entrypoint, instruction/body linkage, code-object digest, stream
  lifetime, completion ownership, or error propagation;
- altered, truncated, or substituted external code-object bytes presented under
  the claimed digest, including a resolver that returns the wrong object;
- an entrypoint resolved from a different code object than the verified bytes;
- a Fold algorithm that changes leaf order, omits a partial tile, overlaps a
  tile, lacks a required barrier, or merges serial chunks with an unverified
  order;
- scalar Maps with tensor domains and tensor Maps with scalar domains;
- unknown attributes, multiple device modules, multiple invocation ABIs, or
  mismatched source-plan fingerprints;
- a device module or invocation ABI swapped with a same-source-plan counterpart,
  including a one-sided recomputation of either child or bundle digest;
- a launch buffer ordinal or access mode that disagrees with the bound
  invocation-ABI slot, even when the tensor types match;
- dispatch based on workload, module, function, fixture, or source-operation
  names.

A test-only interpreter that reads the surviving algebra would not validate
these schemas. It would retest the existing StableHLO round trip while ignoring
buffer slots, launch ordering, predication, barriers, and code objects. Static
NVVM text has the same problem without a loader and invocation ABI. The first
execution gate must call the actual local runtime consumer with the generated
device module and invocation descriptor.

## Local and external gates

Before integration, local checks must include dialect verification, source
rederivation, deterministic fingerprints, all mutations above, CPU execution
through the real consumer, the existing full MLIR suite, Target 1 CPU parity,
Python tests, and repository precommit.

Static target-code inspection becomes meaningful after a code object and ABI
exist. Rebuilt jaxlib is required when the executable consumer enters the XLA
path. A real accelerator is required for device execution, synchronization,
visibility, and runtime-error tests. Those gates remain separate from CPU
semantics and do not imply NVIDIA, H100, GB200, or production acceptance.
