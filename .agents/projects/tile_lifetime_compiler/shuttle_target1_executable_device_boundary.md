# Target 1 executable-device boundary

Status: design stop. No executable device IR is added by this slice.

## Decision

`shuttle.schedule_plan` is the last lossless typed boundary currently supported
by the compiler. It fixes an abstract `simt32` task schedule, logical indexing,
task dependencies, and a resource envelope. It does not contain enough
information to emit or execute device code.

Adding NVVM, a custom call, or a nominal executable-plan operation now would
create an orphan backend. The XLA module-transform hook owns StableHLO-to-
StableHLO rewriting only. It has no buffer-assignment result, executable build
callback, runtime registration, or invocation-time buffer table. The core
pipeline also removes all Shuttle operations before returning StableHLO.
Opt-in materialization and schedule passes remain outside that pipeline and do
not change pipeline ABI 5 or its cache identity.

The next implementation must begin with one explicit runtime owner and one
code-object owner. Those choices are inputs to executable lowering; they cannot
be inferred from tensor types, affine maps, reduction dimensions, numerical
policy, or the abstract `simt32` profile.

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

The first implementation should forbid reuse, donation, nonzero offsets, and
overlapping alias groups. That is a safe candidate policy, not a fact derivable
from StableHLO. A typed XLA FFI handler can observe each argument's dtype, data
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

An executable launch record needs:

- a code-object digest and entrypoint owned by the selected code generator;
- exact buffer-table operands and access modes;
- grid and workgroup dimensions with a target-defined dimensional mapping;
- dynamic scratch ownership and alignment;
- bounds predication for partial tiles;
- dependency events and their release/acquire visibility scope;
- failure and completion semantics visible to the runtime.

The current schedule's task dependencies can order launches. They do not define
intra-workgroup barriers or lane visibility. The `7x13` partial elementwise and
Fold tiles require explicit bounds predication. The `2048x4096` Fold requires a
defined merge of 16 serial feature chunks.

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
leaf order while using only permitted tree association. Until then, the row
Fold launch is not executable.

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
Those ownership decisions remain required even for a standalone local runtime.

## Proposed typed split

Do not add one operation that mixes static code, dynamic pointers, and runtime
state. The first executable implementation should use two closed child schemas
and one closed static binding root.

`shuttle.device_module` is static compiler output. It binds exactly one verified
schedule and contains:

- schema, target-capability, numerical-policy, materialization-plan, and
  schedule-plan fingerprints;
- code-object bytes or an immutable code-object digest and format;
- entrypoints with structural task ordinals, buffer access modes, launch
  dimensions, predication, scratch requirements, and visibility edges;
- a deterministic fingerprint covering every semantic field.

`shuttle.invocation_abi` is a static descriptor consumed by the runtime. It
contains buffer slots, types, required byte spans, strides, offsets, alignments,
address spaces, ownership, alias and donation constraints, tuple-leaf
destinations, and allocation/reuse groups. Actual base pointers, observable
allocation extents supplied by a future ABI extension, and stream/event handles
are dynamic invocation values and must not be serialized into MLIR.

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

The initial candidate may use one launch per schedule task, global buffers,
natural element alignment, zero offsets, unique alias groups, no reuse, and
release/acquire visibility between dependent launches. Each choice must be
encoded as a target/runtime policy and independently checked. None should be
described as inferred from RankedTensorType.

## Behavior-first implementation gates

The executable slice starts only when one local runtime can consume both closed
schemas. Its tests must observe the runtime boundary, not only printed IR.

Positive gates:

- `7x13` source-ordered execution matches an independent finite CPU reference
  exactly under the selected Fold algorithm;
- the post-conversion FAST-policy plan retains
  `tree_association_free_leaf_order_fixed` and either produces the same result
  or documents a separately verified numerical contract;
- `2048x4096`, `7x13`, scalar Maps, and rank-zero buffers produce complete,
  distinct slot and launch bindings;
- symbol renaming preserves device-module and invocation-ABI fingerprints;
- partial `13`, exact `256`, multi-chunk `4096`, and checked byte-span overflow
  use the same public lowering and verifier paths.

Mutation gates must reject:

- missing, duplicated, reordered, or replayed task and buffer bindings;
- changed base-pointer slot, byte span, stride, offset, alignment, address
  space, access mode, alias group, donation relation, tuple-leaf destination,
  reuse group, or allocation owner;
- changed launch grid, workgroup shape, scratch span, predication, dependency,
  visibility, entrypoint, instruction/body linkage, code-object digest, stream
  lifetime, completion ownership, or error propagation;
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
