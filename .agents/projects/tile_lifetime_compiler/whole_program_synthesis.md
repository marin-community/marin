# Shuttle: Whole-Program Synthesis for Machine Learning Systems

## Status

This document describes the long-term Shuttle compiler model and the research/prototype program intended to validate it.

The formalism is a target, not an immediate implementation mandate. During prototyping, simple Python representations, explicit candidate generators, handwritten backend templates, and standalone benchmarks are preferred whenever they answer the relevant research question faster.

The current prototype has already demonstrated substantial proof of life on dense Transformers and distributed MoE. The next objective is to determine whether the same abstractions transfer to qualitatively different architectures before stabilizing them into a production compiler.

---

## 1. Thesis

Modern JAX and PyTorch programs are commonly treated operationally as descriptions of which optimized kernels should be invoked and how those kernels should be chained.

Shuttle starts from a stronger premise:

A high-level ML program is a semantic tensor/state program. Kernel boundaries, layouts, materialization points, communication boundaries, recurrence decompositions, and physical schedules should be outputs of compilation.

Everything eventually becomes kernels and communication operations.

The distinction is between:

```text
kernel-composition compilation:
program
→ recognize known operator/subgraph
→ select existing kernel
→ schedule kernel calls
```

and:

```text
Shuttle:
program
→ whole-program semantic analysis
→ algebraic restructuring
→ state/relation decomposition
→ materialization and layout planning
→ kernel-boundary synthesis
→ producer/consumer schedule synthesis
→ physical kernels and communication
```

Expert-written kernels remain useful:

- as performance oracles;
- as sources of physical implementation primitives;
- as examples from which to extract reusable transformations.

They should not define the semantic optimization space.

A sufficiently mature Shuttle should be capable of producing an implementation that was not present as a complete preassembled kernel in its backend library.

---

## 2. Research ambition

The long-term goal is to obviate entire classes of systems/kernel papers by converting their recurring tricks into compiler transformations.

Examples of currently hand-designed techniques include:

- replacing quadratic attention intermediates with an online reduction state;
- moving residuals, normalizations, activations, and scales into GEMM tile lifetimes;
- changing algebraic factorization to expose efficient contractions;
- choosing whether a transformation belongs before or after a contraction;
- orienting runtime sparse relations to maximize reuse;
- converting ragged work into grouped/segmented contractions;
- deriving bounded producer-consumer pipelines;
- overlapping remote movement with local computation;
- choosing communication granularity and worker allocation;
- converting recurrent definitions into chunkwise or parallel algorithms;
- choosing which intermediates to materialize, recompute, forward, or retain;
- changing kernel boundaries based on connected-region performance rather than operator identity.

CODA demonstrates that a large fraction of non-attention Transformer work can be algebraically moved into GEMM epilogues. (arXiv) FlashAttention-3 demonstrates hardware-specific asynchronous scheduling of tiled attention on Hopper. (arXiv) RedFuser formalizes automatic transformation of cascaded reductions into incremental computation. (arXiv) Nautilus demonstrates end-to-end automatic lowering from mathematical attention expressions to FlashAttention-3-like tiled kernels. (arXiv) Mirage demonstrates joint algebraic and physical superoptimization of tensor programs. (arXiv)

Shuttle should treat these as evidence that expert tricks can be systematized, then operate at a broader region/program scope.

The strongest possible result is not:

Shuttle has implementations of FlashAttention, CODA, MoE, sparse attention, and Gated DeltaNet.

It is:

A small common compiler formalism independently derives the important execution structure of these systems from ordinary program semantics.

---

## 3. Current empirical evidence

### 3.1 Dense Transformer synthesis

The current H100 prototype accepts ordinary StableHLO and recovers an executable dense Transformer plan containing eight major skeletons:

- CODA/QuACK-style contractions.
- Fused RoPE.
- Official FlashAttention-3 attention.
- Two partial RMS reductions.
- Fused SwiGLU.
- Output/down projections.
- Explicit selectable RMS placement policies.

The plan executes without:

- sequence-squared attention intermediates;
- QKV repacking between projection and attention;
- standalone RoPE or SwiGLU kernels for supported cases.

Representative measured results:

```text
sequence length 2048:
    Shuttle / manual-oracle region: 1.456 ms
    JAX/XLA:                       2.501 ms
sequence length 4096:
    Shuttle / manual-oracle region: 3.008 ms
    JAX/XLA:                       6.526 ms
```

A particularly important compiler-generated alternative is RMS scaling in the consumer GEMM prologue:

```text
load unnormalized BF16 A fragment
→ multiply register-resident fragment by FP32 inverse-RMS
→ convert to BF16 WGMMA input
→ execute contraction
```

This preserves the exported source-order BF16 normalization boundary while eliminating the activation-sized materialization.

It is distinct from both:

```text
ordinary execution:
    materialize normalized BF16 activation
```

and:

```text
CODA:
    contract unnormalized activation
    → delay row scale until output epilogue
```

The optimal choice is shape-dependent, which is direct evidence that this should be a compiler search decision rather than an architectural rule.

---

### 3.2 Distributed MoE synthesis

On 4×GB200, a compiler-generated distributed BF16 MoE plan reaches:

```text
Shuttle:          3.984 ms
tuned MoK oracle: 3.613 ms
ratio:            1.103×
```

The generated plan independently derives:

- runtime relation/index planning;
- expert segmentation;
- physical layouts;
- deterministic fixed-slot return/merge;
- buffer dependencies;
- worker allocation;
- communication/computation overlap;
- concatenated `[E, 2I, K]` W13.

It does not invoke the complete MoK forward kernel or reproduce MoK’s event graph.

The selected plan uses:

```text
56 DeepEP communication SMs
concatenated W13
```

and concatenated W13 measured 2.54% faster than separate gate/up projections in the tested configuration.

The merge performs source-ordered FP32 multiply/add without atomics and is bitwise deterministic across repeated execution.

This establishes a second qualitatively different result:

```text
runtime sparse relation
→ segmented computation
→ distributed movement
→ bounded scheduling
```

can also be synthesized competitively from general compiler structure.

---

## 4. Scope of “whole-program”

Whole-program analysis does not imply one enormous kernel.

It means that the compiler retains semantic visibility across boundaries that are conventionally fixed too early:

- framework operator boundaries;
- module boundaries;
- individual GEMMs;
- attention kernels;
- collectives;
- pipeline stages;
- parameter-gather boundaries;
- expert-routing phases.

The output may contain:

- one kernel;
- several persistent kernels;
- library calls;
- communication operations;
- CUDA graphs;
- host/runtime boundaries.

These are physical schedule decisions.

The compiler should be able to move a boundary when doing so improves the complete program.

---

## 5. Semantic foundation

Shuttle should use a small semantic representation that captures the algebraic and dependency structure needed for optimization while remaining independent of CUDA-specific execution details.

The current candidate normal form consists of:

```text
Map
Contract
Fold
Scan
Relation
SegmentedContract
Reshard
Materialize
PersistentState
```

This set is provisional.

A primitive should be added only when experiments demonstrate that an important computation cannot naturally or efficiently be expressed using the existing forms.

---

## 6. Logical dimensions and indexing

Every tensor dimension should have a stable compiler identity.

Human-readable labels such as:

```text
token
hidden
expert
query_head
kv_head
sequence
```

are useful for diagnostics and semantic recovery but are not themselves the identity.

Conceptually:

```text
Axis {
    identity
    extent
    optional label
    optional semantic tags
}
```

Extent may be:

```text
static
symbolic
runtime
segmented
```

Operations carry explicit index relations between axis identities.

Examples:

```text
A[token, hidden]
B[hidden, intermediate]
→ D[token, intermediate]
```

or:

```text
query_head
→ kv_head(query_head)
```

or:

```text
(token, route_slot)
→ expert
```

This representation must distinguish:

- physical transpose from semantic dimension identity;
- reshape from genuine merge/split of logical axes;
- sharding from logical tensor shape;
- dynamic grouping from ordinary dense dimensions.

By the time lowering reaches WGMMA fragments, shared-memory swizzles, or TMA descriptors, semantic axis identities may be compiled away into physical indexing/layout maps.

---

## 7. Values

A logical value should carry enough information to reason about partial availability and distributed execution.

Conceptually:

```text
Value {
    logical axes
    dtype
    index map
    placement
    completeness
    producer
    consumers
    materialization policy
}
```

### Completeness

A value may be complete or an unreduced contribution:

```text
complete
partial {
    merge/reduction operator
    pending domain or mesh axis
}
```

Examples:

```text
partial<sum, tensor_parallel>
partial<sum, data_parallel>
partial<online_softmax_state, selected_kv_blocks>
```

A consumer may require either:

- a complete value;
- or a valid partial state it knows how to extend or merge.

---

## 8. Map

Map represents local transformations that do not fundamentally introduce a reduction or recurrence.

Examples:

```text
residual add
bias
SiLU
GELU
SwiGLU
RoPE
row scaling
column scaling
cast
quantize
dequantize
mask transformation
```

Maps are frequently not materialized as standalone kernels.

The planner may attach a map to:

```text
producer finalization
consumer preparation
state update
transport
materialized standalone transform
```

This attachment decision is part of optimization.

---

## 9. Contract

Contract represents a multilinear contraction.

Conceptually:

```text
Contract {
    key/output axes
    contraction axes
    operands
    accumulator semantics
    optional input preparation
    optional output finalization
    optional auxiliary emissions
}
```

Dense GEMM is the canonical example:

```text
keys: M, N
fold axis: K
acc[M,N] += A[M,K] @ B[K,N]
```

Other examples include:

```text
QKᵀ
PV
tensor-parallel partial contractions
higher-order contractions after factorization
```

The semantic representation should permit transformations on both sides of the contraction.

This is important because the correct compiler question is not simply:

Can operation X be moved into the GEMM epilogue?

It is:

At which legal point during the contraction’s tile lifetime should operation X execute?

Possible choices include:

```text
input preparation
mainloop-associated transform
output finalization
auxiliary state emission
separate materialization
```

The current RMS experiment already demonstrates why both preparation and finalization matter.

---

## 10. Fold

Fold represents a reduction over an unordered or legally reassociable domain.

Conceptually:

```text
Fold {
    key domain
    fold domain
    state
    initialize
    update
    optional merge
    finalize
    numerical/reassociation contract
}
```

Examples:

- RMS sum of squares.
- LayerNorm statistics.
- max/sum-exp.
- online softmax partial states.
- cross-entropy denominator.
- gradient norm.
- expert-output weighted combine.
- top-k summaries.

The state itself may be structured.

For exact attention, a useful partial state is:

```text
AttentionState {
    row_max
    row_sum_exp
    weighted_value_sum
}
```

Two states covering disjoint subsets of K/V positions can be merged by rescaling both to a common maximum and combining their denominator and value accumulators.

This merge structure is critical for both dense and routed sparse attention.

RedFuser’s treatment of cascaded reductions is directly relevant to deriving these incremental forms rather than requiring every recurrence to be named in advance. (arXiv)

---

## 11. Scan

Scan represents ordered state evolution.

Conceptually:

```text
Scan {
    key domain
    ordered axis
    state
    initialize
    update(state, input)
    read(state, input)
    optional ChunkAlgebra
}
```

The defining distinction from Fold is that the logical update order matters.

### Chunk algebra

Many modern recurrent/linear-attention models have an alternative representation that permits training-time parallelism or chunkwise execution.

Represent this explicitly:

```text
ChunkAlgebra {
    summary_type
    summarize(chunk) -> summary
    compose(
        earlier_summary,
        later_summary
    ) -> combined_summary
    apply(
        summary,
        incoming_state
    ) -> outgoing_state
    emit_outputs(
        incoming_state,
        chunk
    )
}
```

Possible physical schedules include:

```text
token recurrent
chunk recurrent
tree/parallel scan
hybrid chunkwise
```

Mamba-2 uses structured state-space duality to expose efficient chunkwise algorithms. (arXiv) Gated DeltaNet introduces a gated delta-rule recurrence with a parallel training algorithm, while Kimi Delta Attention develops a more expressive structured transition and bespoke chunkwise algorithm. (arXiv)

A major future Shuttle test is whether these can be represented by one Scan plus structured transition/factorization rules rather than bespoke architecture kernels.

---

## 12. Relation

Relation is a runtime or static sparse relation among logical roles.

For a binary relation:

```text
R ⊆ Left × Right
```

Examples:

```text
(token route, expert)
(query block, KV block)
(token, selected computation/depth)
(query, retrieved memory page)
```

The relation itself describes semantics.

It does not specify:

- storage order;
- permutation;
- which side should be grouped;
- communication schedule;
- buffer shape;
- physical task ordering.

Those belong to a RelationPlan.

---

## 13. RelationPlan

RelationPlan is the executable/index-plane representation of a relation.

This is now a first-class Shuttle concept.

The important principle is:

Runtime sparse metadata and tensor payload movement are separate planes.

A useful binary plan may contain:

```text
RelationPlan {
    edge count
    left_id[edge]
    right_id[edge]
    edge attributes
    ordering_by_left
    left_offsets
    ordering_by_right
    right_offsets
    original-position mapping
    inverse mapping
    placement/right owner
    local destination
    validity
    padding/capacity information
}
```

Not every implementation needs every representation simultaneously.

The planner may construct whichever orientations are useful.

### Edge attributes

Examples include:

```text
MoE:
    route slot
    router weight
sparse attention:
    mask/bias
    retrieval score
    causal metadata
retrieval:
    page residency
    physical page identity
    estimated fetch cost
```

---

## 14. Relation orientation

A relation can often be executed from either side.

This is a major optimization choice.

### MoE

Token-major:

```text
for each source token:
    send its routes to experts
```

Expert-major:

```text
group routed rows by expert
→ run segmented GMM
```

Both orientations are needed at different points.

### Routed sparse attention

Query-major:

```text
for query block q:
    maintain online attention state
    visit selected KV blocks
```

KV-major:

```text
for KV block k:
    group all query blocks that selected k
    stage K/V once
    process grouped queries
    return partial attention states
```

The second orientation exposes reuse in much the same way expert-major MoE exposes weight reuse.

MoBA explicitly applies MoE-like routing principles to long-context attention by having queries select sparse KV blocks. (arXiv) FlashMoBA is a strong hand-optimized implementation/reference for that computation. (arXiv)

The compiler should be free to choose the orientation based on:

```text
relation degree distribution
payload sizes
resource reuse
placement
communication cost
grouped-compute efficiency
merge cost
```

---

## 15. Higher-arity relations

The long-term relation abstraction should not be fundamentally limited to bipartite graphs.

Conceptually:

```text
Relation {
    roles = [R0, R1, ..., Rn]
    tuples/hyperedges
    tuple attributes
}
```

This enables structures such as:

```text
2-simplicial attention:
(query, key_1, key_2)
```

Higher-order attention then becomes a combination of:

- relation planning;
- contraction factorization;
- structured reduction state.

Recent optimized 2-simplicial attention work demonstrates that the order in which a trilinear interaction is factored into binary tensor-core-compatible computations has major hardware consequences. (arXiv)

This is a future representation test, not a current implementation requirement.

---

## 16. RelationProgram

A common lowering pattern over a relation is:

```text
route/group
→ grouped computation
→ inverse route
→ merge
```

Conceptually:

```text
RelationProgram {
    relation_plan
    group_by_role
    grouped_body
    partial_state_type
    destination_role
    merge_operator
    finalize
}
```

### MoE

```text
relation:
    token route ↔ expert
grouped body:
    W13
    → SwiGLU
    → W2
partial state:
    output vector
merge:
    source-ordered router_weight * value accumulation
```

### Routed sparse attention

```text
relation:
    query block ↔ KV block
grouped body:
    QK
    → local softmax contribution
    → PV
partial state:
    (max, denominator, weighted-value accumulator)
merge:
    exact online-softmax state merge
```

This similarity is deliberate.

A principal near-term research test is whether the existing MoE RelationPlan and scheduling machinery transfers to sparse attention with mostly:

```text
new grouped body
new merge state
attention-specific layouts
```

rather than a new sparse-attention compiler.

---

## 17. SegmentedContract

SegmentedContract represents a contraction over runtime-sized groups.

Conceptually:

```text
SegmentedContract {
    group identity
    runtime offsets or ranges
    per-group operands
    contraction
    preparation/finalization
}
```

For MoE:

```text
group = expert
M_e = number of routed tokens assigned to expert e
D_e = A_e @ W_e
```

The natural physical implementation is grouped/ragged GEMM.

The semantic representation should not assume a particular persistent GMM scheduler.

The scheduler may choose:

```text
expert-major raster
global persistent queue
bucketed experts
padding
coalescing small groups
different task granularities
```

---

## 18. Reshard / placement transition

Logical placement must remain explicit.

A placement transition says:

```text
value with placement A
→ equivalent value with placement B
```

Examples:

```text
token owner → expert owner
expert owner → token owner
parameter shard → consumer-visible weight panel
partial tensor-parallel output → completed destination shard
```

The semantic transition does not initially specify its transport implementation.

Physical candidates may include:

```text
NCCL collective
DeepEP transport
peer TMA
copy engines
source push
pull
remote load/store
device-side collective
```

This allows the compiler to reason about communication at the same level as compute.

---

## 19. Materialization

Materialization is explicit.

Every important intermediate should have one of the following dispositions:

```text
must materialize
may materialize
forward directly
recompute
retain as persistent state
partial-state only
```

A graph edge must not implicitly mean:

materialize the entire tensor to HBM and synchronize.

This is central to Shuttle.

FlashAttention eliminates score/probability materialization.

CODA eliminates many memory-bound Transformer intermediates.

The RMS prologue eliminates normalization materialization without requiring CODA’s numerical reorder.

Distributed Shuttle can eliminate phase-wide MoE barriers by forwarding routed rows through bounded staging buffers.

---

## 20. PersistentState

Some architectures contain state whose lifetime exceeds one local region execution.

Examples:

```text
KV cache
Mamba/GDN state
TTT inner-model state
optimizer state
parameters
routing cache / retrieval index metadata
```

Represent these explicitly rather than encoding them as ordinary ephemeral tensors.

Persistent mutable state carries:

```text
version semantics
ownership
mutation/update rule
visibility
```

This will become important for decode, recurrent architectures, optimizer scheduling, and eventually asynchronous training schemes.

---

## 21. Numerical contracts

Shuttle must not equate real-number algebra with floating-point program equivalence.

Each rewrite or schedule choice should carry an explicit numerical contract.

A useful hierarchy is:

```text
bitwise
source_ordered
cast_equivalent
ordered_fp
bounded_reassociation
real_algebra_equivalent
```

### Current RMS example

Source-ordered prologue:

```text
normalize loaded A
→ cast to BF16 MMA input
→ GEMM
```

preserves the exported low-precision normalization boundary.

CODA delayed scale:

```text
GEMM on unnormalized A
→ scale FP32 accumulator
```

is algebraically equivalent over the reals but changes finite-precision ordering.

Both are legal candidates under different numerical policies.

### Block-scaled formats

MXFP8/FP8 block-scale tensors must be represented semantically.

Scale selection and scale tensor layout are part of the computation, not merely backend metadata.

The compiler should reject unsupported low-precision rewrites rather than silently inventing semantics.

---

## 22. Algebraic transformation library

The compiler should accumulate reusable algebraic facts.

Examples:

### Tile-local movement

```text
elementwise map on a tile
→ candidate producer-finalization or consumer-preparation attachment
```

### Row-scale movement

For a row scalar r:

```text
(r * A) @ W
↔ r * (A @ W)
```

with an explicit floating-point contract.

### Reduction decomposition

```text
reduction(full domain)
→ partial reduction(tile domains)
→ merge(partials)
```

### Online reduction

Derive bounded incremental state for supported cascaded reductions.

### Contraction factorization

For higher-order interactions:

choose binary contraction/factorization tree

based on:

```text
intermediate size
reuse
tensor-core suitability
layout
register/shared-memory footprint
```

### Relation reorientation

```text
iterate left→right
↔ group right←left
```

when semantics permit.

### Recompute/materialize

```text
stored intermediate
↔ recomputed local value
```

when legal and profitable.

### Recurrence chunking

```text
sequential scan
↔ compositional chunk summaries
```

when the recurrence exposes a valid chunk algebra.

---

## 23. Physical schedule representation

After major semantic/algebraic decisions are made, Shuttle lowers to a concrete bounded schedule.

The schedule representation should contain roughly:

```text
TaskFamily
WorkerPool
Buffer
Event
Transport
KernelRegion / KernelBoundary
```

This may later become an MLIR dialect.

For now, Python structures are sufficient.

---

## 24. TaskFamily

A task family represents repeated physical work.

Examples:

```text
QK tile
PV tile
RMS partial tile
expert W13 tile
SwiGLU row block
expert W2 tile
dispatch chunk
KV-block attention group
parameter panel transfer
```

Conceptually:

```text
TaskFamily {
    logical task domain
    physical template
    input readiness
    output readiness
    resource requirements
}
```

---

## 25. WorkerPool

A worker pool reserves or describes execution resources.

Examples:

```text
matrix workers
vector/reduction workers
memory-transfer workers
communication workers
scheduler workers
```

GPU legalization may turn these into:

```text
CTAs
clusters
warpgroups
dedicated communication SMs
```

TPU legalization may turn the same logical roles into different physical execution units.

The high-level schedule should avoid pretending a worker is intrinsically a CUDA warp.

---

## 26. Buffer

A buffer is bounded storage between task families.

Conceptually:

```text
Buffer {
    item domain
    capacity
    placement
    producer
    consumers
    reuse condition
}
```

Correctness requires proving or conservatively ensuring:

- no producer overwrites an unconsumed item;
- bounded capacity cannot deadlock the schedule;
- a buffer generation is not confused with a prior generation.

This is particularly important for:

- MoE inboxes;
- macrobatch rings;
- pipeline stages;
- sparse KV staging;
- FSDP weight panels.

---

## 27. Event

Readiness is often counted rather than Boolean.

Conceptually:

```text
Event {
    domain
    producer task family
    consumer task family
    required arrival count
    phase/generation
    memory ordering scope
}
```

Example:

```text
all W13 output-column tiles for one row block
→ enable SwiGLU row block
```

Arrival counts should be derived from task decomposition.

They should not appear as unexplained hand-coded constants copied from an oracle kernel.

---

## 28. Transport

Transport is the concrete implementation of a placement change.

Conceptually:

```text
Transport {
    semantic placement transition
    chunking
    physical mechanism
    workers/resources
    output buffer
    completion event
}
```

Transport selection is part of scheduling.

The planner may choose different mechanisms by message/tile size.

---

## 29. Kernel boundaries

Kernelization is a schedule decision.

A valid schedule may choose:

```text
one kernel per operator
persistent kernel per region
two-kernel producer/consumer pipeline
one megakernel
mixed library and generated kernels
```

A launch boundary has costs but also provides:

- global synchronization;
- resource-profile reset;
- simpler buffer lifetime;
- easier debugging.

The planner should introduce or remove boundaries based on performance and correctness constraints.

---

## 30. Search strategy

Shuttle does not need blind search over arbitrary programs.

Use hierarchical constrained search.

### Stage 1: semantic alternatives

Enumerate:

```text
algebraic rewrite choices
relation orientation
recurrence/chunk form
materialization choices
contraction factorization
```

### Stage 2: region decomposition

Choose:

```text
skeleton boundaries
layout contracts
kernel boundaries
```

### Stage 3: physical schedules

Choose:

```text
tile sizes
worker allocation
pipeline depth
buffer depth
task ordering
transport
```

### Stage 4: empirical selection

Compile and benchmark a bounded set of competitive candidates.

Initially, a benchmark cache is preferable to a grand analytical cost model.

Analytical estimates should primarily prune obviously bad candidates.

---

## 31. Backend templates

The compiler may reuse excellent low-level physical components.

Examples:

```text
CuTe / CUTLASS / QuACK contraction mainloops
official FlashAttention components
ThunderKittens tile primitives
grouped GEMM implementations
DeepEP communication
remote-copy primitives
reduction kernels
```

This does not violate de novo synthesis.

The distinction is:

```text
acceptable:
    compiler composes generic physical primitives
    into a newly synthesized execution plan
```

versus:

```text
not a synthesis result:
    compiler sees MoE and calls mok.forward()
```

A complete expert implementation may still be registered as:

- baseline;
- oracle;
- fallback backend.

It should not be used to claim that Shuttle synthesized that algorithm.

---

## 32. What “first principles” means operationally

For a workload such as MoE, Shuttle should derive or choose:

```text
runtime relation
relation orientation
expert segmentation
padding
task decomposition
GMM form
map attachment
materialization boundaries
communication granularity
buffering
readiness events
worker allocation
task order
kernel decomposition
return/merge strategy
```

It need not derive:

```text
the instruction encoding of WGMMA
the implementation of a fast remote store
the syntax of TMA
```

Those are backend instruction/library primitives.

---

## 33. Dense Transformer normal form

A dense Transformer region should reduce to combinations of:

```text
Contract
Map attachments
Fold
attention Fold containing Contracts
Materialize
```

A representative optimized region:

```text
QKV Contract
    preparation:
        selected RMS policy
    finalization:
        RoPE
        emit FA3-compatible layout
Streaming attention Fold
    update:
        QK Contract
        online softmax state
        PV Contract
Output Contract
    finalization:
        residual
        RMS partial statistics
RMS Fold
Gate/up Contract
    preparation:
        selected RMS policy
    finalization:
        SwiGLU
Down Contract
    finalization:
        residual
        RMS partial statistics
RMS Fold
```

The current prototype already executes this style of plan.

---

## 34. MoE normal form

A routed MoE forward should normalize approximately to:

```text
routes = top_k(router_logits)
relation =
    Relation(
        left = token × route_slot,
        right = expert,
        edge_weight = router_weight
    )
relation_plan =
    orient_and_index(relation)
routed_x =
    Reshard(
        x through relation_plan,
        destination = expert owner
    )
gate_up =
    SegmentedContract(
        routed_x,
        W_gate_up,
        group = expert
    )
hidden =
    SwiGLU(gate_up)
expert_y =
    SegmentedContract(
        hidden,
        W_down,
        group = expert
    )
returned =
    inverse Reshard(
        expert_y through relation_plan
    )
output =
    Fold(
        key = original token,
        over = route edges,
        update =
            source_ordered router_weight * returned value
    )
```

The existing distributed prototype already demonstrates a competitive lowering of this structure.

---

## 35. Routed sparse attention normal form

The next prototype should test the same relation machinery on:

```text
Relation(
    left = query block,
    right = selected KV block
)
```

Two legal orientations:

### Query-major

```text
for each query block:
    Fold over selected KV blocks
        QK
        online state update
        PV
```

### KV-major

```text
group queries by selected KV block
stage K/V block
→ grouped QK/PV partial computation
→ inverse route partial attention states
→ Fold/merge states by query
```

The state merge is exact online-softmax merge rather than weighted vector addition.

This is the near-term test of whether RelationPlan is genuinely reusable.

MoBA provides a natural sparse-attention semantic workload and explicitly applies MoE-like block selection to attention. (arXiv)

---

## 36. Stateful linear-attention / recurrent normal form

The subsequent prototype should target Gated DeltaNet or Kimi Delta Attention.

The intended representation is:

```text
Scan {
    state = structured fast-weight matrix/state
    update:
        decay / erase
        read correction
        write
    read:
        query against state
    ChunkAlgebra:
        structured summary
        composition
        chunk output generation
}
```

The compiler should ideally recover both:

```text
decode:
    recurrent token update
```

and:

```text
training/prefill:
    chunkwise parallel algorithm
```

without adding a semantic op called GatedDeltaNetKernel.

Gated DeltaNet and Kimi Delta Attention provide strong implementation oracles for this test. (arXiv)

---

## 37. Other architecture stress tests

These are future experiments, not required features.

### Higher-order / simplicial attention

Tests:

```text
n-ary Relation
contraction-tree search
online reduction over product domain
```

### Mixture of Depths / adaptive compute

Tests:

```text
Relation(
    token,
    selected computation instance
)
```

with grouping, bypass, return, and merge.

### Retrieval memory

Tests:

```text
query
→ selected remote/cache pages
```

where relation edges carry placement/cache cost.

### Hyena / long convolution

May demonstrate the need for a dedicated Convolution skeleton if it cannot be expressed naturally and efficiently as existing Contract/Scan forms.

Do not add Convolution to the core normal form until such a prototype shows that it is necessary.

---

## 38. Distributed parallelism

The same semantic model should eventually recover conventional distributed strategies.

### FSDP

Represent:

```text
parameter placement transition
→ streamed Contract consumption
→ gradient partial
→ placement transition / reduction
```

A conventional all-gather/GEMM/reduce-scatter schedule is one candidate.

A panel-streamed schedule is another.

### Tensor parallelism

A sharded contraction naturally produces partial values whose completion requires a reduction/reshard.

### Expert parallelism

Already represented by:

```text
Relation
→ placement transition
→ SegmentedContract
→ inverse transition
→ merge
```

### Pipeline parallelism

No semantic microbatch axis is required.

A stage cut produces a tile stream.

The scheduler chooses groups of ready tiles as stream items and fires downstream regions as their required values become available.

Microbatching is one possible stream grouping, not a fundamental semantic construct.

---

## 39. Pipeline scheduling

Given a cut:

```text
stage A
→ tile channel
→ stage B
```

the scheduler determines:

```text
readiness domain
consumer firing requirement
stream grouping
bounded buffering
forward/backward ordering
```

This can recover:

```text
GPipe
1F1B
interleaved schedules
zero-bubble-like decompositions
```

without encoding those names as fundamental program operators.

Backward-input and weight-gradient work may be exposed as separate task families when their criticality differs.

---

## 40. Target independence

The semantic/Flow representation should not depend on NVIDIA execution concepts.

The same program should in principle lower to:

```text
GPU:
    WGMMA
    TMA
    SMEM
    CTAs/clusters
    NVLink
TPU:
    MXU
    VMEM/SMEM
    DMA
    remote DMA
    ICI
```

Physical schedules will differ substantially.

The value of a TPU backend is initially architectural validation more than expected dense performance gain.

The first production target remains NVIDIA because current GPU compilation leaves much more obvious opportunity.

---

## 41. Long-term compiler layering

A plausible eventual implementation is:

```text
JAX / PyTorch
        ↓
StableHLO / semantic graph
        ↓
Shuttle Flow IR
        ↓
algebraic + relation + state rewrites
        ↓
Shuttle Schedule IR
        ↓
target legalization / backend templates
        ↓
XLA runtime / PJRT
```

Potential MLIR organization:

```text
shuttle
    semantic Flow dialect
shuttle_sched
    bounded physical schedule dialect
```

Reuse existing dialects for:

```text
arith
math
tensor
memref
scf
gpu
nvgpu/nvvm
```

and reuse Shardy rather than creating a new global sharding language.

This is a future engineering direction.

The prototypes should not implement it merely because the architecture diagram contains it.

---

## 42. Relationship to existing compiler work

Shuttle should be evaluated against and learn from several related systems.

### CODA

Shows that constrained GEMM-plus-epilogue programming captures most non-attention Transformer operations while preserving expert GEMM structure. (arXiv)

Shuttle generalizes operation placement beyond the epilogue and reasons across larger regions.

### FlashAttention / FA3

Provides the canonical example of replacing tensor materialization with bounded online state, followed by architecture-specific asynchronous scheduling. FA3 specifically exploits Hopper asynchronous Tensor Cores/TMA and worker specialization. (arXiv)

### RedFuser

Provides formal machinery for converting cascaded reductions into incremental reductions. (arXiv)

This should inform Shuttle’s Fold transformation engine.

### Nautilus

Demonstrates automated math-to-tiled-kernel lowering and automatic discovery of FA3-like attention schedules. (arXiv)

Shuttle’s intended distinction is broader whole-region/state/relation/distributed planning while retaining expert-quality skeletons.

### Mirage

Demonstrates multi-level algebraic and schedule superoptimization with generated custom kernels. (arXiv)

### Axon / Prism

Recent work pushes tensor superoptimization further toward semantic synthesis and symbolic representation of implementation families. (arXiv)

Shuttle should track these closely; the desired contribution is not simply “tensor algebra can be superoptimized,” but a normal form that naturally spans state, runtime sparse relations, distributed placement, and global tile-flow scheduling.

### MoBA / FlashMoBA

Provide the immediate test of relation-plan reuse outside MoE. (arXiv)

### Gated DeltaNet / Kimi Delta Attention / Mamba-2

Provide the immediate tests for the Scan and chunk-algebra representation. (arXiv)

---

## 43. Prototype philosophy

The purpose of the current prototype phase is to falsify or validate the normal form, not to productionize it.

Prefer:

```text
small explicit Python IR
bounded candidate generator
handwritten backend adapter
direct benchmark
```

over:

```text
large generalized compiler subsystem
```

when both answer the same research question.

### Hard-code the search space, not the answer

Acceptable:

```text
try relation orientations A and B
try worker allocations [32, 48, 56]
try buffer depths [2, 4, 8]
benchmark and select
```

Not acceptable as evidence of synthesis:

```text
this is MoE, therefore emit MoK's known schedule
```

The compiler is allowed to know general laws and physical templates.

It should not be handed the winning architecture-specific global plan.

---

## 44. Current checkpoint requirements

Before further experiments, preserve the dense and MoE results so they cannot disappear.

Record:

```text
source revisions
compiler revision
hardware
driver
CUDA
clock/power conditions
benchmark distributions
selected candidate plan
all candidates considered
semantic correctness checks
determinism checks
```

Do not spend excessive time polishing this into publication infrastructure.

The goal is reproducibility sufficient to return to the result later.

---

## 45. Next experiment: routed sparse attention

Active implementation brief: [routed_sparse_attention_brief.md](routed_sparse_attention_brief.md).

Background research and oracle ledger: [routed_sparse_attention_background.md](routed_sparse_attention_background.md).

This is the highest-priority next experiment.

The concrete first-slice plan is tracked in [Routed Sparse Attention: RelationPlan Reuse Experiment](routed_sparse_attention_plan.md).

Question:

Does the MoE relation/index/scheduling machinery actually describe runtime sparse computation, or only MoE?

Use MoBA-like routed block attention initially. MoBA directly applies MoE-style routing to KV blocks, making it a clean semantic stress test. (arXiv)

Required reuse target:

```text
RelationPlan
orientation/grouping
task derivation
buffer representation
event/readiness machinery
transport scheduling
inverse mapping
```

New workload-specific pieces should ideally be mostly:

```text
block-attention grouped body
online-softmax partial state
online-softmax merge
attention layouts
```

Try both:

```text
query-major sparse Fold
```

and:

```text
KV-major RelationProgram
```

The latter is especially useful for testing resource reuse and relation reorientation.

Strong success is not merely high performance.

Strong success means substantial MoE compiler machinery transfers unchanged.

---

## 46. Next experiment after sparse attention: StatefulScan

Target:

```text
Gated DeltaNet
or
Kimi Delta Attention
```

Question:

Can a generic stateful scan representation recover both recurrent and chunkwise high-performance forms?

Add only enough Scan/ChunkAlgebra machinery to execute the experiment.

Avoid architecture-specific semantic nodes.

Measure:

```text
recurrent correctness
chunkwise correctness
training/prefill throughput
decode throughput
backend gap to expert implementation
```

Again, code-reuse/generalization evidence is at least as important as raw latency.

---

## 47. Generality accounting

For every new architecture experiment, classify implementation changes as:

```text
reused unchanged
generalized existing Shuttle machinery
new generic Shuttle machinery
workload-specific semantic recovery
workload-specific physical primitive/backend
```

This is a central metric.

The project is succeeding if the first two categories grow while architecture-specific compiler logic remains small.

---

## 48. When to begin the real XLA fork

Do not begin a major Shuttle/XLA fork merely because dense and MoE work.

Begin serious compiler integration after the normal form survives at least:

```text
dense Contract/Fold composition
runtime Relation/SegmentedContract computation
relation reuse outside MoE
ordered Scan with chunk algebra
```

At that point there is enough evidence to freeze substantial semantic structure.

Then use XLA for:

```text
StableHLO/HLO plumbing
Shardy
autotuning infrastructure
runtime
PJRT
buffer/executable machinery
existing emitters and libraries
```

while Shuttle owns:

```text
semantic normalization
algebraic transformations
relation planning
state decomposition
materialization/layout planning
kernelization
global resource schedule synthesis
```

---

## 49. Whole-training milestone

After the abstraction is stable enough to justify a real compiler implementation, the next major systems target is an entire training step.

That means reasoning jointly about:

```text
forward
backward
recomputation
saved state
dgrad/wgrad decomposition
gradient accumulation
FSDP
TP
EP
pipeline streams
communication overlap
optimizer boundary
```

The important question becomes:

Can Shuttle choose kernel and communication boundaries for the entire training program rather than optimizing isolated forward regions?

This is likely a more consequential milestone than reproducing another single forward kernel.

---

## 50. Long-term success criterion

The project becomes genuinely large if ordinary high-level programs can be compiled into near-expert implementations across several structurally different workloads:

```text
dense Transformer
distributed MoE
routed sparse attention
structured recurrent / linear attention
distributed training
```

using one coherent semantic/scheduling formalism.

A strong eventual statement would be:

Starting from ordinary JAX programs, Shuttle automatically synthesizes implementations competitive with expert-designed systems across dense attention, Transformer MLPs, distributed MoE, routed sparse attention, and structured recurrent models, while jointly optimizing materialization, layout, communication, and kernel boundaries.

An even stronger result would show that the same compiler can synthesize useful new implementations that have no direct hand-written oracle.

---

## 51. Failure criteria

Do not preserve the formalism for aesthetic reasons.

Change it if experiments show that:

- RelationPlan cannot transfer outside MoE;
- stateful linear attention needs fundamentally different structure;
- layout/materialization planning cannot be separated cleanly from physical scheduling;
- numerical contracts are insufficient;
- the candidate space requires architecture-specific special cases everywhere;
- generated kernels are consistently limited by generic backend quality in a way that defeats whole-program optimization.

A failed abstraction experiment is useful.

Do not hide failure by adding:

```text
shuttle.moba
shuttle.gated_deltanet
shuttle.mok
shuttle.flash_attention
```

until the normal form appears to work again.

---

## 52. Design maxim

The shortest statement of Shuttle is:

Analyze the whole semantic program, represent its contractions, state, sparse relations, and placement changes explicitly, then synthesize the lifetimes and movement of tiles before deciding where kernels begin and end.

Or operationally:

```text
tensor/state semantics
    ↓
algebra + relations + recurrence
    ↓
tile-lifetime decisions
    ↓
bounded resource schedule
    ↓
de novo kernelization
```

The project succeeds to the extent that this process turns specialized expert implementation knowledge into reusable compiler knowledge.

---

## 53. Immediate research sequence

```text
NOW
│
├─ Freeze dense + distributed-MoE checkpoint
│
├─ Routed sparse attention
│    └─ test RelationPlan reuse
│
├─ Gated DeltaNet / KDA
│    └─ test Scan + ChunkAlgebra
│
├─ Reevaluate the normal form
│    └─ remove decorative abstractions
│
├─ If still coherent:
│    └─ begin serious XLA/Shuttle implementation
│
├─ Whole training step
│    ├─ backward
│    ├─ recomputation
│    ├─ FSDP / TP / EP
│    └─ pipeline tile streams
│
└─ Further architecture stress tests
     ├─ higher-order/simplicial attention
     ├─ adaptive depth
     ├─ retrieval memory
     └─ long convolutions if needed
```

The next prototypes are intended to earn the right to build the compiler.

The longer-term target is to make expert kernel design increasingly a source of compiler transformations rather than an endless sequence of bespoke implementations.
