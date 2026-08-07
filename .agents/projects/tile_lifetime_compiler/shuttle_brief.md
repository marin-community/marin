# Shuttle: Prototype Brief and Big-Picture Specification

## 1. Mission

Shuttle is an experimental compiler for high-performance ML programs.

The long-term hypothesis is:

High-level JAX/PyTorch programs contain enough semantic structure for a compiler to derive many of the algebraic transformations, kernel boundaries, tile pipelines, communication schedules, and materialization choices that are currently encoded manually in expert-written kernels.

The compiler should perform whole-program analysis and synthesize implementations de novo from a small set of reusable execution primitives and transformation rules.

The output will still ultimately consist of kernels and communication operations. The difference from a conventional kernel-dispatch compiler is that kernel boundaries and schedules are compiler outputs rather than fixed semantic atoms.

Examples of the behavior Shuttle should eventually recover include:

- FlashAttention-style online reductions rather than materializing attention matrices.
- CODA-style placement of Transformer operations around GEMM mainloops.
- Source-ordered RMS scaling in a GEMM prologue when that is a better implementation than either materialization or delayed scaling.
- Layout choices that jointly optimize QKV projection and attention.
- Routed sparse computation organized into grouped/ragged contractions.
- Distributed MoE schedules that overlap routing communication with expert computation.
- Eventually, similar patterns in sparse attention, linear attention/state-space models, FSDP, and pipeline parallelism.

The current project is not to build that whole compiler.

The current project is to gather enough concrete evidence that this compiler architecture is real.

## 2. Current status

The dense Transformer experiment has succeeded.

On branch:

```text
prototype/tile-lifetime-compiler
```

the prototype can take ordinary StableHLO and recover an executable dense Transformer region containing:

- CODA/QuACK GEMMs.
- RMS reduction and scale placement.
- Fused RoPE.
- Official FlashAttention-3.
- Fused SwiGLU.
- Down and output projections.

The generated plan contains eight execution skeletons and runs directly without:

- Sequence-squared attention intermediates.
- QKV repacking between projection and FA3.
- Standalone RoPE or SwiGLU kernels in the supported cases.

Representative H100 results:

```text
Sequence 2048:
    compiler/manual oracle: ~1.456 ms
    JAX/XLA:               ~2.501 ms
Sequence 4096:
    compiler/manual oracle: ~3.008 ms
    JAX/XLA:               ~6.526 ms
```

RMS placement is a real compiler choice rather than a hard-coded transformation.

The prototype supports:

```text
source-ordered consumer prologue:
    normalize the register-resident A fragment
    before conversion to the BF16 WGMMA input
CODA-style delayed epilogue:
    perform GEMM first
    scale the accumulator afterward
```

The source-ordered form preserves the exported BF16 normalization boundary. The delayed form changes floating-point ordering but is often somewhat faster.

Different shapes prefer different implementations, so candidate selection genuinely matters.

Mixture-of-Kittens has also been established as an executable performance oracle on 4xGB200.

Pinned MoK BF16 forward results include approximately:

```text
default:
    3.669 ms
    516.2 TFLOP/s
tuned:
    20 communication SMs
    minibatch = 2048
    macrobatch = 65536
    ~3.61 ms
    ~524 TFLOP/s
```

Official correctness tests pass.

The next goal is not simply to call MoK from the recovered graph.

## 3. The next research question

The next major feasibility question is:

Starting from an ordinary routed-MoE program, can Shuttle use general transformations and schedule synthesis to generate a distributed MoE implementation that is competitive-ish with Mixture-of-Kittens without encoding MoK itself as the lowering?

MoK is the oracle.

It is not the desired compiler output template.

A successful result does not need to beat MoK.

A successful result demonstrates that the same general compiler ideas that worked for dense Transformers extend to irregular, dynamic, distributed computation.

## 4. Conceptual model

The compiler should be thought of as having two conceptual levels.

These do not need to become MLIR dialects or production IRs during the prototype.

Simple Python structures are sufficient.

### 4.1 Flow representation

The Flow representation describes the mathematical program and the dependencies that matter for tiled execution.

Values should retain:

```text
logical dimensions / index relations
dtype
shape
placement
completeness
producers and consumers
```

Logical dimensions should have stable compiler identities. Human-readable names such as token, hidden, or expert are useful but are not themselves the semantics.

The useful operation families appear to be quite small.

### Map

Local transformations:

```text
residual
RoPE
SiLU
SwiGLU
scale
cast
quantize
dequantize
```

### Contract

Dense multilinear contraction:

```text
GEMM
QK contraction
PV contraction
```

A contract may have operations attached to its input preparation or output finalization.

This includes the important choice:

```text
normalize A during input preparation
```

versus:

```text
delay row scale until output finalization
```

### Fold / Scan

Stateful reductions or ordered recurrences:

```text
RMS statistics
softmax state
online attention
gradient accumulation
linear-attention state
```

FlashAttention is conceptually a fold over K/V blocks whose update contains contractions.

### Relation

A runtime sparse relation between logical work items and resources.

Examples:

```text
token route -> expert
query block -> selected KV block
token -> selected depth / computation
```

The relation may be reordered and traversed by either side.

### Segmented Contract

A contraction over runtime-sized groups.

For MoE:

```text
group = expert
M_e = number of routed rows for expert e
D_e = A_e @ W_e
```

The expected physical implementation is a grouped/ragged GEMM.

### Placement change

A value changes logical ownership:

```text
token owner -> expert owner
expert owner -> token owner
```

The Flow representation should describe the placement change without requiring a particular implementation such as NCCL, peer TMA, or source-push remote stores.

### Materialization

A compiler decision that a logical intermediate must exist in memory at some boundary.

Materialization should be explicit rather than implied by every graph edge.

## 5. MoE normal form

A routed MoE forward program should normalize approximately to:

```text
routes = top_k(router_logits)
relation = build_relation(source = token routes, destination = experts)
routed_x = move_by_relation(x, relation, destination = expert owner)
gate = segmented_contract(routed_x, W_gate, group = expert)
up = segmented_contract(routed_x, W_up, group = expert)
hidden = silu(gate) * up
expert_y = segmented_contract(hidden, W_down, group = expert)
returned_y = move_by_inverse_relation(expert_y, relation, destination = original token owner)
output = segmented_merge(
    returned_y,
    group = source token,
    value = router_weight * returned_y,
    operator = sum
)
```

Shared experts can be represented as an independent dense branch merged into the same output.

This is the semantic target.

It should not contain:

```text
MoK
20 communication SMs
minibatch size 2048
specific readiness-counter numbers
specific MoK task ordering
specific ring-buffer phases
```

Those are physical schedule decisions.

## 6. Runtime relation / index plane

The major new capability required for MoE is runtime index planning.

The compiler should be able to generate a compact metadata program from the router output.

A useful runtime structure may contain:

```text
relation_plan {
    source token
    route slot
    destination rank
    destination expert
    destination row
    expert offsets
    tokens per expert
    router weight
    reverse mapping
    validity / padding
}
```

This metadata describes the sparse relation.

The activation payload is handled separately.

This distinction between the index plane and payload plane is important.

The same abstraction should later be usable for routed sparse attention:

```text
query block -> selected KV blocks
```

or other runtime-sparse architectures.

Do not make the index structure more MoE-specific than necessary.

## 7. Schedule representation

After semantic normalization, Shuttle chooses a physical implementation.

The prototype only needs enough schedule representation to express the candidate implementations being tested.

Useful concepts include:

```text
TaskFamily
WorkerPool
Buffer
Event
Transport
KernelBoundary
```

### Task family

Examples:

```text
dispatch row block
expert W13 tile
SwiGLU tile
expert W2 tile
return row block
weighted combine
```

Each task family should specify:

```text
logical task domain
inputs
output
physical template
readiness condition
completion signal
```

### Worker pool

Examples:

```text
communication workers
GMM workers
transform workers
```

For a persistent kernel this might eventually correspond to dedicated SMs or CTA clusters.

### Buffer

A bounded storage object:

```text
capacity
item domain
producer
consumer
reuse condition
```

The important property is that reuse follows actual consumer completion.

### Event

A counted or phased readiness condition.

For example:

```text
all gate/up output-column tasks
for one expert row block
must finish before SwiGLU consumes it
```

Arrival counts should be derived from the task decomposition rather than inserted as unexplained constants.

### Transport

A concrete implementation of a logical placement change:

```text
whole-tensor collective
peer transfer
source push
pull
copy engine
communication worker
```

The initial prototype may implement only one or two choices.

## 8. What “first-principles synthesis” means

The compiler is allowed to reuse high-quality low-level components.

For example:

```text
GMM mainloops
QuACK/CuTe contractions
remote-copy primitives
SwiGLU tile transforms
quantization primitives
```

Reusing these does not invalidate the experiment.

The compiler does not need to synthesize WGMMA instructions from scalar arithmetic.

The compiler contribution is to derive their composition.

A first-principles MoE implementation should derive or choose:

```text
runtime grouping by expert
padding / segmentation
tile decomposition
dispatch granularity
GMM task decomposition
where SwiGLU executes
where intermediate values materialize
communication / computation overlap
buffer capacities
buffer reuse dependencies
worker allocation
task order
kernel boundaries
return and combine granularity
```

from the semantic program, target constraints, and candidate templates.

The implementation is not first-principles if it simply contains:

```text
if MoE:
    emit the MoK event graph
```

or:

```text
if MoE:
    call mok.forward()
```

MoK may be retained as a backend candidate for comparison and as a correctness/performance oracle.

## 9. Numerical contracts

The dense experiment established that algebraically equivalent rewrites can have materially different floating-point semantics.

Continue treating numerical behavior explicitly.

At minimum distinguish:

```text
source_ordered
real_algebra_equivalent
```

For BF16 MoE forward, prefer source-order behavior unless an alternative numerical policy is explicitly selected.

MXFP8 should remain unsupported until scale tensors and block-scale semantics are represented correctly.

Do not treat block scaling as backend metadata.

It changes the actual program.

## 10. Prototype scope

The immediate goal is BF16 MoE forward on GB200.

Focus on one representative configuration compatible with the existing MoK oracle.

The prototype does not need:

- backward
- MXFP8
- production XLA integration
- new MLIR dialects
- general cost modeling
- formal deadlock verification
- arbitrary model families
- cross-platform support
- TPU lowering
- production-quality autotuning

These are future work.

A small number of explicit candidate templates plus empirical benchmarking is enough.

XLA should remain primarily:

```text
StableHLO source
semantic baseline
performance baseline
```

A large XLA patch is not part of this phase.

## 11. Immediate next steps

### Step 1: Finish ordinary JAX MoE semantic recovery

Start from a straightforward JAX implementation of the supported routed MoE and export StableHLO.

Recover:

```text
router logits
top-k experts
router weights
expert grouping
gate/up projections
SwiGLU
down projection
weighted combine
shared-expert branch if present
```

Normalize this into the MoE form described above.

Keep the official MoK binding available as a validation backend, but keep semantic recovery independent of MoK implementation details.

Exit condition: A readable compiler dump shows the recovered semantic MoE program and relation structure from ordinary StableHLO.

### Step 2: Generate the runtime relation plan

Implement the minimal index-plane program needed to convert top-k routes into:

```text
expert counts
expert offsets
destination rank
destination row
source token
route slot
router weight
reverse mapping
padding information
```

Validate it independently against a straightforward reference permutation.

Exit condition: The same relation plan can correctly drive dispatch, inverse dispatch, and weighted combine without relying on MoK’s internal schedule builder.

### Step 3: Lower local expert work to segmented contracts

Represent routed W_gate, routed W_up, SwiGLU, and routed W_down as segmented contracts grouped by expert.

Use an existing high-performance GMM primitive where convenient.

The compiler should generate the runtime task domain from expert offsets.

Exit condition: A single-GPU or already-routed expert batch can execute correctly through the compiler-generated segmented task plan.

### Step 4: Generate a simple distributed plan

Construct a correct distributed BF16 forward schedule from the generic primitives.

The easiest useful candidate is likely a multi-kernel plan.

A plausible decomposition is:

```text
Kernel / phase A:
    dispatch
    W13
    SwiGLU
materialization boundary
Kernel / phase B:
    W2
    return
    weighted combine
```

This exact boundary is not sacred.

Use whatever simple decomposition naturally follows from available primitives and gives a useful experiment.

The important point is that the schedule is generated from relation, segmented contracts, dependencies, buffer requirements, and placement changes rather than copied from MoK.

Exit condition: Compiler-generated distributed execution produces numerically correct MoE output on the target configuration.

### Step 5: Add tile-stream overlap

Once a correct distributed implementation exists, remove unnecessary phase barriers.

Introduce:

```text
bounded inbox / staging buffers
fine-grained readiness
overlap of incoming routed rows with W13
overlap of completed expert work with return
```

Derive the relevant event counts and reuse dependencies from the task graph.

Initial candidate parameters can include:

```text
dispatch chunk size
inbox depth
number of communication workers
number of compute workers
expert-group/task ordering
materialization boundary
```

Do not attempt an enormous search space.

Exit condition: The generated timeline visibly overlaps communication and expert computation.

### Step 6: Empirically choose among generated schedules

Benchmark a bounded candidate set.

Examples:

```text
coarse dispatch -> compute -> return
streamed dispatch + W13
two-kernel source-push-like plan
one persistent kernel if implementation complexity is modest
several worker allocations
several chunk sizes
several inbox depths
```

The compiler may initially select from measured values cached by workload shape.

An elaborate analytical performance model is unnecessary.

Exit condition: The compiler selects a nontrivial schedule based on measured end-to-end latency rather than a globally hard-coded preference.

## 12. What success looks like

### Minimum proof of life

The prototype should demonstrate all of the following:

1. An ordinary JAX MoE StableHLO graph is recovered into a generic routed/segmented semantic representation.
2. The compiler generates the runtime relation/index plan.
3. Expert compute is represented as segmented contracts rather than one opaque MoE operation.
4. At least one distributed BF16 implementation is synthesized without calling the complete MoK forward kernel.
5. Buffer lifetimes and readiness conditions are derived from producer/consumer dependencies.
6. Communication and compute overlap at some granularity.
7. Output matches the semantic reference.
8. The implementation is benchmarked directly against the pinned MoK oracle under the same configuration.

### Performance target

A strong initial target is:

```text
generated BF16 forward latency
<= approximately 1.3x tuned MoK
```

on at least one representative workload.

Given a tuned MoK latency around 3.6 ms, an implementation around 4.5–4.7 ms would already be compelling proof of life.

Within 15–20% would be an excellent result.

Matching MoK is not required.

The result becomes particularly convincing if the compiler independently chooses recognizable high-level strategies such as fine-grained dispatch, expert grouping, bounded buffering, nontrivial communication-worker allocation, communication/compute overlap, and early return of completed work without those choices being encoded as a special MoK lowering.

## 13. What failure would still teach us

A generated implementation that is slower than the performance target is still useful if the gap is isolated.

Record whether the limit is:

```text
GMM throughput
dispatch bandwidth
return bandwidth
communication-worker allocation
buffering
expert imbalance
synchronization overhead
poor task ordering
too-large materialization boundary
```

A result such as:

```text
compiler-generated schedule is 1.5x MoK,
but 80% of the gap is a known GMM primitive difference
```

would still provide evidence for the representation and scheduling model.

Do not hide a useful architectural result because one backend primitive is weak.

## 14. Required compiler diagnostics

Every generated MoE plan should be inspectable.

Produce a plan dump similar to:

```text
MoE Region
Relation
    source domain: token x route_slot
    destination: rank x expert
    routes: runtime
    padding policy: ...
Segmented W13
    groups: experts
    tile: ...
    task count: runtime from expert offsets
Dispatch
    transport: source push
    chunk size: ...
    workers: ...
    buffer depth: ...
Dependencies
    dispatch chunk ready
        -> W13 row tasks
    all gate/up tiles for row block complete
        -> SwiGLU
    hidden row ready
        -> W2 tasks
    W2 output ready
        -> return/combine
Kernel decomposition
    kernel 0: ...
    kernel 1: ...
Buffers
    routed input: ...
    hidden: ...
    returned output: ...
Estimated / measured latency
    ...
```

The purpose is to make it obvious which choices came from the compiler.

## 15. Generality requirements

Do not prematurely implement other architectures.

But avoid MoE-specific abstractions where an equally simple general abstraction exists.

### Runtime sparse relation

Prefer `RelationPlan` over `ExpertRoutePlan` because the same concept should eventually express:

```text
query block -> selected KV block
```

for routed sparse attention.

### Segmented computation

Prefer `SegmentedContract` over `ExpertGMM` because grouped/ragged contractions are useful beyond MoE.

### Result merge

Represent the merge explicitly.

For MoE: weighted vector sum.

For sparse attention this could later be: online-softmax state merge.

This is enough generality for the prototype.

Do not build generic higher-order relation infrastructure or sparse-attention support yet.

## 16. Long-term architecture

If the feasibility work continues to succeed, the emerging compiler model is approximately:

```text
High-level tensor/state program
        ↓
semantic normalization
Map
Contract
Fold / Scan
Relation
SegmentedContract
Placement change
Materialization
        ↓
algebraic and numerical rewrite search
        ↓
tile-lifetime / region planning
        ↓
physical scheduling
tiles
layouts
task families
worker roles
buffers
events
transports
kernel boundaries
        ↓
target-specific primitives
CuTe / QuACK
FlashAttention
grouped GEMM
communication primitives
other expert mainloops
        ↓
kernels and communication operations
```

The reusable intellectual content is:

```text
algebraic restructuring
operation placement during tile lifetimes
structured reduction/state
runtime relation orientation
materialization avoidance
layout planning
bounded producer-consumer pipelines
resource-aware scheduling
```

This can eventually apply to:

- Dense Transformers.
- MoE.
- Routed sparse/retrieval attention.
- Linear attention / Gated DeltaNet / state-space models.
- Higher-order or simplicial attention.
- FSDP communication.
- Pipeline parallelism.

Those are motivations and future validation targets, not current implementation requirements.

## 17. Working philosophy for the prototype

Prefer experiments that answer a compiler-design question.

Do not spend significant time implementing infrastructure whose necessity has not been demonstrated experimentally.

When deciding between implementing a general compiler subsystem and hard-coding one small candidate generator, running it, measuring it, and learning whether the abstraction works, prefer the second.

But distinguish hard-coded search space from hard-coded answer.

It is fine to say:

```text
candidate schedules are:
    A
    B
    C
```

It is not fine to say:

```text
MoE lowers to MoK's exact schedule because we already know it is fast
```

The goal of this stage is to establish that Shuttle can independently recover important implementation structure.

## 18. Immediate definition of done

The present prototype phase is complete when there is a reproducible report showing:

```text
ordinary JAX MoE
    ↓
generic Shuttle semantic recovery
    ↓
runtime sparse relation plan
    ↓
segmented expert contractions
    ↓
compiler-generated distributed tile-flow schedule
    ↓
correct BF16 execution on GB200
    ↓
direct comparison with tuned Mixture-of-Kittens
```

and the generated implementation is close enough to MoK to make further investment in generalized schedule synthesis credible.

At that point, decide based on the measurements whether the next project should be:

- Better MoE schedule synthesis.
- One-kernel persistent generation.
- Backward.
- MXFP8.
- Formal Shuttle MLIR dialects.
- XLA integration.
- Routed sparse attention as the first cross-domain reuse test.

Those are follow-on decisions.

For now the task is simpler:

Show that first-principles whole-program rewrites and scheduling can turn an ordinary routed-MoE graph into a genuinely competitive distributed implementation.

## 19. Working decisions

- Shuttle is the project name. Keep the existing `tile_lifetime` package and module names stable during the prototype.
- BF16 MoE lowering may reorder route-slot accumulation, but execution must be deterministic. Unordered atomic accumulation is not an accepted implementation.
- The headline comparison replays Mixture-of-Kittens' seeded random routing fixture. Exactly balanced, rank-spread routes are retained as a diagnostic workload rather than the primary performance claim.
