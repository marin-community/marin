# Prototype specification: compiling Transformer graphs into tile-lifetime programs

Project framing and the authoritative expert-parallel extension are defined in [Shuttle: Prototype Brief and Big-Picture Specification](shuttle_brief.md). This detailed specification remains the dense H100 experiment and implementation reference.

## TL;DR

Build a research compiler prototype that imports a StableHLO graph for a dense, Llama-style Transformer region and lowers it into H100 execution plans composed from high-performance GEMM mainloops with programmable epilogues, streaming exact attention, and small auxiliary reductions. The compiler must recover semantic operations, prove the legality of algebraic rewrites and tile-lifetime attachments, plan layouts and materializations, emit an executable and inspectable plan, match a JAX reference numerically, and run within 10% of a manually assembled CODA-plus-FlashAttention oracle on supported primary shapes.

## 1. Project summary

Build a research compiler prototype that takes the tensor-operation graph for a dense Transformer region and converts it into a composition of high-performance tiled execution skeletons.

The prototype should demonstrate that a compiler can reason about computation during the lifetime of a tile instead of treating every framework operator as a separate kernel and every operator edge as a fully materialized tensor.

The initial compiler targets one NVIDIA H100 GPU and the forward pass of a Llama-style Transformer. It supports two primary execution skeletons:

1. A fixed high-performance GEMM mainloop with a programmable tile epilogue, following the CODA model. [1, 2]
2. A streaming exact-attention kernel using online softmax, asynchronous memory movement, and producer-consumer scheduling, following the FlashAttention family. [3, 4]

Small auxiliary reductions and unavoidable tensor materializations complete the execution model.

The research question is whether a compiler can start from an ordinary tensor graph and automatically recover substantially the same algebraic decomposition, fusion placement, layouts, and materialization boundaries that an expert would choose when assembling CODA-style GEMMs and FlashAttention-style attention.

The compiler succeeds when it derives these decisions from operation semantics and legality rules, emits an executable plan, and approaches the performance of a manually assembled CODA-plus-FlashAttention implementation.

## 2. Motivation

A framework-level Transformer graph contains operations such as GEMMs, residual additions, RMSNorm, RoPE, SwiGLU, masking, softmax, and attention reductions. These appear as a sequence or DAG of tensor operators.

That representation loses an important fact about efficient GPU execution: many operations can execute while a GEMM or attention tile is already on chip. Materializing the tile and launching another kernel is unnecessary. CODA demonstrates this approach for nearly all non-attention computation in a standard Transformer block, while FlashAttention demonstrates the corresponding IO-aware tiled algorithm for exact attention. [1, 3]

For example, the framework graph may contain:

```text
output projection
-> residual addition
-> RMSNorm
-> gate/up projection
-> SwiGLU
-> down projection
```

An efficient implementation may instead use:

```text
output projection GEMM
  epilogue:
    add residual
    apply RMSNorm weight
    compute partial RMS statistic
small reduction:
    combine partial RMS statistics
gate/up GEMM
  epilogue:
    apply delayed inverse-RMS scale
    apply SwiGLU
down projection GEMM
```

Similarly, semantic attention may appear as:

```text
scores = Q @ K^T
probabilities = softmax(scores + mask)
output = probabilities @ V
```

An efficient implementation does not materialize scores or probabilities. It keeps a query tile resident, streams K/V tiles, and maintains online softmax state.

The compiler must answer questions that ordinary operator scheduling does not:

- During which tile's lifetime can an operation execute?
- Can an operation attach to its producer's epilogue?
- Can it attach to a consumer's input or output path?
- Can a cross-tile reduction be split into tile partials and a small reducer?
- Can an apparent dependency be moved algebraically through a linear map?
- Does a semantic subgraph require a different tiled algorithm, such as online attention?
- Which tensors require global-memory materialization?
- Which layout should a producer emit so that the next skeleton can consume it directly?
- What resource and synchronization costs result from a proposed attachment?

The prototype should make these decisions explicit and inspectable.

## Technical basis and required reading

This project combines and generalizes ideas demonstrated by several existing systems. Treat these systems as executable references and performance oracles. Background reading alone is insufficient.

### CODA

CODA is the primary reference for compiling non-attention Transformer computation into GEMM-plus-epilogue programs. It fixes a high-performance GEMM mainloop and exposes a constrained set of composable epilogue operations, including scaling, accumulation, pairwise transformations, and partial reductions. Many operations ordinarily represented as separate framework kernels can be algebraically reparameterized to run while a GEMM output tile remains on chip. [1, 2]

The compiler should recover CODA-style programs from the input tensor graph instead of requiring the frontend to identify an entire pre-fused CODA operation. Use CODA's implementation and tests as:

- A source of exact algebraic rewrites.
- A reference epilogue language.
- A source of representative Transformer regions.
- A correctness oracle for generated GEMM-plus-epilogue programs.
- A performance oracle for the H100 GEMM backend.

### FlashAttention

FlashAttention is the primary algorithmic reference for lowering semantic attention into an exact tiled computation that avoids materializing the score and probability matrices. Its online-softmax recurrence and IO-aware tiling define the semantic attention skeleton used by this compiler. [3]

FlashAttention-3 is the primary physical-scheduling reference for H100. It uses Hopper's asynchronous Tensor Cores and Tensor Memory Accelerator, warp specialization, and interleaving of matrix multiplication with softmax work. [4]

The compiler is responsible for recognizing semantic attention, selecting the tiled online algorithm, establishing layout contracts, and choosing among supported physical configurations. It may initially instantiate an existing FlashAttention-3 kernel family instead of independently generating every attention instruction.

### ThunderKittens

ThunderKittens is a reference for expressing high-performance kernels using register tiles, shared-memory tiles, asynchronous producer-consumer templates, and persistent grid-level scheduling. Its abstractions cover three useful levels:

- Warp-level tile values and operations.
- Thread-block-level asynchronous pipelines.
- Grid-level persistent execution patterns.

Evaluate ThunderKittens as:

- A backend primitive library.
- An executable model for the compiler's tile and pipeline representation.
- A source of compact H100 GEMM and attention implementations used for comparison.

The compiler's semantic IR should not depend on ThunderKittens-specific types. [5, 6]

### CUTLASS and CuTe

CUTLASS and CuTe are the principal NVIDIA references for hierarchical tensor layouts, tiled copies and matrix operations, TMA pipelines, WGMMA execution, collective mainloops, and programmable epilogues. CODA itself is built on the CUTLASS CuTe DSL. [2, 7, 8]

Treat CUTLASS/CuTe as:

- The default source of high-performance H100 GEMM components.
- The layout and matrix-instruction legalization layer.
- A correctness and performance oracle for generated kernels.
- A possible direct backend for the GEMM epilogue AST.

The compiler owns semantic graph recovery, rewrites, attachment placement, and region planning. CUTLASS/CuTe owns architecture-specific tensor layouts and efficient implementation of selected physical operations.

### Hopper execution model

Ground the backend resource model in NVIDIA's Hopper programming and tuning documentation. Hopper's Tensor Memory Accelerator can move multidimensional tensors between global and shared memory asynchronously without using registers for the transfer, while WGMMA allows asynchronous warpgroup-level matrix operations. These features make producer-consumer worker specialization and explicit pipeline planning central to high-performance H100 execution. [9]

### StableHLO, JAX export, and XLA

StableHLO is the initial graph interchange format. Its specification defines the operation semantics that the importer must preserve. JAX's export API provides a supported path for lowering and serializing JAX functions as StableHLO programs. [10, 11]

XLA:GPU is the principal framework baseline. Its GPU pipeline performs buffer assignment, fusion, code generation through LLVM or Triton, and library selection. Compare the compiler's region-level output against stock XLA and use XLA's behavior to define realistic frontend graph fixtures. [12]

The first prototype may remain standalone after importing StableHLO. Integration into XLA can follow once the semantic and tile-lifetime representations have been validated.

### Related compiler work

Tawa introduces asynchronous references and automatically partitions tiled programs into producer and consumer warps. It is a design reference for representing asynchronous availability without directly embedding CUDA barriers in the high-level IR. [13]

This project's initial scope begins from a tensor graph, performs Transformer-specific algebraic rewriting and execution-skeleton selection, and then generates or instantiates a tiled implementation. Templates or explicit schedule choices may initially provide warp specialization.

## 3. Target and reference workload

### 3.1 Hardware target

Target NVIDIA H100 SXM-class GPUs with:

- Compute capability 9.0.
- BF16 inputs and weights.
- FP32 accumulation where required by the reference semantics.
- TMA-based asynchronous global-to-shared transfers.
- WGMMA-based tensor-core mainloops.
- Hopper warpgroup specialization.

The relevant Hopper mechanisms and constraints are described in NVIDIA's Hopper Tuning Guide. [9]

The generated code may rely on CODA's CuTe DSL implementation, CUTLASS/CuTe, ThunderKittens, or narrowly extracted primitives from those projects. These systems provide architecture-specific kernel machinery; the compiler remains responsible for semantic recovery, rewrites, attachment placement, and region planning. [2, 5–8]

### 3.2 Model family

Use a pre-norm, Llama-style decoder block with:

- RMSNorm.
- Grouped-query attention.
- RoPE.
- Causal masking.
- Q, K, and V projections.
- Attention output projection.
- Gated MLP using SwiGLU.
- Residual connections.
- No bias in linear layers unless a test explicitly enables one.

The canonical semantic block is:

```text
a  = RMSNorm(x, gamma_attn)
q  = a @ Wq
k  = a @ Wk
v  = a @ Wv
q, k = RoPE(q, k)
h  = CausalAttention(q, k, v)
p  = h @ Wo
x1 = x + p
m  = RMSNorm(x1, gamma_mlp)
g  = m @ Wgate
u  = m @ Wup
z  = SiLU(g) * u
d  = z @ Wdown
x2 = x1 + d
```

The optimization region should contain at least two consecutive blocks or equivalent boundary projections. This gives the compiler both a producer and a consumer around each RMSNorm:

```text
QKV projection for block L
-> attention
-> output projection
-> residual/RMSNorm
-> MLP gate/up projection
-> SwiGLU
-> down projection
-> residual/RMSNorm
-> QKV projection for block L+1
```

The principal benchmark region may initially contain one block plus the next block's QKV projection.

### 3.3 Required configurations

Debug configuration:

```text
hidden size:       512
intermediate size: 1408 or another valid gated size
query heads:       8
KV heads:          2
head dimension:    64
sequence length:   128
batch size:        1-2
```

Primary configuration:

```text
hidden size:       4096
intermediate size: 14336
query heads:       32
KV heads:          8
head dimension:    128
sequence lengths:  2048 and 4096
token counts:      selected to exercise large projection GEMMs
```

Add at least one smaller and one larger projection shape where hardware permits. Shape support may be bucketed and static.

## 4. Compiler contract

### 4.1 Input

The compiler receives a normalized tensor graph containing:

- Tensor shapes and dtypes.
- Operation semantics.
- Constant parameters.
- Broadcast dimensions.
- Reduction axes.
- Transpose and reshape semantics.
- Precision and accumulation requirements.
- Causal-mask and grouped-query-attention metadata.
- Optional source locations or frontend operation names.

The acceptance path should begin from StableHLO exported from a JAX implementation of the reference block. StableHLO provides the versioned operation semantics used by the importer, and `jax.export` provides the reference export and serialization path. [10, 11]

Keep StableHLO fixtures in the repository so compiler tests do not depend on repeatedly invoking a particular JAX version. A Python graph builder may also construct equivalent graphs for small unit tests.

The initial importer only needs the operations used by the reference workload:

```text
dot_general
add
multiply
divide
rsqrt
reduce
reshape
transpose
broadcast_in_dim
slice / dynamic_slice where required
concatenate
sine
cosine
exponential
maximum
select
compare
convert
iota
```

The importer should preserve enough provenance to produce readable diagnostics.

### 4.2 Output

The compiler produces an executable region plan containing:

- A sequence or DAG of execution skeleton instances.
- GEMM dimensions and layouts.
- Attached GEMM epilogue programs.
- Attention algorithm and physical schedule parameters.
- Auxiliary reductions.
- Materialized values and their layouts.
- Saved values required by the region.
- Kernel dependencies.
- Buffer allocations and aliases.
- Expected global-memory traffic.
- Estimated resource use.
- Generated source or backend configuration.
- A human-readable explanation of each transformation.

The emitted plan may contain multiple kernels. Kernel count alone is not the optimization objective. The compiler should eliminate unnecessary activation-sized materializations while retaining high-quality compute skeletons.

## 5. Intermediate representation

The compiler should use a small semantic IR followed by a tile-lifetime plan. Avoid building a general GPU instruction IR.

### 5.1 Semantic graph IR

Each value contains:

```text
TensorValue {
    id
    shape
    dtype
    logical_layout
    producer
    consumers
    source_location
}
```

Each operation contains:

```text
TensorOp {
    kind
    inputs
    outputs
    attributes
    exact_semantics
}
```

After semantic recovery, the graph may contain higher-level operations:

```text
Linear
ResidualAdd
RMSNorm
RoPE
SwiGLU
ScaledDotProductAttention
CausalMask
```

These operations must retain exact dimensions, layouts, epsilon values, accumulation types, and grouped-query semantics.

### 5.2 Operation-property annotations

Compiler analysis should infer or assign properties such as:

```text
elementwise
pairwise-local
row-separable
column-separable
tile-local
associative reduction
row reduction
column reduction
indexed selection
stateful online reduction
linear in input
row-scalar multiplication
layout-sensitive
dimension-preserving
dimension-reducing
dimension-expanding
```

These properties determine legal placement and algebraic rewrites.

### 5.3 Execution skeleton IR

The initial skeleton types are:

```text
GemmSkeleton
StreamingAttentionSkeleton
AuxiliaryReductionSkeleton
MaterializedTransformSkeleton
```

A GEMM skeleton contains:

```text
GemmSkeleton {
    A
    B
    D
    M, N, K
    input_layouts
    output_layout
    accumulation_type
    mainloop_variant
    epilogue_program
    emitted_auxiliary_values
    required_auxiliary_values
}
```

A streaming-attention skeleton contains:

```text
StreamingAttentionSkeleton {
    Q
    K
    V
    output
    query_block_size
    key_value_block_size
    head_dimension
    number_of_query_heads
    number_of_kv_heads
    causal
    scale
    input_layouts
    output_layout
    pipeline_stages
    producer_workers
    consumer_workers
    online_state
}
```

An auxiliary reduction contains:

```text
AuxiliaryReductionSkeleton {
    input_partials
    output
    reduction_operator
    reduction_axes
    numerical_policy
}
```

### 5.4 Attachments

An operation that does not need its own skeleton can be attached to a tile lifetime:

```text
Attachment {
    operation
    site
    inputs
    outputs
    legality_requirements
}
```

Valid sites include:

```text
gemm_epilogue
attention_input_transform
attention_score_transform
attention_online_update
attention_output_transform
auxiliary_reduction
materialized_transform
```

Each attachment should report why it is legal and why the selected site is preferred.

### 5.5 Materialization record

Every logical tensor edge should receive one of these dispositions:

```text
materialize
alias
recompute
epilogue-only
partial-reduction-only
internal-attention-state
```

The compiler's textual report must list all activation-sized materializations.

## 6. Execution skeletons

### 6.1 GEMM plus programmable epilogue

Use CODA's H100 implementation as the initial semantic and performance reference for a fixed high-performance GEMM mainloop with a programmable epilogue. CODA is implemented using the CUTLASS CuTe DSL and provides composable epilogue visitors, generated kernel examples, tests, and representative Transformer-level compositions. [1, 2]

The epilogue language should support:

- Elementwise maps.
- Pairwise maps over adjacent features.
- Row-vector loads and broadcasts.
- Column-vector loads and broadcasts.
- Tile loads.
- Tile stores.
- Residual accumulation.
- Type conversion.
- Row and column partial reductions.
- Indexed extraction.
- Stateful max and sum-exp updates where useful.
- Multiple auxiliary outputs.
- Conditional masking for tile tails.

Representative primitives include:

```text
add
multiply
fma
silu
pairwise_swiglu
pairwise_rope
scale_row
scale_column
load_tile
load_row_vector
load_column_vector
store_tile
partial_sum
partial_sum_square
partial_max
partial_logsumexp
select_index
convert
```

Check the initial primitive set against CODA's published abstraction and implementation instead of designing it entirely from scratch. [1, 2]

The epilogue compiler should perform:

- Primitive composition.
- Common-subexpression elimination.
- Load reuse.
- Dead-output elimination.
- Register-liveness estimation.
- Layout validation.
- Resource validation.
- Code generation into the selected backend.

The compiler must preserve a path to the unmodified high-performance mainloop. Epilogue complexity should not silently replace the mainloop with generic scalar CUDA.

### 6.2 Streaming attention

Semantic attention should lower to an exact online-softmax algorithm. This lowering follows the IO-aware exact-attention algorithm introduced by FlashAttention. [3]

For one query tile and a sequence of K/V tiles, maintain:

```text
m = running row maximum
l = running row sum of exp(score - m)
o = running output accumulator
```

For each K/V tile:

```text
s       = scale * Q_tile @ K_tile^T
s       = apply causal or bounds mask
m_new   = max(m, row_max(s))
alpha   = exp(m - m_new)
p       = exp(s - m_new)
l_new   = alpha * l + row_sum(p)
o_new   = alpha * o + p @ V_tile
m, l, o = m_new, l_new, o_new
```

At completion:

```text
output = o / l
```

The physical H100 schedule should support:

- TMA producers loading K/V tiles.
- WGMMA consumers performing QK and PV matrix operations.
- Circular shared-memory buffers.
- Producer-consumer barriers.
- Interleaving softmax work with tensor-core work.
- Causal and tail masking.
- Grouped-query attention.
- BF16 output.
- Optional log-sum-exp output for future backward support.

FlashAttention-3 is the principal reference for overlapping TMA transfers with WGMMA, assigning specialized producer and consumer workers, and interleaving softmax work with matrix operations on Hopper. [4]

The compiler should choose from a finite template family instead of synthesizing arbitrary CUDA control flow.

Initial schedule choices include:

- Q block size.
- K/V block size.
- Number of pipeline stages.
- Number of producer warpgroups.
- Number of consumer warpgroups.
- Q residency strategy.
- K/V shared-memory layout.
- Output layout.

An existing FlashAttention-3-compatible implementation may provide the physical kernel body. The compiler remains responsible for semantic recognition, algorithm selection, shape specialization, layout contracts, and integration with adjacent skeletons.

## 7. Required compiler transformations

### 7.1 Semantic recovery

Recover higher-level operations from normalized StableHLO.

Required recognizers:

- Linear projections from `dot_general`.
- RMSNorm.
- Residual addition.
- RoPE.
- SwiGLU.
- Scaled dot-product attention.
- Causal masking.
- GQA head replication or indexing.

Recognition should tolerate ordinary reshape, transpose, broadcast, and concatenate variations produced by JAX.

Each recognizer must validate exact semantics. A failed recognizer should identify the mismatched condition and leave the graph executable through the reference path.

### 7.2 GEMM-residual-RMSNorm-GEMM reparameterization

Recognize:

```text
h  = x @ W0
u  = h + residual
n  = u * gamma * inverse_rms(u)
y  = n @ W1
```

Rewrite it as:

```text
GEMM 0:
    h = x @ W0
Epilogue 0:
    u           = h + residual
    v           = u * gamma
    rms_partial = partial_sum_square(u)
    store v
Auxiliary reduction:
    r = inverse_sqrt(combine(rms_partial) / hidden_size + epsilon)
GEMM 1:
    t = v @ W1
Epilogue 1:
    y = t * r
```

The legality argument is:

- Residual addition is tile-local.
- Multiplication by gamma is tile-local.
- RMS can be decomposed into tile partials and a small row reduction.
- `r` is one scalar per row.
- Row scaling commutes through right multiplication by `W1`.
- The delayed scale can therefore execute in the second GEMM epilogue.

The generated compiler report should show this derivation instead of reporting only that a pattern matched.

Required checks include:

- The scale is constant across the GEMM reduction dimension.
- Broadcasting semantics match row scaling.
- No intervening consumer observes the canonically normalized activation.
- Epsilon and accumulation precision are preserved.
- Required values are available when each epilogue executes.

### 7.3 SwiGLU fusion

Recognize either:

```text
gate = x @ Wgate
up   = x @ Wup
y    = SiLU(gate) * up
```

or a combined projection whose output dimension contains adjacent gate/up pairs.

Produce a GEMM form whose epilogue applies pairwise SwiGLU before the expanded intermediate is written.

The planner should consider:

- A single concatenated gate/up GEMM.
- A dual-output or grouped GEMM when supported by the backend.
- Pair layout in accumulator registers.
- Output dimension reduction.
- Direct emission in the input layout expected by the down projection.

The execution plan should contain no standalone activation-sized SwiGLU kernel for supported shapes.

### 7.4 RoPE fusion

Recognize Q/K projections followed by pairwise rotary transforms.

Place RoPE in the projection epilogue when:

- Paired rotary dimensions are adjacent or can be made adjacent through the output layout.
- Required sine and cosine vectors can be loaded efficiently.
- The output layout is consumable by the attention skeleton.

The planner should distinguish Q, K, and V output regions and apply RoPE only to Q/K.

The resulting plan should avoid writing unrotated Q/K tensors.

### 7.5 Attention algorithm selection

Recognize semantic scaled dot-product attention and lower it to the streaming-attention skeleton.

The recognition must prove:

- Softmax reduction axis.
- Scale factor.
- Mask semantics.
- Query-head and KV-head mapping.
- Output contraction with V.
- No graph consumer requires materialized score or probability tensors.

The resulting plan must not contain an activation proportional to sequence length squared.

### 7.6 Residual and next-layer planning

After the attention output projection and MLP down projection, repeat the residual/RMSNorm transformation across the following projection boundary.

The optimizer should operate across ordinary module and layer boundaries. Region construction should therefore be based on legal optimization boundaries, not frontend module names.

## 8. Planning algorithm

The planner should use a staged process.

### 8.1 Normalize

- Canonicalize broadcasts.
- Fold trivial reshapes and transposes.
- Normalize reduction axes.
- Canonicalize concatenated versus separate projections.
- Record logical layouts without prematurely forcing physical layouts.

### 8.2 Recover semantics

Replace low-level subgraphs with verified semantic operations.

### 8.3 Enumerate legal transformations

Generate a small set of alternatives:

- Standalone operation.
- Producer epilogue attachment.
- Auxiliary partial reduction.
- Algebraic delayed scaling.
- Pairwise epilogue.
- Streaming-attention lowering.
- Alternative layout at a skeleton boundary.
- Recompute or materialize where both are legal.

### 8.4 Partition into skeletons

Choose GEMM, attention, and reduction skeleton instances.

### 8.5 Plan layouts

Determine:

- GEMM operand layouts.
- Epilogue output layouts.
- Q/K/V attention layouts.
- Attention output layout.
- Whether a transpose is absorbed by a producer or consumer.
- Whether one output needs multiple representations.

### 8.6 Score candidates

Use hard legality constraints followed by a simple analytical score.

The initial score should include:

```text
estimated HBM bytes
number and size of activation materializations
kernel launch count
auxiliary-reduction bytes
layout-conversion bytes
GEMM mainloop quality
epilogue instruction estimate
epilogue register-liveness estimate
shared-memory use
attention pipeline quality
estimated occupancy loss
```

Prioritize preservation of strong GEMM and attention kernels over eliminating a small materialization at severe occupancy cost.

### 8.7 Generate and benchmark

For a small set of competitive plans:

- Generate code or backend configurations.
- Compile and cache artifacts.
- Run short correctness tests.
- Benchmark steady-state latency.
- Persist the best plan by graph fingerprint, shape, dtype, GPU, and toolchain revision.

The first implementation may use empirical tuning only for physical skeleton parameters. Algebraic legality should come from compiler analysis.

Tawa's asynchronous-reference representation is relevant future work for deriving producer-consumer worker roles and pipelines from tiled programs instead of selecting only among explicit templates. [13]

## 9. Backend implementation

### 9.1 GEMM backend

Begin from CODA's H100 GEMM and epilogue implementation, which is built on the CUTLASS CuTe DSL, or from an equivalent CUTLASS/CuTe mainloop with a compatible programmable epilogue interface. Retain ThunderKittens GEMMs as an additional implementation and abstraction reference. [2, 5–8]

The backend interface should accept:

```text
GEMM shape
operand layouts
dtype and accumulation type
epilogue AST
auxiliary inputs
auxiliary outputs
tile shape
pipeline configuration
```

Generate source or template instantiations from the epilogue AST.

The repository should include a small collection of golden epilogue programs and generated-source snapshots.

### 9.2 Attention backend

Use the official FlashAttention-3 Hopper implementation, a ThunderKittens implementation with equivalent semantics, or a narrowly adapted kernel family based on them. [4, 6, 14]

The backend interface should accept:

```text
head dimension
query block size
K/V block size
causal mode
GQA ratio
dtype
input layouts
output layout
pipeline stages
worker configuration
```

The backend may wrap or instantiate an existing implementation initially. The wrapper should expose physical configuration and layout requirements instead of hiding the entire attention implementation behind an opaque framework call.

### 9.3 Auxiliary reductions

Implement lightweight reduction kernels for:

- Combining RMS partial sums.
- Computing inverse RMS.
- Any small row statistics emitted by an epilogue.

These kernels should operate on partial-statistic buffers instead of rereading full activation tensors.

### 9.4 Runtime

Provide a small runtime that:

- Allocates intermediate and partial buffers.
- Launches the skeleton sequence.
- Reuses and aliases buffers according to the plan.
- Records CUDA events.
- Supports CUDA Graph capture for stable repeated execution.
- Reports per-kernel and region timing.
- Caches compiled binaries and tuned configurations.
- Exposes a simple Python callable.

## 10. Correctness and numerical policy

### 10.1 Reference implementation

Provide a straightforward JAX implementation of the same semantic region.

Use it to generate:

- Inputs.
- Parameters.
- Reference outputs.
- Intermediate values for debug tests.

### 10.2 Differential testing

Test:

- Each semantic recognizer.
- Each algebraic rewrite.
- Each epilogue program.
- Each auxiliary reduction.
- Streaming attention.
- The complete generated region.

Use randomized inputs and multiple seeds.

Report:

```text
maximum absolute error
maximum relative error
mean absolute error
selected percentile errors
NaN and infinity counts
```

### 10.3 Rewrite equivalence

For every algebraic rewrite, provide a CPU- or JAX-level equivalence test independent of the GPU backend.

This is especially important for:

- Delayed RMS scaling.
- Gamma placement.
- Concatenated gate/up projections.
- Pairwise RoPE layout.
- GQA head mapping.
- Online-softmax updates.

### 10.4 Precision

Record explicitly:

- Input dtype.
- Weight dtype.
- GEMM accumulation dtype.
- Epilogue computation dtype.
- Partial-reduction dtype.
- Final output dtype.

Do not rely on backend defaults for numerically significant operations.

## 11. Benchmarking

### 11.1 Baselines

Compare three implementations:

1. Framework baseline: the reference JAX block compiled with stock XLA:GPU.
2. Manual oracle: a manually assembled sequence using CODA's H100 kernels and the official FlashAttention-3 implementation, with CUTLASS/CuTe and ThunderKittens component benchmarks where useful. [2, 4, 6–8, 14]
3. Compiler-generated implementation: the plan generated from the StableHLO graph.

The manual oracle defines the practical performance target. The framework baseline determines whether the optimization is useful.

### 11.2 Metrics

Record:

```text
end-to-end region latency
tokens per second
kernel count
activation bytes written to HBM
activation bytes read from HBM
auxiliary-reduction bytes
GEMM TFLOP/s
attention latency
register use
spills
shared-memory use
occupancy
compile time
tuning time
cached-plan hit rate
```

Use Nsight Compute or Nsight Systems selectively for representative runs.

### 11.3 Plan inspection metrics

For each generated plan, also report:

- Number of standalone elementwise kernels.
- Number of standalone normalization kernels.
- Number of layout-conversion kernels.
- Number of activation-sized materializations.
- Operations attached to each GEMM epilogue.
- Attention algorithm and block sizes.
- Estimated versus measured latency.

### 11.4 Performance targets

For supported primary shapes:

- The generated implementation should be within 10% of the manually assembled CODA-plus-attention oracle at region level.
- No standalone residual, RoPE, SwiGLU, or full-activation RMSNorm kernel should remain when the required rewrite is legal.
- Attention should avoid score and probability materialization.
- At least one primary end-to-end configuration should improve region latency over a carefully tuned stock XLA baseline by 10% or more.
- Supported configurations that regress materially should select the framework or library fallback instead of forcing the generated plan.

Performance failures are acceptable research results when the repository identifies the responsible kernel, layout, materialization, or resource cost.

## 12. Structural acceptance tests

### Test A: Residual and RMSNorm recovery

Input graph:

```text
GEMM -> residual add -> RMSNorm -> GEMM
```

Expected plan:

```text
GEMM with residual/gamma/partial-RMS epilogue
small RMS reduction
GEMM with delayed row-scale epilogue
```

### Test B: SwiGLU recovery

Input graph:

```text
gate GEMM
up GEMM
SiLU
multiply
```

Expected plan:

```text
combined or dual gate/up projection
pairwise SwiGLU epilogue
```

### Test C: RoPE recovery

Input graph:

```text
Q/K projection
reshape/split
sine/cosine pairwise rotation
```

Expected plan:

```text
projection GEMM with pairwise RoPE epilogue
attention-consumable output layout
```

### Test D: Attention recovery

Input graph:

```text
QK^T
scale
causal mask
softmax
PV
```

Expected plan:

```text
streaming attention skeleton
online max/sum/output state
no sequence-squared materialization
```

### Test E: Full region

Input graph:

```text
QKV projection
RoPE
attention
output projection
residual/RMSNorm
gate/up
SwiGLU
down projection
residual/RMSNorm
next QKV projection
```

Expected plan:

- QKV projection with RoPE attachment.
- Streaming attention.
- Output projection with residual/gamma/partial-RMS attachment.
- Lightweight RMS reduction.
- Gate/up projection with delayed scale and SwiGLU.
- Down projection with residual/gamma/partial-RMS attachment.
- Lightweight RMS reduction.
- Next QKV projection with delayed scale and RoPE.

## 13. Implementation sequence

### Milestone 0: Reproduce the oracles

- Build and benchmark a representative CODA kernel on H100.
- Build and benchmark the selected FlashAttention implementation.
- Record toolchain and hardware configuration.
- Pin exact commits of CODA, FlashAttention, ThunderKittens, CUTLASS, JAX, and StableHLO in the results report. Record all local modifications.
- Create a manual composed reference region.
- Establish correctness and timing scripts.

Exit condition: the repository can reproduce stable component and region measurements.

### Milestone 1: Semantic graph and importer

- Implement the semantic graph IR.
- Export the reference JAX region to StableHLO.
- Import the required operation subset.
- Canonicalize common reshape, transpose, broadcast, and reduction forms.
- Produce readable graph dumps.

Exit condition: the compiler reconstructs the reference region and compares it structurally with the JAX implementation.

### Milestone 2: CODA recovery

- Add locality/property analysis.
- Implement GEMM epilogue AST.
- Implement residual/RMSNorm/GEMM rewrite.
- Implement SwiGLU attachment.
- Implement RoPE attachment.
- Generate executable GEMM epilogues.
- Add auxiliary RMS reduction.

Exit condition: all required non-attention structural tests pass and generated kernels are competitive with corresponding manually authored CODA kernels.

### Milestone 3: Attention recovery

- Recognize semantic attention.
- Implement the streaming-attention skeleton.
- Integrate the selected H100 attention backend.
- Support causal masking, GQA, and head dimensions 64 and 128.
- Plan layouts from QKV projection into attention.

Exit condition: attention structural tests pass, sequence-squared tensors are absent, and performance is competitive with the selected attention oracle.

### Milestone 4: Full-region planning

- Partition a full Transformer region into skeletons.
- Optimize layout boundaries.
- Allocate and alias buffers.
- Generate a complete executable plan.
- Capture repeated execution in a CUDA Graph where useful.

Exit condition: the compiler-generated region is numerically correct and satisfies the plan-inspection requirements.

### Milestone 5: Cost model and autotuning

- Enumerate a bounded set of legal plans.
- Add analytical pruning.
- Tune physical GEMM and attention configurations.
- Persist results.
- Compare predictions with measurements.

Exit condition: plan selection reliably chooses near-best candidates across the benchmark matrix.

### Milestone 6: Results

- Produce final benchmark tables.
- Save representative Nsight traces.
- Explain remaining materializations.
- Identify the largest gap to the manual oracle.
- Recommend the next compiler feature based on measurements.

## 14. Repository layout

A suggested layout is:

```text
compiler/
    ir.py
    stablehlo_import.py
    canonicalize.py
    semantic_recovery.py
    properties.py
    rewrites/
        rmsnorm.py
        swiglu.py
        rope.py
        attention.py
    skeletons/
        gemm.py
        attention.py
        reduction.py
    planner.py
    cost_model.py
    layouts.py
    plan_format.py
backends/
    h100/
        gemm/
        attention/
        reduction/
        runtime/
reference/
    llama_region.py
    manual_oracle.py
tests/
    importer/
    semantics/
    rewrites/
    plans/
    numerical/
    integration/
benchmarks/
    configs.py
    run_components.py
    run_regions.py
    profile.py
fixtures/
    stablehlo/
    expected_plans/
docs/
    architecture.md
    transformations.md
    results.md
    limitations.md
```

Generated source, binaries, and tuning results should live under a cache directory excluded from version control.

## 15. Required developer-facing outputs

Support these options for every compilation:

```text
--dump-semantic-graph
--dump-candidates
--dump-selected-plan
--explain-rewrites
--dump-layouts
--dump-generated-source
--report-materializations
--report-resource-estimates
```

A selected-plan dump should be understandable without reading generated CUDA. For example:

```text
Region 0
Skeleton 0: QKV GEMM
  Output layout: attention_qkv_hopper
  Epilogue:
    apply delayed RMS row scale
    apply RoPE to Q and K
    store Q/K/V
Skeleton 1: Streaming attention
  Q block: 128
  K/V block: 128
  Pipeline stages: 3
  Causal: true
  GQA ratio: 4
Skeleton 2: Output projection GEMM
  Epilogue:
    add residual x
    multiply gamma_mlp
    emit RMS partial sum of squares
Skeleton 3: RMS partial reduction
Skeleton 4: Gate/up GEMM
  Epilogue:
    apply delayed inverse RMS
    apply pairwise SwiGLU
```

Every rewrite explanation should state:

- The original graph fragment.
- The transformed graph fragment.
- The semantic property used.
- The legality checks.
- The estimated benefit.
- Any numerical change in operation ordering.

## 16. Engineering guidance for an autonomous coding agent

Work in small, testable increments. Keep the repository runnable after each milestone.

Begin by reproducing the reference kernels and measurements. Do not build compiler infrastructure before establishing that the chosen backends work on the available H100 environment.

Treat the semantic graph, expected plans, correctness tests, and benchmark harness as durable artifacts. Generated CUDA may be replaced as the implementation evolves.

Prefer a narrow working importer over a broad incomplete StableHLO implementation. Preserve input fixtures and emit explicit diagnostics for unsupported operations.

Use existing expert-written matrix and attention mainloops. Spend implementation effort on semantic recovery, algebraic rewriting, epilogue generation, layout contracts, and region planning.

Do not count a wrapper around a pre-fused complete Transformer region as compiler recovery. The compiler must independently derive the placement of residual operations, RMS partials, delayed row scales, RoPE, and SwiGLU from the input graph. The attention backend may be templated, but selection of the streaming algorithm and its semantic/layout configuration must be visible in the generated plan.

Whenever performance differs from the oracle, record:

- The exact configuration.
- Generated plan.
- Measured timings.
- Relevant profiler evidence.
- Current hypothesis.
- The smallest next experiment.

Maintain `docs/progress.md` with completed milestones, current blockers, and reproducible commands.

## 17. Deliverables

The completed prototype should contain:

1. A JAX reference implementation of the target Transformer region.
2. StableHLO fixtures for representative configurations.
3. A semantic graph importer and canonicalizer.
4. Semantic recognizers for RMSNorm, RoPE, SwiGLU, and attention.
5. A property and legality analysis.
6. CODA-style algebraic rewrites.
7. An epilogue AST and H100 GEMM backend.
8. A streaming-attention skeleton and H100 backend integration.
9. Auxiliary-reduction kernels.
10. A layout and materialization planner.
11. A bounded cost model and autotuner.
12. Structural, numerical, and integration tests.
13. A manually assembled performance oracle.
14. End-to-end benchmark scripts.
15. Human-readable graph and plan dumps.
16. A results report containing successes, failures, profiler evidence, and recommended follow-up work.

## 18. Definition of done

The project is done when an ordinary JAX implementation of the supported Llama-style region can be exported to StableHLO and passed to the compiler, which then:

1. Recovers the semantic Transformer operations.
2. Reparameterizes the supported non-attention graph into GEMM-plus-epilogue programs and lightweight reductions.
3. Lowers attention to a streamed online-softmax execution skeleton.
4. Selects compatible layouts and materialization boundaries.
5. Emits an executable H100 region plan.
6. Matches the JAX reference numerically.
7. Eliminates the expected standalone memory-bound kernels and sequence-squared attention intermediates.
8. Runs within 10% of the manually assembled CODA-plus-FlashAttention oracle on the primary benchmark configurations.
9. Produces enough explanation and profiling data to understand every remaining performance gap.

Backward compilation, Blackwell scheduling, distributed communication, and expert-parallel MoE are follow-on projects. The dense H100 forward-pass experiment must stand on its own, with clear extension points for those directions.

## 19. Expert-parallel extension: first-principles MoE synthesis

After the dense H100 definition of done is met, extend the compiler to ordinary routed-MoE graphs on a four-GPU GB200 expert-parallel mesh. Mixture-of-Kittens is the pinned correctness and performance oracle. The compiler must not call its complete fused kernel, match an entire MoK graph as one operation, or treat MoK task names as the source IR.

The semantic input uses a global expert axis:

```text
router_logits = x @ router_weight
expert_ids, route_weights = normalized_top_k(router_logits)
shared = shared_down(silu(shared_gate(x)) * shared_up(x))
routed[t, k] = expert_down[expert_ids[t, k]](
    silu(expert_gate[expert_ids[t, k]](x[t]))
    * expert_up[expert_ids[t, k]](x[t])
)
output = shared + sum_k(route_weights[t, k] * routed[t, k])
```

Expert sharding is a physical decision. The semantic graph must not assume that global expert identifiers can index a rank-local weight tensor.

### 19.1 Generic lowering IR

Lower the semantic graph through first-class generic operations:

```text
RouteRelation(source_token, route_slot, global_expert, weight)
ExpertOwnership(global_expert -> owner_rank, local_expert)
GroupBy(owner_rank, local_expert)
CoalesceBy(source_token, owner_rank)  # optional transport optimization
PadSegments(alignment, capacity_policy)
Exchange(grouped_rows, destination=owner_rank)
SegmentedContraction(rows, expert_weights, segment_offsets)
PairwiseMap(SwiGLU)
ReverseExchange(rows, source_relation)
ScatterReduce(destination=source_token, weight, operator=sum)
```

Shared-expert projections remain ordinary dense contractions and may overlap routed dispatch and computation. Each transform records its input relation, output relation, legality proof, ordering semantics, and numerical policy.

The exchange relation need not contain one activation row per route. When several routes for one token target experts on the same rank, the planner should consider sending the token activation once per destination rank and expanding the retained route subrelation after receipt. This is a generic relational projection/coalescing decision and must remain distinct from expert segmentation.

### 19.2 Tile flow and buffers

Lower relation operations to a tile-flow graph whose nodes are generic copies, contractions, reductions, pairwise maps, and exchanges. Edges record:

- logical and physical layout;
- tile shape and valid-row mask;
- producer and consumer worker domains;
- readiness granularity;
- fan-out and fan-in;
- lifetime and reuse interval; and
- whether storage is local, symmetric, or remotely addressable.

Derive dispatch, receive, expert-grouped activation, reverse-exchange, and combine buffers from liveness and capacity analysis. Double or ring buffering must be a scheduling choice derived from overlapping lifetimes, not a hard-coded MoK workspace schema.

### 19.3 Physical alternatives

The planner should initially enumerate a bounded family:

- compact sort/group plus grouped GEMM;
- ragged all-to-all plus segmented GEMM;
- DeepEP transport plus segmented GEMM;
- multi-kernel execution with CUDA Graph capture; and
- a generated persistent schedule that overlaps generic exchange and compute stages.

Existing DeepEP, JAX ragged-all-to-all, Pallas/Triton `ragged_dot`, CUTLASS/CuTe, and ThunderKittens primitives may implement individual generic stages. None may hide the complete routed-MoE region behind one pre-fused call.

### 19.4 Acceptance criteria

For the pinned oracle shape—2048 tokens per rank, 384 global experts, 96 experts per rank, top-6 routing, hidden size 7168, and intermediate size 3072—the compiler must:

1. Recover the ordinary global routed-MoE graph from JAX StableHLO.
2. Derive the relation, ownership, segmentation, exchange, contraction, reverse-exchange, and scatter-reduction stages.
3. Derive explicit tile-flow, buffer, readiness, and worker schedules.
4. Emit an executable BF16 four-GB200 plan without invoking the complete MoK kernel.
5. Match the semantic reference within the declared BF16 numerical policy.
6. Report the contribution of routing, transport, grouped contractions, activation, combine, and synchronization to latency.
7. Tune a bounded legal schedule to within 20–30% of the pinned MoK BF16-forward latency on the same hardware and shape.

If the target is missed, preserve the best plan, profiler evidence, measured gap, and smallest next experiment. Do not close the gap by replacing generic lowering with a MoK-specific complete-region template.

## References

### Core papers and implementations

[1] Han Guo, Jack Zhang, Arjun Menon, Driss Guessous, Vijay Thakkar, Yoon Kim, and Tri Dao. "CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs." 2026. [Paper](https://arxiv.org/abs/2605.19269).

[2] Han Guo et al. `coda-kernels`: CODA and the Rapier GEMM-plus-epilogue infrastructure, implemented with CUTLASS CuTe DSL for NVIDIA H100. [Repository](https://github.com/HanGuo97/coda-kernels).

[3] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022. [Paper](https://arxiv.org/abs/2205.14135).

[4] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." 2024. [Paper](https://arxiv.org/abs/2407.08608).

[5] Benjamin Spector, Simran Arora, Aaryan Singhal, Arjun Parthasarathy, Dan Fu, and Christopher Ré. "ThunderKittens: Simple, Fast, and Adorable AI Kernels." ICLR 2025. [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/05dc08730e32441edff52b0fa6caab5f-Abstract-Conference.html).

[6] Hazy Research. ThunderKittens: register-tile, shared-tile, asynchronous-pipeline, and persistent-kernel primitives for NVIDIA GPUs. [Repository](https://github.com/HazyResearch/ThunderKittens).

### NVIDIA kernel and architecture references

[7] NVIDIA. CUTLASS documentation, including the CUTLASS 3.x GEMM model, collective mainloops, tiled MMA/copy abstractions, and CuTe. [Documentation](https://docs.nvidia.com/cutlass/latest/).

[8] NVIDIA. `cutlass`: CUDA C++ templates and Python/CuTe DSLs for high-performance linear algebra. [Repository](https://github.com/NVIDIA/cutlass).

[9] NVIDIA. "NVIDIA Hopper Tuning Guide." Relevant topics include the Tensor Memory Accelerator, asynchronous data movement, thread-block clusters, shared memory, and Hopper Tensor Cores. [Documentation](https://docs.nvidia.com/cuda/hopper-tuning-guide/).

### Compiler input and integration

[10] OpenXLA. "StableHLO Specification." Defines the versioned operation semantics and interchange format used by the importer. [Specification](https://openxla.org/stablehlo/spec). [Repository](https://github.com/openxla/stablehlo).

[11] JAX. "Exporting and serializing staged-out computations" and the `jax.export` API. [Documentation](https://docs.jax.dev/en/latest/export/export.html).

[12] OpenXLA. "XLA:GPU Architecture Overview." Describes XLA's GPU fusion, buffer assignment, library selection, LLVM and Triton emitters, and runtime pipeline. [Documentation](https://openxla.org/xla/gpu_architecture).

### Related scheduling work

[13] Hongzheng Chen, Bin Fan, Alexander Collins, Bastian Hagedorn, Evghenii Gaburov, Masahiro Masuda, Matthew Brookhart, Chris Sullivan, Jason Knight, Zhiru Zhang, and Vinod Grover. "Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References." 2025. [Paper](https://arxiv.org/abs/2510.14719).

### Attention implementation

[14] Dao AI Lab. `flash-attention`: official implementations of FlashAttention, FlashAttention-2, and the FlashAttention-3 Hopper beta. [Repository](https://github.com/Dao-AILab/flash-attention).

## Source-version policy

Pin exact source revisions instead of depending on moving default branches.

Record the following in `docs/results.md`:

```text
CODA commit:
FlashAttention commit:
ThunderKittens commit:
CUTLASS commit:
StableHLO version:
JAX version:
CUDA toolkit:
NVIDIA driver:
GPU model:
GPU clock or power policy:
```

When code is copied or adapted, preserve upstream license and attribution notices and identify the originating file and revision. Review CODA, CUTLASS, FlashAttention, and ThunderKittens independently because their licenses and dependency structures are not identical.
