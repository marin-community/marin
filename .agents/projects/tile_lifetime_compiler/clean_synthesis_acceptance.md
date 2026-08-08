# Shuttle Clean-Synthesis Prototype

## Handoff and Acceptance Specification

### 1. Mission

The immediate goal is to convert the current Shuttle prototype from a strong
semantic planner around several expert-written kernels into a genuine proof
that Shuttle can synthesize high-performance implementations from ordinary
program semantics.

The prototype should establish clean end-to-end proof points for four
structurally different workloads:

1. Dense Transformer.
2. Distributed routed MoE.
3. Routed sparse attention.
4. Stateful recurrent / linear attention, initially Gated DeltaNet or the
   existing equivalent StatefulScan workload.

For every workload:

Start from an ordinary high-level program, recover a generic Shuttle
representation, generate the physical computation from generic compiler
skeletons and ASTs, and achieve performance no worse than 1.20 times the
natural expert implementation for equivalent semantics. A 1.10-times ratio is
the stretch target and should be reported separately; it is not required to
complete this milestone.

FSDP is not part of this milestone.

Do not begin a major production XLA integration project. Continue in prototype
mode: small representations, generated code, standalone execution, direct
benchmarks, and rapid experiments.

Use sub-agents where independent work can proceed in parallel. Keep one
canonical semantic definition and acceptance checklist so parallel work cannot
drift into different definitions of “clean.”

---

### 2. Why this milestone exists

Previous work established that Shuttle can:

* recover useful structure from ordinary StableHLO;
* make meaningful RMS placement, layout, materialization,
  relation-orientation, buffering, worker-allocation, and scheduling
  decisions;
* achieve strong performance around expert physical implementations;
* cleanly generate at least one generic recurrent kernel core.

However, several headline numbers remain contaminated by workload-specific
physical implementations:

```text
Dense:
    generic planning
    → named QuACK/CODA bodies
    → opaque official FA3
MoE:
    generic RelationPlan and scheduling
    → generic-ish GMM
    → opaque semantic DeepEP combine
Sparse attention:
    generic Relation/Fold planning
    → handwritten sparse-attention Triton body
```

These are good planning results.

They do not yet establish the stronger Shuttle thesis:

Whole-program semantic analysis can synthesize the important expert
implementation structure itself rather than merely selecting and arranging
expert kernels.

This milestone closes that gap.

---

### 3. Non-negotiable definition of clean synthesis

A workload counts as a clean Shuttle result only if all of the following hold.

#### 3.1 Natural frontend

The accepted path begins from an ordinary high-level program.

Preferred route:

```text
ordinary JAX
    ↓
StableHLO
    ↓
Shuttle recovery
```

Frozen StableHLO produced from that program is fine for tests.

An accepted end-to-end benchmark may not begin from:

* precomputed routing fixture;
* prebuilt task list;
* hand-authored attention schedule;
* manually assembled Shuttle semantic IR.

Those remain acceptable for isolated backend/unit tests.

#### 3.2 Semantic recognition may name things, but names must erase

Frontend recovery may recognize familiar semantic structures when that
simplifies canonicalization and validation.

For example, it is acceptable to temporarily recognize:

* RMSNorm;
* Softmax;
* causal mask;
* top-k selection;
* RoPE.

But schedule synthesis must not depend on those architecture/operator names.

Before physical planning, named semantics must lower into generic Shuttle
algebra.

The important rule is:

Semantic recognizers may introduce convenient names, but those names must
erase into generic Shuttle primitives before optimization and physical
synthesis.

##### RMSNorm

Recognizing RMSNorm is acceptable as frontend canonicalization.

It must lower to something equivalent to:

```text
square(x)
→ Fold(sum over hidden)
→ divide by hidden size
→ add epsilon
→ rsqrt
→ broadcast row scalar
→ multiply x
→ multiply gamma
```

or compactly:

```text
Fold(sum_square over hidden)
+
Map(rsqrt / broadcast / scale)
```

The compiler must derive transformations such as:

* emit RMS partials from a producer Contract;
* apply inverse-RMS during the next Contract preparation;
* move the row scalar through a linear map;

under an explicit numerical contract from generic Fold/Map/Contract
properties.

It must not contain:

```text
if RMSNorm before GEMM:
    emit CODA schedule
```

##### Softmax

Recognizing softmax is acceptable.

It must lower into generic reduction/map structure such as:

```text
m = Fold(max over K)
z = exp(x - m)
l = Fold(sum over K, z)
output = z / l
```

The compiler may then derive an incremental or mergeable state:

```text
(max, sum_exp)
```

through generic reduction algebra.

The fact that the frontend operation was called softmax must not itself select
FlashAttention.

##### Causality

Causality is independent of softmax.

Represent it as an index-domain predicate:

```text
valid(q, k) = (k <= q)
```

or equivalent generic structure:

```text
DomainRestriction {
    predicate
}
```

Do not introduce a semantic primitive called:

```text
causal_softmax
```

or require an opaque `causal_attention` operation.

The same mechanism should naturally express:

* causal masks;
* sliding windows;
* prefix masks;
* document boundaries;
* block masks;
* runtime sparse relations.

##### Attention

Attention should emerge from:

```text
Contract(Q, Kᵀ)
→ scale Map
→ DomainRestriction / mask
→ normalized-exponential Fold
→ Contract(probabilities, V)
```

The compiler may recognize an attention-shaped subgraph for diagnostics or
region formation, but the performance lowering must operate on the decomposed
structure.

##### Top-k routing

Recognizing top-k selection is acceptable.

It should lower into:

```text
selection
→ Relation
→ RelationPlan
```

rather than directly into MoE-specific scheduling.

#### 3.3 Examples of acceptable and unacceptable recognition

Acceptable:

```text
recognize RMSNorm
→ lower to Fold + Map
recognize softmax
→ lower to generic normalized-exponential Fold
recognize causal predicate
→ lower to DomainRestriction independently
recognize top-k
→ construct Relation
recognize RoPE
→ lower to pairwise/indexed Map
```

Not acceptable:

```text
recognize RMSNorm-before-GEMM
→ directly emit CODA
recognize QK + causal softmax + PV
→ directly emit FlashAttention
recognize routed MoE
→ directly emit MoK
recognize GDN
→ directly emit published GDN chunk kernel
```

#### 3.4 Machine-checkable name erasure

Name erasure is an executable compiler invariant, not only a documentation
claim.

Define one explicit boundary between semantic recovery and schedule synthesis.
At that boundary, validate that every operation used by the selected schedule
is expressed only through the generic Shuttle vocabulary and scalar/index
expressions. The schedule input may contain:

* Map;
* Contract;
* Fold;
* Scan and ChunkAlgebra;
* Relation and RelationPlan;
* SegmentedContract;
* DomainRestriction;
* Transport;
* Materialize.

It may retain source provenance for diagnostics, but workload names must not
affect legality, candidate enumeration, physical-template selection, or code
generation.

Each accepted plan must emit a name-erasure report containing:

* the named semantic operations recovered by the frontend;
* the generic operations to which each one lowered;
* the exact post-erasure representation consumed by scheduling;
* a validation result proving that no forbidden workload dispatch key remains.

At minimum, the validator should reject schedule-visible dispatch keys or
physical operation kinds containing `RMSNorm`, `FlashAttention`, `MoE`, `MoK`,
`GDN`, `KDA`, `Seer`, or equivalent workload-specific aliases. This string
check is only a backstop; structural validation of the operation types and
backend interfaces is authoritative.

---

### 4. No opaque workload-specific semantic kernel

Generated execution may call low-level implementation primitives.

It may not call an opaque kernel that performs a substantial named-workload
semantic subprogram.

Allowed examples:

* WGMMA / MMA;
* TMA / generic memory copy;
* CuTe layout/copy machinery;
* generic dense GEMM;
* generic grouped/ragged GEMM;
* generic reduction primitive;
* generic remote transport;
* DeepEP dispatch used only as transport/permutation;
* barrier / semaphore primitive.

Not acceptable:

* official FA3 forward;
* MoK forward;
* Seer attention;
* FSA sparse attention;
* FLA GDN;
* handwritten sparse-attention Triton body;
* DeepEP combine that performs semantic accumulation;
* Transformer-specific CODA epilogue body.

The distinction is semantic rather than based on provenance.

A GMM adapted from MoK is allowed if its interface is generic segmented
contraction and it contains no MoE control logic.

---

### 5. Physical bodies must be generated from generic semantics

If changing the semantic body requires editing handwritten workload-specific
CUDA/Triton, the result is not clean.

The intended property is:

```text
semantic body A
    ↓
generic generator
    ↓
physical program A
change semantic body to B
    ↓
same generator
    ↓
physical program B
```

The generator may use a constrained set of high-quality physical templates.

It does not need to be an arbitrary GPU compiler.

---

### 6. Mutation test

Every clean proof point must include at least one semantic mutation
demonstrating that physical code is generated from generic structure rather
than workload identity.

Examples:

#### Dense

Change:

* RMS placement;
* output Map;
* RoPE attachment;
* auxiliary Fold emission;

without changing the GEMM backend by hand.

#### Attention

Change:

* mask/domain predicate;
* scale Map;
* Fold finalization;

without editing attention-specific source.

#### MoE

Change where reasonable:

* activation Map;
* merge expression;
* number of route slots;
* segmented output transform;

through generic generation.

#### StatefulScan

Continue using the same generator for:

* scalar decay;
* per-key decay;
* rank-one update;
* rank-two update.

A workload-name switch that selects different handwritten code does not pass
this test.

---

### 7. Performance acceptance

Each accepted workload must achieve the completion target:

```text
generated Shuttle latency ≤ 1.20 × the natural expert comparison
```

The stretch target is:

```text
generated Shuttle latency ≤ 1.10 × the natural expert comparison
```

The stretch target measures how much headroom remains after the clean path
passes. It does not replace or weaken any semantic, correctness, or synthesis
gate.

The 1.20-times completion target applies to at least one representative primary
configuration for MoE, routed sparse attention, and StatefulScan. Dense must
pass both primary sequence lengths, 2,048 and 4,096, because both boundaries
are already established and exercise materially different attention regimes.

The target matrix is:

| Workload | Completion target | Stretch target | Required coverage |
|---|---:|---:|---|
| Dense Transformer | `≤1.20×` | `≤1.10×` | Both matched `S=2,048` and `S=4,096` boundaries |
| Distributed BF16 MoE | `≤1.20×` | `≤1.10×` | One representative matched natural-program boundary; report post-routing separately |
| Routed sparse attention | `≤1.20×` | `≤1.10×` | One representative matched natural-program boundary; report kernel-only timing separately |
| StatefulScan | `≤1.20×` | `≤1.10×` | One representative matched recurrent or chunkwise semantic boundary |

The completion ratio is:

```text
median(all valid Shuttle samples)
────────────────────────────────────
 median(all matched oracle samples)
```

Capture at least two independent runs with at least 30 steady-state samples
per implementation in each run. Alternate or counterbalance Shuttle and oracle
launch order. Pool every valid predeclared capture when computing the ratio; a
single favorable run cannot establish acceptance. Preserve invalidated runs
with the reason for invalidation. Publish the raw samples, per-run medians,
pooled median, ratio, warmup count, launch order, and clock/power policy.

The comparison must use equivalent:

* semantics;
* precision;
* shape;
* hardware;
* preprocessing and postprocessing boundary;
* materialized inputs and outputs;
* included communication and indexing work.

Every performance claim must publish a benchmark-boundary manifest listing
each included and excluded operation for both Shuttle and the oracle.

The primary acceptance comparison begins from the natural high-level program
and uses matched end-to-end boundaries. A secondary component or post-routing
comparison is useful evidence but cannot substitute for that result.

If a common frontend operation is intentionally excluded from timing, it must
still execute at runtime through the accepted natural path, and it must be
excluded symmetrically from both implementations. A saved route, relation, or
layout fixture may not stand in for runtime frontend execution in the accepted
path.

Absolute thresholds inherited from an unmatched historical boundary are
provisional. Recompute the final 1.20-times threshold from the matched oracle
measurement before declaring acceptance.

Once a matched boundary is established, freeze the oracle samples, boundary
manifest, and derived completion and stretch thresholds. Do not move the target
because a later run is convenient. Rebaseline only when semantics, precision,
shape, hardware, or the measured boundary changes, and preserve the superseded
record.

A useful result outside 20% remains worth recording but does not complete the
milestone.

Do not weaken semantics to hit the target.

---

### 8. Numerical correctness

Every accepted workload must:

* compare against a straightforward semantic reference;
* report maximum and mean error;
* obey its declared numerical policy;
* be deterministic when source-order semantics require it.

Current useful policies include:

* `source_ordered`;
* `real_algebra_equivalent`.

Add finer distinctions only when required.

---

### 9. Prototype normal form

Do not build production IR infrastructure solely because this specification
describes one.

Simple Python structures are enough.

The current useful semantic vocabulary is approximately:

* Map;
* Contract;
* Fold;
* Scan;
* Relation;
* RelationPlan;
* SegmentedContract;
* DomainRestriction;
* Transport;
* Materialize.

These are conceptual primitives.

---

### 10. Map

Local transformations such as:

* add;
* scale;
* RoPE;
* SiLU;
* SwiGLU;
* cast;
* quantize;
* dequantize;
* mask-value transform.

A Map can be assigned to:

* Contract preparation;
* Contract finalization;
* Fold update/finalization;
* Scan update/read;
* Transport;
* standalone materialization.

The attachment site is a compiler decision.

---

### 11. DomainRestriction

DomainRestriction limits which logical index tuples participate in a
computation.

Examples:

```text
causal:
    k <= q
window:
    |q-k| <= W
prefix:
    k <= prefix_end(q)
block sparse:
    block_allowed(q_block, k_block)
document boundary:
    doc(q) == doc(k)
```

It is deliberately independent of:

* softmax;
* attention;
* Contract;
* Fold.

A restriction may later lower to:

* mask values;
* skip physical tiles;
* modify Relation;
* change Fold domain;

depending on profitability.

This separation is important.

---

### 12. Contract

Generic multilinear contraction:

```text
Contract {
    key/output axes
    contraction axes
    operands
    preparation AST
    mainloop
    finalization AST
    auxiliary emissions
}
```

Dense GEMM:

```text
acc[M,N] += A[M,K] @ B[K,N]
```

Other examples:

* QKᵀ;
* PV;
* tensor-parallel partial contractions;
* factored higher-order interactions.

The compiler should decide where surrounding work executes during the tile
lifetime:

* input preparation;
* mainloop-associated transform;
* output finalization;
* auxiliary state emission;
* materialized boundary.

---

### 13. Fold

Generic reducible/stateful computation:

```text
Fold {
    key domain
    fold domain
    state
    initialize
    update
    merge
    finalize
    numerical contract
}
```

Examples:

* RMS statistics;
* max;
* sum;
* log-sum-exp;
* online-softmax state;
* gradient accumulation;
* MoE weighted merge.

The compiler should be able to derive partial-state decomposition from generic
reduction algebra where possible.

---

### 14. Normalized exponential as generic Fold structure

Softmax should not be an opaque physical primitive.

Its decomposed semantics include:

* max reduction;
* `exp(x - max)`;
* sum reduction;
* division.

A streaming lowering may use state:

```text
OnlineNormalizedExpState {
    max
    sum_exp
}
```

Attention extends that state with:

```text
weighted_value_accumulator
```

Partial states over disjoint domains should be mergeable exactly, modulo the
declared floating-point ordering policy.

This machinery must be reusable by:

* dense attention;
* routed sparse attention;
* other normalized weighted reductions.

---

### 15. Scan

Ordered state evolution:

```text
Scan {
    key domain
    ordered axis
    state
    initialize
    update
    read
    optional ChunkAlgebra
}
```

The transition should remain expressed through generic Maps/Contracts where
possible.

A chunk algebra may describe:

* `summarize(chunk)`;
* `compose(summary_a, summary_b)`;
* `apply(summary, incoming_state)`;
* `emit_outputs(incoming_state, chunk)`.

This supports recurrent and chunkwise physical forms without naming GDN or
Mamba kernels.

---

### 16. Relation

A sparse relation among semantic roles.

Examples:

* `(token route, expert)`;
* `(query block, selected KV block)`;
* `(query, memory page)`;
* `(token, selected computation depth)`.

Relation describes semantics only.

---

### 17. RelationPlan

Executable/index-plane representation of a Relation.

A plan may contain:

* left IDs;
* right IDs;
* edge attributes;
* ordering by left;
* offsets by left;
* ordering by right;
* offsets by right;
* inverse/original-position mapping;
* placement;
* validity;
* capacity/padding.

The relation/index metadata plane remains separate from tensor payload
movement.

Both MoE and routed sparse attention must share this infrastructure.

---

### 18. SegmentedContract

A contraction over runtime-sized groups:

```text
group e:
    D_e = A_e @ W_e
```

MoE expert GMM is one instance.

The physical scheduler may choose:

* persistent global queue;
* group-major raster;
* padding;
* bucketing;
* coalescing.

These are not semantic MoE concepts.

---

### 19. Transport

Concrete implementation of a placement transition:

```text
Transport {
    source placement
    destination placement
    chunking
    mechanism
    readiness
}
```

Examples:

* DeepEP dispatch;
* remote copy;
* peer TMA;
* collective;
* copy engine;
* source push.

Transport must remain semantically separate from merge/reduction.

---

### 20. Generic GEMM/Contract skeleton

Build a high-performance generic contraction skeleton around an expert-quality
mainloop:

```text
ContractSkeleton {
    A tile source
    B tile source
    generated preparation AST
    generic matrix mainloop
    generated finalization AST
    generated auxiliary emissions
}
```

Examples of generated preparation:

```text
load A
→ FP32 row scale
→ convert to BF16 operand
```

Examples of finalization:

```text
accumulator
→ residual
→ gamma
→ partial sum-of-squares
→ cast/store
```

or:

```text
gate accumulator
up accumulator
→ SiLU(gate) * up
→ store
```

Using CuTe/CUTLASS/QuACK-derived generic mainloops is allowed.

Transformer-specific preparation/finalization code must be generated.

---

### 21. Generic streaming attention skeleton

Do not call FA3 in the accepted dense path.

The compiler should lower:

```text
Contract(Q, Kᵀ)
→ Map(scale)
→ DomainRestriction
→ normalized-exponential Fold
→ Contract(P, V)
```

into a generic streaming physical skeleton.

A likely generated schedule will contain:

* resident Q tile;
* streamed K/V tiles;
* multistage buffering;
* QK tensor-core work;
* online Fold-state update;
* PV tensor-core work;
* specialized producers/consumers.

The physical template may be FA3-inspired.

The crucial property is that the semantic body is exposed and generated.

Changing the Fold/Map/domain semantics must not require rewriting a named
attention kernel.

An extracted implementation may retain generic architecture machinery such
as:

* TMA descriptors and circular pipelines;
* WGMMA tile mainloops;
* shared-memory layouts and swizzles;
* barriers and producer/consumer worker structure;
* generic tiled reductions and scalar maps.

It may not hide normalized-exponential semantics, attention-state updates, or
mask/domain behavior behind an opaque helper merely because that helper is an
internal class rather than a public attention entry point. In particular, an
imported `Softmax`, attention-state, or mask helper is acceptable only if its
interface and implementation are audited as a generic Fold/Map or
DomainRestriction primitive and the selected state/update/finalize program
remains visible to Shuttle.

The accepted attention artifact must include a helper-level lineage report
showing which code implements generic physical machinery and which code is
generated from the recovered Fold/Map/DomainRestriction semantics.

---

### 22. Generic grouped GEMM skeleton

MoE may use a generic grouped/ragged contraction implementation:

```text
SegmentedContract
    ↓
generic GMM
```

Its interface should be approximately:

* runtime group offsets;
* A/B tensors;
* Contract preparation/finalization AST;
* physical tile configuration.

Concatenated `[E,2I,K]` W13 should emerge as a generic layout/Contract
candidate.

---

### 23. Generic Fold/merge generation

The compiler should generate semantic merges.

#### MoE

```text
state[token] += router_weight * expert_result
```

under source-order constraints.

#### Sparse attention

Merge:

```text
(max, sum_exp, weighted_value_sum)
```

from partial selected-KV states.

Do not hide these operations inside opaque transport calls.

---

### 24. Proof A: Dense Transformer

#### Frontend

Natural JAX Transformer → StableHLO.

#### Semantic recovery

Recover and decompose:

* RMSNorm;
* QKV projections;
* RoPE;
* causal predicate;
* softmax;
* PV;
* output projection;
* residual;
* RMSNorm;
* gate/up;
* SwiGLU;
* down projection.

By schedule time, the important structure should look generically like:

* Fold + Maps;
* Contracts;
* DomainRestriction;
* normalized-exp Fold;

rather than named RMSNorm or `causal_attention` operations.

#### Must generate

* all GEMM preparation/finalization bodies;
* RMS partial Fold handling;
* RoPE Map;
* SwiGLU Map;
* online attention body;
* attention physical pipeline.

No opaque FA3 or Transformer-specific CODA execution.

#### RMS alternatives

Keep:

* source-ordered Contract preparation;
* delayed real-algebra-equivalent Contract finalization;

as compiler-selectable alternatives.

#### Oracle

The frozen matched counterbalanced boundary uses the hand-composed named
QuACK/CODA path plus the FlashAttention-4 CuTe expert attention kernel. This is
oracle-only; the generated path calls neither implementation as a semantic
kernel. Two independent runs contain 30 samples per implementation, with
generated-first and oracle-first process order. Pooled oracle medians are:

```text
S=2048: 1.523838 ms
S=4096: 3.253411 ms
```

Completion thresholds:

```text
S=2048: ≤ 1.828606 ms
S=4096: ≤ 3.904094 ms
```

Stretch thresholds:

```text
S=2048: ≤ 1.676222 ms
S=4096: ≤ 3.578752 ms
```

Current pooled generated results are:

| Sequence | Policy | Median | Ratio | Completion | Stretch |
|---:|---|---:|---:|---|---|
| 2,048 | source-ordered prologue | 1.705818 ms | 1.119422x | pass | miss |
| 2,048 | delayed epilogue | 1.650502 ms | 1.083122x | pass | pass |
| 4,096 | source-ordered prologue | 3.478322 ms | 1.069131x | pass | pass |
| 4,096 | delayed epilogue | 3.390837 ms | 1.042240x | pass | pass |

The earlier official-FA3 manual medians, 1.4561/3.0080 ms, remain conservative
historical checkpoints. All current generated candidates also remain below
their corresponding 1.7472/3.6096-ms completion thresholds. The matched
FlashAttention-4 boundary is the acceptance denominator because it preserves
raw oracle samples and counterbalanced same-toolchain execution.

Evidence is frozen under
`benchmarks/artifacts/dense_clean_synthesis_h100_counterbalanced_v1`.

#### Acceptance

Dense passes only if:

* natural frontend works;
* named semantics erase before scheduling;
* physical Contract bodies are generated;
* attention body/pipeline are generated;
* no opaque FA3/CODA semantic kernel remains;
* correctness passes;
* both primary sequence lengths are within 1.20 times their matched oracle;
* the 1.10-times stretch result is reported for both placements and shapes.

---

### 25. Proof B: Distributed BF16 MoE

#### Frontend

Ordinary JAX routed MoE → StableHLO.

Accepted path may not begin from a saved routing fixture.

#### Recover

```text
top-k
→ Relation
→ RelationPlan
expert grouping
→ SegmentedContract W13
→ SwiGLU Map
→ SegmentedContract W2
inverse relation transport
source-ordered weighted Fold merge
```

#### Allowed

* generic GMM;
* DeepEP dispatch as transport;
* generic communication primitives.

#### Must remove

Opaque DeepEP semantic combine.

Instead:

```text
reverse Transport
→ generated Fold merge
→ generated Map for shared-output combination
```

#### Oracle

The frozen matched natural-program boundary executes the same router logits,
top-k, and normalized route-weight frontend before both Shuttle and MoK. Two
independent captures contain 30 rank-maximum samples per implementation and
counterbalance Shuttle-first and oracle-first launch order. Pooled medians are:

```text
Shuttle:     4.137120 ms
matched MoK: 3.645056 ms
ratio:       1.134995×
```

The frozen natural-boundary targets are:

```text
completion: ≤ 4.374067 ms
stretch:    ≤ 4.009562 ms
```

This passes completion and misses stretch. The complete evidence is under
`benchmarks/artifacts/gb200_moe_natural_boundary_v0`.

The earlier tuned MoK supplied-route replay remains a component checkpoint:

```text
oracle:     3.561696 ms
completion: ≤ 4.274035 ms
stretch:    ≤ 3.917866 ms
```

It is not the acceptance denominator because it begins from supplied routing
data.

For final acceptance, benchmark ordinary router logits through top-k,
RelationPlan construction, dispatch, expert computation, reverse transport,
merge, and shared-output combination. Compare against the MoK oracle with the
same router/top-k/index work included. If the oracle API begins after routing,
prepend the identical measured frontend to the oracle before calculating the
ratio.

Continue publishing both:

* the matched natural-program end-to-end ratio, which determines acceptance;
* the post-routing region ratio, which isolates schedule and kernel quality.

#### Acceptance

* natural frontend;
* RelationPlan generated;
* segmented body generated generically;
* transport and merge separated;
* merge generated from Fold semantics;
* no MoK forward;
* no opaque semantic combine;
* correct/deterministic;
* matched natural-program end-to-end latency ≤1.2× MoK plus the same frontend;
* post-routing latency reported separately against the provisional 4.274035-ms
  completion and 3.917866-ms stretch checkpoints.

---

### 26. Proof C: Routed sparse attention

#### Frontend

Natural MoBA-like sparse-attention program.

#### Semantic structure

Must reduce to approximately:

```text
Relation(query block, selected KV block)
Contract QK
Map scale
DomainRestriction / edge validity
normalized-exp Fold
Contract PV
partial-state Fold merge
```

#### Must reuse

* Relation;
* RelationPlan;
* orientation;
* bounded scheduling;
* readiness;
* generic Fold state;

from existing generic Shuttle infrastructure.

#### Must remove

Handwritten sparse-attention Triton semantic body.

#### Physical choices

Support at least:

* query-major;
* KV-major.

The generated scheduler may choose either.

If needed, add generic right-resource reuse:

```text
stage a right-side resource once
→ process multiple left-side consumers
```

This must be a general RelationPlan scheduling feature rather than KV-cache
special case.

#### Oracle

Oracle refresh, 2026-08-08: the first strong routed acceptance workload is now
MiniMax Sparse Attention (MSA), pinned at
`80434d7f67877c6570ca19cac444b84bc9855dac` with CUTLASS
`eb61c911471867a5fd2466bfd8f29306cea6ebf8`, on B200/GB200. Its natural
program contains index projections, causal token scoring, block-max reduction,
top-k per GQA group, and exact selected attention. Its public SM100 CuTe
implementation exposes runtime indices and a deterministic KV-outer schedule
with real KV staging.

The matched MSA acceptance boundary must include the same natural Index Branch,
top-k policy, relation construction/orientation, and selected attention on both
paths. Payload-only timing from synthetic indices is diagnostic only. The MSA
kernel is oracle-only: Shuttle must generate the index, `RelationPlan`,
QK/normalized-exponential-Fold/PV body, and combine through generic primitives.
The initial target is BF16, Hq/Hkv=64/4, D=128, block 128, top-k 16, causal, at
16K debug and 64K primary sequence length. Acceptance is at most 1.20 times a
locally measured counterbalanced MSA full-route oracle.

The prior FlashMoBA experiment below remains a completed block-shared semantic
and generalization result. Its payload uses an SM80-style physical body for a
more general token/head relation, so it is not the strong performance
denominator for the refreshed row.

The existing pinned MIT Block-Sparse-Attention result is a structural and
correctness checkpoint, not the acceptance denominator. Its physical body is
SM80-oriented even when compiled for SM90, so it cannot establish the
1.20-times Hopper performance gate.

The source audit fixes the oracle policy more precisely:

1. FlashMoBA at `39d9ac043b271d046a2181a9991e99a26b67bca1` is the
   primary payload oracle. Its precomputed-relation interface exactly supports
   BF16, D=128, causal masking, block 128, top-8, and native 32:8 GQA.
2. FlashMoBA's complete wrapper is not an exact frontend match. It routes each
   query token/head against mean-pooled K blocks and forces the current causal
   block. Shuttle routes explicit metadata once per query block, shares the
   relation across heads, and does not force the current block. Do not time the
   native FlashMoBA router as the acceptance denominator.
3. The matched whole-program oracle is the common natural Shuttle router plus
   generic relation reorientation into FlashMoBA's KV-column-major sorted
   query-row lists plus `flash_moba_attn_varlen_func`. Record relation
   reorientation separately. Also report cached-relation FlashMoBA payload
   timing to isolate physical attention quality.
4. MIT Block-Sparse-Attention current HEAD/tag v0.0.2 is
   `49d6c39e4dc0303442cda3bb758b3925d4399c49`, the exact revision already
   benchmarked locally at 1.423632 ms. The December 2025 update added SM90 and
   SM100 build compatibility, not a WGMMA/TMA implementation; the active
   kernel remains SM80 MMA plus `cp.async`. Preserve it as a secondary exact
   semantic/local-H100 control rather than rerunning it as a new Hopper oracle.
5. FlashMLA sparse prefill is excluded from this row. Its supported sparse
   program uses MLA/MQA with Hkv=1, Dqk=512/576, Dv=512, and shared latent K/V,
   not ordinary 32:8 GQA D=128. It may be a separate DSA/MLA experiment.
6. Full FSA is excluded from this row because it implements NSA compressed,
   selected, and sliding branches plus a learned merge. Its selected-attention
   subkernel remains a secondary structural control; whole FSA requires a
   separate natural NSA-semantics row.

The matched boundary must include on both paths:

* the FP32 metadata Contract used by the router;
* the causal block-domain restriction;
* sorted top-k selection and index-plane construction;
* BF16 causal exact attention over the selected blocks with native GQA; and
* BF16 output materialization.

Both paths exclude QKV and output projections. The bounded physical query-group
sweep selected 1024 from `{128, 256, 512, 768, 1024}`. Two independent
counterbalanced captures contain 30 steady-state samples per implementation.
Their pooled medians are:

```text
Shuttle generated full boundary: 0.617200 ms
matched FlashMoBA full boundary:  5.264560 ms
FlashMoBA cached payload:          4.894560 ms
common router only:                0.044080 ms
relation reorientation only:       0.211664 ms
Shuttle / full oracle:             0.117237×
```

Generated and FlashMoBA outputs differ by at most 0.00390625 with mean absolute
difference 0.0000651724 and both repeat bitwise. The fixture contains 95 query
blocks whose selected relation omits the current block. The exact samples,
boundary manifest, source pins, correctness records, and deterministic hashes
are frozen under
`benchmarks/artifacts/sparse_flashmoba_h100_matched_v0`.

This exactly matched comparison closes the old block-shared semantic boundary,
but not the refreshed strong-oracle performance gate. It is a loose physical
denominator: FlashMoBA preserves per-token/per-head
row-list generality and its active kernel remains SM80 MMA plus `cp.async`,
whereas the generated path is specialized to a block-shared relation and uses
a Hopper-native skeleton. Therefore the 0.117237-times result is not a claim
that Shuttle is 8.5 times faster than the best expert implementation of these
semantics. The current MIT 1.423632-ms result remains the tighter secondary
local H100 control. A hand-optimized block-shared WGMMA/TMA implementation, or
a natural workload matching FlashMoBA's native token/head router, is required
before treating the oracle as tight. Further tile tuning against the loose
FlashMoBA denominator is not a priority.

Before timing, validate at least one sparse relation whose selected set omits
the current KV block. This guards against an undocumented MoBA-only assumption
inside the precomputed-relation kernel. Physical query grouping may be tuned
over the bounded set `{128, 256, 512, 768, 1024}` without changing Shuttle's
logical query-block definition of 128.

The historical 2.388208-ms Seer timing remains an unmatched diagnostic only.
It scans causal blocks, mask-tests the selected relation, lacks native GQA, and
excludes K/V expansion from its timed region. Do not use its derived 2.866-ms
checkpoint for acceptance.

The bounded KV-major path executes physically with shared-memory
right-resource staging and deterministic online-state merge, so Proof C's
structural gate is closed. Its first CUDA-core implementation is approximately
188 times slower than query-major and is retained as a negative physical
result; no sequence-squared or per-edge partial state is materialized. The
refreshed performance gate remains open until the natural MSA program passes
the numerical and performance requirements below.

MSA checkpoint, 2026-08-08: the natural 16K program is now synthesized on
SM100 without calling MSA's public score, attention, or combine operations.
The generated path lowers index projections, score reduction, selection,
relation planning, causal restriction, normalized-exponential state, QK/PV,
and deterministic merge from generic Shuttle structure. It retains only
low-level expert-derived CuTe layout, copy, MMA, and pipeline templates.

The isolated matched medians are:

```text
score Contract + block-max Fold + top-k:
    Shuttle 0.637888 ms / MSA 0.707600 ms = 0.9015x
natural index projections + selection:
    Shuttle 0.785760 ms / MSA 0.837360 ms = 0.9384x
natural projections + selection + selected payload:
    Shuttle 4.431920 ms / MSA 3.234160 ms = 1.37035x
```

This closes the natural frontend and generic physical-generation parts of the
MSA row, but does not pass acceptance. The full boundary exceeds 1.20 times,
the 64K primary shape remains unmeasured, and the natural output's maximum
difference from the materialized semantic reference is 0.0536499 versus the
current 0.01 gate. Generated and oracle selectors produce the identical route
hash; the discrepancy from the materialized reference occurs on early causal
rows with underfilled finite domains or an exactly tied top-k cutoff. Exact-
relation generated payload agrees with official MSA to maximum 0.0009765625.

The current decision is to preserve the clean generic implementation rather
than add an MSA-specific combine to manufacture a passing number. The next
performance work, if resumed, should improve the generic deterministic Fold
merge or partial-state representation. Evidence is frozen under
`benchmarks/artifacts/msa_clean_sm100_v0`.

#### Acceptance

* natural frontend;
* generic RelationPlan;
* generated QK/Fold/PV body;
* generated online-state merge;
* no fixed workload-specific attention body;
* numerical correctness/determinism;
* matched natural-program end-to-end latency ≤1.2× the equivalent expert
  oracle;
* matched 1.1× stretch result reported without making it an acceptance gate;
* query-major and KV-major both execute physically through the generic
  RelationPlan/Fold machinery;
* completion is checked against a locally measured matched MSA full-route
  boundary on SM100; FlashMoBA and MIT remain secondary controls;
* unmatched Seer timing is reported only as a historical diagnostic.

---

### 27. Proof D: StatefulScan / Gated DeltaNet

#### Frontend

Ordinary JAX recurrence including natural `stablehlo.while`.

#### Existing clean result

Generic recurrent core:

```text
~0.1385 ms
```

with reuse for:

* scalar decay;
* per-key decay;
* rank-two updates.

#### Complete the proof

Recover generic structure.

From tensor expressions infer:

* structured diagonal decay;
* bounded-rank erase/write;
* state read.

Do not recognize a model name and select its kernel.

#### Generate preparation

Any Q/K/V/gating/preparation operations must come from generic Map/Contract
generation.

#### Recurrent physical skeleton

Connect the clean recurrent generated kernel end-to-end.

#### Chunkwise skeleton

Generate from Scan + ChunkAlgebra.

Do not invoke FLA/GDN whole kernels.

#### Mutation

Existing scalar/per-key/rank mutations must reuse the same generator.

#### Oracle

The accepted pinned FLA delta-rule boundary pools two independent,
counterbalanced captures at the primary configuration:

```text
0.424304 ms
```

The derived targets are:

```text
completion: ≤ 0.509165 ms
stretch:    ≤ 0.466734 ms
```

Shuttle and FLA receive identical BF16 Q/K/V and FP32 log-decay, beta, and
initial-state inputs. Query/key normalization is disabled and the query scale
is one. Each capture contains ten warmup pairs and 50 measured pairs; launch
order alternates within every pair, and the initial order reverses between
captures. The pooled oracle samples and boundary manifest are frozen with the
benchmark artifact. The earlier single-capture 0.420528-ms oracle remains in
the artifact as superseded evidence rather than an acceptance denominator.

#### Acceptance

```text
generated end-to-end latency
≤ 1.20 × oracle
```

for at least one important training/prefill regime.

Report the 1.10-times stretch target separately. The current pooled generated
result, 0.465824 ms or 1.097854 times the matched oracle, passes both completion
and stretch.

---

### 28. Low-level primitive rule

When uncertain whether reuse is allowed, apply:

A primitive is acceptable when its interface naturally applies to
substantially different semantic programs without knowing which model
requested it.

Good:

```text
gemm(A,B)
grouped_gemm(segments,A,B)
mma(fragmentA,fragmentB)
remote_copy(...)
barrier(...)
```

Bad:

```text
sparse_attention_forward(...)
mok_forward(...)
gdn_chunk_forward(...)
deep_ep_moe_combine(...)
```

---

### 29. Generated-code audit

For each accepted workload list every external implementation dependency as:

* hardware/runtime primitive;
* generic compute primitive;
* generic communication primitive;
* generated Shuttle kernel;
* expert/oracle-only.

The accepted generated execution path must contain zero expert/oracle-only
semantic kernels.

The audit must also include:

* the machine-checked name-erasure report;
* the benchmark-boundary manifest for Shuttle and its oracle;
* a runtime call graph or equivalent launch manifest identifying every kernel
  and communication operation executed on the accepted path;
* an assertion that oracle-only modules can be removed or made unavailable
  without changing generated execution.

---

### 30. Source-lineage audit

When adapting generic primitives from expert kernels, record:

* origin;
* what low-level machinery was retained;
* what architecture-specific control was removed;
* resulting generic interface.

Example:

```text
generic grouped GEMM
derived from MoK implementation
retained:
    WGMMA mainloop
    tile/layout machinery
removed:
    routing
    event graph
    MoE scheduling
    combine semantics
```

This prevents “adapted primitive” from quietly becoming “copied solution.”

For FlashAttention-derived code, perform the audit at helper granularity. It is
not sufficient to state that the public `flash_attn` entry point is absent.
Classify layout, pipeline, barrier, copy, contraction, online-state, score-map,
mask, and finalization helpers separately. Any helper that still implements a
substantial attention semantic program must either be generalized behind a
workload-independent Fold/Map/DomainRestriction interface or remain
oracle-only.

---

### 31. Candidate search

Finite explicit candidate sets are encouraged.

Search may cover:

* tile shapes;
* pipeline stages;
* worker allocations;
* buffer depths;
* relation orientation;
* kernel boundaries;
* layouts;
* materialization;
* Map attachment sites;
* transport.

Hard-code the search space when useful.

Do not hard-code the answer.

---

### 32. Generality accounting

For each proof point classify code changes as:

* reused generic Shuttle machinery;
* generalized generic Shuttle machinery;
* new generic Shuttle machinery;
* workload-specific frontend recognition;
* oracle/reference-only code.

A workload-specific physical kernel body should not be needed for acceptance.

---

### 33. Suggested parallel work

Possible sub-agent workstreams:

#### A. Generic Contract generator

* preparation AST;
* finalization AST;
* auxiliary Fold emission;
* dense GEMM path;
* GMM compatibility.

#### B. Generic normalized-exp / attention generator

* generic Fold state;
* QK/PV Contracts;
* DomainRestriction handling;
* online-state merge;
* producer-consumer physical skeleton.

#### C. Clean distributed MoE

* natural JAX path;
* generic semantic task bodies;
* Transport/merge split;
* generated deterministic merge.

#### D. StatefulScan

* `stablehlo.while`;
* transition analysis;
* preparation AST;
* recurrent skeleton;
* chunkwise skeleton.

#### E. Validation

* oracle reproduction;
* semantic references;
* numerical tests;
* generated-code audit;
* performance harness.

---

### 34. Milestone acceptance matrix

All four rows must pass:

| Workload | Natural frontend | Machine-checked name erasure | Generic physical generation | No opaque semantic kernel | Correct | Matched benchmark boundary | ≤1.20× completion | ≤1.10× reported |
|---|---|---|---|---|---|---|---|---|
| Dense Transformer | Required | Required | Required | Required | Required | Required | Required at both shapes | Required, not a gate |
| Distributed BF16 MoE | Required | Required | Required | Required | Required | Required | Required at one primary shape | Required, not a gate |
| Routed sparse attention | Required | Required | Required | Required | Required | Required | Required at one primary shape | Required, not a gate |
| StatefulScan / GDN | Required | Required | Required | Required | Required | Required | Required at one primary shape | Required, not a gate |

Good planning alone does not pass.

Good performance alone does not pass.

Both are required.

---

### 35. Explicit non-goals

Do not make these prerequisites:

* FSDP;
* full training;
* backward MoE;
* MXFP8;
* production MLIR dialects;
* major XLA fork;
* TPU support;
* publication packaging;
* fully analytical cost model;
* arbitrary GPU code generation.

They can be explored only if directly useful to completing the four proof
points.

---

### 36. Acceptance question to ask continuously

For every optimization:

Is this implementation being derived because the compiler understands generic
algebra/state/index structure, or because a recognizer knows the name of the
workload?

And for every physical body:

If I change the mathematical body while keeping the same generic Shuttle
primitives, does the compiler regenerate code, or must someone edit a
workload-specific GPU kernel?

And:

If every named expert kernel for this architecture disappeared, leaving only
generic matrix, reduction, memory, and communication primitives, could Shuttle
still generate this implementation family?

If the answers are favorable, the result is clean.

---

### 37. Final milestone claim

The prototype milestone is complete only when this statement is accurate:

Starting from ordinary high-level programs, Shuttle cleanly synthesizes
competitive physical implementations for dense Transformers, distributed
routed MoE, routed sparse attention, and structured recurrent computation.
Named frontend semantics such as RMSNorm or softmax are canonicalized into
generic algebra before schedule synthesis; causality and other masks are
represented independently as domain restrictions. Across all four workloads,
Shuttle derives operation placement, reduction/state structure, sparse
relations, layouts, materializations, and physical scheduling while using only
generic low-level compute and communication primitives. Each generated
implementation satisfies the matched-boundary protocol in Section 7 and runs
at no more than 1.20 times a strong natural expert comparison. The 1.10-times
stretch result is reported for every accepted workload but is not a completion
gate.

Missing any synthesis, correctness, benchmark-boundary, or 1.20-times
completion gate is partial progress. Missing only the 1.10-times stretch target
does not block the milestone.
