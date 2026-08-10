# Shuttle Synthesis-Boundary Audit

Date: 2026-08-07

The authoritative milestone definition is
[Shuttle Clean-Synthesis Prototype](clean_synthesis_acceptance.md). This audit
tracks current evidence against that stricter acceptance boundary.

## Acceptance vocabulary

Use these terms narrowly:

- **Oracle measurement**: an expert implementation is executed unchanged for
  correctness or performance comparison.
- **Oracle-backed execution**: a Shuttle plan invokes a complete expert kernel
  for one semantic region.
- **Generated schedule**: Shuttle derives routing, dependencies, buffers,
  workers, and kernel boundaries, but one or more physical task bodies remain
  borrowed complete kernels.
- **Synthesized kernel**: Shuttle derives a physical program from generic
  semantic factors and reusable skeletons. It may reuse low-level copy,
  contraction, reduction, and transport primitives, but it does not call a
  complete workload kernel.

Only the final category establishes first-principles kernel synthesis.

## Dense Transformer

Current status: **clean accepted dense proof at both required shapes**.

Ordinary JAX lowers to frozen StableHLO, temporary named semantics erase into
36 generic `Map`, `Contract`, `Fold`, and `DomainRestriction` operations, and
the generic planner constructs the eight-skeleton region. Machine-checked
erasure validation runs before candidate enumeration. The compiler derives RMS
partial emission and consumer placement, RoPE and SwiGLU maps, layouts, and
materialization boundaries from that generic structure.

All GEMM preparation/finalization bodies are emitted from a scalar/tile AST
around a generic QuACK/CuTe mainloop. Pairwise RoPE and SwiGLU arithmetic is
present directly in generated source rather than selected through named
Transformer callbacks. Attention lowers from QK/PV `Contract`s, a domain
restriction, and normalized-exponential `Fold` state into the generated SM90
streaming skeleton. The accepted execution path calls neither official FA3 nor
named CODA/QuACK semantic epilogues.

The first H100 capture gave:

| Sequence | Source-ordered prologue | Delayed epilogue | Matched oracle |
|---:|---:|---:|---:|
| 2,048 | 1.6872 ms (`1.159x`) | 1.6339 ms (`1.122x`) | 1.4561 ms |
| 4,096 | 3.4148 ms (`1.135x`) | 3.3848 ms (`1.125x`) | 3.0080 ms |

All candidates pass the 1.20-times completion ratio at both required shapes.
Generated Contract components are bitwise equal to matching primitive oracles
except for the direct scalar-AST SiLU expression, whose BF16-rounded maximum
error is at most 0.125. The mutation from `SiLU(left) * right` to
`left * right` changes generated arithmetic through the same AST generator.

The final counterbalanced checkpoint uses two independent 30-sample captures
per implementation and reverses generated/oracle process order between runs.
Against a matched hand-composed QuACK/CODA plus FlashAttention-4 CuTe oracle,
the pooled generated ratios are 1.0831/1.1194 times for delayed/prologue at
S=2,048 and 1.0422/1.0691 times at S=4,096. All candidates pass completion;
three of four also pass the 1.10-times stretch target. The generated candidates
also remain below the more conservative historical official-FA3 completion
thresholds.

Final raw evidence is under
`benchmarks/artifacts/dense_clean_synthesis_h100_counterbalanced_v1`; the
earlier `dense_clean_synthesis_h100_20260807` artifact remains the component
and semantic-mutation checkpoint.

## Distributed MoE

Current status: **clean accepted natural-source distributed schedule**.

Shuttle owns:

- `RelationPlan` and destination grouping;
- expert segmentation and padding;
- buffer dependencies and worker-count candidates;
- packing, SwiGLU, deterministic fixed-slot merge, and overlap schedule.

The runtime does not call MoK's complete forward or reproduce its event graph.
The accepted path starts from ordinary JAX StableHLO and executes BF16 router
logits, top-k, and FP32 normalized route weights at runtime. DeepEP performs
only forward payload permutation. A generated GPU RelationPlan constructs
receiver-local counts, padded rows, edge destinations, and ordered weights.
Plain payload-only `all_to_all_single` returns owner partials, followed by a
generated ascending-owner FP32 Fold and shared Map.

Routed and shared expert contractions call the standalone MoK
`grouped_gemm_out` primitive through a generic segmented-contraction contract.
Under the backend rule this is allowed: grouped GEMM is a reusable physical
contraction primitive, not a complete MoE implementation. A generated SwiGLU
Map supplies the activation body.

Two independent captures counterbalance Shuttle-first and oracle-first launch
order, with 30 rank-maximum samples per implementation in each capture. Pooled
medians are 4.137120 ms for Shuttle and 3.645056 ms for matched MoK, or
1.134995 times. The MoK path executes the identical router/top-k frontend
before schedule construction and complete MoK forward. The result meets the
1.20-times completion target and misses the 1.10-times stretch target.

Every device-generated relation is exact with zero overflow. Repeated outputs
are bitwise equal; maximum error against MoK is `0.0001220703125`. Generated
relation, route Fold, and rank Fold code contains no semantic atomic operation.
DeepEP may use readiness counters internally for transport, but it does not
accumulate semantic values or determine their order. DeepEP semantic combine
is legacy-control-only and MoK forward is oracle-only.

The former public `compile_mok_expert_parallel_region` path has been renamed
`compile_mok_oracle_region`, and its plan node is now
`OpaqueMoKOracleSkeleton`. The complete evidence is under
`benchmarks/artifacts/gb200_moe_natural_boundary_v0`.

## Routed sparse attention

Current status: **clean MSA synthesis structure demonstrated; numerical and
1.20-times performance gates remain open**.

The earlier 4.02-ms slot-wave result was Shuttle-owned but used specialized
Triton source. It remains a negative physical experiment rather than clean
synthesis evidence.

The replacement frontend begins with ordinary JAX math containing a metadata
Contract, causal block predicate, top-k selection, selected K/V gathers, QK,
normalized exponential, and PV. StableHLO recovery erases this into generic
`RelationSelectionProgram`, `RelationPlan`, `Contract`, `Map`,
`DomainRestriction`, and `Fold` structure. A machine-checked erasure report is
validated before query-major and KV-major candidates are enumerated. Changing
the runtime metadata changes relation edges while preserving the same generic
streaming program and candidate family.

The query-major SM90 body is generated from that program and the same extracted
QK/online-Fold/PV skeleton used by dense attention. It performs real
three-stage TMA-to-shared-memory K/V staging and does not call Seer, FSA, or a
named sparse-attention entry point.

At S=16,384, block 128, top-8, Hq/Hkv=32/8, and D=128, the natural matched
boundary includes router metadata contraction, causal restriction, GPU top-k
and index forwarding, selected exact attention, and BF16 output on both paths.
The historical target remains based on the first two independent captures. Two
additional 30-pair captures alternate generated→oracle and oracle→generated
for every sample pair. Their pooled medians are 0.617584 ms for generated
Shuttle and 1.423632 ms for pinned Block-Sparse-Attention, or 0.433809 times.
Generated versus oracle maximum/mean differences are
0.00390625/0.0000652, and both outputs repeat bitwise.

The bounded KV-major candidate also executes on H100. It serializes selected
slots, groups each slot by right-side KV block, stages one KV-head block into
dynamic shared memory, reuses it for at most two query blocks, and writes the
online state directly back to its owning query. At the primary non-monotone
relation it covers 996 edges with 671 tasks, uses 65,536 bytes of shared K/V
per CTA and 272,629,760 bytes of global online state, and materializes no
per-edge partial state. It is deterministic and differs from query-major by at
most 0.015625.

The first physical body is intentionally simple and measures 107.879105 ms
versus 0.574656 ms query-major. Its QK/PV work uses CUDA cores and it has no
TMA, WGMMA, cluster multicast, or cross-head K/V sharing. Capacity one changes
the relation task plan from 671 to 996 tasks through the same generated source,
produces bitwise-identical output, and measures 103.355042 ms. This closes the
relation-orientation structural gate while clearly locating the remaining
physical-kernel gap.

The oracle is an SM80-style implementation compiled for SM90, so this result is
a historical matched control rather than a Hopper acceptance comparison.
Complete evidence is under
`benchmarks/artifacts/natural_routed_sparse_attention_h100_matched_v0`.

The stronger MSA checkpoint replaces the old SM80-oriented oracle question.
At 16K on GB200, generic Shuttle score/Fold/Selection is 0.9015 times the
isolated MSA oracle and the full natural boundary is 4.431920/3.234160 ms, or
1.37035 times. Generated and oracle route hashes match exactly. Their common
route differs from the materialized reference only on early causal underfilled
rows or a tied cutoff, but the resulting 0.0536499 maximum output difference
exceeds the current 0.01 numerical gate. Exact-relation payload correctness
passes. Proof C therefore passes synthesis and deterministic exact-relation
execution, but remains provisional on both the natural numerical contract and
the 1.20-times performance target. Evidence is under
`benchmarks/artifacts/msa_clean_sm100_v0`.

## StatefulScan

Current status: **clean accepted StatefulScan proof at the matched core
boundary**.

Ordinary JAX `lax.scan` exports as `stablehlo.while`. The importer reconstructs
logical axes and tensor expressions, then erases the source into generic
`Scan`, `Map`, and `Contract` structure before candidate enumeration. A shared
machine-checked report derives scheduling keys only from ordered extent, state
rank, primitive arity, generic affine transition structure, and numerical
policy. Tests reject workload-named and stale keys.

The generated path is:

```text
AffineIntraChunkPrepare
→ AffineStateScan
→ AffineReadout
```

It uses a generic chunk-64 affine summary and a four-by-four block triangular
inverse over 16-wide subblocks. FLA is imported only by the benchmark harness
as an oracle. Scalar/per-key decay crossed with rank-one/rank-two updates uses
the same recovery, report, candidate generator, and physical stages.

On the matched H100 boundary, both paths receive identical BF16 Q/K/V and FP32
log-decay, beta, and initial state. Two independent counterbalanced captures
pool to 0.465824 ms for Shuttle and 0.424304 ms for pinned FLA, or 1.097854
times. Output/final-state maximum errors are `4.883e-4`/`3.154e-4`; all
generated mutations repeat bitwise. The result passes both the 1.20-times
completion target and the 1.10-times stretch target.

The accepted evidence and reproducible erasure report are under
`benchmarks/artifacts/stateful_scan_affine_pipeline_h100_v0`. This closes the
current core StatefulScan row. Projection, short-convolution, and output-gating
work remain future whole-layer scope rather than part of this frozen matched
boundary.

A later provenance audit separated this matched natural-frontend result from
the earlier 0.138544 ms recurrent-core artifact, whose recovery began at a
hand-authored tensor-expression fixture. Public compilation and mutation
harnesses now enter through JAX-exported `stablehlo.while`; the correction and
remaining GPU replay requirement are recorded in
[stateful_scan_frontend_provenance_20260809.md](stateful_scan_frontend_provenance_20260809.md).

## Backend boundary

Allowed reusable components include:

- TMA/copy primitives;
- WGMMA or generic contraction mainloops;
- reductions and triangular-solve primitives;
- barriers, events, and bounded pipeline machinery;
- DeepEP dispatch or return movement when it does not also implement the
  program's semantic merge.

Disallowed as synthesis evidence:

- official FA3 as the generated attention implementation;
- FLA/FlashQLA as the generated scan implementation;
- MoK complete forward as the generated segmented program;
- DeepEP `combine` as the generated deterministic merge;
- named QuACK Transformer epilogue functions selected from semantic operation
  names rather than generic tile-program primitives.

Expert source remains valuable for abstracting the reusable skeleton and for
defining the performance oracle.

## Milestone acceptance matrix

The table records the strongest current evidence. Sparse attention now has a
matched SM100 oracle, but the clean path does not yet pass its numerical or
performance gate:

| Workload | Natural frontend and name erasure | Generated semantic body | Mutation evidence | Matched ratio | Result |
|---|---|---|---|---:|---|
| Dense Transformer | JAX/StableHLO → 36 generic Flow operations | Contract ASTs plus generated SM90 streaming Fold | pairwise SiLU-product → product through the same AST generator | worst required-shape ratio `1.119422x` | pass |
| Distributed BF16 MoE | JAX/StableHLO router/top-k → RelationPlan | generic segmented Contracts, generated SwiGLU, source-ordered Fold merge | route-slot counts 2 and 6 use the same recovery and generation path | `1.134995x` | pass |
| Routed sparse attention | JAX/StableHLO → Contract/Fold/Selection/Relation/DomainRestriction | generated score/Fold/Selection and routed QK/normalized-exp/PV/merge | relation, block size, score map, and merge schedule retain the generator | `1.37035x` MSA at 16K | provisional: misses performance and natural numerical gates |
| StatefulScan | JAX `stablehlo.while` → Scan/Map/Contract | generated preparation, ordered state scan, and readout | scalar/per-key decay crossed with rank-one/rank-two updates | `1.097854x` | pass |

The accepted numerical contracts remain explicit: dense records source-ordered
and real-algebra-equivalent RMS placements separately; MoE uses fixed
route-slot and rank order without semantic atomics; sparse attention uses
deterministic selected-slot online-state updates; StatefulScan declares
`bounded_reassociation`.

The generated paths contain no expert/oracle semantic kernel. Their external
execution dependencies are generic contraction, grouped contraction, tensor
layout/copy, reduction, triangular-solve, and payload-transport primitives.
Official attention, MoK, Block-Sparse-Attention, and FLA implementations appear
only in oracle paths.

Final local validation on 2026-08-07 passed 179 tile-lifetime tests, Pyrefly
with zero errors, Ruff, and portable checksum verification for the preserved
benchmark artifacts. The prior sparse result remains a historical control.
