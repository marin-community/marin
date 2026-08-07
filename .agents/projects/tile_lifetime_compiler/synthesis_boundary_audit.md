# Shuttle Synthesis-Boundary Audit

Date: 2026-08-07

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

Current status: **oracle-backed execution with generated region planning**.

The compiler recovers the region, selects RMS placement, constructs an
eight-skeleton plan, and determines materialization/layout boundaries. The
measured runtime nevertheless calls:

- named QuACK/CODA epilogue implementations such as RMS partials, SwiGLU,
  RoPE, and A-fragment scaling;
- the official FlashAttention-3 forward kernel for attention.

Therefore the current 1.46/3.01 ms dense results validate semantic recovery,
region decomposition, numerical choices, and an achievable composition. They
do not show that Shuttle synthesized CODA or FA3 kernels.

Required clean result:

- lower generic Map/Contract/Fold attachments to a generated epilogue AST;
- instantiate a generic GEMM skeleton without selecting a named Transformer
  epilogue function;
- lower the attention Fold to a generated Q-resident online-state skeleton
  abstracted from FA3, with FA3 retained only as the oracle.

## Distributed MoE

Current status: **generated distributed schedule with borrowed transport,
borrowed return/merge, and borrowed segmented-contraction kernel**.

Shuttle owns:

- `RelationPlan` and destination grouping;
- expert segmentation and padding;
- buffer dependencies and worker-count candidates;
- packing, SwiGLU, deterministic fixed-slot merge, and overlap schedule.

The runtime does not call MoK's complete forward or reproduce its event graph.
DeepEP dispatch is an allowed ragged transport primitive. The measured path
also calls DeepEP `combine`, which fuses reverse movement with fixed-rank
accumulation and shared-output addition. It is deterministic and non-atomic,
but the accumulation is a semantic Fold rather than transport. Routed and
shared expert contractions additionally call the standalone MoK
`grouped_gemm_out` primitive. Consequently the 3.98 ms result does not
establish synthesis of either expert compute or the final distributed merge.

Required clean result:

- retain DeepEP only as payload `Transport`; return distinct contributions or
  expose a transport-only return path before reduction;
- generate the source-ordered deterministic return/merge Fold;
- derive segmented task domains from generic relation offsets;
- instantiate a Shuttle-owned grouped/ragged contraction skeleton abstracted
  from MoK/QuACK/CuTe;
- generate W13/W2 physical programs from the segmented contraction plan;
- keep the complete MoK forward and grouped-GEMM wrapper oracle-only.

## Routed sparse attention

Current status: **hand-authored Shuttle kernel, not an expert-kernel shell-out**.

The 4.02 ms slot-wave result does not call Seer or FSA. Those implementations
are comparison oracles. The Triton slot-wave body consumes generic
`RelationPlan` fields and is Shuttle-owned.

It is nevertheless specialized source for sparse online attention. Shuttle
does not yet derive that body from generic `Relation + Fold + Contract`
semantics. The result validates relation reuse and one generated schedule, but
not general kernel synthesis.

Required clean result:

- derive online state and merge algebra from the Fold;
- instantiate a generic relation-oriented stateful-contraction skeleton;
- generate query-major and KV-major bodies from the same recovered program;
- use a non-monotone relation and real cluster/shared-memory KV staging for the
  next physical test.

## StatefulScan

Current status: **generic semantic recovery plus one synthesized recurrent
kernel; generated chunkwise execution remains incomplete**.

The affine recurrence and chunk-composition analysis is Shuttle-owned. FLA and
FlashQLA are complete expert kernels and may only be correctness/performance
oracles. Calling them after recognizing GDN/KDA does not count as recovery.

Required clean result:

- connect generic state-affine recovery to StableHLO loop bodies rather than
  relying on tensor-expression fixtures;
- extend the current diagonal and diagonal-plus-low-rank classification to
  block-diagonal transitions where useful;
- instantiate the generated ordered-chunk skeleton;
- compare the generated kernels with FLA/FlashQLA.

The first recurrent result satisfies the kernel-synthesis boundary. Generic
tensor-expression analysis recovers diagonal-plus-bounded-rank structure and
instantiates one Triton skeleton. On H100 the same generated source executes
scalar decay/rank 1, per-key decay/rank 1, and scalar decay/rank 2 without FLA
or FlashQLA installed. At `B1,T64,H32,K=V=128`, medians are 0.138544,
0.138000, and 0.183376 ms. All cases are bitwise deterministic; maximum BF16
output and FP32-state errors are `2.441e-4` and `1.863e-8`. This establishes a
synthesized recurrent core, not yet a synthesized complete GDN/KDA layer.

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
- MoK complete forward or `grouped_gemm_out` as the generated segmented
  contraction;
- DeepEP `combine` as the generated deterministic merge;
- named QuACK Transformer epilogue functions selected by recognized pattern.

Expert source remains valuable for abstracting the reusable skeleton and for
defining the performance oracle.
