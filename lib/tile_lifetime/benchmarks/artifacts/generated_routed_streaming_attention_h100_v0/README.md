# Generated routed streaming attention on H100

Date: 2026-08-07

This checkpoint executes natural `Contract` / score `Map` / online-state
`Fold` semantics over a generic `RelationPlan`. The physical emitter consumes
`RoutedStreamingAttentionCompilation`; it does not call a named dense or
block-sparse attention entry point. It instantiates the same extracted SM90
QK/online-softmax/PV skeleton used by the dense compiler path and supplies
compact relation-derived block lists.

This is a backend/planning proof, not yet the canonical clean end-to-end sparse
acceptance result. The benchmark constructs `RelationPlan` and the generic
streaming program directly. It does not yet start from an ordinary MoBA-like
JAX program and erase router/top-k/attention syntax into
`Relation`/`Contract`/`Map`/`DomainRestriction`/`Fold`. The measured path
therefore passes physical-generation, mutation, correctness, determinism, and
performance checks, but leaves the natural-frontend check open.

The natural frontend is now implemented in
`h100_natural_routed_streaming_attention.py`: ordinary JAX emits the router
Contract, block-domain predicate, top-k, selected K/V gathers, QK, normalized
exponential, and PV. StableHLO recovery erases those names into a generic
runtime `RelationSelectionProgram`, `RelationPlan`, and the same generated
streaming body. The harness forwards GPU top-k results into the relation index
plane and times routing/index generation with attention. Existing measurements
in this directory predate that harness and therefore remain backend-only. A new
matched expert measurement is still required; the precise symmetric boundary
is recorded in `NATURAL_FRONTEND_BOUNDARY.json`. Passing
`--include-block-sparse-oracle` measures the pinned expert kernel in the same
process and wraps it with the identical metadata Contract, block restriction,
top-k, and runtime relation-mask generation.

The kernel moves K/V with TMA into a three-stage circular shared-memory
pipeline. The current SM90 implementation does not multicast K/V across a CTA
cluster, so this checkpoint establishes real shared-memory staging but not the
KV-major cluster-reuse candidate.

## Configuration and results

All runs use one H100 80GB HBM3, BF16, causal GQA `Hq=32`, `Hkv=8`, `D=128`,
sequence 16K, 128-token Q/KV blocks, and at most eight selected KV blocks per
query block. Driver 595.71.05, 700 W power limit, observed 1830 MHz SM and 2619
MHz memory clocks, Torch 2.11.0+cu128, CUTLASS DSL 4.5.2, and
`flash-attn-4==4.0.0b16` supplied the low-level physical helper primitives.

| Relation / score program | Median | Range | Sampled FP32 max abs | Deterministic |
| --- | ---: | ---: | ---: | --- |
| Non-monotone relation, scale `2^-3.5` | 0.491984 ms | 0.490816–0.500672 ms | 0.008165 | yes |
| Historical Seer relation, scale `2^-3.5` | 0.492512 ms | 0.490976–0.502976 ms | 0.008165 | yes |
| Mutated relation, scale `0.125`, tanh softcap 16 | 0.618176 ms | 0.614336–0.659360 ms | 0.007996 | yes |

The primary non-monotone relation has 996 edges and 126 of 128 source rows
traverse destination blocks out of increasing order. Changing both the
relation and score program changes the generated executable while retaining
correctness and determinism.

## Delta from the Seer baseline

The preserved SeerAttention baseline for the same historical 16K/top-8
relation measured 2.388208 ms. The generated SM90 path measures 0.492512 ms,
or 4.85x faster. This is not evidence of a universally better sparse-attention
algorithm: Seer's baseline expands GQA K/V to 32 heads outside the timed region
and scans the dense causal block domain while testing a dense mask. Shuttle
keeps native 8-head GQA and traverses only the 996 selected relation edges.
The comparison isolates exactly the metadata traversal and physical staging
limitations already identified in the Seer artifact.

## Generality accounting

- Reused unchanged: `RelationPlan`, source/destination orientation planning,
  natural QK/score-map/online-state/PV semantics, online merge algebra, H100
  schedule lowering, and the SM90 TMA/shared-memory contraction skeleton.
- Generalized existing machinery: source-major relation traversal now lowers
  to compact block counts and indices without an attention-specific relation
  type.
- New generic machinery: `QueryMajorBlockIndexPlan`, a small index-plane
  lowering that preserves route-slot order.
- Sparse-attention-specific backend code: conversion of compact block lists to
  the physical CuTe tuple expected by the skeleton.
- Sparse-attention-specific semantic code: only the synthetic routed workload;
  ordinary StableHLO router recovery remains future work.

Both query-major and KV-major candidates are produced from the same routed
compilation, and both execute correctly in the backend-neutral reference. Only
query-major executes on H100 in this checkpoint. The KV-major physical plan
would materialize about 2.12 GB of edge partial state at this shape; a bounded
KV-resident cluster implementation remains the meaningful next sparse
iteration.

Raw JSON files contain every timing sample, output hash, relation hash, plan
dump, correctness statistics, and hardware telemetry. `MANIFEST.sha256` pins
the raw records and executed source.
