# Streaming-attention backward schedule boundary

The first GB200 component benchmark at sequence length 2,048 measured
2.200320 ms for Shuttle and 0.155450 ms for Torch SDPA, a 14.155-times gap.
Correctness passed. The old dK/dV grid contained only 512 programs: one program per K/V
tile and K/V head. Each program then revisited 64 query tiles serially for each
of four query heads. The causal path also evaluated the fully invalid upper
triangle in both the dQ and dK/dV traversals. It therefore issued 256 small
reverse-Contract steps per dK/dV program before writing its sole deterministic
dK/dV result.

Grouping mapped query heads and projecting the causal domain reduced the
generated result to 0.864582 ms, compared with a contemporaneous 0.148534 ms
Torch SDPA result. The remaining 5.821-times gap motivates the next bounded
physical change below.

The bounded replacement derives a packed row domain from the QK Contract's
query-head-to-K/V-head index map. All query heads mapped to one K/V head are
placed in one physical Contract row tile. For the 32-head/eight-K/V-head
benchmark this changes each dK/dV fold step from four 32-row Contracts to one
128-row Contract. The same axis transformation now applies to the query-major
dQ traversal: one physical program owns a query-token tile and a K/V head, and
packs the four mapped query heads as Contract rows. K/V tiles are loaded once
for the four mapped heads instead of once per head. Projecting the canonical
lower-triangular domain restriction onto the tiled traversal also skips Q/K
tile pairs that are entirely invalid.
The diagonal tile remains explicitly predicated. The schedule uses no atomics,
uses a deterministic query-row-major/mapped-head-minor tree Fold, and retains
the same score-Map derivative, including the tanh-softcap mutation.
This tree changes finite-precision association relative to the scalar-head
prototype, so the order is stored in the schedule instead of being implicit.

`estimate_streaming_attention_backward_work` reports both logical tile pairs
and physical Contract invocations. This distinguishes useful arithmetic removed
by domain projection from instruction-level coalescing of the same logical GQA
rows. It also reports the packed score-tile footprint so a backend can reject a
coalescing choice that exceeds its register/shared-memory budget.

At the primary GQA ratio of four, packed dQ reduces physical dQ Contract
invocations from 199,680 to 49,920 and K/V tile loads in that traversal from
66,560 to 16,640. Including the already-packed dK/dV traversal, the static
physical Contract count falls from 266,240 to 116,480, a 56.25% reduction.
Logical FLOPs do not change: each remaining Contract has four times as many
rows. Peak packed score state remains 128 by 32, which the existing dK/dV path
already supports. This is a work-granularity and operand-reuse hypothesis, not
a claim of measured speedup.

The preceding query-partition experiment is a negative control. Four query
partitions changed 0.864582 ms to 0.854435 ms at sequence length 2,048 while
adding a 64 MiB FP32 partial buffer and 512 finalizers. It was reverted because
the 1.17% latency reduction did not justify the extra work or storage. Packed
dQ instead reduces physical instruction and load duplication without partial
gradients, atomics, or an additional Fold finalizer.

The next hardware check should compare the packed and scalar-head dQ schedules
with the same 32-by-32 tile configuration. The hypothesis is falsified if the
larger live row tile causes enough register pressure or spilling that generated
latency fails to improve by at least 10%, or if the numerical/determinism checks
change. Even a successful result will not close the remaining gap to the
0.148534 ms SDPA measurement: the Triton prototype still recomputes QK,
probability, dP, and dS separately in the dQ and dK/dV traversals and lacks the
TMA/WGMMA producer-consumer overlap of an expert Hopper pipeline.

## Accepted frontend boundary

JAX owns model differentiation. The accepted training path is:

```text
natural JAX program
    -> JAX VJP HLO
    -> recover generic reverse Contracts, Maps, Folds, and DomainRestrictions
    -> grouped-query/domain-projected physical schedule
```

`derive_streaming_attention_backward` is only a reference symbolic VJP used to
validate recovery and benchmark the physical schedule in isolation. Programs
carry explicit `REFERENCE_SYMBOLIC_VJP` or `JAX_VJP_HLO_RECOVERY` provenance.
The tile scheduler never dispatches on that provenance and does not derive a
model VJP. A full acceptance result still requires recovering the equivalent
generic reverse program from JAX VJP HLO and executing that recovered program.

Torch is used only for the numerical and timing oracle in the standalone
benchmark. It is not a runtime dependency of the compiler-owned schedule or a
physical implementation primitive.

## Physical source lineage

The schedule study used the official FlashAttention repository at revision
`a369df707e1980fb328abcc1733e3457ec10155f` as a physical reference. The useful
generic ideas are a K/V-major persistent work tile, resident FP32 dK/dV
accumulators, staged Q/dO movement, and producer-consumer pipelining around
matrix contractions. Shuttle does not import or invoke FlashAttention's
attention mask, softmax, backward kernel, or model-level dispatch. Its score
Map derivative, DomainRestriction predicate, normalized-exp state, and Fold
order remain compiler-owned and visible.
