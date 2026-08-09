# Streaming-attention backward schedule boundary

The GB200 component benchmark at sequence length 2,048 measured 2.200320 ms
for Shuttle and 0.155450 ms for Torch SDPA, a 14.155-times gap. Correctness
passed. The old dK/dV grid contained only 512 programs: one program per K/V
tile and K/V head. Each program then revisited 64 query tiles serially for each
of four query heads. The causal path also evaluated the fully invalid upper
triangle in both the dQ and dK/dV traversals. It therefore issued 256 small
reverse-Contract steps per dK/dV program before writing its sole deterministic
dK/dV result.

The bounded replacement derives a packed row domain from the QK Contract's
query-head-to-K/V-head index map. All query heads mapped to one K/V head are
placed in one physical Contract row tile. For the 32-head/eight-K/V-head
benchmark this changes each dK/dV fold step from four 32-row Contracts to one
128-row Contract. Projecting the canonical lower-triangular domain restriction
onto the tiled traversal also skips Q/K tile pairs that are entirely invalid.
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
