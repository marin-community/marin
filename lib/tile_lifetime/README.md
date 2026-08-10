# Tile-Lifetime Compiler

This package preserves research prototypes, reference planners, attachment
bridges, benchmarks, and artifacts from Shuttle's early development. The
current compiler package is `lib/shuttle` and targets an in-pipeline MLIR
extension. Python StableHLO recovery and textual HLO rewriting in this package
are experimental evidence only.

One preserved dense prototype imports a frozen portable StableHLO export of a
Llama block through the following QKV/RoPE boundary and recovers:

- packed QKV projections with pairwise RoPE and an FA3-compatible strided BSHD segment layout;
- ordinary causal GQA algebra as an official-FA3 streaming skeleton with online maximum, normalizer, and output state;
- output and down projections with CODA-style residual, gamma, and RMS-partial epilogues;
- gate/up projection with dead-preactivation pairwise SwiGLU; and
- either source-ordered FP32 consumer-prologue RMS scaling or CODA-style delayed consumer-epilogue scaling.

The selected plan contains eight skeletons, no standalone memory-bound transform, and no sequence-squared materialization. A validated plan-driven runtime dispatches the selected QuACK/CODA and official-FA3 skeletons on H100 without QKV repacking. At the primary sequence-2048 shape, the hand-composed delayed and prologue oracles measure 1.456 and 1.480 ms, versus 2.501 ms for the stock JAX/XLA tensor graph; direct plan dispatch overlaps those ranges under unpinned-clock variation. See [results](docs/results.md) for source revisions, numerical comparisons, sequence-4096 results, and limitations.

The follow-on expert-parallel prototype reproduces and tunes the pinned official Mixture-of-Kittens implementation on four GB200 GPUs as an oracle only. Its task graph has been extracted as a comparison artifact, while the compiler path starts from an ordinary global-expert MoE graph and derives route relations, expert ownership, segmentation, tile flow, buffers, and schedules through generic transformations. The target is a generated distributed plan within 20–30% of the MoK BF16-forward oracle. MXFP8 remains out of scope until scale tensors are modeled explicitly.

Run the CPU tests with:

```bash
uv run --frozen --package marin-tile-lifetime --group test pytest lib/tile_lifetime/tests
```

Print the example plan with:

```bash
uv run --frozen --package marin-tile-lifetime python lib/tile_lifetime/examples/rms_region.py
```

The project specification and research brief live under `.agents/projects/tile_lifetime_compiler/`.
