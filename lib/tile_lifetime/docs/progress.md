# Progress

## Current baseline

The compiler imports frozen StableHLO v1.14.1 artifacts and emits an inspectable execution plan. One combined program recovers:

- a CODA-style `GEMM -> residual/RMS partials -> small reduction -> GEMM` region with either delayed consumer-epilogue or source-ordered consumer-prologue scaling;
- exact causal grouped-query attention as an official-FA3 Hopper skeleton with FP32 online maximum, normalizer, and output state;
- score and probability tensors as internal attention state rather than global-memory materializations.

Both physical component paths now execute on H100. Official FA3 runs for causal BF16 GQA with head dimension 128. CODA's delayed scale and an experimental QuACK FP32 A-fragment transform run for the primary projection shapes.

The selected FA3 plan reflects the implementation that was measured: two pipeline stages, packed GQA, a 32-thread producer warp, two consumer warpgroups for head dimension 128, RS PV, intra-warpgroup overlap, and a persistent scheduler.

The semantic planner now composes one connected dense region into eight skeletons: QKV/RoPE, FA3, output projection with residual/RMS partials, RMS reduction, gate/up with RMS scaling and pairwise SwiGLU, down projection with residual/RMS partials, RMS reduction, and the next QKV/RoPE boundary. There are no standalone memory-bound transform skeletons or sequence-squared materializations. This path works both from a Python semantic graph and from a frozen StableHLO v1.14.1 artifact exported by the parameterized JAX reference.

Both RMS placements remain selectable in the connected plan. The default is the FP32 consumer prologue because it preserves the materialized BF16 consumer boundary exactly; CODA-style delayed scaling remains available when its small performance advantage is worth the rounding reorder.

## Reproduce CPU checks

```bash
uv run --frozen --package marin-tile-lifetime --group test pytest lib/tile_lifetime/tests
uv run --frozen --package marin-core --group lint pyrefly check lib/tile_lifetime/src
```

## Current H100 result

At `M=2048`, `K=4096`, and `N=6144`, the tuned FP32 consumer prologue uses a 128x256x64 cooperative QuACK tile. Given the same stored BF16 input, it runs in 0.141 ms, is bitwise equal to materialized pre-scaling, and avoids the normalized-activation materialization. CODA's delayed consumer epilogue measured 0.150-0.164 ms across the recorded runs and has mean absolute consumer-boundary source-order error 0.0838.

At the wider `N=28672` gate/up shape, cluster-N=2 reduces the FP32 prologue to 0.650 ms, versus 0.686 ms for materialized pre-scaling and 0.703 ms for the recorded CODA epilogue. The planner selects this cluster only for wide outputs.

With identical dead-preactivation SwiGLU epilogues, the full gate/up boundary measures 0.6509 ms for FP32 prologue scaling, 0.6430 ms for delayed scaling, and 0.8723 ms for materialized source-order scaling. At the packed QKV/RoPE boundary (`N=6144`), cluster-N=2 gives corresponding results of 0.1467, 0.1357, and 0.2169 ms. The prologue is bitwise equal to the materialized consumer-boundary result in both cases.

Official FA3 runs in 0.0672 ms at sequence length 2048 and 0.2100 ms at sequence length 4096 for `B=1`, `Hq=32`, `Hkv=8`, and `D=128`. It is about 2.4x faster than Torch SDPA's flash backend in these measurements.

The packed QKV/RoPE producer now runs directly into official FA3 in the shared Torch 2.13 runtime. At sequence 2048, the combined boundary measures 0.1944 ms. FA3 accepts Q/K/V segment views with contiguous head dimension and row stride 6144; explicit contiguous repacking is bitwise identical but increases the boundary to 0.2367 ms.

The complete eight-skeleton oracle now executes in that shared runtime. At sequence 2048, consumer-prologue and delayed-epilogue plans measure 1.4800 and 1.4561 ms, compared with 1.9614 ms for a materialized Torch source and 2.5010 ms for stock JAX/XLA. At sequence 4096 the results are 3.0563, 3.0080, 4.0236, and 6.5257 ms, respectively. Thus the delayed plan is 41.8% faster than JAX at sequence 2048 and 53.9% faster at sequence 4096.

Phase profiling shows that the current repeated FP32 scale-strip copies cost 0.0074 ms and 0.0069 ms at the two consumer-prologue boundaries. Gate/up is otherwise tied with delayed scaling within clock variation; next QKV retains about 0.013 ms of A-fragment-transform overhead. A residual/RMS tile sweep selected 128x256x64 cluster-1 for both output and down projections, at 0.1037 ms and 0.3085 ms including their reducers.

The selected plan now executes directly through a validated runtime/backend boundary. Primary-shape StableHLO compilation feeds eight plan skeletons into QuACK/CODA and official FA3 dispatch, with one packed QKV allocation at each projection boundary and physical row-by-tile RMS partial buffers. At sequence 2048, the longest interleaved run measured minimum plan-driven latencies of 1.4199 ms for consumer-prologue scaling and 1.3973 ms for delayed scaling; medians remain noisy because clocks cannot be pinned. The plan-driven output differences exactly match the hand-composed comparison.

## Next checkpoint

The pinned official Mixture-of-Kittens commit now builds and passes its four-rank correctness benchmark on a low-priority 4xGB200 tray. At the published default shape, BF16 forward/backward measure 3.669/9.029 ms and MXFP8 measures 2.521/7.662 ms. No upstream source modifications were required.

A bounded BF16-forward schedule sweep improved the measured forward configuration from the initial 24 communication SMs/minibatch 4096 to 20 communication SMs/minibatch 2048. The isolated sweep measured 3.584 ms/528.4 TFLOP/s; the full official correctness benchmark measured 3.613 ms/524.2 TFLOP/s with unchanged BF16 error. The planner must enforce minibatch divisibility by 256 and exact macrobatch divisibility by minibatch.

The official MoK task/event structure is captured as an oracle description, not a selected backend. The active compiler path now starts from an ordinary global routed-MoE graph and must derive ownership, segmentation, exchange, segmented contractions, tile flow, buffers, and worker schedules using generic transformations. MXFP8 remains deliberately excluded until scale tensors are modeled.

Ordinary JAX MoE StableHLO now recovers into a global semantic graph containing router projection, normalized top-k, shared gated MLP, global routed gated MLP, and weighted merge. The generic EP lowering derives 17 atomic stages, including alternative assignment-row and token-owner-coalesced exchange relations, receiver-local expert segmentation, gate/up layout legalization, explicit tile-flow storage/readiness, and buffer lifetimes. It does not reference the MoK backend.

The runtime index plane now builds a reusable `RelationPlan` from router output. It records source item and route slot, global and owner-local destinations, stable padded destination rows, counts and offsets, coalesced exchange rows, weights, validity, and both reverse maps. The same plan drives assignment dispatch, coalesced dispatch and expansion, inverse dispatch, and an FP32 weighted merge in fixed source-item then ascending-route-slot order. Capacity overflow is rejected before payload movement. The suite has 64 passing tests.

The first exact-route four-GB200 implementation is executable. The native JAX 0.11 ragged-all-to-all runtime paths segfault on first execution on this toolchain, but forcing the XLA ragged collective decomposer is correct. With decomposed transport and Triton segmented GEMMs, the initial 1.25-capacity plan measured 9.460 ms. Tightening the guarded capacity to the exact 12349-row receiver bound improved it to 8.445 ms with zero drops. XLA `ragged_dot` measured 95.100 ms and is not competitive.

The compiler's coalesced DeepEP candidate is now the fastest generated plan. A single global compact assignment batch replaces DeepEP's 49152-row local domain with a guarded 12473-row domain, preserves original token/route-slot positions, reports exact overflow, and merges through deterministic sort/search/gather plus ascending-slot FP32 accumulation. It measures 6.113 ms with zero drops, 46.2% faster than uncapped DeepEP and 1.69 times the tuned MoK oracle.

The standalone Blackwell grouped-GEMM probe now invokes MoK's routed BF16 primitive below the complete event graph. At 96 experts, 256 padded rows per expert, hidden size 7168, and intermediate size 3072, W2 measures 0.943 ms/1148 physical TFLOP/s and two-launch W13 measures 2.036 ms/1063 physical TFLOP/s. A small Torch differential check passes with maximum absolute error 0.0149; the kernel has 255 registers and no spills. The probe does not call full MoK or reuse its communication schedule. Combining these component measurements with raw DeepEP transport predicts approximately 4.92 ms before overlap.

The compiler-produced relation now feeds generated packing, standalone MoK W13, generated SwiGLU, standalone MoK W2, and deterministic pre-combine merge at a simulated DeepEP receive boundary. The exact rank-0 projection contains 6755 receive tokens, 12281 assignments, and 24576 padded rows; an independent route-table scan verifies every mapping. The generated sequence measures 3.455 ms versus 6.747 ms for the matching Torch sequence. Explicit FP32 multiply/add merge is bitwise equal to Torch at 0.361 ms; FMA is 0.003 ms faster with maximum absolute difference `2.38e-7`.

The verified receiver-local path now runs through official DeepEP dispatch and combine on four GB200 ranks. It preserves fixed route-slot FP32 merge, overlaps the generated shared expert with asynchronous dispatch, and passes exact mapping, bitwise schedule-parity, repeat-determinism, and independent semantic-reference checks. A DeepEP worker sweep turned at 96 SMs. The final layout A/B selected concatenated `[E,2I,K]` W13 with 56 communication SMs: two rank-maximum overlap runs measured 3.976 and 3.991 ms, approximately 10.3% above the 3.613-ms MoK oracle. The next experiment should profile this selected plan before changing the communication/computation pipeline.

The dense/MoE checkpoint is now preserved under annotated tag `shuttle-gb200-moe-v1`. A schema-2 replay on one low-priority four-GB200 tray measured 3.9830 ms for the selected plan, 4.0649 ms without overlap, 4.4348 ms with coarse activation materialization, 4.0690 ms with separate gate/up, and 3.5617 ms for the tuned MoK oracle. Raw per-rank distributions, GPU UUID/topology and phase telemetry, deterministic hashes, eight semantic fixtures, source/toolchain pins, candidate fingerprints, and the benchmark cache are checked into `benchmarks/artifacts/gb200_moe_v1`. The replay observed 1950 MHz SM and 3996 MHz memory clocks during every captured benchmark boundary under the cluster-default unpinned policy.
