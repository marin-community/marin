# Progress

## Clean dense and routed-attention synthesis

The dense natural-source path now erases temporary RMSNorm, attention, RoPE,
and SwiGLU names into 36 generic `Map`, `Contract`, `Fold`, and
`DomainRestriction` operations before candidate enumeration. Generic
scalar/tile ASTs generate every Contract preparation/finalization body, and a
generated SM90 QK/online-Fold/PV skeleton replaces official FA3 on the measured
path. At sequence lengths 2,048 and 4,096, both source-ordered prologue and
delayed-epilogue policies passed the first clean capture against the historical
manual oracle. The semantic mutation from SiLU-product to pairwise product
changes generated source through the same AST generator. That component and
mutation checkpoint remains under
`benchmarks/artifacts/dense_clean_synthesis_h100_20260807`.

The earlier block-shared routed-attention path includes metadata contraction, causal
restriction, GPU top-k/index forwarding, selected exact attention, and BF16
output on both Shuttle and oracle paths. At S=16,384, block 128, top-8,
Hq/Hkv=32/8, and D=128, two counterbalanced 30-sample captures pool to
0.617200 ms for generated Shuttle and 5.264560 ms for pinned FlashMoBA on the
same full boundary. FlashMoBA's cached-relation payload is 4.894560 ms; the
common router and generic relation reorientation are 0.044080 and 0.211664 ms.
Generated and oracle outputs are deterministic, with maximum/mean differences
0.00390625/0.0000652.

The same generic `RelationPlan`, score Map, `DomainRestriction`, and
normalized-exponential Fold also generate an executable KV-major slot-wave
schedule. One CTA stages 65,536 bytes of K/V in shared memory and reuses it for
a bounded query group. The primary non-monotone relation covers 996 edges with
671 tasks, uses no per-edge partial-state buffer or semantic atomics, and
repeats bitwise. Its CUDA-core body is intentionally structural and measures
107.879105 ms, so query-major remains selected. That experiment closed the
structural relation-orientation test and its exact FlashMoBA comparison. The FlashMoBA
denominator is physically loose: it preserves per-token/per-head row-list
generality and uses SM80-style MMA plus `cp.async`, while Shuttle is specialized
to the shared block relation and is Hopper-native. The already-measured current
MIT Block-Sparse-Attention result of 1.423632 ms remains a tighter secondary
local H100 control, but it is also SM80-style. Evidence for the new primary
comparison is under
`benchmarks/artifacts/sparse_flashmoba_h100_matched_v0`. It does not close the
refreshed MSA performance gate below.

## Clean MSA synthesis

The first natural MiniMax Sparse Attention path now lowers ordinary JAX and
StableHLO into generic index-projection `Contract`s, score `Contract`, block
maximum `Fold`, `Selection`, `RelationPlan`, causal `DomainRestriction`,
normalized-exponential `Fold`, and QK/PV `Contract`s. The generated path calls
no public MSA score, attention, or combine entry point. It retains only
expert-derived low-level CuTe layout, copy, MMA, and pipeline templates.

At `Q=K=16384`, `Hq/Hkv=64/4`, `D=128`, block 128, top-k 16, causal BF16,
the isolated matched medians are:

| Boundary | Shuttle | MSA oracle | Ratio |
|---|---:|---:|---:|
| Score/Fold/Selection | 0.637888 ms | 0.707600 ms | 0.9015x |
| Natural index projections plus selection | 0.785760 ms | 0.837360 ms | 0.9384x |
| Natural projections, selection, and payload | 4.431920 ms | 3.234160 ms | 1.37035x |

The clean structural proof succeeds, but the written 1.20-times performance
gate remains open. The dominant generic cost is deterministic online-state
merge; under an exact relation it alone costs 1.831552 ms. A BF16x2 merge
variant regressed and was removed rather than retained as dead specialized
code. No further MSA-specific micro-optimization is planned for this
checkpoint; future improvements should strengthen the generic Fold/merge
skeleton.

Generated and oracle selectors have the same deterministic route hash. Both
differ from the materialized reference only on early causal underfilled rows
or a zero cutoff-margin row under the declared `real_algebra_equivalent`
policy. Exact-relation payload correctness passes, but the natural-program
maximum output difference of 0.0536499 exceeds the current 0.01 threshold, so
the numerical gate also remains open. Raw distributions, source audits,
negative results, invalidated pre-causal evidence, commands, and checksums are
under `benchmarks/artifacts/msa_clean_sm100_v0`.

The dense path has also completed the revised statistical protocol. Two
independent 30-sample captures reverse generated/oracle process order and use a
matched hand-composed QuACK/CODA plus FlashAttention-4 CuTe oracle. Pooled
delayed/prologue ratios are 1.0831/1.1194 times at S=2,048 and 1.0422/1.0691
times at S=4,096. Both policies pass completion at both required shapes;
evidence is under
`benchmarks/artifacts/dense_clean_synthesis_h100_counterbalanced_v1`.

## Synthesis-boundary cleanup

The generic expert-parallel schedule now distinguishes payload permutation
from semantic reduction. A clean candidate uses DeepEP dispatch only for the
forward payload movement, `all_to_all_single` for payload-only return, and a
compiler-owned deterministic fixed-rank fold for routed merge plus shared add.
The planner rejects transports such as DeepEP `combine` whose contract also
performs a reduction. The historical 3.9835-ms result remains labeled as an
oracle-assisted composition. The clean four-GB200 path now measures 4.082608
ms: DeepEP performs forward payload dispatch, `all_to_all_single` performs only
reverse payload movement, and a compiler-owned rank-ordered FP32 fold performs
the routed merge and shared-output add. This is 1.1463 times the frozen
3.561696-ms MoK replay and therefore meets the 1.2-times target without calling
DeepEP combine or the MoK forward event graph. The matching sequential region
is 4.229424 ms, and payload-only return plus generated merge is 0.365168 ms.
A second 30-sample confirmation measures 4.142576 ms. Raw distributions,
exact executed sources, pins, and correctness fixtures are preserved under
`benchmarks/artifacts/gb200_moe_clean_merge_v0`.

The matched natural-source boundary is now complete. Ordinary JAX StableHLO
executes router logits, top-k, and normalized FP32 route weights at runtime.
The accepted Shuttle path builds the receiver relation on device, uses DeepEP
only for forward payload dispatch, executes generic segmented W13/W2 plus a
generated SwiGLU Map, returns payloads through `all_to_all_single`, and runs a
generated fixed-rank Fold and shared Map. The matching MoK path prepends the
identical router/top-k frontend.

Two counterbalanced 30-sample captures pool to 4.137120 ms for Shuttle and
3.645056 ms for MoK, or 1.134995 times. This passes the 1.20-times completion
target. A prior scalar-Fold pair pooled to 1.201725 times and remains preserved
as a negative result. BF16x2 vectorization of the generic source-ordered route
and rank Folds supplied the final 0.227104-ms Shuttle reduction. Every generated
relation is exact with zero overflow; repeated outputs are bitwise equal; the
maximum/mean errors against MoK are `1.2207e-4` and at most `2.6670e-6`.
Generated relation and Fold code contains no semantic atomic operation.
Evidence and checksums are under
`benchmarks/artifacts/gb200_moe_natural_boundary_v0`.

## Historical oracle-backed baseline

The earlier executable dense path was a generated region plan around named
QuACK/CODA epilogues and official FA3; it is retained as a historical oracle
composition rather than the current clean synthesized path.
The historical distributed MoE result generated the relation and global
schedule but used DeepEP `combine` for both reverse transport and semantic
reduction. The later supplied-route result separated transport and merge but
still excluded router/top-k and index-plan time. The accepted natural-boundary
result above closes that gap. Its standalone MoK-derived grouped GEMM remains
an allowed generic segmented-contraction skeleton.

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

## Routed sparse-attention prototype

The next phase is isolated on `research/shuttle-routed-sparse-attention`. A generic ragged `RelationPlan` now represents `query block -> selected KV block` with stable traversal from either side. It exposes source/slot order, compact destination offsets, sorted source identities, ownership coordinates, padding, validity, inverse mapping, and coalesced placement changes.

The first CPU implementation executes two schedules from that same relation:

- query-major: keep query state resident and extend it over selected KV blocks;
- KV-major: stage one KV block, compute one partial state for every incident query block, inverse-route those states, and merge by stable selected-slot order.

The state is explicit FP32 `(row_max, row_sum_exp, weighted_value_accumulator)`. Merge uses a shared maximum and rescales both inputs. Both orientations match an independent selected-mask reference for causal GQA, uneven source/destination degree, relation padding, and a padded sequence tail. Repeated KV-major output is deterministic and uses no atomic accumulation.

The planner emits both candidates with derived arrival counts, bounded buffers, worker roles, kernel regions, and materialization costs. The query-major candidate retains partial state internally. The first KV-major candidate deliberately materializes one partial state per valid relation edge so its traffic cost is visible. Both report zero sequence-squared materialization.

At the initial sequence-16K, block-128, top-k-8 configuration, the synthetic causal relation has 996 valid edges. The deterministic two-kernel KV-major baseline would materialize approximately 2.12 GB of FP32 partial state. This does not invalidate KV-major orientation, but it makes bounded forwarding or a more compact state schedule a necessary follow-up if the measured reuse benefit cannot pay for that traffic.

A deterministic bounded alternative now processes the relation in ascending selected-slot waves. Within one wave each query block has at most one incident edge, so KV-major groups update query online state with one writer per query and no atomics. The 16K plan materializes one 272,629,760-byte FP32 online-state buffer rather than 2,121,400,320 bytes of per-edge partials, a 7.78x capacity reduction. The executable Triton candidate uses eight explicit waves for top-k 8.

The H100 oracle phase uses pinned Block-Sparse-Attention, Flash Sparse Attention, and FlashMoBA precomputed-pattern attention. FlashMLA is retained as a GB200 control rather than the primary oracle because its native sparse-prefill path uses materially different MLA/MQA dimensions.

The first query-major GPU adapter is reproducible in `benchmarks/h100_routed_sparse_attention.py`. It feeds the exact compiler-generated relation to MIT Block-Sparse-Attention, times only the cached-relation kernel separately from relation planning, saves every repeated sample, hashes both the route fixture and output, and checks selected query blocks against an independent exact implementation. The holder lacked a system CUDA toolkit, so the pinned MIT source build stopped before compilation at its `nvcc` check.

A pinned SeerAttention Triton fallback completed on H100. At sequence 2048 it measures 0.316752 ms and matches the independent source-ordered reference with maximum absolute error 0.0078125. At sequence 16384 it measures 2.388208 ms over 50 samples, versus 6.282496 ms for dense Torch causal GQA SDPA. The result is deliberately classified as a weak query-major oracle: Seer scans every causal KV block and mask-tests it, and its lack of GQA requires a 201-MB K/V expansion outside the timed kernel. Raw distributions and the exact negative build result are preserved in `benchmarks/artifacts/routed_sparse_attention_h100_v0`.

The FSA KV-major adapter now runs the identical 16K/block-128/top-k-8 Boolean relation, whose raw hash is `b2a57606e303f8af4da0c8002ddea162f86625725696bca7f18b8072a8143427`. The public FSA call measures 12.5392 ms over 30 samples, or 21.322 selected-work TFLOP/s. Sampled reference error is 0.0207922 maximum and 0.000164022 mean; repeated output is bitwise identical. The adapter uses generic `RelationPlan` fields, but FSA expands them to token/head indices and rebuilds its private KV-major relation inside the timed call, so this is an expert oracle rather than Shuttle's physical schedule.

FSA materializes approximately 111.2 MB of partial state/statistics and 20.9 MB of inverse-index data; its measured peak allocator increment is 431.1 MB. Shuttle's coarse all-edge plan declares 2.12 GB. Pinned FSA requires a one-line ephemeral compatibility patch that removes an accidental singleton tuple from a pointer passed to `tl.load`; the pristine failure and exact diff are preserved in `benchmarks/artifacts/routed_sparse_attention_fsa_h100_v0`.

The generated slot-wave implementation now executes the same relation directly. Its selected M32/four-warp physical point measures 0.502016 ms at 2K and 4.017344 ms at 16K, with a 16K range of 4.011936–4.027776 ms over 30 samples and 66.552 selected-work TFLOP/s. Sampled 16K error is 0.00783062 maximum and 0.000124260 mean; output is bitwise deterministic with SHA256 `7fee4b9c61ea72736f203fad5ab212f1f31d9178f750bc967f8c8db2eeb66917`. M16/four-warp and M64/eight-warp 2K candidates were slower at 0.569072 and 0.660096 ms.

The 16K source-order/no-sort ablation is inconclusive by construction: the canonical relation is already KV-monotone inside every selected slot, so it executes the same edge arrays and measures 4.018880 ms versus 4.017344 ms. The next KV-major experiment should use a deliberately non-monotone relation and introduce actual shared KV staging or cluster-level reuse. The current grouped CTAs do not share staged K/V, and eight wave boundaries plus global FP32 state traffic explain the remaining 1.68x gap to the 2.388208-ms query-major Seer smoke. Full candidate records are under `benchmarks/artifacts/routed_sparse_attention_h100_v0/slot_waves`.

The distributed extension is deferred. Relation ownership and coalescing transfer, but the current DeepEP/MoE transport cannot carry structured FP32 attention states or KV-block payloads without a new backend adapter. That is significant new infrastructure, not a small reuse of the existing schedule, so the present conclusion remains single-H100. The full tile-lifetime suite passes 74 tests.

## 2026-08-07: Seer delta closed

- Accounted for the 1.629136-ms gap between the 16K Seer query-major baseline
  and Shuttle's generated KV-major slot waves.
- The generated schedule incurs at least 4.92 GB of global FP32 online-state
  lifecycle traffic plus roughly 0.91 GB of extra Q reads.
- At H100 bandwidth, those bytes predict 1.74--2.33 ms and explain the measured
  gap without attributing it to relation metadata.
- Closed tile-size tuning. The only useful follow-up sparse experiment is a
  non-monotone relation with real cluster/shared-memory KV staging.
- Began the Gated DeltaNet `StatefulScan` prototype on branch
  `research/shuttle-stateful-scan`.

## 2026-08-07: StatefulScan semantics and oracle crossover

- Added generic ordered-state semantics, numerical contracts, exact affine
  chunk composition, and recurrent/chunkwise execution skeletons.
- Represented both scalar-decay Gated DeltaNet and per-channel-decay Kimi Delta
  Attention without adding architecture-specific semantic operation types.
- Independent recurrent and exact-affine chunk executors agree across nonzero
  state, continuation, tail chunks, chunk sizes, decay regimes, scalar/per-key
  diagonals, and bounded update ranks. The focused StatefulScan suite passes
  55 tests; the repository-safe suite passes 1178 tests with 4 skips and 5
  expected failures.
- Measured the Qwen3-Next GDN core with pinned FLA oracles on H100. Matched
  recurrent/chunk medians are 0.084960/0.515104 ms at T=64,
  0.321792/0.532176 ms at T=256, and 3.940768/0.510624 ms at T=2048. The
  empirical winner therefore changes with sequence length.
- Both FLA forms are finite and repeat bitwise. Chunkwise final-state maximum
  error is `5.543e-3` versus `5.364e-7` for recurrent, so the plan records
  bounded reassociation explicitly.
- FlashQLA imports and passes its signature test, but its first kernel JIT is
  blocked by the holder's incomplete split CUDA toolkit. The exact failure is
  preserved without changing the pin.
- StableHLO frontend recovery remains incomplete: JAX emits `lax.scan` as a
  structured `stablehlo.while` plus private recurrence and indexing functions.
  A structured importer now preserves the while condition/body, imports only
  transitively called private functions, resolves the source scan body, and
  reconstructs its logical axes and tensor expressions. Both scalar-decay
  rank-one and per-key-decay rank-two JAX exports recover the same generic
  diagonal-plus-low-rank `StatefulScan` candidate family.
- Added generic tensor-expression linearization and diagonal-plus-low-rank
  recovery. Eighteen gate/diagonal/rank mutations recover the same factor
  family without architecture-specific dispatch.
- Generated and ran a generic recurrent Triton skeleton on H100 with neither
  FLA nor FlashQLA installed. At `B1,T64,H32,K=V=128`, scalar-rank-1,
  per-key-rank-1, and scalar-rank-2 medians are 0.138544, 0.138000, and
  0.183376 ms. All are bitwise deterministic; maximum output/state errors are
  `2.441e-4` and `1.863e-8`.
- Derived exact bounded factored chunk summaries of the form
  `D*S + U*(V^T*S + Z)`, including transformed reads and local outputs. The
  generated GPU ordered-chunk result and FLA comparison are recorded below.
- FSDP remains deferred while the measured chunk-summary materialization is
  removed; that is the narrower prerequisite exposed by this experiment.

## 2026-08-07: Generated ordered factored-chunk path

- Derived chunk factors from generic `RecoveredAffineStateUpdate` terms with
  masked triangular solves. The construction handles scalar or per-key
  diagonals and simultaneous bounded-rank updates without a GDN/KDA dispatch.
- Added a compiler-owned Triton skeleton that retains one FP32 state value
  block across source-ordered chunks and applies BF16 physical summaries under
  a `bounded_reassociation` contract. Scalar-rank-one and per-key-rank-two GPU
  fixtures are finite, bitwise deterministic, and agree with the generated
  recurrent skeleton within `4.883e-4` maximum output error.
- On the Qwen3-Next core at T=2048, the selected C16/BV32 plan measures
  0.665568 ms summary preparation, 0.340032 ms execution, and 0.984496 ms
  combined. That is 1.928x the pinned 0.510624 ms FLA chunk oracle and misses
  the 1.2x target.
- The execution skeleton alone is 0.666x the oracle. The remaining gap is the
  generic preparation plus 84,410,368 bytes of prepared factors, not the
  ordered inter-chunk scan. The next useful implementation experiment is fused
  producer/preparation or a smaller forwarded summary, not another tile sweep.
- Raw distributions, mutation hashes, candidate results, source hashes, and
  environment details are preserved under
  `benchmarks/artifacts/stateful_scan_generated_chunk_h100`.

## 2026-08-07: Natural MoE and routed-attention compiler paths connected

- Added one public StableHLO-to-expert-parallel entry point. Ordinary JAX MoE
  StableHLO is now imported, semantically recovered, and lowered to the same
  generic relation, segmented-contraction, payload-transport, and generated
  merge plan used by the distributed prototype.
- Connected a generic `Contract/Map/Fold`-derived streaming-attention program
  to query-major and KV-major candidates over a shared `RelationPlan`.
  Causal and tanh-softcap score mutations use the same planner, and GQA remains
  an explicit logical-axis index map rather than a named attention mode.
- These connections close compiler plumbing gaps, not physical-performance
  gaps. The MoE performance fixture still begins after router/top-k, and the
  measured sparse slot-wave body predates the generic streaming emitter.
- Quarantined the earlier complete-kernel MoK path behind the explicit
  `compile_mok_oracle_region` and `OpaqueMoKOracleSkeleton` names. It remains a
  baseline only and is no longer presented as a normal Shuttle compiler path.
- The complete local suite passes: 161 tests. A package-wide Pyrefly check
  exposed optional CUDA/CUTLASS/QuACK imports in an in-progress CuTe
  extraction; that source was moved out of the typed core package, and the
  complete source-and-test check is back to zero errors.

## 2026-08-07: Natural routed-attention frontend erased to generic algebra

- Added an ordinary JAX routed-attention program containing a metadata
  contraction, causal block-domain predicate, top-k selection, selected K/V
  gathers, QK, normalized exponential, and PV. A frozen StableHLO fixture keeps
  the accepted compiler test independent of future JAX export changes.
- Recovery lowers the graph to a generic runtime `RelationSelectionProgram`,
  `RelationPlan`, `Contract`, `Map`, `DomainRestriction`, and `Fold` program.
  The shared semantic-erasure validator proves scheduling keys contain no named
  attention or oracle dispatch.
- Two different metadata inputs produce different runtime relation edges while
  retaining the same generated tensor program and query-major/KV-major
  candidate family. The relation-driven online reference matches the natural
  JAX source with maximum/mean error below `0.016`/`0.002`.
- Added an H100 harness that forwards GPU top-k output directly into the
  generated SM90 relation index plane and times router Contract, top-k/index
  generation, and selected exact attention together. The existing 0.491984-ms
  backend checkpoint predates this matched boundary; a fresh run and a
  symmetric expert-oracle measurement remain pending.
- The complete tile-lifetime suite now passes 176 tests; package source and the
  new frontend test pass Pyrefly, and the touched Python files pass Ruff.

## 2026-08-07: Matched StatefulScan clean-synthesis target

- Added a same-process, same-input oracle boundary for the natural JAX
  delta-rule recurrence. Shuttle and pinned FLA receive identical Q/K/V,
  log-decay, beta, and initial state; Q/K normalization is disabled and query
  scale is one on both paths.
- The first matched audit rejected the historical comparison: generated
  execution measured 0.595696 ms while same-run FLA measured 0.434368 ms, a
  1.371-times ratio.
- Profiling isolated repeated diagonal-prefix work in generic factor
  preparation. A K=64 physical tile reduced preparation to 0.281424 ms.
- The final interleaved 50-sample medians are 0.466752 ms generated and
  0.420528 ms FLA, or 1.1099 times. This passes the clean-synthesis performance
  target on the matched boundary.
- Two subsequent independent captures counterbalanced every warmup and measured
  pair and reversed the initial implementation order. Their pooled 100-sample
  medians are 0.465824 ms generated and 0.424304 ms FLA, or 1.097854 times.
  This confirmation passes both the 1.20-times completion target and the
  separately reported 1.10-times stretch target.
- Scalar/per-key decay and rank-one/rank-two mutations still use the same
  generator, remain finite and bitwise deterministic, and have maximum output
  error no greater than `4.883e-4`.
- The shared semantic-erasure validator now executes before candidate
  enumeration. It records only generic `Scan`/`Map`/`Contract` lowering and
  structural scheduling keys, while tests reject workload-named or stale keys.
- The reproducible report and passing status are under
  `benchmarks/artifacts/stateful_scan_affine_pipeline_h100_v0`. This closes the
  current matched StatefulScan core proof.

## 2026-08-08: Clean attention and MoE semantic-body boundaries

- Replaced the fixed-position dense-region recovery with producer/consumer
  dataflow discovery over erased `Contract`, `Map`, `Fold`, and
  `DomainRestriction` operations.
- Removed direct FlashAttention `Softmax`, `AttentionMask`, and score-mod helper
  dependencies from the generated SM90 path. Shuttle now owns the
  normalized-exponential Fold state, score Map, domain restriction, and output
  finalization while retaining the extracted CuTe physical pipeline.
- Replaced handwritten MoE SwiGLU and semantic merge arithmetic with scalar
  CUDA emitted from the recovered plan. Generic pair-Map and ordered-Fold loop
  skeletons call the generated functions, and build/runtime digest guards
  reject plan/source drift.
- On four GB200s, two counterbalanced 30-sample captures give pooled medians of
  4.147536 ms for Shuttle and 3.647136 ms for MoK, a `1.137204×` ratio. The
  generated path is bitwise deterministic and differs from MoK by at most
  `0.0001220703125`.
- The MoE row therefore passes the clean Map/Fold boundary and the 1.20-times
  performance gate. The natural router is still executed by a small generic
  Torch Contract/top-k/Fold adapter rather than a generated router kernel; this
  limitation is recorded explicitly.
- A low-priority H100 validation request remained `SchedulingGated` and was
  removed without consuming a GPU. Dense and routed sparse attention remain
  device-pending after the attention helper extraction.
- Evidence is sealed under
  `benchmarks/artifacts/gb200_moe_clean_map_fold_v1`.

## 2026-08-08: Generic tiled Fold finalization

- Added a backend-neutral tiled Fold-finalization program with dense and
  indexed addressing, validity, explicit physical feature layout, scalar-state
  reduction, vector update/finalize ASTs, and source-ordered versus
  deterministic-tree numerical contracts.
- One SM100 emitter now generates the same 128-bit copy, four-stage shared
  staging, warp-distributed state reduction, vector feature accumulation, and
  deterministic store loop for normalized-exponential attention merge and a
  six-slot indexed non-attention weighted Fold.
- The indexed mutation agrees exactly with its source-order reference and
  measures 0.018128 ms on GB200. A small attention binding is deterministic
  with 0.00134033 maximum error.
- At the 16K natural MSA boundary, pooled isolated medians are 3.823488 ms for
  generated Shuttle and 3.191376 ms for pinned MSA, a 1.198069-times ratio.
  This clears the 3.88-ms objective and lies at the 1.20-times acceptance gate.
- Underfilled top-k slots are explicit invalid relation edges. Six remaining
  routing mismatches come from one exact cutoff tie; exact source-order tie
  selection remains open and is not hidden behind the performance result.
- Evidence is under
  `benchmarks/artifacts/msa_generic_tiled_fold_sm100_v1`.

## 2026-08-08: Training boundary and generic reverse mode

- Exported a natural one-layer Grug MoE train step containing JAX autodiff and
  optimizer update. Reference attention, scatter MoE, and XLA ragged
  contraction produce zero StableHLO custom calls; SGD and AdamW fixtures
  contain 82 `dot_general` operations each.
- Added generic reverse mode for scalar Map ASTs, multilinear Contract
  adjoints, sum-Fold broadcast adjoints, and broadcast-Map reductions.
- A SwiGLU-to-tanh mutation changes the derived VJP without backend source
  changes. A generated RMSNorm-GEMM backward matches independent `dx`,
  `dgamma`, and `dW` formulas within 2e-5.
- The package suite passes 220 tests. Physical backward generation and matched
  H100 timing remain open.
- StableHLO evidence is under
  `benchmarks/artifacts/grug_moe_train_step_stablehlo_v0`; the recommended
  out-of-tree XLA integration sequence is in
  `.agents/projects/shuttle_grug_training_lowering.md`.

## 2026-08-09: Natural Grug GPU region replacement

- Replayed the ordinary one-layer Grug training step on one GB200 and recovered
  a generic Contract plus source-ordered scalar Map directly from its
  `PRE_SCHEDULER` HLO.
- XLA compiled the transformed module, executed one typed-FFI custom call, and
  returned all 53 result leaves bitwise equal to the initial natural baseline.
  The recovered call consumes two BF16 row values and one BF16 Contract weight,
  then returns both the raw Contract and its generated pairwise product Map.
- Across 30 counterbalanced pairs, baseline and transformed whole-step medians
  are 0.552480 and 0.563937 ms, respectively, or `1.020737x`. The handler
  executed 35 times including correctness and warmups.
- Repeated whole-step hashes have four variants on both paths, so this proves
  correct replacement but not bitwise determinism for every operation in the
  surrounding train step.
- The successful environment required coherent CUDA compiler components at
  13.0.88. Pip provided versioned-only cuBLAS/cuDART libraries; the compiler now
  resolves their absolute paths and disables NVCC's implicit cuDART link rather
  than relying on process-global symlinks.
- Raw HLO, generated source, every timing/hash sample, toolchain failures, and
  hardware provenance are under
  `benchmarks/artifacts/grug_contract_map_gpu_gb200_v0`.
