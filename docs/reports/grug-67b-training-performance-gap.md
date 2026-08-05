# Grug 67B training performance: Levanter and MarinSkyRL

Status: the frozen headline cohort, the eager matched-CE isolation control, and
the separately labeled present-day routed-expert attribution are complete and
content-hash verified. The later ring arithmetic oracle is also complete; no
result is pending.

## Executive answer

On the same fixed logical update and four-host/32-H100 topology, Levanter's
fastest valid operational update takes 255.859 seconds. MarinSkyRL's fastest
valid update, with FlashAttention and FSDP2 EP1, takes 6,991.670 seconds: 27.33x
as long, a gap of 6,735.810 seconds. Within MarinSkyRL, FlashAttention saves
222.543 seconds (3.085%) over eager attention. The median FlashAttention run
spends 84.57% of wall time in backward and 15.24% in model forward plus entropy;
the measured policy-loss, optimizer, status-collective, and unclassified spans
are all small by comparison.

The exact four-term time identity puts 6,933.802 seconds in the cross-stack
eager matched-CE term, 27.236 seconds in MarinSkyRL's native eager boundary,
-222.543 seconds in its FlashAttention intervention, and -2.684 seconds in
Levanter's native boundary. The cross-stack term is a raw upper bound, not an
accepted causal attribution. MarinSkyRL's eager and FlashAttention matched
global CE values are respectively 0.018952 and 0.018939 above Levanter's, so
both miss the predeclared absolute 0.01 consistency gate. The two MarinSkyRL
values differ by only 0.000013. The numerical mismatch is therefore not
localized to the attention backend. The headline timing and direct
within-MarinSkyRL interventions remain valid.

A separately labeled present-day matched-CE run partitions its 6,854.120-second
critical-rank wall into 5,583.786 seconds (81.466%) in routed experts and a
1,270.334-second exact nonexpert remainder. Subtracting all 253.176 seconds of
Levanter's matched-CE wall gives a useful 5,330.611-second hybrid diagnostic,
equal to 79.138% of the frozen gap. It is not a lower bound on the frozen gap:
the full-replay stacks fail the represented-parameter/loss comparability gate,
and the 111.351-second source bridge does not partition expert from nonexpert
drift. The remaining correctness blocker and the now-rejected paired result
are stated below.

A post-freeze 32-H100 semantic oracle compared native Levanter ring EP8 with
the exact 12,288 action-log-probability coordinates stored by the eager and
grouped MarinSkyRL arms. The FP32-combine candidate reduced squared distance
from Levanter by 12.9% for eager and 13.4% for grouped relative to their parent
arms. It did not establish a grouped advantage: grouped had 0.014% more squared
error than eager in the candidate artifact. This supports the candidate's
numerical direction, but it neither passes the candidate's failed distributed
semantic gate nor makes its observed 14.23x grouped speedup admissible. The
oracle is excluded from every timing headline.

The comparison uses one content-addressed rollout replay, the June step-630
weights, pinned measurement-freeze source in both repositories, and 32
H100-80GB training GPUs.
Compilation, checkpoint loading, replay preparation, warmup, and profiler cost
are outside every headline timer. Generation, reward work, checkpoint transfer,
and serving synchronization are not measured.

Both stacks ran in `cw-rno2a` on four complete H100x8 hosts. The scheduler did
not pin the same physical GPU UUIDs across the separate jobs, so “same 32
H100s” here means the same count, model, cluster, and complete-host topology,
not identical devices. The final GPU jobs ran sequentially. Levanter and the
fastest MarinSkyRL path each have three samples. The eager operational,
FlashAttention matched-CE, and eager matched-CE interventions each have one
bounded sample; no spread is claimed for them.

## Comparison contract

### Fixed logical update

The replay is an archived 8K RL policy update, reconstructed twice from
independent Ray spill objects. Both reconstructions produced logical batch hash
`e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d`.

| Property | Value |
| --- | ---: |
| Logical sequences | 4,096 |
| Allocated positions | 32,817,152 |
| Non-padding positions | 25,095,420 |
| Loss/response positions | 24,494,588 |
| Sequence tensor shape | `[4096, 8012]` |
| Action/loss tensor shape | `[4096, 6656]` |
| Rank shards | 32 shards of 128 sequences |
| Manifest SHA-256 | `5d2479bbbdcd4ca04a9f7d11de82ce42830fbae878d734cdc3c4a4f123f93b74` |

The manifest and shards live under
`s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/replay-step-1-global/e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d/`.
Each benchmark verifies the manifest, every shard, reconstructed field hashes,
and the logical hash before model initialization.

Separate CPU readbacks reverified all 32 shard hashes and inspected every
stored mask value. All 27,262,976 float32 `loss_mask` values were exactly zero
or one; their sum and nonzero count both equal the manifest's 24,494,588 loss
positions. The integer `response_mask` is also binary and differs from
`loss_mask` at zero positions. Thus the operational action set, summing the
loss mask, and counting selected positions all use the same exact tokens.

The replay was recovered with `scripts/perf/recover_ray_spill.py`.
The manifest embeds the source job, image, revision, spill-object key and
offset, object ETag, payload hash, checked backing-storage reconstruction,
per-field hashes, and every exported shard hash.

### Weights and paths

Both paths start from step 630 of
`june-67b-a2b-sft-s2-thinking`. Levanter reads the native checkpoint at
`s3://marin-us-east-02a/marin/grug/grug_67b_a2b_sft_s2_thinking/2026.07.16/checkpoints/step-630`.
MarinSkyRL reads its BF16 Hugging Face export at revision
`a822321c2c21af099189e7116104b3cf5142c119`.

The MarinSkyRL measurement revision is based on measurement-freeze main
`1388f3ec1e68aad2248c08fdf20c184df45267a5`. Its production-file delta passes
the exact model revision into loading and adds optional benchmark phase hooks;
the hooks are inactive unless this evidence driver installs an event recorder.
The remaining delta is the fixed-replay driver and its validation support.

The native checkpoint keeps FP32 master parameters and applies its pending
query-bias update before BF16 model compute. The Hugging Face export's embedded
provenance says that update was baked into its router biases before all
parameters were stored in BF16; its weight index has router-bias tensors and no
separate pending state. Thus the paths are intended to represent the same
training state, but they are not a bit-identical parameter-precision
experiment. The matched global CE check below independently tests the
numerical consequence instead of accepting provenance alone.

Both operational controls initialize fresh optimizer and scheduler state around
those weights; neither reconstructs the step-630 optimizer moments. Warmup
materializes the optimizer state, then each timer starts again from its exact
fresh state. This comparison measures training-path cost, not the exact numeric
next update of a resumed production optimizer.

The operational headlines deliberately preserve each stack's valid native
training semantics:

| Stack | Operational path |
| --- | --- |
| Levanter | FP32 master parameters with BF16 compute/output, FA4/CuTe attention, stacked 26-layer scan, ring EP8, four-way data sharding, token-mean CE plus logit z-loss, pending QB update, AdamH |
| MarinSkyRL | BF16 model compute, FSDP2 EP1, no packing, layer checkpointing, one sequence per microbatch, policy log-probability and diagnostics, AdamW |

Both paths consume the 4,096 replay rows as 128 global microsteps of 32
sequences, one sequence per H100, and accumulate one gradient boundary. The
Levanter microsteps run inside one compiled scan. MarinSkyRL runs 128 serial
microbatches on each FSDP rank. Levanter applies the checkpoint's pending QB
update exactly before this step. Its next-step QB statistic is the arithmetic
mean of the 128 native per-microstep statistics because the full
token-by-expert statistic does not fit. Its router auxiliary term is likewise
the arithmetic mean of 128 per-microstep native statistics, while CE remains
one exact global token sum. These repacking choices are recorded in each
result.

The matched control removes that semantic difference. It uses the same token
masks and global token-weighted next-token CE through forward and backward,
with no optimizer in either stack. Before reading the MarinSkyRL result, the
cross-stack numerical gate was fixed at a finite global CE within 0.01 absolute
(about 0.5% at the native result), plus finite non-empty gradients on every
rank. This is a consistency gate, not a bit-identity claim.

### Validity gates

| Gate | Levanter | MarinSkyRL |
| --- | --- | --- |
| Source | `3b1000dc5636c446c3aaed1c035e36cc6d6fda53` | `f57b3b60f894606b8b4f4ff0a6fe7fffa2141042` |
| Runtime image | `iris-task@sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab` | `marinskyrl@sha256:505814c8666a6253dd3e00f9a5dd30889ddd3acf1a87b7dc2d0b74d490d9b1b8` |
| World | 4 complete 8-H100 hosts | 4 complete 8-H100 hosts |
| Exact replay verification | Pass | Pass |
| Requested attention/expert path proved | Pass | Pass |
| Fresh state restored before timing | Pass | Pass |
| Fastest path completed optimizer boundary | Pass | Pass |
| Matched CE consistency | Baseline: 1.994753 median | Fail: 2.013692 FlashAttention; 2.013704 eager |

The later Marin commit `7eaea9447` adds required license headers and
Black-only formatting to the evidence scripts. It does not change the executed
benchmark logic, so the content-hashed native results retain their measured
source revision `3b1000dc5`.

The final report branch then merged Marin main `4a0abf3c8` after measurement.
That integration passed the focused code and branch-wide checks, but no
benchmark result is relabeled: every result stays pinned to the source revision
it executed.

One-node Levanter preflights cannot fit this exact 8K state. The operational
compiler proved a 109.82 GiB input/output floor on the one-node mesh; matched
CE requested one contiguous 39.07 GiB buffer. The four-node mesh adds the
four-way data axis used to shard parameters and optimizer state. These are
excluded capacity failures, not timing samples.

The exact-image MarinSkyRL preflight likewise reached model load, materialized
AdamW state, restored fresh state, and then ran out of memory during the first
timed backward pass. That preflight used eight-way full-shard FSDP and left
only 430 MiB free when backward requested 1.56 GiB. The headline instead uses
32-way full-shard FSDP. It formed all 32 ranks and entered policy training.
The one-node failure is path evidence, not a timing sample or a capacity claim
about the four-node topology.

The native preflight also found a correctness defect in FA4's leading-padding
tile pruning. An invalid first query in a 128-query tile carried the sequence
length as its lower key bound, which could prune key tiles needed by later
valid queries. At production head dimension 128, the unfixed kernel disagreed
with a float32 reference in 62.0% of full-attention outputs and 67.5% of
sliding-window outputs. The correction propagates the next valid query's lower
bound through invalid prefixes while leaving the exact score mask unchanged.
Afterward, the full one-H100 file run passed all ten runnable tests, including
every head-dimension-128 reference comparison across full and sliding attention
and forward/backward. Three head-dimension-64 cases were excluded for an
unrelated existing CUTLASS API incompatibility. A one-microbatch 32-H100 model
diagnostic then reported zero non-finite gradient arrays or elements across all
four processes; the same diagnostic before the correction reported 22
non-finite arrays and 21,279,500,800 non-finite elements. Every native headline
below is from the corrected revision.

## Results

### Stack-native operational headline

Rates count the fixed logical batch above. GPU-seconds per logical sequence is
`wall seconds * 32 / 4096`, so it stays meaningful across batch repacking.
For three-sample headlines, spread is the maximum wall minus the minimum; a
bounded one-sample control has no claimed spread.

| Stack and path | Wall seconds | Spread | GPU-s / sequence | Allocated tok/s/GPU | Non-padding tok/s/GPU | Peak memory evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Levanter native, FA4 + ring EP8 | 255.859 | 0.196 | 1.9989 | 4,008.2 | 3,065.1 | 80,057 MiB max device used |
| MarinSkyRL, FlashAttention + FSDP2 EP1 | 6,991.670 | 17.390 | 54.6224 | 146.7 | 112.2 | 47.6/56.4 GiB allocated/reserved |
| MarinSkyRL, eager attention control | 7,214.213 | one bounded sample | 56.3610 | 142.2 | 108.7 | 63.9/76.3 GiB allocated/reserved |

The three Levanter operational walls were 255.766, 255.962, and 255.859
seconds. The median is 2.684 seconds (1.06%) above its matched CE median. All
four JAX processes completed the AdamH boundary and reported zero non-finite
parameter or optimizer-state values. The separate 32-H100 diagnostic above is
the direct gradient-finiteness evidence.

The three MarinSkyRL FlashAttention walls were 7,006.140, 6,991.670, and
6,988.750 seconds. Their 17.390-second spread is 0.249% of the median. All five
planned MarinSkyRL measurements ran in one fixed-order, no-retry allocation.
The extra eager matched-CE isolation ran afterward in its own no-retry
allocation. Separate CPU fetches recomputed every result and payload hash
before applying the frozen acceptance queries. The operational ratio is
27.326x and the absolute wall gap is 6,735.810 seconds.

### Matched CE forward/backward control

| Stack and path | Wall seconds | Spread | GPU-s / sequence | Allocated tok/s/GPU | Non-padding tok/s/GPU | Peak memory evidence | Global CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Levanter matched CE | 253.176 | 0.288 | 1.9779 | 4,050.7 | 3,097.6 | 80,051 MiB max device used | 1.994753 median |
| MarinSkyRL FlashAttention matched CE | 6,965.471 | one bounded sample | 54.4177 | 147.2 | 112.6 | 37.6/47.6 GiB allocated/reserved | 2.013692 |
| MarinSkyRL eager matched CE | 7,186.977 | one bounded sample | 56.1483 | 142.7 | 109.1 | 53.8/72.6 GiB allocated/reserved | 2.013704 |

The FlashAttention and eager matched wall ratios are 27.512x and 28.387x.
Both MarinSkyRL results have finite gradients on every rank, with at least 476
nonempty gradient tensors and 2,096,173,280 gradient elements. Their CE values
are respectively 0.018939 (0.949%) and 0.018952 (0.950%) above the native
median, so both fail the predeclared absolute 0.01 numerical consistency gate.
They differ from each other by only 0.000013. Attention arithmetic therefore
does not explain the cross-stack mismatch. Both rows are valid timing
observations, but neither is an accepted common-objective causal control.

The mismatch is precise at the comparison boundary even though it is not
localized to one tensor. Levanter starts from FP32 master parameters and casts
for BF16 compute; MarinSkyRL starts from parameters already stored in BF16 in
the Hugging Face export. They therefore do not use identical represented
parameter values, and the 0.95% full-replay loss difference shows that
provenance alone is not enough to make them a common-objective control. This
report does not claim that representation rounding explains every part of the
loss difference.

A smaller common-representation oracle provides the alternative semantic
boundary. [Marin commit `6195867835efa48c03b739a3aa67ef06cea78f16`](https://github.com/marin-community/marin/commit/6195867835efa48c03b739a3aa67ef06cea78f16)
exports ordinary FP32 values that both implementations load exactly. On that
fixture, MarinSkyRL matches every selected route exactly; hidden states,
outputs, weights, query-bias candidates, and loss use `2e-5` absolute and
relative tolerances; selected gradients use `5e-5`; and one deterministic SGD
plus query-bias update preserves the next routes exactly. This establishes
implementation agreement when the represented values are shared. It is a
correctness oracle, not a full-replay timing result.

The three native CE values were 1.9947575, 1.9947482, and 1.9947525. Every
warmup and timed JAX process reported zero non-finite gradient arrays or
elements.

The peak-memory cells are validity evidence, not a cross-stack performance
metric. Levanter samples total device memory through `nvidia-smi` during the
timed call. MarinSkyRL reports PyTorch's peak allocated and reserved bytes after
resetting its allocator counters. Those definitions are different, so this
report does not subtract or ratio them.

MFU is omitted. There is no single documented active-MoE FLOP formula that
applies unchanged to both the ring-EP native path and the FSDP2 policy path.

## Causal gap decomposition

Let `Nop` and `Nce` be Levanter's operational and matched-CE walls. Let `Mfop`
and `Meop` be MarinSkyRL's FlashAttention and eager operational walls, and let
`Mece` be its eager matched-CE wall. The exact identity is:

```text
Mfop - Nop = (Mece - Nce) + (Meop - Mece) + (Mfop - Meop) - (Nop - Nce)
```

Algebraically, this separates the cross-stack CE term, each stack's native
boundary, and the direct eager-to-FlashAttention intervention. `Mece` uses
MarinSkyRL eager attention while `Nce` uses native FA4, so the first term also
contains that cross-stack backend difference. Because both MarinSkyRL matched
CE gates failed, the first row below is a raw timing partition, not an accepted
causal label. The other three terms are direct within-stack changes on the same
replay. Table terms are rounded independently for display; the underlying
identity closes with zero error.

| Component | End-to-end seconds | Share of fastest operational gap | Evidence |
| --- | ---: | ---: | --- |
| Cross-stack eager matched-CE term, `Mece - Nce` | +6,933.802 | +102.939% | Exact raw partition; causal label rejected by both CE gates |
| MarinSkyRL eager native-path increment, `Meop - Mece` | +27.236 | +0.404% | Same eager configuration and replay |
| MarinSkyRL FlashAttention intervention, `Mfop - Meop` | -222.543 | -3.304% | Same operational path and replay; attention backend changed |
| Subtract Levanter native-path increment, `-(Nop - Nce)` | -2.684 | -0.040% | Same native configuration and replay |
| Fastest operational gap, `Mfop - Nop` | 6,735.810 | 100% | Stack-native headlines |

### Present-day routed-expert refinement

The frozen operational gap remains 6,735.810478 seconds. A separately labeled
present-day run now measures the routed-expert boundary on the same replay and
matched-CE endpoint. On the critical rank, the 26 `GrugMoeExperts` modules take
5,583.786 seconds: 246.012 seconds in their initial forwards and 5,337.774
seconds in their full backward spans. The backward spans include
gradient-checkpoint recomputation. Layer-level FSDP communication is outside
them. Everything else, including attention, CE, other modules, communication,
Python gaps, and barriers, is the nonnegative 1,270.334-second remainder. These
two spans partition the current 6,854.120-second synchronized wall; the recorded
fields close within `4e-13` seconds of arithmetic roundoff.

| Present-day matched-CE sample | Seconds | Share of wall |
| --- | ---: | ---: |
| Routed experts: initial forward | 246.012 | 3.589% |
| Routed experts: full backward, including recompute | 5,337.774 | 77.877% |
| Routed experts: total | 5,583.786 | 81.466% |
| Nonexpert remainder | 1,270.334 | 18.534% |
| Synchronized wall | 6,854.120 | 100% |

This result is not inserted into the frozen cohort. Let `Nop` and `Nce` be the
frozen Levanter operational and matched-CE walls, `Mfop` and `Mfce` the frozen
MarinSkyRL FlashAttention operational and matched-CE walls, `W` the present-day
instrumented wall, and `E` and `O` its expert and nonexpert parts. The exact
bridge is:

```text
Mfop - Nop = (Mfop - Mfce) + (Mfce - W) + E + O - Nce + (Nce - Nop)
```

| Exact term | Seconds | Meaning |
| --- | ---: | --- |
| Frozen MarinSkyRL operational boundary, `Mfop - Mfce` | +26.199 | Direct frozen within-stack difference |
| Source and measurement bridge, `Mfce - W` | +111.351 | Current-main drift, run noise, and instrumentation cost; not given a mechanism label |
| Present-day routed experts, `E` | +5,583.786 | Non-overlapping critical-rank CUDA spans |
| Present-day nonexpert remainder, `O` | +1,270.334 | Every other operation, communication, and idle gap |
| Subtract frozen Levanter matched CE, `-Nce` | -253.176 | Assigns the complete native CE wall before making a mechanism claim |
| Frozen Levanter boundary, `Nce - Nop` | -2.684 | Direct frozen within-stack difference |
| Frozen fastest operational gap, `Mfop - Nop` | 6,735.810 | Exact algebraic total; recorded fields close within `4e-13` seconds |

The bridge is explicit because the present-day source and hooks were not in the
frozen run. The current CE is `2.0136915405338756`, exactly equal to the
predeclared MarinSkyRL reference and inside the `1e-4` within-stack invariance
gate. Together with the tiny probe's exact loss and gradient comparison, this
supports using the hooks as an attribution instrument; it does not establish
full-scale gradient identity. It also does not repair the failed historical
cross-stack CE gate or claim identical represented weights between stacks.

For scale only, give routed experts all `Nce = 253.175519` seconds of Levanter's
complete matched-CE wall. The resulting hybrid diagnostic is
`H = max(0, E - Nce) = 5,330.611` seconds, or 79.138% of the frozen gap. This is
deliberately **not** called frozen gap closure or a lower bound. The Levanter
full replay starts from FP32 master values while MarinSkyRL starts from the BF16
export; their CE values miss the predeclared gate, their routes are not replayed,
and therefore compatible full-scale expert work has not been established. In
addition, the aggregate `Mfce - W` bridge cannot show whether expert and
nonexpert time moved in opposite directions between revisions.

The measured causal statement is narrower and still useful: routed-expert spans
occupy 5,583.786 seconds, or 81.466%, of the present-day MarinSkyRL matched-CE
wall under the accepted within-stack instrumentation gates. The exact blocker to
transferring that partition into the frozen cross-stack gap is the missing
common-representation full-replay control plus component-wise bridge. The
cheapest discriminator is the paired eager/grouped 32-H100 run below at one
pinned owner source, with identical represented values, replayed routes (or an
exact route/load-shape check), and the same expert/nonexpert hooks in both arms.

### Present-day expert-attribution contract

The contract was frozen before submission. It keeps the exact content-addressed
replay, BF16 Hugging Face revision, 32-H100 topology, 128 microbatches per rank,
global token-weighted CE through backward, and no optimizer. Barriers and CUDA
synchronization bracket the worker timer. Forward hooks count only initial
model forwards. Full module-backward hooks include checkpoint recompute, so the
two expert categories do not overlap. The predeclared invariance gate is a
finite loss within `1e-4` of `2.0136915405338756`, plus finite nonempty
gradients on all ranks, 26 instrumented modules, 3,328 forward calls, 3,328
backward calls, and an exact nonnegative expert/nonexpert partition. The
predeclaration text had SHA-256
`68e0da2da5091a8ba887f015de5258f1c41ac2323b2f445364c312c8d50b020a`
before the job was submitted.

Before the full run, a one-H100 tiny probe found the exact same loss and every
parameter gradient with and without hooks, while recording the expected three
modules and three calls in each phase. Its tiny timings are not used here.

The independent readback passed every gate. This is one bounded sample, so no
spread is claimed. Its source is
[`08f814440579854313a258a8dd658176557f907d`](https://github.com/marin-community/MarinSkyRL/commit/08f814440579854313a258a8dd658176557f907d),
based on MarinSkyRL main `cc8c8e8de2e7242d7e18f0563933fea0a26ac649`
plus the minimum ported fixed-replay and attribution delta. It ran image
`ghcr.io/marin-community/marinskyrl@sha256:5f35056daee57d25f134aa2171126645be6750944c92bec27962cfae412041d3`.
The revision audit from frozen baseline `1388f3ec` to `cc8c8e8d` found one
model-path change, `2088965` (#267), which only accepts the newer four-value
FlashAttention unpadding return; no routed-expert implementation changed. This
narrows the bridge, but does not turn its aggregate wall delta into a
component-wise expert bridge.

### MarinSkyRL phase evidence

The table uses the critical rank that set each synchronized wall. CUDA-event
phases are mutually exclusive. Residual is synchronized worker wall minus the
phase sum, so it includes Python launch gaps, unlabelled post-backward
diagnostics, and barriers; it is not all host overhead.

| MarinSkyRL control | Forward / entropy | Loss / policy | Backward | Optimizer | Status collectives | Residual | Critical rank | Worker spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FlashAttention operational, median sample | 1,065.225 s | 0.056 s | 5,912.595 s | 0.309 s | 4.670 s | 8.815 s | 16 | 0.00118 s |
| Eager operational | 1,125.360 s | 0.058 s | 6,074.906 s | 0.063 s | 5.290 s | 8.536 s | 8 | 0.00102 s |
| FlashAttention matched CE | 1,035.431 s | 5.883 s | 5,923.791 s | — | — | 0.366 s | 31 | 0.00084 s |
| Eager matched CE | 1,100.889 s | 7.314 s | 6,078.474 s | — | — | 0.300 s | 16 | 0.00105 s |

The median FlashAttention operational run assigns 15.236% of wall to model
forward plus entropy and 84.566% to backward. Eager minus FlashAttention sums
exactly to 222.543 seconds: 60.135 seconds in forward/entropy, 162.311 in
backward, 0.002 in policy diagnostics, 0.621 in status collectives, minus 0.246
in optimizer and minus 0.280 in residual. This is the accepted direct
attention intervention. The operational-versus-matched differences are only
bounds on their changed semantics; they do not rescue the rejected
cross-stack label. Eager operational minus eager matched CE is 27.236 seconds.
Its exact phase partition is +24.471 seconds in forward/entropy, -7.256 in
policy diagnostics versus CE loss, -3.568 in backward, +0.063 in optimizer,
+5.290 in status collectives, and +8.235 in residual.

### Focused profiles

The separate Levanter operational profile covered 261.886 seconds and contains
18,473,892 complete XPlane events. Its content-hashed result explicitly marks it
ineligible for the headline. The trace also logged exhausted CUPTI activity
buffers and dropped events, so it supports only qualitative ordering. With
exclusive duration summed across device tracks, all-gather was the hottest
classified device operation (46.245 aggregate seconds), followed by
reduce-scatter (22.390 aggregate seconds). All-reduce contributed 2.962
aggregate seconds and send/receive 0.367. These are overlapping multi-track
durations, not wall shares, and this report does not divide them by step wall
time.

There is no accepted MarinSkyRL trace. The one-node profiled preflight failed
the capacity gate before a timed result or trace was written. The headline's
mutually exclusive CUDA-event phases therefore carry the MarinSkyRL
decomposition; no incomplete trace is used as evidence. Those event records
are inside the MarinSkyRL timed wall. Their recording cost was not separately
measured or subtracted, so it remains part of that path's residual limitation.

The old isolated kernels are roadmap evidence only. In
[MarinSkyRL #248](https://github.com/marin-community/MarinSkyRL/pull/248), at
B=1/S=4096, current FlashAttention measured 4.245 ms for one attention block
forward/backward versus 18.796 ms eager. A separate one-H100 exact-shape sparse
MoE block in the grouped/EP development goal measured 15.844 ms with grouped
experts versus 1,739.815 ms in the Python expert loop, a 109.812x kernel ratio.
Peak allocated memory was 10.204 GiB grouped versus 12.045 GiB eager. The
[measured source `41d5b40`](https://github.com/marin-community/MarinSkyRL/commit/41d5b40)
also passed its four-H100 semantic parity gate. The owner's current
[PR #276](https://github.com/marin-community/MarinSkyRL/pull/276) is open,
non-draft, and at
[`0c213586`](https://github.com/marin-community/MarinSkyRL/commit/0c213586b5491b8046ca7780e965c4b26dc6a2a2),
with later correctness work and green checks. GitHub currently reports merge
conflicts. That newer implementation is not relabeled as the source of the
older measurement. Neither isolated kernel ratio is an end-to-end policy-step
speedup.

### Post-freeze ring arithmetic oracle

The checkpoint recipe used Levanter's default ring EP8 backend, so this oracle
restored the native step-630 FP32 checkpoint and pending query-bias state,
executed the full replay with BF16 compute/output, and retained the same first,
middle, and last action positions as the frozen MarinSkyRL artifacts. It
compares values at exact coordinates and deliberately declares no
cross-framework pass threshold.

| MarinSkyRL artifact | Exact eager/grouped differences | Eager closer / grouped closer | Eager / grouped mean absolute distance | Grouped/eager squared error |
| --- | ---: | ---: | ---: | ---: |
| Parent BF16 combine | 10,563 / 12,288 | 5,181 / 5,382 | 0.482389 / 0.484482 | 1.006460x |
| `fbb1fc8` FP32 combine | 117 / 12,288 | 65 / 52 | 0.403800 / 0.403853 | 1.000139x |

The parent result splits the directional tests: grouped is closer at 201 more
changed coordinates, but its total squared error is 0.646% higher. In the
candidate result, eager is closer at 13 more changed coordinates and grouped's
total squared error is 0.014% higher. Thus native ring does not pick grouped as
the more faithful arm. Comparing corresponding candidate and parent arms is
still informative: candidate squared error is 12.866% lower for eager and
13.413% lower for grouped; mean absolute distance is 16.292% and 16.642% lower.

These are exact floating-point inequality counts, not the earlier frozen
tolerance failures of 1,995 for the parent pair and 3 for the first candidate
pair. Large discrepancies are shared by both frameworks' arms—the candidate's
maximum absolute distance from Levanter is 10.916—and exact tensor identity for
the step-630 BF16 export is provenance-backed but not independently certified.
The result is therefore descriptive evidence for the FP32 correction, not a
new correctness gate or a performance measurement.

## What the old numbers do not prove

The prior 6,814-second MarinSkyRL number used 32 policy H100s, 4K allocation,
eager attention, FSDP2 EP1, 128 serial microbatches per rank, and PPO policy
work. The closest old Levanter number used 64 H100s, synthetic 4K causal-LM
data, ring EP8, and a different batch. Their roughly 14x normalized ratio is
not an apples-to-apples result and is not used here.

The separate grouped-expert/trainer-EP goal had not completed its pinned gates
at the historical measurement freeze, so its later evidence stays out of every
frozen headline and matched-control table. A separate 32-H100 production
`policy_train` observation at the measured owner revision took 429.220 seconds,
but it used a different replay and boundary and was manually stopped after its
required gates. It is unmatched diagnostic evidence, not a bridge into the
frozen arithmetic. [MarinSkyRL #249](https://github.com/marin-community/MarinSkyRL/pull/249)
landed the separate MuonH prerequisite; this comparison keeps AdamW and does
not attribute grouped-expert work to that PR.

The later `fbb1fc8` paired run did use this exact replay on 32 H100s. Eager took
7,692.213 seconds and grouped took 540.642 seconds, an observed 14.228x ratio.
It passed the declared CE and sampled-gradient checks but failed 3 of 12,288
sampled action-log-probability checks. The same artifact therefore rejects its
own performance result. A route-aware follow-up failed at one new coordinate;
the earlier three did not reproduce. Neither run closes
[Marin #7903](https://github.com/marin-community/marin/issues/7903).

After the freeze, [Marin #7820](https://github.com/marin-community/marin/pull/7820)
added a self-contained Levanter FSDP MoE hero for 64 GB200 GPUs. It neither runs
this 32-H100 replay nor implements MarinSkyRL's grouped-expert/trainer-EP path,
so it does not replace any row or roadmap gate in this report.

## Roadmap

| Rank | Opportunity | Measured or bounded end-to-end benefit | Confidence | Effort | Main dependency | Acceptance metric |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Close grouped-expert semantics, then remeasure eager versus grouped on this replay | No accepted end-to-end benefit yet; one rejected candidate pair observed 14.228x, while the directly measured present-day expert span is 5,583.786 seconds | High that this is the dominant present-day MarinSkyRL boundary; realized recovery unmeasured | Medium correctness work plus one measurement | Resolve #7903 and restack conflicting #276 (`0c213586`) on current main | One paired 32-H100 run at a correctness-passed source; identical represented values, replay, routes or exact route/load-shape check, loss, finite gradients, hook counts, and nonexpert accounting; report wall and expert-span deltas |
| 2 | Keep FlashAttention as the Grug training default | 222.543 seconds (3.085% of eager) already realized | High | Done | Packed-mask correctness | Eager/FA matched CE agrees within 0.000013 and FA has no slower fixed-replay wall |
| 3 | Avoid disabled entropy computation in the policy forward | At most 29.794 seconds in the frozen FA operational/matched pair; 24.471 seconds in the eager pair | Medium | Small | Preserve required policy metrics and gradients | Same replay and gradients; beat the 17.390-second FA headline spread |
| 4 | Amortize Python launches, status reductions, post-backward diagnostics, and the final barrier | At most 13.485 seconds from frozen status-plus-residual spans | Low | Medium | Memory-safe batching or graph capture | Same replay and gradients; reduce the directly measured status-plus-residual bound |

The paired 32-H100 discriminator has now run, but its semantic failure makes a
second unchanged run poor value. The smallest useful next tranche is to close
the residual contract in #7903 with focused tests and restack #276 on current
main, then run exactly one accepted fixed-replay A/B with the attribution hooks
left on in both arms. That run will decide whether the 109.812x isolated kernel
ratio survives the 128-microbatch boundary and how much expert and end-to-end
wall it removes. Trainer EP can remain off for this first accepted grouped-only
comparison; add it only after the grouped arm closes its numerical and
accounting gates.

## Reproduction and artifacts

Measurement jobs:

- [Replay loss-mask binary audit](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fgrug-perf-replay-loss-mask-binary-v3-20260801)
- [Replay loss/response-mask alignment audit](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fgrug-perf-replay-mask-alignment-v4-20260801)
- [Levanter matched CE](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fgrug-perf-native-headline-matched-3b1000dc5-r1)
- [Levanter operational](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fgrug-perf-native-headline-operational-3b1000dc5-r1)
- [Levanter operational profile](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fgrug-perf-native-profile-operational-3b1000dc5-r1)
- [MarinSkyRL profiled preflight](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fgrug-perf-msrl-fa-op-preflight-f57b3b60-r4)
- [MarinSkyRL headline](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fgrug-perf-msrl-headline-f57b3b60-r4)
- [MarinSkyRL eager matched-CE isolation](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fgrug-perf-msrl-eager-matched-f57b3b60-r1)
- [Eager matched-CE independent result verifier](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fgrug-perf-fetch-msrl-eager-matched-f57b3b60-r1-20260801)
- [Present-day routed-expert attribution](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fgrug-gap-attr-msrl-ce-08f8144-s1-20260801)
- [Routed-expert independent result verifier](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fgrug-gap-attr-fetch-verify-08f8144-s1-20260801)
- [Post-freeze Levanter ring oracle](https://iris-cw-us-east-02a.oa.dev/#/job/%2Fromain%2Fgrug-ring-oracle-headline-b53145bbc6-r2)
- [Ring-oracle independent result verifier](https://iris-cw-us-east-02a.oa.dev/#/job/%2Fromain%2Fgrug-ring-oracle-readback-b53145bbc6-r1-20260805)

Content-hashed result objects:

- `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/native/headline-matched-ce-3b1000dc5-r1.json`
- `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/native/headline-operational-3b1000dc5-r1.json`
- `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/native/profile-operational-3b1000dc5-r1.json`
- `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/native/profile-operational-3b1000dc5-r1-summary/{summary.json,report.md}`
- `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/msrl/headline-fa-operational-f57b3b60-r4-s{1,2,3}.json`
- `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/msrl/headline-eager-operational-f57b3b60-r4-s1.json`
- `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/msrl/headline-fa-matched-ce-f57b3b60-r4-s1.json`
- `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/msrl/headline-eager-matched-ce-f57b3b60-r1.json`
- `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260801/attribution/headline-fa-matched-ce-expert-08f8144-s1.json`
- `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260805/ring-oracle-headline-b53145bbc6-r2.json`

Each JSON object includes a canonical `result_sha256` over its content excluding
that field. Independent CPU readbacks recomputed and matched these accepted
hashes:

| Result | `result_sha256` |
| --- | --- |
| Levanter matched CE | `f35e977145897844d953cf340ef11f879c9c3d3e240a83ac22abd13ed5bf1add` |
| Levanter operational | `cede4ec49025336166615d3123fb333053c11a196aa8ad7ce82d3b649a7b4e53` |
| Levanter operational profile | `51a8884229dabf4ef9cd70be49575882f188450ac371c5a0ad353674915cabc0` |
| MarinSkyRL FA operational samples 1/2/3 | `0c625580582eeb57b9b593a06bdb8ad0a23e531f6608d1ecdd953aeaf50aa65c`; `9cdbbb88750f80bc143627d9e49581c154f0a1bbf01e3225f8079c5cbae863e3`; `3ac176b5a3dc3710cedd25b0f61bf2bcb2bdd2eb5e1e27dee626cefa53ecbcbd` |
| MarinSkyRL eager operational | `5d036d0af5193b7ee21eabfe201fb9fd4cec8a2906dfa2d151d113b36b81b0d5` |
| MarinSkyRL FA matched CE | `0a205b68f1671abc517a7ab5978b32e1f102071cdb49497f3f05855e5b0f0d1d` |
| MarinSkyRL eager matched CE | `298a9bc2c5aa121c340a769d2931c8c42c29440b6c7917b71f18d6fa99c68e01` |
| MarinSkyRL present-day routed-expert attribution | `51163e4045b5c64d97d6812966680a4e2538f065e20137cd5d7837ffdca3fa22` (payload `b105ae3029bf3239319a44159aeef8808b1f6a982d3779ff3be09447f70e8cbc`) |
| Post-freeze Levanter ring oracle | `c6b5a6d942982fa4ec272c808987140a483314238f67bd5bbac1764a7567e782` (payload `179e859b1db014ba11818102487d3ed8089553e5e4de6fcf4f3b2ff6f7df56ba`) |

The durable benchmark drivers are
`scripts/perf/grug_levanter_fixed_replay_benchmark.py` in Marin and
[`skyrl-train/scripts/grug_fixed_replay_benchmark.py`](https://github.com/marin-community/MarinSkyRL/blob/f57b3b60f894606b8b4f4ff0a6fe7fffa2141042/skyrl-train/scripts/grug_fixed_replay_benchmark.py)
in MarinSkyRL. Each result records source and image identity, configuration,
topology, GPU UUIDs, timing boundary, replay identity, start-state evidence,
per-rank timing, and peak memory.

The final jobs requested four replicas with one complete `H100x8` host per
replica, 48 CPUs, 1,600 GB host memory, 4,000 GB disk, production priority, no
retries, and the exact images above. The MSRL preflight requested one such
host. GPU jobs ran sequentially.

The post-freeze oracle separately used four complete H100x8 hosts in
`cw-us-east-02a`, interactive priority, zero retries, source
`b53145bbc67bb18265c0f29bd308c087890b2542`, and image
`ghcr.io/marin-community/iris-task@sha256:9af9a3d38f57c2ed8dfe1d6f6657a9f4a00c582ec06a5ac2af8fcddbe51da03c`.
Its 400.588-second compile and 69.643-second forward scan are explicitly not
training benchmarks.

<details>
<summary>Exact driver arguments</summary>

Both drivers used these fixed inputs:

```text
--manifest-s3-uri s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/replay-step-1-global/e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d/manifest.json
--manifest-sha256 5d2479bbbdcd4ca04a9f7d11de82ce42830fbae878d734cdc3c4a4f123f93b74
--logical-batch-sha256 e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d
```

Levanter added the native checkpoint, `--source-revision 3b1000dc...`, the
native image digest, `--mode headline`, and either `--objective operational`
or `--objective matched_ce`. Headlines used `--samples 3`; the separate
profile used one operational sample plus `--profile-dir` and
`--profile-s3-prefix`.

MarinSkyRL added the step-630 model and revision, `--source-revision
f57b3b60...`, the MSRL image digest, and `--mode headline`. The allocation ran
FlashAttention operational sample 1, one eager operational sample,
FlashAttention operational samples 2 and 3, then one FlashAttention
`matched_ce` sample. The separate isolation allocation used eager attention and
`--objective matched_ce`. The one-node preflight used
`--mode preflight --attention-backend flash_attention_2 --objective
operational` with a rank-0 profile URI.

The separately labeled present-day attribution used the same model, model
revision, manifest, logical batch, `headline`, `flash_attention_2`, and
`matched_ce` arguments, plus `--expert-attribution`, source
`08f814440579854313a258a8dd658176557f907d`, image
`ghcr.io/marin-community/marinskyrl@sha256:5f35056daee57d25f134aa2171126645be6750944c92bec27962cfae412041d3`,
sample 1, and the 2026-08-01 attribution result URI above.

The post-freeze semantic oracle used `--objective sampled_logprobs --mode
headline --samples 1`, restored the native step-630 checkpoint, and compared
the two frozen MarinSkyRL artifacts after verifying their payload and result
digests. It stores exact first/middle/last action coordinates for every rank and
microbatch. Its result declares `headline_eligible=false`.

</details>
