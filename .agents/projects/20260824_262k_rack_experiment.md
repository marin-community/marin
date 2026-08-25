# Plan: 262K hero-shape probe on one NVL72 rack (integration branch)

**Branch:** `long-context/262k-rack-experiment` (worktree `~/projects/marin.long-context-262k-experiment`)
**Goal:** measure MFU, drop rate, memory, and loss of the `moe_hero_ep` template at seq 262144
on one rack (16 nodes x 4 GB200), restored from the hero's step-6000 checkpoint. This is gates
1–2 of the early-cooldown list in `.agents/projects/20260824_535b_long_context_cooldown.md`
plus a first hero-scale CP datapoint. **No PRs/issues/GitHub posts. Local branch only.**

## Composition

1. Merge, in order: `long-context/mesh-context-axis` (final), `long-context/fa4-sequence-sharding`,
   `long-context/moe-context-sharding`, `long-context/te-cp-backend` (TE inert here; merged for a
   single integration point). Resolve `experiments/grug/moe_hero_ep/model.py` overlaps (FA4 owns
   attention-boundary hunks, MoE owns token-axis hunks — they were written to compose).
2. Port the bench-side launcher delta (`git diff origin/main...bench/8549-capacity-sweep`) for:
   `experiments/grug/moe_hero_ep/launch_mfu_test.py`, `experiments/grug/moe_hero_ep/train.py`,
   `experiments/grug/moe_hero_ep/model.py` (ragged splits field), `experiments/grug/dispatch.py`,
   `experiments/grug/pjrt_wheel.py`, `experiments/grug/checkpointing.py` if the launcher needs it.
   Verify every hunk lands on the intended symbol (backport hazard). Skip the autoresearch/ arm
   harness files.
3. New launcher flags: `--context-axis-size` (→ `GrugTrainerConfig.context_axis_size`),
   `--expert-axis-size` (override `HERO_EP_EXPERT_AXIS_SIZE=64`), `--qk-mult` (model override).
   The seq-len grid logic from `4c49b011fe` already holds tokens/step fixed (batch = tokens/seq).

## The EP+CP parameter-memory change (required, new work)

At EP16xCP4 each expert-rank holds 4x the hero's per-rank parameters: fp32 pinned-host master
+ Muon momentum ~= 1072 GB/node > ~850 GB available → restore would OOM the host. Recorded CP
design (grug-context-parallel-attention logbook): FSDP parameters over the composite
(data, context) so CP does not multiply parameter/optimizer memory.

Implementation: store EP-mode expert weights (and their master/optimizer state) sharded
`P(("expert", "context"), None, None)` instead of `P("expert", None, None)`; `moe_mlp` already
reshards weights to the EP spec before its shard_map, which becomes a per-layer all-gather over
`context` (~1.4 GB per rank per layer for routed weights — cheap on NVLink). Attention/dense
weights similarly extend `_FSDP_AXES` with `context`. Restore is unaffected structurally
(checkpoints record no mesh; the template dictates placement). Gate: at `context_axis_size == 1`
the composite spec is identical to today's.

## Run plan

- Mesh: replica 1 x data 1 x context 4 x expert 16 x model 1 (64 GPUs). Batch 16 x seq 262144
  = 4,194,304 tokens/step (hero-matched). B=16 over batch axes (1*1*16) = 1 sequence per
  expert-coordinate; 65,536 tokens per device.
- Restore: `--restore-from s3://marin-us-east-02a/marin/grug/hero-12d8b6f0-dee637/2026.08.19.2/checkpoints/step-6000`
  `--restore-master-params fp32_pinned_host --master-params fp32_pinned_host` (disabled would put
  268 GB fp32 on device at EP16 — does not fit).
- `--qk-mult 1.84` (= 1.3 x (0.1*ln(262144/4096) + 1), the YaRN-mscale recipe from #6811).
- `--num-steps 6100` (absolute stop step; ~100 measured steps), mixture data (never synthetic —
  a trained router routes by content).
- MoE: pooled transport (hero default), 1 process/node, capacity factors unchanged (1.15/1.15,
  3 waves) — drop rate at 262K vs the 4K baseline (MFU 22.13, drops 1.7e-4 from
  autoresearch/loop-260824-seqlen/results.tsv) is the headline measurement.
- Launch: iris coordinator pattern from `experiments/grug/moe_hero_ep/README.md` on
  `cw-us-east-08a`, `--priority interactive`, unique `IRIS_PORT_JAX`, per-run
  `JAX_COMPILATION_CACHE_DIR`, `IRIS_USER mwittmann`, `MARIN_PREFIX s3://marin-us-east-02a/marin`.

## Pre-flight gates (before the rack run)

1. GPU tests: the FA4 CP kernel changes are GPU-only-testable — run the new GPU-marked tests in
   `lib/levanter/tests/grug/test_fa4_cute_attention.py` (and `tests/test_moe_hero_ep.py -k
   context_parallel`) on a small GB200 allocation (2–4 GPUs) from this integration branch.
2. CPU: full `uv run --no-project infra/ci/run_tests.py` green on the merged branch.
3. A short small-shape rack-free smoke (single node, 4 GPUs, CP4 over a tiny model config) to
   catch integration-level sharding errors before burning a rack slot.

## Pre-flight results (2026-08-25, gates 1 and 3)

Gate 1 passes. `lib/levanter/tests/grug/test_fa4_cute_attention.py` is 19/19 green on one
`cw-us-east-08a` GB200 tray, including all three `..._with_context_sharded_queries`
parametrizations, which are the only hardware coverage of the CuTe `q_offset` arithmetic and the
backward `m_block_min` clamp. `tests/test_moe_hero_ep.py -k "context_parallel or context_sharded"`
is 2/2 green. The context-sharded case first failed at four visible GPUs because it fixed the batch
at one sequence while `compact_grug_mesh` gave `data` the devices `context` left free; the batch now
comes from the mesh (commit a3b0a4b492) and the case passes at two and at four GPUs.

Gate 3 passes on the mesh, on synthetic data: 20 steps of d768 at sequence 32768 on one GB200 tray,
once at `context_axis_size=4` / `expert_axis_size=1`
([W&B](https://wandb.ai/marin-community/marin_moe/runs/cp4-smoke-03)) and once at 2 / 2, which is
the arrangement that puts the pooled-wave transport behind a context-sharded token axis
([W&B](https://wandb.ai/marin-community/marin_moe/runs/cp2ep2-smoke-05)). Both fall monotonically
11.806 -> 3.863 with no sharding error; the 2x2 arm reports finite drops settling from 7.9e-3 to
3.1e-6, all at the receiver.

One thing to watch at rack scale: only the 2x2 arm logs `[SPMD] Involuntary full rematerialization`,
twelve tensors per rank per step, all of them the stacked routed-expert weights (`f32[8, 192, *]`,
that is layers x local experts x width). XLA cannot reshard
`{devices=[1,1,4]<=[2,2]T(1,0)}` to `{devices=[2,1,1,2] last_tile_dim_replicate}` efficiently, so it
replicates and re-partitions. The CP4/EP1 arm logs none, so the cost belongs to the composite
`("expert", "context")` parameter spec and appears only where both axes exceed 1 -- which the
EP16xCP4 rack mesh does. Watch the step time and the peak HBM against the EP-only baseline.

**The run plan's data configuration will not start.** `LmDataConfig` slices every mixture component
to `int(sequences * experiment_budget / target_budget)`, and `experiment_budget` is
`total_schedule_steps * batch * max_seq_len` against an 18.75T target. At `--num-steps 6100`,
batch 16 and sequence 262144 the ratio is 1.36e-3, and because a component's length is counted in
*sequences* (tokens // 262144, 64x fewer than at 4096) three of the mixture's 200 cells (`c13q0`,
`c22q0`, `c27q0`) slice to zero. `MixtureDataset` then raises `ValueError: ... encountered an empty
finite dataset`, before the first step. The same step and batch count at sequence 4096 empties no
cell, so this is specific to the long context. Reproduced twice on hardware, at
`context_axis_size` 4 and 1 alike, so it is not a context-parallel fault. Resolve it before the
rack run: raise the schedule the budget is computed from, drop the simulated-epoching budget for
this probe, or drop the cells that cannot fill one sequence.

## Attempt 3 (2026-08-25 21:50 UTC): the step runs

Run `lc262k-ep16cp4-08251450`, job `/mwittmann/lc262k-ep16cp4-08251450-coord`, on `7962b135ea`
(halo + context-sharded residual). Gang admitted in under a minute, restore of the step-6000
checkpoint complete by 21:54, first optimizer step at 22:00. No OOM, no SConv rejection, no
`Involuntary full rematerialization`, no `hlo_rematerialization` overflow.

**Peak HBM 127.63 GiB against the 138.22 GiB pool limit.** The same step needed 350.36 GiB before
the residual fix, so context-sharding the residual removed roughly 223 GiB per device and left about
10.6 GiB of headroom.

Steady state over steps 6000-6012, from raw `_timestamp` deltas rather than the smoothed tqdm rate:

| quantity | value |
|---|---|
| s/step | 113.5 median (113.3-115.5 after the first three) |
| tokens/s | 37,042 |
| MFU | 8.78 % |
| loss | 1.416 median, restored at 1.336, range 1.09-1.52 |
| `moe/drop_fraction` | 0.1805 median, rising 0.168 -> 0.207 |
| sender / receiver | 0.0806 / 0.1008 |
| peak HBM | 127.63 GiB |

Two results worth separating. Throughput: 8.78 % MFU against roughly 21 % for the 4K hero on one
rack, which is the cost of 262K attention at a hero-matched 4.19M tokens per step.

Routing: **the drop rate is 18 %, against 3.5 % for the live 4K hero and 1.7e-4 for the 4K one-rack
baseline**, and it is drifting up rather than settling. The capacity factors are the hero's 1.15/1.15
and were tuned at sequence 4096 with a global batch of 1024. Here the batch is 16, so each expert
coordinate routes one document's 65,536 tokens instead of sixteen independent 4,096-token
sequences; token-to-expert load within a single long document is far more correlated, and the fixed
per-expert capacity clips it. `train/router/capacity_overflow_rate_mean` tracks `drop_fraction`
exactly, so the loss is entirely capacity clipping and not transport. This is the headline
measurement the probe was for, and it says the 4K capacity settings do not carry to 262K at this
batch.

Read the loss trajectory with the learning-rate caveat below: at ~6.9e-5, about 2 % of the rate the
checkpoint was trained at, the model is barely moving and the step-to-step spread is batch-16 noise
rather than learning.

## The fix: a context-sharded residual, and the halo that unblocks it

`short_conv` now carries a left halo over the context axis (`ad62b7f18a`), so the contract is
shard-local *plus* `kernel_size - 1` tokens from the left neighbour, fetched with one ppermute
inside a shard_map. The Pallas kernel is untouched and its guard still rejects a sharded sequence;
the wrapper sits above it and serves the reference and Pallas bodies alike. The concatenated block
is right-padded to the kernel's block size (safe: a causal convolution never reads right of an
output it keeps), and the halo carries its neighbour's segment ids so packed-document taps are
dropped exactly as unsharded.

`_embedding_gather` then establishes `P(_BATCH_AXES, "context", None)` on the residual
(`0a92ee59de`), which is the only place the layout is set. Everything downstream already composed.
This closes the composition gap between the FA4 and MoE branches: each supported a context-sharded
residual, neither established one.

**One-node gate, d768, tokens per device pinned at 65,536:**

| sequence | context | peak HBM before | peak HBM after | loss at step 20 |
|---|---|---|---|---|
| 65,536 | 1 | 12.68 GiB | 12.68 GiB | 3.7367 |
| 131,072 | 2 | 15.31 GiB | 12.94 GiB | 3.7366 |
| 262,144 | 4 | 19.71 GiB | 13.59 GiB | 3.7365 |

The ladder flattened: per-doubling increments fell from +2.63 and +4.40 GiB to +0.26 and +0.65 GiB.
What remains growing is the K/V all-gather, which is genuinely linear in the sequence and is what
context parallelism buys attention with. CP1 is unchanged to the byte, and the loss trajectories
agree across cp1/cp2/cp4 at every step to four decimals (11.8059 -> 3.7365), the residual spread
being bf16 reduction order on different meshes.

## The residual stream is sequence-replicated, and SConv is why it cannot simply be sharded

**Verified.** Under a CP4 mesh the hero model's residual carries
`P(('replica_dcn','data','expert'), None, None)` at the embedding output, at every `_activation_spec`
round trip through the MoE, and at the final hidden -- identical to the CP1 case. The sequence
dimension is replicated end to end. `_activation_spec`'s own docstring says as much
("seq-replicated today"), and `_embedding_gather` hardcodes `out_specs=P(_BATCH_AXES, None, None)`.
Neither sibling branch establishes the layout: the MoE branch returns the caller's spec by design,
and the FA4 branch reshards only `q`, inside the attention block.

This is the composition gap, and it is the right size. At the rack mesh the batch axes span 16 with a
global batch of 16, so a block input is `[1, 262144, 6144]` bf16 = 3.0 GiB replicated against
0.75 GiB context-sharded. Across 48 layers that is roughly 108 GiB of block inputs alone, the order
of the ~230 GiB long context added at hero width. Forcing the sharded layout at the embedding does
propagate cleanly: `_activation_spec` then carries
`P(('replica_dcn','data','expert'), 'context', None)` through the MoE to the final hidden.

**The fix is blocked on SConv.** The hero runs `sconv=True`, `sconv_kernel=4`,
`sconv_sites=('k','attn','mlp')` -- three calls per layer, 144 across the model. `short_conv` is
shard-local along the sequence with no halo exchange, and its Pallas wrapper refuses a sharded
sequence rather than hide the gather:

```
short_conv requires an unsharded sequence axis for x;
got P(('replica_dcn', 'data', 'expert'), 'context', None).
```

`_assert_local_axes` raises exactly this on a context-sharded activation, and its docstring is
explicit that the point is "to refuse to paper over a real all-gather with a silent reshard". So a
context-sharded residual either fails at every SConv site or is gathered back three times per layer,
which re-materializes the full-sequence activation and returns the memory the change was for.

The missing capability is a left halo of `sconv_kernel - 1` = 3 tokens from the left neighbour in the
context group. The traffic is trivial (3 x 6144 x 2 B = 36 KB per call per boundary); the capability
simply does not exist, and it lives in a shared levanter kernel rather than in this branch. **That was
a design change, and it was signed off and implemented as the halo above.** The CPU verification above does not trip
the guard, because CPU falls back to `short_conv_reference` and never enters the Pallas wrapper; the
guard was exercised directly instead.

## Attribution of the OOM (2026-08-25, one GB200 node, no rack)

**There is no sequence-squared buffer.** Three d768 arms on one `gb200-1node` tray, each holding
tokens per device fixed at 65,536 and varying only the sequence and the context width, all
completed their training steps and logged `memory/peak_gib`:

| sequence | context | peak HBM | run |
|---|---|---|---|
| 65,536 | 1 | 12.68 GiB | `cp1probe-s65536-c1-1400` |
| 131,072 | 2 | 15.31 GiB | `cp2probe-s131072-c2-1400` |
| 262,144 | 4 | 19.71 GiB | `cp4probe-s262144-c4-1358` |

Each doubling of the sequence adds 2.63 then 4.40 GiB. A quadratic term would quadruple the
increment and a linear one would double it, so the sequence-driven cost is at most linear: the whole
4K-to-262K move costs 7.03 GiB on a 12.68 GiB base at this width. The `jit_train_step` at sequence
262144 with context 4 **fits on one tray in 19.71 GiB**.

The 96 GiB and 192 GiB overflows those probes also produced are **not** the train step. Every one is
`jit(accum_for_batch)` -- the evaluator -- and the failing buffer is `bf16[262144, 256, 6, 128]`,
exactly 96.00 GiB, which is K or V for a whole eval batch. `small_scale_abl_launch` fixes
`EVAL_BATCH_SIZE = 256` regardless of `--seq-len`, so its eval batch is 67M tokens at sequence
262144 against a 4.2M-token training step. That is a real bug in the ablation launcher, unrelated to
the hero, which runs with eval disabled.

So the rack OOM is **width-driven, not sequence-driven**. Scaling the sequence term by width (x8
from d768 to d6144) accounts for roughly 56 GiB of the roughly 230 GiB that long context added at
hero width, so most of the gap belongs to something present at `expert_axis_size` 16 and absent at
1. The one-node probes cannot see it: at expert axis 1 the composite `("expert", "context")`
parameter spec is inert, which is why they logged no `Involuntary full rematerialization` at all,
exactly as the pre-flight's CP4/EP1 arm did not.

The 12 replicated parameter copies are themselves small -- summing the shapes the rack logged gives
about 1 GB -- so if the composite spec is the cause, the cost is not the copies but what replicating
them does to the gradients and optimizer state that depend on them. Confirming that needs a mesh
with both axes above 1 at hero width, which one tray cannot provide.

## Attempt 2 (2026-08-25 20:44 UTC): the step does not fit in HBM

Run `lc262k-ep16cp4-08251344`, job `/mwittmann/lc262k-ep16cp4-08251344-coord`, on commit
`cedb42d399` (the mesh-config fix). The gang was admitted in a minute, cleared
`trainer.initialize()`, and all 16 ranks restored the step-6000 checkpoint at 20:47:13 -- about
three and a half minutes for a 535B checkpoint, with no host-memory pressure. Compiling
`jit_train_step` then failed to fit:

```
W hlo_rematerialization.cc:3282] Can't reduce memory use below 193.77GiB (208055883007 bytes) by
  rematerialization; only reduced to 350.36GiB (376197049916 bytes), down from 350.79GiB originally
E gpu_cudamallocasync_allocator.cc:359] cuMemAllocAsync failed to allocate 281923031072 bytes
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 262.56GiB.
  [executable_name='jit_train_step']
```

The step needs 350.36 GiB per device against a 193.77 GiB budget, and rematerialization recovered
0.43 GiB of it. **This is not the `cuda_async` release-threshold trap (#8490).** That one is a
runtime pool-limit effect on a program that fits; here the compiler's own static requirement is 1.8x
physical HBM. The pool limit in the log is 138.22 GiB (the 0.75 default fraction of the 184.3 GiB
usable); raising the fraction to 1.0 buys about 46 GiB against a 157 GiB shortfall, so no memory
setting reaches this. That is why no retry was attempted.

The predicted reshard is real, and it now costs memory rather than only time. Rank 0 logged exactly
12 `[SPMD] Involuntary full rematerialization` warnings, all on `f32` parameter copies: one
`[64,96,6144]`, one `[64,6144,96]`, two `[64,96,1536]`, five `[64,96,3072]`, three `[64,3072,96]`.
Each goes from `{devices=[1,64,1]}` or `{devices=[1,1,64]}` -- a flat 64-way shard -- to
`{devices=[4,1,1,16] last_tile_dim_replicate}`, a 4-way shard on `context` replicated over the 16
`expert` ranks. That target is the composite `("expert", "context")` parameter spec from
`859eeebd25`, and the 96-sized axis does not divide 64, hence the `pad` and then the
give-up-and-replicate. The pre-flight saw this at 2x2; at EP16xCP4 it arrives at full width.

The 262.56 GiB allocation is not attributed yet. Two candidates land near it by arithmetic -- a
sequence-squared attention buffer (262144^2 x 4 B = 256 GiB) and a fully replicated parameter copy --
and telling them apart does not need a rack: lower and compile the step once and read
`.compile().memory_analysis()`, or compile with `--xla_dump_to` and read the buffer assignment. Do
that before requesting the rack again.

Neither the drop rate, the MFU, nor the loss trajectory was measured. Rack occupancy was about six
minutes; all 16 tasks released (1 `failed`, 15 `cosched_failed`) and no GPUs stayed held.

## Attempt 1 (2026-08-25 20:17 UTC): batch 16 is below the device count

Run `lc262k-ep16cp4-08251317`, job `/mwittmann/lc262k-ep16cp4-08251317-coord`. The 16-node gang
queued 14 minutes on Kueue, was admitted at 20:32:46, and every one of the 16 processes died three
seconds later in `trainer.initialize()`:

```
  File "/app/experiments/grug/moe_hero_ep/train.py", line 828, in _run_grug_local
    trainer.initialize()
  File "/app/lib/levanter/src/levanter/trainer.py", line 1145, in _validate_and_set_defaults
    elif self.train_batch_size % (self.per_device_parallelism * self.data_axis_size) != 0:
ZeroDivisionError: integer modulo by zero
```

Grug builds its own `compact_grug_mesh` and never gives `TrainerConfig` a `MeshConfig`, so the
trainer validates against the levanter default, where `axes={"data": -1}` hands `data` all 64
devices and `batch` maps to `(replica_dcn, replica, data)`. `data_axis_size` is therefore 64, not
the 16 that grug's own `_BATCH_AXES = ("replica_dcn", "data", "expert")` gives. At global batch 16
`per_device_parallelism` falls to `16 // 64 == 0` and the next line divides by it.

Nothing context-parallel is involved: any hero-template run whose global batch is below the fleet's
device count hits this, and holding tokens/step fixed at 4,194,304 forces batch 16 at sequence
262144. The hero has never gone below batch 1024. `validate_mesh_axes` did not catch it because it
models grug's batch axes only and never sees levanter's default mesh.

Setting `per_device_parallelism` alone does not fix it -- the same line then rejects
`16 % (1 * 64) != 0`. The fix is to give `TrainerConfig` a `MeshConfig` that matches the compact
grug mesh, with `compute_mapping={"batch": ["replica_dcn", "data", "expert"]}` and the real axis
sizes, so `data_axis_size` is 16. Check the other `data_axis_size` consumers (microbatching,
`per_device_eval_parallelism`) while doing it.

The run never opened the checkpoint and never compiled, so the restore, the SPMD reshard warning,
and the drop measurement are all still unmeasured. Rack occupancy was about two minutes; all 16
tasks released (1 `failed`, 15 `cosched_failed`) and no GPUs stayed held.

## The learning rate this probe actually applies

`--schedule-steps 4470000` sizes the mixture budget correctly (18,748,538,880,000 tokens against
the 18.75T target, ratio 0.99992, no cell slices empty), but it does not put the optimizer
heuristic at the hero's budget. `build_hero_configs` reads `HERO_MODEL.max_seq_len`, which is 4096;
the `--seq-len` override is applied to the model afterwards. The heuristic therefore sees
4470000 x 16 x 4096 = 293B tokens and returns peak `learning_rate` 5.158e-4 against the hero's
3.290e-3. Warmup is `0.01 * 4470000` = 44,700 steps, so at step 6000 the probe applies about
6.9e-5 -- roughly 2 percent of the 3.274e-3 the step-6000 checkpoint was trained at.

Throughput, drop rate, and memory do not depend on this, and a low rate is the conservative side of
a restore. The loss trajectory does: **this run does not diagnose 262K training stability**, only
throughput, memory, and routing. A stability read needs the heuristic computed at the real sequence
length, or an explicit LR override matching the checkpoint's schedule the way the 67B cooldowns did
with `GrugMoeMuonHResumeConfig`.

## Measurements to record

`tokens/s`, s/step (raw elapsed stamps, not smoothed tqdm), MFU, `moe/drop_fraction` +
sender/receiver split, loss trajectory vs the step-6000 baseline, peak HBM + host RSS, compile
time, and any XLA collective anomalies. Compare drop rate against the 4K one-rack baseline.
