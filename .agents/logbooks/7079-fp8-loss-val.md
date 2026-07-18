# FP8 loss-curve validation — bf16 (main) vs FP8 (PR #7079)

Issue: https://github.com/marin-community/marin/issues/7298 (part of #6699)
PR under test: https://github.com/marin-community/marin/pull/7079
Branches: `research/mcwitt/7079-fp8-loss-val` (FP8 arm = PR merged with main @ `512eade96` + launcher) ·
`research/mcwitt/7079-fp8-loss-val-ctl` (control arm = main @ `4a37b09f5` + launcher)
W&B: project `marin_moe`, group `fp8-loss-val-7079`, tags `fp8-loss-val` / `pr7079`
Experiment IDs: `FP8VAL-NNN`

## TL;DR

- (pending) Two-arm A/B at row-13 scale (d2560/26L/64E top-4, ~18B params) on
  4×8 H100 per arm, SlimPajama-6B, full MuonH schedule to zero LR,
  ~5.8B tokens/arm. Acceptance: train-loss gap ≤0.01 at matched steps through
  the cooldown tail (the #6486 threshold).

## Hypothesis queue

| id | hypothesis | status |
|---|---|---|
| H-parity | FP8 (grouped+dense+wire, delayed per-tensor scaling) tracks bf16 within 0.01 loss through the full schedule incl. cooldown | queued |
| H-tail | if a gap appears, it concentrates in the last 10% of the schedule (cooldown numerics, cf. #6486) | queued |

## Entries

### FP8VAL-001 — design + setup (2026-07-16)

Design constraints from the request: ≤8 h wall, ≤64 H100, parallel arms,
CoreWeave-cached data, full loss trajectories.

- **Config:** row-13 shapes exactly as the MFU line of work (`grug_moe_row13`),
  `ring` MoE backend (default on both refs), EP8 intra-node × FSDP 4 nodes,
  `gpu_fa4_cute`, `recompute_all`, capacity factor 1.0 (grug default on both
  refs), batch 128 × seq 4096, seed 0 both arms. Optimizer = production MuonH
  (`lr 1e-3 / adam_lr 1e-4 / warmup 0.1 / min_lr_ratio 0.0`) over the full step
  budget so the cooldown tail is exercised (#6486: bf16-master gaps only
  surfaced in the last 10% of a decay-to-zero schedule).
- **Data:** `slimpajama_6b_dataset()` llama3-tokenized cache (version
  2026.06.28) under `s3://marin-us-east-02a/marin` (in-cluster LOTA endpoint;
  no egress), feistel block shuffle. Eval suites (paloma etc.) skipped: their
  caches are not materialized under the CW prefix and would trigger
  cross-region tokenization; per-step train loss on an identical data stream
  is the comparison instrument.
- **Sizing:** measured single-node H100 throughput for this exact config
  (B200MFU H100 bench, 2026-07-16): bf16 1.501 s/step at B32/8GPU ≈ 87k tok/s;
  FP8 1.372 s/step ≈ 96k tok/s. At 32 GPUs (same per-device batch at B128) with
  cross-node FSDP overhead the bf16 arm should land in 1.6–2.0 s/step →
  11000 steps ≈ 5.8B tokens in ≤ ~6.5 h + compile. Smoke run calibrates before
  the full launch. 5.8B tokens ≈ 4× the budget #6486 needed to resolve
  0.01-level deltas.
- **Control purity:** `fp8-moe-mlp-comms` was 134 commits behind main; merged
  main into the PR (`512eade96`, conflict-free — the branch delta is 18
  FP8-scoped commits) so the arms differ by exactly the PR content. Same
  launcher file (`experiments/grug/moe/launch_fp8_loss_val.py`) on both
  branches; FP8 enabled only via `FP8VAL_FP8=1` (guarded import so the file
  runs unmodified on main).
- **Launch command** (per arm, from the respective worktree):
  `iris --config lib/iris/config/cw-us-east-02a.yaml job run --no-wait --cpu=1 --memory=4G --extra=cpu -e WANDB_API_KEY -e FP8VAL_* -e RUN_ID -- python -m experiments.grug.moe.launch_fp8_loss_val --version dev --run`
  (launcher is a CPU step; `run_grug` dispatches the 4×8-H100 gang via Fray).

Next: FP8VAL-002 smoke (40 steps/arm), then FP8VAL-003 full A/B.

### FP8VAL-002 — smoke round 1: multi-host XLA compile hang (2026-07-17)

`fp8val-{bf16,fp8}-smoke1` (40 steps, 4×8 H100/arm, json_logger tracker — no
WANDB key reachable on the submit box): both arms scheduled instantly, synced
env, logged hparams, then **hung ~2 h inside XLA compilation**. Signature on
every rank (py-spy via `iris task exec`): main thread in
`backend_compile_and_load` (jax 0.10.1 `compiler.py:344`) after ~10 min of real
compile CPU (~600 s utime — matching the known single-node compile cost of the
26-unrolled-layer graph), then **0 CPU / 0% GPU forever**; data-loader queue
full (healthy); last log lines are `spmd_partitioner.cc` involuntary-remat
warnings. Diagnosis: the multi-host **sharded-autotuning rendezvous** (ranks
exchange autotune shards through the coordination service during compile) —
never seen in the single-process 8-GPU benches. Both jobs stopped.

Fix attempt (smoke round 2): resubmitted as `fp8val-{bf16,fp8}-smoke2` with
`XLA_FLAGS=--xla_gpu_shard_autotuning=false` on both arms (the grug dispatcher
forwards `XLA_FLAGS`/`NCCL_*`/`JAX_*` to the train gang).

**Smoke round 2 result:** the flag fixed the hang — compile completed and
execution began — but the bf16 arm **OOM'd in `jit_train_step` (59.78 GiB
allocation)**. Same structural cause as the earlier B128-on-8-GPU OOM: 26
*unrolled* layers (no array-stacked scan on either ref) make XLA hold giant
buffers; here it's the cross-node FSDP (`data=4`) weight-gathers rather than
activation scratch — per-device batch was identical to the single-node fit.

**Topology pivot (smoke rounds 3-6):** drop the data axis entirely —
**2 nodes/arm, EP16, batch 64** (`FP8VAL_GPU_REPLICAS=2 FP8VAL_EXPERT_AXIS=16
FP8VAL_BATCH=64`), 32 H100 total — still OOM'd (remat floor 57.5 GiB, program
peak 89 GiB; the June spike's "B64+L26 hits HBM walls" note holds). Two more
findings and pivots:

- **Round 4 (1 node/arm, EP8, B32 — the proven MFU-bench shape):** discovered
  the dispatcher only forwarded `XLA_FLAGS`, silently dropping
  `XLA_PYTHON_CLIENT_MEM_FRACTION` — train tasks ran at the 0.75 default pool
  (~60 GB) while the bench had used 0.90. Fixed by widening the forward
  prefixes to `XLA_*` (+ later `TF_GPU_ALLOCATOR`) on both branches. With 0.90
  forwarded, **bf16 passes** (40 steps, loss 11.79→10.99, ~89k tok/s ≈ the
  bench's 87.3k), but **FP8 OOMs at the same shape** (42.07 GiB step-scratch).
- **Round 5 (FP8, 0.95 + `cuda_malloc_async`):** ran 9 steps (loss tracking
  bf16 within ~0.004 at matched steps) then OOM'd at step 10 (44.91 GiB) —
  FP8-on-merged-branch needs a few GB more than bf16 where the pre-merge bench
  harness fit FP8 at 0.90. **Open follow-up: FP8 memory regression** — either
  the main merge (haliax scan/core changes came in) or a real-trainer×FP8
  interaction (e.g. state-donation defeated by callback references); the
  standalone bench harness on the merged tree would discriminate.

**Final shape (rounds 6-9): 1 node/arm, EP8, batch 16**, 16 H100 total —
halves activation scratch so both arms have headroom. Three more hazards
eliminated on the way:

- `cuda_malloc_async` seemed implicated in an NCCL clique-acquire stall
  (round 6) — but round 7/8 reproduced the stall *without* it on three
  different nodes, exonerating both the allocator and the hardware.
- The real trigger: **the 10-minute async time-checkpoint save deadlocks the
  bf16 run against the training step's collectives** (every stall sits at the
  first save; smoke9 stepped cleanly at 1.0 s/step until the save fired at
  step ~9, then hung forever). fp8-smoke5's "step-10 OOM" also coincides with
  the first save. fp8-smoke7 survived its save — the hazard is
  timing-dependent. Mitigation for the throwaway validation runs:
  `FP8VAL_CHECKPOINTS=local` (cw_scale's node-local mode, no periodic saves).
- bf16 also stalls at NCCL comm-init with `XLA_PYTHON_CLIENT_MEM_FRACTION`
  0.95 but is fine at 0.90 (prealloc starving NCCL buffers); fp8 needs 0.95
  and runs fine there. Full runs use the per-arm proven fraction (numerics
  unaffected by pool size).

Measured B16 throughput: fp8 ~73.5k tok/s (0.89 s/step, smoke7 full 40
steps); bf16 ~65k tok/s (1.0 s/step, smoke9 pre-save steps).

### FP8VAL-003a — full1 post-mortem (2026-07-17 15:40 UTC)

Both full1 arms wedged within minutes and were killed:

- **bf16-full1** stalled at the same NCCL "Acquire clique" signature at step ~9
  — **with checkpoints disabled**, killing the checkpoint theory. The stall
  follows a "Data loader stalled … queue_size=0" message: it coincides with
  the first block-shuffle refill from object storage (~10-20 min in), i.e. a
  shifted dispatch pattern. py-spy: main thread blocked in C inside the
  `train_step` dispatch (`train.py:522`); the acquire that never completes is
  a **second, lazily-created NCCL communicator** — the signature of XLA's
  nccl **comm splitting**. The proven-clean runs (bf16-smoke4 B32, bench)
  only ever stepped for ~1 min and never reached a refill.
- **fp8-full1** landed on a node with the known **poisoned uv jax cache**
  (`barrier_test` ImportError, same as the MFU-bench day) — g8498e8; purged
  2.7 GiB of cached jax/jaxlib via `task exec` + `uv cache clean jax jaxlib`.
  Bonus find: after the ImportError the process hung in shutdown for 2 h, so
  fray never retried — the job sat "running" while dead.

**Fix attempt (full2):** relaunched both arms 15:41 UTC with
`--xla_gpu_enable_nccl_comm_splitting=false` appended to `XLA_FLAGS` (forces
full communicator init instead of ncclCommSplit — the stock mitigation for
lazy-split clique hangs). Monitor now has explicit stall detection
(no-progress ≈19 min → alert).

### FP8VAL-003b — ACTUAL stall root cause: heavy log reads wedge the training process (2026-07-17 19:40 UTC)

The full2 arms ran cleanly for 3.5 h (bf16 to step 11199) and 1 h (fp8-full3 to
2269) — then **both wedged within ~2 minutes of one event: a heavy
`iris job logs --max-lines 900000` harvest against both running jobs** (for the
mid-run loss plot). Retro-correlating every earlier stall: fp8-full2's 17:43
stall followed two consecutive full-log greps; bf16-full1's 13:52 stall
followed the traceback-investigation reads; smoke9's "checkpoint stall"
coincided with log inspection. The log server is a uvicorn thread *inside* the
training process; a heavy read freezes the host loop mid-dispatch (GPUs 0%,
loader queue full, storage healthy, `Acquire clique` watchdog spam) and never
recovers. The comm-splitting flag, checkpoint, allocator, mem-fraction, and
bad-node theories were all chasing this confound (the mem-fraction and
sharded-autotune fixes remain real, independently verified issues).

**Rules adopted:** full-log harvests only on terminal jobs (history is fully
retained server-side); running jobs get only small-window default-cap reads at
≥2.5 min cadence. Both arms restarted 19:38 UTC **with s3 checkpoints** so any
future wedge costs ≤10 min: `fp8val-bf16-full3` (from scratch, checkpoints on)
and `fp8val-fp8-full3b` (same RUN_ID `fp8val-fp8-full3` → resumes from the
~step-2200 checkpoint). ETA ~02:45 / ~01:15 UTC.

**Interim parity (mid-run snapshot, before restart):** over 6118 matched steps
(fp8-full2 replica vs bf16): mean Δ = +0.0050 (σ 0.0029), window means flat
(+0.002→+0.006→+0.006→+0.004); the independent fp8-full3 series agrees
(+0.0036 over its first 2268 steps). Max single-step |Δ| 0.087 (noise). Well
inside the ≤0.01 bar, no growth trend. Plot artifact:
https://claude.ai/code/artifact/99ec4cac-2537-494c-b9e2-1b285311fa32

### FP8VAL-003c — full3 down: checkpoint saves are a coequal wedge trigger; full4 is the stable config (2026-07-17 21:57 UTC)

full3 lasted <25 min: **bf16-full3 wedged at its first 10-min async checkpoint
save** (no heavy log reads this time — clique-acquire signature at 20:00, ~20
min in, same as smoke9's save-coincident stall), and **fp8-full3b OOM'd on
checkpoint *resume*** (40.64 GiB in `jit_train_step` post-restore — the
restore path leaves enough extra live to push the step arena over; fresh
starts at the same config fit) then hung dead-but-"running" like full1.

Tabulating all attempts: the only runs that lived for hours were the two with
**no periodic checkpoints and only light log polling** (bf16-full2 3.5 h,
fp8-full2 2 h — each ultimately killed by a heavy log read, not by itself).
Conclusion: two independent wedge triggers — **async checkpoint saves** and
**heavy log serves** — both injecting host-side contention during collective
dispatch. Revised FP8VAL-003b: log reads are one trigger, not the whole story.

**full4 (21:57 UTC, final config):** both arms fresh, `FP8VAL_CHECKPOINTS=local`
(no periodic saves, no resume — accepted risk since the no-save config is the
only one with a multi-hour clean record), light monitoring only, nothing else
touches the jobs until terminal. ETA fp8 ~04:15, bf16 ~05:00 UTC.

`/mwittmann/fp8val-bf16-full1` and `/mwittmann/fp8val-fp8-full1`, 24000 steps
× B16 × seq 4096 = **1.57B tokens/arm** (≥ the 1.44B that resolved 0.01-level
deltas in #6486), full MuonH schedule (warmup 2400 steps, cosine-free linear…
schedule per `GrugMoeMuonHConfig` defaults, cooldown to 0), seed 0, identical
data order. Expected: bf16 ~6.7h + compile, fp8 ~6h. Per-step `train/loss` in
job logs via `fp8val.metrics` json_logger; trajectories to be harvested and
attached to the issue at completion.

### FP8VAL-004 — TRUE root cause of every wedge: BFC fragmentation OOM of the step scratch buffer → silent clique deadlock (2026-07-17 22:45 UTC)

Full-history harvest of all terminal jobs (safe post-mortem reads) plus source
dives into iris and levanter kill both prior theories and replace them with
one mechanism, present in **all five** wedges:

```
W bfc_allocator.cc:514 Allocator (GPU_k_bfc) ran out of memory trying to
  allocate 22.66GiB   (bf16 arms; 40.68GiB on fp8 arms)
[~10 s later]
E rendezvous.cc:116 ... Acquire clique: devices=8:[0..7] ... may be stuck
```

| wedge | last step | OOM → stall (UTC) | failing GPUs | chunk |
|---|---|---|---|---|
| bf16-smoke9 | 9 | 12:49:29 | 1 | 22.58 GiB |
| fp8-full2 | 6119 | 17:43:48 | 1 | 40.68 GiB |
| bf16-full2 | 11199 | 19:09:41 | 3 | 22.66 GiB |
| fp8-full3 | 2269 | 19:09:21 | 5 | 40.68 GiB |
| bf16-full3 | 9 | 20:00:48 | 2 | 22.64 GiB |

The allocator dumps show the pool only ~40% occupied at failure — this is
**fragmentation, not exhaustion**: the train step needs one contiguous
22.6 GiB (bf16) / 40.7 GiB (fp8) temp chunk per launch, and after enough pool
churn no hole that large survives. The device threads whose allocation fails
never join the 8-way clique rendezvous; the other ranks wait forever — an
XLA failure mode where alloc failure during collective launch deadlocks
silently instead of raising. That is the entire "wedge".

**Log-read theory (003b): refuted.** `iris job logs` never touches the task
process: reads are served by the standalone finelog Rust server scanning
parquet on disk (`lib/finelog/rust/src/server/log_service.rs`), fed by a
logship sidecar tailing the pod's CRI stdout file
(`src/iris/cluster/backends/k8s/logship.py`). The in-task uvicorn thread is
the Prometheus telltale (`src/iris/runtime/telltale.py`), which serves no
logs. Timeline agrees: bf16-full2 *survived* the heavy 400k/600k-line
harvests at 18:08–18:09 and died at 19:09 with no heavy read within 10 min;
the 900k harvest previously blamed ran at 19:19, *after* both 19:09 stalls.

**Checkpoint-save theory (003c): demoted to perturbation.** On one node the
levanter save path has no barrier or collective (all multihost sync is
`process_count>1`-gated); it is main-thread D2H staging + a background S3
commit thread. Its alloc/free churn plausibly re-carves the BFC pool — both
checkpoint-adjacent wedges landed 3 s / 11 s after the *commit* callback —
but checkpoint-less arms died identically after hours of slow fragmentation,
so saves only modulate timing.

The 19:09 cross-job simultaneity (two nodes, 20 s apart) remains the one
loose end — consistent with a shared external hiccup (monitor cycle, S3/net
blip) perturbing loader/dispatch allocation patterns on both nodes at once,
under a mechanism that only needs a nudge when this fragile.

**Quantified fp8 memory regression:** largest step temp is **40.68 GiB (fp8)
vs 22.66 GiB (bf16)** at the identical B16 config — fp8 needs 57% of the
~72 GiB pool contiguous, which is why fp8 arms always died first (1–2 h vs
3.5 h). This is the concrete target for the planned bisect.

**Mitigation:** the allocator's own hint — `TF_GPU_ALLOCATOR=cuda_malloc_async`
(VMM-backed, no contiguity requirement, immune to BFC fragmentation; already
forwarded by dispatch; previously exonerated of causing stalls). full4 runs
without it and 0/6 full arms have ever finished, so a wedge remains likely;
resubmit-on-wedge commands now carry the flag (a preemptive restart was
declined by policy — investigation was scoped to run alongside full4).
Secondary: leave contiguous headroom (larger mem fraction where NCCL init
tolerates it), and upstream an XLA report: alloc failure inside collective
launch should abort the rendezvous, not hang.

### FP8VAL-005 — fp8 +15 GiB temp-arena regression attributed: float8 dual-write operands + requant scaffolding (2026-07-17 23:55 UTC)

A compile-only probe (`experiments/grug/moe/fp8val_mem_probe.py`, one 8xH100
node, builds the exact row-13 step and AOT-compiles it bf16 then fp8 without
running a step) pins the regression precisely.

`compiled.memory_analysis()` (exact):

| arm | temp arena | args | output | alias |
|---|---|---|---|---|
| bf16 | 22.65 GiB | 29.88 | 29.88 | 29.88 |
| fp8 | 37.67 GiB | 29.89 | 29.89 | 29.89 |

**Persistent state (args/output/alias) is byte-identical** — fp8 adds nothing
to params/grads/opt-state. The entire **+15.0 GiB is transient** peak memory in
the single fused train step (~1.66x bf16). The ~0.3 GiB jitter across probes
(37.67–37.94) is autotune nondeterminism — the same jitter that put bf16 arms
±0.3 GiB across the fragmentation cliff at the step-9 watch executable.

**Per-dtype arena composition** (parsed from XLA
`...-buffer-assignment.txt` `preallocated-temp`; offset-dedup "covered" basis
over-counts absolute peak ~1.85–2.31x because arena offsets are reused across
the step, so read the *deltas and fp8-only dtypes*, not absolute GiB):

| dtype | bf16 | fp8 | Δ | what it is |
|---|---|---|---|---|
| f8e4m3fn | 0 | 5.00 | **+5.00** | forward fp8 operands (e4m3 activations/weights) |
| u8 | 0.02 | 3.29 | **+3.27** | transposed dual-write fp8 operand copies (fp8-as-u8) |
| f8e5m2 | 0 | 0.80 | **+0.80** | backward fp8 gradient operands (e5m2) |
| f32 | 4.88 | 9.01 | +4.13 | dequant / scale / amax-reduce / accumulation scratch |
| bf16 | 47.23 | 51.21 | +3.97 | extra bf16 intermediates around quantize ops |

**Verdict — hypothesis 1 (dual-write / mixed-dtype quantized operands held
live) confirmed as the driver; hypothesis 2 (custom_vjp defeats remat, pinning
*bf16* residuals) is NOT the primary cause.** The bf16 arena has **zero**
float8 tensors; the fp8 arena adds ~9 GiB of float8-family data (e4m3 5.0 +
u8-transpose 3.3 + e5m2 0.8) that simply does not exist in bf16. XLA
materializes each large linear's quantized operand — and its transposed layout
for the backward GEMM — and keeps them resident across the fused
forward+backward rather than requantizing in each GEMM prologue. The remaining
+4 f32 (scale/dequant/amax scratch) and +4 bf16 are the requantization
scaffolding around those ops. bf16 grew only ~4 of the 15 GiB, so remat-pinning
is at most a minor contributor.

**Reducibility:** the dominant cost is the dual-write transpose (e4m3 5.0 +
u8 3.3 ≈ 8.3 GiB) — recomputing the transposed fp8 operand in the backward
instead of stashing it, or not pre-casting operands that a single GEMM
consumes, is the highest-leverage lever. The f32 scale/dequant scratch is
secondary. This is inherent overhead of the current fp8 wiring, not a compiler
accident — a real follow-up for PR #7079, tracked separately from the loss
validation.

**Operational aside (allocator fix landed):** `bfc_allocator` fragmentation
(FP8VAL-004) is defeated by **`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`**
(VMM-backed, no contiguity requirement). NB the fix is this JAX var, NOT
`TF_GPU_ALLOCATOR=cuda_malloc_async` (TF-only; full5 set it and still wedged at
step 9 on BFC). bf16-full6 (cuda_async) cleared the step-9 cliff that killed
full4/full5 and is running; fp8-full4 passed step 6760 (longest fp8 run of the
campaign).
