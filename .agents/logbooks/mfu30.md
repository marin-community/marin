# mfu30: single-rack hero MFU from 28.3% to 30%

Branch `research/mcwitt/mfu30` (worktree `~/projects/marin.mfu30`), based on main `f38da1173d`.
Toolkit: `autoresearch/loop-260930-mfu30/`. Prefix for entries and runs: `M30-`.

## Contract

- Goal (user, 2026-09-30): reach 30% MFU on one GB200 NVL72 rack (64 GPUs, `cw-us-east-08a`) running the
  `moe_hero_ep` recipe restored from a hero checkpoint, with loss trajectories numerically matching main.
- Metric: W&B `throughput/mfu` median over the scored window (MFU is computed from `throughput/duration`,
  so host-side gaps between steps do not count). 30.0% = 13.084 s/step (628.04 PF per step, 64 x 2.5 PF/s).
- Baseline (main): `gcab-control-20260930` (main `44a4188c19`, 360 steps) median **28.31% / 13.864 s**; peak
  103.09 GiB of a 138.2 GiB pool (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`). Gap to goal: -0.78 s/step (-5.6%).
- Compute: interactive priority, one rack. The spare rack is shared FIFO with other sessions; queue waits
  of several hours are expected and do not end the goal. Single-node (1-4 GPU) GB200 jobs run on the
  hero racks' spare nodes and do not wait for the rack.
- Fidelity: no added quantization, no checkpoint-incompatible changes. A candidate's pointwise loss must
  stay within the same-code (C-C) band against a same-seed control from the same checkpoint.

## Protocol

- Checkpoint (pinned for the campaign):
  `s3://marin-us-east-02a/marin/grug/hero-fa4sm100-nomask-step146k/2026.08.19.2/checkpoints/step-180000`.
- Arm: `python -m experiments.grug.moe_hero_ep.launch_diagnostics --run-id <id> --seed 0 --num-steps 180060
  --schedule-steps 390251 --batch-size 1024 --gc-interval 100 --restore-from <ckpt> --version dev --run`
  submitted with `iris --cluster=marin job run --target-cluster cw-us-east-08a --priority interactive
  --enable-extra-resources --cpu 2 --memory 8GB --disk 32GB --timeout 3540`, env `WANDB_API_KEY`,
  `WANDB_PROJECT=marin_moe`, a unique `IRIS_PORT_JAX`. `--num-steps` is an absolute stop step.
- Score steps 180005-180059 (median). Profiled arms add `--profile-start-step 180021 --profile-steps 3`
  and score outside the profiled steps.
- Controls: the three `mhep-ctx4k-s{0,1,2}-20260930` runs (another session's context-length study,
  main `f38da1173d`, same checkpoint, 100 steps, seeds 0-2) are same-code controls at hero geometry. Seed 0
  gives the same-data control for loss comparisons.
- Accept: >= 2 treatment draws, median gain >= max(0.15 MFU, 3 x control sd), loss within the C-C band,
  peak HBM within the pool. Single draws are screens only.

## M30-001 Baseline anatomy (2026-09-30, provisional)

Source: `overlap-remeasure-main-144k` (main `8d70d9cbb0`, Sep 24, step-144000 restore, steps
144010-144013, rank 0). Main has since gained #9278 (Muon sharding of unpadded expert stacks) and host-side
changes; the fresh main trace `pgle-main-trace-144k` (another session, main `f38da1173d`) will replace it.
Profiled span 14.44 s/step vs 14.02 s W&B median for that run (~3% profiler inflation).

Step decomposition (per step): compute-stream busy 11.79 s + exposed collectives 1.68 + exposed host
copies 0.62 + idle 0.35 = 14.44 s. Collective busy 3.96 s at 58% overlap. Rematerialization
(`rematted_computation` scopes) is 2.75 s of the compute stream.

GEMM throughput: every bf16 GEMM family runs at 1.4-1.7 PF/s (56-67% of the 2.5 PF/s dense peak). This
holds for cuBLAS and QuACK alike, and for GEMMs with no collective overlap (attention projections:
1.52-1.68 PF/s). Shared-MLP GEMMs that overlap the ragged all-to-all drop to 1.46 PF/s (1.62-1.73 when
not overlapped). Memory-bound fusions already run at 4.6-5.6 TB/s. Their headroom therefore comes
from removing passes, not from faster kernels.

### Top 10 scopes

Times are compute-stream seconds per step unless noted. SOL uses 1.9 PF/s as the practical bf16 GEMM
ceiling and 7 TB/s as the HBM ceiling; headroom is the optimistic critical-path estimate.

| # | scope | s/step | bound | achieved | SOL est. | headroom |
|---|---|---|---|---|---|---|
| 1 | MoE routed-expert grouped GEMMs (QuACK; fwd 0.85, remat 0.86, bwd 1.82) | 3.53 | compute | 1.57-1.67 PF/s | ~3.0 | ~0.5 |
| 2 | Shared-expert DenseMLP GEMMs (fwd 0.48, remat 0.51, bwd 0.98) | 1.97 | compute | 1.42-1.49 PF/s | ~1.55 | ~0.4 |
| 3 | Exposed collectives (ragged a2a 0.80, FSDP all-gather 0.44, u32 all-reduce 0.24, RS 0.06) | 1.68 exposed | latency / rank skew | 58% overlap | ~0.9 (#9481 reaches 81%) | 0.3-0.7 |
| 4 | Attention Q/K/V/O projections | 1.51 | compute | 1.52-1.68 PF/s | ~1.33 | ~0.2 |
| 5 | FA4 attention kernel (fwd 0.18, remat 0.19, bwd 0.47) | 0.83 | compute | ~0.7 PF/s useful (28%) | ~0.45 | ~0.35 |
| 6 | RMSNorm / GatedNorm | 0.75 | memory | 5.6 TB/s | ~0.5 unfused | ~0.3 (fusion only) |
| 7 | Exposed host-offload copies (opt state at step end ~0.38, carry prefetch in bwd ~0.22) | 0.62 exposed | C2C IO | - | ~0 on device | 0.4-0.6 |
| 8 | MoE latent projections (W_down / W_up) | 0.60 | compute | 1.61-1.68 PF/s | ~0.52 | ~0.08 |
| 9 | Optimizer (MuonH Newton-Schulz GEMMs 0.17 + elementwise 0.29) | 0.46 | mixed | 1.48 PF/s NS | ~0.3 | ~0.15 |
| 10 | Attention elementwise (RoPE, QK norm, gating, layout) | 0.42 | memory | 4.7 TB/s | ~0.26 | ~0.2 (fusion) |

Below the cut: MoE expert elementwise (SwiGLU bwd, masks) 0.37; device idle 0.35; loss / lm_head 0.34
(GEMMs 54-62%); dispatch+combine marshal 0.34 (+ inverse-permutation argsort); short conv 0.19.

The u32 all-reduce is one `psum_invariant` of the MoE drop counts outside the layer loop. It takes
0.24 s on rank 0 and hides no work, so it is almost certainly rank-skew wait. Per #8317's overlap-90
study, removing a sync point moves the wait to the next collective.

### Reward / risk (1-10, higher is better)

| scope | score | reasoning |
|---|---|---|
| Exposed host copies (#7) | 7 | Config-level: keep MuonH state on device (~31-35 GiB against ~35 GiB pool headroom). Exact numerics. Risks: HBM fit and the #8317 donation/aliasing family (partial residency C3' corrupted; full residency is the standard small-run path). |
| Dispatch/combine: inverse routing | 8 | Implemented in `c2b3a4e8db` (hero-optim-round2, never landed): argsort replaced by scatter, exact outputs, +1.60% in one restored pair. Needs a port to main plus a confirmation. |
| Exposed collectives (#3) | 7 | PGLE (#9481 mechanism) on plain main is in flight in another session. New scheduling work beyond it was exhausted by overlap80/90; skew is hardware per-GPU speed. |
| Shared MLP + dense GEMM efficiency (#2, #4, #8) | 6 | 4.1 s of cuBLAS GEMMs at 57-67%. Gate/up interleave prototype exists (CPU-validated only), QKV fusion unimplemented, contention visible. No new kernel. A single-GPU microbenchmark shows the attainable rate cheaply. |
| Norm + attention elementwise fusion (#6, #10) | 4 | ~2 s memory-bound, near roofline per kernel; gains need fused norm/GEMM or multi-op kernels. |
| Loss / lm_head | 5 | Block-size tuning of the fused CE; ~0.1 s. |
| FA4 kernel (#5) | 4 | Native SM100 fwd/bwd just landed (+4% each); further gains are upstream kernel work. |
| Optimizer (#9) | 4 | NS GEMMs small; elementwise over stacked state. |
| MoE expert GEMMs (#1) | 3 | QuACK tiles, clusters and varlen-k exhausted (#8317); fp8 excluded by fidelity. |
| Remat (cross-cut, 2.75 s) | 3 | Structural: MoE bwd needs dispatched rows and gate/up outputs (GB per layer); attention/offload variants measured ~0. |
| Latent projections (#8) | 2 | Near the practical GEMM rate. |

### Ranking and fan-out

1. **A: optimizer-state and host-copy exposure** (on-device MuonH state, then carry prefetch).
2. **B: MoE routing marshal** (port inverse routing; then dispatch/combine/mask elementwise).
3. **C: dense GEMM efficiency** (shared-MLP gate/up fusion, contention, QKV fusion, cuBLAS selection).
4. (peer) PGLE on main: another session's `pgle-main-*` runs. Stack at the end, since a PGLE profile
   is program-specific.
5. Queue for re-assignment: norm/elementwise fusion, loss/lm_head, FA4, optimizer, MoE GEMMs, remat.

## M30-002 Agent C: dense GEMMs are power-capped (direction closed, 2026-09-30)

Single-GB200 microbenchmarks of the hero GEMM shapes (same nvjet kernels as training;
`research/mcwitt/mfu30-gemm` @ 964e813f34, logbook `mfu30-gemm.md`): zero operands stay under the cap
(830-940 W, 2062 MHz) at 2.1-2.3 PF/s; realistic operands pin the 1200 W cap, clocks fall to 1200-1600 MHz,
1.44-1.52 PF/s. Operand bit content sets the power. cuBLASLt off changes <2%; Triton GEMMs are 8-20% slower;
four GPUs of a node at once lose another ~6%. Rack W&B: as MFU went 22.3 -> 26.4 -> 28.4%, mean power
rose 1030 -> 1075 -> 1115 W and median SM clock fell ~1900 -> ~1700 -> ~1570 MHz. In the baseline trace, a
GEMM's rate anticorrelates with tensor activity in the prior 50 ms (r = -0.66; -0.25 at fixed shape).
QKV fusion saves 2.4-3.9% of those GEMMs, gate/up fusion 2.3%: ~0.02-0.03 s/step each, below detection.
Contention: the ragged all-to-all's 32 blocks exclude cuBLAS blocks from 32 of 148 SMs (-22%), matching the
overlapped rate. Flag-only QKV merge for the final stack: `--xla_gpu_dot_merger_threshold_mb~448` (~0.02 s).

Consequence: tensor-core time (~9 s/step) only shrinks through fewer FLOPs. The remaining 0.78 s must come
from exposed communication and copies, idle time, memory-bound passes, and recompute. Removing gaps next to
GEMMs may return ~10-15% of the removed time as slower GEMMs (low confidence).

## M30-003 Agent A: XLA remat counts the host-resident carry as device memory (2026-09-30)

With `--xla_gpu_enable_host_memory_offloading` unset (hero default), post-schedule HloRematerialization
charges the 36 GiB pinned-host carry stack `bf16[48,16,4096,6144]{S(5)}` against the backward loop's device
limit (verified in XLA source at PJRT commit 708c3a4ec79c). The baseline HLO has 143 `*.remat*`
instructions, 84 in the backward body (10.6 GiB of recomputed values, incl. a sync `all-gather.127.remat`
per layer), costing 0.63 s/step on the compute stream (0.40 memory-bound fusions, 0.23 the all-gather,
mostly skew wait). Buffer assignment decoded from the xplane: temp arena 67.49 GiB + 35.6 persistent = the
103.09 peak; arena peaks in the optimizer phase. Host optimizer state is 38.55 GiB/GPU (30.4 expert momentum,
5.9 embedding Adam). Full on-device state (H-A1) would peak ~137.7 GiB (needs fraction ~0.78 and slop >100).
PGLE (#9481 t21 trace) hides the first momentum H2D but leaves the three end-of-step D2H (~0.2 s) and the
XLA remat (0.35 s): orthogonal to H-A4.

Arm `m30a-hmo-01` (flag on, profiled, remat VLOG) queued. Prediction: remat count ~0, -0.25 to -0.45 s/step,
peak +5-10 GiB, first-step loss identical. Next: + pipelined carry prefetch; then H-A1.
