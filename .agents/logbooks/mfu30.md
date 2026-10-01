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
  --enable-extra-resources --cpu 2 --memory 8GB --disk 32GB` (no `--timeout`: it counts Kueue queue time;
  `m30a-hmo-01` died gated at 59m57s), env `WANDB_API_KEY`,
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

## M30-004 Agent B: three bitwise-exact MoE marshal changes stacked (2026-09-30)

Branch `research/mcwitt/mfu30-routing` (logbook `mfu30-routing.md`). Each change gates bitwise-equal to main
on GB200x4 (output, drops, grads of x / combine weights / w13 / w2; drops, padding, one-hot, hero shard shapes):
(A) inverse routing `d1ccdd9959` (port of `c2b3a4e8db`: three argsorts -> index scatter), +0.3% per MoE layer;
(B) chained ragged-a2a cotangents `c67ee1f965` (disjoint chunk writes were differentiated as an overwrite:
a [TK,H] select + add + zero fill per layer), A+B +2.4-3.4%, compiler temp -1.37 GB;
(C) unfilled transport buffers `e612b34244` (a Triton kernel that writes nothing, loop-carried so XLA cannot
hoist and copy it; dropped slots get weight 0), A+B+C +4.4-4.8% per layer, +6.1% in a 3-layer rematted scan.
The trace's "mask `or`" ops are the SwiGLU backward packing gate/up pairs (~7 TB/s). Arm `m30b-unfilled-01`
(A+B+C, profiled) queued; predicted -0.2 to -0.25 s/step, loss equal to seed-0 control if the step is deterministic.

**Fidelity ruling (orchestrator, applies to all agents):** "numerically ~equal" admits rounding, precision
and reassociation changes (f32 instead of bf16 intermediates, reduction order, algebraically identical
gradient formulas) provided loss stays in the same-code band against the seed-0 control. It excludes
training-semantics changes: which tokens drop, capacity or chunking that alters drops, quantization.

**HBM budget order** (headroom ~46 GiB at MEM_FRACTION 0.81, ~35 at 0.75; A's analysis): H-A4 ~10 GiB
(~50 ms/GiB), then B's saved latent MoE output ~19 GiB for a SonicMoE-style backward (~25-30 ms/GiB,
unmeasured), then H-A1 on-device optimizer state 34.6 GiB (~8 ms/GiB) only if room remains.

## M30-005 Same-code noise and in-run drift (2026-09-30)

`gcab-control-20260930` vs `gcab-freeze-20260930` (gc.freeze is host-side, so the step is same-code for MFU):
360-step medians 28.312 vs 28.342; 55-step block medians agree within 0.02-0.05 (28.525/28.541,
28.470/28.503, 28.395/28.409, 28.290/28.336, 28.177/28.203, 28.117/28.151). Both drift down ~0.4 MFU over
330 steps. Consequences for the protocol: (1) compare arms only over matched step windows; the screening
window (restore +5..+59) reads ~0.2 above a long-run median, so main reads ~28.5 there; (2) run-to-run sd
in a matched window is ~0.03, so the 0.15 keep bar is conservative; (3) the goal claim needs a long
confirmation run (>= 200 steps) whose median is >= 30%, not an early-window screen.

B's SonicMoE-style backward (M30B-009, approved, building): dS = rowsum(dh ⊙ h)/w on the expert side and
the latent MoE output saved on device (18.0 GiB), so remat drops the recomputed down GEMM (0.279 s),
return all-to-all exposure (0.18 s) and combine gather-sum; priced -0.40 to -0.50 s/step. Forward, dx,
dW13, dW2 unchanged; dS changes at rounding level. Chunk barrier re-tied from `returned` to h.

## M30-006 Agent C: non-MoE memory passes (2026-09-30)

Single-GPU hero-layer benchmark (real `Block.__call__`, stand-in routed experts, fwd+bwd with remat;
`m30c-blockbench-02..04`; branch `research/mcwitt/mfu30-gemm` @ 8e5ed5b546, entry M30C-005):
- Q/K/V and shared gate/up projection fusion: rejected. Forward bitwise-equal, but XLA's backward gradient
  concatenation costs +1.6-2.5 ms/layer against a 1.0-1.2 ms saving; net +1.5-2.5 ms/layer for both.
  The dot-merger flag noted in M30-002 is dropped with it.
- `--xla_gpu_enable_triton_gemm=false`: -0.85 ms/layer (3 pairs, sd ~0.3). The GatedNorm sigmoid pass
  merges into the following multiply, and gradient sums become cuBLAS in-place epilogues; ~1 ms returns as
  slower GEMM/FA4 time (power cap). ~0.04 s/step: goes to the final stacked run.
- Small items priced below a rack slot: RMSNorm weight-grad reduction (<=0.04), RoPE flipped copy
  (0.02-0.03), XSA reductions (<0.02). Short-conv backward (6.7 passes vs a 3-pass floor) ~0.08 s but needs
  CUDA/CuTe.
- Approved: fused RMSNorm+GatedNorm forward Pallas-Triton kernel (the rank-128 GEMMs sit between the norm
  and the gating multiply, so XLA can't fuse; ~1.5 ms per call vs ~0.4 at roofline; 4 calls per layer),
  ~0.2 s/step, no extra peak. The backward elementwise fusion (~0.1) is optional after that.

## M30-007 Seed-0 control at the pinned checkpoint (2026-09-30)

`mhep-ctx4k-s0-20260930` (main f38da1173d, step-180000 restore, 100 steps): median MFU **28.255** over
180005-180059 (28.256 over 180005-180099), duration median 13.892 s, peak 103.09 GiB. Loss 180000
1.261413, 180001 1.234596, 180002 1.200221, 180003 1.256785. Steps 180000-180010 are a post-restore warmup
(20.6-27.8%); steady state from 180011 is 28.1-28.3 with isolated dips (180013, 180025, 180057). There is
no in-run drift here, unlike the from-step-0 gcab runs (which also read higher, 28.3-28.5). **Gap to 30%
at this checkpoint: 13.892 -> 13.084 s = -0.81 s/step.** Scoring window changed to 180011-180059 (median;
the warmup only adds noise).

## M30-008 Agent C: fused RMSNorm+GatedNorm forward kernel (2026-09-30)

Pallas-Triton kernel `lib/levanter/src/levanter/kernels/pallas/gated_rms_norm/` with a custom_vjp backward
that never materializes y; switch `GrugModelConfig.gated_norm_implementation` /
`launch_diagnostics --gated-norm-implementation pallas_gpu`; parameters unchanged. GB200 correctness at
[16,4096,6144] rank-128 plus padded/odd shapes: output and x / norm-weight / w_down / w_up gradients have
the same error vs f32 as the bf16 XLA reference (ratio 0.99-1.03); max diff 1 bf16 ulp. Temp memory -1.5
GiB per layer. Speed is capped by Pallas-Triton at 2.8-3.2 TB/s: isolated forward 1.046 ms vs XLA 1.21
(1.005 with triton_gemm off). Block benchmark per layer fwd+bwd: flag -1.14 ms, kernel -1.27, kernel+flag
-1.77 (not additive; both remove the sigmoid pass), dot merger +1.07 (rejected). Arm `m30c-grnflag-01`
(kernel + `--xla_gpu_enable_triton_gemm=false`, source 37e3db7b87) queued; predicted -0.06 to -0.09
s/step. Loss will differ at rounding level, so it needs a C-C rerun. Further kernel work (CuTe TMA
forward ~0.1 s, fused backward ~0.1, short-conv backward ~0.08) is deferred. C now owns stacking + PGLE:
branch `research/mcwitt/mfu30-stack`, PR #9481 conflicts with B's transport changes, trace -> profile ->
scored rerun.

## M30-009 Stack branch and PGLE tooling (agent C, 2026-09-30)

`research/mcwitt/mfu30-stack` @ 423e8c50e4 (worktree `~/projects/marin.mfu30-stack`, logbook
`mfu30-stack.md`): campaign base + B's three bitwise-exact commits (always on) + C's norm kernel (switch)
+ PR #9481's six commits. #9481's two transport commits (pipelined expert chunks; mirror transpose
parameters that remove the offset all-to-alls) conflicted with B's chunk-loop / `_ragged_a2a` VJP rewrite
and were ported by hand. B's GB200x4 gate passes on the stack (`m30c-stackgate-02`, bitwise-equal to main
in all six cases). 3-layer rematted scan: main 84.4 ms, stack 77.0 ms (+9.6%; B alone +6.1%). Env
switches for A's flag, C's triton_gemm flag and PGLE go through `stack/arm.sh --xla`. `pgle_build.sh
<trace-run>` builds the profile from the rank-0 xplane. Open: B's SonicMoE backward replaces the chunk loop,
so the #9481 transport ports must be re-folded into it (B to decide). #9481's attention re-gather may be
redundant with A's H-A4 flag.

## M30-010 Agent B: SonicMoE-style backward "D" built and gated (2026-09-30)

`research/mcwitt/mfu30-routing` @ ce112504f1 (entries M30B-010..012). One custom_vjp from dispatch to combine;
the backward returns s = <h, dh> per expert row (fused into the SwiGLU-backward pass), one [C,1] f32
all-to-all per chunk returns it, dS = s/w in f32, dropped/padding slots get 0 via `where`. The offload_carry
policy saves the routed MoE output (`grug_moe_routed_output`, before W_up, 18 GiB). The chunk barrier ties
to the previous chunk's MLP residuals. GB200x4 gate (`m30b-gate-sonic-02`): out/drops/dx/dW13/dW2
value-equal to main in all six cases; dS max 0.6% of the largest gradient, median ~1 ulp. 3-layer
rematted scan at hero per-shard shapes: recompute drops both down-projection QuACK GEMMs, both return
all-to-alls and the combine gather-sum; step 338.8 (main) / 326.0 (A+B+C) / 290.7 ms (A+B+C+D); D alone
-11.8 ms/layer (~-0.57 s/step extrapolated); compiler temp 35.30 / 32.08 / 31.92 GB.
Hazards: D is exact and repeatable only at collective overlap limit 1 (its backward all-to-alls no longer
depend on the recompute's; B forces limit 1 for every ragged run). dS = 0 for an accepted assignment with
weight exactly 0 (sigmoid < 1e-38). D needs A's H-A4 flag: remat already binds on the host-carry mis-count.
Decisions: D arm runs with the H-A4 flag and doubles as the multi-step smoke (loss at 180000 must equal
1.261413); queued now as a second B job (exception to the one-job rule, queue ~5 h). B folds #9481's
mirror parameters and pipelined chunks into `_routed_experts` for the stack; C reviews.

## M30-011 Closed: loss/lm_head, optimizer elementwise, cheap norm-kernel variants (agent C, 2026-09-30)

CE at the hero shape [65536, 6144] x 128256 on GB200x4 (`m30c-ce-01`): production tiles 331-335 ms
fwd+bwd, best larger vocab tiles 320 ms: <= 0.012 s/step. The trace CE is 0.303 s of power-capped GEMMs +
0.03 s elementwise at 6-7 TB/s. Optimizer: 0.19 of its ~0.29 s is QuACK's symmetric Newton-Schulz GEMM
(power-bound); the ~0.1 s elementwise part already runs at 7 TB/s. Raw-Triton rewrite of the fused norm
forward (`m30c-grntriton-01/02`) matches Pallas-Triton. The in-kernel [BT,128]x[128,BD] dot is the limit
(elementwise body alone 0.51 ms at 6.2 TB/s; with the dot >= 0.76 ms); the cheapest re-split is worth
~0.02 s/step. Remaining option: a K=128 CuTe/QuACK GEMM with norm and gate in its epilogue, ~0.1 s/step,
1-2 days, held unless the stack falls short. Queue remainder: SwiGLU dswiglu epilogue (B), short-conv
backward (~0.08, CUDA/CuTe), device idle 0.35 s (unanalyzed), CuTe norm epilogue (~0.1).

## M30-012 Agent B: #9481 fold into D, and E (SwiGLU backward in the dh epilogue) (2026-09-30)

3-layer rematted scan, hero per-shard shapes, GB200x4 (ms/step): main 345.4; A+B+C 332.1; A+B+C + #9481
(C's port) 323.6; D 298.6; D + mirror params 299.0; D + mirror + pipelined chunks 307.4. Pipelining works
inside D (the recompute still drops the return and down GEMM, grads exact) but costs 8.7 ms on 4 GPUs,
so it is dropped (kept in a1d699e67e). Mirror params are neutral on 4 GPUs but remove two offset
all-to-alls per transport, so they are kept: 6a6bb78853 (gate `m30b-gate-mirror-01`: out/drops/dx/dW13/dW2
equal to main, dS within 1 ulp).
E (12643e682c = D + mirror + E): QuACK's grouped dh GEMM runs `dswiglu` in a custom packed-CD epilogue
plus the column reduction for D's <h, dh>. dh is never written and the XLA SwiGLU-backward pass is gone.
One chunk on one GB200: 3.54 vs 5.10 ms. Gate `m30b-gate-epi-02`: out/drops/dW2 equal; dx/dW13/dS ~1 ulp
(median 0.49-0.56%, max <= 0.82%; fp32 SwiGLU backward on the unrounded accumulator). Scan: D 298.7 ->
D+mirror+E 290.3 (-2.8 ms/layer, ~-0.13 s/step).
Arms queued: `m30b-unfilled-02` (A+B+C), `m30b-sonic-01` (D + H-A4 flag, remat VLOG, doubles as smoke).
Next: C integrates 12643e682c into the stack (plus #9481's model commits, pgle_profile, norm kernel)
and queues the stacked trace arm now.

## M30-013 Agent B: idle and exposed-collective classification of the baseline (2026-09-30)

Tool `autoresearch/loop-260930-mfu30/b/exposure.py` (per-instance time above the fastest instance of the
same instruction ~ waiting). Device idle 0.346 s/step: 0.323 step tail (last kernel -> next launch;
outside `throughput/duration`), 0.007 dispatch-to-first-kernel, 0.017 internal gaps (all < 50 us), so it
is not an MFU lever. Exposed collectives 1.681 s/step: chunk-0 ragged transports 0.669 (fwd dispatch
0.183, fwd return 0.153, recompute dispatch 0.153, recompute return 0.181; min ~ median, so this is
transfer, not skew). The scheduler spends each layer's ~10 ms of shared-expert GEMMs on chunk 1 and leaves
chunk 0's ~6.9 ms bare; D removes the recompute return; pipelining (#9481) targets the rest. Rank-skew
waits ~0.40 (u32 drop-count all-reduce 0.241, QB pmin/pmax 0.04, recompute group-size all-gather 0.048, norm
all-reduce 0.019): structural. XLA remat clones 0.246 (A's flag). FSDP gathers + gradient reduce-scatter
~0.27, mostly waits (PGLE). Offset all-to-alls 0.010 (mirror params). Pipelined D variant
`research/mcwitt/mfu30-routing-pipelined` @ 64909b24d0 (values identical). C to queue two stacked trace
arms, pipelined first, then sequential; the winner gets the PGLE build.

## M30-014 Stacked trace arms queued; fidelity criterion refined (2026-09-30)

C's stack = B's 12643e682c (D + mirror + E; `_moe/` identical) + campaign merge + fused norm + #9481's
attention re-gather, MLP-weight prefetch, QB-after-MLP and `pgle_profile.py`. Arms (hero program with
`--xla_gpu_enable_host_memory_offloading=true --xla_gpu_enable_triton_gemm=false`,
`--gated-norm-implementation pallas_gpu`, remat VLOG, profiled 180021-180023):
`m30c-stackpipe-trace-01` (`research/mcwitt/mfu30-stack-pipelined` @ 7393a9ae26) and
`m30c-stackseq-trace-01` (`research/mcwitt/mfu30-stack` @ 4ee7986fb4). Gates `m30c-stackgate-03` /
`-pipe-01`: B's module gate bitwise (d_weights max 0.4%); scan main 339 / D 292.5 / sequential 289.9 /
pipelined 296.0 ms (no shared experts in the scan). C's review of `_routed_experts`: no bugs. Correctness
depends on overlap limit 1 (add an assertion for landing); the backward has no inter-chunk barrier, so
both chunks' [C,H] cotangent buffers (~1.9 GB each) may be live together; watch the EP64 peak.

**Fidelity criterion, refined.** Rounding-level forward changes (fused norm, triton_gemm off) flip
near-tied top-k routing decisions: in a 4-layer EP4 bf16 CPU smoke the routed-MoE gradient leaves move ~6%
relative (every leaf within 1e-4 in f32; loss 7.657223 vs 7.657212). A same-code comparison is
expected to be bitwise (B's exact arms test this), so the C-C band is ~0 and cannot judge rounding changes.
The reference for "numerically ~equal" is therefore a **rounding-only perturbation band**: the loss
divergence of `m30c-grnflag-01` (fused norm + triton_gemm off, no algorithmic change) vs the seed-0
control. A candidate passes if its pointwise loss divergence from the control over 180000-180059 is within
that band (max |d| and mean d, no larger one-signed drift), and drops / balancing loss stay in family.
Exact-equality smoke checks (loss at 180000 == 1.261413) apply only to bitwise-forward arms (A's flag,
B's A+B+C and D).

## M30-015 Agent B: Triton short-conv kernel (2026-09-30)

`sconv_implementation=triton_gpu` (raw Triton via jax_triton; per-chunk register carry of 3 rows +
segment ids; fp32 dw partials summed by the API; inline PTX `mul.rn/add.rn.bf16x2` to stop LLVM contracting
into bf16 FMA). Tuned tiles (commit 5d64137a67): forward chunk 32 / 1024 channels / 4 warps / 8 rows;
backward chunk 128 / 256 / 4 / 8 (register-bound). All 112 sweep rows are bitwise on out and dx. Per call
(ms, Pallas -> Triton, floor): [16,4096,6144] fwd 0.437 -> 0.262 (0.242), bwd 1.065 -> 0.480 (0.350);
[16,4096,1536] fwd 0.119 -> 0.082 (0.073), bwd 0.295 -> 0.151 (0.099). Hero Block on one GB200
(`m30b-sconv-block-02`): short-conv time/layer 4.39 -> 2.20 ms, block step 121.9 -> 120.3 ms, temp
-0.6 GiB, loss bitwise; ~-0.105 s/step estimated. Determinism: Pallas-vs-Triton gradient differences
match Pallas-vs-Pallas rerun differences (~1e-5 rel-rms in attention weights, x, sconv_k), so the
**attention backward is run-to-run nondeterministic** and same-code hero runs are not bitwise. The only
Triton-specific difference is sconv dw at 1.7e-7 (summation order). Patch for the stack:
`b/sconv_on_stack.patch` (9303d20b97); C adds it behind the switch for the final program.

## M30-016 Control replicate; independent scorer (2026-09-30)

`score_arm.py` (median MFU over 180011-180059 minus profiled 180021-180023, peak, drops, pointwise loss vs
the seed-0 control). Controls: s0 28.258 / 13.891 s, s1 28.229 / 13.905 s (same code and checkpoint,
different data order: s1 loss at 180000 is 1.2063 vs 1.2614), so run-to-run MFU spread is ~0.03. Only
seed 0 is a same-data loss reference. Short conv cleared its final GB200 check (`m30b-sconv-03`, 8/8,
out/dx bitwise); it is on both stack branches behind `--sconv-implementation triton_gpu`
(`mfu30-stack` @ cc55c78f45, `-pipelined` @ a7cc657e28), default off.

## M30-017 Control spread; same-code repeat queued (2026-10-01)

Third control `mhep-ctx4k-s2-20260930`: 28.012 / 14.013 s (s0 28.258, s1 28.229). The 0.25 MFU spread
across seeds confounds data order (routing balance) with rack conditions. All campaign arms use seed 0,
so s0 is their same-data reference. A same-code, same-seed repeat of s0 (`m30-ctl-s0-r2`, main
code + logbook only, identical flags to the ctx4k runs, 100 steps) is queued behind the campaign arms. It
calibrates (1) the same-code loss band (attention-backward nondeterminism only) and (2) MFU drift across
the night. Keep bar is provisionally max(0.15, 3 x sd of seed-0 repeats) once the repeat exists.
The resurrected `gcab-freeze` rerun (another session's job, flagged to the user) took the rack at 00:39Z,
ahead of our arms.

## M30-018 First rack results; D arms held for memory settings (2026-10-01)

Scored over 180011-180059 minus 180021-180023 against `mhep-ctx4k-s0` (28.258 / 13.891 s, peak 103.09):

| arm | change | MFU | s/step | dMFU | peak GiB | loss@180000 | dloss max / late mean / late positive |
|---|---|---|---|---|---|---|---|
| m30a-hmo-02 | H-A4 `--xla_gpu_enable_host_memory_offloading=true` | 28.580 | 13.734 | +0.32 | 104.07 | exact | 4.4e-4 / +1.6e-4 / 48 of 49 |
| m30b-unfilled-02 | B's A+B+C (bitwise forward) | 28.668 | 13.692 | +0.41 | 103.75 | exact | 9.6e-4 / +1.8e-4 / 34 of 49 |

A's analysis (M30A-012): zero "Remat via offload"; remat ran only in main (7 instructions; baseline 143);
`all-gather.127.remat` is gone; arena +0.98 GiB. XLA-remat kernel time 0.633 -> 0.0007 s, but the sync
all-gather clone's 0.227 s was skew wait that moved to the backward latent reduce-scatter (0.051 -> 0.255 s
exposed), and 0.111 s of the fusion clones ran under collectives. Remat limit = (pool - persistent) x slop =
(138.22 - 35.09) x 0.85 = 87.66 GiB; main peaks at 89.94, so remat binds. #9481's attention re-gather
looks redundant with the flag.
**Consequence:** D keeps ~18 GiB live across the backward, so at slop 85 XLA remat would cut ~18 GiB and
likely erase D's gain. The same slop sizes the LHS arena, which must stay under the pool (release-
threshold hazard). The three D arms queued at slop 85 (`m30b-sonic-01`, `m30c-stack{pipe,seq}-trace-01`)
were cancelled before they started. A is computing a (MEM_FRACTION, slop) pair; B and C resubmit with it.
Memory settings for D-containing arms (A, fitted to hmo-02's logs): `XLA_PYTHON_CLIENT_MEM_FRACTION=0.78`,
`--xla_gpu_memory_limit_slop_factor=105`. Pool P = fraction x 184.3 = 143.76; remat/LHS limit
L = (P - 35.09) x slop = 114.1 GiB vs D's estimated remat view ~104-106. Remat's view includes the 18.9 GiB
of S(1) collective buffers, so the arena is capped at L - 18.9 = 95.2. Worst-case pool use is 130.8 (13
under the pool); outside the pool 40.5 GiB vs ~28.5 needed (0.83 is the documented failure point).
Verify per arm: "Rematerialized N instructions" <~ 10, "Peak memory for main" <= ~110 (else slop 110),
memory/limit_gib 143.76, memory/peak_gib < ~139. D arms compare only against arms at the same settings;
the goal comparison is the full stack (settings included) vs main at defaults.

## M30-019 Exposure on the current-main arms (agent B, 2026-10-01)

Rank 0, steps 180021-180023 (s/step): span / compute / exposed collectives / exposed copies / idle =
Sep 24 main 14.439 / 11.793 / 1.681 / 0.619 / 0.346; hmo-02 13.766 / 11.627 / 1.443 / 0.618 / 0.077;
unfilled-02 13.762 / 11.533 / 1.516 / 0.631 / 0.082. unfilled-02's gain is routing marshal compute
(-0.37 vs Sep 24; `moe_expert_elementwise` 0.372 -> 0.136, backward -0.23). Ragged a2a exposure is unchanged
(0.838 vs 0.804); the four chunk-0 transports still expose 0.66. In both current-main arms the u32
drop-count all-reduce wait moved to the latent backward `reduce-scatter.18` (0.058 -> 0.305 exposed;
per-instance median 0.18 -> 1.58 ms): ~0.3 s of cross-rank skew. Exposed copies are flat at ~0.63
(carry reload H2D 0.218; end-of-step optimizer-state D2H ~0.21; ~0.37 in the last tenth of the step).
`m30b-sonic-02` (D + H-A4 at 0.78/105) is queued; A is pricing an ordering/split of the optimizer update
that would hide the end-of-step D2H without holding state on device.

## M30-020 m30c-grnflag-01 regresses at EP64 (2026-10-01)

Fused RMSNorm+GatedNorm kernel + `--xla_gpu_enable_triton_gemm=false` (C's arm): steady state ~28.03 vs control
~28.27 on every post-warmup step, i.e. **-0.24 MFU (~+0.12 s/step)**. Interim median 28.021 / 14.008 s (n=40).
memory/peak_gib 117.4-117.7 vs 102.7-103.1 (+14.6 GiB) from step 180000. The single-GPU block benchmark's
-1.77 ms/layer did not transfer: at slop 85 remat binds, so extra memory pressure plausibly turns into
recompute. Rounding band from this arm: loss at 180000 1.2614125 vs 1.2614135; dloss max 3.0e-4, late mean
-3.4e-5, 15/43 positive (balanced). Both bitwise-forward arms instead drifted one-signed (~+1.7e-4 late
mean), which the same-code repeat `m30-ctl-s0-r2` will explain. Stack arms `-02` (which carried both
components) were cancelled before they started; C resubmits `-03` without them and diagnoses the regression.

## M30-021 grnflag-01 regression diagnosed (agent C, 2026-10-01)

Final score 28.023 vs 28.261 (-0.238), 14.007 s (+0.118), peak 117.70 (+14.6), loss within rounding band
(max 3.0e-4, mean -1.9e-5 over 60 steps). Two side effects, neither of them kernel speed:
(1) `--xla_gpu_enable_triton_gemm=false` turns 120 optimizer (Newton-Schulz) GEMMs into cuBLAS calls. The
fallback scheduler costs them at 1000 vs 1 units and reorders the optimizer phase, so the 10.1 GiB
expert-momentum H2D (`copy-start.44`) moves from after the backward to the step start and stays live through the
backward peak: arena 68.5 -> 82.1 GiB, remat kernels 0.633 -> 0.868 s/step. The flag is dropped.
(2) **Memcpy-stream collision**: per layer the compute stream idles ~3.3 ms on weight-slice copies that XLA's
round-robin put on the same memcpy stream as the 4.4 ms forward carry D2H (+0.165 s/step). Either change can
trigger it, and #9481's PGLE trace t21 shows the same stall (0.192 s/step), so it is this collision and not
PGLE's latency model. Any program change, PGLE included, re-draws the assignment. Checks for every trace:
`stack/carry_stall.py` (healthy < ~10 ms/step) and `stack/copy_schedule.py` (copy-start.44 after the
backward). A is making the stall structurally impossible (stream pinning or keeping slices off the memcpy
streams). The fused-norm kernel may return as a post-lineage add-on if its trace passes both checks.
Stack trace arms `-03` (no triton_gemm flag, no fused norm): pipelined @ fcb44f6920, sequential @ a1f483afed.

## M30-022 Stream-collision root cause and XLA patch (agent A, 2026-10-01)

XLA at 708c3a4ec79c: `DynamicSliceCopyFusionAsyncWrapper` makes every dynamic-slice/DUS copy fusion
async, covering both the per-layer stacked-weight slices and the carry DUS-to-host / DS-from-host. Its only
off switch (`xla_gpu_experimental_dynamic_slice_fusion_verify_offsets`) de-asyncs everything and adds runtime
checks. `ExecutionStreamAssignment` round-robins every compute-scope async start over a constant 4
(`kDefaultNumComputeStreams`; `xla_gpu_executable_num_compute_streams` changes allocation, not the
modulus), so a body's stream sharing is post-order position mod 4 and any body change redraws it.
`_xla_stream_annotation` cannot be attached from JAX, because the async wrapper does not copy frontend
attributes. Patch (branch `mcwitt/adhoc-host-transfer-streams`, 283d5b6d98, +74 lines in
`execution_stream_assignment.cc`): with `XLA_GPU_HOST_TRANSFER_STREAMS=1`, async starts touching S(5) go
to dedicated streams (H2D -> 4, D2H -> 5). Zero runtime cost, ordering unchanged, bit-identical when unset.
Approved: branch push to marin-community/xla (no PR or release), wheel built on the cluster with
`pjrt_build_job.sh` and the lock's cuDNN headers (not `marin-pjrt.yaml`, which creates a prerelease). A/B on
the same wheel with the env var on vs off; gate `carry_stall.py` < 10 ms/step. JAX-only fallback (an
integer-zero dependency from slices to carry, ~25 ms/step always paid) is held.

## M30-023 Same-code repeat: noise and loss band (2026-10-01)

`m30-ctl-s0-r2` (main, same flags and seed as `mhep-ctx4k-s0`): MFU 28.235 vs 28.261 over 180011-180059
(28.241 over 180011-180099), duration 13.902 vs 13.889, peak 103.09, loss at 180000 identical. Same-code
loss divergence (attention-backward nondeterminism): first steps 0, -3.6e-6, -1.3e-5, +6.4e-6; max |d|
3.3e-4; late mean -3.9e-5; 35/89 positive (balanced).

| arm vs s0 | max abs d | late mean d | late positive |
|---|---|---|---|
| m30-ctl-s0-r2 (same code) | 3.3e-4 | -3.9e-5 | 35/89 |
| m30c-grnflag-01 (rounding changes) | 3.0e-4 | -3.4e-5 | 15/43 |
| m30a-hmo-02 (remat placement only) | 4.4e-4 | +1.6e-4 | 48/49 |
| m30b-unfilled-02 (bitwise forward, reorganized backward) | 9.6e-4 | +1.8e-4 | 34/49 |

The two bitwise-forward arms drift one-signed (~+1.7e-4), and unfilled-02's max is ~3x the same-code
max. Neither change alters the math, so a systematic bias is implausible: within one pair, chaotic
divergence shares its sign across consecutive steps (prior hero cutovers showed the same relaunch
signature). One pair per arm cannot separate the two. The final program therefore gets a longer
replicated loss check: >= 2 paired runs of >= 100 steps vs same-code controls, judged against the C-C
spread.

## M30-024 Stream-patch wheel built; use on hold pending user approval (2026-10-01)

A built `jax_cuda13_pjrt-0.11.1+marin.283d5b6d98cd` on the cluster (`/mwittmann/m30a-pjrt-build-01`, 15 min;
s3://marin-us-east-02a/marin/research/mcwitt-mfu30/pjrt/283d5b6d98cd/), from branch
`mcwitt/adhoc-host-transfer-streams` pushed to marin-community/xla (branch only: no PR, no release, no CI
dispatch). Build pins: cuDNN headers 9.19.0.56 (uv.lock), `HERMETIC_NCCL_VERSION=2.30.7` (matches the fork's
production build; the copied script had left jax's default). GB200x1 smoke: with the production wheel the
carry D2H shares stream 2 with four weight slices (collision reproduced); with the new wheel and the env var
unset, assignment is identical to production; with `XLA_GPU_HOST_TRANSFER_STREAMS=1`, the carry goes to
streams 4/5 and the slices to 0-3, with bitwise-equal gradients. The `--pip-package` launcher port is
`cf5bc74409` on `research/mcwitt/mfu30-offload`.
**On hold:** running campaign jobs on this custom wheel needs the user's approval (asked 2026-10-01). The
branch push was approved by the orchestrator under an earlier campaign's clarification and is disclosed
to the user. Until then the final program uses the production wheel, `carry_stall.py` is the gate, and A's
JAX-only fallback (~25 ms/step always paid) is the alternative if a final trace shows the stall.

## M30-025 m30b-sonic-02: 29.78% (2026-10-01)

D (dd45f27c17: A+B+C + SonicMoE-style backward, no #9481 commits, no sconv) + H-A4 flag at
MEM_FRACTION 0.78 / slop 105, remat VLOG, profiled 180021-180023: **MFU 29.783 / 13.179 s** vs control 28.258 /
13.891 (+1.52 MFU, -0.712 s/step); steady state 29.70-29.85 with two dips (180010 25.97, 180056 27.89).
memory/limit_gib 143.75 (fraction took effect), peak 125.02. Loss at 180000 exact; dloss max 2.4e-4,
late mean -5.7e-6, 22/49 positive: inside the same-code band. Rough decomposition vs one-draw arms:
unfilled-02 (A+B+C) +0.41, hmo-02 (flag) +0.32, so D plus the memory settings is ~+0.8, sub-additivity aside.
Remaining to 30.0%: -0.095 s/step. Candidates still to land: stack -03 arms (mirror + E + #9481's model commits;
pipelined vs sequential), sconv (~-0.1 est.), PGLE with A's D2H patch (~-0.2 est.).

## M30-026 sonic-02 trace (agent B, M30B-023)

"Rematerialized 0 instructions" on all 64 processes; main peaks at 105.08 GiB vs the 114.1 limit (9 GiB
headroom, both logged values offset by 73.64 GiB of host-space buffers). Rank 0, 180021-180023:
span 13.229 (unfilled-02 13.762), compute 11.188 (11.533), exposed collectives 1.336 (1.516), of which
ragged a2a 0.736 (0.838), exposed copies 0.622, idle 0.084. Compute: D removes the recomputed down GEMM
(-0.263) and the recomputed return/combine (-0.093); the flag removes the XLA remat clones (norms -0.083,
attention elementwise -0.087, shared-MLP recompute -0.120); D's expert-side backward adds ~+0.16; attention
is +0.08 in every phase (unexplained). Remaining ragged exposure per step: forward dispatch c0 0.144
(the only exposed forward transport); recomputed dispatch c0 0.254 (0.111 of it rank-skew wait); recomputed
dispatch c1 0.169 (held behind the c0 recompute by the chunk barrier); reverse return of dy c1 0.149 (issued
with nothing on the compute stream). Row-dot all-to-alls and dy c0's reverse return are covered.
`carry_stall.py` 2.8 ms/step (healthy; unfilled-02 15.6); `copy_schedule.py` healthy. Exposed copies
0.62 (carry reloads 0.219, optimizer D2H 0.211, H2D 44 0.064). Levers: pipelining (forward c0 + recompute
c1, <= ~0.31), backward reorder of dy-c1's reverse return (~0.15), PGLE for FSDP gathers (~0.27) and the
optimizer D2H (A's patch, ~0.2). Skew waits (~0.41) are not schedulable.

## M30-027 User approves the custom wheel (2026-10-01)

The user approved using `jax_cuda13_pjrt-0.11.1+marin.283d5b6d98cd` (A's stream patch) in campaign rack jobs
and keeping branch `mcwitt/adhoc-host-transfer-streams` on marin-community/xla. Plan: after the -03
lineage pick, F1 = winner + wheel + `XLA_GPU_HOST_TRANSFER_STREAMS=1` (traced; carry_stall gate; source for
the PGLE build) and F0 = the same wheel without the env var (isolates the build environment from the
production wheel). Wheel use goes through `--pip-package` (A's cf5bc74409 cherry-picked onto the stack).
Accounting answer to the user: #9481's three model commits, its transport ideas (via B's D: mirror params in
both lineages, pipelined chunks in one) and a regenerated PGLE profile are in the final program; none
of the scored arms so far contain #9481 code. #9374 is excluded (user-parked; measured +0.09% mean).
B's forward-order backward (547bf2ad20 sequential / 2cc470d88f pipelined) gates bitwise and is a paired add-on.

## M30-028 F1 arms on the custom wheel queued (2026-10-01)

C's sandbox denied the wheel plumbing cherry-pick (approval relayed through the orchestrator does not count as
the user's own), so the orchestrator owns all custom-wheel arms. Branches: `research/mcwitt/mfu30-final-seq`
(d4234c88e7 = stack d10320ca2c incl. B's forward-order backward + A's `--pip-package` plumbing cf5bc74409)
and `research/mcwitt/mfu30-final-pipe` (743b5826d2 = 0c30e9dd9a + cf5bc74409). Queued (traced 180021-180023,
H-A4 + slop 105, MEM_FRACTION 0.78, `XLA_GPU_HOST_TRANSFER_STREAMS=1`, `--sconv-implementation triton_gpu
--regather-attention-weights`, wheel via `--pip-package`): `m30-f1-pipe-01` (33010), `m30-f1-seq-01` (33011).
The -03 lineage pick decides which F1 runs (the other is cancelled if it has not started); then F0 (same
wheel, env off) on the winner, the PGLE build from the winning F1 trace (C), and PGLE-scored runs (orchestrator).

## M30-029 Stacked pipelined arm: no net gain over D (2026-10-01)

`m30c-stackpipe-trace-03` (D + mirror + E + #9481 model commits incl. re-gather + Triton sconv + pipelined chunks;
H-A4 at 0.78/105; production wheel): **29.768 / 13.186 s** vs sonic-02 29.783 / 13.179, so net ~0 for
everything added on top of D. Peak 123.20 (vs 125.02). Loss at 180000 exact; dloss max 6.8e-4, late mean
+7.7e-5, 38/49 positive (above the same-code max 3.3e-4). B is attributing the delta from the two traces:
which added component eats the expected E (~-0.13) and sconv (~-0.1) gains.

## M30-030 stackpipe-03 attributed; final lineage = sequential, no re-gather (2026-10-01)

B (M30B-025), comparing profiled steps 1 and 3 (step 2 had a rank stall) of sonic-02 vs stackpipe-03 (s/step):
compute 11.211 -> 10.820 (-0.391), exposed collectives 1.324 -> 1.527 (+0.203), exposed copies 0.620 -> 0.774
(+0.154), idle ~0. Compute: E ~-0.15 (`moe_expert_elementwise` -0.198, +0.05 epilogue cost); Triton sconv
-0.110 (0.218 -> 0.108 s kernel); shared-expert GEMMs freed from contention -0.132 vs expert GEMMs under
pipelined transports +0.081; attention -0.05; router/dispatch/shared elementwise -0.07. Exposure: pipelining
left forward return c1 bare (0.021 -> 0.188) and did not move recomputed dispatch c1 or dy-c1's reverse
return in the backward. #9481 collectives net ~+0.01 (MLP-weight prefetch -0.052; re-gather +0.093 forward
gathers vs -0.061 old FSDP gathers, +0.034 remat_carry gathers). **Carry D2H stall lost the draw: carry_stall
147 ms/step** (sonic-02 2.8), +0.15 s/step. Loss drift (max 6.8e-4) attributed to E's 1-ulp dx/dW13 change.
Decision: sequential chunks, keep mirror + E + sconv + MLP-weight prefetch + QB-after-MLP + B's forward order,
drop the re-gather, add the stream fix. Cancelled `m30-f1-{pipe,seq}-01`; queued `m30-f1-seq-02`
(streams on) and `m30-f0-seq-01` (streams off), both from `research/mcwitt/mfu30-final-seq` @ d4234c88e7,
custom wheel, traced. Expected F1 if the attribution holds: ~13.18 - 0.39 = ~12.8 s (~30.7%).

## M30-031 stackseq-03 attribution (agent B, M30B-026)

Profiled steps 1 and 3, spans without the tail (s/step): sonic-02 13.177 / stackpipe-03 13.144 / stackseq-03
13.087; compute 11.211 / 10.820 / 10.804; exposed collectives 1.324 / 1.527 / 1.491, of which ragged
0.732 / 0.947 / 0.838; exposed copies 0.620 / 0.774 / 0.769. Ragged exposure, sonic-02 / pipe / seq:
forward dispatch c0 0.141 / 0.179 / 0.187; forward return c1 0.021 / 0.188 / 0.184; backward recomputed
dispatch c0 0.254 / 0.280 / 0.292; backward dy-c1 reverse return 0.149 / 0.149 / **0** (sequential put it under
the c0 recompute); backward recomputed dispatch c1 0.167 / 0.150 / 0.167.
(1) In both -03 arms the shared-expert forward GEMMs run after the routed MoE, interleaved with the QB
collectives that #9481's QB-after-MLP commit put behind the MoE output. Forward return c1 is bare (+0.16),
expert GEMMs under the transports +0.12, shared GEMMs alone -0.105. Likely trigger: QB-after-MLP (being
isolated by compile on GB200x4). (2) Carry stall hit again: 141 ms/step (production wheel). (3) Sequential
beats pipelined by 0.109 s of ragged exposure. (4) Compute matches stackpipe-03 (E -0.15, sconv -0.11).
(5) Re-gather nets ~+0.1 s/step (dropped in F1). If QB-after-MLP is confirmed, the final program drops it too
(variant `research/mcwitt/mfu30-final-seq-noqb`).

## M30-032 QB-after-MLP: rack A/B queued (2026-10-01)

B's GB200x4 compile (`m30b-sched-{final,noqb}-01`) could not isolate it: at EP4 both branches put the
shared-expert GEMMs under both returns, unlike the EP64 rack placement (shared GEMMs after the MoE, among the
QB collectives). B's value check is pending (`m30b-qbvalues-*-02`; the model smoke is not run-to-run
deterministic, so it uses spreads). Rack queue (all custom wheel, traced, H-A4 at 0.78/105, sconv on,
re-gather off): `m30-f1-seq-02` (final-seq d4234c88e7, streams on), `m30-f1-noqb-01`
(final-seq-noqb 3b88a218cc = d4234c88e7 + revert of #9481's QB-after-MLP 0b6113396b, streams on),
`m30-f0-seq-02` (final-seq, streams off). F1-seq vs F1-noqb tests the ~0.16 s forward return c1 exposure.
The winner's trace feeds the PGLE build.

## M30-033 stackseq-03 score (2026-10-01)

`m30c-stackseq-trace-03` (sequential D + mirror + E + #9481 model commits incl. re-gather + Triton sconv; production
wheel; H-A4 at 0.78/105): **29.928 / 13.115 s** (+0.15 over sonic-02, +0.16 over stackpipe-03); peak 123.41;
loss at 180000 exact; dloss max 7.0e-4, late mean +8.0e-5, 27/49 positive. It still carries the re-gather
(~+0.1 s), the carry stall (141 ms/step) and the QB-after-MLP placement (~+0.16 s, unconfirmed). Expected
F1-seq-02 if the re-gather and stall attributions hold: ~12.87 s (~30.5%); F1-noqb lower if QB-after-MLP is the trigger.

## M30-034 m30-f1-seq-02: 30.22% (2026-10-01)

Final-seq program (sequential D + mirror + E + B's forward order + Triton sconv + MLP-weight prefetch +
QB-after-MLP, re-gather off, custom wheel with `XLA_GPU_HOST_TRANSFER_STREAMS=1`, H-A4 at 0.78/105; profiled):
**30.219 MFU / 12.990 s** (control 28.258 / 13.891; stackseq-03 29.928 / 13.115). Steady state 30.0-30.34,
slow steps at 180054 (28.67) and 180057 (27.89). Peak 123.50 / limit 143.75. Loss at 180000 exact; dloss
max 3.8e-4, late mean +9.0e-5, 44/49 positive. This is a single 60-step screen, not the goal claim.

**Pre-registered confirmation (written before further data):** final program (f1-seq or f1-noqb, whichever
screens higher with clean checks) x seeds {0, 1, 2} x 100 steps (180000 -> 180100, unprofiled, custom wheel,
streams on). Each is paired with the main control of the same seed and data (mhep-ctx4k-s{0,1,2}; s0 also
m30-ctl-s0-r2). Goal met iff every seed's median MFU over 180011-180099 is >= 30.0 and its loss divergence vs
its same-seed control is within the same-code band (max |d| <= ~1e-3, |late mean| <= ~2e-4, no growth
over the window) with drops in family. If a seed fails, report per-seed values; do not cherry-pick.
Confirmation arms queued (f1-seq program, d4234c88e7, custom wheel, streams on, H-A4 at 0.78/105, sconv,
unprofiled, 180000 -> 180100): `m30-conf-seq-s0` / `-s1` / `-s2` (seeds 0/1/2; ports 33020-33022), paired
against mhep-ctx4k-s0 / s1 / s2. If f1-noqb screens clearly higher with clean checks, it gets its own
confirmation set.

## M30-035 F1-seq-02 attribution (agent B, M30B-027)

Profiled steps 1 and 3 vs stackseq-03 (s/step): span 13.087 -> 12.977 (-0.110; scored -0.125); compute
10.803 -> 10.854 (+0.051: expert GEMMs +0.037 now under dispatch c1, attention projections +0.014); exposed
collectives 1.491 -> 1.473; exposed copies 0.769 -> 0.625 (**-0.144: carry stall 141 -> 3.7 ms/step, the
stream fix**). Re-gather off: -0.025 net (smaller than the earlier +0.1 estimate). Forward-order backward: net 0
(recomputed dispatch c1 0.167 -> 0.010, but dy c0's reverse return now sits bare at 0.149). Remaining ragged
exposure 0.843: fwd dispatch c0 0.192, fwd return c1 0.186 (noqb's target), bwd recomputed dispatch c0 0.300
(~0.14 transfer + skew), bwd dy c0 reverse return 0.149, recomputed dispatch c1 0.010. Other exposure:
latent reduce-scatters 0.284 (skew), backward remat_carry all-gathers 0.168, forward FSDP all-gathers 0.149,
exposed copies 0.625. QB value check: within a job the model smoke is deterministic. Across final-seq and noqb
the QB stats differ at fp32-reassociation level (qb_beta 0.2837 vs 0.2762, loss 1e-7 relative); autotune-off
and same-code repeats are running to tell fusion from autotune. noqb restores main's QB placement.
A (M30A-022) on f1-seq-02: all 16 tasks replaced jax-cuda13-pjrt 0.11.1+marin.708c3a4ec79c with
0.11.1+marin.283d5b6d98cd; job succeeded (verify_ragged_pjrt accepted it). Per step, one stream holds only
carry H2D (48) + optimizer-state H2D (54), another only carry D2H (48) + optimizer-state D2H (56), and the D2D
weight slices use the other four. XLA's stream pool reassigns xprof stream ids across steps, so check per step.
Exposed copies (stackseq-03 / F1 / hmo-02, s/step): optimizer D2H tail 0.261 / 0.261 / 0.260; carry H2D 0.219 /
0.218 / 0.219; **carry D2H 0.141 / 0.004 / 0.006**; optimizer H2D 0.100 / 0.101 / 0.099; total 0.773 / 0.641 / 0.624.

## M30-036 QB-after-MLP revert: no gain (2026-10-01)

`m30-f1-noqb-01` (final-seq-noqb 3b88a218cc, otherwise identical to f1-seq-02): **30.127 / 13.029 s** vs
f1-seq-02 30.219 / 12.990 (-0.09 MFU). Peak 123.50; loss at 180000 exact, dloss max 3.8e-4, late mean
-4.2e-5, 14/49 positive. The revert does not recover the forward return c1 exposure (or costs more
elsewhere), so the shared-expert placement has another trigger. Final program stays final-seq (d4234c88e7);
confirmation arms `m30-conf-seq-s{0,1,2}` already use it.
B (M30B-028): in the noqb trace the shared-expert forward GEMMs return under dispatch c1 / return c0 (as in
sonic-02), but forward return c1 stays bare (0.214) and the QB compute and collectives move back before the MoE
(+0.055 compute, +0.057 exposed dispatch all-gather); QB-after-MLP is a net win. Same code in two jobs is
not bitwise (XLA GEMM autotuning picks per job: 40/88 metrics, 35/36 grads differ), which explains part of the
same-code band. With autotune off, final-seq and noqb agree on loss, qb_beta, margins and bias, with only 1-ulp
diffs in z-loss/LB metrics, so QB-after-MLP leaves the QB statistics unchanged. Margin work started: C builds
PGLE profiles (plain + A's D2H patch) from the f1-seq-02 trace on `research/mcwitt/mfu30-final-pgle`; B
prototypes holding one shared expert back to cover forward return c1 (up to ~0.19 s).

## M30-037 PGLE arms queued (2026-10-01)

C built PGLE profiles from m30-f1-seq-02's trace (rank-0, 180021-180023) on `research/mcwitt/mfu30-final-pgle`
@ 883faa103f (final-seq + two .pbtxt files): 3,674 instruction costs, 3,672 match the f1-seq-02 HLO; 95.5% of
costly instructions covered (all 12 ragged all-to-alls, 110 copy-starts). The D2H variant sets the seven
end-of-step copy-starts to 277.4 ms each. Queued after the confirmations (custom wheel, streams on, traced,
seed 0, 60 steps): `m30-pgle-d2h-01` and `m30-pgle-plain-01`. Watch: carry stall, copy-start.44 hoisting
(+10 GiB), the PGLE accuracy-checker WARN lines (to confirm the profile applied), and the loss band.
B built the shared-expert holdback: `research/mcwitt/mfu30-final-seq-holdback` @ 95fa4631dc (= d4234c88e7 + one
commit; `--held-back-shared-experts N`, default 0, static field). The held shared expert's input passes through
`optimization_barrier((holdback, chunk_residuals[-1].expert_mlp))` after the chunk loop. It is tied to the last
chunk's gate/up residuals, not its down projection, so the recompute does not rerun the GEMM D removed. CPU:
the backward body keeps 8 all-to-alls and 16 ragged dots; one extra barrier; no collectives added. GPU gate
`m30b-holdback-gate-01` is pending. **Landing debt:** on final-seq (d4234c88e7) `tests/test_moe_hero_ep.py` +
`tests/test_moe_context_sharding.py` have 9 CPU failures from the `pip_packages` field added by cf5bc74409; fix
before any PR.

## M30-038 F0: wheel build is neutral; one-signed drift is noise (2026-10-01)

`m30-f0-seq-02` (final-seq, custom wheel, streams OFF, traced): **30.214 / 12.991 s** vs f1-seq-02 (streams ON)
30.219 / 12.990; peak 123.50. F0 drew a good stream assignment, so the wheel's build environment is neutral.
The stream fix is insurance against the draw lost by both -03 arms (~141-147 ms/step). Loss vs control:
F0 max 3.3e-4, late mean -7.4e-5, 10/49 positive; F1 max 3.8e-4, late mean +9.0e-5, 44/49 positive. Same
program, opposite-signed drift, so the one-signed drifts seen in single pairs are chaotic divergence, not bias.
Holdback (B): gate-02 at ab78bbe3ad has a bitwise forward but 4/36 gradient leaves differ (likely GEMM merging
of the two shared experts' gate/up changing wgrad accumulation order; reassociation-level, within the ruling).
A diff job is pending.

## M30-039 CORRECTION: F0 reused F1's executable from the compilation cache (agent A, M30A-023)

`m30-f0-seq-02` never compiled `jit_train_step`. Its remat VLOG printed no `Rematerialized ... in module
jit_train_step` line (F1 printed 64), and per-step `stream_check.py` shows F1's patched layout (carry
D2H/H2D alone on two streams). The persistent compilation cache key covers program, compile options and
XLA flags, but not the `XLA_GPU_HOST_TRANSFER_STREAMS` env var, and stream assignment is fixed at compile time.
So M30-038's "wheel build is neutral / F0 drew a good assignment" is **withdrawn**: F0 measured nothing about
the fix. The fix's value stands on f1-seq-02 vs stackseq-03 (carry D2H exposed 0.141 -> 0.004 s/step;
stackseq-03 shares one stream between carry and ~720 slice copies per step). The drift conclusion becomes
**stronger**: F0 and F1 ran the identical executable and drifted in opposite directions, so single-pair
one-signed drift is run-to-run noise. Hazard: the env switch is cache-unsafe in both directions (whichever
setting compiles first wins). All campaign arms on this wheel set it ON; a real env-off control would need
`JAX_ENABLE_COMPILATION_CACHE=false`. For landing, the switch belongs in DebugOptions (part of the cache key)
or default-on in a promoted wheel. Confirmation arms (env ON, same program) reuse F1's cached executable,
which is consistent.

## M30-040 Holdback arm queued (2026-10-01)

B (M30B-029): the holdback cause is confirmed. Without it, XLA merges both shared experts' gate/up and the latent
down projection into one GEMM on `mlp_in`; with it, the held expert's GEMMs stay separate. Gradient diffs:
`shared[1].w_gate` max_rel 4.2e-3 (0.05% of elements), `w_up` 4.0e-3 (0.06%), `token_embed` 8.3e-3 (2.6%),
i.e. ~1-2 bf16 ulp, accepted under the ruling. Module bitwise in all 3 cases; loss and router metrics bitwise;
recompute resurrects nothing; no new collectives. Queued `m30-holdback-01` (ab78bbe3ad,
`--held-back-shared-experts 1`, otherwise identical to f1-seq-02; custom wheel, streams on, traced) as a paired
comparison against f1-seq-02. Target: forward return c1 (0.186 s/step exposed in F1).
Rack queue: m30-conf-seq-s0/s1/s2 -> m30-pgle-d2h-01 -> m30-pgle-plain-01 -> m30-holdback-01.

## M30-041 Confirmation seed 0: PASS (2026-10-01)

`m30-conf-seq-s0` (final-seq d4234c88e7, custom wheel, streams on, unprofiled, 180000-180100): **MFU median 30.218
over 180011-180099 (n=89), 12.990 s/step**; control mhep-ctx4k-s0 28.260 / 13.890 on the same steps. Peak 123.50.
Loss at 180000 exact; dloss max 3.75e-4, late mean -9.2e-5, 16/89 positive. Divergence growth over thirds of the
window, mean |d| (final vs main | main repeat vs main): 6.6e-5 | 5.0e-5; 1.05e-4 | 1.01e-4; 1.57e-4 | 1.52e-4.
The final program's loss divergence is indistinguishable from same-code divergence. Seeds 1 and 2 pending.

## M30-042 Confirmation seed 1: MFU pass, loss drift above the pre-registered number (2026-10-01)

`m30-conf-seq-s1` vs mhep-ctx4k-s1 (same seed and data): **MFU 30.206 vs 28.235** (12.995 vs 13.902 s), peak 123.50.
Loss at 180000 exact (1.2063123). dloss max 7.7e-4 (within 1e-3) but **late mean +2.65e-4, above the
pre-registered ~2e-4**, 87/89 positive; mean |d| over thirds 1.27e-4 / 2.75e-4 / 3.28e-4, about 2x seed 0's
same-code growth (5.0e-5 / 1.0e-4 / 1.5e-4). The pre-registered bound was calibrated on seed 0 only, so the
honest test is a seed-1 same-code repeat. Queued `m30-ctl-s1-r2` and `m30-ctl-s2-r2` (main code, same flags as
the ctx4k controls) ahead of the margin arms. PGLE and holdback arms were cancelled and resubmitted behind them as
`m30-pgle-{d2h,plain}-02` and `m30-holdback-02`. Verdict: if seed 1's final-vs-main divergence falls within the
seed-1 same-code repeat's divergence, the fidelity criterion holds; otherwise report it as a fidelity
difference and investigate the value-changing components (E's fp32 SwiGLU backward; D's dS).
