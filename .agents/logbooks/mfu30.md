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
