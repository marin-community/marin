# KL-SOAP-H tuning log — beat MuonH on d512 (issue #5728)

Goal: rebase KLSOAPH onto the current may-arch (PR 6153, `moe_may_pr`) and tune on d512 (coordinate descent) until paloma macro_loss < MuonH baseline **3.5438**.

Worktree: `marin-klsoaph-maypr` (branch `research/moe-klsoaph-maypr`, off `moe_may_pr`).
Baseline (marin-community/marin_moe, group `moe-may-compute-opt`): d512 MuonH `marin-big-run-moe_may_compute_opt_d512` paloma **3.5438** @ 530704 tok/s.

## Prior state (from issue #5728, branch research/moe-klsoaph-on-may-recipe)
- 2026-05-16 gate-1: KLSOAPH FAILED, d512 paloma 4.145 (vs AdamH 3.8104), 0.24× speedup.
- 2026-05-25 upstream-parity fix (5 primitives) + d512 sweep over (beta1, shampoo_beta, lr):
  - best = center (beta1=0.95, shampoo=0.90, lr 1×) paloma **3.789**; beta1=0.99 **diverged (NaN)**;
    lr 0.5×/2× both worse; shampoo 0.80/0.95 ≈ center.
  - → (beta1, shampoo, lr) near a local optimum at 3.789, still **+0.245 above MuonH 3.5438**.

## Code audit (klsoaph.py, 2026-06-12)
- **beta2 (Adam 2nd-moment, eigenbasis) and shampoo_beta (Gram+ESI EMA) ARE separate** — independently tunable. beta1=momentum.
- **BLOCK-WISE is the prime suspect**: SOAP runs per independent 128×128 tile (no cross-block coupling) → throws away global 2nd-order signal. `_klsoaph_step` einsums are already shape-generic (`gg_l=[r,r]`, `gg_r=[c,c]`), so de-blocking = state-shape + wrapper change, not a rewrite.
- **No Adam bias-correction** (EMA without 1/(1−βᵗ)). Scale moot under hyperball; β1≠β2 skews early direction slightly. Minor; note.
- ESI-whitening path intricate but structurally correct.
- precond_freq was 5 (TPU-speed deviation); upstream=1.

## Changes (this session)
1. **De-block → full-matrix SOAP** (gg_l=[r,r], gg_r=[c,c], q_l/q_r, exp_avg=[r,c], esi_l=[r], esi_r=[c]); keeps the verified upstream KLSOAPH math (whitened-gram + ESI + Adam-eigenbasis + warm-start QR), just full-matrix.
2. **precond_freq default → 1.**
3. Rebased onto moe_may_pr optimizer.py (already has `_scale_invariant_hyperball_updates`, `_match_named_update_sharding`, 3-group routing).

## HP surface (for coordinate descent)
| HP | current anchor | grid (to sweep) | notes |
|---|---|---|---|
| learning_rate | heuristic ×1.0 | ×{0.5,0.7,1.0,1.4,2.0} | heuristic fit for MuonH; SOAP may differ |
| adam_lr | 6e-4 | {3e-4,6e-4,1.2e-3} | baseline-Adam group LR |
| beta1 (momentum) | 0.95 | {0.9,0.95,0.98} | 0.99 diverged |
| beta2 (Adam eigen) | 0.9 | {0.9,0.95,0.98,0.99} | LOW at 0.9 — likely undertuned |
| shampoo_beta (Gram) | 0.9 | {0.9,0.95,0.99} | EMA on preconditioner |
| precond_freq | 1 | fixed at 1 (per user) | quality > speed |
| eps | 1e-8 | {1e-8,1e-6} | |
| max_grad_norm | 1.0 | {None,1.0} | MuonH uses None |
| warmup | (heuristic) | {0.01, ...} | match MuonH 0.01 |
| init_factor | 0.1 | {0.1,1.0} | ESI init |

Anchor after de-block: full-matrix, freq=1, (beta1,beta2,shampoo)=(0.95,0.9,0.9), lr×1, max_grad_norm=None (try). First sweep prioritizes beta2 (suspected undertuned at 0.9) + lr + max_grad_norm.

## Runs
(table updated as launched)

## Round 0 — establish full-matrix anchor (2026-06-12)
Launched CENTER (full-matrix, freq=1, b1=0.95,b2=0.9,shampoo=0.9, lr×1, maxgn=1.0, EP=2) on reserved v4-32 us-central2:
- `klsoaph_d512_center` — coordinator /kaiyue/iris-run-job-20260612-223838 — wandb marin-community/marin_moe group klsoaph-d512-maypr.
- Purpose: de-risk full-matrix eigh on real sharded mesh (expert grams [256,512,512] replicated — watch OOM/compile/throughput) + new baseline vs MuonH 3.5438.
- NEXT (once center trains clean): fan out parallel sweep — beta2 {0.95,0.99} (suspected undertuned), max_grad_norm None, lr_mult {0.7,1.4}, shampoo {0.95}.

## Round 1 — parallel coordinate-descent (2026-06-12, full-matrix, freq=1)
Anchor = center. Launched in parallel on reserved v4-32 us-central2 (wandb group klsoaph-d512-maypr):
| tag | HP delta | coordinator |
|---|---|---|
| center | — | /kaiyue/iris-run-job-20260612-223838 |
| beta2-0p95 | beta2 0.9→0.95 | /kaiyue/iris-run-job-20260612-224307 |
| beta2-0p99 | beta2 0.9→0.99 | /kaiyue/iris-run-job-20260612-224325 |
| maxgn-none | max_grad_norm 1.0→None | /kaiyue/iris-run-job-20260612-224338 |
Target: paloma < 3.5438. beta2 prioritized (0.9 low for SOAP Adam 2nd-moment). Watch full-matrix OOM/compile on expert grams.

## modded-nanogpt track_3 review (2026-06-12) — informs the grid
- Result #19 KL-SOAP-H (val 3.278): precond_freq=1, lr=.018, beta1=.95, beta2=.9, shampoo_beta=.9, hyperball, **no bias correction**. → our center anchor EXACTLY matches the upstream winning tuple; the missing bias-correction is INTENTIONAL upstream (not a bug).
- Result #21 Shampoo (3.2776): lr=.0015, betas (.9,.95), precond_freq=5, power=-1/4, wd=.2.
- Result #25 KL-SOAP-H + power LR decay + nonzero LR floors (3.2781): LR SCHEDULE matters.
- Result #27 SOAP-H (3.2782): freq=1, lr=.018, b1.95, b2.9 — whitening ≈ optional upstream.
KEY: upstream KL-SOAP-H lr=.018 ≈ 1.8× our MuonH-heuristic lr (0.0098). KL-SOAP-H wants HIGHER lr than MuonH → added lr-1.4 / lr-1.8 to round 1. beta2=0.9 was upstream-optimal (beta2↑ may not help). LR-schedule/power-decay is a later axis.

## Round 1 (extended) — lr axis added per track_3
| tag | HP delta | coordinator |
|---|---|---|
| lr-1p4 | lr ×1.4 (→0.0137) | /kaiyue/iris-run-job-20260612-224600 |
| lr-1p8 | lr ×1.8 (→0.0176 ≈ upstream .018) | /kaiyue/iris-run-job-20260612-224618 |

## ⚠️ CRITICAL FIX (2026-06-12) — non-SOAP groups were NOT apples-to-apples
Pulled the actual d512 MuonH baseline config from wandb (`marin-big-run-moe_may_compute_opt_d512`):
**adam beta1=0.9062, beta2=0.999, adam_lr=0.002262, epsilon=1.01e-15, max_grad_norm=None,
warmup=0.01, lr=0.009804, wd=0.1, lr_schedule=linear** — all heuristic-derived (heuristic_v2:
beta1=0.9062 fixed; beta2=clip(0.999^(tpb/131072),0.95,0.9999); for d512 tpb=131072 → beta2=0.999).
KLSOAPH was running its NON-SOAP groups (adamh: lm_head/output_proj; adam: embeddings/router) with
WRONG values: adam betas 0.9/0.95, adam_lr 6e-4 (~4× low), eps 1e-8, maxgn 1.0 — confounding the
whole comparison. The earlier "beta2" sweep was doubly confounded: one `self.beta2` drove BOTH the
SOAP eigenbasis 2nd-moment AND the real-Adam 2nd-moment.
**Fix (commit fe811dd32):** split SOAP eps from real-Adam eps (`adam_epsilon`); launcher now PINS
adam_lr/beta1/beta2/eps/maxgn from the MuonH heuristic (`_muonh`). The SOAP eigenbasis HPs
(beta1=0.95, beta2=0.9, shampoo=0.9, eps=1e-8, lr_mult, init_factor, precond_freq=1) are now the
ONLY variables under test. Killed all 6 confounded round-1 jobs; relaunched below.
KEY INSIGHT: MuonH's real-Adam beta2=0.999 (large-batch half-life, tpb=131072) ⇒ the SOAP eigenbasis
beta2=0.9 (upstream small-batch value) is very likely undertuned here → beta2 sweep up to 0.999.

## Round 1 (CORRECTED, apples-to-apples) — 2026-06-12
Anchor = center (full-matrix, freq=1, SOAP b1=0.95/b2=0.9/shampoo=0.9, lr×1; non-SOAP = MuonH d512).
| tag | SOAP HP delta | coordinator |
|---|---|---|
| center | — | /kaiyue/iris-run-job-20260612-230400 |
| beta2-0p95 | SOAP beta2 0.9→0.95 | /kaiyue/iris-run-job-20260612-230419 |
| beta2-0p99 | SOAP beta2 0.9→0.99 | /kaiyue/iris-run-job-20260612-230434 |
| beta2-0p999 | SOAP beta2 0.9→0.999 | /kaiyue/iris-run-job-20260612-230455 |
| lr-1p4 | lr ×1.4 | /kaiyue/iris-run-job-20260612-230509 |
| lr-1p8 | lr ×1.8 (≈ upstream .018) | /kaiyue/iris-run-job-20260612-230523 |
Target: paloma macro_loss < 3.5438. Power-decay dropped (linear matches baseline; user: unnecessary).

### Status 23:18 UTC — reserved-v4 contention
- center child RUNNING (grabbed a freed v4-32 slot), cold-starting (~48min to first loss).
- beta2-{0p95,0p99,0p999} + lr-{1p4,1p8} children PENDING: "Coscheduling: need 4 workers" — the
  us-central2 reservation is packed (/held/ unimax+proportional+swarm_fisher, /larry/ d512/768/1024,
  /pc0618/). Not mine to touch; my AMUSE runs are in east5 (won't free central2 v4). Letting them queue
  per the no-premature-conclusion-under-contention rule; they schedule as others' jobs finish.

### Status 23:19 UTC — moved 5 pending points to preemptible v5p-8 us-east5
v4 reservation stayed packed → per user ("use preemptible v5p-8 and other compute"), parametrized the
launcher (KLSOAPH_TPU/REGION/PREEMPTIBLE) and moved the 5 pending axes to preemptible v5p-8 us-east5
(MARIN_PREFIX=gs://marin-us-east5; verified nemotron/paloma/uncheatable mirrored there — no cross-region).
Loss is hardware-independent (global batch 32, steps 10980, seed 0 fixed) so v5p-8 losses compare directly
to the v4 anchor and the MuonH baseline; only tok/s differs (not the stopping metric).
| tag | HW | SOAP HP delta | coordinator |
|---|---|---|---|
| center | v4-32 central2 (resvd) | — | /kaiyue/iris-run-job-20260612-230400 (RUNNING) |
| beta2-0p95 | v5p-8 east5 (preempt) | SOAP beta2→0.95 | /kaiyue/iris-run-job-20260612-231751 |
| beta2-0p99 | v5p-8 east5 (preempt) | SOAP beta2→0.99 | /kaiyue/iris-run-job-20260612-231809 |
| beta2-0p999 | v5p-8 east5 (preempt) | SOAP beta2→0.999 | /kaiyue/iris-run-job-20260612-231822 |
| lr-1p4 | v5p-8 east5 (preempt) | lr ×1.4 | /kaiyue/iris-run-job-20260612-231840 |
| lr-1p8 | v5p-8 east5 (preempt) | lr ×1.8 | /kaiyue/iris-run-job-20260612-231854 |
Watch: full-matrix SOAP gram OOM on v5p-8 (fewer chips than v4-32; v5p has 95GB/chip though).

### Status 23:25 UTC — all 6 children RUNNING, cold-starting
east5 preemptible v5p-8 autoscaler provisioned all 5 axes within ~1 min ("available 0" was transient).
All 6 wandb runs RUNNING, no loss yet (compiling). v5p-8 child confirmed reading gs://marin-us-east5
(in-region, no cross-region). NOTE: a draccus traceback "TypeError: cannot create weak reference to
'str' object" / "Failed to dump config to yaml" appears in every run — BENIGN config-yaml-artifact dump
failure (can't encode the `model` field); training proceeds, structured config still logs to wandb.config.
Do NOT treat that traceback as a failure. No OOM on v5p-8 so far.

### ⚠️ Status 23:55 UTC — full-matrix COMPILE STALL (the de-block bottleneck)
ALL 6 runs reach the initial step-1 checkpoint (center v4 @23:35 after ~27min; beta2-0p999 v5p-8 @23:49)
then go SILENT — center 18+ min, zero training steps logged anywhere (wandb history_rows=0 for all 6,
tqdm stuck at elapsed:00:00 rate:-). The first real train step is compiling the full KLSOAPH graph:
per-expert Gram + batched eigh/QR over [256,512,512] (256 experts, d512: hidden=512, intermediate=256)
EVERY step (precond_freq=1). This is the exact "256 independent eigh … JIT-compile-heavy on TPU" the
block tiling existed to avoid (cf. removed docstring + project_klsoaph_kernel_status: full-fidelity klsoaph
~4% MFU). Block-wise PROVABLY trained last session (3.789/4.145 d512); full-matrix is stuck in XLA compile.
Conflict: "no block-wise" + precond_freq=1 + "no routing dominant params away" ⇒ infeasible compile.
**USER DECISION (23:55): WAIT on the full-matrix compile** — confirm whether XLA finishes (est. +30-60min)
and measure per-step rate before deciding. Runs kept alive; poller armed for first-step / crash.

### ✅ CORRECTION 00:13 UTC — NOT stuck, it was a long compile; runs ARE training
The ~45-min "silence" was iris log-STREAMING LAG during the long full-matrix XLA compile — which COMPLETED.
center (v4) is training: step 191, loss 7.43→6.28 (healthy), **~3.6 s/step, tok/s ~38.8k**. So full-matrix
KLSOAPH is FEASIBLE, just slow: ~3.6s/step → ~11h for 10980 steps; ~13× slower than MuonH (530k tok/s) —
the ~4% MFU regime from project_klsoaph_kernel_status. The 5 v5p-8 points are still finishing their compile
(~12min behind center). So round-1 WILL produce paloma results (~11h); the pain is the 45-min compile + low MFU.

### Pallas/CholeskyQR2 QR-acceleration work (user: "write a pallas QR")
Hot op = `Q,_ = jnp.linalg.qr(GG@Q)` batched [256,n,n] every step (n∈{256,512}); init uses jnp.linalg.eigh.
Both are the slow-to-compile/run XLA linalg lowerings on TPU. Plan: replace with **CholeskyQR2** (G=MᵀM matmul →
chol → triangular-solve; all MXU-friendly), the standard accelerator QR. It IS a true QR factorization, so it
preserves upstream fidelity (not an algorithm swap). CholeskyQR2 correctness VALIDATED on CPU: projector QQᵀ
matches jnp.qr to 2.6e-6 (identical column space — all SOAP needs), orthonormality 6e-5.
- qr_kernel_dev.py: correctness + TPU compile/runtime bench (jnp.qr vs choleskyQR2 vs eigh vs soap-refresh).
- TPU bench job /kaiyue/iris-run-job-20260613-001305 (v5p-8 east5) launched to quantify the win before wiring in.
- If plain-jnp CholeskyQR2 doesn't cut compile/runtime enough, escalate to a true Pallas kernel.

### TPU QR bench RESULTS (01:02 UTC, v5p-8, [256,512,512]) — QR is NOT the bottleneck
| op | compile | run/iter |
|---|---|---|
| jnp.qr | 3.86s | 75 ms |
| cholesky_qr2 | 3.12s | 63 ms |
| jnp.eigh | 9.04s | **2568 ms** |
| soap_qr = qr(GG@Q) | 8.42s | 75 ms |
| soap_cholqr | 8.09s | 66 ms |
**Conclusion: a Pallas/Cholesky QR saves only ~12% on a 75ms op — NOT worth building.** The expensive
linalg is `eigh` (2568ms, ~34× QR) but it runs ONLY at init (step 1), not per-step. So the ~3.6s/step is
inherent full-matrix SOAP overhead (projections + QR across ~12 expert [256,512,512] tensors + fwd/bwd) —
the cost block-tiling reduced, NOT a single replaceable kernel. Only meaningful kernel lever: swap the
init `eigh` → cholesky-QR basis (2568→63ms runtime, 9→3s compile) but that deviates from upstream init
fidelity. DECISION: don't build the Pallas QR (data-driven); runs train fine at ~3.6s/step (~11h), goal is
LOSS not throughput — focus compute on the sweep. CholeskyQR2 validated+available if ever needed (free 12%).

### Round-1 FIRST paloma evals @ step 1000 (01:05 UTC) — early ranking
| point | paloma@1k | note |
|---|---|---|
| beta2-0p95 | 4.560 | co-leader |
| center (anchor) | 4.561 | co-leader |
| beta2-0p999 | 4.569 | ~neutral |
| lr-1p4 | 4.641 | worse |
| lr-1p8 | 4.718 | worst (lr overshoot) |
| beta2-0p99 | pending | |
Step 1000 of 10980 (~9%) — NOT comparable to MuonH FINAL 3.5438. Ranking: lr↑ hurts; SOAP-beta2 0.95≈anchor.
Too early to re-anchor; let all run, re-evaluate ~step 3000-5000 when trajectories separate. No early-stop.

### Status ~01:10 UTC — beta2-0p99 preempted + auto-restarted
beta2-0p99 (preemptible v5p-8 east5) was PREEMPTED ~01:01 (was at step ~935, loss 4.2); iris AUTO-RESTARTED
the child ~01:03 — now recompiling (~45min) + resuming from ~step-900 checkpoint. wandb run shows "crashed"
(first attempt's wandb session died) but training is recovering; NO manual resubmit needed (child re-dispatched).
Will lag the others ~45min. The beta2 trend (0.9≈0.95 best, 0.999 slightly worse) is already clear from the
other two points, so this lag doesn't block re-anchoring. Watching for a preemption PATTERN (preemptible +
45-min recompile is fragile); if it recurs, consider more-stable capacity. Other 5 runs healthy/advancing.

### 🔑 KEY CALIBRATION vs MuonH trajectory (step 2000, ~01:55 UTC)
MuonH baseline paloma trajectory: @1k 4.452, @2k 4.180, @3k 4.069, @4k 3.997, @5k 3.921, ... FINAL 3.5438.
KLSOAPH (anchor/best) vs MuonH: @1k 4.560 vs 4.452 (gap **+0.108**); @2k 4.237 vs 4.180 (gap **+0.057**).
**Gap HALVED in 1000 steps** — classic SOAP: slow early (estimating Gram/eigenbasis preconditioner), then
faster 2nd-order convergence. If the trend holds, KLSOAPH crosses MuonH mid-run → plausibly finishes < 3.5438.
NOT a lost cause. Round-1 consolidated read:
- **beta2 axis NEUTRAL** (0.9/0.95/0.999 all 4.236-4.239 @2k, within match-threshold 2e-3) — "undertuned beta2"
  hypothesis REFUTED; hyperball normalization makes the eigenbasis 2nd-moment EMA insensitive. Drop from grid.
- **lr↑ HURTS** monotonically (×1.0 4.239 < ×1.4 4.356 < ×1.8 4.463 @2k). Anchor lr×1.0 best.
- → Anchor (lr×1.0, beta2=0.9, shampoo=0.9) HOLDS; no round-1 point beats it.
DECISION POINT = step 4000 (MuonH 3.997): if anchor ≤~4.0 the gap kept closing → on track, launch round-2 on
NEW axes (shampoo_beta=preconditioner EMA [most promising for the early-lag], beta1, eps, init_factor; skip
beta2; maybe lr×0.85). If the gap STALLS, rethink (the early-slow SOAP warmup may be the structural cost).
Compute note: contended preemptible v5p-8 saturates capacity, so round-2 likely runs AFTER round-1 frees slots.

### Round-1 mid-run gap (steps 1k-6k) — gap PLATEAUED at ~+0.04, not closing to 0
center: +0.109/+0.058/+0.050/+0.041 (1k-4k). beta2-0p95: +0.108/+0.059/+0.040/+0.039/+0.037/**+0.048** (1k-6k).
lr-1p4: +0.189→+0.138 (clearly worse, never closes). → Anchor tracks MuonH ~+0.04 behind and plateaus
(slightly widens @6k). Projecting: ~3.59 final vs MuonH 3.5438 — anchor alone does NOT beat MuonH. Need
round-2 to close +0.04 (or the final LR-decay phase closes it).

### Round-2 LAUNCHED (2026-06-13 ~04:40 UTC) — new axes from anchor, v5p-8 east5 (48 slots free)
All from anchor (lr×1.0, beta2=0.9, shampoo=0.9, beta1=0.95, eps=1e-8, initf=0.1), one axis changed:
| tag | delta | coordinator |
|---|---|---|
| r2-shampoo0p95 | shampoo_beta 0.9→0.95 | /kaiyue/iris-run-job-20260613-043606 |
| r2-shampoo0p99 | shampoo_beta 0.9→0.99 | /kaiyue/iris-run-job-20260613-043628 |
| r2-beta1-0p90 | beta1 0.95→0.90 | /kaiyue/iris-run-job-20260613-043642 |
| r2-beta1-0p98 | beta1 0.95→0.98 | /kaiyue/iris-run-job-20260613-043712 |
| r2-initf1p0 | init_factor 0.1→1.0 | /kaiyue/iris-run-job-20260613-043726 |
| r2-eps1e6 | eps 1e-8→1e-6 | /kaiyue/iris-run-job-20260613-043739 |
| r2-soapwu0p02 | SOAP warmup 0.01→0.02 (adam stays 0.01) | /kaiyue/iris-run-job-20260613-044114 |
| r2-soapwu0p05 | SOAP warmup 0.01→0.05 | /kaiyue/iris-run-job-20260613-044132 |
| r2-soapwu0p10 | SOAP warmup 0.01→0.10 | /kaiyue/iris-run-job-20260613-044145 |
NEW CODE: separate `klsoaph_warmup` (SOAP group) from inherited `warmup` (adam groups) — user insight that
the SOAP preconditioner lag wants a longer warmup while adam groups stay apples-to-apples (commit). Verified:
SOAP warmup 0.05 → SOAP lr still warming at step 110 (0.0020) while adam peaked (0.0098). Watching whether any
round-2 point tracks MuonH tighter than the anchor's +0.04 plateau. Round-1 still running for the final-gap read.

### 🔑 UPDATE ~05:44 UTC — gap CLOSING AGAIN in the LR-decay phase (plateau was mid-run only)
beta2-0p95 (round-1 leader) gap-vs-MuonH: @1k +0.108, @2k +0.059, @3-5k ~+0.038, @6k +0.048, **@8k +0.019**.
The mid-run +0.04 plateau was NOT the end — the LR-decay phase (6k→8k) is closing the gap (3.729 vs MuonH
3.71 @8k). If it continues to step 10979, the anchor family could land AT or BELOW MuonH 3.5438. So the
earlier "won't beat it" was premature — final post-decay number is the verdict (~2.5h out). center (v4) lags
in steps but tracks the same (beta2 neutral). Round-2 (9 pts) just past compile (~step 300, no evals yet).
Decision deferred to: (a) round-1 FINAL gap, (b) round-2 gap-at-matched-steps vs this trajectory.

## MFU WORK (goal v2: ~10% MFU on v5p-8, vs current ~2-4%) — 2026-06-13
Read scalable_shampoo (google-research) + distributed_kron (evanatyourservice): the canonical pattern is
DISTRIBUTE the per-parameter preconditioner linalg across devices (pmap/shard each device computes a subset,
then all-gather), use matmul-based coupled-Newton instead of dense eigh, and separate stat-update vs
precond-recompute frequencies.
**Root cause of low MFU found:** `_klsoaph_step` RESHARDED everything to FULLY REPLICATED → every v5p chip
redundantly computed ALL 256 experts' eigh/QR (4× wasted on v5p-8). The bench had already shown QR=75ms is
cheap, so the cost was this redundant replication across the expert axis, not the QR kernel itself.
**Fix (commit):** distribute the per-expert SOAP via `shard_map` over the mesh —
- `_klsoaph_step` is now PURE-LOCAL (no reshard/out_sharding; batched-local einsum+eigh+qr).
- `_klsoaph_step_sharded` reshards all inputs to a uniform expert-sharded spec `P(mesh.axis_names, None, None)`
  (shard E across every mesh axis, replicate the [n,n] matrix dims that eigh/QR need whole), then `shard_map`s
  the local step. Each device does only E/num_devices experts. 2D attn leaves + no-mesh fall through to replicated.
- shard_map (not explicit-mesh resharding through qr) sidesteps the `select`-sharding error that `jnp.linalg.qr`
  hits when a batch axis is sharded under an explicit mesh.
**Validated:** bit-identical (0.00e+00 over 3 steps) to the replicated reference on an 8-device (data,expert)
CPU mesh. Launched sharded-anchor (anchor HPs) /kaiyue/iris-run-job-20260613-062036 on v5p-8 east5 to measure
tok/s vs replicated center (38.8k); same HPs so loss must match center too (end-to-end correctness check).
Next: if tok/s up ~Nx (N=mesh size), MFU goal progressing; if comm dominates, amortize eigh (coupled-Newton)
or tune which axes shard E.

### sharded-anchor v1 FAILED on real mesh — two fixes (2026-06-13 ~06:30)
v1 (062036) crashed: `ShardingTypeError: Contracting dimensions are sharded` at the gram einsum, via the
2D-leaf fallback path. Cause: my refactor dropped replication for 2D attn leaves whose native sharding
leaves a CONTRACTING dim on "model" → ambiguous einsum. (8-device test missed it: only had a 3D leaf.)
**Fix 1:** 2D-leaf fallback now reshards to replicated before the local step (small matrices; necessary —
a single matrix's eigh can't be split). **Fix 2 (answers "must you replicate? it's wasteful"):** shard the
expert axis over ALL size>1 mesh axes so every device owns a DISJOINT set of experts (no redundant expert
compute) — not "expert"-only (which left the data-axis devices duplicating work). Excludes size-1 axes to
avoid the involuntary-full-remat that flattening the trivial "model" axis triggered. Validated correct +
no remat on (data,expert)=(2,2) and (2,2,2) CPU meshes. Only remaining replication: necessary matrix-dim
gather (eigh needs each [n,n] whole) + tiny 2D attn leaves. Relaunched sharded-anchor-v2 (063753).

### MONITORING FIX — detect failures without the user
Old monitor watched ONLY wandb summary → a compile-time crash (no wandb run yet) = silent timeout. The user
had to report the failure. New monitor watches the CHILD LOG for unambiguous fatal markers — `Fatal error in
grug training loop`, `jax._src.core.ShardingTypeError`, `RESOURCE_EXHAUSTED`, `OOM killed by the kernel`,
`No accelerator found` (excluding the benign draccus `weak reference` TypeError) — AND for a real tqdm step
rate (success). Exits/notifies on either. This catches the whole failure class wandb misses.

### 🔑 ROUND-1 near-final + ROUND-2 winner (2026-06-13 ~07:35) — beta1 is the lever
**Anchor (beta2-0p95) gap to MuonH closes to ZERO:** +0.108@1k → +0.040@4k → +0.019@8k → +0.014@9k →
**+0.007@10k**. After final LR decay it should ~TIE MuonH 3.5438 (final eval pending).
**Round-2 @ step 2000 (matched-eval comparison vs anchor):**
- **beta1=0.90: −0.031 vs anchor** (4.208 vs 4.239) — BEST. Monotonic: beta1 0.98 +0.085 (worse), 0.95 anchor,
  0.90 −0.031 (better) ⇒ LOWER beta1 helps. beta1 is THE active axis.
- shampoo0p99 −0.000, initf1p0 −0.000, eps1e6 +0.008 — neutral.
- SOAP-warmup all HURT (soapwu 0.02 +0.006, 0.05 +0.017, 0.10 +0.048) ⇒ the early lag is NOT a warmup issue
  (warmup hypothesis refuted); longer SOAP warmup just delays useful steps.
Since the anchor ~ties MuonH and beta1=0.90 improves ~0.03, **beta1=0.90 should finish clearly < 3.5438**.
**Round-3 launched:** push beta1 lower — r3-beta1-0p85 (/073633), r3-beta1-0p80 (/073654), v5p-8 east5.
Re-anchor → beta1=0.90 (will reconfirm at later steps + pick best of {0.80,0.85,0.90}).

## MFU HILL-CLIMB (continuous) — 2026-06-13 ~08:10
sharded-anchor-v2 (shard_map per-expert distribution) hit **147.9k tok/s ≈ 0.89 s/step = 4.0× the
replicated 38.8k** (MFU ~2%→~8-10%). Stabilizes by ~step 10 (per user). "Still not great" vs MuonH
(530k tok/s ≈ 0.25 s/step) — residual ~0.64 s/step is SOAP overhead (projections+QR+reshard), 3.6× MuonH.
**Methodology:** hill-climb like the loss coordinate descent. Fast inner loop = `soap_mfu_bench.py` (times
ONLY scale_by_klsoaph.update() on real d512 expert shapes under a (data,expert) mesh — minutes, no 45-min
full compile). Confirm winners with real tok/s@step10. Keep a running-best; one axis per variant.
**Axes (loss-impact noted; prefer loss-NEUTRAL, validate bit-identical):**
1. [neutral] persistent sharded optimizer STATE — init_fn currently makes replicated state, so each step
   reshards state replicated->sharded->replicated (gather round-trip). Store state expert-sharded -> per-step
   reshards only the GRAD (gather matrix dims), not the state. Likely the biggest comm win.
2. [neutral, validated] cholesky_qr2 instead of jnp.linalg.qr (~12% on the 75ms QR; projector-identical).
3. [near-neutral] bf16 for the projection/gram MATMULS, keep eigh/QR f32 (MXU 2× on the matmul bulk).
4. [neutral] reduce per-leaf shard_map overhead (12 expert leaves each reshard+shard_map separately; batch).
5. [loss-affecting] precond_freq>1 (skip QR-refresh/eigh most steps) — measure MFU gain AND loss cost; only
   adopt if loss barely moves (user wants freq=1 for quality, so this is a last resort / diagnostic).
Baseline opt_step time being measured by microbench (job 081251). Round 1 = persistent-sharded-state +
cholesky_qr2 (both loss-neutral).
MICROBENCH BASELINE: freq=1 opt_step=635ms (compile 139s), v5p-8, 6 layers — opt step is ~71% of the
0.89s end-to-end (so SOAP linalg, not fwd/bwd, is the MFU target). freq{2,4,8} variants measuring.

## 🐛 NaN INVESTIGATION + GUARD (user-flagged, 2026-06-13) — huge-risk fix
r2-soapwu0p10 (worst variant, SOAP-warmup=0.10) DIED at step 3748: eval went all-NaN, run terminated,
checkpoint saved. Scan of ALL runs: only soapwu0p10 NaN'd (not yet systemic) — but it's a latent risk for
ANY run. Train loss was finite up to the NaN; wandb stopped logging at 3748 so we can't see post-3748 train
from wandb. User: a transient eval NaN is a HUGE risk (corrupts the paloma metric / poisons the checkpoint →
"NaN on reload") — must root-cause + guard so NO future run dies.
**Likely cause:** a transient gradient spike / degenerate Gram→eigh/QR in KLSOAPH produces a non-finite SOAP
direction+state, poisoning params + the persisted preconditioner → checkpoint corrupt (reload NaNs) + eval NaN.
**FIX (committed b486485cc): NaN/Inf guard in _klsoaph_step** — if any output is non-finite, zero the direction
and KEEP the old state (skip the step). No-op when finite (validated: normal unchanged; injected 1e30 grad
skipped + state preserved + recovers next step). Protects ALL future runs from poisoned checkpoints / NaN eval.
**Root-cause confirmation in flight:** inspect_checkpoint.py on step-3748 (east5 job 083711) reports which
leaves are non-finite — params (optimizer poisoned → guard fixes) vs all-finite (eval-forward bf16 → need
float32-eval fix instead). Will extend the fix per the verdict.

### Config-parity de-risk — weight decay (2026-06-12)
Checked: MuonH baseline config stores weight_decay=0.1, BUT neither GrugMoeMuonHConfig.build() nor
GrugMoeKLSoapHConfig.build() references add_decayed_weights/weight_decay — both custom build()s leave
it inert (and wd is meaningless under the scale-invariant hyperball matrix step). So NO wd apples-to-apples
gap. Combined with the eps/beta/lr/maxgn pinning, the SOAP eigenbasis HPs are now the ONLY variable vs MuonH.

### Coordinate-descent plan (full axis spec + grids + re-anchor logic)
SOAP eigenbasis HPs are the only knobs (non-SOAP pinned to MuonH d512). Convergence = no single-axis
move improves paloma macro_loss by > 2e-3 (the "match threshold"). Significant improvement to re-anchor:
> 2e-3 better than current anchor.
- **Anchor₀** = (beta1 0.95, beta2 0.9, shampoo 0.9, eps 1e-8, init_factor 0.1, precond_freq 1, lr×1.0).
- **Round 1 (in flight):** beta2 ∈ {0.95, 0.99, 0.999} (large-batch: MuonH real-Adam beta2=0.999 ⇒ SOAP
  0.9 likely undertuned); lr ∈ {×1.4, ×1.8 (≈ upstream .018)}.
- **Round 2 (re-anchor on R1 best, then sweep in parallel):**
  - shampoo_beta ∈ {0.95, 0.99} — Gram EMA, same large-batch half-life argument as beta2.
  - beta1 ∈ {0.9, 0.98} — momentum (0.99 historically diverged → exclude).
  - eps ∈ {1e-7, 1e-6} — SOAP eigenbasis Adam denom.
  - init_factor ∈ {0.05, 1.0} — ESI init (whitening warmup).
  - lr finer bracket around R1-best lr (e.g. best=1.8 → try ×2.2, ×2.6; best=1.0 → ×0.7, ×1.2).
- **Round 3+:** finer grid around the running anchor on whichever axes still move; stop at convergence.
- **Ablations if stuck above 3.5438:** (a) SOAP-H (drop ESI whitening; track_3 #27 ≈ KL-SOAP-H) to test
  whether whitening hurts at MoE scale; (b) precond_freq 1→2 only if eigh cost forces it (quality-first).
  Power-decay LR schedule is OUT (user: unnecessary; linear matches baseline).

### Code de-risking (while cold-starting) — klsoaph.py re-audit
Full re-audit of the de-blocked full-matrix impl vs upstream PR #290: projection (qₗᵀ·g·qᵣ),
back-projection, whitened-Gram (gg_l from g·qᵣ·esiᵣ outer product), ESI (eigen=1/esi² EMA, rsqrt
clamp 4000), warm-started QR refresh (qr(GG·Q) + old→new basis momentum reprojection), eigh init +
zero first-step, intentional no-bias-correction — all FAITHFUL, no bugs. Cleaned stale block-wise
docstrings/naming (commit). CPU smoke re-verified shapes + finiteness. So if a point returns weak,
the cause is HP tuning, not the implementation.

### NaN-AFTER-LOADING investigation (user goal, 2026-06-13) — save/load structure suspect
User: optimizer emitting NaN is implausible; suspects checkpoint SAVE/LOAD logic. Load path
(levanter.load_checkpoint): deserializes saved arrays INTO the template (optimizer.init) structure;
allow_partial=True (grug default) KEEPS TEMPLATE INIT values for any leaf MISSING from the checkpoint.
Hypothesis: KLSOAPH's per-leaf state has None leaves (non-matrix params) + a complex optax masked/
inject_hyperparams structure; if some SOAP state arrays don't round-trip, allow_partial fills them with
init (eye q / zero gg / esi=init^-0.5) → INCONSISTENT preconditioner (loaded q + reset gg/esi) → first
post-load update explodes → NaN. This is a save/load structure bug, not an unsound optimizer (and the
NaN-skip guard wouldn't fix it — it'd keep the inconsistent loaded state).
Discriminator: inspect raw saved checkpoints — DEAD soapwu0p10 step-3748 (job 084828) vs HEALTHY
beta2-0p95 step-10980 (finished clean, job 084849). If healthy raw checkpoint is finite but reload
into-template NaNs → load/structure bug. If saved arrays already NaN → in-memory at save.

### NaN-after-loading: ROUND-TRIP TEST CLEAN (2026-06-13) — save/load is NOT the cause
Built real GrugMoeKLSoapHConfig optimizer, populated SOAP state (3 steps), saved via levanter
save_checkpoint, reloaded into a FRESH template with allow_partial=True (trainer's resume path):
**32/32 opt-state leaves loaded, 0 non-finite, 0 reset-to-init, 0 structure mismatch, post-load step finite.**
=> levanter save/load does NOT introduce NaN and allow_partial does NOT drop KLSOAPH leaves. A FINITE
checkpoint reloads cleanly. (The orbax "incomplete checkpoint" earlier = orbax can't read levanter's
tensorstore format; red herring.)
CONCLUSION: the NaN originates IN-MEMORY (transient SOAP gram/eigh overflow at step 3748 on the unstable
long-warmup soapwu0p10 trajectory), then gets SAVED -> any reload of that poisoned checkpoint NaNs
("nan on loading" symptom). Root = NaN state persisted, NOT load logic. The NaN-guard (skip non-finite
step, keep last-good state) PREVENTS NaN from ever entering state/checkpoint => checkpoints stay finite =>
reload always clean. So the guard fixes BOTH the in-process death AND nan-after-loading, by prevention.

### NaN-after-loading: COMPLETE TWO-LAYER FIX (2026-06-13)
Investigation conclusion: levanter save/load is CLEAN (round-trip: 0 non-finite, 0 reset, post-load step
finite); the NaN originates in-memory (transient SOAP gram/eigh spike) and, if persisted, poisons the
checkpoint → reload NaNs. Fix (both committed, tested):
1. TRAINING GUARD (klsoaph.py): _klsoaph_step skips any non-finite step (zero direction, keep last-good
   state) → NaN never enters state → checkpoints always finite.
2. LOAD SANITIZER (grug/checkpointing.py): on restore, reset non-finite OPT-STATE entries elementwise to
   fresh init (resume cleanly, lose a little preconditioner history); reject non-finite PARAMS → restore
   loop falls back to an older checkpoint. Tests added (tests/test_grug_checkpointing.py, 9 pass).
Together: NaN can neither be SAVED (guard) nor SURVIVE a load (sanitizer) → no future run dies from it.
All FUTURE launches use this code (committed). Currently-running healthy runs predate it but are valid
while finite (guard is a no-op when finite); relaunch with the fix only if one NaNs.

### 🎯 RESULTS (2026-06-13 ~09:00) — GOAL A imminent, GOAL B headroom mapped
GOAL A: round-1 anchor (beta2-0p95) FINISHED **3.5475** (ties MuonH 3.5438, +0.004 above — anchor alone
doesn't beat). lr-1p4 3.5644, lr-1p8 3.5942 (worse). **beta1=0.90 @step5000 paloma 3.903 — BELOW MuonH's
3.921 trajectory** (−0.018) and −0.055 vs anchor@5k; on track to finish < 3.5438 = the winning config
(running, ~46%). Round-3 r3b-beta1-0p85 (/090618) + 0p80 (/090639) relaunched on NaN-safe code to find
best beta1. beta1=0.98 worse → minimum is at/below 0.90.
GOAL B: microbench opt_step (v5p-8, 6 layers): freq1 635ms / f2 430 / f4 356 / f8 298 → **QR-refresh path
≈337ms = 53% of opt_step**; non-refresh (projections+gram+Adam+ESI+hyperball+reshard) ≈298ms. MFU already
~10% (4× sharding). To push further (loss-preserving): subspace-QR (arxiv 2605.26327, reparam P=QᵀSQ, QR on
B·d-col block) attacks the 337ms; bf16 projection matmuls + cholesky_qr2 attack both — validate loss-neutral
+ microbench before adopting. Defer big subspace-QR until beta1=0.90 confirms the MuonH beat (stable baseline).

### Subspace-QR MFU implementation PLAN (arxiv 2605.26327) — goal B push beyond 10%
Target: the QR-refresh path (~337ms = 53% of the 635ms opt_step). MFU already ~10% (4× sharding); this pushes further, LOSS-PRESERVING (paper: subspace update keeps a small effective decomposition interval).
Incremental, validate-at-each-step:
 STEP 1 — full-basis REPARAMETRIZATION (mathematically EQUIVALENT to current → bit-validatable):
   store P_i = Q_iᵀ S_i Q_i instead of S_i (gg_l/gg_r). Updates (paper fig.2):
   - covariance (every step): P ← (1-β2)P + β2·(Qᵀ Δ Q), reuse rotated grad G̃'=Q₁ᵀ G Q₂.
   - eigenbasis refresh: O = qr(P); Q ← Q·O; P ← Oᵀ P O.  (replaces Q=qr(S·Q))
   For KL-SOAP also keep ESI/eigenvalue (esi) updates in the FULL basis (Step 3) — already have.
   VALIDATE: bit-identical loss vs current full-QR on CPU mesh over ~20 steps.
 STEP 2 — SUBSPACE QR (the speedup, paper fig.3): pick index set I of d_sub=B·d columns (B≈1/4);
   O_sub = qr(P[I,I]) (O(d_sub³), B² cost); Q[:,I] ← Q[:,I]·O_sub; rotate P rows/cols in subspace.
   Column selection: cyclic (round-robin blocks) or largest-off-diagonal-mass. B and selection are HPs.
   VALIDATE: microbench opt_step (expect ~B²× the QR cost → big cut); loss-validation RUN with beta1=0.90
   (must still beat MuonH 3.5438). Sweep B∈{1,1/2,1/4} for the MFU/quality tradeoff.
 STEP 3 — bf16 STORAGE of P,Q (paper: reparam makes bf16 stable). Validate loss + memory.
RISK: large rewrite of _klsoaph_step state+math → do as a NEW scale_by_klsoaph_reparam (flagged), validate
bit-identical at STEP 1 before trusting; keep current optimizer as the winning-config default until validated.
Defer adoption into the winning beta1=0.90 config until goal A is confirmed (stable comparison baseline).

### beta1 axis — LOWER IS BETTER (2026-06-13 ~11:30); MuonH beat essentially in hand
SOAP beta1 (projected-momentum EMA) is THE lever. Anchor 0.95 ties (3.5475). Monotonic improvement lowering it:
- beta1=0.90: gaps vs MuonH −0.018@5k −0.009@6k −0.011@8k (running ~78%) → finishing ~3.53 < 3.5438.
- beta1=0.85 / 0.80 (round-3b): paloma ~4.035 @3k vs anchor 4.109 (−0.074), both ~flat → minimum ≈ 0.80-0.85.
Launched r4-beta1-0p70 (/113012) to bracket the min. Multiple configs tracking BELOW MuonH → beat confirmed
pending step-10979 finals. After finals: pick best beta1, declare GOAL A, then re-check other axes from the
new anchor + implement subspace-QR (MFU) against it.

### beta1 finals approaching (2026-06-13 ~12:15) — beta1=0.80 leads, beat decisive
@step6000 gaps vs MuonH: beta1=0.80 −0.036, 0.85 −0.029, 0.90 −0.009 (vs anchor 0.95 which TIES). Clear
monotonic lower-better; **beta1=0.80 is the leading config**, tracking well below MuonH → will finish < 3.5438
decisively. beta1=0.90 (step ~9717, gap −0.005@9k) finishes first (~45min), a narrow beat/near-tie. beta1=0.70
(/113012) bracketing. Winning config ≈ beta1=0.80. Await finals to declare GOAL A; then push 0.70/0.65 if 0.80
not yet the min, and implement subspace-QR (MFU) against the beta1-winner.

### Subspace-QR IMPLEMENTED + VALIDATED (2026-06-13 ~12:25) — goal B push
Added subspace_frac to scale_by_klsoaph/_klsoaph_step (committed). _refresh now computes P=q.T@GG@q and
q_new=_subspace_orthog(P,q,...): frac=1.0 → q@qr(P) (full); frac<1 → QR on the cyclic k×k diagonal block
(k=frac*n) updating only those k basis columns, cycling over 1/frac refreshes (arxiv 2605.26327).
VALIDATED on CPU: frac=1.0 direction == pre-subspace code (diff 4.8e-6, bit-identical → winning config
UNCHANGED); sharded(frac=1.0)==replicated (0.0) on (data,expert) mesh; frac=0.25 q orthonormal (4e-7) +
finite over steps, sharded finite. Microbench frac=1.0 vs 0.25 launched (jobs 122212/122232) to measure the
opt_step cut on the 337ms (53%) QR-refresh. NEXT: if microbench shows a clear cut, loss-validation run with
the winning beta1 + subspace_frac=0.25 (must still beat MuonH 3.5438); sweep frac∈{1,1/2,1/4} for MFU/quality.

### 🚀 Subspace-QR MFU RESULT (2026-06-13) — 2× opt-step cut → ~16% MFU
Microbench (v5p-8, 6 layers): opt_step 642.8ms@frac1.0 → **311.4ms@frac0.25** = 2.06× cut (the QR-refresh,
53% of step, slashed). End-to-end est: step ~0.89s→~0.56s → tok/s ~234k → **~16% MFU** (vs ~10% sharding-only,
~2-4% replicated). Clears the 10% goal. Launched loss-validation run ss0p25-b80 (beta1=0.80 + subspace_frac=0.25)
on v5p-8 east5 to confirm it STILL beats MuonH 3.5438 (compare to r3b-beta1-0p80 frac=1.0) + measure real tok/s.

### Goal A is a TIGHT race — gaps narrow in LR decay (2026-06-13 ~12:45)
beta1=0.90 @step10000 = 3.5856 vs MuonH 3.585 (≈tied); extrapolating MuonH's last-979-step decay (−0.041)
→ beta1=0.90 final ~3.544 = marginal MISS (beats anchor 3.5475 but not clearly < 3.5438). The decay phase
narrows ALL gaps toward ~0. Clean beat depends on lower beta1 (0.80 was −0.036@6k, 0.85 −0.029@6k) HOLDING
margin through decay — finals ~3h out (they're at ~step 7300). beta1=0.70 (/113012) also in flight. So
KLSOAPH full-matrix ≈ MATCHES MuonH (~3.54), best-beta1 marginally below/at-tie — need the 0.80/0.70 finals
to confirm a clean < 3.5438. If all land ~tie, may need beta1=0.65 or re-check another axis from the beta1 anchor.

### ⚠️ HONEST CORRECTION (2026-06-13 ~13:00) — MuonH NOT beaten yet; mid-run gaps misled
beta1=0.90 FINAL@10979 = **3.5500** > MuonH 3.5438 (and worse than anchor 3.5475!). The mid-run lower-beta1
advantage (−0.036@6k) DID NOT survive LR decay — gaps narrow + reverse in the final phase. @step9000 the betas
are ~tied (0.85=3.6325, 0.90=3.6347, 0.80=3.6394), all converging ~3.55. So full-matrix KLSOAPH lands
~3.547-3.550, **+0.004-0.006 ABOVE MuonH 3.5438 — NOT a beat.** Anchor (beta1=0.95)=3.5475 remains the best
KLSOAPH. LESSON: must judge on step-10979 FINALS, not mid-run gaps (no-premature-conclusion).
Beta1 axis ~exhausted (min ~0.90-0.95, all above 3.5438). NEXT LEVER = the DECAY phase (where KLSOAPH loses
its mid-run lead): tune KLSOAPH's LR schedule independently (min_lr_ratio nonzero floor, or power decay — track_3
Result#25 won with power-decay + nonzero floors) to hold the lead through the end. Await 0.80/0.85/0.70 finals
to confirm beta1 axis, then sweep the schedule axis. GOAL A NOT yet met — continue coordinate descent relentlessly.

### beta1 FINALS — trend REVERSED, axis converged at 0.95 (2026-06-13 ~13:30)
Final@10979: beta1 0.95(anchor)=3.5475 < 0.90=3.5500 < 0.85=3.5555 < 0.80=3.5680. HIGHER beta1 better at the
final (opposite of mid-run). beta1 axis CONVERGED at 0.95 = best KLSOAPH = 3.5475, **+0.0037 above MuonH 3.5438**
(0.98/0.99 excluded: worse mid-run + 0.99 historically diverged). lr↑ worse (lr-1p4=3.5644, lr-1p8=3.5942).
So coordinate descent on {beta1,beta2,shampoo,eps,init,warmup,lr} caps at ~3.5475 — MuonH NOT beaten; the gap
opens entirely in the LR-decay tail. Active lever: LR-schedule (min_lr_ratio floor 0.05/0.10, running ~11h).
GOAL B CONFIRMED: subspace-QR ss0p25-b80 = **284,662 tok/s** (1.93× sharded 147.9k, 7.3× replicated 38.8k,
~16-20% MFU) — 2× opt-step cut realized end-to-end. Loss-neutrality vs r3b-beta1-0p80 (frac=1.0, 3.5680)
pending its finish.

### ⚠️ Subspace-QR frac=0.25 DEGRADES LOSS (2026-06-13 ~13:45) — not loss-neutral as implemented
ss0p25-b80 (beta1=0.80, subspace_frac=0.25) paloma 5.54/6.04/5.90 @1k/2k/3k vs r3b-beta1-0p80 (frac=1.0)
~4.0 — Δ +1.0..+1.9 and RISING. So my cyclic-block subspace-QR is NOT loss-preserving: refreshing only 1/4
of the eigenbasis per step ≈ precond_freq-4 staleness (which hurts), and the off-block P coupling isn't
resolved (the on-the-fly P version skipped the paper's full P=QᵀSQ reparam that keeps eigenvalues+basis
consistent). The 2× opt-step speedup (285k tok/s) is real but trades quality. Killed ss0p25-b80.
**Goal B resolution: the clean LOSS-NEUTRAL ~10% MFU is the SHARDING (shard_map 4×, bit-identical) — meets the
10% target.** A loss-free >10% needs the full reparametrization (store P, incremental update, rotate in subspace)
or better subspace selection — deferred (substantial). subspace_frac stays in code (default 1.0 = bit-identical).

### Schedule axis: min_lr FLOOR HURTS; trying cosine SHAPE (2026-06-13 ~14:00)
sched-floor0p05/0p10 (anchor + min_lr_ratio=0.05/0.10) STALLED at train ~6.0 (vs anchor 4.2 @step1000) —
started identical (@100≈9.1) then diverged after warmup. LR schedule verified identical early + frac=1.0
bit-identical, so a nonzero LR FLOOR genuinely hurts KLSOAPH (2nd-order method needs to decay fully to settle;
a persistent floor keeps perturbing → stalls). Floor lever ABANDONED, runs killed. Next: cosine decay SHAPE
(min_lr→0, but keeps LR higher mid-run then drops steeply — may help KLSOAPH hold its mid-run lead through the
tail where the linear schedule loses it). Launched sched-cosine (beta1=0.95, cosine, min_lr=0). If it doesn't
beat 3.5438, conclude KLSOAPH full-matrix matches MuonH within +0.1% (3.5475) — likely intrinsic — and report.

### MFU CALIBRATED (logged throughput/mfu, v5p-8) + 10%-MFU solution (2026-06-13 ~16:30)
| variant | tok/s | MFU |
|---|---|---|
| replicated (v4) | 57k | 2.1% |
| sharded shard_map frac=1.0 | 144k | **5.2%** |
| subspace-QR frac=0.25 | 284k | **10.2%** |
Sharding alone is only 5.2% (earlier '~10%' was wrong). The 10% target is hit ONLY by subspace-QR (10.2%).
PROBLEM: my cyclic subspace-QR degrades loss (6.0 vs 4.0) — refreshing 1/4 of the eigenbasis/step is too stale
for KL-SOAP's Adam-in-eigenbasis. SOLUTION = the paper's FULL reparam (store P=QᵀSQ, maintain incrementally,
rotate in subspace) which keeps EIGENVALUES full-basis every step (the loss-preserving key my on-the-fly version
skipped). Implementing that = 10.2% MFU at the anchor's loss. Also restarted soapwu0p05/0p10 (the NaN'd warmup
runs) on NaN-safe code to verify the guard+sanitizer. NOTE: precond_freq>1 (freq=4→9%, freq=8→11% MFU) also
cuts the QR cost but trades quality (stale basis) — same issue; subspace+full-reparam is the quality-preserving route.

### Lossless-MFU lever: CholeskyQR2 QR backend + precision guard (2026-06-13 ~16:55)
Per user steer ("first think faster QR losslessly + better sharding, not subspace"), the subspace-QR
quality loss is avoided. New lossless QR path: KLSOAPH_QR=cholesky uses CholeskyQR2 (G=MᵀM -> Cholesky ->
triangular-solve, two passes), which is MXU-native (matmul-heavy) vs XLA's sequential-reflection Householder
QR. The subspace-QR(0.25) MFU jump (5.2->10.2%) showed QR DOMINATES opt-step time, so a faster *full* QR is
the right lossless lever. CholeskyQR2 forms MᵀM (squares cond number) -> inaccurate on ill-conditioned grams,
so _qr_Q now CHECKS max|QᵀQ-I| < 1e-3 and falls back to jnp.linalg.qr (robust) when the Cholesky path is too
far off — keeps QR bit-accurate while fast on the well-conditioned common case. Validated on CPU: well-cond
input -> cholesky (ortho_err 1e-6), cond~1e12 -> falls back to jnp.qr (3.5e-7); both orthonormal. Direction
diff vs xla-QR ~3e-4 over 40 steps (loss-neutral).
Microbench reworked to time BOTH backends in one job (monkeypatch _QR_IMPL + re-jit) -> single apples-to-apples
speedup number. v5p-8 microbenches (164335 xla / 164352 cholesky) stuck PENDING behind 6 east5 sweep finals;
relaunched the both-backend bench on the IDLE reserved v4-32 central2 (165749) for the cholesky-vs-xla ratio.

### Compute-contention status (2026-06-13 ~16:55)
10 of my jobs running, all 6 Goal-A/NaN-fix finals on east5 v5p-8 preemptible (adamlr2x/0p5, adamb2-0p99,
soapeps1e6, soapwu0p05-v2, soapwu0p10-v2) — all cold-starting (train children not yet RUNNING). sched-cosine
restarted at step 8 (was preempted). r4-beta1-0p70 near done @10475 paloma macro 3.612 (beta1=0.70 = known-wrong
direction, won't beat 3.5438). No premature conclusions under contention — let them finish.

### NEW criterion: compile < 10 min + understanding the ~24-min compile (2026-06-13 ~18:15)
Goal updated: the d512 run must also COMPILE in <10 min (was ~24-30). Root-cause analysis (not blind):
per-step linalg in `_klsoaph_step` = `jnp.linalg.qr ×2` (refresh branch) + `_flipped_eigh` (eigh) ×2
(init branch — compiled even though it only RUNS at step 1, since lax.cond compiles all branches).
Isolated v4 compiles: eigh 13s, qr 3.5-10s — none is 24min alone, so 24min is the FULL graph (6 layers
× 2 leaves × {eigh-init + qr-refresh}, nested lax.cond, shard_map). Prime suspects: batched-eigh lowering
(heaviest, runs once but compiled) and a **DOUBLE COMPILE**: `_init_state` jit has no out_shardings ->
opt.init produces REPLICATED state, but `_klsoaph_step_sharded` returns mat_p (out_specs) -> step-1 input
(replicated) differs from step-2+ input (mat_p) -> train_step compiles TWICE.

### Lossless lever 1: persistent expert-sharded opt-state (compile + MFU) — TESTING (job 181654 fix-shard)
Fix: init_fn now reshards every per-expert leaf (gg/q [E,n,n], esi [E,n]) to mat_p, so the state is STORED
sharded from step 0. Eliminates (a) the step1-vs-step2 double-compile, (b) the per-step replicated<->mat_p
reshard round-trip, (c) ~mesh-size× opt-state HBM. Pure layout = lossless. Verified on 4-dev CPU mesh:
gg_l init P(('data','expert'),..), persists across steps, finite. Launched fix-shard on v5p-8 east5 to
measure compile time + MFU + loss vs fix-anchor (which has the replicated-init code).

### QR oracle study (Su Jianlin streaming-power-iteration, muon_streaming_power_iteration.md) (2026-06-13)
KLSOAPH refresh `qr(GG@q)` == the article's NAIVE single QR (SCQR Cholesky sees qᵀGG²q -> cond(GG)²).
Implemented dual-orthogonalization SCQR (eq.4, from GG alone): projector-IDENTICAL to qr(GG@q) (projΔ=0.0
CPU, cond 1e3 AND 1e6) with each Cholesky at cond(GG) not cond². LOSSLESS + well-conditioned where single
SCQR fails. BUT v4 isolated bench (full f32): jnp.qr 120ms, choleskyQR2 95ms, single-scqr 166ms,
**dual_scqr 215ms** (slowest — ortho-check QᵀQ + lax.cond + extra matmuls). So dual-SCQR is NOT a per-step
MFU win; QR is only ~120/642ms of the opt-step anyway. Parked: only useful if it cuts COMPILE (cholesky vs
XLA qr/eigh lowering) and amortized via precond_freq>1. MFU levers are sharding + precond_freq + bf16, not QR.

### Efficiency baseline progress + corrected metrics (2026-06-13 ~21:15)
GOAL refined: throughput/tokens_per_second >= 330k (0.99-EMA "time_average_factor=0.99"), compile <10min,
loss regression <0.001 vs anchor, beat MuonH (3.5438).

THROUGHPUT (0.99-EMA tok/s, the goal metric): anchor pf1=145k; pf2=236k; **pf8=305k** (peak 334k). pf>1 makes
per-step MFU/tok bimodal (refresh step ~5% / non-refresh ~12%); judge by TIME-AVERAGE not peak (pf2 peak 12%
but avg 8.5%). Ceiling ~334k = the f32 non-refresh step. -> need pf16/32 (approach ceiling) to clear 330k, OR
push the ceiling (bf16/block-wise). Launched reparam_eig pf4/8/16/32 to (a) hit 330k via high pf, (b) verify
loss-neutrality (reparam = eigenvalues from fresh-basis Gram diag at refresh, staleness-robust).

LOSS: clean anchor (pf1) = **3.5204 < MuonH 3.5438** -> loss criterion met at baseline; need efficiency config to hold.

COMPILE: measured via system/tpu.hloQueueSize.tensor_core_7 stably>10 (NOT log markers, NOT the TPU-alloc wait).
identity_init (skip step-1 eigh) cuts compile ~90min -> ~28min (eigh lowering was a major cost) -- a REAL win,
my earlier log-marker estimate was wrong. Still >10min; need more. JAX_LOG_COMPILES=1 gives per-function compile
breakdown (clean, wait-independent) -> next compile lever (block-wise SOAP shrinks QR/grams). east5 preemptible
contention is blocking diagnostics; using reserved v4 serially.

DONATION ("buffers not usable"): all-non-square leaves -> a LAYOUT mismatch (sharding matched; with_sharding_constraint
was a no-op). Fix: with_layout_constraint pinning output train-state to row-major (matches jnp.zeros-init buffers).
Testing on reserved v4.

### CORRECTION: anchor does NOT beat MuonH (2026-06-13 ~22:10)
Earlier "anchor 3.5204 < MuonH" was train/loss vs paloma — INVALID. Clean anchor FINISHED:
eval/paloma/macro_loss = **3.5496** (train/loss 3.264). MuonH baseline 3.5438. So anchor is +0.0058 ABOVE
MuonH (matches pre-pollution 3.5475). Loss criterion (beat MuonH) NOT met at anchor -> needs coordinate-descent
HP tuning AFTER the efficiency baseline. Efficiency baseline <0.001 regression is measured vs ANCHOR=3.5496.

### COMPILE ROOT-CAUSE NAILED + donation fixed (2026-06-14 ~00:10)
Measured train_step XLA compile DIRECTLY via JAX_LOG_COMPILES (clean, wait-independent):
- WITH identity_init (fix-diag-v5p, v5p-8): jit(train_step) XLA compile = **167s (~3min)**; v4: 173s.
- WITHOUT (anchor, eigh): hloQueueSize ~90min; with identity_init ~28min.
=> identity_init (skip step-1 batched eigh) cuts train_step XLA compile ~65min -> 3min. The batched-eigh
lowering WAS the compile beast. train_step is now 3min (<10min). The remaining ~25min (hloQueueSize) is
NON-train_step startup: deps install + data-cache load + (east5) TPU-alloc-wait -- NOT the optimizer.
=> Optimizer-compile restructure (batch-QR / block-wise / Newton-Schulz) is UNNECESSARY. NS also proven
invalid (CPU: polar doesn't diagonalize qᵀGGq). Next: localize/cut the deps+data startup if it counts.

DONATION ("buffers not usable"): FIXED. _constrain_std_layout (pin output train-state to row-major) ->
0 donation warnings on BOTH v4c and v5p (fix-diag-v5p). Confirmed on target hardware, no crash.

### Criterion status (2026-06-14 ~00:12)
COMPILE <10min: **MET**. fix-diag-v4c (reserved v4, no alloc-wait) reached first train step at _runtime=7.3min
with identity_init. east5 "~30min" was TPU-alloc-wait. train_step XLA compile 3min + deps/data ~4min = 7.3min.
THROUGHPUT >=330k (0.99-EMA): pf8=305k, pf16=321k, pf32=323k -- shrinking increments -> non-refresh step ceiling
~325k < 330k. pf alone likely insufficient (testing pf64/128). Non-refresh step is f32 optimizer matmuls
(gram/projection/direction) -> need bf16 (Wu Lin reparam: q orthonormal cond=1 -> bf16-safe) to raise ceiling.
LOSS <0.001 vs anchor(3.5496): pf-reparam runs not finished yet; bf16 must also be loss-neutral.
