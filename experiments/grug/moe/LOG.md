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
