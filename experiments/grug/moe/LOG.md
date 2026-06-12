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
