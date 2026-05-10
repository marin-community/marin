# Logbook: MuonH paired-experts sweep

Variant of the MuonH matrix sweep where Newton-Schulz iteration runs on
**pairs of experts** instead of one expert at a time. For each 3D expert
weight `(E, A, B)` we reshape to `(E // 2, 2 * A, B)` before NS — so a
layer's MLP runs **32 NS instances instead of 64**.

Motivation: with Grug MoE defaults (E=64, `intermediate_dim = d / 2`),
the per-expert `w_down` is `(i, d) = (d/2, d)` — a 1:2 rectangle. Pairing
makes the resulting matrix `(d, d)` — square. NS converges fastest on
square matrices, so we should get cleaner orthogonalization with half
the per-layer NS overhead. Per-expert `w_gate_up` was already square
`(d, d)`; after pairing it becomes `(d, 2d)`, a 1:2 rectangle (still a
relatively benign aspect ratio for NS).

## Implementation

Single change in `experiments/grug/moe/optimizer.py`:

- New helpers `_pair_3d_leading` and `_unpair_to_original` reshape
  3D expert arrays in / out of the paired layout.
- New transform `scale_with_grug_muonh_paired` runs `_grug_scale_with_muon`
  in the paired space (momentum buffer + hyperball normalization both
  live there), then unpairs the result back to `(E, A, B)` so the trainer
  can apply the update.
- New config `GrugMoeMuonHPairedConfig` mirrors `GrugMoeMuonHConfig` but
  swaps the muonh transform for the paired one. Mask logic is identical
  (matrix leaves -> `muonh_paired`, lm head -> `adamh`, rest -> `adam`).
- New sweep launcher `experiments/grug/moe/muonh_paired_sweep.py`. Same
  gate points and surface as the original MuonH sweep but with run-id
  prefix `muonh-paired-*` and wandb group `muonh-paired-sweep`.

Hyperball normalization runs in **paired space** — each pair is treated
as one matrix for norm preservation. This means w_down's update
magnitude is computed across both paired experts together (single Fro
norm over the `(d, d)` block), not per-expert.

## Gate 1 run

- Submit via:
  ```bash
  .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
    --no-wait --reserve v5p-8 \\
    -e WANDB_API_KEY "$WANDB_API_KEY" \\
    -e MUONH_PAIRED_GATE=1 \\
    -- python -m experiments.grug.moe.muonh_paired_sweep
  ```
- Targets: `(d=512, 2.19e17)` and `(d=768, 1.70e18)`.
- Baseline references from `experiments/grug/moe/README.md`:
  - d512: paloma_macro = 3.8104, 405,630 tok/s, 0.6h
  - d768: paloma_macro = 3.4339, 273,532 tok/s, 2.8h

Comparison metric: effective speedup at the baseline's macro_loss target
(see `agent.md`). Pass = speedup > 1 at both scales.

## Notes / caveats

- NS scaling factor inside `_grug_scale_with_muon` is
  `sqrt(max(1, fan_out / fan_in))`. For paired `w_down` (d, d) this is 1
  (down from `sqrt(2)` for original `(d/2, d)`); for paired `w_gate_up`
  (d, 2d) this is also 1 (down from 1 for `(d, d)`). Net effect: paired
  `w_down` updates are ~`sqrt(2)` smaller in magnitude than per-expert,
  paired `w_gate_up` unchanged.
- Pairs adjacent experts by index (no rotation, no learned grouping).
  Sensible because expert ordering is arbitrary; QB routing balances the
  load across experts regardless of which pair they fall in.
