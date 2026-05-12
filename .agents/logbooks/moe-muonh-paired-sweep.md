# Logbook: MuonH paired-experts sweep

Variant of the MuonH matrix sweep where Newton-Schulz iteration runs on
**pairs of experts** instead of one expert at a time, and the standard
``w_gate_up`` MoE param is **split** into separate ``w_gate`` and
``w_up`` tensors (concatenated only on the forward pass into the kernel).

Combined effect: with Grug MoE defaults (E=64, `intermediate_dim = d / 2`),
the paired NS matrices for ``w_gate``, ``w_up``, and ``w_down`` are all
square ``(d, d)``. NS converges fastest on square matrices.

Counts per MLP layer:

|                              | per-expert MuonH (concat'd ``w_gate_up``) | paired MuonH (split + pair) |
| ---------------------------- | ----------------------------------------- | ---------------------------- |
| NS ops on ``w_gate_up`` / split gate+up | 64 (one ``(d, d)`` each)         | 32 + 32 (two ``(d, d)`` each)|
| NS ops on ``w_down``         | 64 (one ``(d/2, d)`` each, 1:2 aspect)    | 32 (one ``(d, d)`` each)     |
| total NS ops                 | 128                                       | 96                           |
| matrix aspect                | gate_up square, down 1:2                  | all square                   |

## Implementation

Two changes:

1. `experiments/grug/moe/model.py` (`MoEMLP`): store ``w_gate`` and
   ``w_up`` as separate ``(E, d, i)`` tensors. Concatenate to ``(E, d, 2i)``
   on the forward pass before handing to ``moe_mlp``. Random init is
   identical to the concat'd version (same keys, same `_init_weight`
   calls), so the model is mathematically identical at init time.
2. `experiments/grug/moe/optimizer.py`: new helpers `_pair_axis`,
   `_pair_3d_leading`, `_unpair_to_original`. The pair-axis chooser
   picks whichever of `(A, B)` is smaller so the resulting NS matrix is
   most-square. New `scale_with_grug_muonh_paired` runs
   `_grug_scale_with_muon` in paired space (momentum + hyperball
   normalization both there). New `GrugMoeMuonHPairedConfig` mirrors
   `GrugMoeMuonHConfig` with the paired transform swapped in. Mask
   logic is identical (matrix leaves -> `muonh_paired`, lm head ->
   `adamh`, rest -> `adam`).

New launcher `experiments/grug/moe/muonh_paired_sweep.py` mirrors
`muonh_matrix_sweep.py` but with run-id prefix `muonh-paired-*` and
wandb group `muonh-paired-sweep`. Gate via `MUONH_PAIRED_GATE`,
relaunch suffix via `MUONH_PAIRED_RUN_SUFFIX`.

Tests cover pair/unpair along either axis, paired momentum buffer
shape, paired end-to-end update shape preservation, mask routing, and
sweep step naming. All 19 tests in `tests/test_grug_moe_optimizer.py`
pass.

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

Pass = effective speedup > 1 at both scales.

## Notes / caveats

- Splitting `w_gate_up` changes how AdamH on this branch's model would
  treat that param (per-tensor norm vs concat'd-tensor norm). The
  baseline AdamH README runs were trained with the concat'd storage, so
  use the README numbers directly as the comparison target — don't
  re-run a "split-storage AdamH" baseline on this branch.
- NS scaling factor inside `_grug_scale_with_muon` is
  `sqrt(max(1, fan_out / fan_in))`. For all three paired matrices ``(d, d)``
  this is 1. Per-expert reference: `w_gate_up (d, d)` had scale 1;
  `w_down (d/2, d)` had `sqrt(2) ≈ 1.41`. So paired `w_down` updates
  are smaller in magnitude than per-expert by `sqrt(2)`. LR sweep can
  compensate if needed.
- Pairs adjacent experts by index (no rotation, no learned grouping).
  QB routing balances expert load regardless of pair index.
