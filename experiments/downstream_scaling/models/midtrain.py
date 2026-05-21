# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delphi midtraining variants from issue #4547 (best-LR per mix, 3 base scales).

Each entry is one cell of the 36-cell sweep — base Delphi checkpoint
continued-pretrained on `nemotron_cc_math_v1/4plus` with one of two pretrain-
replay mixes:

- `p33m67` = 33% pretrain replay + 67% math (math-heavy)
- `p67m33` = 67% pretrain replay + 33% math (replay-heavy, retention-positive)

The chosen LR per cell is the best from the 4-LR sweep for that mix per the
issue's findings:

- `p33m67` → `lr=0.67` (best math objective at 1e21 and 1e22; ~tied with 0.83 at 1e20)
- `p67m33` → `lr=0.33` (least Paloma retention damage; ~retention-positive)

Bases (for `initialize_from_hf` in the midtrain runs themselves):

- 1e20 scale → `isoflop-3e+20-d2048-L21-B128` (1.9B params, our `1e20_iso`)
- 1e21 scale → `adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021` (3.4B)
- 1e22 scale → `adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e` (9.7B)

All paths are MARIN_PREFIX-relative; the launcher resolves them at plan time
via `discover_hf_checkpoints` (the actual final step is ~7640 for 1e22, not the
issue's stated 7646).
"""

DELPHI_MIDTRAIN_CHECKPOINTS: dict[str, str] = {
    # 1e20 scale
    "1e20_p33m67_lr0.67": "checkpoints/delphi-1e20-p33m67-4p94b-lr0.67-7c32da/hf",
    "1e20_p67m33_lr0.33": "checkpoints/delphi-1e20-p67m33-4p94b-lr0.33-590ea1/hf",
    # 1e21 scale
    "1e21_p33m67_lr0.67": "checkpoints/delphi-1e21-p33m67-9p25b-lr0.67-9cf8da/hf",
    "1e21_p67m33_lr0.33": "checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64/hf",
    # 1e22 scale
    "1e22_p33m67_lr0.67": "checkpoints/delphi-1e22-p33m67-32p07b-lr0.67-54770ae7/hf",
    "1e22_p67m33_lr0.33": "checkpoints/delphi-1e22-p67m33-32p07b-lr0.33-4e8cc7a7/hf",
}
