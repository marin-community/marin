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


# Full 3-scale × 3-mix × 4-LR matrix used by the matrix eval (no SFT). Distinct
# from DELPHI_MIDTRAIN_CHECKPOINTS above: this matrix is keyed on the canonical
# 3e20 ladder point (d2304-L23, ~2.5B), not the off-ladder 1e20-iso d2048-L21
# stand-in. Canonical (state=finished in W&B project marin-community/delphi-
# midtraining) hashes selected per cell after cross-checking W&B status against
# GCS file counts (failed runs left 0 HF subdirs; finished runs left ≥9). See
# the 2026-05-26 plan section of `.agents/logbook/downstream_eval.md`.

# Per-cell finished-in-W&B canonical hashes for 1e21 (9.25B token budget).
_HASHES_1E21: dict[tuple[str, str], str] = {
    ("p33m67", "0.33"): "58ebcb",
    ("p33m67", "0.5"):  "efbc63",
    ("p33m67", "0.67"): "9cf8da",
    ("p33m67", "0.83"): "0cb048",
    ("p50m50", "0.33"): "bccff4",
    ("p50m50", "0.5"):  "973c46",
    ("p50m50", "0.67"): "7e82b3",
    ("p50m50", "0.83"): "f9edd2",
    ("p67m33", "0.33"): "ab4e64",
    ("p67m33", "0.5"):  "114e49",
    ("p67m33", "0.67"): "ecbd27",
    ("p67m33", "0.83"): "a1a261",
}

# Per-cell finished-in-W&B canonical hashes for 1e22 (32.07B token budget).
# Long-hash variants are the canonical ones for the 6 cells with W&B duplicates;
# short-hash variants for those cells are W&B-failed stubs with 0 HF subdirs.
_HASHES_1E22: dict[tuple[str, str], str] = {
    ("p33m67", "0.33"): "e9132105",
    ("p33m67", "0.5"):  "0eeca70d",
    ("p33m67", "0.67"): "54770ae7",
    ("p33m67", "0.83"): "78fd44",
    ("p50m50", "0.33"): "c43ada",
    ("p50m50", "0.5"):  "ecfa99",
    ("p50m50", "0.67"): "e78260",
    ("p50m50", "0.83"): "3c9f70",
    ("p67m33", "0.33"): "4e8cc7a7",
    ("p67m33", "0.5"):  "f60cb12a",
    ("p67m33", "0.67"): "3c17740e",
    ("p67m33", "0.83"): "d35daa",
}

_MIXES: tuple[str, ...] = ("p33m67", "p50m50", "p67m33")
_LRS: tuple[str, ...] = ("0.33", "0.5", "0.67", "0.83")
# `lr0.5` in float form lands in GCS paths as `lr50` for k0p20-naming scales;
# kept as `0.5` for 9p25b/32p07b-naming scales.
_LR_K0P20_SUFFIX: dict[str, str] = {"0.33": "33", "0.5": "50", "0.67": "67", "0.83": "83"}


def _build_midtrain_matrix() -> dict[str, str]:
    matrix: dict[str, str] = {}
    # 3e20 ladder point — d2304-L23 ~2.5B params, k0p20 token budget, replicate a001.
    for mix in _MIXES:
        for lr in _LRS:
            slug = f"3e20_{mix}_lr{lr}"
            matrix[slug] = f"checkpoints/delphi-3e20-{mix}-k0p20-lr{_LR_K0P20_SUFFIX[lr]}-a001/hf"
    # 1e21 ladder point — 3.4B params, 9p25b token budget.
    for mix in _MIXES:
        for lr in _LRS:
            slug = f"1e21_{mix}_lr{lr}"
            matrix[slug] = f"checkpoints/delphi-1e21-{mix}-9p25b-lr{lr}-{_HASHES_1E21[(mix, lr)]}/hf"
    # 1e22 ladder point — 9.7B params, 32p07b token budget.
    for mix in _MIXES:
        for lr in _LRS:
            slug = f"1e22_{mix}_lr{lr}"
            matrix[slug] = f"checkpoints/delphi-1e22-{mix}-32p07b-lr{lr}-{_HASHES_1E22[(mix, lr)]}/hf"
    return matrix


DELPHI_MIDTRAIN_MATRIX: dict[str, str] = _build_midtrain_matrix()
