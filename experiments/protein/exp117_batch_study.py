# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""exp117 batch study: empirical max-batch vs the HBM estimator.

For each TPU slice we measured the largest power-of-2 GLOBAL batch that trains with
``per_device_parallelism = -1`` (the whole per-chip batch in one microbatch, no gradient
accumulation). This module turns those measurements into the study's two tables:

1. ``print_ceiling_summary`` -- the measured max batch per slice, the per-chip ceiling, and the
   HBM ``overhead`` factor that makes the sweep's estimator agree with the measurement.
2. ``print_target_table`` -- for a target global batch that does NOT fit in one microbatch on any
   slice (default 512), the ``(pdp, grad_accum)`` implied by the MEASURED ceiling versus the
   ``(pdp, grad_accum)`` PREDICTED by :func:`tpu_batch_config`, plus the peak HBM utilization each
   slice actually reached (from W&B).

The measured ceilings in :data:`CEILINGS` are the study's ground truth; everything else is derived.
Run: ``python -m experiments.protein.exp117_batch_study`` (see exp117_batch_study.md for method).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import wandb

from experiments.coral.batch_config import (
    adam_optimizer_bytes,
    batch_memory_bytes,
    dense_transformer_bytes,
    tpu_batch_config,
)
from experiments.protein.exp117_sweep import MODEL_CONFIG, SEQ_LEN, VOCAB_SIZE

WANDB_PATH = "eric-czech/marin"
TARGET_BATCH = 512  # a batch too large to fit in one microbatch on any slice, so every slice needs accumulation
PREDICT_OVERHEAD = 1.0  # the estimator's default HBM overhead factor (sweep default)
HBM_METRIC = "hbmMemoryUsage"  # W&B system metric, a percentage, one series per chip: system.tpu.<i>.hbmMemoryUsage
# W&B run-name suffix of a max-fit probe: the sweep folds slice + overhead tag + smoke version +
# batch into the run id (see exp117_sweep.smoke_shape). "-oh1-v2-b<batch>" is overhead 1.0, smoke v2.
RUN_SUFFIX = "-oh1-v2-b{batch}$"


@dataclass(frozen=True)
class SliceCeiling:
    """One slice's measured result. ``max_batch`` is the largest global batch that fit with pdp=-1."""

    tpu: str
    chips: int
    hbm_gib_per_chip: int
    max_batch: int

    @property
    def per_chip_ceiling(self) -> int:
        return self.max_batch // self.chips


# Measured ground truth. Per-chip ceiling is constant within a family (v5e=4, v6e=16, v5p=32) and
# scales with chip count -- a strong internal-consistency check on the measurement.
CEILINGS: list[SliceCeiling] = [
    SliceCeiling("v5litepod-4", 4, 16, 16),
    SliceCeiling("v5litepod-8", 8, 16, 32),
    SliceCeiling("v5litepod-16", 16, 16, 64),
    SliceCeiling("v6e-4", 4, 32, 64),
    SliceCeiling("v6e-8", 8, 32, 128),
    SliceCeiling("v6e-16", 16, 32, 256),
    SliceCeiling("v5p-8", 4, 95, 128),
    SliceCeiling("v5p-16", 8, 95, 256),
    SliceCeiling("v5p-32", 16, 95, 512),
]


def estimated_batch_bytes(global_batch: int, overhead: float) -> int:
    """Aggregate HBM the estimator predicts for a full ``global_batch`` (params + Adam state + acts)."""
    params = MODEL_CONFIG.total_trainable_params(VOCAB_SIZE)
    param_bytes, activation_bytes = dense_transformer_bytes(
        parameter_count=params,
        batch_size=global_batch,
        seq_len=SEQ_LEN,
        hidden_dim=MODEL_CONFIG.hidden_dim,
        intermediate_dim=MODEL_CONFIG.intermediate_dim,
        num_layers=MODEL_CONFIG.num_layers,
    )
    return batch_memory_bytes(
        param_bytes=param_bytes,
        optimizer_bytes=adam_optimizer_bytes(params),
        activation_bytes=activation_bytes,
        overhead_factor=overhead,
    )


def measured_config(c: SliceCeiling, target: int) -> tuple[int, int]:
    """``(pdp, grad_accum)`` the MEASURED ceiling implies to run ``target`` global batch.

    Uses the largest microbatch the slice actually fit; ``pdp = -1`` when the whole target batch
    fits per-chip in one microbatch.
    """
    per_chip_needed = target // c.chips
    if c.per_chip_ceiling >= per_chip_needed:
        return -1, 1
    pdp = c.per_chip_ceiling
    return pdp, target // (pdp * c.chips)


def predicted_config(tpu: str, target: int, overhead: float) -> tuple[int, int]:
    """``(pdp, grad_accum)`` the estimator predicts for ``target`` global batch at ``overhead``."""
    return tpu_batch_config(tpu, target, estimated_batch_bytes(target, overhead))


def matching_overhead(c: SliceCeiling) -> str:
    """Coarsely, the largest overhead at which the estimator's per-chip prediction still reaches the
    measured ceiling (searched on a log2 grid). '>=X' means even the smallest tried overhead
    under-predicts (estimator saturates at pdp=-1 for the fixed global-128 basis)."""
    grid = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    for oh in grid:
        pdp, _ = tpu_batch_config(c.tpu, 128, estimated_batch_bytes(128, oh))
        pred_per_chip = (128 // c.chips) if pdp == -1 else pdp
        if pred_per_chip >= c.per_chip_ceiling:
            return f"{oh:g}"
    return f"<{grid[-1]:g}"


def hbm_peak_util(tpu: str, batch: int, chips: int) -> tuple[float | None, bool]:
    """Peak ``hbmMemoryUsage`` (%) over all chips and steps for the max-fit run(s).

    Fans over every region variant of the run (they horse-raced; any that trained is valid) and
    takes the overall max. Returns ``(util, from_finished)`` where ``from_finished`` is False if no
    completed run was found and provisional data from an unfinished run was used instead.
    """
    api = wandb.Api()
    runs = list(api.runs(WANDB_PATH, filters={"display_name": {"$regex": f"-{tpu}{RUN_SUFFIX.format(batch=batch)}"}}))
    finished = [r for r in runs if r.state == "finished"]
    pool = finished if finished else runs
    peak: float | None = None
    for run in pool:
        try:
            df = run.history(stream="system", samples=100000)
        except Exception:
            continue
        cols = [f"system.tpu.{i}.{HBM_METRIC}" for i in range(chips) if f"system.tpu.{i}.{HBM_METRIC}" in df.columns]
        vals = [float(df[col].max()) for col in cols if df[col].notna().any()]
        if vals:
            peak = max(vals) if peak is None else max(peak, max(vals))
    return peak, bool(finished)


def print_ceiling_summary() -> None:
    print("Measured max global batch (pdp=-1, no grad accum) and the overhead that reproduces it:\n")
    print(f"{'slice':13s} {'chips':>5s} {'HBM/ch':>6s} {'maxBatch':>8s} {'perChip':>7s} {'~overhead':>9s}")
    for c in CEILINGS:
        print(
            f"{c.tpu:13s} {c.chips:>5d} {c.hbm_gib_per_chip:>6d} {c.max_batch:>8d} "
            f"{c.per_chip_ceiling:>7d} {matching_overhead(c):>9s}"
        )


def print_target_table(target: int, overhead: float, with_hbm: bool) -> None:
    print(f"\nTarget global batch = {target}; predicted at overhead = {overhead:g}\n")
    cols = ["slice", "chips", "HBMtot", "HBM/ch", "maxUtil%", "maxB", "pdp_a", "gac_a", "pdp_p", "gac_p"]
    print(" ".join(f"{h:>11s}" for h in cols))
    any_provisional = False
    for c in CEILINGS:
        pdp_a, gac_a = measured_config(c, target)
        pdp_p, gac_p = predicted_config(c.tpu, target, overhead)
        util = "-"
        if with_hbm:
            peak, finished = hbm_peak_util(c.tpu, c.max_batch, c.chips)
            if peak is not None and not finished:
                any_provisional = True
            util = "n/a" if peak is None else (f"{peak:.1f}" if finished else f"{peak:.1f}*")
        row = [
            c.tpu,
            c.chips,
            c.chips * c.hbm_gib_per_chip,
            c.hbm_gib_per_chip,
            util,
            c.max_batch,
            pdp_a,
            gac_a,
            pdp_p,
            gac_p,
        ]
        print(" ".join(f"{v!s:>11s}" for v in row))
    if any_provisional:
        print("\n* = from an unfinished (killed/crashed) run; peak is provisional until a completed run exists.")


def main() -> None:
    parser = argparse.ArgumentParser(description="exp117 batch study analysis")
    parser.add_argument("--target", type=int, default=TARGET_BATCH, help="target global batch for the comparison table")
    parser.add_argument("--overhead", type=float, default=PREDICT_OVERHEAD, help="HBM overhead for the prediction")
    parser.add_argument("--no-hbm", action="store_true", help="skip the W&B HBM-utilization query")
    args = parser.parse_args()
    print_ceiling_summary()
    print_target_table(args.target, args.overhead, with_hbm=not args.no_hbm)


if __name__ == "__main__":
    main()
