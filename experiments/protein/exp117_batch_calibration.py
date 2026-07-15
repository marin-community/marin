# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU batch calibration: tune ``batch_calibration.correction_factor`` against measured ceilings.

Ground truth is the measured per-chip microbatch ceiling for each slice (largest global batch that
trains with ``per_device_parallelism = -1``, divided by chip count) in :data:`CEILINGS`. The
estimator ``tpu_batch_config`` exposes one knob -- ``correction_factor`` -- that scales its byte
estimate. This module finds the single correction factor that best makes the estimator's predicted
per-chip microbatch match the measurement, so the heuristic can predict accumulation for any slice.

Reports: measured ceilings; the correction factor each family needs; the single recommended value; and
the per-slice ``(pdp, accum)`` the estimator then predicts vs. measured, uncorrected (factor 1.0) and
calibrated. ``--hbm`` adds peak HBM utilization from W&B as corroboration.
Run: ``python -m experiments.protein.exp117_batch_calibration`` (see exp117_batch_calibration.md).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import wandb

from experiments.coral.batch_calibration import (
    adam_optimizer_bytes,
    batch_memory_bytes,
    dense_transformer_bytes,
    tpu_batch_config,
)
from experiments.protein.exp117_sweep import MODEL_CONFIG, SEQ_LEN, VOCAB_SIZE

WANDB_PATH = "eric-czech/marin"
UNCORRECTED = 1.0  # correction_factor = 1.0 applies the raw byte estimate (batch_calibration default)
CALIB_TARGET = 2048  # large enough that pdp is never capped; per-chip prediction is stable for target >= 512
TARGET_BATCH = 512  # example target for the (pdp, accum) comparison table
CORRECTION_GRID = [round(0.01 * i, 2) for i in range(1, 201)]  # 0.01 .. 2.00
HBM_METRIC = "hbmMemoryUsage"  # W&B system metric (percent, one series per chip): system.tpu.<i>.hbmMemoryUsage
RUN_SUFFIX = "-oh1-v2-b{batch}$"  # W&B run-name suffix of the max-fit probes (historical smoke v2 run set)


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

    @property
    def family(self) -> str:
        return "v5e" if self.tpu.startswith("v5litepod") else self.tpu.split("-")[0]


# Measured ground truth. Per-chip ceiling is constant within a family (v5e=4, v6e=16, v5p=32) and
# scales with chip count -- an internal consistency check on the measurement.
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


def estimated_batch_bytes(global_batch: int, correction_factor: float) -> int:
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
        correction_factor=correction_factor,
    )


def predicted_per_chip(tpu: str, chips: int, correction_factor: float, target: int = CALIB_TARGET) -> int:
    """Per-chip microbatch the estimator allows at ``correction_factor`` (``pdp``; -1 -> target/chips)."""
    pdp, _ = tpu_batch_config(tpu, target, estimated_batch_bytes(target, correction_factor))
    return target // chips if pdp == -1 else pdp


def accum(per_chip: int, chips: int, target: int) -> int:
    """Gradient-accumulation steps for ``per_chip`` microbatch to reach ``target`` global batch."""
    return max(1, target // (per_chip * chips))


def family_correction_range(family: str) -> tuple[float | None, float | None]:
    """Correction-factor interval on :data:`CORRECTION_GRID` where the estimator reproduces the family's
    measured per-chip ceiling (evaluated on the smallest slice; per-chip is constant within a family)."""
    c = min((c for c in CEILINGS if c.family == family), key=lambda c: c.chips)
    lo = hi = None
    for cf in CORRECTION_GRID:
        if predicted_per_chip(c.tpu, c.chips, cf) == c.per_chip_ceiling:
            lo = cf if lo is None else lo
            hi = cf
    return lo, hi


def recommended_correction_factor() -> float:
    """Smallest correction factor that never over-predicts on any slice (predicted per-chip <= measured,
    so the estimator never under-accumulates into an OOM). Minimizes wasted accumulation subject to that."""
    for cf in CORRECTION_GRID:
        if all(predicted_per_chip(c.tpu, c.chips, cf) <= c.per_chip_ceiling for c in CEILINGS):
            return cf
    return CORRECTION_GRID[-1]


def hbm_peak_util(tpu: str, batch: int, chips: int) -> tuple[float | None, bool]:
    """Peak ``hbmMemoryUsage`` (%) over all chips and steps of the completed max-fit run(s)."""
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


def print_ceilings(with_hbm: bool) -> None:
    print("Measured ceilings (ground truth) — largest global batch with pdp=-1, no accumulation:\n")
    ranges = {fam: family_correction_range(fam) for fam in ("v5e", "v6e", "v5p")}
    head = f"{'slice':13s} {'chips':>5s} {'GiB/chip':>8s} {'max batch':>9s} {'per-chip':>8s} {'corr factor':>12s}"
    print(head + ("  peak HBM%" if with_hbm else ""))
    for c in CEILINGS:
        lo, hi = ranges[c.family]
        rng = f"{lo:.2f}-{hi:.2f}"
        line = (
            f"{c.tpu:13s} {c.chips:>5d} {c.hbm_gib_per_chip:>8d} {c.max_batch:>9d} "
            f"{c.per_chip_ceiling:>8d} {rng:>12s}"
        )
        if with_hbm:
            peak, _ = hbm_peak_util(c.tpu, c.max_batch, c.chips)
            line += f"  {'n/a' if peak is None else f'{peak:.1f}':>8s}"
        print(line)


def print_calibration() -> None:
    # per-family correction-factor ranges are shown in the ceilings table; here just the single value
    rec = recommended_correction_factor()
    print(f"\nRecommended single correction_factor = {rec}  (smallest that never over-predicts on any slice).")


def print_config_table(correction_factor: float) -> None:
    print(f"\nConfig to reach global batch {TARGET_BATCH} at correction_factor {correction_factor:g}:\n")
    cols = ["slice", "chips", "per-chip meas", "per-chip pred", "accum meas", "accum pred", "fit"]
    print(" ".join(f"{h:>13s}" for h in cols))
    for c in CEILINGS:
        meas = c.per_chip_ceiling
        pred = predicted_per_chip(c.tpu, c.chips, correction_factor)
        verdict = "exact" if pred == meas else ("OOM" if pred > meas else f"{meas // pred if pred else 0}x accum")
        row = [
            c.tpu,
            c.chips,
            meas,
            pred,
            accum(meas, c.chips, TARGET_BATCH),
            accum(pred, c.chips, TARGET_BATCH),
            verdict,
        ]
        print(" ".join(f"{v!s:>13s}" for v in row))


def main() -> None:
    parser = argparse.ArgumentParser(description="TPU batch calibration")
    parser.add_argument("--hbm", action="store_true", help="add peak HBM utilization from W&B (slow)")
    args = parser.parse_args()
    print_ceilings(with_hbm=args.hbm)
    print_calibration()
    print_config_table(UNCORRECTED)
    print_config_table(recommended_correction_factor())


if __name__ == "__main__":
    main()
