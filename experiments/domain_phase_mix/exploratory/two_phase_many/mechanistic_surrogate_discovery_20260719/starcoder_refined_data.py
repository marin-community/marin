# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load the persisted refined StarCoder WSD surface for discovery audits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_partially_pooled_phase_bowls as pooled,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFINED_WSD80_DATA = (
    SCRIPT_DIR.parent / "reference_outputs/starcoder_wsd80_surface_refined_20260714/wsd80_observed_metrics.csv"
)
EXPECTED_COORDINATES = 107
STARCODER_DOMAINS = ["nemotron_full", "starcoder"]


def load_refined_wsd80_starcoder(cosine: pooled.Dataset) -> pooled.Dataset:
    """Return all 107 persisted WSD 80/20 coordinates with source checks."""

    frame = pd.read_csv(REFINED_WSD80_DATA)
    coordinate_columns = ["phase_0_starcoder", "phase_1_starcoder"]
    required = [*coordinate_columns, "wsd80_bpb", "wandb_run_id", "wandb_state"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Refined WSD surface is missing columns: {missing}")
    if len(frame) != EXPECTED_COORDINATES:
        raise ValueError(f"Expected {EXPECTED_COORDINATES} refined WSD coordinates, got {len(frame)}")
    if frame.duplicated(coordinate_columns).any():
        raise ValueError("Refined WSD surface contains duplicate coordinates")
    if frame[required].isna().any().any():
        raise ValueError("Refined WSD surface contains incomplete persisted observations")
    if not frame["wandb_state"].eq("finished").all():
        raise ValueError("Refined WSD surface contains unfinished observations")

    p0 = frame["phase_0_starcoder"].to_numpy(dtype=float)
    p1 = frame["phase_1_starcoder"].to_numpy(dtype=float)
    if np.any((p0 < 0.0) | (p0 > 1.0) | (p1 < 0.0) | (p1 > 1.0)):
        raise ValueError("Refined WSD surface contains invalid mixture weights")
    weights = np.stack(
        [np.column_stack([1.0 - p0, p0]), np.column_stack([1.0 - p1, p1])],
        axis=1,
    )
    return pooled.Dataset(
        name="starcoder_wsd_80_20",
        frame=frame,
        y=frame["wsd80_bpb"].to_numpy(dtype=float),
        weights=weights,
        c0=np.asarray(cosine.c0 * (0.8 / 0.5), dtype=float),
        c1=np.asarray(cosine.c1 * (0.2 / 0.5), dtype=float),
        domain_names=list(STARCODER_DOMAINS),
    )
