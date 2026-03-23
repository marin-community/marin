# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for loading observed schedules from the two-phase many-domain sweep."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights

_THIS_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_CSV_PATH = _THIS_DIR / "exploratory" / "two_phase_many" / "two_phase_many.csv"


def load_two_phase_many_phase_weights(run_name: str) -> dict[str, dict[str, float]]:
    """Load normalized phase weights for one observed run from the two-phase-many table."""
    frame = pd.read_csv(TWO_PHASE_MANY_CSV_PATH)
    matched = frame.loc[frame["run_name"] == run_name]
    if len(matched) != 1:
        raise ValueError(f"Expected exactly one row for observed run {run_name!r}, found {len(matched)}")

    row = matched.iloc[0]
    phase_weights: dict[str, dict[str, float]] = {"phase_0": {}, "phase_1": {}}
    for column, value in row.items():
        if pd.isna(value) or not isinstance(column, str):
            continue
        if column.startswith("phase_0_"):
            phase_weights["phase_0"][column.removeprefix("phase_0_")] = float(value)
        elif column.startswith("phase_1_"):
            phase_weights["phase_1"][column.removeprefix("phase_1_")] = float(value)

    return normalize_phase_weights(phase_weights)
