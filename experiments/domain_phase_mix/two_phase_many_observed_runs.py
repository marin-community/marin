# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for loading observed schedules from the two-phase many-domain sweep."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from collections.abc import Sequence

import pandas as pd

from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights

_THIS_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_CSV_PATH = _THIS_DIR / "exploratory" / "two_phase_many" / "two_phase_many.csv"
ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240"
CORE_BASELINE_RUN_NAMES = ("baseline_proportional", "baseline_unimax")
REPRESENTATIVE12_PANEL_RUN_NAMES = (
    "baseline_proportional",
    "baseline_unimax",
    "run_00125",
    "run_00213",
    "run_00152",
    "run_00180",
    "run_00018",
    "run_00021",
    "run_00050",
    "run_00090",
    "run_00056",
    "run_00155",
)
_RUN_NAME_PATTERN = re.compile(r"run_\d{5}")


@dataclass(frozen=True)
class ObservedTwoPhaseManyRun:
    """Observed run metadata and normalized phase weights from the two-phase-many table."""

    source_experiment: str
    run_id: int
    run_name: str
    status: str
    phase_weights: dict[str, dict[str, float]]


def _load_two_phase_many_frame() -> pd.DataFrame:
    return pd.read_csv(TWO_PHASE_MANY_CSV_PATH)


def _row_to_phase_weights(row: pd.Series) -> dict[str, dict[str, float]]:
    phase_weights: dict[str, dict[str, float]] = {"phase_0": {}, "phase_1": {}}
    for column, value in row.items():
        if pd.isna(value) or not isinstance(column, str):
            continue
        if column.startswith("phase_0_"):
            phase_weights["phase_0"][column.removeprefix("phase_0_")] = float(value)
        elif column.startswith("phase_1_"):
            phase_weights["phase_1"][column.removeprefix("phase_1_")] = float(value)
    return normalize_phase_weights(phase_weights)


def load_two_phase_many_phase_weights(run_name: str) -> dict[str, dict[str, float]]:
    """Load normalized phase weights for one observed run from the two-phase-many table."""
    frame = _load_two_phase_many_frame()
    matched = frame.loc[frame["run_name"] == run_name]
    if len(matched) != 1:
        raise ValueError(f"Expected exactly one row for observed run {run_name!r}, found {len(matched)}")

    return _row_to_phase_weights(matched.iloc[0])


def load_original_qsplit240_runs() -> list[ObservedTwoPhaseManyRun]:
    """Load the original qsplit240 sampled swarm runs in run-id order."""
    frame = _load_two_phase_many_frame()
    filtered = frame[
        (frame["source_experiment"] == ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT)
        & frame["run_name"].astype(str).str.fullmatch(_RUN_NAME_PATTERN)
    ].copy()
    filtered["run_id"] = filtered["run_id"].astype(int)
    filtered = filtered.sort_values("run_id").reset_index(drop=True)

    observed_runs = [
        ObservedTwoPhaseManyRun(
            source_experiment=str(row["source_experiment"]),
            run_id=int(row["run_id"]),
            run_name=str(row["run_name"]),
            status=str(row["status"]),
            phase_weights=_row_to_phase_weights(pd.Series(row)),
        )
        for row in filtered.to_dict(orient="records")
    ]
    if len(observed_runs) != 238:
        raise ValueError(f"Expected 238 original qsplit240 runs, found {len(observed_runs)}")
    return observed_runs


def load_original_qsplit240_with_core_baselines() -> list[ObservedTwoPhaseManyRun]:
    """Load baseline_proportional, baseline_unimax, and the original qsplit240 runs in run-id order."""
    frame = _load_two_phase_many_frame()
    filtered = frame[
        frame["run_name"].isin(CORE_BASELINE_RUN_NAMES)
        | (
            (frame["source_experiment"] == ORIGINAL_QSPLIT240_SOURCE_EXPERIMENT)
            & frame["run_name"].astype(str).str.fullmatch(_RUN_NAME_PATTERN)
        )
    ].copy()
    filtered["run_id"] = filtered["run_id"].astype(int)
    filtered = filtered.sort_values("run_id").reset_index(drop=True)

    observed_runs = [
        ObservedTwoPhaseManyRun(
            source_experiment=str(row["source_experiment"]),
            run_id=int(row["run_id"]),
            run_name=str(row["run_name"]),
            status=str(row["status"]),
            phase_weights=_row_to_phase_weights(pd.Series(row)),
        )
        for row in filtered.to_dict(orient="records")
    ]
    if len(observed_runs) != 240:
        raise ValueError(f"Expected 240 baseline+qsplit240 runs, found {len(observed_runs)}")
    return observed_runs


def load_original_qsplit240_named_panel(run_names: Sequence[str]) -> list[ObservedTwoPhaseManyRun]:
    """Load a named qsplit240 panel in exactly the requested run-name order."""
    observed_by_name = {run.run_name: run for run in load_original_qsplit240_with_core_baselines()}
    missing = [run_name for run_name in run_names if run_name not in observed_by_name]
    if missing:
        raise ValueError(f"Unknown qsplit240 panel run names: {missing}")
    return [observed_by_name[run_name] for run_name in run_names]
