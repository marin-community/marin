# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dataset metadata helpers for two-phase many-domain candidate summaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.static_batch_selection import build_dataset_spec_from_frame

DEFAULT_OBJECTIVE_METRIC = "choice_logprob_norm_mean"
EPOCH_METADATA_CSV = Path(__file__).resolve().parent / "two_phase_many_epoch_metadata.csv"
STRATIFIED_60M_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_60m_1p2b"
KNOWN_STRATIFIED_60M_METRICS = {
    "eval/uncheatable_eval/bpb": 1.078909158706665,
}


def build_two_phase_many_loop_config(
    *,
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
    name: str = "two_phase_many_analysis",
) -> Any:
    """Build the canonical loop metadata for the 39-domain two-phase topology."""
    metadata = pd.read_csv(EPOCH_METADATA_CSV)
    phase_fractions = (
        float(metadata["phase_0_fraction"].iloc[0]),
        float(metadata["phase_1_fraction"].iloc[0]),
    )
    target_budget = round(
        float(
            np.median(
                metadata["token_count"].to_numpy(dtype=float)
                * metadata["phase_0_epoch_multiplier"].to_numpy(dtype=float)
                / phase_fractions[0]
            )
        )
    )
    return SimpleNamespace(
        name=name,
        objective_metric=objective_metric,
        model_names=(),
        domain_token_counts=dict(zip(metadata["domain_name"], metadata["token_count"].astype(int), strict=True)),
        phase_fractions=phase_fractions,
        target_budget=target_budget,
    )


def append_two_phase_many_stratified_baseline(frame: pd.DataFrame, *, objective_metric: str) -> pd.DataFrame:
    """Append the known 60M stratified baseline when the requested metric is available."""
    if objective_metric not in KNOWN_STRATIFIED_60M_METRICS:
        return frame
    stratified_run_name = "baseline_stratified"
    if "run_name" in frame.columns and frame["run_name"].astype(str).eq(stratified_run_name).any():
        return frame

    first_phase_columns = [
        column for column in frame.columns if column.startswith("phase_0_") and not column.endswith("_epochs")
    ]
    if not first_phase_columns:
        raise ValueError("Cannot append stratified baseline without phase_0 columns")
    domain_names = [column.removeprefix("phase_0_") for column in first_phase_columns]
    uniform_weight = 1.0 / len(domain_names)
    row: dict[str, float | int | str] = {
        "run_id": 3,
        "run_name": stratified_run_name,
        "source_experiment": STRATIFIED_60M_SOURCE_EXPERIMENT,
        "status": "completed",
        objective_metric: float(KNOWN_STRATIFIED_60M_METRICS[objective_metric]),
    }
    for phase_name in ("phase_0", "phase_1"):
        for domain_name in domain_names:
            row[f"{phase_name}_{domain_name}"] = uniform_weight

    augmented = frame.copy()
    for column in row:
        if column not in augmented.columns:
            augmented[column] = np.nan
    augmented.loc[len(augmented)] = {column: row.get(column, np.nan) for column in augmented.columns}
    return augmented


def load_two_phase_many_candidate_summary_spec(
    csv_path: str | Path,
    *,
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
    name: str = "two_phase_many_candidate_summary",
):
    """Load the candidate summary with real epoch metadata from the training topology."""
    frame = pd.read_csv(csv_path)
    if "status" in frame.columns:
        frame = frame[frame["status"] == "completed"].reset_index(drop=True)
    frame = append_two_phase_many_stratified_baseline(frame, objective_metric=objective_metric)
    loop = build_two_phase_many_loop_config(objective_metric=objective_metric, name=name)
    spec = build_dataset_spec_from_frame(
        frame,
        objective_metric=objective_metric,
        name=name,
        loop=loop,
    )
    return frame, spec, loop
