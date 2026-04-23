# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dataset metadata helpers for two-phase many-domain candidate summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.nextgen.contracts import LoopConfig
from experiments.domain_phase_mix.static_batch_selection import build_dataset_spec_from_frame
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    STRATIFIED_RUN_NAME,
    create_stratified_weight_config,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

DEFAULT_OBJECTIVE_METRIC = "choice_logprob_norm_mean"
STRATIFIED_60M_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_60m_1p2b"
KNOWN_STRATIFIED_60M_METRICS = {
    "eval/uncheatable_eval/bpb": 1.078909158706665,
}


def build_two_phase_many_loop_config(
    *,
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
    name: str = "two_phase_many_analysis",
) -> LoopConfig:
    """Build the canonical loop metadata for the 39-domain two-phase topology."""
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(name=name)
    phase_fractions = tuple(phase.end_fraction - phase.start_fraction for phase in experiment.phase_schedule.phases)
    domain_token_counts = {domain.name: int(domain.total_weight) for domain in experiment.domains}
    return LoopConfig(
        name=name,
        objective_metric=objective_metric,
        model_names=(),
        domain_token_counts=domain_token_counts,
        phase_fractions=phase_fractions,
        target_budget=experiment.target_budget,
    )


def append_two_phase_many_stratified_baseline(frame: pd.DataFrame, *, objective_metric: str) -> pd.DataFrame:
    """Append the known 60M stratified baseline when the requested metric is available."""
    if objective_metric not in KNOWN_STRATIFIED_60M_METRICS:
        return frame
    if "run_name" in frame.columns and frame["run_name"].astype(str).eq(STRATIFIED_RUN_NAME).any():
        return frame

    stratified = create_stratified_weight_config()
    row: dict[str, float | int | str] = {
        "run_id": int(stratified.run_id),
        "run_name": STRATIFIED_RUN_NAME,
        "source_experiment": STRATIFIED_60M_SOURCE_EXPERIMENT,
        "status": "completed",
        objective_metric: float(KNOWN_STRATIFIED_60M_METRICS[objective_metric]),
    }
    for phase_name, phase_weights in stratified.phase_weights.items():
        for domain_name, weight in phase_weights.items():
            row[f"{phase_name}_{domain_name}"] = float(weight)

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
