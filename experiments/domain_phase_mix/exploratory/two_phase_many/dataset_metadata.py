# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dataset metadata helpers for two-phase many-domain candidate summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.nextgen.contracts import LoopConfig
from experiments.domain_phase_mix.static_batch_selection import build_dataset_spec_from_frame
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

DEFAULT_OBJECTIVE_METRIC = "choice_logprob_norm_mean"


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
    loop = build_two_phase_many_loop_config(objective_metric=objective_metric, name=name)
    spec = build_dataset_spec_from_frame(
        frame,
        objective_metric=objective_metric,
        name=name,
        loop=loop,
    )
    return frame, spec, loop
