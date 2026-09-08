# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit an Olmix log-linear baseline on the replicated fixed-subset SL-Verb seedpanel."""

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_sl_verb import (
    OlmixSlVerbFitResult,
    load_fit_from_results,
)

RUN_ID = 244
RUN_NAME = "baseline_olmix_loglinear_sl_verb_choice_logprob_norm_seedpanel_n3"
SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_olmix_sl_verb_choice_logprob_norm_seedpanel_n3"
SOURCE_RESULTS_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_fixed_subset_seedpanel_n3_mmlu_sl_verb_rerun"
)
CANDIDATE_GROUP_COLUMN = "candidate_run_name"


def load_fit_from_seedpanel_results(
    *,
    natural_proportions: np.ndarray,
    phase_fractions: np.ndarray,
) -> OlmixSlVerbFitResult:
    """Fit the SL-Verb Olmix baseline from replicated candidate means."""
    return load_fit_from_results(
        source_experiment=SOURCE_RESULTS_EXPERIMENT,
        natural_proportions=natural_proportions,
        phase_fractions=phase_fractions,
        candidate_group_column=CANDIDATE_GROUP_COLUMN,
    )
