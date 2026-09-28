# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_no_l2_benchmark_aggregates import (
    GSM8K_ACC,
    HUMANEVAL_PASS,
    MMLU_ACC,
    MMLU_BPB,
    OLMO_BASE_EASY_OVERLAP_MACRO_BPB,
    _add_aggregate_targets,
    merge_overlap_metric_columns,
)
from experiments.evals.olmo_base_easy_overlap import OLMO_BASE_EASY_OVERLAP_SUPPORTED_BPB_KEYS


def test_add_aggregate_targets_masks_stale_partial_olmo_base_easy_overlap_macro_bpb():
    frame = pd.DataFrame(
        {
            MMLU_ACC: [0.30, 0.31],
            GSM8K_ACC: [0.10, 0.11],
            HUMANEVAL_PASS: [0.05, 0.06],
            MMLU_BPB: [1.8, 1.7],
            OLMO_BASE_EASY_OVERLAP_MACRO_BPB: [1.0, 2.0],
            "lm_eval/olmo_base_easy_overlap/task_count": [
                float(len(OLMO_BASE_EASY_OVERLAP_SUPPORTED_BPB_KEYS) - 1),
                float(len(OLMO_BASE_EASY_OVERLAP_SUPPORTED_BPB_KEYS)),
            ],
        }
    )

    targets = _add_aggregate_targets(frame)

    assert math.isnan(targets.loc[0, OLMO_BASE_EASY_OVERLAP_MACRO_BPB])
    assert targets.loc[1, OLMO_BASE_EASY_OVERLAP_MACRO_BPB] == 2.0


def test_merge_overlap_metric_columns_prefers_rerun_metrics_over_stale_registry_metrics():
    frame = pd.DataFrame(
        {
            "run_name": ["baseline_proportional"],
            OLMO_BASE_EASY_OVERLAP_MACRO_BPB: [9.9],
            "lm_eval/olmo_base_easy_overlap/task_count": [4.0],
        }
    )
    overlap = pd.DataFrame(
        {
            "run_name": ["baseline_proportional"],
            OLMO_BASE_EASY_OVERLAP_MACRO_BPB: [1.8],
            "lm_eval/olmo_base_easy_overlap/task_count": [10.0],
        }
    )

    merged = merge_overlap_metric_columns(
        frame,
        overlap,
        [OLMO_BASE_EASY_OVERLAP_MACRO_BPB, "lm_eval/olmo_base_easy_overlap/task_count"],
    )

    assert merged.loc[0, OLMO_BASE_EASY_OVERLAP_MACRO_BPB] == 1.8
    assert merged.loc[0, "lm_eval/olmo_base_easy_overlap/task_count"] == 10.0
