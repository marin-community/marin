# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

from experiments.domain_phase_mix.launch_300m_mcq_smooth_proxy_evals import _summarize_mcq_groups


def test_summarize_mcq_groups_single_gold_metrics() -> None:
    rows = [
        {
            "task_alias": "sciq_5shot",
            "doc_id": 0,
            "choice_idx": 0,
            "choice_bytes": 1,
            "is_gold": True,
        },
        {
            "task_alias": "sciq_5shot",
            "doc_id": 0,
            "choice_idx": 1,
            "choice_bytes": 1,
            "is_gold": False,
        },
    ]

    metrics = _summarize_mcq_groups(rows, [(-1.0, False), (-2.0, False)])

    assert metrics["mcq_smooth/sciq_5shot/example_count"] == 1.0
    assert math.isclose(metrics["mcq_smooth/sciq_5shot/choice_prob"], 1.0 / (1.0 + math.exp(-1.0)))
    assert math.isclose(metrics["mcq_smooth/sciq_5shot/nll"], 1.0)
    assert math.isclose(metrics["mcq_smooth/sciq_5shot/bpb"], 1.0 / math.log(2.0))


def test_summarize_mcq_groups_multi_gold_sums_probability_mass() -> None:
    rows = [
        {
            "task_alias": "truthfulqa_mc2_0shot",
            "doc_id": 0,
            "choice_idx": 0,
            "choice_bytes": 1,
            "is_gold": True,
        },
        {
            "task_alias": "truthfulqa_mc2_0shot",
            "doc_id": 0,
            "choice_idx": 1,
            "choice_bytes": 1,
            "is_gold": False,
        },
        {
            "task_alias": "truthfulqa_mc2_0shot",
            "doc_id": 0,
            "choice_idx": 2,
            "choice_bytes": 1,
            "is_gold": True,
        },
    ]

    metrics = _summarize_mcq_groups(rows, [(-1.0, False), (-2.0, False), (-3.0, False)])

    expected_mass = (math.exp(-1.0) + math.exp(-3.0)) / (math.exp(-1.0) + math.exp(-2.0) + math.exp(-3.0))
    assert math.isclose(metrics["mcq_smooth/truthfulqa_mc2_0shot/choice_prob"], expected_mass)
    assert "mcq_smooth/truthfulqa_mc2_0shot/bpb" not in metrics
    assert "mcq_smooth/truthfulqa_mc2_0shot/nll" not in metrics
