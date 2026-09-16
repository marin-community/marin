# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DS-RE predicted baseline realized on a topic-collapsed CC topology."""

from __future__ import annotations

import json
import re
from typing import Any

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.two_phase_many_dsre_predicted_baselines import (
    DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_PHASE_WEIGHTS,
    _copy_phase_weights,
    _phase_mean_total_variation,
    _summarize_variant,
)

DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_RUN_ID = 250
DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_RUN_NAME = "baseline_dsre_ceq_predicted_topic_collapsed"
DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_dsre_predicted_mmlu_bpb_topic_collapsed"
)
_CC_DOMAIN_PATTERN = re.compile(r"^(dolma3_cc/.+)_(high|low)$")


def _collapse_cc_topics(phase_weights: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    collapsed: dict[str, dict[str, float]] = {}
    for phase_name, domain_weights in phase_weights.items():
        phase_result: dict[str, float] = {}
        for domain_name, weight in domain_weights.items():
            match = _CC_DOMAIN_PATTERN.match(domain_name)
            if match is None:
                phase_result[domain_name] = float(weight)
                continue
            topic_name, _quality = match.groups()
            phase_result[topic_name] = phase_result.get(topic_name, 0.0) + float(weight)
        collapsed[phase_name] = phase_result
    return collapsed


DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_PHASE_WEIGHTS = _collapse_cc_topics(
    DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_PHASE_WEIGHTS
)


def _realized_top_level_phase_weights() -> dict[str, dict[str, float]]:
    return _copy_phase_weights(DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_PHASE_WEIGHTS)


def dsre_ceq_predicted_topic_collapsed_summary() -> dict[str, Any]:
    """Return diagnostics for the topic-collapsed DS-RE projected baseline."""
    realized_top_level = _realized_top_level_phase_weights()
    equivalent_tv = _phase_mean_total_variation(realized_top_level, DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_PHASE_WEIGHTS)
    summary = _summarize_variant(
        run_name=DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_RUN_NAME,
        phase_weights=realized_top_level,
        predicted_bpb=None,
    )
    return summary | {
        "collapse_strategy": (
            "sum CC high/low masses into one topic domain; internal component weights follow token counts"
        ),
        "n_collapsed_domains": len(DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_PHASE_WEIGHTS["phase_0"]),
        "equivalent_flat_run_name": "baseline_dsre_ceq_predicted_quality_collapsed",
        "equivalent_flat_tv_distance": equivalent_tv,
    }


def dsre_ceq_predicted_topic_collapsed_summary_json() -> str:
    """Return the topic-collapsed summary JSON."""
    return json.dumps(dsre_ceq_predicted_topic_collapsed_summary(), indent=2, sort_keys=True)


def create_dsre_ceq_predicted_topic_collapsed_weight_config(
    run_id: int = DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_RUN_ID,
) -> WeightConfig:
    """Return the DS-RE optimum on the topic-collapsed topology."""
    return WeightConfig(
        run_id=run_id,
        phase_weights=_copy_phase_weights(DSRE_CEQ_PREDICTED_TOPIC_COLLAPSED_PHASE_WEIGHTS),
    )
