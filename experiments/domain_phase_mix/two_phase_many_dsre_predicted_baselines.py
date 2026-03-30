# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen baselines derived from the saved DS-RE-CEQ predicted optimum."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_TOKEN_COUNTS
from experiments.domain_phase_mix.nextgen.import_sources import NamedWandbRunImportSource
from experiments.domain_phase_mix.two_phase_many_observed_runs import TWO_PHASE_MANY_CSV_PATH

DSRE_PREDICTED_BASELINES_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_dsre_predicted_mmlu_bpb"
DSRE_PREDICTED_OBJECTIVE_METRIC = "lm_eval/mmlu_5shot/bpb"

DSRE_CEQ_PREDICTED_RUN_ID = 246
DSRE_CEQ_PREDICTED_RUN_NAME = "baseline_dsre_ceq_predicted"
DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_ID = 247
DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_NAME = "baseline_dsre_ceq_predicted_quality_collapsed"

_THIS_DIR = Path(__file__).resolve().parent
_EXPLORATORY_DIR = _THIS_DIR / "exploratory" / "two_phase_many"
_OPTIMA_PATH = _EXPLORATORY_DIR / "two_phase_many_model_optima.json"
_SUMMARY_PATH = _EXPLORATORY_DIR / "two_phase_many_dsre_ceq_summary.json"
_CC_DOMAIN_PATTERN = re.compile(r"^(dolma3_cc/.+)_(high|low)$")


def _renormalize_phase(phase_weights: dict[str, float]) -> dict[str, float]:
    total = sum(phase_weights.values())
    if total <= 0:
        raise ValueError(f"Expected positive phase mass, got {total}")
    return {domain_name: float(weight / total) for domain_name, weight in phase_weights.items()}


def _copy_phase_weights(phase_weights: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        phase_name: {domain_name: float(weight) for domain_name, weight in domain_weights.items()}
        for phase_name, domain_weights in phase_weights.items()
    }


def _load_phase_weights() -> dict[str, dict[str, float]]:
    payload = json.loads(_OPTIMA_PATH.read_text())
    phase_weights = _copy_phase_weights(payload["dsre_ceq"])
    return {phase_name: _renormalize_phase(domain_weights) for phase_name, domain_weights in phase_weights.items()}


def _topic_quality_ratios() -> dict[str, tuple[float, float]]:
    counts_by_topic: dict[str, list[float]] = {}
    for domain_name, token_count in TOP_LEVEL_DOMAIN_TOKEN_COUNTS.items():
        match = _CC_DOMAIN_PATTERN.match(domain_name)
        if match is None:
            continue
        topic_prefix, quality = match.groups()
        topic_counts = counts_by_topic.setdefault(topic_prefix, [0.0, 0.0])
        topic_counts[0 if quality == "high" else 1] = float(token_count)

    return {
        topic_prefix: (high_count / (high_count + low_count), low_count / (high_count + low_count))
        for topic_prefix, (high_count, low_count) in counts_by_topic.items()
    }


def _collapse_quality_splits(phase_weights: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    ratios = _topic_quality_ratios()
    collapsed: dict[str, dict[str, float]] = {}
    for phase_name, domain_weights in phase_weights.items():
        phase_result: dict[str, float] = {}
        handled_topics: set[str] = set()
        for domain_name, weight in domain_weights.items():
            match = _CC_DOMAIN_PATTERN.match(domain_name)
            if match is None:
                phase_result[domain_name] = float(weight)
                continue

            topic_prefix, _quality = match.groups()
            if topic_prefix in handled_topics:
                continue

            high_domain = f"{topic_prefix}_high"
            low_domain = f"{topic_prefix}_low"
            total_topic_mass = float(domain_weights.get(high_domain, 0.0) + domain_weights.get(low_domain, 0.0))
            high_ratio, low_ratio = ratios[topic_prefix]
            phase_result[high_domain] = total_topic_mass * high_ratio
            phase_result[low_domain] = total_topic_mass * low_ratio
            handled_topics.add(topic_prefix)

        collapsed[phase_name] = _renormalize_phase(phase_result)
    return collapsed


def _row_to_phase_weights(row: pd.Series) -> dict[str, dict[str, float]]:
    phase_weights: dict[str, dict[str, float]] = {"phase_0": {}, "phase_1": {}}
    for column, value in row.items():
        if pd.isna(value) or not isinstance(column, str):
            continue
        if column.startswith("phase_0_"):
            phase_weights["phase_0"][column.removeprefix("phase_0_")] = float(value)
        elif column.startswith("phase_1_"):
            phase_weights["phase_1"][column.removeprefix("phase_1_")] = float(value)
    return {phase_name: _renormalize_phase(domain_weights) for phase_name, domain_weights in phase_weights.items()}


def _phase_mean_total_variation(left: dict[str, dict[str, float]], right: dict[str, dict[str, float]]) -> float:
    distances: list[float] = []
    for phase_name in left:
        domains = set(left[phase_name]) | set(right[phase_name])
        l1 = sum(
            abs(left[phase_name].get(domain_name, 0.0) - right[phase_name].get(domain_name, 0.0))
            for domain_name in domains
        )
        distances.append(0.5 * l1)
    return float(sum(distances) / len(distances))


def _top_phase_weights(
    phase_weights: dict[str, dict[str, float]], *, top_k: int = 8
) -> dict[str, list[dict[str, float]]]:
    return {
        phase_name: [
            {"domain": domain_name, "weight": float(weight)}
            for domain_name, weight in sorted(domain_weights.items(), key=lambda item: item[1], reverse=True)[:top_k]
        ]
        for phase_name, domain_weights in phase_weights.items()
    }


def _summarize_variant(
    *,
    run_name: str,
    phase_weights: dict[str, dict[str, float]],
    predicted_bpb: float | None,
) -> dict[str, Any]:
    observed = pd.read_csv(TWO_PHASE_MANY_CSV_PATH)
    nearest_run_name = ""
    nearest_tv = float("inf")
    for row in observed.to_dict(orient="records"):
        tv = _phase_mean_total_variation(phase_weights, _row_to_phase_weights(pd.Series(row)))
        if tv < nearest_tv:
            nearest_tv = tv
            nearest_run_name = str(row["run_name"])

    summary: dict[str, Any] = {
        "run_name": run_name,
        "objective_metric": DSRE_PREDICTED_OBJECTIVE_METRIC,
        "nearest_observed_run": nearest_run_name,
        "nearest_observed_tv_distance": nearest_tv,
        "max_phase_weight": max(max(domain.values()) for domain in phase_weights.values()),
        "top_phase_weights": _top_phase_weights(phase_weights),
    }
    if predicted_bpb is not None:
        summary["predicted_bpb"] = predicted_bpb
    return summary


DSRE_CEQ_PREDICTED_PHASE_WEIGHTS = _load_phase_weights()
DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_PHASE_WEIGHTS = _collapse_quality_splits(DSRE_CEQ_PREDICTED_PHASE_WEIGHTS)


def dsre_ceq_predicted_summary() -> dict[str, Any]:
    """Return the saved DS-RE-CEQ optimum summary with geometry diagnostics."""
    payload = json.loads(_SUMMARY_PATH.read_text())["dsre_ceq"]
    return _summarize_variant(
        run_name=DSRE_CEQ_PREDICTED_RUN_NAME,
        phase_weights=DSRE_CEQ_PREDICTED_PHASE_WEIGHTS,
        predicted_bpb=float(payload["predicted_bpb"]),
    ) | {
        "n_params": int(payload["n_params"]),
        "r2": float(payload["r2"]),
        "rmse": float(payload["rmse"]),
        "spearman": float(payload["spearman"]),
        "regret_at_1": float(payload["regret_at_1"]),
    }


def dsre_ceq_predicted_quality_collapsed_summary() -> dict[str, Any]:
    """Return diagnostics for the quality-collapsed DS-RE-CEQ optimum."""
    return _summarize_variant(
        run_name=DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_NAME,
        phase_weights=DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_PHASE_WEIGHTS,
        predicted_bpb=None,
    ) | {
        "derived_from_run_name": DSRE_CEQ_PREDICTED_RUN_NAME,
        "collapse_strategy": "preserve per-topic mass and redistribute using natural high/low token ratios",
    }


def dsre_ceq_predicted_summary_json() -> str:
    """Return a JSON string summarizing both DS-RE predicted baselines."""
    return json.dumps(
        {
            "full_optimum": dsre_ceq_predicted_summary(),
            "quality_collapsed": dsre_ceq_predicted_quality_collapsed_summary(),
        },
        indent=2,
        sort_keys=True,
    )


def create_dsre_ceq_predicted_weight_config(run_id: int = DSRE_CEQ_PREDICTED_RUN_ID) -> WeightConfig:
    """Return the frozen full DS-RE-CEQ predicted optimum baseline."""
    return WeightConfig(run_id=run_id, phase_weights=_copy_phase_weights(DSRE_CEQ_PREDICTED_PHASE_WEIGHTS))


def create_dsre_ceq_predicted_quality_collapsed_weight_config(
    run_id: int = DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_ID,
) -> WeightConfig:
    """Return the quality-collapsed DS-RE-CEQ predicted optimum baseline."""
    return WeightConfig(
        run_id=run_id,
        phase_weights=_copy_phase_weights(DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_PHASE_WEIGHTS),
    )


def create_dsre_ceq_predicted_import_source(
    *,
    local_run_id: int = DSRE_CEQ_PREDICTED_RUN_ID,
    source_experiment: str = DSRE_PREDICTED_BASELINES_SOURCE_EXPERIMENT,
) -> NamedWandbRunImportSource:
    """Return a named-run import source for the full DS-RE-CEQ predicted optimum."""
    return NamedWandbRunImportSource(
        source_experiment=source_experiment,
        local_run_id=local_run_id,
        run_name=DSRE_CEQ_PREDICTED_RUN_NAME,
        phase_weights=DSRE_CEQ_PREDICTED_PHASE_WEIGHTS,
    )


def create_dsre_ceq_predicted_quality_collapsed_import_source(
    *,
    local_run_id: int = DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_ID,
    source_experiment: str = DSRE_PREDICTED_BASELINES_SOURCE_EXPERIMENT,
) -> NamedWandbRunImportSource:
    """Return a named-run import source for the quality-collapsed DS-RE-CEQ optimum."""
    return NamedWandbRunImportSource(
        source_experiment=source_experiment,
        local_run_id=local_run_id,
        run_name=DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_RUN_NAME,
        phase_weights=DSRE_CEQ_PREDICTED_QUALITY_COLLAPSED_PHASE_WEIGHTS,
    )
