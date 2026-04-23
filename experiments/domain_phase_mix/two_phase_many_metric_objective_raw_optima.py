# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw-optimum deployments for metric-specific no-L2 GRP fits."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
import json
from pathlib import Path
from typing import Any, Literal

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import DOMAIN_NAMES

METRIC_OBJECTIVE_RAW_OPTIMA_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_metric_objective_grp_no_l2_raw_optima"
)
METRIC_OBJECTIVE_RAW_OPTIMA_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "metric_registry"
    / "grp_no_l2_metric_objective_fits"
)
METRIC_OBJECTIVE_RAW_OPTIMA_SUMMARY_CSV = METRIC_OBJECTIVE_RAW_OPTIMA_DIR / "summary.csv"
METRIC_OBJECTIVE_RAW_OPTIMA_RUN_IDS = {
    "paloma_macro_bpb": 490,
    "piqa_5shot_choice_logprob": 491,
}
METRIC_OBJECTIVE_RAW_OPTIMA_DEFAULT_SLUGS = ("paloma_macro_bpb", "piqa_5shot_choice_logprob")


@dataclass(frozen=True)
class MetricObjectiveRawOptimumSummary:
    """Summary for one metric-specific no-L2 GRP raw optimum deployment."""

    slug: str
    run_id: int
    run_name: str
    metric_key: str
    display_name: str
    lower_is_better: bool
    predicted_optimum_metric: float
    best_observed_run_name: str
    best_observed_metric: float
    predicted_observed_run_name: str
    predicted_observed_metric: float
    predicted_observed_regret: float
    raw_nearest_observed_run_name: str
    raw_nearest_observed_metric: float
    raw_nearest_observed_regret: float
    raw_nearest_observed_tv: float
    oof_rmse: float
    oof_r2: float
    oof_spearman: float
    phase_weights: dict[str, dict[str, float]]


def parse_metric_objective_raw_optimum_slugs(spec: str) -> tuple[str, ...]:
    """Parse a comma-separated metric-objective raw-optimum slug spec."""
    cleaned = spec.strip().lower()
    if cleaned == "all":
        return METRIC_OBJECTIVE_RAW_OPTIMA_DEFAULT_SLUGS
    slugs = tuple(part.strip() for part in spec.split(",") if part.strip())
    if not slugs:
        raise ValueError("metric-objective slug spec must not be empty")
    invalid = [slug for slug in slugs if slug not in METRIC_OBJECTIVE_RAW_OPTIMA_RUN_IDS]
    if invalid:
        raise ValueError(f"Unsupported metric-objective raw-optimum slugs: {invalid}")
    return slugs


def metric_objective_raw_optimum_run_name(slug: str) -> str:
    """Return the canonical run name for one metric-specific no-L2 GRP raw optimum."""
    if slug == "paloma_macro_bpb":
        return "baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_paloma_macro_bpb"
    if slug == "piqa_5shot_choice_logprob":
        return "baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_piqa_5shot_choice_logprob"
    raise ValueError(f"Unsupported metric-objective raw-optimum slug: {slug}")


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Expected serialized bool, got {value!r}")


@cache
def _summary_rows() -> dict[str, dict[str, str]]:
    if not METRIC_OBJECTIVE_RAW_OPTIMA_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing metric-objective fit summary: {METRIC_OBJECTIVE_RAW_OPTIMA_SUMMARY_CSV}")
    with METRIC_OBJECTIVE_RAW_OPTIMA_SUMMARY_CSV.open(newline="") as handle:
        rows = {row["slug"]: row for row in csv.DictReader(handle)}
    missing = set(METRIC_OBJECTIVE_RAW_OPTIMA_RUN_IDS).difference(rows)
    if missing:
        raise ValueError(f"Missing metric-objective summaries for slugs: {sorted(missing)}")
    return rows


def _weights_csv_for_slug(slug: str) -> Path:
    return METRIC_OBJECTIVE_RAW_OPTIMA_DIR / f"{slug}_raw_optimum_weights.csv"


def _phase_weights_from_csv(slug: str) -> dict[str, dict[str, float]]:
    weights_csv = _weights_csv_for_slug(slug)
    if not weights_csv.exists():
        raise FileNotFoundError(f"Missing metric-objective raw optimum weights: {weights_csv}")

    phase0: dict[str, float] = {}
    phase1: dict[str, float] = {}
    with weights_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            domain_name = row["domain_name"]
            phase0[domain_name] = float(row["phase0_weight"])
            phase1[domain_name] = float(row["phase1_weight"])

    expected = set(DOMAIN_NAMES)
    actual = set(phase0)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(f"Domain mismatch in {weights_csv}: missing={missing}, extra={extra}")

    phase0_sum = sum(phase0.values())
    phase1_sum = sum(phase1.values())
    if abs(phase0_sum - 1.0) > 1e-6 or abs(phase1_sum - 1.0) > 1e-6:
        raise ValueError(f"Weights in {weights_csv} do not sum to 1: phase0={phase0_sum}, phase1={phase1_sum}")

    return {
        "phase_0": {domain_name: phase0[domain_name] for domain_name in DOMAIN_NAMES},
        "phase_1": {domain_name: phase1[domain_name] for domain_name in DOMAIN_NAMES},
    }


def _summary_to_dict(summary: MetricObjectiveRawOptimumSummary) -> dict[str, Any]:
    return {
        "slug": summary.slug,
        "run_id": summary.run_id,
        "run_name": summary.run_name,
        "metric_key": summary.metric_key,
        "display_name": summary.display_name,
        "lower_is_better": summary.lower_is_better,
        "predicted_optimum_metric": summary.predicted_optimum_metric,
        "best_observed_run_name": summary.best_observed_run_name,
        "best_observed_metric": summary.best_observed_metric,
        "predicted_observed_run_name": summary.predicted_observed_run_name,
        "predicted_observed_metric": summary.predicted_observed_metric,
        "predicted_observed_regret": summary.predicted_observed_regret,
        "raw_nearest_observed_run_name": summary.raw_nearest_observed_run_name,
        "raw_nearest_observed_metric": summary.raw_nearest_observed_metric,
        "raw_nearest_observed_regret": summary.raw_nearest_observed_regret,
        "raw_nearest_observed_tv": summary.raw_nearest_observed_tv,
        "oof_rmse": summary.oof_rmse,
        "oof_r2": summary.oof_r2,
        "oof_spearman": summary.oof_spearman,
        "phase_weights": summary.phase_weights,
    }


@cache
def metric_objective_raw_optimum_summaries(
    slugs: tuple[str, ...] = METRIC_OBJECTIVE_RAW_OPTIMA_DEFAULT_SLUGS,
) -> tuple[MetricObjectiveRawOptimumSummary, ...]:
    """Return metric-specific no-L2 GRP raw-optimum summaries."""
    rows = _summary_rows()
    summaries: list[MetricObjectiveRawOptimumSummary] = []
    for slug in slugs:
        row = rows[slug]
        summaries.append(
            MetricObjectiveRawOptimumSummary(
                slug=slug,
                run_id=METRIC_OBJECTIVE_RAW_OPTIMA_RUN_IDS[slug],
                run_name=metric_objective_raw_optimum_run_name(slug),
                metric_key=row["metric_key"],
                display_name=row["display_name"],
                lower_is_better=_parse_bool(row["lower_is_better"]),
                predicted_optimum_metric=float(row["raw_predicted_optimum_metric"]),
                best_observed_run_name=row["best_observed_run_name"],
                best_observed_metric=float(row["best_observed_metric"]),
                predicted_observed_run_name=row["predicted_observed_run_name"],
                predicted_observed_metric=float(row["predicted_observed_metric"]),
                predicted_observed_regret=float(row["predicted_observed_regret"]),
                raw_nearest_observed_run_name=row["raw_nearest_observed_run_name"],
                raw_nearest_observed_metric=float(row["raw_nearest_observed_metric"]),
                raw_nearest_observed_regret=float(row["raw_nearest_observed_regret"]),
                raw_nearest_observed_tv=float(row["raw_nearest_observed_tv"]),
                oof_rmse=float(row["oof_rmse"]),
                oof_r2=float(row["oof_r2"]),
                oof_spearman=float(row["oof_spearman"]),
                phase_weights=_phase_weights_from_csv(slug),
            )
        )
    return tuple(summaries)


def metric_objective_raw_optimum_summaries_json(
    slugs: tuple[str, ...] = METRIC_OBJECTIVE_RAW_OPTIMA_DEFAULT_SLUGS,
) -> str:
    """Return metric-specific no-L2 GRP raw-optimum summaries as JSON."""
    return json.dumps(
        [_summary_to_dict(summary) for summary in metric_objective_raw_optimum_summaries(slugs)],
        indent=2,
        sort_keys=True,
    )


def metric_objective_raw_optimum_summaries_csv(
    slugs: tuple[str, ...] = METRIC_OBJECTIVE_RAW_OPTIMA_DEFAULT_SLUGS,
) -> str:
    """Return a flat CSV summary for metric-specific no-L2 GRP raw optima."""
    fields = (
        "slug",
        "run_id",
        "run_name",
        "metric_key",
        "display_name",
        "lower_is_better",
        "predicted_optimum_metric",
        "best_observed_run_name",
        "best_observed_metric",
        "predicted_observed_run_name",
        "predicted_observed_metric",
        "predicted_observed_regret",
        "raw_nearest_observed_run_name",
        "raw_nearest_observed_metric",
        "raw_nearest_observed_regret",
        "raw_nearest_observed_tv",
        "oof_rmse",
        "oof_r2",
        "oof_spearman",
    )
    rows = [_summary_to_dict(summary) for summary in metric_objective_raw_optimum_summaries(slugs)]
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row[field]) for field in fields))
    return "\n".join(lines) + "\n"


def create_metric_objective_raw_optimum_weight_config(
    slug: Literal["paloma_macro_bpb", "piqa_5shot_choice_logprob"],
) -> WeightConfig:
    """Return the weight config for one metric-specific no-L2 GRP raw optimum."""
    summary = next(summary for summary in metric_objective_raw_optimum_summaries((slug,)) if summary.slug == slug)
    return WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)


def create_metric_objective_raw_optimum_weight_configs(
    slugs: tuple[str, ...] = METRIC_OBJECTIVE_RAW_OPTIMA_DEFAULT_SLUGS,
) -> tuple[WeightConfig, ...]:
    """Return weight configs for metric-specific no-L2 GRP raw optima."""
    return tuple(
        WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
        for summary in metric_objective_raw_optimum_summaries(slugs)
    )
