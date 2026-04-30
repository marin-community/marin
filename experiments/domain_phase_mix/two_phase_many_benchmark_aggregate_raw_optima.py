# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw-optimum deployments for 60M benchmark-aggregate no-L2 GRP fits."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
import io
import json
import math
from pathlib import Path
from typing import Any, Literal

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import DOMAIN_NAMES

BENCHMARK_AGGREGATE_RAW_OPTIMA_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_benchmark_aggregate_grp_no_l2_raw_optima"
)
BENCHMARK_AGGREGATE_RAW_OPTIMA_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "metric_registry"
    / "benchmark_aggregate_60m"
    / "grp_no_l2_fits"
)
BENCHMARK_AGGREGATE_RAW_OPTIMA_TRACKED_DIR = (
    Path(__file__).resolve().parent / "validation_artifacts" / "benchmark_aggregate_raw_optima"
)

OLMO_BASE_EASY_OVERLAP_QA_TECH_SLUG = "olmo_base_easy_overlap_accuracy_raw__qa_tech"
OLMO_BASE_NO_MMLU_HUMANEVAL_SLUG = "olmo_base_plus_humaneval_no_mmlu_accuracy_mean"
BENCHMARK_AGGREGATE_RAW_OPTIMA_DEFAULT_SLUGS = (
    OLMO_BASE_EASY_OVERLAP_QA_TECH_SLUG,
    OLMO_BASE_NO_MMLU_HUMANEVAL_SLUG,
)
BENCHMARK_AGGREGATE_RAW_OPTIMA_RUN_IDS = {
    OLMO_BASE_EASY_OVERLAP_QA_TECH_SLUG: 520,
    OLMO_BASE_NO_MMLU_HUMANEVAL_SLUG: 521,
}
BENCHMARK_AGGREGATE_RAW_OPTIMA_DISPLAY_NAMES = {
    OLMO_BASE_EASY_OVERLAP_QA_TECH_SLUG: "OLMoBase easy-overlap accuracy, raw, qa/tech partition",
    OLMO_BASE_NO_MMLU_HUMANEVAL_SLUG: "OLMoBase-no-MMLU + HumanEval accuracy mean",
}
BENCHMARK_AGGREGATE_RAW_OPTIMA_RUN_NAMES = {
    OLMO_BASE_EASY_OVERLAP_QA_TECH_SLUG: "baseline_benchmark_aggregate_raw_optimum_olmo_easy_overlap_qa_tech",
    OLMO_BASE_NO_MMLU_HUMANEVAL_SLUG: "baseline_benchmark_aggregate_raw_optimum_olmo_no_mmlu_humaneval",
}


@dataclass(frozen=True)
class BenchmarkAggregateRawOptimumSummary:
    """Summary for one 60M benchmark-aggregate raw-optimum deployment."""

    slug: str
    run_id: int
    run_name: str
    display_name: str
    weights_csv: str
    fit_objective: float
    fit_cv_rmse: float
    fit_cv_mae: float
    fit_cv_spearman: float
    fit_cv_foldmean_regret_at_1: float
    fit_lower_tail_optimism: float
    phase0_support_gt_1e4: int
    phase1_support_gt_1e4: int
    phase_weights: dict[str, dict[str, float]]


def parse_benchmark_aggregate_raw_optimum_slugs(spec: str) -> tuple[str, ...]:
    """Parse a comma-separated benchmark-aggregate raw-optimum slug spec."""
    cleaned = spec.strip().lower()
    if cleaned == "all":
        return BENCHMARK_AGGREGATE_RAW_OPTIMA_DEFAULT_SLUGS
    slugs = tuple(part.strip() for part in spec.split(",") if part.strip())
    if not slugs:
        raise ValueError("benchmark-aggregate slug spec must not be empty")
    invalid = [slug for slug in slugs if slug not in BENCHMARK_AGGREGATE_RAW_OPTIMA_RUN_IDS]
    if invalid:
        raise ValueError(f"Unsupported benchmark-aggregate raw-optimum slugs: {invalid}")
    return slugs


def benchmark_aggregate_raw_optimum_run_name(slug: str) -> str:
    """Return the canonical run name for one benchmark-aggregate raw optimum."""
    try:
        return BENCHMARK_AGGREGATE_RAW_OPTIMA_RUN_NAMES[slug]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark-aggregate raw-optimum slug: {slug}") from exc


def _weights_csv_for_slug(slug: str) -> Path:
    return BENCHMARK_AGGREGATE_RAW_OPTIMA_DIR / slug / "raw_optimum_weights.csv"


def _tracked_weights_json_for_slug(slug: str) -> Path:
    return BENCHMARK_AGGREGATE_RAW_OPTIMA_TRACKED_DIR / f"{slug}_weights.json"


def _weights_source_for_slug(slug: str) -> Path:
    weights_csv = _weights_csv_for_slug(slug)
    if weights_csv.exists():
        return weights_csv
    return _tracked_weights_json_for_slug(slug)


def _refine_csv_for_slug(slug: str) -> Path:
    return BENCHMARK_AGGREGATE_RAW_OPTIMA_DIR / slug / "refine.csv"


def _parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


@cache
def _best_refine_row(slug: str) -> dict[str, str]:
    refine_csv = _refine_csv_for_slug(slug)
    if not refine_csv.exists():
        return {}
    with refine_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return min(rows, key=lambda row: _parse_float(row, "objective"))


def _phase_weights_from_csv(slug: str) -> dict[str, dict[str, float]]:
    weights_csv = _weights_csv_for_slug(slug)
    weights_json = _tracked_weights_json_for_slug(slug)
    if weights_csv.exists():
        with weights_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif weights_json.exists():
        with weights_json.open() as handle:
            rows = json.load(handle)
    else:
        raise FileNotFoundError(f"Missing benchmark-aggregate raw optimum weights: {weights_csv} and {weights_json}")

    phase0: dict[str, float] = {}
    phase1: dict[str, float] = {}
    for row in rows:
        domain_name = row["domain_name"]
        phase0[domain_name] = float(row["phase0_weight"])
        phase1[domain_name] = float(row["phase1_weight"])

    expected = set(DOMAIN_NAMES)
    actual = set(phase0)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(f"Domain mismatch for {slug}: missing={missing}, extra={extra}")

    phase0_sum = sum(phase0.values())
    phase1_sum = sum(phase1.values())
    if abs(phase0_sum - 1.0) > 1e-6 or abs(phase1_sum - 1.0) > 1e-6:
        raise ValueError(f"Weights for {slug} do not sum to 1: phase0={phase0_sum}, phase1={phase1_sum}")

    return {
        "phase_0": {domain_name: phase0[domain_name] for domain_name in DOMAIN_NAMES},
        "phase_1": {domain_name: phase1[domain_name] for domain_name in DOMAIN_NAMES},
    }


def _summary_to_dict(summary: BenchmarkAggregateRawOptimumSummary) -> dict[str, Any]:
    return {
        "slug": summary.slug,
        "run_id": summary.run_id,
        "run_name": summary.run_name,
        "display_name": summary.display_name,
        "weights_csv": summary.weights_csv,
        "fit_objective": summary.fit_objective,
        "fit_cv_rmse": summary.fit_cv_rmse,
        "fit_cv_mae": summary.fit_cv_mae,
        "fit_cv_spearman": summary.fit_cv_spearman,
        "fit_cv_foldmean_regret_at_1": summary.fit_cv_foldmean_regret_at_1,
        "fit_lower_tail_optimism": summary.fit_lower_tail_optimism,
        "phase0_support_gt_1e4": summary.phase0_support_gt_1e4,
        "phase1_support_gt_1e4": summary.phase1_support_gt_1e4,
        "phase_weights": summary.phase_weights,
    }


@cache
def benchmark_aggregate_raw_optimum_summaries(
    slugs: tuple[str, ...] = BENCHMARK_AGGREGATE_RAW_OPTIMA_DEFAULT_SLUGS,
) -> tuple[BenchmarkAggregateRawOptimumSummary, ...]:
    """Return 60M benchmark-aggregate no-L2 GRP raw-optimum summaries."""
    summaries: list[BenchmarkAggregateRawOptimumSummary] = []
    for slug in slugs:
        phase_weights = _phase_weights_from_csv(slug)
        best_refine = _best_refine_row(slug)
        phase0 = phase_weights["phase_0"]
        phase1 = phase_weights["phase_1"]
        summaries.append(
            BenchmarkAggregateRawOptimumSummary(
                slug=slug,
                run_id=BENCHMARK_AGGREGATE_RAW_OPTIMA_RUN_IDS[slug],
                run_name=benchmark_aggregate_raw_optimum_run_name(slug),
                display_name=BENCHMARK_AGGREGATE_RAW_OPTIMA_DISPLAY_NAMES[slug],
                weights_csv=str(_weights_source_for_slug(slug)),
                fit_objective=_parse_float(best_refine, "objective"),
                fit_cv_rmse=_parse_float(best_refine, "target_cv_rmse"),
                fit_cv_mae=_parse_float(best_refine, "target_cv_mae"),
                fit_cv_spearman=_parse_float(best_refine, "target_cv_spearman"),
                fit_cv_foldmean_regret_at_1=_parse_float(best_refine, "target_cv_foldmean_regret_at_1"),
                fit_lower_tail_optimism=_parse_float(best_refine, "target_lower_tail_optimism"),
                phase0_support_gt_1e4=sum(weight > 1e-4 for weight in phase0.values()),
                phase1_support_gt_1e4=sum(weight > 1e-4 for weight in phase1.values()),
                phase_weights=phase_weights,
            )
        )
    return tuple(summaries)


def benchmark_aggregate_raw_optimum_summaries_json(
    slugs: tuple[str, ...] = BENCHMARK_AGGREGATE_RAW_OPTIMA_DEFAULT_SLUGS,
) -> str:
    """Return 60M benchmark-aggregate no-L2 GRP raw-optimum summaries as JSON."""
    return json.dumps(
        [_summary_to_dict(summary) for summary in benchmark_aggregate_raw_optimum_summaries(slugs)],
        indent=2,
        sort_keys=True,
    )


def benchmark_aggregate_raw_optimum_summaries_csv(
    slugs: tuple[str, ...] = BENCHMARK_AGGREGATE_RAW_OPTIMA_DEFAULT_SLUGS,
) -> str:
    """Return a flat CSV summary for 60M benchmark-aggregate no-L2 GRP raw optima."""
    fields = (
        "slug",
        "run_id",
        "run_name",
        "display_name",
        "weights_csv",
        "fit_objective",
        "fit_cv_rmse",
        "fit_cv_mae",
        "fit_cv_spearman",
        "fit_cv_foldmean_regret_at_1",
        "fit_lower_tail_optimism",
        "phase0_support_gt_1e4",
        "phase1_support_gt_1e4",
    )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for summary in benchmark_aggregate_raw_optimum_summaries(slugs):
        row = _summary_to_dict(summary)
        writer.writerow({field: row[field] for field in fields})
    return output.getvalue()


def create_benchmark_aggregate_raw_optimum_weight_config(
    slug: Literal["olmo_base_easy_overlap_accuracy_raw__qa_tech", "olmo_base_plus_humaneval_no_mmlu_accuracy_mean"],
) -> WeightConfig:
    """Return the weight config for one 60M benchmark-aggregate raw optimum."""
    summary = next(summary for summary in benchmark_aggregate_raw_optimum_summaries((slug,)) if summary.slug == slug)
    return WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)


def create_benchmark_aggregate_raw_optimum_weight_configs(
    slugs: tuple[str, ...] = BENCHMARK_AGGREGATE_RAW_OPTIMA_DEFAULT_SLUGS,
) -> tuple[WeightConfig, ...]:
    """Return weight configs for 60M benchmark-aggregate raw optima."""
    return tuple(
        WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
        for summary in benchmark_aggregate_raw_optimum_summaries(slugs)
    )
