# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Audit the historical run-noise scale used to interpret the TPP40 bridge.

The bridge's Uncheatable score is the unweighted mean of seven validation BPBs.
This audit estimates ordinary run-to-run noise from historical Delphi rows that
share both phase mixtures exactly and differ only by seed. It deliberately does
not alter the preregistered bridge acceptance thresholds.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_prefix_search_evidence_20260819 as prefix_evidence,
)

REFERENCE_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_tpp40_europe_readiness_20260830"
)
DEFAULT_ACCEPTANCE_REPORT = REFERENCE_DIR / "bridge_acceptance_report_v3.json"
DEFAULT_OUTPUT = REFERENCE_DIR / "bridge_uncheatable_noise_audit_v2.json"
COMPONENT_COLUMNS = (
    "eval/uncheatable_eval/ao3_english/bpb",
    "eval/uncheatable_eval/arxiv_computer_science/bpb",
    "eval/uncheatable_eval/arxiv_physics/bpb",
    "eval/uncheatable_eval/bbc_news/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/wikipedia_english/bpb",
)
TOKEN_WEIGHTED_COLUMN = "eval/uncheatable_eval/bpb"
TABLE9_COLUMN = "table9_macro_bpb"
BRIDGE_MACRO_COLUMN = "bridge_unweighted_macro_bpb"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _probability_within_threshold(*, mean: float, standard_deviation: float, threshold: float) -> float:
    upper = (threshold - mean) / standard_deviation
    lower = (-threshold - mean) / standard_deviation
    return _normal_cdf(upper) - _normal_cdf(lower)


def _repeat_groups(phase_0: np.ndarray, phase_1: np.ndarray) -> list[np.ndarray]:
    groups: dict[tuple[tuple[float, ...], tuple[float, ...]], list[int]] = collections.defaultdict(list)
    for index, (early, late) in enumerate(zip(np.round(phase_0, 8), np.round(phase_1, 8), strict=True)):
        groups[(tuple(early), tuple(late))].append(index)
    return [np.asarray(members, dtype=int) for members in groups.values() if len(members) > 1]


def _within_group_noise(values: np.ndarray, repeat_groups: list[np.ndarray]) -> dict[str, object]:
    groups = [
        {
            "size": len(members),
            "sample_standard_deviation": float(np.std(values[members], ddof=1)),
        }
        for members in repeat_groups
        if np.isfinite(values[members]).all()
    ]
    if not groups:
        raise ValueError("No complete repeated-coordinate groups are available for the requested metric")
    degrees_of_freedom = sum(int(group["size"]) - 1 for group in groups)
    pooled_variance = (
        sum((int(group["size"]) - 1) * float(group["sample_standard_deviation"]) ** 2 for group in groups)
        / degrees_of_freedom
    )
    return {
        "complete_repeat_group_count": len(groups),
        "degrees_of_freedom": degrees_of_freedom,
        "median_single_run_sd": float(np.median([group["sample_standard_deviation"] for group in groups])),
        "pooled_single_run_sd": math.sqrt(pooled_variance),
        "repeat_groups": groups,
    }


def build_audit(acceptance_report: Path) -> dict[str, object]:
    frame, excluded_rows = prefix_evidence.panel()
    geometry = prefix_evidence.geometry(frame)
    repeat_groups = _repeat_groups(geometry["phase_0"], geometry["phase_1"])

    frame = frame.copy()
    frame[BRIDGE_MACRO_COLUMN] = frame[list(COMPONENT_COLUMNS)].mean(axis=1)
    metric_columns = (BRIDGE_MACRO_COLUMN, TOKEN_WEIGHTED_COLUMN, TABLE9_COLUMN, *COMPONENT_COLUMNS)
    noise: dict[str, dict[str, object]] = {}
    for column in metric_columns:
        noise[column] = _within_group_noise(frame[column].to_numpy(float), repeat_groups)

    report = json.loads(acceptance_report.read_text())
    phase_0_pairs = report["uncheatable"]["phase_0"]["pairs"]
    if len(phase_0_pairs) != 1:
        raise ValueError(f"Expected exactly one completed phase-0 bridge pair, found {len(phase_0_pairs)}")
    pair = phase_0_pairs[0]
    observed_delta = float(pair["europe_minus_east5"])
    threshold = float(report["uncheatable"]["phase_0"]["threshold"]["mean_absolute_paired_delta_max"])
    single_run_sd = float(noise[BRIDGE_MACRO_COLUMN]["median_single_run_sd"])
    paired_sd = math.sqrt(2.0) * single_run_sd
    pooled_single_run_sd = float(noise[BRIDGE_MACRO_COLUMN]["pooled_single_run_sd"])
    pooled_paired_sd = math.sqrt(2.0) * pooled_single_run_sd

    component_diagnostics: dict[str, dict[str, float]] = {}
    for column in COMPONENT_COLUMNS:
        component_name = column.removeprefix("eval/").removesuffix("/bpb")
        component_delta = float(pair["component_deltas"][component_name])
        component_sd = float(noise[column]["median_single_run_sd"])
        component_diagnostics[component_name] = {
            "europe_minus_east5": component_delta,
            "historical_single_run_sd": component_sd,
            "standardized_paired_delta": component_delta / (math.sqrt(2.0) * component_sd),
            "historical_pooled_single_run_sd": float(noise[column]["pooled_single_run_sd"]),
            "pooled_standardized_paired_delta": (
                component_delta / (math.sqrt(2.0) * float(noise[column]["pooled_single_run_sd"]))
            ),
        }

    source_dir = prefix_evidence.DELPHI
    source_files = {
        "heldout_current.csv": source_dir / "heldout_current.csv",
        "endpoint_components.csv": source_dir / "endpoint_components.csv",
        "noise_estimator": Path(prefix_evidence.__file__).resolve(),
        "acceptance_report": acceptance_report,
        "audit_script": Path(__file__).resolve(),
    }
    return {
        "schema_version": 2,
        "estimator": {
            "description": (
                "Median sample standard deviation across historical rows sharing both phase mixtures "
                "to 8 decimal places and differing only by seed."
            ),
            "clean_row_count": len(frame),
            "excluded_row_count": excluded_rows,
            "repeat_group_count": len(repeat_groups),
            "repeat_group_size_histogram": {
                str(size): count
                for size, count in sorted(collections.Counter(len(group) for group in repeat_groups).items())
            },
            "bridge_macro_components": list(COMPONENT_COLUMNS),
            "bridge_macro_weighting": "unweighted arithmetic mean",
        },
        "noise": noise,
        "row_2_phase_0": {
            "east5_macro_bpb": float(pair["east5_macro_bpb"]),
            "europe_macro_bpb": float(pair["europe_macro_bpb"]),
            "europe_minus_east5": observed_delta,
            "acceptance_threshold": threshold,
            "historical_single_run_sd": single_run_sd,
            "independent_pair_sd": paired_sd,
            "standardized_paired_delta": observed_delta / paired_sd,
            "historical_pooled_single_run_sd": pooled_single_run_sd,
            "pooled_independent_pair_sd": pooled_paired_sd,
            "pooled_standardized_paired_delta": observed_delta / pooled_paired_sd,
            "pass_probability_under_zero_shift": _probability_within_threshold(
                mean=0.0,
                standard_deviation=paired_sd,
                threshold=threshold,
            ),
            "pass_probability_under_true_plus_0p003_shift": _probability_within_threshold(
                mean=0.003,
                standard_deviation=paired_sd,
                threshold=threshold,
            ),
            "component_diagnostics": component_diagnostics,
        },
        "interpretation_boundary": (
            "The standardized delta compares the bridge discrepancy with historical same-coordinate "
            "seed noise from the 3e18 panel, not from TPP40 bridge replicates. Both the scale transfer and "
            "the independence assumption are limitations. The result is diagnostic rather than a "
            "replacement for the frozen bridge gate and does not estimate accelerator-specific "
            "systematic bias."
        ),
        "source_sha256": {name: _sha256(path) for name, path in source_files.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-report", type=Path, default=DEFAULT_ACCEPTANCE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_audit(args.acceptance_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
