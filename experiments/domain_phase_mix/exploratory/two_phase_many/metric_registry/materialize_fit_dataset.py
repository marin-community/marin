# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas"]
# ///
"""Materialize a fit-ready dataset from the metric registry."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry import (
    METRICS_WIDE_CSV,
    SCRIPT_DIR,
    WEIGHT_PREFIXES,
    canonicalize_metric_key,
)

FIT_DATASET_DIR = SCRIPT_DIR / "fit_datasets"
SAFE_PATH_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_metric_name(metric_key: str) -> str:
    return SAFE_PATH_RE.sub("_", metric_key).strip("_")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", help="Canonical metric key, e.g. lm_eval/piqa_5shot/logprob")
    parser.add_argument("--scale", default="60m_1p2b")
    parser.add_argument("--cohort", default="signal")
    parser.add_argument(
        "--run-set",
        choices=("fit_swarm_60m_default", "qsplit240_core", "all"),
        default="fit_swarm_60m_default",
        help="Logical run subset to materialize after scale/cohort filtering.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def materialize_fit_dataset(
    metric_key: str,
    *,
    scale: str,
    cohort: str,
    run_set: str,
    output: Path | None = None,
) -> pd.DataFrame:
    """Write a fit-ready wide table for one objective metric."""
    canonical_metric = canonicalize_metric_key(metric_key)["canonical_metric_key"]
    frame = pd.read_csv(METRICS_WIDE_CSV, low_memory=False)
    if canonical_metric not in frame.columns:
        raise ValueError(f"{canonical_metric!r} is not present in {METRICS_WIDE_CSV}")

    weight_columns = sorted(column for column in frame.columns if column.startswith(WEIGHT_PREFIXES))
    id_columns = [
        column
        for column in (
            "registry_run_key",
            "run_id",
            "run_name",
            "scale",
            "cohort",
            "source_run_name",
            "source_experiment",
            "wandb_run_id",
            "checkpoint_root",
            "status",
            "is_qsplit240_core",
            "is_baseline_olmix",
            "is_baseline_stratified",
            "is_fit_swarm_60m_default",
        )
        if column in frame.columns
    ]
    mask = frame["scale"].eq(scale) & frame["cohort"].eq(cohort)
    if run_set == "fit_swarm_60m_default":
        mask &= frame["is_fit_swarm_60m_default"].fillna(False)
    elif run_set == "qsplit240_core":
        mask &= frame["is_qsplit240_core"].fillna(False)
    out = frame.loc[mask, id_columns + weight_columns + [canonical_metric]]
    out = out.dropna(subset=[canonical_metric]).rename(columns={canonical_metric: "objective_metric"})
    out["objective_metric_key"] = canonical_metric
    out = out.dropna(axis=1, how="all")
    out = out.reset_index(drop=True)

    if output is None:
        output = FIT_DATASET_DIR / f"{_safe_metric_name(canonical_metric)}__{scale}__{cohort}__{run_set}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Wrote {len(out)} rows to {output}")
    return out


def main() -> None:
    args = _parse_args()
    materialize_fit_dataset(args.metric, scale=args.scale, cohort=args.cohort, run_set=args.run_set, output=args.output)


if __name__ == "__main__":
    main()
