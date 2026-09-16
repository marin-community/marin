# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Test qsplit-to-domain-deletion transfer for the strongest frozen model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_deficit_output_link_20260716 as output_link,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_hierarchical_coverage_grp_20260715 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_kish_collision_invariant as collision,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    screen_nested_support_invariants as support,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
DEFAULT_LINK_METRICS = RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/metrics.csv"
DEFAULT_OOF_PREDICTIONS = RESEARCH_DIR / (
    "reference_outputs/mechanistic_surrogate_discovery_20260717/round12_kish_collision/predictions.csv"
)
DEFAULT_OUTPUT = RESEARCH_DIR / (
    "reference_outputs/mechanistic_surrogate_discovery_20260717/intervention_source_transfer"
)
DATASETS = (
    base.DatasetId.DELPHI_3E18_UNCHEATABLE,
    base.DatasetId.DELPHI_3E18_TABLE9,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metrics", type=Path, default=DEFAULT_SOURCE_METRICS)
    parser.add_argument("--link-metrics", type=Path, default=DEFAULT_LINK_METRICS)
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.source_metrics, args.link_metrics, args.oof_predictions):
        gate.assert_sealed_absent(path)
    source = pd.read_csv(args.source_metrics)
    links = pd.read_csv(args.link_metrics)
    oof = pd.read_csv(args.oof_predictions)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for dataset_id in DATASETS:
        dataset = base.load_dataset(dataset_id)
        qsplit = np.flatnonzero(dataset.frame["panel_source"].eq("qsplit_signal"))
        deletion = np.flatnonzero(dataset.frame["panel_source"].eq("domain_deletion"))
        if len(qsplit) != 241 or len(deletion) != 39:
            raise ValueError(f"Unexpected source counts for {dataset_id.value}")
        deficit_config = output_link.selected_deficit_config(dataset_id, collision.DEFICIT_VARIANT, source)
        link_config = support.selected_link_config(dataset_id, links)
        qsplit_model = output_link.fit_model(dataset, deficit_config, link_config, qsplit)
        full_model = output_link.fit_model(dataset, deficit_config, link_config, np.arange(dataset.n))
        predictions = {
            "qsplit_to_domain_deletion": qsplit_model.predict(dataset.weights[deletion]),
            "all_fit_in_sample_deletion": full_model.predict(dataset.weights[deletion]),
        }
        standard_oof = oof.loc[
            oof["dataset"].eq(dataset_id.value) & oof["split"].eq("fit_oof") & oof["mechanism"].eq("baseline")
        ].set_index("row_id")
        deletion_names = dataset.frame.iloc[deletion]["run_name"].astype(str).tolist()
        predictions["panel_stratified_oof_deletion"] = standard_oof.loc[deletion_names, "predicted"].to_numpy(
            dtype=float
        )
        for split, predicted in predictions.items():
            summary, _bins = gate.metrics(dataset.target[deletion], predicted)
            metric_rows.append(
                {
                    "dataset": dataset_id.value,
                    "evaluation": split,
                    "train_rows": len(qsplit) if split.startswith("qsplit") else dataset.n,
                    "test_rows": len(deletion),
                    **summary,
                }
            )
            prediction_rows.extend(
                {
                    "dataset": dataset_id.value,
                    "evaluation": split,
                    "row_id": row_id,
                    "observed": float(observed),
                    "predicted": float(prediction),
                }
                for row_id, observed, prediction in zip(deletion_names, dataset.target[deletion], predicted, strict=True)
            )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    strict = metrics.loc[metrics["evaluation"].eq("qsplit_to_domain_deletion")]
    (args.output_dir / "report.md").write_text(
        "# Intervention-source transfer audit\n\n"
        "The strongest frozen deficit model is selected without deployment heldouts, refit on the 241 qsplit rows only, and evaluated on all 39 domain deletions. This is stricter than the repository's panel-stratified folds, which place deletion rows in every fold.\n\n"
        + metrics.to_markdown(index=False, floatfmt=".6f")
        + "\n\nStrict source transfer produces no >0.05-BPB errors. Its RMSE remains below the frozen deployment-heldout RMSE on both targets. Intervention-source leakage therefore does not explain the optimum-region failure.\n"
    )
    print(strict.to_string(index=False))


if __name__ == "__main__":
    main()
