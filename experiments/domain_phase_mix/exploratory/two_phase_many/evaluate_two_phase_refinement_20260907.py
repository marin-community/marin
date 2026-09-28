# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Compare refinement candidates with immutable predictions from the broad screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_two_phase_creative_20260907 as common
import fit_two_phase_link_transfer_20260907 as previous
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "reference_outputs/two_phase_refinement_20260907"
PRIOR = HERE / "reference_outputs/two_phase_creative_sweep_20260907"
FAMILIES = ("joint_estimators", "macro_semantic", "compact_semantic")
MATCHED = {
    "CRE2-017": "CRE2-013",
    "CRE2-018": "CRE2-017",
    "CRE2-019": "CRE2-016",
    "CRE2-020": "CRE2-019",
    "CRE2-021": "CRE2-016",
    "CRE2-022": "CRE2-021",
}


def matched_bootstrap(pairs: pd.DataFrame, output: Path) -> None:
    samples = np.random.default_rng(common.BOOTSTRAP_SEED).integers(0, 238, size=(common.BOOTSTRAP_DRAWS, 238))
    rows = []
    for objective in previous.OBJECTIVES:
        group = pairs[(pairs.context == "oof") & (pairs.objective == objective)]
        table = group.pivot(index="asymmetric_row", columns="model", values="predicted_delta").sort_index()
        truth = (
            group.drop_duplicates("asymmetric_row")
            .set_index("asymmetric_row")
            .loc[table.index, "measured_delta"]
            .to_numpy()
        )
        for model, comparator in MATCHED.items():
            if model not in table:
                continue
            prediction, reference = table[model].to_numpy(), table[comparator].to_numpy()
            squared, control_squared = (prediction - truth) ** 2, (reference - truth) ** 2
            regret = common.binary_regret(truth, prediction) - common.binary_regret(truth, reference)
            measures = [
                (
                    "paired_rmse",
                    np.sqrt(squared.mean()) - np.sqrt(control_squared.mean()),
                    np.sqrt(squared[samples].mean(axis=1)) - np.sqrt(control_squared[samples].mean(axis=1)),
                ),
                ("paired_decision_regret", regret.mean(), regret[samples].mean(axis=1)),
            ]
            for metric, point, draws in measures:
                low, high = np.quantile(draws, [0.025, 0.975])
                rows.append(
                    {
                        "objective": objective,
                        "model": model,
                        "comparator": comparator,
                        "metric": metric,
                        "difference": float(point),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "scope": (
                            "Conditional fixed-OOF pair bootstrap; ordinary interval, excludes selection and refitting"
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(output / "matched_bootstrap.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--output", type=Path, default=OUTPUT / "comparison")
    args = parser.parse_args()
    manifest = json.loads((PRIOR / "MANIFEST.json").read_text())
    path = PRIOR / "comparison/all_predictions.csv"
    assert previous.file_hash(path) == manifest["output_sha256"]["comparison/all_predictions.csv"]
    frames, sources = [pd.read_csv(path)], [path, Path(__file__), Path(common.__file__)]
    for family in args.families:
        path = OUTPUT / family / "predictions.csv"
        frames.append(pd.read_csv(path))
        sources.append(path)
    frame = pd.concat(frames, ignore_index=True)
    _, panel, _ = previous.inputs(str(common.PRIOR))
    common.validate_predictions(frame, panel)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "all_predictions.csv", index=False)
    endpoint, pairs = common.collect_metrics(frame, panel, args.output)
    common.bootstrap_pairs(pairs, args.output)
    matched_bootstrap(pairs, args.output)
    previous.write_json(
        args.output / "manifest.json",
        {
            "source_sha256": {str(p): previous.file_hash(p) for p in sources},
            "models": sorted(frame.model.unique()),
            "families": args.families,
            "rows_per_cell": 520,
            "cells_per_model": 8,
            "oof_rows": 518,
            "pairs": 238,
        },
    )
    pair_metrics = pd.read_csv(args.output / "pair_metrics.csv")
    selected = endpoint[(endpoint.context == "oof") & (endpoint.population == "all")]
    selected = selected.merge(
        pair_metrics[pair_metrics.context == "oof"],
        on=["objective", "model", "context"],
        suffixes=("_endpoint", "_pair"),
    )
    selected = selected[selected.model.isin([*MATCHED, "hpr", "separate_heads", "CRE2-013", "CRE2-011", "CRE2-016"])]
    print(
        selected[
            ["objective", "model", "rmse_endpoint", "rmse_pair", "mean_binary_decision_regret", "regret1"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
