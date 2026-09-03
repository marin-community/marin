# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-bucket benefit and harm decomposition of a panel-fitted grid model at chosen bank coordinates.

For each target the model is fitted per component on the canonical panel (heldout inner folds) and the design
columns at the chosen coordinates are multiplied by the fitted amplitudes, aggregated with the target's component
weights. Coordinates are given as suffixes of registry coordinate ids, or ``frontier`` for the measured best.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_dose_anatomy_20260903 as anatomy,
)

PANEL = "delphi_3e18_39bucket"


def fit_parts(model_id: str, target: str, component_index: int, query) -> dict[str, object]:
    panel = harness.load_panel(PANEL)
    entry = registry.ENTRY_BY_ID[model_id]
    group = panel.group(target)
    component = group.components[component_index]
    features = dataclasses.replace(registry.apply_transform(panel.features, entry), component=str(component))
    model = entry.build(features)
    if not isinstance(model, models.GridModel):
        raise TypeError(f"{model_id} is not a grid model")
    fitted = model.fit(
        features,
        group.outcomes[:, component_index],
        np.arange(panel.rows),
        harness.heldout_inner_folds(panel),
        harness._seed(harness.FitTask(model_id, PANEL, target, component_index, component, 0, 0)),
    )
    query_features = dataclasses.replace(registry.apply_transform(query, entry), component=str(component))
    design = model.design(query_features, fitted.shape)
    return {
        "names": design.names,
        "parts": design.values * fitted.head.coefficients[None, :],
        "intercept": fitted.head.intercept,
        "shape": fitted.shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="weibull_softplus_unscaled")
    parser.add_argument("--target", required=True)
    parser.add_argument("--coordinates", nargs="+", required=True, help="'frontier' or coordinate id suffixes")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    harness.HELDOUT_DIR = args.registry_dir.resolve()
    panel = harness.load_panel(PANEL)
    bank, features = harness.heldout_features(panel, args.target)
    _count, mean_column = harness.HELDOUT_TARGET_COLUMNS[args.target]
    indices = []
    for token in args.coordinates:
        if token == "frontier":
            indices.append(int(np.argmin(bank[mean_column].to_numpy(float))))
        else:
            matches = np.flatnonzero(bank["coordinate_id"].str.endswith(token))
            if len(matches) != 1:
                raise ValueError(f"{token}: matched {len(matches)} coordinates")
            indices.append(int(matches[0]))
    query = dataclasses.replace(
        features, exposures=features.exposures[indices], weights=features.weights[indices], label="decomposition"
    )
    group = panel.group(args.target)
    with harness.parallel_config(backend="loky", inner_max_num_threads=1):
        parts = Parallel(n_jobs=args.workers)(
            delayed(fit_parts)(args.model, args.target, index, query) for index in range(len(group.components))
        )
    weights = group.aggregation_weights
    aggregated = sum(weights[index] * part["parts"] for index, part in enumerate(parts))
    intercept = sum(weights[index] * part["intercept"] for index, part in enumerate(parts))
    names = parts[0]["names"]
    benefit = np.array([name.startswith(anatomy.BENEFIT_PREFIXES) for name in names])
    rows = []
    for position, index in enumerate(indices):
        label = args.coordinates[position]
        for bucket_index, bucket in enumerate(panel.buckets):
            rows.append(
                {
                    "coordinate": label,
                    "coordinate_id": bank["coordinate_id"].iloc[index],
                    "measured": float(bank[mean_column].iloc[index]),
                    "predicted": float(intercept + aggregated[position].sum()),
                    "bucket": bucket,
                    "weight": float(query.weights[position, bucket_index]),
                    "epochs": float(query.exposures[position, bucket_index]),
                    "benefit": (
                        float(aggregated[position, benefit][bucket_index])
                        if benefit.sum() == len(panel.buckets)
                        else float("nan")
                    ),
                    "harm": (
                        float(aggregated[position, ~benefit][bucket_index])
                        if (~benefit).sum() == len(panel.buckets)
                        else float("nan")
                    ),
                }
            )
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        args.output_dir / f"frontier_decomposition_{args.target}_{args.model.replace('@', '_')}.csv", index=False
    )
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    totals = table.groupby("coordinate").agg(
        measured=("measured", "first"),
        predicted=("predicted", "first"),
        benefit=("benefit", "sum"),
        harm=("harm", "sum"),
    )
    print(totals.round(4).to_string())
    wide = table.pivot(index="bucket", columns="coordinate", values=["weight", "epochs", "benefit", "harm"]).round(4)
    wide.columns = [f"{kind}:{label}" for kind, label in wide.columns]
    print(wide.to_string())


if __name__ == "__main__":
    main()
