# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Neighbour forecasts for the queued 3e18 validation runs, from the measured bank alone.

`analyze_delphi_top_band_ordering_20260906.py` found that a held-out-source kernel smoother on the bank (TV
bandwidth 0.05) orders the best-measured Table-9 band better than any panel-fitted surrogate. This script writes the
same smoother's forecast for every queued candidate (link, link + hub, coupling, frontier factorial) next to the
nearest measured bank coordinate, so that the morning collection can compare realized values against both the
surrogate's own prediction and the neighbour forecast. No fit, no launch.

usage: uv run python forecast_delphi_queued_runs_20260906.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "reference_outputs"
BENCHMARK = REFERENCE / "delphi_offline_selection_20260906"
OUTPUT = REFERENCE / "delphi_top_band_ordering_20260906" / "queued_run_forecasts.csv"
BANDWIDTHS = (0.05, 0.1)
TABLES = {
    "link": REFERENCE / "delphi_link_validation_3e18_20260906" / "runtime_materialization" / "candidate_weights.csv",
    "hub": REFERENCE / "delphi_link_hub_validation_3e18_20260906" / "runtime_materialization" / "candidate_weights.csv",
    "coupling": (
        REFERENCE / "delphi_coupling_validation_3e18_20260906" / "runtime_materialization" / "candidate_weights.csv"
    ),
    "factorial": REFERENCE / "delphi_frontier_factorial_design_20260906" / "candidate_weights.csv",
    "factorial_replicate": REFERENCE / "delphi_frontier_factorial_design_20260906" / "candidate_weights_replicate.csv",
}


def candidate_matrix(table: pd.DataFrame, buckets: list[str]) -> tuple[list[str], np.ndarray, pd.Series]:
    wide = table.pivot(index="candidate_id", columns="domain", values="weight")
    missing = set(buckets) - set(wide.columns)
    if missing:
        raise ValueError(f"candidate table lacks buckets {sorted(missing)}")
    wide = wide[buckets]
    if not np.allclose(wide.sum(axis=1), 1.0, atol=2e-3):
        raise ValueError("candidate weights do not sum to one")
    targets = table.groupby("candidate_id").target.first().loc[wide.index]
    return list(wide.index), wide.to_numpy(float), targets


def main() -> None:
    panel = np.load(BENCHMARK / "inputs" / "panel.npz", allow_pickle=True)
    buckets = [str(b) for b in panel["buckets"]]
    banks = {}
    for target in ("uncheatable", "table9"):
        labels = pd.read_csv(BENCHMARK / "inputs" / f"{target}_bank_labels.csv")
        features = np.load(BENCHMARK / "inputs" / f"{target}_bank_features.npz", allow_pickle=True)
        order = pd.Index(features["coordinate_id"]).get_indexer(labels.coordinate_id)
        banks[target] = (features["weights"][order], labels)
    rows = []
    for launch, path in TABLES.items():
        ids, weights, targets = candidate_matrix(pd.read_csv(path), buckets)
        for candidate, weight, target in zip(ids, weights, targets, strict=True):
            bank_weights, labels = banks[target]
            distance = np.abs(bank_weights - weight[None, :]).sum(axis=1) / 2
            measured = labels.measured_mean_bpb.to_numpy(float)
            nearest = int(np.argmin(distance))
            row = {
                "launch": launch,
                "candidate_id": candidate,
                "target": target,
                "nearest_bank_tv": float(distance[nearest]),
                "nearest_bank_measured": float(measured[nearest]),
                "nearest_bank_runs": int(labels.run_count.iloc[nearest]),
                "nearest_bank_source": labels.sources.iloc[nearest].replace("archive::", "")[:60],
                "bank_within_tv_0.1": int((distance <= 0.1).sum()),
            }
            for bandwidth in BANDWIDTHS:
                kernel = np.exp(-0.5 * (distance / bandwidth) ** 2)
                row[f"kernel_forecast_tv{bandwidth}"] = float(kernel @ measured / kernel.sum())
                row[f"kernel_mass_tv{bandwidth}"] = float(kernel.sum())
            rows.append(row)
    table = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
