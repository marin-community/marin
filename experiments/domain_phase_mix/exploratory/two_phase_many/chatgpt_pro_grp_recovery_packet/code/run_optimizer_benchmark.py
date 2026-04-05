# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Benchmark nonlinear optimizer choices for GRP retuning."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from grp_packet import (
    CURRENT_BROAD_BETA_START_PARAMS,
    load_packet,
    load_reference_state,
    load_subset_indices,
    subset_packet,
    tune_genericfamily_params,
)

REPRESENTATIVE_SUBSET_SIZES = (40, 100, 180, 220)
METHODS = ("L-BFGS-B", "Powell", "Nelder-Mead")
OBJECTIVE_NAME = "single_foldmean"
RESULTS_CSV = Path(__file__).resolve().parents[1] / "reference_outputs" / "optimizer_benchmark.csv"
SUMMARY_JSON = Path(__file__).resolve().parents[1] / "reference_outputs" / "optimizer_benchmark.json"


def main() -> None:
    packet = load_packet()
    anchors = load_reference_state()
    subset_indices = load_subset_indices()
    valid_weights = np.stack([anchors.validated_global_weights, anchors.validated_pair_weights], axis=0)
    valid_y = np.asarray([anchors.validated_global_bpb, anchors.validated_pair_bpb], dtype=float)

    rows: list[dict[str, object]] = []
    for subset_size in REPRESENTATIVE_SUBSET_SIZES:
        train_packet = subset_packet(packet, np.asarray(subset_indices[subset_size], dtype=int))
        for start_name, start_params in (
            ("current_tuned", anchors.current_tuned_params),
            ("broad_beta", CURRENT_BROAD_BETA_START_PARAMS),
        ):
            for method in METHODS:
                started = perf_counter()
                metrics, result = tune_genericfamily_params(
                    train_packet,
                    valid_weights,
                    valid_y,
                    method=method,
                    objective_name=OBJECTIVE_NAME,
                    start_params=start_params,
                    seed=0,
                )
                rows.append(
                    {
                        "subset_size": subset_size,
                        "start_name": start_name,
                        "method": method,
                        "duration": perf_counter() - started,
                        "success": bool(metrics["success"]),
                        "nit": int(getattr(result, "nit", -1)),
                        "nfev": int(getattr(result, "nfev", -1)),
                        "objective_name": OBJECTIVE_NAME,
                        "objective": float(metrics["objective"]),
                        "cv_rmse": float(metrics["cv_rmse"]),
                        "cv_r2": float(metrics["cv_r2"]),
                        "cv_spearman": float(metrics["cv_spearman"]),
                        "cv_regret_at_1": float(metrics["cv_regret_at_1"]),
                        "cv_foldmean_regret_at_1": float(metrics["cv_foldmean_regret_at_1"]),
                        "anchor_mae": float(metrics["anchor_mae"]),
                        "anchor_rmse": float(metrics["anchor_rmse"]),
                        "alpha": float(metrics["alpha"]),
                        "eta": float(metrics["eta"]),
                        "lam": float(metrics["lam"]),
                        "tau": float(metrics["tau"]),
                        "reg": float(metrics["reg"]),
                        "beta": float(metrics["beta"]),
                        "message": str(metrics["message"]),
                    }
                )

    frame = pd.DataFrame(rows).sort_values(
        ["subset_size", "objective", "duration", "start_name", "method"],
        ascending=[True, True, True, True, True],
    )
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_CSV, index=False)

    best_by_subset = frame.groupby("subset_size", as_index=False).first()
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "representative_subset_sizes": list(REPRESENTATIVE_SUBSET_SIZES),
                "methods": list(METHODS),
                "objective_name": OBJECTIVE_NAME,
                "results_csv": str(RESULTS_CSV),
                "best_by_subset": best_by_subset.to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
