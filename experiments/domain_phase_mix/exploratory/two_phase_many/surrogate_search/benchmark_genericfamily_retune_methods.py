# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Compare optimizer choices for retuning GRP nonlinear parameters on subsets."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    load_generic_family_packet,
)
from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    ccglobalpremium_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    ccpairtotal_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    CSV_PATH,
    GENERICFAMILY_RETUNED_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    OBJECTIVE_METRIC,
    VALIDATED_GLOBAL_BPB,
    VALIDATED_PAIR_BPB,
    _subset_packet,
    _summary_weights,
    tune_genericfamily_subset_params,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = SCRIPT_DIR / "genericfamily_retune_method_benchmark.csv"
SUMMARY_JSON = SCRIPT_DIR / "genericfamily_retune_method_benchmark.json"
METHODS = ("L-BFGS-B", "Powell", "Nelder-Mead")


def main() -> None:
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_genericfamily_retune_method_benchmark",
    )
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    valid_weights = np.stack(
        [
            _summary_weights(ccglobalpremium_retainedtotal_summary(), packet.base.domain_names),
            _summary_weights(ccpairtotal_retainedtotal_summary(), packet.base.domain_names),
        ],
        axis=0,
    )
    valid_y = np.asarray([VALIDATED_GLOBAL_BPB, VALIDATED_PAIR_BPB], dtype=float)

    rows: list[dict[str, float | int | bool | str]] = []
    for subset_size in GENERICFAMILY_RETUNED_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES:
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
        train_packet = _subset_packet(packet, subset_indices)
        for method in METHODS:
            start = perf_counter()
            metrics, result = tune_genericfamily_subset_params(
                train_packet,
                valid_weights,
                valid_y,
                method=method,
                seed=0,
            )
            rows.append(
                {
                    "subset_size": subset_size,
                    "method": method,
                    "duration": perf_counter() - start,
                    "success": bool(metrics["success"]),
                    "nit": int(getattr(result, "nit", -1)),
                    "nfev": int(getattr(result, "nfev", -1)),
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

    frame = pd.DataFrame(rows).sort_values(["subset_size", "objective", "duration"], ascending=[True, True, True])
    frame.to_csv(RESULTS_CSV, index=False)

    best_by_subset = (
        frame.sort_values(["subset_size", "objective", "duration"], ascending=[True, True, True])
        .groupby("subset_size", as_index=False)
        .first()
    )
    best_method_counts = frame.loc[frame.groupby("subset_size")["objective"].idxmin(), "method"].value_counts().to_dict()
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "representative_subset_sizes": list(GENERICFAMILY_RETUNED_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES),
                "methods": list(METHODS),
                "results_csv": str(RESULTS_CSV),
                "best_by_subset": best_by_subset.to_dict(orient="records"),
                "best_method_counts": best_method_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
