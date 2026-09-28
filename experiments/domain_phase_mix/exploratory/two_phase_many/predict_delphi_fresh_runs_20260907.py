# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Out-of-sample calibration of the surrogates on the 2026-09-06 validation runs.

The runs of the link, link-plus-hub, coupling and kappa-floor launches were trained after every surrogate here was
fitted and none of their mixtures is in the fit panel or the frozen bank, so they are a clean test of each
model's level far from the swarm. For each model reconstructed from a selection package, the prediction at every
run's mixture is compared with the measured value on both targets; bias, RMSE and the sign of the predicted gain
over the kappa-0 WSPU control are reported per model.

usage: uv run python predict_delphi_fresh_runs_20260907.py [--selection DIR] [--output DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_delphi_selection_20260906 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import materialize_delphi_link_validation_20260906 as mat

REFERENCE = Path(__file__).resolve().parent / "reference_outputs"
LAUNCHES = {
    "link": REFERENCE / "delphi_link_validation_3e18_20260906",
    "hub": REFERENCE / "delphi_link_hub_validation_3e18_20260906",
    "coupling": REFERENCE / "delphi_coupling_validation_3e18_20260906",
    "kappa_floor": REFERENCE / "delphi_kappa_floor_validation_3e18_20260907",
    "kappa_floor_kl05": REFERENCE / "delphi_kappa_floor_validation_kl05_3e18_20260907",
    "kappa_floor_flat": REFERENCE / "delphi_kappa_floor_flat15_validation_3e18_20260907",
    "frozen_procedure": REFERENCE / "delphi_frozen_procedure_validation_3e18_20260908",
}
CONTROLS = {("uncheatable", 6): 0.9834, ("table9", 6): 1.0722, ("table9", 8): 1.0736}
MODELS = (
    "weibull_softplus_unscaled",
    "weibull_softplus_unscaled@log_deficit_bounded_link",
    "weibull_softplus_unscaled@fitted_floor_link",
)


def measured_runs(buckets: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for launch, directory in LAUNCHES.items():
        weights = pd.read_csv(directory / "runtime_materialization" / "candidate_weights.csv")
        measured = pd.read_csv(directory / "measured_results.csv").set_index("candidate_id")
        for candidate, table in weights.groupby("candidate_id"):
            vector = table.set_index("domain").weight.reindex(buckets).fillna(0.0).to_numpy(float)
            row = measured.loc[candidate]
            if row.status != "measured":
                continue
            rows.append(
                {
                    "launch": launch,
                    "candidate_id": candidate,
                    "target": row.target,
                    "epoch_cap": int(row.epoch_cap),
                    "measured_uncheatable": float(row.uncheatable_bpb),
                    "measured_table9": float(row.table9_macro_bpb),
                    "weights": vector,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, default=REFERENCE / "delphi_fitted_floor_selection_20260907")
    parser.add_argument("--output", type=Path, default=REFERENCE / "delphi_fresh_run_calibration_20260907")
    parser.add_argument("--methods", default=",".join(MODELS), help="registry ids with fold -1 shards in --selection")
    args = parser.parse_args()
    models = tuple(args.methods.split(","))
    args.output.mkdir(parents=True, exist_ok=True)
    data = benchmark.read_npz(benchmark.DEFAULT_OUTPUT / "inputs" / "panel.npz")
    buckets = tuple(map(str, data["buckets"]))
    runs = measured_runs(buckets)
    weights = np.stack(runs.weights.to_list())
    for model in models:
        for target in ("uncheatable", "table9"):
            surrogate = mat.reconstruct(model, target, args.selection)
            runs[f"{model}::{target}"] = surrogate.predict(weights)
    records = []
    for model in models:
        for target in ("uncheatable", "table9"):
            column = f"{model}::{target}"
            measured = runs[f"measured_{target}"].to_numpy(float)
            predicted = runs[column].to_numpy(float)
            fitted_for = runs.target.eq(target).to_numpy()
            records.append(
                {
                    "model": model,
                    "target": target,
                    "runs": len(runs),
                    "bias_all": float(np.mean(measured - predicted)),
                    "rmse_all": float(np.sqrt(np.mean((measured - predicted) ** 2))),
                    "bias_own_optima": float(np.mean((measured - predicted)[fitted_for])),
                    "rmse_own_optima": float(np.sqrt(np.mean(((measured - predicted)[fitted_for]) ** 2))),
                    "spearman_all": float(pd.Series(predicted).corr(pd.Series(measured), method="spearman")),
                }
            )
    summary = pd.DataFrame(records)
    table = runs.drop(columns=["weights"])
    table.to_csv(args.output / "fresh_run_predictions.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    short = {m: m.split("@")[-1] if "@" in m else "wspu" for m in models}
    show = table[["launch", "candidate_id", "target", "measured_uncheatable", "measured_table9"]].copy()
    for model in models:
        for target in ("uncheatable", "table9"):
            show[f"{short[model]}:{target[:2]}"] = table[f"{model}::{target}"]
    print(show.round(4).to_string(index=False))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
