# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Error anatomy of panel-fitted models on the Delphi conditional dose-response runs.

Fits each requested model on the canonical 280-run Delphi panel exactly as the heldout stage does (per
component, heldout inner folds), predicts all 277 dose-response runs, and decomposes the predicted change from
the proportional anchor into the benefit and harm blocks of the design. Uncheatable values come from the
exact final-step recovery table when one is supplied.
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
    prepare_single_phase_heldout_benchmark_20260902 as heldout,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)

PANEL = "delphi_3e18_39bucket"
DOSE_DIR = SCRIPT_DIR / "reference_outputs" / "bucket_epoch_dose_response_20260729"
RAW_DIR = DOSE_DIR / "recovery" / "delphi_3e18_20260902"
ANCHOR = "p000_proportional_anchor"
BENEFIT_PREFIXES = (
    "bucket_signal:",
    "singleton_signal:",
    "pooled_base_signal:",
    "bucket_excess_signal:",
    "family_signal:",
    "pair_signal:",
)


def dose_table(recovery: Path | None) -> pd.DataFrame:
    manifest = pd.read_csv(DOSE_DIR / "full" / "delphi_3e18" / "run_manifest.csv")
    results = pd.read_csv(RAW_DIR / "heldout_results.csv")
    components = pd.read_csv(RAW_DIR / "uncheatable_components.csv")
    table = manifest.merge(results, on="run_name", how="inner")
    if recovery is not None:
        fixed = pd.read_csv(recovery)
        fixed = fixed[fixed["status"].eq("ok")]
        lookup = dict(zip(fixed["run_id"], fixed["final_U"], strict=True))
        table["uncheatable_bpb"] = [
            lookup.get(run_id, value)
            for run_id, value in zip(table["training_wandb_run_id"], table["uncheatable_bpb"], strict=True)
        ]
        for component in heldout.UNCHEATABLE_COMPONENTS:
            values = dict(zip(fixed["run_name"], fixed[component], strict=True))
            mask = components["component"].eq(component)
            components.loc[mask, "bpb"] = [
                values.get(run_name, value)
                for run_name, value in zip(components.loc[mask, "run_name"], components.loc[mask, "bpb"], strict=True)
            ]
    return table, components


def dose_features(panel: harness.BenchPanel, runs: pd.Series):
    weights = pd.read_csv(DOSE_DIR / "full" / "delphi_3e18" / "phase_weights.csv")
    phase = weights[weights["phase"].eq("phase_0")].pivot(index="run_name", columns="domain", values="weight")
    matrix = phase.reindex(index=runs, columns=panel.buckets).to_numpy(float)
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("dose weights are not normalized")
    return dataclasses.replace(
        panel.features, exposures=matrix * panel.features.inventory[None, :], weights=matrix, label=f"{PANEL}|dose"
    )


def fit_component(model_id: str, target: str, component_index: int, query) -> dict[str, np.ndarray]:
    panel = harness.load_panel(PANEL)
    entry = registry.ENTRY_BY_ID[model_id]
    group = panel.group(target)
    component = group.components[component_index]
    features = dataclasses.replace(registry.apply_transform(panel.features, entry), component=str(component))
    model = entry.build(features)
    rows = np.arange(panel.rows)
    fitted = model.fit(
        features,
        group.outcomes[:, component_index],
        rows,
        harness.heldout_inner_folds(panel),
        harness._seed(harness.FitTask(model_id, PANEL, target, component_index, component, 0, 0)),
    )
    query_features = dataclasses.replace(registry.apply_transform(query, entry), component=str(component))
    prediction = np.asarray(model.predict(fitted, query_features, np.arange(query.rows)), dtype=float)
    result = {"prediction": prediction, "benefit": np.full(query.rows, np.nan), "harm": np.full(query.rows, np.nan)}
    if isinstance(model, models.GridModel):
        design = model.design(query_features, fitted.shape)
        parts = design.values * fitted.head.coefficients[None, :]
        benefit = np.array([name.startswith(BENEFIT_PREFIXES) for name in design.names])
        result["benefit"] = parts[:, benefit].sum(axis=1)
        result["harm"] = parts[:, ~benefit].sum(axis=1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="weibull_softplus_unscaled,dsp_total_exposure")
    parser.add_argument("--recovery", type=Path, required=True, help="exact final-step Uncheatable recovery table")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    panel = harness.load_panel(PANEL)
    table, _components = dose_table(args.recovery)
    query = dose_features(panel, table["run_name"])
    anchor = int(np.flatnonzero(table["run_name"].eq(ANCHOR))[0])
    # The anchor and the 39 zero-dose controls are exact panel coordinates: in-sample for every panel fit.
    in_panel = np.abs(query.weights[:, None, :] - panel.features.weights[None, :, :]).sum(axis=-1).min(axis=1) < 1e-8
    print(f"dose runs that are panel coordinates (in sample): {int(in_panel.sum())}", flush=True)
    frames = []
    for model_id in [token.strip() for token in args.models.split(",") if token.strip()]:
        for target, column in (("uncheatable", "uncheatable_bpb"), ("table9", "table9_macro_bpb")):
            group = panel.group(target)
            with harness.parallel_config(backend="loky", inner_max_num_threads=1):
                parts = Parallel(n_jobs=args.workers)(
                    delayed(fit_component)(model_id, target, index, query) for index in range(len(group.components))
                )
            stacked = {
                key: np.stack([part[key] for part in parts], axis=1) @ group.aggregation_weights for key in parts[0]
            }
            measured = table[column].to_numpy(float)
            frame = pd.DataFrame(
                {
                    "model": model_id,
                    "target": target,
                    "run_name": table["run_name"],
                    "focal_domain": table["focal_domain"].fillna("anchor"),
                    "multiplier": table["epoch_multiplier"],
                    "focal_epochs": table["target_simulated_epochs"],
                    "training_state": table["training_wandb_state"],
                    "in_panel": in_panel,
                    "measured": measured,
                    "prediction": stacked["prediction"],
                    "benefit_part": stacked["benefit"],
                    "harm_part": stacked["harm"],
                }
            )
            frame["measured_delta"] = frame["measured"] - measured[anchor]
            frame["predicted_delta"] = frame["prediction"] - stacked["prediction"][anchor]
            frame["benefit_delta"] = frame["benefit_part"] - stacked["benefit"][anchor]
            frame["harm_delta"] = frame["harm_part"] - stacked["harm"][anchor]
            frame["residual"] = frame["prediction"] - frame["measured"]
            frames.append(frame)
    runs = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "dose_anatomy_runs.csv", index=False)
    all_valid = runs[runs["measured"].notna()]
    valid = all_valid[~all_valid["in_panel"]]
    in_sample = all_valid[all_valid["in_panel"]]
    in_sample.to_csv(args.output_dir / "dose_anatomy_in_sample_rows.csv", index=False)
    by_multiplier = (
        valid.groupby(["model", "target", "multiplier"])
        .agg(
            runs=("residual", "size"),
            residual_mean=("residual", "mean"),
            residual_median=("residual", "median"),
            rmse=("residual", lambda values: float(np.sqrt(np.mean(values**2)))),
            measured_delta_median=("measured_delta", "median"),
            predicted_delta_median=("predicted_delta", "median"),
            benefit_delta_median=("benefit_delta", "median"),
            harm_delta_median=("harm_delta", "median"),
        )
        .reset_index()
    )
    by_multiplier.to_csv(args.output_dir / "dose_anatomy_by_multiplier.csv", index=False)
    high = valid[valid["multiplier"] >= 8.0]
    by_bucket = (
        high.groupby(["model", "target", "focal_domain"])
        .agg(
            runs=("residual", "size"),
            residual_mean=("residual", "mean"),
            max_multiplier=("multiplier", "max"),
            measured_delta_at_max=("measured_delta", "last"),
            predicted_delta_at_max=("predicted_delta", "last"),
            harm_delta_at_max=("harm_delta", "last"),
        )
        .reset_index()
    )
    by_bucket.to_csv(args.output_dir / "dose_anatomy_by_bucket_high_dose.csv", index=False)
    pd.set_option("display.width", 250)
    print("out-of-sample dose rows only (the anchor and the zero-dose controls are panel coordinates):")
    print(by_multiplier.round(4).to_string(index=False))
    deletion = in_sample[in_sample["multiplier"].eq(0.0)]
    print("\nin-sample fit at the zero-dose controls (panel coordinates; residual = fitted - measured, not a forecast):")
    print(
        deletion.groupby(["model", "target"])
        .agg(
            rows=("residual", "size"),
            residual_mean=("residual", "mean"),
            rmse=("residual", lambda values: float(np.sqrt(np.mean(values**2)))),
        )
        .round(4)
        .to_string()
    )
    for (model_id, target), subset in valid.groupby(["model", "target"]):
        delta = subset[subset["multiplier"] != 1.0]
        design = np.column_stack([delta["benefit_delta"], delta["harm_delta"]])
        finite = np.isfinite(design).all(axis=1)
        if finite.sum() > 10 and np.abs(design[finite]).max() > 0:
            coef, *_ = np.linalg.lstsq(design[finite], delta["measured_delta"].to_numpy(float)[finite], rcond=None)
            print(
                f"{model_id}/{target}: measured delta ~ {coef[0]:.3f} x benefit delta + {coef[1]:.3f} x harm delta "
                "(1.0 = calibrated)"
            )
        worst = subset.reindex(subset["residual"].abs().sort_values(ascending=False).index).head(8)
        print(f"{model_id}/{target}: largest |residual| runs")
        print(
            worst[["run_name", "multiplier", "training_state", "measured", "prediction", "residual"]]
            .round(4)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
