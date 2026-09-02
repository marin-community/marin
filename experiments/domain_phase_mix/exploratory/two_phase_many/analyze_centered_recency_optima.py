# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Audit KL-regularized optima of the centered recency residual model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_nested_coverage_dsp_optima as optimum,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_centered_recency_residual as centered,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_two_phase_canonical_bowl_candidates_300m as bowl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_lf_sepheads_kl_sweep_300m as separate,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "centered_recency_optima_20260710"
DEFAULT_BENCHMARK_DIR = pooled.REFERENCE_OUTPUTS / "centered_recency_residual_20260710"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
NOISE_SD = {
    "uncheatable": 0.000913,
    "table9": 0.003772,
}


def weights_from_logits(logits: np.ndarray, domains: int) -> np.ndarray:
    weights = np.zeros((2, domains), dtype=float)
    for phase in range(2):
        values = logits[phase * domains : (phase + 1) * domains]
        exponent = np.exp(values - np.max(values))
        weights[phase] = exponent / exponent.sum()
    return weights


def optimize_candidate(
    dataset: pooled.Dataset,
    model: centered.FittedCandidate,
    natural: np.ndarray,
    kl_reg: float,
) -> np.ndarray:
    alpha0, alpha1 = centered.phase_fractions(dataset)
    domains = dataset.m

    def objective(logits: np.ndarray) -> float:
        weights = weights_from_logits(logits, domains)
        prediction = float(model.predict(weights[None, :, :])[0])
        return prediction + kl_reg * optimum.weighted_kl(weights, natural, alpha0, alpha1)

    starts = [np.log(np.clip(np.stack([natural, natural]), 1e-12, 1.0)).reshape(-1)]
    starts.extend(
        np.log(np.clip(dataset.weights[index], 1e-12, 1.0)).reshape(-1) for index in np.argsort(dataset.y)[:12]
    )
    best_value = np.inf
    best_weights = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10, "maxls": 30},
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_weights = weights_from_logits(np.asarray(result.x, dtype=float), domains)
    if best_weights is None:
        raise RuntimeError("No centered-recency optimizer result")
    return best_weights


def local_diagnostics(
    dataset: pooled.Dataset,
    model: centered.FittedCandidate,
    weights: np.ndarray,
    count: int = 3,
) -> tuple[float, float, float]:
    distances = 0.5 * np.abs(dataset.weights - weights[None, :, :]).sum(axis=2).mean(axis=1)
    indices = np.argsort(distances)[:count]
    predictions = model.predict(dataset.weights[indices])
    residual_max = float(np.max(np.abs(predictions - dataset.y[indices])))
    return float(distances[indices[0]]), float(np.min(dataset.y[indices])), residual_max


def selected_l2(benchmark_dir: Path, dataset: str, kind: centered.ResidualKind) -> float:
    selected = pd.read_csv(benchmark_dir / "selected_configs.csv")
    row = selected.loc[selected["dataset"].eq(dataset) & selected["model"].str.startswith(kind.value)]
    if len(row) != 1:
        raise ValueError(f"Expected one selected row for {dataset}/{kind.value}, found {len(row)}")
    return float(str(row.iloc[0]["model"]).rsplit("_", maxsplit=1)[-1])


def oof_rmse(benchmark_dir: Path, dataset: str, kind: centered.ResidualKind) -> float:
    selected = pd.read_csv(benchmark_dir / "selected_configs.csv")
    row = selected.loc[selected["dataset"].eq(dataset) & selected["model"].str.startswith(kind.value)]
    return float(row.iloc[0]["oof_rmse_mean"])


def analyze_objective(
    objective_name: str,
    kl_values: list[float],
    benchmark_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets, _external = centered.load_datasets()
    dataset = datasets[f"300m_{objective_name}"]
    natural = optimum.natural_weights(dataset)
    alpha0, alpha1 = centered.phase_fractions(dataset)
    proportional = np.stack([natural, natural])
    separate_packet, _domains, _natural, _tokens, _budget, _folds = bowl.load_objective(objective_name)
    separate_predictor = separate.build_predictors(separate_packet)["separate_heads"]
    sampled_total_epochs = np.max(
        dataset.weights[:, 0, :] * dataset.c0[None, :] + dataset.weights[:, 1, :] * dataset.c1[None, :],
        axis=1,
    )
    sampled_epoch_p95 = float(np.quantile(sampled_total_epochs, 0.95))
    rows = []
    weight_rows = []
    for kind in centered.ResidualKind:
        l2 = selected_l2(benchmark_dir, dataset.name, kind)
        model = centered.fit_full_candidate(dataset, kind, l2)
        baseline = centered.FittedCandidate(
            backbone=model.backbone,
            residual=None,
            c0=model.c0,
            c1=model.c1,
            alpha0=model.alpha0,
            alpha1=model.alpha1,
        )
        model_oof_rmse = oof_rmse(benchmark_dir, dataset.name, kind)
        proportional_prediction = float(model.predict(proportional[None, :, :])[0])
        for kl_reg in kl_values:
            print(f"Optimizing {dataset.name}/{kind.value} KL={kl_reg:g}", flush=True)
            candidate = optimize_candidate(dataset, model, natural, kl_reg)
            aggregate = alpha0 * candidate[0] + alpha1 * candidate[1]
            tied = np.stack([aggregate, aggregate])
            prediction = float(model.predict(candidate[None, :, :])[0])
            tied_prediction = float(model.predict(tied[None, :, :])[0])
            backbone_prediction = float(baseline.predict(candidate[None, :, :])[0])
            separate_prediction = float(separate_predictor(candidate))
            nearest_tv, local_observed_min, local_residual_max = local_diagnostics(dataset, model, candidate)
            phase0_epochs = candidate[0] * dataset.c0
            phase1_epochs = candidate[1] * dataset.c1
            max_total_epoch = float(np.max(phase0_epochs + phase1_epochs))
            ordering_margin = tied_prediction - prediction
            pair_noise_sd = np.sqrt(2.0) * NOISE_SD[objective_name]
            optimism = float(np.min(dataset.y) - prediction)
            gates = {
                "passes_optimism_gate": optimism <= 2.0 * model_oof_rmse,
                "passes_support_gate": nearest_tv <= 0.2,
                "passes_local_residual_gate": local_residual_max <= 2.0 * model_oof_rmse,
                "passes_local_floor_gate": prediction >= local_observed_min - 2.0 * model_oof_rmse,
                "passes_power_gate": ordering_margin >= 2.0 * pair_noise_sd,
                "passes_epoch_gate": max_total_epoch <= sampled_epoch_p95,
            }
            rows.append(
                {
                    "dataset": dataset.name,
                    "model": kind.value,
                    "residual_l2": l2,
                    "kl_reg": kl_reg,
                    "predicted_target": prediction,
                    "backbone_prediction": backbone_prediction,
                    "separate_head_prediction": separate_prediction,
                    "predicted_gain_vs_proportional": proportional_prediction - prediction,
                    "predicted_ordering_margin_vs_tied": ordering_margin,
                    "ordering_margin_in_3e18_diff_sd": ordering_margin / pair_noise_sd,
                    "panel_best_observed": float(np.min(dataset.y)),
                    "optimism_below_panel_best": optimism,
                    "oof_rmse": model_oof_rmse,
                    "tv_to_proportional": optimum.mean_phase_tv(candidate, proportional),
                    "phase_tv": float(0.5 * np.abs(candidate[0] - candidate[1]).sum()),
                    "nearest_observed_tv": nearest_tv,
                    "local_observed_min": local_observed_min,
                    "local_residual_max": local_residual_max,
                    "max_weight": float(np.max(candidate)),
                    "max_total_epoch": max_total_epoch,
                    "sampled_total_epoch_p95": sampled_epoch_p95,
                    **gates,
                    "passes_all_primary_gates": all(gates.values()),
                }
            )
            for phase in range(2):
                for domain, weight in zip(dataset.domain_names, candidate[phase], strict=True):
                    weight_rows.append(
                        {
                            "dataset": dataset.name,
                            "model": kind.value,
                            "residual_l2": l2,
                            "kl_reg": kl_reg,
                            "phase": phase,
                            "domain": domain,
                            "weight": float(weight),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def write_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    for metric in (
        "predicted_target",
        "predicted_ordering_margin_vs_tied",
        "tv_to_proportional",
        "max_total_epoch",
        "nearest_observed_tv",
    ):
        figure = px.line(
            frame,
            x="kl_reg",
            y=metric,
            color="model",
            facet_col="dataset",
            markers=True,
            color_discrete_sequence=px.colors.diverging.RdYlGn_r,
            title=f"Centered recency KL path: {metric}",
        )
        figure.write_html(
            output_dir / f"kl_path_{metric}.html",
            include_plotlyjs="cdn",
            config=PLOT_CONFIG,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--objectives", default="uncheatable,table9")
    parser.add_argument("--kl-values", default="0,0.05,0.1,0.2,0.3,0.5,1,2,5,10")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    objectives = [part.strip() for part in args.objectives.split(",") if part.strip()]
    unknown = sorted(set(objectives).difference(NOISE_SD))
    if unknown:
        raise ValueError(f"Unknown objectives: {unknown}")
    kl_values = pooled.parse_float_list(args.kl_values)
    result_frames = []
    weight_frames = []
    for objective_name in objectives:
        result, weights = analyze_objective(objective_name, kl_values, args.benchmark_dir)
        result_frames.append(result)
        weight_frames.append(weights)
    diagnostics = pd.concat(result_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)
    diagnostics.to_csv(args.output_dir / "kl_path_diagnostics.csv", index=False)
    weights.to_csv(args.output_dir / "kl_path_weights_long.csv", index=False)
    write_plots(diagnostics, args.output_dir)
    report = [
        "# Centered recency residual optimum audit",
        "",
        diagnostics.to_markdown(index=False),
        "",
        "Primary gates require support, local calibration, plausible epochs, and a predicted ordering effect "
        "large enough to resolve against the 3e18 repeat-noise floor.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(diagnostics.to_string(index=False))
    print(f"Wrote centered-recency optimum audit to {args.output_dir}")


if __name__ == "__main__":
    main()
