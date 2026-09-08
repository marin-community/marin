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
"""Audit one- and two-phase optima of the pooled late-utility model."""

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
    analyze_centered_recency_optima as optimum,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_centered_recency_separate_heads_reorder as reorder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_nested_coverage_dsp_optima as nested_optimum,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_centered_recency_residual as centered,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_global_benefit_crowding as crowding,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_late_utility_bottleneck as bottleneck,
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

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "late_utility_bottleneck_optima_20260710"
DEFAULT_BENCHMARK_DIR = pooled.REFERENCE_OUTPUTS / "late_utility_bottleneck_multiseed_20260710"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
SEPARATE_HEADS_3E18 = {"uncheatable": 0.988712, "table9": 1.067690}


def tied_weights_from_logits(logits: np.ndarray) -> np.ndarray:
    """Map one unconstrained vector to an exact constant-mixture schedule."""
    exponent = np.exp(logits - np.max(logits))
    weights = exponent / exponent.sum()
    return np.stack([weights, weights])


def optimize_candidate(
    dataset: pooled.Dataset,
    model: bottleneck.FittedBottleneckModel,
    natural: np.ndarray,
    kl_reg: float,
    policy: str,
) -> np.ndarray:
    """Optimize either the full or tied policy with the same KL objective."""
    alpha0, alpha1 = centered.phase_fractions(dataset)
    if policy == "two_phase":

        def decode(logits: np.ndarray) -> np.ndarray:
            return optimum.weights_from_logits(logits, dataset.m)

        starts = [np.log(np.clip(np.stack([natural, natural]), 1e-12, 1.0)).reshape(-1)]
        starts.extend(
            np.log(np.clip(dataset.weights[index], 1e-12, 1.0)).reshape(-1) for index in np.argsort(dataset.y)[:12]
        )
    elif policy == "single_phase":

        def decode(logits: np.ndarray) -> np.ndarray:
            return tied_weights_from_logits(logits)

        starts = [np.log(np.clip(natural, 1e-12, 1.0))]
        for index in np.argsort(dataset.y)[:12]:
            aggregate = alpha0 * dataset.weights[index, 0] + alpha1 * dataset.weights[index, 1]
            starts.append(np.log(np.clip(aggregate, 1e-12, 1.0)))
    else:
        raise ValueError(f"Unknown policy {policy!r}")

    def objective(logits: np.ndarray) -> float:
        weights = decode(logits)
        prediction = float(model.predict(weights[None, :, :])[0])
        return prediction + kl_reg * nested_optimum.weighted_kl(weights, natural, alpha0, alpha1)

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
            best_weights = decode(np.asarray(result.x, dtype=float))
    if best_weights is None:
        raise RuntimeError(f"No optimizer result for {policy}")
    return best_weights


def oof_rmse(benchmark_dir: Path, dataset_name: str, model_name: str) -> float:
    """Return the multiseed OOF RMSE for one predeclared cap."""
    summary = pd.read_csv(benchmark_dir / "cv_summary.csv")
    row = summary.loc[summary["dataset"].eq(dataset_name) & summary["model"].eq(model_name)]
    if len(row) != 1:
        raise ValueError(f"Expected one CV row for {dataset_name}/{model_name}, found {len(row)}")
    return float(row.iloc[0]["oof_rmse_mean"])


def analyze_objective(
    objective_name: str,
    caps: list[float],
    kl_values: list[float],
    benchmark_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one objective and audit both policy classes along its KL path."""
    datasets, _external = centered.load_datasets()
    dataset_name = f"300m_{objective_name}"
    dataset = datasets[dataset_name]
    natural = nested_optimum.natural_weights(dataset)
    proportional = np.stack([natural, natural])
    alpha0, alpha1 = centered.phase_fractions(dataset)
    base = crowding.fit_base_model(dataset, np.arange(dataset.n), maxiter=100, coarse_top_k=3)
    incumbent = reorder.weights_from_frame(reorder.separate_heads_path(objective_name), dataset.domain_names)
    separate_packet, _domains, _natural, _tokens, _budget, _folds = bowl.load_objective(objective_name)
    separate_predictor = separate.build_predictors(separate_packet)["separate_heads"]
    sampled_total_epochs = np.max(
        dataset.weights[:, 0, :] * dataset.c0[None, :] + dataset.weights[:, 1, :] * dataset.c1[None, :],
        axis=1,
    )
    sampled_epoch_p95 = float(np.quantile(sampled_total_epochs, 0.95))

    rows: list[dict[str, float | str | bool]] = []
    weight_rows: list[dict[str, float | str | int]] = []
    for cap in caps:
        config = bottleneck.BottleneckConfig(cap)
        model = bottleneck.fit_from_base(dataset, np.arange(dataset.n), config, base)
        model_oof_rmse = oof_rmse(benchmark_dir, dataset_name, config.name)
        proportional_prediction = float(model.predict(proportional[None, :, :])[0])
        incumbent_prediction = float(model.predict(incumbent[None, :, :])[0])
        incumbent_optimism = float(np.min(dataset.y) - incumbent_prediction)
        for policy in ("single_phase", "two_phase"):
            for kl_reg in kl_values:
                print(f"Optimizing {dataset_name}/{config.name}/{policy} KL={kl_reg:g}", flush=True)
                candidate = optimize_candidate(dataset, model, natural, kl_reg, policy)
                aggregate = alpha0 * candidate[0] + alpha1 * candidate[1]
                tied = np.stack([aggregate, aggregate])
                prediction = float(model.predict(candidate[None, :, :])[0])
                tied_prediction = float(model.predict(tied[None, :, :])[0])
                nearest_tv, local_observed_min, local_residual_max = optimum.local_diagnostics(
                    dataset,
                    model,
                    candidate,
                )
                max_total_epoch = float(np.max(candidate[0] * dataset.c0 + candidate[1] * dataset.c1))
                optimism = float(np.min(dataset.y) - prediction)
                pair_noise_sd = np.sqrt(2.0) * optimum.NOISE_SD[objective_name]
                gain_vs_incumbent = incumbent_prediction - prediction
                gates = {
                    "passes_incumbent_calibration_gate": incumbent_optimism <= 2.0 * model_oof_rmse,
                    "passes_incumbent_gain_gate": gain_vs_incumbent >= 2.0 * pair_noise_sd,
                    "passes_optimism_gate": optimism <= 2.0 * model_oof_rmse,
                    "passes_support_gate": nearest_tv <= 0.2,
                    "passes_local_residual_gate": local_residual_max <= 2.0 * model_oof_rmse,
                    "passes_local_floor_gate": prediction >= local_observed_min - 2.0 * model_oof_rmse,
                    "passes_epoch_gate": max_total_epoch <= sampled_epoch_p95,
                }
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": config.name,
                        "cap_factor": cap,
                        "policy": policy,
                        "kl_reg": kl_reg,
                        "predicted_target": prediction,
                        "predicted_gain_vs_proportional": proportional_prediction - prediction,
                        "predicted_gain_vs_separate_heads_incumbent": gain_vs_incumbent,
                        "predicted_ordering_margin_vs_tied": tied_prediction - prediction,
                        "separate_heads_prediction": float(separate_predictor(candidate)),
                        "separate_heads_incumbent_prediction": incumbent_prediction,
                        "separate_heads_incumbent_3e18": SEPARATE_HEADS_3E18[objective_name],
                        "incumbent_optimism_below_panel_best": incumbent_optimism,
                        "tv_to_separate_heads_incumbent": nested_optimum.mean_phase_tv(candidate, incumbent),
                        "panel_best_observed": float(np.min(dataset.y)),
                        "optimism_below_panel_best": optimism,
                        "oof_rmse": model_oof_rmse,
                        "tv_to_proportional": nested_optimum.mean_phase_tv(candidate, proportional),
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
                                "dataset": dataset_name,
                                "model": config.name,
                                "cap_factor": cap,
                                "policy": policy,
                                "kl_reg": kl_reg,
                                "phase": phase,
                                "domain": domain,
                                "weight": float(weight),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def write_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    """Write compact optimum-path diagnostics."""
    for metric in (
        "predicted_target",
        "predicted_gain_vs_separate_heads_incumbent",
        "tv_to_proportional",
        "nearest_observed_tv",
        "max_total_epoch",
    ):
        figure = px.line(
            frame,
            x="kl_reg",
            y=metric,
            color="model",
            line_dash="policy",
            facet_col="dataset",
            markers=True,
            color_discrete_sequence=px.colors.diverging.RdYlGn_r,
            title=f"Late-utility bottleneck optimum path: {metric}",
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
    parser.add_argument("--cap-factors", default="inf,8,4,2,1")
    parser.add_argument("--kl-values", default="0.1,0.2,0.3,0.5,1,2,5,10")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    objectives = [value.strip() for value in args.objectives.split(",") if value.strip()]
    caps = bottleneck.parse_cap_factors(args.cap_factors)
    kl_values = pooled.parse_float_list(args.kl_values)
    result_frames = []
    weight_frames = []
    for objective in objectives:
        result, weights = analyze_objective(objective, caps, kl_values, args.benchmark_dir)
        result_frames.append(result)
        weight_frames.append(weights)
    diagnostics = pd.concat(result_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)
    diagnostics.to_csv(args.output_dir / "kl_path_diagnostics.csv", index=False)
    weights.to_csv(args.output_dir / "kl_path_weights_long.csv", index=False)
    write_plots(diagnostics, args.output_dir)
    launchable = diagnostics.loc[diagnostics["passes_all_primary_gates"]]
    report = [
        "# Late-utility bottleneck optimum audit",
        "",
        "## Candidates passing every launch gate",
        "",
        launchable.to_markdown(index=False) if not launchable.empty else "None.",
        "",
        "## Complete path",
        "",
        diagnostics.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(diagnostics.to_string(index=False))
    print(f"Wrote late-utility bottleneck optimum audit to {args.output_dir}")


if __name__ == "__main__":
    main()
