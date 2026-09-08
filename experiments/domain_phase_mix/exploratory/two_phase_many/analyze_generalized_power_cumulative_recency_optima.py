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
"""Audit KL-regularized optima of generalized cumulative-recency DSP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_centered_recency_optima as centered_optimum,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_centered_recency_separate_heads_reorder as reorder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_nested_coverage_dsp_optima as optimum,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_calibrated_cumulative_recency as generalized,
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

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "generalized_power_cumulative_recency_optima_20260710"
DEFAULT_BENCHMARK_DIR = pooled.REFERENCE_OUTPUTS / "generalized_power_cumulative_recency_300m_l2_sweep_20260710"
DEFAULT_POWER_L2 = 0.001
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
SEPARATE_HEADS_3E18 = {
    "uncheatable": 0.988712,
    "table9": 1.067690,
}


def selected_oof_rmse(benchmark_dir: Path, dataset_name: str, power_l2: float) -> float:
    """Return the OOF RMSE for the preselected shared-power fit."""
    summary = pd.read_csv(benchmark_dir / "cv_summary.csv")
    selected = summary.loc[
        summary["dataset"].eq(dataset_name) & summary["model"].eq(f"shared_power_spread_l2_{power_l2:g}")
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one benchmark row for {dataset_name}, found {len(selected)}")
    return float(selected.iloc[0]["oof_rmse_mean"])


def analyze_objective(
    objective_name: str,
    kl_values: list[float],
    benchmark_dir: Path,
    power_l2: float,
    head_l2: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one objective and trace its KL-regularized proposal path."""
    datasets, _external = centered.load_datasets()
    dataset_name = f"300m_{objective_name}"
    dataset = datasets[dataset_name]
    natural = optimum.natural_weights(dataset)
    alpha0, alpha1 = centered.phase_fractions(dataset)
    proportional = np.stack([natural, natural])
    config = generalized.ModelConfig(generalized.CalibrationKind.SHARED_POWER, power_l2)
    model = generalized.fit_model(
        dataset,
        np.arange(dataset.n),
        config,
        generalized.HEAD_L2_BY_DATASET[dataset_name] if head_l2 is None else head_l2,
        maxiter=100,
        coarse_top_k=3,
    )
    separate_packet, _domains, _natural, _tokens, _budget, _folds = bowl.load_objective(objective_name)
    separate_predictor = separate.build_predictors(separate_packet)["separate_heads"]
    incumbent = reorder.weights_from_frame(reorder.separate_heads_path(objective_name), dataset.domain_names)
    incumbent_prediction = float(model.predict(incumbent[None, :, :])[0])
    model_oof_rmse = selected_oof_rmse(benchmark_dir, dataset_name, power_l2)
    proportional_prediction = float(model.predict(proportional[None, :, :])[0])
    sampled_total_epochs = np.max(
        dataset.weights[:, 0, :] * dataset.c0[None, :] + dataset.weights[:, 1, :] * dataset.c1[None, :],
        axis=1,
    )
    sampled_epoch_p95 = float(np.quantile(sampled_total_epochs, 0.95))
    benefit_power, penalty_power = generalized.response_powers(config.kind, model.shape_parameters)[0]

    rows: list[dict[str, float | str | bool]] = []
    weight_rows: list[dict[str, float | str | int]] = []
    for kl_reg in kl_values:
        print(f"Optimizing {dataset_name} KL={kl_reg:g}", flush=True)
        candidate = centered_optimum.optimize_candidate(dataset, model, natural, kl_reg)
        aggregate = alpha0 * candidate[0] + alpha1 * candidate[1]
        tied = np.stack([aggregate, aggregate])
        prediction = float(model.predict(candidate[None, :, :])[0])
        tied_prediction = float(model.predict(tied[None, :, :])[0])
        nearest_tv, local_observed_min, local_residual_max = centered_optimum.local_diagnostics(
            dataset,
            model,
            candidate,
        )
        max_total_epoch = float(np.max(candidate[0] * dataset.c0 + candidate[1] * dataset.c1))
        optimism = float(np.min(dataset.y) - prediction)
        gates = {
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
                "power_l2": power_l2,
                "head_l2": generalized.HEAD_L2_BY_DATASET[dataset_name] if head_l2 is None else head_l2,
                "kl_reg": kl_reg,
                "benefit_power": benefit_power,
                "penalty_power": penalty_power,
                "predicted_target": prediction,
                "predicted_gain_vs_proportional": proportional_prediction - prediction,
                "predicted_gain_vs_separate_heads_incumbent": incumbent_prediction - prediction,
                "predicted_ordering_margin_vs_tied": tied_prediction - prediction,
                "separate_heads_prediction": float(separate_predictor(candidate)),
                "separate_heads_incumbent_prediction": incumbent_prediction,
                "separate_heads_incumbent_3e18": SEPARATE_HEADS_3E18[objective_name],
                "tv_to_separate_heads_incumbent": optimum.mean_phase_tv(candidate, incumbent),
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
                        "dataset": dataset_name,
                        "model": config.name,
                        "power_l2": power_l2,
                        "head_l2": generalized.HEAD_L2_BY_DATASET[dataset_name] if head_l2 is None else head_l2,
                        "kl_reg": kl_reg,
                        "phase": phase,
                        "domain": domain,
                        "weight": float(weight),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def write_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    """Write compact KL-path diagnostics."""
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
            color="dataset",
            markers=True,
            color_discrete_sequence=px.colors.diverging.RdYlGn_r,
            title=f"Generalized cumulative-recency KL path: {metric}",
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
    parser.add_argument("--power-l2-values", default=str(DEFAULT_POWER_L2))
    parser.add_argument("--head-l2", type=float)
    parser.add_argument("--kl-values", default="0,0.01,0.025,0.05,0.1,0.2,0.3,0.5,1,2,5,10")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    objectives = [part.strip() for part in args.objectives.split(",") if part.strip()]
    kl_values = pooled.parse_float_list(args.kl_values)
    power_l2_values = pooled.parse_float_list(args.power_l2_values)
    result_frames = []
    weight_frames = []
    for power_l2 in power_l2_values:
        for objective_name in objectives:
            result, weights = analyze_objective(
                objective_name,
                kl_values,
                args.benchmark_dir,
                power_l2,
                args.head_l2,
            )
            result_frames.append(result)
            weight_frames.append(weights)
    diagnostics = pd.concat(result_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)
    diagnostics.to_csv(args.output_dir / "kl_path_diagnostics.csv", index=False)
    weights.to_csv(args.output_dir / "kl_path_weights_long.csv", index=False)
    write_plots(diagnostics, args.output_dir)
    report = [
        "# Generalized cumulative-recency optimum audit",
        "",
        diagnostics.to_markdown(index=False),
        "",
        "Candidates must clear support, local calibration, optimism, and epoch gates before validation.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(diagnostics.to_string(index=False))
    print(f"Wrote generalized cumulative-recency optimum audit to {args.output_dir}")


if __name__ == "__main__":
    main()
