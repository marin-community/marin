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
"""Gate late learning by one rank-1 phase-0 capability state.

The calibrated cumulative-recency DSP backbone is kept fixed. Its cumulative
benefit amplitudes define a scalar phase-0 learned state

    z0 = sum_i a_i S_i(e0_i) / sum_i a_i.

One bounded scalar ``gamma`` gates the late benefit score. This is the rank-1
cross-domain interaction ``a_late * gamma * a_early.T`` and costs one parameter
independent of bucket count. ``full`` gates the full late score; ``centered``
gates only the late-response difference from the aggregate-matched tied
schedule, making the added term exactly zero for tied schedules.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_calibrated_cumulative_recency as generalized,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_centered_recency_residual as centered,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "rank1_state_gated_recency_20260710"
POWER_L2 = 0.001
GAMMA_BOUND = 2.0
SCALE_FLOOR = 1e-12
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class GateKind(StrEnum):
    """Whether the rank-1 gate acts on full or centered late response."""

    FULL = "full"
    CENTERED = "centered"


@dataclass(frozen=True)
class Config:
    """Gate variant and scalar ridge strength."""

    kind: GateKind
    gamma_l2: float

    @property
    def name(self) -> str:
        return f"rank1_{self.kind.value}_gamma_l2_{self.gamma_l2:g}"


@dataclass(frozen=True)
class FittedGate:
    """Frozen shared-power backbone plus one bounded state gate."""

    backbone: generalized.FittedModel
    config: Config
    gamma: float
    capability_mean: float

    @property
    def parameter_count(self) -> int:
        return self.backbone.parameter_count + 1

    def predict(self, weights: np.ndarray) -> np.ndarray:
        feature, _capability = gate_feature(
            self.backbone,
            weights,
            self.config.kind,
            self.capability_mean,
        )
        return np.asarray(self.backbone.predict(weights) + self.gamma * feature, dtype=float)


def response_quantities(
    model: generalized.FittedModel,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return phase-0 capability and full/tied late benefit scores."""
    exposure0 = weights[:, 0, :] * model.c0[None, :]
    exposure1 = weights[:, 1, :] * model.c1[None, :]
    cumulative_parameters, recency_parameters, efficiency_offsets = generalized.channel_parameters(
        model.config.kind,
        model.shape_parameters,
        model.num_domains,
    )
    cumulative_rho, cumulative_tau = generalized.calibrated_shape(
        model.cumulative_base,
        *cumulative_parameters,
        efficiency_offsets,
    )
    recency_rho, recency_tau = generalized.calibrated_shape(
        model.recency_base,
        *recency_parameters,
        efficiency_offsets,
    )
    cumulative_powers, recency_powers = generalized.response_powers(model.config.kind, model.shape_parameters)
    phase0_benefit, _phase0_penalty = generalized.channel_features(
        exposure0,
        cumulative_rho,
        cumulative_tau,
        *cumulative_powers,
    )
    late_benefit, _late_penalty = generalized.channel_features(
        exposure1,
        recency_rho,
        recency_tau,
        *recency_powers,
    )
    aggregate = model.alpha0 * weights[:, 0, :] + model.alpha1 * weights[:, 1, :]
    tied_exposure1 = aggregate * model.c1[None, :]
    tied_late_benefit, _tied_penalty = generalized.channel_features(
        tied_exposure1,
        recency_rho,
        recency_tau,
        *recency_powers,
    )
    m = model.num_domains
    cumulative_coef = model.coef[:m]
    late_coef = model.coef[2 * m : 3 * m]
    denominator = max(float(cumulative_coef.sum()), SCALE_FLOOR)
    capability = phase0_benefit @ cumulative_coef / denominator
    late_score = late_benefit @ late_coef
    tied_late_score = tied_late_benefit @ late_coef
    return (
        np.asarray(capability, dtype=float),
        np.asarray(late_score, dtype=float),
        np.asarray(tied_late_score, dtype=float),
    )


def gate_feature(
    model: generalized.FittedModel,
    weights: np.ndarray,
    kind: GateKind,
    capability_mean: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one centered rank-1 cross-phase feature."""
    capability, late_score, tied_late_score = response_quantities(model, weights)
    gated_score = late_score if kind is GateKind.FULL else late_score - tied_late_score
    feature = -(capability - capability_mean) * gated_score
    return np.asarray(feature, dtype=float), capability


def fit_gate(
    backbone: generalized.FittedModel,
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: Config,
) -> FittedGate:
    """Fit one bounded scalar after freezing the backbone."""
    weights = dataset.weights[indices]
    capability, _late_score, _tied = response_quantities(backbone, weights)
    capability_mean = float(np.mean(capability))
    feature, _capability = gate_feature(backbone, weights, config.kind, capability_mean)
    residual = dataset.y[indices] - backbone.predict(weights)
    denominator = float(feature @ feature + config.gamma_l2)
    gamma = float(np.clip(feature @ residual / max(denominator, SCALE_FLOOR), -GAMMA_BOUND, GAMMA_BOUND))
    return FittedGate(
        backbone=backbone,
        config=config,
        gamma=gamma,
        capability_mean=capability_mean,
    )


def benchmark_dataset(
    dataset: pooled.Dataset,
    gate_configs: list[Config],
    seeds: list[int],
    n_splits: int,
    *,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fully refit grouped CV with one shared fold-local backbone."""
    backbone_config = generalized.ModelConfig(generalized.CalibrationKind.SHARED_POWER, POWER_L2)
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = centered.folds_for(dataset, seed, n_splits)
        predictions = {"shared_power_backbone": np.zeros(dataset.n, dtype=float)}
        predictions.update({config.name: np.zeros(dataset.n, dtype=float) for config in gate_configs})
        for fold_id, (train_indices, test_indices) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            backbone = generalized.fit_model(
                dataset,
                train_indices,
                backbone_config,
                generalized.HEAD_L2_BY_DATASET[dataset.name],
                maxiter=maxiter,
                coarse_top_k=coarse_top_k,
            )
            predictions["shared_power_backbone"][test_indices] = backbone.predict(dataset.weights[test_indices])
            for config in gate_configs:
                gate = fit_gate(backbone, dataset, train_indices, config)
                predictions[config.name][test_indices] = gate.predict(dataset.weights[test_indices])
                _feature, capability = gate_feature(
                    backbone,
                    dataset.weights[train_indices],
                    config.kind,
                    gate.capability_mean,
                )
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "gamma": gate.gamma,
                        "gamma_at_bound": abs(gate.gamma) >= GAMMA_BOUND - 1e-8,
                        "capability_sd": float(np.std(capability)),
                    }
                )
        for label, prediction in predictions.items():
            row = asdict(pooled.metrics(dataset, label, seed, prediction, folds))
            row["nominal_param_count"] = 4 * dataset.m + 8 if label != "shared_power_backbone" else 4 * dataset.m + 7
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def starcoder_slice_summary(
    dataset: pooled.Dataset,
    gate_configs: list[Config],
    *,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    """Measure in-sample and leave-slice-out StarCoder response."""
    backbone_config = generalized.ModelConfig(generalized.CalibrationKind.SHARED_POWER, POWER_L2)
    slice_mask = dataset.frame["phase_0_starcoder"].lt(1e-10).to_numpy(dtype=bool)
    slice_indices = np.flatnonzero(slice_mask)
    outside_indices = np.flatnonzero(~slice_mask)
    rows = []
    for protocol, train_indices in (("full_fit", np.arange(dataset.n)), ("leave_slice_out", outside_indices)):
        backbone = generalized.fit_model(
            dataset,
            train_indices,
            backbone_config,
            generalized.HEAD_L2_BY_DATASET[dataset.name],
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
        )
        candidates: list[tuple[str, Any]] = [("shared_power_backbone", backbone)]
        candidates.extend((config.name, fit_gate(backbone, dataset, train_indices, config)) for config in gate_configs)
        for label, candidate in candidates:
            prediction = candidate.predict(dataset.weights[slice_indices])
            targets = dataset.y[slice_indices]
            phase1 = dataset.frame.iloc[slice_indices]["phase_1_starcoder"].to_numpy(dtype=float)
            minimum = int(np.argmin(prediction))
            rows.append(
                {
                    "protocol": protocol,
                    "model": label,
                    "slice_rows": len(slice_indices),
                    "slice_rmse": float(np.sqrt(np.mean((prediction - targets) ** 2))),
                    "slice_spearman": float(spearmanr(targets, prediction).statistic),
                    "predicted_min_phase1_starcoder_weight": float(phase1[minimum]),
                    "predicted_min_bpb": float(prediction[minimum]),
                }
            )
    return pd.DataFrame(rows)


def configs(kinds: str, gamma_l2_values: str) -> list[Config]:
    """Parse the nested gate sweep."""
    return [
        Config(GateKind(kind.strip()), gamma_l2)
        for kind in kinds.split(",")
        if kind.strip()
        for gamma_l2 in pooled.parse_float_list(gamma_l2_values)
    ]


def write_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write a compact cross-swarm comparison."""
    long = summary.melt(
        id_vars=["dataset", "model"],
        value_vars=["oof_rmse_mean", "oof_spearman_mean", "fold_mean_regret_at_1_mean"],
        var_name="metric",
        value_name="value",
    )
    figure = px.bar(
        long,
        x="model",
        y="value",
        color="model",
        facet_row="dataset",
        facet_col="metric",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Rank-1 phase-0 state gate on late response",
    )
    figure.update_layout(showlegend=False, height=1000)
    figure.write_html(output_dir / "crossswarm_cv_comparison.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default=f"{centered.STARCODER_NAME},300m_uncheatable,300m_table9,production_uncheatable",
    )
    parser.add_argument("--kinds", default="full,centered")
    parser.add_argument("--gamma-l2-values", default="0,0.01,0.1,1")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, _external = centered.load_datasets()
    names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(names).difference(datasets))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    gate_configs = configs(args.kinds, args.gamma_l2_values)
    metric_frames = []
    parameter_frames = []
    for name in names:
        metrics, parameters = benchmark_dataset(
            datasets[name],
            gate_configs,
            pooled.parse_int_list(args.seeds),
            args.n_splits,
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
        )
        metric_frames.append(metrics)
        parameter_frames.append(parameters)
    raw = pd.concat(metric_frames, ignore_index=True)
    summary = pooled.summarize(raw)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    slices = starcoder_slice_summary(
        datasets[centered.STARCODER_NAME],
        gate_configs,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
    )
    raw.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "cv_parameters.csv", index=False)
    slices.to_csv(args.output_dir / "starcoder_slice_summary.csv", index=False)
    write_plot(summary, args.output_dir)
    report = [
        "# Rank-1 phase-0 state gate",
        "",
        "The frozen shared-power backbone receives one bounded cross-domain gate parameter.",
        "",
        summary.to_markdown(index=False),
        "",
        "## Gate stability",
        "",
        parameters.groupby(["dataset", "model"])[["gamma", "gamma_at_bound", "capability_sd"]]
        .agg(["mean", "std"])
        .reset_index()
        .to_markdown(index=False),
        "",
        "## StarCoder dense slice",
        "",
        slices.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(parameters.groupby(["dataset", "model"])[["gamma", "capability_sd"]].agg(["mean", "std"]).to_string())
    print(slices.to_string(index=False))
    print(f"Wrote rank-1 state-gated recency benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
