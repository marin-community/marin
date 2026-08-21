# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Quantify the empirical identification ceiling for phase corrections."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_joint_latent_phase_transport_round8 as round8,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_phase_effect_identifiability_round9 as round9,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round18_phase_identification_ceiling"
)
ROUND9_OUTPUT = (
    SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round9_phase_identifiability"
)
SEED = 20260719
RIDGE_ALPHA = 100.0
BOOTSTRAPS = 2000
N_SPLITS = 5
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
BLOCK_NAMES = (
    "relative aggregate",
    "phase contrast",
    "transported contrast",
    "aggregate x contrast",
    "contrast squared",
)


@dataclass(frozen=True)
class FittedRidge:
    scaler: StandardScaler
    model: Ridge

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self.scaler.transform(features)), dtype=float)

    @property
    def coefficients(self) -> np.ndarray:
        return np.asarray(self.model.coef_, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def fit_ridge(features: np.ndarray, target: np.ndarray) -> FittedRidge:
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(features), target)
    return FittedRidge(scaler, model)


def phase_effect_statistics() -> pd.DataFrame:
    _reference, indices, target, panels = round8.aligned_data()
    rows = []
    for column, label in enumerate(round8.OUTPUT_LABELS):
        panel = panels[label]
        one_phase = panel.one_phase_target[indices]
        two_phase = panel.two_phase_target[indices]
        delta = target[:, column]
        rows.append(
            {
                "output": label,
                "coordinate_count": len(indices),
                "one_phase_sd": float(np.std(one_phase, ddof=1)),
                "two_phase_sd": float(np.std(two_phase, ddof=1)),
                "phase_delta_mean": float(np.mean(delta)),
                "phase_delta_sd": float(np.std(delta, ddof=1)),
                "phase_delta_mean_abs": float(np.mean(np.abs(delta))),
                "phase_delta_min": float(np.min(delta)),
                "phase_delta_max": float(np.max(delta)),
                "fraction_two_phase_better": float(np.mean(delta < 0.0)),
                "delta_sd_over_one_phase_sd": float(np.std(delta, ddof=1) / np.std(one_phase, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def design_spectrum(feature_map: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    spectrum_rows = []
    for name, features in feature_map.items():
        standardized = StandardScaler().fit_transform(features)
        singular = np.linalg.svd(standardized, compute_uv=False)
        eigen = singular**2
        numerical_rank = int(np.sum(singular > max(standardized.shape) * np.finfo(float).eps * singular[0]))
        stable_rank = float(eigen.sum() / eigen.max())
        participation = float(eigen.sum() ** 2 / np.sum(eigen**2))
        ridge_df = float(np.sum(eigen / (eigen + RIDGE_ALPHA)))
        summary_rows.append(
            {
                "feature_set": name,
                "rows": standardized.shape[0],
                "columns": standardized.shape[1],
                "numerical_rank": numerical_rank,
                "stable_rank": stable_rank,
                "participation_ratio": participation,
                "ridge_effective_df": ridge_df,
                "rows_per_effective_df": float(standardized.shape[0] / ridge_df),
            }
        )
        fraction = eigen / eigen.sum()
        for component, value in enumerate(fraction, start=1):
            spectrum_rows.append(
                {
                    "feature_set": name,
                    "component": component,
                    "variance_fraction": float(value),
                    "cumulative_variance": float(np.sum(fraction[:component])),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(spectrum_rows)


def coefficient_stability(
    features: np.ndarray, target: np.ndarray, domain_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = fit_ridge(features, target)
    full_coefficients = full.coefficients
    fold_rows = []
    block_rows = []
    folds = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (train, _test) in enumerate(folds.split(features)):
        model = fit_ridge(features[train], target[train])
        coefficients = model.coefficients
        for output, label in enumerate(round8.OUTPUT_LABELS):
            reference = full_coefficients[output]
            candidate = coefficients[output]
            cosine = float(reference @ candidate / max(np.linalg.norm(reference) * np.linalg.norm(candidate), 1e-12))
            threshold = np.quantile(np.abs(reference), 0.75)
            material = np.abs(reference) >= threshold
            sign_agreement = float(np.mean(np.sign(reference[material]) == np.sign(candidate[material])))
            fold_rows.append(
                {
                    "fold": fold,
                    "output": label,
                    "coefficient_cosine_to_full": cosine,
                    "material_sign_agreement": sign_agreement,
                    "coefficient_norm_ratio": float(np.linalg.norm(candidate) / max(np.linalg.norm(reference), 1e-12)),
                }
            )
            block_norms = []
            for block in range(len(BLOCK_NAMES)):
                segment = candidate[block * domain_count : (block + 1) * domain_count]
                block_norms.append(float(np.linalg.norm(segment)))
            total = sum(block_norms)
            for block_name, norm in zip(BLOCK_NAMES, block_norms, strict=True):
                block_rows.append(
                    {
                        "fold": fold,
                        "output": label,
                        "feature_block": block_name,
                        "coefficient_norm_fraction": norm / max(total, 1e-12),
                    }
                )
    return pd.DataFrame(fold_rows), pd.DataFrame(block_rows)


def normalized_transfer_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    slope = float(np.polyfit(predicted, observed, 1)[0]) if np.std(predicted) > 1e-12 else np.nan
    return {
        "normalized_rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "calibration_slope": slope,
        "bias": float(np.mean(residual)),
        "sign_accuracy": float(np.mean(np.sign(observed) == np.sign(predicted))),
    }


def cross_scale_transfer(features: np.ndarray, target: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    transfer_rows = []
    coefficient_rows = []
    objective_pairs = (("uncheatable", 0, 2), ("table9", 1, 3))
    folds = list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(features))
    for objective, index_300m, index_delphi in objective_pairs:
        for source_name, source_index, destination_name, destination_index in (
            ("300m", index_300m, "delphi_3e18", index_delphi),
            ("delphi_3e18", index_delphi, "300m", index_300m),
        ):
            observed = np.full(len(features), np.nan)
            predicted = np.full(len(features), np.nan)
            for train, test in folds:
                source_mean = float(np.mean(target[train, source_index]))
                source_sd = float(np.std(target[train, source_index], ddof=1))
                destination_mean = float(np.mean(target[train, destination_index]))
                destination_sd = float(np.std(target[train, destination_index], ddof=1))
                model = fit_ridge(
                    features[train],
                    ((target[train, source_index] - source_mean) / source_sd)[:, None],
                )
                predicted[test] = model.predict(features[test]).reshape(-1)
                observed[test] = (target[test, destination_index] - destination_mean) / destination_sd
            transfer_rows.append(
                {
                    "objective": objective,
                    "source_scale": source_name,
                    "destination_scale": destination_name,
                    **normalized_transfer_metrics(observed, predicted),
                }
            )

        scaler = StandardScaler().fit(features)
        transformed = scaler.transform(features)
        coefficients = []
        for index in (index_300m, index_delphi):
            standardized_target = (target[:, index] - np.mean(target[:, index])) / np.std(target[:, index], ddof=1)
            coefficients.append(np.asarray(Ridge(alpha=RIDGE_ALPHA).fit(transformed, standardized_target).coef_))
        cosine = float(
            coefficients[0]
            @ coefficients[1]
            / max(np.linalg.norm(coefficients[0]) * np.linalg.norm(coefficients[1]), 1e-12)
        )
        coefficient_rows.append(
            {
                "objective": objective,
                "normalized_coefficient_cosine": cosine,
                "phase_delta_pearson": float(np.corrcoef(target[:, index_300m], target[:, index_delphi])[0, 1]),
                "phase_delta_spearman": float(spearmanr(target[:, index_300m], target[:, index_delphi]).statistic),
            }
        )
    return pd.DataFrame(transfer_rows), pd.DataFrame(coefficient_rows)


def fit_learning_curve(train_size: np.ndarray, rmse_squared: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(train_size)), 1.0 / train_size])
    coefficients, _residual = nnls(design, rmse_squared)
    return float(coefficients[0]), float(coefficients[1])


def learning_curve_ceiling(round9_output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(round9_output / "learning_curve_runs.csv")
    runs = runs.loc[runs["feature_set"].eq("second_order_physics") & runs["diagnostic_family"].eq("ridge")].copy()
    rng = np.random.default_rng(SEED)
    targets = (238, 280, 560, 1120)
    summary_rows = []
    extrapolation_rows = []
    for output, group in runs.groupby("output"):
        means = group.groupby("train_size", as_index=False).agg(
            rmse_squared=("rmse", lambda x: float(np.mean(np.square(x))))
        )
        floor_squared, coefficient = fit_learning_curve(
            means["train_size"].to_numpy(dtype=float),
            means["rmse_squared"].to_numpy(dtype=float),
        )
        boot = np.empty((BOOTSTRAPS, len(targets)), dtype=float)
        for sample in range(BOOTSTRAPS):
            sampled_rows = []
            for train_size, size_group in group.groupby("train_size"):
                draw = rng.choice(size_group["rmse"].to_numpy(dtype=float), size=len(size_group), replace=True)
                sampled_rows.append((float(train_size), float(np.mean(draw**2))))
            sampled = np.asarray(sampled_rows)
            sample_floor, sample_coefficient = fit_learning_curve(sampled[:, 0], sampled[:, 1])
            boot[sample] = np.sqrt(sample_floor + sample_coefficient / np.asarray(targets, dtype=float))
        summary_rows.append(
            {
                "output": output,
                "asymptotic_rmse": float(np.sqrt(floor_squared)),
                "inverse_n_coefficient": coefficient,
                "observed_delta_sd": float(round9.feature_sets()[1][:, round8.OUTPUT_LABELS.index(output)].std(ddof=1)),
            }
        )
        for column, train_size in enumerate(targets):
            point = float(np.sqrt(floor_squared + coefficient / train_size))
            extrapolation_rows.append(
                {
                    "output": output,
                    "train_size": train_size,
                    "predicted_rmse": point,
                    "ci_low": float(np.quantile(boot[:, column], 0.025)),
                    "ci_high": float(np.quantile(boot[:, column], 0.975)),
                    "fraction_of_phase_delta_sd": point
                    / float(round9.feature_sets()[1][:, round8.OUTPUT_LABELS.index(output)].std(ddof=1)),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(extrapolation_rows)


def render_diagnostics(
    target: np.ndarray,
    spectrum: pd.DataFrame,
    extrapolation: pd.DataFrame,
    output: Path,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Cross-scale matched phase deltas: Uncheatable",
            "Cross-scale matched phase deltas: Table-9",
            "Contrast-design cumulative spectrum",
            "Descriptive phase-delta learning-curve extrapolation",
        ),
    )
    colors = {
        "300m_uncheatable": "#1a9850",
        "300m_table9": "#66bd63",
        "delphi_3e18_uncheatable": "#d73027",
        "delphi_3e18_table9": "#f46d43",
    }
    for column, (left, right) in enumerate(((0, 2), (1, 3)), start=1):
        fig.add_trace(
            go.Scatter(
                x=target[:, left],
                y=target[:, right],
                mode="markers",
                marker={"size": 7, "color": "#4575b4", "opacity": 0.65},
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        fig.update_xaxes(title_text=f"{round8.OUTPUT_LABELS[left]} phase delta", row=1, col=column)
        fig.update_yaxes(title_text=f"{round8.OUTPUT_LABELS[right]} phase delta", row=1, col=column)

    spectrum_subset = spectrum.loc[spectrum["feature_set"].eq("raw_aggregate_contrast")].head(80)
    fig.add_trace(
        go.Scatter(
            x=spectrum_subset["component"],
            y=spectrum_subset["cumulative_variance"],
            mode="lines",
            line={"color": "#4575b4", "width": 3},
            name="aggregate + contrast design",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Singular component", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative design variance", range=[0, 1.02], row=2, col=1)

    for output_name, group in extrapolation.groupby("output"):
        fig.add_trace(
            go.Scatter(
                x=group["train_size"],
                y=group["predicted_rmse"],
                error_y={
                    "type": "data",
                    "array": group["ci_high"] - group["predicted_rmse"],
                    "arrayminus": group["predicted_rmse"] - group["ci_low"],
                },
                mode="lines+markers",
                name=output_name,
                line={"color": colors[output_name]},
            ),
            row=2,
            col=2,
        )
    fig.update_xaxes(title_text="Matched training coordinates", type="log", row=2, col=2)
    fig.update_yaxes(title_text="Phase-delta RMSE", row=2, col=2)
    fig.update_layout(
        title="Round 18: what phase-effect signal is identifiable from the matched fit panel?",
        template="plotly_white",
        height=1000,
        width=1500,
        legend={"orientation": "h", "y": -0.12},
    )
    fig.write_html(output, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_map, target, domain_names = round9.feature_sets()

    phase_statistics = phase_effect_statistics()
    design_summary, spectrum = design_spectrum(feature_map)
    stability, block_stability = coefficient_stability(feature_map["second_order_physics"], target, len(domain_names))
    transfer, coefficient_transfer = cross_scale_transfer(feature_map["second_order_physics"], target)
    curve_summary, extrapolation = learning_curve_ceiling(ROUND9_OUTPUT)

    phase_statistics.to_csv(args.output_dir / "phase_effect_statistics.csv", index=False)
    design_summary.to_csv(args.output_dir / "design_effective_dimension.csv", index=False)
    spectrum.to_csv(args.output_dir / "design_spectrum.csv", index=False)
    stability.to_csv(args.output_dir / "coefficient_stability.csv", index=False)
    block_stability.to_csv(args.output_dir / "coefficient_block_stability.csv", index=False)
    transfer.to_csv(args.output_dir / "cross_scale_transfer.csv", index=False)
    coefficient_transfer.to_csv(args.output_dir / "cross_scale_coefficient_alignment.csv", index=False)
    curve_summary.to_csv(args.output_dir / "learning_curve_asymptotes.csv", index=False)
    extrapolation.to_csv(args.output_dir / "learning_curve_extrapolation.csv", index=False)
    render_diagnostics(target, spectrum, extrapolation, args.output_dir / "phase_identification_ceiling.html")

    stability_summary = stability.groupby("output", as_index=False).agg(
        coefficient_cosine_mean=("coefficient_cosine_to_full", "mean"),
        coefficient_cosine_min=("coefficient_cosine_to_full", "min"),
        material_sign_agreement_mean=("material_sign_agreement", "mean"),
        coefficient_norm_ratio_sd=("coefficient_norm_ratio", "std"),
    )
    report = [
        "# Phase-effect identification ceiling",
        "",
        "This is a diagnostic study, not a candidate surrogate. It reads only the 238 coordinate-matched 300M/Delphi fit-panel pairs plus the already-frozen round-9 fit-panel learning curves. It reads no historical or adversarial heldout target.",
        "",
        "The extrapolation fits the descriptive relation RMSE(n)^2 = sigma_inf^2 + c/n. It is not an information-theoretic lower bound; it asks whether ordinary row count alone plausibly closes the observed phase-effect gap.",
        "",
        "## Phase-effect scale",
        "",
        phase_statistics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Effective design dimension",
        "",
        design_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Coefficient stability for the best round-9 diagnostic",
        "",
        stability_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Cross-scale transfer",
        "",
        transfer.to_markdown(index=False, floatfmt=".3f"),
        "",
        coefficient_transfer.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Learning-curve extrapolation",
        "",
        curve_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        extrapolation.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "The audit separates three possibilities: a low-dimensional transferable phase law, a high-dimensional but learnable correction, and a design-limited correction. Stable coefficients plus strong cross-scale transfer would support the first; improving learning curves with weak coefficient transfer support the second; a high asymptotic error fraction, unstable coefficients, and weak leave-region transfer support the third.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "ridge_alpha": RIDGE_ALPHA,
                "coordinate_count": len(target),
                "outputs": round8.OUTPUT_LABELS,
                "historical_targets_read": False,
                "adversarial_targets_read": False,
                "sealed_confirmation_targets_read": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
