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
"""Audit joint identification of a shared phase-transport subspace."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as paired_screen,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round8_joint_latent_phase_transport"
)
SEED = 20260719
N_SPLITS = 5
INNER_SPLITS = 4
BOOTSTRAPS = 2000
NUMERICAL_FLOOR = 1e-12
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
OUTPUT_LABELS = (
    "300m_uncheatable",
    "300m_table9",
    "delphi_3e18_uncheatable",
    "delphi_3e18_table9",
)


@dataclass(frozen=True)
class Config:
    remaining_offset: float
    rank: int
    l2: float
    include_contrast_cost: bool

    @property
    def key(self) -> str:
        return (
            f"remaining={self.remaining_offset:g},rank={self.rank},l2={self.l2:g},cost={int(self.include_contrast_cost)}"
        )


@dataclass(frozen=True)
class Model:
    config: Config
    feature_scale: np.ndarray
    signed_coefficients: np.ndarray
    contrast_coefficients: np.ndarray

    def predict(self, signed: np.ndarray, contrast_cost: np.ndarray) -> np.ndarray:
        normalized = signed / self.feature_scale[None, :]
        prediction = normalized @ self.signed_coefficients
        if self.config.include_contrast_cost:
            prediction += contrast_cost[:, None] * self.contrast_coefficients[None, :]
        return prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[Config]:
    return [
        Config(offset, rank, l2, include_cost)
        for offset in (0.03, 0.1, 0.3, 1.0)
        for rank in (1, 2, 3)
        for l2 in (0.0, 0.01, 0.1, 1.0, 10.0)
        for include_cost in (False, True)
    ]


def aligned_data() -> tuple[paired.PairedPanel, np.ndarray, np.ndarray, dict[str, paired.PairedPanel]]:
    panels = {
        f"{scale}_{target}": paired_screen.load_panel(scale, target)
        for scale in ("300m", "delphi_3e18")
        for target in ("uncheatable", "table9")
    }
    reference = panels["300m_uncheatable"]
    for name, panel in panels.items():
        if panel.domain_names != reference.domain_names:
            raise ValueError(f"Domain mismatch in {name}")
        if not np.allclose(panel.weights, reference.weights, atol=1e-12):
            raise ValueError(f"Coordinate mismatch in {name}")
        if not np.allclose(panel.proportional_weights, reference.proportional_weights, atol=1e-10):
            raise ValueError(f"Proportional-policy mismatch in {name}")
    valid = np.logical_and.reduce([panel.paired_mask for panel in panels.values()])
    indices = np.flatnonzero(valid)
    if len(indices) != 238:
        raise ValueError(f"Expected 238 common matched coordinates, found {len(indices)}")
    values = []
    for label in OUTPUT_LABELS:
        panel = panels[label]
        values.append(panel.two_phase_target[indices] - panel.one_phase_target[indices])
    return reference, indices, np.column_stack(values), panels


def phase_design(panel: paired.PairedPanel, indices: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
    weights = panel.weights[indices]
    aggregate = panel.alpha0 * weights[:, 0, :] + panel.alpha1 * weights[:, 1, :]
    relative_aggregate = aggregate / np.maximum(panel.proportional_weights[None, :], NUMERICAL_FLOOR)
    relative_contrast = (
        panel.alpha0
        * panel.alpha1
        * (weights[:, 1, :] - weights[:, 0, :])
        / np.maximum(panel.proportional_weights[None, :], NUMERICAL_FLOOR)
    )
    transported = relative_contrast / (config.remaining_offset + relative_aggregate)
    family_columns = []
    for members in panel.family_members:
        mass = panel.proportional_weights[members]
        mass = mass / mass.sum()
        family_columns.append(transported[:, members] @ mass)
    signed = np.column_stack(family_columns)
    contrast_cost = np.sum(panel.proportional_weights[None, :] * transported**2, axis=1)
    return signed, contrast_cost


def truncate_rank(coefficients: np.ndarray, rank: int) -> np.ndarray:
    left, singular, right = np.linalg.svd(coefficients, full_matrices=False)
    retained = min(rank, len(singular))
    return (left[:, :retained] * singular[:retained][None, :]) @ right[:retained]


def fit_model(
    signed: np.ndarray,
    contrast_cost: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    config: Config,
) -> Model:
    train_signed = signed[train]
    train_target = target[train]
    scale = np.sqrt(np.mean(train_signed**2, axis=0))
    scale = np.maximum(scale, 1e-8)
    normalized = train_signed / scale[None, :]
    gram = normalized.T @ normalized + config.l2 * np.eye(normalized.shape[1])
    contrast_coefficients = np.zeros(target.shape[1], dtype=float)
    signed_coefficients = np.zeros((normalized.shape[1], target.shape[1]), dtype=float)
    q = contrast_cost[train]
    q_norm = float(q @ q + config.l2)
    for _iteration in range(20):
        residual = train_target - q[:, None] * contrast_coefficients[None, :]
        unconstrained = np.linalg.solve(gram, normalized.T @ residual)
        next_signed = truncate_rank(unconstrained, config.rank)
        if config.include_contrast_cost:
            next_residual = train_target - normalized @ next_signed
            next_contrast = np.maximum(0.0, (q[:, None] * next_residual).sum(axis=0) / max(q_norm, 1e-12))
        else:
            next_contrast = np.zeros_like(contrast_coefficients)
        change = max(
            float(np.max(np.abs(next_signed - signed_coefficients))),
            float(np.max(np.abs(next_contrast - contrast_coefficients))),
        )
        signed_coefficients = next_signed
        contrast_coefficients = next_contrast
        if change < 1e-10:
            break
    return Model(config, scale, signed_coefficients, contrast_coefficients)


def normalized_rmse(observed: np.ndarray, predicted: np.ndarray, scale: np.ndarray) -> float:
    error = np.sqrt(np.mean((predicted - observed) ** 2, axis=0))
    return float(np.mean(error / np.maximum(scale, 1e-8)))


def select_inner(
    reference: paired.PairedPanel,
    indices: np.ndarray,
    target: np.ndarray,
    outer_train: np.ndarray,
) -> tuple[Config, pd.DataFrame]:
    folds = list(KFold(INNER_SPLITS, shuffle=True, random_state=SEED + len(outer_train)).split(outer_train))
    rows: list[dict[str, Any]] = []
    target_scale = np.std(target[outer_train], axis=0, ddof=1)
    for config in configs():
        signed, cost = phase_design(reference, indices, config)
        prediction = np.full_like(target, np.nan)
        for inner_train_local, inner_test_local in folds:
            train = outer_train[inner_train_local]
            test = outer_train[inner_test_local]
            model = fit_model(signed, cost, target, train, config)
            prediction[test] = model.predict(signed[test], cost[test])
        score = normalized_rmse(target[outer_train], prediction[outer_train], target_scale)
        rows.append({"config": config.key, "config_json": json.dumps(asdict(config), sort_keys=True), "score": score})
    table = pd.DataFrame(rows).sort_values(["score", "config"])
    return Config(**json.loads(table.iloc[0]["config_json"])), table


def nested_oof(
    reference: paired.PairedPanel,
    indices: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, list[Model]]:
    prediction = np.full_like(target, np.nan)
    selections = []
    models = []
    folds = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(folds.split(indices)):
        config, inner = select_inner(reference, indices, target, train)
        signed, cost = phase_design(reference, indices, config)
        model = fit_model(signed, cost, target, train, config)
        prediction[test] = model.predict(signed[test], cost[test])
        selections.append(
            {
                "fold": fold,
                "selected_config": config.key,
                "inner_score": float(inner.iloc[0]["score"]),
                "rank": config.rank,
                "remaining_offset": config.remaining_offset,
                "l2": config.l2,
                "include_contrast_cost": config.include_contrast_cost,
            }
        )
        models.append(model)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Nested phase predictions are incomplete")
    return prediction, pd.DataFrame(selections), models


def full_grid_oof(
    reference: paired.PairedPanel,
    indices: np.ndarray,
    target: np.ndarray,
) -> pd.DataFrame:
    folds = list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(indices))
    target_scale = np.std(target, axis=0, ddof=1)
    rows = []
    for config in configs():
        signed, cost = phase_design(reference, indices, config)
        prediction = np.full_like(target, np.nan)
        for train, test in folds:
            model = fit_model(signed, cost, target, train, config)
            prediction[test] = model.predict(signed[test], cost[test])
        rows.append(
            {
                "config": config.key,
                "config_json": json.dumps(asdict(config), sort_keys=True),
                "mean_normalized_rmse": normalized_rmse(target, prediction, target_scale),
                **{
                    f"{label}_rmse": paired_screen.scalar_metrics(target[:, output], prediction[:, output])["rmse"]
                    for output, label in enumerate(OUTPUT_LABELS)
                },
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_normalized_rmse", "config"])


def metric_table(target: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    rows = []
    for output, label in enumerate(OUTPUT_LABELS):
        candidate = paired_screen.scalar_metrics(target[:, output], prediction[:, output])
        zero = paired_screen.scalar_metrics(target[:, output], np.zeros(len(target)))
        rows.append(
            {
                "output": label,
                **candidate,
                "zero_correction_rmse": zero["rmse"],
                "relative_rmse_vs_zero": float(candidate["rmse"] / zero["rmse"] - 1.0),
            }
        )
    return pd.DataFrame(rows)


def correlation_tables(target: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows = []
    for left in range(target.shape[1]):
        for right in range(left + 1, target.shape[1]):
            x = target[:, left]
            y = target[:, right]
            boot = np.empty(BOOTSTRAPS, dtype=float)
            for sample in range(BOOTSTRAPS):
                draw = rng.integers(0, len(x), size=len(x))
                boot[sample] = pearsonr(x[draw], y[draw]).statistic
            rows.append(
                {
                    "left": OUTPUT_LABELS[left],
                    "right": OUTPUT_LABELS[right],
                    "pearson": float(pearsonr(x, y).statistic),
                    "pearson_ci_low": float(np.quantile(boot, 0.025)),
                    "pearson_ci_high": float(np.quantile(boot, 0.975)),
                    "spearman": float(spearmanr(x, y).statistic),
                }
            )
    standardized = (target - target.mean(axis=0)) / target.std(axis=0, ddof=1)
    _left, singular, _right = np.linalg.svd(standardized, full_matrices=False)
    variance = singular**2 / np.sum(singular**2)
    spectrum = pd.DataFrame(
        {
            "component": np.arange(1, len(singular) + 1),
            "singular_value": singular,
            "variance_fraction": variance,
            "cumulative_variance": np.cumsum(variance),
        }
    )
    return pd.DataFrame(rows), spectrum


def subspace_stability(models: list[Model]) -> pd.DataFrame:
    rows = []
    for left in range(len(models)):
        for right in range(left + 1, len(models)):
            a = models[left].signed_coefficients
            b = models[right].signed_coefficients
            ua, _sa, _va = np.linalg.svd(a, full_matrices=False)
            ub, _sb, _vb = np.linalg.svd(b, full_matrices=False)
            rank = min(models[left].config.rank, models[right].config.rank)
            cosine = np.linalg.svd(ua[:, :rank].T @ ub[:, :rank], compute_uv=False)
            rows.append(
                {
                    "fold_left": left,
                    "fold_right": right,
                    "common_rank": rank,
                    "minimum_principal_cosine": float(np.min(cosine)),
                    "mean_principal_cosine": float(np.mean(cosine)),
                }
            )
    return pd.DataFrame(rows)


def render_predictions(target: np.ndarray, prediction: np.ndarray, output: Path) -> None:
    figure = make_subplots(rows=2, cols=2, subplot_titles=OUTPUT_LABELS)
    for index, label in enumerate(OUTPUT_LABELS):
        row, column = divmod(index, 2)
        observed = target[:, index]
        predicted = prediction[:, index]
        low = float(min(observed.min(), predicted.min()))
        high = float(max(observed.max(), predicted.max()))
        figure.add_trace(
            go.Scatter(
                x=predicted,
                y=observed,
                mode="markers",
                marker={
                    "size": 7,
                    "color": observed,
                    "colorscale": "RdYlGn_r",
                    "line": {"width": 0.5, "color": "#1f2937"},
                },
                name=label,
                showlegend=False,
                hovertemplate="predicted=%{x:.5f}<br>observed=%{y:.5f}<extra></extra>",
            ),
            row=row + 1,
            col=column + 1,
        )
        figure.add_trace(
            go.Scatter(
                x=[low, high], y=[low, high], mode="lines", line={"dash": "dash", "color": "#64748b"}, showlegend=False
            ),
            row=row + 1,
            col=column + 1,
        )
        figure.update_xaxes(title_text="Predicted phase delta", row=row + 1, col=column + 1)
        figure.update_yaxes(title_text="Observed phase delta", row=row + 1, col=column + 1)
    figure.update_layout(
        title="Joint latent phase transport: coordinate-grouped nested OOF",
        template="plotly_white",
        height=900,
    )
    figure.write_html(output, include_plotlyjs=True, config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference, indices, target, _panels = aligned_data()
    correlations, spectrum = correlation_tables(target)
    prediction, selections, models = nested_oof(reference, indices, target)
    metrics = metric_table(target, prediction)
    stability = subspace_stability(models)
    grid = full_grid_oof(reference, indices, target)
    rows = []
    for row_index, coordinate in enumerate(indices):
        for output, label in enumerate(OUTPUT_LABELS):
            rows.append(
                {
                    "coordinate_index": int(coordinate),
                    "output": label,
                    "observed_phase_delta": float(target[row_index, output]),
                    "predicted_phase_delta": float(prediction[row_index, output]),
                    "residual": float(prediction[row_index, output] - target[row_index, output]),
                }
            )
    prediction_frame = pd.DataFrame(rows)
    correlations.to_csv(args.output_dir / "phase_delta_correlations.csv", index=False)
    spectrum.to_csv(args.output_dir / "phase_delta_singular_spectrum.csv", index=False)
    selections.to_csv(args.output_dir / "nested_selected_configs.csv", index=False)
    metrics.to_csv(args.output_dir / "nested_oof_metrics.csv", index=False)
    stability.to_csv(args.output_dir / "fold_subspace_stability.csv", index=False)
    grid.to_csv(args.output_dir / "full_grid_oof.csv", index=False)
    prediction_frame.to_csv(args.output_dir / "nested_oof_predictions.csv", index=False)
    render_predictions(target, prediction, args.output_dir / "nested_oof_predictions.html")
    report = [
        "# Round-eight joint latent phase transport identification",
        "",
        "All four outputs share 238 exactly matched policy coordinates. Every coordinate is held out from every output together. No historical or adversarial target value is read.",
        "",
        "## Phase-delta correlations",
        "",
        correlations.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Standardized phase-delta singular spectrum",
        "",
        spectrum.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Nested OOF metrics",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Hyperparameter selections",
        "",
        selections.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Shared-subspace stability",
        "",
        stability.describe().to_markdown(floatfmt=".6f"),
        "",
        "The full non-nested grid is diagnostic only; nested predictions determine promotion.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(metrics.to_string(index=False))
    print(selections.to_string(index=False))


if __name__ == "__main__":
    main()
