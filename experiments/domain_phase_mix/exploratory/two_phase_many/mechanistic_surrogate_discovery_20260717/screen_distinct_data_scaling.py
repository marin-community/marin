# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Screen a target-weighted distinct-data scaling law.

The latent state is retained target-relevant distinct corpus mass rather than
an additive sum of bucket benefits. For bucket ``i`` with relative corpus size
``S_i`` and phase exposures ``e_i^0, e_i^1``, distinct mass is

``U_i^0 = S_i u(e_i^0)`` and ``Delta U_i^1 = S_i[u(e_i^0+e_i^1)-u(e_i^0)]``.

Only a decrease in semantic-family mixture mass across the phase boundary can
forget early distinct state:

``X_i = exp(-lambda gamma_1 [W_f^0-W_f^1]_+) U_i^0 + Delta U_i^1``.

Positive target-specific family values ``q_f`` produce a single effective
distinct-data state ``D=sum_i q_f(i) X_i``. The BPB response is one inverse
scaling law relative to the proportional policy. Phase tying makes the
forgetting factor exactly one, giving the natural fitted single-phase limit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import least_squares
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    freeze_baseline_gate as gate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717.screen_portfolio import (  # noqa: E402
    DASHBOARD,
    PANEL_IDS,
    heldout_data,
    load_panel,
    split_panel_id,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT = (
    RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/round11_distinct_data_scaling"
)
ALPHAS = (0.25, 0.5, 1.0)
PRIOR_SCALES = (0.1, 1.0)
FORGETTING_RATES = (0.0, 2.0, 8.0)
QUALITY_L2S = (0.01, 0.1, 1.0)
SEEDS = (0, 1, 2)
TOP_STAGE1 = 8
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 4}}


class Occupancy(StrEnum):
    WITHOUT_REPLACEMENT = "without_replacement"
    POISSON = "poisson"


class QualityPooling(StrEnum):
    UNIFORM = "uniform"
    FAMILY = "family"


@dataclass(frozen=True)
class Config:
    occupancy: Occupancy
    quality_pooling: QualityPooling
    alpha: float
    prior_scale: float
    forgetting_rate: float
    quality_l2: float

    @property
    def key(self) -> str:
        return (
            f"{self.occupancy.value}__{self.quality_pooling.value}"
            f"__a-{self.alpha:g}__p-{self.prior_scale:g}"
            f"__forget-{self.forgetting_rate:g}__l2-{self.quality_l2:g}"
        )


@dataclass(frozen=True)
class Model:
    config: Config
    intercept: float
    amplitude: float
    family_log_quality: np.ndarray
    parameter_count: int
    effective_degrees_of_freedom: float
    jacobian_condition: float

    def predict(self, panel: Any, weights: np.ndarray) -> np.ndarray:
        feature = scaling_feature(panel, weights, self.config, self.family_log_quality)
        return self.intercept + self.amplitude * feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=DASHBOARD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panels", default=",".join(PANEL_IDS))
    return parser.parse_args()


def configs() -> tuple[Config, ...]:
    output: list[Config] = []
    for occupancy in Occupancy:
        for alpha in ALPHAS:
            for prior in PRIOR_SCALES:
                for forgetting in FORGETTING_RATES:
                    output.append(
                        Config(
                            occupancy=occupancy,
                            quality_pooling=QualityPooling.UNIFORM,
                            alpha=alpha,
                            prior_scale=prior,
                            forgetting_rate=forgetting,
                            quality_l2=0.0,
                        )
                    )
                    output.extend(
                        Config(
                            occupancy=occupancy,
                            quality_pooling=QualityPooling.FAMILY,
                            alpha=alpha,
                            prior_scale=prior,
                            forgetting_rate=forgetting,
                            quality_l2=l2,
                        )
                        for l2 in QUALITY_L2S
                    )
    return tuple(output)


def relative_bucket_sizes(panel: Any) -> np.ndarray:
    sizes = np.stack([panel.phase_fractions[phase] / panel.phase_epoch_factors[phase] for phase in range(2)])
    if not np.isfinite(sizes).all() or np.any(sizes <= 0.0):
        raise ValueError(f"{panel.name}: invalid reconstructed corpus sizes")
    if not np.allclose(sizes[0], sizes[1], rtol=1e-10, atol=1e-12):
        raise ValueError(f"{panel.name}: phase-specific corpus-size reconstructions disagree")
    return sizes.mean(axis=0)


def occupancy(exposure: np.ndarray, law: Occupancy) -> np.ndarray:
    exposure = np.maximum(exposure, 0.0)
    if law is Occupancy.WITHOUT_REPLACEMENT:
        return np.minimum(exposure, 1.0)
    if law is Occupancy.POISSON:
        return -np.expm1(-exposure)
    raise ValueError(law)


def family_weight(panel: Any, weights: np.ndarray) -> np.ndarray:
    return np.column_stack([weights[:, members].sum(axis=1) for members in panel.family_members])


def retained_distinct_state(panel: Any, weights: np.ndarray, config: Config) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    e0 = weights[:, 0] * panel.phase_epoch_factors[0]
    e1 = weights[:, 1] * panel.phase_epoch_factors[1]
    size = relative_bucket_sizes(panel)
    early = size[None, :] * occupancy(e0, config.occupancy)
    total = size[None, :] * occupancy(e0 + e1, config.occupancy)
    late_increment = np.maximum(total - early, 0.0)

    family0 = family_weight(panel, weights[:, 0])
    family1 = family_weight(panel, weights[:, 1])
    drop = np.maximum(family0 - family1, 0.0)
    family_retention = np.exp(-config.forgetting_rate * panel.phase_fractions[1] * drop)
    bucket_retention = np.empty_like(early)
    for family_index, members in enumerate(panel.family_members):
        bucket_retention[:, members] = family_retention[:, family_index, None]
    return early * bucket_retention + late_increment


def family_state(panel: Any, state: np.ndarray) -> np.ndarray:
    return np.column_stack([state[:, members].sum(axis=1) for members in panel.family_members])


def reconstruct_log_quality(free: np.ndarray, family_count: int, pooling: QualityPooling) -> np.ndarray:
    if pooling is QualityPooling.UNIFORM:
        return np.zeros(family_count, dtype=float)
    if len(free) != family_count - 1:
        raise ValueError(f"Expected {family_count - 1} quality contrasts, got {len(free)}")
    return np.concatenate([np.zeros(1, dtype=float), free])


def effective_distinct_ratio(
    panel: Any,
    weights: np.ndarray,
    config: Config,
    family_log_quality: np.ndarray,
) -> np.ndarray:
    state = family_state(panel, retained_distinct_state(panel, weights, config))
    quality = np.exp(np.clip(family_log_quality, -8.0, 8.0))
    effective = state @ quality
    proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
    reference_state = family_state(panel, retained_distinct_state(panel, proportional, config))
    reference = float((reference_state @ quality).item())
    return effective / max(reference, 1e-12)


def scaling_feature(
    panel: Any,
    weights: np.ndarray,
    config: Config,
    family_log_quality: np.ndarray,
) -> np.ndarray:
    ratio = effective_distinct_ratio(panel, weights, config, family_log_quality)
    reference = (1.0 + config.prior_scale) ** (-config.alpha)
    return (np.maximum(ratio, 0.0) + config.prior_scale) ** (-config.alpha) - reference


def fit_model(panel: Any, config: Config, indices: np.ndarray) -> Model:
    indices = np.asarray(indices, dtype=int)
    family_parameters = len(panel.family_names) - 1 if config.quality_pooling is QualityPooling.FAMILY else 0
    parameter_count = 2 + family_parameters

    def residual(parameters: np.ndarray) -> np.ndarray:
        intercept, amplitude = parameters[:2]
        quality = reconstruct_log_quality(parameters[2:], len(panel.family_names), config.quality_pooling)
        prediction = intercept + amplitude * scaling_feature(panel, panel.weights[indices], config, quality)
        output = prediction - panel.observed[indices]
        if family_parameters and config.quality_l2 > 0.0:
            output = np.concatenate([output, math.sqrt(config.quality_l2) * parameters[2:]])
        return output

    y = panel.observed[indices]
    initial = np.concatenate([[float(np.mean(y)), max(float(np.std(y)), 1e-3)], np.zeros(family_parameters)])
    lower = np.concatenate([[0.0, 0.0], np.full(family_parameters, -6.0)])
    upper = np.concatenate([[3.0, 20.0], np.full(family_parameters, 6.0)])
    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=1200,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    if not result.success:
        raise RuntimeError(f"{panel.name}/{config.key}: {result.message}")
    data_jacobian = result.jac[: len(indices)]
    singular = np.linalg.svd(data_jacobian, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))
    ridge = np.zeros(parameter_count, dtype=float)
    ridge[2:] = config.quality_l2
    gram = data_jacobian.T @ data_jacobian
    effective_df = float(np.trace(np.linalg.solve(gram + np.diag(ridge) + 1e-10 * np.eye(parameter_count), gram)))
    quality = reconstruct_log_quality(result.x[2:], len(panel.family_names), config.quality_pooling)
    return Model(
        config=config,
        intercept=float(result.x[0]),
        amplitude=float(result.x[1]),
        family_log_quality=quality,
        parameter_count=parameter_count,
        effective_degrees_of_freedom=effective_df,
        jacobian_condition=condition,
    )


def oof_prediction(panel: Any, dataset: Any, config: Config, seeds: Iterable[int]) -> np.ndarray:
    seed_predictions = []
    for seed in seeds:
        prediction = np.full(panel.n, np.nan, dtype=float)
        for train, test in observatory.folds(dataset, seed):
            model = fit_model(panel, config, train)
            prediction[test] = model.predict(panel, panel.weights[test])
        if not np.isfinite(prediction).all():
            raise ValueError(f"Incomplete OOF prediction for {panel.name}/{config.key}")
        seed_predictions.append(prediction)
    return np.mean(seed_predictions, axis=0)


def leave_region_out_prediction(panel: Any, config: Config) -> np.ndarray:
    coordinates = panel.weights[:, :, 1]
    labels = KMeans(n_clusters=5, random_state=0, n_init=20).fit_predict(coordinates)
    prediction = np.full(panel.n, np.nan, dtype=float)
    for region in sorted(set(labels)):
        test = np.flatnonzero(labels == region)
        train = np.flatnonzero(labels != region)
        prediction[test] = fit_model(panel, config, train).predict(panel, panel.weights[test])
    if not np.isfinite(prediction).all():
        raise ValueError(f"Incomplete leave-region-out prediction for {panel.name}/{config.key}")
    return prediction


def metric_record(
    panel_id: str, config: Config, split: str, observed: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    summary, _bins = gate.metrics(observed, predicted)
    return {"panel": panel_id, "config": config.key, "split": split, **asdict(config), **summary}


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.dashboard)
    bundle = json.loads(args.dashboard.read_text())
    panel_ids = tuple(value.strip() for value in args.panels.split(",") if value.strip())
    all_configs = configs()
    screen_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []

    for panel_id in panel_ids:
        panel, dataset = load_panel(bundle, panel_id)
        stage1 = []
        for config in all_configs:
            prediction = oof_prediction(panel, dataset, config, SEEDS[:1])
            summary, _bins = gate.metrics(panel.observed, prediction)
            row = {"panel": panel_id, "stage": "one_seed", "config": config.key, **asdict(config), **summary}
            screen_rows.append(row)
            stage1.append(row)
        top_keys = {
            row["config"] for row in sorted(stage1, key=lambda row: (row["rmse"], row["regret_at_1"]))[:TOP_STAGE1]
        }
        finalists = [config for config in all_configs if config.key in top_keys]
        final_rows = []
        for config in finalists:
            prediction = oof_prediction(panel, dataset, config, SEEDS)
            row = metric_record(panel_id, config, "fit_oof", panel.observed, prediction)
            screen_rows.append({**row, "stage": "three_seed"})
            final_rows.append(row)
        selected_row = min(final_rows, key=lambda row: (row["rmse"], row["regret_at_1"]))
        selected = next(config for config in finalists if config.key == selected_row["config"])
        metric_rows.append(selected_row)

        full_model = fit_model(panel, selected, np.arange(panel.n))
        for family_name, value in zip(panel.family_names, full_model.family_log_quality, strict=True):
            parameter_rows.append(
                {
                    "panel": panel_id,
                    "config": selected.key,
                    "parameter": f"quality:{family_name}",
                    "raw_value": value,
                    "interpretable_value": math.exp(value),
                    "parameter_count": full_model.parameter_count,
                    "effective_degrees_of_freedom": full_model.effective_degrees_of_freedom,
                    "jacobian_condition": full_model.jacobian_condition,
                }
            )

        if panel.m == 2:
            region_prediction = leave_region_out_prediction(panel, selected)
            metric_rows.append(metric_record(panel_id, selected, "leave_region_out", panel.observed, region_prediction))
        swarm, target = split_panel_id(panel_id)
        heldout = heldout_data(bundle, swarm, target)
        if heldout is not None:
            weights, observed, rows = heldout
            predicted = full_model.predict(panel, weights)
            metric_rows.append(metric_record(panel_id, selected, "heldout_policy_matched", observed, predicted))
            for row, y, prediction in zip(rows, observed, predicted, strict=True):
                prediction_rows.append(
                    {
                        "panel": panel_id,
                        "config": selected.key,
                        "split": "heldout_policy_matched",
                        "row_id": row["name"],
                        "observed": y,
                        "predicted": prediction,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    screen = pd.DataFrame(screen_rows)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    parameters = pd.DataFrame(parameter_rows)
    screen.to_csv(args.output_dir / "hyperparameter_screen.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    parameters.to_csv(args.output_dir / "parameters.csv", index=False)

    if not predictions.empty:
        figure = px.scatter(
            predictions,
            x="predicted",
            y="observed",
            facet_col="panel",
            facet_col_wrap=2,
            color="panel",
            hover_name="row_id",
            color_discrete_sequence=px.colors.diverging.RdYlGn_r,
            title="Target-weighted distinct-data scaling: frozen 3e18 heldouts",
        )
        low = min(predictions["predicted"].min(), predictions["observed"].min())
        high = max(predictions["predicted"].max(), predictions["observed"].max())
        figure.add_shape(type="line", x0=low, x1=high, y0=low, y1=high, line={"dash": "dash", "color": "#777"})
        figure.update_layout(width=1500, height=900, showlegend=False, template="plotly_white")
        figure.write_html(args.output_dir / "heldout_calibration.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    report = [
        "# Target-weighted distinct-data scaling law",
        "",
        "Hyperparameters were selected on fit-panel grouped OOF predictions before any frozen heldout score was computed.",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The phase-tied limit has zero schedule-change hazard by construction. The quality scale is fixed by setting the first family log-quality to zero; only relative family values are identifiable.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
