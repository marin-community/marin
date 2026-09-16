# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
"""Fit matched original-style one- and two-phase separate-heads models.

Both policy classes use the same 279 mixture coordinates. The unmatched
``baseline_stratified`` row is excluded, and each policy's proportional
baseline plus ten shared constant-schedule repeats is collapsed to one mean
target. The models are independently fit and ridge-selected by identical CV
folds before optimizing a common deployment-KL grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_joint_phase_correspondence_dsp as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_two_phase_canonical_bowl_candidates_300m as bowl,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "original_style_matched_sepheads_ablation_20260712"
DEFAULT_GCS_OUTPUT_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_original_style_matched_sepheads_ablation_20260712/mixtures"
)
PRIOR_OUTPUT_DIR = REFERENCE_OUTPUTS / "original_separate_heads_policy_ablation_20260712"
OBJECTIVES = ("uncheatable", "table9")
POLICIES = ("1p", "2p")
TARGET_ABBR = {"uncheatable": "unch", "table9": "t9"}
DEFAULT_L2_VALUES = (
    0.0,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
)
DEFAULT_KL_VALUES = (0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
CV_SEEDS = (0, 1, 2)
N_SPLITS = 5
LOWER_TAIL_FRAC = 0.15
MU_SHIFTS = np.linspace(-2.0, 2.0, 13)
PHASE_FRACTIONS = (0.8, 0.2)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class SeparateHeadsModel:
    policy: str
    l2: float
    mus: tuple[np.ndarray, ...]
    intercept: float
    coefficients: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class CvMetric:
    objective: str
    policy: str
    l2: float
    seed: int
    oof_rmse: float
    oof_spearman: float
    fold_mean_regret_at_1: float
    lower_tail_optimism: float
    low_tail_rmse: float


@dataclass(frozen=True)
class OptimizerResult:
    weights: np.ndarray
    regularized_objective: float
    successful_starts: int


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()


def normalize_panel_source(value: object) -> str:
    source = str(value)
    if "qsplit" in source:
        return "qsplit_signal"
    if source == "domain_deletion":
        return source
    raise ValueError(f"Unexpected matched-panel source {source!r}")


def collapsed_proportional_panel(rows: pd.DataFrame, target: str, policy: str) -> pd.DataFrame:
    proportional = rows.loc[rows["phase_correspondence_key"].eq("baseline_proportional")].copy()
    if len(proportional) != 11:
        raise ValueError(f"{policy}: expected 11 proportional observations, found {len(proportional)}")
    base_rows = proportional.loc[proportional["phase_pair_status"].eq("paired_single_two")]
    if len(base_rows) != 1:
        raise ValueError(f"{policy}: expected one primary proportional row, found {len(base_rows)}")
    mean_row = base_rows.iloc[[0]].copy()
    mean_row[target] = float(proportional[target].mean())
    mean_row["run_name"] = f"matched_{policy}_mean_baseline_proportional"
    mean_row["panel_source"] = "qsplit_signal"
    mean_row["proportional_observation_count"] = len(proportional)
    non_proportional = rows.loc[~rows["phase_correspondence_key"].eq("baseline_proportional")].copy()
    panel = pd.concat([non_proportional, mean_row], ignore_index=True)
    panel["panel_source"] = panel["panel_source"].map(normalize_panel_source)
    panel = panel.sort_values("phase_correspondence_key").reset_index(drop=True)
    if len(panel) != 279:
        raise ValueError(f"{policy}: expected 279 collapsed rows, found {len(panel)}")
    if panel["phase_correspondence_key"].duplicated().any():
        raise ValueError(f"{policy}: collapsed panel contains duplicate correspondence keys")
    return panel


def matched_policy_frames(
    frame: pd.DataFrame,
    target: str,
    domain_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    single = frame.loc[frame["policy_family"].eq("single_phase")].copy()
    repeats = frame.loc[frame["phase_pair_status"].eq("repeat_reference_to_baseline")].copy()
    single_proportional = single.loc[single["phase_correspondence_key"].eq("baseline_proportional")]
    if len(single_proportional) != 1:
        raise ValueError("Expected one single-phase proportional baseline")
    phase_columns = [f"phase_{phase}_{domain}" for phase in (0, 1) for domain in domain_names]
    for column in phase_columns:
        repeats[column] = float(single_proportional.iloc[0][column])
    repeats["policy_family"] = "single_phase"
    one_phase = collapsed_proportional_panel(pd.concat([single, repeats], ignore_index=True), target, "1p")

    two_phase_rows = frame.loc[
        frame["split"].eq("train")
        & frame["phase_pair_status"].isin(("paired_single_two", "repeat_reference_to_baseline"))
    ].copy()
    two_phase = collapsed_proportional_panel(two_phase_rows, target, "2p")
    keys_1p = one_phase["phase_correspondence_key"].tolist()
    keys_2p = two_phase["phase_correspondence_key"].tolist()
    if keys_1p != keys_2p:
        raise ValueError("One- and two-phase panel correspondence keys differ")
    if "baseline_stratified" in keys_1p:
        raise ValueError("Unmatched baseline_stratified leaked into matched panels")
    return one_phase, two_phase


def matched_datasets(frame: pd.DataFrame, objective: str) -> tuple[pooled.Dataset, pooled.Dataset]:
    target = joint.TARGET_COLUMNS[objective]
    domain_names = pooled.load_300m_dataset(objective).domain_names
    one_frame, two_frame = matched_policy_frames(frame, target, domain_names)
    one = joint.dataset_from_frame(objective, one_frame, target)
    two = joint.dataset_from_frame(objective, two_frame, target)
    one_keys = one.frame["phase_correspondence_key"].tolist()
    two_keys = two.frame["phase_correspondence_key"].tolist()
    if one.n != 279 or two.n != 279 or one_keys != two_keys:
        raise ValueError(
            f"{objective}: target filtering broke matched panel alignment " f"(one_phase={one.n}, two_phase={two.n})"
        )
    if not np.allclose(one.weights[:, 0, :], one.weights[:, 1, :], atol=1e-12):
        raise ValueError(f"{objective}: one-phase weights are not tied")
    return one, two


def policy_exposures(dataset: pooled.Dataset, weights: np.ndarray, policy: str) -> tuple[np.ndarray, ...]:
    if policy == "1p":
        if not np.allclose(weights[:, 0, :], weights[:, 1, :], atol=1e-10):
            raise ValueError("One-phase model received phase-varying weights")
        return (weights[:, 0, :] * (dataset.c0 + dataset.c1)[None, :],)
    if policy == "2p":
        return (
            weights[:, 0, :] * dataset.c0[None, :],
            weights[:, 1, :] * dataset.c1[None, :],
        )
    raise ValueError(f"Unknown policy {policy!r}")


def bowl_design(exposure: np.ndarray, mu: np.ndarray) -> np.ndarray:
    delta = np.log1p(exposure) - mu[None, :]
    return np.hstack([np.minimum(delta, 0.0) ** 2, np.maximum(delta, 0.0) ** 2])


def selected_mu(exposure: np.ndarray, target: np.ndarray, l2: float) -> np.ndarray:
    masked = np.where(exposure > 1e-8, exposure, np.nan)
    # Preserve the original separate-heads _gridmu behavior exactly. In
    # particular, deletion zeros make the ordinary median non-finite for a
    # bucket, triggering the shared fallback center before the global shift
    # sweep. A nanmedian would silently define a different surrogate.
    base_mu = np.median(np.log1p(masked), axis=0)
    base_mu = np.where(np.isfinite(base_mu), base_mu, 2.0)
    base_mu = np.clip(base_mu, -2.0, 8.0)
    best_rmse = np.inf
    best_mu = base_mu
    for shift in MU_SHIFTS:
        mu = np.clip(base_mu + shift, -2.0, 8.0)
        design = bowl_design(exposure, mu)
        intercept, coefficients = bowl.fit_head(design, target, l2)
        prediction = intercept + design @ coefficients
        rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_mu = mu
    return best_mu


def fit_model(dataset: pooled.Dataset, indices: np.ndarray, policy: str, l2: float) -> SeparateHeadsModel:
    target = dataset.y[indices]
    exposures = policy_exposures(dataset, dataset.weights[indices], policy)
    mus = tuple(selected_mu(exposure, target, l2) for exposure in exposures)
    designs = [bowl_design(exposure, mu) for exposure, mu in zip(exposures, mus, strict=True)]
    design = np.hstack(designs)
    intercept, coefficients = bowl.fit_head(design, target, l2)
    width = designs[0].shape[1]
    split_coefficients = tuple(
        np.asarray(coefficients[offset : offset + width], dtype=float) for offset in range(0, len(coefficients), width)
    )
    return SeparateHeadsModel(
        policy=policy,
        l2=l2,
        mus=mus,
        intercept=intercept,
        coefficients=split_coefficients,
    )


def predict_model(model: SeparateHeadsModel, dataset: pooled.Dataset, weights: np.ndarray) -> np.ndarray:
    exposures = policy_exposures(dataset, np.asarray(weights, dtype=float), model.policy)
    prediction = np.full(len(weights), model.intercept, dtype=float)
    for exposure, mu, coefficients in zip(exposures, model.mus, model.coefficients, strict=True):
        prediction += bowl_design(exposure, mu) @ coefficients
    return prediction


def cv_metric(
    objective: str,
    policy: str,
    l2: float,
    seed: int,
    target: np.ndarray,
    prediction: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> CvMetric:
    residual = prediction - target
    fold_regrets = []
    for _train_indices, test_indices in folds:
        selected = test_indices[int(np.argmin(prediction[test_indices]))]
        fold_regrets.append(float(target[selected] - np.min(target[test_indices])))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(target))))
    tail = np.argsort(prediction)[:tail_count]
    tail_residual = residual[tail]
    return CvMetric(
        objective=objective,
        policy=policy,
        l2=l2,
        seed=seed,
        oof_rmse=float(np.sqrt(np.mean(residual**2))),
        oof_spearman=float(spearmanr(target, prediction).statistic),
        fold_mean_regret_at_1=float(np.mean(fold_regrets)),
        lower_tail_optimism=float(np.mean(np.maximum(-tail_residual, 0.0))),
        low_tail_rmse=float(np.sqrt(np.mean(tail_residual**2))),
    )


def cross_validate(
    objective: str,
    datasets: dict[str, pooled.Dataset],
    l2_values: tuple[float, ...],
) -> pd.DataFrame:
    rows = []
    fold_frame = datasets["2p"].frame
    for l2 in l2_values:
        for seed in CV_SEEDS:
            folds = component_dsp.panel_stratified_folds(fold_frame, n_splits=N_SPLITS, seed=seed)
            for policy in POLICIES:
                dataset = datasets[policy]
                prediction = np.zeros(dataset.n, dtype=float)
                for fold_index, (train_indices, test_indices) in enumerate(folds, start=1):
                    print(
                        f"{objective}/{policy}: L2={l2:g}, seed={seed}, fold={fold_index}/{N_SPLITS}",
                        flush=True,
                    )
                    model = fit_model(dataset, train_indices, policy, l2)
                    prediction[test_indices] = predict_model(model, dataset, dataset.weights[test_indices])
                rows.append(asdict(cv_metric(objective, policy, l2, seed, dataset.y, prediction, folds)))
    return pd.DataFrame(rows)


def summarize_cv(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["objective", "policy", "l2"], as_index=False)
        .agg(
            oof_rmse_mean=("oof_rmse", "mean"),
            oof_rmse_sd=("oof_rmse", "std"),
            oof_spearman_mean=("oof_spearman", "mean"),
            fold_mean_regret_at_1_mean=("fold_mean_regret_at_1", "mean"),
            lower_tail_optimism_mean=("lower_tail_optimism", "mean"),
            low_tail_rmse_mean=("low_tail_rmse", "mean"),
        )
        .sort_values(["objective", "policy", "oof_rmse_mean", "fold_mean_regret_at_1_mean", "l2"])
        .reset_index(drop=True)
    )


def selected_l2(summary: pd.DataFrame, objective: str, policy: str) -> float:
    selected = summary.loc[summary["objective"].eq(objective) & summary["policy"].eq(policy)]
    return float(selected.iloc[0]["l2"])


def predictor(model: SeparateHeadsModel, dataset: pooled.Dataset) -> Callable[[np.ndarray], float]:
    def predict(weights: np.ndarray) -> float:
        candidate = np.asarray(weights, dtype=float)[None, :, :]
        return float(predict_model(model, dataset, candidate)[0])

    return predict


def weighted_kl(weights: np.ndarray, natural: np.ndarray) -> float:
    return float(olmix.weighted_multiclass_kl(weights, natural, PHASE_FRACTIONS))


def optimize(
    predict: Callable[[np.ndarray], float],
    dataset: pooled.Dataset,
    natural: np.ndarray,
    kl_reg: float,
    policy: str,
) -> OptimizerResult:
    one_phase = policy == "1p"
    m = dataset.m

    def weights_from_logits(logits: np.ndarray) -> np.ndarray:
        if one_phase:
            values = softmax(logits)
            return np.stack([values, values])
        return np.stack([softmax(logits[:m]), softmax(logits[m:])])

    def logits_for_weights(weights: np.ndarray) -> np.ndarray:
        if one_phase:
            return np.log(np.clip(weights[0], 1e-12, 1.0))
        return np.log(np.clip(weights, 1e-12, 1.0)).reshape(-1)

    def objective(logits: np.ndarray) -> float:
        weights = weights_from_logits(logits)
        return predict(weights) + kl_reg * weighted_kl(weights, natural)

    starts = [logits_for_weights(np.stack([natural, natural]))]
    starts.extend(logits_for_weights(dataset.weights[index]) for index in np.argsort(dataset.y)[:8])
    best_value = np.inf
    best_weights = None
    successful_starts = 0
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": 400, "ftol": 1e-10, "maxls": 30},
        )
        if result.success:
            successful_starts += 1
        if np.isfinite(result.fun) and float(result.fun) < best_value:
            best_value = float(result.fun)
            best_weights = weights_from_logits(np.asarray(result.x, dtype=float))
    if best_weights is None:
        raise RuntimeError(f"No finite optimizer result for {policy}, KL={kl_reg:g}")
    return OptimizerResult(best_weights, best_value, successful_starts)


def kl_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def candidate_row(
    objective: str,
    policy: str,
    l2: float,
    kl_reg: float,
    prediction: float,
    result: OptimizerResult,
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
) -> dict[str, object]:
    aggregate = PHASE_FRACTIONS[0] * result.weights[0] + PHASE_FRACTIONS[1] * result.weights[1]
    epochs = olmix.simulated_epochs(result.weights, token_counts, target_budget=target_budget)
    return {
        "candidate": f"origstyle_sep_{TARGET_ABBR[objective]}_{policy}_kl{kl_tag(kl_reg)}",
        "objective": objective,
        "policy": policy,
        "selected_l2": l2,
        "kl_reg": kl_reg,
        "predicted_bpb": prediction,
        "regularized_objective": result.regularized_objective,
        "weighted_kl_to_proportional": weighted_kl(result.weights, natural),
        "aggregate_tv_to_proportional": float(0.5 * np.abs(aggregate - natural).sum()),
        "phase_tv": float(0.5 * np.abs(result.weights[0] - result.weights[1]).sum()),
        "max_weight": float(result.weights.max()),
        "max_simulated_epoch": float(epochs.max()),
        "q95_simulated_epoch": float(np.quantile(epochs, 0.95)),
        "optimizer_successful_starts": result.successful_starts,
    }


def write_candidate(
    output_dir: Path,
    gcs_output_dir: str,
    candidate: str,
    frame: pd.DataFrame,
    upload: bool,
) -> None:
    mixture_dir = output_dir / "mixtures"
    mixture_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(mixture_dir / f"{candidate}.csv", index=False)
    if upload:
        with fsspec.open(f"{gcs_output_dir.rstrip('/')}/{candidate}.csv", "wt") as handle:
            frame.to_csv(handle, index=False)


def mixture_weights(frame: pd.DataFrame, domains: list[str]) -> np.ndarray:
    indexed = frame.set_index("domain").loc[domains]
    return np.stack(
        [
            indexed["phase_0_weight"].to_numpy(dtype=float),
            indexed["phase_1_weight"].to_numpy(dtype=float),
        ]
    )


def prior_candidate_path(objective: str) -> Path:
    return PRIOR_OUTPUT_DIR / f"original_sep_{objective}_cv_selected_2p_kl0p1.csv"


def compare_two_phase_refit(
    objective: str,
    domains: list[str],
    natural: np.ndarray,
    matched_candidate: pd.DataFrame,
) -> dict[str, float | str]:
    prior_path = prior_candidate_path(objective)
    if not prior_path.exists():
        raise FileNotFoundError(f"Missing prior 280-row candidate: {prior_path}")
    prior = mixture_weights(pd.read_csv(prior_path), domains)
    matched = mixture_weights(matched_candidate, domains)
    prior_aggregate = PHASE_FRACTIONS[0] * prior[0] + PHASE_FRACTIONS[1] * prior[1]
    matched_aggregate = PHASE_FRACTIONS[0] * matched[0] + PHASE_FRACTIONS[1] * matched[1]
    return {
        "objective": objective,
        "prior_candidate": str(prior_path),
        "matched_candidate": f"origstyle_sep_{TARGET_ABBR[objective]}_2p_kl0p1",
        "phase_0_tv": float(0.5 * np.abs(prior[0] - matched[0]).sum()),
        "phase_1_tv": float(0.5 * np.abs(prior[1] - matched[1]).sum()),
        "aggregate_tv": float(0.5 * np.abs(prior_aggregate - matched_aggregate).sum()),
        "max_abs_weight_delta": float(np.max(np.abs(prior - matched))),
        "matched_aggregate_tv_to_proportional": float(0.5 * np.abs(matched_aggregate - natural).sum()),
    }


def plot_cv(summary: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(rows=1, cols=2, subplot_titles=["Uncheatable", "Table-9"])
    colors = {"1p": "#6F8190", "2p": "#E36F2C"}
    for column, objective in enumerate(OBJECTIVES, start=1):
        for policy in POLICIES:
            data = summary.loc[
                summary["objective"].eq(objective) & summary["policy"].eq(policy) & summary["l2"].gt(0)
            ].sort_values("l2")
            figure.add_trace(
                go.Scatter(
                    x=data["l2"],
                    y=data["oof_rmse_mean"],
                    error_y={"type": "data", "array": data["oof_rmse_sd"], "visible": True},
                    mode="lines+markers",
                    name=f"{policy} independently fit",
                    legendgroup=policy,
                    showlegend=column == 1,
                    line={"color": colors[policy]},
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(type="log", title_text="ridge L2", row=1, col=column)
        figure.update_yaxes(title_text="grouped OOF RMSE", row=1, col=column)
    figure.update_layout(
        title="Original-style matched separate-heads: independent ridge CV",
        template="plotly_white",
        width=1250,
        height=500,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.17},
        margin={"l": 70, "r": 30, "t": 90, "b": 90},
    )
    figure.write_html(output, include_plotlyjs=True, config=PLOT_CONFIG)


def plot_kl_paths(manifest: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Uncheatable predicted BPB",
            "Table-9 predicted BPB",
            "Uncheatable maximum epochs",
            "Table-9 maximum epochs",
        ],
    )
    colors = {"1p": "#6F8190", "2p": "#E36F2C"}
    for column, objective in enumerate(OBJECTIVES, start=1):
        for policy in POLICIES:
            data = manifest.loc[manifest["objective"].eq(objective) & manifest["policy"].eq(policy)].sort_values(
                "kl_reg"
            )
            figure.add_trace(
                go.Scatter(
                    x=data["kl_reg"],
                    y=data["predicted_bpb"],
                    mode="lines+markers",
                    name=f"{policy} independently fit",
                    legendgroup=policy,
                    showlegend=column == 1,
                    line={"color": colors[policy]},
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=data["kl_reg"],
                    y=data["max_simulated_epoch"],
                    mode="lines+markers",
                    name=f"{policy} independently fit",
                    legendgroup=policy,
                    showlegend=False,
                    line={"color": colors[policy]},
                ),
                row=2,
                col=column,
            )
    figure.update_xaxes(title_text="deployment KL", row=2, col=1)
    figure.update_xaxes(title_text="deployment KL", row=2, col=2)
    figure.update_yaxes(title_text="predicted BPB", row=1, col=1)
    figure.update_yaxes(title_text="predicted BPB", row=1, col=2)
    figure.update_yaxes(title_text="max simulated epochs", row=2, col=1)
    figure.update_yaxes(title_text="max simulated epochs", row=2, col=2)
    figure.update_layout(
        title="Original-style matched one-phase/two-phase KL panel",
        template="plotly_white",
        width=1300,
        height=850,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.1},
        margin={"l": 70, "r": 30, "t": 90, "b": 110},
    )
    figure.write_html(output, include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(
    cv_summary: pd.DataFrame,
    manifest: pd.DataFrame,
    refit_shift: pd.DataFrame,
    output: Path,
) -> None:
    selected = (
        cv_summary.sort_values(["objective", "policy", "oof_rmse_mean"])
        .groupby(["objective", "policy"], as_index=False)
        .first()
    )
    lines = [
        "# Original-style matched separate-heads ablation",
        "",
        "Both policy classes use 279 corresponding coordinates. Each proportional target is the mean of one "
        "policy-specific baseline plus ten constant-schedule repeats. The unmatched `baseline_stratified` "
        "row is excluded.",
        "",
        "## Selected ridge penalties",
        "",
        selected.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Candidate panel",
        "",
        manifest.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Effect of dropping the unmatched two-phase row",
        "",
        "The matched 279-row two-phase KL=0.1 candidate is compared with the prior CV-selected "
        "280-row candidate. Small TV means the independently fitted ablation does not materially "
        "change the established two-phase proposal.",
        "",
        refit_shift.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gcs-output-dir", default=DEFAULT_GCS_OUTPUT_DIR)
    parser.add_argument("--l2-values", default=",".join(str(value) for value in DEFAULT_L2_VALUES))
    parser.add_argument("--kl-values", default=",".join(str(value) for value in DEFAULT_KL_VALUES))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = pooled.load_300m_dataset("table9")
    frame = joint.attach_single_phase_weights(pd.read_csv(args.packet), args.one_phase_source, reference.domain_names)
    l2_values = parse_float_tuple(args.l2_values)
    kl_values = parse_float_tuple(args.kl_values)
    datasets_by_objective: dict[str, dict[str, pooled.Dataset]] = {}
    cv_frames = []
    audit_frames = []
    for objective in OBJECTIVES:
        one, two = matched_datasets(frame, objective)
        datasets_by_objective[objective] = {"1p": one, "2p": two}
        audit_frames.append(
            pd.DataFrame(
                {
                    "objective": objective,
                    "phase_correspondence_key": one.frame["phase_correspondence_key"],
                    "panel_source": one.frame["panel_source"],
                    "one_phase_target": one.y,
                    "two_phase_target": two.y,
                }
            )
        )
        cv_frames.append(cross_validate(objective, datasets_by_objective[objective], l2_values))
    audit = pd.concat(audit_frames, ignore_index=True)
    audit.to_csv(args.output_dir / "matched_panel_targets.csv", index=False)
    cv_metrics = pd.concat(cv_frames, ignore_index=True)
    cv_summary = summarize_cv(cv_metrics)
    cv_metrics.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    cv_summary.to_csv(args.output_dir / "cv_summary.csv", index=False)

    manifest_rows = []
    matched_kl_point_one: dict[str, pd.DataFrame] = {}
    selected_models: dict[str, dict[str, float]] = {}
    for objective in OBJECTIVES:
        _packet, domains, natural, token_counts, target_budget, _folds = bowl.load_objective(objective)
        selected_models[objective] = {}
        for policy in POLICIES:
            dataset = datasets_by_objective[objective][policy]
            l2 = selected_l2(cv_summary, objective, policy)
            selected_models[objective][policy] = l2
            model = fit_model(dataset, np.arange(dataset.n), policy, l2)
            predict = predictor(model, dataset)
            for kl_reg in kl_values:
                result = optimize(predict, dataset, natural, kl_reg, policy)
                prediction = predict(result.weights)
                row = candidate_row(
                    objective,
                    policy,
                    l2,
                    kl_reg,
                    prediction,
                    result,
                    natural,
                    token_counts,
                    target_budget,
                )
                candidate = str(row["candidate"])
                mixture = per_component.mixture_frame(
                    domains=domains,
                    natural=natural,
                    weights=result.weights,
                    token_counts=token_counts,
                    target_budget=target_budget,
                )
                write_candidate(args.output_dir, args.gcs_output_dir, candidate, mixture, args.upload)
                if policy == "2p" and np.isclose(kl_reg, 0.1):
                    matched_kl_point_one[objective] = mixture
                manifest_rows.append(row)
                print(
                    f"{candidate}: pred={prediction:.6f}, max_epoch={row['max_simulated_epoch']:.3f}",
                    flush=True,
                )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    if set(matched_kl_point_one) != set(OBJECTIVES):
        raise ValueError("KL grid must include 0.1 for the two-phase refit comparison")
    refit_shift = pd.DataFrame(
        [
            compare_two_phase_refit(
                objective,
                datasets_by_objective[objective]["2p"].domain_names,
                bowl.load_objective(objective)[2],
                matched_kl_point_one[objective],
            )
            for objective in OBJECTIVES
        ]
    )
    refit_shift.to_csv(args.output_dir / "two_phase_refit_shift.csv", index=False)
    (args.output_dir / "selected_models.json").write_text(json.dumps(selected_models, indent=2) + "\n")
    plot_cv(cv_summary, args.output_dir / "cv_l2_sweep.html")
    plot_kl_paths(manifest, args.output_dir / "kl_policy_paths.html")
    write_report(cv_summary, manifest, refit_shift, args.output_dir / "report.md")
    print(f"Wrote {len(manifest)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
