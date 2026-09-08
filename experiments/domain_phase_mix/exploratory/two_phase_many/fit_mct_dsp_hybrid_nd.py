# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "scipy", "kaleido"]
# ///
"""Evaluate MCT-style scale scaffolds with a DSP mixture anchor.

The baseline MCT-LRQ model is strong on the existing joint-model validation
protocol. This script keeps its centered scale scaffold and swaps the LRQ anchor
for frozen effective-exposure DSP features. It then tests small Apple/DSP-style
interaction additions without retuning the full per-domain DSP geometry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import lsq_linear, minimize
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "analysis_dataset"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "mct_dsp_hybrid_nd_20260515"
IMG_DIR = OUTPUT_DIR / "img"
MCT_REFERENCE_DIR = SCRIPT_DIR / "reference_outputs" / "joint_model_refreshed_20260426" / "mct_lrq_no_barrier_canonical"
FROZEN_DSP_MODEL_PATH = (
    SCRIPT_DIR
    / "reference_outputs"
    / "dsp_canonical_variants_300m_20260510"
    / "dsp_effective_exposure_penalty_nnls"
    / "model.json"
)
PRIMARY_METRIC = "eval/uncheatable_eval/bpb"
N0 = 102_648_576.0
D0 = 5_999_951_872.0
MCT_DROP_EXPONENTS = (0.154791, 0.146425, 0.014295, 1.063376)
RIDGE = 1e-6


@dataclass(frozen=True)
class HybridSpec:
    """One MCT-DSP hybrid candidate."""

    name: str
    interaction_mode: str
    description: str


@dataclass(frozen=True)
class FittedHybrid:
    """Fitted hybrid model."""

    spec: HybridSpec
    theta: np.ndarray
    coef: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    objective: float
    success: bool
    message: str


SPECS = (
    HybridSpec(
        name="mct_dsp_anchor",
        interaction_mode="none",
        description="MCT centered scale scaffold plus frozen effective-exposure DSP anchor.",
    ),
    HybridSpec(
        name="mct_dsp_split_amp",
        interaction_mode="split_amp",
        description="MCT-DSP anchor plus centered benefit/penalty mixture interactions with fitted N amplitudes.",
    ),
    HybridSpec(
        name="mct_dsp_tau_shift",
        interaction_mode="tau_shift",
        description="MCT-DSP anchor with global N/D shifts to the DSP penalty thresholds.",
    ),
    HybridSpec(
        name="mct_dsp_apple_sat",
        interaction_mode="apple_sat",
        description="MCT-DSP anchor with Apple-style shared-r1 repetition discount on saturation exposure only.",
    ),
)


def softplus(x: np.ndarray) -> np.ndarray:
    """Stable softplus."""

    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def effective_repeat(r: np.ndarray, r1: float) -> np.ndarray:
    """Apple-style effective repeat count."""

    repeated = 1.0 + r1 * (1.0 - np.exp(-np.maximum(r - 1.0, 0.0) / r1))
    return np.where(r <= 1.0, r, repeated)


def load_data() -> dict[str, Any]:
    """Load ND packet, MCT split flags, and frozen DSP parameters."""

    frame = pd.read_csv(DATA_DIR / "nd_scale_runs.csv", low_memory=False)
    payload = np.load(DATA_DIR / "nd_scale_packet.npz", allow_pickle=True)
    mask = np.asarray(payload["primary_y_mask"], dtype=bool)
    frame = frame.loc[mask].reset_index(drop=True)
    frozen = json.loads(FROZEN_DSP_MODEL_PATH.read_text())["params"]
    mct_rows = pd.read_csv(MCT_REFERENCE_DIR / "csv" / "row_predictions.csv")
    seed_flags = mct_rows.loc[mct_rows["fit_protocol"].eq("seed7")][
        ["registry_run_key", "seed7_train", "seed7_holdout", "fixed340_holdout", "random_supplement"]
    ].drop_duplicates("registry_run_key")
    leave_flags = mct_rows.loc[mct_rows["fit_protocol"].eq("leave900out")][
        ["registry_run_key", "all900_holdout"]
    ].drop_duplicates("registry_run_key")
    leave_flags["leave900_protocol_row"] = True
    flags = seed_flags.merge(leave_flags, on="registry_run_key", how="outer")
    frame = frame.merge(flags, on="registry_run_key", how="left")
    flag_names = ["seed7_train", "seed7_holdout", "fixed340_holdout", "random_supplement", "all900_holdout"]
    for flag_name in flag_names:
        frame[flag_name] = frame[flag_name].map(lambda value: bool(value) if pd.notna(value) else False)
    frame["leave900_protocol_row"] = frame["leave900_protocol_row"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    frame["non900_train"] = frame["leave900_protocol_row"] & ~frame["all900_holdout"]
    weights = np.asarray(payload["weights"], dtype=float)[mask]
    multipliers = np.asarray(payload["simulated_epoch_multipliers"], dtype=float)[mask]
    return {
        "frame": frame,
        "weights": weights,
        "multipliers": multipliers,
        "y": np.asarray(payload["primary_y"], dtype=float)[mask],
        "domain_names": np.asarray(payload["domain_names"], dtype=object).astype(str),
        "model_sizes": np.asarray(payload["model_sizes"], dtype=float)[mask],
        "realized_train_tokens": np.asarray(payload["realized_train_tokens"], dtype=float)[mask],
        "scale_names": np.asarray(payload["scale_names"], dtype=object).astype(str),
        "scale_index": np.asarray(payload["scale_index"], dtype=np.int64)[mask],
        "frozen_params": {
            "rho": np.asarray(frozen["rho"], dtype=float),
            "tau": np.asarray(frozen["tau"], dtype=float),
            "gamma": float(frozen["gamma"]),
        },
    }


def family_matrix(domain_names: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Return overlapping broad/tech/reasoning family membership."""

    family_names = ["broad_text", "tech_code", "reasoning"]
    matrix = np.zeros((len(domain_names), len(family_names)), dtype=float)
    for idx, domain_name in enumerate(domain_names):
        is_broad = (
            domain_name.startswith("dolma3_cc/")
            or domain_name
            in {
                "dolma3_wikipedia",
                "dolmino_common_crawl_hq",
                "dolmino_olmocr_pdfs_hq",
                "dolmino_stem_heavy_crawl",
            }
            or domain_name.endswith("synth_qa")
        )
        is_tech = any(token in domain_name for token in ("stack_edu", "synth_code", "synth_math")) or domain_name in {
            "dolma3_arxiv",
            "dolma3_finemath_3plus",
        }
        is_reasoning = domain_name in {"dolmino_synth_instruction", "dolmino_synth_thinking"}
        for family_idx, is_member in enumerate((is_broad, is_tech, is_reasoning)):
            matrix[idx, family_idx] = float(is_member)
    return family_names, matrix


def unpack_theta(theta: np.ndarray, spec: HybridSpec) -> dict[str, float]:
    """Decode low-dimensional nonlinear interaction parameters."""

    cursor = 0
    params = {"kappa_benefit": 0.0, "kappa_penalty": 0.0, "eta_n": 0.0, "eta_d": 0.0, "r1": np.inf}
    if spec.interaction_mode == "split_amp":
        params["kappa_benefit"] = float(theta[cursor])
        cursor += 1
        params["kappa_penalty"] = float(theta[cursor])
        cursor += 1
    elif spec.interaction_mode == "tau_shift":
        params["eta_n"] = float(theta[cursor])
        cursor += 1
        params["eta_d"] = float(theta[cursor])
        cursor += 1
    elif spec.interaction_mode == "apple_sat":
        params["r1"] = float(np.exp(theta[cursor]))
        cursor += 1
    elif spec.interaction_mode != "none":
        raise ValueError(f"Unknown interaction mode {spec.interaction_mode}")
    if cursor != len(theta):
        raise ValueError(f"Unused theta for {spec.name}: cursor={cursor}, len={len(theta)}")
    return params


def theta_starts(spec: HybridSpec) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Return deterministic starts and bounds."""

    if spec.interaction_mode == "split_amp":
        return (
            np.asarray([[0.0, 0.0], [-0.5, -0.2], [0.5, 0.2], [-0.5, 0.5]], dtype=float),
            [(-2.0, 2.0), (-2.0, 2.0)],
        )
    if spec.interaction_mode == "tau_shift":
        return (
            np.asarray([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [-0.2, 0.2]], dtype=float),
            [(-2.0, 2.0), (-2.0, 2.0)],
        )
    if spec.interaction_mode == "apple_sat":
        return (
            np.asarray([[np.log(0.5)], [np.log(2.0)], [np.log(10.0)], [np.log(30.0)]], dtype=float),
            [(np.log(1e-3), np.log(100.0))],
        )
    return (np.zeros((1, 0), dtype=float), [])


def dsp_features(
    data: dict[str, Any], indices: np.ndarray, spec: HybridSpec, params: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frozen-geometry DSP features."""

    weights = data["weights"][indices]
    multipliers = data["multipliers"][indices]
    e0 = weights[:, 0, :] * multipliers[:, 0, :]
    e1 = weights[:, 1, :] * multipliers[:, 1, :]
    gamma = data["frozen_params"]["gamma"]
    z = e0 + gamma * e1
    tau = data["frozen_params"]["tau"][None, :]
    if spec.interaction_mode == "tau_shift":
        n = data["model_sizes"][indices] / N0
        d = data["realized_train_tokens"][indices] / D0
        tau = tau + params["eta_n"] * np.log(np.maximum(n, 1e-12))[:, None]
        tau = tau + params["eta_d"] * np.log(np.maximum(d, 1e-12))[:, None]
    saturation_exposure = z
    if spec.interaction_mode == "apple_sat":
        physical = e0 + e1
        repeat = np.maximum(physical, 1e-12)
        saturation_exposure = z * (effective_repeat(repeat, float(params["r1"])) / repeat)
    rho = data["frozen_params"]["rho"][None, :]
    signal = 1.0 - np.exp(-rho * saturation_exposure)
    penalty = softplus(np.log1p(z) - tau) ** 2
    return signal, penalty


def scale_terms(data: dict[str, Any], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centered MCT scale terms and family-head features."""

    alpha, beta, gamma, delta = MCT_DROP_EXPONENTS
    n = data["model_sizes"][indices] / N0
    d = data["realized_train_tokens"][indices] / D0
    u_n = np.power(n, -alpha) - 1.0
    u_d = np.power(d, -beta) - 1.0
    u_nd = np.power(n, -gamma) * np.power(d, -delta) - 1.0
    _family_names, fam = family_matrix(data["domain_names"])
    weights = data["weights"][indices]
    phase0_fam = weights[:, 0, :] @ fam
    phase1_fam = weights[:, 1, :] @ fam
    b_features = np.column_stack([np.ones(len(indices), dtype=float), phase0_fam, phase1_fam])
    return u_n, u_d, u_nd, b_features


def design_matrix(data: dict[str, Any], indices: np.ndarray, spec: HybridSpec, theta: np.ndarray) -> np.ndarray:
    """Build bounded linear design."""

    params = unpack_theta(theta, spec)
    signal, penalty = dsp_features(data, indices, spec, params)
    u_n, u_d, u_nd, b_features = scale_terms(data, indices)
    cols = [
        np.ones(len(indices), dtype=float),
        *[-signal[:, j] for j in range(signal.shape[1])],
        *[penalty[:, j] for j in range(penalty.shape[1])],
        u_n,
        *[u_d * b_features[:, j] for j in range(b_features.shape[1])],
        u_nd,
    ]
    if spec.interaction_mode == "split_amp":
        n = data["model_sizes"][indices] / N0
        benefit_amp = np.power(np.maximum(n, 1e-12), params["kappa_benefit"]) - 1.0
        penalty_amp = np.power(np.maximum(n, 1e-12), params["kappa_penalty"]) - 1.0
        prop_indices = proportional_like_indices(data, indices)
        prop_signal, prop_penalty = dsp_features(data, prop_indices, spec, params)
        delta_signal = signal - prop_signal
        delta_penalty = penalty - prop_penalty
        cols.extend([-(benefit_amp * delta_signal[:, j]) for j in range(signal.shape[1])])
        cols.extend([(penalty_amp * delta_penalty[:, j]) for j in range(penalty.shape[1])])
    return np.column_stack(cols)


def proportional_like_indices(data: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    """Return temporary data rows whose weights/multipliers equal proportional anchors for indices.

    The design builder uses this helper by appending proportional rows to the
    in-memory arrays. It keeps the code simple and deterministic for local
    diagnostics.
    """

    frame = data["frame"]
    prop_lookup: dict[tuple[float, float], int] = {}
    for idx, row in frame.loc[frame["mixture_id"].eq("baseline_proportional")].iterrows():
        prop_lookup[(float(row["model_size"]), float(row["realized_train_tokens"]))] = int(idx)
    return np.asarray(
        [prop_lookup[(float(data["model_sizes"][idx]), float(data["realized_train_tokens"][idx]))] for idx in indices],
        dtype=int,
    )


def fit_linear(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit bounded linear coefficients with intercept free and all heads nonnegative."""

    rest = design[:, 1:]
    mean = rest.mean(axis=0)
    std = rest.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    x = np.column_stack([design[:, :1], (rest - mean) / std])
    if RIDGE > 0:
        penalty = np.sqrt(RIDGE) * np.eye(x.shape[1])
        penalty[0, 0] = 0.0
        x_fit = np.vstack([x, penalty])
        y_fit = np.concatenate([y, np.zeros(x.shape[1], dtype=float)])
    else:
        x_fit = x
        y_fit = y
    lower = np.full(x.shape[1], 0.0, dtype=float)
    upper = np.full(x.shape[1], np.inf, dtype=float)
    lower[0] = -np.inf
    result = lsq_linear(x_fit, y_fit, bounds=(lower, upper), method="trf", tol=1e-10, max_iter=500)
    pred = x @ result.x
    return np.asarray(result.x, dtype=float), mean, std, pred


def fit_model(data: dict[str, Any], train_mask: np.ndarray, spec: HybridSpec) -> FittedHybrid:
    """Fit one hybrid model."""

    train_idx = np.flatnonzero(train_mask)
    starts, bounds = theta_starts(spec)

    def objective(theta: np.ndarray) -> float:
        design = design_matrix(data, train_idx, spec, np.asarray(theta, dtype=float))
        _coef, _mean, _std, pred = fit_linear(design, data["y"][train_idx])
        residual = pred - data["y"][train_idx]
        return float(np.sqrt(np.mean(np.square(residual))))

    best: Any | None = None
    if starts.shape[1] == 0:
        theta = np.zeros(0, dtype=float)
        best = SimpleNamespace(x=theta, fun=objective(theta), success=True, message="no nonlinear parameters")
    else:
        for start in starts:
            result = minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 80, "ftol": 1e-10, "gtol": 1e-7},
            )
            if best is None or float(result.fun) < float(best.fun):
                best = result
    if best is None:
        raise RuntimeError(f"No fit result for {spec.name}")
    design = design_matrix(data, train_idx, spec, np.asarray(best.x, dtype=float))
    coef, mean, std, _pred = fit_linear(design, data["y"][train_idx])
    return FittedHybrid(
        spec=spec,
        theta=np.asarray(best.x, dtype=float),
        coef=coef,
        feature_mean=mean,
        feature_std=std,
        objective=float(best.fun),
        success=bool(best.success),
        message=str(best.message),
    )


def predict_model(model: FittedHybrid, data: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    """Predict selected rows."""

    design = design_matrix(data, indices, model.spec, model.theta)
    rest = design[:, 1:]
    x = np.column_stack([design[:, :1], (rest - model.feature_mean) / model.feature_std])
    return np.asarray(x @ model.coef, dtype=float)


def metric_dict(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Prediction metric dictionary."""

    k = min(8, len(y))
    actual_top = set(np.argsort(y)[:k])
    pred_top = set(np.argsort(pred)[:k])
    return {
        "n": len(y),
        "rmse": float(np.sqrt(np.mean(np.square(pred - y)))),
        "mae": float(np.mean(np.abs(pred - y))),
        "spearman": float(spearmanr(y, pred).statistic) if len(y) > 2 and np.std(pred) > 0 else float("nan"),
        "pearson": float(pearsonr(y, pred).statistic) if len(y) > 2 and np.std(pred) > 0 else float("nan"),
        "regret_at_1": float(y[int(np.argmin(pred))] - np.min(y)),
        "top8_overlap": float(len(actual_top & pred_top) / k),
    }


def evaluate_models(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit all specs on MCT protocols and return summaries."""

    metric_rows = []
    prediction_rows = []
    param_rows = []
    frame = data["frame"]
    all_idx = np.arange(len(frame))
    protocols = [
        (
            "seed7",
            frame["seed7_train"].to_numpy(bool),
            [
                ("seed7_train", frame["seed7_train"].to_numpy(bool)),
                ("seed7_holdout", frame["seed7_holdout"].to_numpy(bool)),
                ("fixed340_holdout", frame["fixed340_holdout"].to_numpy(bool)),
                ("random_supplement", frame["random_supplement"].to_numpy(bool)),
            ],
        ),
        (
            "leave900out",
            frame["non900_train"].to_numpy(bool),
            [
                ("non900_train", frame["non900_train"].to_numpy(bool)),
                ("all900_leave_scale_out", frame["all900_holdout"].to_numpy(bool)),
            ],
        ),
    ]
    for spec in SPECS:
        for protocol, train_mask, splits in protocols:
            print(f"Fitting {spec.name} / {protocol}", flush=True)
            model = fit_model(data, train_mask, spec)
            pred_all = predict_model(model, data, all_idx)
            params = unpack_theta(model.theta, spec)
            param_rows.append(
                {
                    "model": spec.name,
                    "fit_protocol": protocol,
                    "objective": model.objective,
                    "success": model.success,
                    "message": model.message,
                    **params,
                }
            )
            pred_frame = frame[
                [
                    "registry_run_key",
                    "mixture_id",
                    "run_name",
                    "scale",
                    "scale_display_label",
                    "target_budget_multiplier",
                ]
            ].copy()
            pred_frame["model"] = spec.name
            pred_frame["fit_protocol"] = protocol
            pred_frame["actual_bpb"] = data["y"]
            pred_frame["pred_bpb"] = pred_all
            pred_frame["residual_pred_minus_actual"] = pred_all - data["y"]
            prediction_rows.append(pred_frame)
            for split_name, mask in splits:
                row = metric_dict(data["y"][mask], pred_all[mask])
                row.update({"model": spec.name, "fit_protocol": protocol, "split": split_name})
                metric_rows.append(row)
    return (
        pd.DataFrame.from_records(metric_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame.from_records(param_rows),
    )


def mct_reference_metrics() -> pd.DataFrame:
    """Compute canonical MCT-LRQ69-drop metrics for comparison.

    The saved MCT metric table predates the regret/top-8 diagnostics used by
    this script, so compute the shared comparison columns directly from row
    predictions.
    """

    predictions = pd.read_csv(MCT_REFERENCE_DIR / "csv" / "row_predictions.csv")
    predictions = predictions.loc[predictions["model"].eq("mct_lrq69_drop_no_barrier")].copy()
    metric_rows = []
    split_specs = {
        "seed7": (
            ("seed7_holdout", "seed7_holdout"),
            ("fixed340_holdout", "fixed340_holdout"),
            ("random_supplement", "random_supplement"),
        ),
        "leave900out": (("all900_leave_scale_out", "all900_holdout"),),
    }
    for protocol, splits in split_specs.items():
        protocol_rows = predictions.loc[predictions["fit_protocol"].eq(protocol)]
        for split_name, flag_name in splits:
            split_rows = protocol_rows.loc[protocol_rows[flag_name].astype(bool)]
            row = metric_dict(
                split_rows["actual_bpb"].to_numpy(float),
                split_rows["pred_bpb"].to_numpy(float),
            )
            row.update({"model": "mct_lrq69_drop_no_barrier", "fit_protocol": protocol, "split": split_name})
            metric_rows.append(row)
    return pd.DataFrame.from_records(metric_rows)


def write_plots(metrics: pd.DataFrame, mct_metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Write compact diagnostics."""

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    compare = metrics.loc[
        metrics["split"].isin(["seed7_holdout", "fixed340_holdout", "random_supplement", "all900_leave_scale_out"])
    ].copy()
    mct = mct_metrics.rename(columns={"bias_pred_minus_actual": "bias"}).copy()
    mct["model"] = "mct_lrq69_drop_no_barrier"
    cols = ["model", "fit_protocol", "split", "n", "rmse", "spearman", "regret_at_1", "top8_overlap"]
    compare = pd.concat([compare[cols], mct[cols]], ignore_index=True)
    long = compare.melt(
        id_vars=["model", "fit_protocol", "split"],
        value_vars=["rmse", "spearman", "regret_at_1", "top8_overlap"],
        var_name="metric",
        value_name="value",
    )
    fig = px.bar(
        long,
        x="split",
        y="value",
        color="model",
        facet_col="metric",
        barmode="group",
        title="MCT-DSP hybrid vs canonical MCT on MCT validation splits",
        height=750,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(template="plotly_white")
    fig.write_html(IMG_DIR / "mct_dsp_hybrid_metric_comparison.html", include_plotlyjs="cdn")
    try:
        fig.write_image(IMG_DIR / "mct_dsp_hybrid_metric_comparison.png", scale=2)
    except ValueError:
        pass

    plot_frame = predictions.loc[
        predictions["fit_protocol"].eq("seed7") & predictions["model"].isin(["mct_dsp_anchor", "mct_dsp_split_amp"])
    ].copy()
    fig = px.scatter(
        plot_frame,
        x="actual_bpb",
        y="pred_bpb",
        color="scale_display_label",
        facet_col="model",
        hover_name="run_name",
        hover_data=["mixture_id", "target_budget_multiplier", "residual_pred_minus_actual"],
        title="MCT-DSP hybrid seed7 predictions",
        height=650,
    )
    lo = min(plot_frame["actual_bpb"].min(), plot_frame["pred_bpb"].min())
    hi = max(plot_frame["actual_bpb"].max(), plot_frame["pred_bpb"].max())
    fig.add_shape(type="line", x0=lo, x1=hi, y0=lo, y1=hi, line={"dash": "dot", "color": "#333333"})
    fig.update_layout(template="plotly_white")
    fig.write_html(IMG_DIR / "mct_dsp_hybrid_pred_actual.html", include_plotlyjs="cdn")
    try:
        fig.write_image(IMG_DIR / "mct_dsp_hybrid_pred_actual.png", scale=2)
    except ValueError:
        pass


def render_report(metrics: pd.DataFrame, mct_metrics: pd.DataFrame, params: pd.DataFrame) -> str:
    """Render Markdown report."""

    core_splits = ["seed7_holdout", "fixed340_holdout", "random_supplement", "all900_leave_scale_out"]
    compare = metrics.loc[metrics["split"].isin(core_splits)].copy()
    mct = mct_metrics.copy()
    mct["model"] = "mct_lrq69_drop_no_barrier"
    keep = ["model", "fit_protocol", "split", "n", "rmse", "mae", "spearman", "pearson", "regret_at_1", "top8_overlap"]
    compare = pd.concat([mct[keep], compare[keep]], ignore_index=True)
    return "\n".join(
        [
            "# MCT-DSP Hybrid ND Evaluation",
            "",
            "## Data",
            "",
            f"- Source: `{DATA_DIR / 'nd_scale_runs.csv'}`",
            f"- Metric: `{PRIMARY_METRIC}`",
            f"- Frozen DSP geometry: `{FROZEN_DSP_MODEL_PATH}`",
            f"- MCT reference: `{MCT_REFERENCE_DIR}`",
            "",
            "## Tested Forms",
            "",
            "- `mct_dsp_anchor`: MCT centered scale scaffold plus frozen effective-exposure DSP anchor.",
            "- `mct_dsp_split_amp`: adds centered benefit/penalty interaction terms with fitted N amplitudes.",
            "- `mct_dsp_tau_shift`: shifts all DSP penalty thresholds by global `eta_N log(N/N0)+eta_D log(D/D0)`.",
            "- `mct_dsp_apple_sat`: applies Apple-style shared-r1 repetition discount to saturation exposure only.",
            "",
            "## Validation Metrics",
            "",
            compare.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Fitted Low-Dimensional Interaction Parameters",
            "",
            params.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Interpretation",
            "",
            "- The hybrid forms are competitive with the variable-scale DSP screen, but they do not beat canonical MCT on the established MCT validation splits.",
            "- `mct_dsp_tau_shift` is the best hybrid on the seed7/fixed340/random split family, but it generalizes poorly to leave-900-out.",
            "- The plain anchor and Apple-style saturation discount are the safest leave-900-out hybrids, but they are still materially worse than MCT there.",
            "- The split-amplitude interaction improves train fit and then collapses on every held-out split; do not promote it without much stronger regularization.",
            "- The main limitation is that DSP domain geometry is frozen from the 300M fit. A real promotion attempt would need analytic/autodiff gradients for full ND retuning and then raw-optimum/perturbation-gradient diagnostics.",
            "",
        ]
    )


def main() -> None:
    """Run local MCT-DSP hybrid evaluation."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    metrics, predictions, params = evaluate_models(data)
    mct_metrics = mct_reference_metrics()
    metrics.to_csv(OUTPUT_DIR / "metric_summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "row_predictions.csv", index=False)
    params.to_csv(OUTPUT_DIR / "fitted_params.csv", index=False)
    mct_metrics.to_csv(OUTPUT_DIR / "mct_reference_metric_summary.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "metric_summary": metrics.to_dict(orient="records"),
                "fitted_params": params.to_dict(orient="records"),
                "mct_reference_metric_summary": mct_metrics.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    write_plots(metrics, mct_metrics, predictions)
    (OUTPUT_DIR / "report.md").write_text(render_report(metrics, mct_metrics, params))
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
